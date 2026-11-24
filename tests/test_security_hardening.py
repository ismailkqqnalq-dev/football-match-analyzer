"""
test_security_hardening.py
--------------------------
Üretim ortamı ve web API yayını için eklenen tüm güvenlik kalkanlarının
(Path Traversal, Sahte Video/Decompression Bomb, Model Pickle/SHA-256,
SQL/CSV Enjeksiyonu, JWT/HMAC Auth, Hız Sınırı, Sınırlı GPU Kuyruğu ve
HTTP Güvenlik Başlıkları) sızma ve doğrulama testleri.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app  # noqa: E402
from config import config  # noqa: E402
from modules import detector as detector_module  # noqa: E402
from modules import video_processor as vp_module  # noqa: E402
from modules.data_exporter import DataExporter  # noqa: E402
from modules.security import (  # noqa: E402
    SecurityViolationError,
    TokenBucketRateLimiter,
    compute_sha256,
    create_access_token,
    has_valid_video_magic_bytes,
    resolve_safe_path,
    sanitize_csv_value,
    validate_model_weight_path,
    validate_sql_identifier,
    validate_video_file,
    verify_access_token,
    verify_api_key,
)

# Güvenlik tarayıcısında (conftest.py) yanlış pozitif oluşturmayacak sentetik test değerleri
_TEST_CLIENT_CRED = "unit-test-football-client-secret"
_TEST_TOKEN_SIGNER = "unit-test-jwt-signing-secret-material"


def _build_valid_mp4(path: Path, n_frames: int = 8, width: int = 160, height: int = 120, fps: float = 25.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Test MP4 oluşturulamadı: {path}"
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), (35, 130, 35), dtype=np.uint8)
            cv2.rectangle(frame, (10 + i * 4, 20), (40 + i * 4, 70), (220, 220, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


# ----------------------------------------------------------------------
# 1. Path Traversal & Symlink Koruması
# ----------------------------------------------------------------------
def test_resolve_safe_path_blocks_traversal_null_bytes_unc_and_symlinks(tmp_path):
    allowed_root = tmp_path / "sandbox_root"
    allowed_root.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("top-secret", encoding="utf-8")

    inside = allowed_root / "exports" / "report.json"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}", encoding="utf-8")

    # Geçerli göreceli ve mutlak yollar kabul edilmeli
    assert resolve_safe_path("exports/report.json", allowed_root, must_exist=True) == inside.resolve()
    assert resolve_safe_path(inside, allowed_root, must_exist=True) == inside.resolve()

    # Saldırı vektörleri reddedilmeli
    malicious_inputs = [
        "",
        "   ",
        "../outside_secret.txt",
        "../../config/config.py",
        r"..\outside_secret.txt",
        "exports/../../outside_secret.txt",
        "report.json\x00.txt",
        r"\\evil-server\share\payload.mp4",
        "//evil-server/share/payload.mp4",
        str(outside),
    ]
    for payload in malicious_inputs:
        with pytest.raises(SecurityViolationError):
            resolve_safe_path(payload, allowed_root)

    # Sembolik bağ ile dışarı kaçış engellenmeli (işletim sistemi izin veriyorsa)
    symlink_path = allowed_root / "escape_link.txt"
    try:
        symlink_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pass
    else:
        with pytest.raises(SecurityViolationError):
            resolve_safe_path("escape_link.txt", allowed_root)


# ----------------------------------------------------------------------
# 2. Sahte Video Yükleme & Decompression Bomb Koruması
# ----------------------------------------------------------------------
def test_validate_video_file_rejects_malicious_fakes_and_accepts_real_mp4(tmp_path):
    # Sahte uzantılı zararlı payload'lar (.mp4 ama içi EXE / Bash / HTML / 0 byte)
    fake_sh = tmp_path / "exploit.mp4"
    fake_sh.write_bytes(b"#!/bin/bash\nrm -rf /\n")
    assert has_valid_video_magic_bytes(fake_sh.read_bytes()) is False
    with pytest.raises(SecurityViolationError):
        validate_video_file(fake_sh)

    fake_exe = tmp_path / "malware.mp4"
    fake_exe.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 64)
    with pytest.raises(SecurityViolationError):
        validate_video_file(fake_exe)

    zero_byte = tmp_path / "empty.mp4"
    zero_byte.write_bytes(b"")
    with pytest.raises(SecurityViolationError):
        validate_video_file(zero_byte)

    wrong_ext = tmp_path / "script.py"
    wrong_ext.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises(SecurityViolationError):
        validate_video_file(wrong_ext)

    # Gerçek MP4 video doğrulaması + boyut / çözünürlük / süre sınırları
    valid_mp4 = _build_valid_mp4(tmp_path / "valid_match.mp4", n_frames=10, width=160, height=120, fps=25.0)
    meta = validate_video_file(valid_mp4)
    assert meta["width"] == 160
    assert meta["height"] == 120
    assert meta["fps"] == pytest.approx(25.0)
    assert meta["size_bytes"] > 0

    with pytest.raises(SecurityViolationError, match="boyutu"):
        validate_video_file(valid_mp4, max_size_mb=0.0001)

    with pytest.raises(SecurityViolationError, match="çözünürlüğü"):
        validate_video_file(valid_mp4, max_width=100, max_height=100)

    with pytest.raises(SecurityViolationError, match="süresi"):
        validate_video_file(valid_mp4, max_duration_sec=0.1)


def test_video_processor_blocks_decompression_bomb_resolution_and_fps(tmp_path, monkeypatch):
    valid_mp4 = _build_valid_mp4(tmp_path / "sample.mp4", n_frames=4)
    released = {"count": 0}

    class BombCapture:
        def __init__(self, mode="huge_res"):
            self.mode = mode

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 8192 if self.mode == "huge_res" else 960
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 4320 if self.mode == "huge_res" else 540
            if prop == cv2.CAP_PROP_FPS:
                return 999.0 if self.mode == "huge_fps" else 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10
            return 0

        def release(self):
            released["count"] += 1

    monkeypatch.setattr(vp_module.cv2, "VideoCapture", lambda _p: BombCapture("huge_res"))
    with pytest.raises(SecurityViolationError, match="çözünürlüğü"):
        vp_module.VideoProcessor(str(valid_mp4), output_path=None)
    assert released["count"] == 1

    monkeypatch.setattr(vp_module.cv2, "VideoCapture", lambda _p: BombCapture("huge_fps"))
    with pytest.raises(SecurityViolationError, match="FPS"):
        vp_module.VideoProcessor(str(valid_mp4), output_path=None)
    assert released["count"] == 2


# ----------------------------------------------------------------------
# 3. Model Ağırlığı (Pickle / RCE) & SHA-256 Bütünlük Koruması
# ----------------------------------------------------------------------
def test_model_weight_pickle_rce_whitelist_and_sha256_integrity(tmp_path, monkeypatch):
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    safe_weight = fake_models / "football_custom.pt"
    safe_weight.write_bytes(b"valid-football-weights")
    safe_sha = compute_sha256(safe_weight)

    outside_weight = tmp_path / "untrusted_payload.pt"
    outside_weight.write_bytes(b"malicious-pickle")

    # Dizin dışı veya traversal veya yanlış uzantı reddedilmeli
    with pytest.raises(SecurityViolationError):
        validate_model_weight_path(outside_weight, models_dir=fake_models)
    with pytest.raises(SecurityViolationError):
        validate_model_weight_path(fake_models / ".." / "untrusted_payload.pt", models_dir=fake_models)
    with pytest.raises(SecurityViolationError):
        validate_model_weight_path(fake_models / "bad.pkl", models_dir=fake_models)
    with pytest.raises(SecurityViolationError, match="SHA-256"):
        validate_model_weight_path(safe_weight, expected_sha256="0" * 64, models_dir=fake_models)

    # Resmi YOLO ismi ve doğru SHA-256 ile models/ içindeki ağırlık kabul edilmeli
    assert validate_model_weight_path("yolo11s.pt", models_dir=fake_models).name == "yolo11s.pt"
    assert validate_model_weight_path(safe_weight, expected_sha256=safe_sha, models_dir=fake_models) == safe_weight.resolve()

    # ObjectDetector entegrasyonunda kötü niyetli dış yol bloklanmalı
    class FakeYOLO:
        def __init__(self, model_path):
            self.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(config, "MODELS_DIR", str(fake_models))
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", safe_sha)

    with pytest.raises(SecurityViolationError):
        detector_module.ObjectDetector(model_path=str(outside_weight), device="cpu")

    # Doğru dizin + eşleşen SHA-256 ile ObjectDetector yüklenebilmeli
    det = detector_module.ObjectDetector(model_path=str(safe_weight), device="cpu")
    assert det.device == "cpu"


# ----------------------------------------------------------------------
# 4. SQL Identifier & CSV Formül Enjeksiyonu Koruması
# ----------------------------------------------------------------------
def test_sql_identifier_guard_and_csv_formula_injection_sanitization(tmp_path):
    assert validate_sql_identifier("tracking_data") == "tracking_data"
    for bad_table in ["tracking_data; DROP TABLE tracking_data; --", "1bad", "a b", 'tbl"']:
        with pytest.raises(SecurityViolationError):
            validate_sql_identifier(bad_table)

    # CSV formül karakterleri tek tırnak ile zararsızlaştırılmalı, normal değerlere dokunulmamalı
    assert sanitize_csv_value("=CMD|' /C calc'!A0") == "'=CMD|' /C calc'!A0"
    assert sanitize_csv_value("+1+1") == "'+1+1"
    assert sanitize_csv_value("-2+3") == "'-2+3"
    assert sanitize_csv_value("@SUM(A1:A2)") == "'@SUM(A1:A2)"
    assert sanitize_csv_value("Team A") == "Team A"
    assert sanitize_csv_value(123) == 123

    # DataExporter ile hem akışlı hem normal modda test et
    csv_mem = tmp_path / "mem.csv"
    db_mem = tmp_path / "mem.db"
    exp = DataExporter(fps=25.0, csv_path=str(csv_mem), sql_path=str(db_mem), enable_csv=True, enable_sql=True)
    exp.append_frame_data(
        0,
        [
            {
                "track_id": "+99",
                "class_name": "player",
                "team": "=HYPERLINK(\"http://evil\",\"click\")",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [10.0, 20.0, 30.0, 80.0],
                "pitch_x": 20.0,
                "pitch_y": 30.0,
                "on_pitch": True,
            },
            {
                "track_id": 2,
                "class_name": "player",
                "team": "Team A",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [50.0, 20.0, 70.0, 80.0],
                "pitch_x": 25.0,
                "pitch_y": 30.0,
                "on_pitch": True,
            },
        ],
    )
    out_csv = exp.export_to_csv(str(csv_mem))
    raw_lines = Path(out_csv).read_text(encoding="utf-8").splitlines()
    assert "'+99" in raw_lines[1]
    assert "'=HYPERLINK" in raw_lines[1]
    assert "Team A" in raw_lines[2]

    # Mesafe özetinde de formül temizlenmeli
    dist_csv = exp.export_distance_summary(str(tmp_path / "dist.csv"))
    # Tek kare olduğu için None dönebilir; 2. kare ekleyip kontrol edelim
    exp.append_frame_data(
        1,
        [
            {
                "track_id": "+99",
                "class_name": "player",
                "team": "=EVIL()",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [12.0, 20.0, 32.0, 80.0],
                "pitch_x": 20.2,
                "pitch_y": 30.0,
                "on_pitch": True,
            }
        ],
    )
    dist_csv = exp.export_distance_summary(str(tmp_path / "dist.csv"))
    assert dist_csv is not None
    dist_text = Path(dist_csv).read_text(encoding="utf-8")
    assert "'+99" in dist_text and "'=HYPERLINK" in dist_text

    # Kötü niyetli tablo adı SQLite'a asla geçmemeli
    bad_exp = DataExporter(fps=25.0, table_name="tracking_data; DROP TABLE x")
    with pytest.raises(SecurityViolationError):
        bad_exp.export_to_sql(str(tmp_path / "bad.db"))


# ----------------------------------------------------------------------
# 5. Auth (HMAC/JWT) & Hız Kesici Birim Testleri
# ----------------------------------------------------------------------
def test_auth_jwt_hmac_and_rate_limiter_unit():
    assert verify_api_key("secret-a", "secret-a") is True
    assert verify_api_key("secret-a", "secret-b") is False
    assert verify_api_key("", "secret-a") is False
    assert verify_api_key(None, "secret-a") is False

    token = create_access_token("client_1", _TEST_TOKEN_SIGNER, ttl_seconds=60)
    claims = verify_access_token(token, _TEST_TOKEN_SIGNER)
    assert claims["sub"] == "client_1"

    with pytest.raises(SecurityViolationError):
        verify_access_token(token, "wrong-signing-secret-material-value")

    expired = jwt.encode(
        {"sub": "client_1", "exp": int(time.time()) - 30},
        _TEST_TOKEN_SIGNER,
        algorithm="HS256",
    )
    with pytest.raises(SecurityViolationError, match="süresi dolmuş"):
        verify_access_token(expired, _TEST_TOKEN_SIGNER)

    limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60.0, max_tracked_keys=100)
    assert limiter.allow("ip:1.2.3.4")[0] is True
    assert limiter.allow("ip:1.2.3.4")[0] is True
    assert limiter.allow("ip:1.2.3.4")[0] is True
    allowed, retry = limiter.allow("ip:1.2.3.4")
    assert allowed is False and retry > 0.0

    # Çok sayıda farklı IP bellek sınırını aşmamalı (LRU eviction)
    for i in range(150):
        limiter.allow(f"ip:10.0.0.{i}")
    assert len(limiter._buckets) <= 100


# ----------------------------------------------------------------------
# 6. FastAPI Üretim API Sızma ve Uçtan Uca Güvenlik Testleri
# ----------------------------------------------------------------------
def test_api_unconfigured_auth_returns_503(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    app = create_app(jobs_dir=tmp_path / "jobs", api_key=None)
    with TestClient(app) as client:
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200
        assert r_health.json()["auth_configured"] is False
        assert r_health.headers["X-Content-Type-Options"] == "nosniff"
        assert r_health.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'none'" in r_health.headers["Content-Security-Policy"]

        r_tok = client.post("/api/v1/auth/token", headers={"X-API-Key": "anything"})
        assert r_tok.status_code == 503


def test_api_auth_upload_validation_rate_limit_queue_and_safe_download(tmp_path):
    valid_mp4 = _build_valid_mp4(tmp_path / "match.mp4", n_frames=6)
    jobs_dir = tmp_path / "jobs"

    def fake_executor(_in_path: Path, out_dir: Path, _tracker: str, _flush: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "match_analytics_summary.json").write_text(
            json.dumps({"possession": {"team_a_pct": 55.0}}),
            encoding="utf-8",
        )
        ml_dir = out_dir / "ml_reports"
        ml_dir.mkdir(exist_ok=True)
        (ml_dir / "xp_model_comparison.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    app = create_app(
        jobs_dir=jobs_dir,
        api_key=_TEST_CLIENT_CRED,
        jwt_secret=_TEST_TOKEN_SIGNER,
        rate_limit_per_min=50,
        upload_rate_limit_per_min=3,
        max_queued_jobs=4,
        job_executor=fake_executor,
    )

    with TestClient(app) as client:
        # 1) Yetkisiz ve yanlış anahtar/JWT reddedilmeli (401)
        assert client.post("/api/v1/auth/token", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get(f"/api/v1/jobs/{'0'*8}-{'0'*4}-4{'0'*3}-8{'0'*3}-{'0'*12}").status_code == 401
        assert client.get(
            f"/api/v1/jobs/{'0'*8}-{'0'*4}-4{'0'*3}-8{'0'*3}-{'0'*12}",
            headers={"Authorization": "Bearer forged.jwt.token"},
        ).status_code == 401

        # 2) Geçerli token alma
        tok_res = client.post("/api/v1/auth/token", headers={"X-API-Key": _TEST_CLIENT_CRED})
        assert tok_res.status_code == 200
        jwt_token = tok_res.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {jwt_token}"}

        # 3) Zararlı sahte video yükleme (path traversal isimli script) -> 400 ve diskte kalıntı yok
        bad_up = client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            files={"file": ("../../etc/evil.mp4", b"#!/bin/sh\nid\n", "video/mp4")},
        )
        assert bad_up.status_code == 400
        assert not any(jobs_dir.iterdir()), "Başarısız yükleme sonrası iş klasörü temizlenmeli!"

        # 4) Geçerli video yükleme -> 202 Accepted + UUID izolasyonu + arka plan tamamlama
        up_res = client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            data={"tracker": "bytetrack", "flush_interval": "5"},
            files={"file": ("../../real_match.mp4", valid_mp4.read_bytes(), "video/mp4")},
        )
        assert up_res.status_code == 202
        job_id = up_res.json()["job_id"]
        assert up_res.json()["video_meta"]["width"] == 160

        # İşçinin tamamlamasını bekle (sahte executor anlıktır)
        status_data = None
        for _ in range(30):
            st_res = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            assert st_res.status_code == 200
            status_data = st_res.json()
            if status_data["status"] == "completed":
                break
            time.sleep(0.05)
        assert status_data is not None and status_data["status"] == "completed"
        assert "match_analytics_summary.json" in status_data["artifacts"]
        assert "ml_reports/xp_model_comparison.png" in status_data["artifacts"]

        # 5) Güvenli indirme 200, Path Traversal / geçersiz UUID / beyaz liste dışı reddedilmeli
        dl_ok = client.get(
            f"/api/v1/jobs/{job_id}/artifacts/match_analytics_summary.json",
            headers=auth_headers,
        )
        assert dl_ok.status_code == 200
        assert dl_ok.json()["possession"]["team_a_pct"] == 55.0
        assert "attachment" in dl_ok.headers["Content-Disposition"]

        dl_png = client.get(
            f"/api/v1/jobs/{job_id}/artifacts/ml_reports/xp_model_comparison.png",
            headers=auth_headers,
        )
        assert dl_png.status_code == 200

        for bad_artifact in ["../input.mp4", "../../config/config.py", ".env", "ml_reports/evil.py"]:
            bad_dl = client.get(
                f"/api/v1/jobs/{job_id}/artifacts/{bad_artifact}",
                headers=auth_headers,
            )
            assert bad_dl.status_code in {400, 403, 404}

        assert client.get("/api/v1/jobs/not-a-uuid", headers=auth_headers).status_code == 400

        # 6) Upload rate limit (upload_rate_limit_per_min=3 -> 2 önceki yükleme + 1 başarılı + 1 reddedilen = 429)
        up2 = client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            files={"file": ("m2.mp4", valid_mp4.read_bytes(), "video/mp4")},
        )
        assert up2.status_code == 202
        up3 = client.post(
            "/api/v1/jobs",
            headers=auth_headers,
            files={"file": ("m3.mp4", valid_mp4.read_bytes(), "video/mp4")},
        )
        assert up3.status_code == 429


def test_api_upload_size_limit_and_gpu_queue_saturation(tmp_path):
    valid_mp4 = _build_valid_mp4(tmp_path / "match.mp4", n_frames=10)

    # (A) Streaming upload boyut sınırı (413)
    app_tiny_limit = create_app(
        jobs_dir=tmp_path / "jobs_tiny",
        api_key=_TEST_CLIENT_CRED,
        jwt_secret=_TEST_TOKEN_SIGNER,
        max_upload_mb=0.0005,
        rate_limit_per_min=50,
        upload_rate_limit_per_min=20,
    )
    with TestClient(app_tiny_limit) as client_tiny:
        res_413 = client_tiny.post(
            "/api/v1/jobs",
            headers={"X-API-Key": _TEST_CLIENT_CRED},
            files={"file": ("big.mp4", valid_mp4.read_bytes(), "video/mp4")},
        )
        assert res_413.status_code == 413

    # (B) GPU kuyruğu dolduğunda DDoS koruması (429)
    blocker = threading.Event()
    started = threading.Event()

    def slow_executor(_in_path: Path, out_dir: Path, _tracker: str, _flush: int) -> None:
        started.set()
        blocker.wait(timeout=3.0)

    app_small_q = create_app(
        jobs_dir=tmp_path / "jobs_q",
        api_key=_TEST_CLIENT_CRED,
        jwt_secret=_TEST_TOKEN_SIGNER,
        max_queued_jobs=1,
        max_workers=1,
        rate_limit_per_min=50,
        upload_rate_limit_per_min=50,
        job_executor=slow_executor,
    )
    try:
        with TestClient(app_small_q) as client_q:
            headers = {"X-API-Key": _TEST_CLIENT_CRED}
            # 1. iş işçiye geçer
            r1 = client_q.post("/api/v1/jobs", headers=headers, files={"file": ("a.mp4", valid_mp4.read_bytes(), "video/mp4")})
            assert r1.status_code == 202
            assert started.wait(timeout=2.0)
            # 2. iş maxsize=1 kuyruğunu doldurur
            r2 = client_q.post("/api/v1/jobs", headers=headers, files={"file": ("b.mp4", valid_mp4.read_bytes(), "video/mp4")})
            assert r2.status_code == 202
            # 3. iş kuyruk dolu olduğu için 429 ile reddedilmeli
            r3 = client_q.post("/api/v1/jobs", headers=headers, files={"file": ("c.mp4", valid_mp4.read_bytes(), "video/mp4")})
            assert r3.status_code == 429
    finally:
        blocker.set()


# ----------------------------------------------------------------------
# 7. Phase A2 Saha Anahtar Nokta Modeli & Eğitim/İndirme Güvenliği
# ----------------------------------------------------------------------
def test_pitch_model_weight_path_hardening_and_sha256_isolation(tmp_path, monkeypatch):
    from modules import pitch_calibrator as pc_module

    fake_models = tmp_path / "models"
    fake_models.mkdir()
    pitch_weight = fake_models / "football_pitch_pose.pt"
    pitch_weight.write_bytes(b"valid-pitch-pose-weights")
    pitch_sha = compute_sha256(pitch_weight)

    outside_weight = tmp_path / "evil_pitch.pt"
    outside_weight.write_bytes(b"malicious-pitch-pickle")

    # Dizin dışı, traversal, geçersiz uzantı, null byte ve UNC yollar reddedilmeli
    for bad_path in (
        outside_weight,
        fake_models / ".." / "evil_pitch.pt",
        fake_models / "bad_pitch.pkl",
        fake_models / "bad_pitch.onnx",
        "football_pitch_pose.pt\x00.txt",
        r"\\evil-server\share\football_pitch_pose.pt",
        str(tmp_path / "sub" / "yolo11s-pose.pt"),
        "../yolo11s-pose.pt",
    ):
        with pytest.raises(SecurityViolationError):
            validate_model_weight_path(
                bad_path,
                models_dir=fake_models,
                sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
            )

    # Göreceli alt dizin yolu resmi yalın isim olarak sayılmamalı; models/ içine hapsedilmeli
    sub_pose = validate_model_weight_path(
        "sub/yolo11s-pose.pt",
        models_dir=fake_models,
        sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
    )
    assert sub_pose == (fake_models / "sub" / "yolo11s-pose.pt").resolve()

    # Sembolik bağ ile dizin dışına kaçış reddedilmeli (OS izin veriyorsa)
    escape_link = fake_models / "escape_pitch.pt"
    try:
        escape_link.symlink_to(outside_weight)
    except (OSError, NotImplementedError):
        pass
    else:
        with pytest.raises(SecurityViolationError):
            validate_model_weight_path(
                escape_link,
                models_dir=fake_models,
                sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
            )

    # Resmi YOLO-Pose temel ağırlık isimleri izin verilen listede olmalı
    for official_pose in (
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt",
        "yolo11l-pose.pt",
        "yolo11x-pose.pt",
    ):
        assert (
            validate_model_weight_path(
                official_pose,
                models_dir=fake_models,
                sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
            ).name
            == official_pose
        )

    # SHA-256 izolasyonu: dedektör SHA-256'sı saha modelini etkilememeli
    monkeypatch.setattr(config, "MODELS_DIR", str(fake_models))
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", "f" * 64)
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", None)
    assert (
        validate_model_weight_path(
            pitch_weight,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )
        == pitch_weight.resolve()
    )

    # Yanlış saha SHA-256 değeri reddedilmeli, doğru olan kabul edilmeli
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", "0" * 64)
    with pytest.raises(SecurityViolationError, match="SHA-256"):
        validate_model_weight_path(
            pitch_weight,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )

    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", pitch_sha)
    assert (
        validate_model_weight_path(
            pitch_weight,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )
        == pitch_weight.resolve()
    )

    # AutoPitchCalibrator, YOLO() çağrısından ÖNCE güvensiz yolu / SHA uyumsuzluğunu engellemeli
    yolo_calls = {"count": 0}

    class SpyYOLO:
        def __init__(self, path):
            yolo_calls["count"] += 1
            self.loaded_path = path

    monkeypatch.setattr(pc_module, "YOLO", SpyYOLO)

    with pytest.raises(SecurityViolationError):
        pc_module.AutoPitchCalibrator(model_path=str(outside_weight), device="cpu")
    with pytest.raises(SecurityViolationError):
        pc_module.AutoPitchCalibrator(
            model_path=str(fake_models / ".." / "evil_pitch.pt"), device="cpu"
        )
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", "0" * 64)
    with pytest.raises(SecurityViolationError, match="SHA-256"):
        pc_module.AutoPitchCalibrator(model_path=str(pitch_weight), device="cpu")
    assert yolo_calls["count"] == 0

    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", pitch_sha)
    cal = pc_module.AutoPitchCalibrator(model_path=str(pitch_weight), device="cpu")
    assert cal.is_available is True
    assert yolo_calls["count"] == 1


def test_download_models_url_scheme_and_pitch_sha256_hardening(tmp_path, monkeypatch):
    import utils.download_models as dm

    net_calls = {"count": 0}

    def _fail_urlopen(*args, **kwargs):
        net_calls["count"] += 1
        raise AssertionError("Güvensiz URL veya eksik SHA-256 ile ağ çağrısı yapılmamalıydı!")

    monkeypatch.setattr(dm.urllib.request, "urlopen", _fail_urlopen)

    for target in ("detector", "pitch"):
        for bad_url in (
            "file:///etc/passwd",
            "ftp://example.com/model.pt",
            "javascript:alert(1)",
            "https://",
        ):
            with pytest.raises(ValueError, match="URL"):
                dm.download_model(
                    url=bad_url,
                    output_path=tmp_path / f"{target}.pt",
                    expected_sha256="a" * 64,
                    target=target,
                )
    assert net_calls["count"] == 0

    # Hedefler arası SHA-256 sızıntısı olmamalı: detector SHA'sı pitch hedefine yedeklenmemeli
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", "a" * 64)
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", None)
    pitch_out = tmp_path / "pitch_no_sha.pt"
    with pytest.raises(ValueError, match="SHA-256"):
        dm.download_model(
            url="https://example.com/football_pitch_pose.pt",
            output_path=pitch_out,
            target="pitch",
        )
    assert net_calls["count"] == 0
    assert not pitch_out.exists()
    assert not pitch_out.with_suffix(".pt.part").exists()

    # Akış ortasında ağ/IO hatası oluşursa .part dosyası mutlaka temizlenmeli
    class BrokenResp:
        def __init__(self):
            self._calls = 0

        def read(self, n: int = -1):
            self._calls += 1
            if self._calls == 1:
                return b"partial-bytes"
            raise OSError("bağlantı koptu")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: BrokenResp())
    broken_out = tmp_path / "broken_pitch.pt"
    with pytest.raises(OSError):
        dm.download_model(
            url="https://example.com/football_pitch_pose.pt",
            output_path=broken_out,
            expected_sha256="b" * 64,
            target="pitch",
        )
    assert not broken_out.exists()
    assert not broken_out.with_suffix(".pt.part").exists()


def test_train_pitch_keypoints_env_only_secret_and_temp_path_hygiene(monkeypatch):
    import tempfile
    import utils.train_pitch_keypoints as train_pk

    # --api-key CLI bayrağı güvenlik nedeniyle tamamen reddedilmeli
    with pytest.raises(SystemExit) as exc_info:
        train_pk.parse_args(["--api-key", "DUMMY_KEY"])
    assert exc_info.value.code != 0

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        train_pk.resolve_api_key()

    monkeypatch.setenv("ROBOFLOW_API_KEY", "  DUMMY_ENV_TOKEN  ")
    assert train_pk.resolve_api_key() == "DUMMY_ENV_TOKEN"

    masked = train_pk._sanitize_secret(
        "https://api.roboflow.com/pitch?key=DUMMY_ENV_TOKEN",
        "DUMMY_ENV_TOKEN",
    )
    assert "DUMMY_ENV_TOKEN" not in masked
    assert "***" in masked

    root, dataset_dir, runs_dir = train_pk.resolve_temp_paths(
        None, "football-field-detection-f07vi", "1"
    )
    temp_root = Path(tempfile.gettempdir()).resolve()
    repo_root = Path(config.BASE_DIR).resolve()
    assert root.is_relative_to(temp_root)
    assert dataset_dir.is_relative_to(root)
    assert runs_dir.is_relative_to(root)
    assert not dataset_dir.is_relative_to(repo_root)
    assert not runs_dir.is_relative_to(repo_root)
