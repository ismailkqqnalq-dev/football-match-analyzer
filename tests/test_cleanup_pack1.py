"""
Düzeltme Paketi 1 (T1, T2, T4, T5, T8) için yeni kabul testleri.
"""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.ml_analyzer import MLAnalyzer


def _make_few_proxy_passes_df(n_events: int = 8) -> pd.DataFrame:
    """30'dan az gerçek pas olayı içeren kalibre edilmiş takip DataFrame'i üretir."""
    rows = []
    f_cursor = 0
    for ev_idx in range(n_events):
        outcome_same = (ev_idx % 3) != 0
        t1_team = "Team A"
        t2_team = "Team A" if outcome_same else "Team B"
        p1_x = 20.0 + (ev_idx % 4) * 5.0
        p2_x = p1_x + 12.0
        for step in range(12):
            bx = p1_x if step < 4 else (p1_x + (step - 3) * (12.0 / 5.0) if step < 8 else p2_x)
            ts = round(f_cursor / 25.0, 3)
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
                "bbox_x1": bx * 10, "bbox_y1": 340, "bbox_x2": bx * 10 + 10, "bbox_y2": 350,
                "pixel_x": bx * 10, "pixel_y": 340, "pitch_x": bx, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": 10 + ev_idx * 2, "class_name": "player", "team": t1_team, "is_predicted": False,
                "bbox_x1": p1_x * 10, "bbox_y1": 300, "bbox_x2": p1_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p1_x * 10, "pixel_y": 340, "pitch_x": p1_x, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": 11 + ev_idx * 2, "class_name": "player", "team": t2_team, "is_predicted": False,
                "bbox_x1": p2_x * 10, "bbox_y1": 300, "bbox_x2": p2_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p2_x * 10, "pixel_y": 340, "pitch_x": p2_x, "pitch_y": 34.0, "on_pitch": True,
            })
            f_cursor += 1
        f_cursor += 110
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# T1 KABUL TESTLERİ: xP / xT Sentetik Veri Geçişinin Kaldırılması
# ----------------------------------------------------------------------
def test_t1_xp_returns_insufficient_data_below_30_real_events(tmp_path):
    df = _make_few_proxy_passes_df(n_events=8)
    ml = MLAnalyzer(df, fps=25.0, random_state=42)

    xp = ml.run_xp_modeling()
    assert xp["status"] == "insufficient_data"
    assert xp["data_source"] != "calibrated_synthetic_match"
    assert xp["real_events"] < 30
    assert xp["required"] == 30
    assert xp["models"] == []

    # xT hiçbir gerçek pas olmayan kalibre DataFrame'de insufficient_data dönmeli
    no_pass_df = pd.DataFrame([
        {
            "frame_idx": 0, "timestamp_sec": 0.0,
            "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
            "bbox_x1": 500, "bbox_y1": 340, "bbox_x2": 510, "bbox_y2": 350,
            "pixel_x": 505.0, "pixel_y": 345.0, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
        },
        {
            "frame_idx": 0, "timestamp_sec": 0.0,
            "track_id": 1, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 490, "bbox_y1": 300, "bbox_x2": 520, "bbox_y2": 370,
            "pixel_x": 505.0, "pixel_y": 340.0, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
        },
    ])
    no_pass_ml = MLAnalyzer(no_pass_df, fps=25.0, random_state=42)
    xt_empty = no_pass_ml.compute_xt_analysis()
    assert xt_empty["status"] == "insufficient_data"

    # export_all sahte grafik üretmemeli, raporda "Hesaplanamadı – yetersiz veri" yer almalı
    out_dir = tmp_path / "t1_insuff"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)
    assert summary["xp_modeling"]["status"] == "insufficient_data"
    assert not (out_dir / "xp_model_comparison.png").exists()
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Hesaplanamadı – yetersiz veri" in md_text


def test_t1_main_cli_short_video_prints_insufficient_not_roc_auc(tmp_path, monkeypatch, capsys):
    import main as main_module

    sample_df = _make_few_proxy_passes_df(n_events=6)

    class DummyVP:
        def __init__(self, inp, out):
            self.fps = 25.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_frames(self):
            return iter([])

    class DummyObj:
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def get_team_map(self):
            return {}

    class DummyPerspective:
        is_calibrated = True

    class DummyExporter:
        def __init__(self, fps=25.0, **kwargs):
            self.fps = fps

        def postprocess(self, team_map=None):
            return {}

        def export_to_csv(self):
            return None

        def export_to_sql(self):
            return None

        def export_distance_summary(self):
            return None

        def to_dataframe(self):
            return sample_df.copy()

    monkeypatch.setattr(main_module, "VideoProcessor", DummyVP)
    monkeypatch.setattr(main_module, "ObjectDetector", DummyObj)
    monkeypatch.setattr(main_module, "PlayerTracker", DummyObj)
    monkeypatch.setattr(main_module, "BallTracker", DummyObj)
    monkeypatch.setattr(main_module, "TeamClassifier", DummyObj)
    monkeypatch.setattr(main_module, "PerspectiveTransformer", DummyPerspective)
    monkeypatch.setattr(main_module, "CameraMotionTracker", DummyObj)
    monkeypatch.setattr(main_module, "DataExporter", DummyExporter)
    dummy_video = tmp_path / "dummy_short.mp4"
    dummy_video.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["main.py", "--input", str(dummy_video), "--no-video", "--no-analytics", "--export-dir", str(tmp_path)])

    main_module.main()
    stdout = capsys.readouterr().out
    assert "ROC-AUC" not in stdout
    assert "xP hesaplanamadı" in stdout


def test_t1_allow_synthetic_true_runs_old_behavior_with_warning(tmp_path, capsys):
    ml = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42, allow_synthetic=True)
    out_dir = tmp_path / "t1_syn"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)

    assert summary["xp_modeling"]["status"] == "ok"
    assert summary["xp_modeling"]["data_source"] == "calibrated_synthetic_match"
    assert (out_dir / "xp_model_comparison.png").exists()
    assert (out_dir / "xt_spatial_heatmap.png").exists()

    stdout = capsys.readouterr().out
    assert "SENTETİK VERİ – GERÇEK MAÇ SONUCU DEĞİLDİR" in stdout
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "SENTETİK VERİ – GERÇEK MAÇ SONUCU DEĞİLDİR" in md_text


# ----------------------------------------------------------------------
# T2 KABUL TESTLERİ: Kalibrasyonsuz Modda Mekânsal Metriklerin Kapatılması
# ----------------------------------------------------------------------
def _make_uncalibrated_match_df() -> pd.DataFrame:
    """pitch_x / pitch_y tamamen boş ama possession ve pas üretilebilen takip DataFrame'i."""
    rows = []
    for f in range(40):
        ts = round(f / 25.0, 3)
        if f < 6:
            bx = 100.0
        elif f < 29:
            bx = 100.0 + ((f - 5) / 24.0) * 180.0  # ~7.5 px/frame = ~187 px/s (< max_pass_speed)
        else:
            bx = 280.0
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
            "bbox_x1": bx, "bbox_y1": 300.0, "bbox_x2": bx + 10.0, "bbox_y2": 310.0,
            "pixel_x": bx, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 1, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 90.0, "bbox_y1": 280.0, "bbox_x2": 120.0, "bbox_y2": 350.0,
            "pixel_x": 100.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 2, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 270.0, "bbox_y1": 280.0, "bbox_x2": 300.0, "bbox_y2": 350.0,
            "pixel_x": 280.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 3, "class_name": "player", "team": "Team B", "is_predicted": False,
            "bbox_x1": 700.0, "bbox_y1": 280.0, "bbox_x2": 730.0, "bbox_y2": 350.0,
            "pixel_x": 715.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
    return pd.DataFrame(rows)


def test_t2_match_analyzer_uncalibrated_keeps_possession_and_passes_gates_spatial(tmp_path, capsys):
    from modules.match_analytics import MatchAnalyzer

    df = _make_uncalibrated_match_df()
    ma = MatchAnalyzer(df, fps=25.0)
    assert ma.is_calibrated is False

    out_dir = tmp_path / "t2_ma"
    summary = ma.export_all(output_dir=str(out_dir), generate_heatmaps=True)

    assert summary["possession"]["controlled_frames"] > 0
    assert summary["possession"]["team_a_pct"] > 0.0
    assert summary["passes"]["total_completed_passes"] == 1
    assert summary["passes"]["events"][0]["distance"] is None

    assert summary["team_shape"] == {"status": "requires_calibration"}
    assert summary["heatmaps"] == {"status": "requires_calibration"}
    assert not (out_dir / "heatmaps").exists()
    assert list(out_dir.rglob("*.png")) == []

    stdout = capsys.readouterr().out
    notice = "Saha kalibrasyonu olmadığı için konum tabanlı metrikler üretilmedi."
    assert stdout.count(notice) == 1


def test_t2_ml_analyzer_uncalibrated_returns_requires_calibration_no_plots(tmp_path, capsys):
    df = _make_uncalibrated_match_df()
    ml = MLAnalyzer(df, fps=25.0, random_state=42)
    assert ml.is_calibrated is False

    out_dir = tmp_path / "t2_ml"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)

    assert summary["xp_modeling"] == {"status": "requires_calibration"}
    assert summary["xt_analysis"] == {"status": "requires_calibration"}
    assert summary["player_clustering"] == {"status": "requires_calibration"}
    assert summary["tempo_and_anomalies"] == {"status": "requires_calibration"}

    assert list(out_dir.glob("*.png")) == []
    assert not (out_dir / config.EXPORT_ML_PLAYER_PROFILES_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_TEMPO_WINDOWS_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_ANOMALIES_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_PASS_PREDICTIONS_FILENAME).exists()

    notice = "Saha kalibrasyonu olmadığı için konum tabanlı metrikler üretilmedi."
    stdout = capsys.readouterr().out
    assert stdout.count(notice) == 1
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert md_text.count(notice) == 1


def test_t2_calibrated_data_behavior_unchanged(tmp_path):
    from modules.match_analytics import MatchAnalyzer

    df = _make_few_proxy_passes_df(n_events=8)
    ma = MatchAnalyzer(df, fps=25.0)
    assert ma.is_calibrated is True
    ma_out = tmp_path / "t2_cal_ma"
    ma_sum = ma.export_all(output_dir=str(ma_out), generate_heatmaps=True)
    assert ma_sum["team_shape"]["status"] == "ok"
    assert isinstance(ma_sum["heatmaps"], dict) and "team_a" in ma_sum["heatmaps"]
    assert len(list((ma_out / "heatmaps").glob("*.png"))) >= 2

    ml = MLAnalyzer(df, fps=25.0, random_state=42)
    assert ml.is_calibrated is True
    ml_out = tmp_path / "t2_cal_ml"
    ml_sum = ml.export_all(output_dir=str(ml_out), generate_plots=True)
    assert ml_sum["player_clustering"]["status"] == "ok"
    assert ml_sum["tempo_and_anomalies"]["status"] == "ok"
    assert (ml_out / "player_clusters_pca.png").exists()
    assert (ml_out / "tempo_anomaly_timeline.png").exists()


# ----------------------------------------------------------------------
# T4 KABUL TESTLERİ: Kaleci Takımını Çoğunluk Oyuyla Kilitleme & Backfill
# ----------------------------------------------------------------------
def test_t4_goalkeeper_team_locks_after_majority_vote_under_noisy_x(monkeypatch):
    from modules.team_classifier import TeamClassifier

    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    frame = np.full((600, 1000, 3), (35, 140, 35), dtype=np.uint8)

    def _paint(x1, y1, x2, y2, bgr):
        bw, bh = x2 - x1, y2 - y1
        frame[y1 + int(bh * 0.18):y1 + int(bh * 0.52), x1 + int(bw * 0.25):x1 + int(bw * 0.75)] = bgr

    outfield_specs = []
    for i in range(5):
        bx1, by1, bx2, by2 = 140 + i * 45, 120, 180 + i * 45, 250
        _paint(bx1, by1, bx2, by2, (30, 30, 210))
        outfield_specs.append({"track_id": i + 1, "class_name": "player", "bbox": [bx1, by1, bx2, by2]})
    for i in range(5):
        bx1, by1, bx2, by2 = 600 + i * 45, 320, 640 + i * 45, 450
        _paint(bx1, by1, bx2, by2, (210, 40, 30))
        outfield_specs.append({"track_id": i + 6, "class_name": "player", "bbox": [bx1, by1, bx2, by2]})
    _paint(40, 220, 85, 350, (40, 210, 40))

    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=3, vote_window=7)
    # İlk kareyi besleyip Team A / Team B etiketlerini öğren
    first_dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
    first_out = clf.assign_teams(first_dets, frame)
    team_a_lbl = first_out[0]["team"]
    team_b_lbl = first_out[5]["team"]
    assert team_a_lbl != team_b_lbl

    # Gürültülü dizi: ağırlıklı A tarafı (40, 910, 45, 50) -> A olarak kilitlenmeli
    gk_xs = [40, 910, 45, 50]
    assigned_seq = []
    for x in gk_xs:
        dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
        dets.append({"track_id": 101, "class_name": "goalkeeper", "bbox": [x, 220, x + 45, 350], "team": None})
        out = clf.assign_teams(dets, frame)
        assigned_seq.append(next(d["team"] for d in out if d["track_id"] == 101))

    assert 101 in clf._locked_gk_tracks
    assert assigned_seq[-1] == team_a_lbl

    # Kilit sonrası kaleci B tarafına geçse bile takım ASLA değişmemeli
    for _ in range(15):
        dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
        dets.append({"track_id": 101, "class_name": "goalkeeper", "bbox": [910, 220, 955, 350], "team": None})
        out = clf.assign_teams(dets, frame)
        gk_team = next(d["team"] for d in out if d["track_id"] == 101)
        assert gk_team == team_a_lbl
        assert gk_team != team_b_lbl


def test_t4_postprocess_backfills_goalkeeper_placeholder_rows_in_mem_and_stream(tmp_path):
    from modules.data_exporter import DataExporter

    for mode, kwargs in (
        ("mem", {"flush_interval": 0}),
        (
            "stream",
            {
                "flush_interval": 3,
                "csv_path": str(tmp_path / "gk_stream.csv"),
                "sql_path": str(tmp_path / "gk_stream.db"),
                "enable_csv": True,
                "enable_sql": True,
            },
        ),
    ):
        exp = DataExporter(fps=25.0, **kwargs)
        with exp:
            for f in range(12):
                team_val = None if f < 2 else ("Goalkeeper" if f < 7 else "Team A")
                exp.append_frame_data(
                    f,
                    [
                        {
                            "track_id": 101,
                            "class_name": "goalkeeper",
                            "team": team_val,
                            "confidence": 0.9,
                            "is_predicted": False,
                            "bbox": [40.0, 220.0, 85.0, 350.0],
                            "pitch_x": 5.0,
                            "pitch_y": 34.0,
                            "on_pitch": True,
                        }
                    ],
                )
            exp.postprocess(team_map={101: "Team A"})
            df = exp.to_dataframe()
            assert (df["team"] == "Team A").all(), f"{mode} modunda doldurulmamış GK satırı kaldı"
            assert "Goalkeeper" not in set(df["team"].astype(str))
            assert "Team B" not in set(df["team"].astype(str))
            if mode == "stream":
                csv_df = pd.read_csv(kwargs["csv_path"])
                assert (csv_df["team"] == "Team A").all()


# ----------------------------------------------------------------------
# T8 KABUL TESTİ: K-Means Refit Kilitli Takımı Oylamasız Değiştirmemeli
# ----------------------------------------------------------------------
def test_t8_kmeans_refit_votes_instead_of_overwriting_locked_team(monkeypatch):
    from collections import deque
    from modules.team_classifier import TeamClassifier

    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    frame = np.full((600, 1000, 3), (35, 140, 35), dtype=np.uint8)

    def _paint(x1, y1, x2, y2, bgr):
        bw, bh = x2 - x1, y2 - y1
        frame[y1 + int(bh * 0.18):y1 + int(bh * 0.52), x1 + int(bw * 0.25):x1 + int(bw * 0.75)] = bgr

    specs = []
    for i in range(5):
        bx1, by1, bx2, by2 = 40 + i * 80, 100, 96 + i * 80, 240
        _paint(bx1, by1, bx2, by2, (30, 30, 220))
        specs.append({"track_id": i + 1, "class_name": "player", "bbox": [bx1, by1, bx2, by2]})
    for i in range(5):
        bx1, by1, bx2, by2 = 520 + i * 80, 300, 576 + i * 80, 440
        _paint(bx1, by1, bx2, by2, (245, 245, 245))
        specs.append({"track_id": i + 6, "class_name": "player", "bbox": [bx1, by1, bx2, by2]})

    clf = TeamClassifier(
        n_clusters=2,
        min_unique_tracks=10,
        min_samples_for_lock=3,
        vote_window=7,
        max_refits=3,
    )
    clf.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in specs],
        frame,
    )
    assert clf.is_fitted is True
    tid = 1
    orig_team = clf.team_assignment[tid]
    other_team = "Team B" if orig_team == "Team A" else "Team A"
    # Kilitli track'te başlangıçta 2 uyumlu oy olsun (refit sonrası [orig, orig, other] -> henüz %60 eşiğine ulaşmaz)
    clf._vote_history[tid] = deque([orig_team, orig_team], maxlen=clf.vote_window)

    # Tek bir refit track 1'i karşı küme merkezine yaklaştırsa bile hemen takımı değişmemeli
    feat_other = clf._track_feature_bank[6].copy()
    for extra_id in range(20, 26):
        clf._track_feature_bank[extra_id] = clf._track_feature_bank[6].copy()
    clf._track_feature_bank[tid] = feat_other
    clf._samples_since_fit = 10

    clf._maybe_refit_kmeans()
    assert clf._refit_count == 1
    assert clf.team_assignment[tid] == orig_team, "Tek refit oylamasız şekilde kilitli takımı değiştirdi!"

    # Ancak oylama penceresinde min_samples_for_lock ve %60 çoğunluk sağlanınca (2 yeni oyla toplam 3/5) değişmeli
    for _ in range(clf.min_samples_for_lock):
        clf._record_vote_and_maybe_update(tid, other_team)
    assert clf.team_assignment[tid] == other_team


# ----------------------------------------------------------------------
# T5 KABUL TESTLERİ: Model İndirme Betiği ve Taşınabilir Pytest İşaretleri
# ----------------------------------------------------------------------
def test_t5_download_models_empty_url_sha256_and_marker_skip(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm
    import tests.conftest as root_conftest

    # 1) URL boşken anlamlı Türkçe mesajla çıkış kodu 1 dönmeli
    monkeypatch.setattr(config, "MODEL_DOWNLOAD_URL", "")
    rc_empty = dm.main(["--output", str(tmp_path / "m.pt")])
    assert rc_empty == 1
    err_out = capsys.readouterr().err
    assert "MODEL_DOWNLOAD_URL" in err_out or "boş" in err_out

    # 2) Mock URL indirme + SHA-256 doğrulama (başarılı & uyuşmazlık)
    payload = b"FAKE_YOLO_WEIGHTS_BYTES"
    good_sha = hashlib.sha256(payload).hexdigest()

    class FakeResp:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def read(self, n: int = -1):
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos:] if n < 0 else self._data[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))
    out_pt = tmp_path / "downloaded.pt"
    saved = dm.download_model(url="https://example.com/model.pt", output_path=out_pt, expected_sha256=good_sha)
    assert saved.is_file() and saved.read_bytes() == payload

    with pytest.raises(ValueError, match="SHA-256"):
        dm.download_model(
            url="https://example.com/model.pt",
            output_path=tmp_path / "bad.pt",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "bad.pt").exists()

    # 3) Model yokken requires_model işareti beklenen mesajla atlamalı
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", str(tmp_path / "nonexistent.pt"))
    fake_item = SimpleNamespace(
        get_closest_marker=lambda name: object() if name == "requires_model" else None
    )
    with pytest.raises(pytest.skip.Exception) as exc_info:
        root_conftest.pytest_runtest_setup(fake_item)
    assert "models/football_yolo11s_best.pt" in str(exc_info.value)
    assert "Model ağırlıkları" in str(exc_info.value)
