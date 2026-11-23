import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


# ----------------------------------------------------------------------
# R4 KABUL TESTLERİ: Model İndirmede Zorunlu SHA-256 Kontrolü
# ----------------------------------------------------------------------
def test_r4_download_models_rejects_missing_sha256(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    called = {"urlopen": 0}

    def _fail_urlopen(*args, **kwargs):
        called["urlopen"] += 1
        raise AssertionError("SHA-256 olmadan indirme denenmemeliydi!")

    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", None)
    monkeypatch.setattr(dm.urllib.request, "urlopen", _fail_urlopen)

    out_pt = tmp_path / "missing_sha.pt"
    rc = dm.main(["--url", "https://example.com/model.pt", "--output", str(out_pt)])
    assert rc != 0
    assert called["urlopen"] == 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    err_txt = capsys.readouterr().err
    assert "SHA-256" in err_txt


def test_r4_download_models_wrong_sha_removes_part_and_no_target(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    payload = b"FAKE_WEIGHT_BYTES_FOR_SHA_CHECK"

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
    out_pt = tmp_path / "wrong_sha.pt"
    rc = dm.main([
        "--url", "https://example.com/model.pt",
        "--output", str(out_pt),
        "--sha256", "0" * 64,
    ])
    assert rc != 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    err_txt = capsys.readouterr().err
    assert "SHA-256" in err_txt


def test_r4_download_models_insecure_skip_sha256_warns_and_saves(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    payload = b"BYPASS_VERIFIED_BYTES"

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

    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", None)
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))

    out_pt = tmp_path / "insecure_ok.pt"
    rc = dm.main([
        "--url", "https://example.com/model.pt",
        "--output", str(out_pt),
        "--insecure-skip-sha256",
    ])
    assert rc == 0
    assert out_pt.is_file() and out_pt.read_bytes() == payload
    out_txt = capsys.readouterr().out
    assert "UYARI" in out_txt
    assert "SHA-256" in out_txt


# ----------------------------------------------------------------------
# R5 KABUL TESTİ: Geçersiz Video Yolunda Model Yüklenmeden Erken Hata
# ----------------------------------------------------------------------
def test_r5_invalid_input_exits_before_loading_object_detector(tmp_path, monkeypatch, capsys):
    import main as main_mod

    missing_video = tmp_path / "does_not_exist.mp4"
    export_dir = tmp_path / "should_not_create"
    detector_calls = {"count": 0}

    def _forbidden_detector(*args, **kwargs):
        detector_calls["count"] += 1
        raise AssertionError("Geçersiz video yolunda ObjectDetector başlatılmamalıydı!")

    monkeypatch.setattr(main_mod, "ObjectDetector", _forbidden_detector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(missing_video),
            "--export-dir",
            str(export_dir),
            "--no-video",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code != 0
    assert detector_calls["count"] == 0
    assert not export_dir.exists()
    err_txt = capsys.readouterr().err
    assert "bulunamadı" in err_txt


# ----------------------------------------------------------------------
# R7 KABUL TESTLERİ: Tahmini (is_predicted=True) Hayalet Oyuncuların Dışlanması
# ----------------------------------------------------------------------
def test_r7_predicted_ghost_player_excluded_when_real_player_present():
    from modules.match_analytics import MatchAnalyzer
    from modules.ml_analyzer import MLAnalyzer

    rows = []
    # 0..11 kareler: top (50, 34), hayalet oyuncu ID:99 1m (51, 34), gerçek ID:10 2m (52, 34)
    # 12..17 kareler: top uçuşta (52 -> 60m, ~33 m/sn, fiziksel olarak geçerli pas)
    # 18..29 kareler: top (60, 34), hayalet ID:99 1m (61, 34), gerçek ID:11 2m (62, 34)
    for f in range(30):
        ts = round(f / 25.0, 3)
        if f < 12:
            bx, by = 50.0, 34.0
        elif f < 18:
            alpha = (f - 11) / 7.0
            bx, by = 50.0 + alpha * 10.0, 34.0
        else:
            bx, by = 60.0, 34.0

        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
            "team": None, "is_predicted": False,
            "bbox_x1": 495.0, "bbox_y1": 335.0, "bbox_x2": 505.0, "bbox_y2": 345.0,
            "pixel_x": bx * 10.0, "pixel_y": by * 10.0, "pitch_x": bx, "pitch_y": by, "on_pitch": True,
        })
        # Hayalet oyuncu 1m mesafede
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 99, "class_name": "player",
            "team": "Team B", "is_predicted": True,
            "bbox_x1": 500.0, "bbox_y1": 300.0, "bbox_x2": 530.0, "bbox_y2": 370.0,
            "pixel_x": (bx + 1.0) * 10.0, "pixel_y": by * 10.0, "pitch_x": bx + 1.0, "pitch_y": by, "on_pitch": True,
        })
        # Gerçek oyuncular 2m mesafede
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 10, "class_name": "player",
            "team": "Team A", "is_predicted": False,
            "bbox_x1": 510.0, "bbox_y1": 300.0, "bbox_x2": 540.0, "bbox_y2": 370.0,
            "pixel_x": 520.0, "pixel_y": 340.0, "pitch_x": 52.0, "pitch_y": 34.0, "on_pitch": True,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 11, "class_name": "player",
            "team": "Team A", "is_predicted": False,
            "bbox_x1": 610.0, "bbox_y1": 300.0, "bbox_x2": 640.0, "bbox_y2": 370.0,
            "pixel_x": 620.0, "pixel_y": 340.0, "pitch_x": 62.0, "pitch_y": 34.0, "on_pitch": True,
        })

    df = pd.DataFrame(rows)
    ma = MatchAnalyzer(df, fps=25.0)
    tl_df, stats = ma.compute_possession()

    first_half_tids = set(tl_df.loc[tl_df["frame_idx"] < 12, "raw_track_id"].dropna().astype(str))
    assert "10" in first_half_tids
    assert "99" not in set(tl_df["raw_track_id"].dropna().astype(str))
    assert "99" not in set(tl_df["smoothed_track_id"].dropna().astype(str))
    assert stats["team_b_frames"] == 0
    assert stats["team_a_pct"] == 100.0

    ml = MLAnalyzer(df, fps=25.0)
    proxy_df = ml.extract_proxy_pass_events()
    assert not proxy_df.empty
    assert (proxy_df["from_track_id"].astype(str) == "10").any()
    assert "99" not in set(proxy_df["from_track_id"].astype(str))
    assert "99" not in set(proxy_df["to_track_id"].astype(str))
    # Hayalet Team B oyuncusu baskı olarak da sayılmamalı
    assert (proxy_df["opponents_nearby"] == 0).all()


def test_r7_only_predicted_player_nearby_yields_no_possession():
    from modules.match_analytics import MatchAnalyzer
    from modules.ml_analyzer import MLAnalyzer

    rows = []
    for f in range(15):
        ts = round(f / 25.0, 3)
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
            "team": None, "is_predicted": False,
            "bbox_x1": 495.0, "bbox_y1": 335.0, "bbox_x2": 505.0, "bbox_y2": 345.0,
            "pixel_x": 500.0, "pixel_y": 340.0, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 99, "class_name": "player",
            "team": "Team A", "is_predicted": True,
            "bbox_x1": 500.0, "bbox_y1": 300.0, "bbox_x2": 530.0, "bbox_y2": 370.0,
            "pixel_x": 505.0, "pixel_y": 340.0, "pitch_x": 50.5, "pitch_y": 34.0, "on_pitch": True,
        })

    df = pd.DataFrame(rows)
    ma = MatchAnalyzer(df, fps=25.0)
    tl_df, stats = ma.compute_possession()
    assert stats["controlled_frames"] == 0
    assert tl_df["raw_track_id"].isna().all()
    assert tl_df["smoothed_track_id"].isna().all()

    proxy_df = MLAnalyzer(df, fps=25.0).extract_proxy_pass_events()
    assert proxy_df.empty


def test_r7_draw_detections_distinguishes_predicted_and_real_boxes():
    from modules.video_processor import VideoProcessor

    vp = object.__new__(VideoProcessor)
    base_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    real_det = [{"bbox": [60, 60, 140, 180], "track_id": 7, "class_name": "player", "team": "Team A", "is_predicted": False}]
    pred_det = [{"bbox": [60, 60, 140, 180], "track_id": 7, "class_name": "player", "team": "Team A", "is_predicted": True}]

    img_real = vp.draw_detections(base_frame, real_det, inplace=False)
    img_pred = vp.draw_detections(base_frame, pred_det, inplace=False)

    assert not np.array_equal(img_real, img_pred)
    # Gerçek kutunun alt kenarı kesintisiz, tahmini kutunun alt kenarı kesikli olmalı
    bottom_real = img_real[180, 70:130].sum(axis=1) > 0
    bottom_pred = img_pred[180, 70:130].sum(axis=1) > 0
    assert int(bottom_real.sum()) == 60
    assert 0 < int(bottom_pred.sum()) < 60


# ----------------------------------------------------------------------
# R3 KABUL TESTLERİ: Kaleci Kilidi İçin Daha Güçlü Eşik (25 oy, %80 uyum, pencere 40)
# ----------------------------------------------------------------------
def test_r3_goalkeeper_lock_requires_25_votes_and_80_pct_agreement(monkeypatch):
    from modules.team_classifier import TeamClassifier

    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    assert config.TEAM_CLASSIFIER_GK_LOCK_MIN_VOTES == 25
    assert config.TEAM_CLASSIFIER_GK_LOCK_MIN_AGREEMENT == pytest.approx(0.8)
    assert config.TEAM_CLASSIFIER_GK_VOTE_WINDOW >= 25

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

    # 1) 24 tutarlı oy -> kilit yok; 25. oyda kilit var
    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=3, vote_window=7)
    init_dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
    first_out = clf.assign_teams(init_dets, frame)
    team_a_lbl = first_out[0]["team"]

    for _ in range(24):
        dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
        dets.append({"track_id": 201, "class_name": "goalkeeper", "bbox": [40, 220, 85, 350], "team": None})
        out = clf.assign_teams(dets, frame)
        gk_team = next(d["team"] for d in out if d["track_id"] == 201)
        assert gk_team == "Goalkeeper"
        assert 201 not in clf._locked_gk_tracks

    dets25 = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
    dets25.append({"track_id": 201, "class_name": "goalkeeper", "bbox": [40, 220, 85, 350], "team": None})
    out25 = clf.assign_teams(dets25, frame)
    assert next(d["team"] for d in out25 if d["track_id"] == 201) == team_a_lbl
    assert 201 in clf._locked_gk_tracks

    # 2) 30 oyun %70'i (21 A, 9 B) aynı takım -> %80 eşiğini geçemediği için kilit olmamalı
    clf2 = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=3, vote_window=7)
    clf2.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs],
        frame,
    )
    seq_70pct = [40] * 16 + [910] * 9 + [40] * 5  # ilk 24 oy: 16A/8B; 25..30 arası en çok 21/30 (%70) A
    for x in seq_70pct:
        dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in outfield_specs]
        dets.append({"track_id": 202, "class_name": "goalkeeper", "bbox": [x, 220, x + 45, 350], "team": None})
        out_mix = clf2.assign_teams(dets, frame)
        assert 202 not in clf2._locked_gk_tracks
        assert next(d["team"] for d in out_mix if d["track_id"] == 202) == "Goalkeeper"
