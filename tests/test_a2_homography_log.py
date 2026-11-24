"""
test_a2_homography_log.py
-------------------------
Faz A2-2: Kare bazında homografi kaydı (homography_log.csv).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as main_module
from config import config
from modules.pitch_calibrator import CalibrationResult, HomographyLogWriter
from test_a2_jump_gating import (
    FRAME_H,
    FRAME_W,
    H_TRUE,
    FakeFlow,
    NoisyKeypointCalibrator,
    _make_hybrid,
)


def test_a2_2_log_has_one_row_per_frame_and_expected_sources(tmp_path):
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=0)
    hybrid, pt = _make_hybrid(cal, FakeFlow(), refresh=15)
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    log_path = tmp_path / "homography_log.csv"
    with HomographyLogWriter(log_path, flush_every=7) as log:
        for f in range(50):
            m = hybrid.process_frame(f, frame)
            log.write(f, hybrid.frame_info, pt.homography_matrix @ m)

    df = pd.read_csv(log_path)
    assert list(df.columns) == list(HomographyLogWriter.COLUMNS)
    assert len(df) == 50
    assert df["frame_idx"].tolist() == list(range(50))
    src = df["source"].tolist()
    assert src[0] == "keypoint_lock"
    assert set(src[1:15]) == {"flow"}
    assert src[15] == "keypoint_refresh"
    assert set(src[16:30]) == {"flow"}
    assert src[30] == "keypoint_refresh"
    assert set(df["profile"]) == {"broadcast"}
    # Model çalışan karelerde nokta sayısı ve fark dolu; flow karelerinde fark boş
    assert df.loc[15, "num_keypoints"] > 4
    assert df.loc[15, "disagreement_m"] < config.PITCH_CAL_DEADBAND_M
    assert pd.isna(df.loc[3, "disagreement_m"])
    # Kaydedilen tablo gerçek tabloyla aynı noktayı göstermeli
    h0 = df.loc[20, [f"h{r}{c}" for r in range(3) for c in range(3)]].to_numpy(float).reshape(3, 3)
    p = np.array([640.0, 520.0, 1.0])
    a, b = h0 @ p, H_TRUE @ p
    assert np.allclose(a[:2] / a[2], b[:2] / b[2], atol=0.02)  # CSV 6 hane yuvarlama


def test_a2_2_log_writer_empty_matrix_row(tmp_path):
    path = tmp_path / "h.csv"
    with HomographyLogWriter(path) as log:
        log.write(0, {"source": "none"}, None)
    df = pd.read_csv(path)
    assert df.loc[0, "source"] == "none"
    assert pd.isna(df.loc[0, "h00"])


# ----------------------------------------------------------------------
# main.py entegrasyonu
# ----------------------------------------------------------------------
def _patch_pipeline(monkeypatch, n_frames=6):
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(n_frames)]
    H_px_to_m = np.linalg.inv(
        np.array([[5.0, 0.4, 80.0], [0.0, 3.0, 70.0], [0.0, 0.002, 1.0]])
    )

    class DummyVP:
        def __init__(self, inp, out=None):
            self.fps = 25.0
            self.has_writer = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read_frames(self):
            yield from enumerate(frames)

    class DummyDetector:
        device = "cpu"

        def detect_frame(self, frame):
            return []

    class DummyPlayerTracker:
        def __init__(self, *a, **k):
            pass

        def update(self, raw, frame):
            return [{"track_id": 1, "bbox": [260, 150, 300, 230], "class_name": "player"}]

    class DummyBallTracker:
        def update(self, raw, frame_shape=None):
            return None

    class DummyTeamClassifier:
        def assign_teams(self, players, frame):
            for p in players:
                p["team"] = "Team A"
            return players

        def get_team_map(self):
            return {1: "Team A"}

    class DummyExporter:
        def __init__(self, *a, **k):
            pass

        def append_frame_data(self, idx, dets):
            pass

        def postprocess(self, team_map=None):
            return {}

        def export_to_csv(self):
            return None

        def export_to_sql(self):
            return None

        def export_distance_summary(self):
            return None

        def close(self):
            pass

    class StaticFlow:
        scene_cut_detected = False
        recovered_this_frame = False

        def update(self, frame, dets=None):
            return np.eye(3)

        def reanchor(self, frame, dets=None, keep_keyframe_bank=True, prev_cumulative=None):
            return np.eye(3)

    class AutoCal:
        is_available = True

        def __init__(self, *a, **k):
            pass

        def calibrate_frame(self, frame):
            return CalibrationResult(success=True, homography_matrix=H_px_to_m, num_keypoints=12)

    for name, obj in (
        ("VideoProcessor", DummyVP),
        ("ObjectDetector", DummyDetector),
        ("PlayerTracker", DummyPlayerTracker),
        ("BallTracker", DummyBallTracker),
        ("TeamClassifier", DummyTeamClassifier),
        ("DataExporter", DummyExporter),
        ("CameraMotionTracker", StaticFlow),
        ("AutoPitchCalibrator", AutoCal),
    ):
        monkeypatch.setattr(main_module, name, obj)
    monkeypatch.setattr(config, "SOURCE_CALIBRATION_POINTS", {})


def _run_main(monkeypatch, tmp_path, out_name, extra=()):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    out = tmp_path / out_name
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--input", str(video), "--no-video", "--no-analytics", "--no-ml",
         "--export-dir", str(out), *extra],
    )
    main_module.main()
    return out


def test_a2_2_main_writes_log_and_summary_independent_of_no_csv(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=6)
    out = _run_main(monkeypatch, tmp_path, "with_log", extra=("--no-csv",))
    df = pd.read_csv(out / config.EXPORT_HOMOGRAPHY_LOG_FILENAME)
    assert len(df) == 6
    assert df.loc[0, "source"] == "keypoint_lock"
    assert df["h00"].notna().all()
    summary = json.loads((out / config.EXPORT_CALIBRATION_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["profile"] in ("fixed", "broadcast")
    assert "stats" in summary


def test_a2_2_main_no_homography_log_flag(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3)
    out = _run_main(monkeypatch, tmp_path, "no_log", extra=("--no-homography-log",))
    assert not (out / config.EXPORT_HOMOGRAPHY_LOG_FILENAME).exists()
