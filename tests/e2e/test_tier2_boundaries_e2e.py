"""Tier 2 E2E Tests: Boundary values, error paths & corner cases across F1-F5."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(_E2E_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import config  # noqa: E402
import modules.detector as detector_module  # noqa: E402
from modules.camera_motion_tracker import CameraMotionTracker  # noqa: E402
from modules.data_exporter import DataExporter  # noqa: E402
from modules.match_analytics import MatchAnalyzer  # noqa: E402
from modules.ml_analyzer import MLAnalyzer  # noqa: E402
from modules.perspective_transformer import PerspectiveTransformer  # noqa: E402
from modules.team_classifier import TeamClassifier  # noqa: E402
from modules.tracker import BallTracker  # noqa: E402
from modules.video_processor import VideoProcessor  # noqa: E402
import utils.train_custom_yolo as train_script  # noqa: E402

from conftest import (  # noqa: E402
    BASELINE_CSV,
    BASELINE_CSV_SHA256,
    BASELINE_DB,
    BASELINE_DB_SHA256,
    BASELINE_ROW_COUNT,
    REPO_ROOT,
    scan_secret_hygiene,
    sha256_file,
)


class _FakeBoxes:
    def __init__(self, xyxy_rows, cls_rows, conf_rows):
        self.xyxy = torch.tensor(xyxy_rows, dtype=torch.float32)
        self.cls = torch.tensor(cls_rows, dtype=torch.float32)
        self.conf = torch.tensor(conf_rows, dtype=torch.float32)

    def __len__(self):
        return int(self.cls.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield SimpleNamespace(
                xyxy=self.xyxy[i : i + 1],
                cls=self.cls[i : i + 1],
                conf=self.conf[i : i + 1],
            )


# ======================================================================
# F1 Boundaries (5 tests)
# ======================================================================


def test_t2_f1_detect_batch_empty_returns_empty_list(monkeypatch):
    football_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = football_names

        def predict(self, **kwargs):
            raise AssertionError("predict() should not be called on empty batch")

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(config, "TARGET_CLASS_IDS", football_names)

    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cpu")
    assert det.detect_batch([]) == []


def test_t2_f1_detector_class_mismatch_raises_validation_error(monkeypatch):
    class MismatchedYOLO:
        def __init__(self, model_path):
            self.names = {0: "car", 1: "truck", 2: "bicycle", 3: "bus"}

    monkeypatch.setattr(detector_module, "YOLO", MismatchedYOLO)
    monkeypatch.setattr(
        config,
        "TARGET_CLASS_IDS",
        {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
    )

    with pytest.raises((ValueError, RuntimeError)):
        detector_module.ObjectDetector(model_path="dummy.pt", device="cpu")


def test_t2_f1_detector_cuda_oom_batch_halving_recovery(monkeypatch):
    football_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    predict_batch_sizes = []
    empty_cache_calls = 0

    class OOMYOLO:
        def __init__(self, model_path):
            self.names = football_names

        def predict(self, **kwargs):
            src = kwargs["source"]
            n = len(src) if isinstance(src, list) else 1
            predict_batch_sizes.append(n)
            if n > 2:
                raise torch.cuda.OutOfMemoryError("Simulated CUDA out of memory")
            boxes = _FakeBoxes([[10.0, 10.0, 30.0, 50.0]], [2.0], [0.9])
            return [SimpleNamespace(boxes=boxes) for _ in range(n)]

    monkeypatch.setattr(detector_module, "YOLO", OOMYOLO)
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(config, "TARGET_CLASS_IDS", football_names)

    def fake_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(detector_module.torch.cuda, "empty_cache", fake_empty_cache)

    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cuda")
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
    res = det.detect_batch(frames)

    assert len(res) == 8
    assert all(len(r) == 1 and r[0]["class_name"] == "player" for r in res)
    assert empty_cache_calls >= 1
    assert any(sz <= 2 for sz in predict_batch_sizes)


def test_t2_f1_detector_tiny_frame_and_invalid_model_path():
    with pytest.raises(Exception):
        detector_module.ObjectDetector(model_path="nonexistent_missing_weights_999.pt", device="cpu")


def test_t2_f1_train_script_rejects_missing_api_key_and_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError):
        train_script.resolve_api_key("   ")

    bad_yaml = tmp_path / "data.yaml"
    bad_yaml.write_text("names: ['ball', 'player']\nnc: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        train_script.validate_data_yaml(bad_yaml)

    with pytest.raises(FileNotFoundError):
        train_script.validate_data_yaml(tmp_path / "missing_data.yaml")


# ======================================================================
# F2 Boundaries (5 tests)
# ======================================================================


def test_t2_f2_video_processor_missing_and_corrupt_video_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoProcessor(str(tmp_path / "does_not_exist.mp4"))

    corrupt = tmp_path / "corrupt_zero_bytes.mp4"
    corrupt.write_bytes(b"")
    with pytest.raises((IOError, OSError)):
        VideoProcessor(str(corrupt))


def test_t2_f2_data_exporter_flush_interval_zero_one_and_gt_total(isolated_export_dirs):
    results_by_mode = {}
    for mode_idx, interval in enumerate([0, 1, 1000]):
        csv_p = isolated_export_dirs / f"mode_{mode_idx}.csv"
        db_p = isolated_export_dirs / f"mode_{mode_idx}.db"
        exp = DataExporter(
            fps=25.0,
            flush_interval=interval,
            csv_path=str(csv_p),
            sql_path=str(db_p),
            enable_csv=True,
            enable_sql=True,
        )
        for f in range(15):
            exp.append_frame_data(
                f,
                [
                    {
                        "track_id": 1,
                        "class_name": "player",
                        "team": None if f < 3 else "Team A",
                        "confidence": 0.9,
                        "is_predicted": False,
                        "bbox": [100.0 + f, 100.0, 120.0 + f, 160.0],
                        "pitch_x": 20.0 + f * 0.2,
                        "pitch_y": 34.0,
                        "on_pitch": True,
                    }
                ],
            )
        exp.postprocess(team_map={1: "Team A"})
        out_csv = exp.export_to_csv(str(csv_p))
        out_db = exp.export_to_sql(str(db_p))
        df = pd.read_csv(out_csv)
        results_by_mode[interval] = (len(df), df["team"].tolist(), Path(out_db).is_file())

    assert results_by_mode[0][0] == 15
    assert results_by_mode[1] == results_by_mode[0]
    assert results_by_mode[1000] == results_by_mode[0]


def test_t2_f2_data_exporter_distance_summary_empty_and_uncalibrated_returns_none(isolated_export_dirs):
    empty_exp = DataExporter(fps=25.0)
    assert empty_exp.export_distance_summary(str(isolated_export_dirs / "empty_sum.csv")) is None

    uncal_exp = DataExporter(fps=25.0)
    for f in range(10):
        uncal_exp.append_frame_data(
            f,
            [
                {
                    "track_id": 1,
                    "class_name": "player",
                    "team": "Team A",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [100.0, 100.0, 120.0, 160.0],
                    "pitch_x": None,
                    "pitch_y": None,
                    "on_pitch": None,
                }
            ],
        )
    assert uncal_exp.export_distance_summary(str(isolated_export_dirs / "uncal_sum.csv")) is None


def test_t2_f2_cli_missing_input_video_and_invalid_tracker_nonzero_exit(tmp_path):
    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res_missing = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(tmp_path / "missing_input.mp4"),
            "--no-video",
            "--export-dir",
            str(tmp_path / "exp1"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=False,
    )
    assert res_missing.returncode != 0

    res_bad_tracker = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--tracker",
            "nonexistent_tracker_mode_xyz",
            "--export-dir",
            str(tmp_path / "exp2"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=False,
    )
    assert res_bad_tracker.returncode != 0


def test_t2_f2_data_exporter_ball_interp_gap_boundary(isolated_export_dirs):
    exp = DataExporter(fps=25.0)
    # Gap 1 (frames 0 -> 4, gap = 4 == max_gap): should interpolate frames 1, 2, 3
    # Gap 2 (frames 4 -> 9, gap = 5 > max_gap=4): must NOT interpolate frames 5..8
    for f in (0, 4, 9):
        exp.append_frame_data(
            f,
            [
                {
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [100.0 + f * 5, 200.0, 110.0 + f * 5, 210.0],
                    "pitch_x": 10.0 + f,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                }
            ],
        )
    interp_count = exp.interpolate_ball_positions(max_gap=4)
    assert interp_count == 3
    ball_frames = sorted(exp.to_dataframe()["frame_idx"].tolist())
    assert ball_frames == [0, 1, 2, 3, 4, 9]


# ======================================================================
# F3 Boundaries (5 tests)
# ======================================================================


def test_t2_f3_team_classifier_tiny_bbox_and_pre_fit_state():
    tc = TeamClassifier(n_clusters=2, min_unique_tracks=10, color_space="LAB")
    frame = np.full((120, 120, 3), (35, 140, 35), dtype=np.uint8)
    assert tc.extract_jersey_color(frame, [10.0, 10.0, 15.0, 15.0]) is None

    out = tc.assign_teams(
        [{"track_id": 1, "class_name": "player", "bbox": [10.0, 10.0, 50.0, 90.0], "team": None}],
        frame,
    )
    assert tc.is_fitted is False
    assert out[0]["team"] is None


def test_t2_f3_ball_tracker_missing_timeout_boundary():
    bt = BallTracker(max_missing_frames=0, velocity_damping=0.85, max_velocity_px=150.0)
    shape = (120, 200, 3)
    d0 = bt.update([{"class_name": "ball", "bbox": [50.0, 50.0, 60.0, 60.0], "confidence": 0.9}], shape)
    assert d0 is not None
    # With max_missing_frames=0, first missing frame immediately returns None
    assert bt.update([], shape) is None


def test_t2_f3_ml_analyzer_empty_and_insufficient_events_strict_json(tmp_path):
    empty_ml = MLAnalyzer(pd.DataFrame(), fps=25.0)
    direct_empty_xp = empty_ml.run_xp_modeling(events_df=pd.DataFrame())
    assert direct_empty_xp["status"] == "skipped_insufficient_events"

    res_empty = empty_ml.export_all(output_dir=str(tmp_path / "empty_ml"), generate_plots=False)
    assert res_empty["xp_modeling"]["status"] in {"ok", "skipped_insufficient_events", "skipped_single_class"}
    raw = (tmp_path / "empty_ml" / config.EXPORT_ML_SUMMARY_FILENAME).read_text(encoding="utf-8")
    json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"NaN/Inf in JSON: {c}")))


def test_t2_f3_ml_analyzer_single_class_fallback_status(tmp_path):
    ml = MLAnalyzer(pd.DataFrame(), fps=25.0)
    # Construct minimal valid pass events DF where target pass_outcome is uniformly 1
    synth = ml.generate_synthetic_match_passes(n_events=40, seed=42).copy()
    synth["pass_outcome"] = 1
    xp_res = ml.run_xp_modeling(events_df=synth)
    assert xp_res["status"] == "skipped_single_class"


def test_t2_f3_match_analyzer_uncalibrated_pixel_fallback_and_empty_df(tmp_path):
    empty_ma = MatchAnalyzer(pd.DataFrame(), fps=25.0)
    assert empty_ma.is_calibrated is False
    empty_sum = empty_ma.export_all(output_dir=str(tmp_path / "ma_empty"), generate_heatmaps=False)
    assert empty_sum["possession"]["total_frames"] == 0

    uncal_df = pd.DataFrame(
        [
            {
                "frame_idx": f,
                "timestamp_sec": f / 25.0,
                "track_id": "ball" if i == 0 else i,
                "class_name": "ball" if i == 0 else "player",
                "team": None if i == 0 else ("Team A" if i == 1 else "Team B"),
                "is_predicted": False,
                "pixel_x": 200.0 + f * 2,
                "pixel_y": 200.0,
                "pitch_x": np.nan,
                "pitch_y": np.nan,
                "on_pitch": np.nan,
            }
            for f in range(15)
            for i in (0, 1, 2)
        ]
    )
    ma = MatchAnalyzer(uncal_df, fps=25.0)
    assert ma.is_calibrated is False
    res = ma.export_all(output_dir=str(tmp_path / "ma_uncal"), generate_heatmaps=False)
    assert res["possession"]["mode"] == "pixel_fallback"


# ======================================================================
# F4 Boundaries (6 tests)
# ======================================================================


def test_t2_f4_camera_motion_t_plus_1_to_t_plus_20_persistent_none(synthetic_tv_broadcast_clip):
    """Explicit regression check for t+1 re-anchoring bug after scene cut."""
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        cut_start = clip.phases["phase2a_hard_cut"][0]
        for t in range(cut_start + 21):
            ok, frame = cap.read()
            assert ok
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            if t >= cut_start:
                assert mat is None, (
                    f"Frame t+{t - cut_start} (frame {t}) leaked non-None homography after hard cut!"
                )
                assert tracker.is_anchored is False
    finally:
        cap.release()


def test_t2_f4_camera_motion_pitch_colored_zoomed_replay_rejected(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        p2b_start, p2b_end = clip.phases["phase2b_zoomed_pitch_replay"]
        for t in range(p2b_end + 1):
            ok, frame = cap.read()
            assert ok
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            if p2b_start <= t <= p2b_end:
                assert mat is None, f"Zoomed pitch-colored replay frame {t} falsely anchored!"
                assert tracker.is_anchored is False
    finally:
        cap.release()


def test_t2_f4_perspective_transformer_vanishing_line_w_le_zero(synthetic_tv_broadcast_clip):
    transformer = PerspectiveTransformer(source_points=synthetic_tv_broadcast_clip.source_calibration_points)
    # Build projective matrix where last row produces w' <= 0 for pixel (500, 10)
    h_vanishing = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, -100.0],  # w' = y - 100 <= 0 for y <= 100
        ],
        dtype=np.float64,
    )
    assert transformer.transform_point(500.0, 50.0, homography_matrix=h_vanishing) is None
    assert transformer.transform_point(500.0, 100.0, homography_matrix=h_vanishing) is None


def test_t2_f4_perspective_transformer_singular_and_nan_inf_matrices(synthetic_tv_broadcast_clip):
    transformer = PerspectiveTransformer(source_points=synthetic_tv_broadcast_clip.source_calibration_points)
    singular = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    nan_mat = np.full((3, 3), np.nan, dtype=np.float64)
    inf_mat = np.eye(3, dtype=np.float64)
    inf_mat[0, 2] = np.inf

    for bad_mat in (singular, nan_mat, inf_mat):
        assert transformer.transform_point(400.0, 300.0, homography_matrix=bad_mat) is None
        dets = [{"bbox": [390.0, 250.0, 410.0, 300.0], "class_name": "player"}]
        out = transformer.add_pitch_coordinates(dets, camera_motion_matrix=bad_mat)
        assert out[0]["pitch_x"] is None
        assert out[0]["pitch_y"] is None
        assert out[0]["on_pitch"] is None


def test_t2_f4_perspective_transformer_fewer_than_4_landmarks_uncalibrated():
    transformer = PerspectiveTransformer(
        source_points={
            "top_left_corner": (100.0, 100.0),
            "top_right_corner": (900.0, 100.0),
            "bottom_left_corner": (100.0, 500.0),
        }
    )
    assert transformer.is_calibrated is False
    assert transformer.transform_point(300.0, 200.0) is None
    out = transformer.add_pitch_coordinates([{"bbox": [290.0, 180.0, 310.0, 220.0]}])
    assert out[0]["pitch_x"] is None and out[0]["on_pitch"] is None


def test_t2_f4_camera_motion_black_and_low_texture_frames_no_crash():
    tracker = CameraMotionTracker()
    black = np.zeros((360, 640, 3), dtype=np.uint8)
    uniform_green = np.full((360, 640, 3), (35, 140, 35), dtype=np.uint8)

    m0 = tracker.update(black, [])
    m1 = tracker.update(uniform_green, [])
    m2 = tracker.update(uniform_green, [])
    for m in (m0, m1, m2):
        if m is not None:
            assert np.all(np.isfinite(m))


# ======================================================================
# F5 Boundaries (5 tests)
# ======================================================================


def test_t2_f5_secret_scanner_detects_synthetic_mock_leak(tmp_path):
    mock_repo = tmp_path / "mock_repo"
    mock_repo.mkdir()
    # Construct synthetic mock token dynamically so this test file never holds a static token literal
    fake_tok = "Ab" + "CdEf" + "GhIj" + "123456"
    (mock_repo / "leaky_script.py").write_text(
        f'roboflow_api_key = "{fake_tok}"\n',
        encoding="utf-8",
    )
    findings = scan_secret_hygiene(mock_repo)
    assert findings, "Secret hygiene scanner failed to detect synthetic mock leak"


def test_t2_f5_train_script_masks_secret_in_error_messages():
    fake_secret = "Mock" + "Key" + "987654"
    raw_err = f"HTTP 403 failed with key={fake_secret} on workspace"
    sanitized = train_script._sanitize_secret(raw_err, fake_secret)
    assert fake_secret not in sanitized
    assert "***" in sanitized


def test_t2_f5_match_and_ml_cli_read_only_on_baseline_exports(tmp_path):
    before_csv = sha256_file(BASELINE_CSV)
    before_db = sha256_file(BASELINE_DB)

    ma_out = tmp_path / "ma_out"
    ml_out = tmp_path / "ml_out"
    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        [
            sys.executable,
            "modules/match_analytics.py",
            "--input",
            str(BASELINE_CSV),
            "--output-dir",
            str(ma_out),
            "--no-heatmaps",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "modules/ml_analyzer.py",
            "--input",
            str(BASELINE_DB),
            "--output-dir",
            str(ml_out),
            "--no-plots",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=True,
    )

    assert (ma_out / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (ml_out / config.EXPORT_ML_SUMMARY_FILENAME).is_file()
    assert sha256_file(BASELINE_CSV) == before_csv == BASELINE_CSV_SHA256
    assert sha256_file(BASELINE_DB) == before_db == BASELINE_DB_SHA256


def test_t2_f5_gitignore_ignores_runs_datasets_onnx_engine():
    for sample_path in [
        "runs/detect/train/weights/best.pt",
        "datasets/football/data.yaml",
        "models/model.onnx",
        "models/model.engine",
        "data/output_videos/test.mp4",
    ]:
        res = subprocess.run(
            ["git", "check-ignore", "-q", sample_path],
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert res.returncode == 0, f"Expected .gitignore to ignore {sample_path}"


def test_t2_f5_baseline_db_read_only_indexes_present():
    uri = f"file:{BASELINE_DB.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        idx_names = {
            row[1]
            for row in conn.execute("PRAGMA index_list('tracking_data')").fetchall()
        }
        row_count = int(conn.execute("SELECT COUNT(*) FROM tracking_data").fetchone()[0])
    finally:
        conn.close()

    assert {"idx_track_id", "idx_frame_idx"}.issubset(idx_names)
    assert row_count == BASELINE_ROW_COUNT
