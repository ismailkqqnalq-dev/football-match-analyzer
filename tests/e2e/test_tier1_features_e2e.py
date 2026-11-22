"""Tier 1 E2E Tests: Happy-path feature coverage across F1-F5 (>= 5 tests per feature)."""

import json
import math
import os
import shutil
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
from modules.tracker import BallTracker, PlayerTracker  # noqa: E402
from modules.video_processor import VideoProcessor  # noqa: E402
import utils.train_custom_yolo as train_script  # noqa: E402

from conftest import (  # noqa: E402
    BASELINE_CSV,
    BASELINE_CSV_SHA256,
    BASELINE_DB,
    BASELINE_DB_SHA256,
    BASELINE_ROW_COUNT,
    REPO_ROOT,
    build_short_tactical_clip,
    query_sqlite_count_readonly,
    scan_secret_hygiene,
    sha256_file,
)


class _FakeBoxes:
    """Dual-compatible mock Boxes supporting both box iteration and bulk tensor access."""

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


def _make_calibrated_df(n_frames: int = 160, human_classes=("player", "goalkeeper")) -> pd.DataFrame:
    """Create a compact calibrated synthetic DataFrame for MatchAnalyzer / MLAnalyzer E2E."""
    rng = np.random.default_rng(101)
    rows = []
    players = [
        (1, human_classes[0], "Team A", 20.0, 34.0, 0.10, 5.0),
        (2, human_classes[0], "Team A", 35.0, 20.0, 0.14, 8.0),
        (3, human_classes[0], "Team A", 50.0, 46.0, 0.18, 10.0),
        (4, human_classes[-1], "Team A" if human_classes[-1] == "person" else "Goalkeeper", 8.0, 34.0, 0.04, 2.0),
        (5, human_classes[0], "Team B", 80.0, 34.0, 0.10, 5.0),
        (6, human_classes[0], "Team B", 68.0, 22.0, 0.15, 8.0),
        (7, human_classes[0], "Team B", 55.0, 48.0, 0.19, 10.0),
        (8, human_classes[0], "Team B", 94.0, 34.0, 0.05, 3.0),
    ]
    for f in range(n_frames):
        ts = round(f / 25.0, 3)
        carrier = players[(f // 20) % len(players)]
        bx = float(np.clip(carrier[3] + carrier[6] * math.sin(f * carrier[5]) + 0.3, 1.0, 104.0))
        by = float(np.clip(carrier[4] + carrier[6] * math.cos(f * carrier[5] * 0.8), 1.0, 67.0))
        rows.append(
            {
                "frame_idx": f,
                "timestamp_sec": ts,
                "track_id": "ball",
                "class_name": "ball",
                "team": None,
                "confidence": 0.9,
                "is_predicted": False,
                "bbox_x1": bx * 10,
                "bbox_y1": by * 10,
                "bbox_x2": bx * 10 + 10,
                "bbox_y2": by * 10 + 10,
                "pixel_x": bx * 10 + 5,
                "pixel_y": by * 10 + 10,
                "pitch_x": round(bx, 2),
                "pitch_y": round(by, 2),
                "on_pitch": True,
            }
        )
        for tid, cls_name, team, base_x, base_y, amp, rad in players:
            px = float(np.clip(base_x + rad * math.sin(f * amp) + rng.normal(0, 0.02), 1.0, 104.0))
            py = float(np.clip(base_y + rad * math.cos(f * amp * 0.8) + rng.normal(0, 0.02), 1.0, 67.0))
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_sec": ts,
                    "track_id": tid,
                    "class_name": cls_name,
                    "team": team,
                    "confidence": 0.88,
                    "is_predicted": False,
                    "bbox_x1": px * 10,
                    "bbox_y1": py * 10,
                    "bbox_x2": px * 10 + 25,
                    "bbox_y2": py * 10 + 65,
                    "pixel_x": px * 10 + 12,
                    "pixel_y": py * 10 + 65,
                    "pitch_x": round(px, 2),
                    "pitch_y": round(py, 2),
                    "on_pitch": True,
                }
            )
    return pd.DataFrame(rows)


def _make_synthetic_jersey_frame(players_spec):
    frame = np.full((600, 1000, 3), (35, 140, 35), dtype=np.uint8)
    for spec in players_spec:
        x1, y1, x2, y2 = map(int, spec["bbox"])
        bw = x2 - x1
        bh = y2 - y1
        frame[
            y1 + int(bh * 0.22) : y1 + int(bh * 0.50),
            x1 + int(bw * 0.30) : x1 + int(bw * 0.70),
        ] = spec["bgr"]
    return frame


# ======================================================================
# F1: Custom 4-Class Football YOLO Model, Config, Training Hygiene & GPU
# ======================================================================


@pytest.mark.requires_model
def test_f1_trained_football_model_weight_exists_and_configured():
    models_dir = Path(config.MODELS_DIR)
    pt_files = list(models_dir.glob("*.pt"))
    assert pt_files, f"Expected a trained .pt weight file in {models_dir}"
    configured = Path(config.YOLO_MODEL_PATH)
    assert configured.is_file(), f"config.YOLO_MODEL_PATH does not exist: {configured}"
    assert configured.resolve().parent == models_dir.resolve()


def test_f1_config_four_class_football_taxonomy():
    assert set(config.TARGET_CLASS_IDS.values()) == {"ball", "goalkeeper", "player", "referee"}
    assert len(config.TARGET_CLASS_IDS) == 4
    assert config.BALL_CLASS_NAME == "ball"
    assert config.HUMAN_CLASS_NAMES == {"goalkeeper", "player", "referee"}
    assert config.USES_FINE_TUNED_FOOTBALL_MODEL is True
    assert config.TEAM_CLASSIFIER_N_CLUSTERS == 2


def test_f1_detector_validates_model_names_and_runs_inference_mode(monkeypatch):
    inference_mode_states = []
    football_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            self.names = football_names

        def predict(self, **kwargs):
            inference_mode_states.append(torch.is_inference_mode_enabled())
            src = kwargs["source"]
            n = len(src) if isinstance(src, list) else 1
            boxes = _FakeBoxes(
                [[10.0, 20.0, 40.0, 80.0], [100.0, 120.0, 115.0, 135.0]],
                [2.0, 0.0],
                [0.91, 0.84],
            )
            return [SimpleNamespace(boxes=boxes) for _ in range(n)]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(config, "TARGET_CLASS_IDS", football_names)

    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cpu")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    out = det.detect_frame(frame)
    batch_out = det.detect_batch([frame, frame])

    assert inference_mode_states == [True, True]
    assert len(out) == 2
    assert {d["class_name"] for d in out} == {"player", "ball"}
    for item in out:
        assert set(item.keys()) >= {"bbox", "class_id", "class_name", "confidence", "track_id", "team"}
        assert len(item["bbox"]) == 4
        assert isinstance(item["confidence"], float)
    assert len(batch_out) == 2


@pytest.mark.requires_model
@pytest.mark.gpu
def test_f1_detector_batch_cuda_rtx4050_inference():
    model_path = Path(config.YOLO_MODEL_PATH)
    assert model_path.is_file(), f"Missing trained football weights at {model_path}"

    detector = detector_module.ObjectDetector(model_path=str(model_path), device="cuda")
    assert detector.device.startswith("cuda")

    frames = [np.full((540, 960, 3), 60, dtype=np.uint8) for _ in range(4)]
    results = detector.detect_batch(frames)
    assert isinstance(results, list) and len(results) == 4
    for det_list in results:
        assert isinstance(det_list, list)
        for item in det_list:
            assert item["class_name"] in set(config.TARGET_CLASS_IDS.values())
            assert len(item["bbox"]) == 4


def test_f1_train_script_env_var_and_temp_workdir_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "unit_env_key_placeholder")
    assert train_script.resolve_api_key(None) == "unit_env_key_placeholder"

    args = train_script.parse_args([])
    assert args.api_key is None
    assert Path(args.work_dir).resolve().parent == Path(os.path.expandvars("%TEMP%")).resolve() or str(
        Path(args.work_dir)
    ).startswith(str(Path(train_script.DEFAULT_WORK_DIR).parent))

    workers = train_script._resolve_workers(None)
    assert 0 <= workers <= 2

    # Exercise train_custom_yolo.main with mocked Roboflow + YOLO without touching models/
    dataset_dir = tmp_path / "work" / "datasets" / "football-players-detection-3zvbc-v2"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text(
        "names: ['ball', 'goalkeeper', 'player', 'referee']\nnc: 4\n",
        encoding="utf-8",
    )

    train_kwargs_seen = {}
    copied_targets = []

    class FakeVersion:
        version = 2

        def download(self, fmt, location, overwrite=False):
            return SimpleNamespace(location=str(dataset_dir))

    class FakeProject:
        def versions(self):
            return [SimpleNamespace(version="1"), SimpleNamespace(version="2")]

        def version(self, num):
            assert num == 2
            return FakeVersion()

    class FakeRoboflow:
        def __init__(self, api_key):
            assert api_key == "unit_env_key_placeholder"

        def workspace(self, slug):
            return SimpleNamespace(project=lambda p: FakeProject())

    class FakeYOLOTrain:
        def __init__(self, base_model):
            self.trainer = None

        def train(self, **kwargs):
            train_kwargs_seen.update(kwargs)
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"fake_weights")
            self.trainer = SimpleNamespace(save_dir=str(save_dir))

    import ultralytics

    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=FakeRoboflow))
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLOTrain)
    monkeypatch.setattr(
        shutil,
        "copy2",
        lambda src, dst: copied_targets.append((Path(src), Path(dst))),
    )

    train_script.main(["--work-dir", str(tmp_path / "work"), "--epochs", "1"])

    assert train_kwargs_seen["cache"] is False
    assert train_kwargs_seen["amp"] is True
    assert train_kwargs_seen["workers"] <= 2
    assert str(tmp_path / "work") in str(train_kwargs_seen["project"])
    assert len(copied_targets) == 1
    assert copied_targets[0][0].name == "best.pt"


def test_f1_detector_bulk_parse_preserves_four_classes(monkeypatch):
    football_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    class FakeYOLO:
        def __init__(self, model_path):
            self.names = football_names

        def predict(self, **kwargs):
            boxes = _FakeBoxes(
                [
                    [10, 10, 20, 20],
                    [30, 30, 60, 100],
                    [70, 30, 100, 100],
                    [120, 30, 150, 100],
                ],
                [0.0, 1.0, 2.0, 3.0],
                [0.95, 0.89, 0.92, 0.87],
            )
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)
    monkeypatch.setattr(config, "TARGET_CLASS_IDS", football_names)

    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cpu")
    res = det.detect_frame(np.zeros((150, 200, 3), dtype=np.uint8))
    assert [d["class_name"] for d in res] == ["ball", "goalkeeper", "player", "referee"]


# ======================================================================
# F2: CLI Flags, Optional Video/Minimap/Outputs & Streaming DataExporter
# ======================================================================


def test_f2_main_cli_exposes_production_flags():
    res = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=True,
    )
    for flag in [
        "--no-video",
        "--no-minimap",
        "--no-csv",
        "--no-sql",
        "--export-dir",
        "--flush-interval",
        "--tracker",
    ]:
        assert flag in res.stdout, f"Missing CLI flag {flag} in main.py --help"


@pytest.mark.requires_model
def test_f2_main_cli_no_video_and_export_dir(tmp_path):
    clip = build_short_tactical_clip(tmp_path / "in", n_frames=8)
    export_dir = tmp_path / "exports"
    forbidden_video = tmp_path / "should_not_be_created.mp4"

    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip),
            "--output",
            str(forbidden_video),
            "--no-video",
            "--export-dir",
            str(export_dir),
            "--max-frames",
            "8",
            "--no-analytics",
            "--no-ml",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert res.returncode == 0, f"main.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert not forbidden_video.exists(), "--no-video must not write output video"
    assert export_dir.is_dir()
    assert any(export_dir.iterdir()), "Expected exported tracking artifacts in --export-dir"


def test_f2_video_processor_optional_output_and_bare_filename(tmp_path, monkeypatch):
    clip = build_short_tactical_clip(tmp_path / "in", n_frames=6)

    # 1) output_path=None -> no writer created, frames readable
    with VideoProcessor(str(clip), output_path=None) as vp_read:
        assert vp_read.writer is None
        frames = list(vp_read.read_frames())
        assert len(frames) == 6

    # 2) Bare filename output_path="bare_out.mp4" -> must succeed without FileNotFoundError
    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    with VideoProcessor(str(clip), output_path="bare_out.mp4") as vp_write:
        assert vp_write.writer is not None
        for _, frm in vp_write.read_frames():
            vp_write.write_frame(frm)

    out_file = work_dir / "bare_out.mp4"
    assert out_file.is_file() and out_file.stat().st_size > 0


def test_f2_data_exporter_streaming_flush_csv_and_sqlite_wal(tmp_path, isolated_export_dirs):
    csv_path = isolated_export_dirs / "stream.csv"
    db_path = isolated_export_dirs / "stream.db"

    exporter = DataExporter(
        fps=25.0,
        flush_interval=10,
        csv_path=str(csv_path),
        sql_path=str(db_path),
        enable_csv=True,
        enable_sql=True,
    )

    for f in range(35):
        exporter.append_frame_data(
            f,
            [
                {
                    "track_id": 1,
                    "class_name": "player",
                    "team": "Team A",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [100.0 + f, 100.0, 130.0 + f, 170.0],
                    "pitch_x": 20.0 + f * 0.2,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                },
                {
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.85,
                    "is_predicted": False,
                    "bbox": [200.0, 200.0, 210.0, 210.0],
                    "pitch_x": 30.0,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                },
            ],
        )

    exporter.postprocess(team_map={1: "Team A"})
    out_csv = exporter.export_to_csv(str(csv_path))
    out_sql = exporter.export_to_sql(str(db_path))

    assert Path(out_csv).is_file()
    assert Path(out_sql).is_file()

    df = pd.read_csv(out_csv)
    assert len(df.columns) == 16
    assert len(df) == 70
    assert query_sqlite_count_readonly(Path(out_sql)) == 70

    conn = sqlite3.connect(f"file:{Path(out_sql).as_posix()}?mode=ro", uri=True)
    try:
        idx_names = {
            row[1]
            for row in conn.execute("PRAGMA index_list('tracking_data')").fetchall()
        }
    finally:
        conn.close()
    assert {"idx_track_id", "idx_frame_idx"}.issubset(idx_names)


def test_f2_data_exporter_cross_chunk_backfill_and_ball_interpolation(isolated_export_dirs):
    csv_path = isolated_export_dirs / "backfill.csv"
    db_path = isolated_export_dirs / "backfill.db"

    exporter = DataExporter(
        fps=25.0,
        flush_interval=10,
        csv_path=str(csv_path),
        sql_path=str(db_path),
        enable_csv=True,
        enable_sql=True,
    )

    for f in range(30):
        dets = [
            {
                "track_id": 1,
                "class_name": "player",
                "team": None if f < 5 else "Team A",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [100.0 + f, 100.0, 120.0 + f, 160.0],
                "pitch_x": 20.0 + f * 0.1,
                "pitch_y": 30.0,
                "on_pitch": True,
            }
        ]
        # Ball omitted on frames 9, 19, 20 to verify sliding-window interpolation across chunk boundary
        if f not in (9, 19, 20):
            dets.append(
                {
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.85,
                    "is_predicted": False,
                    "bbox": [100.0 + f * 2, 200.0, 110.0 + f * 2, 210.0],
                    "pitch_x": 20.0 + f * 0.5,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                }
            )
        exporter.append_frame_data(f, dets)

    stats = exporter.postprocess(team_map={1: "Team A"})
    assert stats["backfilled_teams"] == 5
    assert stats["interpolated_ball_frames"] == 3

    exporter.export_to_csv(str(csv_path))
    exporter.export_to_sql(str(db_path))

    df_csv = pd.read_csv(csv_path)
    p1 = df_csv[df_csv["track_id"].astype(str) == "1"]
    assert len(p1) == 30
    assert p1["team"].isna().sum() == 0
    assert (p1["team"] == "Team A").all()

    ball_rows = df_csv[df_csv["class_name"] == "ball"].sort_values("frame_idx")
    assert set(ball_rows["frame_idx"].tolist()) == set(range(30))
    f9 = ball_rows[ball_rows["frame_idx"] == 9].iloc[0]
    assert f9["pitch_x"] == pytest.approx(24.5, abs=0.05)


def test_f2_data_exporter_distance_summary_14_columns(isolated_export_dirs):
    exporter = DataExporter(fps=25.0)
    for f in range(30):
        exporter.append_frame_data(
            f,
            [
                {
                    "track_id": 1,
                    "class_name": "player",
                    "team": "Team A",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [100.0 + f, 100.0, 124.0 + f, 164.0],
                    "pitch_x": 20.0 + f * 0.25,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                },
                {
                    "track_id": 2,
                    "class_name": "goalkeeper",
                    "team": "Goalkeeper",
                    "confidence": 0.88,
                    "is_predicted": False,
                    "bbox": [40.0, 200.0, 64.0, 264.0],
                    "pitch_x": 5.0,
                    "pitch_y": 34.0 + f * 0.05,
                    "on_pitch": True,
                },
            ],
        )

    out_summary = exporter.export_distance_summary(
        output_path=str(isolated_export_dirs / "player_distance_summary.csv")
    )
    assert out_summary is not None and Path(out_summary).is_file()
    sum_df = pd.read_csv(out_summary)
    expected_cols = [
        "track_id",
        "class_name",
        "team",
        "frames_tracked",
        "active_time_sec",
        "total_distance_m",
        "avg_speed_kmh",
        "max_speed_kmh",
        "walking_m",
        "jogging_m",
        "running_m",
        "high_speed_running_m",
        "sprint_m",
        "sprint_count",
    ]
    assert list(sum_df.columns) == expected_cols
    assert len(sum_df) == 2


# ======================================================================
# F3: Tracking, Team Memory Bounding & Vectorized Analytics / ML
# ======================================================================


def test_f3_player_tracker_inference_mode_and_scene_cut_hook(monkeypatch):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player", "goalkeeper", "referee"})
    tracker = PlayerTracker(n_init=1, embedder_gpu=False)

    inference_flags = []
    orig_update_tracks = tracker.tracker.update_tracks

    def spy_update_tracks(*args, **kwargs):
        inference_flags.append(torch.is_inference_mode_enabled())
        return orig_update_tracks(*args, **kwargs)

    monkeypatch.setattr(tracker.tracker, "update_tracks", spy_update_tracks)

    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.rectangle(frame, (100, 80), (140, 180), (200, 50, 50), -1)
    det = [{"bbox": [100.0, 80.0, 140.0, 180.0], "confidence": 0.92, "class_name": "player"}]

    tracked = []
    for _ in range(3):
        tracked = tracker.update(det, frame)
    assert inference_flags == [True, True, True]
    assert len(tracked) == 1
    assert set(tracked[0].keys()) >= {"bbox", "track_id", "class_name", "confidence", "team", "is_predicted"}
    assert tracked[0]["class_name"] == "player"

    assert hasattr(tracker, "on_scene_cut") or hasattr(tracker, "reset_on_cut")
    if hasattr(tracker, "on_scene_cut"):
        tracker.on_scene_cut()
    else:
        tracker.reset_on_cut()


def test_f3_ball_tracker_scene_cut_clears_velocity_and_state():
    bt = BallTracker(max_missing_frames=5, velocity_damping=0.9, max_velocity_px=150.0)
    shape = (240, 320, 3)

    bt.update([{"class_name": "ball", "bbox": [50.0, 50.0, 60.0, 60.0], "confidence": 0.9}], shape)
    bt.update([{"class_name": "ball", "bbox": [70.0, 50.0, 80.0, 60.0], "confidence": 0.9}], shape)
    assert bt.velocity == pytest.approx((20.0, 0.0))

    assert hasattr(bt, "on_scene_cut") or hasattr(bt, "reset_on_cut")
    if hasattr(bt, "on_scene_cut"):
        bt.on_scene_cut()
    else:
        bt.reset_on_cut()

    # After scene cut, missing ball must NOT carry over stale pre-cut velocity
    assert bt.update([], frame_shape=shape) is None


def test_f3_team_classifier_purges_color_history_on_lock_and_maps_roles(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    specs = []
    for i in range(4):
        specs.append({"track_id": i + 1, "class_name": "player", "bbox": [30 + i * 70, 80, 75 + i * 70, 210], "bgr": (170, 30, 20)})
    for i in range(4):
        specs.append({"track_id": i + 5, "class_name": "player", "bbox": [30 + i * 70, 260, 75 + i * 70, 390], "bgr": (245, 245, 245)})

    frame = _make_synthetic_jersey_frame(specs)
    tc = TeamClassifier(n_clusters=2, min_unique_tracks=8, color_space="LAB", min_samples_for_lock=2)
    dets = [{"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None} for s in specs]
    tc.assign_teams(dets, frame)
    assert tc.is_fitted is True

    # New track_id 99 locks after 2 samples and must purge color_history[99]
    new_spec = specs + [{"track_id": 99, "class_name": "player", "bbox": [650, 80, 695, 210], "bgr": (170, 30, 20)}]
    frame_99 = _make_synthetic_jersey_frame(new_spec)
    tc.assign_teams([{"track_id": 99, "class_name": "player", "bbox": [650, 80, 695, 210], "team": None}], frame_99)
    assert 99 not in tc.team_assignment
    tc.assign_teams([{"track_id": 99, "class_name": "player", "bbox": [650, 80, 695, 210], "team": None}], frame_99)
    assert 99 in tc.team_assignment
    assert len(tc.color_history.get(99, [])) == 0

    # Referee and goalkeeper map directly
    role_dets = [
        {"track_id": 201, "class_name": "referee", "bbox": [750, 80, 795, 210], "team": None},
        {"track_id": 202, "class_name": "goalkeeper", "bbox": [820, 80, 865, 210], "team": None},
    ]
    out_roles = tc.assign_teams(role_dets, frame_99)
    assert out_roles[0]["team"] == "Referee"
    assert out_roles[1]["team"] == "Goalkeeper"


def test_f3_match_analyzer_from_csv_and_sql_exports_artifacts(tmp_path):
    df = _make_calibrated_df(n_frames=120, human_classes=("player", "goalkeeper"))
    csv_path = tmp_path / "match.csv"
    db_path = tmp_path / "match.db"
    df.to_csv(csv_path, index=False)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("tracking_data", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    csv_out = tmp_path / "out_csv"
    sql_out = tmp_path / "out_sql"

    s_csv = MatchAnalyzer.from_csv(str(csv_path), fps=25.0).export_all(output_dir=str(csv_out), generate_heatmaps=True)
    s_sql = MatchAnalyzer.from_sql(str(db_path), fps=25.0).export_all(output_dir=str(sql_out), generate_heatmaps=True)

    assert s_csv["possession"]["team_a_pct"] == pytest.approx(s_sql["possession"]["team_a_pct"])
    assert s_csv["passes"]["total_completed_passes"] == s_sql["passes"]["total_completed_passes"]
    assert (csv_out / "possession_timeline.csv").is_file()
    assert (csv_out / "match_analytics_summary.json").is_file()
    assert len(list((csv_out / "heatmaps").glob("*.png"))) >= 2


def test_f3_ml_analyzer_from_csv_and_sql_exports_turkish_report_and_strict_json(tmp_path):
    df = _make_calibrated_df(n_frames=160, human_classes=("player", "goalkeeper"))
    csv_path = tmp_path / "ml_in.csv"
    db_path = tmp_path / "ml_in.db"
    df.to_csv(csv_path, index=False)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("tracking_data", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    out_dir = tmp_path / "ml_out"
    res = MLAnalyzer.from_sql(str(db_path), fps=25.0).export_all(output_dir=str(out_dir), generate_plots=True)

    expected_files = [
        config.EXPORT_ML_PLAYER_PROFILES_FILENAME,
        config.EXPORT_ML_TEMPO_WINDOWS_FILENAME,
        config.EXPORT_ML_ANOMALIES_FILENAME,
        config.EXPORT_ML_PASS_PREDICTIONS_FILENAME,
        config.EXPORT_ML_REPORT_FILENAME,
        config.EXPORT_ML_SUMMARY_FILENAME,
        "xp_model_comparison.png",
        "xt_spatial_heatmap.png",
        "player_clusters_pca.png",
        "tempo_anomaly_timeline.png",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 0, f"Missing ML output {fname}"

    report_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert any(kw in report_text for kw in ["xP", "xT", "Taktik", "Anomali", "Rol"])

    raw_json = (out_dir / config.EXPORT_ML_SUMMARY_FILENAME).read_text(encoding="utf-8")
    parsed = json.loads(raw_json, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Bad JSON constant {c}")))
    assert parsed["coord_mode"] == "pitch_meters"
    assert res["is_calibrated"] is True

    # Also smoke-check from_csv parity
    res_csv = MLAnalyzer.from_csv(str(csv_path), fps=25.0).run_all()
    assert res_csv["data_quality"]["total_rows"] == len(df)


def test_f3_analytics_and_ml_accept_both_coco_and_four_class_human_names(tmp_path):
    for label, classes in [("coco", ("person", "person")), ("football4", ("player", "goalkeeper"))]:
        df = _make_calibrated_df(n_frames=120, human_classes=classes)
        ma = MatchAnalyzer(df, fps=25.0)
        summary = ma.export_all(output_dir=str(tmp_path / f"ma_{label}"), generate_heatmaps=False)
        assert summary["possession"]["mode"] == "pitch_meters"
        assert summary["possession"]["ball_detected_frames"] == 120

        ml = MLAnalyzer(df, fps=25.0, analytics_summary=summary)
        ml_res = ml.export_all(output_dir=str(tmp_path / f"ml_{label}"), generate_plots=False)
        assert ml_res["data_quality"]["total_rows"] == len(df)


# ======================================================================
# F4: TV Broadcast Scene-Cut, Fade, Recovery & Homography Guards
# ======================================================================


def test_f4_camera_motion_tracker_main_pan_returns_float64_homography(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        p1_start, p1_end = clip.phases["phase1_main_pan"]
        for t in range(p1_start, p1_end + 1):
            ok, frame = cap.read()
            assert ok, f"Failed to read frame {t}"
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            assert mat is not None, f"Frame {t} in main pan unexpectedly returned None"
            assert mat.shape == (3, 3)
            assert mat.dtype == np.float64
            assert mat[2, 2] == pytest.approx(1.0, abs=1e-6)
            assert tracker.is_anchored is True
            if t > 0:
                assert tracker.scene_cut_detected is False
    finally:
        cap.release()


def test_f4_camera_motion_tracker_hard_cut_invalidates_anchor(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        cut_frame_idx = clip.phases["phase2a_hard_cut"][0]
        for t in range(cut_frame_idx + 1):
            ok, frame = cap.read()
            assert ok
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            if t == cut_frame_idx:
                assert mat is None
                assert tracker.scene_cut_detected is True
                assert tracker.is_anchored is False
    finally:
        cap.release()


def test_f4_camera_motion_tracker_gradual_dissolve_detected_and_none(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        d_start, d_end = clip.phases["phase3_gradual_dissolve"]
        for t in range(d_end + 1):
            ok, frame = cap.read()
            assert ok
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            if d_start <= t <= d_end:
                assert mat is None, f"Gradual dissolve frame {t} should return None"
                assert tracker.is_anchored is False
    finally:
        cap.release()


def test_f4_camera_motion_tracker_auto_reanchors_on_return_to_main_camera(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    recovered_frame = None
    try:
        p4_start, p4_end = clip.phases["phase4_recovered_main"]
        for t in range(clip.total_frames):
            ok, frame = cap.read()
            assert ok
            was_anchored = getattr(tracker, "is_anchored", False)
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            if t >= p4_start and recovered_frame is None and mat is not None and tracker.is_anchored:
                recovered_frame = t
                if hasattr(tracker, "recovered_this_frame"):
                    assert tracker.recovered_this_frame is True or not was_anchored
    finally:
        cap.release()

    assert recovered_frame is not None, "CameraMotionTracker failed to re-anchor on main tactical camera return"
    assert recovered_frame <= clip.phases["phase4_recovered_main"][0] + 15


def test_f4_perspective_transformer_batched_pitch_coordinates_accurate(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    transformer = PerspectiveTransformer(source_points=clip.source_calibration_points)
    assert transformer.is_calibrated is True

    cap = cv2.VideoCapture(str(clip.video_path))
    tracker = CameraMotionTracker()
    try:
        p1_start, p1_end = clip.phases["phase1_main_pan"]
        max_err_m = 0.0
        for t in range(p1_start, p1_end + 1):
            ok, frame = cap.read()
            assert ok
            mat = tracker.update(frame, clip.gt_detections_by_frame[t])
            assert mat is not None
            dets = [dict(d) for d in clip.gt_detections_by_frame[t]]
            out = transformer.add_pitch_coordinates(dets, camera_motion_matrix=mat)
            for item in out:
                tid = item["track_id"]
                gt_x, gt_y = clip.gt_pitch_coords_by_frame[t][tid]
                assert item["on_pitch"] is True
                assert 0.0 <= item["pitch_x"] <= config.PITCH_LENGTH_M
                assert 0.0 <= item["pitch_y"] <= config.PITCH_WIDTH_M
                err = math.hypot(item["pitch_x"] - gt_x, item["pitch_y"] - gt_y)
                max_err_m = max(max_err_m, err)
        assert max_err_m < 1.5, f"Main pan coordinate drift {max_err_m:.2f}m exceeded 1.5m"
    finally:
        cap.release()


def test_f4_draw_pitch_minimap_shape_and_canvas_caching(synthetic_tv_broadcast_clip):
    transformer = PerspectiveTransformer(source_points=synthetic_tv_broadcast_clip.source_calibration_points)
    sample_dets = [
        {"class_name": "player", "team": "Team A", "pitch_x": 30.0, "pitch_y": 25.0, "on_pitch": True},
        {"class_name": "player", "team": "Team B", "pitch_x": 70.0, "pitch_y": 40.0, "on_pitch": True},
        {"class_name": "ball", "team": None, "pitch_x": 52.5, "pitch_y": 34.0, "on_pitch": True},
    ]
    mm1 = transformer.draw_pitch_minimap(sample_dets)
    mm2 = transformer.draw_pitch_minimap([])
    assert mm1.shape == (config.PITCH_MINIMAP_HEIGHT_PX, config.PITCH_MINIMAP_WIDTH_PX, 3)
    assert mm1.shape == (388, 600, 3)
    assert mm1.dtype == np.uint8
    assert not np.array_equal(mm1, mm2)


# ======================================================================
# F5: Secret Hygiene, Git Ignore Compliance & Baseline Immutability
# ======================================================================


def test_f5_gitignore_ignores_pt_weights_and_no_tracked_pt():
    tracked_pt = subprocess.run(
        ["git", "ls-files", "*.pt"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert tracked_pt.stdout.strip() == "", f"Tracked .pt weights found in git: {tracked_pt.stdout}"

    ignore_check = subprocess.run(
        ["git", "check-ignore", "-q", "models/football_yolo11s_best.pt"],
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert ignore_check.returncode == 0, "*.pt must be ignored by .gitignore"


def test_f5_working_tree_has_zero_secret_leaks():
    violations = [v for v in scan_secret_hygiene(REPO_ROOT) if v.startswith("working_tree/")]
    assert not violations, f"Working-tree secret-hygiene violations detected: {violations}"


def test_f5_git_history_has_zero_secret_leaks():
    violations = [v for v in scan_secret_hygiene(REPO_ROOT) if v.startswith("git_history/")]
    assert not violations, f"Git-history secret-hygiene violations detected: {violations}"


def test_f5_config_py_has_no_credentials():
    assert not hasattr(config, "ROBOFLOW_API_KEY")
    cfg_text = (REPO_ROOT / "config" / "config.py").read_text(encoding="utf-8")
    assert "api_key =" not in cfg_text.lower() and "api_key=" not in cfg_text.lower()
    cfg_violations = [
        v for v in scan_secret_hygiene(REPO_ROOT)
        if "config/config.py" in v.replace("\\", "/")
    ]
    assert not cfg_violations


def test_f5_baseline_exports_sha256_and_row_counts_intact():
    assert sha256_file(BASELINE_CSV) == BASELINE_CSV_SHA256
    assert sha256_file(BASELINE_DB) == BASELINE_DB_SHA256
    df = pd.read_csv(BASELINE_CSV)
    assert len(df) == BASELINE_ROW_COUNT
    assert query_sqlite_count_readonly(BASELINE_DB) == BASELINE_ROW_COUNT
