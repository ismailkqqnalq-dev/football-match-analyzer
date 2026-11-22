"""Tier 3 E2E Tests: Pairwise cross-feature combination tests."""

import math
import os
import re
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
from modules.perspective_transformer import PerspectiveTransformer  # noqa: E402
from modules.team_classifier import TeamClassifier  # noqa: E402
from modules.tracker import BallTracker, PlayerTracker  # noqa: E402

from conftest import (  # noqa: E402
    REPO_ROOT,
    build_short_tactical_clip,
    query_sqlite_count_readonly,
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


def _make_jersey_frame(players_spec):
    frame = np.full((540, 960, 3), (35, 140, 35), dtype=np.uint8)
    for spec in players_spec:
        x1, y1, x2, y2 = map(int, spec["bbox"])
        bw = x2 - x1
        bh = y2 - y1
        frame[
            y1 + int(bh * 0.22) : y1 + int(bh * 0.50),
            x1 + int(bw * 0.30) : x1 + int(bw * 0.70),
        ] = spec["bgr"]
    return frame


@pytest.mark.requires_model
def test_t3_no_video_no_minimap_export_dir_with_analytics_and_ml(tmp_path):
    clip = build_short_tactical_clip(tmp_path / "in", n_frames=12)
    export_dir = tmp_path / "exports"
    out_video = tmp_path / "forbidden.mp4"

    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip),
            "--output",
            str(out_video),
            "--no-video",
            "--no-minimap",
            "--export-dir",
            str(export_dir),
            "--max-frames",
            "12",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert res.returncode == 0, f"CLI failed:\n{res.stdout}\n{res.stderr}"
    assert not out_video.exists()
    assert (export_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (export_dir / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).is_file()


@pytest.mark.requires_model
def test_t3_no_csv_flush_interval_sqlite_only_analytics_and_ml(tmp_path):
    clip = build_short_tactical_clip(tmp_path / "in", n_frames=12)
    export_dir = tmp_path / "exports_sql_only"

    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip),
            "--no-video",
            "--no-csv",
            "--flush-interval",
            "5",
            "--export-dir",
            str(export_dir),
            "--max-frames",
            "12",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert res.returncode == 0, f"CLI failed:\n{res.stdout}\n{res.stderr}"
    assert not (export_dir / config.EXPORT_CSV_FILENAME).exists()
    assert (export_dir / config.EXPORT_SQL_FILENAME).is_file()
    assert (export_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (export_dir / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).is_file()


@pytest.mark.requires_model
def test_t3_no_sql_flush_interval_csv_only_analytics_and_ml(tmp_path):
    clip = build_short_tactical_clip(tmp_path / "in", n_frames=12)
    export_dir = tmp_path / "exports_csv_only"

    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip),
            "--no-video",
            "--no-sql",
            "--flush-interval",
            "5",
            "--export-dir",
            str(export_dir),
            "--max-frames",
            "12",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert res.returncode == 0, f"CLI failed:\n{res.stdout}\n{res.stderr}"
    assert not (export_dir / config.EXPORT_SQL_FILENAME).exists()
    assert (export_dir / config.EXPORT_CSV_FILENAME).is_file()
    assert (export_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (export_dir / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).is_file()


def test_t3_scene_cut_coordinates_tracker_resets_and_team_lock_retention(
    synthetic_tv_broadcast_clip, monkeypatch
):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player", "goalkeeper", "referee"})
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    clip = synthetic_tv_broadcast_clip
    cap = cv2.VideoCapture(str(clip.video_path))
    motion_tracker = CameraMotionTracker()
    transformer = PerspectiveTransformer(source_points=clip.source_calibration_points)
    player_tracker = PlayerTracker(n_init=1, embedder_gpu=False)
    ball_tracker = BallTracker(max_missing_frames=5)
    team_classifier = TeamClassifier(n_clusters=2, min_unique_tracks=4, min_samples_for_lock=2)

    assert hasattr(player_tracker, "on_scene_cut") or hasattr(player_tracker, "reset_on_cut")
    assert hasattr(ball_tracker, "on_scene_cut") or hasattr(ball_tracker, "reset_on_cut")

    locked_team_before_cut = {}
    recovered_after_cut = False
    try:
        for t in range(clip.total_frames):
            ok, frame = cap.read()
            assert ok
            dets = clip.gt_detections_by_frame[t]
            tracked_players = player_tracker.update(dets, frame)
            tracked_ball = ball_tracker.update(dets, frame_shape=frame.shape)
            tracked_players = team_classifier.assign_teams(tracked_players, frame)

            all_tracked = list(tracked_players)
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            mat = motion_tracker.update(frame, dets)
            if motion_tracker.scene_cut_detected:
                if hasattr(player_tracker, "on_scene_cut"):
                    player_tracker.on_scene_cut()
                else:
                    player_tracker.reset_on_cut()
                if hasattr(ball_tracker, "on_scene_cut"):
                    ball_tracker.on_scene_cut()
                else:
                    ball_tracker.reset_on_cut()
                # Ball prediction immediately after cut reset must be None on empty detections
                assert ball_tracker.update([], frame_shape=frame.shape) is None

            if mat is not None:
                transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=mat)
            else:
                for item in all_tracked:
                    item["pitch_x"], item["pitch_y"], item["on_pitch"] = None, None, None

            if t == 35:
                locked_team_before_cut = dict(team_classifier.team_assignment)

            if clip.phases["phase2a_hard_cut"][0] <= t <= clip.phases["phase3_gradual_dissolve"][1]:
                assert mat is None
                assert all(item["pitch_x"] is None for item in all_tracked)

            if t >= clip.phases["phase4_recovered_main"][0] and mat is not None:
                recovered_after_cut = True
                # Team locks acquired before cut must remain intact across scene cut
                for tid, team_label in locked_team_before_cut.items():
                    assert team_classifier.team_assignment.get(tid) == team_label
    finally:
        cap.release()

    assert recovered_after_cut is True


def test_t3_streaming_exporter_cross_chunk_backfill_and_ball_interp_parity(isolated_export_dirs):
    stream_csv = isolated_export_dirs / "stream_p.csv"
    stream_db = isolated_export_dirs / "stream_p.db"
    mem_csv = isolated_export_dirs / "mem_p.csv"

    exp_stream = DataExporter(
        fps=25.0,
        flush_interval=7,
        csv_path=str(stream_csv),
        sql_path=str(stream_db),
        enable_csv=True,
        enable_sql=True,
    )
    exp_mem = DataExporter(fps=25.0, flush_interval=0)

    for f in range(45):
        dets = [
            {
                "track_id": 1,
                "class_name": "player",
                "team": None if f < 6 else "Team A",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [100.0 + f * 2, 100.0, 124.0 + f * 2, 164.0],
                "pitch_x": 15.0 + f * 0.25,
                "pitch_y": 30.0,
                "on_pitch": True,
            },
            {
                "track_id": 2,
                "class_name": "player",
                "team": None if f < 6 else "Team B",
                "confidence": 0.89,
                "is_predicted": False,
                "bbox": [400.0 - f, 200.0, 424.0 - f, 264.0],
                "pitch_x": 65.0 - f * 0.2,
                "pitch_y": 40.0,
                "on_pitch": True,
            },
        ]
        # Ball gaps across chunk boundaries (frames 6, 7 and 20, 21)
        if f not in (6, 7, 20, 21):
            dets.append(
                {
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.85,
                    "is_predicted": False,
                    "bbox": [200.0 + f, 250.0, 210.0 + f, 260.0],
                    "pitch_x": 20.0 + f * 0.4,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                }
            )
        exp_stream.append_frame_data(f, dets)
        exp_mem.append_frame_data(f, dets)

    stats_stream = exp_stream.postprocess(team_map={1: "Team A", 2: "Team B"})
    stats_mem = exp_mem.postprocess(team_map={1: "Team A", 2: "Team B"})

    assert stats_stream["backfilled_teams"] == stats_mem["backfilled_teams"] == 12
    assert stats_stream["interpolated_ball_frames"] == stats_mem["interpolated_ball_frames"] == 4

    exp_stream.export_to_csv(str(stream_csv))
    exp_stream.export_to_sql(str(stream_db))
    exp_mem.export_to_csv(str(mem_csv))

    df_stream = pd.read_csv(stream_csv).sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    df_mem = pd.read_csv(mem_csv).sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    assert len(df_stream) == len(df_mem)
    assert query_sqlite_count_readonly(stream_db) == len(df_mem)
    assert df_stream["team"].fillna("").tolist() == df_mem["team"].fillna("").tolist()

    sum_stream = exp_stream.export_distance_summary(str(isolated_export_dirs / "sum_stream.csv"))
    sum_mem = exp_mem.export_distance_summary(str(isolated_export_dirs / "sum_mem.csv"))
    assert pd.read_csv(sum_stream)["total_distance_m"].tolist() == pytest.approx(
        pd.read_csv(sum_mem)["total_distance_m"].tolist(), abs=0.2
    )


def test_t3_cuda_oom_halving_plus_four_class_team_classifier(monkeypatch):
    football_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    monkeypatch.setattr(config, "TARGET_CLASS_IDS", football_names)
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"goalkeeper", "player", "referee"})
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(detector_module.torch.cuda, "empty_cache", lambda: None)

    specs = [
        {"track_id": 1, "class_name": "player", "bbox": [40, 80, 85, 210], "bgr": (170, 30, 20)},
        {"track_id": 2, "class_name": "player", "bbox": [120, 80, 165, 210], "bgr": (170, 30, 20)},
        {"track_id": 3, "class_name": "player", "bbox": [200, 80, 245, 210], "bgr": (245, 245, 245)},
        {"track_id": 4, "class_name": "player", "bbox": [280, 80, 325, 210], "bgr": (245, 245, 245)},
        {"track_id": 5, "class_name": "goalkeeper", "bbox": [360, 80, 405, 210], "bgr": (40, 200, 40)},
        {"track_id": 6, "class_name": "referee", "bbox": [440, 80, 485, 210], "bgr": (0, 230, 245)},
    ]
    frame = _make_jersey_frame(specs)

    class OOMFourClassYOLO:
        def __init__(self, model_path):
            self.names = football_names

        def predict(self, **kwargs):
            src = kwargs["source"]
            n = len(src) if isinstance(src, list) else 1
            if n > 2:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            boxes = _FakeBoxes(
                [s["bbox"] for s in specs] + [[520, 200, 535, 215]],
                [2, 2, 2, 2, 1, 3, 0],
                [0.92, 0.91, 0.9, 0.9, 0.88, 0.89, 0.85],
            )
            return [SimpleNamespace(boxes=boxes) for _ in range(n)]

    monkeypatch.setattr(detector_module, "YOLO", OOMFourClassYOLO)
    det = detector_module.ObjectDetector(model_path="dummy.pt", device="cuda")
    batch = det.detect_batch([frame, frame, frame, frame])
    assert len(batch) == 4

    tc = TeamClassifier(n_clusters=2, min_unique_tracks=4, min_samples_for_lock=2)
    human_dets = [
        {"track_id": i + 1, "class_name": d["class_name"], "bbox": d["bbox"], "team": None}
        for i, d in enumerate(batch[0])
        if d["class_name"] != "ball"
    ]
    for _ in range(2):
        assigned = tc.assign_teams([dict(d) for d in human_dets], frame)
    by_cls = {d["class_name"]: d["team"] for d in assigned}
    assert by_cls["goalkeeper"] == "Team B"
    assert any(d["class_name"] == "goalkeeper" and d["team"] == "Team B" for d in assigned)
    assert by_cls["referee"] == "Referee"
    assert {d["team"] for d in assigned if d["class_name"] == "player"} == {"Team A", "Team B"}


def test_t3_long_pan_with_scorebug_camera_motion_and_batched_perspective(synthetic_tv_broadcast_clip):
    clip = synthetic_tv_broadcast_clip
    transformer = PerspectiveTransformer(source_points=clip.source_calibration_points)
    motion_tracker = CameraMotionTracker()

    cap = cv2.VideoCapture(str(clip.video_path))
    try:
        p1_start, p1_end = clip.phases["phase1_main_pan"]
        for t in range(p1_start, p1_end + 1):
            ok, frame = cap.read()
            assert ok
            dets = [dict(d) for d in clip.gt_detections_by_frame[t]]
            mat = motion_tracker.update(frame, dets)
            assert mat is not None
            batched = transformer.add_pitch_coordinates(dets, camera_motion_matrix=mat)

            # Verify batched add_pitch_coordinates matches transform_point parity & <1.5m drift
            eff = transformer.homography_matrix @ mat
            for item in batched:
                fx, fy = transformer.get_foot_position(item["bbox"])
                pt = transformer.transform_point(fx, fy, homography_matrix=eff)
                assert pt is not None
                assert item["pitch_x"] == pytest.approx(round(pt[0], 2), abs=0.02)
                assert item["pitch_y"] == pytest.approx(round(pt[1], 2), abs=0.02)
                gx, gy = clip.gt_pitch_coords_by_frame[t][item["track_id"]]
                assert math.hypot(item["pitch_x"] - gx, item["pitch_y"] - gy) < 1.5
    finally:
        cap.release()


@pytest.mark.requires_model
def test_t3_tracker_fast_mode_plus_flush_interval_and_no_ml_no_analytics(tmp_path):
    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    help_res = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=True,
    )
    assert "--tracker" in help_res.stdout, "--tracker flag not found in main.py --help"

    # Discover non-deepsort fast tracker choice from --help if listed (e.g. bytetrack / iou / fast)
    tracker_choice = "bytetrack"
    m = re.search(r"--tracker\s+\{([^}]+)\}", help_res.stdout)
    if m:
        choices = [c.strip() for c in m.group(1).split(",") if c.strip()]
        non_default = [c for c in choices if c.lower() != "deepsort"]
        if non_default:
            tracker_choice = non_default[0]

    clip = build_short_tactical_clip(tmp_path / "in", n_frames=10)
    export_dir = tmp_path / "fast_exports"
    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip),
            "--no-video",
            "--tracker",
            tracker_choice,
            "--flush-interval",
            "5",
            "--no-analytics",
            "--no-ml",
            "--export-dir",
            str(export_dir),
            "--max-frames",
            "10",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=False,
    )
    assert res.returncode == 0, f"CLI fast-tracker mode failed:\n{res.stdout}\n{res.stderr}"
    assert (export_dir / config.EXPORT_CSV_FILENAME).is_file() or (
        export_dir / config.EXPORT_SQL_FILENAME
    ).is_file()
    assert not (export_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).exists()
    assert not (export_dir / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).exists()
