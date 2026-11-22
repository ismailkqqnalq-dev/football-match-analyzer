"""Tier 4 E2E Tests: Full real-world application workloads & audits."""

import gc
import json
import math
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

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
from modules.camera_motion_tracker import CameraMotionTracker  # noqa: E402
from modules.data_exporter import DataExporter  # noqa: E402
from modules.detector import ObjectDetector  # noqa: E402
from modules.match_analytics import MatchAnalyzer  # noqa: E402
from modules.ml_analyzer import MLAnalyzer  # noqa: E402
from modules.perspective_transformer import PerspectiveTransformer  # noqa: E402

from conftest import (  # noqa: E402
    BASELINE_CSV,
    BASELINE_CSV_SHA256,
    BASELINE_DB,
    BASELINE_DB_SHA256,
    BASELINE_ROW_COUNT,
    REPO_ROOT,
    iter_90min_frame_batches,
    query_sqlite_count_readonly,
    scan_secret_hygiene,
    sha256_file,
)


def test_t4_scenario1_end_to_end_cli_broadcast_pipeline(
    tmp_path, synthetic_tv_broadcast_clip
):
    """
    Scenario 1: Run synthetic TV broadcast clip end-to-end via `main.py` CLI
    in both video-output and `--no-video` modes with `--export-dir` and `--flush-interval`.
    """
    clip = synthetic_tv_broadcast_clip
    export_dir = tmp_path / "cli_exports"
    out_video = tmp_path / "annotated_broadcast.mp4"

    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip.video_path),
            "--output",
            str(out_video),
            "--export-dir",
            str(export_dir),
            "--flush-interval",
            "25",
            "--max-frames",
            "50",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=False,
    )
    assert res.returncode == 0, f"main.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert out_video.is_file() and out_video.stat().st_size > 0
    assert (export_dir / config.EXPORT_SQL_FILENAME).is_file()
    assert (export_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (export_dir / config.EXPORT_POSSESSION_FILENAME).is_file()

    ml_dir = export_dir / "ml_reports"
    ml_json = ml_dir / config.EXPORT_ML_SUMMARY_FILENAME
    ml_md = ml_dir / config.EXPORT_ML_REPORT_FILENAME
    assert ml_json.is_file() and ml_md.is_file()
    json.loads(
        ml_json.read_text(encoding="utf-8"),
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"NaN/Inf in JSON: {c}")),
    )

    conn = sqlite3.connect(
        f"file:{(export_dir / config.EXPORT_SQL_FILENAME).as_posix()}?mode=ro",
        uri=True,
    )
    try:
        idx_names = {
            row[1]
            for row in conn.execute("PRAGMA index_list('tracking_data')").fetchall()
        }
    finally:
        conn.close()
    assert {"idx_track_id", "idx_frame_idx"}.issubset(idx_names)

    # Also verify --no-video mode isolates exports cleanly
    export_novid = tmp_path / "cli_exports_novid"
    forbidden_vid = tmp_path / "must_not_exist.mp4"
    res_novid = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--input",
            str(clip.video_path),
            "--output",
            str(forbidden_vid),
            "--no-video",
            "--export-dir",
            str(export_novid),
            "--flush-interval",
            "25",
            "--max-frames",
            "25",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env,
        check=False,
    )
    assert res_novid.returncode == 0, f"main.py --no-video failed:\n{res_novid.stderr}"
    assert not forbidden_vid.exists()
    assert (export_novid / config.EXPORT_SQL_FILENAME).is_file()


def test_t4_scenario2_90min_scale_workload_memory_and_analytics_ml(
    isolated_export_dirs, memory_tracker
):
    """
    Scenario 2: Full 90-minute match scale workload (135,000 frames / 2,700,000 rows):
    Streaming DataExporter -> MatchAnalyzer.from_sql -> MLAnalyzer.from_sql
    under bounded psutil RSS / OS peak_wset (NO tracemalloc).
    """
    total_frames = 135_000
    csv_path = isolated_export_dirs / "match_90min.csv"
    db_path = isolated_export_dirs / "match_90min.db"
    dist_path = isolated_export_dirs / "distance_90min.csv"
    ma_dir = isolated_export_dirs / "analytics_90min"
    ml_dir = isolated_export_dirs / "ml_90min"

    exporter = DataExporter(
        fps=25.0,
        flush_interval=2500,
        csv_path=str(csv_path),
        sql_path=str(db_path),
        enable_csv=True,
        enable_sql=True,
    )

    t0 = time.perf_counter()
    for f_idx, frame_dets in iter_90min_frame_batches(total_frames=total_frames, fps=25.0):
        exporter.append_frame_data(f_idx, frame_dets)
        if f_idx % 5000 == 0:
            memory_tracker.sample()

    team_map = {tid: "Team A" for tid in range(1, 10)}
    team_map.update({tid: "Team B" for tid in range(10, 19)})
    team_map[19] = "Referee"

    post_stats = exporter.postprocess(team_map=team_map)
    exporter.export_to_csv(str(csv_path))
    exporter.export_to_sql(str(db_path))
    out_dist = exporter.export_distance_summary(str(dist_path))
    memory_tracker.sample()

    assert post_stats["backfilled_teams"] == 5 * 18
    assert post_stats["interpolated_ball_frames"] > 0
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert db_path.is_file() and db_path.stat().st_size > 0
    assert out_dist is not None and Path(out_dist).is_file()
    assert len(pd.read_csv(out_dist)) == 19

    sql_rows = query_sqlite_count_readonly(db_path)
    assert sql_rows == 2_700_000, f"Expected 2,700,000 rows after ball gap interpolation, got {sql_rows}"

    del exporter
    gc.collect()
    memory_tracker.sample()

    # Stage 2: MatchAnalyzer.from_sql on 2.7M rows
    analyzer = MatchAnalyzer.from_sql(str(db_path), fps=25.0)
    ma_summary = analyzer.export_all(output_dir=str(ma_dir), generate_heatmaps=True)
    memory_tracker.sample()
    assert ma_summary["possession"]["mode"] == "pitch_meters"
    assert ma_summary["possession"]["total_frames"] == total_frames
    assert (ma_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME).is_file()
    assert (ma_dir / config.EXPORT_POSSESSION_FILENAME).is_file()

    del analyzer
    gc.collect()
    memory_tracker.sample()

    # Stage 3: MLAnalyzer.from_sql on 2.7M rows
    ml_analyzer = MLAnalyzer.from_sql(str(db_path), fps=25.0, analytics_summary=ma_summary)
    ml_summary = ml_analyzer.export_all(output_dir=str(ml_dir), generate_plots=True)
    memory_tracker.sample()

    assert ml_summary["data_quality"]["total_rows"] == sql_rows
    assert (ml_dir / config.EXPORT_ML_SUMMARY_FILENAME).is_file()
    assert (ml_dir / config.EXPORT_ML_REPORT_FILENAME).is_file()

    del ml_analyzer
    gc.collect()

    elapsed = time.perf_counter() - t0
    mem_stats = memory_tracker.summary()
    print(
        f"\n[90-Min Stress] rows={sql_rows}, elapsed={elapsed:.2f}s, "
        f"peak_rss_delta={mem_stats['peak_rss_delta_mb']}MB, "
        f"os_peak_wset_delta={mem_stats['os_peak_wset_delta_mb']}MB"
    )
    assert mem_stats["peak_rss_delta_mb"] < 1500.0, (
        f"Peak RSS delta too high for 90-min workload: {mem_stats}"
    )


def test_t4_scenario3_rtx4050_gpu_inference_and_90min_projection(synthetic_tv_broadcast_clip):
    """
    Scenario 3: Real CUDA RTX 4050 GPU inference & VRAM/FPS benchmark on trained
    4-class football YOLO model with 90-minute runtime projection (<= 6 GB VRAM).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable in this environment")

    model_path = Path(config.YOLO_MODEL_PATH)
    assert model_path.is_file(), f"Missing trained football model weights: {model_path}"

    cap = cv2.VideoCapture(str(synthetic_tv_broadcast_clip.video_path))
    frames = []
    try:
        while len(frames) < 120:
            ok, frm = cap.read()
            if not ok:
                break
            frames.append(frm)
    finally:
        cap.release()
    assert len(frames) >= 120

    detector = ObjectDetector(model_path=str(model_path), device="cuda")
    batch_size = max(1, int(config.DETECTION_BATCH_SIZE))

    # Warmup
    _ = detector.detect_batch(frames[:batch_size])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    total_dets = 0
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        batch_out = detector.detect_batch(batch)
        assert len(batch_out) == len(batch)
        for frame_dets in batch_out:
            total_dets += len(frame_dets)
            for d in frame_dets:
                assert d["class_name"] in set(config.TARGET_CLASS_IDS.values())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    fps = len(frames) / max(elapsed, 1e-6)
    projected_90min_min = (135_000 / fps) / 60.0

    print(
        f"\n[RTX 4050 GPU Benchmark] device={torch.cuda.get_device_name(0)}, "
        f"batch_size={batch_size}, frames={len(frames)}, fps={fps:.2f}, "
        f"projected_90min={projected_90min_min:.1f} min, peak_vram={peak_vram_gb:.3f} GiB, "
        f"total_dets={total_dets}"
    )
    assert peak_vram_gb <= 6.0, f"Peak VRAM exceeded 6 GB budget: {peak_vram_gb:.3f} GiB"
    assert fps > 5.0


def test_t4_scenario4_four_phase_tv_broadcast_recovery_accuracy(synthetic_tv_broadcast_clip):
    """
    Scenario 4: Full 4-phase TV broadcast pipeline recovery:
      Phase 1 (0..39): Main pan (< 1.5m error)
      Phase 2a (40..52): Hard cut to non-tactical -> None + is_anchored=False
      Phase 2b (53..64): Zoomed pitch-colored replay -> None + is_anchored=False
      Phase 3 (65..84): 20-frame gradual dissolve -> None + is_anchored=False
      Phase 4 (85..124): Zoomed/panned return to main camera -> auto re-anchor <= 15 frames & < 2.5m error
    """
    clip = synthetic_tv_broadcast_clip
    transformer = PerspectiveTransformer(source_points=clip.source_calibration_points)
    motion_tracker = CameraMotionTracker()
    cap = cv2.VideoCapture(str(clip.video_path))

    p1_max_err = 0.0
    p4_max_err = 0.0
    recovered_frame = None
    p4_start, p4_end = clip.phases["phase4_recovered_main"]
    cut_start = clip.phases["phase2a_hard_cut"][0]
    dissolve_end = clip.phases["phase3_gradual_dissolve"][1]

    try:
        for t in range(clip.total_frames):
            ok, frame = cap.read()
            assert ok
            dets = [dict(d) for d in clip.gt_detections_by_frame[t]]
            mat = motion_tracker.update(frame, dets)

            if t <= clip.phases["phase1_main_pan"][1]:
                assert mat is not None and motion_tracker.is_anchored is True
                out = transformer.add_pitch_coordinates(dets, camera_motion_matrix=mat)
                for item in out:
                    gx, gy = clip.gt_pitch_coords_by_frame[t][item["track_id"]]
                    p1_max_err = max(p1_max_err, math.hypot(item["pitch_x"] - gx, item["pitch_y"] - gy))
            elif cut_start <= t <= dissolve_end:
                assert mat is None, f"Leaked non-None homography during non-tactical frame {t}"
                assert motion_tracker.is_anchored is False
                out = transformer.add_pitch_coordinates(
                    [{"bbox": [100.0, 100.0, 120.0, 160.0]}],
                    camera_motion_matrix=mat,
                ) if mat is not None else [{"pitch_x": None, "pitch_y": None, "on_pitch": None}]
                assert out[0]["pitch_x"] is None
            else:
                # Phase 4: must recover within <= 15 frames and maintain < 2.5m accuracy
                if mat is not None and motion_tracker.is_anchored:
                    if recovered_frame is None:
                        recovered_frame = t
                    out = transformer.add_pitch_coordinates(dets, camera_motion_matrix=mat)
                    for item in out:
                        assert item["on_pitch"] is True
                        gx, gy = clip.gt_pitch_coords_by_frame[t][item["track_id"]]
                        p4_max_err = max(p4_max_err, math.hypot(item["pitch_x"] - gx, item["pitch_y"] - gy))
    finally:
        cap.release()

    assert p1_max_err < 1.5, f"Phase 1 error {p1_max_err:.2f}m >= 1.5m"
    assert recovered_frame is not None, "Failed to re-anchor on return to main tactical camera"
    assert recovered_frame <= p4_start + 15, f"Recovery took {recovered_frame - p4_start} frames (>15)"
    assert p4_max_err < 2.5, f"Phase 4 post-recovery error {p4_max_err:.2f}m >= 2.5m"


def test_t4_scenario5_secret_hygiene_and_baseline_immutability_audit():
    """
    Scenario 5: End-to-end Secret Hygiene (working tree + git log --all -p + .gitignore)
    and Baseline data/exports/tracking_data.{csv,db} SHA-256 immutability audit.
    """
    # 1) Zero tracked .pt weights & .gitignore covers weights/videos/env
    tracked_pt = subprocess.run(
        ["git", "ls-files", "*.pt"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert tracked_pt.stdout.strip() == ""
    for ignored_pattern in ["models/football_yolo11s_best.pt", ".env", "sample.mp4"]:
        ig = subprocess.run(
            ["git", "check-ignore", "-q", ignored_pattern],
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert ig.returncode == 0, f"{ignored_pattern} must be gitignored"

    # 2) Working tree + full git history secret scan
    violations = scan_secret_hygiene(REPO_ROOT)
    assert not violations, f"Secret hygiene violations found: {violations}"

    # 3) Baseline export immutability & row counts
    assert sha256_file(BASELINE_CSV) == BASELINE_CSV_SHA256
    assert sha256_file(BASELINE_DB) == BASELINE_DB_SHA256
    assert BASELINE_CSV.stat().st_size == 441546
    assert BASELINE_DB.stat().st_size == 688128
    assert len(pd.read_csv(BASELINE_CSV)) == BASELINE_ROW_COUNT
    assert query_sqlite_count_readonly(BASELINE_DB) == BASELINE_ROW_COUNT
