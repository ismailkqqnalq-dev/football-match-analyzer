"""Shared fixtures, synthetic generators, and safety guards for E2E tests."""

import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import pytest

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(tempfile.gettempdir(), "ultralytics_cfg"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config  # noqa: E402

BASELINE_CSV = REPO_ROOT / "data" / "exports" / "tracking_data.csv"
BASELINE_DB = REPO_ROOT / "data" / "exports" / "tracking_data.db"
BASELINE_CSV_SHA256 = "723fc5754cf5c777c524b37e8d9083e0c5f35cb3d27b83244ddf8aa25dff2982"
BASELINE_DB_SHA256 = "1667a4b102fb13e12a04b5c785f1721b4ddc29392cd8f188a74f0432acd40672"
BASELINE_ROW_COUNT = 6042

# Generic token detector (requires upper + lower + digit, 16..40 chars)
_MIXED_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9]{16,40}\b)(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z0-9]+\b"
)
_CONTEXT_KEY_RE = re.compile(r"(?:roboflow|api[_-]?key|apikey)", re.IGNORECASE)
_HARDCODED_RF_CALL_RE = re.compile(
    r"""Roboflow\s*\(\s*api_key\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)
_CLI_API_KEY_LITERAL_RE = re.compile(
    r"""--api-key(?:\s+|=)["']?([A-Za-z0-9_\-]{12,})["']?""",
    re.IGNORECASE,
)

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "runs",
    "datasets",
}
_BINARY_SUFFIXES = {
    ".pt",
    ".onnx",
    ".engine",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".png",
    ".jpg",
    ".jpeg",
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".so",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def baseline_data_immutability_guard():
    """Verify baseline data/exports/tracking_data.{csv,db} are untouched before and after E2E."""
    assert BASELINE_CSV.is_file(), f"Missing baseline CSV: {BASELINE_CSV}"
    assert BASELINE_DB.is_file(), f"Missing baseline DB: {BASELINE_DB}"
    before_csv = sha256_file(BASELINE_CSV)
    before_db = sha256_file(BASELINE_DB)
    assert before_csv == BASELINE_CSV_SHA256, (
        f"Baseline CSV hash mismatch BEFORE E2E session: {before_csv}"
    )
    assert before_db == BASELINE_DB_SHA256, (
        f"Baseline DB hash mismatch BEFORE E2E session: {before_db}"
    )
    yield
    after_csv = sha256_file(BASELINE_CSV)
    after_db = sha256_file(BASELINE_DB)
    assert after_csv == BASELINE_CSV_SHA256, (
        f"Baseline CSV was MUTATED during E2E session! after={after_csv}"
    )
    assert after_db == BASELINE_DB_SHA256, (
        f"Baseline DB was MUTATED during E2E session! after={after_db}"
    )


@pytest.fixture
def isolated_export_dirs(tmp_path, monkeypatch):
    """Redirect config.EXPORT_* paths to tmp_path for in-process safety."""
    export_dir = tmp_path / "exports"
    heatmap_dir = export_dir / "heatmaps"
    ml_dir = export_dir / "ml_reports"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "EXPORT_DIR", str(export_dir))
    monkeypatch.setattr(config, "EXPORT_HEATMAP_DIR", str(heatmap_dir))
    monkeypatch.setattr(config, "EXPORT_ML_DIR", str(ml_dir))
    return export_dir


class ProcessMemoryTracker:
    """psutil-based RSS and OS peak_wset tracker (safe for 2.7M-row workloads)."""

    def __init__(self):
        self._proc = psutil.Process(os.getpid())
        mem = self._proc.memory_info()
        self.rss_start_mb = mem.rss / (1024 * 1024)
        self.peak_rss_mb = self.rss_start_mb
        self.peak_wset_start_mb = getattr(mem, "peak_wset", mem.rss) / (1024 * 1024)

    def sample(self) -> float:
        rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        if rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = rss_mb
        return rss_mb

    def summary(self) -> Dict[str, float]:
        mem = self._proc.memory_info()
        rss_end_mb = mem.rss / (1024 * 1024)
        if rss_end_mb > self.peak_rss_mb:
            self.peak_rss_mb = rss_end_mb
        os_peak_wset_mb = getattr(mem, "peak_wset", mem.rss) / (1024 * 1024)
        return {
            "rss_start_mb": round(self.rss_start_mb, 2),
            "rss_end_mb": round(rss_end_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_rss_delta_mb": round(self.peak_rss_mb - self.rss_start_mb, 2),
            "os_peak_wset_mb": round(os_peak_wset_mb, 2),
            "os_peak_wset_delta_mb": round(os_peak_wset_mb - self.rss_start_mb, 2),
        }


@pytest.fixture
def memory_tracker() -> ProcessMemoryTracker:
    return ProcessMemoryTracker()


# ----------------------------------------------------------------------
# Synthetic TV Broadcast Clip & Ground-Truth Homography Generator
# ----------------------------------------------------------------------

@dataclass
class SyntheticBroadcastClip:
    video_path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    source_calibration_points: Dict[str, Tuple[float, float]]
    h_pitch_to_frame0: np.ndarray
    gt_homographies_to_frame0: Dict[int, np.ndarray]
    gt_detections_by_frame: Dict[int, List[Dict]]
    gt_pitch_coords_by_frame: Dict[int, Dict[int | str, Tuple[float, float]]]
    phases: Dict[str, Tuple[int, int]]


def _project_points(h_mat: np.ndarray, pts_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    arr = np.array([pts_xy], dtype=np.float64)
    out = cv2.perspectiveTransform(arr, h_mat.astype(np.float64))[0]
    return [(float(x), float(y)) for x, y in out]


def _make_camera_matrix(cx: float, cy: float, frame_w: int, frame_h: int, scale: float) -> np.ndarray:
    return np.array(
        [
            [scale, 0.0, (frame_w * 0.5) - scale * cx],
            [0.0, scale, (frame_h * 0.5) - scale * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _create_master_pitch_canvas(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a deterministic 2400x1300 textured football pitch + perimeter ad-boards
    and return (canvas_bgr, H_pitch_to_canvas).
    """
    rng = np.random.default_rng(seed)
    canvas_w, canvas_h = 2400, 1300

    # Base textured green turf with mowing stripes + deterministic micro-texture
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    stripe_width = 90
    for sx in range(0, canvas_w, stripe_width):
        color = (34, 132, 34) if (sx // stripe_width) % 2 == 0 else (40, 146, 40)
        canvas[:, sx : min(canvas_w, sx + stripe_width)] = color

    # Subtle deterministic grass texture + scattered turf markers so goodFeaturesToTrack finds 200+ corners
    noise = rng.integers(-14, 15, size=(canvas_h, canvas_w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(450):
        tx = int(rng.integers(40, canvas_w - 40))
        ty = int(rng.integers(240, canvas_h - 40))
        shade = (22, 108, 22) if rng.random() < 0.5 else (52, 168, 52)
        cv2.rectangle(canvas, (tx - 3, ty - 2), (tx + 3, ty + 2), shade, -1)

    # Perimeter stadium ad-boards along top (y: 120..220) with crisp corners
    cv2.rectangle(canvas, (150, 120), (2250, 225), (30, 30, 45), -1)
    ad_colors = [(220, 220, 220), (40, 180, 240), (220, 80, 40), (40, 200, 120)]
    for idx, ax in enumerate(range(170, 2210, 140)):
        c = ad_colors[idx % len(ad_colors)]
        cv2.rectangle(canvas, (ax, 135), (ax + 110, 210), c, 2)
        cv2.line(canvas, (ax + 10, 150), (ax + 95, 195), c, 2)

    # Known perspective trapezoid mapping FIFA pitch (0..105, 0..68) -> canvas
    pitch_corners = np.array(
        [[0.0, 0.0], [105.0, 0.0], [0.0, 68.0], [105.0, 68.0]],
        dtype=np.float64,
    )
    canvas_corners = np.array(
        [[560.0, 310.0], [1840.0, 310.0], [380.0, 1010.0], [2020.0, 1010.0]],
        dtype=np.float64,
    )
    h_pitch_to_canvas = cv2.getPerspectiveTransform(
        pitch_corners.astype(np.float32), canvas_corners.astype(np.float32)
    ).astype(np.float64)

    # Draw crisp white FIFA pitch lines via H_pitch_to_canvas
    white = (245, 245, 245)

    def draw_pitch_line(p1: Tuple[float, float], p2: Tuple[float, float], thickness: int = 3):
        q1, q2 = _project_points(h_pitch_to_canvas, [p1, p2])
        cv2.line(
            canvas,
            (int(round(q1[0])), int(round(q1[1]))),
            (int(round(q2[0])), int(round(q2[1]))),
            white,
            thickness,
        )

    # Outer pitch rectangle
    draw_pitch_line((0.0, 0.0), (105.0, 0.0), 4)
    draw_pitch_line((105.0, 0.0), (105.0, 68.0), 4)
    draw_pitch_line((105.0, 68.0), (0.0, 68.0), 4)
    draw_pitch_line((0.0, 68.0), (0.0, 0.0), 4)
    # Halfway line
    draw_pitch_line((52.5, 0.0), (52.5, 68.0), 3)
    # Left penalty box & six-yard box
    draw_pitch_line((0.0, 13.84), (16.5, 13.84), 3)
    draw_pitch_line((16.5, 13.84), (16.5, 54.16), 3)
    draw_pitch_line((16.5, 54.16), (0.0, 54.16), 3)
    draw_pitch_line((0.0, 24.84), (5.5, 24.84), 3)
    draw_pitch_line((5.5, 24.84), (5.5, 43.16), 3)
    draw_pitch_line((5.5, 43.16), (0.0, 43.16), 3)
    # Right penalty box & six-yard box
    draw_pitch_line((105.0, 13.84), (88.5, 13.84), 3)
    draw_pitch_line((88.5, 13.84), (88.5, 54.16), 3)
    draw_pitch_line((88.5, 54.16), (105.0, 54.16), 3)
    draw_pitch_line((105.0, 24.84), (99.5, 24.84), 3)
    draw_pitch_line((99.5, 24.84), (99.5, 43.16), 3)
    draw_pitch_line((99.5, 43.16), (105.0, 43.16), 3)
    # Center circle polygon
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circle_pts_m = [
        (52.5 + 9.15 * float(np.cos(a)), 34.0 + 9.15 * float(np.sin(a)))
        for a in angles
    ]
    circle_pts_px = _project_points(h_pitch_to_canvas, circle_pts_m)
    poly = np.array([[int(round(x)), int(round(y))] for x, y in circle_pts_px], dtype=np.int32)
    cv2.polylines(canvas, [poly], isClosed=True, color=white, thickness=3)

    return canvas, h_pitch_to_canvas


def _overlay_scorebug(frame: np.ndarray) -> np.ndarray:
    """Draw a static TV broadcast scorebug overlay in top-left."""
    out = frame.copy()
    cv2.rectangle(out, (20, 15), (275, 55), (20, 20, 25), -1)
    cv2.rectangle(out, (20, 15), (275, 55), (235, 235, 235), 2)
    cv2.putText(
        out,
        "MATCH 12:34 | A 1 - 0 B",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
    )
    return out


def build_synthetic_tv_broadcast_clip(
    out_dir: Path,
    filename: str = "tv_broadcast.mp4",
    width: int = 960,
    height: int = 540,
    fps: float = 25.0,
) -> SyntheticBroadcastClip:
    """
    Generate a deterministic 125-frame, 4-phase TV broadcast MP4 in out_dir:
      - Frames  0..39 : Phase 1 — Main tactical camera smooth pan
      - Frames 40..52 : Phase 2a — Hard cut to non-tactical crowd/close-up
      - Frames 53..64 : Phase 2b — Cut to zoomed (2.8x) pitch-colored replay
      - Frames 65..84 : Phase 3 — 20-frame gradual dissolve into off-angle sideline shot
      - Frames 85..124: Phase 4 — Zoomed/panned return to main tactical camera
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / filename

    rng = np.random.default_rng(42)
    master_canvas, h_pitch_to_canvas = _create_master_pitch_canvas(seed=42)

    # Non-tactical crowd/close-up frame (warm non-green tones, high contrast stripes)
    non_tactical_base = np.full((height, width, 3), (45, 55, 175), dtype=np.uint8)
    for row in range(0, height, 36):
        c = (210, 190, 230) if (row // 36) % 2 == 0 else (35, 45, 160)
        cv2.rectangle(non_tactical_base, (0, row), (width, min(height, row + 18)), c, -1)
    for _ in range(90):
        cx = int(rng.integers(30, width - 30))
        cy = int(rng.integers(60, height - 30))
        cv2.circle(non_tactical_base, (cx, cy), 14, (70, 140, 240), -1)

    # Off-angle sideline close-up target for gradual dissolve
    sideline_shot = np.full((height, width, 3), (65, 35, 30), dtype=np.uint8)
    for col in range(0, width, 48):
        cv2.rectangle(sideline_shot, (col, 0), (col + 24, height), (190, 170, 150), -1)
    cv2.circle(sideline_shot, (width // 2, height // 2), 95, (40, 90, 210), -1)

    # Zoomed pitch-colored replay (2.8x zoom + rotated + color shifted)
    replay_center = (1180.0, 680.0)
    replay_scale = 0.52 * 2.8
    rot = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), 14.0, 1.0)
    rot_3x3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    t_replay_base = rot_3x3 @ _make_camera_matrix(
        replay_center[0], replay_center[1], width, height, replay_scale
    )
    replay_pitch_shot = cv2.warpPerspective(master_canvas, t_replay_base, (width, height))
    # Subtle warm lighting shift while remaining predominantly pitch-green
    replay_pitch_shot = np.clip(
        replay_pitch_shot.astype(np.int16) + np.array([15, 8, 22], dtype=np.int16),
        0,
        255,
    ).astype(np.uint8)

    # Frame 0 main camera matrix
    base_cx, base_cy = 1200.0, 660.0
    base_scale = 0.52
    t_cam_0 = _make_camera_matrix(base_cx, base_cy, width, height, base_scale)
    h_pitch_to_frame0 = t_cam_0 @ h_pitch_to_canvas

    calibration_names = [
        "top_left_corner",
        "top_right_corner",
        "bottom_left_corner",
        "bottom_right_corner",
        "halfway_top",
        "halfway_bottom",
        "center_spot",
        "left_penalty_box_top_right",
        "right_penalty_box_top_left",
    ]
    source_calibration_points: Dict[str, Tuple[float, float]] = {}
    pts_m = [config.PITCH_LANDMARKS_METERS[n] for n in calibration_names]
    pts_px = _project_points(h_pitch_to_frame0, pts_m)
    for name, (px, py) in zip(calibration_names, pts_px):
        source_calibration_points[name] = (round(px, 2), round(py, 2))

    # Moving actors in pitch meters: (track_id, class_name, team, base_x, base_y, vx, vy, bgr)
    actors = [
        (1, "player", "Team A", 28.0, 22.0, 0.14, 0.04, (160, 40, 20)),
        (2, "player", "Team A", 40.0, 44.0, 0.12, -0.05, (160, 40, 20)),
        (3, "player", "Team A", 50.0, 32.0, 0.10, 0.03, (160, 40, 20)),
        (4, "player", "Team B", 58.0, 24.0, -0.11, 0.04, (240, 240, 240)),
        (5, "player", "Team B", 70.0, 46.0, -0.13, -0.03, (240, 240, 240)),
        (6, "goalkeeper", "Goalkeeper", 6.0, 34.0, 0.01, 0.05, (230, 50, 220)),
        (7, "referee", "Referee", 52.0, 18.0, 0.06, 0.02, (0, 230, 245)),
        ("ball", "ball", None, 48.0, 34.0, 0.18, -0.02, (0, 180, 255)),
    ]

    total_frames = 125
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter at {video_path}"

    gt_homographies_to_frame0: Dict[int, np.ndarray] = {}
    gt_detections_by_frame: Dict[int, List[Dict]] = {}
    gt_pitch_coords_by_frame: Dict[int, Dict[int | str, Tuple[float, float]]] = {}

    try:
        for t in range(total_frames):
            is_main = (0 <= t <= 39) or (85 <= t <= 124)

            # Compute actor meter positions
            frame_canvas = master_canvas.copy()
            actor_meta = []
            for tid, cls_name, team, bx0, by0, vx, vy, bgr in actors:
                mx = float(np.clip(bx0 + vx * t, 2.0, 103.0))
                my = float(np.clip(by0 + vy * t, 2.0, 66.0))
                cx_px, cy_px = _project_points(h_pitch_to_canvas, [(mx, my)])[0]
                if cls_name == "ball":
                    cv2.circle(frame_canvas, (int(round(cx_px)), int(round(cy_px - 6))), 8, bgr, -1)
                else:
                    # Draw simple torso + legs on master canvas so feet touch (cx_px, cy_px)
                    x1 = int(round(cx_px - 14))
                    x2 = int(round(cx_px + 14))
                    y1 = int(round(cy_px - 58))
                    y2 = int(round(cy_px))
                    cv2.rectangle(frame_canvas, (x1, y1), (x2, y2 - 20), bgr, -1)
                    cv2.rectangle(frame_canvas, (x1 + 4, y2 - 20), (x2 - 4, y2), (30, 30, 30), -1)
                actor_meta.append((tid, cls_name, team, mx, my))

            if 0 <= t <= 39:
                # Phase 1: main tactical camera smooth pan
                t_cam = _make_camera_matrix(
                    base_cx + t * 2.5, base_cy + t * 0.6, width, height, base_scale
                )
                frame = cv2.warpPerspective(frame_canvas, t_cam, (width, height))
            elif 40 <= t <= 52:
                # Phase 2a: hard cut to non-tactical crowd/close-up
                shift = (t - 40) * 3
                frame = np.roll(non_tactical_base, shift, axis=1)
                t_cam = None
            elif 53 <= t <= 64:
                # Phase 2b: zoomed pitch-colored replay (different scale & angle)
                step = t - 53
                t_rep = _make_camera_matrix(
                    replay_center[0] + step * 4.0,
                    replay_center[1] - step * 2.0,
                    width,
                    height,
                    replay_scale,
                )
                frame = cv2.warpPerspective(frame_canvas, rot_3x3 @ t_rep, (width, height))
                frame = np.clip(
                    frame.astype(np.int16) + np.array([15, 8, 22], dtype=np.int16),
                    0,
                    255,
                ).astype(np.uint8)
                t_cam = None
            elif 65 <= t <= 84:
                # Phase 3: 20-frame gradual linear dissolve into sideline shot
                alpha = float(t - 64) / 20.0
                frame = cv2.addWeighted(replay_pitch_shot, 1.0 - alpha, sideline_shot, alpha, 0.0)
                t_cam = None
            else:
                # Phase 4: return to main tactical camera with pan offset + 1.08x zoom
                step = t - 85
                t_cam = _make_camera_matrix(
                    base_cx - 75.0 + step * 2.0,
                    base_cy - 18.0 + step * 0.4,
                    width,
                    height,
                    base_scale * 1.08,
                )
                frame = cv2.warpPerspective(frame_canvas, t_cam, (width, height))

            frame = _overlay_scorebug(frame)
            writer.write(frame)

            if is_main and t_cam is not None:
                m_gt = t_cam_0 @ np.linalg.inv(t_cam)
                m_gt = m_gt / m_gt[2, 2]
                gt_homographies_to_frame0[t] = m_gt

                h_pitch_to_frame_t = t_cam @ h_pitch_to_canvas
                dets: List[Dict] = []
                gt_pts: Dict[int | str, Tuple[float, float]] = {}
                for tid, cls_name, team, mx, my in actor_meta:
                    fx, fy = _project_points(h_pitch_to_frame_t, [(mx, my)])[0]
                    if cls_name == "ball":
                        bbox = [fx - 6.0, fy - 12.0, fx + 6.0, fy]
                    else:
                        bbox = [fx - 12.0, fy - 42.0, fx + 12.0, fy]
                    dets.append(
                        {
                            "track_id": tid,
                            "class_id": {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}[cls_name],
                            "class_name": cls_name,
                            "team": team,
                            "confidence": 0.92,
                            "is_predicted": False,
                            "bbox": bbox,
                        }
                    )
                    gt_pts[tid] = (mx, my)
                gt_detections_by_frame[t] = dets
                gt_pitch_coords_by_frame[t] = gt_pts
            else:
                gt_detections_by_frame[t] = []
                gt_pitch_coords_by_frame[t] = {}
    finally:
        writer.release()

    return SyntheticBroadcastClip(
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        source_calibration_points=source_calibration_points,
        h_pitch_to_frame0=h_pitch_to_frame0,
        gt_homographies_to_frame0=gt_homographies_to_frame0,
        gt_detections_by_frame=gt_detections_by_frame,
        gt_pitch_coords_by_frame=gt_pitch_coords_by_frame,
        phases={
            "phase1_main_pan": (0, 39),
            "phase2a_hard_cut": (40, 52),
            "phase2b_zoomed_pitch_replay": (53, 64),
            "phase3_gradual_dissolve": (65, 84),
            "phase4_recovered_main": (85, 124),
        },
    )


@pytest.fixture(scope="session")
def synthetic_tv_broadcast_clip(tmp_path_factory) -> SyntheticBroadcastClip:
    clip_dir = tmp_path_factory.mktemp("tv_broadcast_clip")
    return build_synthetic_tv_broadcast_clip(clip_dir)


def build_short_tactical_clip(
    out_dir: Path,
    filename: str = "short_tactical.mp4",
    n_frames: int = 25,
    width: int = 960,
    height: int = 540,
    fps: float = 25.0,
) -> Path:
    """Create a fast synthetic tactical video clip in out_dir for CLI/VideoProcessor E2E tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / filename
    master_canvas, _ = _create_master_pitch_canvas(seed=77)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open VideoWriter at {video_path}"
    try:
        for t in range(n_frames):
            t_cam = _make_camera_matrix(1200.0 + t * 1.5, 660.0, width, height, 0.52)
            frame = cv2.warpPerspective(master_canvas, t_cam, (width, height))
            frame = _overlay_scorebug(frame)
            writer.write(frame)
    finally:
        writer.release()
    return video_path


# ----------------------------------------------------------------------
# Fast 90-Minute Synthetic Workload Helpers (135,000 frames / 2.7M rows)
# ----------------------------------------------------------------------

def iter_90min_frame_batches(
    total_frames: int = 135_000,
    fps: float = 25.0,
    ball_gap_period: int = 500,
) -> Iterator[Tuple[int, List[Dict]]]:
    """
    Yield (frame_idx, all_tracked_dicts) for 135,000 frames x 20 objects = 2,700,000 rows:
      - 9 Team A players (track_id 1..9)
      - 9 Team B players (track_id 10..18)
      - 1 Referee (track_id 19)
      - 1 Ball (track_id "ball", with periodic 2-frame detection gaps for interpolation)
    """
    # Precompute static metadata & periodic base coordinates per actor
    actor_specs = []
    for tid in range(1, 10):
        bx = 12.0 + ((tid - 1) % 3) * 18.0
        by = 12.0 + ((tid - 1) // 3) * 20.0
        actor_specs.append((tid, "player", "Team A", bx, by, 0.03 + 0.004 * tid, 3.0))
    for tid in range(10, 19):
        idx = tid - 10
        bx = 55.0 + (idx % 3) * 18.0
        by = 12.0 + (idx // 3) * 20.0
        actor_specs.append((tid, "player", "Team B", bx, by, 0.03 + 0.004 * idx, 3.0))
    actor_specs.append((19, "referee", "Referee", 52.5, 34.0, 0.02, 5.0))

    # Precompute small trig lookup table (period = 1000 frames) for ultra-fast generation
    period = 1000
    t_arr = np.arange(period, dtype=np.float64)
    sin_tbl = np.sin(t_arr * 0.05).tolist()
    cos_tbl = np.cos(t_arr * 0.04).tolist()

    for f in range(total_frames):
        idx = f % period
        s_val = sin_tbl[idx]
        c_val = cos_tbl[idx]

        # Ball orbits near active carrier to produce valid passes/possession
        carrier_idx = (f // 30) % 18
        _, _, _, cbx, cby, _, cr = actor_specs[carrier_idx]
        ball_x = round(max(1.0, min(104.0, cbx + cr * s_val + 0.4)), 2)
        ball_y = round(max(1.0, min(67.0, cby + cr * c_val + 0.2)), 2)

        frame_dets: List[Dict] = []
        # Skip ball on frames f % 500 in {10, 11} to test 2-frame ball interpolation
        if not (ball_gap_period > 0 and (f % ball_gap_period) in (10, 11) and f > 0):
            bx_px = ball_x * 10.0
            by_px = ball_y * 10.0
            frame_dets.append(
                {
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.88,
                    "is_predicted": False,
                    "bbox": [bx_px, by_px, bx_px + 10.0, by_px + 10.0],
                    "pitch_x": ball_x,
                    "pitch_y": ball_y,
                    "on_pitch": True,
                }
            )

        for tid, cls_name, team, bx0, by0, _, rad in actor_specs:
            px = round(max(1.0, min(104.0, bx0 + rad * s_val)), 2)
            py = round(max(1.0, min(67.0, by0 + rad * c_val)), 2)
            px_pix = px * 10.0
            py_pix = py * 10.0
            frame_dets.append(
                {
                    "track_id": tid,
                    "class_name": cls_name,
                    # First 5 frames omit team on players to exercise cross-chunk backfill
                    "team": None if (f < 5 and cls_name == "player") else team,
                    "confidence": 0.91,
                    "is_predicted": False,
                    "bbox": [px_pix, py_pix, px_pix + 24.0, py_pix + 64.0],
                    "pitch_x": px,
                    "pitch_y": py,
                    "on_pitch": True,
                }
            )
        yield f, frame_dets


# ----------------------------------------------------------------------
# Secret Hygiene Scanner (Zero Secret Literals Embedded)
# ----------------------------------------------------------------------

def scan_secret_hygiene(repo_root: Path = REPO_ROOT) -> List[str]:
    """
    Scan working tree text files and `git log --all -p` for:
      1) Non-empty ROBOFLOW_API_KEY env value (if set, NEVER reported in error message).
      2) Mixed-case alphanumeric tokens (16..40 chars, upper+lower+digit) on lines
         referencing roboflow / api_key / api-key.
      3) Hardcoded Roboflow(api_key="...") or `--api-key <token>` literals.
    Returns list of sanitized violation descriptions (empty == clean).
    """
    violations: List[str] = []
    env_secret = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    def _scan_text(label: str, text: str) -> None:
        if env_secret and len(env_secret) >= 8 and env_secret in text:
            violations.append(f"{label}: contains active ROBOFLOW_API_KEY env value")
        for line_no, line in enumerate(text.splitlines(), start=1):
            m_call = _HARDCODED_RF_CALL_RE.search(line)
            if m_call:
                val = m_call.group(1)
                if val not in {"...", "YOUR_API_KEY", "ROBOFLOW_API_KEY"} and len(val) >= 12:
                    violations.append(f"{label}:{line_no}: hardcoded Roboflow(api_key=...) literal")

            m_cli = _CLI_API_KEY_LITERAL_RE.search(line)
            if m_cli:
                val = m_cli.group(1)
                is_upper_placeholder = val.isupper() and ("_" in val or "KEY" in val)
                if not is_upper_placeholder and val.lower() not in {
                    "your_api_key",
                    "placeholder",
                    "test",
                    "dummy",
                    "senin_roboflow_api_keyin",
                }:
                    violations.append(f"{label}:{line_no}: literal --api-key token argument")

            if _CONTEXT_KEY_RE.search(line):
                for tok in _MIXED_TOKEN_RE.findall(line):
                    # Exclude known benign test fixture names / identifiers
                    if tok.lower() in {"roboflowapikey", "testroboflowkey123"}:
                        continue
                    violations.append(
                        f"{label}:{line_no}: suspicious mixed-case token near api_key/roboflow"
                    )

    # 1) Walk working tree
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in _BINARY_SUFFIXES:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = fpath.relative_to(repo_root)
            _scan_text(f"working_tree/{rel}", content)

    # 2) Scan git log --all -p (only when repo_root is a git repository)
    if (repo_root / ".git").exists():
        git_res = subprocess.run(
            ["git", "log", "--all", "-p", "--no-color"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if git_res.returncode == 0 and git_res.stdout:
            current_commit = "unknown"
            for line_no, line in enumerate(git_res.stdout.splitlines(), start=1):
                if line.startswith("commit "):
                    current_commit = line.split()[1][:12]
                    continue
                if env_secret and len(env_secret) >= 8 and env_secret in line:
                    violations.append(
                        f"git_history/{current_commit}: contains active ROBOFLOW_API_KEY"
                    )
                m_call = _HARDCODED_RF_CALL_RE.search(line)
                if m_call:
                    val = m_call.group(1)
                    if val not in {"...", "YOUR_API_KEY", "ROBOFLOW_API_KEY"} and len(val) >= 12:
                        violations.append(
                            f"git_history/{current_commit}:{line_no}: hardcoded Roboflow(api_key=...)"
                        )
                m_cli = _CLI_API_KEY_LITERAL_RE.search(line)
                if m_cli:
                    val = m_cli.group(1)
                    is_upper_placeholder = val.isupper() and ("_" in val or "KEY" in val)
                    if not is_upper_placeholder and val.lower() not in {
                        "your_api_key",
                        "placeholder",
                        "test",
                        "dummy",
                        "senin_roboflow_api_keyin",
                    }:
                        violations.append(
                            f"git_history/{current_commit}:{line_no}: literal --api-key token argument"
                        )
                if _CONTEXT_KEY_RE.search(line):
                    for _ in _MIXED_TOKEN_RE.findall(line):
                        violations.append(
                            f"git_history/{current_commit}:{line_no}: suspicious mixed-case token near api_key/roboflow"
                        )

    return violations


def query_sqlite_count_readonly(db_path: Path, table_name: str = "tracking_data") -> int:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        return int(cur.fetchone()[0])
    finally:
        conn.close()
