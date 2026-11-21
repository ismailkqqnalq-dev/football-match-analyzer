"""
test_r3_camera_recovery.py
--------------------------
R3: TV yayını sahne kesmesi, kademeli geçiş (dissolve), çim renkli tekrar
kesmesi, ana kameraya dönüşte otomatik yeniden hizalanma (recovery),
uzun çekim kayma (drift) kararlılığı ve PerspectiveTransformer dejenere
homografi korumaları için deterministik sentetik test paketi.
"""

import hashlib
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.camera_motion_tracker import CameraMotionTracker
from modules.perspective_transformer import PerspectiveTransformer


# ----------------------------------------------------------------------
# Sentetik yayın sahası ve kamera oluşturucu yardımcılar
# ----------------------------------------------------------------------
PITCH_CORNERS_CANVAS = {
    "top_left_corner": (120.0, 180.0),
    "top_right_corner": (1680.0, 180.0),
    "bottom_left_corner": (120.0, 1020.0),
    "bottom_right_corner": (1680.0, 1020.0),
    "halfway_top": (900.0, 180.0),
    "halfway_bottom": (900.0, 1020.0),
    "center_spot": (900.0, 600.0),
}


def _make_virtual_pitch_canvas(
    width: int = 1800, height: int = 1100, seed: int = 42
) -> np.ndarray:
    """
    Zengin dokulu yeşil çim, FIFA saha çizgileri ve üst tribün/reklam panosu
    şeridi içeren deterministik sanal futbol sahası tuvali üretir.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Üst tribün / reklam panosu şeridi (y < 140)
    canvas[:140, :] = (55, 50, 65)
    for x in range(0, width, 90):
        color = tuple(int(c) for c in rng.integers(70, 220, size=3))
        cv2.rectangle(canvas, (x + 4, 20), (min(width - 2, x + 84), 125), color, -1)
        cv2.putText(
            canvas,
            f"AD{(x // 90) % 10}",
            (x + 16, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )

    # Çim zemini + biçme şeritleri (y >= 140)
    canvas[140:, :] = (36, 138, 36)
    band_w = 130
    for bx in range(0, width, band_w * 2):
        canvas[140:, bx : min(width, bx + band_w)] = (42, 152, 42)

    # Çok ölçekli deterministik çim dokusu (LK + ORB için doğal özellik noktaları)
    noise = rng.integers(-14, 15, size=(height - 140, width, 3), dtype=np.int16)
    turf = np.clip(canvas[140:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    canvas[140:, :] = turf

    for _ in range(900):
        tx = int(rng.integers(20, width - 20))
        ty = int(rng.integers(150, height - 20))
        tw = int(rng.integers(3, 8))
        g_delta = int(rng.integers(-22, 26))
        col = (
            max(10, min(90, 38 + g_delta // 2)),
            max(80, min(200, 142 + g_delta)),
            max(10, min(90, 38 + g_delta // 2)),
        )
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + tw), col, -1)

    # FIFA saha çizgileri (dış sınır, orta çizgi, orta yuvarlak, ceza sahaları)
    white = (242, 245, 242)
    cv2.rectangle(canvas, (120, 180), (1680, 1020), white, 4)
    cv2.line(canvas, (900, 180), (900, 1020), white, 4)
    cv2.circle(canvas, (900, 600), 136, white, 4)
    cv2.circle(canvas, (900, 600), 6, white, -1)

    # Sol ve sağ ceza sahaları + kale sahaları
    cv2.rectangle(canvas, (120, 351), (365, 849), white, 4)
    cv2.rectangle(canvas, (120, 487), (202, 713), white, 4)
    cv2.circle(canvas, (283, 600), 5, white, -1)

    cv2.rectangle(canvas, (1435, 351), (1680, 849), white, 4)
    cv2.rectangle(canvas, (1598, 487), (1680, 713), white, 4)
    cv2.circle(canvas, (1517, 600), 5, white, -1)

    return canvas


def _build_camera_view_matrix(
    out_size: tuple = (960, 540),
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """
    Sanal tuval (1800x1100) koordinatlarını out_size yayın karesi piksellerine
    taşıyan analitik perspektif projeksiyon matrisi (H_view) üretir.
    """
    out_w, out_h = float(out_size[0]), float(out_size[1])
    base_scale_x = out_w / 1450.0
    base_scale_y = out_h / 860.0
    base_scale = min(base_scale_x, base_scale_y)

    # Tuval merkezi (900, 580) -> kare merkezi (out_w/2, out_h/2)
    t_center = np.array(
        [[1.0, 0.0, -900.0], [0.0, 1.0, -580.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    # Hafif gerçekçi yayın perspektif eğimi
    persp = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.00018, 1.0]], dtype=np.float64
    )
    s_base = np.diag([base_scale * zoom, base_scale * zoom, 1.0])
    t_out = np.array(
        [
            [1.0, 0.0, 0.5 * out_w - pan_x],
            [0.0, 1.0, 0.5 * out_h - pan_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    H_view = t_out @ s_base @ persp @ t_center
    return H_view / H_view[2, 2]


def _render_broadcast_view(
    canvas: np.ndarray,
    H_view: np.ndarray,
    out_size: tuple = (960, 540),
    add_scorebug: bool = False,
) -> np.ndarray:
    frame = cv2.warpPerspective(
        canvas,
        H_view,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if add_scorebug:
        cv2.rectangle(frame, (20, 15), (290, 65), (25, 25, 35), -1)
        cv2.rectangle(frame, (20, 15), (290, 65), (235, 235, 235), 2)
        cv2.putText(
            frame,
            "LIVE 45:00 | A 1 - 0 B",
            (32, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return frame


def _project_canvas_landmarks(H_view: np.ndarray) -> dict:
    pts = np.array(
        [[[x, y]] for x, y in PITCH_CORNERS_CANVAS.values()], dtype=np.float64
    )
    proj = cv2.perspectiveTransform(pts, H_view)[:, 0, :]
    return {
        name: (float(px), float(py))
        for name, (px, py) in zip(PITCH_CORNERS_CANVAS.keys(), proj)
    }


def _make_replay_frame(
    kind: str,
    frame_idx: int,
    size: tuple = (960, 540),
    canvas: np.ndarray = None,
) -> np.ndarray:
    """Farklı TV yayın kesmesi senaryoları üretir."""
    w, h = size
    rng = np.random.default_rng(1000 + frame_idx)

    if kind == "closeup_red":
        frame = np.full((h, w, 3), (28, 30, 145), dtype=np.uint8)
        shift = (frame_idx * 14) % max(w // 2, 1)
        cv2.circle(frame, (w // 3 + shift // 2, h // 2), 115, (30, 45, 210), -1)
        cv2.rectangle(
            frame, (50 + shift, 80), (260 + shift, 440), (20, 20, 95), -1
        )
        cv2.putText(
            frame,
            "REPLAY",
            (w - 220, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
        )
        return frame

    if kind == "pitch_colored_replay":
        # Histogramı ana kamera çimine benzeyen ama tamamen farklı yakın çekim düzeni
        assert canvas is not None
        crop_x = 650 + (frame_idx * 18) % 240
        crop_y = 420 + (frame_idx * 9) % 120
        crop = canvas[crop_y : crop_y + 220, crop_x : crop_x + 380]
        frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        # Büyük yakın çekim oyuncu gövdesi / gölge
        cv2.ellipse(frame, (w // 2, h // 2 + 40), (140, 180), 0, 0, 360, (35, 95, 165), -1)
        cv2.circle(frame, (w // 2, h // 2 - 140), 55, (140, 170, 205), -1)
        return frame

    if kind == "green_closeup_nontactical":
        # Yakın çekim saf çim dokusu (çizgisiz / ana kamera geometrisi yok)
        frame = np.full((h, w, 3), (38, 145, 38), dtype=np.uint8)
        noise = rng.integers(-18, 19, size=(h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for _ in range(180):
            gx = int(rng.integers(20, w - 20))
            gy = int(rng.integers(20, h - 20))
            cv2.line(frame, (gx, gy), (gx + 4, gy - 12), (48, 172, 48), 2)
        return frame

    if kind == "green_crowd_nontactical":
        # Yeşil ağırlıklı gürültülü tribün/pankart (ana kamera geometrisi yok)
        frame = np.full((h, w, 3), (45, 130, 50), dtype=np.uint8)
        for _ in range(350):
            rx = int(rng.integers(0, w - 24))
            ry = int(rng.integers(0, h - 24))
            col = (
                int(rng.integers(20, 120)),
                int(rng.integers(100, 210)),
                int(rng.integers(20, 120)),
            )
            cv2.rectangle(frame, (rx, ry), (rx + 22, ry + 18), col, -1)
        return frame

    raise ValueError(f"Bilinmeyen replay türü: {kind}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Test 1: t+1 post-cut bug fix — kesme sonrası replay boyunca None kalmalı
# ----------------------------------------------------------------------
def test_scene_cut_invalidation_persists_across_replay_frames():
    canvas = _make_virtual_pitch_canvas()
    tracker = CameraMotionTracker()

    # Kare 0..4: ana yayın kamerasında yumuşak pan
    for i in range(5):
        H_view = _build_camera_view_matrix(pan_x=i * 6.0, pan_y=0.0, zoom=1.0)
        frame = _render_broadcast_view(canvas, H_view)
        m = tracker.update(frame, [])
        assert m is not None
        assert m.dtype == np.float64
        assert tracker.is_anchored is True
        assert tracker.scene_cut_detected is False

    # Kare 5: ani tekrar kamerasına sert kesme
    cut_frame = _make_replay_frame("closeup_red", 0)
    m_cut = tracker.update(cut_frame, [])
    assert m_cut is None
    assert tracker.scene_cut_detected is True
    assert tracker.is_anchored is False
    assert tracker.shot_state == "NON_TACTICAL"

    # Kare 6..14 (t+1..t+9): hareketli tekrar görüntüleri -> ASLA sahte matris dönmemeli
    for k in range(1, 10):
        replay_frame = _make_replay_frame("closeup_red", k)
        m_rep = tracker.update(replay_frame, [])
        assert m_rep is None, f"Kare t+{k} None dönmeliydi"
        assert tracker.is_anchored is False
        assert tracker.scene_cut_detected is False

    # PerspectiveTransformer ile None sözleşmesi kontrolü
    src_pts = _project_canvas_landmarks(_build_camera_view_matrix())
    transformer = PerspectiveTransformer(source_points=src_pts)
    assert transformer.is_calibrated is True
    assert transformer.transform_point(480.0, 270.0, homography_matrix=None) is not None
    # Kamera kesmesi olduğunda main.py motion_matrix=None için pitch alanlarını None yapar;
    # ayrıca geçersiz matris geçilirse PerspectiveTransformer de None döndürmelidir
    dets = [{"bbox": [460.0, 240.0, 500.0, 300.0]}]
    transformer.add_pitch_coordinates(dets, camera_motion_matrix=np.zeros((3, 3)))
    assert dets[0]["pitch_x"] is None
    assert dets[0]["pitch_y"] is None
    assert dets[0]["on_pitch"] is None


# ----------------------------------------------------------------------
# Test 2: Kademeli geçiş (gradual dissolve / alpha fade) tespiti
# ----------------------------------------------------------------------
def test_gradual_fade_dissolve_detected():
    canvas = _make_virtual_pitch_canvas()
    tracker = CameraMotionTracker()

    last_main = None
    for i in range(6):
        H_view = _build_camera_view_matrix(pan_x=i * 4.0)
        last_main = _render_broadcast_view(canvas, H_view)
        m = tracker.update(last_main, [])
        assert m is not None
        assert tracker.is_anchored is True

    # 10 karelik doğrusal alpha dissolve (ana kamera -> closeup_red)
    cut_detected_during_dissolve = False
    for step in range(1, 11):
        alpha = step / 10.0
        target = _make_replay_frame("closeup_red", step)
        blended = cv2.addWeighted(last_main, 1.0 - alpha, target, alpha, 0.0)
        m = tracker.update(blended, [])
        if m is None:
            cut_detected_during_dissolve = True

    assert cut_detected_during_dissolve is True
    assert tracker.is_anchored is False

    # Dissolve sonrası tekrar kareleri None kalmalı
    for k in range(11, 16):
        m_after = tracker.update(_make_replay_frame("closeup_red", k), [])
        assert m_after is None
        assert tracker.is_anchored is False


# ----------------------------------------------------------------------
# Test 3: Çim renkli tekrar kesmesi (yüksek histogram korelasyonuna rağmen)
# ----------------------------------------------------------------------
def test_pitch_colored_replay_cut_detected():
    canvas = _make_virtual_pitch_canvas()
    tracker = CameraMotionTracker()

    main_frames = []
    for i in range(6):
        H_view = _build_camera_view_matrix(pan_x=i * 3.0)
        f = _render_broadcast_view(canvas, H_view)
        main_frames.append(f)
        m = tracker.update(f, [])
        assert m is not None

    replay0 = _make_replay_frame("pitch_colored_replay", 0, canvas=canvas)

    # Bu sentetik çim-renkli tekrar karesinin gri histogram korelasyonunun gerçekten
    # saf histogram eşiğinin (>= 0.50) üstünde olduğunu doğrula
    h_main = cv2.calcHist([cv2.cvtColor(main_frames[-1], cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    h_rep = cv2.calcHist([cv2.cvtColor(replay0, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    cv2.normalize(h_main, h_main)
    cv2.normalize(h_rep, h_rep)
    gray_corr = cv2.compareHist(h_main, h_rep, cv2.HISTCMP_CORREL)
    assert gray_corr >= 0.55

    # Yine de tracker yapısal/geometrik sinyallerle kesmeyi yakalayıp None dönmeli
    cut_triggered = False
    for k in range(8):
        rf = _make_replay_frame("pitch_colored_replay", k, canvas=canvas)
        m = tracker.update(rf, [])
        if m is None:
            cut_triggered = True
    assert cut_triggered is True
    assert tracker.is_anchored is False
    assert tracker.update(_make_replay_frame("pitch_colored_replay", 9, canvas=canvas), []) is None


# ----------------------------------------------------------------------
# Test 4: Ana kameraya dönüşte recovery (birebir ve pan+zoom yapılmış poz)
# ----------------------------------------------------------------------
def test_recovery_on_return_exact_and_panned_zoomed_pose():
    canvas = _make_virtual_pitch_canvas()
    H_0 = _build_camera_view_matrix(pan_x=0.0, pan_y=0.0, zoom=1.0)
    src_pts = _project_canvas_landmarks(H_0)
    transformer = PerspectiveTransformer(source_points=src_pts)
    assert transformer.is_calibrated is True

    # (a) Birebir kare-0 pozuna dönüş
    tracker_a = CameraMotionTracker()
    for i in range(6):
        H_i = _build_camera_view_matrix(pan_x=i * 5.0)
        assert tracker_a.update(_render_broadcast_view(canvas, H_i), []) is not None

    for k in range(8):
        assert tracker_a.update(_make_replay_frame("closeup_red", k), []) is None
    assert tracker_a.is_anchored is False

    frame_ret_exact = _render_broadcast_view(canvas, H_0)
    m_rec_a = tracker_a.update(frame_ret_exact, [])
    assert m_rec_a is not None
    assert m_rec_a.dtype == np.float64
    assert np.isclose(m_rec_a[2, 2], 1.0)
    assert tracker_a.is_anchored is True
    assert tracker_a.recovered_this_frame is True

    # (b) Pan (+120 px) ve zoom (1.10x) birikimi sonrası tekrar -> pan+zoom pozuna dönüş
    tracker_b = CameraMotionTracker()
    for i in range(21):
        H_i = _build_camera_view_matrix(
            pan_x=i * 6.0, pan_y=i * 0.8, zoom=1.0 + 0.005 * i
        )
        m = tracker_b.update(_render_broadcast_view(canvas, H_i), [])
        assert m is not None

    for k in range(8):
        assert tracker_b.update(_make_replay_frame("closeup_red", k), []) is None
    assert tracker_b.is_anchored is False

    H_ret = _build_camera_view_matrix(pan_x=100.0, pan_y=20.0, zoom=1.08)
    frame_ret = _render_broadcast_view(canvas, H_ret)
    m_rec_b = tracker_b.update(frame_ret, [])
    assert m_rec_b is not None
    assert m_rec_b.dtype == np.float64
    assert np.isclose(m_rec_b[2, 2], 1.0)
    assert tracker_b.is_anchored is True
    assert tracker_b.recovered_this_frame is True

    # Analitik M_gt = H_0 @ inv(H_ret) ile yeniden projeksiyon hatası < 3.5 px
    M_gt = H_0 @ np.linalg.inv(H_ret)
    M_gt /= M_gt[2, 2]

    # Orta yuvarlak / merkez nokta dönüş karedeki piksel koordinatı
    center_ret = cv2.perspectiveTransform(
        np.array([[[900.0, 600.0]]], dtype=np.float64), H_ret
    )[0, 0]
    pred_f0 = cv2.perspectiveTransform(
        center_ret.reshape(1, 1, 2), m_rec_b
    )[0, 0]
    gt_f0 = cv2.perspectiveTransform(
        center_ret.reshape(1, 1, 2), M_gt
    )[0, 0]
    assert float(np.linalg.norm(pred_f0 - gt_f0)) < 3.5

    # PerspectiveTransformer gerçek orta noktayı (52.5m, 34.0m) < 1.0 m hata ile bulmalı
    det_center = [
        {
            "bbox": [
                float(center_ret[0]) - 10.0,
                float(center_ret[1]) - 30.0,
                float(center_ret[0]) + 10.0,
                float(center_ret[1]),
            ]
        }
    ]
    transformer.add_pitch_coordinates(det_center, camera_motion_matrix=m_rec_b)
    assert det_center[0]["on_pitch"] is True
    assert det_center[0]["pitch_x"] == pytest.approx(52.5, abs=1.0)
    assert det_center[0]["pitch_y"] == pytest.approx(34.0, abs=1.0)


# ----------------------------------------------------------------------
# Test 5: Ana kamera dışı yeşil kareler sahte recovery yapmamalı
# ----------------------------------------------------------------------
def test_nontactical_green_closeup_and_crowd_do_not_falsely_recover():
    canvas = _make_virtual_pitch_canvas()
    tracker = CameraMotionTracker()

    for i in range(5):
        H_i = _build_camera_view_matrix(pan_x=i * 4.0)
        assert tracker.update(_render_broadcast_view(canvas, H_i), []) is not None

    # Kesme tetikle
    assert tracker.update(_make_replay_frame("closeup_red", 0), []) is None
    assert tracker.is_anchored is False

    # Yakın çekim saf çim + yeşil tribün kareleri
    for k in range(6):
        m_grass = tracker.update(_make_replay_frame("green_closeup_nontactical", k), [])
        assert m_grass is None
        assert tracker.is_anchored is False

    for k in range(6):
        m_crowd = tracker.update(_make_replay_frame("green_crowd_nontactical", k), [])
        assert m_crowd is None
        assert tracker.is_anchored is False


# ----------------------------------------------------------------------
# Test 6: 600+ karelik uzun kesintisiz çekimde drift sınırı + statik skorbord
# ----------------------------------------------------------------------
def test_long_continuous_shot_drift_bounds_with_scorebug():
    canvas = _make_virtual_pitch_canvas()
    H_0 = _build_camera_view_matrix(pan_x=0.0, pan_y=0.0, zoom=1.0)
    src_pts = _project_canvas_landmarks(H_0)
    transformer = PerspectiveTransformer(source_points=src_pts)

    tracker = CameraMotionTracker()
    total_frames = 620
    return_indices = {200, 400, 600}
    max_abs_tx = 0.0

    center_canvas = np.array([[[900.0, 600.0]]], dtype=np.float64)

    for t in range(total_frames):
        # 200, 400, 600 karelerinde pan/zoom sıfıra döner
        phase = 2.0 * math.pi * (t / 200.0)
        pan_x = 110.0 * math.sin(phase)
        pan_y = 22.0 * math.sin(phase)
        zoom = 1.0 + 0.06 * math.sin(phase)

        H_t = _build_camera_view_matrix(pan_x=pan_x, pan_y=pan_y, zoom=zoom)
        frame = _render_broadcast_view(canvas, H_t, add_scorebug=True)
        m = tracker.update(frame, [])

        assert m is not None, f"Kare {t} beklenmedik biçimde kesildi"
        assert m.dtype == np.float64
        assert np.isclose(m[2, 2], 1.0)
        assert tracker.is_anchored is True

        max_abs_tx = max(max_abs_tx, abs(float(m[0, 2])))

        if t in return_indices:
            center_px = cv2.perspectiveTransform(center_canvas, H_t)[0, 0]
            det = [
                {
                    "bbox": [
                        float(center_px[0]) - 8.0,
                        float(center_px[1]) - 24.0,
                        float(center_px[0]) + 8.0,
                        float(center_px[1]),
                    ]
                }
            ]
            transformer.add_pitch_coordinates(det, camera_motion_matrix=m)
            assert det[0]["on_pitch"] is True
            err_m = math.hypot(det[0]["pitch_x"] - 52.5, det[0]["pitch_y"] - 34.0)
            assert err_m < 1.5, f"Kare {t} kayma (drift) hatası yüksek: {err_m:.2f} m"

    # Statik skorbord optical flow'u I'ya kilitlemedi (gerçek pan izlendi)
    assert max_abs_tx > 50.0


# ----------------------------------------------------------------------
# Test 7: CAMERA_MOTION_SCALE downscaling + S_inv @ D_scaled @ S ölçekleme
# ----------------------------------------------------------------------
def test_camera_motion_scale_downscaled_flow_rescaling(monkeypatch):
    canvas = _make_virtual_pitch_canvas()
    out_size = (1280, 720)
    H_0 = _build_camera_view_matrix(out_size=out_size, pan_x=0.0, pan_y=0.0, zoom=1.0)
    H_1 = _build_camera_view_matrix(out_size=out_size, pan_x=16.0, pan_y=-8.0, zoom=1.04)

    f0 = _render_broadcast_view(canvas, H_0, out_size=out_size)
    f1 = _render_broadcast_view(canvas, H_1, out_size=out_size)

    monkeypatch.setattr(config, "CAMERA_MOTION_SCALE", 0.5, raising=False)
    monkeypatch.setattr(config, "CAMERA_MOTION_SCALE_MIN_WIDTH", 0, raising=False)
    tracker_scaled = CameraMotionTracker()
    assert tracker_scaled.update(f0, []) is not None
    m_scaled = tracker_scaled.update(f1, [])
    assert m_scaled is not None

    monkeypatch.setattr(config, "CAMERA_MOTION_SCALE", 1.0, raising=False)
    tracker_full = CameraMotionTracker()
    assert tracker_full.update(f0, []) is not None
    m_full = tracker_full.update(f1, [])
    assert m_full is not None

    M_gt = H_0 @ np.linalg.inv(H_1)
    M_gt /= M_gt[2, 2]

    test_pt = np.array([[[640.0, 360.0]]], dtype=np.float64)
    p_gt = cv2.perspectiveTransform(test_pt, M_gt)[0, 0]
    p_scaled = cv2.perspectiveTransform(test_pt, m_scaled)[0, 0]
    p_full = cv2.perspectiveTransform(test_pt, m_full)[0, 0]

    assert float(np.linalg.norm(p_scaled - p_gt)) < 1.5
    assert float(np.linalg.norm(p_scaled - p_full)) < 1.5


# ----------------------------------------------------------------------
# Test 8: PerspectiveTransformer dejenere korumalar, vektörleşme & minimap önbellek
# ----------------------------------------------------------------------
def test_perspective_transformer_degeneracy_batched_and_minimap_cache(monkeypatch):
    # 1) Eşdoğrusal / tekrarlı / NaN kalibrasyon -> is_calibrated=False
    collinear_src = {
        "top_left_corner": (100.0, 100.0),
        "top_right_corner": (200.0, 200.0),
        "bottom_left_corner": (300.0, 300.0),
        "bottom_right_corner": (400.0, 400.0),
    }
    t_col = PerspectiveTransformer(source_points=collinear_src)
    assert t_col.is_calibrated is False
    assert t_col.homography_matrix is None

    nan_src = {
        "top_left_corner": (100.0, 100.0),
        "top_right_corner": (900.0, float("nan")),
        "bottom_left_corner": (100.0, 500.0),
        "bottom_right_corner": (900.0, 500.0),
    }
    t_nan = PerspectiveTransformer(source_points=nan_src)
    assert t_nan.is_calibrated is False

    # Bükülmüş (bowtie) kalibrasyon dörtgeni reddedilmeli
    bowtie_src = {
        "top_left_corner": (100.0, 100.0),
        "top_right_corner": (900.0, 100.0),
        "bottom_left_corner": (900.0, 500.0),
        "bottom_right_corner": (100.0, 500.0),
    }
    t_bowtie = PerspectiveTransformer(source_points=bowtie_src)
    assert t_bowtie.is_calibrated is False

    # Boş sözlük açıkça verilirse config'e düşmemeli
    t_empty = PerspectiveTransformer(source_points={})
    assert t_empty.is_calibrated is False

    # 2) Geçerli dik yayın perspektifi (ufuk çizgisi kare içinde y ~ 109)
    steep_src = {
        "top_left_corner": (840.0, 220.0),
        "top_right_corner": (1080.0, 220.0),
        "bottom_left_corner": (80.0, 920.0),
        "bottom_right_corner": (1840.0, 920.0),
    }
    t_steep = PerspectiveTransformer(source_points=steep_src)
    assert t_steep.is_calibrated is True
    # Saha ortası geçerli olmalı, ufuk üstü (y=20) kamera arkası / geçersiz -> None
    assert t_steep.transform_point(960.0, 570.0) is not None
    assert t_steep.transform_point(960.0, 20.0) is None

    # 3) Dejenere transform_point girişleri -> None
    valid_src = {
        "top_left_corner": (100.0, 100.0),
        "top_right_corner": (900.0, 100.0),
        "bottom_left_corner": (100.0, 500.0),
        "bottom_right_corner": (900.0, 500.0),
    }
    transformer = PerspectiveTransformer(source_points=valid_src)
    assert transformer.is_calibrated is True

    assert transformer.transform_point(float("nan"), 300.0) is None
    assert transformer.transform_point(500.0, float("inf")) is None
    assert transformer.transform_point(500.0, 300.0, homography_matrix=np.zeros((3, 3))) is None

    nan_mat = np.eye(3, dtype=np.float64)
    nan_mat[0, 1] = float("nan")
    assert transformer.transform_point(500.0, 300.0, homography_matrix=nan_mat) is None

    # Ufuk çizgisi (w' = 0) ve kamera arkası (w' < 0): OpenCV normalde (0,0) üretirdi!
    H_van = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.01, 0.0, 1.0]], dtype=np.float64)
    assert transformer.transform_point(100.0, 50.0, homography_matrix=H_van) is None  # w'=0
    assert transformer.transform_point(100.0 + 1e-9, 50.0, homography_matrix=H_van) is None
    assert transformer.transform_point(150.0, 50.0, homography_matrix=H_van) is None  # w'=-0.5

    # 4) Vektörel add_pitch_coordinates vs tekil transform_point birebir eşleşmesi
    rng = np.random.default_rng(123)
    dets_batch = []
    dets_ref = []
    for i in range(25):
        if i in (3, 11, 19):
            bbox = [float("nan"), 100.0, 200.0, 200.0]
        else:
            x1 = float(rng.uniform(50.0, 950.0))
            y2 = float(rng.uniform(80.0, 560.0))
            bbox = [x1, y2 - 40.0, x1 + 20.0, y2]
        dets_batch.append({"bbox": bbox, "class_name": "player", "team": "Team A"})
        dets_ref.append({"bbox": bbox, "class_name": "player", "team": "Team A"})

    transformer.add_pitch_coordinates(dets_batch)
    for d in dets_ref:
        fx, fy = transformer.get_foot_position(d["bbox"])
        res = transformer.transform_point(fx, fy)
        if res is None:
            d["pitch_x"], d["pitch_y"], d["on_pitch"] = None, None, None
        else:
            px, py = res
            d["pitch_x"] = round(px, 2)
            d["pitch_y"] = round(py, 2)
            d["on_pitch"] = bool(-1.0 <= px <= 106.0 and -1.0 <= py <= 69.0)

    for b, r in zip(dets_batch, dets_ref):
        assert b["pitch_x"] == r["pitch_x"]
        assert b["pitch_y"] == r["pitch_y"]
        assert b["on_pitch"] == r["on_pitch"]
        if b["on_pitch"] is not None:
            assert type(b["on_pitch"]) is bool

    # Tekil camera_motion_matrix tüm tespitleri güvenli biçimde None yapmalı
    transformer.add_pitch_coordinates(dets_batch, camera_motion_matrix=np.zeros((3, 3)))
    assert all(d["pitch_x"] is None and d["on_pitch"] is None for d in dets_batch)

    # 5) draw_pitch_minimap önbellek izolasyonu ve config invalidasyonu
    sample_dets = [{"pitch_x": 52.5, "pitch_y": 34.0, "on_pitch": True, "team": "Team A"}]
    m1 = transformer.draw_pitch_minimap(sample_dets)
    m1_copy = m1.copy()
    m1[:] = 0  # dışarıdan mutasyon sonraki çağrıyı ve statik önbelleği bozmamalı

    m2 = transformer.draw_pitch_minimap(sample_dets)
    assert np.array_equal(m1_copy, m2)

    # Boş liste ile de yeşil saha temiz dönmeli (m1 mutasyonundan etkilenmeden)
    m_empty = transformer.draw_pitch_minimap([])
    assert m_empty.shape == m2.shape
    assert np.any(m_empty[:, :, 1] > 100)

    # Boyut değişince önbellek yenilenmeli
    monkeypatch.setattr(config, "PITCH_MINIMAP_WIDTH_PX", 420)
    m_resized = transformer.draw_pitch_minimap(sample_dets)
    assert m_resized.shape[1] == 420


# ----------------------------------------------------------------------
# Test 9: Yeşil olmayan sentetik geriye dönük uyumluluk + veri güvenliği
# ----------------------------------------------------------------------
def test_non_green_synthetic_backward_compat_and_data_safety():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "data", "exports", "tracking_data.csv")
    db_path = os.path.join(root, "data", "exports", "tracking_data.db")
    csv_hash_before = _sha256_file(csv_path) if os.path.exists(csv_path) else None
    db_hash_before = _sha256_file(db_path) if os.path.exists(db_path) else None

    # test_pipeline.py'deki 300x400 gri + rastgele kutu sahnesini tekrarla
    rng = np.random.default_rng(7)
    base_img = np.full((300, 400, 3), 80, dtype=np.uint8)
    for _ in range(120):
        cx = int(rng.integers(30, 370))
        cy = int(rng.integers(30, 270))
        color = tuple(int(c) for c in rng.integers(120, 255, size=3))
        cv2.rectangle(base_img, (cx - 4, cy - 4), (cx + 4, cy + 4), color, -1)

    M_shift = np.float32([[1, 0, -10.0], [0, 1, 0]])
    shifted_img = cv2.warpAffine(base_img, M_shift, (400, 300))
    cut_img = np.full((300, 400, 3), 255, dtype=np.uint8)

    tracker = CameraMotionTracker()
    assert tracker.update(base_img, []) is not None
    m1 = tracker.update(shifted_img, [])
    assert m1 is not None
    pt = cv2.perspectiveTransform(np.array([[[190.0, 150.0]]], dtype=np.float32), m1)[0][0]
    assert pt[0] == pytest.approx(200.0, abs=1.5)
    assert pt[1] == pytest.approx(150.0, abs=1.5)

    assert tracker.update(cut_img, []) is None
    assert tracker.scene_cut_detected is True
    assert tracker.is_anchored is False

    # t+1..t+3 beyaz kesme kareleri None dönmeye devam etmeli
    for _ in range(3):
        assert tracker.update(cut_img, []) is None
        assert tracker.is_anchored is False
        assert tracker.scene_cut_detected is False

    # Ana sahneye (base_img) dönülünce yeşil saha kapısı olmadan da kurtarmalı
    m_rec = tracker.update(base_img, [])
    assert m_rec is not None
    assert tracker.is_anchored is True
    assert tracker.recovered_this_frame is True

    if csv_hash_before is not None:
        assert _sha256_file(csv_path) == csv_hash_before
    if db_hash_before is not None:
        assert _sha256_file(db_path) == db_hash_before
