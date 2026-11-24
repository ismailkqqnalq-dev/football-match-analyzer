"""
test_phase_a2_pitch_calibration.py
----------------------------------
Phase A2 (Otomatik Saha Anahtar Nokta Kalibrasyonu ve Hibrit Kamera Takibi)
için deterministik birim ve entegrasyon testleri (R1 + R2).
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module
from config import config
from modules.camera_motion_tracker import CameraMotionTracker
from modules.perspective_transformer import PerspectiveTransformer
from modules.pitch_calibrator import (
    AutoPitchCalibrator,
    CalibrationResult,
    HybridPitchCalibrator,
)
from modules.security import (
    SecurityViolationError,
    compute_sha256,
    validate_model_weight_path,
)


# ----------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------
def _make_synthetic_homography_m_to_px(
    frame_size: tuple = (1280, 720),
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    scale: float = 1.0,
    perspective_tilt: float = 0.00018,
    flip_sign: bool = False,
) -> np.ndarray:
    """
    105m x 68m saha metre koordinatlarını (frame_w x frame_h) yayın karesine
    eşleyen analitik H_m_to_px (3x3) üretir.
    """
    w, h = float(frame_size[0]), float(frame_size[1])
    # Temel saha köşeleri (metre) -> dik yayın görünümü (yamuk)
    src_m = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
        dtype=np.float64,
    )
    dst_px = np.array(
        [
            [w * 0.14 + pan_x, h * 0.22 + pan_y],
            [w * 0.86 + pan_x, h * 0.22 + pan_y],
            [w * 0.96 + pan_x, h * 0.88 + pan_y],
            [w * 0.04 + pan_x, h * 0.88 + pan_y],
        ],
        dtype=np.float64,
    )
    if scale != 1.0:
        center = np.array([w * 0.5, h * 0.5], dtype=np.float64)
        dst_px = center + (dst_px - center) * float(scale)

    H_m_to_px, _ = cv2.findHomography(src_m, dst_px, method=0)
    H_m_to_px = np.asarray(H_m_to_px, dtype=np.float64)
    if perspective_tilt != 0.0:
        H_m_to_px[2, 1] += perspective_tilt * 0.001
    if flip_sign:
        H_m_to_px = -H_m_to_px
    return H_m_to_px


def _project_template_points(
    H_m_to_px: np.ndarray,
    indices: list,
) -> tuple:
    m_pts = np.array(
        [config.PITCH_KEYPOINTS_32_METERS[i] for i in indices], dtype=np.float64
    )
    px_pts = cv2.perspectiveTransform(m_pts.reshape(1, -1, 2), H_m_to_px)[0]
    return px_pts, m_pts


def _make_textured_pitch_frame(
    frame_size: tuple = (640, 360),
    shift_x: int = 0,
    shift_y: int = 0,
    seed: int = 7,
    half: str = "left",
) -> np.ndarray:
    """LK optical flow ve ORB anahtar kare eşleşmesi için dokulu saha karesi üretir."""
    w, h = frame_size
    canvas_w, canvas_h = w * 3, h * 2
    rng = np.random.default_rng(seed if half == "left" else seed + 101)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (36, 138, 36)
    for bx in range(0, canvas_w, 90):
        if (bx // 90) % 2 == 0:
            canvas[:, bx : min(canvas_w, bx + 90)] = (44, 152, 44)

    # Üst reklam bandı
    canvas[:60, :] = (50, 48, 60)
    for ax in range(0, canvas_w, 80):
        col = tuple(int(c) for c in rng.integers(80, 225, size=3))
        cv2.rectangle(canvas, (ax + 4, 8), (min(canvas_w - 2, ax + 74), 52), col, -1)

    # Zengin deterministik çim dokusu
    noise = rng.integers(-18, 19, size=(canvas_h - 60, canvas_w, 3), dtype=np.int16)
    canvas[60:, :] = np.clip(
        canvas[60:, :].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    for _ in range(1400):
        tx = int(rng.integers(10, canvas_w - 15))
        ty = int(rng.integers(65, canvas_h - 15))
        tw = int(rng.integers(4, 10))
        col = tuple(int(c) for c in rng.integers(25, 195, size=3))
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + tw), col, -1)

    # Saha çizgileri (sol/sağ yarı saha birbirinden farklı geometri)
    white = (242, 245, 242)
    if half == "left":
        cv2.rectangle(canvas, (80, 90), (canvas_w - 80, canvas_h - 60), white, 3)
        cv2.rectangle(canvas, (80, 180), (340, 500), white, 3)
        cv2.circle(canvas, (canvas_w // 2, canvas_h // 2), 95, white, 3)
    else:
        # Tamamen farklı köşe işaretçileri ve doku (sıfır ORB çakışması için)
        cv2.rectangle(canvas, (100, 100), (canvas_w - 100, canvas_h - 80), white, 3)
        cv2.rectangle(canvas, (canvas_w - 380, 190), (canvas_w - 100, 510), white, 3)
        cv2.line(canvas, (220, 100), (220, canvas_h - 80), white, 3)

    x0 = max(0, min(canvas_w - w, w // 2 + shift_x))
    y0 = max(0, min(canvas_h - h, h // 4 + shift_y))
    return canvas[y0 : y0 + h, x0 : x0 + w].copy()


def _make_non_tactical_frame(frame_size: tuple = (640, 360), seed: int = 99) -> np.ndarray:
    """Yeşil saha içermeyen yakın çekim / tribün kesmesi üretir."""
    w, h = frame_size
    rng = np.random.default_rng(seed)
    frame = rng.integers(30, 180, size=(h, w, 3), dtype=np.uint8)
    frame[:, :, 1] = np.clip(frame[:, :, 1] // 3, 0, 80)  # düşük yeşil kanal
    cv2.circle(frame, (w // 2, h // 2), 80, (20, 40, 210), -1)
    return frame


# ----------------------------------------------------------------------
# R1 Testleri
# ----------------------------------------------------------------------
def test_r1_config_32_keypoint_template_and_legacy_landmarks():
    assert len(config.PITCH_KEYPOINT_NAMES_32) == 32
    assert len(config.PITCH_KEYPOINTS_32_METERS) == 32
    assert len(set(config.PITCH_KEYPOINT_NAMES_32)) == 32
    assert len(set(config.PITCH_KEYPOINTS_32_METERS)) == 32

    # Roboflow football-field-detection 0..31 sırası ve 105x68 IFAB sınır kontrolü
    expected_spot_checks = {
        0: ("top_left_corner", (0.0, 0.0)),
        5: ("bottom_left_corner", (0.0, 68.0)),
        8: ("left_penalty_spot", (11.0, 34.0)),
        13: ("halfway_top", (52.5, 0.0)),
        14: ("center_circle_top", (52.5, 24.85)),
        15: ("center_circle_bottom", (52.5, 43.15)),
        16: ("halfway_bottom", (52.5, 68.0)),
        21: ("right_penalty_spot", (94.0, 34.0)),
        24: ("top_right_corner", (105.0, 0.0)),
        29: ("bottom_right_corner", (105.0, 68.0)),
        30: ("center_circle_left", (43.35, 34.0)),
        31: ("center_circle_right", (61.65, 34.0)),
    }
    for idx, (exp_name, exp_xy) in expected_spot_checks.items():
        assert config.PITCH_KEYPOINT_NAMES_32[idx] == exp_name
        got_x, got_y = config.PITCH_KEYPOINTS_32_METERS[idx]
        assert got_x == pytest.approx(exp_xy[0], abs=1e-6)
        assert got_y == pytest.approx(exp_xy[1], abs=1e-6)

    for mx, my in config.PITCH_KEYPOINTS_32_METERS:
        assert 0.0 <= mx <= config.PITCH_LENGTH_M
        assert 0.0 <= my <= config.PITCH_WIDTH_M

    # Eski PITCH_LANDMARKS_METERS (29 anahtar) korunmuş olmalı
    assert len(config.PITCH_LANDMARKS_METERS) == 29
    assert config.PITCH_LANDMARKS_METERS["center_spot"] == (52.5, 34.0)

    # Phase A2 sabitleri
    assert config.AUTO_CALIBRATE_PITCH is True
    assert config.PITCH_CALIBRATION_REFRESH_INTERVAL == 15
    assert config.PITCH_CALIBRATION_SMOOTH_FRAMES == 5
    assert config.PITCH_KEYPOINT_CONF_THRESHOLD == pytest.approx(0.50)
    assert config.PITCH_CALIBRATION_MIN_KEYPOINTS == 4
    assert config.PITCH_KEYPOINT_MODEL_DOWNLOAD_URL == ""
    assert "yolo11s-pose.pt" in config.SECURITY_ALLOWED_OFFICIAL_YOLO_BASENAMES


def test_r1_missing_pitch_model_clean_t2_fallback(tmp_path, monkeypatch):
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    missing_model = fake_models / "football_pitch_pose.pt"
    monkeypatch.setattr(config, "MODELS_DIR", str(fake_models))
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_PATH", str(missing_model))

    hough_calls = []

    def _forbidden_hough(*args, **kwargs):
        hough_calls.append((args, kwargs))
        raise AssertionError("OpenCV Hough/çizgi kestirimi çağrılmamalı!")

    monkeypatch.setattr(cv2, "HoughLines", _forbidden_hough)
    monkeypatch.setattr(cv2, "HoughLinesP", _forbidden_hough)

    cal = AutoPitchCalibrator(device="cpu")
    assert cal.is_available is False
    assert cal.model is None

    frame = _make_textured_pitch_frame()
    res = cal.calibrate_frame(frame)
    assert isinstance(res, CalibrationResult)
    assert res.success is False
    assert res.homography_matrix is None
    assert hough_calls == []

    # PerspectiveTransformer T2 modunda kalmalı
    pt = PerspectiveTransformer(source_points={}, quiet=True)
    assert pt.is_calibrated is False
    dets = [{"bbox": [100, 100, 140, 200], "class_name": "player"}]
    pt.add_pitch_coordinates(dets)
    assert dets[0]["pitch_x"] is None
    assert dets[0]["pitch_y"] is None
    assert dets[0]["on_pitch"] is None


def test_r1_pitch_model_security_isolated_from_detector_sha256(tmp_path, monkeypatch):
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    pitch_weight = fake_models / "football_pitch_pose.pt"
    pitch_weight.write_bytes(b"synthetic-pitch-pose-weights")
    pitch_sha = compute_sha256(pitch_weight)

    monkeypatch.setattr(config, "MODELS_DIR", str(fake_models))
    # Dedektör SHA-256 farklı bir değere sabitlenmiş olsa bile saha modeli etkilenmemeli
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", "f" * 64)
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", None)

    resolved = validate_model_weight_path(
        pitch_weight,
        models_dir=fake_models,
        sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
    )
    assert resolved == pitch_weight.resolve()

    # Yanlış SECURITY_PITCH_MODEL_SHA256 engellemeli
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", "0" * 64)
    with pytest.raises(SecurityViolationError, match="SHA-256"):
        validate_model_weight_path(
            pitch_weight,
            models_dir=fake_models,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )

    # Doğru SHA-256 kabul edilmeli
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", pitch_sha)
    assert (
        validate_model_weight_path(
            pitch_weight,
            models_dir=fake_models,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )
        == pitch_weight.resolve()
    )

    # Dizin dışı veya traversal yolu AutoPitchCalibrator tarafından reddedilmeli
    outside_weight = tmp_path / "evil.pt"
    outside_weight.write_bytes(b"evil")
    with pytest.raises(SecurityViolationError):
        AutoPitchCalibrator(model_path=str(outside_weight), device="cpu")
    with pytest.raises(SecurityViolationError):
        AutoPitchCalibrator(
            model_path=str(fake_models / ".." / "evil.pt"), device="cpu"
        )


def test_r1_keypoint_homography_estimation_and_validation_guards():
    cal = AutoPitchCalibrator(model=object(), device="cpu")
    frame_shape = (720, 1280, 3)

    # 1) 10 geçerli + 1 aykırı (outlier) nokta -> RANSAC aykırıyı eleyip başarıyla çözmeli
    H_true = _make_synthetic_homography_m_to_px(frame_size=(1280, 720))
    indices = [0, 1, 4, 5, 9, 12, 13, 14, 15, 16, 24]
    px_pts, m_pts = _project_template_points(H_true, indices)
    px_with_outlier = px_pts.copy()
    px_with_outlier[-1] += np.array([120.0, -90.0])  # aykırı nokta

    res = cal.estimate_homography_from_keypoints(px_with_outlier, m_pts, frame_shape)
    assert res.success is True
    assert res.homography_matrix is not None
    assert res.num_keypoints == len(indices)
    assert res.num_inliers >= 10
    assert res.reproj_error_m is not None and res.reproj_error_m < 0.5
    assert res.reproj_error_px is not None and res.reproj_error_px < 2.0
    assert float(np.linalg.det(res.homography_matrix)) > 0.0

    # 2) Dik yayın açısında ufuk çizgisi yakınken (-H işareti) w' kanonikleştirmesi çalışmalı
    steep_src = np.array(
        [[280.0, 220.0], [1000.0, 220.0], [1220.0, 660.0], [60.0, 660.0], [640.0, 420.0]],
        dtype=np.float64,
    )
    steep_dst = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0], [52.5, 34.0]],
        dtype=np.float64,
    )
    steep_res = cal.estimate_homography_from_keypoints(steep_src, steep_dst, frame_shape)
    assert steep_res.success is True
    assert steep_res.homography_matrix is not None
    assert float(np.linalg.det(steep_res.homography_matrix)) > 0.0

    # 3) < 4 nokta reddedilmeli
    res_few = cal.estimate_homography_from_keypoints(px_pts[:3], m_pts[:3], frame_shape)
    assert res_few.success is False

    # 4) Doğrudaş (collinear) noktalar reddedilmeli
    col_px = np.array(
        [[100.0, 300.0], [300.0, 300.0], [500.0, 300.0], [700.0, 300.0]],
        dtype=np.float64,
    )
    col_m = np.array(
        [[0.0, 34.0], [20.0, 34.0], [52.5, 34.0], [80.0, 34.0]],
        dtype=np.float64,
    )
    assert cal.estimate_homography_from_keypoints(col_px, col_m, frame_shape).success is False

    # 5) Ayna yansıması (det < 0) reddedilmeli
    mirrored_px = px_pts[:6].copy()
    mirrored_px[:, 0] = 1280.0 - mirrored_px[:, 0]
    res_mirror = cal.estimate_homography_from_keypoints(
        mirrored_px, m_pts[:6], frame_shape
    )
    assert res_mirror.success is False

    # 6) Büyük yeniden projeksiyon hatası ve yetersiz RANSAC inlier reddedilmeli
    cal_min6 = AutoPitchCalibrator(model=object(), device="cpu", min_keypoints=6)
    noisy_px = px_pts[:6].copy()
    noisy_px[0] += [60.0, -50.0]
    noisy_px[1] += [-55.0, 45.0]
    res_inliers = cal_min6.estimate_homography_from_keypoints(noisy_px, m_pts[:6], frame_shape)
    assert res_inliers.success is False
    assert res_inliers.reason == "insufficient_inliers"

    cal_reproj = AutoPitchCalibrator(
        model=object(),
        device="cpu",
        min_keypoints=5,
        ransac_thresh_px=25.0,
        max_reproj_err_px=4.0,
    )
    jittered_px = px_pts[:6].copy()
    jittered_px += np.array(
        [[10.0, -8.0], [-9.0, 9.0], [8.0, 10.0], [-10.0, -7.0], [9.0, -9.0], [-8.0, 8.0]]
    )
    res_reproj = cal_reproj.estimate_homography_from_keypoints(
        jittered_px, m_pts[:6], frame_shape
    )
    assert res_reproj.success is False
    assert res_reproj.reason in {"reproj_error_px_exceeded", "reproj_error_m_exceeded"}

    # 7) Bükülmüş (bowtie) homografi reddedilmeli
    bowtie_px = np.array(
        [[200.0, 200.0], [1000.0, 200.0], [200.0, 620.0], [1000.0, 620.0]],
        dtype=np.float64,
    )
    bowtie_m = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
        dtype=np.float64,
    )
    assert cal.estimate_homography_from_keypoints(bowtie_px, bowtie_m, frame_shape).success is False

    # 8) Mock YOLO-Pose tensörü: düşük güvenli keypointleri filtrelemeli
    full_xy = np.zeros((1, 32, 2), dtype=np.float32)
    full_conf = np.zeros((1, 32), dtype=np.float32)
    all_px, _ = _project_template_points(H_true, list(range(32)))
    for k in [0, 5, 9, 12, 13, 16, 24, 29]:
        full_xy[0, k] = all_px[k]
        full_conf[0, k] = 0.92
    # Güven eşiği altındaki bozuk keypointler elenmeli
    full_xy[0, 1] = [5000.0, 5000.0]
    full_conf[0, 1] = 0.15
    full_xy[0, 4] = [-400.0, -400.0]
    full_conf[0, 4] = 0.49

    class FakePoseModel:
        def predict(self, **kwargs):
            kpts = SimpleNamespace(xy=full_xy, conf=full_conf)
            return [SimpleNamespace(keypoints=kpts, boxes=None)]

    cal_mock = AutoPitchCalibrator(model=FakePoseModel(), device="cpu")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_res = cal_mock.calibrate_frame(frame)
    assert mock_res.success is True
    assert mock_res.num_keypoints == 8


def test_r1_general_position_rejects_three_collinear_inliers_and_keeps_valid_sets():
    cal = AutoPitchCalibrator(model=object(), device="cpu")
    frame_shape = (720, 1280, 3)
    H_true = _make_synthetic_homography_m_to_px(frame_size=(1280, 720))

    # 1) Üst taç çizgisi üzerinde 3 doğrudaş + 1 çizgi dışı nokta ([0, 13, 24, 14]) — tam ve +8px gürültülü
    px_top3, m_top3 = _project_template_points(H_true, [0, 13, 24, 14])
    res_top3_exact = cal.estimate_homography_from_keypoints(px_top3, m_top3, frame_shape)
    assert res_top3_exact.success is False
    assert res_top3_exact.reason in {"collinear_keypoints", "collinear_inliers"}

    px_top3_noisy = px_top3.copy()
    px_top3_noisy[1] += np.array([0.0, 8.0])
    res_top3_noisy = cal.estimate_homography_from_keypoints(px_top3_noisy, m_top3, frame_shape)
    assert res_top3_noisy.success is False

    # 2) Alt taç çizgisi üzerinde 3 doğrudaş + 1 çizgi dışı nokta ([5, 16, 29, 15]) — tam ve +4px gürültülü
    px_bot3, m_bot3 = _project_template_points(H_true, [5, 16, 29, 15])
    res_bot3_exact = cal.estimate_homography_from_keypoints(px_bot3, m_bot3, frame_shape)
    assert res_bot3_exact.success is False

    px_bot3_noisy = px_bot3.copy()
    px_bot3_noisy[1] += np.array([0.0, 4.0])
    res_bot3_noisy = cal.estimate_homography_from_keypoints(px_bot3_noisy, m_bot3, frame_shape)
    assert res_bot3_noisy.success is False

    # 3) (N-1) doğrudaş + 1 çizgi dışı: orta saha çizgisi üzerinde 4 nokta [13, 14, 15, 16] + 1 köşe [0]
    px_n_minus_1, m_n_minus_1 = _project_template_points(H_true, [13, 14, 15, 16, 0])
    res_n_minus_1 = cal.estimate_homography_from_keypoints(px_n_minus_1, m_n_minus_1, frame_shape)
    assert res_n_minus_1.success is False

    # 4) RANSAC sonrası dejenere inlier çökmesi: 4 nokta (3'ü doğrudaş) + 1 uzak aykırı nokta
    px_with_outlier, m_with_outlier = _project_template_points(H_true, [0, 13, 24, 14, 29])
    px_with_outlier[-1] += np.array([150.0, -120.0])
    res_post_ransac = cal.estimate_homography_from_keypoints(
        px_with_outlier, m_with_outlier, frame_shape
    )
    assert res_post_ransac.success is False
    assert res_post_ransac.reason == "collinear_inliers"

    # 5) Metre uzayı genel konumda olsa bile piksel uzayında 3 doğrudaş + 1 çizgi dışı reddedilmeli
    _, m_corners = _project_template_points(H_true, [0, 24, 29, 5])
    px_collinear_3_of_4 = np.array(
        [[200.0, 200.0], [640.0, 200.0], [1080.0, 200.0], [640.0, 620.0]],
        dtype=np.float64,
    )
    assert cal.estimate_homography_from_keypoints(
        px_collinear_3_of_4, m_corners, frame_shape
    ).success is False

    # 6) Geçerli setler (< 0.5 m hata ile) geçmeye devam etmeli: 4-köşe (dik -H dahil), 6-nokta (3 üst + 3 alt), 10-nokta
    H_steep_neg = _make_synthetic_homography_m_to_px(
        frame_size=(1280, 720), perspective_tilt=0.00028, flip_sign=True
    )
    px_4c, m_4c = _project_template_points(H_steep_neg, [0, 24, 29, 5])
    res_4c = cal.estimate_homography_from_keypoints(px_4c, m_4c, frame_shape)
    assert res_4c.success is True
    assert res_4c.reproj_error_m is not None and res_4c.reproj_error_m < 0.5
    assert float(np.linalg.det(res_4c.homography_matrix)) > 0.0

    px_6pt, m_6pt = _project_template_points(H_true, [0, 13, 24, 5, 16, 29])
    res_6pt = cal.estimate_homography_from_keypoints(px_6pt, m_6pt, frame_shape)
    assert res_6pt.success is True
    assert res_6pt.reproj_error_m is not None and res_6pt.reproj_error_m < 0.5

    px_10pt, m_10pt = _project_template_points(
        H_true, [0, 1, 4, 5, 9, 12, 13, 16, 24, 29]
    )
    res_10pt = cal.estimate_homography_from_keypoints(px_10pt, m_10pt, frame_shape)
    assert res_10pt.success is True
    assert res_10pt.reproj_error_m is not None and res_10pt.reproj_error_m < 0.5


# ----------------------------------------------------------------------
# R2 Testleri
# ----------------------------------------------------------------------
def test_r2_perspective_transformer_dynamic_hooks(capsys):
    pt = PerspectiveTransformer(source_points={}, quiet=True)
    out = capsys.readouterr().out
    assert out == ""
    assert pt.is_calibrated is False

    H_m_to_px = _make_synthetic_homography_m_to_px((1280, 720))
    H_px_to_m = np.linalg.inv(H_m_to_px)
    assert pt.set_homography(H_px_to_m) is True
    assert pt.is_calibrated is True
    saved_h = pt.homography_matrix.copy()

    # Geçersiz matrisler mevcut kalibrasyonu bozmamalı
    assert pt.set_homography(None) is False
    assert pt.set_homography(np.zeros((3, 3))) is False
    nan_h = saved_h.copy()
    nan_h[0, 0] = np.nan
    assert pt.set_homography(nan_h) is False
    assert pt.is_calibrated is True
    assert np.allclose(pt.homography_matrix, saved_h)

    # clear_calibration() -> T2 moduna dönüş
    pt.clear_calibration()
    assert pt.is_calibrated is False
    assert pt.homography_matrix is None
    dets = [{"bbox": [600, 400, 640, 500], "class_name": "player"}]
    pt.add_pitch_coordinates(dets)
    assert dets[0]["pitch_x"] is None
    assert dets[0]["on_pitch"] is None


def test_r2_camera_motion_tracker_reanchor_and_bank_rebase():
    tracker = CameraMotionTracker()
    frame_size = (640, 360)

    # 1) Pan hareketi boyunca takip ve keyframe birikimi
    frames = [_make_textured_pitch_frame(frame_size, shift_x=i * 12) for i in range(6)]
    m_prev = None
    for f in frames:
        m_prev = tracker.update(f)
        assert m_prev is not None
    assert len(tracker._keyframe_bank) >= 1

    old_bank = [dict(kf, M_k=kf["M_k"].copy()) for kf in tracker._keyframe_bank]
    m_old = m_prev.copy()

    reanchored = tracker.reanchor(
        frames[-1], keep_keyframe_bank=True, prev_cumulative=m_old
    )
    assert np.allclose(reanchored, np.eye(3))
    assert np.allclose(tracker.cumulative_matrix, np.eye(3))
    assert tracker.is_anchored is True
    assert tracker.shot_state == "ANCHORED"

    inv_old = np.linalg.inv(m_old)
    for old_kf, new_kf in zip(old_bank, tracker._keyframe_bank):
        expected_m = inv_old @ old_kf["M_k"]
        expected_m = expected_m / expected_m[2, 2]
        assert np.allclose(new_kf["M_k"], expected_m, atol=1e-6)
        exp_center = tracker._project_center_ref(expected_m, frames[-1].shape)
        assert np.allclose(new_kf["view_center_ref"], exp_center, atol=1e-4)

    # 2) Sahne kesmesi sonrası Kare 0'a dönüldüğünde bankadaki yeniden tabanlanmış
    # M_k üzerinden Kare 0'a geri dönüş (< 4 px köşe hatası)
    cut_frame = _make_non_tactical_frame(frame_size)
    assert tracker.update(cut_frame) is None
    assert tracker.scene_cut_detected is True
    assert tracker.update(cut_frame) is None

    rec_m = tracker.update(frames[0])
    assert rec_m is not None
    assert tracker.recovered_this_frame is True

    corners = np.array(
        [[[0.0, 0.0]], [[640.0, 0.0]], [[640.0, 360.0]], [[0.0, 360.0]]],
        dtype=np.float64,
    )
    p_rec = cv2.perspectiveTransform(corners, rec_m)[:, 0, :]
    p_exp = cv2.perspectiveTransform(corners, inv_old)[:, 0, :]
    corner_err = float(np.mean(np.linalg.norm(p_rec - p_exp, axis=1)))
    assert corner_err < 4.0

    # 3) keep_keyframe_bank=False bankayı temizlemeli ve scene_cut_detected bayrağını korumalı
    tracker.scene_cut_detected = True
    new_angle = _make_textured_pitch_frame(frame_size, half="right")
    tracker.reanchor(new_angle, keep_keyframe_bank=False)
    assert tracker.scene_cut_detected is True
    assert len(tracker._keyframe_bank) <= 1
    if tracker._keyframe_bank:
        assert tracker._keyframe_bank[0]["pinned"] is True
        assert np.allclose(tracker._keyframe_bank[0]["M_k"], np.eye(3))


def test_r2_hybrid_calibrator_lifecycle_cut_opposite_half_and_smooth_blend():
    frame_size = (640, 360)
    H_left_px_to_m = np.linalg.inv(_make_synthetic_homography_m_to_px(frame_size, pan_x=0.0))
    H_left_shifted = np.linalg.inv(_make_synthetic_homography_m_to_px(frame_size, pan_x=4.0))
    H_right_px_to_m = np.linalg.inv(_make_synthetic_homography_m_to_px(frame_size, pan_x=-60.0))

    f_closeup = _make_non_tactical_frame(frame_size, seed=11)
    f_left = _make_textured_pitch_frame(frame_size, shift_x=0, half="left")
    f_left_2 = _make_textured_pitch_frame(frame_size, shift_x=4, half="left")
    f_replay = _make_non_tactical_frame(frame_size, seed=22)
    f_right = _make_textured_pitch_frame(frame_size, shift_x=0, half="right")

    # Kare indeksine göre sahte AutoPitchCalibrator yanıtları
    scripted_results = {}

    class ScriptedCalibrator:
        is_available = True

        def __init__(self):
            self.current_idx = -1

        def calibrate_frame(self, frame):
            return scripted_results.get(
                self.current_idx, CalibrationResult(success=False, reason="no_kpts")
            )

    cal = ScriptedCalibrator()
    pt = PerspectiveTransformer(source_points={}, quiet=True)
    mt = CameraMotionTracker()
    hybrid = HybridPitchCalibrator(
        calibrator=cal,
        transformer=pt,
        motion_tracker=mt,
        refresh_interval=5,
        smooth_frames=4,
    )

    # Kare 0-1: Giriş yakın çekimi (anahtar nokta yok) -> kalibrasyonsuz (None)
    for idx in (0, 1):
        cal.current_idx = idx
        assert hybrid.process_frame(idx, f_closeup) is None
        assert pt.is_calibrated is False

    # Kare 2: İlk taktik kare -> initial_lock
    cal.current_idx = 2
    scripted_results[2] = CalibrationResult(success=True, homography_matrix=H_left_px_to_m)
    m2 = hybrid.process_frame(2, f_left)
    assert m2 is not None
    assert pt.is_calibrated is True
    assert hybrid.last_calibration_reason == "initial_lock"
    assert hybrid.last_calibration_frame == 2

    # Kare 3-6: Ara kareler optical flow ile ilerler
    for idx in range(3, 7):
        cal.current_idx = idx
        m_step = hybrid.process_frame(idx, f_left)
        assert m_step is not None

    # Kare 7: Periyodik yenileme (5 kare sonra). A2-3: model flow ile uyuşuyorsa
    # (fark <= PITCH_CAL_DEADBAND_M) tablo DEĞİŞMEZ ve çapa taşınmaz. İşaret ters
    # (-H) olsa bile fark işaret hizalı ölçülmeli.
    # (f_left_2 içeriği 4 px sola kayar -> model tablosu pan_x=-4)
    H_left_agree = np.linalg.inv(_make_synthetic_homography_m_to_px(frame_size, pan_x=-4.0))
    H_left_off = np.linalg.inv(_make_synthetic_homography_m_to_px(frame_size, pan_x=-8.0))
    pt.homography_matrix = -pt.homography_matrix  # ters işaret testi
    h_before = pt.homography_matrix.copy()
    cal.current_idx = 7
    scripted_results[7] = CalibrationResult(success=True, homography_matrix=H_left_agree)
    m7 = hybrid.process_frame(7, f_left_2)
    assert m7 is not None
    assert hybrid.frame_info["source"] == "keypoint_refresh"
    assert hybrid.frame_info["disagreement_m"] <= config.PITCH_CAL_DEADBAND_M
    assert hybrid._blend_total == 0
    assert np.allclose(pt.homography_matrix, h_before)
    assert hybrid.last_calibration_frame == 7

    # Kare 12: ~4 px (DEADBAND < fark <= SOFT) -> hemen kabul, referans noktaları
    # üzerinden yumuşak geçiş; süre farkın PITCH_CAL_BLEND_MAX_STEP_M'ye bölümü kadar
    for idx in range(8, 12):
        cal.current_idx = idx
        hybrid.process_frame(idx, f_left_2)
    cal.current_idx = 12
    scripted_results[12] = CalibrationResult(success=True, homography_matrix=H_left_off)
    m12 = hybrid.process_frame(12, f_left_2)
    d12 = hybrid.frame_info["disagreement_m"]
    assert config.PITCH_CAL_DEADBAND_M < d12 <= config.PITCH_CAL_SOFT_M
    assert np.allclose(m12, np.eye(3))
    assert hybrid.last_calibration_reason == "periodic_refresh"
    assert hybrid.frame_info["source"] == "blend"
    expected_total = max(4, int(np.ceil(d12 / config.PITCH_CAL_BLEND_MAX_STEP_M)))
    assert hybrid._blend_total == expected_total
    assert hybrid._blend_step == 1

    # Kare 13-14: geçiş sürüyor
    for idx in (13, 14):
        cal.current_idx = idx
        hybrid.process_frame(idx, f_left_2)
        assert hybrid.frame_info["source"] == "blend"
    assert hybrid._blend_total == expected_total

    # Kare 15: Tekrar (replay) kesmesi -> None, aktif geçiş iptal
    cal.current_idx = 15
    assert hybrid.process_frame(15, f_replay) is None
    assert mt.scene_cut_detected is True
    assert hybrid._blend_total == 0
    assert hybrid.was_non_tactical is True

    # Kare 16: Karşı yarı sahaya dönüş (sıfır ORB çakışması) -> keypoint ile anında toparlanma
    cal.current_idx = 16
    scripted_results[16] = CalibrationResult(success=True, homography_matrix=H_right_px_to_m)
    m16 = hybrid.process_frame(16, f_right)
    assert m16 is not None
    assert hybrid.last_calibration_reason == "post_cut_recovery"
    assert hybrid.frame_info["source"] == "keypoint_lock"
    assert hybrid.frame_info["calib_reset"] is True
    assert pt.is_calibrated is True

    # Kare 21: Periyodik yenileme başarısız olursa last_calibration_frame korunup Kare 22'de yeniden denenmeli
    cal.current_idx = 21
    assert hybrid.process_frame(21, f_right) is not None
    assert hybrid.last_calibration_frame == 16  # güncellenmedi

    cal.current_idx = 22
    scripted_results[22] = CalibrationResult(success=True, homography_matrix=H_right_px_to_m)
    hybrid.process_frame(22, f_right)
    assert hybrid.last_calibration_frame == 22
    assert hybrid.frame_info["source"] == "keypoint_refresh"  # aynı yer -> değişiklik yok


def test_r2_main_cli_priority_and_no_auto_calibrate_flag(tmp_path, monkeypatch):
    # 1) --help çıktısı --no-auto-calibrate bayrağını içermeli
    help_proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_proc.returncode == 0
    assert "--no-auto-calibrate" in help_proc.stdout

    # Ortak sahte pipeline bileşenleri
    frames = [_make_textured_pitch_frame((640, 360)) for _ in range(3)]
    H_true = _make_synthetic_homography_m_to_px((640, 360))
    H_px_to_m = np.linalg.inv(H_true)
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"")

    class DummyVP:
        def __init__(self, inp, out=None):
            self.fps = 25.0
            self.has_writer = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read_frames(self):
            for idx, f in enumerate(frames):
                yield idx, f

    class DummyDetector:
        device = "cpu"

        def detect_frame(self, frame):
            return [{"bbox": [260, 150, 300, 230], "class_name": "player", "confidence": 0.9}]

    class DummyPlayerTracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, raw_dets, frame):
            return [
                {
                    "track_id": 1,
                    "bbox": [260, 150, 300, 230],
                    "class_name": "player",
                    "confidence": 0.9,
                }
            ]

    class DummyBallTracker:
        def update(self, raw_dets, frame_shape=None):
            return None

    class DummyTeamClassifier:
        def assign_teams(self, players, frame):
            for p in players:
                p["team"] = "Team A"
            return players

        def get_team_map(self):
            return {1: "Team A"}

    exported_records = []

    class DummyExporter:
        def __init__(self, fps=25.0, output_dir=None, **kwargs):
            self.fps = fps
            self.output_dir = output_dir

        def append_frame_data(self, frame_idx, all_tracked):
            exported_records.append((frame_idx, [dict(d) for d in all_tracked]))

        def postprocess(self, team_map=None):
            return {}

        def export_to_csv(self):
            return None

        def export_to_sql(self):
            return None

        def export_distance_summary(self):
            return None

        def to_dataframe(self):
            import pandas as pd
            return pd.DataFrame()

        def close(self):
            pass

    monkeypatch.setattr(main_module, "VideoProcessor", DummyVP)
    monkeypatch.setattr(main_module, "ObjectDetector", DummyDetector)
    monkeypatch.setattr(main_module, "PlayerTracker", DummyPlayerTracker)
    monkeypatch.setattr(main_module, "BallTracker", DummyBallTracker)
    monkeypatch.setattr(main_module, "TeamClassifier", DummyTeamClassifier)
    monkeypatch.setattr(main_module, "DataExporter", DummyExporter)

    auto_cal_calls = []

    class AvailableAutoCal:
        is_available = True

        def __init__(self, *args, **kwargs):
            auto_cal_calls.append("init")

        def calibrate_frame(self, frame):
            auto_cal_calls.append("calibrate")
            return CalibrationResult(success=True, homography_matrix=H_px_to_m)

    # A) Manuel SOURCE_CALIBRATION_POINTS doluysa otomatik kalibrasyon ÇAĞRILMAMALI
    manual_pts = {
        "top_left_corner": (90.0, 80.0),
        "top_right_corner": (550.0, 80.0),
        "bottom_right_corner": (610.0, 315.0),
        "bottom_left_corner": (30.0, 315.0),
    }
    monkeypatch.setattr(config, "SOURCE_CALIBRATION_POINTS", manual_pts)
    monkeypatch.setattr(main_module, "AutoPitchCalibrator", AvailableAutoCal)
    exported_records.clear()
    auto_cal_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(dummy_video),
            "--no-video",
            "--no-analytics",
            "--no-ml",
            "--export-dir",
            str(tmp_path / "manual"),
        ],
    )
    main_module.main()
    assert auto_cal_calls == []
    assert exported_records[0][1][0]["pitch_x"] is not None

    # B) SOURCE_CALIBRATION_POINTS boş + --no-auto-calibrate -> T2 kalibrasyonsuz mod
    monkeypatch.setattr(config, "SOURCE_CALIBRATION_POINTS", {})
    exported_records.clear()
    auto_cal_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(dummy_video),
            "--no-video",
            "--no-analytics",
            "--no-ml",
            "--no-auto-calibrate",
            "--export-dir",
            str(tmp_path / "no_auto"),
        ],
    )
    main_module.main()
    assert auto_cal_calls == []
    assert exported_records[0][1][0]["pitch_x"] is None

    # C) SOURCE_CALIBRATION_POINTS boş + eksik model -> temiz T2 fallback
    class MissingAutoCal:
        is_available = False

    monkeypatch.setattr(main_module, "AutoPitchCalibrator", MissingAutoCal)
    exported_records.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(dummy_video),
            "--no-video",
            "--no-analytics",
            "--no-ml",
            "--export-dir",
            str(tmp_path / "missing_model"),
        ],
    )
    main_module.main()
    assert exported_records[0][1][0]["pitch_x"] is None

    # D) SOURCE_CALIBRATION_POINTS boş + mevcut otomatik model -> saha koordinatları üretilmeli
    monkeypatch.setattr(main_module, "AutoPitchCalibrator", AvailableAutoCal)
    exported_records.clear()
    auto_cal_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input",
            str(dummy_video),
            "--no-video",
            "--no-analytics",
            "--no-ml",
            "--export-dir",
            str(tmp_path / "auto_ok"),
        ],
    )
    main_module.main()
    assert "init" in auto_cal_calls
    assert "calibrate" in auto_cal_calls
    assert exported_records[0][1][0]["pitch_x"] is not None
    assert exported_records[0][1][0]["pitch_y"] is not None


def test_r2_baseline_exports_unmodified():
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "exports" / "tracking_data.csv"
    db_path = repo_root / "data" / "exports" / "tracking_data.db"
    assert csv_path.is_file()
    assert db_path.is_file()

    before_csv = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    before_db = hashlib.sha256(db_path.read_bytes()).hexdigest()

    # Kalibratör eksik modelle çalıştırıldığında örnek dışa aktarımlara dokunmamalı
    cal = AutoPitchCalibrator(model=None, device="cpu") if Path(config.PITCH_KEYPOINT_MODEL_PATH).is_file() else AutoPitchCalibrator(device="cpu")
    cal.calibrate_frame(_make_textured_pitch_frame())

    after_csv = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    after_db = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert before_csv == after_csv
    assert before_db == after_db


@pytest.mark.requires_pitch_model
def test_r1_real_pitch_model_smoke():
    cal = AutoPitchCalibrator(device="cpu")
    assert cal.is_available is True
    frame = _make_textured_pitch_frame((1280, 720))
    res = cal.calibrate_frame(frame)
    assert isinstance(res, CalibrationResult)


# ----------------------------------------------------------------------
# R3 KABUL TESTLERİ: Saha Anahtar Nokta Eğitimi, Model İndirme & İşaretçi
# ----------------------------------------------------------------------
def test_r3_train_pitch_keypoints_env_only_api_key_and_no_cli_flag(monkeypatch, capsys):
    import utils.train_pitch_keypoints as train_pk

    parsed = train_pk.parse_args([])
    assert not hasattr(parsed, "api_key")

    with pytest.raises(SystemExit) as exc_info:
        train_pk.parse_args(["--api-key", "YOUR_API_KEY"])
    assert exc_info.value.code != 0

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError):
        train_pk.resolve_api_key()

    monkeypatch.setenv("ROBOFLOW_API_KEY", "   ")
    with pytest.raises(ValueError):
        train_pk.resolve_api_key()

    monkeypatch.setenv("ROBOFLOW_API_KEY", "  DUMMY_ENV_TOKEN  ")
    assert train_pk.resolve_api_key() == "DUMMY_ENV_TOKEN"

    masked = train_pk._sanitize_secret(
        "Hata: https://api.roboflow.com/auth?token=DUMMY_ENV_TOKEN",
        "DUMMY_ENV_TOKEN",
    )
    assert "DUMMY_ENV_TOKEN" not in masked
    assert "***" in masked

    captured = capsys.readouterr()
    assert "DUMMY_ENV_TOKEN" not in captured.out
    assert "DUMMY_ENV_TOKEN" not in captured.err


def test_r3_train_pitch_keypoints_temp_paths_and_safe_cleanup(tmp_path, monkeypatch):
    import tempfile
    import utils.train_pitch_keypoints as train_pk

    root, dataset_dir, runs_dir = train_pk.resolve_temp_paths(
        None, "football-field-detection-f07vi", "1"
    )
    temp_root = Path(tempfile.gettempdir()).resolve()
    repo_root = Path(config.BASE_DIR).resolve()

    assert root.is_relative_to(temp_root)
    assert root.name == "football_pitch_pose_training"
    assert dataset_dir.is_relative_to(root)
    assert runs_dir.is_relative_to(root)
    assert not dataset_dir.is_relative_to(repo_root)
    assert not runs_dir.is_relative_to(repo_root)

    work_dir = tmp_path / "pose_work"
    external_keep = tmp_path / "external_keep"
    external_keep.mkdir()
    (external_keep / "marker.txt").write_text("keep", encoding="utf-8")

    save_dir = work_dir / "runs" / "football_pitch_pose"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_bytes(b"POSE_BEST_WEIGHTS")

    class FakeVersion:
        version = "4"

        def download(self, fmt, location=None, overwrite=False):
            loc = Path(location)
            loc.mkdir(parents=True, exist_ok=True)
            (loc / "data.yaml").write_text(
                "names:\n  - pitch\nkpt_shape: [32, 3]\n",
                encoding="utf-8",
            )
            return SimpleNamespace(location=str(loc))

    class FakeProject:
        def versions(self):
            return [FakeVersion()]

        def version(self, ver_num):
            return FakeVersion()

    class FakeRoboflow:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def workspace(self, ws):
            return SimpleNamespace(project=lambda p: FakeProject())

    class FakeYOLO:
        def __init__(self, base_model):
            self.trainer = SimpleNamespace(save_dir=str(save_dir), best=str(best_pt))

        def train(self, **kwargs):
            return SimpleNamespace()

    import ultralytics

    out_target = tmp_path / "models" / "football_pitch_pose.pt"
    monkeypatch.setenv("ROBOFLOW_API_KEY", "DUMMY_ENV_TOKEN")
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_PATH", str(out_target))
    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=FakeRoboflow))
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    train_pk.main(["--work-dir", str(work_dir), "--epochs", "1", "--cleanup"])

    assert out_target.is_file()
    assert out_target.read_bytes() == b"POSE_BEST_WEIGHTS"
    assert not (work_dir / "datasets" / "football-field-detection-f07vi-v4").exists()
    assert not save_dir.exists()
    assert (external_keep / "marker.txt").is_file()


def test_r3_validate_pose_data_yaml_contract_and_edge_cases(tmp_path):
    import utils.train_pitch_keypoints as train_pk

    valid_list = tmp_path / "valid_list.yaml"
    valid_list.write_text(
        "names:\n  - pitch\nkpt_shape: [32, 3]\n",
        encoding="utf-8",
    )
    meta_list = train_pk.validate_pose_data_yaml(valid_list)
    assert meta_list["names"] == {0: "pitch"}
    assert meta_list["kpt_shape"] == [32, 3]
    assert meta_list["flip_idx"] is None

    flip_perm = list(reversed(range(32)))
    valid_dict = tmp_path / "valid_dict.yaml"
    valid_dict.write_text(
        f"names:\n  0: football-pitch\nkpt_shape: [32, 2]\nflip_idx: {flip_perm}\n",
        encoding="utf-8",
    )
    meta_dict = train_pk.validate_pose_data_yaml(valid_dict)
    assert meta_dict["names"] == {0: "football-pitch"}
    assert meta_dict["kpt_shape"] == [32, 2]
    assert meta_dict["flip_idx"] == flip_perm

    with pytest.raises(FileNotFoundError):
        train_pk.validate_pose_data_yaml(tmp_path / "missing.yaml")

    invalid_yaml_cases = [
        ("empty_names.yaml", "names: []\nkpt_shape: [32, 3]\n"),
        ("missing_names.yaml", "kpt_shape: [32, 3]\n"),
        ("blank_name.yaml", "names:\n  - ''\nkpt_shape: [32, 3]\n"),
        ("missing_kpt_shape.yaml", "names:\n  - pitch\n"),
        ("wrong_kpt_count_17.yaml", "names:\n  - pitch\nkpt_shape: [17, 3]\n"),
        ("wrong_kpt_count_3.yaml", "names:\n  - pitch\nkpt_shape: [3, 3]\n"),
        ("wrong_dims_4.yaml", "names:\n  - pitch\nkpt_shape: [32, 4]\n"),
        ("bool_kpt_shape.yaml", "names:\n  - pitch\nkpt_shape: [true, 3]\n"),
        ("flip_short.yaml", "names:\n  - pitch\nkpt_shape: [32, 3]\nflip_idx: [0, 1, 2]\n"),
        (
            "flip_oob.yaml",
            f"names:\n  - pitch\nkpt_shape: [32, 3]\nflip_idx: {list(range(31)) + [32]}\n",
        ),
        (
            "flip_neg.yaml",
            f"names:\n  - pitch\nkpt_shape: [32, 3]\nflip_idx: {[-1] + list(range(1, 32))}\n",
        ),
        (
            "flip_dup.yaml",
            f"names:\n  - pitch\nkpt_shape: [32, 3]\nflip_idx: {[0] + list(range(31))}\n",
        ),
    ]
    for fname, content in invalid_yaml_cases:
        p = tmp_path / fname
        p.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="."):
            train_pk.validate_pose_data_yaml(p)


def test_r3_train_pitch_keypoints_version_selection_vram_kwargs_and_copy(tmp_path, monkeypatch, capsys):
    import tempfile
    import utils.train_pitch_keypoints as train_pk

    requested = []

    class VersionProj:
        def versions(self):
            return [
                SimpleNamespace(version="2"),
                SimpleNamespace(version="12"),
                SimpleNamespace(version="9"),
            ]

        def version(self, ver_num):
            requested.append(ver_num)
            return SimpleNamespace(version=str(ver_num))

    vp = VersionProj()
    assert train_pk.select_dataset_version(vp, None).version == "12"
    assert train_pk.select_dataset_version(vp, 9).version == "9"
    assert requested == [12, 9]

    work_dir = tmp_path / "work_vram"
    save_dir = work_dir / "runs" / "football_pitch_pose"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True)
    expected_best = weights_dir / "best.pt"
    expected_best.write_bytes(b"PITCH_POSE_WEIGHTS_V12")
    expected_sha = hashlib.sha256(b"PITCH_POSE_WEIGHTS_V12").hexdigest()

    train_calls = []
    init_models = []

    class FakeVersion:
        version = "12"

        def download(self, fmt, location=None, overwrite=False):
            loc = Path(location)
            loc.mkdir(parents=True, exist_ok=True)
            (loc / "data.yaml").write_text(
                "names:\n  0: pitch\nkpt_shape: [32, 3]\n",
                encoding="utf-8",
            )
            return SimpleNamespace(location=str(loc))

    class FakeProject:
        def versions(self):
            return [SimpleNamespace(version="2"), SimpleNamespace(version="12")]

        def version(self, ver_num):
            return FakeVersion()

    class FakeRoboflow:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def workspace(self, ws):
            return SimpleNamespace(project=lambda p: FakeProject())

    class FakeYOLO:
        def __init__(self, base_model):
            init_models.append(base_model)
            self.trainer = SimpleNamespace(save_dir=str(save_dir), best=str(expected_best))

        def train(self, **kwargs):
            train_calls.append(kwargs)
            return SimpleNamespace()

    import ultralytics

    default_dest = tmp_path / "models" / "football_pitch_pose.pt"
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "DUMMY_ENV_TOKEN")
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_PATH", str(default_dest))
    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=FakeRoboflow))
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    train_pk.main(["--work-dir", str(work_dir), "--epochs", "2"])

    assert init_models == ["yolo11s-pose.pt"]
    assert len(train_calls) == 1
    kwargs = train_calls[0]
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 4
    assert kwargs["cache"] is False
    assert kwargs["amp"] is True
    assert kwargs["workers"] <= 2
    assert Path(kwargs["project"]).resolve() == (work_dir / "runs").resolve()

    assert default_dest.is_file()
    assert default_dest.read_bytes() == b"PITCH_POSE_WEIGHTS_V12"
    yolo_cfg = Path(os.environ["YOLO_CONFIG_DIR"]).resolve()
    assert yolo_cfg.is_relative_to(Path(tempfile.gettempdir()).resolve())

    out_txt = capsys.readouterr().out
    assert "Eğitim tamamlandı" in out_txt
    assert expected_sha in out_txt
    assert "DUMMY_ENV_TOKEN" not in out_txt

    # --output argümanı config.PITCH_KEYPOINT_MODEL_PATH yerine özel yolu kullanmalı
    custom_dest = tmp_path / "custom" / "my_pitch.pt"
    train_pk.main(["--work-dir", str(work_dir), "--epochs", "1", "--output", str(custom_dest)])
    assert custom_dest.is_file()
    assert custom_dest.read_bytes() == b"PITCH_POSE_WEIGHTS_V12"


def test_r3_download_models_target_pitch_resolution_and_cli_aliases(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    args_default = dm.parse_args([])
    assert args_default.target == "detector"
    assert args_default.output is None

    assert dm.parse_args(["--target", "pitch"]).target == "pitch"
    assert dm.parse_args(["--target", "pitch"]).output is None
    assert dm.parse_args(["--model", "pitch"]).target == "pitch"

    with pytest.raises(SystemExit):
        dm.parse_args(["--target", "invalid_target"])

    # Boş PITCH_KEYPOINT_MODEL_DOWNLOAD_URL anlamlı Türkçe hata döndürmeli
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_DOWNLOAD_URL", "")
    rc_empty = dm.main(["--target", "pitch"])
    assert rc_empty == 1
    err_empty = capsys.readouterr().err
    assert "PITCH_KEYPOINT_MODEL_DOWNLOAD_URL" in err_empty
    assert "train_pitch_keypoints.py" in err_empty

    payload = b"FAKE_PITCH_POSE_MODEL_BYTES"
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

    pitch_dest = tmp_path / "models" / "football_pitch_pose.pt"
    det_dest = tmp_path / "models" / "football_yolo11s_best.pt"
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_PATH", str(pitch_dest))
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_DOWNLOAD_URL", "https://example.com/pitch.pt")
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", good_sha)
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))

    assert dm.main(["--target", "pitch"]) == 0
    assert pitch_dest.is_file() and pitch_dest.read_bytes() == payload

    assert dm.main(["--model", "pitch", "--overwrite"]) == 0
    assert pitch_dest.read_bytes() == payload

    custom_out = tmp_path / "custom_pitch.pt"
    assert dm.main(["--target", "pitch", "--output", str(custom_out)]) == 0
    assert custom_out.is_file() and custom_out.read_bytes() == payload

    # Varsayılan detector hedefi config.YOLO_MODEL_PATH / MODEL_DOWNLOAD_URL / SECURITY_MODEL_SHA256 kullanmalı
    det_payload = b"FAKE_DETECTOR_BYTES"
    det_sha = hashlib.sha256(det_payload).hexdigest()
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", str(det_dest))
    monkeypatch.setattr(config, "MODEL_DOWNLOAD_URL", "https://example.com/det.pt")
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", det_sha)
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(det_payload))

    assert dm.main([]) == 0
    assert det_dest.is_file() and det_dest.read_bytes() == det_payload


def test_r3_download_models_pitch_sha256_mandatory_bypass_and_part_cleanup(tmp_path, monkeypatch, capsys):
    import utils.download_models as dm

    called = {"urlopen": 0}

    def _fail_urlopen(*args, **kwargs):
        called["urlopen"] += 1
        raise AssertionError("SHA-256 olmadan veya geçersiz URL ile indirme yapılmamalı!")

    # SECURITY_MODEL_SHA256 dolu olsa bile pitch hedefi için SECURITY_PITCH_MODEL_SHA256 zorunlu
    monkeypatch.setattr(config, "SECURITY_MODEL_SHA256", "a" * 64)
    monkeypatch.setattr(config, "SECURITY_PITCH_MODEL_SHA256", None)
    monkeypatch.setattr(dm.urllib.request, "urlopen", _fail_urlopen)

    out_pt = tmp_path / "pitch_target.pt"
    rc_no_sha = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
    ])
    assert rc_no_sha != 0
    assert called["urlopen"] == 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    assert "SHA-256" in capsys.readouterr().err

    # Geçersiz URL şeması ağ çağrısı yapmadan reddedilmeli
    for bad_url in ("ftp://example.com/pitch.pt", "file:///tmp/pitch.pt"):
        rc_bad_url = dm.main([
            "--target", "pitch",
            "--url", bad_url,
            "--output", str(out_pt),
            "--sha256", "a" * 64,
        ])
        assert rc_bad_url != 0
        assert called["urlopen"] == 0

    payload = b"PITCH_MODEL_PAYLOAD_BYTES"
    good_sha = hashlib.sha256(payload).hexdigest()

    class FakeResp:
        def __init__(self, data: bytes, fail_after_first: bool = False):
            self._data = data
            self._pos = 0
            self._fail = fail_after_first
            self._reads = 0

        def read(self, n: int = -1):
            self._reads += 1
            if self._fail and self._reads > 1:
                raise OSError("akış hatası")
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos:self._pos + 8] if n < 0 else self._data[self._pos:self._pos + min(n, 8)]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    # Yanlış SHA-256 -> .part temizlenmeli ve hedef oluşturulmamalı
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))
    rc_bad_sha = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--sha256", "0" * 64,
    ])
    assert rc_bad_sha != 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()
    assert "SHA-256" in capsys.readouterr().err

    # Akış ortasında OSError -> .part temizlenmeli ve hedef oluşturulmamalı
    monkeypatch.setattr(
        dm.urllib.request,
        "urlopen",
        lambda req, timeout=120.0: FakeResp(payload, fail_after_first=True),
    )
    rc_io_err = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--sha256", good_sha,
    ])
    assert rc_io_err != 0
    assert not out_pt.exists()
    assert not out_pt.with_suffix(".pt.part").exists()

    # --insecure-skip-sha256 uyarısı ve temiz kayıt
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))
    rc_skip = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--insecure-skip-sha256",
    ])
    assert rc_skip == 0
    assert out_pt.is_file() and out_pt.read_bytes() == payload
    assert not out_pt.with_suffix(".pt.part").exists()
    out_skip = capsys.readouterr().out
    assert "UYARI" in out_skip and "SHA-256" in out_skip

    # Mevcut dosya koruması: eşleşen SHA-256 ağ çağrısı yapmadan geçer
    monkeypatch.setattr(dm.urllib.request, "urlopen", _fail_urlopen)
    rc_existing_ok = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--sha256", good_sha,
    ])
    assert rc_existing_ok == 0
    assert called["urlopen"] == 0

    # Bozuk mevcut dosya --overwrite olmadan reddedilir, --overwrite ile yenilenir
    out_pt.write_bytes(b"corrupted")
    rc_existing_bad = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--sha256", good_sha,
    ])
    assert rc_existing_bad != 0
    assert "--overwrite" in capsys.readouterr().err

    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda req, timeout=120.0: FakeResp(payload))
    rc_overwrite = dm.main([
        "--target", "pitch",
        "--url", "https://example.com/pitch.pt",
        "--output", str(out_pt),
        "--sha256", good_sha,
        "--overwrite",
    ])
    assert rc_overwrite == 0
    assert out_pt.read_bytes() == payload


def test_r3_requires_pitch_model_marker_skip_isolated_from_detector(tmp_path, monkeypatch):
    import tests.conftest as root_conftest

    det_file = tmp_path / "detector.pt"
    det_file.write_bytes(b"det")
    pitch_file = tmp_path / "pitch.pt"

    monkeypatch.setattr(config, "YOLO_MODEL_PATH", str(det_file))
    monkeypatch.setattr(config, "PITCH_KEYPOINT_MODEL_PATH", str(pitch_file))

    pitch_item = SimpleNamespace(
        get_closest_marker=lambda name: object() if name == "requires_pitch_model" else None
    )
    det_item = SimpleNamespace(
        get_closest_marker=lambda name: object() if name == "requires_model" else None
    )

    # Dedektör modeli mevcut olsa bile saha modeli yoksa requires_pitch_model atlamalı
    with pytest.raises(pytest.skip.Exception) as exc_info:
        root_conftest.pytest_runtest_setup(pitch_item)
    assert "models/football_pitch_pose.pt" in str(exc_info.value)
    assert "Model ağırlıkları" in str(exc_info.value)

    # requires_model ise dedektör mevcut olduğu için atlamamalı
    root_conftest.pytest_runtest_setup(det_item)

    # Saha modeli oluşturulduğunda requires_pitch_model da atlamamalı
    pitch_file.write_bytes(b"pitch")
    root_conftest.pytest_runtest_setup(pitch_item)
