"""
pitch_calibrator.py
-------------------
PHASE A2: OTOMATİK SAHA ANAHTAR NOKTA (32 KEYPOINT) KALİBRASYONU VE HİBRİT TAKİP

Bu modül iki ana bileşen sunar:
    1) AutoPitchCalibrator:
       YOLO-Pose saha anahtar nokta modeliyle (`models/football_pitch_pose.pt`)
       32 standart IFAB 105m x 68m saha noktasını tespit eder, piksel alanında
       RANSAC + geometrik doğrulama (yeniden projeksiyon hatası, determinant,
       yerel Jacobian, w' işareti ve ölçek sınırları) ile homografi hesaplar.
       Model ağırlığı yoksa OpenCV çizgi/Hough kestirimine düşmeden temiz
       şekilde kalibrasyonsuz (T2) moda geçer.

    2) HybridPitchCalibrator:
       AutoPitchCalibrator + PerspectiveTransformer + CameraMotionTracker
       üçlüsünü birleştirir:
         - İlk taktik karede (giriş/yakın çekim geçildikten sonra) kilitler.
         - Sahne kesmesi / tekrar sonrası dönüşte (ORB çakışması olmasa bile,
           ör. karşı yarı sahaya kesmede) anahtar noktalarla yeniden kilitler.
         - Periyodik aralıklarla (varsayılan 15 kare) kamera takipçi çapasını
           ve anahtar kare bankasını taşıyarak (reanchor) kaymayı yumuşak
           geçişle (smooth blend) sıfırlar.
"""

from dataclasses import dataclass
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.camera_motion_tracker import CameraMotionTracker
from modules.perspective_transformer import PerspectiveTransformer
from modules.security import validate_model_weight_path

_REAL_YOLO_MODULE = YOLO.__module__


@dataclass
class CalibrationResult:
    """Tek bir kare için otomatik anahtar nokta kalibrasyonu çıktısı."""

    success: bool
    homography_matrix: Optional[np.ndarray] = None
    num_keypoints: int = 0
    num_inliers: int = 0
    reproj_error_px: Optional[float] = None
    reproj_error_m: Optional[float] = None
    reason: str = ""


class AutoPitchCalibrator:
    """
    YOLO-Pose tabanlı 32-keypoint otomatik saha kalibrasyon sınıfı.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, os.PathLike]] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        half: Optional[bool] = None,
        det_conf: Optional[float] = None,
        keypoint_conf_threshold: Optional[float] = None,
        min_keypoints: Optional[int] = None,
        ransac_thresh_px: Optional[float] = None,
        max_reproj_err_px: Optional[float] = None,
        max_reproj_err_m: Optional[float] = None,
        keypoint_template_meters: Optional[Sequence[Tuple[float, float]]] = None,
        model: Optional[Any] = None,
    ):
        self.model_path = (
            os.fspath(model_path)
            if model_path is not None
            else str(getattr(config, "PITCH_KEYPOINT_MODEL_PATH", os.path.join(config.MODELS_DIR, "football_pitch_pose.pt")))
        )
        raw_device = str(device if device is not None else getattr(config, "DEVICE", "cpu"))
        if raw_device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = raw_device

        raw_half = bool(half if half is not None else getattr(config, "PITCH_KEYPOINT_HALF", True))
        self.half = bool(raw_half and self.device.startswith("cuda"))
        self.imgsz = int(imgsz if imgsz is not None else getattr(config, "PITCH_KEYPOINT_IMGSZ", 640))
        self.det_conf = float(det_conf if det_conf is not None else getattr(config, "PITCH_KEYPOINT_DET_CONF_THRESHOLD", 0.25))
        self.keypoint_conf_threshold = float(
            keypoint_conf_threshold
            if keypoint_conf_threshold is not None
            else getattr(config, "PITCH_KEYPOINT_CONF_THRESHOLD", 0.50)
        )
        self.min_keypoints = max(
            4,
            int(
                min_keypoints
                if min_keypoints is not None
                else getattr(config, "PITCH_CALIBRATION_MIN_KEYPOINTS", 4)
            ),
        )
        self.ransac_thresh_px = float(
            ransac_thresh_px
            if ransac_thresh_px is not None
            else getattr(config, "PITCH_CALIBRATION_RANSAC_PX", 8.0)
        )
        self.max_reproj_err_px = float(
            max_reproj_err_px
            if max_reproj_err_px is not None
            else getattr(config, "PITCH_CALIBRATION_MAX_REPROJ_ERR_PX", 12.0)
        )
        self.max_reproj_err_m = float(
            max_reproj_err_m
            if max_reproj_err_m is not None
            else getattr(config, "PITCH_CALIBRATION_MAX_REPROJ_ERR_M", 2.0)
        )

        template = (
            keypoint_template_meters
            if keypoint_template_meters is not None
            else getattr(config, "PITCH_KEYPOINTS_32_METERS", ())
        )
        self.keypoint_template_meters: Tuple[Tuple[float, float], ...] = tuple(
            (float(pt[0]), float(pt[1])) for pt in template
        )

        self.model: Optional[Any] = None
        self.is_available: bool = False
        self.unavailable_reason: str = ""

        if model is not None:
            self.model = model
            self.is_available = True
            return

        is_mock_yolo = getattr(YOLO, "__module__", "") != _REAL_YOLO_MODULE
        if is_mock_yolo and str(self.model_path) == "dummy.pt":
            self.model = YOLO("dummy.pt")
            self.is_available = True
            return

        validated_path = validate_model_weight_path(
            self.model_path,
            sha256_config_attr="SECURITY_PITCH_MODEL_SHA256",
        )
        if not validated_path.is_file():
            self.unavailable_reason = f"Saha keypoint modeli bulunamadı: {validated_path}"
            print(
                f"[AutoPitchCalibrator] BİLGİ: {self.unavailable_reason}. "
                f"Kalibrasyonsuz (T2) modda devam ediliyor."
            )
            return

        self.model = YOLO(str(validated_path))
        self.is_available = True
        print(
            f"[AutoPitchCalibrator] Saha anahtar nokta modeli yüklendi: {validated_path} "
            f"(device={self.device}, imgsz={self.imgsz}, half={self.half})"
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _to_numpy(val: Any) -> Optional[np.ndarray]:
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        if hasattr(val, "detach"):
            val = val.detach()
        if hasattr(val, "cpu"):
            val = val.cpu()
        if hasattr(val, "numpy"):
            return np.asarray(val.numpy())
        try:
            return np.asarray(val)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------
    @staticmethod
    def _is_non_collinear(
        pts: np.ndarray,
        min_sv0: float,
        min_sv1: float,
        min_ratio: float,
    ) -> bool:
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return False
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        try:
            sv = np.linalg.svd(centered, compute_uv=False)
        except np.linalg.LinAlgError:
            return False
        if len(sv) < 2:
            return False
        sv0 = float(sv[0])
        sv1 = float(sv[1])
        if not (math.isfinite(sv0) and math.isfinite(sv1)):
            return False
        if sv0 < min_sv0 or sv1 < min_sv1:
            return False
        return (sv1 / max(sv0, 1e-12)) > min_ratio

    # ----------------------------------------------------------------
    @staticmethod
    def _triangle_non_collinear(
        pts: np.ndarray,
        i: int,
        j: int,
        k: int,
        min_altitude: float,
        min_ratio: float,
    ) -> bool:
        """Üç noktanın oluşturduğu üçgenin dik yüksekliğini ve en-boy oranını doğrular."""
        ax, ay = float(pts[i, 0]), float(pts[i, 1])
        bx, by = float(pts[j, 0]), float(pts[j, 1])
        cx, cy = float(pts[k, 0]), float(pts[k, 1])

        d1_sq = (bx - ax) ** 2 + (by - ay) ** 2
        d2_sq = (cx - bx) ** 2 + (cy - by) ** 2
        d3_sq = (ax - cx) ** 2 + (ay - cy) ** 2
        l_max_sq = max(d1_sq, d2_sq, d3_sq)
        if l_max_sq <= 1e-18:
            return False

        l_max = math.sqrt(l_max_sq)
        area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        h_perp = area2 / l_max
        return h_perp >= min_altitude and (h_perp / l_max) >= min_ratio

    # ----------------------------------------------------------------
    @staticmethod
    def _has_at_least_n_minus_1_on_exact_line(
        pts: np.ndarray,
        a_idx: int,
        b_idx: int,
        tol: float,
    ) -> bool:
        """a_idx-b_idx doğrusuna tol mesafe içinde en az N-1 nokta bulunup bulunmadığını kontrol eder."""
        n = int(pts.shape[0])
        if n < 4 or a_idx == b_idx:
            return False
        dx = float(pts[b_idx, 0] - pts[a_idx, 0])
        dy = float(pts[b_idx, 1] - pts[a_idx, 1])
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-12:
            return False
        rel = pts - pts[a_idx]
        perp = np.abs(dx * rel[:, 1] - dy * rel[:, 0]) / seg_len
        return int(np.sum(perp <= tol)) >= (n - 1)

    # ----------------------------------------------------------------
    @classmethod
    def _has_general_position_4tuple(
        cls,
        px_pts: np.ndarray,
        m_pts: np.ndarray,
        min_px_altitude: float = 4.0,
        min_px_ratio: float = 0.015,
        min_m_altitude: float = 0.5,
        min_m_ratio: float = 0.02,
    ) -> bool:
        """
        N >= 4 nokta içinde hem piksel hem de metre uzayında 4 üçlüsü de
        doğrudaş olmayan (general position) en az bir 4'lü alt küme arar.
        """
        n = int(px_pts.shape[0])
        if n < 4 or int(m_pts.shape[0]) != n:
            return False

        def pair_tri_ok(a: int, b: int, c: int) -> bool:
            return cls._triangle_non_collinear(
                m_pts, a, b, c, min_m_altitude, min_m_ratio
            ) and cls._triangle_non_collinear(
                px_pts, a, b, c, min_px_altitude, min_px_ratio
            )

        # Hızlı yol (O(N)): en uzak metre çiftinden başlayarak geçerli 4'lü ara
        d0 = np.sum((m_pts - m_pts[0]) ** 2, axis=1)
        i0 = int(np.argmax(d0))
        d1 = np.sum((m_pts - m_pts[i0]) ** 2, axis=1)
        i1 = int(np.argmax(d1))
        chosen_i2: Optional[int] = None
        if i0 != i1:
            for i2 in range(n):
                if i2 == i0 or i2 == i1:
                    continue
                if pair_tri_ok(i0, i1, i2):
                    chosen_i2 = i2
                    for i3 in range(i2 + 1, n):
                        if i3 == i0 or i3 == i1:
                            continue
                        if (
                            pair_tri_ok(i0, i1, i3)
                            and pair_tri_ok(i0, i2, i3)
                            and pair_tri_ok(i1, i2, i3)
                        ):
                            return True
                    break

        # En az N-1 nokta tek bir doğru üzerindeyse hiçbir 4'lü genel konumda olamaz (O(N) hızlı ret)
        if i0 != i1 and (
            cls._has_at_least_n_minus_1_on_exact_line(m_pts, i0, i1, 1e-6)
            or cls._has_at_least_n_minus_1_on_exact_line(px_pts, i0, i1, 1e-4)
        ):
            return False
        if chosen_i2 is not None:
            for a, b in ((i0, chosen_i2), (i1, chosen_i2)):
                if (
                    cls._has_at_least_n_minus_1_on_exact_line(m_pts, a, b, 1e-6)
                    or cls._has_at_least_n_minus_1_on_exact_line(px_pts, a, b, 1e-4)
                ):
                    return False

        # Budamalı tam arama: geçersiz (i, j, k) üçlülerini l döngüsüne girmeden eler
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    if not pair_tri_ok(i, j, k):
                        continue
                    for l in range(k + 1, n):
                        if (
                            pair_tri_ok(i, j, l)
                            and pair_tri_ok(i, k, l)
                            and pair_tri_ok(j, k, l)
                        ):
                            return True
        return False

    # ----------------------------------------------------------------
    @staticmethod
    def _local_jacobian_det(
        H: np.ndarray,
        u: float,
        v: float,
        X: float,
        Y: float,
        w: float,
    ) -> float:
        del u, v
        j00 = (float(H[0, 0]) - X * float(H[2, 0])) / w
        j01 = (float(H[0, 1]) - X * float(H[2, 1])) / w
        j10 = (float(H[1, 0]) - Y * float(H[2, 0])) / w
        j11 = (float(H[1, 1]) - Y * float(H[2, 1])) / w
        return float(j00 * j11 - j01 * j10)

    # ----------------------------------------------------------------
    def detect_keypoints(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Tek bir kare üzerinde YOLO-Pose çalıştırır ve güven eşiği üzerindeki
        saha anahtar noktalarını (piksel, metre, indeks) olarak döndürür.
        """
        if not self.is_available or self.model is None:
            return None
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return None
        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        if frame_h <= 0 or frame_w <= 0 or frame.size == 0:
            return None

        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                conf=self.det_conf,
                device=self.device,
                imgsz=self.imgsz,
                half=self.half,
                verbose=False,
            )

        if not results:
            return None

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None

        xy_arr = self._to_numpy(getattr(keypoints, "xy", None))
        conf_arr = self._to_numpy(getattr(keypoints, "conf", None))
        data_arr = self._to_numpy(getattr(keypoints, "data", None))

        if xy_arr is None and data_arr is not None:
            data_f = np.asarray(data_arr, dtype=np.float64)
            if data_f.ndim == 2 and data_f.shape[-1] >= 2:
                data_f = data_f[np.newaxis, ...]
            if data_f.ndim == 3 and data_f.shape[-1] >= 2:
                xy_arr = data_f[:, :, :2]
                if conf_arr is None and data_f.shape[-1] >= 3:
                    conf_arr = data_f[:, :, 2]

        if xy_arr is None:
            return None

        xy_arr = np.asarray(xy_arr, dtype=np.float64)
        if xy_arr.ndim == 2 and xy_arr.shape[-1] >= 2:
            xy_arr = xy_arr[np.newaxis, :, :2]
        elif xy_arr.ndim == 3 and xy_arr.shape[-1] >= 2:
            xy_arr = xy_arr[:, :, :2]
        else:
            return None

        n_inst, n_kpts, _ = xy_arr.shape
        if n_inst == 0 or n_kpts == 0:
            return None

        if conf_arr is None and data_arr is not None:
            data_f = np.asarray(data_arr, dtype=np.float64)
            if data_f.ndim == 2 and data_f.shape[-1] >= 3:
                data_f = data_f[np.newaxis, ...]
            if data_f.ndim == 3 and data_f.shape[0] == n_inst and data_f.shape[1] == n_kpts and data_f.shape[2] >= 3:
                conf_arr = data_f[:, :, 2]

        if conf_arr is None:
            finite_xy = np.isfinite(xy_arr[:, :, 0]) & np.isfinite(xy_arr[:, :, 1])
            non_zero = (xy_arr[:, :, 0] != 0.0) | (xy_arr[:, :, 1] != 0.0)
            conf_arr = np.where(finite_xy & non_zero, 1.0, 0.0)
        else:
            conf_arr = np.asarray(conf_arr, dtype=np.float64)
            if conf_arr.ndim == 1 and conf_arr.shape[0] == n_kpts:
                conf_arr = conf_arr[np.newaxis, :]
            if conf_arr.shape != (n_inst, n_kpts):
                return None

        # Birden fazla saha tespiti varsa en güvenilir örneği seç
        best_idx = 0
        if n_inst > 1:
            boxes = getattr(result, "boxes", None)
            box_conf = self._to_numpy(getattr(boxes, "conf", None)) if boxes is not None else None
            if box_conf is not None:
                box_conf_1d = np.asarray(box_conf, dtype=np.float64).ravel()
                if len(box_conf_1d) == n_inst and bool(np.isfinite(box_conf_1d).any()):
                    best_idx = int(np.nanargmax(box_conf_1d))
                else:
                    box_conf = None
            if box_conf is None:
                best_score = (-1, -1.0)
                for idx in range(n_inst):
                    c_row = conf_arr[idx]
                    valid_mask = np.isfinite(c_row) & (c_row >= self.keypoint_conf_threshold)
                    cnt = int(valid_mask.sum())
                    mean_c = float(np.mean(c_row[valid_mask])) if cnt > 0 else -1.0
                    score = (cnt, mean_c)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

        k_limit = min(n_kpts, len(self.keypoint_template_meters))
        min_x = -0.05 * float(frame_w)
        max_x = 1.05 * float(frame_w)
        min_y = -0.05 * float(frame_h)
        max_y = 1.05 * float(frame_h)

        pixel_list: List[Tuple[float, float]] = []
        meter_list: List[Tuple[float, float]] = []
        idx_list: List[int] = []

        for k in range(k_limit):
            px = float(xy_arr[best_idx, k, 0])
            py = float(xy_arr[best_idx, k, 1])
            c = float(conf_arr[best_idx, k])
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(c)):
                continue
            if c < self.keypoint_conf_threshold:
                continue
            if not (min_x <= px <= max_x and min_y <= py <= max_y):
                continue
            mx, my = self.keypoint_template_meters[k]
            pixel_list.append((px, py))
            meter_list.append((float(mx), float(my)))
            idx_list.append(k)

        if not pixel_list:
            return None

        return (
            np.asarray(pixel_list, dtype=np.float64),
            np.asarray(meter_list, dtype=np.float64),
            np.asarray(idx_list, dtype=np.int64),
        )

    # ----------------------------------------------------------------
    def estimate_homography_from_keypoints(
        self,
        pixel_pts: np.ndarray,
        meter_pts: np.ndarray,
        frame_shape: Tuple[int, ...],
    ) -> CalibrationResult:
        """
        Piksel <-> metre anahtar nokta eşleşmelerinden RANSAC ile piksel->metre
        homografi matrisi üretir ve tüm geometrik tutarlılık kontrollerinden geçirir.
        """
        if frame_shape is None or len(frame_shape) < 2:
            return CalibrationResult(success=False, reason="invalid_frame_shape")
        try:
            frame_h = float(frame_shape[0])
            frame_w = float(frame_shape[1])
        except (TypeError, ValueError):
            return CalibrationResult(success=False, reason="invalid_frame_shape")
        if not (math.isfinite(frame_h) and math.isfinite(frame_w)) or frame_h <= 0.0 or frame_w <= 0.0:
            return CalibrationResult(success=False, reason="invalid_frame_shape")

        try:
            px_arr = np.asarray(pixel_pts, dtype=np.float64)
            m_arr = np.asarray(meter_pts, dtype=np.float64)
        except (TypeError, ValueError):
            return CalibrationResult(success=False, reason="invalid_keypoint_arrays")

        if (
            px_arr.ndim != 2
            or m_arr.ndim != 2
            or px_arr.shape[1] != 2
            or m_arr.shape[1] != 2
            or px_arr.shape[0] != m_arr.shape[0]
        ):
            return CalibrationResult(success=False, reason="invalid_keypoint_shapes")

        n_pts = int(px_arr.shape[0])
        if not (bool(np.isfinite(px_arr).all()) and bool(np.isfinite(m_arr).all())):
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="non_finite_keypoints")

        min_req = max(4, self.min_keypoints)
        if n_pts < min_req:
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="insufficient_keypoints")

        unique_m = {(round(float(x), 3), round(float(y), 3)) for x, y in m_arr}
        if len(unique_m) < 4:
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="duplicate_meter_keypoints")

        unique_px = {(round(float(x), 1), round(float(y), 1)) for x, y in px_arr}
        if len(unique_px) < 4:
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="duplicate_pixel_keypoints")

        # 2D doğrudaş olmama ve 4-nokta genel konum (general position) kontrolü
        if (
            not self._is_non_collinear(px_arr, 20.0, 4.0, 0.03)
            or not self._is_non_collinear(m_arr, 2.0, 0.5, 0.03)
            or not self._has_general_position_4tuple(px_arr, m_arr)
        ):
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="collinear_keypoints")

        # Piksel alanında RANSAC (8.0 px eşiği doğrudan piksel hatasına uygulanır)
        H_m_to_px, mask = cv2.findHomography(
            m_arr,
            px_arr,
            cv2.RANSAC,
            float(self.ransac_thresh_px),
        )
        if H_m_to_px is None:
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="find_homography_failed")

        H_m_to_px = np.asarray(H_m_to_px, dtype=np.float64)
        if H_m_to_px.shape != (3, 3) or not bool(np.isfinite(H_m_to_px).all()):
            return CalibrationResult(success=False, num_keypoints=n_pts, reason="non_finite_homography")

        inlier_mask = (
            mask.ravel().astype(bool)
            if mask is not None
            else np.ones(n_pts, dtype=bool)
        )
        num_inliers = int(inlier_mask.sum())
        if num_inliers < min_req:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="insufficient_inliers",
            )

        px_in = px_arr[inlier_mask]
        m_in = m_arr[inlier_mask]
        if (
            not self._is_non_collinear(px_in, 20.0, 4.0, 0.03)
            or not self._is_non_collinear(m_in, 2.0, 0.5, 0.03)
            or not self._has_general_position_4tuple(px_in, m_in)
        ):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="collinear_inliers",
            )

        try:
            H_px_to_m_raw = np.linalg.inv(H_m_to_px)
        except np.linalg.LinAlgError:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="singular_homography",
            )

        if not bool(np.isfinite(H_px_to_m_raw).all()):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="non_finite_inverse_homography",
            )

        # Dik yayın açılarında det > 0 kontrolünden ÖNCE görünür inlier noktalar üzerinde w' işaretini kanonikleştir
        w_cal = (
            H_px_to_m_raw[2, 0] * px_in[:, 0]
            + H_px_to_m_raw[2, 1] * px_in[:, 1]
            + H_px_to_m_raw[2, 2]
        )
        if not bool(np.isfinite(w_cal).all()):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="non_finite_w_inliers",
            )
        if float(np.median(w_cal)) < 0.0:
            H_px_to_m_raw = -H_px_to_m_raw

        min_abs_det, max_cond, w_eps, row2_min_norm = (
            PerspectiveTransformer._get_homography_thresholds()
        )
        H_px_to_m = PerspectiveTransformer._validate_and_normalize_matrix(
            H_px_to_m_raw,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )
        if H_px_to_m is None:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="degenerate_or_ill_conditioned",
            )

        w_in = H_px_to_m[2, 0] * px_in[:, 0] + H_px_to_m[2, 1] * px_in[:, 1] + H_px_to_m[2, 2]
        w_m_in = H_m_to_px[2, 0] * m_in[:, 0] + H_m_to_px[2, 1] * m_in[:, 1] + H_m_to_px[2, 2]
        if (
            not bool(np.isfinite(w_in).all())
            or not bool(np.all(w_in > w_eps))
            or not bool(np.isfinite(w_m_in).all())
            or not bool(np.all(np.abs(w_m_in) > 1e-9))
        ):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reason="horizon_or_bowtie_homography",
            )

        # Piksel ve metre alanında yeniden projeksiyon hatası sınırları
        proj_px = cv2.perspectiveTransform(m_in.reshape(1, -1, 2), H_m_to_px)[0]
        err_px = np.linalg.norm(proj_px - px_in, axis=1)
        mean_err_px = float(np.mean(err_px))
        if not math.isfinite(mean_err_px) or mean_err_px > self.max_reproj_err_px:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=mean_err_px if math.isfinite(mean_err_px) else None,
                reason="reproj_error_px_exceeded",
            )

        proj_m = cv2.perspectiveTransform(px_in.reshape(1, -1, 2), H_px_to_m)[0]
        err_m = np.linalg.norm(proj_m - m_in, axis=1)
        mean_err_m = float(np.mean(err_m))
        if not math.isfinite(mean_err_m) or mean_err_m > self.max_reproj_err_m:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=mean_err_m if math.isfinite(mean_err_m) else None,
                reason="reproj_error_m_exceeded",
            )

        # Yönelim (ayna yansımasını reddet) ve her inlier için yerel Jacobian > 0 kontrolü
        try:
            det_h = float(np.linalg.det(H_px_to_m))
        except np.linalg.LinAlgError:
            det_h = 0.0
        if not math.isfinite(det_h) or det_h <= 0.0:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="invalid_orientation_det",
            )

        for i in range(num_inliers):
            det_j = self._local_jacobian_det(
                H_px_to_m,
                float(px_in[i, 0]),
                float(px_in[i, 1]),
                float(proj_m[i, 0]),
                float(proj_m[i, 1]),
                float(w_in[i]),
            )
            if not math.isfinite(det_j) or det_j <= 0.0:
                return CalibrationResult(
                    success=False,
                    num_keypoints=n_pts,
                    num_inliers=num_inliers,
                    reproj_error_px=round(mean_err_px, 4),
                    reproj_error_m=round(mean_err_m, 4),
                    reason="invalid_local_jacobian",
                )

        # Kare merkezi saha sınırları ve yerel ölçek akla yatkınlık kontrolü
        cx = 0.5 * frame_w
        cy = 0.5 * frame_h
        w_c = float(H_px_to_m[2, 0] * cx + H_px_to_m[2, 1] * cy + H_px_to_m[2, 2])
        if not math.isfinite(w_c) or w_c <= w_eps:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="invalid_frame_center_w",
            )

        center_m = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float64), H_px_to_m
        )[0, 0]
        cx_m = float(center_m[0])
        cy_m = float(center_m[1])
        if not (math.isfinite(cx_m) and math.isfinite(cy_m)):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="non_finite_frame_center",
            )

        pitch_len = float(getattr(config, "PITCH_LENGTH_M", 105.0))
        pitch_wid = float(getattr(config, "PITCH_WIDTH_M", 68.0))
        if not (-35.0 <= cx_m <= pitch_len + 35.0 and -25.0 <= cy_m <= pitch_wid + 25.0):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="frame_center_out_of_bounds",
            )

        det_j_c = self._local_jacobian_det(H_px_to_m, cx, cy, cx_m, cy_m, w_c)
        if not math.isfinite(det_j_c) or det_j_c <= 0.0:
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="invalid_center_jacobian",
            )

        scale_m_per_px = math.sqrt(det_j_c)
        if not (0.005 <= scale_m_per_px <= 1.0):
            return CalibrationResult(
                success=False,
                num_keypoints=n_pts,
                num_inliers=num_inliers,
                reproj_error_px=round(mean_err_px, 4),
                reproj_error_m=round(mean_err_m, 4),
                reason="implausible_pitch_scale",
            )

        return CalibrationResult(
            success=True,
            homography_matrix=H_px_to_m.copy(),
            num_keypoints=n_pts,
            num_inliers=num_inliers,
            reproj_error_px=round(mean_err_px, 4),
            reproj_error_m=round(mean_err_m, 4),
            reason="ok",
        )

    # ----------------------------------------------------------------
    def calibrate_frame(self, frame: np.ndarray) -> CalibrationResult:
        """Verilen video karesi için tam otomatik homografi kalibrasyonu çalıştırır."""
        if not self.is_available:
            return CalibrationResult(
                success=False,
                reason=self.unavailable_reason or "pitch_model_unavailable",
            )
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
            return CalibrationResult(success=False, reason="invalid_frame")

        detected = self.detect_keypoints(frame)
        if detected is None:
            return CalibrationResult(success=False, reason="no_valid_keypoints")

        pixel_pts, meter_pts, _indices = detected
        return self.estimate_homography_from_keypoints(pixel_pts, meter_pts, frame.shape)


class HybridPitchCalibrator:
    """
    AutoPitchCalibrator + PerspectiveTransformer + CameraMotionTracker hibrit yöneticisi.
    """

    def __init__(
        self,
        calibrator: AutoPitchCalibrator,
        transformer: PerspectiveTransformer,
        motion_tracker: CameraMotionTracker,
        refresh_interval: Optional[int] = None,
        smooth_frames: Optional[int] = None,
    ):
        self.calibrator = calibrator
        self.transformer = transformer
        self.motion_tracker = motion_tracker
        self.refresh_interval = max(
            1,
            int(
                refresh_interval
                if refresh_interval is not None
                else getattr(config, "PITCH_CALIBRATION_REFRESH_INTERVAL", 15)
            ),
        )
        self.smooth_frames = max(
            1,
            int(
                smooth_frames
                if smooth_frames is not None
                else getattr(config, "PITCH_CALIBRATION_SMOOTH_FRAMES", 5)
            ),
        )
        self.last_calibration_frame: Optional[int] = None
        self.was_non_tactical: bool = False
        self._blend_start_h: Optional[np.ndarray] = None
        self._blend_target_h: Optional[np.ndarray] = None
        self._blend_step: int = 0
        self._blend_total: int = 0
        self.calibration_count: int = 0
        self.last_calibration_reason: str = ""

    # ----------------------------------------------------------------
    def _cancel_blend(self) -> None:
        self._blend_start_h = None
        self._blend_target_h = None
        self._blend_step = 0
        self._blend_total = 0

    # ----------------------------------------------------------------
    @staticmethod
    def _normalize_matrix(matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        min_abs_det, max_cond, _w_eps, row2_min_norm = (
            PerspectiveTransformer._get_homography_thresholds()
        )
        return PerspectiveTransformer._validate_and_normalize_matrix(
            matrix,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _center_displacement_m(
        h_a: np.ndarray, h_b: np.ndarray, frame_shape: Tuple[int, ...]
    ) -> Optional[float]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy = 0.5 * float(w), 0.5 * float(h)
        _, _, w_eps, _ = PerspectiveTransformer._get_homography_thresholds()
        wa = float(h_a[2, 0] * cx + h_a[2, 1] * cy + h_a[2, 2])
        wb = float(h_b[2, 0] * cx + h_b[2, 1] * cy + h_b[2, 2])
        if not (math.isfinite(wa) and math.isfinite(wb)) or wa <= w_eps or wb <= w_eps:
            return None
        pt = np.array([[[cx, cy]]], dtype=np.float64)
        pa = cv2.perspectiveTransform(pt, h_a)[0, 0]
        pb = cv2.perspectiveTransform(pt, h_b)[0, 0]
        dist = float(math.hypot(float(pa[0] - pb[0]), float(pa[1] - pb[1])))
        return dist if math.isfinite(dist) else None

    # ----------------------------------------------------------------
    def _apply_immediate_lock(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        h_kp: np.ndarray,
        reason: str,
    ) -> Optional[np.ndarray]:
        self._cancel_blend()
        if not self.transformer.set_homography(h_kp):
            return None
        reanchored = self.motion_tracker.reanchor(
            frame, raw_detections, keep_keyframe_bank=False
        )
        self.last_calibration_frame = frame_idx
        self.was_non_tactical = False
        self.calibration_count += 1
        self.last_calibration_reason = reason
        return reanchored

    # ----------------------------------------------------------------
    def _start_periodic_smooth_refresh(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion_matrix: np.ndarray,
        h_kp_raw: np.ndarray,
    ) -> np.ndarray:
        h_kp = self._normalize_matrix(h_kp_raw)
        if h_kp is None:
            return motion_matrix

        h_flow_eff = None
        if self.transformer.homography_matrix is not None:
            h_flow_eff = self._normalize_matrix(
                self.transformer.homography_matrix @ motion_matrix
            )

        reanchored = self.motion_tracker.reanchor(
            frame,
            raw_detections,
            keep_keyframe_bank=True,
            prev_cumulative=motion_matrix,
        )

        self.last_calibration_frame = frame_idx
        self.calibration_count += 1
        self.last_calibration_reason = "periodic_refresh"

        # Yumuşak geçiş uygulanabilir mi? (sign-align + sonlu merkez sapması <= 25 m)
        use_blend = False
        if h_flow_eff is not None and self.smooth_frames > 1:
            if float(np.sum(h_flow_eff * h_kp)) < 0.0:
                h_flow_eff = -h_flow_eff
            disp_m = self._center_displacement_m(h_flow_eff, h_kp, frame.shape)
            if disp_m is not None and disp_m <= 25.0:
                mid_h = self._normalize_matrix(0.5 * h_flow_eff + 0.5 * h_kp)
                use_blend = mid_h is not None

        if not use_blend or h_flow_eff is None:
            self._cancel_blend()
            self.transformer.set_homography(h_kp)
            return reanchored

        self._blend_start_h = h_flow_eff.copy()
        self._blend_target_h = h_kp.copy()
        self._blend_total = self.smooth_frames
        self._blend_step = 1

        alpha = 1.0 / float(self._blend_total)
        h_blend = (1.0 - alpha) * self._blend_start_h + alpha * self._blend_target_h
        if not self.transformer.set_homography(h_blend):
            self.transformer.set_homography(h_kp)
            self._cancel_blend()
        elif self._blend_step >= self._blend_total:
            self._cancel_blend()

        return reanchored

    # ----------------------------------------------------------------
    def _advance_active_blend(self) -> None:
        if (
            self._blend_total <= 0
            or self._blend_start_h is None
            or self._blend_target_h is None
        ):
            return
        self._blend_step += 1
        alpha = min(1.0, float(self._blend_step) / float(self._blend_total))
        h_blend = (1.0 - alpha) * self._blend_start_h + alpha * self._blend_target_h
        if not self.transformer.set_homography(h_blend):
            self.transformer.set_homography(self._blend_target_h)
            self._cancel_blend()
            return
        if self._blend_step >= self._blend_total:
            self._cancel_blend()

    # ----------------------------------------------------------------
    def process_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]] = None,
    ) -> Optional[np.ndarray]:
        cal_start = int(getattr(config, "CALIBRATION_FRAME_INDEX", 0))
        if frame_idx < cal_start:
            return None

        motion_matrix = self.motion_tracker.update(frame, raw_detections)
        scene_cut = bool(getattr(self.motion_tracker, "scene_cut_detected", False))
        recovered = bool(getattr(self.motion_tracker, "recovered_this_frame", False))

        if scene_cut:
            self._cancel_blend()
            self.was_non_tactical = True

        # Durum 1: İlk taktik kare kilidi veya kesme/tekrar sonrası geri dönüş
        # (ORB eşleşmesi olmasa bile -- örn. karşı yarı sahaya kesme -- keypoint ile yakala)
        needs_immediate_lock = (
            (not self.transformer.is_calibrated)
            or (motion_matrix is None)
            or recovered
            or self.was_non_tactical
        )
        if needs_immediate_lock:
            cal_res = self.calibrator.calibrate_frame(frame)
            if (
                getattr(cal_res, "success", False)
                and getattr(cal_res, "homography_matrix", None) is not None
            ):
                reason = (
                    "initial_lock"
                    if not self.transformer.is_calibrated
                    else "post_cut_recovery"
                )
                locked_m = self._apply_immediate_lock(
                    frame_idx,
                    frame,
                    raw_detections,
                    cal_res.homography_matrix,
                    reason=reason,
                )
                if locked_m is not None:
                    return locked_m

            # Keypoint kalibrasyonu bu karede başarısız oldu:
            # - Çapa yoksa (motion_matrix is None) veya hiç kalibrasyon yapılmadıysa None dön (NON_TACTICAL / T2).
            # - ORB eski ana kameraya başarıyla döndüyse ve önceden geçerli kalibrasyon varsa
            #   ORB matrisini döndür, ancak was_non_tactical=True bırakarak sonraki karede keypoint yenilemeyi dene.
            if motion_matrix is None or not self.transformer.is_calibrated:
                return None
            return motion_matrix

        # Durum 2: Periyodik keypoint yenileme (drift sıfırlama + yumuşak geçiş)
        periodic_due = (
            self.last_calibration_frame is not None
            and (frame_idx - self.last_calibration_frame) >= self.refresh_interval
        )
        if periodic_due:
            cal_res = self.calibrator.calibrate_frame(frame)
            if (
                getattr(cal_res, "success", False)
                and getattr(cal_res, "homography_matrix", None) is not None
            ):
                return self._start_periodic_smooth_refresh(
                    frame_idx,
                    frame,
                    raw_detections,
                    motion_matrix,
                    cal_res.homography_matrix,
                )
            # Başarısız olursa last_calibration_frame güncellenmez -> sonraki karede yeniden dener.

        # Durum 3: Ara karelerde devam eden yumuşak geçişi ilerlet
        self._advance_active_blend()
        return motion_matrix
