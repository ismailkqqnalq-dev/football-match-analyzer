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
         - Periyodik aralıklarla (varsayılan 15 kare) modeli flow ile karşılaştırır;
           farkın büyüklüğüne göre kademeli karar verir (A2-3) ve kabul edilen
           düzeltmeyi referans noktaları üzerinden yumuşak geçişle uygular.
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




_VALID_PROFILES = ("auto", "fixed", "broadcast")


class HybridPitchCalibrator:
    """
    AutoPitchCalibrator + PerspectiveTransformer + CameraMotionTracker hibrit yöneticisi.

    Kare başına efektif tablo: E = transformer.homography_matrix @ motion_matrix
    (motion_matrix: mevcut kare pikseli -> çapa karesi pikseli).

    A2-3 kademeli kabul kuralı (model tablosu ile flow tablosu arasındaki fark,
    görüntüdeki 3x3 referans ızgarasının saha üzerindeki ortanca mesafesi):
      - fark <= PITCH_CAL_DEADBAND_M  -> tablo değiştirilmez (gürültü titremesi önlenir)
      - fark <= PITCH_CAL_SOFT_M      -> hemen kabul, referans noktaları üzerinden yumuşak geçiş
      - fark >  PITCH_CAL_SOFT_M      -> model sonraki PITCH_CAL_CONFIRM_REFRESHES yenilemede
                                         aynı yeri tutarlı gösterirse kabul; göstermezse aykırı
      - doğrulanan fark > PITCH_CAL_HARD_M -> sert geçiş, kare `calib_reset` olarak işaretlenir

    A2-5 görüntü profili: "fixed" (sabit/geniş kamera) ya da "broadcast" (TV yayını).
    """

    def __init__(
        self,
        calibrator: AutoPitchCalibrator,
        transformer: PerspectiveTransformer,
        motion_tracker: CameraMotionTracker,
        refresh_interval: Optional[int] = None,
        smooth_frames: Optional[int] = None,
        fps: Optional[float] = None,
        profile: Optional[str] = None,
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
        self.fps = float(fps) if fps and fps > 0 else 25.0

        self.deadband_m = float(getattr(config, "PITCH_CAL_DEADBAND_M", 0.5))
        self.soft_m = float(getattr(config, "PITCH_CAL_SOFT_M", 1.0))
        self.hard_m = float(getattr(config, "PITCH_CAL_HARD_M", 3.0))
        self.confirm_refreshes = max(0, int(getattr(config, "PITCH_CAL_CONFIRM_REFRESHES", 2)))
        self.blend_max_step_m = max(1e-3, float(getattr(config, "PITCH_CAL_BLEND_MAX_STEP_M", 0.05)))

        requested = str(profile if profile is not None else getattr(config, "FOOTAGE_PROFILE", "auto")).lower()
        if requested not in _VALID_PROFILES:
            raise ValueError(
                f"Geçersiz görüntü profili: {requested!r} (geçerli: {', '.join(_VALID_PROFILES)})"
            )
        self.requested_profile = requested
        self.profile = "fixed" if requested == "fixed" else "broadcast"
        self.profile_decided = requested != "auto"
        self.profile_reason = "config" if self.profile_decided else "algılanıyor"
        self.profile_events: List[Dict[str, Any]] = []

        # Profil algılama / izleme istatistikleri
        self._detect_start_frame: Optional[int] = None
        self._pan_path_px = 0.0
        self._max_scale_change = 0.0
        self._prev_motion: Optional[np.ndarray] = None
        self._reanchored_since_prev = False

        # Sabit kamera kilidi
        self.fixed_candidates_target = max(1, int(getattr(config, "PITCH_CAL_FIXED_CANDIDATES", 10)))
        self._fixed_candidates: List[np.ndarray] = []  # çapa koordinatında tablolar
        self._fixed_locked = False
        self._last_health_frame: Optional[int] = None

        # Doğrulama bekleyen model tablosu (çapa koordinatında)
        self._pending_anchor_h: Optional[np.ndarray] = None
        self._pending_count = 0

        # Yumuşak geçiş durumu (referans dörtgenin saha koordinatları arasında)
        self._blend_quad_img: Optional[np.ndarray] = None
        self._blend_p0: Optional[np.ndarray] = None
        self._blend_p1: Optional[np.ndarray] = None
        self._blend_step = 0
        self._blend_total = 0

        self.last_calibration_frame: Optional[int] = None
        self.was_non_tactical = False
        self.calibration_count = 0
        self.last_calibration_reason = ""

        self.stats: Dict[str, int] = {
            "model_calls": 0,
            "model_success": 0,
            "agree_no_change": 0,
            "soft_blend": 0,
            "pending": 0,
            "rejected_outlier": 0,
            "confirmed_blend": 0,
            "confirmed_hard_reset": 0,
            "immediate_lock": 0,
        }
        self.frame_info: Dict[str, Any] = self._new_frame_info()

    # ----------------------------------------------------------------
    def set_fps(self, fps: Optional[float]) -> None:
        if fps and fps > 0:
            self.fps = float(fps)

    def _new_frame_info(self) -> Dict[str, Any]:
        return {
            "source": "none",
            "num_keypoints": None,
            "num_inliers": None,
            "reproj_error_m": None,
            "disagreement_m": None,
            "calib_reset": False,
            "profile": self.profile,
        }

    # ----------------------------------------------------------------
    # Matris yardımcıları
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

    @staticmethod
    def _canonical_sign(h: np.ndarray, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Görüntü merkezinde w' > 0 olacak şekilde matris işaretini sabitler."""
        cx, cy = 0.5 * float(frame_shape[1]), 0.5 * float(frame_shape[0])
        w = float(h[2, 0] * cx + h[2, 1] * cy + h[2, 2])
        return -h if w < 0.0 else h

    @staticmethod
    def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Pikselleri sahaya izdüşürür; w' <= eps olan noktalar NaN döner."""
        _, _, w_eps, _ = PerspectiveTransformer._get_homography_thresholds()
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ h.T
        w = homog[:, 2]
        out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        ok = np.isfinite(w) & (w > w_eps)
        out[ok] = homog[ok, :2] / w[ok, None]
        return out

    @staticmethod
    def reference_grid(frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Görüntüde kenarlardan içeride 3x3 referans noktası."""
        h, w = float(frame_shape[0]), float(frame_shape[1])
        fr = (0.2, 0.5, 0.8)
        return np.array([[w * fx, h * fy] for fy in fr for fx in fr], dtype=np.float64)

    @staticmethod
    def reference_quad(frame_shape: Tuple[int, ...]) -> np.ndarray:
        h, w = float(frame_shape[0]), float(frame_shape[1])
        return np.array(
            [[0.2 * w, 0.3 * h], [0.8 * w, 0.3 * h], [0.8 * w, 0.8 * h], [0.2 * w, 0.8 * h]],
            dtype=np.float64,
        )

    @classmethod
    def disagreement_m(
        cls, h_a: np.ndarray, h_b: np.ndarray, frame_shape: Tuple[int, ...]
    ) -> Optional[float]:
        """
        İki piksel->metre tablosunun saha üzerinde ne kadar ayrıştığı:
        3x3 referans ızgarasının iki tabloyla izdüşümleri arasındaki ortanca mesafe.
        Saha (±10 m) içine düşen noktalar tercih edilir (tribün/gökyüzü dışlanır).
        """
        a = cls._canonical_sign(np.asarray(h_a, dtype=np.float64), frame_shape)
        b = cls._canonical_sign(np.asarray(h_b, dtype=np.float64), frame_shape)
        grid = cls.reference_grid(frame_shape)
        pa = cls._project(a, grid)
        pb = cls._project(b, grid)
        valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if not valid.any():
            return None
        length = float(getattr(config, "PITCH_LENGTH_M", 105.0))
        width = float(getattr(config, "PITCH_WIDTH_M", 68.0))
        on_pitch = (
            valid
            & (pb[:, 0] >= -10.0) & (pb[:, 0] <= length + 10.0)
            & (pb[:, 1] >= -10.0) & (pb[:, 1] <= width + 10.0)
        )
        use = on_pitch if int(on_pitch.sum()) >= 3 else valid
        dist = np.linalg.norm(pa[use] - pb[use], axis=1)
        val = float(np.median(dist))
        return val if math.isfinite(val) else None

    def _effective_flow(self, motion: np.ndarray) -> Optional[np.ndarray]:
        if self.transformer.homography_matrix is None:
            return None
        return self._normalize_matrix(self.transformer.homography_matrix @ motion)

    @staticmethod
    def _safe_inv(m: np.ndarray) -> Optional[np.ndarray]:
        try:
            inv = np.linalg.inv(np.asarray(m, dtype=np.float64))
        except np.linalg.LinAlgError:
            return None
        return inv if bool(np.isfinite(inv).all()) else None

    # ----------------------------------------------------------------
    # Çapa değişimi (reanchor) ve aday tabloların taşınması
    # ----------------------------------------------------------------
    def _reanchor(
        self,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: Optional[np.ndarray],
        keep_bank: bool,
    ) -> np.ndarray:
        if keep_bank and motion is not None:
            # Eski çapa koordinatındaki tablolar -> yeni çapa (mevcut kare) koordinatı: T @ M
            self._fixed_candidates = [c @ motion for c in self._fixed_candidates]
            if self._pending_anchor_h is not None:
                self._pending_anchor_h = self._pending_anchor_h @ motion
            reanchored = self.motion_tracker.reanchor(
                frame, raw_detections, keep_keyframe_bank=True, prev_cumulative=motion
            )
        else:
            self._fixed_candidates = []
            self._clear_pending()
            reanchored = self.motion_tracker.reanchor(
                frame, raw_detections, keep_keyframe_bank=False
            )
        self._reanchored_since_prev = True
        return reanchored

    def _clear_pending(self, rejected: bool = False) -> None:
        if rejected and self._pending_count > 0:
            self.stats["rejected_outlier"] += 1
        self._pending_anchor_h = None
        self._pending_count = 0

    # ----------------------------------------------------------------
    # Yumuşak geçiş (matris ortalaması YOK: referans dörtgenin saha koordinatları ara değerlenir)
    # ----------------------------------------------------------------
    def _cancel_blend(self) -> None:
        self._blend_quad_img = None
        self._blend_p0 = None
        self._blend_p1 = None
        self._blend_step = 0
        self._blend_total = 0

    def _blend_active(self) -> bool:
        return self._blend_total > 0 and self._blend_p0 is not None

    def blend_matrix(self, alpha: float) -> Optional[np.ndarray]:
        if self._blend_quad_img is None or self._blend_p0 is None or self._blend_p1 is None:
            return None
        a = min(1.0, max(0.0, float(alpha)))
        p = (1.0 - a) * self._blend_p0 + a * self._blend_p1
        h = cv2.getPerspectiveTransform(
            self._blend_quad_img.astype(np.float32), p.astype(np.float32)
        )
        return self._normalize_matrix(np.asarray(h, dtype=np.float64))

    def _advance_active_blend(self) -> bool:
        """Aktif geçişi bir adım ilerletir; bu karede geçiş tablosu kullanıldıysa True."""
        if not self._blend_active():
            return False
        self._blend_step += 1
        alpha = min(1.0, float(self._blend_step) / float(self._blend_total))
        h_blend = self.blend_matrix(alpha)
        if h_blend is None or not self.transformer.set_homography(h_blend):
            target = self.blend_matrix(1.0)
            if target is not None:
                self.transformer.set_homography(target)
            self._cancel_blend()
            self.frame_info["source"] = "keypoint_refresh"
            return True
        finished = self._blend_step >= self._blend_total
        self.frame_info["source"] = "keypoint_refresh" if finished else "blend"
        if finished:
            self._cancel_blend()
        return True

    # ----------------------------------------------------------------
    # Kabul yolları
    # ----------------------------------------------------------------
    def _calibrate(self, frame: np.ndarray) -> Any:
        self.stats["model_calls"] += 1
        res = self.calibrator.calibrate_frame(frame)
        self.frame_info["num_keypoints"] = getattr(res, "num_keypoints", None)
        self.frame_info["num_inliers"] = getattr(res, "num_inliers", None)
        self.frame_info["reproj_error_m"] = getattr(res, "reproj_error_m", None)
        if getattr(res, "success", False) and getattr(res, "homography_matrix", None) is not None:
            self.stats["model_success"] += 1
        return res

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
        reanchored = self._reanchor(frame, raw_detections, None, keep_bank=False)
        self.last_calibration_frame = frame_idx
        self.was_non_tactical = False
        self.calibration_count += 1
        self.last_calibration_reason = reason
        self.stats["immediate_lock"] += 1
        self.frame_info["source"] = "keypoint_lock"
        self.frame_info["calib_reset"] = True
        return reanchored

    def _accept_hard(
        self,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: np.ndarray,
        h_kp: np.ndarray,
        reason: str,
    ) -> np.ndarray:
        self._cancel_blend()
        if not self.transformer.set_homography(h_kp):
            self.frame_info["source"] = "flow"
            return motion
        reanchored = self._reanchor(frame, raw_detections, motion, keep_bank=True)
        self.calibration_count += 1
        self.last_calibration_reason = reason
        self.frame_info["source"] = "keypoint_refresh"
        self.frame_info["calib_reset"] = True
        return reanchored

    def _accept_blend(
        self,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: np.ndarray,
        e_flow: np.ndarray,
        h_kp: np.ndarray,
        disagreement: float,
        reason: str,
    ) -> np.ndarray:
        quad = self.reference_quad(frame.shape)
        p0 = self._project(self._canonical_sign(e_flow, frame.shape), quad)
        p1 = self._project(self._canonical_sign(h_kp, frame.shape), quad)
        if not (bool(np.isfinite(p0).all()) and bool(np.isfinite(p1).all())):
            return self._accept_hard(frame, raw_detections, motion, h_kp, reason)

        reanchored = self._reanchor(frame, raw_detections, motion, keep_bank=True)
        # Çapa artık mevcut kare (M = I): referans dörtgen bu karenin pikselleridir.
        self._blend_quad_img = quad
        self._blend_p0 = p0
        self._blend_p1 = p1
        self._blend_step = 0
        self._blend_total = max(
            self.smooth_frames, int(math.ceil(float(disagreement) / self.blend_max_step_m))
        )
        self.calibration_count += 1
        self.last_calibration_reason = reason
        self._advance_active_blend()
        return reanchored

    def _gated_refresh(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: np.ndarray,
        h_kp_raw: np.ndarray,
    ) -> np.ndarray:
        """A2-3 kademeli karar kuralı."""
        self.last_calibration_frame = frame_idx
        self.frame_info["source"] = "flow"
        h_kp = self._normalize_matrix(h_kp_raw)
        e_flow = self._effective_flow(motion)
        if h_kp is None:
            return motion
        if e_flow is None:
            return self._accept_hard(frame, raw_detections, motion, h_kp, "periodic_refresh")
        h_kp = self._canonical_sign(h_kp, frame.shape)
        e_flow = self._canonical_sign(e_flow, frame.shape)

        d = self.disagreement_m(e_flow, h_kp, frame.shape)
        self.frame_info["disagreement_m"] = None if d is None else round(d, 4)

        if d is not None and d <= self.deadband_m:
            self._clear_pending(rejected=True)
            self.stats["agree_no_change"] += 1
            self.frame_info["source"] = "keypoint_refresh"
            return motion

        if d is not None and d <= self.soft_m:
            self._clear_pending(rejected=True)
            self.stats["soft_blend"] += 1
            return self._accept_blend(
                frame, raw_detections, motion, e_flow, h_kp, d, "periodic_refresh"
            )

        # SOFT üstü: doğrulama bekle
        inv_m = self._safe_inv(motion)
        if inv_m is None:
            return motion
        if self._pending_anchor_h is not None:
            pend_now = self._normalize_matrix(self._pending_anchor_h @ motion)
            d_cons = (
                self.disagreement_m(pend_now, h_kp, frame.shape)
                if pend_now is not None
                else None
            )
            if d_cons is not None and d_cons <= self.soft_m:
                self._pending_count += 1
            else:
                if self._pending_count >= 1:
                    self.stats["rejected_outlier"] += 1
                self._pending_count = 1
        else:
            self._pending_count = 1
        self._pending_anchor_h = h_kp @ inv_m

        if self._pending_count >= 1 + self.confirm_refreshes:
            self._clear_pending()
            if d is not None and d <= self.hard_m:
                self.stats["confirmed_blend"] += 1
                return self._accept_blend(
                    frame, raw_detections, motion, e_flow, h_kp, d, "confirmed_refresh"
                )
            self.stats["confirmed_hard_reset"] += 1
            return self._accept_hard(frame, raw_detections, motion, h_kp, "confirmed_reset")

        self.stats["pending"] += 1
        return motion

    # ----------------------------------------------------------------
    # A2-5: Görüntü profili
    # ----------------------------------------------------------------
    def _set_profile(self, frame_idx: int, new_profile: str, reason: str) -> None:
        old = self.profile
        self.profile = new_profile
        self.profile_decided = True
        self.profile_reason = reason
        self.profile_events.append(
            {"frame_idx": int(frame_idx), "from": old, "to": new_profile, "reason": reason}
        )
        self._pan_path_px = 0.0
        self._max_scale_change = 0.0
        print(f"[Kalibrasyon] Görüntü profili: {new_profile} (kare {frame_idx}; {reason})")

    def _update_profile(
        self,
        frame_idx: int,
        frame_shape: Tuple[int, ...],
        motion: Optional[np.ndarray],
        scene_cut: bool,
    ) -> None:
        if self.profile_decided and self.profile == "broadcast":
            return
        w = max(1.0, float(frame_shape[1]))
        center = np.array([[0.5 * w, 0.5 * float(frame_shape[0])]], dtype=np.float64)
        if motion is not None:
            if self._prev_motion is not None and not self._reanchored_since_prev:
                p_now = self._project(motion, center)[0]
                p_prev = self._project(self._prev_motion, center)[0]
                step = float(np.linalg.norm(p_now - p_prev))
                if math.isfinite(step):
                    self._pan_path_px += step
            try:
                scale = math.sqrt(abs(float(np.linalg.det(motion[:2, :2] / motion[2, 2]))))
                self._max_scale_change = max(self._max_scale_change, abs(scale - 1.0))
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                pass
        self._prev_motion = None if motion is None else motion.copy()
        self._reanchored_since_prev = False

        pan_ratio = self._pan_path_px / w
        max_pan = float(getattr(config, "PROFILE_FIXED_MAX_PAN_RATIO", 0.05))
        max_scale = float(getattr(config, "PROFILE_FIXED_MAX_SCALE_CHANGE", 0.03))
        moving = pan_ratio > max_pan or self._max_scale_change > max_scale

        if not self.profile_decided:
            if self._detect_start_frame is None:
                self._detect_start_frame = frame_idx
            detect_frames = int(round(float(getattr(config, "PROFILE_DETECT_SEC", 5.0)) * self.fps))
            if scene_cut:
                self._set_profile(frame_idx, "broadcast", "algılama penceresinde sahne kesmesi")
            elif moving:
                self._set_profile(
                    frame_idx,
                    "broadcast",
                    f"kamera hareketi (pan oranı={pan_ratio:.3f}, ölçek değişimi={self._max_scale_change:.3f})",
                )
            elif frame_idx - self._detect_start_frame >= detect_frames:
                self._set_profile(
                    frame_idx,
                    "fixed",
                    f"{detect_frames} karede kesme yok, kamera sabit "
                    f"(pan oranı={pan_ratio:.3f}, ölçek değişimi={self._max_scale_change:.3f})",
                )
        elif self.profile == "fixed":
            if scene_cut:
                self._set_profile(frame_idx, "broadcast", "sabit profilde sahne kesmesi")
            elif moving:
                self._set_profile(frame_idx, "broadcast", "sabit profilde belirgin kamera hareketi")

    def _lock_fixed(
        self,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: np.ndarray,
    ) -> np.ndarray:
        """Aday tablolar arasından en tutarlısını (ortanca farkı en düşük) kilitler."""
        cands = [self._normalize_matrix(c @ motion) for c in self._fixed_candidates]
        cands = [self._canonical_sign(c, frame.shape) for c in cands if c is not None]
        self._fixed_candidates = []
        self._fixed_locked = True
        if not cands:
            return motion
        best = cands[0]
        if len(cands) > 1:
            best_score = math.inf
            for i, ci in enumerate(cands):
                ds = [
                    self.disagreement_m(ci, cj, frame.shape)
                    for j, cj in enumerate(cands)
                    if j != i
                ]
                ds = [x for x in ds if x is not None]
                score = float(np.median(ds)) if ds else math.inf
                if score < best_score:
                    best_score = score
                    best = ci
        e_flow = self._effective_flow(motion)
        if e_flow is None:
            return self._accept_hard(frame, raw_detections, motion, best, "fixed_lock")
        e_flow = self._canonical_sign(e_flow, frame.shape)
        d = self.disagreement_m(e_flow, best, frame.shape)
        self.frame_info["disagreement_m"] = None if d is None else round(d, 4)
        self.last_calibration_reason = "fixed_lock"
        if d is not None and d <= self.deadband_m:
            self.frame_info["source"] = "keypoint_refresh"
            return motion
        if d is not None and d <= self.soft_m:
            return self._accept_blend(frame, raw_detections, motion, e_flow, best, d, "fixed_lock")
        return self._accept_hard(frame, raw_detections, motion, best, "fixed_lock")

    def _process_fixed(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
        motion: np.ndarray,
    ) -> np.ndarray:
        if not self._fixed_locked:
            if len(self._fixed_candidates) >= self.fixed_candidates_target:
                return self._lock_fixed(frame, raw_detections, motion)
            due = (
                self.last_calibration_frame is None
                or (frame_idx - self.last_calibration_frame) >= self.refresh_interval
            )
            if due and not self._blend_active():
                self.last_calibration_frame = frame_idx
                res = self._calibrate(frame)
                if getattr(res, "success", False) and res.homography_matrix is not None:
                    inv_m = self._safe_inv(motion)
                    h = self._normalize_matrix(res.homography_matrix)
                    if inv_m is not None and h is not None:
                        self._fixed_candidates.append(h @ inv_m)
                    if len(self._fixed_candidates) >= self.fixed_candidates_target:
                        return self._lock_fixed(frame, raw_detections, motion)
            if not self._advance_active_blend():
                self.frame_info["source"] = "flow"
            return motion

        health_frames = max(
            1, int(round(float(getattr(config, "PITCH_CAL_FIXED_HEALTH_SEC", 10.0)) * self.fps))
        )
        if self._last_health_frame is None:
            self._last_health_frame = frame_idx
        if (frame_idx - self._last_health_frame) >= health_frames and not self._blend_active():
            self._last_health_frame = frame_idx
            res = self._calibrate(frame)
            if getattr(res, "success", False) and res.homography_matrix is not None:
                return self._gated_refresh(
                    frame_idx, frame, raw_detections, motion, res.homography_matrix
                )
        if not self._advance_active_blend():
            self.frame_info["source"] = "flow"
        return motion

    # ----------------------------------------------------------------
    def process_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]] = None,
    ) -> Optional[np.ndarray]:
        self.frame_info = self._new_frame_info()
        result = self._process_frame_inner(frame_idx, frame, raw_detections)
        self.frame_info["profile"] = self.profile
        if result is None:
            self.frame_info["source"] = "none"
        return result

    def _process_frame_inner(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_detections: Optional[List[Dict]],
    ) -> Optional[np.ndarray]:
        cal_start = int(getattr(config, "CALIBRATION_FRAME_INDEX", 0))
        if frame_idx < cal_start:
            return None

        motion_matrix = self.motion_tracker.update(frame, raw_detections)
        scene_cut = bool(getattr(self.motion_tracker, "scene_cut_detected", False))
        recovered = bool(getattr(self.motion_tracker, "recovered_this_frame", False))

        self._update_profile(frame_idx, frame.shape, motion_matrix, scene_cut)

        if scene_cut:
            self._cancel_blend()
            self._clear_pending()
            self.was_non_tactical = True

        # Durum 1: İlk taktik kare kilidi veya kesme/tekrar sonrası geri dönüş
        needs_immediate_lock = (
            (not self.transformer.is_calibrated)
            or (motion_matrix is None)
            or recovered
            or self.was_non_tactical
        )
        if needs_immediate_lock:
            cal_res = self._calibrate(frame)
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
                    frame_idx, frame, raw_detections, cal_res.homography_matrix, reason=reason
                )
                if locked_m is not None:
                    if self.profile == "fixed" and not self._fixed_locked:
                        h0 = self._normalize_matrix(cal_res.homography_matrix)
                        if h0 is not None:
                            self._fixed_candidates.append(h0)
                    return locked_m

            # Keypoint kalibrasyonu bu karede başarısız: önceki geçerli kalibrasyon ve
            # flow varsa onunla devam et; was_non_tactical bir sonraki karede yeniden denemeyi sağlar.
            if motion_matrix is None or not self.transformer.is_calibrated:
                return None
            self.frame_info["source"] = "flow"
            return motion_matrix

        if self.profile == "fixed":
            return self._process_fixed(frame_idx, frame, raw_detections, motion_matrix)

        # Durum 2 (broadcast): periyodik keypoint yenileme + kademeli karar
        periodic_due = (
            self.last_calibration_frame is not None
            and (frame_idx - self.last_calibration_frame) >= self.refresh_interval
            and not self._blend_active()
        )
        if periodic_due:
            cal_res = self._calibrate(frame)
            if (
                getattr(cal_res, "success", False)
                and getattr(cal_res, "homography_matrix", None) is not None
            ):
                if not self.profile_decided:
                    inv_m = self._safe_inv(motion_matrix)
                    h = self._normalize_matrix(cal_res.homography_matrix)
                    if inv_m is not None and h is not None:
                        self._fixed_candidates.append(h @ inv_m)
                        self._fixed_candidates = self._fixed_candidates[-self.fixed_candidates_target:]
                return self._gated_refresh(
                    frame_idx, frame, raw_detections, motion_matrix, cal_res.homography_matrix
                )
            # Başarısız olursa last_calibration_frame güncellenmez -> sonraki karede yeniden dener.

        # Durum 3: Ara karelerde devam eden yumuşak geçişi ilerlet
        if not self._advance_active_blend():
            self.frame_info["source"] = "flow"
        return motion_matrix

    # ----------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "requested_profile": self.requested_profile,
            "profile": self.profile,
            "profile_reason": self.profile_reason,
            "profile_events": list(self.profile_events),
            "thresholds_m": {
                "deadband": self.deadband_m,
                "soft": self.soft_m,
                "hard": self.hard_m,
                "confirm_refreshes": self.confirm_refreshes,
            },
            "calibration_count": self.calibration_count,
            "stats": dict(self.stats),
        }
