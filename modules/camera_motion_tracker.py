"""
camera_motion_tracker.py
--------------------------
KAMERA HAREKETİ TELAFİSİ (Adım 5'in eklentisi)

SORUN: perspective_transformer.py, kalibrasyonu SADECE İLK KAREDE yapıp
tüm video boyunca AYNI homografi matrisini kullanıyordu. Ama yayın kamerası
maç sırasında pan/tilt/zoom yapar -- bu durumda ilk karede doğru olan
matris, 10. saniyede tamamen yanlış sonuç verir.

ÇÖZÜM STRATEJİSİ: OPTICAL FLOW İLE KAMERA HAREKETİ TAHMİNİ
    Sahadaki SABİT noktaları (çizgiler, reklam panosu kenarları -- oyuncu
    veya top DEĞİL) kare kare takip ediyoruz. Bu noktaların pikseldeki
    kayması bize "kamera bu iki kare arasında ne kadar hareket etti" bilgisini
    veriyor. Bu kare-kare hareketleri zincirleme çarparak, "şu anki karenin
    pikselleri, kalibrasyonun yapıldığı İLK KARENİN piksel sistemine göre
    nereye denk gelir" sorusuna cevap buluyoruz. Bu birikimli (cumulative)
    dönüşüm matrisini, statik homografi matrisiyle birleştirerek her kare
    için "efektif" bir homografi elde ediyoruz.

MATEMATİK (özet):
    D_t : kare (t) piksellerini kare (t-1) piksellerine eşleyen homografi
          (optical flow ile eşleşen nokta çiftlerinden hesaplanır)
    M_t : kare (t) piksellerini kare (0) [kalibrasyon karesi] piksellerine
          eşleyen BİRİKİMLİ dönüşüm = M_(t-1) @ D_t
    H_0 : kare (0) piksellerini saha metrelerine eşleyen STATİK homografi
          (perspective_transformer.py'de zaten hesaplanan matris)
    H_efektif(t) = H_0 @ M_t   -> kare (t) piksellerini DOĞRUDAN saha
                                   metrelerine eşler

BİLİNEN SINIRLAMA (drift):
    Her kare tahmini küçük bir hata taşır ve bu hatalar zincirleme çarpımla
    ZAMANLA BİRİKİR (drift). Kısa-orta süreli klipler (birkaç dakika) için
    kabul edilebilir düzeydedir, ancak çok uzun kesintisiz çekimlerde
    (10+ dakika) periyodik olarak yeniden kalibrasyon (drift correction)
    yapılması önerilir -- bu, ileri seviye bir geliştirme olarak roadmap'te
    bırakılmıştır.

SAHNE KESMESİ (SCENE CUT):
    Yayın başka bir kameraya (tekrar izleme, close-up) aniden geçtiğinde,
    optical flow'un takip ettiği noktalar ANLAMSIZ hale gelir (tamamen farklı
    bir görüntü). Bunu histogram karşılaştırmasıyla tespit edip, o andan
    itibaren "artık güvenilir değil" diyoruz -- yanlış veri üretmektense hiç
    üretmemeyi tercih ediyoruz.
"""

from collections import deque
import math
import os
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class CameraMotionTracker:
    """
    Optical flow ile kare-kare kamera hareketini tahmin edip, kalibrasyon
    karesine göre BİRİKİMLİ bir dönüşüm matrisi üreten ve TV yayınındaki
    sahne kesmeleri / tekrar görüntüleri sonrasında ana kameraya dönüşte
    otomatik olarak yeniden hizalanan (recovery) sınıf.

    Kullanım örneği:
        motion_tracker = CameraMotionTracker()
        for frame_idx, frame in vp.read_frames():
            ...
            motion_matrix = motion_tracker.update(frame, raw_detections)
            if motion_matrix is not None:
                all_tracked = transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=motion_matrix)
            else:
                # sahne kesmesi ya da ana kamera dışı çekim -- bu kare için saha koordinatı üretme
                ...
    """

    def __init__(self):
        # Lucas-Kanade optical flow parametreleri
        self.lk_params = dict(
            winSize=config.OPTICAL_FLOW_WIN_SIZE,
            maxLevel=config.OPTICAL_FLOW_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.OPTICAL_FLOW_CRITERIA_COUNT,
                config.OPTICAL_FLOW_CRITERIA_EPS,
            ),
        )

        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._orb_cache_nfeatures: Optional[int] = None
        self._orb: Optional[cv2.ORB] = None

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_gray_flow: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self._prev_flow_scale: Tuple[float, float] = (1.0, 1.0)

        # M_t: şu anki karenin piksellerini, kalibrasyon karesinin (kare 0)
        # piksel sistemine eşleyen birikimli matris (float64).
        self.cumulative_matrix: np.ndarray = np.eye(3, dtype=np.float64)

        self.frame_count: int = 0
        self.tracking_lost: bool = False        # geçici takip kaybı veya ana kamera dışı durum
        self.scene_cut_detected: bool = False   # yalnızca kesmenin tespit edildiği karede True
        self.is_anchored: bool = False          # kare 0 koordinat sistemine bağlı mı?
        self.recovered_this_frame: bool = False # yalnızca yeniden hizalanmanın gerçekleştiği karede True
        self.shot_state: str = "UNINITIALIZED"  # "UNINITIALIZED" | "ANCHORED" | "NON_TACTICAL"
        self.consecutive_lost_frames: int = 0

        self._healthy_anchored_frames: int = 0
        self._fade_degraded_count: int = 0
        self._recovery_streak: int = 0
        self._anchor_green_ratio: float = 0.0

        self._prev_thumb_gray: Optional[np.ndarray] = None
        self._prev_thumb_edges: Optional[np.ndarray] = None
        self._prev_hist_gray: Optional[np.ndarray] = None
        self._prev_hist_hsv: Optional[np.ndarray] = None
        self._baseline_hist_gray: Optional[np.ndarray] = None
        self._baseline_hist_hsv: Optional[np.ndarray] = None
        self._hist_window: deque = deque(maxlen=16)

        self._keyframe_bank: List[Dict] = []
        self._last_kf_frame: int = -1000
        self._last_reanchor_frame: int = 0

        print("[CameraMotionTracker] Oluşturuldu. İlk karede referans noktalar toplanacak.")

    # ----------------------------------------------------------------
    def _get_orb(self) -> cv2.ORB:
        nfeatures = int(
            getattr(
                config,
                "CAMERA_ORB_MAX_FEATURES",
                getattr(config, "CAMERA_KEYFRAME_ORB_FEATURES", 600),
            )
        )
        if self._orb is None or self._orb_cache_nfeatures != nfeatures:
            self._orb = cv2.ORB_create(
                nfeatures=nfeatures, fastThreshold=15, edgeThreshold=15
            )
            self._orb_cache_nfeatures = nfeatures
        return self._orb

    # ----------------------------------------------------------------
    def _build_background_mask(
        self, frame_shape: Tuple[int, ...], detections: Optional[List[Dict]]
    ) -> np.ndarray:
        """
        Oyuncu/top bounding box'larını, ekran kenarlarını ve sabit TV skor/logo
        bölgelerini maskeleyerek (0 yaparak) sabit arka plan üzerinde nokta arar.
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])
        mask = np.full((h, w), 255, dtype=np.uint8)
        pad = int(config.PLAYER_MASK_PADDING_PX)

        if detections:
            for det in detections:
                bbox = det.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                except (TypeError, ValueError):
                    continue
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        bbox_only_mask = mask.copy()

        # Kenar sınır maskesi
        border_ratio = float(getattr(config, "CAMERA_BORDER_MASK_RATIO", 0.02))
        if border_ratio > 0.0:
            bx = int(w * border_ratio)
            by = int(h * border_ratio)
            if by > 0:
                mask[:by, :] = 0
                mask[max(0, h - by) :, :] = 0
            if bx > 0:
                mask[:, :bx] = 0
                mask[:, max(0, w - bx) :] = 0

        # Sabit TV skorbord / logo bölgeleri (küçük sentetik test karelerini koru)
        min_mask_h = int(getattr(config, "CAMERA_SCOREBUG_MASK_MIN_HEIGHT", 360))
        if h >= min_mask_h:
            zones = getattr(
                config,
                "CAMERA_SCOREBUG_MASK_ZONES",
                (
                    (0.0, 0.0, 0.32, 0.14),
                    (0.68, 0.0, 1.0, 0.14),
                    (0.15, 0.88, 0.85, 1.0),
                ),
            )
            for zone in zones:
                nx1, ny1, nx2, ny2 = zone
                zx1 = max(0, min(w, int(round(nx1 * w))))
                zy1 = max(0, min(h, int(round(ny1 * h))))
                zx2 = max(0, min(w, int(round(nx2 * w))))
                zy2 = max(0, min(h, int(round(ny2 * h))))
                if zx2 > zx1 and zy2 > zy1:
                    mask[zy1:zy2, zx1:zx2] = 0

            extra_rects = getattr(config, "CAMERA_STATIC_MASK_RECTS", None)
            if extra_rects:
                for zone in extra_rects:
                    nx1, ny1, nx2, ny2 = zone
                    zx1 = max(0, min(w, int(round(nx1 * w))))
                    zy1 = max(0, min(h, int(round(ny1 * h))))
                    zx2 = max(0, min(w, int(round(nx2 * w))))
                    zy2 = max(0, min(h, int(round(ny2 * h))))
                    if zx2 > zx1 and zy2 > zy1:
                        mask[zy1:zy2, zx1:zx2] = 0

        if cv2.countNonZero(mask) < int(0.15 * h * w):
            return bbox_only_mask

        return mask

    # ----------------------------------------------------------------
    def _compute_frame_signals(self, frame: np.ndarray) -> Dict[str, object]:
        """Küçültülmüş thumbnail üzerinden histogram, MAD, kenar yoğunluğu ve çim oranı üretir."""
        h, w = frame.shape[:2]
        thumb_w = int(getattr(config, "CAMERA_ANALYSIS_THUMB_WIDTH", 160))
        scale_t = min(1.0, float(thumb_w) / float(max(w, 1)))
        tw = max(16, int(round(w * scale_t)))
        th = max(12, int(round(h * scale_t)))

        if (tw, th) != (w, h):
            thumb_bgr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            thumb_bgr = frame

        thumb_gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
        thumb_hsv = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2HSV)

        hist_gray = cv2.calcHist([thumb_gray], [0], None, [64], [0, 256]).astype(
            np.float32
        )
        cv2.normalize(hist_gray, hist_gray)

        hist_hsv = cv2.calcHist(
            [thumb_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
        ).astype(np.float32)
        cv2.normalize(hist_hsv, hist_hsv)

        gx = cv2.Sobel(thumb_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(thumb_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(gx, gy)
        edge_density = float(np.mean(edge_mag > 25.0))

        y0 = int(0.10 * th)
        y1 = max(y0 + 1, int(0.92 * th))
        roi = thumb_hsv[y0:y1, :]
        green_mask = (
            (roi[:, :, 0] >= 28)
            & (roi[:, :, 0] <= 95)
            & (roi[:, :, 1] >= 30)
            & (roi[:, :, 2] >= 30)
        )
        green_ratio = float(np.mean(green_mask))

        return {
            "thumb_gray": thumb_gray,
            "thumb_edges": edge_mag,
            "hist_gray": hist_gray,
            "hist_hsv": hist_hsv,
            "edge_density": edge_density,
            "green_ratio": green_ratio,
        }

    # ----------------------------------------------------------------
    def _store_prev_signals(
        self,
        gray: np.ndarray,
        gray_flow: np.ndarray,
        flow_scale: Tuple[float, float],
        signals: Dict[str, object],
        reset_baseline: bool = False,
    ) -> None:
        self.prev_gray = gray
        self.prev_gray_flow = gray_flow
        self._prev_flow_scale = flow_scale

        self._prev_thumb_gray = signals["thumb_gray"]  # type: ignore[assignment]
        self._prev_thumb_edges = signals["thumb_edges"]  # type: ignore[assignment]
        hist_gray = signals["hist_gray"]  # type: ignore[assignment]
        hist_hsv = signals["hist_hsv"]  # type: ignore[assignment]
        self._prev_hist_gray = hist_gray
        self._prev_hist_hsv = hist_hsv

        win_len = max(2, int(getattr(config, "SCENE_CUT_FADE_WINDOW", 8)))
        if self._hist_window.maxlen != win_len:
            self._hist_window = deque(list(self._hist_window), maxlen=win_len)
        self._hist_window.append((hist_gray.copy(), hist_hsv.copy()))

        if reset_baseline or self._baseline_hist_gray is None:
            self._baseline_hist_gray = hist_gray.copy()
            self._baseline_hist_hsv = hist_hsv.copy()
            self._hist_window.clear()
            self._hist_window.append((hist_gray.copy(), hist_hsv.copy()))

    # ----------------------------------------------------------------
    def _update_baseline_ema(
        self, signals: Dict[str, object], corr_min: float, corr_baseline_min: float, inlier_ratio: float
    ) -> None:
        if self._baseline_hist_gray is None or self._baseline_hist_hsv is None:
            return
        if inlier_ratio >= 0.55 and corr_min >= 0.94 and corr_baseline_min >= 0.75:
            alpha = float(getattr(config, "SCENE_CUT_EMA_ALPHA", 0.04))
            hist_gray = signals["hist_gray"]  # type: ignore[assignment]
            hist_hsv = signals["hist_hsv"]  # type: ignore[assignment]
            self._baseline_hist_gray = (
                (1.0 - alpha) * self._baseline_hist_gray + alpha * hist_gray
            ).astype(np.float32)
            cv2.normalize(self._baseline_hist_gray, self._baseline_hist_gray)
            self._baseline_hist_hsv = (
                (1.0 - alpha) * self._baseline_hist_hsv + alpha * hist_hsv
            ).astype(np.float32)
            cv2.normalize(self._baseline_hist_hsv, self._baseline_hist_hsv)

    # ----------------------------------------------------------------
    def _get_flow_scale_and_gray(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        h, w = gray.shape[:2]
        cfg_scale = getattr(config, "CAMERA_MOTION_SCALE", None)
        min_width = int(getattr(config, "CAMERA_MOTION_SCALE_MIN_WIDTH", 640))

        if cfg_scale is None:
            scale = 0.5 if w >= min_width else 1.0
        else:
            scale_val = float(cfg_scale)
            scale = scale_val if (0.0 < scale_val < 1.0 and w >= min_width) else 1.0

        if scale >= 1.0:
            return gray, 1.0, 1.0

        ws = max(64, int(round(w * scale)))
        hs = max(48, int(round(h * scale)))
        gray_flow = cv2.resize(gray, (ws, hs), interpolation=cv2.INTER_AREA)
        return gray_flow, float(ws) / float(w), float(hs) / float(h)

    # ----------------------------------------------------------------
    def _detect_scene_cut(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        """
        Geriye dönük uyumluluk için korunan histogram tabanlı kesme kontrolü.
        """
        hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])

        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)

        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return correlation < config.SCENE_CUT_CORRELATION_THRESHOLD

    # ----------------------------------------------------------------
    def _detect_features(
        self, gray_flow: np.ndarray, mask_flow: np.ndarray, scale_avg: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Kareyi mekansal ızgaraya bölerek arka plan üzerinde homojen dağılmış
        köşe noktaları tespit eder.
        """
        h, w = gray_flow.shape[:2]
        max_corners = int(config.CAMERA_MOTION_MAX_CORNERS)
        min_dist = max(4.0, float(config.CAMERA_MOTION_MIN_DISTANCE) * scale_avg)
        grid_rows = int(getattr(config, "CAMERA_MOTION_GRID_ROWS", 4))
        grid_cols = int(getattr(config, "CAMERA_MOTION_GRID_COLS", 6))

        if grid_rows <= 1 or grid_cols <= 1 or w < 120 or h < 90:
            return cv2.goodFeaturesToTrack(
                gray_flow,
                maxCorners=max_corners,
                qualityLevel=config.CAMERA_MOTION_QUALITY_LEVEL,
                minDistance=min_dist,
                mask=mask_flow,
            )

        per_cell = max(8, math.ceil(max_corners / float(grid_rows * grid_cols)))
        collected: List[np.ndarray] = []

        for r in range(grid_rows):
            y0 = (r * h) // grid_rows
            y1 = ((r + 1) * h) // grid_rows
            if y1 <= y0:
                continue
            for c in range(grid_cols):
                x0 = (c * w) // grid_cols
                x1 = ((c + 1) * w) // grid_cols
                if x1 <= x0:
                    continue
                cell_mask = mask_flow[y0:y1, x0:x1]
                if cv2.countNonZero(cell_mask) == 0:
                    continue
                pts = cv2.goodFeaturesToTrack(
                    gray_flow[y0:y1, x0:x1],
                    maxCorners=per_cell,
                    qualityLevel=config.CAMERA_MOTION_QUALITY_LEVEL,
                    minDistance=min_dist,
                    mask=cell_mask,
                )
                if pts is not None and len(pts) > 0:
                    pts = pts.copy()
                    pts[:, 0, 0] += float(x0)
                    pts[:, 0, 1] += float(y0)
                    collected.append(pts)

        if collected:
            merged = np.vstack(collected)
            if len(merged) >= int(config.CAMERA_MOTION_MIN_TRACKED_POINTS):
                return merged

        return cv2.goodFeaturesToTrack(
            gray_flow,
            maxCorners=max_corners,
            qualityLevel=config.CAMERA_MOTION_QUALITY_LEVEL,
            minDistance=min_dist,
            mask=mask_flow,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _validate_homography(
        matrix: Optional[np.ndarray],
        frame_shape: Tuple[int, ...],
        is_step: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Homografi matrisini float64 hassasiyetinde doğrular ve M[2,2]=1 olarak normalize eder.
        [-1,1] normalize koordinat koşul sayısı, determinant, tekil değer oranı,
        ufuk çizgisi (w > 0.15) ve dışbükey dörtgen yönelimini kontrol eder.
        """
        if matrix is None:
            return None

        try:
            m = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if m.ndim != 2 or m.shape != (3, 3) or not bool(np.isfinite(m).all()):
            return None

        if abs(float(m[2, 2])) <= 1e-9:
            return None

        m = m / float(m[2, 2])
        if not bool(np.isfinite(m).all()):
            return None

        h, w = int(frame_shape[0]), int(frame_shape[1])
        wf = float(max(w, 1))
        hf = float(max(h, 1))

        try:
            det = float(np.linalg.det(m))
        except np.linalg.LinAlgError:
            return None

        if is_step:
            min_det = float(getattr(config, "CAMERA_STEP_MIN_DET", 0.25))
            max_det = float(getattr(config, "CAMERA_STEP_MAX_DET", 4.0))
            max_cond = float(getattr(config, "CAMERA_STEP_MAX_COND", 500.0))
        else:
            min_det = float(getattr(config, "CAMERA_HOMOGRAPHY_MIN_DET", 0.05))
            max_det = float(getattr(config, "CAMERA_HOMOGRAPHY_MAX_DET", 20.0))
            max_cond = float(getattr(config, "CAMERA_HOMOGRAPHY_MAX_COND", 1000.0))

        if not math.isfinite(det) or det < min_det or det > max_det:
            return None

        # Piksel ötelemelerinden bağımsız [-1, 1] normalize koordinat koşul sayısı
        norm_mat = np.diag([2.0 / wf, 2.0 / hf, 1.0])
        norm_inv = np.diag([wf / 2.0, hf / 2.0, 1.0])
        try:
            m_norm = norm_mat @ m @ norm_inv
            cond = float(np.linalg.cond(m_norm))
            sv = np.linalg.svd(m[:2, :2], compute_uv=False)
        except np.linalg.LinAlgError:
            return None

        if not math.isfinite(cond) or cond > max_cond:
            return None

        sv_max = float(sv[0])
        sv_min = float(sv[-1])
        if is_step:
            if sv_min <= 0.40 or sv_max >= 2.50 or (sv_max / max(sv_min, 1e-12)) > 2.5:
                return None
        else:
            if sv_min <= 0.15 or sv_max >= 6.0 or (sv_max / max(sv_min, 1e-12)) > 5.0:
                return None

        # Ufuk / w > 0.15 kontrolü (4 köşe + merkez)
        test_xy = np.array(
            [
                [0.0, 0.0],
                [wf, 0.0],
                [wf, hf],
                [0.0, hf],
                [0.5 * wf, 0.5 * hf],
            ],
            dtype=np.float64,
        )
        w_vals = m[2, 0] * test_xy[:, 0] + m[2, 1] * test_xy[:, 1] + m[2, 2]
        if not bool(np.all(w_vals > 0.15)):
            return None

        # 4 köşe dışbükey ve yönelim koruyucu dörtgen oluşturmalı
        corners = test_xy[:4]
        proj_x = (m[0, 0] * corners[:, 0] + m[0, 1] * corners[:, 1] + m[0, 2]) / w_vals[:4]
        proj_y = (m[1, 0] * corners[:, 0] + m[1, 1] * corners[:, 1] + m[1, 2]) / w_vals[:4]
        if not (bool(np.isfinite(proj_x).all()) and bool(np.isfinite(proj_y).all())):
            return None

        pts = np.stack([proj_x, proj_y], axis=1)
        for i in range(4):
            e1 = pts[(i + 1) % 4] - pts[i]
            e2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
            cross = float(e1[0] * e2[1] - e1[1] * e2[0])
            if cross <= 0.0:
                return None

        return m

    # ----------------------------------------------------------------
    def _extract_orb_full(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = gray.shape[:2]
        orb_max_w = int(getattr(config, "CAMERA_ORB_MATCH_WIDTH", 640))
        if w > orb_max_w:
            scale = float(orb_max_w) / float(w)
            ow = orb_max_w
            oh = max(48, int(round(h * scale)))
            gray_s = cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            inv_sx = float(w) / float(ow)
            inv_sy = float(h) / float(oh)
        else:
            gray_s = gray
            mask_s = mask
            inv_sx = 1.0
            inv_sy = 1.0

        orb = self._get_orb()
        kps, des = orb.detectAndCompute(gray_s, mask_s)
        if not kps or des is None or len(kps) == 0:
            return None, None

        pts_full = np.array(
            [[kp.pt[0] * inv_sx, kp.pt[1] * inv_sy] for kp in kps],
            dtype=np.float32,
        )
        return pts_full, des

    # ----------------------------------------------------------------
    @staticmethod
    def _project_center_ref(matrix: np.ndarray, frame_shape: Tuple[int, ...]) -> np.ndarray:
        h, w = frame_shape[:2]
        cx, cy = 0.5 * float(w), 0.5 * float(h)
        denom = float(matrix[2, 0] * cx + matrix[2, 1] * cy + matrix[2, 2])
        if abs(denom) <= 1e-9:
            return np.array([cx, cy], dtype=np.float64)
        rx = float(matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2]) / denom
        ry = float(matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2]) / denom
        return np.array([rx, ry], dtype=np.float64)

    # ----------------------------------------------------------------
    def _maybe_insert_keyframe(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        frame_shape: Tuple[int, ...],
        pinned: bool = False,
    ) -> None:
        max_size = int(
            getattr(
                config,
                "CAMERA_KEYFRAME_BANK_SIZE",
                getattr(config, "CAMERA_KEYFRAME_BANK_MAX_SIZE", 16),
            )
        )
        kf_interval = int(getattr(config, "CAMERA_KEYFRAME_INTERVAL", 75))
        min_shift_ratio = float(getattr(config, "CAMERA_KEYFRAME_MIN_SHIFT_RATIO", 0.15))
        h, w = frame_shape[:2]

        view_center = self._project_center_ref(self.cumulative_matrix, frame_shape)

        if not pinned and len(self._keyframe_bank) > 0:
            min_dist = min(
                float(np.linalg.norm(view_center - kf["view_center_ref"]))
                for kf in self._keyframe_bank
            )
            spatial_new = min_dist >= (min_shift_ratio * float(w))
            time_elapsed = (self.frame_count - self._last_kf_frame) >= kf_interval
            if not spatial_new and not (time_elapsed and len(self._keyframe_bank) < max_size):
                return

        pts_full, des = self._extract_orb_full(gray, mask)
        if pts_full is None or des is None or len(pts_full) < 12:
            return

        entry = {
            "frame_idx": self.frame_count,
            "M_k": self.cumulative_matrix.copy(),
            "pts_full": pts_full,
            "descriptors": des,
            "view_center_ref": view_center,
            "pinned": bool(pinned),
        }

        if len(self._keyframe_bank) >= max_size:
            # Sabit (pinned) olmayan ve diğer anahtar karelere en yakın (en az mekansal
            # çeşitlilik sağlayan) anahtar kareyi çıkar; böylece erken/yaygın kapsama korunur.
            evict_idx = None
            smallest_nn = float("inf")
            for idx, kf in enumerate(self._keyframe_bank):
                if kf["pinned"]:
                    continue
                nn_dist = min(
                    (
                        float(np.linalg.norm(kf["view_center_ref"] - other["view_center_ref"]))
                        for j, other in enumerate(self._keyframe_bank)
                        if j != idx
                    ),
                    default=0.0,
                )
                if nn_dist < smallest_nn:
                    smallest_nn = nn_dist
                    evict_idx = idx
            if evict_idx is not None:
                self._keyframe_bank.pop(evict_idx)
            else:
                return

        self._keyframe_bank.append(entry)
        self._last_kf_frame = self.frame_count

    # ----------------------------------------------------------------
    def _match_against_keyframe_bank(
        self,
        pts_curr_full: Optional[np.ndarray],
        des_curr: Optional[np.ndarray],
        frame_shape: Tuple[int, ...],
        min_inliers: int,
        min_inlier_ratio: float,
        earliest_first: bool = False,
    ) -> Optional[Tuple[np.ndarray, int, float]]:
        if (
            pts_curr_full is None
            or des_curr is None
            or len(pts_curr_full) < min_inliers
            or not self._keyframe_bank
        ):
            return None

        h, w = frame_shape[:2]
        lowe_ratio = float(getattr(config, "CAMERA_ORB_LOWE_RATIO", 0.75))
        ransac_thresh = float(getattr(config, "CAMERA_RECOVERY_RANSAC_THRESHOLD", 5.0))
        max_reproj_err = float(
            getattr(config, "CAMERA_RECOVERY_MAX_REPROJ_ERR_PX", 6.0)
        )

        # Kare 0 her zaman ilk sırada; geri kalanlar erken-önce veya yeni-önce sıralanır
        pinned_kfs = [kf for kf in self._keyframe_bank if kf["pinned"]]
        other_kfs = [kf for kf in self._keyframe_bank if not kf["pinned"]]
        if not earliest_first:
            other_kfs = list(reversed(other_kfs))
        ordered_kfs = pinned_kfs + other_kfs

        best_result: Optional[Tuple[np.ndarray, int, float]] = None
        best_score = -1

        for kf in ordered_kfs:
            des_kf = kf["descriptors"]
            pts_kf = kf["pts_full"]
            if des_kf is None or len(des_kf) < min_inliers:
                continue

            raw_matches = self._matcher.knnMatch(des_curr, des_kf, k=2)
            # Lowe oranı + trainIdx tekilleştirme (aynı referans noktasına çoklu eşleşmeyi önle)
            best_by_train: Dict[int, cv2.DMatch] = {}
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m1, m2 = pair[0], pair[1]
                if m1.distance < lowe_ratio * m2.distance:
                    prev_m = best_by_train.get(m1.trainIdx)
                    if prev_m is None or m1.distance < prev_m.distance:
                        best_by_train[m1.trainIdx] = m1

            good_matches = list(best_by_train.values())
            if len(good_matches) < min_inliers:
                continue

            src_m = np.array(
                [pts_curr_full[m.queryIdx] for m in good_matches], dtype=np.float32
            )
            dst_m = np.array(
                [pts_kf[m.trainIdx] for m in good_matches], dtype=np.float32
            )

            H_curr_to_kf, inlier_mask = cv2.findHomography(
                src_m, dst_m, cv2.RANSAC, ransac_thresh
            )
            if H_curr_to_kf is None or inlier_mask is None:
                continue

            inliers = inlier_mask.ravel().astype(bool)
            inlier_count = int(inliers.sum())
            inlier_ratio = inlier_count / float(len(good_matches))
            if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
                continue

            # Mekansal yayılım kontrolü (lokal çim/logo sahte eşleşmelerini reddet)
            in_curr = src_m[inliers]
            in_kf = dst_m[inliers]
            if (
                float(in_curr[:, 0].max() - in_curr[:, 0].min()) < 0.20 * float(w)
                or float(in_curr[:, 1].max() - in_curr[:, 1].min()) < 0.15 * float(h)
                or float(in_kf[:, 0].max() - in_kf[:, 0].min()) < 0.20 * float(w)
                or float(in_kf[:, 1].max() - in_kf[:, 1].min()) < 0.15 * float(h)
            ):
                continue

            H_step_valid = self._validate_homography(
                H_curr_to_kf, frame_shape, is_step=False
            )
            if H_step_valid is None:
                continue

            # Yeniden projeksiyon hatası kontrolü
            proj_dst = cv2.perspectiveTransform(
                in_curr.reshape(1, -1, 2).astype(np.float64), H_step_valid
            )[0]
            reproj_errs = np.linalg.norm(
                proj_dst - in_kf.astype(np.float64), axis=1
            )
            if float(np.median(reproj_errs)) > max_reproj_err:
                continue

            M_cand = kf["M_k"] @ H_step_valid
            M_valid = self._validate_homography(M_cand, frame_shape, is_step=False)
            if M_valid is None:
                continue

            if inlier_count > best_score:
                best_score = inlier_count
                best_result = (M_valid, inlier_count, inlier_ratio)

            if inlier_count >= 30 and inlier_ratio >= 0.45:
                break

        return best_result

    # ----------------------------------------------------------------
    def _try_recover_from_keyframe_bank(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        frame_shape: Tuple[int, ...],
        signals: Dict[str, object],
    ) -> Optional[np.ndarray]:
        if not self._keyframe_bank:
            return None

        # Adaptif ana kamera kapısı (kare 0 yeşil saha ise aday da yeşil saha olmalı)
        green_min_anchor = float(getattr(config, "CAMERA_MAIN_GREEN_MIN_ANCHOR", 0.15))
        if self._anchor_green_ratio >= green_min_anchor:
            min_green = max(
                float(getattr(config, "CAMERA_MAIN_GATE_MIN_GREEN_RATIO", 0.12)),
                float(getattr(config, "CAMERA_MAIN_GATE_GREEN_FACTOR", 0.40))
                * self._anchor_green_ratio,
            )
            max_green = float(getattr(config, "CAMERA_MAIN_GREEN_MAX_RATIO", 1.0))
            green_ratio = float(signals["green_ratio"])  # type: ignore[arg-type]
            if not (min_green <= green_ratio <= max_green):
                return None

        min_edge_density = float(getattr(config, "CAMERA_MAIN_MIN_EDGE_DENSITY", 0.005))
        if float(signals["edge_density"]) < min_edge_density:  # type: ignore[arg-type]
            return None

        pts_curr_full, des_curr = self._extract_orb_full(gray, mask)
        min_inliers = int(getattr(config, "CAMERA_RECOVERY_MIN_INLIERS", 12))
        min_inlier_ratio = float(getattr(config, "CAMERA_RECOVERY_MIN_INLIER_RATIO", 0.30))

        match = self._match_against_keyframe_bank(
            pts_curr_full,
            des_curr,
            frame_shape,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            earliest_first=False,
        )
        if match is None:
            return None

        return match[0]

    # ----------------------------------------------------------------
    def _maybe_reanchor_drift(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        frame_shape: Tuple[int, ...],
    ) -> None:
        interval = int(getattr(config, "CAMERA_DRIFT_REANCHOR_INTERVAL", 60))
        if (self.frame_count - self._last_reanchor_frame) < interval:
            return
        if not self._keyframe_bank:
            return

        self._last_reanchor_frame = self.frame_count
        pts_curr_full, des_curr = self._extract_orb_full(gray, mask)
        min_inliers = int(getattr(config, "CAMERA_DRIFT_REANCHOR_MIN_INLIERS", 20))
        match = self._match_against_keyframe_bank(
            pts_curr_full,
            des_curr,
            frame_shape,
            min_inliers=min_inliers,
            min_inlier_ratio=0.40,
            earliest_first=True,
        )
        if match is None:
            return

        M_kf, _inliers, _ratio = match
        h, w = frame_shape[:2]
        corners = np.array(
            [[[0.0, 0.0]], [[float(w), 0.0]], [[float(w), float(h)]], [[0.0, float(h)]]],
            dtype=np.float64,
        )
        p_kf = cv2.perspectiveTransform(corners, M_kf)[:, 0, :]
        p_cur = cv2.perspectiveTransform(corners, self.cumulative_matrix)[:, 0, :]
        mean_disp = float(np.mean(np.linalg.norm(p_kf - p_cur, axis=1)))
        max_px = float(getattr(config, "CAMERA_DRIFT_REANCHOR_MAX_PX", 25.0))
        if math.isfinite(mean_disp) and mean_disp <= max_px:
            self.cumulative_matrix = M_kf

    # ----------------------------------------------------------------
    def _invalidate_anchor(
        self,
        gray: np.ndarray,
        gray_flow: np.ndarray,
        flow_scale: Tuple[float, float],
        signals: Dict[str, object],
    ) -> None:
        """
        Sahne kesmesinde mevcut çapayı (anchor) geçersiz kılar; ANCAK ana
        kamera anahtar kare bankasını (_keyframe_bank) ve _anchor_green_ratio'yu
        korur ki ana kameraya dönüldüğünde otomatik recovery yapılabilsin.
        """
        self.is_anchored = False
        self.scene_cut_detected = True
        self.tracking_lost = True
        self.shot_state = "NON_TACTICAL"
        self.consecutive_lost_frames = 0
        self._fade_degraded_count = 0
        self._recovery_streak = 0
        self.cumulative_matrix = np.eye(3, dtype=np.float64)
        self.prev_points = None
        self._store_prev_signals(
            gray, gray_flow, flow_scale, signals, reset_baseline=True
        )

    # ----------------------------------------------------------------
    def reset(self):
        """
        Takipçiyi tamamen sıfırlar (yeni bir video başladığında).
        Anahtar kare bankası dahil tüm dahili durum temizlenir.
        """
        self.prev_gray = None
        self.prev_gray_flow = None
        self.prev_points = None
        self._prev_flow_scale = (1.0, 1.0)
        self.cumulative_matrix = np.eye(3, dtype=np.float64)
        self.frame_count = 0
        self.tracking_lost = False
        self.scene_cut_detected = False
        self.is_anchored = False
        self.recovered_this_frame = False
        self.shot_state = "UNINITIALIZED"
        self.consecutive_lost_frames = 0
        self._healthy_anchored_frames = 0
        self._fade_degraded_count = 0
        self._recovery_streak = 0
        self._anchor_green_ratio = 0.0
        self._prev_thumb_gray = None
        self._prev_thumb_edges = None
        self._prev_hist_gray = None
        self._prev_hist_hsv = None
        self._baseline_hist_gray = None
        self._baseline_hist_hsv = None
        self._hist_window.clear()
        self._keyframe_bank.clear()
        self._last_kf_frame = -1000
        self._last_reanchor_frame = 0

    # ----------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None,
        frame_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Bir kareyi işler, kamera hareketini tahmin eder, sahne kesmelerini ve
        kademeli geçişleri yakalar, ana kameraya dönüşte yeniden hizalanır.

        Parametreler
        ------------
        frame : np.ndarray
            Mevcut kare (BGR).
        detections : List[Dict], optional
            Bu karedeki ham tespitler (oyuncu + top) -- arka plan maskesi için.
        frame_idx : int, optional
            Opsiyonel harici kare indeksi.

        Dönüş
        -----
        Optional[np.ndarray]
            Ana kameraya bağlıyken float64 3x3 birikimli dönüşüm matrisi (M_t),
            sahne kesmesi veya ana kamera dışı çekimlerde None.
        """
        self.scene_cut_detected = False
        self.recovered_this_frame = False
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._build_background_mask(frame.shape, detections)
        signals = self._compute_frame_signals(frame)

        gray_flow, sx, sy = self._get_flow_scale_and_gray(gray)
        if (sx, sy) != (1.0, 1.0):
            mask_flow = cv2.resize(
                mask,
                (gray_flow.shape[1], gray_flow.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_flow = mask
        scale_avg = 0.5 * (sx + sy)

        # --- İlk kare: ana referans çapası ve Pinned Keyframe 0 ---
        if self.prev_gray is None:
            self.cumulative_matrix = np.eye(3, dtype=np.float64)
            self.is_anchored = True
            self.tracking_lost = False
            self.shot_state = "ANCHORED"
            self.consecutive_lost_frames = 0
            self._healthy_anchored_frames = 1
            self._anchor_green_ratio = float(signals["green_ratio"])  # type: ignore[arg-type]
            self.prev_points = self._detect_features(gray_flow, mask_flow, scale_avg)
            self._store_prev_signals(
                gray, gray_flow, (sx, sy), signals, reset_baseline=True
            )
            self._maybe_insert_keyframe(gray, mask, frame.shape, pinned=True)
            self._last_reanchor_frame = self.frame_count
            return self.cumulative_matrix.copy()

        # --- Ana kamera dışı / çapasız durum (t+1, t+2, ...): recovery dene ---
        if not self.is_anchored:
            recovered_m = self._try_recover_from_keyframe_bank(
                gray, mask, frame.shape, signals
            )
            if recovered_m is None:
                self._recovery_streak = 0
                self.tracking_lost = True
                self.shot_state = "NON_TACTICAL"
                self.prev_points = None
                self._store_prev_signals(
                    gray, gray_flow, (sx, sy), signals, reset_baseline=False
                )
                return None

            self._recovery_streak += 1
            confirm_frames = int(getattr(config, "CAMERA_RECOVERY_CONFIRM_FRAMES", 1))
            if self._recovery_streak < confirm_frames:
                self.tracking_lost = True
                self.shot_state = "NON_TACTICAL"
                self._store_prev_signals(
                    gray, gray_flow, (sx, sy), signals, reset_baseline=False
                )
                return None

            # Yeniden hizalanma başarılı!
            self.cumulative_matrix = recovered_m
            self.is_anchored = True
            self.recovered_this_frame = True
            self.tracking_lost = False
            self.shot_state = "ANCHORED"
            self.consecutive_lost_frames = 0
            self._fade_degraded_count = 0
            self._recovery_streak = 0
            self._last_reanchor_frame = self.frame_count
            self.prev_points = self._detect_features(gray_flow, mask_flow, scale_avg)
            self._store_prev_signals(
                gray, gray_flow, (sx, sy), signals, reset_baseline=True
            )
            return self.cumulative_matrix.copy()

        # --- Çapalı (ANCHORED) durum: çok sinyalli sahne kesmesi & geçiş kontrolü ---
        thumb_gray = signals["thumb_gray"]  # type: ignore[assignment]
        thumb_edges = signals["thumb_edges"]  # type: ignore[assignment]
        hist_gray = signals["hist_gray"]  # type: ignore[assignment]
        hist_hsv = signals["hist_hsv"]  # type: ignore[assignment]

        corr_gray = float(
            cv2.compareHist(self._prev_hist_gray, hist_gray, cv2.HISTCMP_CORREL)
        )
        corr_hsv = float(
            cv2.compareHist(self._prev_hist_hsv, hist_hsv, cv2.HISTCMP_CORREL)
        )
        corr_min = min(corr_gray, corr_hsv)

        mad_prev = float(
            np.mean(cv2.absdiff(thumb_gray, self._prev_thumb_gray))
        ) / 255.0
        edge_mad_prev = float(
            np.mean(np.abs(thumb_edges - self._prev_thumb_edges))
        ) / 255.0

        base_corr_gray = float(
            cv2.compareHist(self._baseline_hist_gray, hist_gray, cv2.HISTCMP_CORREL)
        )
        base_corr_hsv = float(
            cv2.compareHist(self._baseline_hist_hsv, hist_hsv, cv2.HISTCMP_CORREL)
        )
        corr_baseline_min = min(base_corr_gray, base_corr_hsv)

        thresh_gray = float(config.SCENE_CUT_CORRELATION_THRESHOLD)
        thresh_hsv = float(getattr(config, "SCENE_CUT_HSV_CORR_THRESHOLD", 0.45))
        thresh_mad = float(getattr(config, "SCENE_CUT_MAD_THRESHOLD", 0.30))
        thresh_mad_hard = float(getattr(config, "SCENE_CUT_MAD_HARD_THRESHOLD", 0.45))
        thresh_fade = float(getattr(config, "SCENE_CUT_FADE_CORR_THRESHOLD", 0.55))

        pre_flow_cut = (
            corr_gray < thresh_gray
            or corr_hsv < thresh_hsv
            or (mad_prev > thresh_mad and corr_min < 0.72)
            or mad_prev > thresh_mad_hard
            or corr_baseline_min < thresh_fade
        )

        if pre_flow_cut:
            print(
                f"[CameraMotionTracker] UYARI: Kare {self.frame_count} civarında sahne "
                f"kesmesi/geçişi tespit edildi (corr_gray={corr_gray:.2f}, "
                f"corr_hsv={corr_hsv:.2f}, corr_base={corr_baseline_min:.2f})."
            )
            self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
            return None

        # Ölçek değişikliği olduysa önceki akış karesini hizala
        if (
            self.prev_gray_flow is None
            or self.prev_gray_flow.shape != gray_flow.shape
            or self._prev_flow_scale != (sx, sy)
        ):
            self.prev_gray_flow = (
                cv2.resize(
                    self.prev_gray,
                    (gray_flow.shape[1], gray_flow.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                if (sx, sy) != (1.0, 1.0)
                else self.prev_gray
            )
            self.prev_points = self._detect_features(
                self.prev_gray_flow, mask_flow, scale_avg
            )
            self._prev_flow_scale = (sx, sy)

        min_tracked = int(config.CAMERA_MOTION_MIN_TRACKED_POINTS)
        max_lost = int(getattr(config, "CAMERA_MAX_CONSECUTIVE_LOST_FRAMES", 15))

        def _handle_flow_loss() -> Optional[np.ndarray]:
            if corr_min < 0.78 or mad_prev > 0.12 or edge_mad_prev > 0.16:
                self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
                return None

            self.consecutive_lost_frames += 1
            self.tracking_lost = True
            if self.consecutive_lost_frames >= max_lost:
                recovered_m = self._try_recover_from_keyframe_bank(
                    gray, mask, frame.shape, signals
                )
                if recovered_m is not None:
                    self.cumulative_matrix = recovered_m
                    self.tracking_lost = False
                    self.consecutive_lost_frames = 0
                    self.prev_points = self._detect_features(
                        gray_flow, mask_flow, scale_avg
                    )
                    self._store_prev_signals(
                        gray, gray_flow, (sx, sy), signals, reset_baseline=True
                    )
                    return self.cumulative_matrix.copy()
                self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
                return None

            self.prev_points = self._detect_features(gray_flow, mask_flow, scale_avg)
            self._store_prev_signals(
                gray, gray_flow, (sx, sy), signals, reset_baseline=False
            )
            return self.cumulative_matrix.copy()

        if self.prev_points is None or len(self.prev_points) < min_tracked:
            return _handle_flow_loss()

        # --- Optical Flow ---
        curr_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_flow, gray_flow, self.prev_points, None, **self.lk_params
        )
        if curr_points is None or status is None:
            return _handle_flow_loss()

        status_flat = status.reshape(-1)
        good_prev_flow = self.prev_points[status_flat == 1]
        good_curr_flow = curr_points[status_flat == 1]

        if len(good_prev_flow) < min_tracked:
            return _handle_flow_loss()

        # Tam çözünürlük noktalarını hazırla ve D_t hesapla
        inv_scale = np.array([[1.0 / sx, 1.0 / sy]], dtype=np.float64)
        good_prev_full = good_prev_flow.reshape(-1, 2).astype(np.float64) * inv_scale
        good_curr_full = good_curr_flow.reshape(-1, 2).astype(np.float64) * inv_scale

        ransac_thresh_full = float(config.CAMERA_MOTION_RANSAC_THRESHOLD)
        ransac_thresh_flow = max(1.0, ransac_thresh_full * scale_avg)

        D_flow, inlier_mask = cv2.findHomography(
            good_curr_flow,
            good_prev_flow,
            cv2.RANSAC,
            ransac_thresh_flow,
        )

        D_full: Optional[np.ndarray] = None
        if D_flow is not None:
            D_flow64 = np.asarray(D_flow, dtype=np.float64)
            if (sx, sy) != (1.0, 1.0):
                S = np.diag([sx, sy, 1.0])
                S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
                D_full = S_inv @ D_flow64 @ S
            else:
                D_full = D_flow64
            if abs(float(D_full[2, 2])) > 1e-9:
                D_full = D_full / float(D_full[2, 2])
            else:
                D_full = None

        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = inlier_count / float(len(good_prev_flow))

        # --- Akış sonrası geometrik kesme / çim-renkli tekrar kesmesi kontrolü ---
        geom_min_inlier_ratio = float(
            getattr(config, "SCENE_CUT_GEOM_MIN_INLIER_RATIO", 0.25)
        )
        if (D_full is None or inlier_ratio < geom_min_inlier_ratio or inlier_count < 10) and (
            corr_min < 0.88
            or mad_prev > 0.08
            or edge_mad_prev > 0.12
            or corr_baseline_min < 0.75
        ):
            print(
                f"[CameraMotionTracker] UYARI: Kare {self.frame_count} civarında geometrik "
                f"sahne kesmesi tespit edildi (inlier_ratio={inlier_ratio:.2f}, "
                f"inliers={inlier_count})."
            )
            self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
            return None

        soft_fade_corr = float(
            getattr(config, "SCENE_CUT_FADE_SOFT_CORR_THRESHOLD", 0.72)
        )
        soft_fade_min_frames = int(getattr(config, "SCENE_CUT_FADE_MIN_FRAMES", 3))
        if corr_baseline_min < soft_fade_corr and inlier_ratio < 0.45:
            self._fade_degraded_count += 1
            if self._fade_degraded_count >= soft_fade_min_frames:
                self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
                return None
        else:
            self._fade_degraded_count = 0

        # Adım homografisi doğrulama + aşırı projektif terimlerde afin geri dönüş (affine fallback)
        min_step_inliers = int(getattr(config, "CAMERA_MOTION_MIN_INLIERS", 10))
        min_step_ratio = float(getattr(config, "CAMERA_MOTION_MIN_INLIER_RATIO", 0.30))

        step_valid = None
        if (
            D_full is not None
            and inlier_count >= min_step_inliers
            and inlier_ratio >= min_step_ratio
        ):
            h, w = frame.shape[:2]
            proj_mag = abs(float(D_full[2, 0])) * float(w) + abs(float(D_full[2, 1])) * float(h)
            if proj_mag <= 0.08:
                step_valid = self._validate_homography(
                    D_full, frame.shape, is_step=True
                )

        if step_valid is None and len(good_curr_full) >= min_step_inliers:
            M_aff, aff_inliers = cv2.estimateAffine2D(
                good_curr_full,
                good_prev_full,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh_full,
            )
            if M_aff is not None and aff_inliers is not None:
                aff_count = int(aff_inliers.sum())
                aff_ratio = aff_count / float(len(good_curr_full))
                if aff_count >= min_step_inliers and aff_ratio >= min_step_ratio:
                    D_aff = np.eye(3, dtype=np.float64)
                    D_aff[:2, :] = np.asarray(M_aff, dtype=np.float64)
                    step_valid = self._validate_homography(
                        D_aff, frame.shape, is_step=True
                    )
                    if step_valid is not None:
                        inlier_count = aff_count
                        inlier_ratio = aff_ratio

        if step_valid is None:
            return _handle_flow_loss()

        # --- Birikimli matrisi güncelle ve doğrula ---
        M_cand = self.cumulative_matrix @ step_valid
        M_valid = self._validate_homography(M_cand, frame.shape, is_step=False)
        if M_valid is None:
            recovered_m = self._try_recover_from_keyframe_bank(
                gray, mask, frame.shape, signals
            )
            if recovered_m is not None:
                M_valid = recovered_m
            else:
                self._invalidate_anchor(gray, gray_flow, (sx, sy), signals)
                return None

        self.cumulative_matrix = M_valid
        self.tracking_lost = False
        self.consecutive_lost_frames = 0
        self._healthy_anchored_frames += 1

        # Uzun çekimlerde biriken kaymayı erken anahtar karelerle düzelt ve yeni anahtar kare ekle
        if inlier_ratio >= 0.65:
            self._maybe_reanchor_drift(gray, mask, frame.shape)
            self._maybe_insert_keyframe(gray, mask, frame.shape, pinned=False)

        self._update_baseline_ema(signals, corr_min, corr_baseline_min, inlier_ratio)
        self.prev_points = self._detect_features(gray_flow, mask_flow, scale_avg)
        self._store_prev_signals(
            gray, gray_flow, (sx, sy), signals, reset_baseline=False
        )

        return self.cumulative_matrix.copy()


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa Adım 1+2+3+4+5'i, DİNAMİK kamera hareketi
# telafisiyle birlikte uçtan uca test eder.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modules.video_processor import VideoProcessor
    from modules.detector import ObjectDetector
    from modules.tracker import PlayerTracker, BallTracker
    from modules.team_classifier import TeamClassifier
    from modules.perspective_transformer import PerspectiveTransformer

    detector = ObjectDetector()
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    team_classifier = TeamClassifier()
    transformer = PerspectiveTransformer()
    motion_tracker = CameraMotionTracker()

    if not transformer.is_calibrated:
        print("\nKalibrasyon eksik. Önce utils/calibrate_homography.py ile kalibrasyon yap.\n")
    else:
        with VideoProcessor(config.DEFAULT_INPUT_VIDEO, config.DEFAULT_OUTPUT_VIDEO) as vp:
            for frame_idx, frame in vp.read_frames():
                raw_detections = detector.detect_frame(frame)
                tracked_players = player_tracker.update(raw_detections, frame)
                tracked_ball = ball_tracker.update(raw_detections)
                tracked_players = team_classifier.assign_teams(tracked_players, frame)

                all_tracked = tracked_players.copy()
                if tracked_ball is not None:
                    all_tracked.append(tracked_ball)

                # Kamera hareketini tahmin et
                motion_matrix = motion_tracker.update(frame, raw_detections)

                if motion_matrix is not None:
                    all_tracked = transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=motion_matrix)
                else:
                    # Sahne kesmesi -- bu kare için saha koordinatı üretme
                    for det in all_tracked:
                        det["pitch_x"], det["pitch_y"], det["on_pitch"] = None, None, None

                annotated_frame = vp.draw_detections(frame, all_tracked)
                vp.write_frame(annotated_frame)

        print("Dinamik kamera hareketi telafisi testi tamamlandı.")
