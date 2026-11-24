"""
perspective_transformer.py
---------------------------
ADIM 5: PERSPEKTİF DÖNÜŞÜMÜ (HOMOGRAPHY)

Bu modülün görevi: yayın kamerasının 2D perspektifinden gelen piksel
koordinatlarını (x, y), kuş bakışı (bird's-eye view) bir saha şablonu
üzerindeki GERÇEK METRİK koordinatlara (örn. "sahanın sol üst köşesinden
32.4 metre içeride, 18.1 metre kenarda") çevirmek.

NEDEN HOMOGRAFİ (HOMOGRAPHY)?
    Yayın kamerası sahayı bir AÇIDAN çekiyor (perspektif projeksiyon).
    Bu yüzden sahadaki paralel çizgiler (yan çizgiler gibi) görüntüde
    birbirine yakınsıyormuş gibi görünür, ve saha üzerindeki eşit
    mesafeler ekranda eşit piksel mesafelerine karşılık gelmez (kameraya
    yakın bölge daha büyük, uzak bölge daha küçük görünür).

    Eğer sahanın DÜZ bir zemin olduğunu varsayarsak (ki futbol sahaları
    düzdür), kameradaki herhangi bir nokta ile sahadaki gerçek karşılığı
    arasında matematik olarak bir "homografi matrisi" (3x3, 8 serbestlik
    dereceli) ile TAM bir dönüşüm kurulabilir. Bu matris bulunduktan sonra,
    görüntüdeki HERHANGİ bir (x, y) piksel noktası, sahadaki gerçek (X, Y)
    metre koordinatına çevrilebilir.

NASIL BULUNUYOR?
    En az 4 tane "eşleşen nokta çifti" gerekiyor: hem piksel koordinatını
    hem de gerçek metre koordinatını bildiğimiz noktalar (örn. ceza sahası
    köşeleri, orta yuvarlak, köşe bayrakları). Bu noktalar config.py'de
    PITCH_LANDMARKS_METERS (gerçek koordinatlar, sabit) ve
    SOURCE_CALIBRATION_POINTS (senin videondaki piksel karşılıkları, elle
    kalibre edilir) olarak tanımlı. cv2.findHomography() bu eşleşmelerden
    3x3'lük matrisi hesaplıyor.

ÖNEMLİ VARSAYIM / SINIRLAMA:
    Bu implementasyon TEK BİR SABİT KAMERA AÇISI varsayar -- yani kalibrasyonu
    bir kere yapıp tüm video boyunca AYNI homografi matrisini kullanırız.
    Eğer yayın kamerası maç sırasında pan/tilt/zoom yapıyorsa (ki çoğu
    profesyonel yayında yapar), bu matris zamanla GEÇERSİZLEŞİR ve
    dönüştürülen koordinatlar hatalı çıkar. Gerçek bir yayın kamerası için
    üretim seviyesinde bir sistem, HER KAREDE saha çizgilerini otomatik
    tespit edip (bir "pitch keypoint detection" modeliyle) homografiyi
    dinamik olarak güncellemelidir. Bu, ileri seviye bir geliştirme olarak
    roadmap'te tutulabilir; şimdilik tek kamera açılı / sabit geniş açı
    çekimler (örn. taktik kamera, drone, sabit yayın kamerası) için
    tasarlanmıştır.

AYAK NOKTASI NEDEN KULLANILIYOR (BBOX MERKEZİ DEĞİL)?
    Homografi sadece DÜZLEM (zemin) üzerindeki noktalar için geçerlidir.
    Bir oyuncunun bounding box'ının MERKEZİ, oyuncunun gövdesinin ortasına
    (yerden ~90 cm yükseklikte) denk gelir -- bu nokta zeminde DEĞİLDİR,
    dolayısıyla homografi ile çevirmek yanlış sonuç verir. Oysa oyuncunun
    AYAKLARI (bbox'ın alt-orta noktası) zeminle temas halindedir. Bu yüzden
    her zaman bbox'ın "alt-orta" noktasını (bottom-center) kullanıyoruz.
"""

import math
import os
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class PerspectiveTransformer:
    """
    Piksel koordinatlarını kuş bakışı saha (metre) koordinatlarına çeviren sınıf.

    Kullanım örneği:
        transformer = PerspectiveTransformer()  # config'deki kalibrasyon noktalarını kullanır
        tracked_players = transformer.add_pitch_coordinates(tracked_players)
        # artık her det["pitch_x"], det["pitch_y"] (metre) ve det["on_pitch"] (bool) dolu
    """

    def __init__(
        self,
        source_points: Optional[Dict[str, Tuple[float, float]]] = None,
        destination_points_meters: Optional[Dict[str, Tuple[float, float]]] = None,
        quiet: bool = False,
    ):
        """
        Parametreler
        ------------
        source_points : Dict[str, Tuple[float, float]], optional
            Landmark ismi -> videodaki piksel koordinatı. Verilmezse
            config.SOURCE_CALIBRATION_POINTS kullanılır.
        destination_points_meters : Dict[str, Tuple[float, float]], optional
            Landmark ismi -> sahadaki gerçek metre koordinatı. Verilmezse
            config.PITCH_LANDMARKS_METERS kullanılır.
        quiet : bool, optional
            True verilirse eksik kalibrasyon noktası başlangıç uyarısı bastırılır.
        """
        self.source_points = (
            source_points if source_points is not None else config.SOURCE_CALIBRATION_POINTS
        )
        self.destination_points_meters = (
            destination_points_meters
            if destination_points_meters is not None
            else config.PITCH_LANDMARKS_METERS
        )
        self.quiet = bool(quiet)

        self.homography_matrix: Optional[np.ndarray] = None
        self.is_calibrated = False
        self._cached_pitch_canvas: Optional[np.ndarray] = None
        self._cached_pitch_canvas_key: Optional[Tuple] = None

        self._compute_homography()

    # ----------------------------------------------------------------
    @staticmethod
    def _get_homography_thresholds() -> Tuple[float, float, float, float]:
        min_abs_det = float(getattr(config, "HOMOGRAPHY_MIN_ABS_DET", 1e-10))
        max_cond = float(
            getattr(
                config,
                "HOMOGRAPHY_MAX_CONDITION_NUMBER",
                getattr(config, "HOMOGRAPHY_MAX_COND", 1e8),
            )
        )
        w_eps = float(
            getattr(
                config,
                "HOMOGRAPHY_MIN_HOMOGENEOUS_W",
                getattr(config, "HOMOGRAPHY_DIVISOR_EPS", 1e-6),
            )
        )
        row2_min_norm = float(getattr(config, "HOMOGRAPHY_ROW2_MIN_NORM", 1e-12))
        return min_abs_det, max_cond, w_eps, row2_min_norm

    # ----------------------------------------------------------------
    @staticmethod
    def _validate_and_normalize_matrix(
        matrix: Optional[np.ndarray],
        min_abs_det: float = 1e-10,
        max_cond: float = 1e8,
        row2_min_norm: float = 1e-12,
    ) -> Optional[np.ndarray]:
        """
        3x3 homografi matrisini float64 hassasiyetinde doğrular ve son satırının
        L2 normuyla normalize eder (w' işaretini korur, sıfıra bölmeyi önler).
        Tekil, sonlu olmayan veya aşırı koşullu matrisler için None döndürür.
        """
        if matrix is None:
            return None

        try:
            m = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if m.ndim != 2 or m.shape != (3, 3):
            return None

        if not bool(np.isfinite(m).all()):
            return None

        row2_norm = float(np.linalg.norm(m[2, :]))
        if not math.isfinite(row2_norm) or row2_norm <= row2_min_norm:
            return None

        m_norm = m / row2_norm
        if not bool(np.isfinite(m_norm).all()):
            return None

        try:
            det = float(np.linalg.det(m_norm))
            if not math.isfinite(det) or abs(det) <= min_abs_det:
                return None

            cond = float(np.linalg.cond(m_norm))
            if not math.isfinite(cond) or cond > max_cond:
                return None
        except np.linalg.LinAlgError:
            return None

        return m_norm

    # ----------------------------------------------------------------
    def _compute_homography(self):
        """
        config'deki eşleşen nokta çiftlerinden 3x3 homografi matrisini hesaplar.
        En az 4 nokta gereklidir (bir düzlem-düzlem dönüşümünün serbestlik derecesi 8'dir,
        her nokta 2 denklem verir -> 4 nokta = 8 denklem).
        """
        self.homography_matrix = None
        self.is_calibrated = False

        # Sadece HEM piksel HEM metre karşılığı bilinen (yani iki sözlükte de
        # ortak olan) landmark isimlerini kullanıyoruz.
        common_landmarks = [
            name for name in self.source_points
            if name in self.destination_points_meters
        ]

        if len(common_landmarks) < 4:
            if not self.quiet:
                print(
                    f"[PerspectiveTransformer] UYARI: Sadece {len(common_landmarks)} kalibrasyon "
                    f"noktası bulundu (en az 4 gerekli). Homografi hesaplanamadı -- "
                    f"config.py -> SOURCE_CALIBRATION_POINTS'i doldurmalısın "
                    f"(bkz. utils/calibrate_homography.py). Saha koordinatları üretilmeyecek."
                )
            return

        # Piksel noktaları (kaynak) ve metre noktaları (hedef) dizilerini hazırla.
        try:
            src_pts = np.array(
                [self.source_points[name] for name in common_landmarks], dtype=np.float64
            )
            dst_pts = np.array(
                [self.destination_points_meters[name] for name in common_landmarks],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            print("[PerspectiveTransformer] HATA: Kalibrasyon noktaları sayısal değil.")
            return

        if (
            src_pts.ndim != 2
            or src_pts.shape[1] != 2
            or dst_pts.shape != src_pts.shape
            or not bool(np.isfinite(src_pts).all())
            or not bool(np.isfinite(dst_pts).all())
        ):
            print("[PerspectiveTransformer] HATA: Kalibrasyon noktaları geçersiz veya sonlu değil.")
            return

        # cv2.RANSAC: 4'ten fazla nokta varsa, tıklama hatası gibi aykırı
        # noktaları otomatik eleyerek daha sağlam bir matris bulur.
        H, mask = cv2.findHomography(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32),
            cv2.RANSAC,
            config.HOMOGRAPHY_RANSAC_THRESHOLD,
        )

        if H is None:
            print("[PerspectiveTransformer] HATA: Homografi matrisi hesaplanamadı (noktalar dejenere olabilir).")
            return

        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not bool(np.isfinite(H).all()):
            print("[PerspectiveTransformer] HATA: Hesaplanan homografi matrisi sonlu değil.")
            return

        inlier_mask = (
            mask.ravel().astype(bool)
            if mask is not None
            else np.ones(len(common_landmarks), dtype=bool)
        )
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 4:
            print(
                f"[PerspectiveTransformer] HATA: Yetersiz RANSAC inlier sayısı ({inlier_count} < 4)."
            )
            return

        min_abs_det, max_cond, w_eps, row2_min_norm = self._get_homography_thresholds()

        # Bilinen görünür kalibrasyon noktaları üzerinde w işaretini kanonikleştir:
        # dik yayın kameralarında (ufuk kare içindeyken) findHomography H'yi ters
        # işaretle döndürebilir; görünür noktalarda medyan w > 0 olacak şekilde hizala.
        inlier_src = src_pts[inlier_mask]
        w_cal = H[2, 0] * inlier_src[:, 0] + H[2, 1] * inlier_src[:, 1] + H[2, 2]
        if not bool(np.isfinite(w_cal).all()):
            print("[PerspectiveTransformer] HATA: Kalibrasyon noktalarında w değeri sonlu değil.")
            return

        if float(np.median(w_cal)) < 0.0:
            H = -H
            w_cal = -w_cal

        row2_norm = float(np.linalg.norm(H[2, :]))
        if not math.isfinite(row2_norm) or row2_norm <= row2_min_norm:
            print("[PerspectiveTransformer] HATA: Homografi son satır normu dejenere.")
            return

        # Bükülmüş (bowtie) veya ufuk çizgisiyle çakışan kalibrasyonları reddet
        if not bool(np.all(w_cal > w_eps * row2_norm)):
            print("[PerspectiveTransformer] HATA: Kalibrasyon noktaları tutarsız/bükülmüş homografi üretti.")
            return

        validated_H = self._validate_and_normalize_matrix(
            H,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )
        if validated_H is None:
            print("[PerspectiveTransformer] HATA: Homografi matrisi tekil veya aşırı koşullu.")
            return

        self.homography_matrix = validated_H
        self.is_calibrated = True
        print(
            f"[PerspectiveTransformer] Homografi başarıyla hesaplandı. "
            f"{len(common_landmarks)} nokta kullanıldı ({inlier_count} tanesi RANSAC inlier)."
        )

    # ----------------------------------------------------------------
    def set_homography(
        self,
        homography_matrix: Optional[np.ndarray],
        source_points: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> bool:
        """
        Harici bir kalibratörden gelen 3x3 piksel->metre homografi matrisini
        doğrulayıp normalize ederek uygular. Geçersiz matriste mevcut durumu korur.
        """
        if homography_matrix is None:
            return False

        try:
            H = np.asarray(homography_matrix, dtype=np.float64)
        except (TypeError, ValueError):
            return False

        if H.ndim != 2 or H.shape != (3, 3) or not bool(np.isfinite(H).all()):
            return False

        min_abs_det, max_cond, w_eps, row2_min_norm = self._get_homography_thresholds()

        if source_points is not None and len(source_points) > 0:
            try:
                pts = np.array(list(source_points.values()), dtype=np.float64)
            except (TypeError, ValueError):
                return False
            if pts.ndim != 2 or pts.shape[1] != 2 or not bool(np.isfinite(pts).all()):
                return False

            w_cal = H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + H[2, 2]
            if not bool(np.isfinite(w_cal).all()):
                return False
            if float(np.median(w_cal)) < 0.0:
                H = -H
                w_cal = -w_cal
            row2_norm = float(np.linalg.norm(H[2, :]))
            if not math.isfinite(row2_norm) or row2_norm <= row2_min_norm:
                return False
            if not bool(np.all(w_cal > w_eps * row2_norm)):
                return False

        validated_H = self._validate_and_normalize_matrix(
            H,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )
        if validated_H is None:
            return False

        self.homography_matrix = validated_H.copy()
        self.is_calibrated = True
        if source_points is not None:
            self.source_points = dict(source_points)
        return True

    # ----------------------------------------------------------------
    def clear_calibration(self) -> None:
        """Aktif homografi kalibrasyonunu temizleyip kalibrasyonsuz (T2) moda döner."""
        self.homography_matrix = None
        self.is_calibrated = False

    # ----------------------------------------------------------------
    def transform_point(
        self,
        pixel_x: float,
        pixel_y: float,
        homography_matrix: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Tek bir piksel noktasını sahadaki metre koordinatına çevirir.

        Parametreler
        ------------
        pixel_x, pixel_y : float
            Çevrilecek piksel koordinatı.
        homography_matrix : np.ndarray, optional
            Kullanılacak homografi matrisi. Verilmezse STATİK temel matris
            (self.homography_matrix) kullanılır -- yani kamera hareketi
            telafisi YAPILMAZ. Kamera hareketi telafili sonuç için,
            camera_motion_tracker.py'den gelen efektif matrisi buraya vermelisin
            (bkz. add_pitch_coordinates).

        Dönüş
        -----
        Optional[Tuple[float, float]]
            (pitch_x_metre, pitch_y_metre) ya da kalibrasyon/projeksiyon geçersizse None.
        """
        try:
            px = float(pixel_x)
            py = float(pixel_y)
        except (TypeError, ValueError):
            return None

        if not (math.isfinite(px) and math.isfinite(py)):
            return None

        min_abs_det, max_cond, w_eps, row2_min_norm = self._get_homography_thresholds()
        raw_matrix = homography_matrix if homography_matrix is not None else self.homography_matrix
        m_norm = self._validate_and_normalize_matrix(
            raw_matrix,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )
        if m_norm is None:
            return None

        # Ufuk çizgisi (w' ~ 0) ve kamera arkası (w' <= 0) projeksiyonlarını engelle
        w_prime = float(m_norm[2, 0] * px + m_norm[2, 1] * py + m_norm[2, 2])
        if not math.isfinite(w_prime) or w_prime <= w_eps:
            return None

        point = np.array([[[px, py]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(point, m_norm)

        pitch_x = float(transformed[0, 0, 0])
        pitch_y = float(transformed[0, 0, 1])
        if not (math.isfinite(pitch_x) and math.isfinite(pitch_y)):
            return None

        return pitch_x, pitch_y

    # ----------------------------------------------------------------
    @staticmethod
    def get_foot_position(bbox: List[float]) -> Tuple[float, float]:
        """
        Bir bounding box'tan "ayak noktasını" (alt-orta) hesaplar.
        Homografi dönüşümü SADECE zemine değen noktalar için doğru sonuç
        verdiğinden, oyuncunun kutusunun merkezi DEĞİL, alt-orta noktası
        kullanılmalıdır (bkz. modül başındaki açıklama).
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = y2  # kutunun alt kenarı
        return foot_x, foot_y

    # ----------------------------------------------------------------
    @staticmethod
    def _clear_pitch_fields(detections: List[Dict]) -> List[Dict]:
        for det in detections:
            det["pitch_x"] = None
            det["pitch_y"] = None
            det["on_pitch"] = None
        return detections

    # ----------------------------------------------------------------
    def add_pitch_coordinates(
        self,
        detections: List[Dict],
        camera_motion_matrix: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Bir tespit/takip listesindeki her elemana saha (metre) koordinatlarını ekler.

        Her elemana şu alanlar eklenir:
            "pitch_x": float veya None    -> sahanın uzun kenarı boyunca metre (0-105)
            "pitch_y": float veya None    -> sahanın kısa kenarı boyunca metre (0-68)
            "on_pitch": bool veya None    -> hesaplanan konum saha sınırları İÇİNDE mi?

        Parametreler
        ------------
        detections : List[Dict]
            "bbox" alanı olan herhangi bir tespit/takip listesi (oyuncu veya top).
        camera_motion_matrix : np.ndarray, optional
            camera_motion_tracker.py'den gelen, "şu anki karenin piksellerini
            kalibrasyon karesinin piksel sistemine taşıyan" birikimli matris.
            Verilirse, EFEKTİF homografi = self.homography_matrix @ camera_motion_matrix
            olarak hesaplanır -- yani kamera hareketi telafi edilmiş olur.
            Verilmezse (None), SABİT kamera varsayımıyla self.homography_matrix
            doğrudan kullanılır (Adım 5'in ilk, statik hali ile geriye dönük uyumluluk).

        Dönüş
        -----
        List[Dict]
            Aynı liste, saha koordinatları eklenmiş halde.
        """
        if not detections:
            return detections

        if not self.is_calibrated or self.homography_matrix is None:
            return self._clear_pitch_fields(detections)

        min_abs_det, max_cond, w_eps, row2_min_norm = self._get_homography_thresholds()
        base_m = self._validate_and_normalize_matrix(
            self.homography_matrix,
            min_abs_det=min_abs_det,
            max_cond=max_cond,
            row2_min_norm=row2_min_norm,
        )
        if base_m is None:
            return self._clear_pitch_fields(detections)

        # Efektif homografiyi belirle: dinamik (kamera hareketi telafili) mi, statik mi?
        if camera_motion_matrix is not None:
            motion_m = self._validate_and_normalize_matrix(
                camera_motion_matrix,
                min_abs_det=min_abs_det,
                max_cond=max_cond,
                row2_min_norm=row2_min_norm,
            )
            if motion_m is None:
                return self._clear_pitch_fields(detections)
            effective_m = self._validate_and_normalize_matrix(
                base_m @ motion_m,
                min_abs_det=min_abs_det,
                max_cond=max_cond,
                row2_min_norm=row2_min_norm,
            )
            if effective_m is None:
                return self._clear_pitch_fields(detections)
        else:
            effective_m = base_m

        n = len(detections)
        pts = np.zeros((n, 2), dtype=np.float64)
        finite_input = np.zeros(n, dtype=bool)

        for i, det in enumerate(detections):
            det["pitch_x"] = None
            det["pitch_y"] = None
            det["on_pitch"] = None

            bbox = det.get("bbox")
            if bbox is None:
                continue
            try:
                fx, fy = self.get_foot_position(bbox)
                fx_f = float(fx)
                fy_f = float(fy)
            except (TypeError, ValueError):
                continue

            if math.isfinite(fx_f) and math.isfinite(fy_f):
                pts[i, 0] = fx_f
                pts[i, 1] = fy_f
                finite_input[i] = True

        w_prime = (
            pts[:, 0] * effective_m[2, 0]
            + pts[:, 1] * effective_m[2, 1]
            + effective_m[2, 2]
        )
        valid_mask = finite_input & np.isfinite(w_prime) & (w_prime > w_eps)

        if bool(np.any(valid_mask)):
            valid_pts = pts[valid_mask].reshape(1, -1, 2)
            projected = cv2.perspectiveTransform(valid_pts, effective_m)[0]
            valid_indices = np.flatnonzero(valid_mask)

            margin = float(getattr(config, "PITCH_ON_PITCH_MARGIN_M", 1.0))
            pitch_length = float(config.PITCH_LENGTH_M)
            pitch_width = float(config.PITCH_WIDTH_M)

            for idx, (px_raw, py_raw) in zip(valid_indices, projected):
                px_f = float(px_raw)
                py_f = float(py_raw)
                if not (math.isfinite(px_f) and math.isfinite(py_f)):
                    continue

                det = detections[int(idx)]
                det["pitch_x"] = round(px_f, 2)
                det["pitch_y"] = round(py_f, 2)
                det["on_pitch"] = bool(
                    -margin <= px_f <= pitch_length + margin
                    and -margin <= py_f <= pitch_width + margin
                )

        return detections

    # ----------------------------------------------------------------
    def _get_static_pitch_canvas(
        self,
        w: int,
        h: int,
        margin: int,
        pitch_length: float,
        pitch_width: float,
        center_circle_radius: float,
    ) -> np.ndarray:
        cache_key = (w, h, margin, pitch_length, pitch_width, center_circle_radius)
        if (
            self._cached_pitch_canvas is not None
            and self._cached_pitch_canvas_key == cache_key
        ):
            return self._cached_pitch_canvas

        canvas = np.full((h, w, 3), (60, 140, 30), dtype=np.uint8)  # BGR koyu yeşil
        scale_x = (w - 2 * margin) / pitch_length
        scale_y = (h - 2 * margin) / pitch_width

        def to_px(mx: float, my: float) -> Tuple[int, int]:
            return int(margin + mx * scale_x), int(margin + my * scale_y)

        top_left = to_px(0.0, 0.0)
        bottom_right = to_px(pitch_length, pitch_width)
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)

        half_x = pitch_length / 2.0
        halfway_top = to_px(half_x, 0.0)
        halfway_bottom = to_px(half_x, pitch_width)
        cv2.line(canvas, halfway_top, halfway_bottom, (255, 255, 255), 2)

        center = to_px(half_x, pitch_width / 2.0)
        radius_px = int(center_circle_radius * scale_x)
        cv2.circle(canvas, center, radius_px, (255, 255, 255), 2)

        self._cached_pitch_canvas = canvas
        self._cached_pitch_canvas_key = cache_key
        return canvas

    # ----------------------------------------------------------------
    def draw_pitch_minimap(self, detections: List[Dict]) -> np.ndarray:
        """
        Kuş bakışı 2D bir saha şablonu çizip, üzerine oyuncuların/topun
        güncel konumlarını nokta olarak yerleştirir. Bu, kalibrasyonun
        doğru çalışıp çalışmadığını GÖRSEL olarak doğrulamak için de
        çok kullanışlıdır (örn. eğer noktalar sahanın çok dışına düşüyorsa,
        kalibrasyon noktalarında hata var demektir).

        Dönüş
        -----
        np.ndarray
            BGR formatında, sahayı ve üzerindeki oyuncu/top konumlarını
            gösteren bir görüntü (config.PITCH_MINIMAP_WIDTH_PX x HEIGHT_PX boyutunda).
        """
        w = int(config.PITCH_MINIMAP_WIDTH_PX)
        h = int(config.PITCH_MINIMAP_HEIGHT_PX)
        margin = int(config.PITCH_MINIMAP_MARGIN_PX)
        pitch_length = float(config.PITCH_LENGTH_M)
        pitch_width = float(config.PITCH_WIDTH_M)
        center_circle_radius = float(getattr(config, "CENTER_CIRCLE_RADIUS_M", 9.15))

        minimap = self._get_static_pitch_canvas(
            w, h, margin, pitch_length, pitch_width, center_circle_radius
        ).copy()

        # Metre -> minimap piksel dönüşümü için ölçek faktörü
        scale_x = (w - 2 * margin) / pitch_length
        scale_y = (h - 2 * margin) / pitch_width

        def to_minimap_px(meter_x: float, meter_y: float) -> Tuple[int, int]:
            px = int(margin + meter_x * scale_x)
            py = int(margin + meter_y * scale_y)
            return px, py

        # --- Oyuncu/top konumlarını çiz ---
        for det in detections:
            if not det.get("on_pitch"):
                continue  # saha dışına düşen (muhtemelen hatalı) noktaları çizme

            px_m = det.get("pitch_x")
            py_m = det.get("pitch_y")
            if px_m is None or py_m is None:
                continue

            try:
                px_f = float(px_m)
                py_f = float(py_m)
            except (TypeError, ValueError):
                continue

            if not (math.isfinite(px_f) and math.isfinite(py_f)):
                continue

            px, py = to_minimap_px(px_f, py_f)

            if det.get("class_name") == "ball":
                cv2.circle(minimap, (px, py), 5, config.BBOX_COLOR_BALL, -1)
            elif det.get("team") == "Team A":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_TEAM_A, -1)
            elif det.get("team") == "Team B":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_TEAM_B, -1)
            elif det.get("team") == "Referee":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_REFEREE, -1)
            elif det.get("team") == "Goalkeeper" or det.get("class_name") == "goalkeeper":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_GOALKEEPER, -1)
            else:
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_DEFAULT, -1)

        return minimap


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa Adım 1+2+3+4+5'i uçtan uca test eder ve
# minimap'i ayrı bir video olarak da kaydeder (kalibrasyonu görsel doğrulamak için).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modules.video_processor import VideoProcessor
    from modules.detector import ObjectDetector
    from modules.tracker import PlayerTracker, BallTracker
    from modules.team_classifier import TeamClassifier

    detector = ObjectDetector()
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    team_classifier = TeamClassifier()
    transformer = PerspectiveTransformer()

    if not transformer.is_calibrated:
        print(
            "\nKalibrasyon eksik olduğu için saha koordinatları üretilemeyecek.\n"
            "Önce şunu çalıştır: python utils/calibrate_homography.py --input <video_yolu>\n"
        )
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

                # Adım 5: saha koordinatlarını ekle
                all_tracked = transformer.add_pitch_coordinates(all_tracked)

                annotated_frame = vp.draw_detections(frame, all_tracked)
                vp.write_frame(annotated_frame)

        print("Homografi testi tamamlandı.")
