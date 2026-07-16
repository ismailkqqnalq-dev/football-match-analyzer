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
        """
        self.source_points = source_points or config.SOURCE_CALIBRATION_POINTS
        self.destination_points_meters = destination_points_meters or config.PITCH_LANDMARKS_METERS

        self.homography_matrix: Optional[np.ndarray] = None
        self.is_calibrated = False

        self._compute_homography()

    # ----------------------------------------------------------------
    def _compute_homography(self):
        """
        config'deki eşleşen nokta çiftlerinden 3x3 homografi matrisini hesaplar.
        En az 4 nokta gereklidir (bir düzlem-düzlem dönüşümünün serbestlik derecesi 8'dir,
        her nokta 2 denklem verir -> 4 nokta = 8 denklem).
        """
        # Sadece HEM piksel HEM metre karşılığı bilinen (yani iki sözlükte de
        # ortak olan) landmark isimlerini kullanıyoruz.
        common_landmarks = [
            name for name in self.source_points
            if name in self.destination_points_meters
        ]

        if len(common_landmarks) < 4:
            print(
                f"[PerspectiveTransformer] UYARI: Sadece {len(common_landmarks)} kalibrasyon "
                f"noktası bulundu (en az 4 gerekli). Homografi hesaplanamadı -- "
                f"config.py -> SOURCE_CALIBRATION_POINTS'i doldurmalısın "
                f"(bkz. utils/calibrate_homography.py). Saha koordinatları üretilmeyecek."
            )
            self.is_calibrated = False
            return

        # Piksel noktaları (kaynak) ve metre noktaları (hedef) dizilerini hazırla.
        # cv2.findHomography float32 array bekler, şekil (N, 2).
        src_pts = np.array([self.source_points[name] for name in common_landmarks], dtype=np.float32)
        dst_pts = np.array([self.destination_points_meters[name] for name in common_landmarks], dtype=np.float32)

        # cv2.RANSAC: 4'ten fazla nokta varsa, tıklama hatası gibi aykırı
        # noktaları otomatik eleyerek daha sağlam bir matris bulur.
        # Tam 4 nokta varsa RANSAC'a gerek yok ama fonksiyon yine de çalışır.
        self.homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, config.HOMOGRAPHY_RANSAC_THRESHOLD
        )

        if self.homography_matrix is None:
            print("[PerspectiveTransformer] HATA: Homografi matrisi hesaplanamadı (noktalar dejenere olabilir).")
            self.is_calibrated = False
            return

        self.is_calibrated = True
        inlier_count = int(mask.sum()) if mask is not None else len(common_landmarks)
        print(
            f"[PerspectiveTransformer] Homografi başarıyla hesaplandı. "
            f"{len(common_landmarks)} nokta kullanıldı ({inlier_count} tanesi RANSAC inlier)."
        )

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
            (pitch_x_metre, pitch_y_metre) ya da kalibrasyon yapılmadıysa None.
        """
        matrix = homography_matrix if homography_matrix is not None else self.homography_matrix

        if matrix is None:
            return None

        # cv2.perspectiveTransform, noktaları (1, N, 2) şeklinde bekler.
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, matrix)

        pitch_x, pitch_y = transformed[0][0]
        return float(pitch_x), float(pitch_y)

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
            "on_pitch": bool               -> hesaplanan konum saha sınırları İÇİNDE mi?

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
        if not self.is_calibrated:
            for det in detections:
                det["pitch_x"] = None
                det["pitch_y"] = None
                det["on_pitch"] = None
            return detections

        # Efektif homografiyi belirle: dinamik (kamera hareketi telafili) mi,
        # statik mi?
        if camera_motion_matrix is not None:
            effective_matrix = self.homography_matrix @ camera_motion_matrix
        else:
            effective_matrix = self.homography_matrix

        for det in detections:
            foot_x, foot_y = self.get_foot_position(det["bbox"])
            result = self.transform_point(foot_x, foot_y, homography_matrix=effective_matrix)

            if result is None:
                det["pitch_x"] = None
                det["pitch_y"] = None
                det["on_pitch"] = None
                continue

            pitch_x, pitch_y = result
            det["pitch_x"] = round(pitch_x, 2)
            det["pitch_y"] = round(pitch_y, 2)

            # Küçük bir tolerans payı (1m) ile saha sınırları içinde mi kontrol et.
            margin = 1.0
            det["on_pitch"] = (
                -margin <= pitch_x <= config.PITCH_LENGTH_M + margin
                and -margin <= pitch_y <= config.PITCH_WIDTH_M + margin
            )

        return detections

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
        w = config.PITCH_MINIMAP_WIDTH_PX
        h = config.PITCH_MINIMAP_HEIGHT_PX
        margin = config.PITCH_MINIMAP_MARGIN_PX

        # Yeşil saha zemini oluştur
        minimap = np.full((h, w, 3), (60, 140, 30), dtype=np.uint8)  # BGR koyu yeşil

        # Metre -> minimap piksel dönüşümü için ölçek faktörü
        scale_x = (w - 2 * margin) / config.PITCH_LENGTH_M
        scale_y = (h - 2 * margin) / config.PITCH_WIDTH_M

        def to_minimap_px(meter_x, meter_y):
            px = int(margin + meter_x * scale_x)
            py = int(margin + meter_y * scale_y)
            return px, py

        # --- Saha çizgilerini çiz (dış çizgi, orta çizgi, orta yuvarlak) ---
        top_left = to_minimap_px(0, 0)
        bottom_right = to_minimap_px(config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)
        cv2.rectangle(minimap, top_left, bottom_right, (255, 255, 255), 2)

        halfway_top = to_minimap_px(52.5, 0)
        halfway_bottom = to_minimap_px(52.5, 68)
        cv2.line(minimap, halfway_top, halfway_bottom, (255, 255, 255), 2)

        center = to_minimap_px(52.5, 34)
        radius_px = int(9.15 * scale_x)  # orta yuvarlak yarıçapı FIFA standardı: 9.15m
        cv2.circle(minimap, center, radius_px, (255, 255, 255), 2)

        # --- Oyuncu/top konumlarını çiz ---
        for det in detections:
            if not det.get("on_pitch"):
                continue  # saha dışına düşen (muhtemelen hatalı) noktaları çizme

            px, py = to_minimap_px(det["pitch_x"], det["pitch_y"])

            if det.get("class_name") == "ball":
                cv2.circle(minimap, (px, py), 5, config.BBOX_COLOR_BALL, -1)
            elif det.get("team") == "Team A":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_TEAM_A, -1)
            elif det.get("team") == "Team B":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_TEAM_B, -1)
            elif det.get("team") == "Referee":
                cv2.circle(minimap, (px, py), 6, config.BBOX_COLOR_REFEREE, -1)
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
