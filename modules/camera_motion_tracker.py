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

import os
import sys
from typing import List, Dict, Optional

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class CameraMotionTracker:
    """
    Optical flow ile kare-kare kamera hareketini tahmin edip, kalibrasyon
    karesine göre BİRİKİMLİ bir dönüşüm matrisi üreten sınıf.

    Kullanım örneği:
        motion_tracker = CameraMotionTracker()
        for frame_idx, frame in vp.read_frames():
            ...
            motion_matrix = motion_tracker.update(frame, raw_detections)
            if motion_matrix is not None:
                all_tracked = transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=motion_matrix)
            else:
                # sahne kesmesi ya da takip kaybı -- bu kare için saha koordinatı üretme
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

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

        # M_t: şu anki karenin piksellerini, kalibrasyon karesinin (kare 0)
        # piksel sistemine eşleyen birikimli matris. Başlangıçta birim matris
        # (henüz hiç hareket olmadı, kare 0 = kare 0).
        self.cumulative_matrix = np.eye(3, dtype=np.float32)

        self.frame_count = 0
        self.tracking_lost = False       # bu karede yeterli nokta eşleşmedi mi?
        self.scene_cut_detected = False  # bu karede sahne kesmesi tespit edildi mi?

        print("[CameraMotionTracker] Oluşturuldu. İlk karede referans noktalar toplanacak.")

    # ----------------------------------------------------------------
    def _build_background_mask(self, frame_shape, detections: List[Dict]) -> np.ndarray:
        """
        Oyuncu/top bounding box'larının olduğu bölgeleri MASKELEYEREK (0 yaparak),
        goodFeaturesToTrack'in sadece SABİT arka plan (çim, çizgi, reklam panosu)
        üzerinde nokta aramasını sağlar. Hareketli nesneler üzerinde nokta
        aranırsa, kamera hareketi yerine OYUNCU hareketi ölçülmüş olur --
        bu da tüm hesaplamayı yanlış yapar.
        """
        mask = np.full(frame_shape[:2], 255, dtype=np.uint8)
        pad = config.PLAYER_MASK_PADDING_PX

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame_shape[1], x2 + pad)
            y2 = min(frame_shape[0], y2 + pad)
            mask[y1:y2, x1:x2] = 0

        return mask

    # ----------------------------------------------------------------
    def _detect_scene_cut(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        """
        İki ardışık kare arasındaki histogram korelasyonuna bakarak ani bir
        sahne değişimi (kamera kesmesi) olup olmadığını tespit eder.
        Korelasyon 1.0'a yakınsa kareler benzer, 0'a yakın/negatifse çok
        farklıdır (muhtemelen farklı bir kamera açısına geçilmiştir).
        """
        hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])

        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)

        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return correlation < config.SCENE_CUT_CORRELATION_THRESHOLD

    # ----------------------------------------------------------------
    def _detect_features(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Mevcut karede takip edilecek yeni özellik noktalarını (köşeleri) bulur."""
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=config.CAMERA_MOTION_MAX_CORNERS,
            qualityLevel=config.CAMERA_MOTION_QUALITY_LEVEL,
            minDistance=config.CAMERA_MOTION_MIN_DISTANCE,
            mask=mask,
        )

    # ----------------------------------------------------------------
    def reset(self):
        """
        Takipçiyi sıfırlar (sahne kesmesi sonrası ya da yeni bir klip
        başladığında). Birikimli matris tekrar birim matrise döner --
        yani "sıfırlama sonrası ilk kare, yeni referans karesi" olur.
        """
        self.prev_gray = None
        self.prev_points = None
        self.cumulative_matrix = np.eye(3, dtype=np.float32)
        self.frame_count = 0
        self.tracking_lost = False
        self.scene_cut_detected = False

    # ----------------------------------------------------------------
    def update(self, frame: np.ndarray, detections: List[Dict]) -> Optional[np.ndarray]:
        """
        Bir kareyi işler, kamera hareketini tahmin eder ve birikimli
        dönüşüm matrisini günceller.

        Parametreler
        ------------
        frame : np.ndarray
            Mevcut kare (BGR).
        detections : List[Dict]
            Bu karedeki ham tespitler (oyuncu + top) -- arka plan maskesi
            için "hangi bölgeler hareketli nesne, hangileri sabit arka plan"
            ayrımını yapmakta kullanılır.

        Dönüş
        -----
        Optional[np.ndarray]
            3x3 birikimli dönüşüm matrisi (M_t), ya da güvenilir bir tahmin
            yapılamadıysa (sahne kesmesi tespit edildiyse) None.
            NOT: Takip geçici olarak kaybolduysa (ör. motion blur) ama sahne
            kesmesi DEĞİLSE, matris DEĞİŞTİRİLMEDEN (son bilinen haliyle)
            döndürülür -- bu, video akışının aniden durmasını engeller.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._build_background_mask(frame.shape, detections)

        # --- İlk kare: sadece referans noktaları topla, hareket hesaplama ---
        if self.prev_gray is None:
            self.prev_points = self._detect_features(gray, mask)
            self.prev_gray = gray
            self.frame_count = 1
            return self.cumulative_matrix.copy()

        # --- Sahne kesmesi kontrolü ---
        if self._detect_scene_cut(self.prev_gray, gray):
            print(
                f"[CameraMotionTracker] UYARI: Kare {self.frame_count} civarında sahne "
                f"kesmesi tespit edildi (muhtemelen farklı bir kameraya geçildi). "
                f"Kamera takibi sıfırlanıyor -- bu segment için yeniden kalibrasyon gerekir."
            )
            self.reset()
            self.scene_cut_detected = True  # reset() sonrasında set ediyoruz ki reset onu geri False yapmasın
            self.prev_points = self._detect_features(gray, mask)
            self.prev_gray = gray
            self.frame_count = 1
            return None  # bu kare için güvenilir bir dönüşüm YOK

        self.scene_cut_detected = False

        # --- Yeterli referans noktası yoksa (nadir, örn. kamera tamamen
        # oyunculara/topa odaklandıysa) -- son bilinen matrisle devam et ---
        if self.prev_points is None or len(self.prev_points) < config.CAMERA_MOTION_MIN_TRACKED_POINTS:
            self.tracking_lost = True
            self.prev_points = self._detect_features(gray, mask)
            self.prev_gray = gray
            return self.cumulative_matrix.copy()

        # --- Optical Flow: önceki karedeki noktaların bu karede nereye
        # kaydığını hesapla ---
        curr_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if curr_points is None or status is None:
            self.tracking_lost = True
            self.prev_points = self._detect_features(gray, mask)
            self.prev_gray = gray
            return self.cumulative_matrix.copy()

        status = status.reshape(-1)
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) < config.CAMERA_MOTION_MIN_TRACKED_POINTS:
            # Yeterli sayıda nokta başarıyla takip edilemedi (ör. motion blur,
            # hızlı kamera hareketi). Matrisi DEĞİŞTİRMEDEN, son bilinen haliyle
            # devam ediyoruz -- ani sıçramalardansa hafif "donmuş" bir tahmin
            # daha güvenlidir.
            self.tracking_lost = True
            self.prev_points = self._detect_features(gray, mask)
            self.prev_gray = gray
            return self.cumulative_matrix.copy()

        self.tracking_lost = False

        # --- Kare-kare hareket homografisini hesapla: D_t (kare t -> kare t-1) ---
        D, _inlier_mask = cv2.findHomography(
            good_curr, good_prev, cv2.RANSAC, config.CAMERA_MOTION_RANSAC_THRESHOLD
        )

        if D is not None:
            # Birikimli matrisi güncelle: M_t = M_(t-1) @ D_t
            # (kare t pikseli -> D_t ile kare t-1'e -> M_(t-1) ile kare 0'a taşınır)
            self.cumulative_matrix = self.cumulative_matrix @ D
        # D None ise (dejenere nokta kümesi gibi nadir bir durum), matrisi
        # olduğu gibi bırakıyoruz -- en azından son geçerli tahminle devam ederiz.

        # --- Bir sonraki kare için referans noktaları YENİDEN tespit et.
        # Her karede yeniden tohumlama (re-seeding), noktaların zamanla
        # kaybolup takip kalitesinin düşmesini engeller. ---
        self.prev_points = self._detect_features(gray, mask)
        self.prev_gray = gray
        self.frame_count += 1

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
