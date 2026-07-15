"""
tracker.py
----------
ADIM 3: NESNE TAKİBİ (OBJECT TRACKING) MODÜLÜ

Bu modülün görevi: detector.py'nin ürettiği "her karede sıfırdan tespit"
listesini alıp, kareler arasında **kalıcı kimlik (ID) sürekliliği** sağlamak.

Neden bu ayrı bir problem?
    YOLO her kareyi BAĞIMSIZ olarak işler; yani 5. karedeki "oyuncu" ile
    6. karedeki "oyuncu" arasında hiçbir bağlantı kurmaz, ikisi de sıfırdan
    tespit edilir. Tracking, bu tespitleri zaman içinde eşleştirip her
    nesneye "Bu, 3. karede gördüğümüz 7 numaralı kişiyle aynı kişi" diyen
    algoritmadır.

Neden DeepSORT (ByteTrack değil)?
    ByteTrack SADECE hareket (Kalman filtresi + IOU eşleştirme) kullanır.
    Bir oyuncu kamera açısından birkaç saniyeliğine çıkıp geri girdiğinde,
    ByteTrack onun "yeni bir kişi" olduğunu düşünüp yeni bir ID verir çünkü
    hareket geçmişi kopmuştur.
    DeepSORT ise her track için bir "görünüm embedding'i" (forma rengi,
    doku, şekil gibi görsel özellikler) saklar. Oyuncu geri döndüğünde,
    konumu tutmasa bile GÖRÜNÜMÜ üzerinden eski ID'sini geri alabilir.
    Bu yüzden yayın kameralarında (oyuncuların sık sık kadraj dışına
    çıktığı senaryo) DeepSORT daha uygun bir tercih.

Top (ball) NEDEN AYRI YÖNETİLİYOR?
    DeepSORT'un görünüm modeli, "kişi" görüntüleri üzerinde eğitilmiş
    CNN'lere dayanır ve küçük/tekdüze nesneler (top gibi) için güvenilir
    embedding üretmez. Sahada zaten tek bir top olduğundan, karmaşık bir
    takip algoritmasına gerek yok: her karede en yüksek güven skorlu tek
    top tespitini alıp sabit "ball" ID'si atıyoruz, kayıp karelerde ise
    kısa süreliğine son bilinen konumu koruyoruz.
"""

import os
import sys
from typing import List, Dict, Optional

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class PlayerTracker:
    """
    Oyuncu/hakem (class_name == "person") tespitlerini DeepSORT ile takip eden sınıf.

    Kullanım örneği:
        tracker = PlayerTracker()
        tracked = tracker.update(detections, frame)
        # tracked -> track_id alanı doldurulmuş detection listesi
    """

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        n_init: int = config.TRACKER_N_INIT,
        max_cosine_distance: float = config.TRACKER_MAX_COSINE_DISTANCE,
        nn_budget: int = config.TRACKER_NN_BUDGET,
        embedder: str = config.TRACKER_EMBEDDER,
        embedder_gpu: bool = config.TRACKER_EMBEDDER_GPU,
    ):
        print(
            f"[PlayerTracker] DeepSORT başlatılıyor "
            f"(max_age={max_age}, n_init={n_init}, embedder={embedder})"
        )

        # DeepSort nesnesi hem Kalman filtresi (hareket tahmini) hem de
        # appearance embedder'ı (görünüm karşılaştırması) içeride yönetir.
        self.tracker = DeepSort(
            max_age=max_age,                       # ID'nin "hayatta" kalma süresi (frame)
            n_init=n_init,                         # onay için gereken ardışık eşleşme sayısı
            max_cosine_distance=max_cosine_distance,  # görünüm benzerliği eşiği
            nn_budget=nn_budget,                   # track başına saklanan embedding sayısı
            embedder=embedder,                     # "mobilenet" -> hafif CNN
            embedder_gpu=embedder_gpu,
            bgr=True,                              # OpenCV kareleri BGR formatındadır
            half=False,                            # float16 hızlandırma (GPU'da True yapılabilir)
        )

    # ----------------------------------------------------------------
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Bir karedeki "person" tespitlerini tracker'a besleyip, kalıcı track_id
        atanmış güncel detection listesini döndürür.

        ÖNEMLİ: Bu metod SADECE class_name == "person" olan tespitleri işler.
        Top, ayrı olarak BallTracker ile yönetilir (aşağıda).

        Parametreler
        ------------
        detections : List[Dict]
            detector.py'den gelen, o karedeki ham (henüz track_id'siz) tespitler.
        frame : np.ndarray
            Mevcut kare. DeepSORT, görünüm embedding'ini çıkarmak için bounding
            box içindeki piksel verisine (yani frame'e) ihtiyaç duyar.

        Dönüş
        -----
        List[Dict]
            Her biri track_id doldurulmuş, sadece ONAYLANMIŞ (confirmed) track'leri
            içeren detection listesi. Henüz onaylanmamış (n_init'i tamamlamamış)
            "aday" track'ler bu listede görünmez -- bu, gürültülü/geçici yanlış
            tespitlerin ID almasını engeller.
        """
        # Sadece kişi (oyuncu/hakem) tespitlerini alıyoruz
        person_detections = [d for d in detections if d["class_name"] == "person"]

        # DeepSORT'un beklediği format: [([left, top, width, height], confidence, class_name), ...]
        # Bizim bbox formatımız [x1, y1, x2, y2] olduğu için width/height'a çeviriyoruz.
        raw_detections = []
        for det in person_detections:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            raw_detections.append(([x1, y1, width, height], det["confidence"], det["class_name"]))

        # update_tracks: hem Kalman tahminini günceller hem yeni tespitlerle
        # eşleştirme (appearance + motion) yapar hem de görünüm embedding'ini
        # frame'den çıkarır.
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        tracked_detections: List[Dict] = []

        for track in tracks:
            # is_confirmed(): track'in n_init eşiğini geçip "gerçek" kabul
            # edildiğini belirtir. Onaylanmamış track'leri (henüz kararsız
            # adaylar) çıktıya dahil etmiyoruz.
            if not track.is_confirmed():
                continue

            # time_since_update > 0 demek: bu karede EŞLEŞME bulunamadı,
            # Kalman filtresi tahmini konumu veriyor (nesne muhtemelen
            # geçici olarak kapanmış/görünmez). İstersen bu bilgiyi
            # ileride "interpolasyon" olarak işaretlemek için kullanabilirsin.
            l, t, w, h = track.to_ltwh()
            x1, y1, x2, y2 = float(l), float(t), float(l + w), float(t + h)

            tracked_detections.append({
                "bbox": [x1, y1, x2, y2],
                "track_id": int(track.track_id),
                "class_name": "person",
                "confidence": None,  # DeepSORT çıktısı orijinal güven skorunu tutmaz
                "team": None,        # Adım 4'te (K-Means) doldurulacak
                "is_predicted": track.time_since_update > 0,  # bu karede gerçek tespit yok, tahmin var
            })

        return tracked_detections


class BallTracker:
    """
    Topu takip eden basit, tekil (singleton) mantıklı sınıf.

    DeepSORT gibi bir görünüm modeli kullanmaz; sadece:
      1) Her karede en yüksek güven skorlu top tespitini seçer.
      2) Top hiç tespit edilmezse, kısa bir süre (config.BALL_MAX_MISSING_FRAMES)
         boyunca son bilinen konumu "tahmini" olarak döndürür.
      3) Bu süre aşılırsa topu "kayıp" sayar ve None döndürür.
    """

    def __init__(self, max_missing_frames: int = config.BALL_MAX_MISSING_FRAMES):
        self.max_missing_frames = max_missing_frames
        self.last_known_detection: Optional[Dict] = None
        self.missing_frame_count = 0

    # ----------------------------------------------------------------
    def update(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Parametreler
        ------------
        detections : List[Dict]
            O karedeki tüm ham tespitler (person + ball karışık halde).

        Dönüş
        -----
        Optional[Dict]
            Topun güncel konumunu içeren tek bir detection sözlüğü, ya da
            top uzun süredir kayıpsa None.
        """
        ball_detections = [d for d in detections if d["class_name"] == "ball"]

        if ball_detections:
            # Birden fazla "top" tespiti varsa (yanlış pozitif olabilir),
            # en yüksek güven skoruna sahip olanı gerçek top kabul ediyoruz.
            best_detection = max(ball_detections, key=lambda d: d["confidence"])

            best_detection = dict(best_detection)  # kopya al, orijinali bozma
            best_detection["track_id"] = "ball"
            best_detection["is_predicted"] = False

            self.last_known_detection = best_detection
            self.missing_frame_count = 0
            return best_detection

        # Bu karede top tespit edilemedi
        self.missing_frame_count += 1

        if self.last_known_detection is not None and self.missing_frame_count <= self.max_missing_frames:
            # Son bilinen konumu "tahmini" olarak döndürüyoruz.
            predicted = dict(self.last_known_detection)
            predicted["is_predicted"] = True
            return predicted

        # Top uzun süredir görünmüyor -> kayıp kabul et
        return None


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa Adım 1+2+3'ü uçtan uca test eder:
# Video oku -> tespit et -> takip et -> çiz -> yeni videoya yaz.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from modules.video_processor import VideoProcessor
    from modules.detector import ObjectDetector

    detector = ObjectDetector()
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    with VideoProcessor(config.DEFAULT_INPUT_VIDEO, config.DEFAULT_OUTPUT_VIDEO) as vp:
        for frame_idx, frame in vp.read_frames():
            # 1) Ham tespitler (Adım 2)
            raw_detections = detector.detect_frame(frame)

            # 2) Oyuncu/hakem takibi (DeepSORT, kalıcı ID)
            tracked_players = player_tracker.update(raw_detections, frame)

            # 3) Top takibi (basit singleton mantık)
            tracked_ball = ball_tracker.update(raw_detections)

            # 4) İkisini birleştirip çizim için tek listeye koy
            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            # 5) Çiz ve kaydet
            annotated_frame = vp.draw_detections(frame, all_tracked)
            vp.write_frame(annotated_frame)

    print("Takip testi tamamlandı. Her oyuncunun artık kalıcı bir ID'si olmalı.")
