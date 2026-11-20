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
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class PlayerTracker:
    """
    "İnsan" sayılan (config.HUMAN_CLASS_NAMES'te tanımlı) tespitleri DeepSORT
    ile takip eden sınıf. Bu isim kümesi, kullanılan YOLO modeline göre değişir:
        - COCO pretrained model: {"person"} (oyuncu/hakem henüz ayrılmamış)
        - Fine-tune futbol modeli: {"player", "goalkeeper", "referee"}
    Bu sayede aynı PlayerTracker kodu, hangi model kullanılırsa kullanılsın
    değişiklik gerektirmeden çalışır.

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
        half: bool = config.TRACKER_EMBEDDER_HALF,
    ):
        if embedder_gpu and not torch.cuda.is_available():
            print("[PlayerTracker] UYARI: embedder_gpu=True istendi ancak CUDA yok; CPU'ya düşülüyor.")
            embedder_gpu = False

        use_half = bool(half and embedder_gpu)

        print(
            f"[PlayerTracker] DeepSORT başlatılıyor "
            f"(max_age={max_age}, n_init={n_init}, embedder={embedder}, "
            f"embedder_gpu={embedder_gpu}, half={use_half})"
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
            half=use_half,                         # GPU'da FP16 hızlandırma
        )

    # ----------------------------------------------------------------
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Bir karedeki "person" tespitlerini tracker'a besleyip, kalıcı track_id
        atanmış güncel detection listesini döndürür.

        ÖNEMLİ: Bu metod SADECE config.HUMAN_CLASS_NAMES içindeki class_name'e
        sahip tespitleri işler. Top, ayrı olarak BallTracker ile yönetilir (aşağıda).

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
        # Sadece "insan" sayılan tespitleri alıyoruz (oyuncu/kaleci/hakem --
        # hangi sınıf isimlerinin buraya dahil olduğu config.HUMAN_CLASS_NAMES
        # ile TARGET_CLASS_IDS'ten otomatik türetilir, bkz. config.py).
        person_detections = [d for d in detections if d["class_name"] in config.HUMAN_CLASS_NAMES]

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

            # ÖNEMLİ DÜZELTME: Daha önce burası sabit "person" yazıyordu --
            # bu, fine-tune futbol modeli kullanıldığında "goalkeeper"/"referee"
            # gibi orijinal sınıf bilgisini SESSİZCE kaybederdi (hepsi "person"
            # olarak görünürdü). deep_sort_realtime, update_tracks()'e üçüncü
            # eleman olarak verdiğimiz class_name'i track.det_class üzerinden
            # geri veriyor -- onu kullanıyoruz. Track'in class'ı hiç
            # ayarlanmadıysa (teorik olarak olmamalı, ama savunmacı programlama
            # için) HUMAN_CLASS_NAMES'ten ilk değeri yedek olarak kullanıyoruz.
            original_class_name = (
                (track.get_det_class() if hasattr(track, "get_det_class") else getattr(track, "det_class", None))
                or next(iter(config.HUMAN_CLASS_NAMES))
            )
            det_conf = track.get_det_conf() if hasattr(track, "get_det_conf") else None

            tracked_detections.append({
                "bbox": [x1, y1, x2, y2],
                "track_id": int(track.track_id),
                "class_name": original_class_name,
                "confidence": round(float(det_conf), 4) if det_conf is not None else None,
                "team": None,        # Adım 4'te (K-Means) doldurulacak
                "is_predicted": track.time_since_update > 0,  # bu karede gerçek tespit yok, tahmin var
            })

        return tracked_detections


class BallTracker:
    """
    Topu takip eden sönümlenen hızlı (damped velocity) tekil (singleton) sınıf.

    DeepSORT gibi bir görünüm modeli kullanmaz; bunun yerine:
      1) Her karede en yüksek güven skorlu top tespitini seçer ve piksel hızını günceller.
      2) Top geçici olarak tespit edilemezse, (config.BALL_MAX_MISSING_FRAMES)
         kare boyunca son bilinen konumu sönümlenen hızla ileri taşır ("tahmini").
      3) Bu süre aşılırsa topu "kayıp" sayar ve None döndürür.
    """

    def __init__(
        self,
        max_missing_frames: int = config.BALL_MAX_MISSING_FRAMES,
        velocity_damping: float = config.BALL_VELOCITY_DAMPING,
        max_velocity_px: float = config.BALL_MAX_VELOCITY_PX_PER_FRAME,
    ):
        self.max_missing_frames = max_missing_frames
        self.velocity_damping = velocity_damping
        self.max_velocity_px = max_velocity_px
        self.last_known_detection: Optional[Dict] = None
        self.missing_frame_count: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_real_center: Optional[Tuple[float, float]] = None
        self._frames_since_real: int = 0

    # ----------------------------------------------------------------
    def reset(self) -> None:
        """Top takip durumunu sıfırlar."""
        self.last_known_detection = None
        self.missing_frame_count = 0
        self.velocity = (0.0, 0.0)
        self._last_real_center = None
        self._frames_since_real = 0

    # ----------------------------------------------------------------
    @staticmethod
    def _clamp_bbox_to_frame(bbox: List[float], frame_shape: Optional[Tuple[int, ...]]) -> List[float]:
        """Verilen bbox'ı kare boyutları içine (0..w, 0..h) kelepçeler."""
        if frame_shape is None or len(frame_shape) < 2:
            return bbox
        h, w = float(frame_shape[0]), float(frame_shape[1])
        x1, y1, x2, y2 = map(float, bbox)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= w:
            x1 = max(0.0, min(w - bw, x1))
            x2 = x1 + bw
        else:
            x1, x2 = 0.0, w
        if bh <= h:
            y1 = max(0.0, min(h - bh, y1))
            y2 = y1 + bh
        else:
            y1, y2 = 0.0, h
        return [x1, y1, x2, y2]

    # ----------------------------------------------------------------
    def update(
        self,
        detections: List[Dict],
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[Dict]:
        """
        Parametreler
        ------------
        detections : List[Dict]
            O karedeki tüm ham tespitler (person + ball karışık halde).
        frame_shape : Tuple[int, ...], optional
            Tahmin edilen konumun kare sınırları dışına taşmaması için (H, W, ...).

        Dönüş
        -----
        Optional[Dict]
            Topun güncel konumunu içeren tek bir detection sözlüğü, ya da
            top uzun süredir kayıpsa None.
        """
        ball_detections = [d for d in detections if d["class_name"] == config.BALL_CLASS_NAME]

        if ball_detections:
            # Birden fazla "top" tespiti varsa (yanlış pozitif olabilir),
            # en yüksek güven skoruna sahip olanı gerçek top kabul ediyoruz.
            best_detection = dict(max(ball_detections, key=lambda d: d["confidence"]))
            x1, y1, x2, y2 = map(float, best_detection["bbox"])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self._last_real_center is not None:
                self._frames_since_real += 1
                if self._frames_since_real <= self.max_missing_frames:
                    gap = max(1, self._frames_since_real)
                    vx = (cx - self._last_real_center[0]) / gap
                    vy = (cy - self._last_real_center[1]) / gap
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    if speed > self.max_velocity_px > 0:
                        scale = self.max_velocity_px / speed
                        vx *= scale
                        vy *= scale
                    self.velocity = (vx, vy)
                else:
                    self.velocity = (0.0, 0.0)
            else:
                self.velocity = (0.0, 0.0)

            self._last_real_center = (cx, cy)
            self._frames_since_real = 0

            best_detection["track_id"] = "ball"
            best_detection["is_predicted"] = False

            self.last_known_detection = dict(best_detection)
            self.last_known_detection["bbox"] = [x1, y1, x2, y2]
            self.missing_frame_count = 0
            return best_detection

        # Bu karede top tespit edilemedi
        self.missing_frame_count += 1
        if self._last_real_center is not None:
            self._frames_since_real += 1

        if self.last_known_detection is not None and self.missing_frame_count <= self.max_missing_frames:
            vx = self.velocity[0] * self.velocity_damping
            vy = self.velocity[1] * self.velocity_damping
            self.velocity = (vx, vy)

            x1, y1, x2, y2 = map(float, self.last_known_detection["bbox"])
            shifted_bbox = self._clamp_bbox_to_frame(
                [x1 + vx, y1 + vy, x2 + vx, y2 + vy],
                frame_shape,
            )
            self.last_known_detection["bbox"] = shifted_bbox

            predicted = dict(self.last_known_detection)
            predicted["bbox"] = list(shifted_bbox)
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
