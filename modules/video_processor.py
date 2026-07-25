"""
video_processor.py
-------------------
ADIM 1: VİDEO İŞLEME MODÜLÜ

Bu modül, tek bir sorumluluğa sahiptir (Single Responsibility Principle):
- MP4 videoyu frame frame okumak
- Her frame üzerine (varsa) bounding box + ID çizmek
- İşlenmiş frame'leri yeni bir video dosyasına yazmak

Nesne tespiti (YOLO) veya takip (tracking) mantığı BU DOSYADA YER ALMAZ.
Bu sınıf sadece "video giriş/çıkış" ve "çizim" işini üstlenir.
Böylece ileride tracker'ı veya detector'ı değiştirdiğimizde bu dosyaya
hiç dokunmamıza gerek kalmaz (Open/Closed Principle).
"""

import os
import cv2
import numpy as np
from typing import Generator, List, Dict, Optional
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class VideoProcessor:
    """
    Video okuma / yazma / kare üzerine çizim işlemlerinden sorumlu sınıf.

    Kullanım örneği:
        vp = VideoProcessor(input_path="match.mp4", output_path="match_out.mp4")
        for frame_idx, frame in vp.read_frames():
            # frame üzerinde detection/tracking yap...
            annotated_frame = vp.draw_detections(frame, detections)
            vp.write_frame(annotated_frame)
        vp.release()
    """

    def __init__(self, input_path: str, output_path: Optional[str] = None):
        """
        Parametreler
        ------------
        input_path : str
            İşlenecek giriş videosunun (.mp4) tam dosya yolu.
        output_path : str, optional
            Çizimli çıktı videosunun kaydedileceği yol. Verilmezse sadece
            okuma yapılabilir, yazma yapılamaz.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Giriş videosu bulunamadı: {input_path}")

        self.input_path = input_path
        self.output_path = output_path

        # cv2.VideoCapture: video dosyasını frame frame okumamızı sağlayan nesne
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise IOError(f"Video açılamadı: {input_path}")

        # Video meta verilerini (genişlik, yükseklik, fps, toplam frame sayısı) al
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"[VideoProcessor] Video yüklendi: {os.path.basename(input_path)} | "
            f"Çözünürlük: {self.frame_width}x{self.frame_height} | "
            f"FPS: {self.fps:.2f} | Toplam Frame: {self.total_frames}"
        )

        # Eğer çıktı yolu verildiyse, VideoWriter nesnesini hazırla.
        self.writer = None
        if self.output_path:
            self._init_writer()

    # ----------------------------------------------------------------
    def _init_writer(self):
        """cv2.VideoWriter nesnesini video meta verilerine göre başlatır."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Codec'i config'den al (örn. 'mp4v') ve fourcc koduna çevir
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)

        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        print(f"[VideoProcessor] Çıktı videosu hazırlanıyor: {self.output_path}")

    # ----------------------------------------------------------------
    def read_frames(self) -> Generator:
        """
        Videoyu baştan sona frame frame okuyan bir generator (jeneratör).

        Generator kullanmamızın sebebi: Tüm videoyu tek seferde belleğe
        yüklemek yerine (bu uzun maçlarda RAM'i patlatır), her seferinde
        tek bir frame üretip bellekte tutmak. Bu, büyük video dosyaları
        için bellek verimliliği sağlar.

        Yields
        ------
        (frame_index, frame) : Tuple[int, np.ndarray]
            frame_index -> videodaki sıra numarası (0'dan başlar)
            frame -> BGR formatında NumPy dizisi (OpenCV varsayılanı)
        """
        frame_index = 0

        # tqdm ile terminalde ilerleme çubuğu gösteriyoruz
        progress_bar = tqdm(total=self.total_frames, desc="Video okunuyor", unit="frame")

        while True:
            # cap.read() -> (başarı_bool, frame) döner
            success, frame = self.cap.read()

            if not success:
                # Video bitti veya bir okuma hatası oluştu
                break

            yield frame_index, frame

            frame_index += 1
            progress_bar.update(1)

        progress_bar.close()

    # ----------------------------------------------------------------
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """
        Verilen tespit/takip listesine göre frame üzerine bounding box ve ID çizer.

        Parametreler
        ------------
        frame : np.ndarray
            Üzerine çizim yapılacak orijinal kare.
        detections : List[Dict]
            Her biri şu formatta olan tespit sözlüklerinin listesi:
            {
                "bbox": [x1, y1, x2, y2],   # piksel koordinatları
                "track_id": int veya None,   # takip ID'si (3. adımda doldurulacak)
                "class_name": "person"/"ball" (COCO modeli) veya
                              "player"/"goalkeeper"/"referee"/"ball" (fine-tune model),
                "team": "Team A" / "Team B" / "Referee" / None,  # 4. adımda doldurulacak
                "confidence": float,
            }

        Dönüş
        -----
        np.ndarray
            Üzerine kutu ve ID çizilmiş yeni frame (orijinali bozulmaz, kopyası döner).
        """
        # Orijinal frame'i bozmamak için kopyasını alıyoruz
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det.get("class_name", "object")
            track_id = det.get("track_id", None)
            team = det.get("team", None)

            # Renk seçimi: top turuncu; takım bilgisi varsa (Adım 4'ten sonra
            # dolar) takıma özel renk; yoksa varsayılan yeşil kullanılır.
            if class_name == "ball":
                color = config.BBOX_COLOR_BALL
            elif team == "Team A":
                color = config.BBOX_COLOR_TEAM_A
            elif team == "Team B":
                color = config.BBOX_COLOR_TEAM_B
            elif team == "Referee":
                color = config.BBOX_COLOR_REFEREE
            else:
                color = config.BBOX_COLOR_DEFAULT

            # --- Bounding box çizimi ---
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            # --- Etiket metni oluşturma (ID + varsa takım bilgisi) ---
            label_parts = []
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            if team:
                label_parts.append(team)
            if not label_parts:
                label_parts.append(class_name)

            label = " | ".join(label_parts)

            # Etiketin arkasına okunabilirlik için dolu bir dikdörtgen (arka plan) çizelim
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.ID_FONT_SCALE, config.ID_FONT_THICKNESS
            )
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 4, y1),
                color,
                thickness=-1,  # -1 -> içi dolu dikdörtgen
            )

            # Etiket metnini yaz (siyah renk, dolu arka plan üzerinde okunaklı olsun diye)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.ID_FONT_SCALE,
                (0, 0, 0),
                config.ID_FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        return annotated

    # ----------------------------------------------------------------
    def overlay_minimap(
        self,
        frame: np.ndarray,
        minimap: np.ndarray,
        position: str = None,
        scale: float = None,
        margin: int = None,
    ) -> np.ndarray:
        """
        Kuş bakışı saha minimap'ini (perspective_transformer.draw_pitch_minimap()'ten
        gelir), ana video karesinin bir köşesine "resim-içinde-resim" (picture-in-
        picture) olarak gömer -- yayın kanallarındaki "taktik kamera" görünümü gibi.

        Parametreler
        ------------
        frame : np.ndarray
            Üzerine gömme yapılacak ana video karesi (BGR). BU KARE YERİNDE
            (in-place) DEĞİŞTİRİLİR -- ayrıca dönüş değeri olarak da verilir.
        minimap : np.ndarray
            perspective_transformer.draw_pitch_minimap()'ten gelen kuş bakışı görüntü.
        position, scale, margin : opsiyonel
            Verilmezse config.py'deki MINIMAP_OVERLAY_* varsayılanları kullanılır.

        Dönüş
        -----
        np.ndarray
            Minimap'i köşesinde taşıyan ana kare.
        """
        position = position or config.MINIMAP_OVERLAY_POSITION
        scale = scale if scale is not None else config.MINIMAP_OVERLAY_SCALE
        margin = margin if margin is not None else config.MINIMAP_OVERLAY_MARGIN_PX

        frame_h, frame_w = frame.shape[:2]
        minimap_h, minimap_w = minimap.shape[:2]

        # Minimap'i, video genişliğinin belirli bir oranı kadar olacak şekilde
        # küçült (en-boy oranını koruyarak).
        target_w = max(1, int(frame_w * scale))
        target_h = max(1, int(target_w * minimap_h / minimap_w))
        resized = cv2.resize(minimap, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Ana videodan görsel olarak ayrışsın diye ince beyaz bir kenarlık ekle.
        bordered = cv2.copyMakeBorder(resized, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        b_h, b_w = bordered.shape[:2]

        # Köşe konumunu hesapla. Video kareden küçükse (aşırı küçük çıktı
        # boyutu gibi uç durumlarda) taşmayı önlemek için sınırlıyoruz.
        if position == "bottom_right":
            x1, y1 = frame_w - b_w - margin, frame_h - b_h - margin
        elif position == "bottom_left":
            x1, y1 = margin, frame_h - b_h - margin
        elif position == "top_right":
            x1, y1 = frame_w - b_w - margin, margin
        else:  # "top_left" (varsayılan/bilinmeyen değer için de güvenli düşüş)
            x1, y1 = margin, margin

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x1 + b_w), min(frame_h, y1 + b_h)

        # Kare, minimap'ten küçükse (nadir ama olası) kırp -- çökmeyi önle.
        frame[y1:y2, x1:x2] = bordered[: y2 - y1, : x2 - x1]

        return frame

    # ----------------------------------------------------------------
    def write_frame(self, frame: np.ndarray):
        """İşlenmiş tek bir frame'i çıktı videosuna yazar."""
        if self.writer is None:
            raise RuntimeError(
                "VideoWriter başlatılmadı. VideoProcessor'ı output_path parametresiyle oluşturmalısın."
            )
        self.writer.write(frame)

    # ----------------------------------------------------------------
    def release(self):
        """Video okuma/yazma nesnelerini serbest bırakır (dosya kilitlerini kapatır)."""
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        print("[VideoProcessor] Kaynaklar serbest bırakıldı. İşlem tamamlandı.")

    # ----------------------------------------------------------------
    def __enter__(self):
        """Context manager desteği: 'with VideoProcessor(...) as vp:' kullanımı için."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with bloğu bittiğinde otomatik olarak kaynakları serbest bırakır."""
        self.release()


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa basit bir test yapar
# (sadece video okuma/yazma pipeline'ının çalıştığını doğrular, tespit yapmaz).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_input = config.DEFAULT_INPUT_VIDEO
    test_output = config.DEFAULT_OUTPUT_VIDEO

    with VideoProcessor(test_input, test_output) as vp:
        for frame_idx, frame in vp.read_frames():
            # Henüz detection yok, sadece frame numarasını sol üst köşeye yazalım
            cv2.putText(
                frame, f"Frame: {frame_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            vp.write_frame(frame)

    print("Test tamamlandı. Çıktı videosunu kontrol edin.")
