"""
detector.py
-----------
ADIM 2: NESNE TESPİTİ (OBJECT DETECTION) MODÜLÜ

Bu modül, Ultralytics YOLOv8 modelini sarmalayan (wrap eden) bir sınıf sunar.
Sorumluluğu net: bir frame al, o frame içindeki oyuncu/hakem/top nesnelerini
bounding box + sınıf + güven skoru olarak döndür.

Tasarım notu (OOP - Dependency Inversion):
    Bu sınıf sadece "detect_frame(frame) -> List[Dict]" arayüzünü sağlar.
    main.py bu arayüzü kullanır, YOLO'nun iç detaylarını bilmesine gerek kalmaz.
    İleride YOLOv8'i başka bir modelle (örn. RT-DETR veya özel eğitilmiş bir
    futbol modeliyle) değiştirmek istersen, sadece bu dosyayı güncellersin;
    main.py ve video_processor.py hiç değişmez.
"""

import os
import sys
from typing import List, Dict

import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class ObjectDetector:
    """
    YOLOv8 tabanlı nesne tespit sınıfı.

    Kullanım örneği:
        detector = ObjectDetector()
        detections = detector.detect_frame(frame)
        # detections -> [{"bbox": [...], "class_name": "person", "confidence": 0.87}, ...]
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DETECTION_IOU_THRESHOLD,
        device: str = config.DEVICE,
    ):
        """
        Parametreler
        ------------
        model_path : str
            YOLO ağırlık dosyasının (.pt) yolu. Eğer dosya yoksa Ultralytics
            otomatik olarak COCO ile önceden eğitilmiş standart modeli indirir
            (ör. "yolov8x.pt" ismi verilirse internetten çeker).
        confidence_threshold : float
            Bu değerin altındaki tespitler filtrelenir (gürültü azaltma).
        iou_threshold : float
            Non-Max Suppression (örtüşen kutuları birleştirme) eşiği.
        device : str
            "cuda" (GPU) veya "cpu". GPU yoksa otomatik "cpu"ya düşer.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # --- ÖNEMLİ DÜZELTME ---
        # Daha önce burada "device" parametresi hiç kontrol edilmeden doğrudan
        # kullanılıyordu -- docstring "GPU yoksa otomatik cpu'ya düşer" diyordu
        # ama bu DAVRANIŞ HİÇ YAZILMAMIŞTI. Sonuç: config.py'de DEVICE="cuda"
        # iken CUDA'sız bir makinede çalıştırılınca Ultralytics/torch,
        # "Invalid CUDA 'device=0' requested" hatasıyla çöküyordu. Şimdi
        # gerçekten kontrol edip otomatik düşüyoruz.
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(
                f"[ObjectDetector] UYARI: '{device}' istendi ama bu makinede CUDA kullanılamıyor "
                f"(torch.cuda.is_available() == False). CPU'ya düşülüyor.\n"
                f"                 CPU'da özellikle yolov8x gibi büyük modeller ÇOK YAVAŞ çalışabilir "
                f"(kare başına saniyeler sürebilir). İki seçeneğin var:\n"
                f"                   1) NVIDIA GPU'n varsa CUDA destekli PyTorch kur:\n"
                f"                      pip uninstall torch torchvision -y\n"
                f"                      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                f"                      (kendi CUDA sürücü versiyonuna uygun linki https://pytorch.org/get-started/locally/ adresinden bul)\n"
                f"                   2) GPU'n yoksa, config.py'de YOLO_MODEL_PATH'i daha küçük/hızlı bir\n"
                f"                      modele çevir: YOLO_MODEL_PATH = os.path.join(MODELS_DIR, \"yolov8n.pt\")"
            )
            self.device = "cpu"
        else:
            self.device = device

        print(f"[ObjectDetector] Model yükleniyor: {model_path} (device={self.device})")

        # YOLO modelini yüklüyoruz. Ultralytics kütüphanesi hem .pt dosyasını
        # yerelden okur hem de yoksa otomatik indirir.
        self.model = YOLO(model_path)

        # Bizim ilgilendiğimiz sınıf ID'leri (config.py içinde tanımlı):
        #   0  -> person (oyuncu/hakem, henüz ayrılmamış)
        #   32 -> sports ball (top)
        self.target_class_ids = config.TARGET_CLASS_IDS

        print("[ObjectDetector] Model başarıyla yüklendi.")

    # ----------------------------------------------------------------
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Tek bir frame üzerinde nesne tespiti yapar.

        Parametreler
        ------------
        frame : np.ndarray
            BGR formatında (OpenCV varsayılanı) tek bir video karesi.

        Dönüş
        -----
        List[Dict]
            Her biri şu formatta bir sözlük listesi:
            {
                "bbox": [x1, y1, x2, y2],     # piksel koordinatları (int)
                "class_id": int,               # orijinal YOLO/COCO sınıf ID'si
                "class_name": str,              # "person" veya "ball"
                "confidence": float,            # 0-1 arası güven skoru
                "track_id": None,               # 3. adımda (tracking) doldurulacak
                "team": None,                   # 4. adımda (K-Means) doldurulacak
            }
        """
        # model.predict(...) tek bir frame için çıkarım (inference) yapar.
        # verbose=False -> konsola gereksiz log basmasın diye kapatıyoruz.
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict] = []

        # results tek elemanlı bir liste döner (tek frame verdiğimiz için)
        result = results[0]

        # result.boxes -> tespit edilen tüm kutuların bilgilerini tutan nesne
        for box in result.boxes:
            class_id = int(box.cls[0])

            # Sadece bizim ilgilendiğimiz sınıfları (person, ball) alıyoruz.
            # Örneğin COCO'daki "car", "dog" gibi alakasız sınıfları eliyoruz.
            if class_id not in self.target_class_ids:
                continue

            # Bounding box koordinatlarını al (x1, y1, x2, y2 formatında, piksel cinsinden)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": self.target_class_ids[class_id],
                "confidence": round(confidence, 4),
                "track_id": None,   # 3. adımda (Tracking) doldurulacak
                "team": None,       # 4. adımda (K-Means Team Classification) doldurulacak
            })

        return detections

    # ----------------------------------------------------------------
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Birden fazla frame'i tek seferde (batch) işleyerek GPU verimliliğini artırır.
        Büyük videolarda frame-by-frame yerine batch halinde tahmin yapmak
        çıkarım süresini ciddi oranda kısaltır.

        Parametreler
        ------------
        frames : List[np.ndarray]
            İşlenecek frame'lerin listesi.

        Dönüş
        -----
        List[List[Dict]]
            Her frame için ayrı bir detection listesi içeren liste.
        """
        results = self.model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        all_detections: List[List[Dict]] = []

        for result in results:
            frame_detections: List[Dict] = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.target_class_ids:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                frame_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": self.target_class_ids[class_id],
                    "confidence": round(confidence, 4),
                    "track_id": None,
                    "team": None,
                })

            all_detections.append(frame_detections)

        return all_detections


# ----------------------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa video_processor ile birlikte uçtan uca test eder:
# Videoyu okur -> her frame'de tespit yapar -> kutuları çizer -> yeni videoya yazar.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Modülü paket dışından da çalıştırılabilir yapmak için import'u burada tekrar yapıyoruz
    from modules.video_processor import VideoProcessor

    detector = ObjectDetector()

    with VideoProcessor(config.DEFAULT_INPUT_VIDEO, config.DEFAULT_OUTPUT_VIDEO) as vp:
        for frame_idx, frame in vp.read_frames():
            # 1) Bu frame'deki oyuncu/top tespitlerini al
            detections = detector.detect_frame(frame)

            # 2) Tespitleri frame üzerine çiz (henüz track_id/team yok, sadece kutu+sınıf)
            annotated_frame = vp.draw_detections(frame, detections)

            # 3) Çizilmiş frame'i çıktı videosuna yaz
            vp.write_frame(annotated_frame)

    print("Tespit testi tamamlandı. Çıktı videosunda oyuncu ve top kutularını görebilirsiniz.")
