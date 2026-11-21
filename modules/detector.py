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
from typing import Dict, List, Optional

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
        imgsz: int = config.DETECTION_IMG_SIZE,
        half: bool = config.DETECTION_HALF,
        target_class_ids: Optional[Dict[int, str]] = None,
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
        imgsz : int
            Çıkarım sırasında kullanılacak giriş görüntü boyutu (piksel).
        half : bool
            FP16 yarı duyarlıklı çıkarım (sadece CUDA aktifken etkinleşir).
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

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

        self.half = bool(half and self.device.startswith("cuda"))

        print(
            f"[ObjectDetector] Model yükleniyor: {model_path} "
            f"(device={self.device}, imgsz={self.imgsz}, half={self.half})"
        )

        # YOLO modelini yüklüyoruz. Ultralytics kütüphanesi hem .pt dosyasını
        # yerelden okur hem de yoksa otomatik indirir.
        self.model = YOLO(model_path)

        raw_targets = target_class_ids if target_class_ids is not None else config.TARGET_CLASS_IDS
        self.target_class_ids = {int(k): str(v) for k, v in raw_targets.items()}
        self._validate_model_names()

        print("[ObjectDetector] Model başarıyla yüklendi.")

    # ----------------------------------------------------------------
    def _validate_model_names(self) -> None:
        """
        Model ağırlığındaki `model.names` sınıf haritasının `self.target_class_ids`
        ile uyumlu olduğunu doğrular. Mock modellerde `names` yoksa atlanır.
        """
        if not self.target_class_ids:
            raise ValueError("[ObjectDetector] target_class_ids boş olamaz.")

        raw_names = getattr(self.model, "names", None)
        if raw_names is None:
            return

        if isinstance(raw_names, dict):
            model_names = {int(k): str(v).strip() for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            model_names = {i: str(v).strip() for i, v in enumerate(raw_names)}
        else:
            return

        mismatches = []
        for cls_id, expected_name in self.target_class_ids.items():
            expected_norm = expected_name.strip().lower()
            if cls_id not in model_names:
                mismatches.append(f"id={cls_id}: beklenen='{expected_name}', modelde yok")
                continue

            actual_name = model_names[cls_id]
            actual_norm = actual_name.strip().lower()
            is_coco_ball_alias = (
                cls_id == 32 and expected_norm == "ball" and actual_norm == "sports ball"
            )
            if actual_norm != expected_norm and not is_coco_ball_alias:
                mismatches.append(
                    f"id={cls_id}: beklenen='{expected_name}', modeldeki='{actual_name}'"
                )

        if mismatches:
            raise ValueError(
                "[ObjectDetector] Model sınıf eşlemesi config.TARGET_CLASS_IDS ile uyuşmuyor "
                "(config.YOLO_MODEL_PATH ve config.TARGET_CLASS_IDS ayarlarını kontrol edin). "
                f"Beklenen: {self.target_class_ids} | Model: {model_names} | "
                f"Uyuşmazlıklar: {'; '.join(mismatches)}"
            )

    # ----------------------------------------------------------------
    def _parse_result(self, result) -> List[Dict]:
        """Tek bir YOLO tahmin sonucunu standart tespit sözlüğü listesine dönüştürür."""
        if result is None or getattr(result, "boxes", None) is None:
            return []

        boxes = result.boxes
        if len(boxes) == 0:
            return []

        detections: List[Dict] = []

        # Gerçek Ultralytics Boxes nesnesi için tek seferde toplu GPU->CPU->NumPy dönüşümü
        if hasattr(boxes, "xyxy") and hasattr(boxes, "cls") and hasattr(boxes, "conf"):
            xyxy_raw = boxes.xyxy
            cls_raw = boxes.cls
            conf_raw = boxes.conf

            xyxy_arr = (
                xyxy_raw.detach().cpu().numpy()
                if hasattr(xyxy_raw, "detach")
                else (xyxy_raw.cpu().numpy() if hasattr(xyxy_raw, "cpu") else np.asarray(xyxy_raw))
            )
            cls_arr = (
                cls_raw.detach().cpu().numpy()
                if hasattr(cls_raw, "detach")
                else (cls_raw.cpu().numpy() if hasattr(cls_raw, "cpu") else np.asarray(cls_raw))
            )
            conf_arr = (
                conf_raw.detach().cpu().numpy()
                if hasattr(conf_raw, "detach")
                else (conf_raw.cpu().numpy() if hasattr(conf_raw, "cpu") else np.asarray(conf_raw))
            )

            for i in range(len(cls_arr)):
                class_id = int(cls_arr[i])
                if class_id not in self.target_class_ids:
                    continue

                x1, y1, x2, y2 = (float(v) for v in xyxy_arr[i])
                confidence = float(conf_arr[i])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": self.target_class_ids[class_id],
                    "confidence": round(confidence, 4),
                    "track_id": None,
                    "team": None,
                })
            return detections

        # Test mock'ları (list[FakeBox]) için uyumluluk yolu
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id not in self.target_class_ids:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
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
                "bbox": [x1, y1, x2, y2],     # piksel koordinatları (float)
                "class_id": int,               # YOLO sınıf ID'si
                "class_name": str,             # sınıf adı (ör. "player", "ball")
                "confidence": float,           # 0-1 arası güven skoru
                "track_id": None,              # 3. adımda (tracking) doldurulacak
                "team": None,                  # 4. adımda (K-Means) doldurulacak
            }
        """
        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                imgsz=self.imgsz,
                half=self.half,
                classes=list(self.target_class_ids.keys()),
                verbose=False,
            )

        if not results:
            return []

        return self._parse_result(results[0])

    # ----------------------------------------------------------------
    def _predict_batch_with_oom_fallback(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch çıkarımı çalıştırır; CUDA OutOfMemoryError durumunda önbelleği
        temizleyip batch'i fiziksel olarak yarıya bölerek tekrar dener.
        """
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=frames,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=self.imgsz,
                    half=self.half,
                    batch=len(frames),
                    classes=list(self.target_class_ids.keys()),
                    verbose=False,
                )
            return [self._parse_result(result) for result in results]
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
            if not is_oom:
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(frames) <= 1:
                raise torch.cuda.OutOfMemoryError(
                    f"[ObjectDetector] batch_size=1 iken bile CUDA belleği yetersiz "
                    f"(imgsz={self.imgsz}, half={self.half}). Orijinal hata: {exc}"
                ) from exc

            mid = len(frames) // 2
            return (
                self._predict_batch_with_oom_fallback(frames[:mid])
                + self._predict_batch_with_oom_fallback(frames[mid:])
            )

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
        if not frames:
            return []

        return self._predict_batch_with_oom_fallback(frames)


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
