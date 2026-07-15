# Futbol Maç Analiz ve Oyuncu Takip Sistemi

## Kurulum
```bash
pip install -r requirements.txt
```

YOLO model ağırlığı ilk çalıştırmada otomatik indirilir (internet gerektirir),
ya da `models/` klasörüne manuel olarak `.pt` dosyası koyabilirsiniz.

## Çalıştırma (şu anki durum: Adım 1 + Adım 2)
```bash
python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
```

## Klasör Yapısı
Detaylı açıklama için sohbetteki mesaja bakınız. Özet:

- `config/config.py`      -> tüm sabitler, yollar, eşik değerleri
- `modules/video_processor.py` -> Adım 1: video okuma/yazma/çizim
- `modules/detector.py`         -> Adım 2: YOLO nesne tespiti
- `modules/tracker.py`          -> Adım 3 (yakında): DeepSORT/ByteTrack
- `modules/team_classifier.py`  -> Adım 4 (yakında): K-Means forma renk analizi
- `modules/perspective_transformer.py` -> Adım 5 (yakında): homografi
- `modules/data_exporter.py`    -> Adım 6 (yakında): Pandas + SQL
- `main.py`               -> pipeline'ı baştan sona çalıştıran orkestratör
- `data/`                 -> giriş videoları, çıktı videoları, dışa aktarılan veriler
- `models/`                -> YOLO ağırlık dosyaları (.pt)
