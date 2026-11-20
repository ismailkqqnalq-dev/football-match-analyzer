# Futbol Maç Analiz ve Oyuncu Takip Sistemi

Video → oyuncu/top tespiti → kalıcı takip → takım sınıflandırma → kamera
hareketi telafili saha (metre) koordinatları → görsel çıktı + CSV/SQL veri
dışa aktarımı.

## Kurulum
```bash
pip install -r requirements.txt
```

YOLO model ağırlığı ilk çalıştırmada otomatik indirilir (internet gerektirir),
ya da `models/` klasörüne manuel olarak `.pt` dosyası koyabilirsiniz.

## Çalıştırma öncesi: Homografi kalibrasyonu (Adım 5 için gerekli)
Saha (metre) koordinatları üretebilmek için, videonun uygun bir karesinde birkaç
bilinen saha noktasını (köşe, orta yuvarlak vb.) işaretlemeniz gerekiyor:
```bash
python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4
```
Bu adımı atlarsanız pipeline yine çalışır, sadece `pitch_x`/`pitch_y` alanları
`None` kalır (piksel koordinatları ve video çıktısı etkilenmez).

## Çalıştırma
```bash
python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
```

Ek CLI seçenekleri:
- `--batch-size N`: YOLO çıkarımı için batch boyutu (varsayılan: CUDA aktifse `8`, CPU'da `1`).
- `--max-frames N`: Hızlı test/önizleme için yalnızca ilk `N` kareyi işler.
- `--no-analytics`: Video bitiminde Adım 7 taktik analiz ve ısı haritası üretimini atlar.

Mevcut takip verisinden videoyu tekrar çalıştırmadan bağımsız analiz üretmek için:
```bash
python modules/match_analytics.py --input data/exports/tracking_data.csv
```

Çıktılar:
- `data/output_videos/`  -> kutular + ID + takım renkleri + (kalibre ise) taktik minimap ile işaretlenmiş video
- `data/exports/tracking_data.csv` ve `.db`  -> kare-kare ham takip verisi (backfill + top interpolasyonlu)
- `data/exports/player_distance_summary.csv`  -> oyuncu başına yumuşatılmış mesafe, hız ve FIFA hız bandı / sprint özeti
- `data/exports/match_analytics_summary.json` ve `possession_timeline.csv` -> topa sahip olma, pas ağı ve takım şekli özeti
- `data/exports/heatmaps/` -> takım ve öne çıkan oyuncular için 2D ısı haritası PNG görselleri

## Pipeline Adımları
| Adım | Modül | Görev |
|---|---|---|
| 1 | `modules/video_processor.py` | Video okuma/yazma/çizim + minimap bindirme |
| 2 | `modules/detector.py` | YOLO nesne tespiti (batch + CUDA FP16 destekli oyuncu/top) |
| 3 | `modules/tracker.py` | DeepSORT (GPU embedder) ile kalıcı ID + sönümlenen hızlı top takibi |
| 4 | `modules/team_classifier.py` | CIELAB/BGR K-Means forma renk kümeleme + önbellekli Takım A/B/Hakem/Kaleci ayrımı |
| 5 | `modules/perspective_transformer.py` + `modules/camera_motion_tracker.py` | Homografi + optical flow ile kamera hareketi telafisi -> saha (metre) koordinatları |
| 6 | `modules/data_exporter.py` | Pandas ile biriktirme, takım backfill + top interpolasyonu, CSV/SQLite + FIFA hız bantlı özet |
| 7 | `modules/match_analytics.py` | Topa sahip olma, pas tespiti/ağı, takım şekli ve 2D ısı haritası üretimi |

`config/config.py` tüm sabitleri (yollar, eşik değerleri, model parametreleri) tek yerde toplar.

## Testler
Otomatik birim ve entegrasyon testlerini çalıştırmak için:
```bash
pytest -v
```

## Varsayılan YOLO modeli hakkında (COCO, sınırlı)
Şu anki varsayılan `yolov8x.pt`, COCO veri setiyle eğitilmiştir ve sadece
`person` + `sports ball` sınıflarını tanır -- hakem/oyuncu ayrımını YOLO
seviyesinde YAPAMAZ (bunu Adım 4'teki K-Means renk kümelemesi tahmin eder).
Daha isabetli sonuç için aşağıdaki fine-tune seçeneğine bakın.

## (Opsiyonel) Futbola özel fine-tune edilmiş YOLO modeli kullanma
`utils/train_custom_yolo.py` betiği, Roboflow'daki halka açık
"football-players-detection" veri setiyle (ball/goalkeeper/player/referee
sınıfları) kendi fine-tune modelinizi eğitmenizi sağlar.

**Bu betik GPU ve internet gerektirir -- bu proje ortamında değil, Google
Colab gibi ücretsiz bir GPU ortamında çalıştırılmalıdır.** Adım adım talimat
için dosyanın başındaki docstring'e bakın; özetle:
```bash
pip install ultralytics roboflow
python train_custom_yolo.py --api-key SENIN_ROBOFLOW_API_KEYIN --epochs 100
```
Eğitim bitince betik, `config/config.py` içinde değiştirmeniz gereken TAM İKİ
satırı (`YOLO_MODEL_PATH` ve `TARGET_CLASS_IDS`) ekrana yazdırır. Başka
hiçbir dosyaya (main.py, tracker.py, team_classifier.py) dokunmanıza gerek
yok -- pipeline, bu iki değerden `HUMAN_CLASS_NAMES`, `BALL_CLASS_NAME` ve
`USES_FINE_TUNED_FOOTBALL_MODEL` gibi ayarları otomatik türetip kendini
buna göre ayarlar (ör. hakem artık K-Means'e hiç sokulmadan doğrudan
etiketlenir, K-Means küme sayısı 3'ten 2'ye düşer).

## Bilinen Sınırlamalar
- **Kamera hareketi telafisi (Adım 5)**: Optical flow tabanlı olduğu için
  zamanla küçük bir "drift" (kayma hatası) birikir; çok uzun kesintisiz
  çekimlerde periyodik yeniden kalibrasyon gerekebilir. Yayın başka bir
  kameraya (tekrar izleme vb.) geçtiğinde sistem bunu tespit edip o segment
  için saha koordinatı üretmeyi durdurur (otomatik toparlanmaz).
- **Mesafe/hız hesabı (Adım 6)**: Sadece ardışık karelerdeki (kayıp kare
  olmayan), gerçek (tahmini olmayan) tespitler arası mesafe sayılır.

## Klasör Yapısı
```
football_analysis_system/
├── main.py                          # pipeline'ı baştan sona çalıştıran orkestratör
├── requirements.txt
├── config/config.py                 # tüm sabitler tek yerde
├── modules/
│   ├── video_processor.py           # Adım 1
│   ├── detector.py                  # Adım 2
│   ├── tracker.py                   # Adım 3
│   ├── team_classifier.py           # Adım 4
│   ├── perspective_transformer.py   # Adım 5
│   ├── camera_motion_tracker.py     # Adım 5 eklentisi (kamera hareketi telafisi)
│   ├── data_exporter.py             # Adım 6
│   └── match_analytics.py           # Adım 7 (taktik analiz ve ısı haritaları)
├── tests/
│   └── test_pipeline.py             # otomatik birim/entegrasyon testleri
├── utils/
│   ├── calibrate_homography.py      # homografi kalibrasyon yardımcı aracı
│   └── train_custom_yolo.py         # (opsiyonel) fine-tune eğitim betiği
├── models/                          # YOLO .pt ağırlıkları
└── data/
    ├── input_videos/
    ├── output_videos/
    └── exports/                     # CSV + SQLite çıktıları
```
