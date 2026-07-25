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
Saha (metre) koordinatları üretebilmek için, videonun ilk karesinde birkaç
bilinen saha noktasını (köşe, orta yuvarlak vb.) işaretlemeniz gerekiyor:
```bash
python utils/calibrate_homography.py --video data/input_videos/match_sample.mp4
```
Bu adımı atlarsanız pipeline yine çalışır, sadece `pitch_x`/`pitch_y` alanları
`None` kalır (piksel koordinatları ve video çıktısı etkilenmez).

## Çalıştırma
```bash
python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
```

Çıktılar:
- `data/output_videos/`  -> kutular + ID + takım renkleriyle işaretlenmiş video
- `data/exports/tracking_data.csv` ve `.db`  -> kare-kare ham takip verisi
- `data/exports/player_distance_summary.csv`  -> oyuncu başına mesafe/hız özeti

## Pipeline Adımları
| Adım | Modül | Görev |
|---|---|---|
| 1 | `modules/video_processor.py` | Video okuma/yazma/çizim |
| 2 | `modules/detector.py` | YOLO nesne tespiti (oyuncu/top) |
| 3 | `modules/tracker.py` | DeepSORT ile kalıcı ID takibi |
| 4 | `modules/team_classifier.py` | K-Means forma renk kümeleme -> Takım A/B/Hakem |
| 5 | `modules/perspective_transformer.py` + `modules/camera_motion_tracker.py` | Homografi + optical flow ile kamera hareketi telafisi -> saha (metre) koordinatları |
| 6 | `modules/data_exporter.py` | Pandas ile biriktirme, CSV + SQLite'a yazma, mesafe/hız özeti |

`config/config.py` tüm sabitleri (yollar, eşik değerleri, model parametreleri) tek yerde toplar.

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
│   └── data_exporter.py             # Adım 6
├── utils/
│   ├── calibrate_homography.py      # homografi kalibrasyon yardımcı aracı
│   └── train_custom_yolo.py         # (opsiyonel) fine-tune eğitim betiği
├── models/                          # YOLO .pt ağırlıkları
└── data/
    ├── input_videos/
    ├── output_videos/
    └── exports/                     # CSV + SQLite çıktıları
```
