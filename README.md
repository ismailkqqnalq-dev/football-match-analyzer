# Futbol Maç Analiz ve Oyuncu Takip Sistemi

Video → futbola özel 4 sınıflı YOLO ile oyuncu/kaleci/hakem/top tespiti →
kalıcı takip (DeepSORT / ByteTrack) → takım sınıflandırma → TV yayını sahne
kesmesi ve tekrar dönüşü telafili saha (metre) koordinatları → görsel çıktı +
90 dakikalık maç ölçeğinde akışlı (streaming) CSV/SQLite veri dışa aktarımı +
taktik analiz ve makine öğrenmesi raporları.

## Kurulum
```bash
pip install -r requirements.txt
```

Varsayılan yapılandırma `models/football_yolo11s_best.pt` konumundaki 4 sınıflı
(`0: ball`, `1: goalkeeper`, `2: player`, `3: referee`) futbola özel eğitilmiş
YOLO11s modelini kullanır.

## Çalıştırma öncesi: Homografi kalibrasyonu (Adım 5 için gerekli)
Saha (metre) koordinatları üretebilmek için, videonun uygun bir karesinde en az
4 bilinen saha noktasını (köşe, orta yuvarlak, ceza sahası köşesi vb.)
işaretlemeniz gerekiyor:
```bash
python utils/calibrate_homography.py --input data/input_videos/match_sample.mp4
```
Bu adımı atlarsanız pipeline yine çalışır, sadece `pitch_x`/`pitch_y` alanları
`None` kalır (piksel koordinatları, `pixel_fallback` modundaki analizler ve
video çıktısı etkilenmez).

## Çalıştırma
```bash
python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
```

### 90 Dakikalık Maç ve Üretim CLI Seçenekleri
- `--no-video`: Çıktı videosu oluşturmayı ve kare çizimini atlar; 90 dakikalık (~135.000 kare / ~2,7 milyon satır) maç analizlerinde disk ve işlem süresini önemli ölçüde azaltır.
- `--no-minimap`: Çıktı videosu üzerine kuş bakışı 2D taktik minimap bindirmeyi kapatır.
- `--no-csv` / `--no-sql`: İsteğe bağlı olarak CSV veya SQLite çıktılarından birini devre dışı bırakır.
- `--export-dir DİZİN`: Tüm veri, ısı haritası ve ML rapor çıktılarının yazılacağı hedef klasörü belirler (varsayılan: `data/exports`).
- `--flush-interval N`: Kare-kare takip verilerini her `N` karede bir SQLite (WAL modu) ve CSV'ye akıtarak RAM kullanımını sabit tutar (varsayılan: `2500`, `0` = sadece video sonunda yaz).
- `--tracker {deepsort,bytetrack}`: Oyuncu takipçisi seçimi (`deepsort` = görünüm + hareket, `bytetrack` = yüksek hızlı Kalman + IoU).
- `--embedding-stride N`: DeepSORT modunda görünüm embedding çıkarma aralığı (`1` = her kare, `>1` = ara karelerde hızlı IoU eşleştirme).
- `--batch-size N`: YOLO çıkarımı için batch boyutu (varsayılan: CUDA aktifse `8`, CPU'da `1`; CUDA OOM durumunda otomatik yarıya bölerek kurtarır).
- `--max-frames N`: Hızlı test/önizleme için yalnızca ilk `N` kareyi işler.
- `--no-analytics`: Video bitiminde Adım 7 taktik analiz ve ısı haritası üretimini atlar.
- `--no-ml`: Video bitiminde Adım 8 taktik ve scouting ML analizini (`xP`/`xT`, rol kümeleme, tempo/anomali) atlar.

Mevcut takip verisinden (CSV veya SQLite `.db`) videoyu tekrar çalıştırmadan bağımsız analiz üretmek için:
```bash
python modules/match_analytics.py --input data/exports/tracking_data.db
python modules/ml_analyzer.py --input data/exports/tracking_data.db
```

Çıktılar:
- `data/output_videos/` -> kutular + ID + takım renkleri + (kalibre ise) taktik minimap ile işaretlenmiş video
- `data/exports/tracking_data.csv` ve `.db` -> kare-kare ham takip verisi (parçalar arası takım backfill + kayan pencereli top interpolasyonlu)
- `data/exports/player_distance_summary.csv` -> oyuncu başına yumuşatılmış mesafe, hız ve FIFA hız bandı / sprint özeti
- `data/exports/match_analytics_summary.json` ve `possession_timeline.csv` -> topa sahip olma, pas ağı ve takım şekli özeti
- `data/exports/heatmaps/` -> takım ve öne çıkan oyuncular için 2D ısı haritası PNG görselleri
- `data/exports/ml_reports/` -> Adım 8 ML çıktıları (`ml_analysis_summary.json`, `ml_analysis_report.md`, `ml_player_profiles.csv`, `ml_tempo_windows.csv`, `ml_anomalies.csv`, `ml_pass_predictions.csv`, `xp_model_comparison.png`, `xt_spatial_heatmap.png`, `player_clusters_pca.png`, `tempo_anomaly_timeline.png`)

## Pipeline Adımları
| Adım | Modül | Görev |
|---|---|---|
| 1 | `modules/video_processor.py` | Video okuma/yazma/çizim (`--no-video` ve yerinde çizim destekli) + minimap bindirme |
| 2 | `modules/detector.py` | 4 sınıflı futbola özel YOLO11s nesne tespiti (batch + CUDA FP16 + OOM yarıya bölme korumalı) |
| 3 | `modules/tracker.py` | DeepSORT (GPU embedder + stride) ve ByteTrack (Kalman+IoU) ile kalıcı ID + sahne kesmesi sıfırlamalı top takibi |
| 4 | `modules/team_classifier.py` | CIELAB/BGR K-Means forma renk kümeleme + kilit sonrası geçmiş temizlemeli Takım A/B/Hakem/Kaleci ayrımı |
| 5 | `modules/perspective_transformer.py` + `modules/camera_motion_tracker.py` | Homografi + optical flow + ORB anahtar kare bankası ile TV sahne kesmesi/tekrar dönüşü kurtarmalı saha (metre) koordinatları |
| 6 | `modules/data_exporter.py` | Periyodik disk akışı (SQLite WAL + CSV), parçalar arası takım backfill + top interpolasyonu, FIFA hız bantlı özet |
| 7 | `modules/match_analytics.py` | Vektörel topa sahip olma, pas tespiti/ağı, takım şekli ve 2D ısı haritası üretimi |
| 8 | `modules/ml_analyzer.py` | Taktik & Scouting ML Motoru: `xP` pas başarı olasılığı, `xT` mekansal tehdit, Silhouette + 2B PCA oyuncu rol kümeleme, tempo/yorgunluk ve anomali tespiti |

`config/config.py` tüm sabitleri (yollar, eşik değerleri, model parametreleri) tek yerde toplar.

## Futbola Özel 4 Sınıflı YOLO Modeli Eğitimi (`utils/train_custom_yolo.py`)
`utils/train_custom_yolo.py` betiği, Roboflow üzerindeki `football-players-detection`
veri setini (`ball`, `goalkeeper`, `player`, `referee`) geçici `%TEMP%\football_yolo_training`
dizinine indirerek yerel GPU (örn. RTX 4050 6 GB VRAM) üzerinde `yolo11s.pt` taban
modeliyle (`imgsz=960`, `batch=4`, `amp=True`, `cache=False`) eğitir ve en iyi ağırlığı
`models/football_yolo11s_best.pt` konumuna kopyalar:

```powershell
$env:ROBOFLOW_API_KEY = "SENIN_ROBOFLOW_API_KEYIN"
$env:YOLO_CONFIG_DIR = "$env:TEMP\ultralytics_cfg"
python utils/train_custom_yolo.py --epochs 60 --imgsz 960 --batch 4
Remove-Item Env:ROBOFLOW_API_KEY -ErrorAction SilentlyContinue
```

## TV Yayını Sahne Kesmesi ve Uzun Çekim Kayma (Drift) Dayanıklılığı (Adım 5)
- **Çok Sinyalli Kesme ve Kademeli Geçiş (Dissolve) Tespiti**: `CameraMotionTracker`, gri/HSV histogram korelasyonu, kenar yoğunluğu, yeşil çim oranı ve optical flow RANSAC inlier oranını birlikte değerlendirerek sert kesmeleri, yakın çekim tekrarları ve kademeli geçişleri (fade/dissolve) yakalar.
- **Ana Kameraya Dönüşte Otomatik Yeniden Hizalanma (Recovery)**: Tekrar veya yakın çekim sırasında `update()` güvenli biçimde `None` döndürür (sahte saha koordinatı üretilmez); yayın ana taktik kameraya döndüğünde ORB anahtar kare bankası (`_keyframe_bank`) ile kalibrasyon koordinat sistemine otomatik olarak yeniden kilitlenir.
- **Skorbord Maskeleme ve Uzun Çekim Kararlılığı**: Sabit TV skorbord/logo bölgeleri ve oyuncu kutuları maskelenir, ayrıca periyodik anahtar kare doğrulaması ile uzun çekimlerde birikimli kayma (drift) sınırlandırılır.

## Testler
Otomatik birim, stres ve uçtan uca (E2E) testlerini çalıştırmak için:
```powershell
$env:YOLO_CONFIG_DIR="$env:TEMP\ultralytics_cfg"
pytest -v
```
