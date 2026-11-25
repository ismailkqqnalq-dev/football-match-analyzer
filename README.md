# Futbol Maç Analiz ve Oyuncu Takip Sistemi

[![Tests](https://github.com/ismailkqqnalq-dev/football-match-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/ismailkqqnalq-dev/football-match-analyzer/actions/workflows/tests.yml)

Video → futbola özel 4 sınıflı YOLO ile oyuncu/kaleci/hakem/top tespiti →
kalıcı takip (DeepSORT / ByteTrack) → takım sınıflandırma → TV yayını sahne
kesmesi ve tekrar dönüşü telafili saha (metre) koordinatları → görsel çıktı +
90 dakikalık maç ölçeğinde akışlı (streaming) CSV/SQLite veri dışa aktarımı +
taktik analiz ve makine öğrenmesi raporları.

Ölçülen isabet, hedefler ve bilinen sınırlar için: [PROJE_DURUMU.md](PROJE_DURUMU.md).

## Kurulum
```bash
pip install -r requirements.txt
```

Varsayılan yapılandırma `models/football_yolo11s_best.pt` konumundaki 4 sınıflı
(`0: ball`, `1: goalkeeper`, `2: player`, `3: referee`) futbola özel eğitilmiş
YOLO11s modelini kullanır.

## Model ağırlıkları
`.pt` ağırlık dosyaları boyut ve lisans/dağıtım hijyeni nedeniyle Git deposuna
dahil edilmez. Yeni bir makinede modeli kurmak için iki yol vardır:

1. **Hazır ağırlığı indirme (`utils/download_models.py`)**:
   ```powershell
   python utils/download_models.py --url <MODEL_INDIRME_URL>
   ```
   `config/config.py` içindeki `MODEL_DOWNLOAD_URL` alanını doldurduysanız
   `python utils/download_models.py` komutu doğrudan çalışır ve
   `SECURITY_MODEL_SHA256` (`FOOTBALL_MODEL_SHA256` ortam değişkeni) tanımlıysa
   indirme sonrası SHA-256 bütünlük kontrolü yapar.

2. **Kendi GPU ortamınızda yeniden eğitme (`utils/train_custom_yolo.py`)**:
   Model Roboflow üzerindeki `football-players-detection` veri setiyle `yolo11s.pt`
   tabanından eğitilmiştir; yeniden eğitim komutları aşağıda yer almaktadır.

Model dosyası veya CUDA GPU bulunmayan makinelerde `pytest` çalıştırıldığında,
`requires_model` ve `gpu` işaretli testler hata vermeden gerekçesiyle atlanır (`skipped`).

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

### Kalibrasyon isabetini ölçme (Faz A2)
Otomatik kalibrasyonun gerçek videoda kaç metre şaştığını ölçmek için iki klipte
(bir TV yayını, bir sabit/geniş kamera) 20'şer kare etiketleyin:

1. Pipeline'ı çalıştırın; export dizinine kare bazında `homography_log.csv` yazılır
   (`--no-homography-log` ile kapatılabilir):
   ```powershell
   python main.py --input data/input_videos/klip.mp4 --no-video --export-dir data/exports/klip
   ```
2. Aynı klipte bilinen saha noktalarına tıklayarak etiketleyin (`n`: sonraki kare,
   `q`: bitir; nokta adını isim ya da listedeki numarayla girin):
   ```powershell
   python utils/calibrate_homography.py --input data/input_videos/klip.mp4 --label-mode --every-sec 15 --labels-out data/exports/klip/labels.csv
   ```
   Belirli kareler için `--every-sec` yerine `--frames 100,350,900` kullanılabilir.
3. Ölçümü çalıştırın; Türkçe özet ekrana, ayrıntılar `homography_eval.json`'a yazılır:
   ```powershell
   python utils/evaluate_homography.py --log data/exports/klip/homography_log.csv --labels data/exports/klip/labels.csv
   ```
   Özet; ortanca ve %90'lık dilim hatasını, etiketli karelerde geçerli tablo oranını
   ve kaynağa göre (`keypoint_*`, `flow`, `blend`) kırılımı verir. Etiketlenip tablosu
   olmayan kare "tablo yok" sayılır. Hedef: TV yayınında ortanca hata < 1 m, sabit
   kamerada < 0,5 m ve etiketli karelerin en az %90'ında geçerli tablo.

### Pas ve top sahipliği doğrulaması (Faz V)
Sistemin bulduğu pasların ve top kayıplarının gerçeğe ne kadar uyduğunu ölçmek için
5-10 dakikalık bir klibi elle etiketleyip karşılaştırın:

1. Pipeline'ı analizle birlikte çalıştırın; export dizinine `match_analytics_summary.json`
   (pas/kayıp olayları) ve `possession_timeline.csv` (kare bazında sahiplik) yazılır.
2. `utils/label_events.html` dosyasını tarayıcıda açın, aynı videoyu seçin ve izlerken
   işaretleyin: `1`/`2` top Takım A/B'de, `0` top kimsede değil, `P` tamamlanan pas
   (topun ayaktan çıktığı an), `T` top kaybı, `U` geri al, `[`/`]` yavaşlat/hızlandır.
   Bitince "CSV indir" ile `gt_events.csv` ve `gt_possession.csv` alınır. Video ve
   işaretler bilgisayardan çıkmaz.
3. Karşılaştırın:
   ```powershell
   python utils/validate_events.py --summary data/exports/klip/match_analytics_summary.json --timeline data/exports/klip/possession_timeline.csv --gt-events gt_events.csv --gt-possession gt_possession.csv --start-sec 0 --end-sec 300
   ```
   Bir olay, aynı türdeki etiketle 1 sn içindeyse doğru sayılır (`--tolerance` ile
   değişir). Çıktı; pas ve kayıp için kesinlik (sistemin dediklerinin kaçı doğru),
   duyarlılık (gerçek olayların kaçı yakalandı) ve F1, ayrıca saniye bazında sahiplik
   uyumunu verir. Etiketlerdeki Takım A/B sistemle ters düştüyse otomatik çevrilir.

## Çalıştırma
```bash
python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
```

### Video yerine video/maç linki vermek
`--input` parametresine dosya yolu yerine bir `http(s)` video/maç linki verilebilir:
```bash
python main.py --input "https://ornek-site.com/mac-ozeti" --no-video --max-frames 200
```
Desteklenen link türleri:
- Video paylaşım siteleri: `https://www.youtube.com/watch?v=...`
- İçinde gömülü video olan sayfalar (haber, kulüp, maç siteleri): `https://ornek-kulup.com/haberler/mac-ozeti`
- Doğrudan video dosyaları: `https://cdn.ornek.com/mac.mp4` (`.webm`, `.mkv` de olur)
- HLS / DASH akışları: `https://yayin.ornek.com/mac/index.m3u8`, `.../manifest.mpd`

Sayfada birden çok video varsa ilki alınır. Sayfada indirilebilir video yoksa veya video korumalıysa (DRM, ör. ücretli yayın platformları) anlaşılır bir hata mesajı verilir.

- Video `yt-dlp` ile en fazla 1080p (tercihen mp4/h264, sessiz) olarak geçici klasöre (`%TEMP%\fma_url_cache`) bir kez indirilir; aynı link tekrar verilirse önbellekteki dosya kullanılır. Yer açmak için bu klasör silinebilir.
- `--max-frames` verilse bile videonun tamamı indirilir (90 dk 1080p birkaç GB olabilir).
- Video siteleri sık değiştiği için indirme hata verirse önce `pip install -U yt-dlp` deneyin.
- Yasal not: video sitelerinin ve platformların kullanım şartları video indirmeyi kısıtlayabilir. Bu özellik kişisel analiz içindir ve kullanım sorumluluğu kullanıcıya aittir; proje ticari bir ürüne dönüşürse ayrıca değerlendirilmelidir.

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

## Otomatik Saha Kalibrasyonu ve Saha Anahtar Nokta Modeli (Faz A2)
`config.SOURCE_CALIBRATION_POINTS` boşsa ve `models/football_pitch_pose.pt` varsa pipeline
sahayı 32 anahtar noktayla (YOLO-Pose) otomatik kalibre eder; model yoksa kalibrasyonsuz
(T2) modda devam eder. Önemli davranışlar:
- **Kademeli kabul**: Model, optical flow tablosuyla düzenli olarak karşılaştırılır.
  Fark 0,5 m altındaysa tablo değiştirilmez; 1 m'ye kadar yumuşak geçişle kabul edilir;
  daha büyük fark ancak sonraki 2 yenilemede doğrulanırsa kabul edilir (tek seferlik hatalı
  tespitler oyuncuları sahada kaydırmaz). 3 m üstü doğrulanmış düzeltmeler sert geçişle
  uygulanır ve o kare `calib_reset` olarak işaretlenip mesafe/hız hesaplarından dışlanır.
- **Görüntü profili** (`config.FOOTAGE_PROFILE = "auto" | "fixed" | "broadcast"`): ilk 5 sn'de
  kamera hareketi ve sahne kesmesine bakılır. Sabit kamerada tek, en tutarlı tablo kilitlenir
  ve yalnızca 10 sn'de bir sağlık kontrolü yapılır; kesme ya da kamera hareketi görülürse
  yayın profiline geçilir (tersine geçiş yok). Karar `calibration_summary.json`'a yazılır.
- **Kayıt**: kare bazında kullanılan tablo `homography_log.csv`'ye yazılır (isabet ölçümü için).

### Saha anahtar nokta modelini eğitme (RTX 4050 6 GB)
Roboflow API anahtarı yalnızca ortam değişkeniyle verilir, hiçbir dosyaya yazılmaz.
1. Veri setini indirip **nokta sırasını kontrol edin** (eğitim başlamaz):
   ```powershell
   $env:ROBOFLOW_API_KEY = "SENIN_ROBOFLOW_API_KEYIN"
   $env:YOLO_CONFIG_DIR = "$env:TEMP\ultralytics_cfg"
   python utils/train_pitch_keypoints.py --check-order-only
   ```
   `runs/pitch_order_check/*.png` görsellerinde her nokta, `config.PITCH_KEYPOINT_NAMES_32`'deki
   adımızla çizilir. Adlar noktaların gerçek yerleriyle (köşe, ceza sahası köşesi, orta yuvarlak
   vb.) eşleşmiyorsa **eğitime geçmeyin**: sıra kaymış demektir ve tüm kalibrasyon sessizce
   yanlış olur. Betik ayrıca `kpt_shape == [32, 3]` kontrolü yapar; veri setinde `flip_idx`
   yoksa yatay çevirme eşlemesini bizim şablondan üretir, veri setindeki eşleme şablonla
   uyuşmuyorsa çevirmeyi kapatıp uyarır.
2. Görseller doğruysa eğitimi başlatın (yaklaşık 15-40 dk, veri seti boyutuna bağlı):
   ```powershell
   python utils/train_pitch_keypoints.py --epochs 60 --imgsz 640 --batch 4
   Remove-Item Env:ROBOFLOW_API_KEY -ErrorAction SilentlyContinue
   ```
3. Betik en iyi ağırlığı `models/football_pitch_pose.pt` olarak kaydeder, SHA-256 özetini ve
   `$env:FOOTBALL_PITCH_MODEL_SHA256` / `SECURITY_PITCH_MODEL_SHA256` satırlarını yazdırır.

## TV Yayını Sahne Kesmesi ve Uzun Çekim Kayma (Drift) Dayanıklılığı (Adım 5)
- **Çok Sinyalli Kesme ve Kademeli Geçiş (Dissolve) Tespiti**: `CameraMotionTracker`, gri/HSV histogram korelasyonu, kenar yoğunluğu, yeşil çim oranı ve optical flow RANSAC inlier oranını birlikte değerlendirerek sert kesmeleri, yakın çekim tekrarları ve kademeli geçişleri (fade/dissolve) yakalar.
- **Ana Kameraya Dönüşte Otomatik Yeniden Hizalanma (Recovery)**: Tekrar veya yakın çekim sırasında `update()` güvenli biçimde `None` döndürür (sahte saha koordinatı üretilmez); yayın ana taktik kameraya döndüğünde ORB anahtar kare bankası (`_keyframe_bank`) ile kalibrasyon koordinat sistemine otomatik olarak yeniden kilitlenir.
- **Skorbord Maskeleme ve Uzun Çekim Kararlılığı**: Sabit TV skorbord/logo bölgeleri ve oyuncu kutuları maskelenir, ayrıca periyodik anahtar kare doğrulaması ile uzun çekimlerde birikimli kayma (drift) sınırlandırılır.

## Testler
Otomatik birim, güvenlik ve uçtan uca (E2E) testlerini çalıştırmak için:
```powershell
$env:YOLO_CONFIG_DIR="$env:TEMP\ultralytics_cfg"
pytest -v
```

## Güvenlik Mimarisi ve Üretim (API / Docker) Yayını
Proje dış dünyaya/müşteriye bir servis olarak açıldığında olası saldırılara karşı **5 katmanlı stadyum güvenlik çemberi** ile korunur:
1. **Kimlik Doğrulama ve Hız Sınırı (`api/app.py`, `modules/security.py`)**: `POST /api/v1/auth/token` üzerinden sabit zamanlı (`hmac.compare_digest`) `X-API-Key` kontrolüyle kısa ömürlü JWT bileti üretir.
   Bellek sınırlandırmalı `TokenBucketRateLimiter` brute-force ve DDoS saldırılarını `429 Too Many Requests` ile durdurur.
2. **Derin Video Denetçisi (`validate_video_file`)**: Yalnızca `.mp4` uzantısına bakmaz; ilk baytlardaki konteyner imzasını (`magic bytes`), 4 GB boyut sınırını, `<= 4K` çözünürlüğü (decompression bomb koruması), FPS ve süre (`<= 125 dk`) sınırlarını doğrular; yüklenen dosyayı izole `UUIDv4` klasörüne kaydeder.
3. **Model Bütünlük ve Pickle/RCE Koruması (`validate_model_weight_path`)**: PyTorch `.pt` dosyalarının yalnızca salt-okunur `models/` dizininden (veya resmi YOLO temel isimlerinden) yüklenmesini zorunlu kılar; `FOOTBALL_MODEL_SHA256` tanımlanırsa yükleme öncesi SHA-256 doğrulaması yapar.
4. **Enjeksiyon ve Yol Gezinme Kalkanı (`resolve_safe_path`, `validate_sql_identifier`, `sanitize_csv_value`)**: SQLite tablo adlarını doğrular, parametrik SQL sorguları kullanır, CSV dışa aktarımlarında Excel formül enjeksiyonunu (`=`, `+`, `-`, `@`) zararsızlaştırır ve `../../` dizin gezinme girişimlerini engeller.
5. **Konteyner İzolasyonu ve Sıralı GPU Kuyruğu (`Dockerfile`, `docker-compose.yml`)**: `BoundedJobQueue` ekran kartının aynı anda fazla istekle kilitlenmesini önler; Docker konteyneri yetkisiz `analyzer` kullanıcısı (`UID 10001`), salt-okunur kök dosya sistemi (`read_only: true`), `cap_drop: [ALL]` ve `8 GB` RAM / `256` PID sınırıyla çalışır. Büyük video yüklemelerinde RAM tüketimini ve `/tmp` taşmasını önlemek için `TMPDIR` yazılabilir veri hacmine (`/app/data/jobs/_tmp`) yönlendirilir. CUDA GPU hızlandırması için ana makinede **NVIDIA Container Toolkit** kurulu olmalıdır (GPU yoksa `detector.py` otomatik olarak CPU'ya düşer).

Yerel üretim API'sini güvenli şekilde başlatmak için:
```powershell
$env:FOOTBALL_API_KEY = "GUCLU_RASTGELE_API_ANAHTARINIZ"
$env:FOOTBALL_JWT_SECRET = "EN_AZ_32_KARAKTERLIK_GIZLI_ANAHTAR"
$env:YOLO_CONFIG_DIR = "$env:TEMP\ultralytics_cfg"
uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Docker Compose ile başlatma:
- **CPU modu (varsayılan, GPU gerektirmez):**
  ```bash
  docker compose up
  ```
- **GPU modu (ana makinede NVIDIA Container Toolkit gerekir):**
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
  ```
