# Test Senaryoları (2026-11-26 durumu)

Bu belge projenin bugünkü özelliklerini elle doğrulamak içindir. Her senaryoda girdi,
adımlar ve beklenen sonuç var; "Sonuç" sütununa **Geçti / Kaldı / Atlandı** ve kısa not
yazılır. Otomatik testler ayrıca `tests/` altında (bkz. senaryo R-01).

**Genel kurallar**
- Çıktılar her zaman `$env:TEMP` altına yazılır. `data/exports/tracking_data.csv` ve
  `tracking_data.db` örnek dosyalarının üzerine asla yazılmaz.
- Komutlardan önce bir kez:
  ```powershell
  $env:YOLO_CONFIG_DIR="$env:TEMP\ultralytics_cfg"
  ```
- Örnek girdiler:
  - **WIDE:** Geniş/taktik kamera klibi, en az 2 dk.
  - **TV:** TV yayını klibi.
  - Kendi videolarınızı kullanın; videolar repoya eklenmez.

---

## 1. Kurulum

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| K-01 | Bağımlılıklar | `pip install -r requirements.txt` | Hatasız biter | |
| K-02 | Model dosyaları | `python utils/download_models.py` | `models/` altında tespit ve saha modelleri bulunur ya da indirme yönergesi basılır; anahtar/token ekrana yazılmaz | |
| K-03 | Yardım çıktısı | `python main.py --help` | Tüm seçenekler Türkçe açıklamayla listelenir | |

## 2. Video işleme (CLI)

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| P-01 | Kısa deneme | `python main.py --input WIDE --max-frames 300 --export-dir $env:TEMP\t1` | Hatasız biter; `$env:TEMP\t1` altında `tracking_data.csv`, `tracking_data.db`, `match_analytics_summary.json` oluşur | |
| P-02 | Kutulu çıktı videosu | `python main.py --input WIDE --max-frames 300 --output $env:TEMP\t2.mp4 --export-dir $env:TEMP\t2` | Video açılır. Oyuncu, kaleci, hakem ve top kutuları ile kuş bakışı küçük harita görünür | |
| P-03 | Küçük harita kapalı | P-02 + `--no-minimap` | Videoda küçük harita yok | |
| P-04 | Hızlı mod | `--no-video` ile WIDE'ın tamamı | Çıktı videosu oluşmaz; veri dosyaları oluşur; süre P-02'ye göre belirgin kısadır | |
| P-05 | CSV / SQL atlama | `--no-csv` sonra `--no-sql` | İlkinde `tracking_data.csv`, ikincisinde `tracking_data.db` oluşmaz | |
| P-06 | Analiz atlama | `--no-analytics --no-ml` | `match_analytics_summary.json` ve `ml_reports/` oluşmaz | |
| P-07 | Takipçi seçimi | `--tracker` için desteklenen değerleri ayrı ayrı dene | Her biri hatasız biter; geçersiz değer anlaşılır hata verir | |
| P-08 | Bozuk / olmayan dosya | `--input yok.mp4` | Çökme izi (traceback) yerine anlaşılır Türkçe hata | |
| P-09 | Link girişi | `python main.py --input "https://...mp4" --no-video --max-frames 200 --export-dir $env:TEMP\t9` | Video indirilip işlenir; desteklenmeyen uzantı veya adres reddedilir | |

## 3. Otomatik saha tanıma (Faz A2)

Hedefler, ana kamera karelerinde:

| Ölçüt | Geniş kamera | TV |
|---|---|---|
| Ortanca hata | < 0,5 m | < 1 m |
| Geçerli tablo oranı | ≥ %90 | ≥ %90 |
| Duran nesne hızı | < 1 km/sa | < 1 km/sa |

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| A-01 | Kayıt dosyası | P-04'ü WIDE ile çalıştır | `homography_log.csv` ve `calibration_summary.json` oluşur; profil "geniş" olarak seçilir | |
| A-02 | İsabet ölçümü (WIDE) | README "Kalibrasyon isabetini ölçme" adımları: `calibrate_homography.py --label-mode` ile etiketle, `evaluate_homography.py` ile ölç | Ortanca hata < 0,5 m (bugün 0,28 m) | |
| A-03 | İsabet ölçümü (TV) | A-02'yi TV ile tekrarla | **Bilinen sınır:** bugün 4,6 m, hedef < 1 m. Kalması beklenir; sonucu not et | |
| A-04 | Yanlış tablo reddi | Kamera hızlı dönerken veya yakın çekimde `homography_log.csv`'ye bak | Doğrulanamayan karelerde yeni tablo kabul edilmez, önceki tablo korunur; haritada oyuncular sahadan fırlamaz | |
| A-05 | Duran nesne | Çıktı videosunda çizgi üstünde duran hakem veya oyuncuyu 1 sn izle | Haritadaki nokta titremez, kaymaz | |
| A-06 | Otomatik kalibrasyon kapalı | `--no-auto-calibrate` | Manuel tablo varsa o kullanılır, yoksa piksel modunda çalışır; çökme yok | |
| A-07 | Koyu / soluk çim | Gece maçı veya kış çimi klibiyle A-01 | Saha maskesi oyun alanını kapsar; kalibrasyon hiç bulunamıyorsa sebebi kayda yazılır | |

## 4. Maç analizi (pas, sahiplik, takım biçimi)

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| M-01 | Videosuz yeniden analiz | `python modules/match_analytics.py --input <bir tracking_data.db kopyası>` | Video tekrar işlenmeden özet üretilir | |
| M-02 | Pas mantığı | Özetteki birkaç pası videoda izle | Pas veren ve alan aynı takımdaysa "tamamlanan pas", değilse "top kaybı" | |
| M-03 | Kısa rakip teması | Topun bir rakibin dibinden geçtiği bir pası bul | Tek pas sayılır; iki sahte top kaybı yazılmaz | |
| M-04 | Kale arkasındaki top | Kale ağında yedek top görünen bir an | Sahiplik zaman çizelgesine girmez | |
| M-05 | Takım biçimi kapsaması | `match_analytics_summary.json` içindeki takım biçimi | Her metrikte kapsama etiketi var. Geniş kamera genelde "high", TV "medium" | |
| M-06 | Isı haritası ve mesafe | `player_distance_summary.csv` ve ısı haritaları | Mesafe ve hız değerleri futbol için makul (oyuncu başına < 37 km/sa) | |

## 5. ML analizi

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| ML-01 | Rapor üretimi | `python modules/ml_analyzer.py --input <tracking_data.db kopyası>` | `ml_reports/` altında README'deki 10 dosya oluşur; rapor Türkçe | |
| ML-02 | Az veri | Çok kısa bir klip (≤ 300 kare) | Model eğitimi atlanır veya sentetik veriye düşülür ve bu raporda açıkça yazar; çökme yok | |
| ML-03 | Analizörler tutarlı | Aynı veride `ml_pass_predictions.csv` ile `match_analytics_summary.json` paslarını karşılaştır | Pas zamanları büyük ölçüde örtüşür (aynı ortak kurallar kullanılıyor) | |

## 6. Faz V: pas ve sahiplik doğrulaması

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| V-01 | Etiketleme aracı | `utils/label_events.html`'i tarayıcıda aç, videoyu seç, birkaç pas ve kayıp işaretle, "CSV indir" | `gt_events.csv` ve `gt_possession.csv` iner; sütunlar `time_sec,event_type,team` ve `start_sec,end_sec,team` | |
| V-02 | Kısayollar | Araçta klavye kısayollarıyla işaretleme | Video odaktayken de kısayollar çalışır | |
| V-03 | Karşılaştırma | README'deki `validate_events.py` komutu, `--json-out $env:TEMP\v.json` | Kesinlik, duyarlılık, F1 ve sahiplik uyumu yazılır; takım harfleri ters etiketlenmişse otomatik eşlenir | |
| V-04 | **İnsan etiketiyle ölçüm (WIDE, görülmemiş bölüm)** | 5–10 dk etiketle, V-03'ü çalıştır | Hedef yok, gerçek değeri kaydet. Yapay zekâ etiketiyle bugün pas F1 ~0,70, sahiplik ~%84, top kaybı F1 0,20–0,40 | |
| V-05 | **İnsan etiketiyle ölçüm (TV)** | V-04'ü TV ile tekrarla | Değeri kaydet; TV kalibrasyonu zayıf olduğu için düşük çıkması beklenir | |

## 7. API ve güvenlik

Başlatma:
```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| S-01 | Sağlık kontrolü | `GET /api/v1/health` | 200, kuyruk bilgisi | |
| S-02 | Kimlik doğrulama | `POST /api/v1/auth/token`, önce yanlış sonra doğru `X-API-Key` ile | Yanlış: 401. Doğru: geçerli token | |
| S-03 | Yetkisiz erişim | Token olmadan `GET /api/v1/jobs/<uuid>` | 401 | |
| S-04 | İş gönderme | Token ile kısa bir `.mp4` yükle | 202, `job_id`; durum `queued` → `running` → `completed` | |
| S-05 | Kötü dosya | `.exe` veya video olmayan dosya yükle | 400, anlaşılır mesaj; iş klasörü kalmaz | |
| S-06 | Boyut / süre sınırı | Sınırdan büyük video yükle | 413 veya 400 | |
| S-07 | Çıktı indirme | `GET .../artifacts/match_analytics_summary.json` | Dosya iner. `../` ile yol denemesi veya beyaz listede olmayan ad: 403 | |
| S-08 | Hız sınırı | Dakikada sınırdan fazla istek gönder | 429 | |
| S-09 | Kuyruk dolu | Kuyruk kapasitesinden fazla iş gönder | 429 "kuyruk dolu" | |
| S-10 | Eski iş temizliği | `API_JOB_TTL_SECONDS`'ı kısalt (ör. 60), bir iş bitir, bekle, yeni iş gönder | Eski işin kaydı (404) ve `data/jobs/<uuid>` klasörü silinir; çalışan işe dokunulmaz | |
| S-11 | Docker | `docker compose up` (GPU için `-f docker-compose.gpu.yml` eklenir) | Konteyner yetkisiz kullanıcıyla ayağa kalkar; S-01 çalışır | |

## 8. Regresyon

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| R-01 | Otomatik testler | `python -m pytest -q` (repo kökünde; `pytest.ini` ile `tests/` çalışır) | Tamamı geçer. Tam paket uzun sürebilir; CI'da da çalışır | |
| R-02 | Örnek veri korunuyor | R-01 öncesi ve sonrası `data/exports/tracking_data.csv` ve `.db` tarihlerine bak | Değişmemiş | |

---

## Bilinen sınırlar (kalması beklenen senaryolar)
- **A-03:** TV kalibrasyonu hedefin altında. Daha çok etiketli TV verisiyle ince ayar gerekiyor.
- **V-04 / V-05:** Top kaybı tespiti zayıf. Faz V henüz insan etiketiyle doğrulanmadı.
- Takip izleri kısa (10 dk'da 439 iz, ortanca 1,5 sn). Oyuncu bazlı tutarlılık beklenmemeli.
