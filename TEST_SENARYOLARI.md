# Test Senaryoları (2026-11-26 durumu)

> **Son çalıştırma: 2026-11-26.** Geniş/taktik klip (10 dk, 50 fps), CUDA'lı makine, main `61b9f8e`.
> "Sizde" = göz veya insan etiketi gerekiyor; "Atlandı" = bu ortamda girdi yok.

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
| K-01 | Bağımlılıklar | `pip install -r requirements.txt` | Hatasız biter || **Geçti.** `pip check`: bozuk bağımlılık yok |
| K-02 | Model dosyaları | `python utils/download_models.py` | `models/` altında tespit ve saha modelleri bulunur ya da indirme yönergesi basılır; anahtar/token ekrana yazılmaz || **Kaldı (küçük).** Modeller `models/` altında olduğu hâlde betik "indirme adresi tanımlı değil" diyip hata koduyla (1) çıkıyor; önce dosyanın varlığına bakmıyor. Anahtar ekrana basılmıyor |
| K-03 | Yardım çıktısı | `python main.py --help` | Tüm seçenekler Türkçe açıklamayla listelenir || **Geçti.** 16 seçenek, Türkçe açıklamalar |

## 2. Video işleme (CLI)

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| P-01 | Kısa deneme | `python main.py --input WIDE --max-frames 300 --export-dir $env:TEMP\t1` | Hatasız biter; `$env:TEMP\t1` altında `tracking_data.csv`, `tracking_data.db`, `match_analytics_summary.json` oluşur || **Geçti.** 300 kare 115 sn; tüm dosyalar oluştu |
| P-02 | Kutulu çıktı videosu | `python main.py --input WIDE --max-frames 300 --output $env:TEMP\t2.mp4 --export-dir $env:TEMP\t2` | Video açılır. Oyuncu, kaleci, hakem ve top kutuları ile kuş bakışı küçük harita görünür || **Geçti.** Kutular, takım renkleri, top ve kuş bakışı harita doğru (250. kare incelendi) |
| P-03 | Küçük harita kapalı | P-02 + `--no-minimap` | Videoda küçük harita yok || **Geçti.** Harita yok, kutular var |
| P-04 | Hızlı mod | `--no-video` ile WIDE'ın tamamı | Çıktı videosu oluşmaz; veri dosyaları oluşur; süre P-02'ye göre belirgin kısadır || **Geçti.** 30.000 kare 76,6 dk (GPU başka işlerle paylaşılırken); tüm veri dosyaları oluştu |
| P-05 | CSV / SQL atlama | `--no-csv` sonra `--no-sql` | İlkinde `tracking_data.csv`, ikincisinde `tracking_data.db` oluşmaz || **Geçti.** `--no-csv`: yalnız `.db`; `--no-sql`: yalnız `.csv` |
| P-06 | Analiz atlama | `--no-analytics --no-ml` | `match_analytics_summary.json` ve `ml_reports/` oluşmaz || **Geçti.** Analiz ve ML çıktısı yok. Not: `player_distance_summary.csv` yine yazılıyor (işleme adımında üretiliyor) |
| P-07 | Takipçi seçimi | `--tracker` için desteklenen değerleri ayrı ayrı dene | Her biri hatasız biter; geçersiz değer anlaşılır hata verir || **Geçti.** deepsort, bytetrack, fast, iou, kalman_iou hatasız. `bogus` değeri izin verilen listeyle reddedildi |
| P-08 | Bozuk / olmayan dosya | `--input yok.mp4` | Çökme izi (traceback) yerine anlaşılır Türkçe hata || **Geçti.** `[HATA] Giriş video dosyası bulunamadı`, çıkış kodu 1, çökme izi yok |
| P-09 | Link girişi | `python main.py --input "https://...mp4" --no-video --max-frames 200 --export-dir $env:TEMP\t9` | Video indirilip işlenir; desteklenmeyen uzantı veya adres reddedilir || **Atlandı.** Dış siteden indirme gerektiriyor (izin istenmedi). Uzantı reddi otomatik testlerde var |

## 3. Otomatik saha tanıma (Faz A2)

Hedefler, ana kamera karelerinde:

| Ölçüt | Geniş kamera | TV |
|---|---|---|
| Ortanca hata | < 0,5 m | < 1 m |
| Geçerli tablo oranı | ≥ %90 | ≥ %90 |
| Duran nesne hızı | < 1 km/sa | < 1 km/sa |

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| A-01 | Kayıt dosyası | P-04'ü WIDE ile çalıştır | `homography_log.csv` ve `calibration_summary.json` oluşur; kamera hareketine uygun profil (`fixed` veya `broadcast`) seçilir | **Geçti (belge düzeltildi).** Dosyalar oluştu; 46. karede `broadcast` profiline geçiş doğru davranıştır. Ölçüm (`tactical_10min.mp4`, ilk 6 sn): kare 46'da net kayma `67,1 px` (`0,035w`), kare 299'da `619,3 px` (`0,323w`) ve ölçek değişimi `0,172` (`PROFILE_FIXED_MAX_PAN_RATIO = 0,05`, `PROFILE_FIXED_MAX_SCALE_CHANGE = 0,03` eşiklerini aştığı için `fixed` yerine `broadcast` seçilir; `fixed` profili yalnızca pan/zoom yapmayan sabit kameralar içindir) |
| A-02 | İsabet ölçümü (WIDE) | README "Kalibrasyon isabetini ölçme" adımları: `calibrate_homography.py --label-mode` ile etiketle, `evaluate_homography.py` ile ölç | Ortanca hata < 0,5 m (bugün 0,28 m) | **KALDI (gerileme).** Tam 10 dk: ortanca 0,60 m, %90 2,31 m. Sebep doğrulandı: #9'daki `PITCH_CAL_BLEND_MAX_SPEED_MPS = 0.25`. İlk 11.001 karede aynı 5 etiketli kare (43 nokta): main 0,68 m; hız sınırı 0 → 0,31 m; #9 öncesi → 0,37 m. #20 maske ve #21 bulanıklaştırma suçsuz. Düzeltme bekliyor (hedef: < 0,5 m ve duran oyuncu < 1 km/sa birlikte) |
| A-03 | İsabet ölçümü (TV) | A-02'yi TV ile tekrarla | **Bilinen sınır:** bugün 4,6 m, hedef < 1 m. Kalması beklenir; sonucu not et || **Atlandı.** TV klibi bu ortamda yok (bilinen sınır: 4,6 m) |
| A-04 | Yanlış tablo reddi | Kamera hızlı dönerken veya yakın çekimde `homography_log.csv`'ye bak | Doğrulanamayan karelerde yeni tablo kabul edilmez, önceki tablo korunur; haritada oyuncular sahadan fırlamaz || **Sizde.** Videoda göz kontrolü gerekiyor. Kayıtta: 1967 model çağrısı, 179 aykırı tablo reddedildi |
| A-05 | Duran nesne | Çıktı videosunda çizgi üstünde duran hakem veya oyuncuyu 1 sn izle | Haritadaki nokta titremez, kaymaz || **Sizde.** Göz kontrolü gerekiyor |
| A-06 | Otomatik kalibrasyon kapalı | `--no-auto-calibrate` | Manuel tablo varsa o kullanılır, yoksa piksel modunda çalışır; çökme yok || **Geçti.** Kalibrasyonsuz çalıştı, çökme yok; `calibration_summary.json` yazılmadı (beklenen) |
| A-07 | Koyu / soluk çim | Gece maçı veya kış çimi klibiyle A-01 | Saha maskesi oyun alanını kapsar; kalibrasyon hiç bulunamıyorsa sebebi kayda yazılır || **Atlandı.** Gece/kış çimi klibi yok. Otomatik test var; normal çimde maske eskisiyle birebir aynı (20 kare) |

## 4. Maç analizi (pas, sahiplik, takım biçimi)

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| M-01 | Videosuz yeniden analiz | `python modules/match_analytics.py --input <bir tracking_data.db kopyası>` | Video tekrar işlenmeden özet üretilir || **Geçti.** 10 dk'lık maç 10 sn'de yeniden analiz edildi |
| M-02 | Pas mantığı | Özetteki birkaç pası videoda izle | Pas veren ve alan aynı takımdaysa "tamamlanan pas", değilse "top kaybı" || **Sizde.** Videoda izleme gerekiyor |
| M-03 | Kısa rakip teması | Topun bir rakibin dibinden geçtiği bir pası bul | Tek pas sayılır; iki sahte top kaybı yazılmaz || **Sizde.** Videoda izleme gerekiyor (otomatik test var) |
| M-04 | Kale arkasındaki top | Kale ağında yedek top görünen bir an | Sahiplik zaman çizelgesine girmez || **Sizde.** Videoda izleme gerekiyor (otomatik test var) |
| M-05 | Takım biçimi kapsaması | `match_analytics_summary.json` içindeki takım biçimi | Her metrikte kapsama etiketi var. Geniş kamera genelde "high", TV "medium" || **Geçti (not).** Kapsama etiketi var: Team A "high", ortanca 11 oyuncu. Not: takım derinliği 106 m çıkıyor (tüm maç boyunca uç konumlar); tek başına anlamlı değil |
| M-06 | Isı haritası ve mesafe | `player_distance_summary.csv` ve ısı haritaları | Mesafe ve hız değerleri futbol için makul (oyuncu başına < 37 km/sa) || **Kısmen.** Ortanca azami hız 20,5 km/sa (makul), ama 449 izin 107'si 37–38 km/sa sınırına dayanıyor. Muhtemelen konum titremesi veya kimlik karışması; sınır (38) değeri gizliyor |

## 5. ML analizi

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| ML-01 | Rapor üretimi | `python modules/ml_analyzer.py --input <tracking_data.db kopyası>` | `ml_reports/` altında README'deki 10 dosya oluşur; rapor Türkçe || **Geçti.** 10 dosyanın hepsi oluştu, 34 sn; rapor Türkçe |
| ML-02 | Az veri | Çok kısa bir klip (≤ 300 kare) | Model eğitimi atlanır veya sentetik veriye düşülür ve bu raporda açıkça yazar; çökme yok || **Geçti.** 300 karede "Hesaplanamadı – yetersiz veri (1/30 pas)" açıkça yazıyor |
| ML-03 | Analizörler tutarlı | Aynı veride `ml_pass_predictions.csv` ile `match_analytics_summary.json` paslarını karşılaştır | Pas zamanları büyük ölçüde örtüşür (aynı ortak kurallar kullanılıyor) || **Geçti.** Ana analizörün 143 olayının %98,6'sının ML'de 1 sn içinde karşılığı var |

## 6. Faz V: pas ve sahiplik doğrulaması

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| V-01 | Etiketleme aracı | `utils/label_events.html`'i tarayıcıda aç, videoyu seç, birkaç pas ve kayıp işaretle, "CSV indir" | `gt_events.csv` ve `gt_possession.csv` iner; sütunlar `time_sec,event_type,team` ve `start_sec,end_sec,team` || **Sizde.** Tarayıcıda yerel video seçmek gerekiyor |
| V-02 | Kısayollar | Araçta klavye kısayollarıyla işaretleme | Video odaktayken de kısayollar çalışır || **Sizde.** |
| V-03 | Karşılaştırma | README'deki `validate_events.py` komutu, `--json-out $env:TEMP\v.json` | Kesinlik, duyarlılık, F1 ve sahiplik uyumu yazılır; takım harfleri ters etiketlenmişse otomatik eşlenir || **Geçti.** JSON çıktısı; 0–120 sn'de pas F1 0,71 (yapay zekâ etiketiyle) |
| V-04 | **İnsan etiketiyle ölçüm (WIDE, görülmemiş bölüm)** | 5–10 dk etiketle, V-03'ü çalıştır | Hedef yok, gerçek değeri kaydet. Yapay zekâ etiketiyle bugün pas F1 ~0,70, sahiplik ~%84, top kaybı F1 0,20–0,40 || **Sizde.** İnsan etiketi gerekiyor |
| V-05 | **İnsan etiketiyle ölçüm (TV)** | V-04'ü TV ile tekrarla | Değeri kaydet; TV kalibrasyonu zayıf olduğu için düşük çıkması beklenir || **Sizde.** İnsan etiketi ve TV klibi gerekiyor |

## 7. API ve güvenlik

Başlatma:
```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| S-01 | Sağlık kontrolü | `GET /api/v1/health` | 200, kuyruk bilgisi || **Geçti.** 200, kuyruk bilgisi |
| S-02 | Kimlik doğrulama | `POST /api/v1/auth/token`, önce yanlış sonra doğru `X-API-Key` ile | Yanlış: 401. Doğru: geçerli token || **Geçti.** Yanlış anahtar 401, doğru anahtar token |
| S-03 | Yetkisiz erişim | Token olmadan `GET /api/v1/jobs/<uuid>` | 401 || **Geçti.** 401 |
| S-04 | İş gönderme | Token ile kısa bir `.mp4` yükle | 202, `job_id`; durum `queued` → `running` → `completed` || **Geçti.** 202 → running → completed, 19 çıktı dosyası (5 sn'lik klip) |
| S-05 | Kötü dosya | `.exe` veya video olmayan dosya yükle | 400, anlaşılır mesaj; iş klasörü kalmaz || **Geçti.** `.exe` ve sahte `.mp4` 400; iş klasörü kalmadı |
| S-06 | Boyut / süre sınırı | Sınırdan büyük video yükle | 413 veya 400 || **Geçti.** 413 (sınır 3 MB'a çekilerek denendi) |
| S-07 | Çıktı indirme | `GET .../artifacts/match_analytics_summary.json` | Dosya iner. `../` ile yol denemesi veya beyaz listede olmayan ad: 403 || **Geçti.** Çıktı indi. `../` yol denemeleri, `input.mp4` ve listede olmayan adlar 403, geçersiz kimlik 400 |
| S-08 | Hız sınırı | Dakikada sınırdan fazla istek gönder | 429 || **Geçti.** 40 istekte 36×200, 4×429 |
| S-09 | Kuyruk dolu | Kuyruk kapasitesinden fazla iş gönder | 429 "kuyruk dolu" || **Geçti.** 4. işte 429 "kuyruk dolu" |
| S-10 | Eski iş temizliği | `API_JOB_TTL_SECONDS`'ı kısalt (ör. 60), bir iş bitir, bekle, yeni iş gönder | Eski işin kaydı (404) ve `data/jobs/<uuid>` klasörü silinir; çalışan işe dokunulmaz || **Geçti.** Süre 60 sn'ye çekildi; süresi dolan iş 404 ve klasörü silindi, kuyruktaki işler duruyor |
| S-11 | Docker | `docker compose up` (GPU için `-f docker-compose.gpu.yml` eklenir) | Konteyner yetkisiz kullanıcıyla ayağa kalkar; S-01 çalışır || **Atlandı.** Docker kurulu ama çalışmıyor. `docker compose config` geçerli |

## 8. Regresyon

| No | Senaryo | Adımlar | Beklenen | Sonuç |
|---|---|---|---|---|
| R-01 | Otomatik testler | `python -m pytest -q` (repo kökünde; `pytest.ini` ile `tests/` çalışır) | Tamamı geçer. Tam paket uzun sürebilir; CI'da da çalışır || **Geçti.** 346 test, 18 dk 46 sn |
| R-02 | Örnek veri korunuyor | R-01 öncesi ve sonrası `data/exports/tracking_data.csv` ve `.db` tarihlerine bak | Değişmemiş || **Geçti.** Örnek veri dosyalarının tarihleri değişmedi |

---

## Bilinen sınırlar (kalması beklenen senaryolar)
- **A-03:** TV kalibrasyonu hedefin altında. Daha çok etiketli TV verisiyle ince ayar gerekiyor.
- **V-04 / V-05:** Top kaybı tespiti zayıf. Faz V henüz insan etiketiyle doğrulanmadı.
- Takip izleri kısa (10 dk'da 439 iz, ortanca 1,5 sn). Oyuncu bazlı tutarlılık beklenmemeli.
