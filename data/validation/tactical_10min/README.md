# Doğrulama etiketleri: geniş/taktik kamera, 10 dk

Video repoda yok (50 fps, 1920x1080, 30.000 kare). Etiketler bu klibe aittir ve
`utils/benchmark_validation.py` ile kullanılır.

| Dosya | İçerik | Kaynak |
|---|---|---|
| `a2_labels.csv` | 12 karede 105 saha noktasının piksel konumu (`frame_idx,landmark_name,pixel_x,pixel_y`) | Elle etiket (`utils/calibrate_homography.py --label-mode`) |
| `gt_events_000_120.csv`, `gt_possession_000_120.csv` | 0–120 sn pas / top kaybı ve sahiplik | Yapay zekâ etiketi (insan değil); düzeltmelerin ayarlandığı aralık |
| `gt_events_120_240.csv`, `gt_possession_120_240.csv` | 120–240 sn | Yapay zekâ etiketi; kontrol aralığı |

Faz V etiketleri insan etiketi değildir; sonuçlar kaba gösterge olarak okunmalı
(bkz. `PROJE_DURUMU.md`). İnsan etiketi gelince aynı adlarla değiştirilebilir.
