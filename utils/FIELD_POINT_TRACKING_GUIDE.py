"""
FUTBOL SAHASI NOKTASI İZLEME (FIELD POINT TRACKING) - KKULLANIM KILAVUZU
==========================================================================

Bu kılavuz, futbol sahası köşe noktalarını otomatik olarak tespit etme ve
sahne kesintilerine (scene cut) rağmen noktaların sabit kalmasını sağlama
sistemini açıklar.

═══════════════════════════════════════════════════════════════════════════

1. GENEL BAKIŞ
══════════════

Futbol maçı videosunda kamera hareketi, yakınlaştırma (zoom) ve sahne
kesintileri nedeniyle saha çizgileri ve noktaları "kayabilir" - yani piksel
koordinatlarında dengesiz hale gelir.

Bu sistem:
✓ Futbol sahası köşelerini VİDEO BAŞINDA otomatik olarak tespit eder
✓ Tespit edilen noktaları SABİT tutar (diğer karelerde YENIDEN tespit EDİLMEZ)
✓ Sahne kesintilerini (scene cut) algılar
✓ Algılanan kesintilerde noktaları YENİDEN tespit eder (kayması önler)
✓ Tüm noktaları JSON ve pickle formatında kaydeder

SONUÇ: Saha noktaları video boyunca istikrarlı kalır, minimap ve taktik
görünüm daha güvenilir hale gelir.

═══════════════════════════════════════════════════════════════════════════

2. FUTBOL SAHASI NOKTASI TÜRLERİ
════════════════════════════════

Standart futbol sahası için tespit edilen noktalar:

┌─────────────────────────────────────────────────────────┐
│                    SAHA ÖZETİ (TOP GÖRÜNÜŞ)             │
│                                                          │
│   (TL)─────(CT)─────(TR)                                │
│     └─────────┘ └─────────┘                             │
│   (PT-TL)   center_circle    (PT-TR)                   │
│     │       /        \          │                       │
│     │      /          \         │                       │
│    (PL)    center_circle      (PR)                     │
│     │      \          /         │                       │
│     │       \        /          │                       │
│   (PT-BL)                   (PT-BR)                    │
│     └─────────┘ └─────────┘                             │
│   (BL)─────(CB)─────(BR)                                │
│                                                          │
│  TL/TR/BR/BL     = Köşe bayrağı (corner)               │
│  CT/CB           = Top çizgisi ortası                   │
│  PT-TL/TR/BL/BR  = Ceza sahası köşesi                  │
│  PL/PR           = Ceza sahası kenarı                  │
│  CL              = Merkez dairesi                       │
└─────────────────────────────────────────────────────────┘

Tespit edilen nokta türleri:
  "corner"        : 4 köşe bayrağı
  "goal_line"     : 2 gol çizgisi uç noktası
  "penalty_corner": 4 ceza sahası köşesi
  "penalty_area"  : Ceza sahası kenarları
  "center_circle" : Orta daire merkezi
  "center_line"   : Orta çizgi işaretleri

═══════════════════════════════════════════════════════════════════════════

3. KULLANIM
═══════════

A) OTOMATIK AYIK (main pipeline'ın içinde)
──────────────────────────────────────────

main.py çalıştırdığınızda, saha noktaları otomatik olarak tespit edilir:

    python main.py --input data/input_videos/match.mp4 \\
                   --output data/output_videos/match_annotated.mp4

ÇIKTI:
  • data/output_videos/match_annotated.mp4 → Noktaların çizili video
  • data/exports/field_points.json         → Tespit edilen noktalar (JSON)
  • data/exports/field_points.pkl          → Tespit edilen noktalar (pickle)
  • data/exports/field_point_detection_history.json → Tespit geçmişi

B) STANDALONE TESPIT (hızlı test için)
──────────────────────────────────────

Sadece saha noktalarını tespit etmek istiyorsanız:

    python utils/detect_field_points.py \\
        --input data/input_videos/match.mp4 \\
        --output data/exports/field_points

Tek bir frame'den tespit:

    python utils/detect_field_points.py \\
        --input data/frames/sample_frame_0050.png \\
        --output data/exports

OPTIONS:
  --input FILE              : Video (.mp4) veya frame (.png/.jpg) dosyası
  --output DIR              : Çıktı dizini
  --no-annotated-video      : Annotated video'yu kaydetme
  --quiet                   : Terminal çıktısını kısalt

═══════════════════════════════════════════════════════════════════════════

4. ÇIKTI DOSYALARI HAKKINDA
────────────────────────────

A) field_points.json
────────────────────

Tespit edilen saha noktalarını JSON formatında saklar. Yapısı:

    {
      "field_points": {
        "corner_tl": {
          "point_id": "corner_tl",
          "point_type": "corner",
          "pixel_x": 123.45,
          "pixel_y": 67.89,
          "pitch_x": null,          ← Kalibrasyon yapılmışsa dolu
          "pitch_y": null,          ← Kalibrasyon yapılmışsa dolu
          "confidence": 0.95,
          "frame_detected": 0,
          "is_stable": true
        },
        ...
      },
      "metadata": {
        "pitch_length_m": 105.0,
        "pitch_width_m": 68.0,
        "total_detections": 42
      }
    }

Alanların anlamı:
  • point_id: Noktanın benzersiz tanımlayıcısı
  • point_type: Nokta türü (corner, goal_line, penalty_corner, vb)
  • pixel_x, pixel_y: Piksel koordinatları (video frame'de)
  • pitch_x, pitch_y: Saha koordinatları metre cinsinden (kalibrasyon varsa)
  • confidence: Harris corner detection güveni (0-1)
  • frame_detected: İlk tespit kare numarası
  • is_stable: Sahne kesintisi sonrası yeniden tespit edildiyse False → True

B) field_points.pkl
────────────────────

Python pickle formatında binary veri. Python kodu içinden yükleme:

    import pickle
    with open('data/exports/field_points.pkl', 'rb') as f:
        field_points = pickle.load(f)
        for point_id, point_obj in field_points.items():
            print(f"{point_id}: ({point_obj.pixel_x}, {point_obj.pixel_y})")

C) field_point_detection_history.json
──────────────────────────────────────

Her kare için tespit edilen noktaları kaydeder (DEBUG amaçlı).
Dosya çok büyük olabilir; sadece gerekirse kullanın.

═══════════════════════════════════════════════════════════════════════════

5. NASIL ÇALIŞIYOR? (TEKNIK DETAYLAR)
═══════════════════════════════════════

1) İLK KARE TESPİTİ (Calibration Frame)
   ────────────────────────────────────
   
   • config.CALIBRATION_FRAME_INDEX numaralı kare (varsayılan: 0) işlenir
   • Harris corner detection → köşe aday noktaları
   • Canny edge + Hough line detection → saha çizgilerinin kesişme noktaları
   • Tüm adaylar birleştirilip futbol sahası öğelerine eşleştirilir
   • Tespit edilen noktalar self.field_points'e kaydedilir

2) SAHNE KESİNTİSİ ALGILAMA (Scene Cut Detection)
   ──────────────────────────────────────────────
   
   • Her kare için histogram hesaplanır
   • Mevcut frame'in histogramı önceki kare'nin histogramı ile karşılaştırılır
   • Bhattacharyya distance > 0.3 eşiğini aşarsa → SCENE CUT
   • Sahne kesintileri = usta çekimi (master shot), yardımcı kamera, grafik
   
   Örnek:
   ┌─────────────────┬──────────────┐
   │  Frame 0-100   │  Scene Cut   │  Frame 101-200  │ Scene Cut │ ...
   │ Normal çekim   │ ⚠ Kamera    │ Farklı açı    │ ⚠ Grafik │
   └─────────────────┴──────────────┴─────────────────┴──────────┴─...
                     │ Noktalar YENİDEN        │ Noktalar YENİDEN
                     │ TESPİT EDİLİR          │ TESPİT EDİLİR

3) SAHNE KESİNTİSİNDE YENİDEN TESPİT
   ──────────────────────────────────
   
   • Sahne kesintisi algılandı → detect_field_points_in_frame() yeniden çağrılır
   • Yeni tespit edilen noktalar self.field_points'e birleştirilir
   • is_stable = False işaretlenir (sonra True'ya dönüştürülebilir)
   • SONUÇ: Noktaların yeni sahnede "kayması" engellenir

4) HOMAGRAFİ DÖNÜŞTÜRME (Homography Transformation)
   ──────────────────────────────────────────────
   
   • Eğer utils/calibrate_homography.py ile kalibrasyon yapılmışsa
   • Piksel koordinatları → saha koordinatları (metre)
   • PerspectiveTransformer.pixel_to_pitch() kullanılır
   • CSV/JSON çıktısı hem piksel hem saha koordinatlarını içerir

═══════════════════════════════════════════════════════════════════════════

6. AYARLAR (CONFIG)
═══════════════════

config.py içinde ayarlanabilir parametreler:

    CALIBRATION_FRAME_INDEX = 0
        → Hangi kare'deki kalibrasyon ve ilk tespit yapılacak
        → Motion tracker bu kareyi referans olarak kullanır
    
    SHOW_TACTICAL_MINIMAP = True
        → Kuş bakışı saha minimap'i göster

modules/field_point_tracker.py içinde:

    PITCH_LENGTH = 105.0  (metre)
    PITCH_WIDTH = 68.0    (metre)
    GOAL_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 40.32
    CENTER_CIRCLE_RADIUS = 9.15
        → Standart futbol sahası ölçüleri (UEFA)
        → Farklı ligler/saçılar için değiştirilebilir
    
    scene_cut_threshold = 0.3
        → Histogram farkı eşiği (0.0-1.0)
        → Düşük → Daha fazla sahne kesintisi algılanır (yanlış pozitif riski)
        → Yüksek → Daha az algılanır (bazı kesintiler atlanabilir)
        → Varsayılan 0.3 çoğu video için iyidir

═══════════════════════════════════════════════════════════════════════════

7. HATA AYIKLAMA
═════════════════

Problem: Noktalar tespit edilmiyor
───────────────────────────────────
→ Çözüm 1: Video karesinin çok karanlık/parlak olduğunu kontrol et
→ Çözüm 2: config.CALIBRATION_FRAME_INDEX'i değiştir (başka bir kare dene)
→ Çözüm 3: Saha çizgileri net görülüyor mu? Tespit sensitivity ayarla

Problem: Sahne kesintileri hiç algılanmıyor
─────────────────────────────────────────────
→ Çözüm: field_point_tracker.scene_cut_threshold'ı düşür (0.2 veya 0.1)

Problem: Çok fazla yanlış sahne kesintisi algılanıyor
───────────────────────────────────────────────────────
→ Çözüm: field_point_tracker.scene_cut_threshold'ı artır (0.4 veya 0.5)

Problem: JSON dosyası pitch_x/pitch_y'si None (boş)
────────────────────────────────────────────────────
→ Neden: Kalibrasyon yapılmamış
→ Çözüm: python utils/calibrate_homography.py ile kalibrasyon yap
→ İleri: Kalibrasyon yapıldıktan sonra main.py'i yeniden çalıştır

═══════════════════════════════════════════════════════════════════════════

8. İLERİ KULLANIMı
══════════════════

Python kodu içinden field points'i kullanma:

    import json
    from modules.field_point_tracker import FieldPointTracker
    
    # JSON'dan yükle
    with open('data/exports/field_points.json', 'r') as f:
        data = json.load(f)
        field_points = data['field_points']
        
        for pid, point in field_points.items():
            print(f"{pid}: pixel=({point['pixel_x']}, {point['pixel_y']})")
            if point['pitch_x'] is not None:
                print(f"         pitch=({point['pitch_x']}, {point['pitch_y']}) metre")

Field point tracker'ı programı içinde kullanma:

    from modules.field_point_tracker import FieldPointTracker
    
    tracker = FieldPointTracker(use_line_detection=True)
    
    # İlk tespit
    points = tracker.detect_field_points_in_frame(frame, frame_idx=0)
    
    # Sahne kesintisi kontrol et
    is_cut = tracker.detect_scene_cut(frame)
    if is_cut:
        new_points = tracker.redetect_points_on_scene_cut(frame, frame_idx)
    
    # Dışa aktarım
    tracker.export_to_json('field_points.json')
    tracker.export_to_pickle('field_points.pkl')

═══════════════════════════════════════════════════════════════════════════

9. PERFORMANS İPUÇLARı
═══════════════════════

Büyük videolarda (>1 saat) işlem yavaş gidiyorsa:

1) use_line_detection=False yap (daha hızlı ama daha az doğru)
   tracker = FieldPointTracker(use_line_detection=False)

2) Sahne kesintisi algılamasını devre dışı bırak (çok kare işleme yapıyor)
   # main.py içinde:
   # is_scene_cut = tracker.detect_scene_cut(frame)  # Bu satırı yorum yap

3) Sadece her Nth kare'de tespit et (ör. 10. kareler)
   if frame_idx % 10 == 0:
       if is_scene_cut: ...

═══════════════════════════════════════════════════════════════════════════

10. KAYNAKLAR
═══════════════

Modül Dosyaları:
  • modules/field_point_tracker.py  : Ana modül
  • utils/detect_field_points.py    : Standalone tespit script
  • main.py                         : Pipeline entegrasyonu

İlgili Modüller:
  • modules/perspective_transformer.py : Homografi kalibrasyon
  • utils/calibrate_homography.py      : Etkileşimli kalibrasyon aracı

Diğer Kaynaklar:
  • config/config.py : Konfigürasyon parametreleri
  • README.md        : Proje ana kılavuzu

═══════════════════════════════════════════════════════════════════════════

11. SOZI
══════════

Sorularınız veya sorunlarınız varsa:
  1) Terminal çıktısı kontrol et (error/warning mesajları)
  2) field_point_detection_history.json debug çıktısını inceleyebilir:
     python -c "import json; print(json.dumps(json.load(open(...)), indent=2))" | head -100
  3) Annotated video'yu izle (noktaların nereye tespit edildiğini gör)

═══════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(__doc__)
