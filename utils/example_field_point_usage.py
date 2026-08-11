"""
example_field_point_usage.py
─────────────────────────────

Field Point Tracking systeminin pratik kullanim ornekleri.

CALISTIRILMASI:
    python utils/example_field_point_usage.py

"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.field_point_tracker import FieldPointTracker, FieldPoint


def example_1_basic_detection():
    """
    Örnek 1: Temel tespit
    
    Saha noktalarını tespit etme ve JSON olarak kaydetme.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 1: TEMEL TESPIT (Basic Detection)")
    print("=" * 60)
    
    # FieldPointTracker oluştur
    tracker = FieldPointTracker(use_line_detection=True)
    
    # Örnek: Manuel olarak noktaları ekle (gerçekte detect_field_points_in_frame'den gelir)
    sample_points = {
        'corner_tl': FieldPoint(
            point_id='corner_tl',
            point_type='corner',
            pixel_x=100.0,
            pixel_y=120.0,
            pitch_x=0.0,
            pitch_y=0.0,
            confidence=0.95
        ),
        'corner_br': FieldPoint(
            point_id='corner_br',
            point_type='corner',
            pixel_x=1820.0,
            pixel_y=1020.0,
            pitch_x=105.0,
            pitch_y=68.0,
            confidence=0.92
        ),
        'center_circle': FieldPoint(
            point_id='center_circle',
            point_type='center_circle',
            pixel_x=960.0,
            pixel_y=570.0,
            pitch_x=52.5,
            pitch_y=34.0,
            confidence=0.88
        ),
    }
    
    tracker.field_points.update(sample_points)
    
    print(f"\nTespit edilen {len(sample_points)} nokta:")
    for point_id, point in sample_points.items():
        print(f"  {point_id:20s}: pixel=({point.pixel_x:7.1f}, {point.pixel_y:7.1f}) "
              f"pitch=({point.pitch_x:6.1f}, {point.pitch_y:6.1f}) "
              f"confidence={point.confidence:.2f}")
    
    # Dışa aktarım
    output_file = 'test_field_points.json'
    tracker.export_to_json(output_file)
    print(f"\nNoktalar '{output_file}' olarak kaydedildi.")
    
    # Yükleme ve okuma
    with open(output_file, 'r') as f:
        data = json.load(f)
        print(f"\nJSON içeriği okundu:")
        print(f"  Toplam nokta: {len(data['field_points'])}")
        print(f"  Saha boyutu: {data['metadata']['pitch_length_m']} x {data['metadata']['pitch_width_m']} m")
    
    # Cleanup
    if os.path.exists(output_file):
        os.remove(output_file)


def example_2_scene_cut_detection():
    """
    Örnek 2: Sahne kesintisi algılama
    
    Farklı frame'lerin histogram'larını karşılaştırarak
    sahne kesintisini algılama.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 2: SAHNE KESİNTİSİ ALGILAMA (Scene Cut Detection)")
    print("=" * 60)
    
    tracker = FieldPointTracker(use_line_detection=False)
    
    # Örnek: Threshold değerini göster
    print(f"\nSahne kesintisi algılama threshold: {tracker.scene_cut_threshold}")
    print(f"Bu değer histogram Bhattacharyya distance'ıdır.")
    print(f"\nÖrnek açıklamalar:")
    print(f"  • Frame 0→1 histogram farkı = 0.05 → Scene CUT DEĞİL")
    print(f"  • Frame 50→51 histogram farkı = 0.35 → SCENE CUT (kamera değişti)")
    print(f"  • Frame 100→101 histogram farkı = 0.25 → Scene CUT DEĞİL")
    print(f"\nThreshold değerini düşürüp yükseltmek duyarlılığı değiştirir:")
    print(f"  Düşük (0.1):  Çok sayıda yanlış alarm (yanlış pozitifleri)")
    print(f"  Normal (0.3): Dengeli algılama")
    print(f"  Yüksek (0.5): Sadece çok net kesintileri yakalar")


def example_3_point_classification():
    """
    Örnek 3: Nokta sınıflandırması
    
    Tespit edilen köşelerin futbol sahası öğelerine nasıl
    eşleştirildiğini göster.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 3: NOKTA SINIFLANDIRILMASI (Point Classification)")
    print("=" * 60)
    
    print("\nFutbol sahası köşe noktası türleri:")
    print("\nGRUP 1: Köşe bayrağı (4 noktası)")
    print("  corner_tl : Top-left corner")
    print("  corner_tr : Top-right corner")
    print("  corner_br : Bottom-right corner")
    print("  corner_bl : Bottom-left corner")
    
    print("\nGRUP 2: Ceza sahası (4 köşe + 2 kenar)")
    print("  penalty_tl : Top-left penalty corner")
    print("  penalty_tr : Top-right penalty corner")
    print("  penalty_bl : Bottom-left penalty corner")
    print("  penalty_br : Bottom-right penalty corner")
    print("  penalty_l  : Left penalty area")
    print("  penalty_r  : Right penalty area")
    
    print("\nGRUP 3: Merkez alan")
    print("  center_circle : Midfield center circle")
    print("  center_left   : Center line left")
    print("  center_right  : Center line right")
    
    print("\nGRUP 4: Gol çizgisi")
    print("  goal_line_left  : Goal line center (left side)")
    print("  goal_line_right : Goal line center (right side)")
    
    print("\nSTABİLİTE DURUMU:")
    print("  is_stable = True  : Noktası ilk tespit edilen ve henüz değişmedi")
    print("  is_stable = False : Sahne kesintisinden sonra yeniden tespit edildi")


def example_4_coordinates_info():
    """
    Örnek 4: Koordinat sistemi ve türleri
    
    Piksel vs saha koordinatlarını ve ne zaman kullanıldığını açıkla.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 4: KOORDİNAT SİSTEMLERİ (Coordinate Systems)")
    print("=" * 60)
    
    print("\nPİKSEL KOORDİNATLARI (pixel_x, pixel_y):")
    print("  • Video frame'deki konumu piksel cinsinden")
    print("  • Aralık: 0 ≤ x ≤ frame_width, 0 ≤ y ≤ frame_height")
    print("  • HER ZAMAN dolu (her tespit'te kullanılır)")
    print("  • Örnek: (1920, 1080)")
    
    print("\nSAHA KOORDİNATLARI (pitch_x, pitch_y):")
    print("  • Saha üzerindeki gerçek konumu metre cinsinden")
    print("  • Aralık: 0 ≤ x ≤ 105m, 0 ≤ y ≤ 68m (UEFA standart)")
    print("  • SADECE kalibrasyon yapılmışsa dolu (aksi takdirde None)")
    print("  • Örnek: (52.5, 34.0) = sahanın ortası")
    
    print("\nKALİBRASYON:")
    print("  1. utils/calibrate_homography.py ile kalibrasyon yap")
    print("  2. Homografi matrisi hesaplanır")
    print("  3. update_with_homography() ile noktaları dönüştür")
    print("  4. Artık saha koordinatları dolu olur")
    
    print("\nORTA NOKTALAR (standart UEFA sahası):")
    corner_tl = (0.0, 0.0)
    corner_br = (105.0, 68.0)
    center = (52.5, 34.0)
    penalty_top = (16.5, 34.0)
    print(f"  Top-left corner     : {corner_tl}")
    print(f"  Bottom-right corner : {corner_br}")
    print(f"  Center circle       : {center}")
    print(f"  Penalty area start  : {penalty_top}")


def example_5_export_formats():
    """
    Örnek 5: Dışa aktarım formatları
    
    JSON ve pickle formatlarının farkını göster.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 5: DIŞA AKTARIM FORMATLARI (Export Formats)")
    print("=" * 60)
    
    # Örnek nokta oluştur
    point = FieldPoint(
        point_id='corner_tl',
        point_type='corner',
        pixel_x=123.45,
        pixel_y=67.89,
        pitch_x=0.0,
        pitch_y=0.0,
        confidence=0.95,
        frame_detected=0
    )
    
    print("\nJSON FORMAŞ:")
    print("  ✓ İnsan tarafından okunabilir")
    print("  ✓ Diğer programlama dillerine aktarılabilir")
    print("  ✓ Metin editörü ile açılabilir")
    print("  ✗ Dosya boyutu biraz daha büyük")
    print("  ✗ Yükleme sırasında parsing gerekir")
    
    import json
    json_str = json.dumps({
        'point_id': point.point_id,
        'point_type': point.point_type,
        'pixel_x': point.pixel_x,
        'pixel_y': point.pixel_y,
        'confidence': point.confidence
    }, indent=2)
    print("\n  Örnek JSON:")
    for line in json_str.split('\n'):
        print(f"    {line}")
    
    print("\nPİCKLE FORMAŞ:")
    print("  ✓ Dosya boyutu daha küçük")
    print("  ✓ Yükleme daha hızlı")
    print("  ✓ Python objelerini doğrudan kaydeder")
    print("  ✗ Sadece Python'da okunabilir")
    print("  ✗ Güvenlik riski (untrusted pickle'ı açma)")
    
    print("\n  Kullanım örneği (Python):")
    print("    import pickle")
    print("    with open('field_points.pkl', 'rb') as f:")
    print("        points = pickle.load(f)")


def example_6_workflow():
    """
    Örnek 6: Tipik iş akışı
    
    Baştan sona nasıl kullanıldığını göster.
    """
    print("\n" + "=" * 60)
    print("ÖRNEK 6: TİPİK İŞ AKIŞI (Typical Workflow)")
    print("=" * 60)
    
    print("""
    ADIM 1: Kalibrasyon
    ├─ python utils/calibrate_homography.py
    ├─ Video'da saha noktalarını tıkla
    └─ Homografi matrisi kaydedilir
    
    ADIM 2: Ana işlem
    ├─ python main.py --input match.mp4
    ├─ Video frame-frame işlenir
    ├─ Frame 0: Noktalar otomatik tespit edilir
    ├─ Frame 1-N: Sahne kesintileri kontrol edilir
    │            → Kesinti bulunursa noktalar yeniden tespit
    └─ Tüm veriler dışa aktarılır
    
    ADIM 3: Verileri kullan
    ├─ field_points.json → Diğer programlara gönder
    ├─ field_points.pkl  → Python kodunda kullan
    ├─ Minimap görünümü → Taktik analiz
    └─ Oyuncu konumları → İstatistik hesaplaması
    
    ÇIKTI DOSYALARI:
    ├─ data/exports/field_points.json
    ├─ data/exports/field_points.pkl
    ├─ data/exports/field_point_detection_history.json
    ├─ data/output_videos/match_annotated.mp4 (noktaların çizili)
    └─ data/exports/tracking_data.csv
    """)


def main():
    """Ana fonksiyon - tüm örnekleri çalıştır."""
    print("\n" + "=" * 60)
    print("FUTBOL SAHASI NOKTASI İZLEME - KULLANIM ÖRNEKLERİ")
    print("Field Point Tracking - Usage Examples")
    print("=" * 60)
    
    examples = [
        ("1", example_1_basic_detection, "Temel tespit"),
        ("2", example_2_scene_cut_detection, "Sahne kesintisi algılama"),
        ("3", example_3_point_classification, "Nokta sınıflandırması"),
        ("4", example_4_coordinates_info, "Koordinat sistemleri"),
        ("5", example_5_export_formats, "Dışa aktarım formatları"),
        ("6", example_6_workflow, "Tipik iş akışı"),
    ]
    
    for key, func, desc in examples:
        try:
            func()
        except Exception as e:
            print(f"\n[ERROR] Örnek {key} hatası: {e}")
    
    print("\n" + "=" * 60)
    print("TÜM ÖRNEKLER TAMAMLANDI")
    print("=" * 60)
    print("""
    SONRAKI ADIMLAR:
    
    1. utils/detect_field_points.py kullanarak sahası noktalarını tespit et:
       python utils/detect_field_points.py --input video.mp4 --output exports
    
    2. Kalibrasyon yap:
       python utils/calibrate_homography.py
    
    3. Ana pipeline'ı çalıştır:
       python main.py --input video.mp4 --output output.mp4
    
    4. Çıktı dosyalarını incele:
       • exports/field_points.json → Noktaların koordinatları
       • exports/field_points_annotated.mp4 → Noktaların çizili videosu
    """)


if __name__ == "__main__":
    main()
