"""
detect_field_points.py
----------------------
Futbol sahası noktalarını video üzerinde otomatik olarak tespit eden ve
kaydeden utility script.

KULLANIM:
    python utils/detect_field_points.py --input data/input_videos/match.mp4 --output data/exports

ÇIKTI:
    - field_points.json     : Tespit edilen saha noktalarının koordinatları
    - field_points.pkl      : Binary format (Python pickle)
    - annotated_video.mp4   : Noktaların çizili olduğu annotated video
"""

import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.field_point_tracker import FieldPointTracker
from modules.perspective_transformer import PerspectiveTransformer


def process_video_for_field_points(input_video: str, output_dir: str, 
                                   show_detection: bool = True,
                                   save_annotated_video: bool = True) -> None:
    """
    Videonun tamamını işle ve saha noktalarını tespit et.
    
    Args:
        input_video: İnput video dosya yolu
        output_dir: Çıktı dosyalarının kaydedileceği dizin
        show_detection: Terminal'de tespit sonuçlarını göster
        save_annotated_video: Noktaların çizili olduğu videoyu kaydet
    """
    
    print("=" * 60)
    print("FUTBOL SAHASI NOKTASI TESPİTİ (Field Point Detection)")
    print("=" * 60)
    
    if not os.path.exists(input_video):
        print(f"[ERROR] Video bulunamadı: {input_video}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Video açılır
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Video açılamadı: {input_video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {input_video}")
    print(f"  Çözünürlük: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Toplam kare: {total_frames}")
    
    # FieldPointTracker oluştur
    tracker = FieldPointTracker(use_line_detection=True)
    
    # Perspektif transformer (kalibrasyon varsa saha koordinatlarına dönüştürmek için)
    transformer = PerspectiveTransformer()
    
    # Çıktı video writer
    output_video_path = None
    out = None
    if save_annotated_video:
        output_video_path = os.path.join(output_dir, "field_points_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Video işle
    frame_idx = 0
    calibration_frame_done = False
    
    print(f"\nVideo işleniyor...")
    print(f"[Frame] Tespit edilen noktalar: Scene Cut Algılandı?")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # İlk kare (kalibrasyon karesini) işle
        if not calibration_frame_done:
            print(f"[{frame_idx:5d}] İlk kare → Noktalar tespit ediliyor...")
            field_points = tracker.detect_field_points_in_frame(frame, frame_idx)
            tracker.field_points.update(field_points)
            
            if show_detection:
                print(f"        └─ {len(field_points)} nokta tespit edildi:")
                for pid, pt in field_points.items():
                    confidence_pct = pt.confidence * 100
                    print(f"           • {pid}: ({pt.pixel_x:.1f}, {pt.pixel_y:.1f}) [conf: {confidence_pct:.1f}%]")
            
            calibration_frame_done = True
        
        # Sonraki kareler: sahne kesintisi kontrol et
        else:
            is_scene_cut = tracker.detect_scene_cut(frame)
            
            if is_scene_cut:
                print(f"[{frame_idx:5d}] ⚠ SAHNE KESİNTİSİ → Noktalar YENİDEN TESPİT ediliyor...")
                new_points = tracker.redetect_points_on_scene_cut(frame, frame_idx)
                if show_detection:
                    print(f"        └─ {len(new_points)} nokta yeniden tespit edildi")
            
            # Her 100 kareyi göster
            if frame_idx % 100 == 0 and frame_idx > 0:
                print(f"[{frame_idx:5d}] İşleniyor... ({frame_idx}/{total_frames})")
        
        # Noktaları çiz
        annotated = tracker.draw_field_points(frame.copy())
        
        # Frame bilgisi ekle
        cv2.putText(
            annotated, f"Frame: {frame_idx}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Çıktı videoya yaz
        if out is not None:
            out.write(annotated)
        
        frame_idx += 1
    
    cap.release()
    if out is not None:
        out.release()
    
    # Noktaları dönüştür (kalibrasyon varsa)
    if transformer.is_calibrated:
        print(f"\n[Homografi] Kalibrasyon bulundu. Noktalar saha koordinatlarına dönüştürülüyor...")
        tracker.update_with_homography(transformer)
    else:
        print(f"\n[Homografi] Kalibrasyon yapılmamış. Noktalar piksel koordinatlarında kalacak.")
    
    # Dışa aktarım
    print("\n" + "=" * 60)
    print("DIŞA AKTARIMI (Export)")
    print("=" * 60)
    
    json_path = os.path.join(output_dir, "field_points.json")
    tracker.export_to_json(json_path)
    
    pkl_path = os.path.join(output_dir, "field_points.pkl")
    tracker.export_to_pickle(pkl_path)
    
    history_path = os.path.join(output_dir, "field_point_detection_history.json")
    tracker.export_detection_history(history_path)
    
    print(f"\n✓ Çıktılar {output_dir} içinde kaydedildi:")
    print(f"  • field_points.json               : Tespit edilen noktalar (JSON)")
    print(f"  • field_points.pkl                : Tespit edilen noktalar (pickle)")
    print(f"  • field_point_detection_history.json : Tespit geçmişi (debug)")
    if save_annotated_video:
        print(f"  • field_points_annotated.mp4      : Noktaların çizili videosu")
    
    print(f"\n✓ Toplam {len(tracker.field_points)} saha noktası kalıcı hale getirildi.")
    print(f"✓ Video işleme tamamlandı ({total_frames} kare işlendi).")


def detect_field_points_from_single_frame(frame_path: str, output_dir: str) -> None:
    """
    Tek bir frame'den saha noktalarını tespit et (hızlı test için).
    
    Args:
        frame_path: Frame görüntüsü dosya yolu
        output_dir: Çıktı dizini
    """
    
    print("=" * 60)
    print("FUTBOL SAHASI NOKTASI TESPİTİ - TEK KARE")
    print("=" * 60)
    
    if not os.path.exists(frame_path):
        print(f"[ERROR] Görüntü bulunamadı: {frame_path}")
        return
    
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[ERROR] Görüntü okunamadı: {frame_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Görüntü: {frame_path}")
    print(f"Çözünürlük: {frame.shape[1]}x{frame.shape[0]}")
    
    # Noktaları tespit et
    tracker = FieldPointTracker(use_line_detection=True)
    field_points = tracker.detect_field_points_in_frame(frame, frame_idx=0)
    tracker.field_points.update(field_points)
    
    print(f"\n{len(field_points)} saha noktası tespit edildi:")
    for pid, pt in field_points.items():
        confidence_pct = pt.confidence * 100
        print(f"  • {pid}: ({pt.pixel_x:.1f}, {pt.pixel_y:.1f}) [conf: {confidence_pct:.1f}%]")
    
    # Annotated çıktı
    annotated = tracker.draw_field_points(frame.copy())
    
    output_img_path = os.path.join(output_dir, "field_points_detected.png")
    cv2.imwrite(output_img_path, annotated)
    print(f"\n✓ Annotated görüntü kaydedildi: {output_img_path}")
    
    # Dışa aktarım
    json_path = os.path.join(output_dir, "field_points.json")
    tracker.export_to_json(json_path)
    print(f"✓ Noktalar kaydedildi: {json_path}")


def parse_args():
    """Komut satırı argümanlarını oku."""
    parser = argparse.ArgumentParser(
        description="Futbol sahası noktalarını otomatik olarak tespit et"
    )
    
    parser.add_argument(
        "--input", type=str, default=config.DEFAULT_INPUT_VIDEO,
        help="İnput video (.mp4) veya frame (.png/.jpg) dosya yolu"
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(config.EXPORT_DIR, "field_points"),
        help="Çıktı dosyalarının kaydedileceği dizin"
    )
    parser.add_argument(
        "--no-annotated-video", action="store_true",
        help="Annotated video'yu kaydetme"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Terminal çıktısını kısalt"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Input türünü belirle
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video işle
        process_video_for_field_points(
            args.input,
            args.output,
            show_detection=(not args.quiet),
            save_annotated_video=(not args.no_annotated_video)
        )
    elif input_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        # Tek kare işle
        detect_field_points_from_single_frame(args.input, args.output)
    else:
        print(f"[ERROR] Desteklenmeyen dosya formatı: {input_ext}")
        print("Lütfen video (.mp4, .avi, .mov, .mkv) veya frame (.png, .jpg) dosyası sağla.")
