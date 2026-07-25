"""
main.py
-------
Projenin giriş noktası (entry point).

ŞU ANKİ DURUM (Adım 1 + 2 + 3 + 4 + 5 + 6 tamamlandı):
    Video okunuyor -> YOLO ile oyuncu/top tespiti yapılıyor ->
    DeepSORT ile oyunculara kalıcı ID atanıyor, top ayrıca takip ediliyor ->
    forma renklerine göre K-Means ile Takım A / Takım B / Hakem ayrımı yapılıyor ->
    optical flow ile kamera hareketi (pan/tilt/zoom) her karede tahmin edilip
    homografi buna göre güncelleniyor, piksel koordinatları kuş bakışı saha
    (metre) koordinatlarına çevriliyor (KALİBRASYON GEREKTİRİR, bkz.
    utils/calibrate_homography.py) -> kutular + ID'ler + takım renkleri
    çiziliyor -> yeni video olarak kaydediliyor -> HER KAREDEKİ tüm veri
    (piksel + saha koordinatları, takım, ID) biriktirilip video bitince
    CSV + SQLite veritabanına ve oyuncu başına mesafe/hız özetine yazılıyor.

Kullanım:
    python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from modules.video_processor import VideoProcessor
from modules.detector import ObjectDetector
from modules.tracker import PlayerTracker, BallTracker
from modules.team_classifier import TeamClassifier
from modules.perspective_transformer import PerspectiveTransformer
from modules.camera_motion_tracker import CameraMotionTracker
from modules.data_exporter import DataExporter


def parse_args():
    """Komut satırı argümanlarını okur (video yolu vs. hardcode etmemek için)."""
    parser = argparse.ArgumentParser(description="Futbol Maç Analiz ve Oyuncu Takip Sistemi")
    parser.add_argument(
        "--input", type=str, default=config.DEFAULT_INPUT_VIDEO,
        help="İşlenecek giriş videosunun yolu (.mp4)"
    )
    parser.add_argument(
        "--output", type=str, default=config.DEFAULT_OUTPUT_VIDEO,
        help="Kutuların çizileceği çıktı videosunun yolu (.mp4)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("FUTBOL MAÇ ANALİZ SİSTEMİ - Adım 1-5 Pipeline (dinamik kamera telafili)")
    print("=" * 60)

    # --- Adım 2 için: YOLO dedektörünü bir kere yükle (her frame'de yeniden yükleme!) ---
    detector = ObjectDetector()

    # --- Adım 3 için: Tracker'ları da video döngüsünden ÖNCE, bir kere oluşturuyoruz.
    # Tracker'lar İÇ DURUM (state) taşır -- yani "geçmiş kareleri hatırlar".
    # Her karede yeniden oluşturursak hafızaları sıfırlanır ve ID sürekliliği bozulur.
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    # --- Adım 4 için: TeamClassifier de state taşıdığı için (renk geçmişi,
    # fit edilmiş K-Means modeli) döngü dışında bir kere oluşturuluyor.
    team_classifier = TeamClassifier()

    # --- Adım 5 için: PerspectiveTransformer, kalibrasyon noktalarından
    # homografi matrisini SADECE BİR KERE hesaplar (__init__ içinde).
    # Kalibrasyon eksikse (config.SOURCE_CALIBRATION_POINTS boşsa) uyarı
    # verip is_calibrated=False olarak devam eder; pipeline çökmez, sadece
    # saha koordinatları None olarak kalır.
    transformer = PerspectiveTransformer()
    if not transformer.is_calibrated:
        print(
            "\n[UYARI] Homografi kalibrasyonu yapılmamış. Saha (metre) koordinatları "
            "üretilmeyecek VE taktik minimap görünümü çıktı videosuna eklenmeyecek "
            "(minimap, saha koordinatlarına ihtiyaç duyar). Kalibrasyon için: "
            "python utils/calibrate_homography.py\n"
        )

    # CameraMotionTracker de state taşır (önceki karenin gri tonlaması,
    # takip noktaları, birikimli hareket matrisi) -- döngü dışında bir kere oluşturuluyor.
    motion_tracker = CameraMotionTracker()

    # --- Adım 1 için: Video okuma/yazma nesnesini context manager ile aç ---
    # 'with' kullanımı, video dosyalarının işlem bitince (hata olsa bile)
    # otomatik olarak düzgünce kapatılmasını garanti eder.
    with VideoProcessor(args.input, args.output) as vp:

        # --- Adım 6 için: DataExporter, vp.fps'e ihtiyaç duyduğundan (zaman
        # damgası hesaplamak için) VideoProcessor açıldıktan SONRA, ama video
        # döngüsünden ÖNCE oluşturuluyor. O da state taşır (biriken kayıt
        # listesi) -- döngü içinde her seferinde yeniden oluşturulmamalı.
        exporter = DataExporter(fps=vp.fps)

        for frame_idx, frame in vp.read_frames():

            # 1) Bu frame üzerinde nesne tespiti yap (oyuncu + top) -- Adım 2
            raw_detections = detector.detect_frame(frame)

            # 2) Oyuncu/hakem tespitlerine kalıcı track_id ata -- Adım 3
            tracked_players = player_tracker.update(raw_detections, frame)

            # 3) Topu ayrı mantıkla takip et -- Adım 3
            tracked_ball = ball_tracker.update(raw_detections)

            # 4) Forma rengine göre Takım A / Takım B / Hakem ata -- Adım 4
            tracked_players = team_classifier.assign_teams(tracked_players, frame)

            # 5) Çizim için oyuncu + top listelerini birleştir
            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            # 6) Kamera hareketini (pan/tilt/zoom) tahmin et -- Adım 5 eklentisi
            motion_matrix = motion_tracker.update(frame, raw_detections)

            # 7) Piksel koordinatlarını saha (metre) koordinatlarına çevir -- Adım 5
            if motion_matrix is not None:
                # Normal durum: kamera hareketi telafili efektif homografi kullanılır
                all_tracked = transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=motion_matrix)
            else:
                # motion_matrix None ise CameraMotionTracker bir SAHNE KESMESİ
                # tespit etmiştir (ör. tekrar izlemeye geçildi). Yanlış saha
                # koordinatı üretmektense, bu kare için hiç üretmiyoruz.
                for det in all_tracked:
                    det["pitch_x"], det["pitch_y"], det["on_pitch"] = None, None, None

            # 8) Bu karenin tüm verisini dışa aktarım için biriktir -- Adım 6
            exporter.append_frame_data(frame_idx, all_tracked)

            # 9) Tespitleri frame üzerine görsel olarak çiz
            annotated_frame = vp.draw_detections(frame, all_tracked)

            # 10) Frame numarasını debug amaçlı sol üste yazalım
            import cv2
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx} | Oyuncu: {len(tracked_players)} | Top: {'Var' if tracked_ball else 'Yok'}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # 11) Taktik görünüm (kuş bakışı saha minimap'i) -- SADECE kalibrasyon
            # yapılmışsa üretilir/gömülür (kalibrasyon yoksa pitch_x/y zaten None'dur,
            # anlamlı bir minimap çizilemez).
            if config.SHOW_TACTICAL_MINIMAP and transformer.is_calibrated:
                minimap = transformer.draw_pitch_minimap(all_tracked)
                annotated_frame = vp.overlay_minimap(annotated_frame, minimap)

            # 12) İşlenmiş frame'i çıktı videosuna yaz
            vp.write_frame(annotated_frame)

        # --- Video döngüsü bitti (hâlâ 'with VideoProcessor' bloğunun içindeyiz,
        # yani vp.fps gibi bilgiler hâlâ erişilebilir). Şimdi Adım 6: biriken
        # tüm veriyi diske yaz. ---
        print("\n" + "=" * 60)
        print("VERİ DIŞA AKTARIMI (Adım 6)")
        print("=" * 60)
        exporter.export_to_csv()
        exporter.export_to_sql()
        exporter.export_distance_summary()

    print("\nİşlem tamamlandı!")
    print(f"Çıktı videosu şurada: {args.output}")
    print(f"Dışa aktarılan veriler şurada: {config.EXPORT_DIR}")


if __name__ == "__main__":
    main()
