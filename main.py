"""
main.py
-------
Projenin giriş noktası (entry point).

ŞU ANKİ DURUM (Adım 1 + 2 + 3 tamamlandı):
    Video okunuyor -> YOLO ile oyuncu/top tespiti yapılıyor ->
    DeepSORT ile oyunculara kalıcı ID atanıyor, top ayrıca takip ediliyor ->
    kutular + ID'ler çiziliyor -> yeni video olarak kaydediliyor.

İLERİDE EKLENECEK ADIMLAR (bir sonraki mesajlarda birlikte yazacağız):
    4) TeamClassifier (K-Means)      -> forma rengine göre Takım A / B / Hakem ayır
    5) PerspectiveTransformer        -> piksel koordinatlarını saha koordinatına çevir
    6) DataExporter (Pandas/SQL)     -> tüm veriyi CSV + SQL'e yaz

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
    print("FUTBOL MAÇ ANALİZ SİSTEMİ - Adım 1 & 2 Pipeline")
    print("=" * 60)

    # --- Adım 2 için: YOLO dedektörünü bir kere yükle (her frame'de yeniden yükleme!) ---
    detector = ObjectDetector()

    # --- Adım 3 için: Tracker'ları da video döngüsünden ÖNCE, bir kere oluşturuyoruz.
    # Tracker'lar İÇ DURUM (state) taşır -- yani "geçmiş kareleri hatırlar".
    # Her karede yeniden oluşturursak hafızaları sıfırlanır ve ID sürekliliği bozulur.
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    # --- Adım 1 için: Video okuma/yazma nesnesini context manager ile aç ---
    # 'with' kullanımı, video dosyalarının işlem bitince (hata olsa bile)
    # otomatik olarak düzgünce kapatılmasını garanti eder.
    with VideoProcessor(args.input, args.output) as vp:
        for frame_idx, frame in vp.read_frames():

            # 1) Bu frame üzerinde nesne tespiti yap (oyuncu + top) -- Adım 2
            raw_detections = detector.detect_frame(frame)

            # 2) Oyuncu/hakem tespitlerine kalıcı track_id ata -- Adım 3
            tracked_players = player_tracker.update(raw_detections, frame)

            # 3) Topu ayrı mantıkla takip et -- Adım 3
            tracked_ball = ball_tracker.update(raw_detections)

            # 4) Çizim için oyuncu + top listelerini birleştir
            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            # NOT: Bir sonraki adımda burada şu satır eklenecek:
            #   all_tracked = team_classifier.assign_teams(all_tracked, frame)
            # Bu, her oyuncuya forma rengine göre "team" bilgisini dolduracak.

            # 5) Tespitleri frame üzerine görsel olarak çiz
            annotated_frame = vp.draw_detections(frame, all_tracked)

            # 6) Frame numarasını debug amaçlı sol üste yazalım
            import cv2
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx} | Oyuncu: {len(tracked_players)} | Top: {'Var' if tracked_ball else 'Yok'}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # 7) İşlenmiş frame'i çıktı videosuna yaz
            vp.write_frame(annotated_frame)

    print("\nİşlem tamamlandı!")
    print(f"Çıktı videosu şurada: {args.output}")


if __name__ == "__main__":
    main()
