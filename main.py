"""
main.py
-------
Projenin giriş noktası (entry point).

ŞU ANKİ DURUM (Adım 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 tamamlandı):
    Video okunuyor -> YOLO ile oyuncu/top tespiti yapılıyor ->
    DeepSORT ile oyunculara kalıcı ID atanıyor, top ayrıca takip ediliyor ->
    forma renklerine göre K-Means ile Takım A / Takım B / Hakem ayrımı yapılıyor ->
    optical flow ile kamera hareketi (pan/tilt/zoom) her karede tahmin edilip
    homografi buna göre güncelleniyor, piksel koordinatları kuş bakışı saha
    (metre) koordinatlarına çevriliyor (KALİBRASYON GEREKTİRİR, bkz.
    utils/calibrate_homography.py) -> kutular + ID'ler + takım renkleri
    çiziliyor -> yeni video olarak kaydediliyor -> HER KAREDEKİ tüm veri
    (piksel + saha koordinatları, takım, ID) biriktirilip video bitince
    CSV + SQLite veritabanına, oyuncu mesafe/hız özetine, Adım 7 taktik analiz
    raporuna ve Adım 8 Taktik & Scouting ML raporlarına (xP/xT, oyuncu rol
    kümeleme, tempo/anomali tespiti) yazılıyor.

Kullanım:
    python main.py --input data/input_videos/match_sample.mp4 --output data/output_videos/match_sample_annotated.mp4
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from modules.video_processor import VideoProcessor
from modules.detector import ObjectDetector
from modules.tracker import PlayerTracker, BallTracker
from modules.team_classifier import TeamClassifier
from modules.perspective_transformer import PerspectiveTransformer
from modules.camera_motion_tracker import CameraMotionTracker
from modules.data_exporter import DataExporter
from modules.match_analytics import MatchAnalyzer
from modules.ml_analyzer import MLAnalyzer


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
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="YOLO batch çıkarım boyutu (varsayılan: CUDA'da config.DETECTION_BATCH_SIZE, CPU'da 1)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Test için yalnızca ilk N kareyi işle (varsayılan: tüm video)"
    )
    parser.add_argument(
        "--no-analytics", action="store_true",
        help="Video bitiminde Adım 7 taktik analiz ve ısı haritası üretimini atla"
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Video bitiminde Adım 8 taktik ve scouting ML analizini (xP/xT, rol kümeleme, tempo/anomali) atla"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("FUTBOL MAÇ ANALİZ SİSTEMİ - Adım 1-8 Pipeline (dinamik kamera telafili)")
    print("=" * 60)

    # --- Adım 2 için: YOLO dedektörünü bir kere yükle (her frame'de yeniden yükleme!) ---
    detector = ObjectDetector()
    batch_size = (
        max(1, args.batch_size)
        if args.batch_size is not None
        else (config.DETECTION_BATCH_SIZE if detector.device.startswith("cuda") else 1)
    )
    print(f"[Pipeline] YOLO batch_size={batch_size}")

    # --- Adım 3 için: Tracker'ları da video döngüsünden ÖNCE, bir kere oluşturuyoruz.
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    # --- Adım 4 için: TeamClassifier state taşıdığı için döngü dışında bir kere oluşturuluyor. ---
    team_classifier = TeamClassifier()

    # --- Adım 5 için: PerspectiveTransformer ---
    transformer = PerspectiveTransformer()
    if not transformer.is_calibrated:
        print(
            "\n[UYARI] Homografi kalibrasyonu yapılmamış. Saha (metre) koordinatları "
            "üretilmeyecek VE taktik minimap görünümü çıktı videosuna eklenmeyecek "
            "(minimap, saha koordinatlarına ihtiyaç duyar). Kalibrasyon için: "
            "python utils/calibrate_homography.py --input <video_yolu>\n"
        )

    motion_tracker = CameraMotionTracker()

    with VideoProcessor(args.input, args.output) as vp:
        exporter = DataExporter(fps=vp.fps)

        def process_frame(frame_idx, frame, raw_detections):
            # 2) Oyuncu/hakem tespitlerine kalıcı track_id ata -- Adım 3
            tracked_players = player_tracker.update(raw_detections, frame)

            # 3) Topu sönümlenen hızlı tekil takipçiyle izle -- Adım 3
            tracked_ball = ball_tracker.update(raw_detections, frame_shape=frame.shape)

            # 4) Forma rengine göre Takım A / Takım B / Hakem / Kaleci ata -- Adım 4
            tracked_players = team_classifier.assign_teams(tracked_players, frame)

            # 5) Çizim için oyuncu + top listelerini birleştir
            all_tracked = tracked_players.copy()
            if tracked_ball is not None:
                all_tracked.append(tracked_ball)

            # 6) Kamera hareketini (pan/tilt/zoom) tahmin et -- Adım 5 eklentisi
            if frame_idx >= config.CALIBRATION_FRAME_INDEX:
                motion_matrix = motion_tracker.update(frame, raw_detections)
            else:
                motion_matrix = None

            # 7) Piksel koordinatlarını saha (metre) koordinatlarına çevir -- Adım 5
            if motion_matrix is not None:
                all_tracked = transformer.add_pitch_coordinates(all_tracked, camera_motion_matrix=motion_matrix)
            else:
                for det in all_tracked:
                    det["pitch_x"], det["pitch_y"], det["on_pitch"] = None, None, None

            # 8) Bu karenin tüm verisini dışa aktarım için biriktir -- Adım 6
            exporter.append_frame_data(frame_idx, all_tracked)

            # 9) Tespitleri frame üzerine görsel olarak çiz
            annotated_frame = vp.draw_detections(frame, all_tracked)

            # 10) Frame numarasını debug amaçlı sol üste yazalım
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx} | Oyuncu: {len(tracked_players)} | Top: {'Var' if tracked_ball else 'Yok'}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # 11) Taktik görünüm (kuş bakışı saha minimap'i)
            if config.SHOW_TACTICAL_MINIMAP and transformer.is_calibrated:
                minimap = transformer.draw_pitch_minimap(all_tracked)
                annotated_frame = vp.overlay_minimap(annotated_frame, minimap)

            # 12) İşlenmiş frame'i çıktı videosuna yaz
            vp.write_frame(annotated_frame)

        frame_batch = []
        for frame_idx, frame in vp.read_frames():
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            frame_batch.append((frame_idx, frame))
            if len(frame_batch) >= batch_size:
                if batch_size > 1:
                    batch_detections = detector.detect_batch([f for _, f in frame_batch])
                else:
                    batch_detections = [detector.detect_frame(frame_batch[0][1])]

                for (f_idx, f), raw_detections in zip(frame_batch, batch_detections):
                    process_frame(f_idx, f, raw_detections)
                frame_batch.clear()

        if frame_batch:
            if batch_size > 1 and len(frame_batch) > 1:
                batch_detections = detector.detect_batch([f for _, f in frame_batch])
            else:
                batch_detections = [detector.detect_frame(f) for _, f in frame_batch]

            for (f_idx, f), raw_detections in zip(frame_batch, batch_detections):
                process_frame(f_idx, f, raw_detections)
            frame_batch.clear()

        # --- Video döngüsü bitti: Adım 6 dışa aktarım ve Adım 7 taktik analiz ---
        print("\n" + "=" * 60)
        print("VERİ DIŞA AKTARIMI (Adım 6)")
        print("=" * 60)
        exporter.postprocess(team_map=team_classifier.get_team_map())
        exporter.export_to_csv()
        exporter.export_to_sql()
        exporter.export_distance_summary()

        analytics_summary = None
        if config.GENERATE_ANALYTICS_REPORT and not args.no_analytics:
            print("\n" + "=" * 60)
            print("TAKTİK ANALİZ VE ISI HARİTASI (Adım 7)")
            print("=" * 60)
            analyzer = MatchAnalyzer(exporter.to_dataframe(), fps=vp.fps)
            analytics_summary = analyzer.export_all()
            poss = analytics_summary["possession"]
            passes = analytics_summary["passes"]
            print(
                f"Topa Sahip Olma ({poss['mode']}) -> "
                f"Team A: %{poss['team_a_pct']} | Team B: %{poss['team_b_pct']}"
            )
            print(
                f"Pas Özeti -> Tamamlanan: {passes['total_completed_passes']} | "
                f"Top Kaybı: {passes['total_turnovers']}"
            )

        if config.GENERATE_ML_REPORT and not args.no_ml:
            print("\n" + "=" * 60)
            print("TAKTİK VE SCOUTING ML MOTORU (Adım 8)")
            print("=" * 60)
            ml_analyzer = MLAnalyzer(
                exporter.to_dataframe(),
                fps=vp.fps,
                analytics_summary=analytics_summary,
            )
            ml_summary = ml_analyzer.export_all()
            xp = ml_summary["xp_modeling"]
            best_m = next((m for m in xp["models"] if m["model_name"] == xp["selected_model"]), None)
            cl = ml_summary["player_clustering"]
            tp = ml_summary["tempo_and_anomalies"]
            if best_m is not None:
                print(
                    f"xP Model ({xp['selected_model']}) -> ROC-AUC: {best_m['test']['roc_auc']} | "
                    f"PR-AUC: {best_m['test']['pr_auc']} | Brier: {best_m['test']['brier_score']}"
                )
            print(
                f"Rol Kümeleme -> Oyuncu: {cl['eligible_players']} | "
                f"Optimal k: {cl['optimal_k']} | Silhouette: {cl['best_silhouette']}"
            )
            print(
                f"Tempo & Anomali -> Tempo Düşüşü: {tp['tempo_drop_count']} | "
                f"Yorgunluk: {tp['fatigue_players_count']} | Anomali: {tp['total_anomalies']}"
            )

    print("\nİşlem tamamlandı!")
    print(f"Çıktı videosu şurada: {args.output}")
    print(f"Dışa aktarılan veriler şurada: {config.EXPORT_DIR}")


if __name__ == "__main__":
    main()
