"""
main.py
-------
Projenin giriş noktası (entry point).

ŞU ANKİ DURUM (Adım 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 tamamlandı):
    Video okunuyor -> YOLO ile 4 sınıflı (ball, goalkeeper, player, referee)
    oyuncu/top tespiti yapılıyor -> DeepSORT veya ByteTrack ile oyunculara
    kalıcı ID atanıyor, top ayrıca takip ediliyor -> forma renklerine göre
    K-Means ile Takım A / Takım B / Hakem / Kaleci ayrımı yapılıyor ->
    optical flow + ORB anahtar kare bankası ile kamera hareketi (pan/tilt/zoom)
    ve TV yayın sahne kesmesi / tekrar dönüşü her karede tahmin edilip
    homografi buna göre güncelleniyor, piksel koordinatları kuş bakışı saha
    (metre) koordinatlarına çevriliyor -> (isteğe bağlı) kutular + ID'ler +
    takım renkleri + taktik minimap çizilip çıktı videosuna kaydediliyor ->
    90 dakikalık maç ölçeğinde periyodik disk akışı (SQLite WAL + CSV) ile
    bellek şişmeden kaydediliyor -> Adım 7 taktik analiz ve Adım 8 Taktik &
    Scouting ML raporları (xP/xT, oyuncu rol kümeleme, tempo/anomali tespiti)
    üretiliyor.

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
    """Komut satırı argümanlarını okur (video yolu ve 90 dk üretim bayrakları)."""
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
        "--no-video", action="store_true",
        help="Çıktı videosu oluşturmayı ve kare çizimini devre dışı bırak (90 dk hızlı analiz modu)"
    )
    parser.add_argument(
        "--no-minimap", action="store_true",
        help="Çıktı videosu üzerine kuş bakışı 2D taktik minimap bindirmeyi devre dışı bırak"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Ham takip verisini CSV dosyasına (tracking_data.csv) yazmayı atla"
    )
    parser.add_argument(
        "--no-sql", action="store_true",
        help="Ham takip verisini SQLite veritabanına (tracking_data.db) yazmayı atla"
    )
    parser.add_argument(
        "--export-dir", type=str, default=config.EXPORT_DIR,
        help="Tüm veri ve rapor çıktılarının yazılacağı dizin (varsayılan: config.EXPORT_DIR)"
    )
    parser.add_argument(
        "--flush-interval", type=int,
        default=2500,
        help="Verilerin kaç karede bir diske (SQLite WAL / CSV) akıtılacağı (0 = sadece video sonunda yaz)"
    )
    parser.add_argument(
        "--tracker", "--tracker-type", dest="tracker",
        choices=["deepsort", "bytetrack", "fast", "iou", "kalman_iou"],
        default=getattr(config, "TRACKER_TYPE", "deepsort"),
        help="Oyuncu takip algoritması: 'deepsort' (görünüm + hareket) veya 'bytetrack' (hızlı Kalman+IoU)"
    )
    parser.add_argument(
        "--embedding-stride", type=int,
        default=getattr(config, "TRACKER_EMBEDDING_STRIDE", 1),
        help="DeepSORT modunda görünüm embedding hesaplama adımı (1 = her kare, >1 = ara karelerde IoU)"
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

    export_dir = os.path.abspath(args.export_dir) if args.export_dir else os.path.abspath(config.EXPORT_DIR)
    os.makedirs(export_dir, exist_ok=True)
    output_video_path = None if args.no_video else args.output
    enable_csv = not args.no_csv
    enable_sql = not args.no_sql
    show_minimap = config.SHOW_TACTICAL_MINIMAP and (not args.no_minimap)

    print("=" * 60)
    print("FUTBOL MAÇ ANALİZ SİSTEMİ - Adım 1-8 Pipeline (90 Dk Ölçeklenebilir & Sahne Kesmesi Korumalı)")
    print("=" * 60)

    # --- Adım 2 için: YOLO dedektörünü bir kere yükle ---
    detector = ObjectDetector()
    batch_size = (
        max(1, args.batch_size)
        if args.batch_size is not None
        else (config.DETECTION_BATCH_SIZE if detector.device.startswith("cuda") else 1)
    )
    print(
        f"[Pipeline] YOLO batch_size={batch_size} | tracker={args.tracker} | "
        f"flush_interval={args.flush_interval} | export_dir={export_dir}"
    )

    # --- Adım 3 için: Tracker'ları video döngüsünden ÖNCE bir kere oluştur ---
    try:
        player_tracker = PlayerTracker(
            tracker_type=args.tracker,
            embedding_stride=args.embedding_stride,
        )
    except TypeError:
        player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    # --- Adım 4 için: TeamClassifier ---
    team_classifier = TeamClassifier()

    # --- Adım 5 için: PerspectiveTransformer & CameraMotionTracker ---
    transformer = PerspectiveTransformer()
    if not transformer.is_calibrated:
        print(
            "\n[UYARI] Homografi kalibrasyonu yapılmamış. Saha (metre) koordinatları "
            "üretilmeyecek VE taktik minimap görünümü çıktı videosuna eklenmeyecek "
            "(minimap, saha koordinatlarına ihtiyaç duyar). Kalibrasyon için: "
            "python utils/calibrate_homography.py --input <video_yolu>\n"
        )

    motion_tracker = CameraMotionTracker()

    with VideoProcessor(args.input, output_video_path) as vp:
        try:
            exporter = DataExporter(
                fps=vp.fps,
                output_dir=export_dir,
                flush_interval=args.flush_interval,
                enable_csv=enable_csv,
                enable_sql=enable_sql,
            )
        except TypeError:
            exporter = DataExporter(fps=vp.fps)

        try:
            def process_frame(frame_idx, frame, raw_detections):
                # 1) Kamera hareketini (pan/tilt/zoom) ve sahne kesmesini tahmin et -- Adım 5
                if frame_idx >= config.CALIBRATION_FRAME_INDEX:
                    motion_matrix = motion_tracker.update(frame, raw_detections)
                else:
                    motion_matrix = None

                # Sahne kesmesi tespit edildiyse aktif oyuncu/top izlerini sıfırla
                if getattr(motion_tracker, "scene_cut_detected", False):
                    if hasattr(player_tracker, "on_scene_cut"):
                        player_tracker.on_scene_cut()
                    if hasattr(ball_tracker, "on_scene_cut"):
                        ball_tracker.on_scene_cut()

                # 2) Oyuncu/hakem/kaleci tespitlerine kalıcı track_id ata -- Adım 3
                tracked_players = player_tracker.update(raw_detections, frame)

                # 3) Topu sönümlenen hızlı tekil takipçiyle izle -- Adım 3
                tracked_ball = ball_tracker.update(raw_detections, frame_shape=frame.shape)

                # 4) Forma rengine göre Takım A / Takım B / Hakem / Kaleci ata -- Adım 4
                tracked_players = team_classifier.assign_teams(tracked_players, frame)

                # 5) Oyuncu + top listelerini birleştir
                all_tracked = tracked_players.copy()
                if tracked_ball is not None:
                    all_tracked.append(tracked_ball)

                # 6) Piksel koordinatlarını saha (metre) koordinatlarına çevir -- Adım 5
                if motion_matrix is not None:
                    all_tracked = transformer.add_pitch_coordinates(
                        all_tracked, camera_motion_matrix=motion_matrix
                    )
                else:
                    for det in all_tracked:
                        det["pitch_x"], det["pitch_y"], det["on_pitch"] = None, None, None

                # 7) Bu karenin tüm verisini dışa aktarım için biriktir / akıt -- Adım 6
                exporter.append_frame_data(frame_idx, all_tracked)

                # 8) Çıktı videosu aktifse kare üzerine çiz ve yaz
                if getattr(vp, "has_writer", output_video_path is not None):
                    try:
                        annotated_frame = vp.draw_detections(frame, all_tracked, inplace=True)
                    except TypeError:
                        annotated_frame = vp.draw_detections(frame, all_tracked)
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_idx} | Oyuncu: {len(tracked_players)} | Top: {'Var' if tracked_ball else 'Yok'}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    if show_minimap and transformer.is_calibrated and motion_matrix is not None:
                        minimap = transformer.draw_pitch_minimap(all_tracked)
                        annotated_frame = vp.overlay_minimap(annotated_frame, minimap)
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

            # --- Video döngüsü bitti: Adım 6 dışa aktarım ---
            print("\n" + "=" * 60)
            print("VERİ DIŞA AKTARIMI (Adım 6)")
            print("=" * 60)
            exporter.postprocess(team_map=team_classifier.get_team_map())
            csv_out_path = exporter.export_to_csv() if enable_csv else None
            sql_out_path = exporter.export_to_sql() if enable_sql else None
            exporter.export_distance_summary()

            # Adım 7 ve Adım 8 için veri kaynağını seç (SQLite -> CSV -> bellek)
            def _load_match_analyzer() -> MatchAnalyzer:
                if sql_out_path and os.path.isfile(sql_out_path):
                    return MatchAnalyzer.from_sql(sql_out_path, fps=vp.fps)
                if csv_out_path and os.path.isfile(csv_out_path):
                    return MatchAnalyzer.from_csv(csv_out_path, fps=vp.fps)
                return MatchAnalyzer(exporter.to_dataframe(), fps=vp.fps)

            def _load_ml_analyzer(analytics_sum) -> MLAnalyzer:
                if sql_out_path and os.path.isfile(sql_out_path):
                    return MLAnalyzer.from_sql(
                        sql_out_path, fps=vp.fps, analytics_summary=analytics_sum
                    )
                if csv_out_path and os.path.isfile(csv_out_path):
                    return MLAnalyzer.from_csv(
                        csv_out_path, fps=vp.fps, analytics_summary=analytics_sum
                    )
                return MLAnalyzer(
                    exporter.to_dataframe(), fps=vp.fps, analytics_summary=analytics_sum
                )

            analytics_summary = None
            if config.GENERATE_ANALYTICS_REPORT and not args.no_analytics:
                print("\n" + "=" * 60)
                print("TAKTİK ANALİZ VE ISI HARİTASI (Adım 7)")
                print("=" * 60)
                analyzer = _load_match_analyzer()
                analytics_summary = analyzer.export_all(output_dir=export_dir)
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
                ml_out_dir = os.path.join(export_dir, "ml_reports")
                ml_analyzer = _load_ml_analyzer(analytics_summary)
                ml_summary = ml_analyzer.export_all(output_dir=ml_out_dir)
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
        finally:
            if hasattr(exporter, "close"):
                exporter.close()

    print("\nİşlem tamamlandı!")
    if output_video_path:
        print(f"Çıktı videosu şurada: {output_video_path}")
    print(f"Dışa aktarılan veriler şurada: {export_dir}")


if __name__ == "__main__":
    main()
