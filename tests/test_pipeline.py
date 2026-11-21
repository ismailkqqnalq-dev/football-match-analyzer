import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.camera_motion_tracker import CameraMotionTracker
from modules.data_exporter import DataExporter
import modules.detector as detector_module
from modules.match_analytics import MatchAnalyzer
from modules.perspective_transformer import PerspectiveTransformer
from modules.team_classifier import TeamClassifier
from modules.tracker import BallTracker


def _make_synthetic_jersey_frame(players_spec):
    """Yeşil çim arka planlı sentetik kare ve her oyuncu için forma kutusu oluşturur."""
    frame = np.full((600, 1000, 3), (35, 140, 35), dtype=np.uint8)
    for spec in players_spec:
        x1, y1, x2, y2 = map(int, spec["bbox"])
        bw = x2 - x1
        bh = y2 - y1
        # Gövde kırpımının (0.15..0.55 x 0.20..0.80) iç kısmını forma rengiyle boya;
        # köşelerde yeşil çim arka planı kalsın.
        jy1 = y1 + int(bh * 0.22)
        jy2 = y1 + int(bh * 0.50)
        jx1 = x1 + int(bw * 0.30)
        jx2 = x1 + int(bw * 0.70)
        frame[jy1:jy2, jx1:jx2] = spec["bgr"]
    return frame


def test_team_classifier_lab_cache_and_finetune(monkeypatch):
    # 1) Standart LAB modu: 2 takım + 1 hakem grubu + önbellek çağrı sayısı doğrulaması
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", False)

    players_spec = []
    # 5 oyuncu koyu lacivert (Team A adayı), 5 oyuncu beyaz (Team B adayı), 2 hakem sarı
    for idx in range(5):
        players_spec.append({
            "track_id": idx + 1,
            "class_name": "person",
            "bbox": [40 + idx * 80, 100, 90 + idx * 80, 240],
            "bgr": (150, 30, 20),  # koyu mavi
        })
    for idx in range(5):
        players_spec.append({
            "track_id": idx + 6,
            "class_name": "person",
            "bbox": [40 + idx * 80, 300, 90 + idx * 80, 440],
            "bgr": (245, 245, 245),  # beyaz
        })
    for idx in range(2):
        players_spec.append({
            "track_id": idx + 11,
            "class_name": "person",
            "bbox": [600 + idx * 90, 200, 650 + idx * 90, 340],
            "bgr": (0, 230, 245),  # parlak sarı
        })

    frame = _make_synthetic_jersey_frame(players_spec)
    classifier = TeamClassifier(
        n_clusters=3,
        min_unique_tracks=12,
        color_space="LAB",
        min_samples_for_lock=2,
    )

    extract_calls = 0
    orig_extract = classifier.extract_jersey_color

    def tracked_extract(frm, bbox):
        nonlocal extract_calls
        extract_calls += 1
        return orig_extract(frm, bbox)

    monkeypatch.setattr(classifier, "extract_jersey_color", tracked_extract)

    dets_frame1 = [
        {"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None}
        for s in players_spec
    ]
    out1 = classifier.assign_teams(dets_frame1, frame)
    assert classifier.is_fitted is True
    assert extract_calls == 12

    blue_teams = {d["team"] for d in out1[:5]}
    white_teams = {d["team"] for d in out1[5:10]}
    ref_teams = {d["team"] for d in out1[10:]}
    assert len(blue_teams) == 1
    assert len(white_teams) == 1
    assert blue_teams != white_teams
    assert ref_teams == {"Referee"}

    # İkinci karede tüm fit edilen ID'ler önbellekten dönmeli (extract çağrılmamalı)
    dets_frame2 = [
        {"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None}
        for s in players_spec
    ]
    out2 = classifier.assign_teams(dets_frame2, frame)
    assert extract_calls == 12
    assert {d["team"] for d in out2[:5]} == blue_teams
    assert {d["team"] for d in out2[5:10]} == white_teams
    assert {d["team"] for d in out2[10:]} == {"Referee"}

    # Fit sonrası yeni gelen 13. oyuncu min_samples_for_lock=2 örnek sonra kilitlenmeli
    new_spec = players_spec + [{
        "track_id": 13,
        "class_name": "person",
        "bbox": [820, 100, 870, 240],
        "bgr": (150, 30, 20),
    }]
    frame_with_13 = _make_synthetic_jersey_frame(new_spec)
    new_det = [{"track_id": 13, "class_name": "person", "bbox": [820, 100, 870, 240], "team": None}]
    out_new1 = classifier.assign_teams([dict(new_det[0])], frame_with_13)
    assert out_new1[0]["team"] == next(iter(blue_teams))
    assert extract_calls == 13
    assert 13 not in classifier.team_assignment

    out_new2 = classifier.assign_teams([dict(new_det[0])], frame_with_13)
    assert out_new2[0]["team"] == next(iter(blue_teams))
    assert extract_calls == 14
    assert 13 in classifier.team_assignment

    # 3. çağrıda artık önbellekte olduğu için extract çağrılmamalı
    classifier.assign_teams([dict(new_det[0])], frame_with_13)
    assert extract_calls == 14

    # 2) Fine-tune modu: referee ve goalkeeper renk kümelemesine hiç girmemeli
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    ft_classifier = TeamClassifier(n_clusters=2, min_unique_tracks=4, color_space="LAB")
    ft_calls = 0
    orig_ft_extract = ft_classifier.extract_jersey_color

    def ft_spy(frm, bbox):
        nonlocal ft_calls
        ft_calls += 1
        return orig_ft_extract(frm, bbox)

    monkeypatch.setattr(ft_classifier, "extract_jersey_color", ft_spy)

    ft_dets = [
        {"track_id": 101, "class_name": "referee", "bbox": [600, 200, 650, 340], "team": None},
        {"track_id": 102, "class_name": "goalkeeper", "bbox": [700, 200, 750, 340], "team": None},
    ]
    ft_out = ft_classifier.assign_teams(ft_dets, frame)
    assert ft_out[0]["team"] == "Referee"
    assert ft_out[1]["team"] == "Goalkeeper"
    assert ft_calls == 0
    assert 101 not in ft_classifier.color_history
    assert 102 not in ft_classifier.color_history


def test_ball_tracker_damped_velocity_and_timeout():
    tracker = BallTracker(max_missing_frames=3, velocity_damping=0.8, max_velocity_px=100.0)
    frame_shape = (100, 200, 3)

    # Kare 0 ve 1: top sağa doğru 20 px/kare hareket ediyor
    d0 = tracker.update(
        [{"class_name": "ball", "bbox": [140.0, 40.0, 150.0, 50.0], "confidence": 0.9}],
        frame_shape=frame_shape,
    )
    assert d0 is not None and d0["is_predicted"] is False

    d1 = tracker.update(
        [{"class_name": "ball", "bbox": [160.0, 40.0, 170.0, 50.0], "confidence": 0.92}],
        frame_shape=frame_shape,
    )
    assert d1 is not None and d1["is_predicted"] is False
    assert tracker.velocity == pytest.approx((20.0, 0.0))

    # Kare 2 (kayıp #1): hız 20 * 0.8 = 16 px sağa taşır -> [176, 40, 186, 50]
    p1 = tracker.update([], frame_shape=frame_shape)
    assert p1 is not None and p1["is_predicted"] is True
    assert p1["bbox"][0] == pytest.approx(176.0)

    # Kare 3 (kayıp #2): hız 16 * 0.8 = 12.8 px -> [188.8, 40, 198.8, 50]
    p2 = tracker.update([], frame_shape=frame_shape)
    assert p2 is not None and p2["is_predicted"] is True
    assert p2["bbox"][0] == pytest.approx(188.8)

    # Kare 4 (kayıp #3): hız 10.24 px -> sınır 200'e kelepçelenir ([190, 40, 200, 50])
    p3 = tracker.update([], frame_shape=frame_shape)
    assert p3 is not None and p3["is_predicted"] is True
    assert p3["bbox"][2] <= 200.0
    assert p3["bbox"][0] >= 0.0

    # Kare 5 (kayıp #4 > max_missing_frames=3): None döner
    p4 = tracker.update([], frame_shape=frame_shape)
    assert p4 is None


def test_data_exporter_finetune_smoothing_backfill_and_ball_interp(tmp_path, monkeypatch):
    # Fine-tune sınıflarıyla export_distance_summary boş dönmemeli
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player", "goalkeeper", "referee"})
    monkeypatch.setattr(config, "SPEED_SMOOTHING_WINDOW_FRAMES", 5)
    monkeypatch.setattr(config, "MAX_REALISTIC_PLAYER_SPEED_KMH", 38.0)

    exporter = DataExporter(fps=25.0)
    # 50 kare boyunca player (id=1): ~7.5 m/s (27 km/h -> 0.30 m/frame) + +/-0.05 m jitter
    # Kare 25'te 1 karelik 15m sahte sıçrama ekleyip geri eski rotaya döndür veya zıplat
    for f in range(50):
        jitter = 0.05 if f % 2 == 0 else -0.05
        px = 10.0 + f * 0.30 + jitter
        if f >= 25:
            # 25. karede 12 m ani ışınlanma (gerçek dışı hız)
            px += 12.0
        team_val = None if f < 5 else "Team A"
        exporter.append_frame_data(f, [{
            "track_id": 1,
            "class_name": "player",
            "team": team_val,
            "confidence": 0.9,
            "is_predicted": False,
            "bbox": [100.0 + f, 100.0, 130.0 + f, 180.0],
            "pitch_x": round(px, 2),
            "pitch_y": 30.0,
            "on_pitch": True,
        }])

        # Top kare 0 ve kare 4'te gerçek tespit, arada 1-3 kare boşluk/tahmin
        if f == 0:
            exporter.append_frame_data(f, [{
                "track_id": "ball",
                "class_name": "ball",
                "team": None,
                "confidence": 0.85,
                "is_predicted": False,
                "bbox": [100.0, 200.0, 110.0, 210.0],
                "pitch_x": 20.0,
                "pitch_y": 30.0,
                "on_pitch": True,
            }])
        elif f == 2:
            # Var olan tahmin kaydı interpolasyonla güncellenmeli
            exporter.append_frame_data(f, [{
                "track_id": "ball",
                "class_name": "ball",
                "team": None,
                "confidence": 0.85,
                "is_predicted": True,
                "bbox": [100.0, 200.0, 110.0, 210.0],
                "pitch_x": 20.0,
                "pitch_y": 30.0,
                "on_pitch": True,
            }])
        elif f == 4:
            exporter.append_frame_data(f, [{
                "track_id": "ball",
                "class_name": "ball",
                "team": None,
                "confidence": 0.88,
                "is_predicted": False,
                "bbox": [140.0, 240.0, 150.0, 250.0],
                "pitch_x": 24.0,
                "pitch_y": 34.0,
                "on_pitch": True,
            }])

    post_stats = exporter.postprocess(team_map={1: "Team A"})
    assert post_stats["backfilled_teams"] == 5
    assert post_stats["interpolated_ball_frames"] == 3

    df_after = exporter.to_dataframe()
    assert df_after[df_after["track_id"] == 1]["team"].isna().sum() == 0

    ball_f2 = df_after[(df_after["class_name"] == "ball") & (df_after["frame_idx"] == 2)].iloc[0]
    assert ball_f2["pitch_x"] == pytest.approx(22.0)
    assert ball_f2["pitch_y"] == pytest.approx(32.0)
    assert bool(ball_f2["is_predicted"]) is True

    summary_path = str(tmp_path / "player_distance_summary.csv")
    out_csv = exporter.export_distance_summary(output_path=summary_path)
    assert out_csv is not None and os.path.exists(out_csv)

    sum_df = pd.read_csv(out_csv)
    assert len(sum_df) == 1
    row = sum_df.iloc[0]
    assert row["class_name"] == "player"
    assert row["max_speed_kmh"] <= 38.0
    # 12m ışınlanma filtrelendiği için toplam mesafe ~14.7m civarında olmalı (25+ değil)
    assert 10.0 <= row["total_distance_m"] <= 18.0
    assert row["sprint_m"] > 0.0
    assert row["sprint_count"] >= 1


def test_perspective_and_camera_motion():
    src = {
        "top_left_corner": (100.0, 100.0),
        "top_right_corner": (900.0, 100.0),
        "bottom_left_corner": (100.0, 500.0),
        "bottom_right_corner": (900.0, 500.0),
    }
    transformer = PerspectiveTransformer(source_points=src)
    assert transformer.is_calibrated is True

    center_m = transformer.transform_point(500.0, 300.0)
    assert center_m is not None
    assert center_m[0] == pytest.approx(52.5, abs=0.1)
    assert center_m[1] == pytest.approx(34.0, abs=0.1)

    # Sentetik dokulu arka plan oluştur ve 10 px sola kaydır (kamera sağa pan yapmış gibi)
    rng = np.random.default_rng(7)
    base_img = np.full((300, 400, 3), 80, dtype=np.uint8)
    for _ in range(120):
        cx = int(rng.integers(30, 370))
        cy = int(rng.integers(30, 270))
        color = tuple(int(c) for c in rng.integers(120, 255, size=3))
        cv2.rectangle(base_img, (cx - 4, cy - 4), (cx + 4, cy + 4), color, -1)

    shift_x = -10.0
    M_shift = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted_img = cv2.warpAffine(base_img, M_shift, (400, 300))

    tracker = CameraMotionTracker()
    m0 = tracker.update(base_img, [])
    assert m0 is not None
    m1 = tracker.update(shifted_img, [])
    assert m1 is not None

    # Shifted görüntüdeki (190, 150) noktası referans karedeki (200, 150)'ye dönmeli
    pt = cv2.perspectiveTransform(np.array([[[190.0, 150.0]]], dtype=np.float32), m1)[0][0]
    assert pt[0] == pytest.approx(200.0, abs=1.5)
    assert pt[1] == pytest.approx(150.0, abs=1.5)

    # Ani tamamen parlak/beyaz kare sahne kesmesi tetiklemeli
    cut_img = np.full((300, 400, 3), 255, dtype=np.uint8)
    m_cut = tracker.update(cut_img, [])
    assert m_cut is None
    assert tracker.scene_cut_detected is True


def test_match_analytics_synthetic_and_real_csv(tmp_path):
    # 1) Sentetik kalibre veri: paslaşma, top kaybı, takım şekli ve ısı haritası
    rows = []
    # Kare 0-9: Team A oyuncu 1 topa sahip (x=20->25)
    # Kare 10-19: Team A oyuncu 2 topa sahip (x=35->40) => completed_pass (1 -> 2)
    # Kare 20-29: Team B oyuncu 3 topa sahip (x=55->50) => turnover (2 -> 3)
    for f in range(30):
        if f < 10:
            bx, by = 20.0 + (f * 0.5), 34.0
            p1x, p2x, p3x = bx + 0.5, 35.0, 55.0
        elif f < 20:
            bx, by = 35.0 + ((f - 10) * 0.5), 24.0
            p1x, p2x, p3x = 25.0, bx + 0.4, 55.0
        else:
            bx, by = 55.0 - ((f - 20) * 0.5), 44.0
            p1x, p2x, p3x = 25.0, 40.0, bx - 0.5

        ts = round(f / 25.0, 3)
        rows.append({"frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False, "pixel_x": bx * 10, "pixel_y": by * 10, "pitch_x": bx, "pitch_y": by, "on_pitch": True})
        rows.append({"frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": "person", "team": "Team A", "is_predicted": False, "pixel_x": p1x * 10, "pixel_y": 340.0, "pitch_x": p1x, "pitch_y": 34.0, "on_pitch": True})
        rows.append({"frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": "person", "team": "Team A", "is_predicted": False, "pixel_x": p2x * 10, "pixel_y": 240.0, "pitch_x": p2x, "pitch_y": 24.0, "on_pitch": True})
        rows.append({"frame_idx": f, "timestamp_sec": ts, "track_id": 3, "class_name": "person", "team": "Team B", "is_predicted": False, "pixel_x": p3x * 10, "pixel_y": 440.0, "pitch_x": p3x, "pitch_y": 44.0, "on_pitch": True})

    syn_out = tmp_path / "syn_analytics"
    analyzer = MatchAnalyzer(pd.DataFrame(rows), fps=25.0)
    assert analyzer.is_calibrated is True
    summary = analyzer.export_all(output_dir=str(syn_out), generate_heatmaps=True)

    assert summary["possession"]["mode"] == "pitch_meters"
    assert summary["possession"]["team_a_pct"] > summary["possession"]["team_b_pct"]
    assert summary["passes"]["total_completed_passes"] == 1
    assert summary["passes"]["total_turnovers"] == 1
    assert summary["team_shape"]["teams"]["Team A"]["width"] == pytest.approx(10.0, abs=0.1)

    assert os.path.exists(summary["exports"]["analytics_summary_json"])
    assert os.path.exists(summary["exports"]["possession_timeline_csv"])
    assert len(summary["exports"]["heatmaps"]) >= 2
    for img_path in summary["exports"]["heatmaps"].values():
        assert os.path.exists(img_path)

    # 2) Gerçek repo çıktısı data/exports/tracking_data.csv (pixel_fallback modu)
    real_csv = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    real_analyzer = MatchAnalyzer.from_csv(real_csv)
    assert real_analyzer.is_calibrated is False

    real_out = tmp_path / "real_analytics"
    real_summary = real_analyzer.export_all(output_dir=str(real_out), generate_heatmaps=True)
    assert real_summary["possession"]["mode"] == "pixel_fallback"
    assert real_summary["possession"]["ball_detected_frames"] > 0
    with open(real_summary["exports"]["analytics_summary_json"], "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["fps"] == pytest.approx(25.0)


def test_detector_gpu_cpu_config(monkeypatch):
    predict_calls = []

    class FakeTensor:
        def __init__(self, vals):
            self._vals = vals

        def tolist(self):
            return self._vals

    class FakeBox:
        def __init__(self, cls_id, xyxy, conf):
            self.cls = [cls_id]
            self.xyxy = [FakeTensor(xyxy)]
            self.conf = [conf]

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            src = kwargs["source"]
            n = len(src) if isinstance(src, list) else 1
            return [SimpleNamespace(boxes=[FakeBox(0, [10.0, 20.0, 30.0, 60.0], 0.91)]) for _ in range(n)]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    # CPU modu: half mutlaka False olmalı
    det_cpu = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cpu",
        imgsz=640,
        half=True,
    )
    assert det_cpu.device == "cpu"
    assert det_cpu.half is False
    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det_cpu.detect_frame(dummy_frame)
    assert len(out) == 1 and out[0]["class_name"] == config.TARGET_CLASS_IDS[0]
    assert predict_calls[-1]["half"] is False
    assert predict_calls[-1]["imgsz"] == 640
    assert predict_calls[-1]["classes"] == list(config.TARGET_CLASS_IDS.keys())

    # CUDA simülasyonu: half True olmalı ve detect_batch doğru çalışmalı
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: True)
    det_cuda = detector_module.ObjectDetector(
        model_path="dummy.pt",
        device="cuda",
        imgsz=960,
        half=True,
    )
    assert det_cuda.device == "cuda"
    assert det_cuda.half is True
    batch_out = det_cuda.detect_batch([dummy_frame, dummy_frame])
    assert len(batch_out) == 2
    assert predict_calls[-1]["half"] is True
    assert predict_calls[-1]["imgsz"] == 960
    assert predict_calls[-1]["classes"] == list(config.TARGET_CLASS_IDS.keys())
