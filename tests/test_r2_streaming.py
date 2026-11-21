import csv
import hashlib
import os
import sqlite3
import sys
import tracemalloc
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
import modules.tracker as tracker_module
from modules.data_exporter import DataExporter
from modules.team_classifier import TeamClassifier
from modules.tracker import BallTracker, PlayerTracker
from modules.video_processor import VideoProcessor


def _sha256_or_none(path: str):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_synthetic_video(path: str, n_frames: int = 5, width: int = 64, height: int = 48, fps: float = 25.0):
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter açılamadı: {path}"
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), (20, 100 + i * 10, 40), dtype=np.uint8)
            cv2.rectangle(frame, (5 + i * 3, 10), (20 + i * 3, 35), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _make_jersey_frame(players_spec):
    frame = np.full((600, 1000, 3), (35, 140, 35), dtype=np.uint8)
    for spec in players_spec:
        x1, y1, x2, y2 = map(int, spec["bbox"])
        bw = x2 - x1
        bh = y2 - y1
        jy1 = y1 + int(bh * 0.22)
        jy2 = y1 + int(bh * 0.50)
        jx1 = x1 + int(bw * 0.30)
        jx2 = x1 + int(bw * 0.70)
        frame[jy1:jy2, jx1:jx2] = spec["bgr"]
    return frame


# ====================================================================
# 1. PlayerTracker & BallTracker Testleri
# ====================================================================


def test_player_tracker_deepsort_inference_mode_and_cut_reset(monkeypatch):
    monkeypatch.setattr(
        config,
        "HUMAN_CLASS_NAMES",
        {"player", "goalkeeper", "referee"},
        raising=False,
    )
    inference_states = []

    class FakeTrack:
        def __init__(self, ltwh, track_id, det_class, det_conf):
            self._ltwh = list(ltwh)
            self.track_id = str(track_id)
            self.det_class = det_class
            self.det_conf = det_conf
            self.time_since_update = 0
            self.features = [np.ones(4, dtype=np.float32)]

        def is_confirmed(self):
            return True

        def to_ltwh(self):
            return self._ltwh

        def get_det_class(self):
            return self.det_class

        def get_det_conf(self):
            return self.det_conf

    class FakeDeepSort:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tracker = SimpleNamespace(
                tracks=[],
                del_tracks_ids=[],
                _next_id=1,
                metric=SimpleNamespace(samples={}),
            )

        def update_tracks(self, raw_detections, embeds=None, frame=None):
            inference_states.append(torch.is_inference_mode_enabled())
            tracks = []
            for ltwh, conf, cls_name in raw_detections:
                tid = self.tracker._next_id
                self.tracker._next_id += 1
                t = FakeTrack(ltwh, tid, cls_name, conf)
                tracks.append(t)
                self.tracker.metric.samples[str(tid)] = [np.ones(4, dtype=np.float32)]
            self.tracker.tracks = tracks
            return tracks

    monkeypatch.setattr(tracker_module, "DeepSort", FakeDeepSort)

    tracker = PlayerTracker(tracker_type="deepsort", embedder_gpu=False)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [
        {"bbox": [10.0, 10.0, 30.0, 60.0], "confidence": 0.91, "class_name": "player"},
        {"bbox": [40.0, 10.0, 60.0, 60.0], "confidence": 0.85, "class_name": "goalkeeper"},
        {"bbox": [70.0, 10.0, 85.0, 60.0], "confidence": 0.88, "class_name": "referee"},
        {"bbox": [50.0, 50.0, 58.0, 58.0], "confidence": 0.95, "class_name": "ball"},
    ]
    out = tracker.update(dets, frame=frame)
    assert inference_states == [True]
    assert [o["class_name"] for o in out] == ["player", "goalkeeper", "referee"]
    assert [o["track_id"] for o in out] == [1, 2, 3]

    # Sahne kesmesi: aktif track'ler ve metric.samples temizlenmeli, _next_id korunmalı
    tracker.on_scene_cut()
    assert tracker.tracker.tracker.tracks == []
    assert tracker.tracker.tracker.metric.samples == {}
    assert tracker.tracker.tracker._next_id == 4

    out_after_cut = tracker.update(
        [{"bbox": [10.0, 10.0, 30.0, 60.0], "confidence": 0.9, "class_name": "player"}],
        frame=frame,
    )
    assert out_after_cut[0]["track_id"] == 4

    # reset_on_cut() takma adı da aynı davranışı sergilemeli
    tracker.reset_on_cut()
    assert tracker.tracker.tracker.tracks == []
    assert tracker.tracker.tracker._next_id == 5


def test_player_tracker_deepsort_embedding_stride(monkeypatch):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player"}, raising=False)
    from deep_sort_realtime.deep_sort.detection import Detection
    from deep_sort_realtime.deep_sort.kalman_filter import KalmanFilter
    from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric
    from deep_sort_realtime.deep_sort.tracker import Tracker

    embed_calls = []
    full_update_calls = []

    class FakeEmbedderDeepSort:
        def __init__(self, **kwargs):
            metric = NearestNeighborDistanceMetric("cosine", 0.3, 100)
            self.tracker = Tracker(metric, max_age=kwargs.get("max_age", 30), n_init=kwargs.get("n_init", 2))
            self.tracker.kf = KalmanFilter()

        def generate_embeds(self, frame, raw_dets):
            embed_calls.append((len(raw_dets), torch.is_inference_mode_enabled()))
            return [np.ones(8, dtype=np.float32) for _ in raw_dets]

        def create_detections(self, raw_dets, embeds):
            return [
                Detection(ltwh, conf, emb, class_name=cls_name)
                for (ltwh, conf, cls_name), emb in zip(raw_dets, embeds)
            ]

        def update_tracks(self, raw_detections, embeds=None, frame=None):
            full_update_calls.append((len(raw_detections), torch.is_inference_mode_enabled()))
            if embeds is None:
                embeds = self.generate_embeds(frame, raw_detections)
            dets = self.create_detections(raw_detections, embeds)
            self.tracker.predict()
            self.tracker.update(dets)
            return self.tracker.tracks

    monkeypatch.setattr(tracker_module, "DeepSort", FakeEmbedderDeepSort)

    tracker = PlayerTracker(n_init=2, max_age=10, tracker_type="deepsort", embedding_stride=3, embedder_gpu=False)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    p1 = {"bbox": [10.0, 10.0, 30.0, 60.0], "confidence": 0.9, "class_name": "player"}
    p2 = {"bbox": [60.0, 10.0, 80.0, 60.0], "confidence": 0.85, "class_name": "player"}

    # Kare 1 (stride): 1 tespit -> full update_tracks + 1 embed
    tracker.update([p1], frame)
    assert len(full_update_calls) == 1
    assert full_update_calls[-1] == (1, True)
    assert len(embed_calls) == 1

    # Kare 2 (non-stride): p1 eşleşir (embed yok), p2 yeni eşleşmeyen -> sadece p2 için 1 embed
    out2 = tracker.update([p1, p2], frame)
    assert len(full_update_calls) == 1  # non-stride path
    assert len(embed_calls) == 2
    assert embed_calls[-1] == (1, True)
    assert len(out2) == 1 and out2[0]["track_id"] == 1
    assert "1" in tracker.tracker.tracker.metric.samples

    # Kare 3 (non-stride): hem p1 hem p2 eşleşir -> 0 embed çağrısı!
    out3 = tracker.update([p1, p2], frame)
    assert len(embed_calls) == 2
    assert sorted(o["track_id"] for o in out3) == [1, 2]
    assert "2" in tracker.tracker.tracker.metric.samples

    # Kare 4 (stride): full update_tracks sorunsuz çalışır (boş feature IndexError olmamalı)
    out4 = tracker.update([p1, p2], frame)
    assert len(full_update_calls) == 2
    assert sorted(o["track_id"] for o in out4) == [1, 2]


def test_player_tracker_bytetrack_schema_classes_and_cut_reset(monkeypatch):
    monkeypatch.setattr(
        config,
        "HUMAN_CLASS_NAMES",
        {"player", "goalkeeper", "referee"},
        raising=False,
    )
    monkeypatch.setattr(config, "TRACKER_TYPE", "kalman_iou", raising=False)

    tracker = PlayerTracker(max_age=5, n_init=2)
    assert tracker.tracker_type == "bytetrack"

    dets_0 = [
        {"bbox": [10.0, 20.0, 40.0, 90.0], "confidence": 0.9, "class_name": "player"},
        {"bbox": [100.0, 20.0, 130.0, 90.0], "confidence": 0.85, "class_name": "goalkeeper"},
        {"bbox": [200.0, 20.0, 230.0, 90.0], "confidence": 0.80, "class_name": "referee"},
        {"bbox": [50.0, 50.0, 60.0, 60.0], "confidence": 0.95, "class_name": "ball"},
    ]
    # n_init=2 -> ilk karede henüz onaylı çıktı dönmemeli
    out0 = tracker.update(dets_0)
    assert out0 == []

    # 2. karede 3 insan sınıfı onaylanır, top hariç tutulur
    dets_1 = [
        {"bbox": [12.0, 20.0, 42.0, 90.0], "confidence": 0.92, "class_name": "player"},
        {"bbox": [102.0, 20.0, 132.0, 90.0], "confidence": 0.87, "class_name": "goalkeeper"},
        {"bbox": [202.0, 20.0, 232.0, 90.0], "confidence": 0.83, "class_name": "referee"},
    ]
    out1 = tracker.update(dets_1)
    assert len(out1) == 3
    for item in out1:
        assert set(item.keys()) == {"bbox", "track_id", "class_name", "confidence", "team", "is_predicted"}
        assert isinstance(item["track_id"], int)
        assert item["is_predicted"] is False
        assert item["team"] is None
    assert {o["class_name"] for o in out1} == {"player", "goalkeeper", "referee"}

    # 3. kare: oyuncu geçici olarak kaybolsun (0 tespit) -> tahmin edilen kutu dönmeli
    out2 = tracker.update([])
    assert len(out2) == 3
    assert all(o["is_predicted"] is True and o["confidence"] is None for o in out2)

    # Sahne kesmesi: hayalet tahminler silinmeli ve ID sayacı sıfırlanmamalı
    prev_max_id = max(o["track_id"] for o in out2)
    tracker.on_scene_cut()
    assert tracker.update([]) == []

    # Kesme sonrası yeniden onaylanan track eski ID'lerden büyük yeni ID almalı
    tracker.update([{"bbox": [10.0, 20.0, 40.0, 90.0], "confidence": 0.9, "class_name": "player"}])
    out_new = tracker.update([{"bbox": [11.0, 20.0, 41.0, 90.0], "confidence": 0.9, "class_name": "player"}])
    assert len(out_new) == 1
    assert out_new[0]["track_id"] > prev_max_id


def test_ball_tracker_on_scene_cut_clears_state():
    tracker = BallTracker(max_missing_frames=5, velocity_damping=0.8, max_velocity_px=100.0)
    tracker.update([{"class_name": "ball", "bbox": [10.0, 10.0, 20.0, 20.0], "confidence": 0.9}])
    tracker.update([{"class_name": "ball", "bbox": [30.0, 10.0, 40.0, 20.0], "confidence": 0.9}])
    assert tracker.velocity == pytest.approx((20.0, 0.0))

    tracker.on_scene_cut()
    assert tracker.last_known_detection is None
    assert tracker.velocity == (0.0, 0.0)
    assert tracker.update([]) is None

    tracker.update([{"class_name": "ball", "bbox": [50.0, 50.0, 60.0, 60.0], "confidence": 0.9}])
    tracker.reset_on_cut()
    assert tracker.last_known_detection is None
    assert tracker.update([]) is None


# ====================================================================
# 2. TeamClassifier Bellek Sınırlandırma Testi
# ====================================================================


def test_team_classifier_purges_color_history_on_lock(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", False, raising=False)
    monkeypatch.setattr(config, "TEAM_COLOR_HISTORY_MAX_UNLOCKED_TRACKS", 64, raising=False)

    players_spec = []
    for i in range(5):
        players_spec.append({
            "track_id": i + 1,
            "class_name": "person",
            "bbox": [40 + i * 70, 80, 90 + i * 70, 220],
            "bgr": (150, 30, 20),
        })
    for i in range(5):
        players_spec.append({
            "track_id": i + 6,
            "class_name": "person",
            "bbox": [40 + i * 70, 280, 90 + i * 70, 420],
            "bgr": (245, 245, 245),
        })

    frame = _make_jersey_frame(players_spec)
    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, color_space="LAB", min_samples_for_lock=3)

    dets = [
        {"track_id": s["track_id"], "class_name": "person", "bbox": s["bbox"], "team": None}
        for s in players_spec
    ]
    clf.assign_teams(dets, frame)
    assert clf.is_fitted is True
    assert len(clf.team_assignment) == 10
    # İlk K-Means fit sonrası tüm kilitlenen track'lerin color_history'si silinmiş olmalı
    assert len(clf.color_history) == 0

    # Fit sonrası 250 yeni track: min_samples_for_lock=3 sonrası hepsi kilitlenip temizlenmeli
    for tid in range(100, 350):
        det = {"track_id": tid, "class_name": "person", "bbox": [40, 80, 90, 220], "team": None}
        for _ in range(3):
            clf.assign_teams([dict(det)], frame)
        assert tid in clf.team_assignment
        assert tid not in clf.color_history

    # Tek karelik gürültü track'leri bile üst sınırla (64) sınırlı kalmalı
    for tid in range(1000, 1200):
        clf.assign_teams(
            [{"track_id": tid, "class_name": "person", "bbox": [40, 280, 90, 420], "team": None}],
            frame,
        )
    assert len(clf.color_history) <= 64
    team_map = clf.get_team_map()
    assert 1 in team_map and 100 in team_map


# ====================================================================
# 3. VideoProcessor Testleri
# ====================================================================


def test_video_processor_output_none_and_empty(tmp_path):
    in_mp4 = str(tmp_path / "sample_in.mp4")
    _create_synthetic_video(in_mp4, n_frames=5)

    for out_arg in (None, ""):
        with VideoProcessor(in_mp4, output_path=out_arg) as vp:
            assert vp.output_path is None
            assert vp.writer is None
            assert vp.has_writer is False

            frames = list(vp.read_frames())
            assert len(frames) == 5
            for _, frm in frames:
                assert vp.write_frame(frm) is False

            # İçeride release + __exit__ çifte release sorunsuz olmalı
            vp.release()
            assert vp.has_writer is False


def test_video_processor_bare_filename_output(tmp_path, monkeypatch):
    in_mp4 = str(tmp_path / "sample_in.mp4")
    _create_synthetic_video(in_mp4, n_frames=5)
    monkeypatch.chdir(tmp_path)

    with VideoProcessor(in_mp4, output_path="bare.mp4") as vp:
        assert vp.has_writer is True
        assert vp.writer is not None
        for _, frm in vp.read_frames():
            assert vp.write_frame(frm) is True

    assert vp.has_writer is False
    assert vp.writer is None
    assert vp.write_frame(np.zeros((48, 64, 3), dtype=np.uint8)) is False

    out_file = tmp_path / "bare.mp4"
    assert out_file.exists() and out_file.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_file))
    try:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        assert count == 5
    finally:
        cap.release()


def test_video_processor_nested_dir_early_break_and_guards(tmp_path):
    in_mp4 = str(tmp_path / "sample_in.mp4")
    _create_synthetic_video(in_mp4, n_frames=6)
    out_mp4 = tmp_path / "sub" / "dir" / "out.mp4"

    with VideoProcessor(in_mp4, output_path=str(out_mp4)) as vp:
        seen = 0
        for _, frm in vp.read_frames():
            copy_img = vp.draw_detections(frm, [], inplace=False)
            assert not np.shares_memory(copy_img, frm)

            inplace_img = vp.draw_detections(frm, [], inplace=True)
            assert inplace_img is frm

            assert vp.overlay_minimap(frm, None) is frm
            vp.write_frame(inplace_img)
            seen += 1
            if seen == 2:
                break

    assert out_mp4.exists()


# ====================================================================
# 4. DataExporter Akış, Sınıflar ve 135k-Kare Stres Testleri
# ====================================================================


def test_flush_interval_zero_backward_compatibility(tmp_path):
    for idx, kwargs in enumerate(({}, {"flush_interval": 0})):
        in_dir = tmp_path / f"custom_out_{idx}"
        exp = DataExporter(fps=25.0, output_dir=str(in_dir), **kwargs)
        for f in range(50):
            exp.append_frame_data(f, [{
                "track_id": 1,
                "class_name": "player",
                "team": "Team A",
                "confidence": 0.9,
                "is_predicted": False,
                "bbox": [10.0 + f, 20.0, 30.0 + f, 80.0],
                "pitch_x": 20.0 + f * 0.2,
                "pitch_y": 30.0,
                "on_pitch": True,
            }])
        exp.postprocess()
        assert not (in_dir / config.EXPORT_CSV_FILENAME).exists()
        assert not (in_dir / config.EXPORT_SQL_FILENAME).exists()

        df = exp.to_dataframe()
        assert len(df) == 50
        assert tuple(df.columns) == DataExporter.COLUMNS

        csv_out = exp.export_to_csv()
        sql_out = exp.export_to_sql()
        assert os.path.dirname(csv_out) == str(in_dir)
        assert os.path.dirname(sql_out) == str(in_dir)
        assert os.path.exists(csv_out) and os.path.exists(sql_out)


def test_streaming_vs_nonstreaming_exact_equivalence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player", "goalkeeper", "referee"}, raising=False)
    monkeypatch.setattr(config, "TEAM_CLASSIFIER_BACKFILL_TEAMS", True, raising=False)

    stream_dir = tmp_path / "stream"
    mem_dir = tmp_path / "mem"
    stream_dir.mkdir()
    mem_dir.mkdir()

    exp_mem = DataExporter(fps=25.0, output_dir=str(mem_dir), flush_interval=0)
    exp_str = DataExporter(
        fps=25.0,
        output_dir=str(stream_dir),
        flush_interval=20,
        max_ball_gap=15,
    )

    with exp_str:
        for f in range(120):
            frame_dets = []
            # track_id=1: kare 0..19 team=None (fit öncesi), 20+ Team A
            frame_dets.append({
                "track_id": 1,
                "class_name": "player",
                "team": None if f < 20 else "Team A",
                "confidence": np.float32(0.92),
                "is_predicted": np.bool_(False),
                "bbox": [100.0 + f, 100.0, 130.0 + f, 180.0],
                "pitch_x": np.float32(10.0 + f * 0.25),
                "pitch_y": np.float32(34.0),
                "on_pitch": np.bool_(True),
            })
            # track_id=2: goalkeeper, kare 20..30 team=None -> 45+ Goalkeeper (flush sonrası SQL/CSV patch testi)
            if f >= 20:
                frame_dets.append({
                    "track_id": 2,
                    "class_name": "goalkeeper",
                    "team": None if f < 45 else "Goalkeeper",
                    "confidence": 0.88,
                    "is_predicted": False,
                    "bbox": [20.0, 200.0, 50.0, 280.0],
                    "pitch_x": 5.0 + (f - 20) * 0.05,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                })
            # track_id=3: on_pitch=False ve legacy 'person' sınıfı testi
            frame_dets.append({
                "track_id": 3,
                "class_name": "person",
                "team": "Referee",
                "confidence": 0.80,
                "is_predicted": False,
                "bbox": [300.0, 50.0, 320.0, 120.0],
                "pitch_x": 50.0 + f * 0.1,
                "pitch_y": 10.0 if f % 10 != 0 else -10.0,
                "on_pitch": f % 10 != 0,
            })

            # Top: kare 18 (real) -> 19..24 (flush sınırı f=20'yi aşan boşluk!) -> kare 25 (real)
            # Ayrıca kare 60 (real) -> 61..85 (gap=25 > 15 -> interpole edilmemeli) -> kare 86 (real)
            if f in {0, 18, 25, 40, 60, 86, 110}:
                frame_dets.append({
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.87,
                    "is_predicted": False,
                    "bbox": [200.0 + f, 200.0, 210.0 + f, 210.0],
                    "pitch_x": 20.0 + f * 0.3,
                    "pitch_y": 34.0,
                    "on_pitch": True,
                })
            elif f in {20, 22}:
                frame_dets.append({
                    "track_id": "ball",
                    "class_name": "ball",
                    "team": None,
                    "confidence": 0.6,
                    "is_predicted": True,
                    "bbox": [190.0, 190.0, 200.0, 200.0],
                    "pitch_x": 19.0,
                    "pitch_y": 30.0,
                    "on_pitch": True,
                })

            exp_mem.append_frame_data(f, frame_dets)
            exp_str.append_frame_data(f, frame_dets)

        mem_stats = exp_mem.postprocess()
        str_stats = exp_str.postprocess()

    assert str_stats == mem_stats
    assert len(exp_str.records) == 0

    mem_csv = exp_mem.export_to_csv()
    str_csv = exp_str.export_to_csv()
    mem_sql = exp_mem.export_to_sql()
    str_sql = exp_str.export_to_sql()

    # İdempotency: postprocess sonrası export_to_csv / export_to_sql üzerine yazıp bozmamalı
    assert os.path.abspath(str_csv) == os.path.abspath(str(stream_dir / config.EXPORT_CSV_FILENAME))
    assert os.path.abspath(str_sql) == os.path.abspath(str(stream_dir / config.EXPORT_SQL_FILENAME))

    df_mem_csv = pd.read_csv(mem_csv)
    df_str_csv = pd.read_csv(str_csv)
    pd.testing.assert_frame_equal(df_mem_csv, df_str_csv, check_dtype=False)

    # SQLite BLOB yok, indeksler mevcut ve satırlar denk
    conn = sqlite3.connect(str_sql)
    try:
        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert {"idx_track_id", "idx_frame_idx"}.issubset(indexes)

        sample_row = conn.execute(
            f"SELECT frame_idx, pitch_x, is_predicted, on_pitch FROM {config.EXPORT_SQL_TABLE_NAME} LIMIT 1"
        ).fetchone()
        assert all(not isinstance(val, (bytes, bytearray)) for val in sample_row)

        df_str_sql = pd.read_sql_query(f"SELECT * FROM {config.EXPORT_SQL_TABLE_NAME}", conn)
    finally:
        conn.close()

    conn_mem = sqlite3.connect(mem_sql)
    try:
        df_mem_sql = pd.read_sql_query(f"SELECT * FROM {config.EXPORT_SQL_TABLE_NAME}", conn_mem)
    finally:
        conn_mem.close()

    pd.testing.assert_frame_equal(df_mem_sql, df_str_sql, check_dtype=False)

    # Mesafe özeti eşitliği
    mem_dist = exp_mem.export_distance_summary()
    str_dist = exp_str.export_distance_summary()
    assert mem_dist is not None and str_dist is not None
    pd.testing.assert_frame_equal(pd.read_csv(mem_dist), pd.read_csv(str_dist))


def test_distance_summary_accepts_four_class_and_legacy_person(tmp_path, monkeypatch):
    # 4 sınıflı modda player/goalkeeper/referee ve legacy "person" satırları birlikte özetlenebilmeli;
    # COCO modunda ("person") ise "person" özetlenebilmeli.
    cases = [
        ({"player", "goalkeeper", "referee"}, {"player", "goalkeeper", "referee", "person"}),
        ({"person"}, {"person"}),
    ]
    for idx, (human_set, expected_classes) in enumerate(cases):
        monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", human_set, raising=False)
        exp = DataExporter(fps=25.0, output_dir=str(tmp_path))
        for f in range(10):
            exp.append_frame_data(f, [
                {
                    "track_id": 1,
                    "class_name": "player",
                    "team": "Team A",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [10.0, 10.0, 30.0, 60.0],
                    "pitch_x": 10.0 + f * 0.2,
                    "pitch_y": 20.0,
                    "on_pitch": True,
                },
                {
                    "track_id": 2,
                    "class_name": "goalkeeper",
                    "team": "Goalkeeper",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [20.0, 10.0, 40.0, 60.0],
                    "pitch_x": 15.0 + f * 0.2,
                    "pitch_y": 22.0,
                    "on_pitch": True,
                },
                {
                    "track_id": 3,
                    "class_name": "referee",
                    "team": "Referee",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [30.0, 10.0, 50.0, 60.0],
                    "pitch_x": 20.0 + f * 0.2,
                    "pitch_y": 24.0,
                    "on_pitch": True,
                },
                {
                    "track_id": 4,
                    "class_name": "person",
                    "team": "Team B",
                    "confidence": 0.9,
                    "is_predicted": False,
                    "bbox": [50.0, 10.0, 70.0, 60.0],
                    "pitch_x": 30.0 + f * 0.2,
                    "pitch_y": 25.0,
                    "on_pitch": True,
                },
            ])
        out_path = str(tmp_path / f"dist_{idx}.csv")
        res = exp.export_distance_summary(output_path=out_path)
        assert res is not None
        df = pd.read_csv(res)
        assert set(df["class_name"]) == expected_classes


def test_streaming_csv_only_and_sql_only_modes(tmp_path):
    # 1) Yalnızca CSV akış modu
    csv_dir = tmp_path / "csv_only"
    exp_csv = DataExporter(
        fps=25.0,
        output_dir=str(csv_dir),
        flush_interval=10,
        enable_csv=True,
        enable_sql=False,
    )
    for f in range(30):
        exp_csv.append_frame_data(f, [{
            "track_id": 1,
            "class_name": "player",
            "team": "Team A",
            "confidence": 0.9,
            "is_predicted": False,
            "bbox": [10.0, 10.0, 30.0, 60.0],
            "pitch_x": 10.0 + f * 0.2,
            "pitch_y": 30.0,
            "on_pitch": True,
        }])
    exp_csv.postprocess()
    assert (csv_dir / config.EXPORT_CSV_FILENAME).exists()
    assert not (csv_dir / config.EXPORT_SQL_FILENAME).exists()

    dist_csv = exp_csv.export_distance_summary()
    assert dist_csv is not None and os.path.exists(dist_csv)
    sql_after = exp_csv.export_to_sql()
    assert os.path.exists(sql_after)

    # 2) Yalnızca SQLite akış modu + kısa video (< flush_interval)
    sql_dir = tmp_path / "sql_only"
    exp_sql = DataExporter(
        fps=25.0,
        output_dir=str(sql_dir),
        flush_interval=2500,
        enable_csv=False,
        enable_sql=True,
    )
    for f in range(15):
        exp_sql.append_frame_data(f, [{
            "track_id": 5,
            "class_name": "goalkeeper",
            "team": "Goalkeeper",
            "confidence": 0.9,
            "is_predicted": False,
            "bbox": [10.0, 10.0, 30.0, 60.0],
            "pitch_x": 5.0 + f * 0.1,
            "pitch_y": 34.0,
            "on_pitch": True,
        }])
    exp_sql.postprocess()
    assert (sql_dir / config.EXPORT_SQL_FILENAME).exists()
    assert not (sql_dir / config.EXPORT_CSV_FILENAME).exists()

    csv_after = exp_sql.export_to_csv()
    assert os.path.exists(csv_after)
    assert len(pd.read_csv(csv_after)) == 15


def test_streaming_90min_135k_frames_stress_and_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"player", "goalkeeper", "referee"}, raising=False)

    # Mevcut örnek veri dosyalarının SHA-256 değerlerini kaydet (asla değişmemeli)
    real_csv = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    real_db = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)
    before_csv_hash = _sha256_or_none(real_csv)
    before_db_hash = _sha256_or_none(real_db)

    stress_dir = tmp_path / "stress_90min"
    stress_dir.mkdir()
    csv_path = str(stress_dir / "tracking_90m.csv")
    sql_path = str(stress_dir / "tracking_90m.db")
    dist_path = str(stress_dir / "distance_summary_90m.csv")

    n_frames = 135_000
    n_players = 19  # 18 player + 1 goalkeeper + 1 ball = 20 tespit/kare -> 2.7M satır

    # Nesne tahsis yükünü azaltmak için şablon tespitleri önceden hazırla
    player_templates = []
    for pid in range(1, n_players + 1):
        cls_name = "goalkeeper" if pid == 19 else "player"
        team_name = "Goalkeeper" if pid == 19 else ("Team A" if pid <= 10 else "Team B")
        player_templates.append((pid, cls_name, team_name))

    tracemalloc.start()
    try:
        with DataExporter(
            fps=25.0,
            flush_interval=2500,
            csv_path=csv_path,
            sql_path=sql_path,
            max_ball_gap=15,
        ) as exp:
            for f in range(n_frames):
                dets = []
                # İlk 15 karede team=None (fit öncesi backfill testi)
                is_prefit = f < 15
                dx = (f % 400) * 0.15 if (f // 400) % 2 == 0 else (400 - (f % 400)) * 0.15
                for pid, cls_name, team_name in player_templates:
                    dets.append({
                        "track_id": pid,
                        "class_name": cls_name,
                        "team": None if is_prefit else team_name,
                        "confidence": 0.9,
                        "is_predicted": False,
                        "bbox": [50.0 + pid * 20.0, 100.0, 75.0 + pid * 20.0, 170.0],
                        "pitch_x": 10.0 + pid * 3.0 + dx * 0.2,
                        "pitch_y": 15.0 + (pid % 5) * 8.0,
                        "on_pitch": True,
                    })

                # Her 500 karede 5 karelik top boşluğu (interpolasyon akış testi)
                mod = f % 500
                if not (100 < mod <= 105):
                    dets.append({
                        "track_id": "ball",
                        "class_name": "ball",
                        "team": None,
                        "confidence": 0.88,
                        "is_predicted": False,
                        "bbox": [300.0, 200.0, 312.0, 212.0],
                        "pitch_x": 52.5 + (mod * 0.05),
                        "pitch_y": 34.0,
                        "on_pitch": True,
                    })

                exp.append_frame_data(f, dets)

            stats = exp.postprocess()
            _, peak_stream_bytes = tracemalloc.get_traced_memory()

            assert len(exp.records) == 0
            assert stats["backfilled_teams"] == 15 * n_players
            assert stats["interpolated_ball_frames"] == (n_frames // 500) * 5

        tracemalloc.reset_peak()
        out_dist = exp.export_distance_summary(output_path=dist_path)
        _, peak_dist_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_stream_bytes < 350 * 1024 * 1024, f"Streaming RAM çok yüksek: {peak_stream_bytes / 1e6:.1f} MB"
    assert peak_dist_bytes < 150 * 1024 * 1024, f"Distance summary RAM çok yüksek: {peak_dist_bytes / 1e6:.1f} MB"

    # SQLite doğrulaması
    expected_rows = n_frames * n_players + n_frames  # interpolasyon sonrası her karede 1 top
    conn = sqlite3.connect(sql_path)
    try:
        sql_count = conn.execute(f"SELECT COUNT(*) FROM {config.EXPORT_SQL_TABLE_NAME}").fetchone()[0]
        assert sql_count == expected_rows

        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert {"idx_track_id", "idx_frame_idx"}.issubset(indexes)

        null_teams = conn.execute(
            f"SELECT COUNT(*) FROM {config.EXPORT_SQL_TABLE_NAME} WHERE class_name != 'ball' AND team IS NULL"
        ).fetchone()[0]
        assert null_teams == 0

        sample = conn.execute(
            f"SELECT frame_idx, pitch_x, is_predicted, on_pitch FROM {config.EXPORT_SQL_TABLE_NAME} LIMIT 1"
        ).fetchone()
        assert all(not isinstance(v, (bytes, bytearray)) for v in sample)
    finally:
        conn.close()

    # CSV satır sayısı doğrulaması (belleğe yüklemeden)
    with open(csv_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    assert line_count == expected_rows + 1

    # Mesafe özeti doğrulaması
    assert out_dist is not None and os.path.exists(out_dist)
    dist_df = pd.read_csv(out_dist)
    assert len(dist_df) == n_players
    assert (dist_df["total_distance_m"] > 0).all()
    assert (dist_df["max_speed_kmh"] <= 38.0).all()

    # Veri güvenliği: örnek dosyalar kesinlikle değişmemiş olmalı
    assert _sha256_or_none(real_csv) == before_csv_hash
    assert _sha256_or_none(real_db) == before_db_hash
