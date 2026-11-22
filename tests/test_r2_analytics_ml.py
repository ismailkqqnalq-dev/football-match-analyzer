import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
import tracemalloc

import cv2
import numpy as np
import pandas as pd
import psutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.match_analytics import MatchAnalyzer
from modules.ml_analyzer import MLAnalyzer


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_json_load(path: str):
    text = open(path, "r", encoding="utf-8").read()
    return json.loads(
        text,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {c}")),
    )


def _write_df_to_sql(df: pd.DataFrame, db_path: str, table_name: str = config.EXPORT_SQL_TABLE_NAME) -> None:
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def _make_controlled_match_df(use_four_class: bool = True) -> pd.DataFrame:
    """
    MatchAnalyzer ve MLAnalyzer parite/kenar-durum testleri için kontrollü,
    kalibre edilmiş çok-oyunculu veri seti üretir.
    """
    p_cls = "player" if use_four_class else "person"
    gk_cls = "goalkeeper" if use_four_class else "person"
    ref_cls = "referee" if use_four_class else "person"
    rows = []

    # 46 karelik kontrollü senaryo (gerçekçi çok-kareli pas uçuşları ile):
    # - Kare 0-4: Oyuncu 1 (Team A) topa sahip; kare 2'de top yok (köprü testi), kare 0'da duplicate ball (is_predicted=False kazanmalı)
    # - Kare 5-16: Top 1 -> 2 arasında süzülüyor
    # - Kare 17-21: Oyuncu 2 (Team A) topa sahip (pas 1 -> 2)
    # - Kare 22-41: Top 2 -> 3 arasında süzülüyor
    # - Kare 42-45: Oyuncu 3 (Team B) topa sahip (top kaybı 2 -> 3)
    # - Kare 6'da Oyuncu 4 on_pitch=False (dışlanmalı)
    # - Hakem (99) topun dibinde dursa bile top kontrolü almamalı
    for f in range(46):
        ts = round(f / 25.0, 3)
        if f < 5:
            bx, by = 20.0 + f * 0.4, 34.0
        elif f < 17:
            t_f = 0.22 + (f - 5) / 11.0 * 0.56
            bx = 21.6 + t_f * (34.0 - 21.6)
            by = 34.0 + t_f * (22.0 - 34.0)
        elif f < 22:
            bx, by = 34.0 + (f - 17) * 0.4, 22.0
        elif f < 42:
            t_f = 0.14 + (f - 22) / 19.0 * 0.72
            bx = 35.6 + t_f * (58.0 - 35.6)
            by = 22.0 + t_f * (46.0 - 22.0)
        else:
            bx, by = 58.0 - (f - 42) * 0.4, 46.0

        # Kare 2'de top yok -> 1 karelik boşluk majority vote ile köprülenmeli
        if f != 2:
            rows.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
                "team": None, "confidence": 0.9, "is_predicted": False,
                "bbox_x1": bx * 10, "bbox_y1": by * 10, "bbox_x2": bx * 10 + 10, "bbox_y2": by * 10 + 10,
                "pixel_x": bx * 10 + 5, "pixel_y": by * 10 + 5,
                "pitch_x": bx, "pitch_y": by, "on_pitch": True,
            })
        if f == 0:
            # Aynı karede is_predicted=True yanlış top konumu (tercih edilmemeli)
            rows.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
                "team": None, "confidence": np.nan, "is_predicted": True,
                "bbox_x1": 900.0, "bbox_y1": 600.0, "bbox_x2": 910.0, "bbox_y2": 610.0,
                "pixel_x": 905.0, "pixel_y": 605.0,
                "pitch_x": 90.0, "pitch_y": 60.0, "on_pitch": True,
            })

        p1x, p1y = (bx + 0.5, 34.0) if f < 5 else (22.0, 34.0)
        p2x, p2y = (bx + 0.4, 22.0) if 17 <= f < 22 else (36.0, 22.0)
        p3x, p3y = (bx - 0.5, 46.0) if f >= 42 else (58.0, 46.0)

        rows.extend([
            {
                "frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": gk_cls,
                "team": "Team A", "confidence": 0.85, "is_predicted": False,
                "bbox_x1": p1x * 10, "bbox_y1": p1y * 10, "bbox_x2": p1x * 10 + 30, "bbox_y2": p1y * 10 + 70,
                "pixel_x": p1x * 10 + 15, "pixel_y": p1y * 10 + 70,
                "pitch_x": p1x, "pitch_y": p1y, "on_pitch": True,
            },
            {
                "frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": p_cls,
                "team": "Team A", "confidence": 0.88, "is_predicted": False,
                "bbox_x1": p2x * 10, "bbox_y1": p2y * 10, "bbox_x2": p2x * 10 + 30, "bbox_y2": p2y * 10 + 70,
                "pixel_x": p2x * 10 + 15, "pixel_y": p2y * 10 + 70,
                "pitch_x": p2x, "pitch_y": p2y, "on_pitch": True,
            },
            {
                "frame_idx": f, "timestamp_sec": ts, "track_id": 3, "class_name": p_cls,
                "team": "Team B", "confidence": 0.87, "is_predicted": False,
                "bbox_x1": p3x * 10, "bbox_y1": p3y * 10, "bbox_x2": p3x * 10 + 30, "bbox_y2": p3y * 10 + 70,
                "pixel_x": p3x * 10 + 15, "pixel_y": p3y * 10 + 70,
                "pitch_x": p3x, "pitch_y": p3y, "on_pitch": True,
            },
            {
                "frame_idx": f, "timestamp_sec": ts, "track_id": 4, "class_name": p_cls,
                "team": "Team B", "confidence": 0.80, "is_predicted": False,
                "bbox_x1": 700.0, "bbox_y1": 300.0, "bbox_x2": 730.0, "bbox_y2": 370.0,
                "pixel_x": 715.0, "pixel_y": 370.0,
                "pitch_x": 70.0 if f != 6 else 120.0, "pitch_y": 30.0,
                "on_pitch": False if f == 6 else pd.NA,
            },
            {
                "frame_idx": f, "timestamp_sec": ts, "track_id": 99, "class_name": ref_cls,
                "team": "Referee", "confidence": 0.90, "is_predicted": False,
                "bbox_x1": bx * 10, "bbox_y1": by * 10, "bbox_x2": bx * 10 + 30, "bbox_y2": by * 10 + 70,
                "pixel_x": bx * 10 + 15, "pixel_y": by * 10 + 70,
                "pitch_x": bx, "pitch_y": by, "on_pitch": True,
            },
        ])
    return pd.DataFrame(rows)


def _make_90min_tracking_df(n_frames: int = 135_000, fps: float = 25.0) -> pd.DataFrame:
    """
    90 dakikalık (~135.000 kare x 22 nesne = 2.970.000 satır) sentetik takip
    verisini vektörel NumPy dizileriyle <1.5 sn'de kompakt tiplerle üretir.
    """
    n_entities = 22  # 10 Team A (1 kaleci + 9 oyuncu) + 10 Team B + 1 Hakem + 1 Top
    total_rows = n_frames * n_entities

    frames_1d = np.arange(n_frames, dtype=np.int32)
    ts_1d = np.round(frames_1d.astype(np.float64) / fps, 3)

    # 21 insan + 1 top
    tid_labels = [str(i) for i in range(1, 22)] + ["ball"]
    cls_labels = (
        ["goalkeeper"] + ["player"] * 9
        + ["goalkeeper"] + ["player"] * 9
        + ["referee", "ball"]
    )
    team_labels = ["Team A"] * 10 + ["Team B"] * 10 + ["Referee", None]

    base_x = np.array(
        [8.0, 22.0, 24.0, 26.0, 35.0, 46.0, 50.0, 56.0, 72.0, 82.0,
         97.0, 82.0, 80.0, 78.0, 68.0, 58.0, 52.0, 46.0, 30.0, 22.0,
         52.5, 52.5],
        dtype=np.float32,
    )
    base_y = np.array(
        [34.0, 12.0, 28.0, 52.0, 34.0, 14.0, 34.0, 54.0, 18.0, 42.0,
         34.0, 14.0, 30.0, 52.0, 34.0, 16.0, 36.0, 54.0, 20.0, 44.0,
         34.0, 34.0],
        dtype=np.float32,
    )
    # Rol çeşitliliği için farklı hız genlikleri
    amps = np.array(
        [0.02, 0.06, 0.07, 0.08, 0.11, 0.15, 0.16, 0.14, 0.21, 0.19,
         0.02, 0.06, 0.07, 0.08, 0.11, 0.15, 0.16, 0.14, 0.21, 0.19,
         0.10, 0.0],
        dtype=np.float32,
    )

    # Kümülatif faz ile son %20'de gerçekçi tempo düşüşü + oyuncu 9'da kademeli yorgunluk
    tempo_1d = np.where(frames_1d >= int(n_frames * 0.80), 0.35, 1.0).astype(np.float32)
    phase_1d = np.cumsum(tempo_1d, dtype=np.float64) * 0.12

    # (n_frames, 22) matrisinde konumları hesapla
    phases = phase_1d[:, None] * (amps[None, :] / 0.12)
    # Oyuncu 9 (indeks 8) zamanla yorulsun
    fatigue_mult = np.linspace(1.3, 0.25, n_frames, dtype=np.float32)
    phases[:, 8] = np.cumsum(tempo_1d * fatigue_mult, dtype=np.float64) * (amps[8] / 0.12) * 0.12

    px_mat = np.clip(base_x[None, :] + 7.5 * np.sin(phases), 1.0, 104.0).astype(np.float32)
    py_mat = np.clip(base_y[None, :] + 6.5 * np.cos(phases * 0.85), 1.0, 67.0).astype(np.float32)

    # Oyuncu 2 (indeks 1) için kısa süreli formasyon dışı konum anomalisi (kare 60_000..60_500)
    px_mat[60_000:60_500, 1] = np.float32(101.0)
    py_mat[60_000:60_500, 1] = np.float32(65.0)

    # Top (indeks 21): her 50 karede sırayla oyuncular arasında gezerek pas/turnover üretir
    carrier_idx = (frames_1d // 50) % 20
    # Her 50 karelik bloğun son 10 karesinde havada/ara konumda geçiş
    in_flight = (frames_1d % 50) >= 40
    next_carrier = (carrier_idx + 1) % 20
    ball_px = np.where(
        in_flight,
        0.5 * (px_mat[frames_1d, carrier_idx] + px_mat[frames_1d, next_carrier]),
        px_mat[frames_1d, carrier_idx] + np.float32(0.35),
    ).astype(np.float32)
    ball_py = np.where(
        in_flight,
        0.5 * (py_mat[frames_1d, carrier_idx] + py_mat[frames_1d, next_carrier]),
        py_mat[frames_1d, carrier_idx] + np.float32(0.25),
    ).astype(np.float32)
    px_mat[:, 21] = ball_px
    py_mat[:, 21] = ball_py

    pitch_x = px_mat.reshape(-1)
    pitch_y = py_mat.reshape(-1)
    pixel_x = (pitch_x * np.float32(15.0) + np.float32(120.0)).astype(np.float32)
    pixel_y = (pitch_y * np.float32(12.0) + np.float32(80.0)).astype(np.float32)

    bbox_x1 = (pixel_x - np.float32(15.0)).astype(np.float32)
    bbox_y1 = (pixel_y - np.float32(60.0)).astype(np.float32)
    bbox_x2 = (pixel_x + np.float32(15.0)).astype(np.float32)
    bbox_y2 = pixel_y.copy()

    frame_idx = np.repeat(frames_1d, n_entities)
    timestamp_sec = np.repeat(ts_1d, n_entities)

    tid_codes = np.tile(np.arange(n_entities, dtype=np.int16), n_frames)
    track_id = pd.Categorical.from_codes(tid_codes, categories=tid_labels)

    cls_unique = ["goalkeeper", "player", "referee", "ball"]
    cls_map = np.array([cls_unique.index(c) for c in cls_labels], dtype=np.int8)
    class_name = pd.Categorical.from_codes(np.tile(cls_map, n_frames), categories=cls_unique)

    tm_unique = ["Team A", "Team B", "Referee"]
    tm_map = np.array([0] * 10 + [1] * 10 + [2, -1], dtype=np.int8)
    team = pd.Categorical.from_codes(np.tile(tm_map, n_frames), categories=tm_unique)

    is_predicted = np.zeros(total_rows, dtype=bool)
    on_pitch = np.ones(total_rows, dtype=bool)
    confidence = np.full(total_rows, 0.85, dtype=np.float32)

    return pd.DataFrame({
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "track_id": track_id,
        "class_name": class_name,
        "team": team,
        "confidence": confidence,
        "is_predicted": is_predicted,
        "bbox_x1": bbox_x1,
        "bbox_y1": bbox_y1,
        "bbox_x2": bbox_x2,
        "bbox_y2": bbox_y2,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "on_pitch": on_pitch,
    })


# ----------------------------------------------------------------------
# PART (A): PARİTE, M1 4-SINIF UYUMU VE KENAR DURUMLARI
# ----------------------------------------------------------------------
def test_match_analyzer_parity_4class_and_legacy_person(tmp_path, monkeypatch):
    # M1 4-sınıf konfigürasyonunu simüle et ("person" config.HUMAN_CLASS_NAMES dışında)
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"goalkeeper", "player", "referee"})

    for use_four in (True, False):
        df = _make_controlled_match_df(use_four_class=use_four)
        df_orig = df.copy(deep=True)

        # In-memory zero-copy non-mutation testi
        ma_mem = MatchAnalyzer(df, fps=25.0)
        assert ma_mem.is_calibrated is True
        tl_mem, pos_mem = ma_mem.compute_possession()
        pass_mem = ma_mem.detect_passes(tl_mem)
        shape_mem = ma_mem.compute_team_shape()
        assert df.equals(df_orig), "MatchAnalyzer(copy=False) girdi DataFrame'ini bozmamalı"

        # Kare 2 (top yok) majority vote ile Oyuncu 1'e köprülenmiş olmalı; Hakem 99 kontrol almamalı
        assert len(tl_mem) == 45
        row_f2_check = tl_mem[tl_mem["frame_idx"] == 1].iloc[0]
        assert row_f2_check["smoothed_track_id"] == "1"
        assert "99" not in {p["track_id"] for p in pos_mem["player_control"]}
        assert pos_mem["mode"] == "pitch_meters"
        assert pass_mem["total_completed_passes"] == 1
        assert pass_mem["total_turnovers"] == 1
        # Kare 6'da on_pitch=False olan Oyuncu 4 dışlanmış olmalı (45 kare kalır)
        tb_pos = {p["track_id"]: p for p in shape_mem["teams"]["Team B"]["player_positions"]}
        assert tb_pos["4"]["frames"] == 45

        # CSV ve SQLite üzerinden aynı sonuçların üretildiğini doğrula
        tag = "4cls" if use_four else "legacy"
        csv_p = str(tmp_path / f"match_{tag}.csv")
        db_p = str(tmp_path / f"match_{tag}.db")
        df.to_csv(csv_p, index=False)
        _write_df_to_sql(df, db_p)

        ma_csv = MatchAnalyzer.from_csv(csv_p, fps=25.0)
        ma_sql = MatchAnalyzer.from_sql(db_p, fps=25.0)

        out_mem = ma_mem.export_all(output_dir=str(tmp_path / f"out_mem_{tag}"), generate_heatmaps=True)
        out_csv = ma_csv.export_all(output_dir=str(tmp_path / f"out_csv_{tag}"), generate_heatmaps=True)
        out_sql = ma_sql.export_all(output_dir=str(tmp_path / f"out_sql_{tag}"), generate_heatmaps=True)

        assert out_mem["possession"] == out_csv["possession"] == out_sql["possession"]
        assert out_mem["passes"] == out_csv["passes"] == out_sql["passes"]
        assert out_mem["team_shape"] == out_csv["team_shape"] == out_sql["team_shape"]
        for hm_path in out_csv["exports"]["heatmaps"].values():
            img = cv2.imread(hm_path)
            assert img is not None and img.size > 0


def test_match_analyzer_edge_cases_and_pixel_fallback(tmp_path, monkeypatch):
    # Boş DataFrame
    empty_ma = MatchAnalyzer(pd.DataFrame(), allow_pixel_fallback=True)
    assert empty_ma.is_calibrated is False
    tl_e, pos_e = empty_ma.compute_possession()
    assert tl_e.empty and pos_e["total_frames"] == 0
    assert empty_ma.detect_passes(tl_e)["total_completed_passes"] == 0
    assert empty_ma.compute_team_shape()["teams"] == {}
    assert empty_ma.generate_heatmaps(output_dir=str(tmp_path / "empty_hm")) == {}

    # Eksik team/is_predicted/timestamp_sec + karma int/str/NaN track_id + göreceli heatmap yolu
    df_min = pd.DataFrame([
        {"frame_idx": 0, "track_id": "ball", "class_name": "ball", "pixel_x": 100.0, "pixel_y": 200.0},
        {"frame_idx": 0, "track_id": 1, "class_name": "player", "pixel_x": 105.0, "pixel_y": 202.0},
        {"frame_idx": 1, "track_id": "ball", "class_name": "ball", "pixel_x": 180.0, "pixel_y": 200.0},
        {"frame_idx": 1, "track_id": "2", "class_name": "player", "pixel_x": 182.0, "pixel_y": 201.0},
        {"frame_idx": 1, "track_id": np.nan, "class_name": "player", "pixel_x": 500.0, "pixel_y": 500.0},
    ])
    ma_min = MatchAnalyzer(df_min, allow_pixel_fallback=True)
    assert ma_min.fps == 25.0
    assert ma_min.is_calibrated is False
    tl_m, pos_m = ma_min.compute_possession()
    assert pos_m["mode"] == "pixel_fallback"
    assert pos_m["controlled_frames"] == 2
    shape_m = ma_min.compute_team_shape()
    assert shape_m["teams"] == {}  # team sütunu yok

    # _render_heatmap_image boş dirname ile çökmemeli
    monkeypatch.chdir(tmp_path)
    out_rel = ma_min._render_heatmap_image(
        xs=np.array([100.0, 180.0]),
        ys=np.array([200.0, 201.0]),
        bounds=(0.0, 300.0, 0.0, 300.0),
        title="Relative Path Test",
        draw_pitch_lines=False,
        output_path="rel_heatmap.png",
    )
    assert os.path.exists(tmp_path / out_rel)


def test_ml_analyzer_4class_compat_step_speeds_cache_and_parity(monkeypatch):
    monkeypatch.setattr(config, "HUMAN_CLASS_NAMES", {"goalkeeper", "player", "referee"})

    # 1) Segment boşluğu, hız sıçraması (spike) ve nedensel team=NaN -> Team A testi
    rows = []
    for f in list(range(12)) + list(range(20, 26)):
        ts = round(f / 25.0, 3)
        # Kare 8'de dev ışınlanma sıçraması (x=1000m), sonra normale dönüş
        if f == 8:
            px = 1000.0
        elif f < 12:
            px = 20.0 + f * 0.20  # 5.0 m/sn = 18.0 km/sa
        else:
            px = 25.0 + (f - 20) * 0.20  # 18.0 km/sa

        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 1,
            "class_name": "player" if f % 2 == 0 else "person",
            "team": None if f < 4 else "Team A",
            "confidence": 0.9, "is_predicted": False,
            "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 130.0, "bbox_y2": 270.0,
            "pixel_x": 115.0, "pixel_y": 270.0,
            "pitch_x": px, "pitch_y": 34.0, "on_pitch": True,
        })
    # Top (kalibrasyon algılaması için)
    rows.append({
        "frame_idx": 0, "timestamp_sec": 0.0, "track_id": "ball", "class_name": "ball",
        "team": None, "confidence": 0.9, "is_predicted": False,
        "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 110.0, "bbox_y2": 210.0,
        "pixel_x": 105.0, "pixel_y": 205.0, "pitch_x": 20.0, "pitch_y": 34.0, "on_pitch": True,
    })

    df = pd.DataFrame(rows)
    df_orig = df.copy(deep=True)
    analyzer = MLAnalyzer(df, fps=25.0)
    assert analyzer.is_calibrated is True

    spd_1, spikes_1 = analyzer._compute_player_step_speeds()
    assert df.equals(df_orig), "MLAnalyzer(copy=False) girdi DataFrame'ini bozmamalı"
    # Kare 8 ve kare 9 (1000 -> 21.8) sıçrama olarak dışlanmalı; kare 10..11 ve 21..25 sabit 18.0 km/sa kalmalı
    assert len(spikes_1) >= 1
    assert any(sp["frame_idx"] == 8 for sp in spikes_1)
    assert 8 not in set(spd_1["frame_idx"].tolist())
    assert 9 not in set(spd_1["frame_idx"].tolist())
    tail_spds = spd_1[spd_1["frame_idx"] >= 21]["speed"].to_numpy(dtype=float)
    np.testing.assert_allclose(tail_spds, 18.0, atol=1e-6)
    # Kare 1..3'te team Unknown, kare 4+ Team A olmalı
    assert set(spd_1[spd_1["frame_idx"] < 4]["team"].astype(str)) == {"Unknown"}
    assert set(spd_1[spd_1["frame_idx"] >= 4]["team"].astype(str)) == {"Team A"}

    # Dışarıdan döndürülen speed_df değiştirilse bile önbellek bozulmamalı
    spd_1["speed"] = -999.0
    spd_2, _ = analyzer._compute_player_step_speeds()
    assert (spd_2["speed"] > 0).all()

    # Önbellek temizleme ve df değişiminde geçersizleştirme
    analyzer.clear_cache()
    spd_3, _ = analyzer._compute_player_step_speeds()
    pd.testing.assert_frame_equal(spd_2, spd_3)

    # run_all içinde gerçek hesaplama yalnız 1 kez çalışmalı
    real_compute = MLAnalyzer._compute_player_step_speeds
    call_count = {"compute": 0, "outer": 0}

    def _spy(self, exclude_referees=True):
        call_count["outer"] += 1
        was_cached = bool(exclude_referees) in self._speed_cache
        res = real_compute(self, exclude_referees=exclude_referees)
        if not was_cached:
            call_count["compute"] += 1
        return res

    analyzer.clear_cache()
    monkeypatch.setattr(MLAnalyzer, "_compute_player_step_speeds", _spy)
    analyzer.run_all()
    assert call_count["outer"] == 3
    assert call_count["compute"] == 1


def test_ml_analyzer_proxy_passes_causal_non_leakage_and_pressures():
    # Kare 0..3: Oyuncu 1 (Team A) topa sahip; f<=3 iken team=None, f>=4 iken Team A (pas f=3->f=22 çok-kareli uçuşla)
    # Aynı kare içinde duplicate (tid=1, fid=3) olsa bile aynı karede sızıntı olmamalı
    rows = []
    for f in range(26):
        ts = round(f / 25.0, 3)
        if f <= 3:
            bx = 15.0
        elif f < 22:
            t_f = 0.14 + (f - 4) / 17.0 * 0.72
            bx = 15.0 + t_f * 25.0
        else:
            bx = 40.0
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
            "team": None, "is_predicted": False,
            "bbox_x1": bx * 10, "bbox_y1": 300, "bbox_x2": bx * 10 + 10, "bbox_y2": 310,
            "pixel_x": bx * 10, "pixel_y": 305, "pitch_x": bx, "pitch_y": 34.0, "on_pitch": True,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": "player",
            "team": None if f < 4 else "Team A", "is_predicted": False,
            "bbox_x1": 150, "bbox_y1": 300, "bbox_x2": 180, "bbox_y2": 370,
            "pixel_x": 165, "pixel_y": 340, "pitch_x": 15.0, "pitch_y": 34.0, "on_pitch": True,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": "player",
            "team": "Team A", "is_predicted": False,
            "bbox_x1": 400, "bbox_y1": 300, "bbox_x2": 430, "bbox_y2": 370,
            "pixel_x": 415, "pixel_y": 340, "pitch_x": 40.0, "pitch_y": 34.0, "on_pitch": True,
        })
        # Rakip (3): serbest bırakma karesinde (f=3) alıcıya 12m uzakta, varışta (f=4) 1m yakınlaşır
        opp_x = 52.0 if f <= 3 else 41.0
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 3, "class_name": "player",
            "team": "Team B", "is_predicted": False,
            "bbox_x1": opp_x * 10, "bbox_y1": 300, "bbox_x2": opp_x * 10 + 30, "bbox_y2": 370,
            "pixel_x": opp_x * 10, "pixel_y": 340, "pitch_x": opp_x, "pitch_y": 34.0, "on_pitch": True,
        })

    ev_df = MLAnalyzer(pd.DataFrame(rows), fps=25.0).extract_proxy_pass_events()
    assert len(ev_df) == 1
    ev = ev_df.iloc[0]
    assert ev["is_team_missing"] == 1
    assert pd.isna(ev["from_team"])
    assert ev["to_team"] == "Team A"
    assert ev["receiver_pressure_dist"] == pytest.approx(12.0, abs=1e-3)


def test_ml_analyzer_roles_tempo_and_io_parity(tmp_path):
    # Nullable BooleanDtype (<NA> içeren) ile 4-sınıf verisi + CSV + SQLite paritesi
    try:
        from tests.test_ml_analyzer import _make_calibrated_tracking_df
    except ModuleNotFoundError:
        from test_ml_analyzer import _make_calibrated_tracking_df

    df = _make_calibrated_tracking_df(
        n_frames=220,
        fps=25.0,
        inject_tempo_drop=True,
        inject_fatigue_player=True,
        inject_spatial_outlier=True,
        inject_low_coverage_tail=False,
    )
    df["class_name"] = df["class_name"].replace({"person": "player"})
    df["on_pitch"] = pd.array(
        [True if i % 25 != 0 else pd.NA for i in range(len(df))],
        dtype="boolean",
    )

    an_mem = MLAnalyzer(df, fps=25.0, random_state=42)
    prof_mem, cl_mem = an_mem.cluster_player_roles()
    win_mem, anom_mem, tp_mem = an_mem.detect_tempo_and_anomalies(window_sec=2.0)

    assert cl_mem["eligible_players"] == 8
    assert tp_mem["tempo_drop_count"] >= 1
    assert tp_mem["fatigue_players_count"] >= 1
    assert "spatial_off_formation" in set(anom_mem["anomaly_type"])

    csv_p = str(tmp_path / "ml_io.csv")
    db_p = str(tmp_path / "ml_io.db")
    df.to_csv(csv_p, index=False)
    df_sql = df.copy()
    df_sql["on_pitch"] = df_sql["on_pitch"].astype(object)
    _write_df_to_sql(df_sql, db_p)

    an_csv = MLAnalyzer.from_csv(csv_p, fps=25.0, random_state=42)
    an_sql = MLAnalyzer.from_sql(db_p, fps=25.0, random_state=42)

    prof_csv, cl_csv = an_csv.cluster_player_roles()
    prof_sql, cl_sql = an_sql.cluster_player_roles()
    assert cl_mem["optimal_k"] == cl_csv["optimal_k"] == cl_sql["optimal_k"]
    assert cl_mem["best_silhouette"] == cl_csv["best_silhouette"] == cl_sql["best_silhouette"]
    pd.testing.assert_frame_equal(prof_mem, prof_csv)
    pd.testing.assert_frame_equal(prof_mem, prof_sql)

    win_csv, anom_csv, tp_csv = an_csv.detect_tempo_and_anomalies(window_sec=2.0)
    win_sql, anom_sql, tp_sql = an_sql.detect_tempo_and_anomalies(window_sec=2.0)
    assert tp_mem == tp_csv == tp_sql
    pd.testing.assert_frame_equal(win_mem, win_csv)
    pd.testing.assert_frame_equal(win_mem, win_sql)
    pd.testing.assert_frame_equal(anom_mem, anom_csv)
    pd.testing.assert_frame_equal(anom_mem, anom_sql)


def test_baseline_sample_exports_parity_and_data_safety():
    csv_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    db_path = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)
    csv_hash_before = _sha256(csv_path)
    db_hash_before = _sha256(db_path)

    ma_csv = MatchAnalyzer.from_csv(csv_path)
    ma_sql = MatchAnalyzer.from_sql(db_path)
    tl_c, pos_c = ma_csv.compute_possession()
    tl_s, pos_s = ma_sql.compute_possession()
    assert pos_c == pos_s
    assert ma_csv.detect_passes(tl_c) == ma_sql.detect_passes(tl_s)
    assert ma_csv.compute_team_shape() == ma_sql.compute_team_shape()

    ml_csv = MLAnalyzer.from_csv(csv_path, random_state=42)
    ml_sql = MLAnalyzer.from_sql(db_path, random_state=42)
    dq_c = ml_csv.audit_data_quality()
    dq_s = ml_sql.audit_data_quality()
    assert dq_c["total_rows"] == 6042 == dq_s["total_rows"]
    assert dq_c["domain_anomalies"] == dq_s["domain_anomalies"]

    prof_c, cl_c = ml_csv.cluster_player_roles()
    prof_s, cl_s = ml_sql.cluster_player_roles()
    assert cl_c == cl_s
    pd.testing.assert_frame_equal(prof_c, prof_s)

    assert _sha256(csv_path) == csv_hash_before
    assert _sha256(db_path) == db_hash_before


# ----------------------------------------------------------------------
# PART (B): 135.000 KARE / ~2.97M SATIRLIK 90 DAKİKA STRES VE BELLEK TESTLERİ
# ----------------------------------------------------------------------
def test_r2_match_analyzer_90min_stress(tmp_path):
    csv_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    db_path = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)
    csv_hash_before = _sha256(csv_path)
    db_hash_before = _sha256(db_path)

    df = _make_90min_tracking_df(n_frames=135_000, fps=25.0)
    assert len(df) == 2_970_000

    proc = psutil.Process(os.getpid())
    gc.collect()
    rss_before = proc.memory_info().rss
    peak_wset_before = getattr(proc.memory_info(), "peak_wset", rss_before)

    t0 = time.perf_counter()
    analyzer = MatchAnalyzer(df, fps=25.0)
    out_dir = tmp_path / "match_90min_out"
    summary = analyzer.export_all(output_dir=str(out_dir), generate_heatmaps=True)
    elapsed = time.perf_counter() - t0

    mem_after = proc.memory_info()
    rss_after = mem_after.rss
    rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)
    peak_wset_delta = max(0, getattr(mem_after, "peak_wset", rss_after) - peak_wset_before)
    peak_mb = max(rss_delta_mb, peak_wset_delta / (1024 * 1024))
    print(f"\n[Stress MatchAnalyzer 135k] elapsed={elapsed:.2f}s, peak_delta={peak_mb:.1f}MB, rss_delta={rss_delta_mb:.1f}MB")

    assert elapsed < 45.0
    assert rss_delta_mb < 1536.0
    assert summary["is_calibrated"] is True
    assert summary["possession"]["ball_detected_frames"] == 135_000
    assert summary["passes"]["total_completed_passes"] + summary["passes"]["total_turnovers"] > 100

    pos_csv = out_dir / config.EXPORT_POSSESSION_FILENAME
    sum_json = out_dir / config.EXPORT_ANALYTICS_SUMMARY_FILENAME
    assert pos_csv.exists() and pos_csv.stat().st_size > 0
    assert sum_json.exists() and sum_json.stat().st_size > 0
    _strict_json_load(str(sum_json))
    assert len(summary["exports"]["heatmaps"]) >= 2
    for hm_p in summary["exports"]["heatmaps"].values():
        assert os.path.exists(hm_p) and os.path.getsize(hm_p) > 1000

    del analyzer, df
    gc.collect()

    assert _sha256(csv_path) == csv_hash_before
    assert _sha256(db_path) == db_hash_before


def test_r2_ml_analyzer_90min_stress(tmp_path):
    csv_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    db_path = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)
    csv_hash_before = _sha256(csv_path)
    db_hash_before = _sha256(db_path)

    df = _make_90min_tracking_df(n_frames=135_000, fps=25.0)
    assert len(df) == 2_970_000

    proc = psutil.Process(os.getpid())
    gc.collect()
    rss_before = proc.memory_info().rss
    peak_wset_before = getattr(proc.memory_info(), "peak_wset", rss_before)

    t0 = time.perf_counter()
    ml_analyzer = MLAnalyzer(df, fps=25.0, random_state=42)
    out_dir = tmp_path / "ml_90min_out"
    summary = ml_analyzer.export_all(output_dir=str(out_dir), generate_plots=True)
    elapsed = time.perf_counter() - t0

    mem_after = proc.memory_info()
    rss_after = mem_after.rss
    rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)
    peak_wset_delta = max(0, getattr(mem_after, "peak_wset", rss_after) - peak_wset_before)
    peak_mb = max(rss_delta_mb, peak_wset_delta / (1024 * 1024))
    print(f"\n[Stress MLAnalyzer 135k] elapsed={elapsed:.2f}s, peak_delta={peak_mb:.1f}MB, rss_delta={rss_delta_mb:.1f}MB")

    assert elapsed < 60.0
    assert rss_delta_mb < 1536.0
    assert summary["data_quality"]["total_rows"] == 2_970_000
    assert summary["data_quality"]["unique_frames"] == 135_000
    assert summary["player_clustering"]["eligible_players"] == 20
    assert summary["xp_modeling"]["status"] == "ok"
    assert summary["xp_modeling"]["data_source"] == "real_tracking_events"
    assert summary["tempo_and_anomalies"]["total_windows"] >= 20

    # Katı JSON ve Türkçe Markdown doğrulaması
    sum_json = out_dir / config.EXPORT_ML_SUMMARY_FILENAME
    rep_md = out_dir / config.EXPORT_ML_REPORT_FILENAME
    prof_csv = out_dir / config.EXPORT_ML_PLAYER_PROFILES_FILENAME
    win_csv = out_dir / config.EXPORT_ML_TEMPO_WINDOWS_FILENAME
    anom_csv = out_dir / config.EXPORT_ML_ANOMALIES_FILENAME
    pass_csv = out_dir / config.EXPORT_ML_PASS_PREDICTIONS_FILENAME

    assert sum_json.exists() and sum_json.stat().st_size > 0
    _strict_json_load(str(sum_json))

    md_text = rep_md.read_text(encoding="utf-8")
    assert "## 1. Veri Kalitesi" in md_text
    assert "## 4. Oyuncu Rol Kümeleme" in md_text
    assert "## 5. Tempo Düşüşü" in md_text

    for csv_out in (prof_csv, win_csv, anom_csv, pass_csv):
        assert csv_out.exists() and csv_out.stat().st_size > 0

    for png_name in (
        "xp_model_comparison.png",
        "xt_spatial_heatmap.png",
        "player_clusters_pca.png",
        "tempo_anomaly_timeline.png",
    ):
        png_p = out_dir / png_name
        assert png_p.exists() and png_p.stat().st_size > 1000

    del ml_analyzer, df
    gc.collect()

    assert _sha256(csv_path) == csv_hash_before
    assert _sha256(db_path) == db_hash_before
