import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.match_analytics import MatchAnalyzer
from modules.ml_analyzer import MLAnalyzer


def _row(
    f: int,
    tid,
    cls: str,
    team,
    px: float,
    py: float,
    fps: float = 25.0,
    is_pred: bool = False,
) -> dict:
    return {
        "frame_idx": f,
        "timestamp_sec": round(f / fps, 3),
        "track_id": tid,
        "class_name": cls,
        "team": team,
        "confidence": 0.9,
        "is_predicted": is_pred,
        "bbox_x1": px * 10.0,
        "bbox_y1": py * 10.0,
        "bbox_x2": px * 10.0 + 24.0,
        "bbox_y2": py * 10.0 + 64.0,
        "pixel_x": px * 10.0 + 12.0,
        "pixel_y": py * 10.0 + 64.0,
        "pitch_x": float(px),
        "pitch_y": float(py),
        "on_pitch": True,
    }


# ----------------------------------------------------------------------
# R1: KALECİ GERİ PAS, OYUN KURMA VE KONTROL SÜRESİ ENTEGRASYONU
# ----------------------------------------------------------------------
def test_r1_gk_backpass_and_buildup_not_turnover_in_both_analyzers():
    # Senaryo:
    # - Kare 0..4: Team A stoper (tid=2, x=22) topa sahip
    # - Kare 5..14: Geri pas uçuşu (22 -> 8, 10 kare, ~32 m/s)
    # - Kare 15..20: Team A kalecisi (tid=1, class_name="goalkeeper", team="Goalkeeper", x=8) topu kontrol eder
    # - Kare 21..34: Oyun kurma pası (8 -> 34, 14 kare, ~43 m/s -> 15 kare kullan ~39 m/s)
    # - Kare 35..40: Team A orta saha (tid=3, x=34) topa sahip => toplam 2 completed_pass!
    # - Kare 41..50: Top kaybı pası (34 -> 48, 10 kare, ~32 m/s)
    # - Kare 51..56: Team B forvet (tid=11, x=48) topa sahip => 1 turnover!
    rows = []
    for f in range(57):
        if f <= 4:
            bx, by = 22.0, 34.0
        elif f <= 14:
            tf = (f - 4) / 11.0
            bx, by = 22.0 + tf * (8.0 - 22.0), 34.0
        elif f <= 20:
            bx, by = 8.0, 34.0
        elif f <= 35:
            tf = (f - 20) / 16.0
            bx, by = 8.0 + tf * (34.0 - 8.0), 34.0
        elif f <= 40:
            bx, by = 34.0, 34.0
        elif f <= 50:
            tf = (f - 40) / 11.0
            bx, by = 34.0 + tf * (48.0 - 34.0), 34.0
        else:
            bx, by = 48.0, 34.0

        rows.append(_row(f, "ball", "ball", None, bx, by))
        rows.append(_row(f, 1, "goalkeeper", "Goalkeeper", 8.0, 34.0))
        rows.append(_row(f, 2, "player", "Team A", 22.0, 34.0))
        rows.append(_row(f, 3, "player", "Team A", 34.0, 34.0))
        rows.append(_row(f, 11, "player", "Team B", 48.0, 34.0))
        rows.append(_row(f, 12, "player", "Team B", 78.0, 34.0))
        rows.append(_row(f, 10, "goalkeeper", "Goalkeeper", 97.0, 34.0))

    df = pd.DataFrame(rows)
    df_orig = df.copy(deep=True)

    ma = MatchAnalyzer(df, fps=25.0)
    tl, pos = ma.compute_possession()
    passes = ma.detect_passes(tl)
    shape = ma.compute_team_shape()

    assert df.equals(df_orig), "MatchAnalyzer girdi DataFrame'ini değiştirmemeli"
    assert pos["controlled_frames"] > 0
    assert pos["team_a_frames"] + pos["team_b_frames"] == pos["controlled_frames"]
    gk_ctrl = next(p for p in pos["player_control"] if p["track_id"] == "1")
    assert gk_ctrl["team"] == "Team A"
    assert gk_ctrl["frames"] >= 5

    # Geri pas ve oyun kurma completed_pass olmalı (turnover değil!)
    assert passes["total_completed_passes"] == 2
    assert passes["total_turnovers"] == 1
    assert passes["team_stats"]["Team A"]["completed_passes"] == 2
    assert passes["team_stats"]["Team A"]["turnovers"] == 1

    # Team shape içinde kaleci Team A savunma hattında yer almalı
    assert shape["teams"]["Team A"]["defensive_line_x"] == pytest.approx(8.0, abs=0.1)

    # MLAnalyzer paritesi
    ml = MLAnalyzer(df, fps=25.0)
    ev = ml.extract_proxy_pass_events()
    assert df.equals(df_orig), "MLAnalyzer girdi DataFrame'ini değiştirmemeli"
    assert len(ev) == 3
    assert set(ev["from_team"].dropna().unique()) == {"Team A"}
    assert "Goalkeeper" not in set(ev["from_team"].dropna()) | set(ev["to_team"].dropna())
    assert int((ev["pass_outcome"] == 1).sum()) == 2
    assert int((ev["pass_outcome"] == 0).sum()) == 1

    profiles, _ = ml.cluster_player_roles(min_track_frames=20)
    gk_prof = profiles[profiles["track_id"] == "1"].iloc[0]
    assert gk_prof["team"] == "Team A"


# ----------------------------------------------------------------------
# R2: OTOMATİK HÜCUM YÖNÜ VE DEVRE ARASI TESPİTİ TESTLERİ
# ----------------------------------------------------------------------
def test_r2_reversed_attack_directions_and_halftime_side_swap(tmp_path):
    # 1. Yarı (kare 0..99): Team A SAĞDA (x=70..96, sağdan sola -x hücum), Team B SOLDA (x=9..35, soldan sağa +x hücum)
    # 2. Yarı (kare 100..199): DEVRE ARASI YÖN DEĞİŞİMİ! Team A SOLDA (+x), Team B SAĞDA (-x)
    rows = []
    for f in range(200):
        half1 = f < 100
        if half1:
            # 1. yarıda Team A sağdan sola ileri pas: 78 -> 66 (dx < 0)
            # Kare 10..15: Team A oyuncu 2 (x=78); kare 16..27: uçuş; kare 28..33: Team A oyuncu 3 (x=66)
            if f < 16:
                bx, by = 78.0, 20.0
            elif f <= 27:
                tf = (f - 15) / 13.0
                bx, by = 78.0 + tf * (66.0 - 78.0), 20.0
            else:
                bx, by = 66.0, 20.0

            rows.append(_row(f, "ball", "ball", None, bx, by))
            rows.append(_row(f, 1, "goalkeeper", "Team A", 96.0, 34.0))
            rows.append(_row(f, 2, "player", "Team A", 78.0, 20.0))
            rows.append(_row(f, 3, "player", "Team A", 66.0, 20.0))
            rows.append(_row(f, 10, "goalkeeper", "Team B", 8.0, 34.0))
            rows.append(_row(f, 11, "player", "Team B", 25.0, 48.0))
            rows.append(_row(f, 12, "player", "Team B", 38.0, 48.0))
        else:
            # 2. yarıda Team A soldan sağa ileri pas: 27 -> 39 (dx > 0)
            lf = f - 100
            if lf < 16:
                bx, by = 27.0, 48.0
            elif lf <= 27:
                tf = (lf - 15) / 13.0
                bx, by = 27.0 + tf * (39.0 - 27.0), 48.0
            else:
                bx, by = 39.0, 48.0

            rows.append(_row(f, "ball", "ball", None, bx, by))
            rows.append(_row(f, 1, "goalkeeper", "Team A", 8.0, 34.0))
            rows.append(_row(f, 2, "player", "Team A", 27.0, 48.0))
            rows.append(_row(f, 3, "player", "Team A", 39.0, 48.0))
            rows.append(_row(f, 10, "goalkeeper", "Team B", 96.0, 34.0))
            rows.append(_row(f, 11, "player", "Team B", 78.0, 20.0))
            rows.append(_row(f, 12, "player", "Team B", 66.0, 20.0))

    df = pd.DataFrame(rows)
    ma = MatchAnalyzer(df, fps=25.0)
    shape = ma.compute_team_shape()

    assert shape["halves_detected"] == 2
    assert len(shape["by_half"]) == 2
    h1, h2 = shape["by_half"]
    assert h1["attack_directions"]["Team A"] == "-x"
    assert h1["attack_directions"]["Team B"] == "+x"
    assert h1["teams"]["Team A"]["defensive_line_x"] == pytest.approx(96.0, abs=0.1)
    assert h1["teams"]["Team A"]["attacking_line_x"] == pytest.approx(66.0, abs=0.1)

    assert h2["attack_directions"]["Team A"] == "+x"
    assert h2["attack_directions"]["Team B"] == "-x"
    assert h2["teams"]["Team A"]["defensive_line_x"] == pytest.approx(8.0, abs=0.1)
    assert h2["teams"]["Team A"]["attacking_line_x"] == pytest.approx(39.0, abs=0.1)

    ml = MLAnalyzer(df, fps=25.0)
    ev = ml.extract_proxy_pass_events()
    assert len(ev) == 2
    # Her iki yarıdaki ileri pas da forward_progress_dx > 0 üretmeli!
    assert (ev["forward_progress_dx"] > 0).all()
    assert ev.iloc[0]["attack_dir"] == -1
    assert ev.iloc[1]["attack_dir"] == 1
    # 1. yarıda Team A sağdan sola hücum ederken y=20 (< 1/3) takım perspektifinden sağ kanat ("sag") olur
    assert ev.iloc[0]["pitch_zone_start"].endswith("_sag")

    xt_res = ml.compute_xt_analysis(events_df=ev)
    assert xt_res["halves_detected"] == 2
    scored_ev = xt_res["_scored_xt_eval_df"]
    assert (scored_ev["delta_xT"] > 0).all()

    profiles, _ = ml.cluster_player_roles(min_track_frames=20)
    gk_a = profiles[profiles["track_id"] == "1"].iloc[0]
    assert gk_a["norm_mean_x"] < 0.25

    out_dir = tmp_path / "ml_r2_out"
    ml.export_all(output_dir=str(out_dir), generate_plots=False)
    md_text = (out_dir / "ml_analysis_report.md").read_text(encoding="utf-8")
    assert "Tespit Edilen Devre Sayısı" in md_text
    assert "1. Yarı" in md_text and "2. Yarı" in md_text


def test_r2_time_gap_halftime_and_short_clip_no_false_split():
    # Açık zaman boşluğu (> 60 sn) kısa klipte bile 2 yarı yakalamalı
    rows_gap = []
    for f in range(20):
        rows_gap.append(_row(f, 1, "goalkeeper", "Team A", 8.0, 34.0))
        rows_gap.append(_row(f, 2, "player", "Team A", 30.0, 34.0))
        rows_gap.append(_row(f, 10, "goalkeeper", "Team B", 96.0, 34.0))
        rows_gap.append(_row(f, 11, "player", "Team B", 75.0, 34.0))
    for f in range(2500, 2520):  # +100 sn boşluk
        rows_gap.append(_row(f, 1, "goalkeeper", "Team A", 96.0, 34.0))
        rows_gap.append(_row(f, 2, "player", "Team A", 75.0, 34.0))
        rows_gap.append(_row(f, 10, "goalkeeper", "Team B", 8.0, 34.0))
        rows_gap.append(_row(f, 11, "player", "Team B", 30.0, 34.0))

    shape_gap = MatchAnalyzer(pd.DataFrame(rows_gap), fps=25.0).compute_team_shape()
    assert shape_gap["halves_detected"] == 2

    # Kısa tek-devre klip (< 60 kare): sahte bölünme olmamalı
    rows_short = []
    for f in range(40):
        rows_short.append(_row(f, 1, "player", "Team A", 35.0 + (f % 5), 34.0))
        rows_short.append(_row(f, 2, "player", "Team B", 65.0 - (f % 5), 34.0))
    shape_short = MatchAnalyzer(pd.DataFrame(rows_short), fps=25.0).compute_team_shape()
    assert shape_short["halves_detected"] == 1


# ----------------------------------------------------------------------
# R3: FİZİKSEL PAS, TEK KARE IŞINLANMA VE HAVA TOPU FİLTRESİ TESTLERİ
# ----------------------------------------------------------------------
def test_r3_rejects_single_frame_teleport_pred_snap_and_aerial_flyby():
    # Senaryo 1: Tek karelik imkânsız ışınlanma (0.04s, 30m) + donmuş is_predicted sıçraması
    rows_tp = []
    for f in range(30):
        is_p = False
        if f < 10:
            bx = 20.0
        elif f < 15:
            # Donmuş tahmin tutuşu
            bx = 20.0
            is_p = True
        else:
            # Kare 15'te 30m ışınlanma (x=50)
            bx = 50.0
        rows_tp.append(_row(f, "ball", "ball", None, bx, 34.0, is_pred=is_p))
        rows_tp.append(_row(f, 1, "player", "Team A", 20.0, 34.0))
        rows_tp.append(_row(f, 2, "player", "Team A", 50.0, 34.0))

    df_tp = pd.DataFrame(rows_tp)
    ma_tp = MatchAnalyzer(df_tp, fps=25.0)
    tl_tp, _ = ma_tp.compute_possession()
    passes_tp = ma_tp.detect_passes(tl_tp)
    assert passes_tp["total_completed_passes"] == 0
    assert passes_tp["total_turnovers"] == 0

    ml_tp = MLAnalyzer(df_tp, fps=25.0)
    assert len(ml_tp.extract_proxy_pass_events()) == 0

    # Senaryo 2: Hava topu (aerial flyby) — top 25 m/s hızla düz uçarak rakip (tid=99) üzerinden geçiyor
    # Pasör (tid=1, x=15) -> Alıcı (tid=2, x=45); ortada x=30'da duran rakip (tid=99) üzerinden 25 m/s ile geçer.
    # Gerçekçi 1.20s uçuş (30 kare @ 1.0 m/kare = 25 m/s > 15 m/s kontrol hız sınırı, 0 deg sapma, %0 yavaşlama).
    rows_air = []
    for f in range(46):
        if f < 6:
            bx = 15.0
        elif f < 36:
            bx = 15.0 + (f - 5) * 1.0  # 1.0 m/kare = 25 m/s düz uçuş
        else:
            bx = 45.0
        rows_air.append(_row(f, "ball", "ball", None, bx, 34.0))
        rows_air.append(_row(f, 1, "player", "Team A", 15.0, 34.0))
        rows_air.append(_row(f, 99, "player", "Team B", 30.0, 34.5))  # 2D olarak tam yol üstünde!
        rows_air.append(_row(f, 2, "player", "Team A", 45.0, 34.0))

    df_air = pd.DataFrame(rows_air)
    ma_air = MatchAnalyzer(df_air, fps=25.0)
    tl_air, pos_air = ma_air.compute_possession()
    passes_air = ma_air.detect_passes(tl_air)

    ctrl_ids = {p["track_id"] for p in pos_air["player_control"]}
    assert "99" not in ctrl_ids, "Hava topu altındaki savunmacı sahte kontrol almamalı"
    assert passes_air["total_turnovers"] == 0
    assert passes_air["total_completed_passes"] == 1
    assert passes_air["events"][0]["from_track_id"] == "1"
    assert passes_air["events"][0]["to_track_id"] == "2"

    ml_air = MLAnalyzer(df_air, fps=25.0)
    ev_air = ml_air.extract_proxy_pass_events()
    assert len(ev_air) == 1
    assert ev_air.iloc[0]["from_track_id"] == "1"
    assert ev_air.iloc[0]["to_track_id"] == "2"
    assert ev_air.iloc[0]["pass_outcome"] == 1


def test_r3_fps_scaling_min_control_frames_and_deflection_control():
    # 50 FPS'de min_control_frames = round(0.08 * 50) = 4 kare olmalı;
    # 2 karelik kısa dokunuş elenirken, 45 deg sapmalı & yavaşlayan gerçek kontrol (>4 kare) kabul edilmeli.
    rows = []
    fps = 50.0
    for f in range(40):
        if f < 8:
            bx, by = 20.0, 34.0
        elif f < 10:
            # Sadece 2 karelik kısa temas (< 4 kare @ 50fps)
            bx, by = 26.0, 34.0
        elif f < 22:
            # 12 kare uçuş (~22 m/s, ~12 * 0.44m)
            tf = (f - 9) / 13.0
            bx, by = 26.0 + tf * (33.0 - 26.0), 34.0
        else:
            # Yavaşlayıp yön değiştirerek (45 deg) kontrol ediliyor (18 kare)
            bx, by = 33.0, 34.0 + (f - 22) * 0.08
        rows.append(_row(f, "ball", "ball", None, bx, by, fps=fps))
        rows.append(_row(f, 1, "player", "Team A", 20.0, 34.0, fps=fps))
        rows.append(_row(f, 2, "player", "Team B", 26.0, 34.0, fps=fps))
        rows.append(_row(f, 3, "player", "Team A", 33.0, 34.0, fps=fps))

    df = pd.DataFrame(rows)
    ma = MatchAnalyzer(df, fps=fps)
    tl, pos = ma.compute_possession()
    ctrl_map = {p["track_id"]: p["frames"] for p in pos["player_control"]}
    assert "2" not in ctrl_map, "50fps'de 2 karelik kısa temas min_control_frames=4 ile elenmeli"
    assert "1" in ctrl_map and "3" in ctrl_map
    passes = ma.detect_passes(tl)
    assert passes["total_turnovers"] == 0
    assert passes["total_completed_passes"] == 1


def test_pass_detected_when_kalman_froze_during_flight():
    # Gerçek veride görülen desen (50 fps): top pasörün ayağından çıkınca tespit kopar,
    # Kalman tahmini pasörün yanında donar (15 kare), top uçuşta yeniden görünür.
    # Donmuş tahminden yeni konuma adım 4 m / 0.08 sn = 50 m/sn "sıçrama" gibi görünür;
    # ama son gerçek tespitten (kare 19) hız 4 m / 0.38 sn ~ 10.5 m/sn: gerçek pas.
    fps = 50.0
    rows = []
    for f in range(101):
        is_p = False
        if f < 20:
            bx = 60.0
        elif f < 35:
            bx, is_p = 60.0, True
        elif f < 38:
            continue  # top hiç görünmüyor
        elif f < 70:
            bx = 64.0 + (f - 38) * 0.25  # ~12.5 m/sn uçuş
        else:
            bx = 72.0
        rows.append(_row(f, "ball", "ball", None, bx, 34.0, fps=fps, is_pred=is_p))
    for f in range(101):
        rows.append(_row(f, 1, "player", "Team A", 60.0, 34.0, fps=fps))
        rows.append(_row(f, 2, "player", "Team A", 72.0, 34.0, fps=fps))

    ma = MatchAnalyzer(pd.DataFrame(rows), fps=fps)
    tl, _ = ma.compute_possession()
    passes = ma.detect_passes(tl)
    assert passes["total_completed_passes"] == 1
    assert passes["events"][0]["from_track_id"] == "1"
    assert passes["events"][0]["to_track_id"] == "2"


def test_reused_track_id_takes_team_of_current_spell():
    # Takipçi 1 numaralı kimliği önce uzun süre Team B'li bir oyuncuya, sonra Team A'lı
    # bir oyuncuya vermiş. İzin ömür boyu çoğunluğu B; ama top A'lı oyuncudayken
    # A takım arkadaşına giden pas top kaybı değil, tamamlanan pas sayılmalı.
    rows = []
    for f in range(200):  # kimlik 1 = Team B oyuncusu, top uzakta
        rows.append(_row(f, 1, "player", "Team B", 10.0, 10.0))
        rows.append(_row(f, "ball", "ball", None, 80.0, 60.0))
    for f in range(200, 260):  # kimlik 1 artık Team A oyuncusu, topa sahip, sonra pas
        bx = 40.0 if f < 220 else (40.0 + (f - 219) * 0.5 if f < 240 else 50.0)
        rows.append(_row(f, "ball", "ball", None, bx, 34.0))
        rows.append(_row(f, 1, "player", "Team A", 40.0, 34.0))
        rows.append(_row(f, 2, "player", "Team A", 50.0, 34.0))

    ma = MatchAnalyzer(pd.DataFrame(rows), fps=25.0)
    tl, _ = ma.compute_possession()
    passes = ma.detect_passes(tl)
    assert passes["total_turnovers"] == 0
    assert passes["total_completed_passes"] == 1


def test_off_pitch_ball_detection_ignored_for_possession():
    # Kale ağındaki yedek top (saha dışı, on_pitch=False) sahiplik zaman çizelgesine girmemeli.
    rows = []
    for f in range(20):
        ball = _row(f, "ball", "ball", None, 108.0, 34.6)
        ball["on_pitch"] = False
        rows.append(ball)
        rows.append(_row(f, 1, "player", "Team A", 104.5, 34.6))
    for f in range(20, 40):
        rows.append(_row(f, "ball", "ball", None, 50.0, 34.0))
        rows.append(_row(f, 1, "player", "Team A", 50.5, 34.0))

    ma = MatchAnalyzer(pd.DataFrame(rows), fps=25.0)
    tl, _ = ma.compute_possession()
    assert tl["frame_idx"].min() == 20


def test_r3_rejects_medium_pred_snap_12m_after_freeze():
    # Challenger Bug 1 regresyonu: >=0.3sn donmuş is_predicted=True tutuşundan sonra
    # 12m (300 m/sn @ 25fps) is_predicted=True -> False sıçraması her iki analizörde de 0 pas üretmeli.
    rows = []
    for f in range(26):
        is_p = False
        if f < 8:
            bx = 20.0
        elif f < 16:
            # 8 kare (0.32s >= 0.30s) donmuş tahmin
            bx = 20.0
            is_p = True
        else:
            # Kare 16'da 12m ani sıçrama (x=32 -> 300 m/s)
            bx = 32.0
        rows.append(_row(f, "ball", "ball", None, bx, 34.0, is_pred=is_p))
        rows.append(_row(f, 1, "player", "Team A", 20.0, 34.0))
        rows.append(_row(f, 2, "player", "Team A", 32.0, 34.0))

    df = pd.DataFrame(rows)
    ma = MatchAnalyzer(df, fps=25.0)
    tl, _ = ma.compute_possession()
    passes = ma.detect_passes(tl)
    assert passes["total_completed_passes"] == 0
    assert passes["total_turnovers"] == 0

    ml = MLAnalyzer(df, fps=25.0)
    assert len(ml.extract_proxy_pass_events()) == 0


def test_r1_untracked_opposing_goalkeepers_resolved_individually():
    # Challenger Bug 2 regresyonu: track_id=None olan iki karşıt kaleci (x=8 ve x=96)
    # tek medyan havuzunda birleştirilmeden ayrı ayrı Team A ve Team B olarak çözümlenmeli.
    from modules.pitch_orientation import resolve_goalkeeper_teams

    rows = []
    for f in range(20):
        rows.append(_row(f, None, "goalkeeper", "Goalkeeper", 8.0, 34.0))
        rows.append(_row(f, None, "goalkeeper", "Goalkeeper", 96.0, 34.0))
        rows.append(_row(f, 1, "player", "Team A", 25.0, 34.0))
        rows.append(_row(f, 2, "player", "Team B", 78.0, 34.0))

    df = pd.DataFrame(rows)
    codes, tlist, _ = resolve_goalkeeper_teams(df, fps=25.0)
    ta_c = tlist.index("Team A")
    tb_c = tlist.index("Team B")

    gk_left_mask = (df["class_name"] == "goalkeeper") & (df["pitch_x"] < 50.0)
    gk_right_mask = (df["class_name"] == "goalkeeper") & (df["pitch_x"] > 50.0)
    assert set(codes[gk_left_mask.to_numpy()]) == {ta_c}
    assert set(codes[gk_right_mask.to_numpy()]) == {tb_c}


def test_r1_detect_tempo_and_anomalies_gk_only_frame_centroid_alignment():
    # Reviewer Minor Finding 2 regresyonu: obs_idx kaleci satırlarını da
    # çözümlenmiş takıma eşleyerek kalecinin tek başına kaldığı karede centroid hizalamasını korur.
    rows = []
    for f in range(60):
        in_anom_win = f >= 50
        gk_x = 60.0 if in_anom_win else 8.0
        rows.append(_row(f, 1, "goalkeeper", "Goalkeeper", gk_x, 34.0))
        if not in_anom_win:
            rows.append(_row(f, 2, "player", "Team A", 30.0, 34.0))
            rows.append(_row(f, 3, "player", "Team A", 40.0, 34.0))
        elif f < 59:
            # Tek karelik gözlemler: speed_df'e girmez ama obs_idx centroid'inde yer alır;
            # f=59'da ise yalnızca Team A kalecisi gözlemlenir.
            rows.append(_row(f, 20 + f, "player", "Team A", 30.0, 34.0))
            rows.append(_row(f, 40 + f, "player", "Team A", 40.0, 34.0))
        rows.append(_row(f, 10, "goalkeeper", "Goalkeeper", 96.0, 34.0))
        rows.append(_row(f, 11, "player", "Team B", 75.0, 34.0))

    df = pd.DataFrame(rows)
    ml = MLAnalyzer(df, fps=25.0)
    _, anom_df, _ = ml.detect_tempo_and_anomalies(window_sec=0.4)
    gk_spatial = anom_df[
        (anom_df["anomaly_type"] == "spatial_off_formation")
        & (anom_df["track_id"] == "1")
    ]
    assert not gk_spatial.empty
    assert gk_spatial.iloc[0]["team"] == "Team A"
    assert gk_spatial.iloc[0]["metric_value"] > 0.15
