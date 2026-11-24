"""
test_quality_pack3.py
---------------------
Kod kalitesi taraması (paket 3) düzeltmeleri:
  1) Top seçimi önceki konuma süreklilik gözetir.
  2) Kaleci takım eşleştirmesi kalibrasyon varsa saha metresiyle yapılır.
  3) Takımı eksik oyuncu satırları için yedek etiket çoğunluk oyuyla seçilir.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.match_analytics import MatchAnalyzer
from modules.team_classifier import TeamClassifier
from modules.tracker import BallTracker


def _ball(cx, cy, conf, size=10.0):
    h = size / 2.0
    return {"bbox": [cx - h, cy - h, cx + h, cy + h], "confidence": conf, "class_name": config.BALL_CLASS_NAME}


def _center(det):
    x1, y1, x2, y2 = det["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ----------------------------------------------------------------------
# 1) Top seçimi
# ----------------------------------------------------------------------
def test_ball_prefers_candidate_near_track_over_far_higher_conf():
    bt = BallTracker(max_missing_frames=15, velocity_damping=0.85, max_velocity_px=150.0)
    for i in range(5):
        bt.update([_ball(100.0 + 5.0 * i, 500.0, 0.6)])
    # Gerçek top (125, 500) civarında düşük güvenle; uzakta (1500, 300) daha yüksek güvenli sahte top
    out = bt.update([_ball(126.0, 500.0, 0.35), _ball(1500.0, 300.0, 0.40)])
    assert abs(_center(out)[0] - 126.0) < 1e-6


def test_ball_still_picks_highest_conf_without_history():
    bt = BallTracker()
    out = bt.update([_ball(126.0, 500.0, 0.35), _ball(1500.0, 300.0, 0.40)])
    assert abs(_center(out)[0] - 1500.0) < 1e-6


def test_ball_reacquires_far_ball_when_it_is_the_only_candidate():
    bt = BallTracker(max_missing_frames=15, velocity_damping=0.85, max_velocity_px=150.0)
    bt.update([_ball(100.0, 500.0, 0.6)])
    out = bt.update([_ball(1500.0, 300.0, 0.40)])
    assert abs(_center(out)[0] - 1500.0) < 1e-6


def test_ball_uses_plain_confidence_after_track_is_lost():
    bt = BallTracker(max_missing_frames=3, velocity_damping=0.85, max_velocity_px=150.0)
    bt.update([_ball(100.0, 500.0, 0.6)])
    for _ in range(5):  # max_missing_frames aşıldı -> eski konum geçersiz
        bt.update([])
    out = bt.update([_ball(110.0, 500.0, 0.35), _ball(1500.0, 300.0, 0.40)])
    assert abs(_center(out)[0] - 1500.0) < 1e-6


# ----------------------------------------------------------------------
# 2) Kaleci eşleştirmesi
# ----------------------------------------------------------------------
def _player(tid, team, px, pitch_x, cls="player"):
    return {
        "track_id": tid, "class_name": cls, "team": team,
        "bbox": [px - 10.0, 400.0, px + 10.0, 460.0],
        "pitch_x": pitch_x, "pitch_y": 34.0 if pitch_x is not None else None,
    }


def _fitted_classifier():
    tc = TeamClassifier()
    tc.is_fitted = True
    return tc


def test_goalkeeper_uses_pitch_meters_when_available():
    """
    Piksel uzayında Team B bloğu kaleciye yakın görünüyor (kamera sağa dönmüş),
    ama saha metresinde kaleci açıkça Team A'nın (sol) savunma bloğunun önünde.
    """
    tc = _fitted_classifier()
    outfield = (
        [_player(i, "Team A", 1500.0 + 10 * i, 15.0 + i) for i in range(1, 6)]
        + [_player(10 + i, "Team B", 300.0 + 10 * i, 70.0 + i) for i in range(1, 6)]
    )
    tc._update_team_spatial_ema(outfield)
    gk = _player(99, None, 400.0, 3.0, cls="goalkeeper")
    assert tc._resolve_goalkeeper_team(gk, outfield) == "Team A"


def test_goalkeeper_falls_back_to_pixels_without_calibration():
    tc = _fitted_classifier()
    outfield = (
        [_player(i, "Team A", 200.0 + 10 * i, None) for i in range(1, 6)]
        + [_player(10 + i, "Team B", 1500.0 + 10 * i, None) for i in range(1, 6)]
    )
    tc._update_team_spatial_ema(outfield)
    gk = _player(99, None, 50.0, None, cls="goalkeeper")
    assert tc._resolve_goalkeeper_team(gk, outfield) == "Team A"


def test_goalkeeper_pitch_and_pixel_emas_do_not_mix():
    tc = _fitted_classifier()
    pix_only = (
        [_player(i, "Team A", 200.0, None) for i in range(1, 4)]
        + [_player(10 + i, "Team B", 1500.0, None) for i in range(1, 4)]
    )
    tc._update_team_spatial_ema(pix_only)
    with_pitch = (
        [_player(i, "Team A", 200.0, 20.0) for i in range(1, 4)]
        + [_player(10 + i, "Team B", 1500.0, 80.0) for i in range(1, 4)]
    )
    tc._update_team_spatial_ema(with_pitch)
    assert tc._team_x_ema["pixel"]["Team A"] == 200.0
    assert tc._team_x_ema["pitch"]["Team A"] == 20.0
    assert tc._team_x_ema["pitch"]["Team B"] == 80.0


# ----------------------------------------------------------------------
# 3) Yedek takım etiketi
# ----------------------------------------------------------------------
def test_possession_fallback_team_uses_majority_not_first_label():
    """
    Oyuncu 7 ilk karede yanlışlıkla Team B etiketlenmiş, sonraki tüm karelerde Team A.
    Topa sahip olduğu karede takım etiketi eksikse Team A sayılmalı.
    """
    rows = []
    for f in range(0, 10):
        team = "Team B" if f == 0 else "Team A"
        rows.append({
            "frame_idx": f, "timestamp_sec": f / 25.0, "track_id": "7", "class_name": "player",
            "team": team, "pixel_x": 500.0, "pixel_y": 500.0,
            "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True, "is_predicted": False,
        })
    for f in range(10, 30):
        rows.append({
            "frame_idx": f, "timestamp_sec": f / 25.0, "track_id": "7", "class_name": "player",
            "team": None, "pixel_x": 500.0, "pixel_y": 500.0,
            "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True, "is_predicted": False,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": f / 25.0, "track_id": "ball", "class_name": config.BALL_CLASS_NAME,
            "team": None, "pixel_x": 505.0, "pixel_y": 500.0,
            "pitch_x": 50.5, "pitch_y": 34.0, "on_pitch": True, "is_predicted": False,
        })
    # İki takımın da var olması için uzakta başka bir Team B oyuncusu
    for f in range(0, 30):
        rows.append({
            "frame_idx": f, "timestamp_sec": f / 25.0, "track_id": "20", "class_name": "player",
            "team": "Team B", "pixel_x": 1500.0, "pixel_y": 500.0,
            "pitch_x": 90.0, "pitch_y": 34.0, "on_pitch": True, "is_predicted": False,
        })
    analyzer = MatchAnalyzer(pd.DataFrame(rows), fps=25.0)
    timeline, stats = analyzer.compute_possession()
    controlled = timeline[timeline["raw_track_id"] == "7"]
    assert len(controlled) > 0
    assert set(controlled["raw_team"].dropna()) == {"Team A"}
