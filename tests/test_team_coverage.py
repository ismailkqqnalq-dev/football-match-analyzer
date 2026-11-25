import numpy as np
import pandas as pd

from modules.match_analytics import MatchAnalyzer, _team_coverage


def _row(f, tid, team, x, y, cls="player"):
    return {
        "frame_idx": f, "timestamp_sec": f / 25.0, "track_id": tid, "class_name": cls, "team": team,
        "confidence": 0.9, "is_predicted": False,
        "bbox_x1": x * 10, "bbox_y1": y * 10, "bbox_x2": x * 10 + 24, "bbox_y2": y * 10 + 64,
        "pixel_x": x * 10 + 12, "pixel_y": y * 10 + 64, "pitch_x": float(x), "pitch_y": float(y),
        "on_pitch": True,
    }


def test_team_coverage_counts_missing_frames_as_zero():
    # Kapsamdaki kareler 0..3; takım yalnızca 0 ve 1. karelerde 10'ar oyuncuyla görünüyor.
    subset = np.repeat([0, 1, 2, 3], 10)
    team = np.repeat([0, 1], 10)
    cov = _team_coverage(subset, team)
    assert cov["frames"] == 4
    assert cov["median_visible_players"] == 5.0
    assert cov["mean_visible_players"] == 5.0
    assert cov["frames_with_high_coverage_pct"] == 50.0
    assert cov["level"] == "medium"


def test_team_shape_carries_coverage_level():
    rows = []
    for f in range(20):
        for i in range(10):  # Team A: geniş açı, 10 oyuncu
            rows.append(_row(f, f"a{i}", "Team A", 10 + 4 * i, 10 + 4 * i))
        for i in range(3):   # Team B: yalnızca 3 oyuncu görünüyor (yakın çekim gibi)
            rows.append(_row(f, f"b{i}", "Team B", 60 + 4 * i, 30))
    shape = MatchAnalyzer(pd.DataFrame(rows), fps=25.0).compute_team_shape()
    assert shape["teams"]["Team A"]["coverage"]["level"] == "high"
    assert shape["teams"]["Team A"]["coverage"]["median_visible_players"] == 10.0
    assert shape["teams"]["Team B"]["coverage"]["level"] == "low"
    assert shape["teams"]["Team B"]["coverage"]["median_visible_players"] == 3.0
