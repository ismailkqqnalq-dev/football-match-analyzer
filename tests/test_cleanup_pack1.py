"""
Düzeltme Paketi 1 (T1, T2, T4, T5, T8) için yeni kabul testleri.
"""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.ml_analyzer import MLAnalyzer


def _make_few_proxy_passes_df(n_events: int = 8) -> pd.DataFrame:
    """30'dan az gerçek pas olayı içeren kalibre edilmiş takip DataFrame'i üretir."""
    rows = []
    f_cursor = 0
    for ev_idx in range(n_events):
        outcome_same = (ev_idx % 3) != 0
        t1_team = "Team A"
        t2_team = "Team A" if outcome_same else "Team B"
        p1_x = 20.0 + (ev_idx % 4) * 5.0
        p2_x = p1_x + 12.0
        for step in range(12):
            bx = p1_x if step < 4 else (p1_x + (step - 3) * (12.0 / 5.0) if step < 8 else p2_x)
            ts = round(f_cursor / 25.0, 3)
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
                "bbox_x1": bx * 10, "bbox_y1": 340, "bbox_x2": bx * 10 + 10, "bbox_y2": 350,
                "pixel_x": bx * 10, "pixel_y": 340, "pitch_x": bx, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": 10 + ev_idx * 2, "class_name": "player", "team": t1_team, "is_predicted": False,
                "bbox_x1": p1_x * 10, "bbox_y1": 300, "bbox_x2": p1_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p1_x * 10, "pixel_y": 340, "pitch_x": p1_x, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": ts,
                "track_id": 11 + ev_idx * 2, "class_name": "player", "team": t2_team, "is_predicted": False,
                "bbox_x1": p2_x * 10, "bbox_y1": 300, "bbox_x2": p2_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p2_x * 10, "pixel_y": 340, "pitch_x": p2_x, "pitch_y": 34.0, "on_pitch": True,
            })
            f_cursor += 1
        f_cursor += 110
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# T1 KABUL TESTLERİ: xP / xT Sentetik Veri Geçişinin Kaldırılması
# ----------------------------------------------------------------------
def test_t1_xp_returns_insufficient_data_below_30_real_events(tmp_path):
    df = _make_few_proxy_passes_df(n_events=8)
    ml = MLAnalyzer(df, fps=25.0, random_state=42)

    xp = ml.run_xp_modeling()
    assert xp["status"] == "insufficient_data"
    assert xp["data_source"] != "calibrated_synthetic_match"
    assert xp["real_events"] < 30
    assert xp["required"] == 30
    assert xp["models"] == []

    # xT hiçbir gerçek pas olmayan boş DataFrame'de insufficient_data dönmeli
    empty_ml = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42)
    xt_empty = empty_ml.compute_xt_analysis()
    assert xt_empty["status"] == "insufficient_data"

    # export_all sahte grafik üretmemeli, raporda "Hesaplanamadı – yetersiz veri" yer almalı
    out_dir = tmp_path / "t1_insuff"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)
    assert summary["xp_modeling"]["status"] == "insufficient_data"
    assert not (out_dir / "xp_model_comparison.png").exists()
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Hesaplanamadı – yetersiz veri" in md_text


def test_t1_main_cli_short_video_prints_insufficient_not_roc_auc(tmp_path, monkeypatch, capsys):
    import main as main_module

    sample_df = _make_few_proxy_passes_df(n_events=6)

    class DummyVP:
        def __init__(self, inp, out):
            self.fps = 25.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_frames(self):
            return iter([])

    class DummyObj:
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def get_team_map(self):
            return {}

    class DummyPerspective:
        is_calibrated = True

    class DummyExporter:
        def __init__(self, fps=25.0, **kwargs):
            self.fps = fps

        def postprocess(self, team_map=None):
            return {}

        def export_to_csv(self):
            return None

        def export_to_sql(self):
            return None

        def export_distance_summary(self):
            return None

        def to_dataframe(self):
            return sample_df.copy()

    monkeypatch.setattr(main_module, "VideoProcessor", DummyVP)
    monkeypatch.setattr(main_module, "ObjectDetector", DummyObj)
    monkeypatch.setattr(main_module, "PlayerTracker", DummyObj)
    monkeypatch.setattr(main_module, "BallTracker", DummyObj)
    monkeypatch.setattr(main_module, "TeamClassifier", DummyObj)
    monkeypatch.setattr(main_module, "PerspectiveTransformer", DummyPerspective)
    monkeypatch.setattr(main_module, "CameraMotionTracker", DummyObj)
    monkeypatch.setattr(main_module, "DataExporter", DummyExporter)
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-video", "--no-analytics", "--export-dir", str(tmp_path)])

    main_module.main()
    stdout = capsys.readouterr().out
    assert "ROC-AUC" not in stdout
    assert "xP hesaplanamadı" in stdout


def test_t1_allow_synthetic_true_runs_old_behavior_with_warning(tmp_path, capsys):
    ml = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42, allow_synthetic=True)
    out_dir = tmp_path / "t1_syn"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)

    assert summary["xp_modeling"]["status"] == "ok"
    assert summary["xp_modeling"]["data_source"] == "calibrated_synthetic_match"
    assert (out_dir / "xp_model_comparison.png").exists()
    assert (out_dir / "xt_spatial_heatmap.png").exists()

    stdout = capsys.readouterr().out
    assert "SENTETİK VERİ – GERÇEK MAÇ SONUCU DEĞİLDİR" in stdout
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "SENTETİK VERİ – GERÇEK MAÇ SONUCU DEĞİLDİR" in md_text
