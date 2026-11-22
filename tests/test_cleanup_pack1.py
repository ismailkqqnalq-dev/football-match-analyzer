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

    # xT hiçbir gerçek pas olmayan kalibre DataFrame'de insufficient_data dönmeli
    no_pass_df = pd.DataFrame([
        {
            "frame_idx": 0, "timestamp_sec": 0.0,
            "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
            "bbox_x1": 500, "bbox_y1": 340, "bbox_x2": 510, "bbox_y2": 350,
            "pixel_x": 505.0, "pixel_y": 345.0, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
        },
        {
            "frame_idx": 0, "timestamp_sec": 0.0,
            "track_id": 1, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 490, "bbox_y1": 300, "bbox_x2": 520, "bbox_y2": 370,
            "pixel_x": 505.0, "pixel_y": 340.0, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
        },
    ])
    no_pass_ml = MLAnalyzer(no_pass_df, fps=25.0, random_state=42)
    xt_empty = no_pass_ml.compute_xt_analysis()
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


# ----------------------------------------------------------------------
# T2 KABUL TESTLERİ: Kalibrasyonsuz Modda Mekânsal Metriklerin Kapatılması
# ----------------------------------------------------------------------
def _make_uncalibrated_match_df() -> pd.DataFrame:
    """pitch_x / pitch_y tamamen boş ama possession ve pas üretilebilen takip DataFrame'i."""
    rows = []
    for f in range(40):
        ts = round(f / 25.0, 3)
        if f < 6:
            bx = 100.0
        elif f < 29:
            bx = 100.0 + ((f - 5) / 24.0) * 180.0  # ~7.5 px/frame = ~187 px/s (< max_pass_speed)
        else:
            bx = 280.0
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
            "bbox_x1": bx, "bbox_y1": 300.0, "bbox_x2": bx + 10.0, "bbox_y2": 310.0,
            "pixel_x": bx, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 1, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 90.0, "bbox_y1": 280.0, "bbox_x2": 120.0, "bbox_y2": 350.0,
            "pixel_x": 100.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 2, "class_name": "player", "team": "Team A", "is_predicted": False,
            "bbox_x1": 270.0, "bbox_y1": 280.0, "bbox_x2": 300.0, "bbox_y2": 350.0,
            "pixel_x": 280.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts,
            "track_id": 3, "class_name": "player", "team": "Team B", "is_predicted": False,
            "bbox_x1": 700.0, "bbox_y1": 280.0, "bbox_x2": 730.0, "bbox_y2": 350.0,
            "pixel_x": 715.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
    return pd.DataFrame(rows)


def test_t2_match_analyzer_uncalibrated_keeps_possession_and_passes_gates_spatial(tmp_path, capsys):
    from modules.match_analytics import MatchAnalyzer

    df = _make_uncalibrated_match_df()
    ma = MatchAnalyzer(df, fps=25.0)
    assert ma.is_calibrated is False

    out_dir = tmp_path / "t2_ma"
    summary = ma.export_all(output_dir=str(out_dir), generate_heatmaps=True)

    assert summary["possession"]["controlled_frames"] > 0
    assert summary["possession"]["team_a_pct"] > 0.0
    assert summary["passes"]["total_completed_passes"] == 1
    assert summary["passes"]["events"][0]["distance"] is None

    assert summary["team_shape"] == {"status": "requires_calibration"}
    assert summary["heatmaps"] == {"status": "requires_calibration"}
    assert not (out_dir / "heatmaps").exists()
    assert list(out_dir.rglob("*.png")) == []

    stdout = capsys.readouterr().out
    notice = "Saha kalibrasyonu olmadığı için konum tabanlı metrikler üretilmedi."
    assert stdout.count(notice) == 1


def test_t2_ml_analyzer_uncalibrated_returns_requires_calibration_no_plots(tmp_path, capsys):
    df = _make_uncalibrated_match_df()
    ml = MLAnalyzer(df, fps=25.0, random_state=42)
    assert ml.is_calibrated is False

    out_dir = tmp_path / "t2_ml"
    summary = ml.export_all(output_dir=str(out_dir), generate_plots=True)

    assert summary["xp_modeling"] == {"status": "requires_calibration"}
    assert summary["xt_analysis"] == {"status": "requires_calibration"}
    assert summary["player_clustering"] == {"status": "requires_calibration"}
    assert summary["tempo_and_anomalies"] == {"status": "requires_calibration"}

    assert list(out_dir.glob("*.png")) == []
    assert not (out_dir / config.EXPORT_ML_PLAYER_PROFILES_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_TEMPO_WINDOWS_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_ANOMALIES_FILENAME).exists()
    assert not (out_dir / config.EXPORT_ML_PASS_PREDICTIONS_FILENAME).exists()

    notice = "Saha kalibrasyonu olmadığı için konum tabanlı metrikler üretilmedi."
    stdout = capsys.readouterr().out
    assert stdout.count(notice) == 1
    md_text = (out_dir / config.EXPORT_ML_REPORT_FILENAME).read_text(encoding="utf-8")
    assert md_text.count(notice) == 1


def test_t2_calibrated_data_behavior_unchanged(tmp_path):
    from modules.match_analytics import MatchAnalyzer

    df = _make_few_proxy_passes_df(n_events=8)
    ma = MatchAnalyzer(df, fps=25.0)
    assert ma.is_calibrated is True
    ma_out = tmp_path / "t2_cal_ma"
    ma_sum = ma.export_all(output_dir=str(ma_out), generate_heatmaps=True)
    assert ma_sum["team_shape"]["status"] == "ok"
    assert isinstance(ma_sum["heatmaps"], dict) and "team_a" in ma_sum["heatmaps"]
    assert len(list((ma_out / "heatmaps").glob("*.png"))) >= 2

    ml = MLAnalyzer(df, fps=25.0, random_state=42)
    assert ml.is_calibrated is True
    ml_out = tmp_path / "t2_cal_ml"
    ml_sum = ml.export_all(output_dir=str(ml_out), generate_plots=True)
    assert ml_sum["player_clustering"]["status"] == "ok"
    assert ml_sum["tempo_and_anomalies"]["status"] == "ok"
    assert (ml_out / "player_clusters_pca.png").exists()
    assert (ml_out / "tempo_anomaly_timeline.png").exists()
