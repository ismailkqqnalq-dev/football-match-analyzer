import hashlib
import json
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.ml_analyzer import MLAnalyzer


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_calibrated_tracking_df(
    n_frames: int = 250,
    fps: float = 25.0,
    inject_tempo_drop: bool = False,
    inject_fatigue_player: bool = False,
    inject_spatial_outlier: bool = False,
    inject_low_coverage_tail: bool = False,
) -> pd.DataFrame:
    """
    Testler için çok rollü, kalibre edilmiş (105m x 68m) sentetik takip verisi üretir.
    İsteğe bağlı olarak tempo düşüşü, yorgunluk eğilimi, formasyon dışı konum
    anomalisi ve düşük kapsam penceresi enjekte eder.
    """
    rng = np.random.default_rng(123)
    rows = []

    # 8 oyuncu (4 Team A, 4 Team B) + 1 Hakem + 1 geçici 5 karelik iz
    players_cfg = [
        # (track_id, team, base_x, base_y, speed_amp, radius_x, radius_y)
        (1, "Team A", 20.0, 34.0, 0.08, 4.0, 6.0),    # Sabit savunma
        (2, "Team A", 28.0, 18.0, 0.10, 6.0, 7.0),    # Sol bek
        (3, "Team A", 52.0, 34.0, 0.16, 12.0, 10.0),  # Merkez oyun kurucu
        (4, "Team A", 74.0, 52.0, 0.25, 16.0, 12.0),  # Yüksek tempolu kanat
        (5, "Team B", 82.0, 34.0, 0.08, 4.0, 6.0),    # Savunma B
        (6, "Team B", 72.0, 16.0, 0.11, 6.0, 7.0),    # Sağ bek B
        (7, "Team B", 50.0, 36.0, 0.17, 11.0, 10.0),  # Orta saha B
        (8, "Team B", 30.0, 50.0, 0.24, 15.0, 11.0),  # Kanat B
    ]

    for f in range(n_frames):
        ts = round(f / fps, 3)
        # Düşük kapsam kuyruğu (son %20'de yalnızca 1 oyuncu kadrajda kalsın)
        if inject_low_coverage_tail and f >= int(n_frames * 0.80):
            active_subset = [players_cfg[0]]
        else:
            active_subset = players_cfg

        # Tempo düşüşü enjeksiyonu: son yarıda tüm oyuncuların hızını %70 azalt
        tempo_mult = 0.25 if (inject_tempo_drop and f >= int(n_frames * 0.65)) else 1.0

        # Topu sırayla oyuncuların ayağına yakın gezdir (proxy pass üretimi için)
        carrier = active_subset[(f // 25) % len(active_subset)]
        c_tid, _, c_bx, c_by, c_amp, c_rx, c_ry = carrier
        c_phase = f * c_amp * tempo_mult
        bx = float(np.clip(c_bx + c_rx * math_sin(c_phase) + 0.3, 1.0, 104.0))
        by = float(np.clip(c_by + c_ry * math_cos(c_phase * 0.8), 1.0, 67.0))
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
            "team": None, "confidence": 0.85, "is_predicted": False,
            "bbox_x1": bx * 10, "bbox_y1": by * 10, "bbox_x2": bx * 10 + 10, "bbox_y2": by * 10 + 10,
            "pixel_x": bx * 10 + 5, "pixel_y": by * 10 + 10,
            "pitch_x": round(bx, 2), "pitch_y": round(by, 2), "on_pitch": True,
        })

        for tid, team, base_x, base_y, amp, rx, ry in active_subset:
            local_mult = tempo_mult
            # Oyuncu 4'te kademeli yorgunluk eğilimi (zamanla hız %75 düşer)
            if inject_fatigue_player and tid == 4:
                progress = f / max(1, n_frames - 1)
                local_mult *= max(0.15, 1.2 - 1.05 * progress)

            # Sabit adım ilerlemesi için kümülatif faz yaklaşımı
            phase = f * amp * local_mult
            px = float(np.clip(base_x + rx * math_sin(phase) + rng.normal(0, 0.03), 1.0, 104.0))
            py = float(np.clip(base_y + ry * math_cos(phase * 0.8) + rng.normal(0, 0.03), 1.0, 67.0))

            # Oyuncu 2 için orta pencerede formasyon dışı aşırı konum anomalisi
            if inject_spatial_outlier and tid == 2 and int(n_frames * 0.45) <= f <= int(n_frames * 0.55):
                px = 102.0
                py = 66.0

            rows.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": tid, "class_name": "person",
                "team": None if f < 3 else team,  # İlk 3 karede eksik team (nedensel test)
                "confidence": np.nan, "is_predicted": False,
                "bbox_x1": px * 10, "bbox_y1": py * 10, "bbox_x2": px * 10 + 30, "bbox_y2": py * 10 + 70,
                "pixel_x": px * 10 + 15, "pixel_y": py * 10 + 70,
                "pitch_x": round(px, 2), "pitch_y": round(py, 2), "on_pitch": True,
            })

        # Hakem izi (kümelemeden hariç tutulmalı)
        ref_x = 52.0 + 10.0 * math_sin(f * 0.12)
        ref_y = 34.0 + 8.0 * math_cos(f * 0.10)
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 99, "class_name": "person",
            "team": "Referee", "confidence": np.nan, "is_predicted": False,
            "bbox_x1": ref_x * 10, "bbox_y1": ref_y * 10, "bbox_x2": ref_x * 10 + 30, "bbox_y2": ref_y * 10 + 70,
            "pixel_x": ref_x * 10 + 15, "pixel_y": ref_y * 10 + 70,
            "pitch_x": round(ref_x, 2), "pitch_y": round(ref_y, 2), "on_pitch": True,
        })

        # Geçici 5 karelik iz (kümelemeden hariç tutulmalı)
        if f < 5:
            rows.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": 77, "class_name": "person",
                "team": "Team A", "confidence": np.nan, "is_predicted": False,
                "bbox_x1": 100.0, "bbox_y1": 100.0, "bbox_x2": 130.0, "bbox_y2": 170.0,
                "pixel_x": 115.0, "pixel_y": 170.0,
                "pitch_x": 10.0, "pitch_y": 10.0, "on_pitch": True,
            })

    return pd.DataFrame(rows)


def math_sin(x: float) -> float:
    return float(np.sin(x))


def math_cos(x: float) -> float:
    return float(np.cos(x))


# ----------------------------------------------------------------------
# 1. EKSİK / NULL DENETİMİ VE NEDENSEL İŞLEME
# ----------------------------------------------------------------------
def test_missing_null_audit_and_causal_team_handling():
    real_csv = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    analyzer = MLAnalyzer.from_csv(real_csv)
    audit = analyzer.audit_data_quality()

    assert audit["total_rows"] == 6042
    assert audit["coord_mode"] == "pixel_fallback"
    assert audit["is_calibrated"] is False
    assert audit["missing_by_column"]["pitch_x"]["missing_count"] == 6042
    assert audit["missing_by_column"]["team"]["missing_count"] == 824
    assert audit["domain_anomalies"]["team_missing_ball"] == 590
    assert audit["domain_anomalies"]["team_missing_human"] == 234
    assert audit["domain_anomalies"]["confidence_missing_human"] == 5452
    assert audit["domain_anomalies"]["invalid_bbox_total"] == 744
    assert audit["domain_anomalies"]["oob_pixel_predicted"] >= 250
    assert audit["domain_anomalies"]["transient_track_count"] == 9
    assert len(audit["handling_strategy"]) >= 4

    # Nedensellik testi: ilk karelerde team=NaN olan bir iz, gelecek karelerden team almamalı
    rows = []
    for f in range(32):
        ts = round(f / 25.0, 3)
        if f < 10:
            bx = 10.0
        elif f < 22:
            bx = 13.0 + ((f - 10) / 11.0) * 9.0
        else:
            bx = 25.0
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
            "team": None, "is_predicted": False,
            "bbox_x1": bx * 10, "bbox_y1": 340, "bbox_x2": bx * 10 + 10, "bbox_y2": 350,
            "pixel_x": bx * 10, "pixel_y": 340, "pitch_x": bx, "pitch_y": 34.0, "on_pitch": True,
        })
        # Oyuncu 1: f < 12 iken team=NaN, f >= 12 iken Team A (pas f=9 -> f=10'da gerçekleşir)
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": "person",
            "team": None if f < 12 else "Team A", "is_predicted": False,
            "bbox_x1": 100, "bbox_y1": 300, "bbox_x2": 130, "bbox_y2": 370,
            "pixel_x": 100.0, "pixel_y": 340.0, "pitch_x": 10.0, "pitch_y": 34.0, "on_pitch": True,
        })
        rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": "person",
            "team": "Team A", "is_predicted": False,
            "bbox_x1": 250, "bbox_y1": 300, "bbox_x2": 280, "bbox_y2": 370,
            "pixel_x": 250.0, "pixel_y": 340.0, "pitch_x": 25.0, "pitch_y": 34.0, "on_pitch": True,
        })
    causal_analyzer = MLAnalyzer(pd.DataFrame(rows), fps=25.0)
    proxy_ev = causal_analyzer.extract_proxy_pass_events()
    assert len(proxy_ev) == 1
    assert proxy_ev.iloc[0]["is_team_missing"] == 1
    assert pd.isna(proxy_ev.iloc[0]["from_team"])


# ----------------------------------------------------------------------
# 2. KESİN KRONOLOJİK BÖLÜNME VE SIFIR VERİ SIZINTISI TESTİ
# ----------------------------------------------------------------------
def test_xp_chronological_split_and_zero_data_leakage():
    base_analyzer = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42)
    events_clean = base_analyzer.generate_synthetic_match_passes(n_events=600, seed=42)
    res_clean = base_analyzer.run_xp_modeling(events_df=events_clean)

    assert res_clean["chronological_split_verified"] is True
    s_info = res_clean["split_info"]
    assert s_info["train_frame_range"][1] <= s_info["val_frame_range"][0]
    assert s_info["val_frame_range"][1] <= s_info["test_frame_range"][0]

    # Test setindeki son %20'yi ekstrem değerlerle boz:
    # Eğitim setindeki SimpleImputer, StandardScaler, OneHotEncoder ve LogisticRegression
    # parametreleri BİT-FOR-BİT aynı kalmalıdır (sıfır veri sızıntısı)!
    events_mutated = events_clean.sort_values(["frame_idx", "timestamp_sec"]).reset_index(drop=True).copy()
    test_start_idx = s_info["train_count"] + s_info["validation_count"]
    events_mutated.loc[test_start_idx:, "pass_length"] = 9999.0
    events_mutated.loc[test_start_idx:, "nearest_opponent_dist"] = 9999.0
    events_mutated.loc[test_start_idx:, "from_team"] = "LeakageTestTeam"
    events_mutated.loc[test_start_idx:, "pass_outcome"] = 1 - events_mutated.loc[test_start_idx:, "pass_outcome"]

    mutated_analyzer = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42)
    mutated_analyzer.run_xp_modeling(events_df=events_mutated)

    pipe_clean = base_analyzer.xp_models_["Logistic Regression"]
    pipe_mutated = mutated_analyzer.xp_models_["Logistic Regression"]

    prep_c = pipe_clean.named_steps["preprocessor"]
    prep_m = pipe_mutated.named_steps["preprocessor"]

    # Sayısal imputer medyanları ve scaler ortalamaları / std'leri birebir aynı olmalı
    np.testing.assert_allclose(
        prep_c.named_transformers_["num"].named_steps["imputer"].statistics_,
        prep_m.named_transformers_["num"].named_steps["imputer"].statistics_,
    )
    np.testing.assert_allclose(
        prep_c.named_transformers_["num"].named_steps["scaler"].mean_,
        prep_m.named_transformers_["num"].named_steps["scaler"].mean_,
    )
    np.testing.assert_allclose(
        prep_c.named_transformers_["num"].named_steps["scaler"].scale_,
        prep_m.named_transformers_["num"].named_steps["scaler"].scale_,
    )

    # OneHotEncoder kategorileri test setindeki "LeakageTestTeam"i içermemeli
    ohe_c = prep_c.named_transformers_["cat"].named_steps["ohe"]
    ohe_m = prep_m.named_transformers_["cat"].named_steps["ohe"]
    for cats_c, cats_m in zip(ohe_c.categories_, ohe_m.categories_):
        assert list(cats_c) == list(cats_m)
        assert "LeakageTestTeam" not in list(cats_m)

    # LogisticRegression ağırlıkları birebir aynı kalmalı
    np.testing.assert_allclose(
        pipe_clean.named_steps["model"].coef_,
        pipe_mutated.named_steps["model"].coef_,
    )
    np.testing.assert_allclose(
        pipe_clean.named_steps["model"].intercept_,
        pipe_mutated.named_steps["model"].intercept_,
    )


# ----------------------------------------------------------------------
# 3. MODEL KARŞILAŞTIRMASI, METRİKLER VE XT ANALİZİ
# ----------------------------------------------------------------------
def test_xp_model_comparison_beats_baseline_and_xt_grid():
    analyzer = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42)
    xp_res = analyzer.run_xp_modeling()

    assert xp_res["data_source"] == "calibrated_synthetic_match"
    model_names = [m["model_name"] for m in xp_res["models"]]
    assert model_names == [
        "Naive Baseline (DummyClassifier)",
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ]

    for m in xp_res["models"]:
        t = m["test"]
        for key in ("accuracy", "f1", "f1_macro", "roc_auc", "pr_auc", "brier_score", "log_loss", "confusion_matrix"):
            assert key in t
        assert len(t["confusion_matrix"]) == 2
        assert len(t["confusion_matrix"][0]) == 2
        assert len(m["operational_tradeoff_tr"]) > 10

    assert xp_res["beats_naive_baseline"]["passed"] is True
    assert xp_res["beats_naive_baseline"]["roc_auc_gain"] > 0.15
    assert xp_res["beats_naive_baseline"]["brier_improvement"] > 0.02

    xt_res = analyzer.compute_xt_analysis(xp_results=xp_res)
    grid = np.asarray(xt_res["xt_grid"], dtype=float)
    assert grid.shape == (config.ML_XT_GRID_ROWS, config.ML_XT_GRID_COLS)
    # Rakip kale bölgesi (sağ orta) kendi yarı alanından yüksek xT'ye sahip olmalı
    assert grid[3, 11] > grid[3, 0]
    assert xt_res["evaluated_events_count"] > 0


# ----------------------------------------------------------------------
# 4. OYUNCU ROL KÜMELEME (SILHOUETTE + 2B PCA) VE KENAR DURUMLAR
# ----------------------------------------------------------------------
def test_player_role_clustering_silhouette_pca_and_exclusions(tmp_path):
    df = _make_calibrated_tracking_df(n_frames=200)
    analyzer = MLAnalyzer(df, fps=25.0, random_state=42)
    profiles_df, cl_sum = analyzer.cluster_player_roles()

    assert cl_sum["eligible_players"] == 8
    # Hakem (99) ve geçici 5 karelik iz (77) hariç tutulmalı
    excluded_ids = {e["track_id"] for e in cl_sum["excluded_tracks"]}
    assert "99" in excluded_ids
    assert "77" in excluded_ids
    assert len(cl_sum["candidate_k_scores"]) >= 2
    assert 2 <= cl_sum["optimal_k"] <= 6
    assert cl_sum["best_silhouette"] > 0.1
    assert len(cl_sum["pca_explained_variance_ratio"]) == 2
    assert {"pca_1", "pca_2", "tactical_role_tr", "pitch_coverage_area"}.issubset(profiles_df.columns)

    # Kümeleme grafiği üretimi
    plot_path = str(tmp_path / "pca.png")
    analyzer._plot_player_clusters_pca(profiles_df, cl_sum, plot_path)
    assert os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000

    # < 3 oyuncu durumunda çökmeden zarif fallback
    tiny_df = df[df["track_id"].isin(["ball", 1, 2])].copy()
    tiny_analyzer = MLAnalyzer(tiny_df, fps=25.0, random_state=42)
    tiny_prof, tiny_sum = tiny_analyzer.cluster_player_roles()
    assert tiny_sum["eligible_players"] == 2
    assert tiny_sum["optimal_k"] == 1
    assert len(tiny_prof) == 2


# ----------------------------------------------------------------------
# 5. TEMPO DÜŞÜŞÜ / YORGUNLUK VE KONUM ANOMALİSİ TESPİTİ
# ----------------------------------------------------------------------
def test_tempo_drop_fatigue_and_spatial_anomaly_detection():
    df = _make_calibrated_tracking_df(
        n_frames=300,
        fps=25.0,
        inject_tempo_drop=True,
        inject_fatigue_player=True,
        inject_spatial_outlier=True,
        inject_low_coverage_tail=False,
    )
    analyzer = MLAnalyzer(df, fps=25.0, random_state=42)
    win_df, anom_df, tp_sum = analyzer.detect_tempo_and_anomalies(window_sec=2.0)

    assert tp_sum["tempo_drop_count"] >= 1
    assert win_df["tempo_drop_flag"].any()
    assert tp_sum["fatigue_players_count"] >= 1
    anom_types = set(anom_df["anomaly_type"].tolist())
    assert "tempo_drop" in anom_types
    assert "fatigue_trend" in anom_types
    assert "spatial_off_formation" in anom_types

    # Düşük kapsam farkındalığı: kadraj dışı kalan son bölüm tempo_drop değil low_coverage sayılmalı
    df_cov = _make_calibrated_tracking_df(
        n_frames=250,
        fps=25.0,
        inject_tempo_drop=False,
        inject_low_coverage_tail=True,
    )
    cov_analyzer = MLAnalyzer(df_cov, fps=25.0, random_state=42)
    cov_win, cov_anom, _ = cov_analyzer.detect_tempo_and_anomalies(window_sec=2.0)
    team_b_tail = cov_win[(cov_win["team"] == "Team B") & (cov_win["window_start_sec"] >= 8.0)]
    assert not team_b_tail.empty
    assert team_b_tail["low_coverage_flag"].all()
    assert not team_b_tail["tempo_drop_flag"].any()
    assert "low_coverage" in set(cov_anom["anomaly_type"].tolist())


# ----------------------------------------------------------------------
# 6. GERÇEK TRACKING_DATA.CSV / .DB + BAĞIMSIZ CLI + VERİ GÜVENLİĞİ
# ----------------------------------------------------------------------
def test_real_tracking_data_cli_csv_and_sql_and_data_safety(tmp_path):
    csv_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    db_path = os.path.join(config.EXPORT_DIR, config.EXPORT_SQL_FILENAME)

    csv_hash_before = _sha256(csv_path)
    db_hash_before = _sha256(db_path)

    out_csv_dir = tmp_path / "cli_csv_out"
    res_csv = subprocess.run(
        [sys.executable, "modules/ml_analyzer.py", "--input", csv_path, "--output-dir", str(out_csv_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[ML Özeti]" in res_csv.stdout

    expected_files = [
        config.EXPORT_ML_SUMMARY_FILENAME,
        config.EXPORT_ML_REPORT_FILENAME,
        config.EXPORT_ML_PLAYER_PROFILES_FILENAME,
        config.EXPORT_ML_TEMPO_WINDOWS_FILENAME,
        config.EXPORT_ML_ANOMALIES_FILENAME,
        config.EXPORT_ML_PASS_PREDICTIONS_FILENAME,
        "xp_model_comparison.png",
        "xt_spatial_heatmap.png",
        "player_clusters_pca.png",
        "tempo_anomaly_timeline.png",
    ]
    for fname in expected_files:
        fpath = out_csv_dir / fname
        assert fpath.exists(), f"Eksik çıktı dosyası: {fname}"
        assert fpath.stat().st_size > 0

    with open(out_csv_dir / config.EXPORT_ML_SUMMARY_FILENAME, "r", encoding="utf-8") as f:
        summary_json = json.load(f)
    assert summary_json["coord_mode"] == "pixel_fallback"
    assert summary_json["player_clustering"]["eligible_players"] == 10
    assert summary_json["xt_analysis"]["evaluated_events_count"] >= 1
    assert summary_json["xp_modeling"]["beats_naive_baseline"]["passed"] is True

    # SQLite CLI + --no-plots doğrulaması
    out_db_dir = tmp_path / "cli_db_out"
    subprocess.run(
        [sys.executable, "modules/ml_analyzer.py", "--input", db_path, "--output-dir", str(out_db_dir), "--no-plots"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (out_db_dir / config.EXPORT_ML_SUMMARY_FILENAME).exists()
    assert not (out_db_dir / "xp_model_comparison.png").exists()

    # Veri güvenliği: orijinal CSV ve DB kesinlikle değişmemiş olmalı
    assert _sha256(csv_path) == csv_hash_before
    assert _sha256(db_path) == db_hash_before


# ----------------------------------------------------------------------
# 7. MAIN.PY ADIM 8 ENTEGRASYONU (--no-ml VE AKTİF ADIM 8)
# ----------------------------------------------------------------------
def test_main_step8_integration_and_no_ml_flag(tmp_path, monkeypatch):
    import main as main_module

    sample_df = _make_calibrated_tracking_df(n_frames=90, fps=25.0)

    class DummyVP:
        def __init__(self, inp, out):
            self.fps = 25.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_frames(self):
            return iter([])

        def draw_detections(self, frame, dets):
            return frame

        def overlay_minimap(self, frame, mm):
            return frame

        def write_frame(self, frame):
            pass

    class DummyDetector:
        device = "cpu"

    class DummyTracker:
        pass

    class DummyTeamClassifier:
        def get_team_map(self):
            return {}

    class DummyPerspective:
        is_calibrated = False

    class DummyExporter:
        def __init__(self, fps=25.0):
            self.fps = fps

        def postprocess(self, team_map=None):
            return {}

        def export_to_csv(self):
            return str(tmp_path / "dummy.csv")

        def export_to_sql(self):
            return str(tmp_path / "dummy.db")

        def export_distance_summary(self):
            return None

        def to_dataframe(self):
            return sample_df.copy()

    monkeypatch.setattr(main_module, "VideoProcessor", DummyVP)
    monkeypatch.setattr(main_module, "ObjectDetector", DummyDetector)
    monkeypatch.setattr(main_module, "PlayerTracker", DummyTracker)
    monkeypatch.setattr(main_module, "BallTracker", DummyTracker)
    monkeypatch.setattr(main_module, "TeamClassifier", DummyTeamClassifier)
    monkeypatch.setattr(main_module, "PerspectiveTransformer", DummyPerspective)
    monkeypatch.setattr(main_module, "CameraMotionTracker", DummyTracker)
    monkeypatch.setattr(main_module, "DataExporter", DummyExporter)
    monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "EXPORT_ML_DIR", str(tmp_path / "ml_reports"))
    monkeypatch.setattr(config, "EXPORT_HEATMAP_DIR", str(tmp_path / "heatmaps"))

    # 1) --no-ml verildiğinde Adım 8 atlanmalı
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-analytics", "--no-ml"])
    main_module.main()
    assert not (tmp_path / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).exists()

    # 2) Varsayılan modda Adım 8 çalışıp rapor ve grafik üretmeli
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-analytics"])
    main_module.main()
    assert (tmp_path / "ml_reports" / config.EXPORT_ML_SUMMARY_FILENAME).exists()
    assert (tmp_path / "ml_reports" / config.EXPORT_ML_REPORT_FILENAME).exists()
    assert (tmp_path / "ml_reports" / "xp_model_comparison.png").exists()


# ----------------------------------------------------------------------
# 8. DEJENERE GİRDİLER VE SIFIR VARYANSLI PCA REGRESYON TESTİ
# ----------------------------------------------------------------------
def test_degenerate_inputs_and_zero_variance_pca(tmp_path):
    base_row = {
        "frame_idx": 0, "timestamp_sec": 0.0, "track_id": 1, "class_name": "person",
        "team": "Team A", "confidence": np.nan, "is_predicted": False,
        "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 140.0, "bbox_y2": 280.0,
        "pixel_x": 120.0, "pixel_y": 280.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
    }
    cases = {
        "empty_df": pd.DataFrame(),
        "single_row_df": pd.DataFrame([base_row]),
        "all_is_predicted_true": pd.DataFrame([{**base_row, "frame_idx": f, "is_predicted": True} for f in range(40)]),
        "inverted_bboxes": pd.DataFrame([{**base_row, "frame_idx": f, "bbox_x1": 150.0, "bbox_x2": 100.0} for f in range(40)]),
    }
    for name, df_case in cases.items():
        out_dir = tmp_path / name
        analyzer = MLAnalyzer(df_case, fps=25.0, random_state=42)
        summary = analyzer.export_all(output_dir=str(out_dir), generate_plots=True)
        assert summary["data_quality"]["duration_sec"] == 0.0 or name != "empty_df"
        expected_range = [0, 0] if name in ("empty_df", "single_row_df") else [0, 39]
        assert summary["data_quality"]["frame_range"] == expected_range
        assert "excluded_tracks_count" in summary["player_clustering"]
        assert "excluded_tracks" in summary["player_clustering"]
        assert "anomaly_counts_by_type" in summary["tempo_and_anomalies"]
        json_text = (out_dir / config.EXPORT_ML_SUMMARY_FILENAME).read_text(encoding="utf-8")
        # Katı JSON: NaN/Infinity kabul edilmez
        json.loads(json_text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {c}")))
        assert (out_dir / config.EXPORT_ML_REPORT_FILENAME).exists()

    # compute_xt_analysis içinde NaN koordinatlar ve predicted_xp çökmemeli
    xt_nan_events = pd.DataFrame([
        {
            "frame_idx": 1, "from_track_id": "1", "to_track_id": "2", "from_team": "Team A",
            "start_norm_x": np.nan, "start_norm_y": 0.5, "end_norm_x": 0.7, "end_norm_y": np.nan,
            "predicted_xp": np.nan,
        },
        {
            "frame_idx": 2, "from_track_id": "2", "to_track_id": "3", "from_team": "Team B",
            "start_norm_x": None, "start_norm_y": None, "end_norm_x": 0.2, "end_norm_y": 0.3,
            "predicted_xp": 0.8,
        },
    ])
    xt_res = MLAnalyzer(pd.DataFrame()).compute_xt_analysis(events_df=xt_nan_events)
    assert xt_res["evaluated_events_count"] == 2
    assert np.isfinite(xt_res["mean_delta_xT"])
    assert np.isfinite(xt_res["mean_expected_xT"])

    # Özdeş özellikli oyuncular: PCA uyarısız, NaN'sız [0.0, 0.0] ve geçerli grafik üretmeli
    identical_rows = []
    for f in range(40):
        for tid in (1, 2, 3, 4):
            identical_rows.append({
                "frame_idx": f, "timestamp_sec": round(f / 25.0, 3),
                "track_id": tid, "class_name": "person", "team": "Team A",
                "confidence": np.nan, "is_predicted": False,
                "bbox_x1": 200.0, "bbox_y1": 200.0, "bbox_x2": 230.0, "bbox_y2": 270.0,
                "pixel_x": 215.0, "pixel_y": 270.0, "pitch_x": 40.0, "pitch_y": 30.0, "on_pitch": True,
            })
    id_analyzer = MLAnalyzer(pd.DataFrame(identical_rows), fps=25.0, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prof_df, cl_sum = id_analyzer.cluster_player_roles()
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warns) == 0
    assert cl_sum["eligible_players"] == 4
    assert cl_sum["pca_explained_variance_ratio"] == [0.0, 0.0]
    assert not prof_df[["pca_1", "pca_2"]].isna().any().any()
    pca_png = tmp_path / "identical_pca.png"
    id_analyzer._plot_player_clusters_pca(prof_df, cl_sum, str(pca_png))
    assert pca_png.exists() and pca_png.stat().st_size > 1000


# ----------------------------------------------------------------------
# 9. KRONOLOJİK BÖLÜNME SONRASI TEK SINIFLI y_train REGRESYON TESTİ
# ----------------------------------------------------------------------
def test_single_class_y_train_after_chronological_split(tmp_path):
    base = MLAnalyzer(pd.DataFrame(), fps=25.0, random_state=42)
    syn = base.generate_synthetic_match_passes(n_events=100, seed=42).sort_values("frame_idx").reset_index(drop=True)
    # İlk 65 olay (train %60 = 60) tamamen sınıf 1, son 35 olay sınıf 0 (tüm sette 2 sınıf var ama y_train tek sınıf!)
    syn.loc[:64, "pass_outcome"] = 1
    syn.loc[65:, "pass_outcome"] = 0

    res_explicit = base.run_xp_modeling(events_df=syn)
    assert res_explicit["status"] == "skipped_single_class"
    assert res_explicit["models"] == []
    # Rapor ve grafik çökmeden üretilebilmeli
    plot_p = tmp_path / "xp_skipped.png"
    base._plot_xp_comparison(res_explicit, str(plot_p))
    assert plot_p.exists()

    # y_train iki sınıflı ama y_val / y_test tek sınıflı durum da çökmemeli
    syn_val_one = base.generate_synthetic_match_passes(n_events=100, seed=42).sort_values("frame_idx").reset_index(drop=True)
    syn_val_one.loc[60:, "pass_outcome"] = 1
    res_val_one = base.run_xp_modeling(events_df=syn_val_one)
    assert res_val_one["status"] == "ok"
    assert len(res_val_one["models"]) == 4

    # events_df=None otomatik modda: >=30 gerçek proxy olay olup y_train tek sınıf kalırsa sentetiğe düşmeli
    rows = []
    f_cursor = 0
    for ev_idx in range(36):
        # Her pas olayında: 3 kare tutuş + 5 kare gerçekçi uçuş + 3 kare tutuş;
        # ilk 24 pas Team A->Team A (1), son 12 pas Team A->Team B (0)
        outcome_same_team = ev_idx < 24
        t1_team = "Team A"
        t2_team = "Team A" if outcome_same_team else "Team B"
        p1_x, p2_x = 20.0, 28.0
        flight_xs = [22.6, 23.4, 24.0, 24.6, 25.4]
        for step in range(11):
            if step < 3:
                bx = p1_x
            elif step < 8:
                bx = flight_xs[step - 3]
            else:
                bx = p2_x
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": round(f_cursor / 25.0, 3),
                "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
                "bbox_x1": bx * 10, "bbox_y1": 340, "bbox_x2": bx * 10 + 10, "bbox_y2": 350,
                "pixel_x": bx * 10, "pixel_y": 340, "pitch_x": bx, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": round(f_cursor / 25.0, 3),
                "track_id": 100 + ev_idx * 2, "class_name": "person", "team": t1_team, "is_predicted": False,
                "bbox_x1": p1_x * 10, "bbox_y1": 300, "bbox_x2": p1_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p1_x * 10, "pixel_y": 340, "pitch_x": p1_x, "pitch_y": 34.0, "on_pitch": True,
            })
            rows.append({
                "frame_idx": f_cursor, "timestamp_sec": round(f_cursor / 25.0, 3),
                "track_id": 101 + ev_idx * 2, "class_name": "person", "team": t2_team, "is_predicted": False,
                "bbox_x1": p2_x * 10, "bbox_y1": 300, "bbox_x2": p2_x * 10 + 30, "bbox_y2": 370,
                "pixel_x": p2_x * 10, "pixel_y": 340, "pitch_x": p2_x, "pitch_y": 34.0, "on_pitch": True,
            })
            f_cursor += 1
        f_cursor += 105

    auto_analyzer = MLAnalyzer(pd.DataFrame(rows), fps=25.0, random_state=42)
    proxy_check = auto_analyzer.extract_proxy_pass_events()
    assert len(proxy_check) >= config.ML_MIN_SUPERVISED_EVENTS
    assert proxy_check["pass_outcome"].nunique() == 2

    all_res = auto_analyzer.run_all()
    assert all_res["xp_modeling"]["status"] == "ok"
    assert all_res["xp_modeling"]["data_source"] == "calibrated_synthetic_match"
    assert all_res["xp_modeling"]["fallback_reason"] == "single_class_y_train_after_chronological_split"


# ----------------------------------------------------------------------
# 10. NEDENSEL HIZ, SINIRDA YARILANMAMA VE GELECEK KARE BASKI SIZINTISIZLIĞI
# ----------------------------------------------------------------------
def test_causal_step_speeds_no_boundary_halving_and_no_future_leakage():
    # Sabit 75.0 px/s hareket (fps=25 -> kare başına 3.0 px) + 2 karelik ayrı segment
    const_rows = []
    for f in list(range(10)) + [20, 21]:
        x_px = 100.0 + (f * 3.0 if f < 10 else (f - 20) * 3.0)
        const_rows.append({
            "frame_idx": f, "timestamp_sec": round(f / 25.0, 3),
            "track_id": 1, "class_name": "person", "team": "Team A", "is_predicted": False,
            "bbox_x1": x_px, "bbox_y1": 200.0, "bbox_x2": x_px + 30.0, "bbox_y2": 270.0,
            "pixel_x": x_px, "pixel_y": 270.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
        })
    const_analyzer = MLAnalyzer(pd.DataFrame(const_rows), fps=25.0)
    spd_df, spikes = const_analyzer._compute_player_step_speeds()
    assert not spikes
    assert len(spd_df) == 10  # ilk segmentte 9 adım + 2 karelik segmentte 1 adım
    np.testing.assert_allclose(spd_df["speed"].to_numpy(dtype=float), 75.0, atol=1e-6)

    # Gelecek kare sızıntısı testi: t=0..10 arası sabit duran pasör t=11,12'de sıçrasa bile
    # ve t>=11'de alıcı yanına rakip ışınlansa bile t=10 özellikleri değişmemeli
    def _build_pass_df(future_burst: bool) -> pd.DataFrame:
        r = []
        for f in range(52):
            ts = round(f / 25.0, 3)
            # Top t<=10'da pasör (1)'de, t=11..45 uçuşta (gerçekçi hız), t>=46'da alıcı (2)'de
            if f <= 10 or f >= 46:
                bx = 100.0 if f <= 10 else 300.0
                r.append({
                    "frame_idx": f, "timestamp_sec": ts, "track_id": "ball", "class_name": "ball",
                    "team": None, "is_predicted": False,
                    "bbox_x1": bx, "bbox_y1": 300.0, "bbox_x2": bx + 10.0, "bbox_y2": 310.0,
                    "pixel_x": bx, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
                })
            # Pasör (1): t<=10 sabit 100.0, future_burst ise t>=11'de +25 px/kare sıçrar
            p1_x = 100.0 if (not future_burst or f <= 10) else 100.0 + (f - 10) * 25.0
            r.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": "person",
                "team": "Team A", "is_predicted": False,
                "bbox_x1": p1_x, "bbox_y1": 280.0, "bbox_x2": p1_x + 30.0, "bbox_y2": 350.0,
                "pixel_x": p1_x, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
            })
            # Alıcı (2): sabit 300.0
            r.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": "person",
                "team": "Team A", "is_predicted": False,
                "bbox_x1": 300.0, "bbox_y1": 280.0, "bbox_x2": 330.0, "bbox_y2": 350.0,
                "pixel_x": 300.0, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
            })
            # Rakip (3): t<=10'da 500.0'da (200 px uzak), future_burst ise t>=11'de alıcının yanına (305.0) geçer
            opp_x = 500.0 if (not future_burst or f <= 10) else 305.0
            r.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": 3, "class_name": "person",
                "team": "Team B", "is_predicted": False,
                "bbox_x1": opp_x, "bbox_y1": 280.0, "bbox_x2": opp_x + 30.0, "bbox_y2": 350.0,
                "pixel_x": opp_x, "pixel_y": 305.0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
            })
        return pd.DataFrame(r)

    ev_calm = MLAnalyzer(_build_pass_df(False), fps=25.0).extract_proxy_pass_events()
    ev_burst = MLAnalyzer(_build_pass_df(True), fps=25.0).extract_proxy_pass_events()
    assert len(ev_calm) == 1 and len(ev_burst) == 1
    assert ev_calm.iloc[0]["passer_speed"] == 0.0
    assert ev_burst.iloc[0]["passer_speed"] == 0.0
    assert ev_calm.iloc[0]["receiver_pressure_dist"] == 200.0
    assert ev_burst.iloc[0]["receiver_pressure_dist"] == 200.0


# ----------------------------------------------------------------------
# 11. KÜÇÜK DALGALANMALARDA YANLIŞ POZİTİF TEMPO_DROP OLMAMASI
# ----------------------------------------------------------------------
def test_tempo_drop_ignores_small_fluctuations_and_flags_real_drops():
    # 3 pencere (her biri 2 sn = 50 kare @ 25 fps): W0 ~ 19.8 km/sa, W1 ~ 19.9 km/sa, W2 ~ 19.2 km/sa (~%3.3 düşüş)
    # Oyuncular sabit hızlı doğrusal yürüyüşle net hız üretir
    def _make_window_speed_df(speeds_kmh_per_win):
        rows = []
        fps = 25.0
        for tid_idx, tid in enumerate((1, 2, 3, 4)):
            team = "Team A" if tid <= 2 else "Team B"
            x = 10.0
            y = 10.0 + tid_idx * 12.0
            f = 0
            for w_spd_kmh in speeds_kmh_per_win:
                step_m = (w_spd_kmh / 3.6) / fps
                for _ in range(50):
                    x += step_m
                    if x > 95.0:
                        x -= 80.0
                    rows.append({
                        "frame_idx": f, "timestamp_sec": round(f / fps, 3),
                        "track_id": tid, "class_name": "person", "team": team, "is_predicted": False,
                        "bbox_x1": x * 10, "bbox_y1": y * 10, "bbox_x2": x * 10 + 30, "bbox_y2": y * 10 + 70,
                        "pixel_x": x * 10, "pixel_y": y * 10, "pitch_x": x, "pitch_y": y, "on_pitch": True,
                    })
                    f += 1
        # Kalibre mod için top satırları
        for f_b in range(len(speeds_kmh_per_win) * 50):
            rows.append({
                "frame_idx": f_b, "timestamp_sec": round(f_b / fps, 3),
                "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
                "bbox_x1": 500, "bbox_y1": 300, "bbox_x2": 510, "bbox_y2": 310,
                "pixel_x": 505, "pixel_y": 305, "pitch_x": 50.0, "pitch_y": 34.0, "on_pitch": True,
            })
        return pd.DataFrame(rows)

    # Küçük dalgalanma (%3.3): kesinlikle tempo_drop olmamalı
    steady_df = _make_window_speed_df([19.8, 19.9, 19.2])
    win_s, anom_s, tp_s = MLAnalyzer(steady_df, fps=25.0).detect_tempo_and_anomalies(window_sec=2.0)
    assert tp_s["tempo_drop_count"] == 0
    assert not win_s["tempo_drop_flag"].any()
    assert "tempo_drop" not in set(anom_s["anomaly_type"].tolist()) if not anom_s.empty else True

    # Gerçek düşüş (%30): kesinlikle tempo_drop olarak yakalanmalı
    drop_df = _make_window_speed_df([20.0, 20.0, 14.0])
    win_d, anom_d, tp_d = MLAnalyzer(drop_df, fps=25.0).detect_tempo_and_anomalies(window_sec=2.0)
    assert tp_d["tempo_drop_count"] >= 1
    assert win_d["tempo_drop_flag"].any()
    assert "tempo_drop" in set(anom_d["anomaly_type"].tolist())


# ----------------------------------------------------------------------
# 12. KALİBRESİZ TRACKING_DATA.CSV İÇİN PITCH-PROJEDELİ XP OLASILIKLARI
# ----------------------------------------------------------------------
def test_uncalibrated_xp_pitch_projection_non_saturated():
    csv_path = os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME)
    analyzer = MLAnalyzer.from_csv(csv_path)
    res = analyzer.run_all()
    assert res["xp_modeling"]["scoring_unit_mode"] == "normalized_pitch_projection_105x68"
    pred_df = res["_artifacts"]["pass_predictions_df"]
    assert not pred_df.empty
    probs = pred_df["predicted_xp"].to_numpy(dtype=float)
    assert np.all((probs > 0.0) & (probs < 1.0))
    # Piksel doygunluğu (hepsi ~0.00 veya ~1.00) yaşanmadığını doğrula
    assert float(probs.min()) > 0.05
    assert float(probs.max()) < 0.99
    if len(probs) >= 2:
        assert float(np.std(probs)) > 0.01
    else:
        model_probs = res["xp_modeling"]["_scored_events_df"]["predicted_xp"].to_numpy(dtype=float)
        assert float(np.std(model_probs)) > 0.01


# ----------------------------------------------------------------------
# 13. REAL_TRACKING_EVENTS + PIXEL_FALLBACK EĞİTİM SINIRI SIZINTISIZLIĞI
# ----------------------------------------------------------------------
def test_real_tracking_pixel_fallback_train_bounds_invariance():
    def _build_pixel_real_events_df(inject_extreme_test: bool) -> pd.DataFrame:
        rows = []
        f_cursor = 0
        for ev_idx in range(36):
            outcome_same_team = (ev_idx % 3) != 0
            t1_team = "Team A"
            t2_team = "Team A" if outcome_same_team else "Team B"

            p1_x = 180.0 + (ev_idx % 5) * 30.0
            p1_y = 200.0 + (ev_idx % 3) * 80.0
            p2_x = 460.0 + (ev_idx % 4) * 35.0
            p2_y = 220.0 + ((ev_idx + 1) % 3) * 75.0
            opp_x = p1_x + 90.0 + (ev_idx % 4) * 30.0
            opp_y = p1_y + 25.0

            if inject_extreme_test and ev_idx >= 32:
                p1_x = 9200.0 + (ev_idx % 5) * 20.0
                p1_y = 6380.0 + (ev_idx % 3) * 40.0
                p2_x = 9520.0
                p2_y = 6540.0
                opp_x = p1_x + 90.0
                opp_y = p1_y + 25.0

            for step in range(86):
                ts = round(f_cursor / 25.0, 3)
                if step < 3 or step >= 83:
                    bx, by = (p1_x, p1_y) if step < 3 else (p2_x, p2_y)
                    rows.append({
                        "frame_idx": f_cursor, "timestamp_sec": ts,
                        "track_id": "ball", "class_name": "ball", "team": None, "is_predicted": False,
                        "bbox_x1": bx, "bbox_y1": by, "bbox_x2": bx + 10.0, "bbox_y2": by + 10.0,
                        "pixel_x": bx, "pixel_y": by, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
                    })
                rows.append({
                    "frame_idx": f_cursor, "timestamp_sec": ts,
                    "track_id": 100 + ev_idx * 3, "class_name": "person", "team": t1_team, "is_predicted": False,
                    "bbox_x1": p1_x, "bbox_y1": p1_y, "bbox_x2": p1_x + 30.0, "bbox_y2": p1_y + 70.0,
                    "pixel_x": p1_x, "pixel_y": p1_y, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
                })
                rows.append({
                    "frame_idx": f_cursor, "timestamp_sec": ts,
                    "track_id": 101 + ev_idx * 3, "class_name": "person", "team": t2_team, "is_predicted": False,
                    "bbox_x1": p2_x, "bbox_y1": p2_y, "bbox_x2": p2_x + 30.0, "bbox_y2": p2_y + 70.0,
                    "pixel_x": p2_x, "pixel_y": p2_y, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
                })
                rows.append({
                    "frame_idx": f_cursor, "timestamp_sec": ts,
                    "track_id": 102 + ev_idx * 3, "class_name": "person", "team": "Team B", "is_predicted": False,
                    "bbox_x1": opp_x, "bbox_y1": opp_y, "bbox_x2": opp_x + 30.0, "bbox_y2": opp_y + 70.0,
                    "pixel_x": opp_x, "pixel_y": opp_y, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
                })
                f_cursor += 1
            f_cursor += 105  # > PASS_MAX_GAP_SEC (4.0s @ 25fps), olaylar arası sahte devir önlenir
        return pd.DataFrame(rows)

    r_clean = MLAnalyzer(_build_pixel_real_events_df(False), fps=25.0, random_state=42).run_all()
    r_extreme = MLAnalyzer(_build_pixel_real_events_df(True), fps=25.0, random_state=42).run_all()

    for res in (r_clean, r_extreme):
        assert res["xp_modeling"]["data_source"] == "real_tracking_events"
        assert res["xp_modeling"]["bounds_source"] == "train_split_only"
        assert res["xp_modeling"]["scoring_unit_mode"] == "native_pixel_train_bounds"

        proxy_scored = res["xp_modeling"]["_proxy_scored_df"]
        model_scored = res["xp_modeling"]["_scored_events_df"]
        merged = proxy_scored.merge(
            model_scored,
            on=["frame_idx", "end_frame", "from_track_id", "to_track_id"],
            suffixes=("_proxy", "_model"),
        )
        assert len(merged) == len(model_scored)
        assert float(np.max(np.abs(merged["predicted_xp_proxy"] - merged["predicted_xp_model"]))) == 0.0

    train_count = int(r_clean["xp_modeling"]["split_info"]["train_count"])
    assert train_count >= 20
    pred_c = r_clean["_artifacts"]["pass_predictions_df"].iloc[:train_count]
    pred_e = r_extreme["_artifacts"]["pass_predictions_df"].iloc[:train_count]

    max_end_norm_x_diff = float(np.max(np.abs(pred_c["end_norm_x"].to_numpy() - pred_e["end_norm_x"].to_numpy())))
    max_xp_diff = float(np.max(np.abs(pred_c["predicted_xp"].to_numpy() - pred_e["predicted_xp"].to_numpy())))
    max_expected_xt_diff = float(np.max(np.abs(pred_c["expected_xT"].to_numpy() - pred_e["expected_xT"].to_numpy())))

    assert max_end_norm_x_diff == 0.0
    assert max_xp_diff == 0.0
    assert max_expected_xt_diff == 0.0


# ----------------------------------------------------------------------
# 14. DÜŞÜK KAPSAMDA LEGEND UYARISIZ ÇALIŞMA VE MODAL TAKIM ÇÖZÜMLEMESİ
# ----------------------------------------------------------------------
def test_all_low_coverage_no_legend_warning_and_modal_team_fatigue(tmp_path):
    # Tüm pencerelerin düşük kapsamlı (<15 örnek) ve tempo düşüşsüz olduğu senaryo
    low_cov_rows = []
    for f in range(6):
        ts = round(f / 25.0, 3)
        for tid, team, y0 in ((1, "Team A", 200.0), (2, "Team B", 350.0)):
            x0 = 150.0 + f * 4.0
            low_cov_rows.append({
                "frame_idx": f, "timestamp_sec": ts, "track_id": tid, "class_name": "person",
                "team": team, "is_predicted": False,
                "bbox_x1": x0, "bbox_y1": y0, "bbox_x2": x0 + 30.0, "bbox_y2": y0 + 70.0,
                "pixel_x": x0, "pixel_y": y0, "pitch_x": np.nan, "pitch_y": np.nan, "on_pitch": np.nan,
            })

    out_dir = tmp_path / "all_low_cov"
    low_cov_analyzer = MLAnalyzer(pd.DataFrame(low_cov_rows), fps=25.0, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = low_cov_analyzer.export_all(output_dir=str(out_dir), generate_plots=True)

    legend_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "No artists with labels found to put in legend" in str(w.message)
    ]
    assert len(legend_warnings) == 0
    assert summary["tempo_and_anomalies"]["total_windows"] >= 1
    assert summary["tempo_and_anomalies"]["covered_windows"] == 0
    assert (out_dir / "tempo_anomaly_timeline.png").exists()

    # Başta team=NaN sonra Team B olan oyuncuda modal bilinen takım çözümlenmeli; hiç takımı olmayan Unknown kalmalı
    fatigue_rows = []
    x1, x2 = 10.0, 15.0
    for f in range(150):
        ts = round(f / 25.0, 3)
        x1 += 0.20 if f < 75 else 0.04
        x2 += 0.12
        fatigue_rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 1, "class_name": "person",
            "team": np.nan if f < 5 else "Team B", "is_predicted": False,
            "bbox_x1": x1 * 10, "bbox_y1": 200.0, "bbox_x2": x1 * 10 + 30.0, "bbox_y2": 270.0,
            "pixel_x": x1 * 10, "pixel_y": 270.0, "pitch_x": x1, "pitch_y": 20.0, "on_pitch": True,
        })
        fatigue_rows.append({
            "frame_idx": f, "timestamp_sec": ts, "track_id": 2, "class_name": "person",
            "team": np.nan, "is_predicted": False,
            "bbox_x1": x2 * 10, "bbox_y1": 400.0, "bbox_x2": x2 * 10 + 30.0, "bbox_y2": 470.0,
            "pixel_x": x2 * 10, "pixel_y": 470.0, "pitch_x": x2, "pitch_y": 40.0, "on_pitch": True,
        })
    fatigue_rows.append({
        "frame_idx": 0, "timestamp_sec": 0.0, "track_id": "ball", "class_name": "ball",
        "team": None, "is_predicted": False,
        "bbox_x1": 500.0, "bbox_y1": 300.0, "bbox_x2": 510.0, "bbox_y2": 310.0,
        "pixel_x": 505.0, "pixel_y": 305.0, "pitch_x": 50.0, "pitch_y": 30.0, "on_pitch": True,
    })

    _, _, tp_sum = MLAnalyzer(pd.DataFrame(fatigue_rows), fps=25.0, random_state=42).detect_tempo_and_anomalies(window_sec=2.0)
    trends_by_id = {ft["track_id"]: ft for ft in tp_sum["player_fatigue_trends"]}
    assert trends_by_id["1"]["team"] == "Team B"
    assert trends_by_id["2"]["team"] == "Unknown"


