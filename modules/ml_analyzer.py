"""
ml_analyzer.py
--------------
ADIM 8: TAKTİK VE SCOUTING MAKİNE ÖĞRENMESİ MOTORU (MLAnalyzer)

Kare-kare takip verisi (DataExporter çıktısı, tracking_data.csv veya .db)
üzerinden uçtan uca makine öğrenmesi tabanlı futbol analitiği ve scouting
raporları üretir:
  - Veri Kalitesi & Eksik/NULL Denetimi (nedensel, gelecek sızıntısız işleme)
  - R1: Pas Başarı Olasılığı (xP) Denetimli Modelleme & Kalibrasyon
        (DummyClassifier vs LogisticRegression vs RandomForest vs GradientBoosting;
        kesin kronolojik train/val/test bölünmesi, sıfır veri sızıntısı,
        F1, ROC-AUC, PR-AUC, Brier skoru, karmaşıklık matrisi, operasyonel ödünleşimler)
  - R1: Markov Değer İterasyonu ile Mekansal Beklenen Tehdit (xT) Izgarası (12x8)
  - R2: Oyuncu Rol ve Profil Kümeleme (Silhouette ile optimal k seçimi + 2B PCA)
  - R3: Zaman Pencereli Tempo Düşüşü / Yorgunluk Eğilimi ve Konum/Hız Anomalisi
        (Kapsam farkındalıklı pencereleme, doğrusal eğim, MAD z-skoru + IsolationForest)
  - R4: Yapılandırılmış JSON, Türkçe Markdown raporu, CSV tabloları, PNG grafikler
        ve bağımsız CLI arayüzü.

Hem kalibre edilmiş saha (metre) koordinatlarıyla (`pitch_x`, `pitch_y`),
hem de kalibrasyon yapılmamış kayıtlarda piksel fallback moduyla (`pixel_x`, `pixel_y`)
sorunsuz çalışır.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import ConvexHull
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# ----------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------------------------------
def _clean_str(val: Any) -> Optional[str]:
    """NaN / None / boş metinleri güvenli biçimde None'a dönüştürür."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)) and (math.isnan(float(val)) or math.isinf(float(val))):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _to_python_scalar(obj: Any) -> Any:
    """NumPy / Pandas nesnelerini katı JSON (allow_nan=False) uyumlu Python tiplerine çevirir."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_python_scalar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_scalar(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_python_scalar(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (str, bytes)):
        return obj.decode("utf-8", errors="replace") if isinstance(obj, bytes) else obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _robust_mad_zscore(values: np.ndarray) -> np.ndarray:
    """Medyan Mutlak Sapma (MAD) tabanlı dayanıklı z-skoru hesaplar."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    med = np.nanmedian(arr)
    abs_dev = np.abs(arr - med)
    mad = np.nanmedian(abs_dev)
    if mad <= 1e-9 or math.isnan(mad):
        std = float(np.nanstd(arr))
        if std <= 1e-9 or math.isnan(std):
            return np.zeros_like(arr, dtype=float)
        return (arr - med) / std
    return 0.67448975 * (arr - med) / mad


def _zone_label_from_norm(nx: float, ny: float) -> str:
    """Normalize [0..1]x[0..1] saha koordinatından kategorik bölge adı üretir."""
    if nx < 1.0 / 3.0:
        depth = "savunma"
    elif nx < 2.0 / 3.0:
        depth = "orta_saha"
    else:
        depth = "hucum"

    if ny < 1.0 / 3.0:
        lane = "sol"
    elif ny < 2.0 / 3.0:
        lane = "merkez"
    else:
        lane = "sag"
    return f"{depth}_{lane}"


# ----------------------------------------------------------------------
# ANA ML ANALİZ SINIFI
# ----------------------------------------------------------------------
class MLAnalyzer:
    """
    Futbol takip verisi ve kalibre edilmiş sentetik maç verisi üzerinde
    xP (Pas Başarı Olasılığı), xT (Beklenen Tehdit), Oyuncu Rol Kümeleme
    ve Tempo/Yorgunluk/Anomali Tespiti çalıştıran taktik ML motoru.
    """

    XP_NUMERIC_FEATURES = [
        "ball_player_dist",
        "nearest_opponent_dist",
        "receiver_pressure_dist",
        "opponents_nearby",
        "pass_length",
        "forward_progress_dx",
        "pass_angle_cos",
        "pass_angle_sin",
        "passer_speed",
        "start_norm_x",
        "start_norm_y",
        "end_norm_x",
        "end_norm_y",
        "is_team_missing",
        "is_opponent_dist_missing",
        "is_ball_predicted",
        "is_passer_predicted",
    ]
    XP_CATEGORICAL_FEATURES = ["pitch_zone_start", "pitch_zone_end", "from_team"]

    def __init__(
        self,
        df: pd.DataFrame,
        fps: Optional[float] = None,
        analytics_summary: Optional[Dict[str, Any]] = None,
        random_state: int = config.ML_RANDOM_STATE,
    ):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.fps = float(fps) if fps and fps > 0 else self._infer_fps()
        self.analytics_summary = analytics_summary
        self.random_state = int(random_state)

        self.has_pitch_players, self.has_pitch_ball, self.is_calibrated = self._detect_calibration()
        self.coord_mode = "pitch_meters" if self.is_calibrated else "pixel_fallback"
        self.bounds = self._compute_coordinate_bounds()

        # İnceleme / test erişimi için eğitilmiş modeller ve son sonuçlar
        self.xp_models_: Dict[str, Pipeline] = {}
        self.best_xp_model_name_: Optional[str] = None
        self.best_xp_pipeline_: Optional[Pipeline] = None
        self.last_results_: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        fps: Optional[float] = None,
        analytics_summary: Optional[Dict[str, Any]] = None,
        random_state: int = config.ML_RANDOM_STATE,
    ) -> "MLAnalyzer":
        """CSV takip dosyasından MLAnalyzer örneği oluşturur."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        df = pd.read_csv(csv_path)
        return cls(df=df, fps=fps, analytics_summary=analytics_summary, random_state=random_state)

    # ----------------------------------------------------------------
    @classmethod
    def from_sql(
        cls,
        db_path: str,
        table_name: Optional[str] = None,
        fps: Optional[float] = None,
        analytics_summary: Optional[Dict[str, Any]] = None,
        random_state: int = config.ML_RANDOM_STATE,
    ) -> "MLAnalyzer":
        """SQLite veritabanından MLAnalyzer örneği oluşturur."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite dosyası bulunamadı: {db_path}")
        table = table_name or config.EXPORT_SQL_TABLE_NAME
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        finally:
            conn.close()
        return cls(df=df, fps=fps, analytics_summary=analytics_summary, random_state=random_state)

    # ----------------------------------------------------------------
    def _infer_fps(self) -> float:
        """frame_idx ve timestamp_sec farklarından FPS değerini çıkarır (varsayılan: 25.0)."""
        if self.df.empty or "frame_idx" not in self.df.columns or "timestamp_sec" not in self.df.columns:
            return 25.0

        ts_df = (
            self.df[["frame_idx", "timestamp_sec"]]
            .dropna()
            .drop_duplicates(subset=["frame_idx"])
            .sort_values("frame_idx")
        )
        if len(ts_df) < 2:
            return 25.0

        df_idx = ts_df["frame_idx"].diff()
        dt = ts_df["timestamp_sec"].diff()
        valid = (df_idx > 0) & (dt > 0)
        if not valid.any():
            return 25.0

        est = float((df_idx[valid] / dt[valid]).median())
        if est <= 0 or math.isnan(est) or math.isinf(est):
            return 25.0
        if abs(est - round(est)) < 0.05:
            return float(round(est))
        return round(est, 2)

    # ----------------------------------------------------------------
    def _detect_calibration(self) -> Tuple[bool, bool, bool]:
        """Hem oyuncu hem top satırlarında geçerli saha (metre) koordinatı olup olmadığını denetler."""
        if (
            self.df.empty
            or "pitch_x" not in self.df.columns
            or "pitch_y" not in self.df.columns
            or "class_name" not in self.df.columns
        ):
            return False, False, False

        has_valid_pitch = self.df["pitch_x"].notna() & self.df["pitch_y"].notna()
        is_ball = self.df["class_name"] == config.BALL_CLASS_NAME
        team_not_ref = (self.df["team"] != "Referee") if "team" in self.df.columns else True
        is_player = (
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (self.df["class_name"] != "referee")
            & team_not_ref
        )
        has_pitch_players = bool((has_valid_pitch & is_player).any())
        has_pitch_ball = bool((has_valid_pitch & is_ball).any())
        return has_pitch_players, has_pitch_ball, bool(has_pitch_players and has_pitch_ball)

    # ----------------------------------------------------------------
    def _get_valid_bbox_mask(self, df: pd.DataFrame) -> pd.Series:
        """Ters veya sıfır boyutlu bounding box hatalarını işaretler (True = geçerli bbox)."""
        if df.empty:
            return pd.Series(dtype=bool)
        required = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
        if not required.issubset(df.columns):
            return pd.Series(True, index=df.index)
        return (
            df["bbox_x1"].notna()
            & df["bbox_y1"].notna()
            & df["bbox_x2"].notna()
            & df["bbox_y2"].notna()
            & (df["bbox_x2"] > df["bbox_x1"])
            & (df["bbox_y2"] > df["bbox_y1"])
        )

    # ----------------------------------------------------------------
    def _compute_coordinate_bounds(self, max_frame_idx: Optional[int] = None) -> Tuple[float, float, float, float]:
        """Kalibre modda [0..105, 0..68], piksel modda kaçak tahminlerden arınmış alan sınırları."""
        if self.is_calibrated:
            return (0.0, float(config.PITCH_LENGTH_M), 0.0, float(config.PITCH_WIDTH_M))

        if self.df.empty or "pixel_x" not in self.df.columns or "pixel_y" not in self.df.columns:
            return (0.0, 1920.0, 0.0, 1080.0)

        obs = self.df.dropna(subset=["pixel_x", "pixel_y"]).copy()
        if max_frame_idx is not None and "frame_idx" in obs.columns:
            obs_split = obs[obs["frame_idx"] <= int(max_frame_idx)]
            if not obs_split.empty:
                obs = obs_split
        if "is_predicted" in obs.columns:
            obs_real = obs[obs["is_predicted"] != True]  # noqa: E712
            if not obs_real.empty:
                obs = obs_real
        valid_bbox = self._get_valid_bbox_mask(obs)
        if valid_bbox.any():
            obs = obs[valid_bbox]
        in_frame = obs[(obs["pixel_x"] >= 0.0) & (obs["pixel_y"] >= 0.0)]
        if not in_frame.empty:
            obs = in_frame

        if obs.empty:
            return (0.0, 1920.0, 0.0, 1080.0)

        min_x = float(obs["pixel_x"].min())
        max_x = float(obs["pixel_x"].max())
        min_y = float(obs["pixel_y"].min())
        max_y = float(obs["pixel_y"].max())
        if max_x - min_x < 10.0:
            max_x = min_x + 100.0
        if max_y - min_y < 10.0:
            max_y = min_y + 100.0
        return (min_x, max_x, min_y, max_y)

    # ----------------------------------------------------------------
    def _normalize_xy(
        self,
        x: float,
        y: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float]:
        """Koordinatı [0..1] x [0..1] normalize alana çevirir."""
        min_x, max_x, min_y, max_y = bounds if bounds is not None else self.bounds
        nx = float(np.clip((x - min_x) / max(1e-6, max_x - min_x), 0.0, 1.0))
        ny = float(np.clip((y - min_y) / max(1e-6, max_y - min_y), 0.0, 1.0))
        return nx, ny

    # ----------------------------------------------------------------
    def _get_xy_columns(self) -> Tuple[str, str]:
        return ("pitch_x", "pitch_y") if self.is_calibrated else ("pixel_x", "pixel_y")

    # ----------------------------------------------------------------
    def audit_data_quality(self) -> Dict[str, Any]:
        """
        Eksik/NULL değerleri, ters bbox'ları, kadraj dışı Kalman tahminlerini ve
        kalibrasyon durumunu açıkça denetler; veri kaybı olmadan işleme politikasını raporlar.
        """
        if self.df.empty:
            return {
                "total_rows": 0,
                "total_columns": int(len(self.df.columns)),
                "unique_frames": 0,
                "frame_range": [0, 0],
                "duration_sec": 0.0,
                "fps": self.fps,
                "coord_mode": self.coord_mode,
                "is_calibrated": self.is_calibrated,
                "missing_by_column": {},
                "domain_anomalies": {
                    "ball_rows": 0,
                    "human_rows": 0,
                    "predicted_rows_total": 0,
                    "predicted_human_rows": 0,
                    "predicted_ball_rows": 0,
                    "team_missing_ball": 0,
                    "team_missing_human": 0,
                    "confidence_missing_human": 0,
                    "confidence_missing_ball": 0,
                    "invalid_bbox_total": 0,
                    "invalid_bbox_predicted": 0,
                    "oob_pixel_total": 0,
                    "oob_pixel_predicted": 0,
                    "on_pitch_false_count": 0,
                    "transient_track_count": 0,
                    "transient_track_ids": [],
                },
                "handling_strategy": [
                    "Veri seti boş olduğu için yalnızca varsayılan şema raporlandı."
                ],
            }

        n_rows = int(len(self.df))
        n_cols = int(len(self.df.columns))
        unique_frames = int(self.df["frame_idx"].nunique()) if "frame_idx" in self.df.columns else 0
        has_valid_frames = "frame_idx" in self.df.columns and self.df["frame_idx"].notna().any()
        frame_min = int(self.df["frame_idx"].min()) if has_valid_frames else 0
        frame_max = int(self.df["frame_idx"].max()) if has_valid_frames else 0
        duration_sec = (
            round(float(self.df["timestamp_sec"].max() - self.df["timestamp_sec"].min()), 2)
            if "timestamp_sec" in self.df.columns and self.df["timestamp_sec"].notna().any()
            else round(unique_frames / self.fps, 2)
        )

        missing_by_column: Dict[str, Dict[str, Any]] = {}
        for col in self.df.columns:
            m_cnt = int(self.df[col].isna().sum())
            missing_by_column[str(col)] = {
                "dtype": str(self.df[col].dtype),
                "missing_count": m_cnt,
                "missing_pct": round((m_cnt / n_rows) * 100.0, 2) if n_rows > 0 else 0.0,
            }

        is_ball = (
            self.df["class_name"] == config.BALL_CLASS_NAME
            if "class_name" in self.df.columns
            else pd.Series(False, index=self.df.index)
        )
        is_human = (
            self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
            if "class_name" in self.df.columns
            else pd.Series(False, index=self.df.index)
        )
        is_pred = (
            self.df["is_predicted"].fillna(False).astype(bool)
            if "is_predicted" in self.df.columns
            else pd.Series(False, index=self.df.index)
        )

        team_na = (
            self.df["team"].isna()
            if "team" in self.df.columns
            else pd.Series(True, index=self.df.index)
        )
        team_na_ball = int((team_na & is_ball).sum())
        team_na_human = int((team_na & is_human).sum())

        conf_na_human = (
            int((self.df["confidence"].isna() & is_human).sum())
            if "confidence" in self.df.columns
            else n_rows
        )
        conf_na_ball = (
            int((self.df["confidence"].isna() & is_ball).sum())
            if "confidence" in self.df.columns
            else n_rows
        )

        valid_bbox = self._get_valid_bbox_mask(self.df)
        invalid_bbox_total = int((~valid_bbox).sum())
        invalid_bbox_pred = int((~valid_bbox & is_pred).sum())

        oob_total = 0
        oob_pred = 0
        if "pixel_x" in self.df.columns and "pixel_y" in self.df.columns:
            max_px_x = max(1920.0, float(self.bounds[1]))
            max_px_y = max(1080.0, float(self.bounds[3]))
            oob_mask = (
                (self.df["pixel_x"] < 0.0)
                | (self.df["pixel_y"] < 0.0)
                | (self.df["pixel_x"] > max_px_x)
                | (self.df["pixel_y"] > max_px_y)
            )
            oob_total = int(oob_mask.sum())
            oob_pred = int((oob_mask & is_pred).sum())

        on_pitch_false = (
            int((self.df["on_pitch"] == False).sum())  # noqa: E712
            if "on_pitch" in self.df.columns
            else 0
        )

        # Takip izlerinin gözlenen kare sayıları (geçici kısa izleri tespit et)
        transient_tracks: List[str] = []
        if "track_id" in self.df.columns:
            human_obs = self.df[is_human & (~is_pred)]
            if not human_obs.empty:
                obs_counts = human_obs["track_id"].astype(str).value_counts()
                transient_tracks = [
                    tid for tid, cnt in obs_counts.items() if int(cnt) < config.ML_MIN_TRACK_FRAMES
                ]

        handling_strategy = [
            f"Eksik 'team' değerleri (top: {team_na_ball}, oyuncu: {team_na_human}): "
            "Gelecek karelerden veri sızıntısı olmaması için kronolojik ayrım öncesi global "
            "backfill uygulanmadı; yalnızca geçmiş karelerden nedensel (causal) çözümleme ve "
            "eksik bayrağı ('is_team_missing') + eğitim setine fit edilen 'Unknown' kodlaması kullanıldı.",
            f"Eksik 'confidence' (insan: {conf_na_human}, top: {conf_na_ball}): "
            "Takip katmanında insan tespitlerinde güven skoru boş bırakıldığından model özelliği "
            "olarak zorlanmadı; bunun yerine 'is_predicted' bayrağı korundu.",
            f"Kalibrasyon durumu ({self.coord_mode}): "
            + (
                "Saha metre koordinatları (105x68m) mevcut; on_pitch=False noktalar hız/kümeleme dışında tutuldu."
                if self.is_calibrated
                else "pitch_x/pitch_y %100 NULL olduğundan otomatik piksel fallback ve [0..1] normalizasyon uygulandı."
            ),
            f"Tahmini (is_predicted=True) ve hatalı kutular (ters bbox: {invalid_bbox_total}, "
            f"kadraj dışı: {oob_total}, bunun {oob_pred} adedi tahmin kaynaklı): "
            "Hız, rol kümeleme ve tempo hesaplarında yalnızca gözlemlenen (is_predicted=False) ve "
            "geçerli kutulu satırlar kullanıldı; kaçak sıçramalar anomali olarak ayrıca raporlandı.",
            f"Kısa/geçici izler (< {config.ML_MIN_TRACK_FRAMES} gözlemlenen kare): "
            f"{len(transient_tracks)} iz rol kümelemeden hariç tutulup denetim listesine kaydedildi.",
        ]

        return {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "unique_frames": unique_frames,
            "frame_range": [frame_min, frame_max],
            "duration_sec": duration_sec,
            "fps": self.fps,
            "coord_mode": self.coord_mode,
            "is_calibrated": self.is_calibrated,
            "missing_by_column": missing_by_column,
            "domain_anomalies": {
                "ball_rows": int(is_ball.sum()),
                "human_rows": int(is_human.sum()),
                "predicted_rows_total": int(is_pred.sum()),
                "predicted_human_rows": int((is_pred & is_human).sum()),
                "predicted_ball_rows": int((is_pred & is_ball).sum()),
                "team_missing_ball": team_na_ball,
                "team_missing_human": team_na_human,
                "confidence_missing_human": conf_na_human,
                "confidence_missing_ball": conf_na_ball,
                "invalid_bbox_total": invalid_bbox_total,
                "invalid_bbox_predicted": invalid_bbox_pred,
                "oob_pixel_total": oob_total,
                "oob_pixel_predicted": oob_pred,
                "on_pitch_false_count": on_pitch_false,
                "transient_track_count": len(transient_tracks),
                "transient_track_ids": transient_tracks,
            },
            "handling_strategy": handling_strategy,
        }

    # ----------------------------------------------------------------
    def _compute_player_step_speeds(self, exclude_referees: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Gözlemlenen ardışık kareler arasında oyuncu adım hızlarını hesaplar.
        Gerçek dışı hız sıçramalarını (ışınlanma / kamera kesmesi) hız ortalamalarından
        ayırır ve anomali adayı olarak döndürür.
        """
        x_col, y_col = self._get_xy_columns()
        empty_cols = [
            "frame_idx", "timestamp_sec", "track_id", "team",
            "x", "y", "norm_x", "norm_y", "speed", "dx", "dy", "dt",
        ]
        if (
            self.df.empty
            or x_col not in self.df.columns
            or y_col not in self.df.columns
            or "class_name" not in self.df.columns
            or "frame_idx" not in self.df.columns
        ):
            return pd.DataFrame(columns=empty_cols), []

        valid_bbox = self._get_valid_bbox_mask(self.df)
        mask = self.df["class_name"].isin(config.HUMAN_CLASS_NAMES) & valid_bbox
        if exclude_referees:
            team_not_ref = (self.df["team"] != "Referee") if "team" in self.df.columns else True
            mask &= (self.df["class_name"] != "referee") & team_not_ref
        if "is_predicted" in self.df.columns:
            mask &= self.df["is_predicted"] != True  # noqa: E712
        if self.is_calibrated and "on_pitch" in self.df.columns:
            mask &= self.df["on_pitch"] != False  # noqa: E712

        sub = self.df[mask].dropna(subset=[x_col, y_col, "frame_idx"]).copy()
        if sub.empty:
            return pd.DataFrame(columns=empty_cols), []

        if not self.is_calibrated:
            min_x, max_x, min_y, max_y = self.bounds
            sub = sub[
                (sub[x_col] >= min_x - 30.0)
                & (sub[x_col] <= max_x + 30.0)
                & (sub[y_col] >= min_y - 30.0)
                & (sub[y_col] <= max_y + 30.0)
            ]

        max_speed_cap = (
            float(config.MAX_REALISTIC_PLAYER_SPEED_KMH)
            if self.is_calibrated
            else float(config.ML_PIXEL_MAX_SPEED_PX_PER_SEC)
        )
        max_gap = max(1, int(config.DISTANCE_CALC_MAX_FRAME_GAP))
        smooth_win = min(3, max(1, int(config.SPEED_SMOOTHING_WINDOW_FRAMES)))

        speed_rows: List[Dict[str, Any]] = []
        spike_anomalies: List[Dict[str, Any]] = []

        # Nedensel takım çözümlemesi için track bazında son bilinen takım
        for tid_raw, grp in sub.groupby("track_id"):
            tid = str(tid_raw)
            grp = grp.sort_values("frame_idx").drop_duplicates(subset=["frame_idx"]).reset_index(drop=True)
            if len(grp) < 2:
                continue

            fdiff = grp["frame_idx"].diff()
            seg_ids = (fdiff.isna() | (fdiff > max_gap)).cumsum()

            # Nedensel takım: ilk geçerli takım gözlemi sonrası taşınır
            running_team: Optional[str] = None
            resolved_teams: List[Optional[str]] = []
            team_series = grp["team"].tolist() if "team" in grp.columns else [None] * len(grp)
            for t_val in team_series:
                ct = _clean_str(t_val)
                if ct is not None:
                    running_team = ct
                resolved_teams.append(ct if ct is not None else running_team)
            grp["_causal_team"] = resolved_teams

            for _, seg in grp.groupby(seg_ids):
                seg = seg.reset_index(drop=True)
                if len(seg) < 2:
                    continue

                raw_xs = seg[x_col].to_numpy(dtype=float)
                raw_ys = seg[y_col].to_numpy(dtype=float)
                # Kesinlikle nedensel (center=False) geriye dönük hareketli ortalama için adım tamponu
                valid_step_speeds: List[float] = []

                for i in range(1, len(seg)):
                    df_i = int(seg.loc[i, "frame_idx"] - seg.loc[i - 1, "frame_idx"])
                    if df_i <= 0 or df_i > max_gap:
                        valid_step_speeds.clear()
                        continue
                    dt = df_i / self.fps
                    if dt <= 0:
                        valid_step_speeds.clear()
                        continue

                    dx = float(raw_xs[i] - raw_xs[i - 1])
                    dy = float(raw_ys[i] - raw_ys[i - 1])
                    dist = math.hypot(dx, dy)
                    raw_speed = (dist / dt) * 3.6 if self.is_calibrated else (dist / dt)

                    ts_raw = seg.loc[i, "timestamp_sec"] if "timestamp_sec" in seg.columns else None
                    ts = (
                        float(ts_raw)
                        if ts_raw is not None and not pd.isna(ts_raw)
                        else float(seg.loc[i, "frame_idx"]) / self.fps
                    )
                    x_val = float(raw_xs[i])
                    y_val = float(raw_ys[i])
                    nx, ny = self._normalize_xy(x_val, y_val)

                    if raw_speed > max_speed_cap:
                        unit = "km/sa" if self.is_calibrated else "px/sn"
                        spike_anomalies.append({
                            "anomaly_type": "speed_spike",
                            "severity": "high",
                            "frame_idx": int(seg.loc[i, "frame_idx"]),
                            "timestamp_sec": round(ts, 2),
                            "track_id": tid,
                            "team": seg.loc[i, "_causal_team"] or "Unknown",
                            "metric_value": round(raw_speed, 2),
                            "threshold": max_speed_cap,
                            "description_tr": (
                                f"Oyuncu {tid} için fiziksel sınır üstü hız sıçraması "
                                f"({raw_speed:.1f} {unit} > {max_speed_cap:.1f} {unit}); ortalama hesaptan dışlandı."
                            ),
                        })
                        # Sıçrama sonraki adımların nedensel penceresini kirletmesin
                        valid_step_speeds.clear()
                        continue

                    valid_step_speeds.append(raw_speed)
                    speed = float(np.mean(valid_step_speeds[-smooth_win:]))

                    speed_rows.append({
                        "frame_idx": int(seg.loc[i, "frame_idx"]),
                        "timestamp_sec": round(ts, 3),
                        "track_id": tid,
                        "team": seg.loc[i, "_causal_team"] or "Unknown",
                        "x": x_val,
                        "y": y_val,
                        "norm_x": nx,
                        "norm_y": ny,
                        "speed": speed,
                        "dx": dx,
                        "dy": dy,
                        "dt": dt,
                    })

        return pd.DataFrame(speed_rows, columns=empty_cols), spike_anomalies

    # ----------------------------------------------------------------
    def extract_proxy_pass_events(self) -> pd.DataFrame:
        """
        Ham takip verisindeki top-oyuncu dinamiklerinden vekil (proxy) pas ve
        top el değiştirme olaylarını çıkarır. Takım çözümlemesi KESİNLİKLE
        nedenseldir (yalnızca geçmiş kareler kullanılır, gelecek sızıntısı yoktur).
        """
        event_cols = [
            "frame_idx", "timestamp_sec", "end_frame", "end_timestamp_sec",
            "from_track_id", "to_track_id", "from_team", "to_team",
            "start_x", "start_y", "end_x", "end_y",
            "start_norm_x", "start_norm_y", "end_norm_x", "end_norm_y",
            "ball_player_dist", "nearest_opponent_dist", "receiver_pressure_dist",
            "opponents_nearby", "pass_length", "forward_progress_dx",
            "pass_angle_cos", "pass_angle_sin", "passer_speed",
            "pitch_zone_start", "pitch_zone_end",
            "is_team_missing", "is_opponent_dist_missing",
            "is_ball_predicted", "is_passer_predicted", "pass_outcome",
        ]
        x_col, y_col = self._get_xy_columns()
        if (
            self.df.empty
            or x_col not in self.df.columns
            or y_col not in self.df.columns
            or "class_name" not in self.df.columns
            or "frame_idx" not in self.df.columns
        ):
            return pd.DataFrame(columns=event_cols)

        df_sorted = self.df.sort_values("frame_idx").copy()

        # Top verisi (aynı karede varsa is_predicted=False tercih edilir)
        ball_df = df_sorted[df_sorted["class_name"] == config.BALL_CLASS_NAME].dropna(subset=[x_col, y_col]).copy()
        if ball_df.empty:
            return pd.DataFrame(columns=event_cols)
        if "is_predicted" in ball_df.columns:
            ball_df["_pred_sort"] = ball_df["is_predicted"].fillna(False).astype(bool)
            ball_df = ball_df.sort_values(["frame_idx", "_pred_sort"]).drop_duplicates(subset=["frame_idx"])
        else:
            ball_df["_pred_sort"] = False
            ball_df = ball_df.sort_values("frame_idx").drop_duplicates(subset=["frame_idx"])

        valid_bbox = self._get_valid_bbox_mask(df_sorted)
        team_not_ref = (df_sorted["team"] != "Referee") if "team" in df_sorted.columns else True
        players_df = df_sorted[
            df_sorted["class_name"].isin(config.HUMAN_CLASS_NAMES)
            & (df_sorted["class_name"] != "referee")
            & team_not_ref
            & valid_bbox
        ].dropna(subset=[x_col, y_col]).copy()
        if self.is_calibrated and "on_pitch" in players_df.columns:
            players_df = players_df[players_df["on_pitch"] != False]  # noqa: E712

        if players_df.empty:
            return pd.DataFrame(columns=event_cols)

        max_ctrl_dist = (
            float(config.POSSESSION_MAX_PLAYER_BALL_DIST_M)
            if self.is_calibrated
            else float(config.POSSESSION_MAX_PLAYER_BALL_DIST_PX)
        )
        pressure_radius = 5.0 if self.is_calibrated else 150.0
        min_move = (
            float(config.PASS_MIN_BALL_MOVE_M)
            if self.is_calibrated
            else float(config.PASS_MIN_BALL_MOVE_PX)
        )

        # Oyuncu hızlarını nedensel sözlük olarak hazırla
        speed_df, _ = self._compute_player_step_speeds(exclude_referees=True)
        speed_lookup: Dict[Tuple[int, str], float] = {}
        if not speed_df.empty:
            for _, sr in speed_df.iterrows():
                speed_lookup[(int(sr["frame_idx"]), str(sr["track_id"]))] = float(sr["speed"])

        players_by_frame = {int(fid): grp for fid, grp in players_df.groupby("frame_idx")}
        ball_by_frame = {int(brow["frame_idx"]): brow for _, brow in ball_df.iterrows()}

        # NEDENSEL (CAUSAL) takip: yalnızca kare t ve öncesinden takım öğren
        causal_team_map: Dict[str, str] = {}
        causal_teams_by_frame: Dict[int, Dict[str, Optional[str]]] = {}
        frame_snapshots: List[Dict[str, Any]] = []

        all_frames = sorted(set(ball_by_frame.keys()) | set(players_by_frame.keys()))
        for fid in all_frames:
            f_players = players_by_frame.get(fid)
            if f_players is not None and not f_players.empty:
                # Önce kare içindeki anlık ve geçmiş takımları çöz (gelecekten okuma YOK)
                resolved_list: List[Optional[str]] = []
                raw_missing_list: List[int] = []
                frame_team_dict: Dict[str, Optional[str]] = {}
                for _, prow in f_players.iterrows():
                    tid = _clean_str(prow.get("track_id"))
                    raw_team = _clean_str(prow.get("team"))
                    if raw_team is not None:
                        resolved_team = raw_team
                        raw_missing_list.append(0)
                    else:
                        resolved_team = causal_team_map.get(tid) if tid else None
                        raw_missing_list.append(1)
                    resolved_list.append(resolved_team)
                    if tid is not None:
                        frame_team_dict[tid] = resolved_team
                causal_teams_by_frame[fid] = frame_team_dict

                # Kare bittikten sonra bilinen takımları gelecek kareler için güncelle
                for _, prow in f_players.iterrows():
                    tid = _clean_str(prow.get("track_id"))
                    raw_team = _clean_str(prow.get("team"))
                    if tid and raw_team is not None:
                        causal_team_map[tid] = raw_team

                brow = ball_by_frame.get(fid)
                if brow is None:
                    continue

                bx = float(brow[x_col])
                by = float(brow[y_col])
                ts = (
                    float(brow["timestamp_sec"])
                    if "timestamp_sec" in brow and not pd.isna(brow["timestamp_sec"])
                    else fid / self.fps
                )
                pxs = f_players[x_col].to_numpy(dtype=float)
                pys = f_players[y_col].to_numpy(dtype=float)
                dists = np.hypot(pxs - bx, pys - by)
                min_idx = int(np.argmin(dists))
                min_d = float(dists[min_idx])

                chosen = f_players.iloc[min_idx]
                chosen_tid = _clean_str(chosen.get("track_id"))
                chosen_team = resolved_list[min_idx]
                is_team_missing = raw_missing_list[min_idx]
                is_passer_pred = int(bool(chosen.get("is_predicted"))) if "is_predicted" in chosen and not pd.isna(chosen.get("is_predicted")) else 0
                is_ball_pred = int(bool(brow.get("_pred_sort")))

                if min_d <= max_ctrl_dist and chosen_tid is not None:
                    # Rakip baskı mesafesi (farklı bilinen takımdaki en yakın oyuncu)
                    opp_dists: List[float] = []
                    for j in range(len(f_players)):
                        if j == min_idx:
                            continue
                        other_team = resolved_list[j]
                        if chosen_team is not None and other_team is not None and other_team != chosen_team:
                            d_opp = float(math.hypot(pxs[j] - pxs[min_idx], pys[j] - pys[min_idx]))
                            opp_dists.append(d_opp)

                    nearest_opp = min(opp_dists) if opp_dists else np.nan
                    opp_nearby = sum(1 for d in opp_dists if d <= pressure_radius)

                    frame_snapshots.append({
                        "frame_idx": fid,
                        "timestamp_sec": ts,
                        "ball_x": bx,
                        "ball_y": by,
                        "track_id": chosen_tid,
                        "team": chosen_team,
                        "ball_player_dist": min_d,
                        "nearest_opponent_dist": nearest_opp,
                        "opponents_nearby": opp_nearby,
                        "passer_speed": speed_lookup.get((fid, chosen_tid), 0.0),
                        "is_team_missing": is_team_missing,
                        "is_ball_predicted": is_ball_pred,
                        "is_passer_predicted": is_passer_pred,
                    })

        if len(frame_snapshots) < 2:
            return pd.DataFrame(columns=event_cols)

        # Ardışık kontrol periyotları (spells) oluştur
        spells: List[Dict[str, Any]] = []
        for snap in frame_snapshots:
            if spells and spells[-1]["track_id"] == snap["track_id"]:
                spells[-1]["end"] = snap
            else:
                spells.append({"track_id": snap["track_id"], "start": snap, "end": snap})

        events: List[Dict[str, Any]] = []
        for i in range(len(spells) - 1):
            s1_end = spells[i]["end"]
            s2_start = spells[i + 1]["start"]
            if s1_end["track_id"] == s2_start["track_id"]:
                continue

            gap_sec = (s2_start["frame_idx"] - s1_end["frame_idx"]) / self.fps
            dx = s2_start["ball_x"] - s1_end["ball_x"]
            dy = s2_start["ball_y"] - s1_end["ball_y"]
            pass_len = math.hypot(dx, dy)

            if gap_sec > config.PASS_MAX_GAP_SEC or pass_len < min_move:
                continue

            sx_n, sy_n = self._normalize_xy(s1_end["ball_x"], s1_end["ball_y"])
            ex_n, ey_n = self._normalize_xy(s2_start["ball_x"], s2_start["ball_y"])
            angle = math.atan2(dy, dx) if pass_len > 1e-6 else 0.0

            from_team = s1_end["team"] or "Unknown"
            to_team = s2_start["team"] or "Unknown"
            if from_team != "Unknown" and to_team != "Unknown":
                outcome: Any = 1 if from_team == to_team else 0
            else:
                outcome = np.nan

            # Yönlü ilerleme (bilinen Team B için ters eksen varsayımı yalnızca yön bilgisi tutarlıysa)
            fwd_dx = (sx_n - ex_n) if from_team == "Team B" else (ex_n - sx_n)

            cols_n = int(config.ML_XT_GRID_COLS)
            rows_n = int(config.ML_XT_GRID_ROWS)
            opp_dist = s1_end["nearest_opponent_dist"]

            # Alıcı baskı mesafesini gelecek kareden (s2_start) DEĞİL, pasın bırakıldığı
            # serbest bırakma karesinden (s1_end) nedensel olarak ölç (sıfır gelecek sızıntısı).
            rec_pressure: Any = np.nan
            rel_fid = int(s1_end["frame_idx"])
            to_tid_str = str(s2_start["track_id"])
            rel_players = players_by_frame.get(rel_fid)
            rel_teams = causal_teams_by_frame.get(rel_fid, {})
            if rel_players is not None and not rel_players.empty:
                rel_tids = rel_players["track_id"].astype(str).tolist()
                if to_tid_str in rel_tids:
                    rec_idx = rel_tids.index(to_tid_str)
                    rel_xs = rel_players[x_col].to_numpy(dtype=float)
                    rel_ys = rel_players[y_col].to_numpy(dtype=float)
                    rec_x, rec_y = float(rel_xs[rec_idx]), float(rel_ys[rec_idx])
                    rec_team = rel_teams.get(to_tid_str)
                    ref_team = rec_team if rec_team is not None else (from_team if from_team != "Unknown" else None)
                    if ref_team is not None:
                        rec_opp_dists = [
                            float(math.hypot(rel_xs[j] - rec_x, rel_ys[j] - rec_y))
                            for j, tid_j in enumerate(rel_tids)
                            if tid_j != to_tid_str
                            and rel_teams.get(tid_j) is not None
                            and rel_teams.get(tid_j) != ref_team
                        ]
                        if rec_opp_dists:
                            rec_pressure = min(rec_opp_dists)

            events.append({
                "frame_idx": int(s1_end["frame_idx"]),
                "timestamp_sec": round(float(s1_end["timestamp_sec"]), 3),
                "end_frame": int(s2_start["frame_idx"]),
                "end_timestamp_sec": round(float(s2_start["timestamp_sec"]), 3),
                "from_track_id": str(s1_end["track_id"]),
                "to_track_id": str(s2_start["track_id"]),
                "from_team": from_team if s1_end["team"] is not None else np.nan,
                "to_team": to_team if s2_start["team"] is not None else np.nan,
                "start_x": round(float(s1_end["ball_x"]), 2),
                "start_y": round(float(s1_end["ball_y"]), 2),
                "end_x": round(float(s2_start["ball_x"]), 2),
                "end_y": round(float(s2_start["ball_y"]), 2),
                "start_norm_x": round(sx_n, 4),
                "start_norm_y": round(sy_n, 4),
                "end_norm_x": round(ex_n, 4),
                "end_norm_y": round(ey_n, 4),
                "ball_player_dist": round(float(s1_end["ball_player_dist"]), 3),
                "nearest_opponent_dist": round(float(opp_dist), 3) if not pd.isna(opp_dist) else np.nan,
                "receiver_pressure_dist": round(float(rec_pressure), 3) if not pd.isna(rec_pressure) else np.nan,
                "opponents_nearby": int(s1_end["opponents_nearby"]),
                "pass_length": round(float(pass_len), 3),
                "forward_progress_dx": round(float(fwd_dx), 4),
                "pass_angle_cos": round(float(math.cos(angle)), 4),
                "pass_angle_sin": round(float(math.sin(angle)), 4),
                "passer_speed": round(float(s1_end["passer_speed"]), 3),
                "start_zone_x": min(cols_n - 1, max(0, int(sx_n * cols_n))),
                "start_zone_y": min(rows_n - 1, max(0, int(sy_n * rows_n))),
                "end_zone_x": min(cols_n - 1, max(0, int(ex_n * cols_n))),
                "end_zone_y": min(rows_n - 1, max(0, int(ey_n * rows_n))),
                "pitch_zone_start": _zone_label_from_norm(sx_n, sy_n),
                "pitch_zone_end": _zone_label_from_norm(ex_n, ey_n),
                "is_team_missing": int(s1_end["is_team_missing"]),
                "is_opponent_dist_missing": 1 if pd.isna(opp_dist) else 0,
                "is_ball_predicted": int(s1_end["is_ball_predicted"]),
                "is_passer_predicted": int(s1_end["is_passer_predicted"]),
                "pass_outcome": outcome,
            })

        return pd.DataFrame(events, columns=event_cols)

    # ----------------------------------------------------------------
    def generate_synthetic_match_passes(
        self,
        n_events: int = config.ML_SYNTHETIC_MATCH_EVENTS,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        105m x 68m kalibre edilmiş saha üzerinde deterministik, fizik temelli
        kronolojik sentetik maç pas veri seti üretir. Gerçekçi eksik takım/sensör
        gürültüsü ve doğrusal olmayan futbol fiziği içerir.
        """
        rng = np.random.default_rng(self.random_state if seed is None else int(seed))
        n = max(100, int(n_events))
        pitch_l = float(config.PITCH_LENGTH_M)
        pitch_w = float(config.PITCH_WIDTH_M)
        fps = float(self.fps or 25.0)
        cols_n = int(config.ML_XT_GRID_COLS)
        rows_n = int(config.ML_XT_GRID_ROWS)

        # Kronolojik zaman damgaları (0 .. 5400 sn)
        timestamps = np.sort(rng.uniform(1.0, 5400.0, size=n))
        frame_indices = np.round(timestamps * fps).astype(int)

        rows: List[Dict[str, Any]] = []
        for i in range(n):
            t_sec = float(timestamps[i])
            f_idx = int(frame_indices[i])
            team = "Team A" if rng.random() < 0.52 else "Team B"
            attack_dir = 1.0 if team == "Team A" else -1.0

            # Saha başlangıç pozisyonu (orta saha ağırlıklı beta dağılımı)
            sx = float(np.clip(rng.beta(2.2, 2.2) * pitch_l, 2.0, pitch_l - 2.0))
            sy = float(np.clip(rng.normal(pitch_w * 0.5, 16.0), 1.5, pitch_w - 1.5))

            # Pas mesafesi ve açısı
            pass_len = float(np.clip(rng.gamma(shape=2.4, scale=7.0) + 2.5, 2.5, 58.0))
            # Takım hücum yönüne hafif eğilimli açı
            base_angle = 0.0 if attack_dir > 0 else math.pi
            angle = float(base_angle + rng.normal(0.0, 1.05))
            ex = float(np.clip(sx + pass_len * math.cos(angle), 0.5, pitch_l - 0.5))
            ey = float(np.clip(sy + pass_len * math.sin(angle), 0.5, pitch_w - 0.5))
            actual_len = math.hypot(ex - sx, ey - sy)
            actual_angle = math.atan2(ey - sy, ex - sx) if actual_len > 1e-6 else 0.0

            # Kinematik & baskı özellikleri
            ball_player_dist = float(np.clip(rng.gamma(1.8, 0.45), 0.15, 2.45))
            nearest_opp_dist = float(np.clip(rng.gamma(2.5, 2.1), 0.5, 22.0))
            receiver_press_dist = float(np.clip(rng.gamma(2.4, 2.0), 0.5, 22.0))
            opp_nearby = int(np.clip(rng.poisson(max(0.2, 3.2 - nearest_opp_dist * 0.35)), 0, 5))
            passer_speed = float(np.clip(rng.gamma(2.0, 1.3), 0.1, 8.8))

            sx_n = sx / pitch_l
            sy_n = sy / pitch_w
            ex_n = ex / pitch_l
            ey_n = ey / pitch_w
            fwd_dx = ((ex - sx) * attack_dir) / pitch_l

            # Futbol fiziğine dayalı doğrusal olmayan pas başarı olasılığı (xP)
            # - Uzun pas + yakın baskı etkileşimi başarıyı keskin düşürür
            # - Ceza sahasına / kalabalık alana dikine paslar daha risklidir
            # - Kontrollü top + düşük baskı yüksek xP verir
            pressure_term = -2.45 * math.exp(-nearest_opp_dist / 2.4) - 1.95 * math.exp(-receiver_press_dist / 2.2)
            length_term = -0.095 * max(0.0, actual_len - 9.0)
            interaction_term = -0.28 * (max(0.0, actual_len - 10.0) / 8.0) * max(0.0, 4.0 - nearest_opp_dist)
            density_term = -0.55 * opp_nearby - 0.65 * max(0.0, ball_player_dist - 0.9) - 0.14 * passer_speed
            # Hücum 3. bölge sıkışıklığı
            target_attack_depth = ex_n if attack_dir > 0 else (1.0 - ex_n)
            zone_term = -0.85 * max(0.0, target_attack_depth - 0.60) - 0.65 * max(0.0, fwd_dx)

            logit = 3.95 + pressure_term + length_term + interaction_term + density_term + zone_term
            p_success = 1.0 / (1.0 + math.exp(-logit))
            outcome = int(rng.binomial(1, p_success))

            # Gerçekçi eksik/gürültülü veri enjeksiyonu (%6 eksik team, %5 eksik rakip mesafesi, %4 tahmin top)
            team_missing = 1 if rng.random() < 0.06 else 0
            opp_missing = 1 if rng.random() < 0.05 else 0
            ball_pred = 1 if rng.random() < 0.04 else 0
            passer_pred = 1 if rng.random() < 0.03 else 0

            from_id = str(int(rng.integers(1, 12))) if team == "Team A" else str(int(rng.integers(12, 23)))
            to_id = from_id
            while to_id == from_id:
                if outcome == 1:
                    to_id = str(int(rng.integers(1, 12))) if team == "Team A" else str(int(rng.integers(12, 23)))
                else:
                    to_id = str(int(rng.integers(12, 23))) if team == "Team A" else str(int(rng.integers(1, 12)))

            rows.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(t_sec, 3),
                "end_frame": f_idx + int(max(3, round(actual_len * 1.4))),
                "end_timestamp_sec": round(t_sec + max(0.2, actual_len / 14.0), 3),
                "from_track_id": from_id,
                "to_track_id": to_id,
                "from_team": np.nan if team_missing else team,
                "to_team": team if outcome == 1 else ("Team B" if team == "Team A" else "Team A"),
                "start_x": round(sx, 2),
                "start_y": round(sy, 2),
                "end_x": round(ex, 2),
                "end_y": round(ey, 2),
                "start_norm_x": round(sx_n, 4),
                "start_norm_y": round(sy_n, 4),
                "end_norm_x": round(ex_n, 4),
                "end_norm_y": round(ey_n, 4),
                "ball_player_dist": round(ball_player_dist, 3),
                "nearest_opponent_dist": np.nan if opp_missing else round(nearest_opp_dist, 3),
                "receiver_pressure_dist": round(receiver_press_dist, 3),
                "opponents_nearby": opp_nearby,
                "pass_length": round(actual_len, 3),
                "forward_progress_dx": round(fwd_dx, 4),
                "pass_angle_cos": round(math.cos(actual_angle), 4),
                "pass_angle_sin": round(math.sin(actual_angle), 4),
                "passer_speed": round(passer_speed, 3),
                "start_zone_x": min(cols_n - 1, max(0, int(sx_n * cols_n))),
                "start_zone_y": min(rows_n - 1, max(0, int(sy_n * rows_n))),
                "end_zone_x": min(cols_n - 1, max(0, int(ex_n * cols_n))),
                "end_zone_y": min(rows_n - 1, max(0, int(ey_n * rows_n))),
                "pitch_zone_start": _zone_label_from_norm(sx_n, sy_n),
                "pitch_zone_end": _zone_label_from_norm(ex_n, ey_n),
                "is_team_missing": team_missing,
                "is_opponent_dist_missing": opp_missing,
                "is_ball_predicted": ball_pred,
                "is_passer_predicted": passer_pred,
                "pass_outcome": outcome,
            })

        return pd.DataFrame(rows)

    # ----------------------------------------------------------------
    def _build_xp_preprocessor(self, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        """Yalnızca eğitim setine fit edilmek üzere ön işleme boru hattı kurar."""
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _evaluate_binary_classifier(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Accuracy ötesi tüm sınıflandırma ve kalibrasyon metriklerini hesaplar."""
        acc = float(accuracy_score(y_true, y_pred))
        f1_bin = float(f1_score(y_true, y_pred, zero_division=0))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            roc = float(roc_auc_score(y_true, y_prob))
            pr = float(average_precision_score(y_true, y_prob))
            ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6), labels=[0, 1]))
            prob_true, prob_pred = calibration_curve(y_true, np.clip(y_prob, 0.0, 1.0), n_bins=8, strategy="uniform")
            cal_curve = {
                "prob_true": [round(float(v), 4) for v in prob_true],
                "prob_pred": [round(float(v), 4) for v in prob_pred],
            }
        else:
            roc = 0.5
            pr = float(np.mean(y_true))
            ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6), labels=[0, 1]))
            cal_curve = {"prob_true": [], "prob_pred": []}

        brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0), pos_label=1))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        return {
            "accuracy": round(acc, 4),
            "f1": round(f1_bin, 4),
            "f1_macro": round(f1_mac, 4),
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "confusion_matrix": cm,
            "calibration_curve": cal_curve,
        }

    # ----------------------------------------------------------------
    def _project_proxy_events_to_pitch_units(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Piksel fallback modundaki vekil olay özelliklerini (px, px/sn) kalibre
        sentetik xP modeliyle uyumlu [0..105]x[0..68] metre ölçeğine izdüşürür.
        """
        df_proj = df_in.copy()
        min_x, max_x, min_y, max_y = self.bounds
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        scale_x = float(config.PITCH_LENGTH_M) / span_x
        scale_y = float(config.PITCH_WIDTH_M) / span_y
        scale_iso = math.sqrt(scale_x * scale_y)

        if {"start_norm_x", "start_norm_y", "end_norm_x", "end_norm_y"}.issubset(df_proj.columns):
            dx_m = (df_proj["end_norm_x"].astype(float) - df_proj["start_norm_x"].astype(float)) * float(config.PITCH_LENGTH_M)
            dy_m = (df_proj["end_norm_y"].astype(float) - df_proj["start_norm_y"].astype(float)) * float(config.PITCH_WIDTH_M)
            df_proj["pass_length"] = np.hypot(dx_m, dy_m)
        elif "pass_length" in df_proj.columns:
            df_proj["pass_length"] = df_proj["pass_length"].astype(float) * scale_iso

        for col in ("ball_player_dist", "nearest_opponent_dist", "receiver_pressure_dist", "passer_speed"):
            if col in df_proj.columns:
                df_proj[col] = df_proj[col].astype(float) * scale_iso
        return df_proj

    # ----------------------------------------------------------------
    def _recompute_event_normalization_with_bounds(
        self,
        df_events: pd.DataFrame,
        bounds: Tuple[float, float, float, float],
    ) -> pd.DataFrame:
        """Verilen piksel sınırlarıyla normalize saha ve bölge özelliklerini yeniden hesaplar."""
        if df_events.empty or not {"start_x", "start_y", "end_x", "end_y"}.issubset(df_events.columns):
            return df_events
        out = df_events.copy()
        cols_n = int(config.ML_XT_GRID_COLS)
        rows_n = int(config.ML_XT_GRID_ROWS)

        sx_ns: List[float] = []
        sy_ns: List[float] = []
        ex_ns: List[float] = []
        ey_ns: List[float] = []
        fwd_dxs: List[float] = []
        sz_xs: List[int] = []
        sz_ys: List[int] = []
        ez_xs: List[int] = []
        ez_ys: List[int] = []
        pz_s: List[str] = []
        pz_e: List[str] = []

        for _, row in out.iterrows():
            sx_n, sy_n = self._normalize_xy(float(row["start_x"]), float(row["start_y"]), bounds=bounds)
            ex_n, ey_n = self._normalize_xy(float(row["end_x"]), float(row["end_y"]), bounds=bounds)
            from_team = _clean_str(row.get("from_team")) or "Unknown"
            fwd_dx = (sx_n - ex_n) if from_team == "Team B" else (ex_n - sx_n)
            sx_ns.append(round(sx_n, 4))
            sy_ns.append(round(sy_n, 4))
            ex_ns.append(round(ex_n, 4))
            ey_ns.append(round(ey_n, 4))
            fwd_dxs.append(round(float(fwd_dx), 4))
            sz_xs.append(min(cols_n - 1, max(0, int(sx_n * cols_n))))
            sz_ys.append(min(rows_n - 1, max(0, int(sy_n * rows_n))))
            ez_xs.append(min(cols_n - 1, max(0, int(ex_n * cols_n))))
            ez_ys.append(min(rows_n - 1, max(0, int(ey_n * rows_n))))
            pz_s.append(_zone_label_from_norm(sx_n, sy_n))
            pz_e.append(_zone_label_from_norm(ex_n, ey_n))

        out["start_norm_x"] = sx_ns
        out["start_norm_y"] = sy_ns
        out["end_norm_x"] = ex_ns
        out["end_norm_y"] = ey_ns
        out["forward_progress_dx"] = fwd_dxs
        out["start_zone_x"] = sz_xs
        out["start_zone_y"] = sz_ys
        out["end_zone_x"] = ez_xs
        out["end_zone_y"] = ez_ys
        out["pitch_zone_start"] = pz_s
        out["pitch_zone_end"] = pz_e
        return out

    # ----------------------------------------------------------------
    def run_xp_modeling(self, events_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Pas Başarı Olasılığı (xP) denetimli modellerini kesin kronolojik
        train/val/test bölünmesiyle (sıfır veri sızıntısı) eğitir ve karşılaştırır.
        """
        total_ratio = float(config.ML_TRAIN_RATIO) + float(config.ML_VAL_RATIO) + float(config.ML_TEST_RATIO)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"ML_TRAIN_RATIO + ML_VAL_RATIO + ML_TEST_RATIO toplamı 1.0 olmalıdır (şu an: {total_ratio:.4f})."
            )

        proxy_events = self.extract_proxy_pass_events()
        fallback_reason: Optional[str] = None

        if events_df is not None:
            model_df = events_df.copy()
            data_source = "provided_events"
        else:
            valid_proxy = proxy_events.dropna(subset=["pass_outcome"]) if not proxy_events.empty else pd.DataFrame()
            if len(valid_proxy) < config.ML_MIN_SUPERVISED_EVENTS:
                fallback_reason = (
                    f"insufficient_real_events ({len(valid_proxy)} < {config.ML_MIN_SUPERVISED_EVENTS})"
                )
                model_df = self.generate_synthetic_match_passes(
                    n_events=config.ML_SYNTHETIC_MATCH_EVENTS,
                    seed=self.random_state,
                )
                data_source = "calibrated_synthetic_match"
            elif valid_proxy["pass_outcome"].nunique() < 2:
                fallback_reason = "single_class_real_events"
                model_df = self.generate_synthetic_match_passes(
                    n_events=config.ML_SYNTHETIC_MATCH_EVENTS,
                    seed=self.random_state,
                )
                data_source = "calibrated_synthetic_match"
            else:
                model_df = valid_proxy.copy()
                data_source = "real_tracking_events"

        def _prepare_and_split(df_in: pd.DataFrame) -> Dict[str, Any]:
            df_work = df_in.copy()
            total_in = int(len(df_work))
            if "pass_outcome" not in df_work.columns:
                return {
                    "total_input_events": total_in,
                    "missing_target_count": total_in,
                    "model_df": pd.DataFrame(),
                    "n": 0,
                    "num_cols": [],
                    "cat_cols": [],
                    "feature_cols": [],
                    "feature_missing_audit": {},
                    "train_df": pd.DataFrame(),
                    "val_df": pd.DataFrame(),
                    "test_df": pd.DataFrame(),
                    "chronological_ok": False,
                }

            miss_target = int(df_work["pass_outcome"].isna().sum())
            df_work = df_work.dropna(subset=["pass_outcome"]).copy()
            if df_work.empty:
                return {
                    "total_input_events": total_in,
                    "missing_target_count": miss_target,
                    "model_df": df_work,
                    "n": 0,
                    "num_cols": [],
                    "cat_cols": [],
                    "feature_cols": [],
                    "feature_missing_audit": {},
                    "train_df": pd.DataFrame(),
                    "val_df": pd.DataFrame(),
                    "test_df": pd.DataFrame(),
                    "chronological_ok": False,
                }

            df_work["pass_outcome"] = df_work["pass_outcome"].astype(int)

            # Eksik özellik sütunlarını güvenli biçimde tamamla (satır bazlı, sızıntısız)
            if "is_team_missing" not in df_work.columns and "from_team" in df_work.columns:
                df_work["is_team_missing"] = df_work["from_team"].isna().astype(int)
            if "is_opponent_dist_missing" not in df_work.columns and "nearest_opponent_dist" in df_work.columns:
                df_work["is_opponent_dist_missing"] = df_work["nearest_opponent_dist"].isna().astype(int)

            num_c = [c for c in self.XP_NUMERIC_FEATURES if c in df_work.columns]
            cat_c = [c for c in self.XP_CATEGORICAL_FEATURES if c in df_work.columns]
            feat_c = num_c + cat_c

            miss_audit = {
                col: {
                    "missing_count": int(df_work[col].isna().sum()),
                    "missing_pct": round((int(df_work[col].isna().sum()) / max(1, len(df_work))) * 100.0, 2),
                }
                for col in feat_c
            }

            # KESİN KRONOLOJİK SIRALAMA VE BÖLÜNME (ÖN İŞLEMEDEN ÖNCE!)
            sort_cols = [c for c in ("frame_idx", "timestamp_sec") if c in df_work.columns]
            if sort_cols:
                df_work = df_work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
            else:
                df_work = df_work.reset_index(drop=True)

            n_obs = len(df_work)
            if n_obs < 15:
                return {
                    "total_input_events": total_in,
                    "missing_target_count": miss_target,
                    "model_df": df_work,
                    "n": n_obs,
                    "num_cols": num_c,
                    "cat_cols": cat_c,
                    "feature_cols": feat_c,
                    "feature_missing_audit": miss_audit,
                    "train_df": df_work,
                    "val_df": pd.DataFrame(columns=df_work.columns),
                    "test_df": pd.DataFrame(columns=df_work.columns),
                    "chronological_ok": True,
                }

            train_end = max(5, int(round(n_obs * float(config.ML_TRAIN_RATIO))))
            test_count = max(2, int(round(n_obs * float(config.ML_TEST_RATIO))))
            val_end = max(train_end + 2, n_obs - test_count)
            val_end = min(val_end, n_obs - 2)

            tr_df = df_work.iloc[:train_end].copy()
            va_df = df_work.iloc[train_end:val_end].copy()
            te_df = df_work.iloc[val_end:].copy()

            chrono_ok = True
            if "frame_idx" in df_work.columns:
                chrono_ok = bool(
                    tr_df["frame_idx"].max() <= va_df["frame_idx"].min()
                    and va_df["frame_idx"].max() <= te_df["frame_idx"].min()
                )

            return {
                "total_input_events": total_in,
                "missing_target_count": miss_target,
                "model_df": df_work,
                "n": n_obs,
                "num_cols": num_c,
                "cat_cols": cat_c,
                "feature_cols": feat_c,
                "feature_missing_audit": miss_audit,
                "train_df": tr_df,
                "val_df": va_df,
                "test_df": te_df,
                "chronological_ok": chrono_ok,
            }

        prep_info = _prepare_and_split(model_df)
        n = prep_info["n"]
        train_df = prep_info["train_df"]
        y_train_check = train_df["pass_outcome"].to_numpy(dtype=int) if not train_df.empty else np.array([], dtype=int)

        # Kronolojik bölünme SONRASI eğitim setinde tek sınıf veya yetersiz olay varsa güvenli işlem
        if n < 15 or len(np.unique(y_train_check)) < 2:
            if events_df is None and data_source == "real_tracking_events":
                fallback_reason = (
                    "single_class_y_train_after_chronological_split"
                    if n >= 15
                    else f"insufficient_events_after_cleaning ({n} < 15)"
                )
                model_df = self.generate_synthetic_match_passes(
                    n_events=config.ML_SYNTHETIC_MATCH_EVENTS,
                    seed=self.random_state,
                )
                data_source = "calibrated_synthetic_match"
                prep_info = _prepare_and_split(model_df)
                n = prep_info["n"]
                train_df = prep_info["train_df"]
                y_train_check = train_df["pass_outcome"].to_numpy(dtype=int) if not train_df.empty else np.array([], dtype=int)
            else:
                self.xp_models_ = {}
                self.best_xp_model_name_ = None
                self.best_xp_pipeline_ = None
                model_df = prep_info["model_df"]
                val_df = prep_info["val_df"]
                test_df = prep_info["test_df"]
                is_single_class = (n > 0 and model_df["pass_outcome"].nunique() < 2) or (
                    len(y_train_check) > 0 and len(np.unique(y_train_check)) < 2
                )
                status = "skipped_single_class" if is_single_class else "skipped_insufficient_events"
                skip_reason = (
                    "single_class_y_train_after_chronological_split"
                    if is_single_class
                    else f"insufficient_events ({n} < 15)"
                )
                scored_events_df = model_df.copy()
                if not scored_events_df.empty:
                    scored_events_df["predicted_xp"] = np.nan
                return {
                    "status": status,
                    "fallback_reason": skip_reason,
                    "data_source": data_source,
                    "scoring_unit_mode": "skipped",
                    "bounds_source": "global_or_calibrated",
                    "total_input_events": prep_info["total_input_events"],
                    "dropped_missing_target": prep_info["missing_target_count"],
                    "modeled_events": n,
                    "positive_rate": round(float(model_df["pass_outcome"].mean()), 4) if n > 0 else 0.0,
                    "chronological_split_verified": prep_info["chronological_ok"],
                    "split_info": {
                        "train_count": int(len(train_df)),
                        "validation_count": int(len(val_df)),
                        "test_count": int(len(test_df)),
                        "train_frame_range": (
                            [int(train_df["frame_idx"].min()), int(train_df["frame_idx"].max())]
                            if not train_df.empty and "frame_idx" in train_df.columns
                            else None
                        ),
                        "val_frame_range": (
                            [int(val_df["frame_idx"].min()), int(val_df["frame_idx"].max())]
                            if not val_df.empty and "frame_idx" in val_df.columns
                            else None
                        ),
                        "test_frame_range": (
                            [int(test_df["frame_idx"].min()), int(test_df["frame_idx"].max())]
                            if not test_df.empty and "frame_idx" in test_df.columns
                            else None
                        ),
                    },
                    "feature_missing_audit": prep_info["feature_missing_audit"],
                    "models": [],
                    "selected_model": None,
                    "beats_naive_baseline": {
                        "roc_auc_gain": 0.0,
                        "pr_auc_gain": 0.0,
                        "brier_improvement": 0.0,
                        "passed": False,
                    },
                    "top_features": [],
                    "proxy_events_count": int(len(proxy_events)),
                    "_scored_events_df": scored_events_df,
                    "_proxy_scored_df": proxy_events.copy(),
                }

        model_df = prep_info["model_df"]
        train_df = prep_info["train_df"]
        val_df = prep_info["val_df"]
        test_df = prep_info["test_df"]
        num_cols = prep_info["num_cols"]
        cat_cols = prep_info["cat_cols"]
        feature_cols = prep_info["feature_cols"]
        total_input_events = prep_info["total_input_events"]
        missing_target_count = prep_info["missing_target_count"]
        feature_missing_audit = prep_info["feature_missing_audit"]
        chronological_ok = prep_info["chronological_ok"]

        # Kalibresiz gerçek takip verisiyle eğitimde test karesi koordinat sızıntısını önle:
        # normalizasyon sınırlarını yalnızca eğitim (train) split karelerinden hesapla
        bounds_source = "global_or_calibrated"
        train_bounds: Optional[Tuple[float, float, float, float]] = None
        if data_source == "real_tracking_events" and not self.is_calibrated:
            train_max_frame = int(train_df["frame_idx"].max()) if "frame_idx" in train_df.columns else None
            train_bounds = self._compute_coordinate_bounds(max_frame_idx=train_max_frame)
            model_df = self._recompute_event_normalization_with_bounds(model_df, train_bounds)
            train_len = len(train_df)
            val_len = len(val_df)
            train_df = model_df.iloc[:train_len].copy()
            val_df = model_df.iloc[train_len:train_len + val_len].copy()
            test_df = model_df.iloc[train_len + val_len:].copy()
            bounds_source = "train_split_only"

        X_train, y_train = train_df[feature_cols], train_df["pass_outcome"].to_numpy(dtype=int)
        X_val, y_val = val_df[feature_cols], val_df["pass_outcome"].to_numpy(dtype=int)
        X_test, y_test = test_df[feature_cols], test_df["pass_outcome"].to_numpy(dtype=int)

        candidate_estimators = [
            (
                "Naive Baseline (DummyClassifier)",
                DummyClassifier(strategy="most_frequent"),
                "Sıfır özellik öğrenimi; yalnızca çoğunluk sınıfını tahmin eder. En düşük gecikme, temel referans eşiği.",
            ),
            (
                "Logistic Regression",
                LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state),
                "Yüksek yorumlanabilirlik (doğrusal katsayılar), çok düşük çıkarım gecikmesi (<1 ms) ve doğal kalibre olasılıklar; "
                "ancak doğrusal olmayan baskı × mesafe etkileşimlerini sınırlı yakalar.",
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=6,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                "Doğrusal olmayan etkileşimleri güçlü yakalar ve aykırı değerlere dayanıklıdır; "
                "ancak model boyutu/gecikmesi daha yüksektir ve uç olasılıklarda kalibrasyon kayması gösterebilir.",
            ),
            (
                "Gradient Boosting",
                GradientBoostingClassifier(
                    n_estimators=90,
                    max_depth=2,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                    subsample=0.75,
                    random_state=self.random_state,
                ),
                "Ardışık hata düzeltme ile en yüksek ayırt edicilik (ROC-AUC/PR-AUC) ve dengeli kalibrasyon (düşük Brier); "
                "eğitim süresi daha uzundur ve hiperparametre/aşırı öğrenme (overfitting) takibi gerektirir.",
            ),
        ]

        self.xp_models_ = {}
        model_results: List[Dict[str, Any]] = []

        for model_name, estimator, tradeoff_note in candidate_estimators:
            preprocessor = self._build_xp_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_ms = (time.perf_counter() - t0) * 1000.0

            # Train, Validation ve Test olasılıkları
            train_prob = pipe.predict_proba(X_train)[:, 1]
            train_pred = pipe.predict(X_train)

            val_prob = pipe.predict_proba(X_val)[:, 1]
            val_pred = pipe.predict(X_val)

            t_inf0 = time.perf_counter()
            test_prob = pipe.predict_proba(X_test)[:, 1]
            test_pred = pipe.predict(X_test)
            inf_ms = (time.perf_counter() - t_inf0) * 1000.0

            train_metrics = self._evaluate_binary_classifier(y_train, train_prob, train_pred)
            val_metrics = self._evaluate_binary_classifier(y_val, val_prob, val_pred)
            test_metrics = self._evaluate_binary_classifier(y_test, test_prob, test_pred)

            overfit_auc_gap = round(train_metrics["roc_auc"] - test_metrics["roc_auc"], 4)
            self.xp_models_[model_name] = pipe

            model_results.append({
                "model_name": model_name,
                "is_baseline": model_name.startswith("Naive Baseline"),
                "train_time_ms": round(train_ms, 2),
                "test_inference_ms": round(inf_ms, 3),
                "latency_per_1000_events_ms": round((inf_ms / max(1, len(X_test))) * 1000.0, 2),
                "overfitting_auc_gap": overfit_auc_gap,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "operational_tradeoff_tr": tradeoff_note,
            })

        # En iyi modeli YALNIZCA doğrulama (validation) setine göre seç (test sızıntısı yok!)
        non_baseline = [m for m in model_results if not m["is_baseline"]]
        best_entry = max(
            non_baseline,
            key=lambda m: (m["validation"]["roc_auc"], -m["validation"]["brier_score"], m["validation"]["f1"]),
        )
        self.best_xp_model_name_ = best_entry["model_name"]
        self.best_xp_pipeline_ = self.xp_models_[self.best_xp_model_name_]

        # Test seti ve mevcutsa gerçek proxy olayları için xP tahminleri üret
        scored_events_df = model_df.copy()
        scored_events_df["split"] = (
            ["train"] * len(train_df) + ["validation"] * len(val_df) + ["test"] * len(test_df)
        )
        scored_events_df["predicted_xp"] = np.round(
            self.best_xp_pipeline_.predict_proba(model_df[feature_cols])[:, 1], 4
        )

        # Birim hizalama modu: kalibresiz piksel verisi metre eğitimli modele verilirken orantılı saha ölçeği uygula
        if not self.is_calibrated and data_source == "calibrated_synthetic_match":
            scoring_unit_mode = "normalized_pitch_projection_105x68"
        elif not self.is_calibrated and data_source == "real_tracking_events":
            scoring_unit_mode = "native_pixel_train_bounds"
        elif self.is_calibrated:
            scoring_unit_mode = "native_pitch_meters"
        else:
            scoring_unit_mode = "provided_events_native"

        # Eğer model sentetik veride eğitildiyse, tracking_data'dan çıkan gerçek proxy olayları da xP ile skorla
        proxy_scored_df = pd.DataFrame()
        if not proxy_events.empty:
            proxy_scored_df = proxy_events.copy()
            # Eksik kolon varsa NaN ile hizala
            for fc in feature_cols:
                if fc not in proxy_scored_df.columns:
                    proxy_scored_df[fc] = np.nan
            if data_source == "real_tracking_events" and not self.is_calibrated and train_bounds is not None:
                proxy_scored_df = self._recompute_event_normalization_with_bounds(proxy_scored_df, train_bounds)
            if not self.is_calibrated and data_source == "calibrated_synthetic_match":
                scoring_proxy_df = self._project_proxy_events_to_pitch_units(proxy_scored_df)
            else:
                scoring_proxy_df = proxy_scored_df
            proxy_scored_df["predicted_xp"] = np.round(
                self.best_xp_pipeline_.predict_proba(scoring_proxy_df[feature_cols])[:, 1], 4
            )

        # En iyi modelin özellik önemlerini / katsayılarını çıkar
        feature_importance: List[Dict[str, Any]] = []
        try:
            prep_fitted = self.best_xp_pipeline_.named_steps["preprocessor"]
            est_fitted = self.best_xp_pipeline_.named_steps["model"]
            feat_names = list(prep_fitted.get_feature_names_out())
            if hasattr(est_fitted, "feature_importances_"):
                vals = est_fitted.feature_importances_
            elif hasattr(est_fitted, "coef_"):
                vals = np.abs(est_fitted.coef_[0])
            else:
                vals = np.zeros(len(feat_names))
            pairs = sorted(zip(feat_names, vals), key=lambda kv: float(kv[1]), reverse=True)[:10]
            feature_importance = [{"feature": str(k), "importance": round(float(v), 4)} for k, v in pairs]
        except Exception:
            feature_importance = []

        baseline_test = next(m["test"] for m in model_results if m["is_baseline"])
        best_test = best_entry["test"]

        return {
            "status": "ok",
            "fallback_reason": fallback_reason,
            "data_source": data_source,
            "scoring_unit_mode": scoring_unit_mode,
            "bounds_source": bounds_source,
            "total_input_events": total_input_events,
            "dropped_missing_target": missing_target_count,
            "modeled_events": n,
            "positive_rate": round(float(model_df["pass_outcome"].mean()), 4),
            "chronological_split_verified": chronological_ok,
            "split_info": {
                "train_count": int(len(train_df)),
                "validation_count": int(len(val_df)),
                "test_count": int(len(test_df)),
                "train_frame_range": [int(train_df["frame_idx"].min()), int(train_df["frame_idx"].max())] if "frame_idx" in train_df.columns else None,
                "val_frame_range": [int(val_df["frame_idx"].min()), int(val_df["frame_idx"].max())] if "frame_idx" in val_df.columns else None,
                "test_frame_range": [int(test_df["frame_idx"].min()), int(test_df["frame_idx"].max())] if "frame_idx" in test_df.columns else None,
            },
            "feature_missing_audit": feature_missing_audit,
            "models": model_results,
            "selected_model": self.best_xp_model_name_,
            "beats_naive_baseline": {
                "roc_auc_gain": round(best_test["roc_auc"] - baseline_test["roc_auc"], 4),
                "pr_auc_gain": round(best_test["pr_auc"] - baseline_test["pr_auc"], 4),
                "brier_improvement": round(baseline_test["brier_score"] - best_test["brier_score"], 4),
                "passed": bool(
                    best_test["roc_auc"] > baseline_test["roc_auc"]
                    and best_test["brier_score"] < baseline_test["brier_score"]
                ),
            },
            "top_features": feature_importance,
            "proxy_events_count": int(len(proxy_scored_df)),
            "_scored_events_df": scored_events_df,
            "_proxy_scored_df": proxy_scored_df,
        }

    # ----------------------------------------------------------------
    def _build_xt_grid(self, events_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        12x8 Markov Değer İterasyonu (Expected Threat - xT) ızgarası hesaplar.
        Hücum yönü soldan sağa (col=0 -> col=cols-1) normalize [0..1]x[0..1] uzaydadır.
        """
        cols = int(config.ML_XT_GRID_COLS)
        rows = int(config.ML_XT_GRID_ROWS)

        shot_prob = np.zeros((rows, cols), dtype=float)
        goal_prob = np.zeros((rows, cols), dtype=float)

        for r in range(rows):
            y_norm = (r + 0.5) / rows
            y_center_dist = abs(y_norm - 0.5)
            for c in range(cols):
                x_norm = (c + 0.5) / cols
                goal_dist = math.hypot(1.0 - x_norm, (y_norm - 0.5) * 0.65)
                # Şut olasılığı ceza sahası yakınında yükselir
                sp = 0.005 + 0.42 * math.exp(-((goal_dist / 0.22) ** 2))
                # Gol olasılığı kale ağzına yaklaştıkça ve merkezde artar
                gp = 0.01 + 0.34 * math.exp(-((goal_dist / 0.16) ** 2)) * max(0.25, 1.0 - y_center_dist * 1.4)
                shot_prob[r, c] = float(np.clip(sp, 0.002, 0.55))
                goal_prob[r, c] = float(np.clip(gp, 0.005, 0.45))

        move_prob = 1.0 - shot_prob

        # Uzamsal geçiş matrisi (ileri doğru yumuşak Gauss prior + yeterli veri varsa ampirik karışım)
        n_cells = rows * cols
        T_prior = np.zeros((n_cells, n_cells), dtype=float)
        for r1 in range(rows):
            for c1 in range(cols):
                idx1 = r1 * cols + c1
                weights = np.zeros(n_cells, dtype=float)
                for r2 in range(rows):
                    for c2 in range(cols):
                        idx2 = r2 * cols + c2
                        dc = c2 - c1
                        dr = r2 - r1
                        # İleri ve yakın komşulara doğal pas/sürüş dağılımı
                        forward_bias = 1.35 if dc > 0 else (0.75 if dc == 0 else 0.45)
                        w = forward_bias * math.exp(-((dc - 1.0) ** 2) / 6.5 - (dr ** 2) / 4.0)
                        weights[idx2] = w
                T_prior[idx1] = weights / max(1e-9, float(weights.sum()))

        T = T_prior.copy()
        if (
            events_df is not None
            and len(events_df) >= 50
            and {"start_zone_x", "start_zone_y", "end_zone_x", "end_zone_y"}.issubset(events_df.columns)
        ):
            counts = np.zeros((n_cells, n_cells), dtype=float)
            for _, row in events_df.iterrows():
                try:
                    c1 = int(np.clip(int(row["start_zone_x"]), 0, cols - 1))
                    r1 = int(np.clip(int(row["start_zone_y"]), 0, rows - 1))
                    c2 = int(np.clip(int(row["end_zone_x"]), 0, cols - 1))
                    r2 = int(np.clip(int(row["end_zone_y"]), 0, rows - 1))
                    counts[r1 * cols + c1, r2 * cols + c2] += 1.0
                except (TypeError, ValueError):
                    continue
            # Dirichlet / Laplace karışımı (prior gücü = 15.0)
            row_sums = counts.sum(axis=1, keepdims=True)
            T = (counts + 15.0 * T_prior) / (row_sums + 15.0)

        # Markov Değer İterasyonu (25 adım)
        xt = np.zeros(n_cells, dtype=float)
        immediate = (shot_prob * goal_prob).reshape(-1)
        move_flat = move_prob.reshape(-1)
        for _ in range(25):
            xt = immediate + move_flat * (T @ xt)

        return xt.reshape((rows, cols))

    # ----------------------------------------------------------------
    def compute_xt_analysis(
        self,
        events_df: Optional[pd.DataFrame] = None,
        xp_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Mekansal xT (Expected Threat) ızgarasını hesaplar ve hem gerçek takip
        vekil olaylarını hem de kalibre olayları xT_start, xT_end, delta_xT ve
        expected_xT (= xP * delta_xT) ile skorlar.
        """
        cols = int(config.ML_XT_GRID_COLS)
        rows = int(config.ML_XT_GRID_ROWS)

        if events_df is not None:
            eval_df = events_df.copy()
            eval_source = "provided_events"
        elif xp_results is not None:
            proxy_df = xp_results.get("_proxy_scored_df")
            if isinstance(proxy_df, pd.DataFrame) and not proxy_df.empty:
                eval_df = proxy_df.copy()
                eval_source = "real_tracking_proxy_events"
            else:
                eval_df = xp_results.get("_scored_events_df", pd.DataFrame()).copy()
                eval_source = xp_results.get("data_source", "synthetic_events")
        else:
            proxy_df = self.extract_proxy_pass_events()
            if not proxy_df.empty:
                eval_df = proxy_df
                eval_source = "real_tracking_proxy_events"
            else:
                eval_df = self.generate_synthetic_match_passes(n_events=300, seed=self.random_state)
                eval_source = "calibrated_synthetic_match"

        # Grid'i eğitilmiş/zengin olay setiyle kur (varsa model olayları)
        grid_ref_df = (
            xp_results.get("_scored_events_df")
            if xp_results is not None and isinstance(xp_results.get("_scored_events_df"), pd.DataFrame)
            else eval_df
        )
        xt_grid = self._build_xt_grid(grid_ref_df)

        def _safe_norm(val: Any) -> float:
            if val is None:
                return 0.5
            try:
                if pd.isna(val):
                    return 0.5
            except (TypeError, ValueError):
                return 0.5
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return 0.5
            return float(np.clip(fval, 0.0, 0.9999))

        def score_df_with_xt(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty:
                return df_in.copy()
            df_out = df_in.copy()
            xt_starts: List[float] = []
            xt_ends: List[float] = []
            deltas: List[float] = []
            exp_xts: List[float] = []

            for _, row in df_out.iterrows():
                sx_n = _safe_norm(row.get("start_norm_x", 0.5))
                sy_n = _safe_norm(row.get("start_norm_y", 0.5))
                ex_n = _safe_norm(row.get("end_norm_x", 0.5))
                ey_n = _safe_norm(row.get("end_norm_y", 0.5))

                # Team B sağdan sola hücum ediyorsa x eksenini kendi hücum yönüne çevir
                team_val = _clean_str(row.get("from_team"))
                if team_val == "Team B":
                    sx_eff = 1.0 - sx_n
                    ex_eff = 1.0 - ex_n
                else:
                    sx_eff = sx_n
                    ex_eff = ex_n

                c1 = min(cols - 1, max(0, int(sx_eff * cols)))
                r1 = min(rows - 1, max(0, int(sy_n * rows)))
                c2 = min(cols - 1, max(0, int(ex_eff * cols)))
                r2 = min(rows - 1, max(0, int(ey_n * rows)))

                xt_s = float(xt_grid[r1, c1])
                xt_e = float(xt_grid[r2, c2])
                d_xt = xt_e - xt_s
                xp_val = row.get("predicted_xp")
                if xp_val is None or pd.isna(xp_val) or (isinstance(xp_val, (float, np.floating)) and (math.isnan(float(xp_val)) or math.isinf(float(xp_val)))):
                    xp_val = 0.75
                exp_xt = float(xp_val) * d_xt

                xt_starts.append(round(xt_s, 5))
                xt_ends.append(round(xt_e, 5))
                deltas.append(round(d_xt, 5))
                exp_xts.append(round(exp_xt, 5))

            df_out["xT_start"] = xt_starts
            df_out["xT_end"] = xt_ends
            df_out["delta_xT"] = deltas
            df_out["expected_xT"] = exp_xts
            return df_out

        scored_eval = score_df_with_xt(eval_df)
        scored_model = (
            score_df_with_xt(xp_results["_scored_events_df"])
            if xp_results is not None and isinstance(xp_results.get("_scored_events_df"), pd.DataFrame)
            else scored_eval
        )

        team_xt_summary: Dict[str, Dict[str, Any]] = {}
        if not scored_eval.empty and "from_team" in scored_eval.columns:
            for tname in ("Team A", "Team B"):
                t_df = scored_eval[scored_eval["from_team"] == tname]
                if not t_df.empty:
                    team_xt_summary[tname] = {
                        "events": int(len(t_df)),
                        "total_delta_xT": round(float(t_df["delta_xT"].sum()), 4),
                        "mean_delta_xT": round(float(t_df["delta_xT"].mean()), 4),
                        "total_expected_xT": round(float(t_df["expected_xT"].sum()), 4),
                    }

        top_events: List[Dict[str, Any]] = []
        if not scored_eval.empty:
            top_df = scored_eval.sort_values("expected_xT", ascending=False).head(5)
            for _, row in top_df.iterrows():
                top_events.append({
                    "frame_idx": int(row["frame_idx"]) if "frame_idx" in row and not pd.isna(row["frame_idx"]) else None,
                    "from_track_id": _clean_str(row.get("from_track_id")),
                    "to_track_id": _clean_str(row.get("to_track_id")),
                    "from_team": _clean_str(row.get("from_team")) or "Unknown",
                    "predicted_xp": float(row["predicted_xp"]) if "predicted_xp" in row and not pd.isna(row["predicted_xp"]) else None,
                    "xT_start": float(row["xT_start"]),
                    "xT_end": float(row["xT_end"]),
                    "delta_xT": float(row["delta_xT"]),
                    "expected_xT": float(row["expected_xT"]),
                })

        return {
            "coord_mode": self.coord_mode,
            "eval_source": eval_source,
            "grid_shape": [rows, cols],
            "min_xt": round(float(xt_grid.min()), 5),
            "max_xt": round(float(xt_grid.max()), 5),
            "xt_grid": [[round(float(v), 5) for v in row] for row in xt_grid],
            "evaluated_events_count": int(len(scored_eval)),
            "mean_delta_xT": round(float(scored_eval["delta_xT"].mean()), 5) if not scored_eval.empty else 0.0,
            "mean_expected_xT": round(float(scored_eval["expected_xT"].mean()), 5) if not scored_eval.empty else 0.0,
            "team_xt_summary": team_xt_summary,
            "top_threat_events": top_events,
            "_scored_xt_eval_df": scored_eval,
            "_scored_xt_model_df": scored_model,
        }

    # ----------------------------------------------------------------
    def cluster_player_roles(
        self,
        min_track_frames: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Oyuncuları ortalama pozisyon, saha kapsama alanı (konveks örtü),
        hız/sprint profili ve top kontrol metrikleriyle K-Means kullanarak kümeleyip
        Silhouette skoruyla optimal k seçer ve 2B PCA projeksiyonu üretir.
        """
        min_frames = int(min_track_frames if min_track_frames is not None else config.ML_MIN_TRACK_FRAMES)
        profile_cols = [
            "track_id", "team", "observed_frames", "mean_x", "mean_y",
            "norm_mean_x", "norm_mean_y", "std_x", "std_y", "pitch_coverage_area",
            "mean_speed", "p95_speed", "high_speed_ratio",
            "ball_control_pct", "mean_dist_to_ball",
            "cluster_id", "tactical_role_tr", "pca_1", "pca_2",
        ]
        if self.df.empty or "class_name" not in self.df.columns or "track_id" not in self.df.columns:
            return pd.DataFrame(columns=profile_cols), {
                "coord_mode": self.coord_mode,
                "min_track_frames": min_frames,
                "eligible_players": 0,
                "excluded_tracks_count": 0,
                "excluded_tracks": [],
                "candidate_k_scores": {},
                "optimal_k": None,
                "best_silhouette": None,
                "pca_explained_variance_ratio": [],
                "cluster_summaries": [],
            }

        x_col, y_col = self._get_xy_columns()
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return pd.DataFrame(columns=profile_cols), {
                "coord_mode": self.coord_mode,
                "min_track_frames": min_frames,
                "eligible_players": 0,
                "excluded_tracks_count": 0,
                "excluded_tracks": [],
                "candidate_k_scores": {},
                "optimal_k": None,
                "best_silhouette": None,
                "pca_explained_variance_ratio": [],
                "cluster_summaries": [],
            }
        valid_bbox = self._get_valid_bbox_mask(self.df)
        human_df = self.df[self.df["class_name"].isin(config.HUMAN_CLASS_NAMES) & valid_bbox].copy()
        if "is_predicted" in human_df.columns:
            human_df = human_df[human_df["is_predicted"] != True]  # noqa: E712
        if self.is_calibrated and "on_pitch" in human_df.columns:
            human_df = human_df[human_df["on_pitch"] != False]  # noqa: E712
        human_df = human_df.dropna(subset=[x_col, y_col])

        if not self.is_calibrated and not human_df.empty:
            min_x, max_x, min_y, max_y = self.bounds
            human_df = human_df[
                (human_df[x_col] >= min_x - 30.0)
                & (human_df[x_col] <= max_x + 30.0)
                & (human_df[y_col] >= min_y - 30.0)
                & (human_df[y_col] <= max_y + 30.0)
            ]

        # Hız ve top mesafesi yardımcı tabloları
        speed_df, _ = self._compute_player_step_speeds(exclude_referees=True)
        hsr_thresh = (
            float(config.SPEED_ZONE_RUNNING_MAX_KMH)
            if self.is_calibrated
            else (
                float(speed_df["speed"].quantile(0.75))
                if not speed_df.empty and len(speed_df) >= 10
                else 120.0
            )
        )

        # Top-oyuncu mesafesi ve kontrol oranı (kare bazında)
        ball_df = self.df[self.df["class_name"] == config.BALL_CLASS_NAME].dropna(subset=[x_col, y_col]).copy()
        if "is_predicted" in ball_df.columns:
            ball_df["_ps"] = ball_df["is_predicted"].fillna(False).astype(bool)
            ball_df = ball_df.sort_values(["frame_idx", "_ps"]).drop_duplicates(subset=["frame_idx"])
        ball_coords = {
            int(r["frame_idx"]): (float(r[x_col]), float(r[y_col]))
            for _, r in ball_df.iterrows()
        }
        max_ctrl_dist = (
            float(config.POSSESSION_MAX_PLAYER_BALL_DIST_M)
            if self.is_calibrated
            else float(config.POSSESSION_MAX_PLAYER_BALL_DIST_PX)
        )

        # Her karede topa en yakın oyuncuyu bul
        ball_ctrl_counts: Counter = Counter()
        ball_dists_by_track: Dict[str, List[float]] = {}
        team_not_ref_h = (human_df["team"] != "Referee") if "team" in human_df.columns else True
        players_non_ref = human_df[
            (human_df["class_name"] != "referee") & team_not_ref_h
        ]
        for fid, fgrp in players_non_ref.groupby("frame_idx"):
            bxy = ball_coords.get(int(fid))
            if bxy is None or fgrp.empty:
                continue
            bx, by = bxy
            pxs = fgrp[x_col].to_numpy(dtype=float)
            pys = fgrp[y_col].to_numpy(dtype=float)
            dists = np.hypot(pxs - bx, pys - by)
            for idx_row, tid_val in enumerate(fgrp["track_id"].astype(str).tolist()):
                ball_dists_by_track.setdefault(tid_val, []).append(float(dists[idx_row]))
            min_idx = int(np.argmin(dists))
            if float(dists[min_idx]) <= max_ctrl_dist:
                ctrl_tid = str(fgrp.iloc[min_idx]["track_id"])
                ball_ctrl_counts[ctrl_tid] += 1

        total_ball_frames = max(1, len(ball_coords))
        excluded_tracks: List[Dict[str, Any]] = []
        player_rows: List[Dict[str, Any]] = []

        for tid_raw, grp in human_df.groupby("track_id"):
            tid = str(tid_raw)
            n_obs = int(len(grp))
            team_vals = [_clean_str(v) for v in grp["team"].tolist() if _clean_str(v) is not None] if "team" in grp.columns else []
            resolved_team = Counter(team_vals).most_common(1)[0][0] if team_vals else "Unknown"
            is_ref = bool((grp["class_name"] == "referee").any() or resolved_team == "Referee")

            if is_ref:
                excluded_tracks.append({
                    "track_id": tid,
                    "team": resolved_team,
                    "observed_frames": n_obs,
                    "reason": "Hakem izi (takım rolü kümelemesinden hariç tutuldu)",
                })
                continue

            if n_obs < min_frames:
                excluded_tracks.append({
                    "track_id": tid,
                    "team": resolved_team,
                    "observed_frames": n_obs,
                    "reason": f"Yetersiz gözlemlenen kare ({n_obs} < {min_frames})",
                })
                continue

            xs = grp[x_col].to_numpy(dtype=float)
            ys = grp[y_col].to_numpy(dtype=float)
            norm_pts = np.array([self._normalize_xy(x, y) for x, y in zip(xs, ys)], dtype=float)
            nxs = norm_pts[:, 0]
            nys = norm_pts[:, 1]

            # Saha kapsama alanı: Konveks Örtü (ConvexHull), kolineer/az nokta durumunda bbox alanı
            cov_area = 0.0
            if len(norm_pts) >= 3:
                try:
                    hull = ConvexHull(norm_pts)
                    cov_area = float(hull.volume)  # 2B'de volume == alan
                except Exception:
                    cov_area = float((nxs.max() - nxs.min()) * (nys.max() - nys.min()))
            else:
                cov_area = float((nxs.max() - nxs.min()) * (nys.max() - nys.min()))

            p_spds = speed_df[speed_df["track_id"] == tid]["speed"].to_numpy(dtype=float)
            if p_spds.size > 0:
                mean_spd = float(np.mean(p_spds))
                p95_spd = float(np.percentile(p_spds, 95))
                hsr_ratio = float(np.mean(p_spds >= hsr_thresh))
            else:
                mean_spd, p95_spd, hsr_ratio = 0.0, 0.0, 0.0

            b_dists = ball_dists_by_track.get(tid, [])
            mean_b_dist = float(np.mean(b_dists)) if b_dists else np.nan
            ctrl_pct = (ball_ctrl_counts.get(tid, 0) / total_ball_frames) * 100.0

            player_rows.append({
                "track_id": tid,
                "team": resolved_team,
                "observed_frames": n_obs,
                "mean_x": round(float(np.mean(xs)), 2),
                "mean_y": round(float(np.mean(ys)), 2),
                "norm_mean_x": round(float(np.mean(nxs)), 4),
                "norm_mean_y": round(float(np.mean(nys)), 4),
                "std_x": round(float(np.std(nxs)), 4),
                "std_y": round(float(np.std(nys)), 4),
                "pitch_coverage_area": round(cov_area, 5),
                "mean_speed": round(mean_spd, 2),
                "p95_speed": round(p95_spd, 2),
                "high_speed_ratio": round(hsr_ratio, 4),
                "ball_control_pct": round(ctrl_pct, 2),
                "mean_dist_to_ball": round(mean_b_dist, 2) if not math.isnan(mean_b_dist) else np.nan,
            })

        if not player_rows:
            return pd.DataFrame(columns=profile_cols), {
                "coord_mode": self.coord_mode,
                "min_track_frames": min_frames,
                "eligible_players": 0,
                "excluded_tracks_count": int(len(excluded_tracks)),
                "excluded_tracks": excluded_tracks,
                "candidate_k_scores": {},
                "optimal_k": None,
                "best_silhouette": None,
                "pca_explained_variance_ratio": [],
                "cluster_summaries": [],
            }

        profiles_df = pd.DataFrame(player_rows)
        cluster_feat_cols = [
            "norm_mean_x", "norm_mean_y", "std_x", "std_y",
            "pitch_coverage_area", "mean_speed", "p95_speed",
            "high_speed_ratio", "ball_control_pct", "mean_dist_to_ball",
        ]

        X_raw = profiles_df[cluster_feat_cols].to_numpy(dtype=float)
        # Eğer bir sütunun tamamı NaN ise (örn. top hiç yoksa) 0 ile güvenli doldur
        col_all_nan = np.isnan(X_raw).all(axis=0)
        if col_all_nan.any():
            X_raw[:, col_all_nan] = 0.0
        X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)
        X_scaled = StandardScaler().fit_transform(X_imp)

        n_players = len(profiles_df)
        candidate_k_scores: Dict[int, float] = {}
        optimal_k = 1
        best_sil: Optional[float] = None

        if n_players >= 3:
            k_min = max(2, int(config.ML_CLUSTER_K_MIN))
            k_max = min(int(config.ML_CLUSTER_K_MAX), n_players - 1)
            best_labels = None

            for k in range(k_min, k_max + 1):
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                labels = km.fit_predict(X_scaled)
                if len(set(labels)) < 2:
                    continue
                sil = float(silhouette_score(X_scaled, labels))
                candidate_k_scores[k] = round(sil, 4)
                if best_sil is None or sil > best_sil:
                    best_sil = sil
                    optimal_k = k
                    best_labels = labels

            if best_labels is None:
                best_labels = np.zeros(n_players, dtype=int)
                optimal_k = 1
        else:
            best_labels = np.zeros(n_players, dtype=int)

        profiles_df["cluster_id"] = best_labels.astype(int)

        # 2B PCA projeksiyonu (sıfır varyanslı özdeş oyuncu durumunu uyarısız / NaN'sız işle)
        total_var = float(np.var(X_scaled))
        if n_players >= 2 and X_scaled.shape[1] >= 2 and total_var > 1e-12:
            pca = PCA(n_components=2, random_state=self.random_state)
            coords_2d = np.nan_to_num(pca.fit_transform(X_scaled), nan=0.0, posinf=0.0, neginf=0.0)
            raw_ev = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0, posinf=0.0, neginf=0.0)
            exp_var = [round(float(v), 4) for v in raw_ev]
        else:
            coords_2d = np.zeros((n_players, 2), dtype=float)
            exp_var = [0.0, 0.0]

        profiles_df["pca_1"] = np.round(coords_2d[:, 0], 4)
        profiles_df["pca_2"] = np.round(coords_2d[:, 1], 4)

        # Küme merkezlerinin z-skorlarına göre Türkçe taktik rol açıklaması çıkar
        global_means = profiles_df[cluster_feat_cols].mean(numeric_only=True)
        global_stds = profiles_df[cluster_feat_cols].std(ddof=0, numeric_only=True).replace(0.0, 1.0)

        cluster_summaries: List[Dict[str, Any]] = []
        role_map: Dict[int, str] = {}

        for cid, cgrp in profiles_df.groupby("cluster_id"):
            cid_int = int(cid)
            z_speed = float((cgrp["mean_speed"].mean() - global_means["mean_speed"]) / max(1e-6, global_stds["mean_speed"]))
            z_ctrl = float((cgrp["ball_control_pct"].mean() - global_means["ball_control_pct"]) / max(1e-6, global_stds["ball_control_pct"]))
            z_cov = float((cgrp["pitch_coverage_area"].mean() - global_means["pitch_coverage_area"]) / max(1e-6, global_stds["pitch_coverage_area"]))
            z_x = float((cgrp["norm_mean_x"].mean() - global_means["norm_mean_x"]) / max(1e-6, global_stds["norm_mean_x"]))

            if z_ctrl >= 0.45:
                role_tr = "Top Dağıtıcı / Merkez Oyun Kurucu"
            elif z_speed >= 0.45 and z_cov >= 0.25:
                role_tr = "Yüksek Tempolu Kanat / Dinamik Orta Saha"
            elif z_x >= 0.55 and z_speed >= 0.0:
                role_tr = "İleri Uç / Hücum Baskı Rolü"
            elif z_cov <= -0.35 and z_speed <= -0.25:
                role_tr = "Sabit Bölge / Derin Savunma Bloğu"
            else:
                role_tr = "Dengeli Kutu-Kutu / Destek Rolü"

            role_map[cid_int] = role_tr
            cluster_summaries.append({
                "cluster_id": cid_int,
                "tactical_role_tr": role_tr,
                "player_count": int(len(cgrp)),
                "track_ids": cgrp["track_id"].tolist(),
                "mean_speed": round(float(cgrp["mean_speed"].mean()), 2),
                "mean_coverage_area": round(float(cgrp["pitch_coverage_area"].mean()), 4),
                "mean_ball_control_pct": round(float(cgrp["ball_control_pct"].mean()), 2),
            })

        profiles_df["tactical_role_tr"] = profiles_df["cluster_id"].map(role_map)
        profiles_df = profiles_df.sort_values(["cluster_id", "observed_frames"], ascending=[True, False]).reset_index(drop=True)

        summary = {
            "coord_mode": self.coord_mode,
            "min_track_frames": min_frames,
            "eligible_players": int(n_players),
            "excluded_tracks_count": int(len(excluded_tracks)),
            "excluded_tracks": excluded_tracks,
            "candidate_k_scores": {str(k): v for k, v in candidate_k_scores.items()},
            "optimal_k": int(optimal_k),
            "best_silhouette": round(float(best_sil), 4) if best_sil is not None else None,
            "pca_explained_variance_ratio": exp_var,
            "cluster_summaries": cluster_summaries,
        }
        return profiles_df, summary

    # ----------------------------------------------------------------
    def detect_tempo_and_anomalies(
        self,
        window_sec: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Zaman pencereli hareket yoğunluğu ve tempo düşüşlerini (kapsam farkındalıklı),
        oyuncu bazlı yorgunluk eğilimlerini ve IsolationForest + MAD z-skoru tabanlı
        konum/hız anomalilerini tespit eder.
        """
        win_sec = float(window_sec if window_sec is not None else config.ML_TEMPO_WINDOW_SEC)
        window_cols = [
            "window_idx", "window_start_sec", "window_end_sec", "team",
            "active_players", "sample_count", "mean_speed", "p90_speed",
            "high_speed_ratio", "coverage_ok", "low_coverage_flag",
            "baseline_speed", "speed_drop_pct", "tempo_z_score", "tempo_drop_flag",
        ]
        anom_cols = [
            "anomaly_type", "severity", "frame_idx", "timestamp_sec",
            "track_id", "team", "metric_value", "threshold", "description_tr",
        ]

        speed_df, spike_anomalies = self._compute_player_step_speeds(exclude_referees=True)
        anomalies_list: List[Dict[str, Any]] = list(spike_anomalies)

        if speed_df.empty:
            early_anom_df = pd.DataFrame(anomalies_list, columns=anom_cols)
            early_counts = (
                early_anom_df["anomaly_type"].value_counts().to_dict()
                if not early_anom_df.empty
                else {}
            )
            return pd.DataFrame(columns=window_cols), early_anom_df, {
                "coord_mode": self.coord_mode,
                "window_sec": round(win_sec, 2),
                "total_windows": 0,
                "covered_windows": 0,
                "low_coverage_windows": 0,
                "tempo_drop_count": 0,
                "fatigue_players_count": 0,
                "total_anomalies": int(len(early_anom_df)),
                "anomaly_counts_by_type": early_counts,
                "player_fatigue_trends": [],
            }

        if "timestamp_sec" in self.df.columns and self.df["timestamp_sec"].notna().any():
            t_min = float(self.df["timestamp_sec"].min())
            t_max = float(self.df["timestamp_sec"].max())
        else:
            t_min = float(speed_df["timestamp_sec"].min())
            t_max = float(speed_df["timestamp_sec"].max())
        total_dur = max(0.1, t_max - t_min)
        if window_sec is None and total_dur > 600.0:
            win_sec = max(win_sec, round(total_dur / 30.0, 1))

        speed_df = speed_df.copy()
        speed_df["window_idx"] = np.floor((speed_df["timestamp_sec"] - t_min) / max(0.5, win_sec)).astype(int)
        max_w_idx = max(
            int(speed_df["window_idx"].max()),
            int(math.floor(max(0.0, t_max - t_min - 1e-6) / max(0.5, win_sec))),
        )

        hsr_thresh = (
            float(config.SPEED_ZONE_RUNNING_MAX_KMH)
            if self.is_calibrated
            else float(speed_df["speed"].quantile(0.75))
        )
        min_samples = int(config.ML_TEMPO_MIN_SAMPLES_PER_WINDOW)

        teams_to_eval = ["ALL"] + [t for t in ("Team A", "Team B") if (speed_df["team"] == t).any()]
        window_rows: List[Dict[str, Any]] = []

        for team_label in teams_to_eval:
            t_sub = speed_df if team_label == "ALL" else speed_df[speed_df["team"] == team_label]
            # Önceki kapsanan pencerelerin nedensel ortalaması için tarihçe
            prior_covered_means: List[float] = []

            for w_idx in range(max_w_idx + 1):
                w_start = t_min + w_idx * win_sec
                w_end = w_start + win_sec
                w_df = t_sub[t_sub["window_idx"] == w_idx]
                s_cnt = int(len(w_df))
                act_p = int(w_df["track_id"].nunique()) if not w_df.empty else 0
                cov_ok = bool(s_cnt >= min_samples and act_p >= 1)
                low_cov = not cov_ok

                if not w_df.empty:
                    m_spd = float(w_df["speed"].mean())
                    p90_spd = float(np.percentile(w_df["speed"], 90))
                    hsr_r = float(np.mean(w_df["speed"] >= hsr_thresh))
                else:
                    m_spd, p90_spd, hsr_r = 0.0, 0.0, 0.0

                baseline_spd = float(np.mean(prior_covered_means)) if prior_covered_means else np.nan
                drop_pct = (
                    (baseline_spd - m_spd) / max(1e-6, baseline_spd)
                    if cov_ok and not math.isnan(baseline_spd) and baseline_spd > 0
                    else 0.0
                )
                if cov_ok and len(prior_covered_means) >= 2 and baseline_spd > 0:
                    p_std = float(np.std(prior_covered_means, ddof=0))
                    p_std_floored = max(p_std, 0.08 * baseline_spd, 1e-6)
                    z_val = (m_spd - baseline_spd) / p_std_floored
                elif cov_ok and len(prior_covered_means) == 1 and baseline_spd > 0:
                    p_std_floored = max(0.08 * baseline_spd, 1e-6)
                    z_val = (m_spd - baseline_spd) / p_std_floored
                else:
                    z_val = 0.0

                tempo_drop = bool(
                    cov_ok
                    and len(prior_covered_means) >= 1
                    and (
                        drop_pct >= config.ML_TEMPO_DROP_PCT_THRESHOLD
                        or (drop_pct >= 0.12 and z_val <= config.ML_TEMPO_DROP_Z_THRESHOLD)
                    )
                )

                if cov_ok:
                    prior_covered_means.append(m_spd)

                if tempo_drop:
                    anomalies_list.append({
                        "anomaly_type": "tempo_drop",
                        "severity": "high" if drop_pct >= 0.35 else "medium",
                        "frame_idx": int(w_df["frame_idx"].min()) if not w_df.empty else None,
                        "timestamp_sec": round(w_start, 2),
                        "track_id": "TEAM",
                        "team": team_label,
                        "metric_value": round(m_spd, 2),
                        "threshold": round(baseline_spd * (1.0 - config.ML_TEMPO_DROP_PCT_THRESHOLD), 2),
                        "description_tr": (
                            f"{team_label} için [{w_start:.1f}-{w_end:.1f}s] penceresinde ani tempo düşüşü: "
                            f"hız {baseline_spd:.1f} -> {m_spd:.1f} (%{drop_pct * 100:.1f} düşüş, z={z_val:.2f})."
                        ),
                    })
                elif low_cov and team_label in ("Team A", "Team B"):
                    anomalies_list.append({
                        "anomaly_type": "low_coverage",
                        "severity": "info",
                        "frame_idx": int(w_df["frame_idx"].min()) if not w_df.empty else None,
                        "timestamp_sec": round(w_start, 2),
                        "track_id": "TEAM",
                        "team": team_label,
                        "metric_value": s_cnt,
                        "threshold": min_samples,
                        "description_tr": (
                            f"{team_label} için [{w_start:.1f}-{w_end:.1f}s] penceresinde düşük kapsam/kamera çıkışı "
                            f"({s_cnt} örnek < {min_samples}); yorgunluk olarak etiketlenmedi."
                        ),
                    })

                window_rows.append({
                    "window_idx": w_idx,
                    "window_start_sec": round(w_start, 2),
                    "window_end_sec": round(w_end, 2),
                    "team": team_label,
                    "active_players": act_p,
                    "sample_count": s_cnt,
                    "mean_speed": round(m_spd, 2),
                    "p90_speed": round(p90_spd, 2),
                    "high_speed_ratio": round(hsr_r, 4),
                    "coverage_ok": cov_ok,
                    "low_coverage_flag": low_cov,
                    "baseline_speed": round(baseline_spd, 2) if not math.isnan(baseline_spd) else np.nan,
                    "speed_drop_pct": round(drop_pct * 100.0, 2),
                    "tempo_z_score": round(z_val, 3),
                    "tempo_drop_flag": tempo_drop,
                })

        windows_df = pd.DataFrame(window_rows, columns=window_cols)

        # Oyuncu bazlı yorgunluk (hız düşüş eğimi) analizi
        fatigue_trends: List[Dict[str, Any]] = []
        for tid, pgrp in speed_df.groupby("track_id"):
            if len(pgrp) < 12:
                continue
            pgrp = pgrp.sort_values("timestamp_sec")
            ts_arr = pgrp["timestamp_sec"].to_numpy(dtype=float)
            spd_arr = pgrp["speed"].to_numpy(dtype=float)
            if float(ts_arr.max() - ts_arr.min()) < max(2.0, win_sec):
                continue

            reg = scipy_stats.linregress(ts_arr, spd_arr)
            half_idx = len(spd_arr) // 2
            first_mean = float(np.mean(spd_arr[:half_idx]))
            second_mean = float(np.mean(spd_arr[half_idx:]))
            half_drop_pct = ((first_mean - second_mean) / max(1e-6, first_mean)) if first_mean > 0 else 0.0

            is_fatigue = bool(
                reg.slope < 0
                and half_drop_pct >= 0.20
                and (math.isnan(reg.pvalue) or reg.pvalue < 0.15)
            )
            known_teams = [
                t for t in (_clean_str(v) for v in pgrp["team"].tolist())
                if t is not None and t.lower() != "unknown"
            ]
            p_team = Counter(known_teams).most_common(1)[0][0] if known_teams else "Unknown"

            fatigue_trends.append({
                "track_id": str(tid),
                "team": p_team,
                "samples": int(len(pgrp)),
                "speed_slope_per_sec": round(float(reg.slope), 4),
                "p_value": round(float(reg.pvalue), 4) if not math.isnan(reg.pvalue) else None,
                "first_half_mean_speed": round(first_mean, 2),
                "second_half_mean_speed": round(second_mean, 2),
                "half_drop_pct": round(half_drop_pct * 100.0, 2),
                "fatigue_trend_flag": is_fatigue,
            })

            if is_fatigue:
                anomalies_list.append({
                    "anomaly_type": "fatigue_trend",
                    "severity": "medium",
                    "frame_idx": int(pgrp["frame_idx"].iloc[half_idx]),
                    "timestamp_sec": round(float(ts_arr[half_idx]), 2),
                    "track_id": str(tid),
                    "team": p_team,
                    "metric_value": round(half_drop_pct * 100.0, 2),
                    "threshold": 20.0,
                    "description_tr": (
                        f"Oyuncu {tid} ({p_team}) zaman içinde yorgunluk/hız düşüş eğilimi gösterdi "
                        f"(ilk yarı {first_mean:.1f} -> ikinci yarı {second_mean:.1f}, "
                        f"eğim={reg.slope:.3f}/sn, p={reg.pvalue:.3f})."
                    ),
                })

        # Konum & Hareket Anomalisi (Takım merkezine uzaklık + MAD z-skoru + IsolationForest)
        # Kare bazında takım formasyon merkezini hız sıçramasından dışlanmayan TÜM geçerli
        # gözlemlenen oyuncular üzerinden (aykırıya dayanıklı medyan ile) nedensel hesapla
        x_col, y_col = self._get_xy_columns()
        used_obs_centroids = False
        if (
            not self.df.empty
            and x_col in self.df.columns
            and y_col in self.df.columns
            and "class_name" in self.df.columns
            and "frame_idx" in self.df.columns
        ):
            valid_bbox_obs = self._get_valid_bbox_mask(self.df)
            team_not_ref_obs = (self.df["team"] != "Referee") if "team" in self.df.columns else True
            obs_mask = (
                self.df["class_name"].isin(config.HUMAN_CLASS_NAMES)
                & (self.df["class_name"] != "referee")
                & team_not_ref_obs
                & valid_bbox_obs
            )
            if "is_predicted" in self.df.columns:
                obs_mask &= self.df["is_predicted"] != True  # noqa: E712
            if self.is_calibrated and "on_pitch" in self.df.columns:
                obs_mask &= self.df["on_pitch"] != False  # noqa: E712
            obs_pos = self.df[obs_mask].dropna(subset=[x_col, y_col, "frame_idx"]).copy()
            if not self.is_calibrated and not obs_pos.empty:
                min_x, max_x, min_y, max_y = self.bounds
                obs_pos = obs_pos[
                    (obs_pos[x_col] >= min_x - 30.0)
                    & (obs_pos[x_col] <= max_x + 30.0)
                    & (obs_pos[y_col] >= min_y - 30.0)
                    & (obs_pos[y_col] <= max_y + 30.0)
                ]
            if not obs_pos.empty:
                obs_pos = obs_pos.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
                obs_teams: List[str] = []
                for _, tgrp in obs_pos.groupby("track_id", sort=False):
                    r_team: Optional[str] = None
                    t_list = tgrp["team"].tolist() if "team" in tgrp.columns else [None] * len(tgrp)
                    for tv in t_list:
                        ct = _clean_str(tv)
                        if ct is not None:
                            r_team = ct
                        obs_teams.append((ct if ct is not None else r_team) or "Unknown")
                obs_pos["_obs_team"] = obs_teams
                min_bx, max_bx, min_by, max_by = self.bounds
                obs_pos["_obs_nx"] = np.clip((obs_pos[x_col].to_numpy(dtype=float) - min_bx) / max(1e-6, max_bx - min_bx), 0.0, 1.0)
                obs_pos["_obs_ny"] = np.clip((obs_pos[y_col].to_numpy(dtype=float) - min_by) / max(1e-6, max_by - min_by), 0.0, 1.0)
                obs_cent = (
                    obs_pos.groupby(["frame_idx", "_obs_team"], as_index=False)[["_obs_nx", "_obs_ny"]]
                    .median()
                    .rename(columns={"_obs_team": "team", "_obs_nx": "_cx", "_obs_ny": "_cy"})
                )
                merged_cent = speed_df[["frame_idx", "team"]].merge(
                    obs_cent, on=["frame_idx", "team"], how="left"
                )
                if not merged_cent["_cx"].isna().any():
                    speed_df["dist_from_team_centroid"] = np.hypot(
                        speed_df["norm_x"].to_numpy(dtype=float) - merged_cent["_cx"].to_numpy(dtype=float),
                        speed_df["norm_y"].to_numpy(dtype=float) - merged_cent["_cy"].to_numpy(dtype=float),
                    )
                    used_obs_centroids = True

        if not used_obs_centroids:
            centroids = (
                speed_df.groupby(["frame_idx", "team"])[["norm_x", "norm_y"]]
                .transform("median")
            )
            speed_df["dist_from_team_centroid"] = np.hypot(
                speed_df["norm_x"] - centroids["norm_x"],
                speed_df["norm_y"] - centroids["norm_y"],
            )

        # Oyuncu-pencere özetleri üzerinden anomali değerlendirmesi (gürültüyü azaltır)
        pw_df = (
            speed_df.groupby(["window_idx", "track_id", "team"], as_index=False)
            .agg(
                frame_idx=("frame_idx", "min"),
                timestamp_sec=("timestamp_sec", "mean"),
                norm_x=("norm_x", "mean"),
                norm_y=("norm_y", "mean"),
                speed=("speed", "mean"),
                dist_from_team_centroid=("dist_from_team_centroid", "mean"),
                samples=("speed", "count"),
            )
        )
        pw_df = pw_df[pw_df["samples"] >= 3].reset_index(drop=True)

        if len(pw_df) >= 6:
            dist_mad_z = _robust_mad_zscore(pw_df["dist_from_team_centroid"].to_numpy(dtype=float))
            spd_mad_z = _robust_mad_zscore(pw_df["speed"].to_numpy(dtype=float))
            pw_df["dist_mad_z"] = dist_mad_z
            pw_df["speed_mad_z"] = spd_mad_z

            iso_flags = np.zeros(len(pw_df), dtype=bool)
            if len(pw_df) >= 20:
                iso_feats = StandardScaler().fit_transform(
                    pw_df[["norm_x", "norm_y", "dist_from_team_centroid", "speed"]].to_numpy(dtype=float)
                )
                iso = IsolationForest(
                    contamination=float(config.ML_ANOMALY_CONTAMINATION),
                    random_state=self.random_state,
                )
                iso_flags = iso.fit_predict(iso_feats) == -1

            mad_thresh = float(config.ML_ANOMALY_MAD_Z_THRESHOLD)
            for idx, row in pw_df.iterrows():
                is_spatial_mad = abs(float(row["dist_mad_z"])) >= mad_thresh
                is_speed_mad = abs(float(row["speed_mad_z"])) >= mad_thresh
                is_iso = bool(iso_flags[idx])

                if is_spatial_mad or (is_iso and float(row["dist_mad_z"]) >= 2.0):
                    anomalies_list.append({
                        "anomaly_type": "spatial_off_formation",
                        "severity": "high" if abs(float(row["dist_mad_z"])) >= mad_thresh else "medium",
                        "frame_idx": int(row["frame_idx"]),
                        "timestamp_sec": round(float(row["timestamp_sec"]), 2),
                        "track_id": str(row["track_id"]),
                        "team": str(row["team"]),
                        "metric_value": round(float(row["dist_from_team_centroid"]), 4),
                        "threshold": mad_thresh,
                        "description_tr": (
                            f"Oyuncu {row['track_id']} ({row['team']}) {row['timestamp_sec']:.1f}s civarında "
                            f"formasyon dışı konumlanma sergiledi (takım merkezine uzaklık={row['dist_from_team_centroid']:.3f}, "
                            f"MAD z={row['dist_mad_z']:.2f}, IsolationForest={'Aykırı' if is_iso else 'Normal'})."
                        ),
                    })
                elif is_speed_mad or is_iso:
                    anomalies_list.append({
                        "anomaly_type": "movement_anomaly",
                        "severity": "medium",
                        "frame_idx": int(row["frame_idx"]),
                        "timestamp_sec": round(float(row["timestamp_sec"]), 2),
                        "track_id": str(row["track_id"]),
                        "team": str(row["team"]),
                        "metric_value": round(float(row["speed"]), 2),
                        "threshold": mad_thresh,
                        "description_tr": (
                            f"Oyuncu {row['track_id']} ({row['team']}) {row['timestamp_sec']:.1f}s civarında "
                            f"olağandışı hareket yoğunluğu gösterdi (hız={row['speed']:.1f}, "
                            f"MAD z={row['speed_mad_z']:.2f}, IsolationForest={'Aykırı' if is_iso else 'Normal'})."
                        ),
                    })

        anomalies_df = pd.DataFrame(anomalies_list, columns=anom_cols)
        if not anomalies_df.empty:
            anomalies_df = anomalies_df.sort_values(
                ["timestamp_sec", "severity"], ascending=[True, True]
            ).reset_index(drop=True)

        all_win = windows_df[windows_df["team"] == "ALL"] if not windows_df.empty else windows_df
        summary = {
            "coord_mode": self.coord_mode,
            "window_sec": round(win_sec, 2),
            "total_windows": int(len(all_win)),
            "covered_windows": int(all_win["coverage_ok"].sum()) if not all_win.empty else 0,
            "low_coverage_windows": int(all_win["low_coverage_flag"].sum()) if not all_win.empty else 0,
            "tempo_drop_count": int(windows_df["tempo_drop_flag"].sum()) if not windows_df.empty else 0,
            "fatigue_players_count": sum(1 for ft in fatigue_trends if ft["fatigue_trend_flag"]),
            "total_anomalies": int(len(anomalies_df)),
            "anomaly_counts_by_type": (
                anomalies_df["anomaly_type"].value_counts().to_dict()
                if not anomalies_df.empty
                else {}
            ),
            "player_fatigue_trends": fatigue_trends,
        }
        return windows_df, anomalies_df, summary

    # ----------------------------------------------------------------
    def _plot_xp_comparison(self, xp_res: Dict[str, Any], out_path: str) -> str:
        """xP model karşılaştırması (ROC-AUC, PR-AUC, F1, 1-Brier) ve kalibrasyon eğrileri grafiği."""
        models = xp_res.get("models", [])
        if not models:
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
            ax.text(0.5, 0.5, "xP model karşılaştırması için yeterli sınıf çeşitliliği yok", ha="center", va="center")
            ax.set_axis_off()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path
        names = [m["model_name"].replace(" (DummyClassifier)", "\n(Baseline)") for m in models]
        roc_aucs = [m["test"]["roc_auc"] for m in models]
        pr_aucs = [m["test"]["pr_auc"] for m in models]
        f1s = [m["test"]["f1"] for m in models]
        briers = [1.0 - m["test"]["brier_score"] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

        x = np.arange(len(names))
        w = 0.2
        ax1.bar(x - 1.5 * w, roc_aucs, width=w, label="ROC-AUC", color="#1f77b4")
        ax1.bar(x - 0.5 * w, pr_aucs, width=w, label="PR-AUC", color="#2ca02c")
        ax1.bar(x + 0.5 * w, f1s, width=w, label="F1 Skoru", color="#ff7f0e")
        ax1.bar(x + 1.5 * w, briers, width=w, label="1 - Brier", color="#9467bd")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=8.5)
        ax1.set_ylim(0.0, 1.08)
        ax1.set_ylabel("Test Skoru")
        ax1.set_title(f"xP Model Karşılaştırması ({xp_res['data_source']})")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Mükemmel Kalibrasyon")
        colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
        for idx, m in enumerate(models):
            cc = m["test"]["calibration_curve"]
            if cc["prob_pred"] and cc["prob_true"]:
                ax2.plot(
                    cc["prob_pred"],
                    cc["prob_true"],
                    marker="o",
                    linewidth=1.8,
                    color=colors[idx % len(colors)],
                    label=f"{m['model_name'].split(' (')[0]} (Brier={m['test']['brier_score']:.3f})",
                )
        ax2.set_xlabel("Tahmin Edilen Olasılık (xP)")
        ax2.set_ylabel("Gerçekleşen Başarı Oranı")
        ax2.set_title("Test Seti Olasılık Kalibrasyon Eğrisi")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ----------------------------------------------------------------
    def _plot_xt_heatmap(self, xt_res: Dict[str, Any], out_path: str) -> str:
        """12x8 Markov xT (Expected Threat) mekansal ısı haritası."""
        grid = np.asarray(xt_res["xt_grid"], dtype=float)
        rows, cols = grid.shape

        fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150)
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Beklenen Tehdit (xT)", fontsize=9)

        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                txt_col = "white" if val > grid.max() * 0.55 else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", color=txt_col, fontsize=7.5)

        mode_label = (
            "Kalibre Saha (105x68m)"
            if xt_res["coord_mode"] == "pitch_meters"
            else "Kalibresiz Piksel Fallback (Göreli Saha)"
        )
        ax.set_title(f"Mekansal Beklenen Tehdit (xT) Izgarası - {mode_label}\n(Hücum Yönü: Soldan Sağa →)")
        ax.set_xlabel("Saha Uzunluk Bölgesi (1 → 12)")
        ax.set_ylabel("Saha Genişlik Koridoru (1 → 8)")
        ax.set_xticks(range(cols))
        ax.set_xticklabels([str(i + 1) for i in range(cols)])
        ax.set_yticks(range(rows))
        ax.set_yticklabels([str(i + 1) for i in range(rows)])

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ----------------------------------------------------------------
    def _plot_player_clusters_pca(
        self,
        profiles_df: pd.DataFrame,
        cluster_summary: Dict[str, Any],
        out_path: str,
    ) -> str:
        """2B PCA oyuncu rol kümeleme görselleştirmesi."""
        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)

        if profiles_df.empty:
            ax.text(0.5, 0.5, "Kümeleme için yeterli oyuncu verisi yok", ha="center", va="center")
            ax.set_axis_off()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        markers = {"Team A": "o", "Team B": "s"}

        for cid, cgrp in profiles_df.groupby("cluster_id"):
            cid_int = int(cid)
            role = str(cgrp["tactical_role_tr"].iloc[0])
            col = palette[cid_int % len(palette)]
            for team_name, tgrp in cgrp.groupby("team"):
                mk = markers.get(str(team_name), "^")
                ax.scatter(
                    tgrp["pca_1"],
                    tgrp["pca_2"],
                    c=col,
                    marker=mk,
                    s=115,
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.9,
                    label=f"Küme {cid_int}: {role} ({team_name})",
                )
                for _, row in tgrp.iterrows():
                    ax.annotate(
                        f"ID:{row['track_id']}",
                        (row["pca_1"], row["pca_2"]),
                        textcoords="offset points",
                        xytext=(6, 5),
                        fontsize=8,
                    )

        ev = cluster_summary.get("pca_explained_variance_ratio", [0.0, 0.0])
        ev1_raw = ev[0] if len(ev) >= 1 and ev[0] is not None and not pd.isna(ev[0]) else 0.0
        ev2_raw = ev[1] if len(ev) >= 2 and ev[1] is not None and not pd.isna(ev[1]) else 0.0
        ev1 = float(ev1_raw) * 100.0
        ev2 = float(ev2_raw) * 100.0
        sil = cluster_summary.get("best_silhouette")
        opt_k = cluster_summary.get("optimal_k")
        sil_str = f"Silhouette={sil:.3f}" if sil is not None else "Silhouette=N/A"

        ax.set_xlabel(f"PCA Bileşen 1 (Açıklanan Varyans: %{ev1:.1f})")
        ax.set_ylabel(f"PCA Bileşen 2 (Açıklanan Varyans: %{ev2:.1f})")
        ax.set_title(f"Oyuncu Rol Kümeleme 2B PCA Projeksiyonu (Optimal k={opt_k}, {sil_str})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ----------------------------------------------------------------
    def _plot_tempo_and_anomalies(
        self,
        windows_df: pd.DataFrame,
        anomalies_df: pd.DataFrame,
        tempo_summary: Dict[str, Any],
        out_path: str,
    ) -> str:
        """Zaman pencereli takım temposu, kapsam uyarıları ve anomali zaman çizelgesi."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), dpi=150, gridspec_kw={"height_ratios": [2.2, 1.2]})

        unit = "km/sa" if tempo_summary["coord_mode"] == "pitch_meters" else "px/sn"
        if windows_df.empty:
            ax1.text(0.5, 0.5, "Tempo analizi için yeterli hız örneği yok", ha="center", va="center")
            ax2.set_axis_off()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path

        team_colors = {"ALL": "#222222", "Team A": "#1f77b4", "Team B": "#d62728"}
        for tname, tgrp in windows_df.groupby("team"):
            tgrp = tgrp.sort_values("window_start_sec")
            t_mid = 0.5 * (tgrp["window_start_sec"] + tgrp["window_end_sec"])
            cov_df = tgrp[tgrp["coverage_ok"]]
            if not cov_df.empty:
                ax1.plot(
                    0.5 * (cov_df["window_start_sec"] + cov_df["window_end_sec"]),
                    cov_df["mean_speed"],
                    marker="o",
                    linewidth=2.0 if tname == "ALL" else 1.6,
                    color=team_colors.get(tname, "#555555"),
                    label=f"{tname} Ortalama Hız",
                )
            drop_df = tgrp[tgrp["tempo_drop_flag"]]
            if not drop_df.empty:
                ax1.scatter(
                    0.5 * (drop_df["window_start_sec"] + drop_df["window_end_sec"]),
                    drop_df["mean_speed"],
                    color="red",
                    s=110,
                    zorder=5,
                    marker="v",
                    label=f"{tname} Tempo Düşüşü",
                )
            if tname == "ALL":
                low_df = tgrp[tgrp["low_coverage_flag"]]
                for _, lrow in low_df.iterrows():
                    ax1.axvspan(
                        lrow["window_start_sec"],
                        lrow["window_end_sec"],
                        color="#cccccc",
                        alpha=0.35,
                    )

        ax1.set_ylabel(f"Ortalama Hız ({unit})")
        ax1.set_title(
            f"Zaman Pencereli Tempo Eğilimi ve Düşük Kapsam Bölgeleri (Pencere={tempo_summary['window_sec']}s)"
        )
        ax1.grid(True, linestyle="--", alpha=0.4)
        if ax1.get_legend_handles_labels()[0]:
            ax1.legend(loc="best", fontsize=8)

        # Alt panel: Anomali türlerinin zaman çizelgesi (Türkçe y-ekseni etiketleri)
        anomaly_labels_tr = {
            "tempo_drop": "Tempo Düşüşü",
            "fatigue_trend": "Yorgunluk Eğilimi",
            "speed_spike": "Hız Sıçraması",
            "spatial_off_formation": "Pozisyon Anomalisi",
            "movement_anomaly": "Hareket Anomalisi",
            "low_coverage": "Düşük Kapsama",
        }
        if not anomalies_df.empty:
            type_map = {t: idx for idx, t in enumerate(sorted(anomalies_df["anomaly_type"].unique()))}
            sev_colors = {"high": "#d62728", "medium": "#ff7f0e", "info": "#7f7f7f"}
            for _, arow in anomalies_df.iterrows():
                y_pos = type_map[arow["anomaly_type"]]
                ax2.scatter(
                    float(arow["timestamp_sec"]),
                    y_pos,
                    color=sev_colors.get(str(arow["severity"]), "#1f77b4"),
                    s=70,
                    alpha=0.85,
                )
            ax2.set_yticks(list(type_map.values()))
            ax2.set_yticklabels([anomaly_labels_tr.get(k, k) for k in type_map.keys()], fontsize=8.5)
        else:
            ax2.text(0.5, 0.5, "Bayraklanan anomali yok", ha="center", va="center")

        ax2.set_xlabel("Maç Zamanı (saniye)")
        ax2.set_title("Tespit Edilen Tempo Düşüşleri, Yorgunluk ve Konum/Hız Anomalileri")
        ax2.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ----------------------------------------------------------------
    def _build_turkish_markdown_report(self, results: Dict[str, Any], export_paths: Dict[str, Any]) -> str:
        """Tüm ML bulgularını ml-best-practices uyumlu Türkçe Markdown raporu olarak derler."""
        dq = results.get("data_quality") or {}
        xp = results.get("xp_modeling") or {}
        xt = results.get("xt_analysis") or {}
        cl = results.get("player_clustering") or {}
        tp = results.get("tempo_and_anomalies") or {}

        coord_mode = dq.get("coord_mode", self.coord_mode)
        is_cal = bool(dq.get("is_calibrated", self.is_calibrated))
        total_rows = dq.get("total_rows", 0)
        unique_frames = dq.get("unique_frames", 0)
        duration_sec = dq.get("duration_sec", 0.0)
        fps_val = dq.get("fps", self.fps)

        lines: List[str] = [
            "# Futbol Maç Analizi — Taktik ve Scouting ML Raporu (Adım 8)",
            "",
            f"- **Koordinat Modu:** `{coord_mode}` (Kalibre: `{'Evet' if is_cal else 'Hayır'}`)",
            f"- **Toplam Kayıt / Kare:** {total_rows} satır / {unique_frames} kare ({duration_sec} sn @ {fps_val} fps)",
            "",
            "## 1. Veri Kalitesi ve Eksik/NULL Denetimi",
            "",
            "| Sütun | Tip | Eksik Sayısı | Eksik % |",
            "|---|---|---:|---:|",
        ]
        for col, info in (dq.get("missing_by_column") or {}).items():
            lines.append(
                f"| `{col}` | `{info.get('dtype', 'unknown')}` | {info.get('missing_count', 0)} | %{float(info.get('missing_pct', 0.0)):.2f} |"
            )

        lines.extend(["", "### İşleme Stratejisi (Veri Sızıntısız)"])
        for note in dq.get("handling_strategy") or []:
            lines.append(f"- {note}")

        split_info = xp.get("split_info") or {}
        xp_models = xp.get("models") or []
        scoring_unit_mode = xp.get("scoring_unit_mode", "native")
        fallback_reason = xp.get("fallback_reason")
        lines.extend([
            "",
            "## 2. Pas Başarı Olasılığı (xP) Modelleme ve Operasyonel Ödünleşimler",
            "",
            f"- **Eğitim Veri Kaynağı:** `{xp.get('data_source', 'N/A')}` ({xp.get('modeled_events', 0)} olay, pozitif sınıf oranı: %{float(xp.get('positive_rate', 0.0)) * 100:.1f})",
            f"- **Skorlama Birim Modu:** `{scoring_unit_mode}`"
            + (" (piksel vekil olaylar `[0..105]x[0..68]` m orantılı saha ölçeğine izdüşürüldü)" if scoring_unit_mode == "normalized_pitch_projection_105x68" else ""),
            f"- **Kronolojik Bölünme (Ön İşlemeden Önce):** Eğitim={split_info.get('train_count', 0)} | Doğrulama={split_info.get('validation_count', 0)} | Test={split_info.get('test_count', 0)} (Kronoloji doğrulandı: `{xp.get('chronological_split_verified', False)}`)",
            f"- **Seçilen En İyi Model (Doğrulama Setine Göre):** **{xp.get('selected_model') or 'Yok'}**",
        ])
        if fallback_reason:
            lines.append(f"- **Yedek (Fallback) Nedeni:** `{fallback_reason}`")

        if xp_models:
            lines.extend([
                "",
                "| Model | Test Accuracy | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier (Düşük İyi) | Aşırı Öğrenme (Train-Test AUC) | Gecikme (ms/1000 olay) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for m in xp_models:
                tm = m.get("test") or {}
                lines.append(
                    f"| {m.get('model_name', 'Model')} | {float(tm.get('accuracy', 0.0)):.4f} | {float(tm.get('f1', 0.0)):.4f} | {float(tm.get('roc_auc', 0.5)):.4f} | "
                    f"{float(tm.get('pr_auc', 0.0)):.4f} | {float(tm.get('brier_score', 0.0)):.4f} | {float(m.get('overfitting_auc_gap', 0.0)):+.4f} | {float(m.get('latency_per_1000_events_ms', 0.0)):.2f} |"
                )

            lines.extend(["", "### Karmaşıklık Matrisleri (Test Seti) ve Operasyonel Ödünleşimler"])
            for m in xp_models:
                cm = (m.get("test") or {}).get("confusion_matrix", [])
                lines.append(f"- **{m.get('model_name', 'Model')}**: CM=`{cm}` — {m.get('operational_tradeoff_tr', '')}")

        grid_shape = xt.get("grid_shape") or [int(config.ML_XT_GRID_ROWS), int(config.ML_XT_GRID_COLS)]
        g_rows = grid_shape[0] if len(grid_shape) >= 1 else int(config.ML_XT_GRID_ROWS)
        g_cols = grid_shape[1] if len(grid_shape) >= 2 else int(config.ML_XT_GRID_COLS)
        lines.extend([
            "",
            "## 3. Mekansal Beklenen Tehdit (xT) Analizi",
            "",
            f"- **Değerlendirilen Kaynak:** `{xt.get('eval_source', 'N/A')}` ({xt.get('evaluated_events_count', 0)} olay)",
            f"- **Izgara Boyutu & Aralık:** {g_rows}x{g_cols}, `xT` aralığı: `[{float(xt.get('min_xt', 0.0)):.4f}, {float(xt.get('max_xt', 0.0)):.4f}]`",
            f"- **Ortalama `delta_xT` / `expected_xT`:** `{float(xt.get('mean_delta_xT', 0.0)):.4f}` / `{float(xt.get('mean_expected_xT', 0.0)):.4f}`",
        ])
        for tname, tinfo in (xt.get("team_xt_summary") or {}).items():
            lines.append(
                f"- **{tname}:** {tinfo.get('events', 0)} olay | Toplam `delta_xT`={float(tinfo.get('total_delta_xT', 0.0)):.4f} | Toplam `expected_xT`={float(tinfo.get('total_expected_xT', 0.0)):.4f}"
            )

        excluded_tracks = cl.get("excluded_tracks") or []
        excluded_count = cl.get("excluded_tracks_count", len(excluded_tracks))
        lines.extend([
            "",
            "## 4. Oyuncu Rol Kümeleme (Silhouette & 2B PCA)",
            "",
            f"- **Uygun Oyuncu Sayısı:** {cl.get('eligible_players', 0)} (Hariç tutulan iz: {excluded_count})",
            f"- **Optimal Küme Sayısı (k):** `{cl.get('optimal_k')}` (En iyi Silhouette skoru: `{cl.get('best_silhouette')}`)",
            f"- **Aday k Silhouette Skorları:** `{cl.get('candidate_k_scores', {})}`",
            f"- **2B PCA Açıklanan Varyans Oranı:** `{cl.get('pca_explained_variance_ratio', [])}`",
            "",
            "| Küme ID | Taktik Rol (Türkçe) | Oyuncu Sayısı | Oyuncu ID'leri | Ort. Hız | Ort. Top Kontrol % |",
            "|---:|---|---:|---|---:|---:|",
        ])
        for cs in cl.get("cluster_summaries") or []:
            lines.append(
                f"| {cs.get('cluster_id', 0)} | {cs.get('tactical_role_tr', '')} | {cs.get('player_count', 0)} | "
                f"{', '.join(str(tid) for tid in (cs.get('track_ids') or []))} | {float(cs.get('mean_speed', 0.0)):.2f} | %{float(cs.get('mean_ball_control_pct', 0.0)):.2f} |"
            )

        lines.extend([
            "",
            "## 5. Tempo Düşüşü, Yorgunluk Eğilimi ve Anomali Tespiti",
            "",
            f"- **Pencere Süresi / Sayısı:** {tp.get('window_sec', 0.0)} sn | Toplam={tp.get('total_windows', 0)}, Kapsanan={tp.get('covered_windows', 0)}, Düşük Kapsam={tp.get('low_coverage_windows', 0)}",
            f"- **Bayraklanan Tempo Düşüşü / Yorgunluk:** {tp.get('tempo_drop_count', 0)} tempo düşüşü, {tp.get('fatigue_players_count', 0)} yorgunluk eğilimli oyuncu",
            f"- **Toplam Anomali / Uyarı Kaydı:** {tp.get('total_anomalies', 0)} (`{tp.get('anomaly_counts_by_type', {})}`)",
            "",
            "## 6. Üretilen Çıktı Dosyaları",
            "",
        ])
        for key, path in (export_paths or {}).items():
            if path:
                lines.append(f"- `{key}`: `{path}`")

        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        """Tüm ML analiz aşamalarını sırayla çalıştırır (diske yazmadan)."""
        dq = self.audit_data_quality()
        xp_res = self.run_xp_modeling()
        xt_res = self.compute_xt_analysis(xp_results=xp_res)
        profiles_df, cluster_summary = self.cluster_player_roles()
        windows_df, anomalies_df, tempo_summary = self.detect_tempo_and_anomalies()

        results = {
            "fps": self.fps,
            "coord_mode": self.coord_mode,
            "is_calibrated": self.is_calibrated,
            "data_quality": dq,
            "xp_modeling": xp_res,
            "xt_analysis": xt_res,
            "player_clustering": cluster_summary,
            "tempo_and_anomalies": tempo_summary,
            "_artifacts": {
                "player_profiles_df": profiles_df,
                "tempo_windows_df": windows_df,
                "anomalies_df": anomalies_df,
                "pass_predictions_df": xt_res.get("_scored_xt_eval_df", pd.DataFrame()),
            },
        }
        self.last_results_ = results
        return results

    # ----------------------------------------------------------------
    def export_all(
        self,
        output_dir: Optional[str] = None,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Tüm ML analizlerini çalıştırır; CSV tablolarını, PNG grafikleri,
        Türkçe Markdown raporunu ve katı JSON özetini diske kaydeder.
        """
        out_dir = output_dir or config.EXPORT_ML_DIR
        os.makedirs(out_dir, exist_ok=True)

        results = self.run_all()
        artifacts = results.pop("_artifacts")

        # Dahili DataFrame anahtarlarını serileştirilebilir özetten çıkar
        xp_res = results["xp_modeling"]
        xp_res.pop("_scored_events_df", None)
        xp_res.pop("_proxy_scored_df", None)
        xt_res = results["xt_analysis"]
        xt_res.pop("_scored_xt_eval_df", None)
        xt_res.pop("_scored_xt_model_df", None)

        profiles_df: pd.DataFrame = artifacts["player_profiles_df"]
        windows_df: pd.DataFrame = artifacts["tempo_windows_df"]
        anomalies_df: pd.DataFrame = artifacts["anomalies_df"]
        pass_pred_df: pd.DataFrame = artifacts["pass_predictions_df"]

        profiles_csv = os.path.join(out_dir, config.EXPORT_ML_PLAYER_PROFILES_FILENAME)
        windows_csv = os.path.join(out_dir, config.EXPORT_ML_TEMPO_WINDOWS_FILENAME)
        anomalies_csv = os.path.join(out_dir, config.EXPORT_ML_ANOMALIES_FILENAME)
        passes_csv = os.path.join(out_dir, config.EXPORT_ML_PASS_PREDICTIONS_FILENAME)

        profiles_df.to_csv(profiles_csv, index=False)
        windows_df.to_csv(windows_csv, index=False)
        anomalies_df.to_csv(anomalies_csv, index=False)
        pass_pred_df.to_csv(passes_csv, index=False)

        plots: Dict[str, str] = {}
        if generate_plots:
            plots["xp_model_comparison_png"] = self._plot_xp_comparison(
                xp_res, os.path.join(out_dir, "xp_model_comparison.png")
            )
            plots["xt_spatial_heatmap_png"] = self._plot_xt_heatmap(
                xt_res, os.path.join(out_dir, "xt_spatial_heatmap.png")
            )
            plots["player_clusters_pca_png"] = self._plot_player_clusters_pca(
                profiles_df, results["player_clustering"], os.path.join(out_dir, "player_clusters_pca.png")
            )
            plots["tempo_anomaly_timeline_png"] = self._plot_tempo_and_anomalies(
                windows_df, anomalies_df, results["tempo_and_anomalies"], os.path.join(out_dir, "tempo_anomaly_timeline.png")
            )

        report_md_path = os.path.join(out_dir, config.EXPORT_ML_REPORT_FILENAME)
        summary_json_path = os.path.join(out_dir, config.EXPORT_ML_SUMMARY_FILENAME)

        export_paths: Dict[str, Any] = {
            "player_profiles_csv": profiles_csv,
            "tempo_windows_csv": windows_csv,
            "anomalies_csv": anomalies_csv,
            "pass_predictions_csv": passes_csv,
            "ml_report_md": report_md_path,
            "ml_summary_json": summary_json_path,
            **plots,
        }

        md_text = self._build_turkish_markdown_report(results, export_paths)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        results["exports"] = export_paths
        clean_summary = _to_python_scalar(results)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(clean_summary, f, indent=2, ensure_ascii=False, allow_nan=False)

        print(f"[MLAnalyzer] Taktik & Scouting ML raporu kaydedildi: {summary_json_path}")
        return clean_summary


# ----------------------------------------------------------------------
# BAĞIMSIZ CLI GİRİŞ NOKTASI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Takip verisinden xP/xT, oyuncu rol kümeleme ve tempo/anomali ML analizleri üretir."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME),
        help="Giriş takip dosyası (.csv veya .db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.EXPORT_ML_DIR,
        help="ML raporlarının ve grafiklerin kaydedileceği klasör",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS değeri (belirtilmezse veriden otomatik çıkarılır)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="PNG grafik üretimini atla",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input.lower().endswith(".db"):
        ml_analyzer = MLAnalyzer.from_sql(args.input, fps=args.fps)
    else:
        ml_analyzer = MLAnalyzer.from_csv(args.input, fps=args.fps)

    summary = ml_analyzer.export_all(
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )
    xp = summary.get("xp_modeling", {})
    models_list = xp.get("models", [])
    best_m = next((m for m in models_list if m["model_name"] == xp.get("selected_model")), None)
    cl = summary.get("player_clustering", {})
    tp = summary.get("tempo_and_anomalies", {})

    if best_m is not None:
        print(
            f"\n[ML Özeti] Mod: {summary['coord_mode']} | "
            f"xP Model ({xp.get('selected_model')}) -> ROC-AUC: {best_m['test']['roc_auc']} | "
            f"PR-AUC: {best_m['test']['pr_auc']} | Brier: {best_m['test']['brier_score']}"
        )
    else:
        print(f"\n[ML Özeti] Mod: {summary['coord_mode']} | xP Durum: {xp.get('status', 'skipped')}")
    print(
        f"[ML Özeti] Rol Kümeleme -> Uygun Oyuncu: {cl['eligible_players']} | "
        f"Optimal k: {cl['optimal_k']} | Silhouette: {cl['best_silhouette']}"
    )
    print(
        f"[ML Özeti] Tempo & Anomali -> Tempo Düşüşü: {tp['tempo_drop_count']} | "
        f"Yorgunluk Eğilimi: {tp['fatigue_players_count']} | Toplam Anomali: {tp['total_anomalies']}"
    )
