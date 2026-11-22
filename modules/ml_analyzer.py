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
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
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
from modules.pitch_orientation import (
    compute_ball_kinematic_filter,
    has_discontinuity_between,
    lookup_attack_directions,
    resolve_goalkeeper_teams,
    smooth_possession_codes,
)


# ----------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------------------------------
STANDARD_TRACKING_COLUMNS = (
    "frame_idx",
    "timestamp_sec",
    "track_id",
    "class_name",
    "team",
    "confidence",
    "is_predicted",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pixel_x",
    "pixel_y",
    "pitch_x",
    "pitch_y",
    "on_pitch",
)
_STANDARD_TRACKING_COL_SET = frozenset(STANDARD_TRACKING_COLUMNS)
_ML_CAT_COLS = ("track_id", "class_name", "team")
_ML_FLOAT32_COLS = (
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pixel_x",
    "pixel_y",
    "pitch_x",
    "pitch_y",
)
_ML_BOOL_COLS = ("is_predicted", "on_pitch")


def _get_human_class_names() -> set:
    """M1 4-sınıf (goalkeeper/player/referee) ve eski COCO ('person') uyumlu insan sınıf kümesi."""
    return set(getattr(config, "HUMAN_CLASS_NAMES", ())) | {"person"}


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


def _bool_mask(series: pd.Series, na_value: bool) -> np.ndarray:
    """Nullable boolean, float32, int, object ve category serilerini Kleene hatasız bool maskeye çevirir."""
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=bool)
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return series.fillna(na_value).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(dtype):
        arr = series.to_numpy(dtype=np.float32, copy=False)
        out = arr != 0
        nan_m = np.isnan(arr)
        if nan_m.any():
            out = out.copy()
            out[nan_m] = na_value
        return out
    codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=True)
    u_bool = np.empty(len(uniques), dtype=bool)
    for i, u in enumerate(uniques):
        if u is None or (isinstance(u, float) and math.isnan(u)):
            u_bool[i] = na_value
        elif isinstance(u, (bool, np.bool_)):
            u_bool[i] = bool(u)
        elif isinstance(u, (int, float, np.integer, np.floating)):
            u_bool[i] = float(u) != 0.0
        else:
            s = str(u).strip().lower()
            if s in ("true", "1", "1.0", "yes"):
                u_bool[i] = True
            elif s in ("false", "0", "0.0", "no"):
                u_bool[i] = False
            else:
                u_bool[i] = na_value
    out = np.full(n, na_value, dtype=bool)
    valid = codes >= 0
    if valid.any():
        out[valid] = u_bool[codes[valid]]
    return out


def _compact_bool_series(series: pd.Series) -> pd.Series:
    """Boolean sütunu bool (non-null) veya Kleene-hatasız float32 (1.0/0.0/NaN) tipine sıkıştırır."""
    dtype = series.dtype
    if dtype == np.dtype(bool):
        return series
    if pd.api.types.is_bool_dtype(dtype):
        if not series.isna().any():
            return series.to_numpy(dtype=bool)
        arr = np.full(len(series), np.nan, dtype=np.float32)
        notna = series.notna().to_numpy(dtype=bool)
        arr[notna] = series.loc[notna].to_numpy(dtype=np.float32)
        return pd.Series(arr, index=series.index, dtype=np.float32)
    if pd.api.types.is_numeric_dtype(dtype):
        if not series.isna().any():
            arr_num = series.to_numpy(dtype=np.float64, copy=False)
            if np.all((arr_num == 0) | (arr_num == 1)):
                return pd.Series(arr_num != 0, index=series.index, dtype=bool)
        return pd.to_numeric(series, errors="coerce").astype(np.float32)

    codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=True)
    u_vals = np.full(len(uniques), np.nan, dtype=np.float32)
    for i, u in enumerate(uniques):
        if u is None or (isinstance(u, float) and math.isnan(u)):
            continue
        if isinstance(u, (bool, np.bool_)):
            u_vals[i] = 1.0 if bool(u) else 0.0
        elif isinstance(u, (int, float, np.integer, np.floating)):
            u_vals[i] = 1.0 if float(u) != 0.0 else 0.0
        else:
            s = str(u).strip().lower()
            if s in ("true", "1", "1.0", "yes"):
                u_vals[i] = 1.0
            elif s in ("false", "0", "0.0", "no"):
                u_vals[i] = 0.0
    out = np.full(len(series), np.nan, dtype=np.float32)
    valid = codes >= 0
    if valid.any():
        out[valid] = u_vals[codes[valid]]
    if not np.isnan(out).any():
        return pd.Series(out != 0.0, index=series.index, dtype=bool)
    return pd.Series(out, index=series.index, dtype=np.float32)


def _compact_tracking_df(df: pd.DataFrame) -> pd.DataFrame:
    """Yüklenen 16-sütunluk takip DataFrame'ini kompakt tiplere indirger."""
    if "frame_idx" in df.columns:
        f_num = pd.to_numeric(df["frame_idx"], errors="coerce")
        if not f_num.isna().any():
            df["frame_idx"] = f_num.astype(np.int32)
        else:
            df["frame_idx"] = f_num.astype(np.float32)

    if "timestamp_sec" in df.columns:
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce").astype(np.float64)

    for col in _ML_FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    for col in _ML_BOOL_COLS:
        if col in df.columns:
            df[col] = _compact_bool_series(df[col])

    for col in _ML_CAT_COLS:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            s = df[col]
            if pd.api.types.is_numeric_dtype(s.dtype):
                s_str = s.astype(str).where(s.notna(), np.nan)
                df[col] = pd.Categorical(s_str)
            else:
                df[col] = pd.Categorical(s)
    return df


def _encode_clean_strings(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Serideki benzersiz değerleri _clean_str ile bir kez temizleyip int32 kodlara çevirir (-1 = None)."""
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=np.int32), []
    if isinstance(series.dtype, pd.CategoricalDtype):
        raw_codes = series.cat.codes.to_numpy(dtype=np.int32)
        uniques = series.cat.categories
    else:
        raw_codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=True)
        raw_codes = raw_codes.astype(np.int32, copy=False)

    cleaned_list: List[str] = []
    str_to_code: Dict[str, int] = {}
    remap = np.full(len(uniques), -1, dtype=np.int32)
    for i, u in enumerate(uniques):
        cs = _clean_str(u)
        if cs is not None:
            c = str_to_code.get(cs)
            if c is None:
                c = len(cleaned_list)
                cleaned_list.append(cs)
                str_to_code[cs] = c
            remap[i] = c

    out = np.full(n, -1, dtype=np.int32)
    valid = raw_codes >= 0
    if valid.any():
        out[valid] = remap[raw_codes[valid]]
    return out, cleaned_list


def _encode_sorted_track_ids(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """track_id serisini str(val) sırasına uyumlu (groupby düzeni) int32 kodlara ve benzersiz etiket listesine çevirir."""
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=np.int32), []
    if isinstance(series.dtype, pd.CategoricalDtype):
        raw_codes = series.cat.codes.to_numpy(dtype=np.int32)
        uniques = list(series.cat.categories)
    else:
        raw_codes, uniques_idx = pd.factorize(series, sort=False, use_na_sentinel=True)
        raw_codes = raw_codes.astype(np.int32, copy=False)
        uniques = list(uniques_idx)

    # Aynı str() temsiline sahip benzersizleri tekilleştir ve sırala
    unique_items: Dict[str, Any] = {}
    raw_to_str: List[str] = []
    for u in uniques:
        su = str(u)
        raw_to_str.append(su)
        if su not in unique_items:
            unique_items[su] = u

    try:
        sorted_pairs = sorted(unique_items.items(), key=lambda kv: kv[1])
    except TypeError:
        sorted_pairs = sorted(unique_items.items(), key=lambda kv: kv[0])

    sorted_strs = [kv[0] for kv in sorted_pairs]
    str_to_sorted = {s: idx for idx, s in enumerate(sorted_strs)}
    remap = np.array([str_to_sorted[su] for su in raw_to_str], dtype=np.int32)

    out = np.full(n, -1, dtype=np.int32)
    valid = raw_codes >= 0
    if valid.any():
        out[valid] = remap[raw_codes[valid]]
    return out, sorted_strs


class _VectorizedSpikeAnomalies(Sequence):
    """10k+ hız sıçraması anomalilerini milyonlarca dict üretmeden tembel sunan dizi."""

    __slots__ = (
        "frame_idx",
        "timestamp_sec",
        "tid_codes",
        "tid_strs",
        "team_codes",
        "team_strs",
        "metric_values",
        "threshold",
        "unit",
    )

    def __init__(
        self,
        frame_idx: np.ndarray,
        timestamp_sec: np.ndarray,
        tid_codes: np.ndarray,
        tid_strs: List[str],
        team_codes: np.ndarray,
        team_strs: List[str],
        metric_values: np.ndarray,
        threshold: float,
        unit: str,
    ) -> None:
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.tid_codes = tid_codes
        self.tid_strs = tid_strs
        self.team_codes = team_codes
        self.team_strs = team_strs
        self.metric_values = metric_values
        self.threshold = float(threshold)
        self.unit = str(unit)

    def __len__(self) -> int:
        return int(len(self.frame_idx))

    def _item_at(self, idx: int) -> Dict[str, Any]:
        tid_s = self.tid_strs[int(self.tid_codes[idx])]
        tmc = int(self.team_codes[idx])
        team_s = self.team_strs[tmc] if 0 <= tmc < len(self.team_strs) else "Unknown"
        ts_v = round(float(self.timestamp_sec[idx]), 2)
        r_spd = float(self.metric_values[idx])
        return {
            "anomaly_type": "speed_spike",
            "severity": "high",
            "frame_idx": int(self.frame_idx[idx]),
            "timestamp_sec": ts_v,
            "track_id": tid_s,
            "team": team_s,
            "metric_value": round(r_spd, 2),
            "threshold": self.threshold,
            "description_tr": (
                f"Oyuncu {tid_s} için fiziksel sınır üstü hız sıçraması "
                f"({r_spd:.1f} {self.unit} > {self.threshold:.1f} {self.unit}); ortalama hesaptan dışlandı."
            ),
        }

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.frame_idx))
            return [self._item_at(i) for i in range(start, stop, step)]
        n = len(self.frame_idx)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError("index out of range")
        return self._item_at(int(index))

    def __iter__(self):
        for i in range(len(self.frame_idx)):
            yield self._item_at(i)

    def to_dataframe(self, columns: List[str]) -> pd.DataFrame:
        n = len(self.frame_idx)
        if n == 0:
            return pd.DataFrame(columns=columns)
        tid_lut = np.asarray(self.tid_strs, dtype=object)
        tid_arr = tid_lut[self.tid_codes]
        if self.team_strs:
            tm_lut = np.asarray(self.team_strs, dtype=object)
            tm_arr = np.where(
                self.team_codes >= 0,
                tm_lut[np.maximum(0, self.team_codes)],
                "Unknown",
            )
        else:
            tm_arr = np.full(n, "Unknown", dtype=object)
        unit = self.unit
        thresh = self.threshold
        mvals = self.metric_values
        desc_list = [
            f"Oyuncu {t} için fiziksel sınır üstü hız sıçraması ({v:.1f} {unit} > {thresh:.1f} {unit}); ortalama hesaptan dışlandı."
            for t, v in zip(tid_arr.tolist(), mvals.tolist())
        ]
        df = pd.DataFrame({
            "anomaly_type": np.full(n, "speed_spike", dtype=object),
            "severity": np.full(n, "high", dtype=object),
            "frame_idx": self.frame_idx,
            "timestamp_sec": np.round(self.timestamp_sec, 2),
            "track_id": tid_arr,
            "team": tm_arr,
            "metric_value": np.round(mvals, 2),
            "threshold": np.full(n, thresh, dtype=np.float64),
            "description_tr": desc_list,
        })
        return df[columns]


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
        copy: bool = False,
    ):
        self.df = df.copy() if (df is not None and copy) else (df if df is not None else pd.DataFrame())
        self.fps = float(fps) if fps and fps > 0 else self._infer_fps()
        self.analytics_summary = analytics_summary
        self.random_state = int(random_state)

        self.has_pitch_players, self.has_pitch_ball, self.is_calibrated = self._detect_calibration()
        self.coord_mode = "pitch_meters" if self.is_calibrated else "pixel_fallback"
        self.bounds = self._compute_coordinate_bounds()

        # İnceleme / test erişimi için eğitilmiş modeller, hız önbelleği ve son sonuçlar
        self._speed_cache: Dict[bool, Tuple[Tuple[Any, ...], pd.DataFrame, Sequence[Dict[str, Any]]]] = {}
        self._orientation_cache: Optional[Tuple[int, bool, np.ndarray, List[str], List[Dict[str, Any]]]] = None
        self.xp_models_: Dict[str, Pipeline] = {}
        self.best_xp_model_name_: Optional[str] = None
        self.best_xp_pipeline_: Optional[Pipeline] = None
        self.last_results_: Optional[Dict[str, Any]] = None

    def clear_cache(self) -> None:
        """Önbelleğe alınmış adım hızı ve yönelim tablolarını temizler."""
        self._speed_cache.clear()
        self._orientation_cache = None

    def _get_resolved_teams_and_halves(self) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Girdi DataFrame'ini değiştirmeden kaleci eşlemesini ve devre/hücum yönlerini önbellekli döndürür."""
        n_rows = len(self.df)
        if (
            self._orientation_cache is not None
            and self._orientation_cache[0] == n_rows
            and self._orientation_cache[1] == self.is_calibrated
        ):
            _, _, r_codes, r_list, halves = self._orientation_cache
            return r_codes, r_list, halves

        r_codes, r_list, halves = resolve_goalkeeper_teams(
            df=self.df,
            fps=self.fps,
            bounds=self.bounds,
            is_calibrated=self.is_calibrated,
        )
        self._orientation_cache = (n_rows, self.is_calibrated, r_codes, r_list, halves)
        return r_codes, r_list, halves

    # ----------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        fps: Optional[float] = None,
        analytics_summary: Optional[Dict[str, Any]] = None,
        random_state: int = config.ML_RANDOM_STATE,
    ) -> "MLAnalyzer":
        """CSV takip dosyasından sütun budamalı ve kompakt tipli MLAnalyzer örneği oluşturur."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

        try:
            header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            return cls(
                df=pd.DataFrame(columns=list(STANDARD_TRACKING_COLUMNS)),
                fps=fps,
                analytics_summary=analytics_summary,
                random_state=random_state,
                copy=False,
            )
        usecols = [c for c in header_cols if c in _STANDARD_TRACKING_COL_SET]
        usecols_arg = usecols if usecols else None

        dtype_hint = {
            "track_id": "category",
            "class_name": "category",
            "team": "category",
            "timestamp_sec": "float64",
            "confidence": "float32",
            "bbox_x1": "float32",
            "bbox_y1": "float32",
            "bbox_x2": "float32",
            "bbox_y2": "float32",
            "pixel_x": "float32",
            "pixel_y": "float32",
            "pitch_x": "float32",
            "pitch_y": "float32",
        }
        try:
            df = pd.read_csv(csv_path, usecols=usecols_arg, dtype=dtype_hint)
        except (ValueError, TypeError):
            df = pd.read_csv(csv_path, usecols=usecols_arg)
        df = _compact_tracking_df(df)
        return cls(df=df, fps=fps, analytics_summary=analytics_summary, random_state=random_state, copy=False)

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
        """SQLite veritabanından parçalı (chunked) ve kompakt tipli MLAnalyzer örneği oluşturur."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite dosyası bulunamadı: {db_path}")
        table = table_name or config.EXPORT_SQL_TABLE_NAME
        safe_table = str(table).replace('"', '""')
        sql_chunksize = max(1, int(getattr(config, "ANALYTICS_SQL_CHUNKSIZE", 250_000)))

        conn = sqlite3.connect(db_path)
        try:
            info_df = pd.read_sql_query(f'PRAGMA table_info("{safe_table}")', conn)
            existing_cols = info_df["name"].tolist() if not info_df.empty else []
            selected_cols = [c for c in STANDARD_TRACKING_COLUMNS if c in existing_cols]
            cols_sql = ", ".join(f'"{c}"' for c in selected_cols) if selected_cols else "*"
            query = f'SELECT {cols_sql} FROM "{safe_table}"'

            chunks: List[pd.DataFrame] = []
            for chunk in pd.read_sql_query(query, conn, chunksize=sql_chunksize):
                chunks.append(_compact_tracking_df(chunk))
        finally:
            conn.close()

        if not chunks:
            df = pd.DataFrame(columns=selected_cols)
        elif len(chunks) == 1:
            df = chunks[0]
        else:
            cat_data = {}
            for col in _ML_CAT_COLS:
                if col in chunks[0].columns:
                    cat_data[col] = union_categoricals([c[col] for c in chunks])
            non_cat_cols = [c for c in chunks[0].columns if c not in cat_data]
            if non_cat_cols:
                df = pd.concat([c[non_cat_cols] for c in chunks], ignore_index=True)
            else:
                df = pd.DataFrame(index=range(sum(len(c) for c in chunks)))
            for col, cat_vals in cat_data.items():
                df[col] = cat_vals
            ordered_cols = [c for c in chunks[0].columns if c in df.columns]
            df = df[ordered_cols]

        return cls(df=df, fps=fps, analytics_summary=analytics_summary, random_state=random_state, copy=False)

    # ----------------------------------------------------------------
    def _infer_fps(self) -> float:
        """frame_idx ve timestamp_sec farklarından FPS değerini çıkarır (varsayılan: 25.0)."""
        if self.df.empty or "frame_idx" not in self.df.columns or "timestamp_sec" not in self.df.columns:
            return 25.0

        f_arr = pd.to_numeric(self.df["frame_idx"], errors="coerce").to_numpy(dtype=np.float64)
        t_arr = pd.to_numeric(self.df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float64)
        valid_ft = np.isfinite(f_arr) & np.isfinite(t_arr)
        if int(valid_ft.sum()) < 2:
            return 25.0

        f_arr = f_arr[valid_ft]
        t_arr = t_arr[valid_ft]
        if np.any(f_arr[1:] < f_arr[:-1]):
            order = np.argsort(f_arr, kind="mergesort")
            f_arr = f_arr[order]
            t_arr = t_arr[order]

        first = np.empty(len(f_arr), dtype=bool)
        first[0] = True
        first[1:] = f_arr[1:] != f_arr[:-1]
        f_u = f_arr[first]
        t_u = t_arr[first]
        if len(f_u) < 2:
            return 25.0

        df_idx = np.diff(f_u)
        dt = np.diff(t_u)
        valid = (df_idx > 0) & (dt > 0)
        if not valid.any():
            return 25.0

        est = float(np.median(df_idx[valid] / dt[valid]))
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

        has_valid_pitch = (self.df["pitch_x"].notna() & self.df["pitch_y"].notna()).to_numpy(dtype=bool)
        cls_s = self.df["class_name"]
        is_ball = (cls_s == config.BALL_CLASS_NAME).fillna(False).to_numpy(dtype=bool)
        human_names = _get_human_class_names()
        team_not_ref = (
            ~((self.df["team"] == "Referee").fillna(False).to_numpy(dtype=bool))
            if "team" in self.df.columns
            else True
        )
        is_player = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
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

        mask = (self.df["pixel_x"].notna() & self.df["pixel_y"].notna()).to_numpy(dtype=bool)
        if not mask.any():
            return (0.0, 1920.0, 0.0, 1080.0)

        if max_frame_idx is not None and "frame_idx" in self.df.columns:
            m = mask & (self.df["frame_idx"] <= int(max_frame_idx)).fillna(False).to_numpy(dtype=bool)
            if m.any():
                mask = m
        if "is_predicted" in self.df.columns:
            m = mask & (~_bool_mask(self.df["is_predicted"], na_value=False))
            if m.any():
                mask = m
        valid_bbox = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
        m = mask & valid_bbox
        if m.any():
            mask = m
        in_frame = mask & ((self.df["pixel_x"] >= 0.0) & (self.df["pixel_y"] >= 0.0)).fillna(False).to_numpy(dtype=bool)
        if in_frame.any():
            mask = in_frame

        if not mask.any():
            return (0.0, 1920.0, 0.0, 1080.0)

        px_vals = self.df.loc[mask, "pixel_x"].to_numpy(dtype=np.float64)
        py_vals = self.df.loc[mask, "pixel_y"].to_numpy(dtype=np.float64)
        min_x = float(px_vals.min())
        max_x = float(px_vals.max())
        min_y = float(py_vals.min())
        max_y = float(py_vals.max())
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
            (self.df["class_name"] == config.BALL_CLASS_NAME).fillna(False).to_numpy(dtype=bool)
            if "class_name" in self.df.columns
            else np.zeros(n_rows, dtype=bool)
        )
        human_names = _get_human_class_names()
        is_human = (
            self.df["class_name"].isin(human_names).fillna(False).to_numpy(dtype=bool)
            if "class_name" in self.df.columns
            else np.zeros(n_rows, dtype=bool)
        )
        is_pred = (
            _bool_mask(self.df["is_predicted"], na_value=False)
            if "is_predicted" in self.df.columns
            else np.zeros(n_rows, dtype=bool)
        )

        team_na = (
            self.df["team"].isna().to_numpy(dtype=bool)
            if "team" in self.df.columns
            else np.ones(n_rows, dtype=bool)
        )
        team_na_ball = int((team_na & is_ball).sum())
        team_na_human = int((team_na & is_human).sum())

        conf_na_human = (
            int((self.df["confidence"].isna().to_numpy(dtype=bool) & is_human).sum())
            if "confidence" in self.df.columns
            else n_rows
        )
        conf_na_ball = (
            int((self.df["confidence"].isna().to_numpy(dtype=bool) & is_ball).sum())
            if "confidence" in self.df.columns
            else n_rows
        )

        valid_bbox = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
        invalid_bbox_total = int((~valid_bbox).sum())
        invalid_bbox_pred = int((~valid_bbox & is_pred).sum())

        oob_total = 0
        oob_pred = 0
        if "pixel_x" in self.df.columns and "pixel_y" in self.df.columns:
            max_px_x = max(1920.0, float(self.bounds[1]))
            max_px_y = max(1080.0, float(self.bounds[3]))
            px_x = self.df["pixel_x"].to_numpy(dtype=np.float64, copy=False)
            px_y = self.df["pixel_y"].to_numpy(dtype=np.float64, copy=False)
            oob_mask = (
                (px_x < 0.0)
                | (px_y < 0.0)
                | (px_x > max_px_x)
                | (px_y > max_px_y)
            )
            oob_total = int(oob_mask.sum())
            oob_pred = int((oob_mask & is_pred).sum())

        on_pitch_false = (
            int((~_bool_mask(self.df["on_pitch"], na_value=True)).sum())
            if "on_pitch" in self.df.columns
            else 0
        )

        # Takip izlerinin gözlenen kare sayıları (geçici kısa izleri tespit et)
        transient_tracks: List[str] = []
        if "track_id" in self.df.columns:
            tid_sub = self.df.loc[is_human & (~is_pred), "track_id"]
            if not tid_sub.empty:
                if isinstance(tid_sub.dtype, pd.CategoricalDtype) or pd.api.types.is_integer_dtype(tid_sub.dtype):
                    obs_counts = tid_sub.value_counts()
                else:
                    obs_counts = tid_sub.astype(str).value_counts()
                transient_tracks = [
                    str(tid) for tid, cnt in obs_counts.items()
                    if 0 < int(cnt) < config.ML_MIN_TRACK_FRAMES
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
    def _compute_player_step_speeds(self, exclude_referees: bool = True) -> Tuple[pd.DataFrame, Sequence[Dict[str, Any]]]:
        """
        Gözlemlenen ardışık kareler arasında oyuncu adım hızlarını hesaplar.
        Gerçek dışı hız sıçramalarını (ışınlanma / kamera kesmesi) hız ortalamalarından
        ayırır ve anomali adayı olarak döndürür.
        """
        cache_key = bool(exclude_referees)
        cache_sig = (
            id(self.df),
            self.df.shape,
            tuple(self.df.columns),
            self.is_calibrated,
            self.fps,
            self.bounds,
        )
        cached = self._speed_cache.get(cache_key)
        if cached is not None and cached[0] == cache_sig:
            cached_spikes = cached[2]
            spikes_ret = (
                cached_spikes
                if isinstance(cached_spikes, _VectorizedSpikeAnomalies)
                else [dict(a) for a in cached_spikes]
            )
            return cached[1].copy(deep=True), spikes_ret

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
            or "track_id" not in self.df.columns
        ):
            empty_df = pd.DataFrame(columns=empty_cols)
            self._speed_cache[cache_key] = (cache_sig, empty_df, [])
            return empty_df.copy(deep=True), []

        valid_bbox = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
        cls_s = self.df["class_name"]
        human_names = _get_human_class_names()
        mask = cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool) & valid_bbox
        if exclude_referees:
            team_not_ref = (
                ~((self.df["team"] == "Referee").fillna(False).to_numpy(dtype=bool))
                if "team" in self.df.columns
                else True
            )
            mask &= ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool)) & team_not_ref
        if "is_predicted" in self.df.columns:
            mask &= ~_bool_mask(self.df["is_predicted"], na_value=False)
        if self.is_calibrated and "on_pitch" in self.df.columns:
            mask &= _bool_mask(self.df["on_pitch"], na_value=True)

        mask &= (
            self.df[x_col].notna().to_numpy(dtype=bool)
            & self.df[y_col].notna().to_numpy(dtype=bool)
            & self.df["frame_idx"].notna().to_numpy(dtype=bool)
            & self.df["track_id"].notna().to_numpy(dtype=bool)
        )

        if not self.is_calibrated:
            min_x, max_x, min_y, max_y = self.bounds
            x_all = self.df[x_col].to_numpy(dtype=np.float64, copy=False)
            y_all = self.df[y_col].to_numpy(dtype=np.float64, copy=False)
            mask &= (
                (x_all >= min_x - 30.0)
                & (x_all <= max_x + 30.0)
                & (y_all >= min_y - 30.0)
                & (y_all <= max_y + 30.0)
            )

        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            empty_df = pd.DataFrame(columns=empty_cols)
            self._speed_cache[cache_key] = (cache_sig, empty_df, [])
            return empty_df.copy(deep=True), []

        tid_codes, tid_strs = _encode_sorted_track_ids(self.df["track_id"].iloc[idx])
        fids = self.df["frame_idx"].iloc[idx].to_numpy(dtype=np.int64)
        xs = np.round(self.df[x_col].iloc[idx].to_numpy(dtype=np.float64), 4)
        ys = np.round(self.df[y_col].iloc[idx].to_numpy(dtype=np.float64), 4)
        ts_arr = (
            np.round(self.df["timestamp_sec"].iloc[idx].to_numpy(dtype=np.float64), 4)
            if "timestamp_sec" in self.df.columns
            else np.full(len(idx), np.nan, dtype=np.float64)
        )
        if "team" in self.df.columns:
            team_codes, team_strs = _encode_clean_strings(self.df["team"].iloc[idx])
        else:
            team_codes = np.full(len(idx), -1, dtype=np.int32)
            team_strs = []

        # R1: Yalnızca kaleci satırlarının ("Goalkeeper"/goalkeeper) takımını çözümle;
        # saha oyuncularının eksik (None) etiketlerine dokunma (nedensellik korunur).
        res_codes_all, res_team_list, _ = self._get_resolved_teams_and_halves()
        gk_str_c = team_strs.index("Goalkeeper") if "Goalkeeper" in team_strs else -2
        is_gk_idx = (cls_s.iloc[idx] == "goalkeeper").fillna(False).to_numpy(dtype=bool)
        if gk_str_c >= 0:
            is_gk_idx |= (team_codes == gk_str_c)
        if is_gk_idx.any():
            team_strs = list(team_strs)
            str_to_c = {s: i for i, s in enumerate(team_strs)}
            for s in res_team_list:
                if s not in str_to_c:
                    str_to_c[s] = len(team_strs)
                    team_strs.append(s)
            res_remap = np.array([str_to_c[s] for s in res_team_list], dtype=np.int32)
            res_slice = res_codes_all[idx]
            gk_valid = is_gk_idx & (res_slice >= 0)
            if gk_valid.any():
                team_codes = team_codes.copy()
                team_codes[gk_valid] = res_remap[res_slice[gk_valid]]

        # Kararlı (track_id, frame_idx) sıralaması ve yinelenen karelerin ilkini tutma
        order = np.lexsort((fids, tid_codes))
        tid_codes = tid_codes[order]
        fids = fids[order]
        xs = xs[order]
        ys = ys[order]
        ts_arr = ts_arr[order]
        team_codes = team_codes[order]

        dup_keep = np.empty(len(fids), dtype=bool)
        dup_keep[0] = True
        dup_keep[1:] = (tid_codes[1:] != tid_codes[:-1]) | (fids[1:] != fids[:-1])
        if not dup_keep.all():
            tid_codes = tid_codes[dup_keep]
            fids = fids[dup_keep]
            xs = xs[dup_keep]
            ys = ys[dup_keep]
            ts_arr = ts_arr[dup_keep]
            team_codes = team_codes[dup_keep]

        n_rows = len(fids)
        if n_rows < 2:
            empty_df = pd.DataFrame(columns=empty_cols)
            self._speed_cache[cache_key] = (cache_sig, empty_df, [])
            return empty_df.copy(deep=True), []

        # Nedensel takım ileri doldurma (track başına)
        idx_all = np.arange(n_rows, dtype=np.int64)
        is_t_start = np.empty(n_rows, dtype=bool)
        is_t_start[0] = True
        is_t_start[1:] = tid_codes[1:] != tid_codes[:-1]
        row_t_start = np.maximum.accumulate(np.where(is_t_start, idx_all, 0))
        last_valid_pos = np.maximum.accumulate(np.where(team_codes >= 0, idx_all, -1))
        has_team = last_valid_pos >= row_t_start
        resolved_team_codes = np.where(has_team, team_codes[np.maximum(0, last_valid_pos)], -1)

        max_speed_cap = (
            float(config.MAX_REALISTIC_PLAYER_SPEED_KMH)
            if self.is_calibrated
            else float(config.ML_PIXEL_MAX_SPEED_PX_PER_SEC)
        )
        max_gap = max(1, int(config.DISTANCE_CALC_MAX_FRAME_GAP))
        smooth_win = min(3, max(1, int(config.SPEED_SMOOTHING_WINDOW_FRAMES)))

        same_track = tid_codes[1:] == tid_codes[:-1]
        df_i = fids[1:] - fids[:-1]
        step_ok = same_track & (df_i > 0) & (df_i <= max_gap)
        dt_step = np.where(step_ok, df_i / self.fps, 1.0)
        dx_step = xs[1:] - xs[:-1]
        dy_step = ys[1:] - ys[:-1]
        dist_step = np.hypot(dx_step, dy_step)
        scale_spd = 3.6 if self.is_calibrated else 1.0
        raw_spd_step = np.where(step_ok, (dist_step / dt_step) * scale_spd, 0.0)

        step_ok_full = np.r_[False, step_ok]
        raw_spd_full = np.r_[0.0, raw_spd_step]
        dx_full = np.r_[0.0, dx_step]
        dy_full = np.r_[0.0, dy_step]
        dt_full = np.r_[0.0, dt_step]

        is_spike = step_ok_full & (raw_spd_full > max_speed_cap)
        n_spikes = int(is_spike.sum())
        spike_anomalies: Any = []
        if n_spikes > 10_000:
            unit = "km/sa" if self.is_calibrated else "px/sn"
            sp_idx = np.flatnonzero(is_spike)
            sp_fids = fids[sp_idx]
            sp_ts_raw = ts_arr[sp_idx]
            sp_ts = np.where(np.isfinite(sp_ts_raw), sp_ts_raw, sp_fids.astype(np.float64) / self.fps)
            spike_anomalies = _VectorizedSpikeAnomalies(
                frame_idx=sp_fids.astype(np.int64, copy=False),
                timestamp_sec=sp_ts.astype(np.float64, copy=False),
                tid_codes=tid_codes[sp_idx].astype(np.int32, copy=False),
                tid_strs=tid_strs,
                team_codes=resolved_team_codes[sp_idx].astype(np.int32, copy=False),
                team_strs=team_strs,
                metric_values=raw_spd_full[sp_idx].astype(np.float64, copy=False),
                threshold=max_speed_cap,
                unit=unit,
            )
        elif n_spikes > 0:
            unit = "km/sa" if self.is_calibrated else "px/sn"
            for si in np.flatnonzero(is_spike):
                s_idx = int(si)
                tid_s = tid_strs[int(tid_codes[s_idx])]
                tmc = int(resolved_team_codes[s_idx])
                team_s = team_strs[tmc] if tmc >= 0 else "Unknown"
                ts_s = float(ts_arr[s_idx]) if np.isfinite(ts_arr[s_idx]) else float(fids[s_idx]) / self.fps
                r_spd = float(raw_spd_full[s_idx])
                spike_anomalies.append({
                    "anomaly_type": "speed_spike",
                    "severity": "high",
                    "frame_idx": int(fids[s_idx]),
                    "timestamp_sec": round(ts_s, 2),
                    "track_id": tid_s,
                    "team": team_s,
                    "metric_value": round(r_spd, 2),
                    "threshold": max_speed_cap,
                    "description_tr": (
                        f"Oyuncu {tid_s} için fiziksel sınır üstü hız sıçraması "
                        f"({r_spd:.1f} {unit} > {max_speed_cap:.1f} {unit}); ortalama hesaptan dışlandı."
                    ),
                })

        is_keep = step_ok_full & (~is_spike)
        if not is_keep.any():
            empty_df = pd.DataFrame(columns=empty_cols)
            self._speed_cache[cache_key] = (cache_sig, empty_df, spike_anomalies)
            spikes_ret = (
                spike_anomalies
                if isinstance(spike_anomalies, _VectorizedSpikeAnomalies)
                else [dict(a) for a in spike_anomalies]
            )
            return empty_df.copy(deep=True), spikes_ret

        # Kesinlikle nedensel geriye dönük yumuşatma (segment ve sıçrama sınırlarında sıfırlanır)
        last_reset = np.maximum.accumulate(np.where(~is_keep, idx_all, -1))
        run_len = idx_all - last_reset
        w_eff = np.minimum(run_len, smooth_win)

        keep_idx = np.flatnonzero(is_keep)
        w_k = w_eff[keep_idx].astype(np.float64)
        spd_sum = raw_spd_full[keep_idx].copy()
        for offset in range(1, smooth_win):
            has_prev = w_eff[keep_idx] > offset
            if has_prev.any():
                spd_sum[has_prev] += raw_spd_full[keep_idx[has_prev] - offset]
        smoothed_spd = spd_sum / w_k

        ts_keep_raw = ts_arr[keep_idx]
        ts_keep = np.where(np.isfinite(ts_keep_raw), ts_keep_raw, fids[keep_idx] / self.fps)
        x_keep = xs[keep_idx]
        y_keep = ys[keep_idx]
        min_bx, max_bx, min_by, max_by = self.bounds
        nx_keep = np.clip((x_keep - min_bx) / max(1e-6, max_bx - min_bx), 0.0, 1.0)
        ny_keep = np.clip((y_keep - min_by) / max(1e-6, max_by - min_by), 0.0, 1.0)

        team_cats = list(team_strs)
        if "Unknown" in team_cats:
            unknown_code = team_cats.index("Unknown")
        else:
            unknown_code = len(team_cats)
            team_cats.append("Unknown")
        keep_team_codes = np.where(
            resolved_team_codes[keep_idx] >= 0,
            resolved_team_codes[keep_idx],
            unknown_code,
        )

        speed_df = pd.DataFrame({
            "frame_idx": fids[keep_idx].astype(np.int32),
            "timestamp_sec": np.round(ts_keep, 3),
            "track_id": pd.Categorical.from_codes(tid_codes[keep_idx], categories=tid_strs),
            "team": pd.Categorical.from_codes(keep_team_codes, categories=team_cats),
            "x": x_keep,
            "y": y_keep,
            "norm_x": nx_keep,
            "norm_y": ny_keep,
            "speed": smoothed_spd,
            "dx": dx_full[keep_idx],
            "dy": dy_full[keep_idx],
            "dt": dt_full[keep_idx],
        })

        self._speed_cache[cache_key] = (cache_sig, speed_df, spike_anomalies)
        spikes_ret = (
            spike_anomalies
            if isinstance(spike_anomalies, _VectorizedSpikeAnomalies)
            else [dict(a) for a in spike_anomalies]
        )
        return speed_df.copy(deep=True), spikes_ret

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
            "start_zone_x", "start_zone_y", "end_zone_x", "end_zone_y",
            "pitch_zone_start", "pitch_zone_end", "attack_dir",
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

        cls_s = self.df["class_name"]
        f_notna = self.df["frame_idx"].notna().to_numpy(dtype=bool)
        xy_notna = (self.df[x_col].notna() & self.df[y_col].notna()).to_numpy(dtype=bool)

        # Top verisi (aynı karede varsa is_predicted=False tercih edilir)
        b_mask = (cls_s == config.BALL_CLASS_NAME).fillna(False).to_numpy(dtype=bool) & xy_notna & f_notna
        b_idx = np.flatnonzero(b_mask)
        if len(b_idx) == 0:
            return pd.DataFrame(columns=event_cols)

        b_fids = self.df["frame_idx"].iloc[b_idx].to_numpy(dtype=np.int64)
        b_xs = self.df[x_col].iloc[b_idx].to_numpy(dtype=np.float64)
        b_ys = self.df[y_col].iloc[b_idx].to_numpy(dtype=np.float64)
        b_ts = (
            self.df["timestamp_sec"].iloc[b_idx].to_numpy(dtype=np.float64)
            if "timestamp_sec" in self.df.columns
            else np.full(len(b_idx), np.nan, dtype=np.float64)
        )
        b_pred = (
            _bool_mask(self.df["is_predicted"].iloc[b_idx], na_value=False).astype(np.uint8)
            if "is_predicted" in self.df.columns
            else np.zeros(len(b_idx), dtype=np.uint8)
        )
        b_ord = np.lexsort((b_pred, b_fids))
        b_fids = b_fids[b_ord]
        b_xs = b_xs[b_ord]
        b_ys = b_ys[b_ord]
        b_ts = b_ts[b_ord]
        b_pred = b_pred[b_ord]

        b_first = np.empty(len(b_fids), dtype=bool)
        b_first[0] = True
        b_first[1:] = b_fids[1:] != b_fids[:-1]
        ball_fids = b_fids[b_first]
        ball_xs = b_xs[b_first]
        ball_ys = b_ys[b_first]
        ball_ts = b_ts[b_first]
        ball_is_pred = b_pred[b_first]
        n_ball = len(ball_fids)

        kin = compute_ball_kinematic_filter(
            ball_fids=ball_fids,
            ball_xs=ball_xs,
            ball_ys=ball_ys,
            ball_is_pred=ball_is_pred,
            fps=self.fps,
            is_calibrated=self.is_calibrated,
            bounds=self.bounds,
        )

        res_codes_all, res_team_list, halves = self._get_resolved_teams_and_halves()
        ref_res_code = res_team_list.index("Referee") if "Referee" in res_team_list else -2

        valid_bbox = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
        human_names = _get_human_class_names()
        team_not_ref = (
            (res_codes_all != ref_res_code)
            if ref_res_code >= 0
            else (
                ~((self.df["team"] == "Referee").fillna(False).to_numpy(dtype=bool))
                if "team" in self.df.columns
                else np.ones(len(self.df), dtype=bool)
            )
        )
        p_mask = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
            & team_not_ref
            & valid_bbox
            & xy_notna
            & f_notna
        )
        if self.is_calibrated and "on_pitch" in self.df.columns:
            p_mask &= _bool_mask(self.df["on_pitch"], na_value=True)

        p_idx = np.flatnonzero(p_mask)
        if len(p_idx) == 0:
            return pd.DataFrame(columns=event_cols)

        p_fids = self.df["frame_idx"].iloc[p_idx].to_numpy(dtype=np.int64)
        p_xs = self.df[x_col].iloc[p_idx].to_numpy(dtype=np.float64)
        p_ys = self.df[y_col].iloc[p_idx].to_numpy(dtype=np.float64)
        p_tid_codes, tid_list = (
            _encode_clean_strings(self.df["track_id"].iloc[p_idx])
            if "track_id" in self.df.columns
            else (np.full(len(p_idx), -1, dtype=np.int32), [])
        )
        p_team_codes, team_list = (
            _encode_clean_strings(self.df["team"].iloc[p_idx])
            if "team" in self.df.columns
            else (np.full(len(p_idx), -1, dtype=np.int32), [])
        )
        p_is_gk = (cls_s.iloc[p_idx] == "goalkeeper").fillna(False).to_numpy(dtype=bool)
        gk_raw_c = team_list.index("Goalkeeper") if "Goalkeeper" in team_list else -2
        if gk_raw_c >= 0:
            p_is_gk |= (p_team_codes == gk_raw_c)

        # Çözümlenmiş kaleci takımlarını raw listeyle birleştir (yalnızca GK satırlarında)
        p_res_slice = res_codes_all[p_idx]
        if p_is_gk.any():
            team_list = list(team_list)
            t_str_to_c = {s: i for i, s in enumerate(team_list)}
            for s in res_team_list:
                if s not in t_str_to_c:
                    t_str_to_c[s] = len(team_list)
                    team_list.append(s)
            res_remap = np.array([t_str_to_c[s] for s in res_team_list], dtype=np.int32)
            gk_valid = p_is_gk & (p_res_slice >= 0)
            if gk_valid.any():
                p_team_codes = p_team_codes.copy()
                p_team_codes[gk_valid] = res_remap[p_res_slice[gk_valid]]

        p_is_pred = (
            _bool_mask(self.df["is_predicted"].iloc[p_idx], na_value=False).astype(np.uint8)
            if "is_predicted" in self.df.columns
            else np.zeros(len(p_idx), dtype=np.uint8)
        )

        # Kare sırasına göre kararlı sıralama (kare içi sıra korunur)
        if len(p_fids) > 1 and np.any(p_fids[1:] < p_fids[:-1]):
            p_ord = np.argsort(p_fids, kind="mergesort")
            p_fids = p_fids[p_ord]
            p_xs = p_xs[p_ord]
            p_ys = p_ys[p_ord]
            p_tid_codes = p_tid_codes[p_ord]
            p_team_codes = p_team_codes[p_ord]
            p_is_pred = p_is_pred[p_ord]

        # NEDENSEL (CAUSAL) takım çözümleme: Yalnızca kesinlikle önceki karelerden (fid' < fid)
        n_p = len(p_fids)
        order = np.lexsort((p_fids, p_tid_codes))
        tc = p_tid_codes[order]
        fc = p_fids[order]
        tm = p_team_codes[order]
        idx_p = np.arange(n_p, dtype=np.int64)

        is_t_start = np.empty(n_p, dtype=bool)
        is_t_start[0] = True
        is_t_start[1:] = tc[1:] != tc[:-1]
        is_tf_start = is_t_start.copy()
        if n_p > 1:
            is_tf_start[1:] |= (fc[1:] != fc[:-1])

        row_t_start = np.maximum.accumulate(np.where(is_t_start, idx_p, 0))
        row_tf_start = np.maximum.accumulate(np.where(is_tf_start, idx_p, 0))
        last_valid_pos = np.maximum.accumulate(np.where(tm >= 0, idx_p, -1))
        prev_pos = np.where(row_tf_start > 0, last_valid_pos[np.maximum(0, row_tf_start - 1)], -1)
        has_prior = (prev_pos >= row_t_start) & (tc >= 0)
        resolved_sorted = np.where(tm >= 0, tm, np.where(has_prior, tm[np.maximum(0, prev_pos)], -1))

        p_resolved_team_codes = np.empty(n_p, dtype=np.int32)
        p_resolved_team_codes[order] = resolved_sorted
        p_raw_missing = (p_team_codes < 0).astype(np.uint8)

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

        # Oyuncu hızlarını np.searchsorted tabanlı arama dizisi olarak hazırla
        speed_df, _ = self._compute_player_step_speeds(exclude_referees=True)
        tid_to_code = {s: i for i, s in enumerate(tid_list)}
        sp_keys_s = np.empty(0, dtype=np.int64)
        sp_vals_s = np.empty(0, dtype=np.float64)
        sp_fallback: Dict[Tuple[int, int], float] = {}
        if not speed_df.empty and tid_to_code:
            sp_fids = speed_df["frame_idx"].to_numpy(dtype=np.int64)
            sp_raw_c, sp_uniqs = pd.factorize(speed_df["track_id"], sort=False)
            sp_remap = np.array([tid_to_code.get(str(u), -1) for u in sp_uniqs], dtype=np.int32)
            sp_tids = np.where(sp_raw_c >= 0, sp_remap[np.maximum(0, sp_raw_c)], -1)
            sp_spds = speed_df["speed"].to_numpy(dtype=np.float64)
            sp_in_range = (sp_tids >= 0) & (sp_fids >= 0) & (sp_fids < (1 << 31))
            if sp_in_range.any():
                sp_keys = (sp_fids[sp_in_range] << 32) | sp_tids[sp_in_range].astype(np.int64)
                sp_ord = np.argsort(sp_keys, kind="mergesort")
                sp_keys_s = sp_keys[sp_ord]
                sp_vals_s = sp_spds[sp_in_range][sp_ord]
            sp_out_range = (sp_tids >= 0) & (~sp_in_range)
            if sp_out_range.any():
                for fk, tk, sv in zip(sp_fids[sp_out_range], sp_tids[sp_out_range], sp_spds[sp_out_range]):
                    sp_fallback[(int(fk), int(tk))] = float(sv)

        def _lookup_speed(fid_val: int, tid_c: int) -> float:
            if len(sp_keys_s) > 0 and 0 <= fid_val < (1 << 31):
                k = (int(fid_val) << 32) | int(tid_c)
                pos = int(np.searchsorted(sp_keys_s, k))
                if pos < len(sp_keys_s) and sp_keys_s[pos] == k:
                    return float(sp_vals_s[pos])
            if sp_fallback:
                return sp_fallback.get((int(fid_val), int(tid_c)), 0.0)
            return 0.0

        starts = np.searchsorted(p_fids, ball_fids, side="left")
        ends = np.searchsorted(p_fids, ball_fids, side="right")
        active = np.flatnonzero(ends > starts)

        raw_tid_codes = np.full(n_ball, -1, dtype=np.int32)
        raw_team_codes = np.full(n_ball, -1, dtype=np.int32)
        raw_win_idx = np.full(n_ball, -1, dtype=np.int64)
        raw_min_d = np.full(n_ball, np.nan, dtype=np.float64)
        raw_opp_d = np.full(n_ball, np.nan, dtype=np.float64)
        raw_opp_n = np.zeros(n_ball, dtype=np.int32)

        if len(active) > 0:
            act_s = starts[active]
            act_e = ends[active]
            act_lens = act_e - act_s
            first_len = int(act_lens[0])
            # 90 dakikalık düzenli kare dilimlerinde (sabit oyuncu sayısı/kare) %100 vektörel hızlı yol
            if first_len > 0 and np.all(act_lens == first_len):
                n_act = len(active)
                k_players = first_len
                b_xs_act = ball_xs[active]
                b_ys_act = ball_ys[active]

                base_idx = act_s[:, None] + np.arange(k_players, dtype=np.int64)[None, :]
                px_mat = p_xs[base_idx]
                py_mat = p_ys[base_idx]
                dx_mat = px_mat - b_xs_act[:, None]
                dy_mat = py_mat - b_ys_act[:, None]
                d2_mat = dx_mat * dx_mat + dy_mat * dy_mat
                min_locs = np.argmin(d2_mat, axis=1)
                row_arange = np.arange(n_act, dtype=np.int64)
                min_dists = np.sqrt(d2_mat[row_arange, min_locs])
                win_global_idx = act_s + min_locs
                chosen_tids = p_tid_codes[win_global_idx]

                valid_snap = (min_dists <= max_ctrl_dist) & (chosen_tids >= 0)
                v_rows = np.flatnonzero(valid_snap)
                if len(v_rows) > 0:
                    v_bi = active[v_rows]
                    v_s = act_s[v_rows]
                    v_win_idx = win_global_idx[v_rows]
                    v_tids = chosen_tids[v_rows]
                    v_teams = p_resolved_team_codes[v_win_idx]

                    v_base = v_s[:, None] + np.arange(k_players, dtype=np.int64)[None, :]
                    v_px_mat = p_xs[v_base]
                    v_py_mat = p_ys[v_base]
                    v_tm_mat = p_resolved_team_codes[v_base]
                    win_px = p_xs[v_win_idx][:, None]
                    win_py = p_ys[v_win_idx][:, None]

                    opp_mask_mat = (
                        (v_teams[:, None] >= 0)
                        & (v_tm_mat >= 0)
                        & (v_tm_mat != v_teams[:, None])
                    )
                    opp_d_mat = np.hypot(v_px_mat - win_px, v_py_mat - win_py)
                    opp_d_masked = np.where(opp_mask_mat, opp_d_mat, np.inf)
                    min_opp_d = opp_d_masked.min(axis=1)
                    has_opp = np.isfinite(min_opp_d)

                    raw_tid_codes[v_bi] = v_tids
                    raw_team_codes[v_bi] = v_teams
                    raw_win_idx[v_bi] = v_win_idx
                    raw_min_d[v_bi] = min_dists[v_rows]
                    raw_opp_d[v_bi] = np.where(has_opp, min_opp_d, np.nan)
                    raw_opp_n[v_bi] = (opp_d_masked <= pressure_radius).sum(axis=1)
            else:
                for b_i in active:
                    bi = int(b_i)
                    s = int(starts[bi])
                    e = int(ends[bi])
                    bx = float(ball_xs[bi])
                    by = float(ball_ys[bi])

                    pxs_slice = p_xs[s:e]
                    pys_slice = p_ys[s:e]
                    dx_s = pxs_slice - bx
                    dy_s = pys_slice - by
                    d2 = dx_s * dx_s + dy_s * dy_s
                    min_local = int(np.argmin(d2))
                    min_d = math.hypot(float(dx_s[min_local]), float(dy_s[min_local]))
                    if min_d > max_ctrl_dist:
                        continue

                    chosen_tid_c = int(p_tid_codes[s + min_local])
                    if chosen_tid_c < 0:
                        continue

                    chosen_team_c = int(p_resolved_team_codes[s + min_local])
                    if chosen_team_c >= 0:
                        teams_slice = p_resolved_team_codes[s:e]
                        opp_mask = (teams_slice >= 0) & (teams_slice != chosen_team_c)
                        if opp_mask.any():
                            opp_d = np.hypot(
                                pxs_slice[opp_mask] - pxs_slice[min_local],
                                pys_slice[opp_mask] - pys_slice[min_local],
                            )
                            nearest_opp = float(opp_d.min())
                            opp_nearby = int((opp_d <= pressure_radius).sum())
                        else:
                            nearest_opp = np.nan
                            opp_nearby = 0
                    else:
                        nearest_opp = np.nan
                        opp_nearby = 0

                    raw_tid_codes[bi] = chosen_tid_c
                    raw_team_codes[bi] = chosen_team_c
                    raw_win_idx[bi] = s + min_local
                    raw_min_d[bi] = min_d
                    raw_opp_d[bi] = nearest_opp
                    raw_opp_n[bi] = opp_nearby

        # R3: Hava topu / tek karelik imkânsız sıçrama karelerinde kontrolü reddet
        valid_ctrl_mask = kin["valid_ctrl_mask"]
        if len(valid_ctrl_mask) == n_ball:
            raw_tid_codes[~valid_ctrl_mask] = -1
            raw_team_codes[~valid_ctrl_mask] = -1

        # MatchAnalyzer ile birebir uyumlu çoğunluk oyu yumuşatma + min kontrol süresi
        tid_to_team_code = np.full(len(tid_list), -1, dtype=np.int32)
        win = max(1, int(config.POSSESSION_SMOOTHING_FRAMES))
        sm_tids, sm_teams = smooth_possession_codes(
            raw_tid_codes=raw_tid_codes,
            raw_team_codes=raw_team_codes,
            tid_to_team_code=tid_to_team_code,
            win=win,
            min_control_frames=int(kin["min_control_frames"]),
            preserve_frame_team=True,
        )

        valid_sm = np.flatnonzero(sm_tids >= 0)
        if len(valid_sm) < 2:
            return pd.DataFrame(columns=event_cols)

        sm_t_vals = sm_tids[valid_sm]
        change_idx = np.flatnonzero(sm_t_vals[1:] != sm_t_vals[:-1]) + 1
        sp_starts_idx = np.r_[0, change_idx]
        sp_ends_idx = np.r_[change_idx - 1, len(valid_sm) - 1]

        def _build_snap(bi: int, tid_c: int, team_c: int, fallback_bi: int) -> Dict[str, Any]:
            fid = int(ball_fids[bi])
            ts = float(ball_ts[bi]) if np.isfinite(ball_ts[bi]) else fid / self.fps
            bx = float(ball_xs[bi])
            by = float(ball_ys[bi])

            if raw_tid_codes[bi] == tid_c and raw_win_idx[bi] >= 0:
                w_idx = int(raw_win_idx[bi])
                b_p_dist = float(raw_min_d[bi])
                opp_d = float(raw_opp_d[bi]) if np.isfinite(raw_opp_d[bi]) else np.nan
                opp_n = int(raw_opp_n[bi])
                raw_miss = int(p_raw_missing[w_idx])
                p_pred = int(p_is_pred[w_idx])
            else:
                # Yumuşatma köprüsü karesi: oyuncuyu bu karede (yoksa en yakın gözlenen karede) bul
                s_sl = int(starts[bi])
                e_sl = int(ends[bi])
                matched_loc = -1
                if e_sl > s_sl:
                    locs = np.flatnonzero(p_tid_codes[s_sl:e_sl] == tid_c)
                    if len(locs) > 0:
                        matched_loc = s_sl + int(locs[0])
                if matched_loc >= 0:
                    px_v = float(p_xs[matched_loc])
                    py_v = float(p_ys[matched_loc])
                    b_p_dist = math.hypot(px_v - bx, py_v - by)
                    opp_d = np.nan
                    opp_n = 0
                    if team_c >= 0:
                        tms = p_resolved_team_codes[s_sl:e_sl]
                        om = (tms >= 0) & (tms != team_c)
                        if om.any():
                            od = np.hypot(p_xs[s_sl:e_sl][om] - px_v, p_ys[s_sl:e_sl][om] - py_v)
                            opp_d = float(od.min())
                            opp_n = int((od <= pressure_radius).sum())
                    raw_miss = int(p_raw_missing[matched_loc])
                    p_pred = int(p_is_pred[matched_loc])
                else:
                    fb_w = int(raw_win_idx[fallback_bi]) if fallback_bi >= 0 and raw_win_idx[fallback_bi] >= 0 else -1
                    b_p_dist = float(raw_min_d[fallback_bi]) if fallback_bi >= 0 and np.isfinite(raw_min_d[fallback_bi]) else 0.0
                    opp_d = float(raw_opp_d[fallback_bi]) if fallback_bi >= 0 and np.isfinite(raw_opp_d[fallback_bi]) else np.nan
                    opp_n = int(raw_opp_n[fallback_bi]) if fallback_bi >= 0 else 0
                    raw_miss = 1 if team_c < 0 else (int(p_raw_missing[fb_w]) if fb_w >= 0 else 0)
                    p_pred = int(p_is_pred[fb_w]) if fb_w >= 0 else 0

            return {
                "b_idx": bi,
                "frame_idx": fid,
                "timestamp_sec": ts,
                "ball_x": bx,
                "ball_y": by,
                "tid_code": tid_c,
                "track_id": tid_list[tid_c],
                "team_code": team_c,
                "team": team_list[team_c] if team_c >= 0 else None,
                "ball_player_dist": b_p_dist,
                "nearest_opponent_dist": opp_d,
                "opponents_nearby": opp_n,
                "passer_speed": _lookup_speed(fid, tid_c),
                "is_team_missing": raw_miss,
                "is_ball_predicted": int(ball_is_pred[bi]),
                "is_passer_predicted": p_pred,
            }

        spells: List[Dict[str, Any]] = []
        min_cf = int(kin["min_control_frames"])
        for s_pos, e_pos in zip(sp_starts_idx, sp_ends_idx):
            s_i = int(s_pos)
            e_i = int(e_pos)
            if (e_i - s_i + 1) < min_cf:
                continue
            bi_s = int(valid_sm[s_i])
            bi_e = int(valid_sm[e_i])
            tid_c = int(sm_tids[bi_s])
            # En yakın gerçek tespit yedek indeksini bul (köprü durumları için)
            sub_bis = valid_sm[s_i:e_i + 1]
            obs_mask = raw_tid_codes[sub_bis] == tid_c
            fb_s = int(sub_bis[obs_mask][0]) if obs_mask.any() else -1
            fb_e = int(sub_bis[obs_mask][-1]) if obs_mask.any() else -1

            spells.append({
                "tid_code": tid_c,
                "start": _build_snap(bi_s, tid_c, int(sm_teams[bi_s]), fb_s),
                "end": _build_snap(bi_e, tid_c, int(sm_teams[bi_e]), fb_e),
            })

        if len(spells) < 2:
            return pd.DataFrame(columns=event_cols)

        cols_n = int(config.ML_XT_GRID_COLS)
        rows_n = int(config.ML_XT_GRID_ROWS)
        min_pass_dur = float(kin["min_pass_duration_sec"])
        max_pass_spd = float(kin["max_pass_speed"])
        disc_starts = kin["discontinuity_starts"]
        disc_ends = kin["discontinuity_ends"]
        single_half = len(halves) == 1
        if single_half:
            ad0 = halves[0].get("attack_directions") or {}
            da0 = int(ad0.get("Team A", 1))
            db0 = int(ad0.get("Team B", -1))
            half_ranges: List[Tuple[int, int, int, int]] = []
        else:
            da0, db0 = 1, -1
            half_ranges = [
                (
                    int(h["start_frame"]),
                    int(h["end_frame"]),
                    int((h.get("attack_directions") or {}).get("Team A", 1)),
                    int((h.get("attack_directions") or {}).get("Team B", -1)),
                )
                for h in halves
            ]
        events: List[Dict[str, Any]] = []

        for i in range(len(spells) - 1):
            s1_end = spells[i]["end"]
            s2_start = spells[i + 1]["start"]
            if s1_end["tid_code"] == s2_start["tid_code"]:
                continue

            gap_sec = (s2_start["frame_idx"] - s1_end["frame_idx"]) / self.fps
            dx = s2_start["ball_x"] - s1_end["ball_x"]
            dy = s2_start["ball_y"] - s1_end["ball_y"]
            pass_len = math.hypot(dx, dy)

            if gap_sec > config.PASS_MAX_GAP_SEC or pass_len < min_move:
                continue
            if gap_sec + 1e-6 < min_pass_dur:
                continue
            if (pass_len / max(gap_sec, 1.0 / self.fps)) > max_pass_spd:
                continue
            if has_discontinuity_between(s1_end["frame_idx"], s2_start["frame_idx"], disc_starts, disc_ends):
                continue

            sx_n, sy_n = self._normalize_xy(s1_end["ball_x"], s1_end["ball_y"])
            ex_n, ey_n = self._normalize_xy(s2_start["ball_x"], s2_start["ball_y"])
            angle = math.atan2(dy, dx) if pass_len > 1e-6 else 0.0

            from_team = s1_end["team"] or "Unknown"
            to_team = s2_start["team"] or "Unknown"
            if from_team in ("Team A", "Team B") and to_team in ("Team A", "Team B"):
                outcome: Any = 1 if from_team == to_team else 0
            elif from_team != "Unknown" and to_team != "Unknown":
                outcome = 1 if from_team == to_team else 0
            else:
                outcome = np.nan

            # R2: Otomatik tespit edilen devre / takım hücum yönünü kullan
            if single_half:
                adir = da0 if from_team == "Team A" else (db0 if from_team == "Team B" else 1)
            else:
                fid_ev = int(s1_end["frame_idx"])
                adir = 1
                n_h = len(half_ranges)
                for h_idx, (hs, he, da_h, db_h) in enumerate(half_ranges):
                    in_h = (
                        (fid_ev <= he)
                        if h_idx == 0
                        else ((fid_ev >= hs) if h_idx == n_h - 1 else (hs <= fid_ev <= he))
                    )
                    if in_h:
                        adir = da_h if from_team == "Team A" else (db_h if from_team == "Team B" else 1)
                        break
            fwd_dx = (ex_n - sx_n) * float(adir)
            if adir < 0:
                sx_eff, sy_eff = 1.0 - sx_n, 1.0 - sy_n
                ex_eff, ey_eff = 1.0 - ex_n, 1.0 - ey_n
            else:
                sx_eff, sy_eff = sx_n, sy_n
                ex_eff, ey_eff = ex_n, ey_n

            opp_dist = s1_end["nearest_opponent_dist"]

            # Alıcı baskı mesafesini gelecek kareden (s2_start) DEĞİL, serbest bırakma karesinden (s1_end) ölç
            rec_pressure: Any = np.nan
            s1_bi = int(s1_end["b_idx"])
            rs = int(starts[s1_bi])
            re = int(ends[s1_bi])
            if re > rs:
                to_tid_c = int(s2_start["tid_code"])
                rel_tids_sl = p_tid_codes[rs:re]
                rec_matches = np.flatnonzero(rel_tids_sl == to_tid_c)
                if len(rec_matches) > 0:
                    rec_local = int(rec_matches[0])
                    rel_teams_sl = p_resolved_team_codes[rs:re]
                    rec_team_c = int(rel_teams_sl[rec_local])
                    ref_team_c = rec_team_c if rec_team_c >= 0 else int(s1_end["team_code"])
                    if ref_team_c >= 0:
                        opp_m = (
                            (rel_tids_sl != to_tid_c)
                            & (rel_teams_sl >= 0)
                            & (rel_teams_sl != ref_team_c)
                        )
                        if opp_m.any():
                            rec_x = p_xs[rs + rec_local]
                            rec_y = p_ys[rs + rec_local]
                            rec_pressure = float(
                                np.hypot(p_xs[rs:re][opp_m] - rec_x, p_ys[rs:re][opp_m] - rec_y).min()
                            )

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
                "start_zone_x": min(cols_n - 1, max(0, int(sx_eff * cols_n))),
                "start_zone_y": min(rows_n - 1, max(0, int(sy_eff * rows_n))),
                "end_zone_x": min(cols_n - 1, max(0, int(ex_eff * cols_n))),
                "end_zone_y": min(rows_n - 1, max(0, int(ey_eff * rows_n))),
                "pitch_zone_start": _zone_label_from_norm(sx_eff, sy_eff),
                "pitch_zone_end": _zone_label_from_norm(ex_eff, ey_eff),
                "attack_dir": adir,
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

        min_x, max_x, min_y, max_y = bounds
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)

        sx_n = np.clip((out["start_x"].to_numpy(dtype=np.float64) - min_x) / span_x, 0.0, 1.0)
        sy_n = np.clip((out["start_y"].to_numpy(dtype=np.float64) - min_y) / span_y, 0.0, 1.0)
        ex_n = np.clip((out["end_x"].to_numpy(dtype=np.float64) - min_x) / span_x, 0.0, 1.0)
        ey_n = np.clip((out["end_y"].to_numpy(dtype=np.float64) - min_y) / span_y, 0.0, 1.0)

        if "attack_dir" in out.columns:
            adir_arr = pd.to_numeric(out["attack_dir"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
            is_rtl = adir_arr < 0
        elif "frame_idx" in out.columns and "from_team" in out.columns and not self.df.empty:
            _, _, halves = self._get_resolved_teams_and_halves()
            adir_arr = lookup_attack_directions(
                out["frame_idx"].fillna(0).to_numpy(dtype=np.int64),
                out["from_team"].fillna("Unknown").astype(str).to_numpy(),
                halves,
            )
            is_rtl = adir_arr < 0
        elif "from_team" in out.columns:
            team_codes, team_list = _encode_clean_strings(out["from_team"])
            tb_code = team_list.index("Team B") if "Team B" in team_list else -2
            is_rtl = team_codes == tb_code
        else:
            is_rtl = np.zeros(len(out), dtype=bool)
        fwd_dx = np.where(is_rtl, sx_n - ex_n, ex_n - sx_n)
        sx_eff = np.where(is_rtl, 1.0 - sx_n, sx_n)
        sy_eff = np.where(is_rtl, 1.0 - sy_n, sy_n)
        ex_eff = np.where(is_rtl, 1.0 - ex_n, ex_n)
        ey_eff = np.where(is_rtl, 1.0 - ey_n, ey_n)

        out["start_norm_x"] = np.round(sx_n, 4)
        out["start_norm_y"] = np.round(sy_n, 4)
        out["end_norm_x"] = np.round(ex_n, 4)
        out["end_norm_y"] = np.round(ey_n, 4)
        out["forward_progress_dx"] = np.round(fwd_dx, 4)
        out["start_zone_x"] = np.clip((sx_eff * cols_n).astype(int), 0, cols_n - 1)
        out["start_zone_y"] = np.clip((sy_eff * rows_n).astype(int), 0, rows_n - 1)
        out["end_zone_x"] = np.clip((ex_eff * cols_n).astype(int), 0, cols_n - 1)
        out["end_zone_y"] = np.clip((ey_eff * rows_n).astype(int), 0, rows_n - 1)
        out["pitch_zone_start"] = [_zone_label_from_norm(float(x), float(y)) for x, y in zip(sx_eff, sy_eff)]
        out["pitch_zone_end"] = [_zone_label_from_norm(float(x), float(y)) for x, y in zip(ex_eff, ey_eff)]
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
            z_sx = pd.to_numeric(events_df["start_zone_x"], errors="coerce").to_numpy(dtype=np.float64)
            z_sy = pd.to_numeric(events_df["start_zone_y"], errors="coerce").to_numpy(dtype=np.float64)
            z_ex = pd.to_numeric(events_df["end_zone_x"], errors="coerce").to_numpy(dtype=np.float64)
            z_ey = pd.to_numeric(events_df["end_zone_y"], errors="coerce").to_numpy(dtype=np.float64)
            valid_z = np.isfinite(z_sx) & np.isfinite(z_sy) & np.isfinite(z_ex) & np.isfinite(z_ey)
            if valid_z.any():
                c1 = np.clip(z_sx[valid_z].astype(int), 0, cols - 1)
                r1 = np.clip(z_sy[valid_z].astype(int), 0, rows - 1)
                c2 = np.clip(z_ex[valid_z].astype(int), 0, cols - 1)
                r2 = np.clip(z_ey[valid_z].astype(int), 0, rows - 1)
                np.add.at(counts, (r1 * cols + c1, r2 * cols + c2), 1.0)
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

        def _safe_norm_arr(df_in: pd.DataFrame, col: str) -> np.ndarray:
            if col not in df_in.columns:
                return np.full(len(df_in), 0.5, dtype=np.float64)
            arr = pd.to_numeric(df_in[col], errors="coerce").to_numpy(dtype=np.float64)
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = arr.copy()
                arr[bad] = 0.5
            return np.clip(arr, 0.0, 0.9999)

        _, _, halves = self._get_resolved_teams_and_halves()

        def score_df_with_xt(df_in: pd.DataFrame, use_halves: bool = True) -> pd.DataFrame:
            if df_in.empty:
                return df_in.copy()
            df_out = df_in.copy()
            sx_n = _safe_norm_arr(df_out, "start_norm_x")
            sy_n = _safe_norm_arr(df_out, "start_norm_y")
            ex_n = _safe_norm_arr(df_out, "end_norm_x")
            ey_n = _safe_norm_arr(df_out, "end_norm_y")

            # Hücum yönü sağdan sola (-1) ise koordinatları takım perspektifine çevir
            if "attack_dir" in df_out.columns:
                adir_arr = pd.to_numeric(df_out["attack_dir"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
                is_rtl = adir_arr < 0
            elif (
                use_halves
                and not self.df.empty
                and "frame_idx" in df_out.columns
                and "from_team" in df_out.columns
                and eval_source not in ("provided_events", "calibrated_synthetic_match")
            ):
                adir_arr = lookup_attack_directions(
                    df_out["frame_idx"].fillna(0).to_numpy(dtype=np.int64),
                    df_out["from_team"].fillna("Unknown").astype(str).to_numpy(),
                    halves,
                )
                is_rtl = adir_arr < 0
            elif "from_team" in df_out.columns:
                ft_codes, ft_list = _encode_clean_strings(df_out["from_team"])
                tb_c = ft_list.index("Team B") if "Team B" in ft_list else -2
                is_rtl = ft_codes == tb_c
            else:
                is_rtl = np.zeros(len(df_out), dtype=bool)

            sx_eff = np.where(is_rtl, 1.0 - sx_n, sx_n)
            sy_eff = np.where(is_rtl, 1.0 - sy_n, sy_n)
            ex_eff = np.where(is_rtl, 1.0 - ex_n, ex_n)
            ey_eff = np.where(is_rtl, 1.0 - ey_n, ey_n)

            c1 = np.clip((sx_eff * cols).astype(int), 0, cols - 1)
            r1 = np.clip((sy_eff * rows).astype(int), 0, rows - 1)
            c2 = np.clip((ex_eff * cols).astype(int), 0, cols - 1)
            r2 = np.clip((ey_eff * rows).astype(int), 0, rows - 1)

            xt_s = xt_grid[r1, c1]
            xt_e = xt_grid[r2, c2]
            d_xt = xt_e - xt_s

            if "predicted_xp" in df_out.columns:
                xp_arr = pd.to_numeric(df_out["predicted_xp"], errors="coerce").to_numpy(dtype=np.float64)
                bad_xp = ~np.isfinite(xp_arr)
                if bad_xp.any():
                    xp_arr = xp_arr.copy()
                    xp_arr[bad_xp] = 0.75
            else:
                xp_arr = np.full(len(df_out), 0.75, dtype=np.float64)
            exp_xt = xp_arr * d_xt

            df_out["xT_start"] = np.round(xt_s, 5)
            df_out["xT_end"] = np.round(xt_e, 5)
            df_out["delta_xT"] = np.round(d_xt, 5)
            df_out["expected_xT"] = np.round(exp_xt, 5)
            return df_out

        scored_eval = score_df_with_xt(eval_df, use_halves=True)
        model_uses_halves = (
            xp_results is not None
            and xp_results.get("data_source") == "real_tracking_events"
        )
        scored_model = (
            score_df_with_xt(xp_results["_scored_events_df"], use_halves=model_uses_halves)
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

        halves_summary = [
            {
                "period": int(h["period"]),
                "start_frame": int(h["start_frame"]),
                "end_frame": int(h["end_frame"]),
                "detection_reason": h.get("detection_reason", "initial"),
                "attack_directions": {
                    k: ("+x" if int(v) >= 0 else "-x")
                    for k, v in (h.get("attack_directions") or {}).items()
                },
            }
            for h in halves
        ]

        return {
            "coord_mode": self.coord_mode,
            "eval_source": eval_source,
            "halves_detected": len(halves_summary),
            "attack_directions_by_half": halves_summary,
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

        cls_s = self.df["class_name"]
        human_names = _get_human_class_names()
        valid_bbox = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
        h_mask = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & valid_bbox
            & self.df[x_col].notna().to_numpy(dtype=bool)
            & self.df[y_col].notna().to_numpy(dtype=bool)
            & self.df["track_id"].notna().to_numpy(dtype=bool)
        )
        if "is_predicted" in self.df.columns:
            h_mask &= ~_bool_mask(self.df["is_predicted"], na_value=False)
        if self.is_calibrated and "on_pitch" in self.df.columns:
            h_mask &= _bool_mask(self.df["on_pitch"], na_value=True)
        if not self.is_calibrated and h_mask.any():
            min_x, max_x, min_y, max_y = self.bounds
            x_all = self.df[x_col].to_numpy(dtype=np.float64, copy=False)
            y_all = self.df[y_col].to_numpy(dtype=np.float64, copy=False)
            h_mask &= (
                (x_all >= min_x - 30.0)
                & (x_all <= max_x + 30.0)
                & (y_all >= min_y - 30.0)
                & (y_all <= max_y + 30.0)
            )

        # Hız tablosu ve ön-gruplama (O(N_tracks x N_rows) taramayı önler)
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
        speed_by_track: Dict[str, np.ndarray] = {}
        if not speed_df.empty:
            s_codes, s_uniques = pd.factorize(speed_df["track_id"], sort=False)
            s_vals = speed_df["speed"].to_numpy(dtype=np.float64)
            s_ord = np.argsort(s_codes, kind="mergesort")
            sc_s = s_codes[s_ord]
            sv_s = s_vals[s_ord]
            s_brk = np.flatnonzero(sc_s[1:] != sc_s[:-1]) + 1
            for st, en in zip(np.r_[0, s_brk], np.r_[s_brk, len(sc_s)]):
                c_val = int(sc_s[st])
                if c_val >= 0:
                    speed_by_track[str(s_uniques[c_val])] = sv_s[st:en]

        h_idx = np.flatnonzero(h_mask)
        if len(h_idx) == 0:
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

        h_tid_codes, tid_strs = _encode_sorted_track_ids(self.df["track_id"].iloc[h_idx])
        n_tracks = len(tid_strs)
        h_xs = self.df[x_col].iloc[h_idx].to_numpy(dtype=np.float64)
        h_ys = self.df[y_col].iloc[h_idx].to_numpy(dtype=np.float64)
        h_is_ref_cls = (cls_s.iloc[h_idx] == "referee").fillna(False).to_numpy(dtype=bool)
        res_codes_all, res_team_list, halves = self._get_resolved_teams_and_halves()
        if "team" in self.df.columns:
            h_team_codes = res_codes_all[h_idx]
            h_team_strs = list(res_team_list)
            ref_res_c = h_team_strs.index("Referee") if "Referee" in h_team_strs else -2
            h_team_not_ref = (h_team_codes != ref_res_c) if ref_res_c >= 0 else np.ones(len(h_idx), dtype=bool)
        else:
            h_team_codes = np.full(len(h_idx), -1, dtype=np.int32)
            h_team_strs = []
            h_team_not_ref = np.ones(len(h_idx), dtype=bool)

        h_fids_arr = (
            self.df["frame_idx"].iloc[h_idx].to_numpy(dtype=np.int64)
            if "frame_idx" in self.df.columns
            else np.zeros(len(h_idx), dtype=np.int64)
        )
        h_is_rtl = np.zeros(len(h_idx), dtype=bool)
        ta_c = h_team_strs.index("Team A") if "Team A" in h_team_strs else -2
        tb_c = h_team_strs.index("Team B") if "Team B" in h_team_strs else -2
        for idx_h, h in enumerate(halves):
            if len(halves) == 1:
                hm = np.ones(len(h_idx), dtype=bool)
            elif idx_h == 0:
                hm = h_fids_arr <= int(h["end_frame"])
            elif idx_h == len(halves) - 1:
                hm = h_fids_arr >= int(h["start_frame"])
            else:
                hm = (h_fids_arr >= int(h["start_frame"])) & (h_fids_arr <= int(h["end_frame"]))
            ad = h.get("attack_directions", {})
            if ta_c >= 0 and int(ad.get("Team A", 1)) < 0:
                h_is_rtl |= hm & (h_team_codes == ta_c)
            if tb_c >= 0 and int(ad.get("Team B", -1)) < 0:
                h_is_rtl |= hm & (h_team_codes == tb_c)

        # Top-oyuncu mesafesi ve kontrol oranı (np.searchsorted + np.bincount)
        b_mask = (
            (cls_s == config.BALL_CLASS_NAME).fillna(False).to_numpy(dtype=bool)
            & self.df[x_col].notna().to_numpy(dtype=bool)
            & self.df[y_col].notna().to_numpy(dtype=bool)
        )
        has_fidx_col = "frame_idx" in self.df.columns
        if has_fidx_col:
            b_mask &= self.df["frame_idx"].notna().to_numpy(dtype=bool)
        b_idx = np.flatnonzero(b_mask)

        ball_frames = np.empty(0, dtype=np.int64)
        ball_xs = np.empty(0, dtype=np.float64)
        ball_ys = np.empty(0, dtype=np.float64)
        if len(b_idx) > 0 and has_fidx_col:
            b_fids = self.df["frame_idx"].iloc[b_idx].to_numpy(dtype=np.int64)
            b_xs_raw = self.df[x_col].iloc[b_idx].to_numpy(dtype=np.float64)
            b_ys_raw = self.df[y_col].iloc[b_idx].to_numpy(dtype=np.float64)
            if "is_predicted" in self.df.columns:
                b_pred = _bool_mask(self.df["is_predicted"].iloc[b_idx], na_value=False).astype(np.uint8)
                b_ord = np.lexsort((b_pred, b_fids))
                b_fids = b_fids[b_ord]
                b_xs_raw = b_xs_raw[b_ord]
                b_ys_raw = b_ys_raw[b_ord]
                b_keep = np.empty(len(b_fids), dtype=bool)
                b_keep[0] = True
                b_keep[1:] = b_fids[1:] != b_fids[:-1]
            else:
                b_ord = np.argsort(b_fids, kind="mergesort")
                b_fids = b_fids[b_ord]
                b_xs_raw = b_xs_raw[b_ord]
                b_ys_raw = b_ys_raw[b_ord]
                b_keep = np.empty(len(b_fids), dtype=bool)
                b_keep[-1] = True
                b_keep[:-1] = b_fids[:-1] != b_fids[1:]
            ball_frames = b_fids[b_keep]
            ball_xs = b_xs_raw[b_keep]
            ball_ys = b_ys_raw[b_keep]

        max_ctrl_dist = (
            float(config.POSSESSION_MAX_PLAYER_BALL_DIST_M)
            if self.is_calibrated
            else float(config.POSSESSION_MAX_PLAYER_BALL_DIST_PX)
        )

        ball_dist_sum = np.zeros(n_tracks, dtype=np.float64)
        ball_dist_cnt = np.zeros(n_tracks, dtype=np.int64)
        ball_ctrl_arr = np.zeros(n_tracks, dtype=np.int64)
        total_ball_frames = max(1, len(ball_frames))

        if len(ball_frames) > 0 and has_fidx_col:
            h_fids = self.df["frame_idx"].iloc[h_idx].to_numpy(dtype=np.float64)
            nr_mask = (~h_is_ref_cls) & h_team_not_ref & np.isfinite(h_fids)
            if nr_mask.any():
                nr_fids = h_fids[nr_mask].astype(np.int64)
                nr_xs = h_xs[nr_mask]
                nr_ys = h_ys[nr_mask]
                nr_tids = h_tid_codes[nr_mask]

                b_pos = np.searchsorted(ball_frames, nr_fids)
                b_pos_safe = np.minimum(b_pos, len(ball_frames) - 1)
                matched = (b_pos < len(ball_frames)) & (ball_frames[b_pos_safe] == nr_fids)
                if matched.any():
                    m_fids = nr_fids[matched]
                    m_bidx = b_pos_safe[matched]
                    m_xs = nr_xs[matched]
                    m_ys = nr_ys[matched]
                    m_tids = nr_tids[matched]

                    m_dists = np.hypot(m_xs - ball_xs[m_bidx], m_ys - ball_ys[m_bidx])
                    ball_dist_sum = np.bincount(m_tids, weights=m_dists, minlength=n_tracks)
                    ball_dist_cnt = np.bincount(m_tids, minlength=n_tracks)

                    m_orig_pos = np.arange(len(m_fids), dtype=np.int64)
                    # Düzenli 90-dk matrislerinde (her karede sabit k oyuncu) O(N) reshape ile min bul
                    n_m = len(m_fids)
                    n_bf = len(ball_frames)
                    if n_bf > 0 and n_m % n_bf == 0 and np.all(m_fids[:: n_m // n_bf] == ball_frames):
                        k_p = n_m // n_bf
                        d_mat = m_dists.reshape(n_bf, k_p)
                        min_col = np.argmin(d_mat, axis=1)
                        min_rows = np.arange(n_bf, dtype=np.int64) * k_p + min_col
                    else:
                        m_ord = np.lexsort((m_orig_pos, m_dists, m_fids))
                        f_sorted = m_fids[m_ord]
                        is_first_f = np.empty(len(f_sorted), dtype=bool)
                        is_first_f[0] = True
                        is_first_f[1:] = f_sorted[1:] != f_sorted[:-1]
                        min_rows = m_ord[is_first_f]
                    ctrl_rows = min_rows[m_dists[min_rows] <= max_ctrl_dist]
                    if len(ctrl_rows) > 0:
                        ball_ctrl_arr = np.bincount(m_tids[ctrl_rows], minlength=n_tracks)

        excluded_tracks: List[Dict[str, Any]] = []
        player_rows: List[Dict[str, Any]] = []

        # Track bazlı gruplama için tek O(N log N) sıralama
        h_ord = np.argsort(h_tid_codes, kind="mergesort")
        hc_s = h_tid_codes[h_ord]
        hxs_s = h_xs[h_ord]
        hys_s = h_ys[h_ord]
        href_s = h_is_ref_cls[h_ord]
        htm_s = h_team_codes[h_ord]
        hrtl_s = h_is_rtl[h_ord]

        h_brk = np.flatnonzero(hc_s[1:] != hc_s[:-1]) + 1
        min_bx, max_bx, min_by, max_by = self.bounds
        inv_span_x = 1.0 / max(1e-6, max_bx - min_bx)
        inv_span_y = 1.0 / max(1e-6, max_by - min_by)

        for st, en in zip(np.r_[0, h_brk], np.r_[h_brk, len(hc_s)]):
            st_i = int(st)
            en_i = int(en)
            t_code = int(hc_s[st_i])
            if t_code < 0:
                continue
            tid = tid_strs[t_code]
            n_obs = en_i - st_i

            tm_slice = htm_s[st_i:en_i]
            valid_tm = tm_slice[tm_slice >= 0]
            if len(valid_tm) == 0:
                resolved_team = "Unknown"
            elif len(valid_tm) == 1:
                resolved_team = h_team_strs[int(valid_tm[0])]
            else:
                tm_cnts = np.bincount(valid_tm)
                max_tc = int(tm_cnts.max())
                tied = np.flatnonzero(tm_cnts == max_tc)
                if len(tied) == 1:
                    resolved_team = h_team_strs[int(tied[0])]
                else:
                    tied_set = set(tied.tolist())
                    resolved_team = "Unknown"
                    for c_val in valid_tm:
                        if int(c_val) in tied_set:
                            resolved_team = h_team_strs[int(c_val)]
                            break

            is_ref = bool(href_s[st_i:en_i].any() or resolved_team == "Referee")
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

            xs = hxs_s[st_i:en_i]
            ys = hys_s[st_i:en_i]
            nxs = np.clip((xs - min_bx) * inv_span_x, 0.0, 1.0)
            nys = np.clip((ys - min_by) * inv_span_y, 0.0, 1.0)
            rtl_slice = hrtl_s[st_i:en_i]
            nxs_eff = np.where(rtl_slice, 1.0 - nxs, nxs)
            nys_eff = np.where(rtl_slice, 1.0 - nys, nys)

            span_nx = float(nxs.max() - nxs.min())
            span_ny = float(nys.max() - nys.min())
            if len(nxs) >= 3 and span_nx > 1e-12 and span_ny > 1e-12:
                try:
                    hull = ConvexHull(np.column_stack((nxs, nys)))
                    cov_area = float(hull.volume)  # 2B'de volume == alan
                except Exception:
                    cov_area = span_nx * span_ny
            else:
                cov_area = span_nx * span_ny

            p_spds = speed_by_track.get(tid)
            if p_spds is not None and p_spds.size > 0:
                mean_spd = float(np.mean(p_spds))
                p95_spd = float(np.percentile(p_spds, 95))
                hsr_ratio = float(np.mean(p_spds >= hsr_thresh))
            else:
                mean_spd, p95_spd, hsr_ratio = 0.0, 0.0, 0.0

            b_cnt = int(ball_dist_cnt[t_code])
            mean_b_dist = float(ball_dist_sum[t_code] / b_cnt) if b_cnt > 0 else np.nan
            ctrl_pct = (int(ball_ctrl_arr[t_code]) / total_ball_frames) * 100.0

            player_rows.append({
                "track_id": tid,
                "team": resolved_team,
                "observed_frames": n_obs,
                "mean_x": round(float(np.mean(xs)), 2),
                "mean_y": round(float(np.mean(ys)), 2),
                "norm_mean_x": round(float(np.mean(nxs_eff)), 4),
                "norm_mean_y": round(float(np.mean(nys_eff)), 4),
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
                if n_players > 2000:
                    sil = float(silhouette_score(X_scaled, labels, sample_size=2000, random_state=self.random_state))
                else:
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

        for cid, cgrp in profiles_df.groupby("cluster_id", observed=True):
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
        spike_seq = spike_anomalies if isinstance(spike_anomalies, _VectorizedSpikeAnomalies) else None
        anomalies_list: List[Dict[str, Any]] = [] if spike_seq is not None else list(spike_anomalies)

        if speed_df.empty:
            early_anom_df = (
                spike_seq.to_dataframe(anom_cols)
                if spike_seq is not None
                else pd.DataFrame(anomalies_list, columns=anom_cols)
            )
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

        win_denom = max(0.5, win_sec)
        s_ts = speed_df["timestamp_sec"].to_numpy(dtype=np.float64)
        s_fidx = speed_df["frame_idx"].to_numpy(dtype=np.int64)
        s_spd = speed_df["speed"].to_numpy(dtype=np.float64)
        s_nx = speed_df["norm_x"].to_numpy(dtype=np.float64)
        s_ny = speed_df["norm_y"].to_numpy(dtype=np.float64)
        s_win = np.maximum(0, np.floor((s_ts - t_min) / win_denom).astype(np.int64))
        max_w_idx = max(
            int(s_win.max()),
            int(math.floor(max(0.0, t_max - t_min - 1e-6) / win_denom)),
        )
        n_wins = max_w_idx + 1

        s_tid_codes, s_tid_strs = _encode_sorted_track_ids(speed_df["track_id"])
        s_team_raw_codes, s_team_strs = _encode_clean_strings(speed_df["team"])

        hsr_thresh = (
            float(config.SPEED_ZONE_RUNNING_MAX_KMH)
            if self.is_calibrated
            else float(np.quantile(s_spd, 0.75))
        )
        min_samples = int(config.ML_TEMPO_MIN_SAMPLES_PER_WINDOW)

        team_a_code = s_team_strs.index("Team A") if "Team A" in s_team_strs else -2
        team_b_code = s_team_strs.index("Team B") if "Team B" in s_team_strs else -2
        teams_to_eval = ["ALL"]
        if team_a_code >= 0 and (s_team_raw_codes == team_a_code).any():
            teams_to_eval.append("Team A")
        if team_b_code >= 0 and (s_team_raw_codes == team_b_code).any():
            teams_to_eval.append("Team B")

        window_rows: List[Dict[str, Any]] = []
        for team_label in teams_to_eval:
            if team_label == "ALL":
                t_win = s_win
                t_tid = s_tid_codes
                t_spd = s_spd
                t_fidx = s_fidx
            else:
                tc_target = team_a_code if team_label == "Team A" else team_b_code
                t_m = s_team_raw_codes == tc_target
                t_win = s_win[t_m]
                t_tid = s_tid_codes[t_m]
                t_spd = s_spd[t_m]
                t_fidx = s_fidx[t_m]

            if len(t_win) > 0:
                ord_w = np.lexsort((t_tid, t_win))
                w_s = t_win[ord_w]
                tid_s = t_tid[ord_w]
                spd_s = t_spd[ord_w]
                fidx_s = t_fidx[ord_w]

                s_cnt_arr = np.bincount(w_s, minlength=n_wins)
                spd_sum_arr = np.bincount(w_s, weights=spd_s, minlength=n_wins)
                hsr_sum_arr = np.bincount(w_s, weights=(spd_s >= hsr_thresh), minlength=n_wins)
                is_new_tid = np.empty(len(tid_s), dtype=bool)
                is_new_tid[0] = True
                is_new_tid[1:] = (tid_s[1:] != tid_s[:-1]) | (w_s[1:] != w_s[:-1])
                act_p_arr = np.bincount(w_s, weights=is_new_tid, minlength=n_wins).astype(int)

                w_brk = np.flatnonzero(w_s[1:] != w_s[:-1]) + 1
                w_starts_nz = np.r_[0, w_brk]
                w_ends_nz = np.r_[w_brk, len(w_s)]
                win_slices: List[Optional[Tuple[int, int]]] = [None] * n_wins
                for st_w, en_w in zip(w_starts_nz, w_ends_nz):
                    win_slices[int(w_s[st_w])] = (int(st_w), int(en_w))
            else:
                s_cnt_arr = np.zeros(n_wins, dtype=np.int64)
                spd_sum_arr = np.zeros(n_wins, dtype=np.float64)
                hsr_sum_arr = np.zeros(n_wins, dtype=np.float64)
                act_p_arr = np.zeros(n_wins, dtype=int)
                spd_s = np.empty(0, dtype=np.float64)
                fidx_s = np.empty(0, dtype=np.int64)
                win_slices = [None] * n_wins

            prior_covered_means: List[float] = []
            for w_idx in range(n_wins):
                w_start = t_min + w_idx * win_sec
                w_end = w_start + win_sec
                s_cnt = int(s_cnt_arr[w_idx])
                act_p = int(act_p_arr[w_idx])
                cov_ok = bool(s_cnt >= min_samples and act_p >= 1)
                low_cov = not cov_ok

                sl = win_slices[w_idx]
                if sl is not None:
                    st_w, en_w = sl
                    m_spd = float(spd_sum_arr[w_idx] / s_cnt)
                    p90_spd = float(np.percentile(spd_s[st_w:en_w], 90))
                    hsr_r = float(hsr_sum_arr[w_idx] / s_cnt)
                    min_fidx = int(fidx_s[st_w:en_w].min())
                else:
                    m_spd, p90_spd, hsr_r = 0.0, 0.0, 0.0
                    min_fidx = None

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
                        "frame_idx": min_fidx,
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
                        "frame_idx": min_fidx,
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
        unknown_tm_code = next((i for i, s in enumerate(s_team_strs) if s.lower() == "unknown"), -1)
        f_ord = np.lexsort((s_ts, s_tid_codes))
        ft_tid = s_tid_codes[f_ord]
        ft_ts = s_ts[f_ord]
        ft_spd = s_spd[f_ord]
        ft_fidx = s_fidx[f_ord]
        ft_tm = s_team_raw_codes[f_ord]

        ft_brk = np.flatnonzero(ft_tid[1:] != ft_tid[:-1]) + 1
        min_dur_fatigue = max(2.0, win_sec)

        for st_f, en_f in zip(np.r_[0, ft_brk], np.r_[ft_brk, len(ft_tid)]):
            st_i = int(st_f)
            en_i = int(en_f)
            if en_i - st_i < 12:
                continue
            ts_arr = ft_ts[st_i:en_i]
            if float(ts_arr.max() - ts_arr.min()) < min_dur_fatigue:
                continue

            spd_arr = ft_spd[st_i:en_i]
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
            tm_sl = ft_tm[st_i:en_i]
            known_tm = tm_sl[(tm_sl >= 0) & (tm_sl != unknown_tm_code)]
            if len(known_tm) == 0:
                p_team = "Unknown"
            elif len(known_tm) == 1:
                p_team = s_team_strs[int(known_tm[0])]
            else:
                k_cnts = np.bincount(known_tm)
                max_k = int(k_cnts.max())
                tied_k = np.flatnonzero(k_cnts == max_k)
                if len(tied_k) == 1:
                    p_team = s_team_strs[int(tied_k[0])]
                else:
                    tied_set = set(tied_k.tolist())
                    p_team = "Unknown"
                    for c_val in known_tm:
                        if int(c_val) in tied_set:
                            p_team = s_team_strs[int(c_val)]
                            break

            tid = s_tid_strs[int(ft_tid[st_i])]
            fatigue_trends.append({
                "track_id": tid,
                "team": p_team,
                "samples": en_i - st_i,
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
                    "frame_idx": int(ft_fidx[st_i + half_idx]),
                    "timestamp_sec": round(float(ts_arr[half_idx]), 2),
                    "track_id": tid,
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
        x_col, y_col = self._get_xy_columns()
        used_obs_centroids = False
        dist_from_centroid = np.empty(len(s_fidx), dtype=np.float64)

        # Birleşik takım kodlaması ("Unknown" dahil)
        unified_teams = list(s_team_strs)
        if "Unknown" not in unified_teams:
            unified_teams.append("Unknown")
        u_team_to_code = {t: i for i, t in enumerate(unified_teams)}
        s_team_uni = np.array(
            [u_team_to_code[s_team_strs[c]] if c >= 0 else u_team_to_code["Unknown"] for c in s_team_raw_codes],
            dtype=np.int64,
        )

        if (
            not self.df.empty
            and x_col in self.df.columns
            and y_col in self.df.columns
            and "class_name" in self.df.columns
            and "frame_idx" in self.df.columns
            and "track_id" in self.df.columns
        ):
            cls_s = self.df["class_name"]
            human_names = _get_human_class_names()
            valid_bbox_obs = self._get_valid_bbox_mask(self.df).to_numpy(dtype=bool)
            team_not_ref_obs = (
                ~((self.df["team"] == "Referee").fillna(False).to_numpy(dtype=bool))
                if "team" in self.df.columns
                else True
            )
            obs_mask = (
                cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
                & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
                & team_not_ref_obs
                & valid_bbox_obs
                & self.df[x_col].notna().to_numpy(dtype=bool)
                & self.df[y_col].notna().to_numpy(dtype=bool)
                & self.df["frame_idx"].notna().to_numpy(dtype=bool)
                & self.df["track_id"].notna().to_numpy(dtype=bool)
            )
            if "is_predicted" in self.df.columns:
                obs_mask &= ~_bool_mask(self.df["is_predicted"], na_value=False)
            if self.is_calibrated and "on_pitch" in self.df.columns:
                obs_mask &= _bool_mask(self.df["on_pitch"], na_value=True)
            if not self.is_calibrated and obs_mask.any():
                min_x, max_x, min_y, max_y = self.bounds
                x_all = self.df[x_col].to_numpy(dtype=np.float64, copy=False)
                y_all = self.df[y_col].to_numpy(dtype=np.float64, copy=False)
                obs_mask &= (
                    (x_all >= min_x - 30.0)
                    & (x_all <= max_x + 30.0)
                    & (y_all >= min_y - 30.0)
                    & (y_all <= max_y + 30.0)
                )

            obs_idx = np.flatnonzero(obs_mask)
            if len(obs_idx) > 0:
                o_tids, _ = _encode_sorted_track_ids(self.df["track_id"].iloc[obs_idx])
                o_fids = self.df["frame_idx"].iloc[obs_idx].to_numpy(dtype=np.int64)
                o_xs = self.df[x_col].iloc[obs_idx].to_numpy(dtype=np.float64)
                o_ys = self.df[y_col].iloc[obs_idx].to_numpy(dtype=np.float64)

                o_ord = np.lexsort((o_fids, o_tids))
                o_tids = o_tids[o_ord]
                o_fids = o_fids[o_ord]
                o_xs = o_xs[o_ord]
                o_ys = o_ys[o_ord]

                if "team" in self.df.columns:
                    o_tm_codes, o_tm_strs = _encode_clean_strings(self.df["team"].iloc[obs_idx])
                    o_is_gk = (cls_s.iloc[obs_idx] == "goalkeeper").fillna(False).to_numpy(dtype=bool)
                    gk_raw_c = o_tm_strs.index("Goalkeeper") if "Goalkeeper" in o_tm_strs else -2
                    if gk_raw_c >= 0:
                        o_is_gk |= (o_tm_codes == gk_raw_c)
                    if o_is_gk.any():
                        res_codes_all, res_team_list, _ = self._get_resolved_teams_and_halves()
                        o_tm_strs = list(o_tm_strs)
                        o_str_to_c = {s: i for i, s in enumerate(o_tm_strs)}
                        for s in res_team_list:
                            if s not in o_str_to_c:
                                o_str_to_c[s] = len(o_tm_strs)
                                o_tm_strs.append(s)
                        res_remap = np.array([o_str_to_c[s] for s in res_team_list], dtype=np.int32)
                        o_res_slice = res_codes_all[obs_idx]
                        gk_valid = o_is_gk & (o_res_slice >= 0)
                        if gk_valid.any():
                            o_tm_codes = o_tm_codes.copy()
                            o_tm_codes[gk_valid] = res_remap[o_res_slice[gk_valid]]
                    o_tm_codes = o_tm_codes[o_ord]
                    for ts_val in o_tm_strs:
                        if ts_val not in u_team_to_code:
                            u_team_to_code[ts_val] = len(unified_teams)
                            unified_teams.append(ts_val)
                    o_remap = np.array([u_team_to_code[s] for s in o_tm_strs], dtype=np.int64)
                    n_o = len(o_tids)
                    idx_o = np.arange(n_o, dtype=np.int64)
                    is_ot_start = np.empty(n_o, dtype=bool)
                    is_ot_start[0] = True
                    if n_o > 1:
                        is_ot_start[1:] = o_tids[1:] != o_tids[:-1]
                    row_ot_start = np.maximum.accumulate(np.where(is_ot_start, idx_o, 0))
                    last_valid_o = np.maximum.accumulate(np.where(o_tm_codes >= 0, idx_o, -1))
                    has_ot = last_valid_o >= row_ot_start
                    o_resolved_team = np.where(
                        has_ot,
                        o_remap[o_tm_codes[np.maximum(0, last_valid_o)]],
                        u_team_to_code["Unknown"],
                    )
                else:
                    o_resolved_team = np.full(len(obs_idx), u_team_to_code["Unknown"], dtype=np.int64)

                min_bx, max_bx, min_by, max_by = self.bounds
                obs_nx = np.clip((o_xs - min_bx) / max(1e-6, max_bx - min_bx), 0.0, 1.0)
                obs_ny = np.clip((o_ys - min_by) / max(1e-6, max_by - min_by), 0.0, 1.0)

                team_stride = len(unified_teams) + 1
                fid_offset = min(int(o_fids.min()), int(s_fidx.min()))
                obs_keys = (o_fids - fid_offset) * team_stride + o_resolved_team
                obs_cent = pd.DataFrame({"key": obs_keys, "nx": obs_nx, "ny": obs_ny}).groupby("key", sort=True).median()
                cent_keys = obs_cent.index.to_numpy(dtype=np.int64)
                spd_keys = (s_fidx - fid_offset) * team_stride + s_team_uni
                pos = np.searchsorted(cent_keys, spd_keys)
                pos_safe = np.minimum(pos, max(0, len(cent_keys) - 1))
                matched = (len(cent_keys) > 0) & (pos < len(cent_keys)) & (cent_keys[pos_safe] == spd_keys)
                if np.all(matched):
                    cx = obs_cent["nx"].to_numpy(dtype=np.float64)[pos_safe]
                    cy = obs_cent["ny"].to_numpy(dtype=np.float64)[pos_safe]
                    dist_from_centroid = np.hypot(s_nx - cx, s_ny - cy)
                    used_obs_centroids = True

        if not used_obs_centroids:
            team_stride = len(unified_teams) + 1
            fid_offset = int(s_fidx.min())
            spd_keys = (s_fidx - fid_offset) * team_stride + s_team_uni
            spd_cent = pd.DataFrame({"key": spd_keys, "nx": s_nx, "ny": s_ny}).groupby("key", sort=True).median()
            cent_keys = spd_cent.index.to_numpy(dtype=np.int64)
            pos = np.searchsorted(cent_keys, spd_keys)
            cx = spd_cent["nx"].to_numpy(dtype=np.float64)[pos]
            cy = spd_cent["ny"].to_numpy(dtype=np.float64)[pos]
            dist_from_centroid = np.hypot(s_nx - cx, s_ny - cy)

        # Oyuncu-pencere özetleri üzerinden anomali değerlendirmesi (gürültüyü azaltır)
        pw_in = pd.DataFrame({
            "window_idx": s_win,
            "track_id": speed_df["track_id"].to_numpy(),
            "team": speed_df["team"].to_numpy(),
            "frame_idx": s_fidx,
            "timestamp_sec": s_ts,
            "norm_x": s_nx,
            "norm_y": s_ny,
            "speed": s_spd,
            "dist_from_team_centroid": dist_from_centroid,
        })
        pw_df = (
            pw_in.groupby(["window_idx", "track_id", "team"], as_index=False, observed=True, sort=True)
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
            dist_vals = pw_df["dist_from_team_centroid"].to_numpy(dtype=float)
            spd_vals = pw_df["speed"].to_numpy(dtype=float)
            dist_mad_z = _robust_mad_zscore(dist_vals)
            spd_mad_z = _robust_mad_zscore(spd_vals)

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
            is_spatial_mad = np.abs(dist_mad_z) >= mad_thresh
            is_speed_mad = np.abs(spd_mad_z) >= mad_thresh
            cond_spatial = is_spatial_mad | (iso_flags & (dist_mad_z >= 2.0))
            cond_move = (~cond_spatial) & (is_speed_mad | iso_flags)

            flagged_indices = np.flatnonzero(cond_spatial | cond_move)
            if len(flagged_indices) > 0:
                pw_fidx = pw_df["frame_idx"].to_numpy(dtype=np.int64)
                pw_ts = pw_df["timestamp_sec"].to_numpy(dtype=np.float64)
                pw_tid = pw_df["track_id"].astype(str).to_numpy()
                pw_tm = pw_df["team"].astype(str).to_numpy()

                for idx in flagged_indices:
                    ii = int(idx)
                    is_iso = bool(iso_flags[ii])
                    tid_s = str(pw_tid[ii])
                    tm_s = str(pw_tm[ii])
                    ts_v = float(pw_ts[ii])
                    f_v = int(pw_fidx[ii])
                    if cond_spatial[ii]:
                        dmz = float(dist_mad_z[ii])
                        d_cent = float(dist_vals[ii])
                        anomalies_list.append({
                            "anomaly_type": "spatial_off_formation",
                            "severity": "high" if abs(dmz) >= mad_thresh else "medium",
                            "frame_idx": f_v,
                            "timestamp_sec": round(ts_v, 2),
                            "track_id": tid_s,
                            "team": tm_s,
                            "metric_value": round(d_cent, 4),
                            "threshold": mad_thresh,
                            "description_tr": (
                                f"Oyuncu {tid_s} ({tm_s}) {ts_v:.1f}s civarında "
                                f"formasyon dışı konumlanma sergiledi (takım merkezine uzaklık={d_cent:.3f}, "
                                f"MAD z={dmz:.2f}, IsolationForest={'Aykırı' if is_iso else 'Normal'})."
                            ),
                        })
                    else:
                        smz = float(spd_mad_z[ii])
                        sp_v = float(spd_vals[ii])
                        anomalies_list.append({
                            "anomaly_type": "movement_anomaly",
                            "severity": "medium",
                            "frame_idx": f_v,
                            "timestamp_sec": round(ts_v, 2),
                            "track_id": tid_s,
                            "team": tm_s,
                            "metric_value": round(sp_v, 2),
                            "threshold": mad_thresh,
                            "description_tr": (
                                f"Oyuncu {tid_s} ({tm_s}) {ts_v:.1f}s civarında "
                                f"olağandışı hareket yoğunluğu gösterdi (hız={sp_v:.1f}, "
                                f"MAD z={smz:.2f}, IsolationForest={'Aykırı' if is_iso else 'Normal'})."
                            ),
                        })

        other_anom_df = pd.DataFrame(anomalies_list, columns=anom_cols)
        if spike_seq is not None:
            spike_df = spike_seq.to_dataframe(anom_cols)
            anomalies_df = (
                pd.concat([spike_df, other_anom_df], ignore_index=True)
                if not other_anom_df.empty
                else spike_df
            )
        else:
            anomalies_df = other_anom_df
        if not anomalies_df.empty:
            anomalies_df = anomalies_df.sort_values(
                ["timestamp_sec", "severity"], ascending=[True, True], kind="mergesort"
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

        for cid, cgrp in profiles_df.groupby("cluster_id", observed=True):
            cid_int = int(cid)
            role = str(cgrp["tactical_role_tr"].iloc[0])
            col = palette[cid_int % len(palette)]
            for team_name, tgrp in cgrp.groupby("team", observed=True):
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
                for tid_v, px_v, py_v in zip(tgrp["track_id"], tgrp["pca_1"], tgrp["pca_2"]):
                    ax.annotate(
                        f"ID:{tid_v}",
                        (px_v, py_v),
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
        for tname, tgrp in windows_df.groupby("team", observed=True):
            tgrp = tgrp.sort_values("window_start_sec")
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
                if not low_df.empty:
                    l_starts = low_df["window_start_sec"].to_numpy(dtype=float)
                    l_ends = low_df["window_end_sec"].to_numpy(dtype=float)
                    cur_s, cur_e = float(l_starts[0]), float(l_ends[0])
                    for ns, ne in zip(l_starts[1:], l_ends[1:]):
                        if float(ns) <= cur_e + 1e-6:
                            cur_e = max(cur_e, float(ne))
                        else:
                            ax1.axvspan(cur_s, cur_e, color="#cccccc", alpha=0.35)
                            cur_s, cur_e = float(ns), float(ne)
                    ax1.axvspan(cur_s, cur_e, color="#cccccc", alpha=0.35)

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
            atype_s = anomalies_df["anomaly_type"]
            sev_s = anomalies_df["severity"]
            ts_all = anomalies_df["timestamp_sec"].to_numpy(dtype=np.float64)
            unique_types = sorted(atype_s.unique())
            unique_sevs = sorted(sev_s.unique())
            type_map = {t: idx for idx, t in enumerate(unique_types)}
            sev_colors = {"high": "#d62728", "medium": "#ff7f0e", "info": "#7f7f7f"}
            for atype in unique_types:
                y_pos = type_map[atype]
                m_type = (atype_s == atype).to_numpy(dtype=bool)
                for sev in unique_sevs:
                    m_pair = m_type & (sev_s == sev).to_numpy(dtype=bool)
                    if not m_pair.any():
                        continue
                    ts_vals = ts_all[m_pair]
                    if len(ts_vals) > 5000:
                        ts_vals = ts_vals[np.isfinite(ts_vals)]
                        if len(ts_vals) > 0:
                            t_span = float(np.max(ts_vals) - np.min(ts_vals))
                            step = max(0.05, t_span / 2000.0)
                            ts_vals = np.unique(np.round(ts_vals / step) * step)
                    ax2.scatter(
                        ts_vals,
                        np.full(len(ts_vals), y_pos, dtype=float),
                        color=sev_colors.get(str(sev), "#1f77b4"),
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
        halves_list = xt.get("attack_directions_by_half") or []
        if halves_list:
            lines.append(f"- **Tespit Edilen Devre Sayısı:** {xt.get('halves_detected', len(halves_list))}")
            for hinfo in halves_list:
                ad_map = hinfo.get("attack_directions") or {}
                ta_dir_tr = "Soldan Sağa (+x)" if ad_map.get("Team A") == "+x" else "Sağdan Sola (-x)"
                tb_dir_tr = "Soldan Sağa (+x)" if ad_map.get("Team B") == "+x" else "Sağdan Sola (-x)"
                lines.append(
                    f"  - **{hinfo.get('period', 1)}. Yarı (Kare {hinfo.get('start_frame', 0)}..{hinfo.get('end_frame', 0)}):** "
                    f"Team A → {ta_dir_tr} | Team B → {tb_dir_tr}"
                )
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
