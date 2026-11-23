"""
match_analytics.py
------------------
ADIM 7: TAKTİK ANALİZ VE ISI HARİTASI MODÜLÜ

Kare-kare takip verisi (DataExporter çıktısı, tracking_data.csv veya .db)
üzerinden futbol odaklı taktiksel metrikler ve görseller üretir:
  - Topa sahip olma yüzdeleri (Team A vs Team B) ve oyuncu kontrol süreleri
  - Pas tespiti, pas isabeti, top kayıpları ve pas ağı (pass network)
  - Takım ortalama pozisyonları, saha yayılımı (width / depth) ve hat mesafeleri
  - OpenCV tabanlı 2D takım ve oyuncu ısı haritaları (heatmap PNG)

Hem kalibre edilmiş saha (metre) koordinatlarıyla (`pitch_x`, `pitch_y`),
hem de kalibrasyon yapılmamış kayıtlarda piksel koordinatlı (`pixel_x`, `pixel_y`)
fallback modunda sorunsuz çalışır.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.pitch_orientation import (
    compute_ball_kinematic_filter,
    compute_pixel_scale_px_per_m,
    has_discontinuity_between,
    resolve_goalkeeper_teams,
    smooth_possession_codes,
)


_ANALYTICS_COLUMNS = (
    "frame_idx",
    "timestamp_sec",
    "track_id",
    "class_name",
    "team",
    "is_predicted",
    "pixel_x",
    "pixel_y",
    "pitch_x",
    "pitch_y",
    "on_pitch",
)
_ANALYTICS_COL_SET = frozenset(_ANALYTICS_COLUMNS)
_CAT_COLS = ("track_id", "class_name", "team")
_FLOAT32_COLS = ("pixel_x", "pixel_y", "pitch_x", "pitch_y")
_BOOL_COLS = ("is_predicted", "on_pitch")


def _get_human_class_names() -> set:
    """M1 4-sınıf (goalkeeper/player/referee) ve eski COCO ('person') uyumlu insan sınıf kümesi."""
    return set(getattr(config, "HUMAN_CLASS_NAMES", ())) | {"person"}


def _clean_str(val: Any) -> Optional[str]:
    """NaN / None / boş değerleri güvenli biçimde None'a normalize eder."""
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
    """NumPy / Pandas tiplerini JSON serileştirilebilir standart Python tiplerine çevirir."""
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
    if isinstance(obj, (str, bytes)):
        return obj.decode("utf-8", errors="replace") if isinstance(obj, bytes) else obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _bool_mask(series: pd.Series, na_value: bool) -> np.ndarray:
    """Nullable boolean, float32, int, object ve category serilerini Kleene hatasız bool NumPy maskesine çevirir."""
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
    """Boolean sütunu bellek dostu bool (non-null) veya float32 (1.0/0.0/NaN) tipine sıkıştırır."""
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


def _compact_analytics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Yüklenen takip DataFrame'ini bellek dostu tiplere (int32/float32/float64/category) indirger."""
    if "frame_idx" in df.columns:
        f_num = pd.to_numeric(df["frame_idx"], errors="coerce")
        if not f_num.isna().any():
            df["frame_idx"] = f_num.astype(np.int32)
        else:
            df["frame_idx"] = f_num.astype(np.float32)

    if "timestamp_sec" in df.columns:
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce").astype(np.float64)

    for col in _FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = _compact_bool_series(df[col])

    for col in _CAT_COLS:
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


class MatchAnalyzer:
    """
    Takip verisinden topa sahip olma, pas ağı, takım şekli ve ısı haritası
    üreten taktik analiz sınıfı.
    """

    _CALIBRATION_NOTICE_TR = "Saha kalibrasyonu olmadığı için konum tabanlı metrikler üretilmedi."

    def __init__(
        self,
        df: pd.DataFrame,
        fps: Optional[float] = None,
        copy: bool = False,
        allow_pixel_fallback: Optional[bool] = None,
    ):
        self.df = pd.DataFrame() if df is None else (df.copy() if copy else df)
        self.fps = float(fps) if fps and fps > 0 else self._infer_fps()
        self.has_pitch_players, self.has_pitch_ball, self.is_calibrated = self._detect_calibration()
        self.allow_pixel_fallback = bool(
            getattr(config, "ALLOW_PIXEL_FALLBACK_METRICS", False)
            if allow_pixel_fallback is None
            else allow_pixel_fallback
        )
        self._orientation_cache: Optional[Tuple[int, bool, np.ndarray, List[str], List[Dict[str, Any]]]] = None
        self._last_kinematic_info: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------------
    def _compute_bounds(self) -> Tuple[float, float, float, float]:
        """Kalibreli modda [0..105, 0..68], piksel modda geçerli piksel koordinat sınırlarını döndürür."""
        if self.is_calibrated:
            return (0.0, float(config.PITCH_LENGTH_M), 0.0, float(config.PITCH_WIDTH_M))
        if self.df.empty or "pixel_x" not in self.df.columns or "pixel_y" not in self.df.columns:
            return (0.0, 1920.0, 0.0, 1080.0)
        mask = (self.df["pixel_x"].notna() & self.df["pixel_y"].notna()).to_numpy(dtype=bool)
        if "is_predicted" in self.df.columns:
            m = mask & (~_bool_mask(self.df["is_predicted"], na_value=False))
            if m.any():
                mask = m
        if not mask.any():
            return (0.0, 1920.0, 0.0, 1080.0)
        px = self.df.loc[mask, "pixel_x"].to_numpy(dtype=np.float64)
        py = self.df.loc[mask, "pixel_y"].to_numpy(dtype=np.float64)
        min_x, max_x = float(px.min()), float(px.max())
        min_y, max_y = float(py.min()), float(py.max())
        if max_x - min_x < 10.0:
            max_x = min_x + 100.0
        if max_y - min_y < 10.0:
            max_y = min_y + 100.0
        return (min_x, max_x, min_y, max_y)

    # ----------------------------------------------------------------
    def _get_resolved_teams_and_halves(
        self,
        use_calibrated: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Girdi DataFrame'ini değiştirmeden kaleci eşlemesini ve devre/hücum yönlerini önbellekli döndürür."""
        cal_flag = self.is_calibrated if use_calibrated is None else bool(use_calibrated)
        n_rows = len(self.df)
        if (
            self._orientation_cache is not None
            and self._orientation_cache[0] == n_rows
            and self._orientation_cache[1] == cal_flag
        ):
            _, _, r_codes, r_list, halves = self._orientation_cache
            return r_codes, r_list, halves

        bounds = (
            (0.0, float(config.PITCH_LENGTH_M), 0.0, float(config.PITCH_WIDTH_M))
            if cal_flag
            else self._compute_bounds()
        )
        r_codes, r_list, halves = resolve_goalkeeper_teams(
            df=self.df,
            fps=self.fps,
            bounds=bounds,
            is_calibrated=cal_flag,
        )
        self._orientation_cache = (n_rows, cal_flag, r_codes, r_list, halves)
        return r_codes, r_list, halves

    # ----------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        fps: Optional[float] = None,
        allow_pixel_fallback: Optional[bool] = None,
    ) -> "MatchAnalyzer":
        """CSV takip dosyasından sütun budamalı ve kompakt tipli MatchAnalyzer oluşturur."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

        try:
            header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            return cls(
                df=pd.DataFrame(columns=list(_ANALYTICS_COLUMNS)),
                fps=fps,
                copy=False,
                allow_pixel_fallback=allow_pixel_fallback,
            )
        usecols = [c for c in header_cols if c in _ANALYTICS_COL_SET]
        usecols_arg = usecols if usecols else None

        dtype_hint = {
            "track_id": "category",
            "class_name": "category",
            "team": "category",
            "timestamp_sec": "float64",
            "pixel_x": "float32",
            "pixel_y": "float32",
            "pitch_x": "float32",
            "pitch_y": "float32",
        }
        try:
            df = pd.read_csv(csv_path, usecols=usecols_arg, dtype=dtype_hint)
        except (ValueError, TypeError):
            df = pd.read_csv(csv_path, usecols=usecols_arg)
        df = _compact_analytics_df(df)
        return cls(df=df, fps=fps, copy=False, allow_pixel_fallback=allow_pixel_fallback)

    # ----------------------------------------------------------------
    @classmethod
    def from_sql(
        cls,
        db_path: str,
        table_name: Optional[str] = None,
        fps: Optional[float] = None,
        allow_pixel_fallback: Optional[bool] = None,
    ) -> "MatchAnalyzer":
        """SQLite veritabanından parçalı (chunked) ve kompakt tipli MatchAnalyzer oluşturur."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite dosyası bulunamadı: {db_path}")
        table = table_name or config.EXPORT_SQL_TABLE_NAME
        safe_table = str(table).replace('"', '""')
        sql_chunksize = max(1, int(getattr(config, "ANALYTICS_SQL_CHUNKSIZE", 250_000)))

        conn = sqlite3.connect(db_path)
        try:
            info_df = pd.read_sql_query(f'PRAGMA table_info("{safe_table}")', conn)
            existing_cols = info_df["name"].tolist() if not info_df.empty else []
            selected_cols = [c for c in _ANALYTICS_COLUMNS if c in existing_cols]
            cols_sql = ", ".join(f'"{c}"' for c in selected_cols) if selected_cols else "*"
            query = f'SELECT {cols_sql} FROM "{safe_table}"'

            chunks: List[pd.DataFrame] = []
            for chunk in pd.read_sql_query(query, conn, chunksize=sql_chunksize):
                chunks.append(_compact_analytics_df(chunk))
        finally:
            conn.close()

        if not chunks:
            df = pd.DataFrame(columns=selected_cols)
        elif len(chunks) == 1:
            df = chunks[0]
        else:
            cat_data = {}
            for col in _CAT_COLS:
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

        return cls(df=df, fps=fps, copy=False, allow_pixel_fallback=allow_pixel_fallback)

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
        """Hem top hem oyuncu satırlarında geçerli saha (metre) koordinatı olup olmadığını kontrol eder."""
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
        has_any_ball = bool(is_ball.any())
        is_calibrated = bool(has_pitch_players and (has_pitch_ball or not has_any_ball))
        return has_pitch_players, has_pitch_ball, is_calibrated

    # ----------------------------------------------------------------
    def compute_possession(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Her karede topa en yakın oyuncuyu belirler, çoğunluk oyuyla titreşimleri
        yumuşatır ve takım/oyuncu topa sahip olma istatistiklerini hesaplar.
        """
        coord_space = "pitch_meters" if self.is_calibrated else "pixel_fallback"
        x_col, y_col = ("pitch_x", "pitch_y") if self.is_calibrated else ("pixel_x", "pixel_y")
        max_dist = (
            float(config.POSSESSION_MAX_PLAYER_BALL_DIST_M)
            if self.is_calibrated
            else float(config.POSSESSION_MAX_PLAYER_BALL_DIST_PX)
        )

        empty_cols = [
            "frame_idx", "timestamp_sec", "ball_x", "ball_y",
            "raw_track_id", "raw_team", "nearest_dist",
            "smoothed_track_id", "smoothed_team",
        ]
        if (
            self.df.empty
            or x_col not in self.df.columns
            or y_col not in self.df.columns
            or "class_name" not in self.df.columns
            or "frame_idx" not in self.df.columns
        ):
            return pd.DataFrame(columns=empty_cols), {
                "mode": coord_space,
                "max_distance_threshold": max_dist,
                "total_frames": 0,
                "ball_detected_frames": 0,
                "controlled_frames": 0,
                "team_a_frames": 0,
                "team_b_frames": 0,
                "team_a_pct": 0.0,
                "team_b_pct": 0.0,
                "player_control": [],
            }

        total_frames = int(self.df["frame_idx"].nunique())

        cls_s = self.df["class_name"]
        f_notna = self.df["frame_idx"].notna().to_numpy(dtype=bool)
        xy_notna = (self.df[x_col].notna() & self.df[y_col].notna()).to_numpy(dtype=bool)

        b_mask = (cls_s == config.BALL_CLASS_NAME).fillna(False).to_numpy(dtype=bool) & xy_notna & f_notna
        b_idx = np.flatnonzero(b_mask)

        if len(b_idx) == 0:
            return pd.DataFrame(columns=empty_cols), {
                "mode": coord_space,
                "max_distance_threshold": max_dist,
                "total_frames": total_frames,
                "ball_detected_frames": 0,
                "controlled_frames": 0,
                "team_a_frames": 0,
                "team_b_frames": 0,
                "team_a_pct": 0.0,
                "team_b_pct": 0.0,
                "player_control": [],
            }

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
        n = len(ball_fids)

        bounds = self._compute_bounds()
        kin = compute_ball_kinematic_filter(
            ball_fids=ball_fids,
            ball_xs=ball_xs,
            ball_ys=ball_ys,
            ball_is_pred=ball_is_pred,
            fps=self.fps,
            is_calibrated=self.is_calibrated,
            bounds=bounds,
        )
        self._last_kinematic_info = kin

        resolved_team_codes_all, resolved_team_list, _ = self._get_resolved_teams_and_halves(
            use_calibrated=self.is_calibrated
        )

        human_names = _get_human_class_names()
        ref_code = resolved_team_list.index("Referee") if "Referee" in resolved_team_list else -2
        team_not_ref = (
            (resolved_team_codes_all != ref_code)
            if ref_code >= 0
            else np.ones(len(self.df), dtype=bool)
        )
        p_mask = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
            & team_not_ref
            & xy_notna
            & f_notna
        )
        if "is_predicted" in self.df.columns:
            p_mask &= ~_bool_mask(self.df["is_predicted"], na_value=False)
        if self.is_calibrated and "on_pitch" in self.df.columns:
            p_mask &= _bool_mask(self.df["on_pitch"], na_value=True)

        p_idx = np.flatnonzero(p_mask)
        if len(p_idx) > 0:
            p_tid_codes, tid_list = (
                _encode_clean_strings(self.df["track_id"].iloc[p_idx])
                if "track_id" in self.df.columns
                else (np.full(len(p_idx), -1, dtype=np.int32), [])
            )
            p_team_codes = resolved_team_codes_all[p_idx]
            team_list = list(resolved_team_list)
            tid_to_team_code = np.full(len(tid_list), -1, dtype=np.int32)
            valid_tt = (p_tid_codes >= 0) & (p_team_codes >= 0)
            if valid_tt.any():
                vt_tids = p_tid_codes[valid_tt]
                vt_teams = p_team_codes[valid_tt]
                _, first_occ = np.unique(vt_tids, return_index=True)
                tid_to_team_code[vt_tids[first_occ]] = vt_teams[first_occ]

            p_fids = self.df["frame_idx"].iloc[p_idx].to_numpy(dtype=np.int64)
            p_xs = self.df[x_col].iloc[p_idx].to_numpy(dtype=np.float64)
            p_ys = self.df[y_col].iloc[p_idx].to_numpy(dtype=np.float64)

            if len(p_fids) > 1 and np.any(p_fids[1:] < p_fids[:-1]):
                p_ord = np.argsort(p_fids, kind="mergesort")
                p_fids = p_fids[p_ord]
                p_xs = p_xs[p_ord]
                p_ys = p_ys[p_ord]
                p_tid_codes = p_tid_codes[p_ord]
                p_team_codes = p_team_codes[p_ord]

            starts = np.searchsorted(p_fids, ball_fids, side="left")
            ends = np.searchsorted(p_fids, ball_fids, side="right")
        else:
            tid_list = []
            team_list = []
            tid_to_team_code = np.empty(0, dtype=np.int32)
            p_xs = np.empty(0, dtype=np.float64)
            p_ys = np.empty(0, dtype=np.float64)
            p_tid_codes = np.empty(0, dtype=np.int32)
            p_team_codes = np.empty(0, dtype=np.int32)
            starts = np.zeros(n, dtype=np.int64)
            ends = np.zeros(n, dtype=np.int64)

        raw_tid_codes = np.full(n, -1, dtype=np.int32)
        raw_team_codes = np.full(n, -1, dtype=np.int32)
        nearest_dists = np.full(n, np.nan, dtype=np.float64)

        active_frames = np.flatnonzero(ends > starts)
        for i in active_frames:
            s = int(starts[i])
            e = int(ends[i])
            dx = p_xs[s:e] - ball_xs[i]
            dy = p_ys[s:e] - ball_ys[i]
            dists_sq = dx * dx + dy * dy
            min_off = int(np.argmin(dists_sq))
            min_d = math.hypot(float(dx[min_off]), float(dy[min_off]))
            nearest_dists[i] = round(min_d, 3)
            if min_d <= max_dist:
                tc = int(p_tid_codes[s + min_off])
                raw_tid_codes[i] = tc
                tmc = int(p_team_codes[s + min_off])
                if tmc < 0 and tc >= 0:
                    tmc = int(tid_to_team_code[tc])
                raw_team_codes[i] = tmc

        # R3: Hava topu / tek karelik imkânsız sıçrama karelerinde kontrolü reddet
        valid_ctrl_mask = kin["valid_ctrl_mask"]
        if len(valid_ctrl_mask) == n:
            raw_tid_codes[~valid_ctrl_mask] = -1
            raw_team_codes[~valid_ctrl_mask] = -1

        # Çoğunluk oyuyla yumuşatma + minimum kontrol süresi filtresi
        win = max(1, int(config.POSSESSION_SMOOTHING_FRAMES))
        smoothed_tid_codes, smoothed_team_codes = smooth_possession_codes(
            raw_tid_codes=raw_tid_codes,
            raw_team_codes=raw_team_codes,
            tid_to_team_code=tid_to_team_code,
            win=win,
            min_control_frames=int(kin["min_control_frames"]),
            preserve_frame_team=False,
        )

        ts_out = np.where(
            np.isfinite(ball_ts),
            np.round(ball_ts, 3),
            np.round(ball_fids / self.fps, 3),
        )
        raw_tids_out = [tid_list[c] if c >= 0 else None for c in raw_tid_codes]
        raw_teams_out = [team_list[c] if c >= 0 else None for c in raw_team_codes]
        sm_tids_out = [tid_list[c] if c >= 0 else None for c in smoothed_tid_codes]
        sm_teams_out = [team_list[c] if c >= 0 else None for c in smoothed_team_codes]

        timeline_df = pd.DataFrame({
            "frame_idx": ball_fids,
            "timestamp_sec": ts_out,
            "ball_x": np.round(ball_xs, 3),
            "ball_y": np.round(ball_ys, 3),
            "raw_track_id": raw_tids_out,
            "raw_team": raw_teams_out,
            "nearest_dist": nearest_dists,
            "smoothed_track_id": sm_tids_out,
            "smoothed_team": sm_teams_out,
        })

        team_a_code = team_list.index("Team A") if "Team A" in team_list else -2
        team_b_code = team_list.index("Team B") if "Team B" in team_list else -2
        controlled_mask = smoothed_tid_codes >= 0
        controlled_frames = int(controlled_mask.sum())
        team_a_frames = int((smoothed_team_codes == team_a_code).sum()) if team_a_code >= 0 else 0
        team_b_frames = int((smoothed_team_codes == team_b_code).sum()) if team_b_code >= 0 else 0
        ab_total = team_a_frames + team_b_frames

        team_a_pct = round((team_a_frames / ab_total) * 100.0, 2) if ab_total > 0 else 0.0
        team_b_pct = round((team_b_frames / ab_total) * 100.0, 2) if ab_total > 0 else 0.0

        player_control: List[Dict[str, Any]] = []
        if controlled_frames > 0 and len(tid_list) > 0:
            ctrl_tids = smoothed_tid_codes[controlled_mask]
            ctrl_teams = smoothed_team_codes[controlled_mask]
            tid_counts = np.bincount(ctrl_tids, minlength=len(tid_list))

            first_team_per_tid = np.full(len(tid_list), -1, dtype=np.int32)
            has_ctrl_team = ctrl_teams >= 0
            if has_ctrl_team.any():
                ct_valid_tids = ctrl_tids[has_ctrl_team]
                ct_valid_teams = ctrl_teams[has_ctrl_team]
                _, first_idx = np.unique(ct_valid_tids, return_index=True)
                first_team_per_tid[ct_valid_tids[first_idx]] = ct_valid_teams[first_idx]

            active_tids = np.flatnonzero(tid_counts > 0)
            for tc in active_tids:
                f_cnt = int(tid_counts[tc])
                tmc = int(first_team_per_tid[tc])
                player_control.append({
                    "track_id": tid_list[tc],
                    "team": team_list[tmc] if tmc >= 0 else None,
                    "frames": f_cnt,
                    "seconds": round(f_cnt / self.fps, 2),
                })
            player_control.sort(key=lambda r: r["frames"], reverse=True)

        stats = {
            "mode": coord_space,
            "max_distance_threshold": max_dist,
            "total_frames": total_frames,
            "ball_detected_frames": int(n),
            "controlled_frames": controlled_frames,
            "team_a_frames": team_a_frames,
            "team_b_frames": team_b_frames,
            "team_a_pct": team_a_pct,
            "team_b_pct": team_b_pct,
            "player_control": player_control,
        }
        return timeline_df, stats

    # ----------------------------------------------------------------
    def detect_passes(self, timeline_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Top kontrolündeki oyuncu geçişlerinden başarılı pasları, top kayıplarını
        ve takım pas ağını (pass network) çıkarır.
        """
        empty_summary = {
            "mode": "pitch_meters" if self.is_calibrated else "pixel_fallback",
            "total_completed_passes": 0,
            "total_turnovers": 0,
            "team_stats": {
                "Team A": {"completed_passes": 0, "turnovers": 0, "attempted_passes": 0, "pass_accuracy_pct": 0.0},
                "Team B": {"completed_passes": 0, "turnovers": 0, "attempted_passes": 0, "pass_accuracy_pct": 0.0},
            },
            "pass_network": [],
            "top_pass_connections": [],
            "events": [],
        }
        if timeline_df is None or timeline_df.empty or "smoothed_track_id" not in timeline_df.columns:
            return empty_summary

        tid_codes, tid_list = _encode_clean_strings(timeline_df["smoothed_track_id"])
        valid = tid_codes >= 0
        if not valid.any():
            return empty_summary

        c_tids = tid_codes[valid]
        team_codes, team_list = (
            _encode_clean_strings(timeline_df["smoothed_team"])
            if "smoothed_team" in timeline_df.columns
            else (np.full(len(timeline_df), -1, dtype=np.int32), [])
        )
        c_teams = team_codes[valid]
        c_fids = timeline_df["frame_idx"].to_numpy(dtype=np.int64)[valid]
        c_bx = timeline_df["ball_x"].to_numpy(dtype=np.float64)[valid]
        c_by = timeline_df["ball_y"].to_numpy(dtype=np.float64)[valid]

        # Ardışık aynı track_id karelerini kontrol periyotlarına (spell) dönüştür
        change_idx = np.flatnonzero(c_tids[1:] != c_tids[:-1]) + 1
        spell_starts = np.r_[0, change_idx]
        spell_ends = np.r_[change_idx - 1, len(c_tids) - 1]

        kin = self._last_kinematic_info
        if kin is not None:
            min_control_frames = int(kin["min_control_frames"])
            min_pass_duration_sec = float(kin["min_pass_duration_sec"])
            max_pass_speed = float(kin["max_pass_speed"])
            disc_starts = kin["discontinuity_starts"]
            disc_ends = kin["discontinuity_ends"]
        else:
            scale = 1.0 if self.is_calibrated else compute_pixel_scale_px_per_m(self._compute_bounds())
            min_control_frames = max(
                int(getattr(config, "POSSESSION_MIN_CONTROL_FRAMES", 2)),
                int(round(float(getattr(config, "POSSESSION_MIN_CONTROL_SEC", 0.08)) * self.fps)),
            )
            min_pass_duration_sec = float(getattr(config, "PASS_MIN_DURATION_SEC", 0.08))
            max_pass_speed = float(getattr(config, "PASS_MAX_SPEED_MPS", 42.0)) * scale
            disc_starts = np.empty(0, dtype=np.int64)
            disc_ends = np.empty(0, dtype=np.int64)

        spells: List[Dict[str, Any]] = []
        for st, en in zip(spell_starts, spell_ends):
            st_i = int(st)
            en_i = int(en)
            if (en_i - st_i + 1) < min_control_frames:
                continue
            tmc = int(c_teams[st_i])
            if tmc < 0 and en_i > st_i:
                sub_t = c_teams[st_i + 1:en_i + 1]
                pos = np.flatnonzero(sub_t >= 0)
                if len(pos) > 0:
                    tmc = int(sub_t[pos[0]])
            spells.append({
                "track_id": tid_list[int(c_tids[st_i])],
                "team": team_list[tmc] if tmc >= 0 else None,
                "start_frame": int(c_fids[st_i]),
                "end_frame": int(c_fids[en_i]),
                "start_x": float(c_bx[st_i]),
                "start_y": float(c_by[st_i]),
                "end_x": float(c_bx[en_i]),
                "end_y": float(c_by[en_i]),
            })

        min_move = (
            float(config.PASS_MIN_BALL_MOVE_M)
            if self.is_calibrated
            else float(config.PASS_MIN_BALL_MOVE_PX)
        )

        events: List[Dict[str, Any]] = []
        edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        team_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"completed": 0, "turnovers": 0})

        for i in range(len(spells) - 1):
            s1 = spells[i]
            s2 = spells[i + 1]
            if s1["track_id"] == s2["track_id"]:
                continue

            gap_sec = (s2["start_frame"] - s1["end_frame"]) / self.fps
            move_dist = math.hypot(s2["start_x"] - s1["end_x"], s2["start_y"] - s1["end_y"])

            if gap_sec > config.PASS_MAX_GAP_SEC or move_dist < min_move:
                continue
            if gap_sec + 1e-6 < min_pass_duration_sec:
                continue
            pass_speed = move_dist / max(gap_sec, 1.0 / self.fps)
            if pass_speed > max_pass_speed:
                continue
            if has_discontinuity_between(s1["end_frame"], s2["start_frame"], disc_starts, disc_ends):
                continue
            if s1["team"] not in ("Team A", "Team B") or s2["team"] not in ("Team A", "Team B"):
                continue

            if s1["team"] == s2["team"]:
                event_type = "completed_pass"
                team_counts[s1["team"]]["completed"] += 1
                edge_counts[(s1["track_id"], s2["track_id"], s1["team"])] += 1
            else:
                event_type = "turnover"
                team_counts[s1["team"]]["turnovers"] += 1

            events.append({
                "event_type": event_type,
                "from_track_id": s1["track_id"],
                "to_track_id": s2["track_id"],
                "from_team": s1["team"],
                "to_team": s2["team"],
                "start_frame": s1["end_frame"],
                "end_frame": s2["start_frame"],
                "duration_sec": round(gap_sec, 2),
                "distance": round(move_dist, 2) if (self.is_calibrated or self.allow_pixel_fallback) else None,
            })

        team_stats: Dict[str, Dict[str, Any]] = {}
        for tname in ("Team A", "Team B"):
            comp = team_counts[tname]["completed"]
            turn = team_counts[tname]["turnovers"]
            att = comp + turn
            acc = round((comp / att) * 100.0, 2) if att > 0 else 0.0
            team_stats[tname] = {
                "completed_passes": comp,
                "turnovers": turn,
                "attempted_passes": att,
                "pass_accuracy_pct": acc,
            }

        pass_network = [
            {
                "from_track_id": f_tid,
                "to_track_id": t_tid,
                "team": t_name,
                "count": cnt,
            }
            for (f_tid, t_tid, t_name), cnt in edge_counts.items()
        ]
        pass_network.sort(key=lambda e: e["count"], reverse=True)

        return {
            "mode": "pitch_meters" if self.is_calibrated else "pixel_fallback",
            "total_completed_passes": sum(1 for e in events if e["event_type"] == "completed_pass"),
            "total_turnovers": sum(1 for e in events if e["event_type"] == "turnover"),
            "team_stats": team_stats,
            "pass_network": pass_network,
            "top_pass_connections": pass_network[:10],
            "events": events,
        }

    # ----------------------------------------------------------------
    def compute_team_shape(self) -> Dict[str, Any]:
        """
        Team A ve Team B için ortalama pozisyon (centroid), saha genişliği/derinliği
        ve hücum yönüne göre normalize edilmiş savunma/hücum hatlarını hesaplar.
        """
        if not self.is_calibrated and not self.allow_pixel_fallback:
            return {"status": "requires_calibration"}

        use_pitch = self.has_pitch_players
        x_col, y_col = ("pitch_x", "pitch_y") if use_pitch else ("pixel_x", "pixel_y")
        mode = "pitch_meters" if use_pitch else "pixel_fallback"

        result: Dict[str, Any] = {"status": "ok", "mode": mode, "teams": {}}
        if (
            self.df.empty
            or x_col not in self.df.columns
            or y_col not in self.df.columns
            or "class_name" not in self.df.columns
            or "team" not in self.df.columns
        ):
            return result

        resolved_team_codes_all, resolved_team_list, halves = self._get_resolved_teams_and_halves(
            use_calibrated=use_pitch
        )
        ta_code = resolved_team_list.index("Team A") if "Team A" in resolved_team_list else -2
        tb_code = resolved_team_list.index("Team B") if "Team B" in resolved_team_list else -2
        ab_mask = np.zeros(len(self.df), dtype=bool)
        if ta_code >= 0:
            ab_mask |= resolved_team_codes_all == ta_code
        if tb_code >= 0:
            ab_mask |= resolved_team_codes_all == tb_code

        cls_s = self.df["class_name"]
        human_names = _get_human_class_names()
        p_mask = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
            & ab_mask
            & self.df[x_col].notna().to_numpy(dtype=bool)
            & self.df[y_col].notna().to_numpy(dtype=bool)
        )
        if use_pitch and "on_pitch" in self.df.columns:
            p_mask &= _bool_mask(self.df["on_pitch"], na_value=True)

        p_idx = np.flatnonzero(p_mask)
        if len(p_idx) == 0:
            result["teams"]["Team A"] = None
            result["teams"]["Team B"] = None
            return result

        xs_all = self.df[x_col].iloc[p_idx].to_numpy(dtype=np.float64)
        ys_all = self.df[y_col].iloc[p_idx].to_numpy(dtype=np.float64)
        fids_all = (
            self.df["frame_idx"].iloc[p_idx].to_numpy(dtype=np.int64)
            if "frame_idx" in self.df.columns
            else np.zeros(len(p_idx), dtype=np.int64)
        )
        team_codes_sub = resolved_team_codes_all[p_idx]
        if "track_id" in self.df.columns:
            tid_codes_all, tid_uniques = pd.factorize(
                self.df["track_id"].iloc[p_idx], sort=False, use_na_sentinel=True
            )
            tid_strs = [str(u) for u in tid_uniques]
        else:
            tid_codes_all = np.full(len(p_idx), -1, dtype=np.int32)
            tid_strs = []

        def _shape_for_subset(sub_mask: np.ndarray, attack_dirs: Dict[str, int]) -> Dict[str, Any]:
            teams_out: Dict[str, Any] = {}
            for tname, tcode in (("Team A", ta_code), ("Team B", tb_code)):
                t_mask = sub_mask & (team_codes_sub == tcode) if tcode >= 0 else np.zeros(len(p_idx), dtype=bool)
                if not t_mask.any():
                    teams_out[tname] = None
                    continue

                xs_t = xs_all[t_mask]
                ys_t = ys_all[t_mask]
                t_codes = tid_codes_all[t_mask]
                valid_tid = t_codes >= 0

                player_positions: List[Dict[str, Any]] = []
                if valid_tid.any():
                    vt_codes = t_codes[valid_tid]
                    vt_xs = xs_t[valid_tid]
                    vt_ys = ys_t[valid_tid]
                    k_len = len(tid_strs)
                    cnts = np.bincount(vt_codes, minlength=k_len)
                    sum_x = np.bincount(vt_codes, weights=vt_xs, minlength=k_len)
                    sum_y = np.bincount(vt_codes, weights=vt_ys, minlength=k_len)
                    active = np.flatnonzero(cnts > 0)
                    for c in active:
                        cnt_c = int(cnts[c])
                        player_positions.append({
                            "track_id": tid_strs[c],
                            "avg_x": round(float(sum_x[c] / cnt_c), 2),
                            "avg_y": round(float(sum_y[c] / cnt_c), 2),
                            "frames": cnt_c,
                        })
                    player_positions.sort(key=lambda p: p["frames"], reverse=True)

                if not player_positions:
                    teams_out[tname] = None
                    continue

                xs = np.array([p["avg_x"] for p in player_positions], dtype=float)
                ys = np.array([p["avg_y"] for p in player_positions], dtype=float)
                adir = int(attack_dirs.get(tname, 1 if tname == "Team A" else -1))
                def_line = float(xs.min()) if adir >= 0 else float(xs.max())
                att_line = float(xs.max()) if adir >= 0 else float(xs.min())

                teams_out[tname] = {
                    "player_count": len(player_positions),
                    "attack_direction": "+x" if adir >= 0 else "-x",
                    "centroid_x": round(float(xs_t.mean()), 2),
                    "centroid_y": round(float(ys_t.mean()), 2),
                    "defensive_line_x": round(def_line, 2),
                    "attacking_line_x": round(att_line, 2),
                    "depth": round(float(xs.max() - xs.min()), 2),
                    "width": round(float(ys.max() - ys.min()), 2),
                    "std_x": round(float(xs_t.std(ddof=0)), 2),
                    "std_y": round(float(ys_t.std(ddof=0)), 2),
                    "player_positions": player_positions,
                }
            return teams_out

        by_half: List[Dict[str, Any]] = []
        for idx_h, h in enumerate(halves):
            hs = int(h["start_frame"])
            he = int(h["end_frame"])
            if len(halves) == 1:
                h_mask = np.ones(len(p_idx), dtype=bool)
            elif idx_h == 0:
                h_mask = fids_all <= he
            elif idx_h == len(halves) - 1:
                h_mask = fids_all >= hs
            else:
                h_mask = (fids_all >= hs) & (fids_all <= he)
            h_dirs = h.get("attack_directions", {"Team A": 1, "Team B": -1})
            by_half.append({
                "period": int(h["period"]),
                "start_frame": hs,
                "end_frame": he,
                "detection_reason": h.get("detection_reason", "initial"),
                "attack_directions": {
                    k: ("+x" if int(v) >= 0 else "-x") for k, v in h_dirs.items()
                },
                "teams": _shape_for_subset(h_mask, h_dirs),
            })

        result["halves_detected"] = len(by_half)
        result["by_half"] = by_half
        if len(by_half) == 1:
            result["teams"] = by_half[0]["teams"]
            result["attack_directions"] = by_half[0]["attack_directions"]
        else:
            # Çok devreli maçta geriye dönük uyumlu üst düzey özet: oyuncu kare toplamları
            # tam maçtan gelirken, savunma/hücum hatları birincil (1. yarı) yönü yansıtır.
            primary_dirs = halves[0].get("attack_directions", {"Team A": 1, "Team B": -1})
            full_teams = _shape_for_subset(np.ones(len(p_idx), dtype=bool), primary_dirs)
            for tname in ("Team A", "Team B"):
                if full_teams.get(tname) is None:
                    continue
                first_h = next(
                    (bh["teams"].get(tname) for bh in by_half if bh["teams"].get(tname) is not None),
                    None,
                )
                if first_h is not None:
                    full_teams[tname]["centroid_x"] = first_h["centroid_x"]
                    full_teams[tname]["centroid_y"] = first_h["centroid_y"]
                    full_teams[tname]["defensive_line_x"] = first_h["defensive_line_x"]
                    full_teams[tname]["attacking_line_x"] = first_h["attacking_line_x"]
                    full_teams[tname]["depth"] = first_h["depth"]
                    full_teams[tname]["width"] = first_h["width"]
                    full_teams[tname]["attack_direction"] = first_h["attack_direction"]
            result["teams"] = full_teams
            result["attack_directions"] = by_half[0]["attack_directions"]

        return result

    # ----------------------------------------------------------------
    @staticmethod
    def _draw_pitch_canvas(
        width_px: int = 840,
        height_px: int = 544,
        margin_px: int = 36,
        draw_lines: bool = True,
        title: str = "",
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """2D saha tuvali oluşturur ve saha iç sınırlarını döndürür."""
        total_w = width_px + 2 * margin_px
        total_h = height_px + 2 * margin_px
        canvas = np.full((total_h, total_w, 3), (45, 115, 30), dtype=np.uint8)

        x0, y0 = margin_px, margin_px
        x1, y1 = margin_px + width_px, margin_px + height_px

        if draw_lines:
            sx = width_px / config.PITCH_LENGTH_M
            sy = height_px / config.PITCH_WIDTH_M

            def to_px(mx: float, my: float) -> Tuple[int, int]:
                return int(round(x0 + mx * sx)), int(round(y0 + my * sy))

            line_col = (240, 240, 240)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), line_col, 2)
            cv2.line(canvas, to_px(52.5, 0), to_px(52.5, 68), line_col, 2)
            cv2.circle(canvas, to_px(52.5, 34), int(round(9.15 * sx)), line_col, 2)
            cv2.circle(canvas, to_px(52.5, 34), 4, line_col, -1)

            # Sol ve sağ ceza sahaları + kale sahaları
            cv2.rectangle(canvas, to_px(0, 13.84), to_px(16.5, 54.16), line_col, 2)
            cv2.rectangle(canvas, to_px(88.5, 13.84), to_px(105.0, 54.16), line_col, 2)
            cv2.rectangle(canvas, to_px(0, 24.84), to_px(5.5, 43.16), line_col, 2)
            cv2.rectangle(canvas, to_px(99.5, 24.84), to_px(105.0, 43.16), line_col, 2)
        else:
            # Kalibresiz piksel alanı için çerçeve ve hafif ızgara
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 220, 200), 2)
            for frac in (0.25, 0.5, 0.75):
                gx = int(round(x0 + width_px * frac))
                gy = int(round(y0 + height_px * frac))
                cv2.line(canvas, (gx, y0), (gx, y1), (70, 135, 55), 1)
                cv2.line(canvas, (x0, gy), (x1, gy), (70, 135, 55), 1)

        if title:
            cv2.putText(
                canvas, title, (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return canvas, x0, y0, width_px, height_px

    # ----------------------------------------------------------------
    def _render_heatmap_image(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        bounds: Tuple[float, float, float, float],
        title: str,
        draw_pitch_lines: bool,
        output_path: str,
    ) -> str:
        """Verilen nokta bulutunu Gaussian yoğunluk haritası olarak PNG'ye kaydeder."""
        canvas, x0, y0, pw, ph = self._draw_pitch_canvas(
            draw_lines=draw_pitch_lines, title=title
        )
        min_x, max_x, min_y, max_y = bounds
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)

        density = np.zeros((ph, pw), dtype=np.float32)
        pxs = np.clip(((xs - min_x) / span_x) * (pw - 1), 0, pw - 1).astype(int)
        pys = np.clip(((ys - min_y) / span_y) * (ph - 1), 0, ph - 1).astype(int)
        np.add.at(density, (pys, pxs), 1.0)

        blurred = cv2.GaussianBlur(density, (0, 0), sigmaX=24, sigmaY=24)
        max_val = float(blurred.max())
        if max_val > 0:
            norm = np.clip((blurred / max_val) * 255.0, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

            roi = canvas[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
            alpha = np.where(norm > 10, (norm.astype(np.float32) / 255.0) * 0.72, 0.0)[..., None]
            blended = roi * (1.0 - alpha) + colored.astype(np.float32) * alpha
            canvas[y0:y0 + ph, x0:x0 + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    # ----------------------------------------------------------------
    def generate_heatmaps(
        self,
        output_dir: Optional[str] = None,
        top_n_players: int = 3,
    ) -> Dict[str, Any]:
        """
        Takım A, Takım B ve en sık takip edilen N oyuncu için 2D ısı haritası PNG'leri üretir.
        """
        if not self.is_calibrated and not self.allow_pixel_fallback:
            return {"status": "requires_calibration"}

        out_dir = output_dir or config.EXPORT_HEATMAP_DIR
        os.makedirs(out_dir, exist_ok=True)

        use_pitch = self.has_pitch_players
        x_col, y_col = ("pitch_x", "pitch_y") if use_pitch else ("pixel_x", "pixel_y")

        saved_paths: Dict[str, str] = {}
        if (
            self.df.empty
            or x_col not in self.df.columns
            or y_col not in self.df.columns
            or "class_name" not in self.df.columns
        ):
            return saved_paths

        resolved_team_codes_all, resolved_team_list, _ = self._get_resolved_teams_and_halves(
            use_calibrated=use_pitch
        )
        cls_s = self.df["class_name"]
        human_names = _get_human_class_names()
        ref_code = resolved_team_list.index("Referee") if "Referee" in resolved_team_list else -2
        team_not_ref = (
            (resolved_team_codes_all != ref_code)
            if ref_code >= 0
            else np.ones(len(self.df), dtype=bool)
        )
        p_mask = (
            cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
            & ~((cls_s == "referee").fillna(False).to_numpy(dtype=bool))
            & team_not_ref
            & self.df[x_col].notna().to_numpy(dtype=bool)
            & self.df[y_col].notna().to_numpy(dtype=bool)
        )
        if use_pitch and "on_pitch" in self.df.columns:
            p_mask &= _bool_mask(self.df["on_pitch"], na_value=True)

        p_idx = np.flatnonzero(p_mask)
        if len(p_idx) == 0:
            return saved_paths

        all_x = self.df[x_col].iloc[p_idx].to_numpy(dtype=np.float64)
        all_y = self.df[y_col].iloc[p_idx].to_numpy(dtype=np.float64)

        if use_pitch:
            bounds = (0.0, float(config.PITCH_LENGTH_M), 0.0, float(config.PITCH_WIDTH_M))
            space_suffix = "Pitch Meters"
        else:
            min_ax = float(all_x.min())
            max_ax = float(all_x.max())
            min_ay = float(all_y.min())
            max_ay = float(all_y.max())
            pad_x = max(50.0, (max_ax - min_ax) * 0.05)
            pad_y = max(50.0, (max_ay - min_ay) * 0.05)
            bounds = (
                max(0.0, min_ax - pad_x),
                max_ax + pad_x,
                max(0.0, min_ay - pad_y),
                max_ay + pad_y,
            )
            space_suffix = "Pixel Space (Uncalibrated)"

        if "team" in self.df.columns:
            p_team_codes = resolved_team_codes_all[p_idx]
            team_list = list(resolved_team_list)
        else:
            p_team_codes = np.full(len(p_idx), -1, dtype=np.int32)
            team_list = []

        for team_label, key, fname in (
            ("Team A", "team_a", "team_a_heatmap.png"),
            ("Team B", "team_b", "team_b_heatmap.png"),
        ):
            if team_label in team_list:
                t_code = team_list.index(team_label)
                tm = p_team_codes == t_code
                if tm.any():
                    out_path = os.path.join(out_dir, fname)
                    self._render_heatmap_image(
                        xs=all_x[tm],
                        ys=all_y[tm],
                        bounds=bounds,
                        title=f"{team_label} Heatmap ({space_suffix})",
                        draw_pitch_lines=use_pitch,
                        output_path=out_path,
                    )
                    saved_paths[key] = out_path

        if top_n_players > 0 and "track_id" in self.df.columns:
            tid_s = self.df["track_id"].iloc[p_idx]
            if isinstance(tid_s.dtype, pd.CategoricalDtype):
                raw_codes = tid_s.cat.codes.to_numpy(dtype=np.int32)
                uniques = tid_s.cat.categories
            else:
                raw_codes, uniques = pd.factorize(tid_s, sort=False, use_na_sentinel=True)
                raw_codes = raw_codes.astype(np.int32, copy=False)

            str_to_code: Dict[str, int] = {}
            tid_strs: List[str] = []
            remap = np.full(len(uniques), -1, dtype=np.int32)
            for i, u in enumerate(uniques):
                s = str(u)
                c = str_to_code.get(s)
                if c is None:
                    c = len(tid_strs)
                    tid_strs.append(s)
                    str_to_code[s] = c
                remap[i] = c

            nan_str_code: Optional[int] = None
            if (raw_codes < 0).any():
                nan_str_code = str_to_code.get("nan")
                if nan_str_code is None:
                    nan_str_code = len(tid_strs)
                    tid_strs.append("nan")
                    str_to_code["nan"] = nan_str_code
            mapped_codes = np.where(raw_codes >= 0, remap[np.maximum(0, raw_codes)], nan_str_code or 0)

            counts = np.bincount(mapped_codes, minlength=len(tid_strs))
            top_order = np.argsort(-counts, kind="mergesort")[:top_n_players]

            for tc in top_order:
                if counts[tc] <= 0:
                    continue
                tid_str = tid_strs[int(tc)]
                pm = mapped_codes == tc
                p_teams = p_team_codes[pm]
                valid_pt = p_teams >= 0
                team_tag = team_list[int(p_teams[valid_pt][0])] if valid_pt.any() else "Unknown"
                out_path = os.path.join(out_dir, f"player_{tid_str}_heatmap.png")
                self._render_heatmap_image(
                    xs=all_x[pm],
                    ys=all_y[pm],
                    bounds=bounds,
                    title=f"Player ID:{tid_str} ({team_tag}) - {space_suffix}",
                    draw_pitch_lines=use_pitch,
                    output_path=out_path,
                )
                saved_paths[f"player_{tid_str}"] = out_path

        return saved_paths

    # ----------------------------------------------------------------
    def export_all(
        self,
        output_dir: Optional[str] = None,
        generate_heatmaps: bool = True,
        suppress_calibration_notice: bool = False,
    ) -> Dict[str, Any]:
        """
        Tüm taktik analizleri çalıştırır ve JSON özeti, top kontrol zaman serisi CSV'si
        ve (kalibre ise) ısı haritası PNG'lerini diske kaydeder.
        """
        out_dir = output_dir or config.EXPORT_DIR
        os.makedirs(out_dir, exist_ok=True)

        timeline_df, possession_stats = self.compute_possession()
        pass_stats = self.detect_passes(timeline_df)
        team_shape = self.compute_team_shape()

        if not self.is_calibrated and not self.allow_pixel_fallback:
            heatmaps: Dict[str, Any] = {"status": "requires_calibration"}
        elif generate_heatmaps:
            heatmaps = self.generate_heatmaps(output_dir=os.path.join(out_dir, "heatmaps"))
        else:
            heatmaps = {}

        possession_csv_path = os.path.join(out_dir, config.EXPORT_POSSESSION_FILENAME)
        timeline_df.to_csv(possession_csv_path, index=False)

        cal_notice = (
            self._CALIBRATION_NOTICE_TR
            if (not self.is_calibrated and not self.allow_pixel_fallback)
            else None
        )
        summary = _to_python_scalar({
            "fps": self.fps,
            "is_calibrated": self.is_calibrated,
            "has_pitch_players": self.has_pitch_players,
            "has_pitch_ball": self.has_pitch_ball,
            "possession": possession_stats,
            "passes": pass_stats,
            "team_shape": team_shape,
            "heatmaps": heatmaps,
            "calibration_notice": cal_notice,
            "exports": {
                "possession_timeline_csv": possession_csv_path,
                "heatmaps": heatmaps,
            },
        })

        summary_json_path = os.path.join(out_dir, config.EXPORT_ANALYTICS_SUMMARY_FILENAME)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["exports"]["analytics_summary_json"] = summary_json_path

        if cal_notice and not suppress_calibration_notice:
            print(f"[MatchAnalyzer] {cal_notice}")
        print(f"[MatchAnalyzer] Taktik analiz özeti kaydedildi: {summary_json_path}")
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Takip verisinden taktik maç analizi ve ısı haritası üretir.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(config.EXPORT_DIR, config.EXPORT_CSV_FILENAME),
        help="Giriş takip dosyası (.csv veya .db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.EXPORT_DIR,
        help="Analiz çıktılarının kaydedileceği klasör",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS değeri (belirtilmezse veriden otomatik çıkarılır)",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Isı haritası PNG dosyalarını üretmeyi atla",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input.lower().endswith(".db"):
        analyzer = MatchAnalyzer.from_sql(args.input, fps=args.fps)
    else:
        analyzer = MatchAnalyzer.from_csv(args.input, fps=args.fps)

    summary = analyzer.export_all(
        output_dir=args.output_dir,
        generate_heatmaps=not args.no_heatmaps,
    )
    poss = summary["possession"]
    passes = summary["passes"]
    print(
        f"\n[Analiz Özeti] Mod: {poss['mode']} | "
        f"Topa Sahip Olma -> Team A: %{poss['team_a_pct']} | Team B: %{poss['team_b_pct']}"
    )
    print(
        f"[Analiz Özeti] Paslar -> Tamamlanan: {passes['total_completed_passes']} | "
        f"Top Kaybı: {passes['total_turnovers']}"
    )
