"""
analytics_common.py
-------------------
match_analytics.py ve ml_analyzer.py tarafından ortak kullanılan küçük veri
yardımcıları (önceden iki dosyada birebir kopyaydı). Biri düzeltilip diğeri
unutulursa iki rapor aynı veriden farklı sonuç vermesin diye tek yerde tutulur.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


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
