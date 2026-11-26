"""
Mevcut durumun sözleşme testleri: doğrudan testi olmayan ortak yardımcılar,
iki analizörün paylaştığı pas kuralları ve API iş süresi ayarı.
Manuel kabul senaryoları için TEST_SENARYOLARI.md'ye bakın.
"""
import math
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.analytics_common import (
    _bool_mask,
    _clean_str,
    _compact_bool_series,
    _encode_clean_strings,
    _get_human_class_names,
)
from modules.pitch_orientation import (
    compute_pixel_scale_px_per_m,
    filter_ball_outliers,
    has_discontinuity_between,
    sandwich_spell_keep_indices,
)


# ----------------------------------------------------------------------
# analytics_common
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "val, expected",
    [
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (pd.NA, None),
        ("", None),
        ("  ", None),
        ("nan", None),
        ("None", None),
        ("NULL", None),
        ("  Team A ", "Team A"),
        (7, "7"),
        (7.0, "7.0"),
    ],
)
def test_clean_str_normalizes_missing_values(val, expected):
    assert _clean_str(val) == expected


def test_human_class_names_include_legacy_person():
    names = _get_human_class_names()
    assert "person" in names
    assert set(getattr(config, "HUMAN_CLASS_NAMES", ())) <= names


@pytest.mark.parametrize(
    "series, na_value, expected",
    [
        (pd.Series([True, False, None], dtype="boolean"), False, [True, False, False]),
        (pd.Series([True, False, None], dtype="boolean"), True, [True, False, True]),
        (pd.Series([1.0, 0.0, np.nan], dtype=np.float32), True, [True, False, True]),
        (pd.Series([2, 0, 1]), False, [True, False, True]),
        (pd.Series(["true", "False", "yes", "no", "1", "0", "maybe", None]), False,
         [True, False, True, False, True, False, False, False]),
        (pd.Series(["True", None], dtype="category"), True, [True, True]),
    ],
)
def test_bool_mask_handles_all_dtypes(series, na_value, expected):
    out = _bool_mask(series, na_value=na_value)
    assert out.dtype == bool
    assert out.tolist() == expected


def test_bool_mask_empty_series():
    assert _bool_mask(pd.Series([], dtype=object), na_value=True).shape == (0,)


def test_compact_bool_series_keeps_missing_as_nan():
    full = _compact_bool_series(pd.Series(["true", "false"]))
    assert full.dtype == bool and full.tolist() == [True, False]

    with_na = _compact_bool_series(pd.Series([True, None], dtype="boolean"))
    assert with_na.dtype == np.float32
    assert with_na.iloc[0] == 1.0 and math.isnan(with_na.iloc[1])

    numeric = _compact_bool_series(pd.Series([0, 1, 1]))
    assert numeric.dtype == bool and numeric.tolist() == [False, True, True]


def test_encode_clean_strings_merges_equivalent_values():
    codes, uniques = _encode_clean_strings(pd.Series(["Team A", " Team A", None, "nan", "Team B"]))
    assert uniques == ["Team A", "Team B"]
    assert codes.tolist() == [0, 0, -1, -1, 1]

    cat_codes, cat_uniques = _encode_clean_strings(pd.Series(["x", None, "y"], dtype="category"))
    assert cat_uniques == ["x", "y"] and cat_codes.tolist() == [0, -1, 1]

    empty_codes, empty_uniques = _encode_clean_strings(pd.Series([], dtype=object))
    assert len(empty_codes) == 0 and empty_uniques == []


# ----------------------------------------------------------------------
# pitch_orientation: ortak yardımcılar
# ----------------------------------------------------------------------
def test_pixel_scale_uses_pitch_span_and_safe_default():
    assert compute_pixel_scale_px_per_m(None) == 15.0
    assert compute_pixel_scale_px_per_m((0.0, 10.0, 0.0, 10.0)) == 15.0  # dejenere açıklık
    assert compute_pixel_scale_px_per_m((0.0, 1050.0, 0.0, 680.0)) == pytest.approx(10.0)
    assert compute_pixel_scale_px_per_m((0.0, 100.0, 0.0, 50.0)) == 3.0  # alt sınır


def test_has_discontinuity_between_overlap_rules():
    starts = np.array([10, 50])
    ends = np.array([12, 55])
    assert has_discontinuity_between(5, 11, starts, ends)
    assert has_discontinuity_between(52, 60, starts, ends)
    assert not has_discontinuity_between(13, 49, starts, ends)
    assert not has_discontinuity_between(20, 20, starts, ends)  # boş aralık
    assert not has_discontinuity_between(0, 100, np.array([]), np.array([]))


def _spells(spec):
    """spec: [(start_frame, end_frame, team), ...] -> sandwich_spell_keep_indices argümanları."""
    s = [a for a, _, _ in spec]
    e = [b for _, b, _ in spec]
    t = [c for _, _, c in spec]
    return s, e, t, t


def test_sandwich_rule_drops_short_opponent_spell_inside_pass():
    # A (0-20) -> B 0,2 sn (25-30) -> A (40-80): B atlanır
    s, e, ts, te = _spells([(0, 20, "Team A"), (25, 30, "Team B"), (40, 80, "Team A")])
    assert sandwich_spell_keep_indices(s, e, ts, te, fps=25.0) == [0, 2]


@pytest.mark.parametrize(
    "spec, why",
    [
        ([(0, 20, "Team A"), (25, 50, "Team B"), (60, 80, "Team A")], "rakip periyodu uzun (1 sn)"),
        ([(0, 20, "Team A"), (25, 30, "Team B"), (40, 80, "Team B")], "sonraki periyot rakipte"),
        ([(0, 20, "Team A"), (25, 30, "Team A"), (40, 80, "Team A")], "ortadaki aynı takım"),
        ([(0, 20, None), (25, 30, "Team B"), (40, 80, None)], "takım bilinmiyor"),
        ([(0, 20, "Team A"), (25, 30, "Team B"), (200, 240, "Team A")], "arada pas süresinden uzun boşluk"),
    ],
)
def test_sandwich_rule_keeps_real_spells(spec, why):
    s, e, ts, te = _spells(spec)
    assert sandwich_spell_keep_indices(s, e, ts, te, fps=25.0) == [0, 1, 2], why


def test_sandwich_rule_disabled_and_short_lists(monkeypatch):
    s, e, ts, te = _spells([(0, 20, "Team A"), (25, 30, "Team B"), (40, 80, "Team A")])
    monkeypatch.setattr(config, "PASS_SANDWICH_MAX_SEC", 0.0)
    assert sandwich_spell_keep_indices(s, e, ts, te, fps=25.0) == [0, 1, 2]
    assert sandwich_spell_keep_indices([0, 10], [5, 20], ["Team A"] * 2, ["Team A"] * 2, 25.0) == [0, 1]


def test_filter_ball_outliers_scales_thresholds_in_pixel_mode():
    # Piksel modunda (10 px/m) 20 m'lik sahte sıçrama = 200 px; metre eşikleri ölçeklenmeli
    fps = 50.0
    xs = np.r_[np.full(50, 500.0), np.full(20, 700.0), np.full(50, 505.0)]
    fids = np.arange(len(xs))
    ys = np.full(len(xs), 340.0)
    keep = filter_ball_outliers(fids, xs, ys, None, fps, is_calibrated=False, bounds=(0.0, 1050.0, 0.0, 680.0))
    assert keep[:50].all() and keep[70:].all() and not keep[50:70].any()

    # 5 px (0,5 m) titreme, ölçeklenmiş 8 m sıçrama eşiğinin altında: hiçbir kare atılmaz
    small = np.r_[np.full(50, 500.0), np.full(20, 505.0), np.full(50, 500.0)]
    keep2 = filter_ball_outliers(fids, small, ys, None, fps, is_calibrated=False, bounds=(0.0, 1050.0, 0.0, 680.0))
    assert keep2.all()


# ----------------------------------------------------------------------
# API: iş süresi ayarı create_app'ten kuyruğa ulaşıyor
# ----------------------------------------------------------------------
def test_create_app_passes_job_ttl_and_purges_on_startup(tmp_path):
    from fastapi.testclient import TestClient

    from api.app import create_app

    stale = tmp_path / str(uuid.uuid4())
    stale.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = tmp_path / str(uuid.uuid4())
    fresh.mkdir()

    app = create_app(jobs_dir=tmp_path, api_key="k", jwt_secret="s" * 32, job_ttl_seconds=3600,
                     job_executor=lambda *a: None)
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
    assert not stale.exists() and fresh.exists()


def test_job_ttl_zero_disables_cleanup(tmp_path):
    from api.app import BoundedJobQueue

    stale = tmp_path / str(uuid.uuid4())
    stale.mkdir()
    old = time.time() - 10 ** 6
    os.utime(stale, (old, old))
    q = BoundedJobQueue(jobs_dir=tmp_path, executor=lambda *a: None, job_ttl_sec=0)
    assert q.purge_expired() == []
    assert stale.exists()
