"""
pitch_orientation.py
--------------------
Faz A1 ortak vektörel saha yönelimi, kaleci eşleştirme, devre arası tespiti
ve top kinematik / hava topu filtreleme yardımcıları.

Hem `MatchAnalyzer` hem de `MLAnalyzer` tarafından kullanılarak tutarlılık ve
O(N) vektörel performans sağlar; girdi DataFrame'lerini kesinlikle değiştirmez.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def _clean_str_local(val: Any) -> Optional[str]:
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


def _encode_clean_strings_local(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
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
        cs = _clean_str_local(u)
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


def _bool_mask_local(series: pd.Series, na_value: bool) -> np.ndarray:
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


# ----------------------------------------------------------------------
# R3: PİKSEL ÖLÇEK VE TOP KİNEMATİK / HAVA TOPU FİLTRESİ
# ----------------------------------------------------------------------
def compute_pixel_scale_px_per_m(
    bounds: Optional[Tuple[float, float, float, float]],
    default_px_per_m: float = 15.0,
) -> float:
    """
    Piksel fallback modunda saha yatay açıklığından piksel/metre oranını türetir.
    Dejenere açıklıklarda (< 50 px) güvenli varsayılan (15.0 px/m) döner.
    """
    if bounds is None or len(bounds) < 2:
        return float(default_px_per_m)
    span_x = float(bounds[1]) - float(bounds[0])
    if not math.isfinite(span_x) or span_x < 50.0:
        return float(default_px_per_m)
    pitch_len = max(1.0, float(getattr(config, "PITCH_LENGTH_M", 105.0)))
    return max(3.0, span_x / pitch_len)


def compute_ball_kinematic_filter(
    ball_fids: np.ndarray,
    ball_xs: np.ndarray,
    ball_ys: np.ndarray,
    ball_is_pred: Optional[np.ndarray],
    fps: float,
    is_calibrated: bool,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    reset_frames: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Top yörüngesi üzerinde vektörel O(N) kinematik denetim uygular:
      - Hava topu (aerial flyby): hızlı geçişte (v > max_ctrl_ball_speed)
        belirgin yön değişimi (>= 25 deg) ve hız düşüşü (>= %35) yoksa
        oyuncu kontrolü reddedilir.
      - Tek karelik ışınlanma sıçramaları (v_in > max_pass_speed ve v_out > max_pass_speed)
        kontrol dışı bırakılır.
      - Donmuş Kalman tahmininden (is_predicted=True -> False) ani sıçrama
        sınırları pas geçişi engelleyici süreksizlik olarak işaretlenir.
      - A2-3: Kalibrasyon sıfırlama karesine (calib_reset) giren adımlar hız
        denetiminde kullanılmaz ve süreksizlik olarak işaretlenir.
    """
    fps_safe = float(fps) if fps and fps > 0 else 25.0
    scale = 1.0 if is_calibrated else compute_pixel_scale_px_per_m(bounds)

    max_ctrl_ball_speed = float(getattr(config, "POSSESSION_MAX_BALL_CONTROL_SPEED_MPS", 15.0)) * scale
    max_pass_speed = float(getattr(config, "PASS_MAX_SPEED_MPS", 42.0)) * scale
    min_deflection_deg = float(getattr(config, "BALL_AERIAL_DIRECTION_CHANGE_MIN_DEG", 25.0))
    min_decel_ratio = float(getattr(config, "BALL_AERIAL_DECEL_MIN_RATIO", 0.35))
    min_ctrl_sec = float(getattr(config, "POSSESSION_MIN_CONTROL_SEC", 0.08))
    min_ctrl_cfg = int(getattr(config, "POSSESSION_MIN_CONTROL_FRAMES", 2))
    min_control_frames = max(min_ctrl_cfg, int(round(min_ctrl_sec * fps_safe)))
    min_pass_duration_sec = float(getattr(config, "PASS_MIN_DURATION_SEC", 0.08))

    n = len(ball_fids)
    if n == 0:
        empty_b = np.empty(0, dtype=bool)
        empty_i = np.empty(0, dtype=np.int64)
        return {
            "valid_ctrl_mask": empty_b,
            "aerial_reject_mask": empty_b,
            "spike_reject_mask": empty_b,
            "discontinuity_starts": empty_i,
            "discontinuity_ends": empty_i,
            "max_ctrl_ball_speed": max_ctrl_ball_speed,
            "max_pass_speed": max_pass_speed,
            "min_control_frames": min_control_frames,
            "min_pass_duration_sec": min_pass_duration_sec,
            "scale_px_per_m": scale,
        }

    pred_arr = (
        np.asarray(ball_is_pred, dtype=np.uint8)
        if ball_is_pred is not None and len(ball_is_pred) == n
        else np.zeros(n, dtype=np.uint8)
    )

    aerial_reject = np.zeros(n, dtype=bool)
    spike_reject = np.zeros(n, dtype=bool)
    disc_starts = np.empty(0, dtype=np.int64)
    disc_ends = np.empty(0, dtype=np.int64)

    if n >= 2:
        df_step = ball_fids[1:] - ball_fids[:-1]
        dt_step = np.maximum(df_step.astype(np.float64) / fps_safe, 1.0 / fps_safe)
        dx_step = ball_xs[1:] - ball_xs[:-1]
        dy_step = ball_ys[1:] - ball_ys[:-1]
        dist_step = np.hypot(dx_step, dy_step)
        speed_step = dist_step / dt_step

        max_vel_gap_frames = max(3, int(round(0.25 * fps_safe)))
        valid_step = (df_step >= 1) & (df_step <= max_vel_gap_frames)
        reset_step = (
            np.isin(ball_fids[1:], np.asarray(reset_frames, dtype=np.int64))
            if reset_frames is not None and len(reset_frames) > 0
            else np.zeros(n - 1, dtype=bool)
        )
        valid_step &= ~reset_step

        # Donmuş Kalman tahmininden (is_predicted=1 -> 0) imkânsız hızlı sıçrama süreksizlikleri.
        # Tahmin pas sırasında pasörün yanında donup kaldığı için hız, donmuş tahminden değil
        # tahmin öncesindeki son GERÇEK tespitten ölçülür; aksi halde her gerçek pas sıçrama sayılır.
        min_snap_dist = float(getattr(config, "PASS_MIN_BALL_MOVE_M", 2.0)) * scale
        pos = np.arange(n)
        last_real = np.maximum.accumulate(np.where(pred_arr == 0, pos, -1))[:-1]
        has_real = last_real >= 0
        lr = np.where(has_real, last_real, 0)
        real_dt = np.maximum((ball_fids[1:] - ball_fids[lr]).astype(np.float64) / fps_safe, 1.0 / fps_safe)
        real_speed = np.where(
            has_real,
            np.hypot(ball_xs[1:] - ball_xs[lr], ball_ys[1:] - ball_ys[lr]) / real_dt,
            speed_step,
        )
        pred_snap_mask = (
            (df_step >= 1)
            & (df_step <= max(4, int(round(0.20 * fps_safe))))
            & (pred_arr[:-1] == 1)
            & (pred_arr[1:] == 0)
            & (speed_step > max_pass_speed)
            & (real_speed > max_pass_speed)
            & (dist_step > min_snap_dist)
            & ~reset_step
        )
        disc_mask = pred_snap_mask | reset_step
        if disc_mask.any():
            disc_starts = ball_fids[:-1][disc_mask]
            disc_ends = ball_fids[1:][disc_mask]

        if n >= 3:
            both_valid = valid_step[:-1] & valid_step[1:]
            v_in = speed_step[:-1]
            v_out = speed_step[1:]
            dx_in = dx_step[:-1]
            dy_in = dy_step[:-1]
            dx_out = dx_step[1:]
            dy_out = dy_step[1:]

            # Tek karelik izole sıçrama: geliş ve gidiş imkânsız hızlı
            spike_mid = both_valid & (v_in > max_pass_speed) & (v_out > max_pass_speed)
            spike_reject[1:-1] = spike_mid

            # Hava topu (aerial flyby): hızlı geliş+gidiş, düşük sapma açısı, düşük yavaşlama
            fast_both = both_valid & (v_in > max_ctrl_ball_speed) & (v_out > max_ctrl_ball_speed)
            if fast_both.any():
                dot = dx_in * dx_out + dy_in * dy_out
                cross = np.abs(dx_in * dy_out - dy_in * dx_out)
                angles_deg = np.degrees(np.arctan2(cross, dot))
                decel_ratio = (v_in - v_out) / np.maximum(v_in, 1e-9)
                aerial_mid = (
                    fast_both
                    & (angles_deg < min_deflection_deg)
                    & (decel_ratio < min_decel_ratio)
                )
                aerial_reject[1:-1] = aerial_mid

    valid_ctrl_mask = ~(aerial_reject | spike_reject)
    return {
        "valid_ctrl_mask": valid_ctrl_mask,
        "aerial_reject_mask": aerial_reject,
        "spike_reject_mask": spike_reject,
        "discontinuity_starts": disc_starts,
        "discontinuity_ends": disc_ends,
        "max_ctrl_ball_speed": max_ctrl_ball_speed,
        "max_pass_speed": max_pass_speed,
        "min_control_frames": min_control_frames,
        "min_pass_duration_sec": min_pass_duration_sec,
        "scale_px_per_m": scale,
    }


def has_discontinuity_between(
    start_frame: int,
    end_frame: int,
    disc_starts: np.ndarray,
    disc_ends: np.ndarray,
) -> bool:
    """[start_frame, end_frame] aralığında top tahmin/ışınlanma süreksizliği olup olmadığını O(log D) kontrol eder."""
    if disc_starts is None or len(disc_starts) == 0 or end_frame <= start_frame:
        return False
    idx = int(np.searchsorted(disc_starts, start_frame, side="left"))
    for k in (idx - 1, idx):
        if 0 <= k < len(disc_starts):
            if disc_starts[k] < end_frame and disc_ends[k] > start_frame:
                return True
    return False


def reject_ball_outlier_segments(
    ball_fids: np.ndarray,
    ball_xs: np.ndarray,
    ball_ys: np.ndarray,
    ball_is_pred: Optional[np.ndarray],
    fps: float,
    max_speed: float,
    max_outlier_sec: float = 1.0,
    min_jump: float = 8.0,
) -> np.ndarray:
    """
    Top dedektörü iki nesne (gerçek top ve sahte bir nokta) arasında gidip gelebiliyor.
    Gerçek tespitler, ardışık adımın hem max_speed'i hem min_jump mesafesini aşmadığı
    parçalara ayrılır (küçük top-ayak titremeleri parça kesmez). İki komşu parça
    arasında fiziksel olarak geçilebiliyorsa aradaki kısa (<= max_outlier_sec) parça sahte
    sayılır. Atılan ya da imkânsız bir sıçramayı örten tahmin/ara değer kareleri de atılır.
    Döndürür: tutulacak satırlar için bool maske.
    """
    n = len(ball_fids)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    fps_safe = float(fps) if fps and fps > 0 else 25.0
    pred = (
        np.asarray(ball_is_pred, dtype=bool)
        if ball_is_pred is not None and len(ball_is_pred) == n
        else np.zeros(n, dtype=bool)
    )
    real = np.flatnonzero(~pred)
    if len(real) < 3:
        return keep

    def speed(i: int, j: int) -> float:
        dt = max((ball_fids[j] - ball_fids[i]) / fps_safe, 1.0 / fps_safe)
        return math.hypot(ball_xs[j] - ball_xs[i], ball_ys[j] - ball_ys[i]) / dt

    segs: List[List[int]] = [[int(real[0]), int(real[0])]]
    for a, b in zip(real[:-1], real[1:]):
        a, b = int(a), int(b)
        if speed(a, b) > max_speed and math.hypot(ball_xs[b] - ball_xs[a], ball_ys[b] - ball_ys[a]) > min_jump:
            segs.append([int(b), int(b)])
        else:
            segs[-1][1] = int(b)

    dropped: List[Tuple[int, int]] = []
    changed = True
    while changed and len(segs) >= 3:
        changed = False
        k = 1
        while k < len(segs) - 1:
            s, e = segs[k]
            short = (ball_fids[e] - ball_fids[s]) / fps_safe <= max_outlier_sec
            if short and speed(segs[k - 1][1], segs[k + 1][0]) <= max_speed:
                dropped.append((s, e))
                segs[k - 1][1] = segs[k + 1][1]
                del segs[k:k + 2]
                changed = True
            else:
                k += 1

    for s, e in dropped:
        keep[s:e + 1] &= pred[s:e + 1]  # parçanın gerçek tespitleri
    kept_real = np.flatnonzero(~pred & keep)
    if len(kept_real) == 0:
        return keep
    dropped_starts = (
        np.sort(np.fromiter((s for s, _ in dropped), dtype=np.int64, count=len(dropped)))
        if dropped
        else np.empty(0, dtype=np.int64)
    )
    # Tahmin karelerini en yakın tutulan gerçek komşularına göre O(log D) denetle
    for p in np.flatnonzero(pred):
        pos = int(np.searchsorted(kept_real, p))
        if pos == 0 or pos >= len(kept_real):
            continue
        a, b = int(kept_real[pos - 1]), int(kept_real[pos])
        idx_d = int(np.searchsorted(dropped_starts, a, side="right"))
        between_dropped = idx_d < len(dropped_starts) and int(dropped_starts[idx_d]) < b
        if between_dropped or speed(a, b) > max_speed:
            keep[p] = False
    return keep


def smooth_possession_codes(
    raw_tid_codes: np.ndarray,
    raw_team_codes: np.ndarray,
    tid_to_team_code: np.ndarray,
    win: int,
    min_control_frames: int,
    preserve_frame_team: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Çoğunluk oyuyla titreşim yumuşatma + minimum kontrol süresi (spell) filtresi.
    `preserve_frame_team=True` iken (MLAnalyzer nedensellik koruması için), mevcut
    karede takım kodu -1 ise gelecek karelerden takım bilgisi taşınmaz.
    """
    n = len(raw_tid_codes)
    if n == 0:
        return raw_tid_codes.copy(), raw_team_codes.copy()

    w = max(1, int(win))
    half = w // 2
    if w <= 1:
        sm_tids = raw_tid_codes.copy()
        sm_teams = raw_team_codes.copy()
    elif half <= 7:
        eff_w = 2 * half + 1
        pad_tid = np.pad(raw_tid_codes, half, constant_values=-1)
        win_mat = np.column_stack([pad_tid[off:off + n] for off in range(eff_w)])
        eq = (win_mat[:, :, None] == win_mat[:, None, :]) & (win_mat[:, :, None] >= 0)
        cnts = eq.sum(axis=2)
        best_col = np.argmax(cnts, axis=1)
        row_idx = np.arange(n)
        top_cnt = cnts[row_idx, best_col]
        top_tc = np.where(top_cnt > 0, win_mat[row_idx, best_col], -1)

        if half > 0:
            left_has = (win_mat[:, :half] == top_tc[:, None]).any(axis=1)
            right_has = (win_mat[:, half + 1:] == top_tc[:, None]).any(axis=1)
        else:
            left_has = np.zeros(n, dtype=bool)
            right_has = np.zeros(n, dtype=bool)

        keep = (top_cnt > 0) & (
            (raw_tid_codes >= 0)
            | (top_cnt >= 3)
            | ((top_cnt == 2) & left_has & right_has)
        )
        sm_tids = np.where(keep, top_tc, -1).astype(np.int32)
        sm_teams = np.full(n, -1, dtype=np.int32)

        pad_team = np.pad(raw_team_codes, half, constant_values=-1)
        team_mat = np.column_stack([pad_team[off:off + n] for off in range(eff_w)])

        if preserve_frame_team:
            same_mask = keep & (sm_tids == raw_tid_codes)
            sm_teams[same_mask] = raw_team_codes[same_mask]
            diff_idx = np.flatnonzero(keep & (sm_tids != raw_tid_codes))
            if len(diff_idx) > 0:
                rev_tid = win_mat[diff_idx, half::-1]
                rev_tm = team_mat[diff_idx, half::-1]
                hit = (rev_tid == sm_tids[diff_idx, None]) & (rev_tm >= 0)
                has_hit = hit.any(axis=1)
                first_col = np.argmax(hit, axis=1)
                sm_teams[diff_idx] = np.where(
                    has_hit, rev_tm[np.arange(len(diff_idx)), first_col], -1
                )
        else:
            valid_idx = np.flatnonzero(keep)
            if len(valid_idx) > 0:
                vt = sm_tids[valid_idx]
                in_lut = vt < len(tid_to_team_code)
                if in_lut.any():
                    sm_teams[valid_idx[in_lut]] = tid_to_team_code[vt[in_lut]]
                fb_idx = valid_idx[sm_teams[valid_idx] < 0]
                if len(fb_idx) > 0:
                    hit = (win_mat[fb_idx] == sm_tids[fb_idx, None]) & (team_mat[fb_idx] >= 0)
                    has_hit = hit.any(axis=1)
                    first_col = np.argmax(hit, axis=1)
                    sm_teams[fb_idx] = np.where(
                        has_hit, team_mat[fb_idx, first_col], -1
                    )
    else:
        sm_tids = np.full(n, -1, dtype=np.int32)
        sm_teams = np.full(n, -1, dtype=np.int32)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            counts: Dict[int, int] = {}
            for k in range(lo, hi):
                tc = int(raw_tid_codes[k])
                if tc >= 0:
                    counts[tc] = counts.get(tc, 0) + 1
            if not counts:
                continue
            top_tc = max(counts, key=counts.__getitem__)
            top_cnt = counts[top_tc]
            # Tek karelik boşluklarda köprü kur ama uzun boşlukları/uçuş karelerini yapay uzatma
            if raw_tid_codes[i] < 0:
                if top_cnt < 2:
                    continue
                if top_cnt < 3:
                    has_left = any(raw_tid_codes[k] == top_tc for k in range(lo, i))
                    has_right = any(raw_tid_codes[k] == top_tc for k in range(i + 1, hi))
                    if not (has_left and has_right):
                        continue
            smoothed_val = top_tc
            sm_tids[i] = smoothed_val
            if preserve_frame_team:
                if raw_tid_codes[i] == top_tc:
                    sm_teams[i] = int(raw_team_codes[i])
                else:
                    # Yalnızca geçmiş veya mevcut kareden (k <= i) al; gelecek kareden alma
                    stm = -1
                    for k in range(i, lo - 1, -1):
                        if raw_tid_codes[k] == top_tc and raw_team_codes[k] >= 0:
                            stm = int(raw_team_codes[k])
                            break
                    sm_teams[i] = stm
            else:
                stm = int(tid_to_team_code[top_tc]) if 0 <= top_tc < len(tid_to_team_code) else -1
                if stm < 0:
                    for k in range(lo, hi):
                        if raw_tid_codes[k] == top_tc and raw_team_codes[k] >= 0:
                            stm = int(raw_team_codes[k])
                            break
                sm_teams[i] = stm

    # Minimum kontrol süresi filtresi: min_control_frames altındaki tekil/kısa kontrol
    # periyotlarını sahte temas olarak temizle (kısa boşluklardan sonra devir üretmez).
    min_cf = max(1, int(min_control_frames))
    if min_cf > 1 and n > 0:
        valid_pos = sm_tids >= 0
        if valid_pos.any():
            # Ardışık aynı geçerli track_id koşularını bul
            change = np.empty(n, dtype=bool)
            change[0] = True
            change[1:] = sm_tids[1:] != sm_tids[:-1]
            starts = np.flatnonzero(change)
            ends = np.r_[starts[1:] - 1, n - 1]
            bad_runs = np.flatnonzero((sm_tids[starts] >= 0) & ((ends - starts + 1) < min_cf))
            for r_i in bad_runs:
                sm_tids[starts[r_i]:ends[r_i] + 1] = -1
                sm_teams[starts[r_i]:ends[r_i] + 1] = -1

    # Takipçi aynı kimliği zamanla farklı oyunculara verebiliyor; izin tüm ömrü boyunca
    # çoğunluk takımı o anki oyuncuyu temsil etmeyebilir. Her kontrol periyodunun takımı,
    # periyot içindeki karelerin çoğunluk takımıdır (yoksa önceki değer korunur).
    if not preserve_frame_team and n > 0:
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = sm_tids[1:] != sm_tids[:-1]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            tc = int(sm_tids[s])
            if tc < 0:
                continue
            seg_tid = raw_tid_codes[s:e]
            seg_team = raw_team_codes[s:e]
            votes = seg_team[(seg_tid == tc) & (seg_team >= 0)]
            if len(votes) > 0:
                sm_teams[s:e] = int(np.bincount(votes).argmax())

    return sm_tids, sm_teams


# ----------------------------------------------------------------------
# R1 & R2: DEVRE ARASI TESPİTİ, HÜCUM YÖNÜ VE KALECİ EŞLEŞTİRME
# ----------------------------------------------------------------------
def _infer_half_attack_directions(
    fids: np.ndarray,
    xs: np.ndarray,
    team_codes: np.ndarray,
    is_gk: np.ndarray,
    ta_code: int,
    tb_code: int,
    start_f: int,
    end_f: int,
    mid_x: float,
) -> Dict[str, int]:
    """Tek bir devre diliminde kaleci/savunma bloğu konumundan Team A ve Team B hücum yönünü (+1 / -1) çıkarır."""
    in_seg = (fids >= start_f) & (fids <= end_f)
    if not in_seg.any():
        return {"Team A": 1, "Team B": -1}

    seg_x = xs[in_seg]
    seg_t = team_codes[in_seg]
    seg_gk = is_gk[in_seg]

    ma = (seg_t == ta_code) if ta_code >= 0 else np.zeros(len(seg_t), dtype=bool)
    mb = (seg_t == tb_code) if tb_code >= 0 else np.zeros(len(seg_t), dtype=bool)

    has_a = bool(ma.any())
    has_b = bool(mb.any())

    gk_a_x = float(np.median(seg_x[ma & seg_gk])) if (ma & seg_gk).any() else None
    gk_b_x = float(np.median(seg_x[mb & seg_gk])) if (mb & seg_gk).any() else None
    out_a = seg_x[ma & ~seg_gk]
    out_b = seg_x[mb & ~seg_gk]
    med_a = float(np.median(out_a)) if len(out_a) > 0 else (float(np.median(seg_x[ma])) if has_a else None)
    med_b = float(np.median(out_b)) if len(out_b) > 0 else (float(np.median(seg_x[mb])) if has_b else None)

    if has_a and has_b:
        if gk_a_x is not None and gk_b_x is not None and abs(gk_a_x - gk_b_x) > 1e-3:
            dir_a = 1 if gk_a_x < gk_b_x else -1
            return {"Team A": dir_a, "Team B": -dir_a}
        if gk_a_x is not None and med_a is not None and abs(gk_a_x - med_a) > 1e-3:
            dir_a = 1 if gk_a_x < med_a else -1
            return {"Team A": dir_a, "Team B": -dir_a}
        if gk_b_x is not None and med_b is not None and abs(gk_b_x - med_b) > 1e-3:
            dir_b = 1 if gk_b_x < med_b else -1
            return {"Team A": -dir_b, "Team B": dir_b}
        if med_a is not None and med_b is not None:
            dir_a = 1 if med_a <= med_b else -1
            return {"Team A": dir_a, "Team B": -dir_a}
    elif has_a:
        if gk_a_x is not None and med_a is not None and abs(gk_a_x - med_a) > 1e-3:
            dir_a = 1 if gk_a_x < med_a else -1
        elif med_a is not None:
            dir_a = 1 if med_a <= mid_x else -1
        else:
            dir_a = 1
        return {"Team A": dir_a, "Team B": -dir_a}
    elif has_b:
        if gk_b_x is not None and med_b is not None and abs(gk_b_x - med_b) > 1e-3:
            dir_b = 1 if gk_b_x < med_b else -1
        elif med_b is not None:
            dir_b = 1 if med_b <= mid_x else -1
        else:
            dir_b = -1
        return {"Team A": -dir_b, "Team B": dir_b}

    return {"Team A": 1, "Team B": -1}


def _detect_side_swap_splits(
    fids: np.ndarray,
    xs: np.ndarray,
    team_codes: np.ndarray,
    is_gk: np.ndarray,
    ta_code: int,
    tb_code: int,
    seg_start: int,
    seg_end: int,
    min_sep: float,
    min_frames_for_swap: int,
) -> List[Tuple[int, int, str]]:
    """Sürekli video diliminde kalıcı takım/kaleci taraf değişimi (devre arası) arar."""
    in_seg = (fids >= seg_start) & (fids <= seg_end)
    if not in_seg.any():
        return [(seg_start, seg_end, "initial")]

    sf = fids[in_seg]
    sx = xs[in_seg]
    st = team_codes[in_seg]
    sgk = is_gk[in_seg]
    if len(sf) > 1 and np.any(sf[1:] < sf[:-1]):
        ord_s = np.argsort(sf, kind="mergesort")
        sf = sf[ord_s]
        sx = sx[ord_s]
        st = st[ord_s]
        sgk = sgk[ord_s]

    u_fids = np.unique(sf)
    if len(u_fids) < min_frames_for_swap:
        return [(seg_start, seg_end, "initial")]

    n_win = min(30, max(6, len(u_fids) // 15))
    if n_win < 4:
        return [(seg_start, seg_end, "initial")]

    bin_edges = np.linspace(0, len(u_fids), n_win + 1, dtype=int)
    w_start_f = [int(u_fids[bin_edges[w]]) for w in range(n_win)]
    w_end_f = [int(u_fids[bin_edges[w + 1] - 1]) for w in range(n_win)]

    w_lo = np.searchsorted(sf, w_start_f, side="left")
    w_hi = np.searchsorted(sf, w_end_f, side="right")
    d_vals = np.full(n_win, np.nan, dtype=np.float64)
    for w in range(n_win):
        lo_w = int(w_lo[w])
        hi_w = int(w_hi[w])
        if hi_w <= lo_w:
            continue
        wx = sx[lo_w:hi_w]
        wt = st[lo_w:hi_w]
        wgk = sgk[lo_w:hi_w]

        ma = (wt == ta_code) if ta_code >= 0 else np.zeros(len(wt), dtype=bool)
        mb = (wt == tb_code) if tb_code >= 0 else np.zeros(len(wt), dtype=bool)

        if ma.any() and mb.any():
            if (ma & wgk).any() and (mb & wgk).any():
                ax = float(np.median(wx[ma & wgk]))
                bx = float(np.median(wx[mb & wgk]))
            else:
                ax = float(np.median(wx[ma]))
                bx = float(np.median(wx[mb]))
            d_vals[w] = ax - bx
        elif ma.any() and (ma & wgk).any() and (ma & ~wgk).any():
            d_vals[w] = float(np.median(wx[ma & wgk]) - np.median(wx[ma & ~wgk]))
        elif mb.any() and (mb & wgk).any() and (mb & ~wgk).any():
            d_vals[w] = float(np.median(wx[mb & ~wgk]) - np.median(wx[mb & wgk]))

    has_gks = bool(sgk.any())
    req_purity = 0.75 if has_gks else 0.85
    req_sep = min_sep if has_gks else max(min_sep, 2.5 * min_sep)

    best_k = -1
    best_score = -1.0
    min_side_wins = max(2, n_win // 5)

    for k in range(min_side_wins, n_win - min_side_wins + 1):
        left = d_vals[:k]
        right = d_vals[k:]
        lv = left[np.isfinite(left)]
        rv = right[np.isfinite(right)]
        if len(lv) < min_side_wins or len(rv) < min_side_wins:
            continue

        med_l = float(np.median(lv))
        med_r = float(np.median(rv))
        if abs(med_l) < req_sep or abs(med_r) < req_sep:
            continue
        if med_l * med_r >= 0:
            continue

        sign_l = 1.0 if med_l > 0 else -1.0
        # Yarı sonunda eski tarafa geri dönüş varsa (kontratak salınımı) devre bölünmesi değildir
        if np.median(lv[:min(2, len(lv))]) * sign_l <= 0:
            continue
        if np.median(rv[-min(2, len(rv)):]) * (-sign_l) <= 0:
            continue

        purity_l = float(np.mean((lv * sign_l) > (0.25 * min_sep)))
        purity_r = float(np.mean((rv * (-sign_l)) > (0.25 * min_sep)))
        if purity_l < req_purity or purity_r < req_purity:
            continue

        score = (abs(med_l) + abs(med_r)) * purity_l * purity_r
        if score > best_score:
            best_score = score
            best_k = k

    if best_k < 0:
        return [(seg_start, seg_end, "initial")]

    # Sınır karesini pencere (best_k - 1) .. best_k içinde işaret değişimine göre rafine et
    split_lo = w_start_f[max(0, best_k - 1)]
    split_hi = w_end_f[min(n_win - 1, best_k)]
    cand_fids = u_fids[(u_fids >= split_lo) & (u_fids <= split_hi)]
    split_frame = w_start_f[best_k]

    if len(cand_fids) >= 2:
        med_l_sign = 1.0 if float(np.nanmedian(d_vals[:best_k])) > 0 else -1.0
        cf_lo = np.searchsorted(sf, cand_fids, side="left")
        cf_hi = np.searchsorted(sf, cand_fids, side="right")
        for idx_cf, cf in enumerate(cand_fids):
            lo_c = int(cf_lo[idx_cf])
            hi_c = int(cf_hi[idx_cf])
            if hi_c <= lo_c:
                continue
            cx = sx[lo_c:hi_c]
            ct = st[lo_c:hi_c]
            cgk = sgk[lo_c:hi_c]
            ma = (ct == ta_code) if ta_code >= 0 else np.zeros(len(ct), dtype=bool)
            mb = (ct == tb_code) if tb_code >= 0 else np.zeros(len(ct), dtype=bool)
            diff_cf: Optional[float] = None
            if ma.any() and mb.any():
                if (ma & cgk).any() and (mb & cgk).any():
                    diff_cf = float(np.median(cx[ma & cgk]) - np.median(cx[mb & cgk]))
                else:
                    diff_cf = float(np.median(cx[ma]) - np.median(cx[mb]))
            if diff_cf is not None and (diff_cf * (-med_l_sign)) > 0:
                split_frame = int(cf)
                break

    if split_frame <= seg_start or split_frame > seg_end:
        return [(seg_start, seg_end, "initial")]

    return [
        (seg_start, split_frame - 1, "initial"),
        (split_frame, seg_end, "side_swap"),
    ]


def _extract_player_orientation_arrays(
    df: pd.DataFrame,
    is_calibrated: bool,
    resolved_team_codes: Optional[np.ndarray] = None,
    resolved_team_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if df is None or df.empty or "frame_idx" not in df.columns or "class_name" not in df.columns:
        return {"empty": True}

    x_col = "pitch_x" if is_calibrated and "pitch_x" in df.columns else "pixel_x"
    if x_col not in df.columns:
        return {"empty": True}

    cls_s = df["class_name"]
    human_names = set(getattr(config, "HUMAN_CLASS_NAMES", ())) | {"person"}
    is_ref_cls = (cls_s == "referee").fillna(False).to_numpy(dtype=bool)
    f_notna = df["frame_idx"].notna().to_numpy(dtype=bool)
    x_notna = df[x_col].notna().to_numpy(dtype=bool)

    if resolved_team_codes is not None and resolved_team_list is not None:
        t_codes_all = resolved_team_codes
        t_list = list(resolved_team_list)
    elif "team" in df.columns:
        t_codes_all, t_list = _encode_clean_strings_local(df["team"])
    else:
        t_codes_all = np.full(len(df), -1, dtype=np.int32)
        t_list = []

    ref_code = t_list.index("Referee") if "Referee" in t_list else -2
    is_ref_team = (t_codes_all == ref_code) if ref_code >= 0 else np.zeros(len(df), dtype=bool)
    gk_team_code = t_list.index("Goalkeeper") if "Goalkeeper" in t_list else -2
    is_gk_all = (cls_s == "goalkeeper").fillna(False).to_numpy(dtype=bool) | (
        (t_codes_all == gk_team_code) if gk_team_code >= 0 else False
    )

    p_mask = (
        cls_s.isin(human_names).fillna(False).to_numpy(dtype=bool)
        & (~is_ref_cls)
        & (~is_ref_team)
        & f_notna
        & x_notna
    )
    if is_calibrated and "on_pitch" in df.columns:
        p_mask &= _bool_mask_local(df["on_pitch"], na_value=True)

    p_idx = np.flatnonzero(p_mask)
    if len(p_idx) == 0:
        return {"empty": True}

    fids = df["frame_idx"].iloc[p_idx].to_numpy(dtype=np.int64)
    xs = df[x_col].iloc[p_idx].to_numpy(dtype=np.float64)
    ts = (
        pd.to_numeric(df["timestamp_sec"].iloc[p_idx], errors="coerce").to_numpy(dtype=np.float64)
        if "timestamp_sec" in df.columns
        else np.full(len(p_idx), np.nan, dtype=np.float64)
    )
    t_codes = t_codes_all[p_idx]
    gk_mask = is_gk_all[p_idx]
    tid_codes, tid_list = (
        _encode_clean_strings_local(df["track_id"].iloc[p_idx])
        if "track_id" in df.columns
        else (np.full(len(p_idx), -1, dtype=np.int32), [])
    )

    return {
        "empty": False,
        "p_idx": p_idx,
        "fids": fids,
        "xs": xs,
        "ts": ts,
        "t_codes": t_codes,
        "t_list": t_list,
        "gk_mask": gk_mask,
        "tid_codes": tid_codes,
        "tid_list": tid_list,
    }


def _detect_match_halves_from_arrays(
    arr: Dict[str, Any],
    fps_safe: float,
    bounds: Optional[Tuple[float, float, float, float]],
    is_calibrated: bool,
) -> List[Dict[str, Any]]:
    if arr.get("empty"):
        return [{
            "period": 1,
            "start_frame": 0,
            "end_frame": 0,
            "attack_directions": {"Team A": 1, "Team B": -1},
            "detection_reason": "default",
        }]

    fids = arr["fids"]
    xs = arr["xs"]
    ts = arr["ts"]
    t_codes = arr["t_codes"]
    t_list = arr["t_list"]
    gk_mask = arr["gk_mask"]

    ta_code = t_list.index("Team A") if "Team A" in t_list else -2
    tb_code = t_list.index("Team B") if "Team B" in t_list else -2

    ord_f = np.argsort(fids, kind="mergesort")
    fids_s = fids[ord_f]
    ts_s = ts[ord_f]
    first_f = np.empty(len(fids_s), dtype=bool)
    first_f[0] = True
    first_f[1:] = fids_s[1:] != fids_s[:-1]
    u_fids = fids_s[first_f]
    u_ts = ts_s[first_f]

    min_f = int(u_fids[0])
    max_f = int(u_fids[-1])

    # 1) Belirgin zaman/kare boşluğuna göre kaba devre bölütleri
    gap_thresh_sec = float(getattr(config, "HALFTIME_MIN_GAP_SEC", 60.0))
    gap_segments: List[Tuple[int, int, str]] = []
    if len(u_fids) >= 2:
        df_u = np.diff(u_fids).astype(np.float64) / fps_safe
        if np.isfinite(u_ts).all():
            dt_u = np.diff(u_ts)
            eff_dt = np.where(dt_u > 0, dt_u, df_u)
        else:
            eff_dt = df_u
        gap_idx = np.flatnonzero(eff_dt > gap_thresh_sec)
        cur_start = min_f
        reason = "initial"
        for gi in gap_idx:
            seg_end = int(u_fids[gi])
            gap_segments.append((cur_start, seg_end, reason))
            cur_start = int(u_fids[gi + 1])
            reason = "time_gap"
        gap_segments.append((cur_start, max_f, reason))
    else:
        gap_segments.append((min_f, max_f, "initial"))

    if bounds is not None and len(bounds) >= 2:
        span_x = max(1.0, float(bounds[1]) - float(bounds[0]))
        mid_x = 0.5 * (float(bounds[0]) + float(bounds[1]))
    else:
        span_x = max(1.0, float(np.max(xs) - np.min(xs))) if len(xs) > 0 else 105.0
        mid_x = 52.5 if is_calibrated else (0.5 * float(np.min(xs) + np.max(xs)))

    min_sep = max(6.0, 0.06 * 105.0) if is_calibrated else max(30.0, 0.06 * span_x)
    min_frames_for_swap = int(getattr(config, "HALFTIME_MIN_FRAMES_FOR_SWAP", 60))

    # 2) Her boşluk-bölütünün içinde sürekli taraf değişimi ara
    refined_segments: List[Tuple[int, int, str]] = []
    for seg_s, seg_e, seg_reason in gap_segments:
        sub_segs = _detect_side_swap_splits(
            fids=fids,
            xs=xs,
            team_codes=t_codes,
            is_gk=gk_mask,
            ta_code=ta_code,
            tb_code=tb_code,
            seg_start=seg_s,
            seg_end=seg_e,
            min_sep=min_sep,
            min_frames_for_swap=min_frames_for_swap,
        )
        for idx_s, (ss, se, sr) in enumerate(sub_segs):
            actual_reason = seg_reason if (idx_s == 0 and seg_reason == "time_gap") else sr
            refined_segments.append((ss, se, actual_reason))

    halves: List[Dict[str, Any]] = []
    for period_idx, (hs, he, reason) in enumerate(refined_segments, start=1):
        dirs = _infer_half_attack_directions(
            fids=fids,
            xs=xs,
            team_codes=t_codes,
            is_gk=gk_mask,
            ta_code=ta_code,
            tb_code=tb_code,
            start_f=hs,
            end_f=he,
            mid_x=mid_x,
        )
        halves.append({
            "period": period_idx,
            "start_frame": int(hs),
            "end_frame": int(he),
            "attack_directions": dirs,
            "detection_reason": reason,
        })

    return halves


def detect_match_halves(
    df: pd.DataFrame,
    fps: float = 25.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    is_calibrated: Optional[bool] = None,
    resolved_team_codes: Optional[np.ndarray] = None,
    resolved_team_list: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Takip DataFrame'inden maç devrelerini ve her devrede Team A / Team B'nin
    gerçek hücum yönünü (+1 soldan sağa, -1 sağdan sola) otomatik çıkarır:
      1) Belirgin zaman/kare boşluğu (> HALFTIME_MIN_GAP_SEC = 60s)
      2) Sürekli videoda kalıcı kaleci / takım ağırlık merkezi taraf değişimi
         (kısa tek-devre kliplerde sahte bölünmeyi önleyen histerezis ile)
    """
    fps_safe = float(fps) if fps and fps > 0 else 25.0
    if is_calibrated is None:
        is_calibrated = bool(
            df is not None
            and not df.empty
            and "pitch_x" in df.columns
            and df["pitch_x"].notna().any()
        )

    arr = _extract_player_orientation_arrays(
        df,
        is_calibrated=bool(is_calibrated),
        resolved_team_codes=resolved_team_codes,
        resolved_team_list=resolved_team_list,
    )
    return _detect_match_halves_from_arrays(arr, fps_safe, bounds, bool(is_calibrated))


def resolve_goalkeeper_teams(
    df: pd.DataFrame,
    fps: float = 25.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    is_calibrated: Optional[bool] = None,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Girdi DataFrame'ini DEĞİŞTİRMEDEN (copy=False uyumlu), her yarıdaki savunma
    bloğunun ortalama x konumuna göre 'Goalkeeper' veya takımsız kaleci satırlarını
    'Team A' / 'Team B' olarak çözümler ve devre bilgisini döndürür.

    Önemli: Yalnızca kaleci satırlarını (class_name == 'goalkeeper' veya
    team == 'Goalkeeper') günceller; saha oyuncularının eksik (None) takım
    etiketlerine dokunmaz (nedensel non-leakage korunur).
    """
    n_total = 0 if df is None else len(df)
    if n_total == 0:
        return np.empty(0, dtype=np.int32), [], [{
            "period": 1,
            "start_frame": 0,
            "end_frame": 0,
            "attack_directions": {"Team A": 1, "Team B": -1},
            "detection_reason": "default",
        }]

    fps_safe = float(fps) if fps and fps > 0 else 25.0
    if is_calibrated is None:
        is_calibrated = bool(
            "pitch_x" in df.columns and "class_name" in df.columns and df["pitch_x"].notna().any()
        )

    if "team" in df.columns:
        team_codes, team_list = _encode_clean_strings_local(df["team"])
    else:
        team_codes = np.full(n_total, -1, dtype=np.int32)
        team_list = []

    team_list = list(team_list)
    if "Team A" not in team_list:
        team_list.append("Team A")
    if "Team B" not in team_list:
        team_list.append("Team B")

    ta_code = team_list.index("Team A")
    tb_code = team_list.index("Team B")
    gk_str_code = team_list.index("Goalkeeper") if "Goalkeeper" in team_list else -2

    cls_s = df["class_name"] if "class_name" in df.columns else pd.Series([None] * n_total)
    is_gk_row = (cls_s == "goalkeeper").fillna(False).to_numpy(dtype=bool)
    if gk_str_code >= 0:
        is_gk_row |= (team_codes == gk_str_code)

    unresolved_gk = is_gk_row & ((team_codes == gk_str_code) | (team_codes < 0))

    arr = _extract_player_orientation_arrays(
        df,
        is_calibrated=bool(is_calibrated),
        resolved_team_codes=team_codes,
        resolved_team_list=team_list,
    )
    halves = _detect_match_halves_from_arrays(arr, fps_safe, bounds, bool(is_calibrated))

    if not unresolved_gk.any() or arr.get("empty"):
        return team_codes, team_list, halves

    p_idx = arr["p_idx"]
    fids = arr["fids"]
    xs = arr["xs"]
    t_codes = arr["t_codes"]
    gk_mask = arr["gk_mask"]
    tid_codes = arr["tid_codes"]

    resolved_codes = team_codes.copy()
    p_resolved = t_codes.copy()
    single_h = len(halves) == 1
    any_updated = False

    for half in halves:
        if single_h:
            out_a = xs[(~gk_mask) & (t_codes == ta_code)]
            out_b = xs[(~gk_mask) & (t_codes == tb_code)]
        else:
            hs = int(half["start_frame"])
            he = int(half["end_frame"])
            in_h = (fids >= hs) & (fids <= he)
            if not in_h.any():
                continue
            out_a = xs[in_h & (~gk_mask) & (t_codes == ta_code)]
            out_b = xs[in_h & (~gk_mask) & (t_codes == tb_code)]

        if len(out_a) == 0 and len(out_b) == 0:
            continue

        # Savunma bloğu ortalama/kuantil x konumu
        if len(out_a) > 0 and len(out_b) > 0:
            med_a = float(np.median(out_a))
            med_b = float(np.median(out_b))
            if med_a <= med_b:
                block_a = float(np.mean(out_a[out_a <= np.quantile(out_a, 0.50)]))
                block_b = float(np.mean(out_b[out_b >= np.quantile(out_b, 0.50)]))
            else:
                block_a = float(np.mean(out_a[out_a >= np.quantile(out_a, 0.50)]))
                block_b = float(np.mean(out_b[out_b <= np.quantile(out_b, 0.50)]))
        else:
            block_a = float(np.mean(out_a)) if len(out_a) > 0 else None
            block_b = float(np.mean(out_b)) if len(out_b) > 0 else None

        if single_h:
            unres_in_h = gk_mask & ((t_codes == gk_str_code) | (t_codes < 0))
        else:
            unres_in_h = in_h & gk_mask & ((t_codes == gk_str_code) | (t_codes < 0))
        if not unres_in_h.any():
            continue

        u_pos = np.flatnonzero(unres_in_h)
        u_tids = tid_codes[u_pos]

        # Takipli kaleciler (track_id mevcut): iz başına medyan x ile tutarlı eşle
        pos_mask = u_tids >= 0
        if pos_mask.any():
            pos_u = u_pos[pos_mask]
            pos_tids = u_tids[pos_mask]
            for tid_c in np.unique(pos_tids):
                sub = pos_u[pos_tids == tid_c]
                gk_x = float(np.median(xs[sub]))
                if block_a is not None and block_b is not None:
                    assigned_code = ta_code if abs(gk_x - block_a) <= abs(gk_x - block_b) else tb_code
                elif block_a is not None:
                    assigned_code = ta_code
                else:
                    assigned_code = tb_code
                p_resolved[sub] = assigned_code
                resolved_codes[p_idx[sub]] = assigned_code
            any_updated = True

        # İz numarası eksik kaleciler (track_id=None/NaN): satır bazlı vektörel çöz (havuzlama yok)
        neg_mask = ~pos_mask
        if neg_mask.any():
            neg_u = u_pos[neg_mask]
            if block_a is not None and block_b is not None:
                row_xs = xs[neg_u]
                assigned_neg = np.where(
                    np.abs(row_xs - block_a) <= np.abs(row_xs - block_b),
                    ta_code,
                    tb_code,
                ).astype(np.int32)
            elif block_a is not None:
                assigned_neg = np.full(len(neg_u), ta_code, dtype=np.int32)
            else:
                assigned_neg = np.full(len(neg_u), tb_code, dtype=np.int32)
            p_resolved[neg_u] = assigned_neg
            resolved_codes[p_idx[neg_u]] = assigned_neg
            any_updated = True

    # Çözümlenmiş kalecilerle devreleri ve hücum yönlerini son kez rafine et
    if any_updated:
        arr["t_codes"] = p_resolved
        halves = _detect_match_halves_from_arrays(arr, fps_safe, bounds, bool(is_calibrated))
    return resolved_codes, team_list, halves


def lookup_attack_directions(
    frame_indices: np.ndarray,
    teams: np.ndarray,
    halves: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Her olay / gözlem satırı için (frame_idx, team) çiftine karşılık gelen
    hücum yönünü (+1 soldan sağa, -1 sağdan sola) O(N) vektörel döndürür.
    """
    n = len(frame_indices)
    dirs = np.ones(n, dtype=np.int8)
    if n == 0 or not halves:
        return dirs

    f_arr = np.asarray(frame_indices, dtype=np.int64)
    t_arr = np.asarray(teams, dtype=object)
    is_a = t_arr == "Team A"
    is_b = t_arr == "Team B"

    if len(halves) == 1:
        ad = halves[0].get("attack_directions", {})
        da = int(ad.get("Team A", 1))
        db = int(ad.get("Team B", -1))
        return np.where(is_a, da, np.where(is_b, db, 1)).astype(np.int8)

    for idx_h, h in enumerate(halves):
        hs = int(h["start_frame"])
        he = int(h["end_frame"])
        if idx_h == 0:
            mask = f_arr <= he
        elif idx_h == len(halves) - 1:
            mask = f_arr >= hs
        else:
            mask = (f_arr >= hs) & (f_arr <= he)
        if not mask.any():
            continue
        ad = h.get("attack_directions", {})
        da = int(ad.get("Team A", 1))
        db = int(ad.get("Team B", -1))
        dirs[mask & is_a] = da
        dirs[mask & is_b] = db

    return dirs
