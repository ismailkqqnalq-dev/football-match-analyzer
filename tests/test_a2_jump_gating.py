"""
test_a2_jump_gating.py
----------------------
Faz A2-3: Model–flow uyuşmazlığına göre kademeli kabul (sıçrama/titreme koruması),
referans noktaları üzerinden yumuşak geçiş ve `calib_reset` karelerinin mesafe,
hız ve top kinematik hesaplarından dışlanması.

Simülasyon: gerçek kamera SABİT. Keypoint modeli gerçek tablodan gürültülü noktalarla
tablo kestirir; optical flow sahte bir takipçiyle (isteğe bağlı birikimli hata) verilir.
"""

import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.data_exporter import DataExporter
from modules.perspective_transformer import PerspectiveTransformer
from modules.pitch_calibrator import (
    AutoPitchCalibrator,
    CalibrationResult,
    HybridPitchCalibrator,
)
from modules.pitch_orientation import compute_ball_kinematic_filter

FRAME_W, FRAME_H = 1280, 720
FPS = 25.0
PLAYER_PX = np.array([[640.0, 520.0]])


def _true_h_m_to_px() -> np.ndarray:
    w, h = float(FRAME_W), float(FRAME_H)
    src_m = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    dst_px = np.array(
        [[w * 0.14, h * 0.22], [w * 0.86, h * 0.22], [w * 0.96, h * 0.88], [w * 0.04, h * 0.88]]
    )
    H, _ = cv2.findHomography(src_m, dst_px, method=0)
    return np.asarray(H, dtype=np.float64)


H_M_TO_PX = _true_h_m_to_px()
H_TRUE = np.linalg.inv(H_M_TO_PX)


class FakeFlow:
    """Sabit kamera; flow'un kestirdiği öteleme `err_fn(t)` (piksel) kadar hatalı."""

    def __init__(self, err_fn=None):
        self.err_fn = err_fn or (lambda t: np.zeros(2))
        self.t = 0
        self.anchor_err = np.zeros(2)
        self.scene_cut_detected = False
        self.recovered_this_frame = False

    def update(self, frame, detections=None):
        self.t += 1
        d = np.asarray(self.err_fn(self.t), dtype=np.float64) - self.anchor_err
        m = np.eye(3)
        m[0, 2], m[1, 2] = d[0], d[1]
        return m

    def reanchor(self, frame, detections=None, keep_keyframe_bank=True, prev_cumulative=None):
        self.anchor_err = np.asarray(self.err_fn(self.t), dtype=np.float64)
        return np.eye(3)


class NoisyKeypointCalibrator:
    """Gerçek tablodan σ piksel gürültülü keypoint'lerle tablo kestirir."""

    is_available = True

    def __init__(self, sigma_px=3.0, seed=0, meter_offset_fn=None, fail_fn=None):
        self.sigma = float(sigma_px)
        self.rng = np.random.default_rng(seed)
        self.meter_offset_fn = meter_offset_fn or (lambda call: (0.0, 0.0))
        self.fail_fn = fail_fn or (lambda call: False)
        self.calls = 0
        self.est = AutoPitchCalibrator(model=object(), device="cpu")
        m_all = np.asarray(config.PITCH_KEYPOINTS_32_METERS, dtype=np.float64)
        px_all = cv2.perspectiveTransform(m_all.reshape(1, -1, 2), H_M_TO_PX)[0]
        inside = (
            (px_all[:, 0] >= 0) & (px_all[:, 0] < FRAME_W)
            & (px_all[:, 1] >= 0) & (px_all[:, 1] < FRAME_H)
        )
        self.m_pts = m_all[inside]
        self.px_pts = px_all[inside]

    def calibrate_frame(self, frame):
        self.calls += 1
        if self.fail_fn(self.calls):
            return CalibrationResult(success=False, reason="no_valid_keypoints")
        px = self.px_pts + self.rng.normal(0.0, self.sigma, size=self.px_pts.shape)
        off = np.asarray(self.meter_offset_fn(self.calls), dtype=np.float64)
        return self.est.estimate_homography_from_keypoints(
            px, self.m_pts + off, (FRAME_H, FRAME_W, 3)
        )


def _make_hybrid(cal, flow, profile="broadcast", refresh=15):
    pt = PerspectiveTransformer(source_points={}, quiet=True)
    hybrid = HybridPitchCalibrator(
        calibrator=cal,
        transformer=pt,
        motion_tracker=flow,
        refresh_interval=refresh,
        smooth_frames=5,
        fps=FPS,
        profile=profile,
    )
    return hybrid, pt


def _player_pitch(pt, motion):
    e = pt.homography_matrix @ motion
    return HybridPitchCalibrator._project(e, PLAYER_PX)[0]


def _run(hybrid, pt, n_frames):
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    positions, infos = [], []
    for f in range(n_frames):
        m = hybrid.process_frame(f, frame)
        assert m is not None, f"kare {f}: kalibrasyon yok"
        positions.append(_player_pitch(pt, m))
        infos.append(dict(hybrid.frame_info))
    return np.asarray(positions), infos


def _true_player_pitch():
    return HybridPitchCalibrator._project(H_TRUE, PLAYER_PX)[0]


# ----------------------------------------------------------------------
def test_a2_3_stationary_player_noise_does_not_create_fake_distance():
    """σ = 3 px keypoint gürültüsü, 25 fps'de 90 sn: sabit oyuncu < 1 km/s ve < 10 m."""
    n = int(90 * FPS)
    cal = NoisyKeypointCalibrator(sigma_px=3.0, seed=42)
    hybrid, pt = _make_hybrid(cal, FakeFlow())
    pos, infos = _run(hybrid, pt, n)

    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    total_m = float(steps.sum())
    avg_kmh = total_m / (n / FPS) * 3.6
    assert total_m < 10.0, f"sahte mesafe {total_m:.2f} m"
    assert avg_kmh < 1.0, f"sahte ortalama hız {avg_kmh:.3f} km/s"
    # Konum gerçeğe yakın kalmalı
    assert float(np.linalg.norm(pos[-1] - _true_player_pitch())) < 1.0
    assert cal.calls > 100  # broadcast profilinde periyodik yenileme gerçekten çalıştı


def test_a2_3_single_outlier_refresh_causes_no_jump():
    """Bir yenilemede tablo 5 m kaymış: hiçbir karede 0,5 m'den büyük sıçrama yok."""
    outlier_call = 8
    cal = NoisyKeypointCalibrator(
        sigma_px=1.0,
        seed=3,
        meter_offset_fn=lambda c: (5.0, 0.0) if c == outlier_call else (0.0, 0.0),
    )
    hybrid, pt = _make_hybrid(cal, FakeFlow())
    pos, infos = _run(hybrid, pt, 400)

    jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert float(jumps.max()) < 0.5, f"en büyük sıçrama {jumps.max():.2f} m"
    disagreements = [i["disagreement_m"] for i in infos if i["disagreement_m"] is not None]
    assert max(disagreements) > config.PITCH_CAL_SOFT_M  # aykırı gerçekten görüldü
    assert hybrid.stats["rejected_outlier"] >= 1
    assert hybrid.stats["confirmed_hard_reset"] == 0
    assert not any(i["calib_reset"] for i in infos[1:])
    assert float(np.linalg.norm(pos[-1] - _true_player_pitch())) < 0.5


def test_a2_3_real_flow_drift_is_corrected_after_confirmation_and_flagged():
    """
    Model 100-400. karelerde nokta bulamıyor; bu sırada flow birikimli ~4 m kayıyor.
    Model geri gelince fark > HARD: 1 + CONFIRM tutarlı yenilemeden sonra sert geçiş,
    o kare calib_reset; mesafe özetinde o adım sayılmıyor.
    """
    drift_px = 45.0  # ~4 m

    def err(t):
        if t < 100:
            return np.zeros(2)
        return np.array([drift_px * min(1.0, (t - 100) / 300.0), 0.0])

    cal = NoisyKeypointCalibrator(sigma_px=1.0, seed=5)
    flow = FakeFlow(err)
    hybrid, pt = _make_hybrid(cal, flow)

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    pos, infos = [], []
    for f in range(700):
        cal.fail_fn = (lambda c, f=f: 100 <= f < 400)
        m = hybrid.process_frame(f, frame)
        pos.append(_player_pitch(pt, m))
        infos.append(dict(hybrid.frame_info))
    pos = np.asarray(pos)

    drift_m = float(np.linalg.norm(pos[399] - _true_player_pitch()))
    assert drift_m > config.PITCH_CAL_HARD_M, f"kurulum: kayma {drift_m:.2f} m"
    reset_frames = [f for f, i in enumerate(infos) if i["calib_reset"] and f > 0]
    assert len(reset_frames) == 1, reset_frames
    rf = reset_frames[0]
    assert hybrid.stats["confirmed_hard_reset"] == 1
    # Doğrulama: ilk ret değil, 1 + CONFIRM yenileme sonra (15 karelik aralıkla)
    assert rf >= 400 + config.PITCH_CAL_CONFIRM_REFRESHES * hybrid.refresh_interval
    assert float(np.linalg.norm(pos[rf] - _true_player_pitch())) < 0.5
    assert float(np.linalg.norm(pos[-1] - _true_player_pitch())) < 0.5

    # Mesafe özeti: sıfırlama karesine giren sıçrama sayılmamalı
    frames = np.arange(len(pos))
    common = dict(
        track_id="7", frame_idxs=frames, class_names=["player"] * len(pos),
        teams=["Team A"] * len(pos), pitch_xs=pos[:, 0], pitch_ys=pos[:, 1], fps=FPS,
        smooth_win=1, min_sprint_frames=25, max_frame_gap=1, max_realistic_speed_kmh=1e9,
    )
    with_jump = DataExporter._summarize_track_rows(**common)
    without_jump = DataExporter._summarize_track_rows(
        **common, reset_frames=np.asarray(reset_frames)
    )
    jump_m = float(np.linalg.norm(pos[rf] - pos[rf - 1]))
    assert jump_m > config.PITCH_CAL_HARD_M
    assert with_jump["total_distance_m"] - without_jump["total_distance_m"] == pytest.approx(
        jump_m, abs=0.05
    )


def test_a2_3_soft_blend_uses_reference_points_and_valid_matrices():
    """Yumuşak geçişte her ara tablo geçerli, referans noktaları α ile doğrusal ilerliyor."""
    # Flow'da 0.8 m'lik küçük kayma -> SOFT bandında, hemen yumuşak geçiş
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=1)
    flow = FakeFlow(lambda t: np.array([0.0 if t < 2 else 8.0, 0.0]))
    hybrid, pt = _make_hybrid(cal, flow, refresh=15)
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for f in range(15):
        hybrid.process_frame(f, frame)
    hybrid.process_frame(15, frame)
    d = hybrid.frame_info["disagreement_m"]
    assert config.PITCH_CAL_DEADBAND_M < d <= config.PITCH_CAL_SOFT_M
    assert hybrid.frame_info["source"] == "blend"
    assert hybrid._blend_total == max(5, math.ceil(d / hybrid.blend_step_m))

    quad = hybrid._blend_quad_img
    p0, p1 = hybrid._blend_p0.copy(), hybrid._blend_p1.copy()
    min_abs_det, max_cond, _w, row2 = PerspectiveTransformer._get_homography_thresholds()
    for a in np.linspace(0.0, 1.0, 11):
        h = hybrid.blend_matrix(a)
        assert h is not None
        assert PerspectiveTransformer._validate_and_normalize_matrix(
            h, min_abs_det=min_abs_det, max_cond=max_cond, row2_min_norm=row2
        ) is not None
        proj = HybridPitchCalibrator._project(h, quad)
        assert np.allclose(proj, (1 - a) * p0 + a * p1, atol=1e-3)

    # Kare başına kayma blend_step_m (saniyelik sınır / fps) değerini aşmamalı
    prev = None
    max_step = 0.0
    for f in range(16, 16 + hybrid._blend_total + 2):
        m = hybrid.process_frame(f, frame)
        p = _player_pitch(pt, m)
        if prev is not None:
            max_step = max(max_step, float(np.linalg.norm(p - prev)))
        prev = p
    assert max_step <= hybrid.blend_step_m * 1.5
    assert hybrid._blend_total == 0


def test_a2_3_matrix_averaging_removed():
    import inspect
    from modules import pitch_calibrator

    src = inspect.getsource(pitch_calibrator.HybridPitchCalibrator)
    assert "(1.0 - alpha) * self._blend_start_h" not in src
    assert "disp_m <= 25.0" not in src


# ----------------------------------------------------------------------
# calib_reset: dışa aktarım, ML hızları, top kinematik filtresi
# ----------------------------------------------------------------------
def _exporter_with_jump(tmp_path, flag_reset: bool) -> DataExporter:
    exp = DataExporter(
        fps=FPS, output_dir=str(tmp_path), flush_interval=0, enable_csv=True, enable_sql=True
    )
    for f in range(50):
        # 25. karede 1 m kalibrasyon sıçraması (5 karelik yumuşatmayla hız sınırının altında
        # kaldığı için hız filtresi yakalamaz; yalnızca calib_reset bölmesi dışlar)
        x = 30.0 + (1.0 if f >= 25 else 0.0)
        exp.append_frame_data(
            f,
            [{
                "track_id": 1, "class_name": "player", "team": "Team A", "confidence": 0.9,
                "bbox": [100, 100, 120, 160], "pitch_x": x, "pitch_y": 30.0, "on_pitch": True,
                "calib_reset": flag_reset and f == 25,
            }],
        )
    return exp


@pytest.mark.parametrize("stream", [False, True])
def test_a2_3_distance_summary_skips_calib_reset_step(tmp_path, stream):
    for flag in (False, True):
        out = tmp_path / f"{stream}_{flag}"
        exp = _exporter_with_jump(out, flag)
        if stream:
            exp.export_to_sql()
            exp._has_flushed_any = True
            exp._active_sql_path = exp._get_sql_path(None)
        path = exp.export_distance_summary()
        df = pd.read_csv(path)
        dist = float(df.loc[0, "total_distance_m"])
        if flag:
            assert dist == pytest.approx(0.0, abs=1e-6)
        else:
            assert dist > 0.8
        exp.close()


def test_a2_3_calib_reset_column_written_to_csv_and_sql(tmp_path):
    exp = _exporter_with_jump(tmp_path, True)
    csv_path = exp.export_to_csv()
    sql_path = exp.export_to_sql()
    df = pd.read_csv(csv_path)
    assert "calib_reset" in df.columns
    assert df.loc[df["frame_idx"] == 25, "calib_reset"].astype(str).str.lower().iloc[0] == "true"
    import sqlite3

    conn = sqlite3.connect(sql_path)
    try:
        table = exp._get_table_name()
        n = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE calib_reset = 1').fetchone()[0]
    finally:
        conn.close()
    assert n == 1
    exp.close()


def test_a2_3_ml_step_speeds_exclude_calib_reset():
    from modules.ml_analyzer import MLAnalyzer

    rows = []
    for f in range(60):
        x = 30.0 + 0.1 * f + (6.0 if f >= 30 else 0.0)
        rows.append({
            "frame_idx": f, "timestamp_sec": f / FPS, "track_id": "1", "class_name": "player",
            "team": "Team A", "confidence": 0.9, "is_predicted": False,
            "bbox_x1": 100.0, "bbox_y1": 100.0, "bbox_x2": 120.0, "bbox_y2": 160.0,
            "pixel_x": 110.0, "pixel_y": 160.0, "pitch_x": x, "pitch_y": 30.0,
            "on_pitch": True, "calib_reset": f == 30,
        })
    df = pd.DataFrame(rows)
    speeds_flag, _ = MLAnalyzer(df.copy(), fps=FPS)._compute_player_step_speeds()
    speeds_no, spikes_no = MLAnalyzer(df.drop(columns=["calib_reset"]), fps=FPS)._compute_player_step_speeds()
    assert float(speeds_flag["dx"].abs().max()) < 1.0
    assert float(speeds_no["dx"].abs().max()) > 5.0 or len(spikes_no) > 0


def test_a2_3_ball_kinematic_filter_marks_reset_as_discontinuity():
    fids = np.arange(10, dtype=np.int64)
    xs = np.array([10.0 + 0.2 * i + (6.0 if i >= 5 else 0.0) for i in range(10)])
    ys = np.full(10, 30.0)
    kin_no = compute_ball_kinematic_filter(fids, xs, ys, None, FPS, True)
    kin = compute_ball_kinematic_filter(fids, xs, ys, None, FPS, True, reset_frames=np.array([5]))
    assert 5 not in set(kin_no["discontinuity_ends"].tolist())
    assert list(kin["discontinuity_starts"]) == [4]
    assert list(kin["discontinuity_ends"]) == [5]
    assert bool(kin["valid_ctrl_mask"][4]) and bool(kin["valid_ctrl_mask"][5])


def test_blend_step_is_limited_per_second_independent_of_fps():
    """Yumuşak geçişin kayma hızı saniye cinsinden sınırlı olmalı (duran oyuncuya sahte hız vermesin)."""
    h = HybridPitchCalibrator.__new__(HybridPitchCalibrator)
    h.blend_max_step_m = config.PITCH_CAL_BLEND_MAX_STEP_M
    h.blend_max_speed_mps = config.PITCH_CAL_BLEND_MAX_SPEED_MPS
    for fps in (25.0, 50.0, 60.0):
        h.fps = fps
        assert h.blend_step_m * fps <= config.PITCH_CAL_BLEND_MAX_SPEED_MPS + 1e-9
        assert h.blend_step_m <= config.PITCH_CAL_BLEND_MAX_STEP_M
    # Saniyelik sınır, duran oyuncu kabul ölçütünün (1 km/sa) altında
    assert config.PITCH_CAL_BLEND_MAX_SPEED_MPS * 3.6 < 1.0
