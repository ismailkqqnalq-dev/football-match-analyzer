"""
test_a2_footage_profile.py
--------------------------
Faz A2-5: Görüntü profili (sabit kamera / TV yayını) algılama ve davranışı.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from modules.perspective_transformer import PerspectiveTransformer
from modules.pitch_calibrator import HybridPitchCalibrator
from test_a2_jump_gating import (
    FPS,
    FRAME_H,
    FRAME_W,
    FakeFlow,
    NoisyKeypointCalibrator,
    _make_hybrid,
    _player_pitch,
    _true_player_pitch,
)

FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _run(hybrid, flow, frames, cut_frames=()):
    infos = []
    for f in frames:
        flow.scene_cut_detected = f in cut_frames
        m = hybrid.process_frame(f, FRAME)
        infos.append((m, dict(hybrid.frame_info)))
    return infos


def test_a2_5_auto_detects_fixed_and_stops_periodic_refresh():
    cal = NoisyKeypointCalibrator(sigma_px=2.0, seed=11)
    flow = FakeFlow()
    hybrid, pt = _make_hybrid(cal, flow, profile="auto")
    detect_frames = int(config.PROFILE_DETECT_SEC * FPS)

    _run(hybrid, flow, range(0, detect_frames + 1))
    assert hybrid.profile == "fixed"
    assert hybrid.profile_decided is True
    assert "kamera sabit" in hybrid.profile_reason
    assert hybrid._fixed_locked is True  # algılama penceresindeki adaylardan kilit
    assert hybrid.profile_events[-1]["to"] == "fixed"

    # Kilitten sonraki 10 sn'de model yalnızca sağlık kontrolü için ~1 kez çağrılır
    calls_before = cal.calls
    infos = _run(hybrid, flow, range(detect_frames + 1, detect_frames + 1 + int(10 * FPS)))
    assert cal.calls - calls_before <= 1
    assert {i["profile"] for _, i in infos} == {"fixed"}
    # Broadcast olsaydı 10 sn'de ~16 çağrı olurdu
    assert cal.calls - calls_before < int(10 * FPS) // hybrid.refresh_interval
    m, _ = infos[-1]
    assert np.linalg.norm(_player_pitch(pt, m) - _true_player_pitch()) < 0.5


def test_a2_5_forced_fixed_locks_most_consistent_candidate():
    cal = NoisyKeypointCalibrator(sigma_px=2.0, seed=4)
    flow = FakeFlow()
    hybrid, pt = _make_hybrid(cal, flow, profile="fixed")
    assert hybrid.profile == "fixed" and hybrid.profile_reason == "config"
    n_cand = hybrid.fixed_candidates_target
    _run(hybrid, flow, range(0, n_cand * hybrid.refresh_interval + 2))
    assert hybrid._fixed_locked is True
    assert cal.calls == n_cand
    _run(hybrid, flow, range(n_cand * hybrid.refresh_interval + 2, 600))
    health_frames = int(config.PITCH_CAL_FIXED_HEALTH_SEC * FPS)
    assert cal.calls <= n_cand + (600 // health_frames) + 1


def test_a2_5_panning_camera_detected_as_broadcast():
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=1)
    flow = FakeFlow(lambda t: np.array([3.0 * t, 0.0]))  # sürekli pan
    hybrid, _ = _make_hybrid(cal, flow, profile="auto")
    _run(hybrid, flow, range(0, 60))
    assert hybrid.profile == "broadcast"
    assert hybrid.profile_decided is True
    assert "kamera hareketi" in hybrid.profile_reason


def test_a2_5_scene_cut_in_detection_window_means_broadcast():
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=1)
    flow = FakeFlow()
    hybrid, _ = _make_hybrid(cal, flow, profile="auto")
    _run(hybrid, flow, range(0, 40), cut_frames={20})
    assert hybrid.profile == "broadcast"
    assert "sahne kesmesi" in hybrid.profile_reason


def test_a2_5_fixed_switches_to_broadcast_on_cut_one_way(capsys):
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=1)
    flow = FakeFlow()
    hybrid, _ = _make_hybrid(cal, flow, profile="auto")
    detect_frames = int(config.PROFILE_DETECT_SEC * FPS)
    _run(hybrid, flow, range(0, detect_frames + 1))
    assert hybrid.profile == "fixed"
    cut = detect_frames + 30
    _run(hybrid, flow, range(detect_frames + 1, detect_frames + 200), cut_frames={cut})
    assert hybrid.profile == "broadcast"
    ev = hybrid.profile_events[-1]
    assert ev == {"frame_idx": cut, "from": "fixed", "to": "broadcast", "reason": "sabit profilde sahne kesmesi"}
    assert "Görüntü profili: broadcast" in capsys.readouterr().out
    # Tek yönlü: sabit görüntü sürse de fixed'e geri dönülmez
    _run(hybrid, flow, range(detect_frames + 200, detect_frames + 600))
    assert hybrid.profile == "broadcast"
    summary = hybrid.summary()
    assert [e["to"] for e in summary["profile_events"]] == ["fixed", "broadcast"]


def test_a2_5_invalid_profile_rejected():
    pt = PerspectiveTransformer(source_points={}, quiet=True)
    with pytest.raises(ValueError):
        HybridPitchCalibrator(
            calibrator=NoisyKeypointCalibrator(), transformer=pt, motion_tracker=FakeFlow(),
            profile="drone",
        )


def test_a2_5_fixed_profile_survives_long_flow_jitter():
    """10 dk boyunca kare başına ±0,5 px flow titremesi sabit profili yayına çevirmemeli."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 0.5, size=(20000, 2))
    cal = NoisyKeypointCalibrator(sigma_px=1.0, seed=2)
    flow = FakeFlow(lambda t: jitter[t])
    hybrid, _ = _make_hybrid(cal, flow, profile="auto")
    _run(hybrid, flow, range(0, int(10 * 60 * FPS)))
    assert hybrid.profile == "fixed", hybrid.profile_events


def test_a2_5_fixed_profile_switches_on_real_pan():
    cal = NoisyKeypointCalibrator(sigma_px=0.0, seed=1)
    detect_frames = int(config.PROFILE_DETECT_SEC * FPS)
    start = detect_frames + 50
    flow = FakeFlow(lambda t: np.array([max(0, t - start) * 2.0, 0.0]))
    hybrid, _ = _make_hybrid(cal, flow, profile="auto")
    _run(hybrid, flow, range(0, detect_frames + 1))
    assert hybrid.profile == "fixed"
    _run(hybrid, flow, range(detect_frames + 1, start + 100))
    assert hybrid.profile == "broadcast"
    assert "net kayma" in hybrid.profile_events[-1]["reason"]
