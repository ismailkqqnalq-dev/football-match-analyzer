"""
test_a2_evaluate_homography.py
------------------------------
Faz A2-6: Etiketleme yardımcıları ve kalibrasyon isabet ölçümü.
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.calibrate_homography as ch
import utils.evaluate_homography as ev
from config import config
from modules.pitch_calibrator import HomographyLogWriter


def _h_m_to_px():
    src = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    dst = np.array([[180.0, 160.0], [1100.0, 160.0], [1230.0, 630.0], [50.0, 630.0]])
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


H_M_TO_PX = _h_m_to_px()
H_PX_TO_M = np.linalg.inv(H_M_TO_PX)
LABEL_NAMES = [
    "top_left_corner", "halfway_top", "center_spot", "left_penalty_spot",
    "left_penalty_box_bottom_right", "right_six_yard_box_top_left",
]


def _labels(frames):
    lm = ev.label_landmarks_meters()
    rows = []
    for f in frames:
        for n in LABEL_NAMES:
            px = cv2.perspectiveTransform(np.array([[lm[n]]], dtype=np.float64), H_M_TO_PX)[0, 0]
            rows.append((f, n, px[0], px[1]))
    return rows


def _write_log(path, frames_h):
    with HomographyLogWriter(path) as log:
        for f, (src, h) in frames_h.items():
            log.write(f, {"source": src, "profile": "broadcast"}, h)


def test_a2_6_perfect_table_gives_zero_error(tmp_path):
    labels = tmp_path / "labels.csv"
    ch.write_labels_csv(str(labels), _labels([10, 20, 30]))
    log = tmp_path / "homography_log.csv"
    _write_log(log, {f: ("keypoint_refresh", H_PX_TO_M) for f in range(40)})
    res = ev.evaluate(pd.read_csv(log), pd.read_csv(labels))
    assert res["labeled_frames"] == 3
    assert res["valid_table_ratio"] == 1.0
    assert res["points_evaluated"] == 3 * len(LABEL_NAMES)
    assert res["median_error_m"] < 0.05  # piksel 0,1'e yuvarlandı
    assert res["by_source"]["keypoint_refresh"]["frames"] == 3


def test_a2_6_shifted_table_gives_two_meter_error_and_source_breakdown(tmp_path):
    labels = tmp_path / "labels.csv"
    ch.write_labels_csv(str(labels), _labels([5, 15]))
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    log = tmp_path / "homography_log.csv"
    _write_log(log, {5: ("flow", shift @ H_PX_TO_M), 15: ("blend", H_PX_TO_M)})
    res = ev.evaluate(pd.read_csv(log), pd.read_csv(labels))
    assert res["by_source"]["flow"]["median_error_m"] == pytest.approx(2.0, abs=0.05)
    assert res["by_source"]["blend"]["median_error_m"] < 0.05
    assert res["p90_error_m"] == pytest.approx(2.0, abs=0.05)


def test_a2_6_missing_frame_counts_as_no_table(tmp_path):
    labels = tmp_path / "labels.csv"
    ch.write_labels_csv(str(labels), _labels([1, 2, 3, 4]))
    log = tmp_path / "homography_log.csv"
    _write_log(log, {1: ("flow", H_PX_TO_M), 2: ("none", None), 4: ("flow", H_PX_TO_M)})
    res = ev.evaluate(pd.read_csv(log), pd.read_csv(labels))
    assert res["labeled_frames"] == 4
    assert res["frames_with_table"] == 2
    assert res["frames_without_table"] == 2  # 2: tablo boş, 3: log'da yok
    assert res["valid_table_ratio"] == 0.5
    statuses = {fr["frame_idx"]: fr["status"] for fr in res["frames"]}
    assert statuses == {1: "ok", 2: "tablo yok", 3: "tablo yok", 4: "ok"}


def test_a2_6_cli_writes_turkish_summary_and_json(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    rows = _labels([7]) + [(7, "olmayan_nokta", 10.0, 10.0)]
    ch.write_labels_csv(str(labels), rows)
    log = tmp_path / "homography_log.csv"
    _write_log(log, {7: ("keypoint_lock", H_PX_TO_M)})
    out_json = tmp_path / "eval.json"
    ev.main(["--log", str(log), "--labels", str(labels), "--json-out", str(out_json)])
    text = capsys.readouterr().out
    assert "KALİBRASYON İSABETİ" in text
    assert "ortanca hata" in text
    assert "olmayan_nokta" in text
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["unknown_landmarks"] == ["olmayan_nokta"]
    assert data["points_evaluated"] == len(LABEL_NAMES)


def test_a2_6_label_helpers():
    assert ch.resolve_label_frames(1000, 25.0, frames_arg="350, 100,100") == [100, 350]
    assert ch.resolve_label_frames(1000, 25.0, every_sec=15) == [0, 375, 750]
    with pytest.raises(ValueError):
        ch.resolve_label_frames(100, 25.0, frames_arg="50,200")
    with pytest.raises(ValueError):
        ch.resolve_label_frames(100, 25.0)
    names = ch.label_landmark_names()
    assert set(config.PITCH_KEYPOINT_NAMES_32) <= set(names)
    assert set(config.PITCH_LANDMARKS_METERS) <= set(names)
    assert ch.parse_landmark_answer("center_spot", names) == "center_spot"
    assert ch.parse_landmark_answer(str(names.index("halfway_top")), names) == "halfway_top"
    assert ch.parse_landmark_answer("", names) is None
    assert ch.parse_landmark_answer("yok_boyle", names) is None
    assert ch.parse_landmark_answer("9999", names) is None
