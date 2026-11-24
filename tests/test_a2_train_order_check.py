"""
test_a2_train_order_check.py
----------------------------
Faz A2-7: Eğitim öncesi nokta sırası doğrulaması (kpt_shape, flip_idx, görsel kontrol).
"""

import os
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.train_pitch_keypoints as tpk
from config import config


def _write_yaml(path, **kw):
    data = {"names": ["pitch"], "kpt_shape": [32, 3]}
    data.update(kw)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a2_7_wrong_kpt_shape_gives_clear_error(tmp_path):
    p = _write_yaml(tmp_path / "data.yaml", kpt_shape=[30, 3])
    with pytest.raises(ValueError) as exc:
        tpk.validate_pose_data_yaml(p, expected_num_keypoints=32)
    assert "kpt_shape" in str(exc.value)


def test_a2_7_template_flip_idx_is_mirror_involution():
    flip = tpk.template_flip_idx()
    assert flip is not None and len(flip) == 32
    pts = np.asarray(config.PITCH_KEYPOINTS_32_METERS)
    for i, j in enumerate(flip):
        assert flip[j] == i  # çevirmenin çevirmesi kendisi
        assert pts[j][0] == pytest.approx(config.PITCH_LENGTH_M - pts[i][0])
        assert pts[j][1] == pytest.approx(pts[i][1])
    names = config.PITCH_KEYPOINT_NAMES_32
    assert names[flip[names.index("top_left_corner")]] == "top_right_corner"
    assert names[flip[names.index("left_penalty_spot")]] == "right_penalty_spot"


def test_a2_7_missing_flip_idx_is_generated_into_data_yaml(tmp_path):
    p = _write_yaml(tmp_path / "data.yaml")
    meta = tpk.validate_pose_data_yaml(p)
    pol = tpk.resolve_flip_policy(meta, p, fliplr=0.5)
    assert pol["action"] == "generated"
    assert pol["fliplr"] == 0.5
    written = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert written["flip_idx"] == tpk.template_flip_idx()
    assert tpk.validate_pose_data_yaml(p)["flip_idx"] == written["flip_idx"]


def test_a2_7_mismatching_dataset_flip_idx_disables_flip_and_warns(tmp_path):
    p = _write_yaml(tmp_path / "data.yaml", flip_idx=list(range(32)))
    meta = tpk.validate_pose_data_yaml(p)
    pol = tpk.resolve_flip_policy(meta, p, fliplr=0.5)
    assert pol["action"] == "dataset_mismatch"
    assert pol["fliplr"] == 0.0
    assert "nokta sırası" in pol["message"]


def test_a2_7_matching_dataset_flip_idx_kept_and_zero_fliplr(tmp_path):
    p = _write_yaml(tmp_path / "data.yaml", flip_idx=tpk.template_flip_idx())
    pol = tpk.resolve_flip_policy(tpk.validate_pose_data_yaml(p), p, fliplr=0.5)
    assert pol["action"] == "dataset_ok" and pol["fliplr"] == 0.5
    p2 = _write_yaml(tmp_path / "d2.yaml")
    pol2 = tpk.resolve_flip_policy(tpk.validate_pose_data_yaml(p2), p2, fliplr=0.0)
    assert pol2["action"] == "not_needed"
    assert "flip_idx" not in yaml.safe_load(p2.read_text(encoding="utf-8"))


def test_a2_7_order_check_png_drawn_with_our_names(tmp_path):
    ds = tmp_path / "ds"
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "train" / "labels").mkdir(parents=True)
    img = np.full((360, 640, 3), (40, 140, 40), dtype=np.uint8)
    cv2.imwrite(str(ds / "train" / "images" / "frame_a.jpg"), img)
    kpts = []
    for i in range(32):
        vis = 2 if i < 6 else 0
        kpts += [0.1 + 0.02 * i, 0.5, vis] if vis else [0.0, 0.0, 0]
    line = "0 0.5 0.5 1.0 1.0 " + " ".join(f"{v:.4f}" for v in kpts)
    (ds / "train" / "labels" / "frame_a.txt").write_text(line + "\n", encoding="utf-8")

    xy, vis = tpk.read_pose_label(ds / "train" / "labels" / "frame_a.txt", 640, 360, 32)
    assert int(vis.sum()) == 6
    assert xy[0] == pytest.approx([64.0, 180.0])

    out_dir = tmp_path / "order"
    pngs = tpk.write_order_check_images(ds, out_dir)
    assert len(pngs) == 1 and pngs[0].is_file()
    drawn = cv2.imread(str(pngs[0]))
    assert drawn.shape == img.shape
    assert not np.array_equal(drawn, cv2.imread(str(ds / "train" / "images" / "frame_a.jpg")))


def test_a2_7_check_order_only_flag_parsed():
    args = tpk.parse_args(["--check-order-only", "--fliplr", "0"])
    assert args.check_order_only is True
    assert args.fliplr == 0.0
    assert not hasattr(args, "api_key")  # anahtar asla argümanla alınmaz
