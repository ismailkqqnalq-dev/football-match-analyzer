import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from modules.team_classifier import TeamClassifier


def _draw_player_patch(
    frame: np.ndarray,
    bbox,
    colors_bgr,
    pattern: str = "solid",
    brightness_scale: float = 1.0,
) -> None:
    """Sentetik oyuncu gövde yaması çizer (düz, dikey çubuklu veya yarı yarıya parçalı + gölge ölçekli)."""
    x1, y1, x2, y2 = map(int, bbox)
    bw = x2 - x1
    bh = y2 - y1
    jy1 = y1 + int(bh * 0.18)
    jy2 = y1 + int(bh * 0.52)
    jx1 = x1 + int(bw * 0.25)
    jx2 = x1 + int(bw * 0.75)

    def _scale(bgr):
        return tuple(int(np.clip(round(c * brightness_scale), 0, 255)) for c in bgr)

    c1 = _scale(colors_bgr[0])
    c2 = _scale(colors_bgr[1]) if len(colors_bgr) > 1 else c1

    if pattern == "solid":
        frame[jy1:jy2, jx1:jx2] = c1
    elif pattern == "stripes":
        stripe_w = max(2, (jx2 - jx1) // 6)
        for idx, cx in enumerate(range(jx1, jx2, stripe_w)):
            col = c1 if idx % 2 == 0 else c2
            frame[jy1:jy2, cx:min(jx2, cx + stripe_w)] = col
    elif pattern == "split":
        mid_x = (jx1 + jx2) // 2
        frame[jy1:jy2, jx1:mid_x] = c1
        frame[jy1:jy2, mid_x:jx2] = c2
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _make_pitch_frame(specs, width: int = 1000, height: int = 600) -> np.ndarray:
    frame = np.full((height, width, 3), (35, 140, 35), dtype=np.uint8)
    for sp in specs:
        _draw_player_patch(
            frame,
            sp["bbox"],
            sp["colors"],
            pattern=sp.get("pattern", "solid"),
            brightness_scale=sp.get("scale", 1.0),
        )
    return frame


# ----------------------------------------------------------------------
# R1: KALECİ TAKIM EŞLEŞTİRME TESTLERİ
# ----------------------------------------------------------------------
def test_r1_goalkeeper_matched_by_defensive_block_and_preserves_class_name(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    specs = []
    # Team A (kırmızı) sol yarı sahada: x=140..360
    for i in range(5):
        specs.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [140 + i * 45, 120, 180 + i * 45, 250],
            "colors": [(30, 30, 210)],
        })
    # Team B (mavi) sağ yarı sahada: x=600..820
    for i in range(5):
        specs.append({
            "track_id": i + 6, "class_name": "player",
            "bbox": [600 + i * 45, 320, 640 + i * 45, 450],
            "colors": [(210, 40, 30)],
        })
    # Kaleciler farklı kaleci forması giyiyor (yeşil ve turuncu)
    specs.append({
        "track_id": 101, "class_name": "goalkeeper",
        "bbox": [40, 220, 85, 350],
        "colors": [(40, 210, 40)],
    })
    specs.append({
        "track_id": 102, "class_name": "goalkeeper",
        "bbox": [910, 220, 955, 350],
        "colors": [(20, 150, 240)],
    })

    frame = _make_pitch_frame(specs)
    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)
    for _ in range(int(config.TEAM_CLASSIFIER_GK_LOCK_MIN_VOTES)):
        dets = [
            {"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None}
            for s in specs
        ]
        out = clf.assign_teams(dets, frame)

    team_a_label = out[0]["team"]
    team_b_label = out[5]["team"]
    assert team_a_label in ("Team A", "Team B")
    assert team_b_label in ("Team A", "Team B")
    assert team_a_label != team_b_label

    gk_left = next(d for d in out if d["track_id"] == 101)
    gk_right = next(d for d in out if d["track_id"] == 102)
    assert gk_left["class_name"] == "goalkeeper"
    assert gk_right["class_name"] == "goalkeeper"
    assert gk_left["team"] == team_a_label
    assert gk_right["team"] == team_b_label

    tmap = clf.get_team_map()
    assert tmap[101] == team_a_label
    assert tmap[102] == team_b_label


def test_r1_goalkeeper_follows_halftime_side_swap(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    def _build_half(a_on_left: bool):
        specs = []
        ax_base = 150 if a_on_left else 620
        bx_base = 620 if a_on_left else 150
        gk_a_x = 45 if a_on_left else 910
        gk_b_x = 910 if a_on_left else 45
        for i in range(5):
            specs.append({
                "track_id": i + 1, "class_name": "player",
                "bbox": [ax_base + i * 40, 120, ax_base + i * 40 + 35, 250],
                "colors": [(25, 25, 220)],  # Team A (kırmızı)
            })
            specs.append({
                "track_id": i + 6, "class_name": "player",
                "bbox": [bx_base + i * 40, 320, bx_base + i * 40 + 35, 450],
                "colors": [(240, 240, 240)],  # Team B (beyaz)
            })
        specs.append({
            "track_id": 101, "class_name": "goalkeeper",
            "bbox": [gk_a_x, 220, gk_a_x + 40, 350],
            "colors": [(40, 210, 40)],
        })
        specs.append({
            "track_id": 102, "class_name": "goalkeeper",
            "bbox": [gk_b_x, 220, gk_b_x + 40, 350],
            "colors": [(20, 160, 240)],
        })
        return specs, _make_pitch_frame(specs)

    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)

    # 1. Yarı: kaleci kilidi eşiği kadar kare (25 oy)
    specs1, frame1 = _build_half(a_on_left=True)
    for _ in range(int(config.TEAM_CLASSIFIER_GK_LOCK_MIN_VOTES)):
        dets1 = [{"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None} for s in specs1]
        out1 = clf.assign_teams(dets1, frame1)

    team_a_lbl = out1[0]["team"]
    team_b_lbl = out1[1]["team"]
    assert next(d for d in out1 if d["track_id"] == 101)["team"] == team_a_lbl
    assert next(d for d in out1 if d["track_id"] == 102)["team"] == team_b_lbl

    # 2. Yarı (devre arası taraf değişimi): 30 kare
    specs2, frame2 = _build_half(a_on_left=False)
    for _ in range(30):
        dets2 = [{"track_id": s["track_id"], "class_name": s["class_name"], "bbox": s["bbox"], "team": None} for s in specs2]
        out2 = clf.assign_teams(dets2, frame2)

    assert next(d for d in out2 if d["track_id"] == 101)["team"] == team_a_lbl
    assert next(d for d in out2 if d["track_id"] == 102)["team"] == team_b_lbl


# ----------------------------------------------------------------------
# R4: ÇUBUKLU / PARÇALI FORMA, GÖLGE DAYANIKLILIĞI VE GÜNCELLEME TESTLERİ
# ----------------------------------------------------------------------
def test_r4_striped_and_split_kits_classification(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", False)

    # Senaryo 1: Kırmızı-Beyaz çubuklu vs Düz Beyaz (eski ortalama renk bunları karıştırabiliyordu) + Sarı hakem
    specs = []
    for i in range(5):
        specs.append({
            "track_id": i + 1, "class_name": "person",
            "bbox": [40 + i * 80, 100, 96 + i * 80, 240],
            "colors": [(30, 30, 220), (245, 245, 245)],  # Kırmızı-Beyaz çubuklu
            "pattern": "stripes",
        })
    for i in range(5):
        specs.append({
            "track_id": i + 6, "class_name": "person",
            "bbox": [40 + i * 80, 300, 96 + i * 80, 440],
            "colors": [(245, 245, 245)],  # Düz Beyaz
            "pattern": "solid",
        })
    for i in range(2):
        specs.append({
            "track_id": i + 11, "class_name": "person",
            "bbox": [650 + i * 90, 200, 706 + i * 90, 340],
            "colors": [(0, 230, 245)],  # Sarı hakem
            "pattern": "solid",
        })

    frame = _make_pitch_frame(specs)
    clf = TeamClassifier(n_clusters=3, min_unique_tracks=12, min_samples_for_lock=2)
    dets = [{"track_id": s["track_id"], "class_name": "person", "bbox": s["bbox"], "team": None} for s in specs]
    out = clf.assign_teams(dets, frame)

    t1 = {d["team"] for d in out[:5]}
    t2 = {d["team"] for d in out[5:10]}
    t_ref = {d["team"] for d in out[10:]}
    assert len(t1) == 1 and t1 <= {"Team A", "Team B"}
    assert len(t2) == 1 and t2 <= {"Team A", "Team B"}
    assert t1 != t2
    assert t_ref == {"Referee"}

    # Senaryo 2: Sarı-Kırmızı yarı-parçalı vs Mavi-Siyah çubuklu (fine-tuned 2 küme)
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)
    specs_split = []
    for i in range(5):
        specs_split.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + i * 80, 100, 96 + i * 80, 240],
            "colors": [(0, 215, 245), (30, 30, 215)],  # Sarı-Kırmızı parçalı
            "pattern": "split",
        })
    for i in range(5):
        specs_split.append({
            "track_id": i + 6, "class_name": "player",
            "bbox": [520 + i * 80, 300, 576 + i * 80, 440],
            "colors": [(200, 40, 20), (25, 25, 25)],  # Mavi-Siyah çubuklu
            "pattern": "stripes",
        })
    frame_split = _make_pitch_frame(specs_split)
    clf2 = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)
    out2 = clf2.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in specs_split],
        frame_split,
    )
    a_set = {d["team"] for d in out2[:5]}
    b_set = {d["team"] for d in out2[5:10]}
    assert len(a_set) == 1 and len(b_set) == 1 and a_set != b_set


def test_r4_stadium_shadow_invariance(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    specs = []
    # 5 kırmızı oyuncu: 3 güneş (1.0x), 2 derin stadyum gölgesi (0.4x)
    for i in range(5):
        specs.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + i * 80, 100, 96 + i * 80, 240],
            "colors": [(30, 30, 220)],
            "scale": 1.0 if i < 3 else 0.4,
        })
    # 5 mavi oyuncu: 3 güneş (1.0x), 2 derin stadyum gölgesi (0.4x)
    for i in range(5):
        specs.append({
            "track_id": i + 6, "class_name": "player",
            "bbox": [520 + i * 80, 300, 576 + i * 80, 440],
            "colors": [(220, 50, 25)],
            "scale": 1.0 if i < 3 else 0.4,
        })

    frame = _make_pitch_frame(specs)
    clf = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)
    dets = [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in specs]
    out = clf.assign_teams(dets, frame)

    red_teams = {d["team"] for d in out[:5]}
    blue_teams = {d["team"] for d in out[5:10]}
    assert len(red_teams) == 1
    assert len(blue_teams) == 1
    assert red_teams != blue_teams


def test_r4_majority_vote_corrects_erroneous_initial_lock(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    base_specs = []
    for i in range(5):
        base_specs.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + i * 80, 100, 96 + i * 80, 240],
            "colors": [(30, 30, 220)],  # Kırmızı
        })
    for i in range(5):
        base_specs.append({
            "track_id": i + 6, "class_name": "player",
            "bbox": [520 + i * 80, 300, 576 + i * 80, 440],
            "colors": [(245, 245, 245)],  # Beyaz
        })
    frame_base = _make_pitch_frame(base_specs)

    clf = TeamClassifier(
        n_clusters=2,
        min_unique_tracks=10,
        min_samples_for_lock=3,
        refresh_interval=5,
        vote_window=7,
    )
    clf.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in base_specs],
        frame_base,
    )
    assert clf.is_fitted is True
    red_label = clf.team_assignment[1]
    white_label = clf.team_assignment[6]
    assert red_label != white_label

    # Yeni track 99 ilk 3 karede yanlış/gölgeli-örtüşük beyaz piksel görsün -> yanlış kilit!
    bad_frame = _make_pitch_frame(base_specs + [{
        "track_id": 99, "class_name": "player",
        "bbox": [200, 450, 256, 580],
        "colors": [(245, 245, 245)],
    }])
    for _ in range(3):
        clf.assign_teams([{"track_id": 99, "class_name": "player", "bbox": [200, 450, 256, 580], "team": None}], bad_frame)
    assert clf.team_assignment[99] == white_label
    assert 99 not in clf.color_history

    # Ardından 25 kare boyunca gerçek kırmızı formasıyla görünsün -> çoğunluk oylaması düzeltmeli
    good_frame = _make_pitch_frame(base_specs + [{
        "track_id": 99, "class_name": "player",
        "bbox": [200, 450, 256, 580],
        "colors": [(30, 30, 220)],
    }])
    for _ in range(25):
        out_99 = clf.assign_teams([{"track_id": 99, "class_name": "player", "bbox": [200, 450, 256, 580], "team": None}], good_frame)

    assert clf.team_assignment[99] == red_label
    assert out_99[0]["team"] == red_label


def test_r4_bounded_refit_recovers_skewed_initial_fit(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    # Başlangıçta 9 kırmızı + 1 mavi oyuncu ile çarpık ilk fit (kümeler başlangıçta kırmızı tonları bölebilir)
    skew_specs = []
    for i in range(9):
        skew_specs.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + (i % 5) * 80, 80 + (i // 5) * 160, 96 + (i % 5) * 80, 210 + (i // 5) * 160],
            "colors": [(30, 30, 220)],
            "scale": 1.0 if i < 5 else 0.45,
        })
    skew_specs.append({
        "track_id": 10, "class_name": "player",
        "bbox": [600, 300, 656, 440],
        "colors": [(220, 50, 25)],
        "scale": 1.0,
    })

    clf = TeamClassifier(
        n_clusters=2,
        min_unique_tracks=10,
        min_samples_for_lock=2,
        refresh_interval=5,
        vote_window=7,
        max_refits=3,
    )
    frame_skew = _make_pitch_frame(skew_specs)
    clf.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in skew_specs],
        frame_skew,
    )
    assert clf.is_fitted is True

    # Sonradan 10 yeni mavi ve kırmızı oyuncu + tekrar gözlemlerle sınırlı refit devreye girmeli
    extra_specs = list(skew_specs)
    for j in range(6):
        extra_specs.append({
            "track_id": 20 + j, "class_name": "player",
            "bbox": [520 + (j % 4) * 80, 80 + (j // 4) * 160, 576 + (j % 4) * 80, 210 + (j // 4) * 160],
            "colors": [(220, 50, 25)],
            "scale": 1.0,
        })
    frame_extra = _make_pitch_frame(extra_specs)
    for _ in range(8):
        out_all = clf.assign_teams(
            [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in extra_specs],
            frame_extra,
        )

    assert clf._refit_count >= 1
    red_labels = {d["team"] for d in out_all if d["track_id"] in range(1, 10)}
    blue_labels = {d["team"] for d in out_all if d["track_id"] in ({10} | set(range(20, 26)))}
    assert len(red_labels) == 1
    assert len(blue_labels) == 1
    assert red_labels != blue_labels


@pytest.mark.parametrize("shadow_scale", [0.45, 0.40, 0.35])
def test_r4_red_white_stripes_vs_solid_red_and_achromatic_under_deep_shadow(monkeypatch, shadow_scale):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    # Kırmızı-Beyaz çubuklu vs Düz Kırmızı (yarısı güneş, yarısı derin gölgede)
    specs_rw = []
    for i in range(6):
        specs_rw.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + i * 75, 80, 96 + i * 75, 220],
            "colors": [(30, 30, 220), (245, 245, 245)],
            "pattern": "stripes",
            "scale": 1.0 if i < 3 else shadow_scale,
        })
    for i in range(6):
        specs_rw.append({
            "track_id": i + 7, "class_name": "player",
            "bbox": [40 + i * 75, 300, 96 + i * 75, 440],
            "colors": [(30, 30, 220)],
            "pattern": "solid",
            "scale": 1.0 if i < 3 else shadow_scale,
        })

    frame_rw = _make_pitch_frame(specs_rw)
    clf_rw = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)
    out_rw = clf_rw.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in specs_rw],
        frame_rw,
    )
    stripes_teams = {d["team"] for d in out_rw[:6]}
    solid_teams = {d["team"] for d in out_rw[6:12]}
    assert len(stripes_teams) == 1
    assert len(solid_teams) == 1
    assert stripes_teams != solid_teams


def test_r4_white_vs_black_under_035_shadow(monkeypatch):
    monkeypatch.setattr(config, "USES_FINE_TUNED_FOOTBALL_MODEL", True)

    specs_wb = []
    for i in range(6):
        specs_wb.append({
            "track_id": i + 1, "class_name": "player",
            "bbox": [40 + i * 75, 80, 96 + i * 75, 220],
            "colors": [(245, 245, 245)],
            "pattern": "solid",
            "scale": 1.0 if i < 3 else 0.35,
        })
    for i in range(6):
        specs_wb.append({
            "track_id": i + 7, "class_name": "player",
            "bbox": [40 + i * 75, 300, 96 + i * 75, 440],
            "colors": [(25, 25, 25)],
            "pattern": "solid",
            "scale": 1.0 if i < 3 else 0.35,
        })

    frame_wb = _make_pitch_frame(specs_wb)
    clf_wb = TeamClassifier(n_clusters=2, min_unique_tracks=10, min_samples_for_lock=2)
    out_wb = clf_wb.assign_teams(
        [{"track_id": s["track_id"], "class_name": "player", "bbox": s["bbox"], "team": None} for s in specs_wb],
        frame_wb,
    )
    white_teams = {d["team"] for d in out_wb[:6]}
    black_teams = {d["team"] for d in out_wb[6:12]}
    assert len(white_teams) == 1
    assert len(black_teams) == 1
    assert white_teams != black_teams
