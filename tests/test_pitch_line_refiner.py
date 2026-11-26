"""Saha çizgisine oturtma (modules/pitch_line_refiner.py) testleri — sentetik saha görüntüsüyle."""

import cv2
import numpy as np
import pytest

from config import config
from modules.pitch_calibrator import AutoPitchCalibrator
from modules.pitch_line_refiner import PitchLineRefiner, pitch_line_samples

W, H = 1920, 1080
# Gerçek tablo (metre -> piksel): orta saha ağırlıklı bir yayın açısı
_M_QUAD = np.float32([[35, 0], [80, 0], [85, 68], [30, 68]])
_P_QUAD = np.float32([[150, 120], [1750, 120], [2200, 1050], [-300, 1050]])
G_TRUE = cv2.getPerspectiveTransform(_M_QUAD, _P_QUAD).astype(np.float64)
H_TRUE = np.linalg.inv(G_TRUE)  # piksel -> metre


def _project(G, pts):
    hom = np.c_[pts, np.ones(len(pts))] @ G.T
    return hom[:, :2] / hom[:, 2:3]


def _render_pitch(G):
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 140, 40)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    img[:60] = (90, 60, 50)  # tribün şeridi (yeşil değil)
    for p in _project(G, pitch_line_samples(0.05)):
        if -5 <= p[0] < W + 5 and 60 <= p[1] < H + 5:
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2, (235, 235, 235), -1)
    return img


def _perturbed_h(shift_px):
    """Gerçek tabloyu 4 köşede shift_px kadar bozar (model sapmasını taklit eder)."""
    offs = np.float32([[shift_px, -shift_px], [-shift_px, shift_px], [shift_px, shift_px], [-shift_px, -shift_px]])
    G = cv2.getPerspectiveTransform(_M_QUAD, _P_QUAD + offs).astype(np.float64)
    return np.linalg.inv(G)


def _err_m(h_px_to_m, meter_pts):
    px = _project(G_TRUE, meter_pts)
    return np.linalg.norm(_project(h_px_to_m, px) - meter_pts, axis=1)


CHECK_PTS = np.array([[52.5, 0.0], [52.5, 24.85], [52.5, 43.15], [43.35, 34.0], [61.65, 34.0], [70.0, 60.0]])


@pytest.fixture(scope="module")
def pitch_img():
    return _render_pitch(G_TRUE)


def test_refine_recovers_true_homography(pitch_img):
    h0 = _perturbed_h(30.0)
    assert np.median(_err_m(h0, CHECK_PTS)) > 0.5  # başlangıç gerçekten bozuk
    res = PitchLineRefiner().refine(pitch_img, h0)
    assert res.success, res.reason
    assert res.inlier_ratio_after > res.inlier_ratio_before
    assert np.max(_err_m(res.homography_matrix, CHECK_PTS)) < 0.3


def test_refine_picks_best_of_multiple_candidates(pitch_img):
    bad = _perturbed_h(90.0)
    good = _perturbed_h(20.0)
    res = PitchLineRefiner().refine(pitch_img, [bad, good])
    assert res.success, res.reason
    assert np.max(_err_m(res.homography_matrix, CHECK_PTS)) < 0.3


def test_refine_fails_cleanly_without_lines():
    blank = np.zeros((H, W, 3), np.uint8)
    blank[:] = (40, 140, 40)
    res = PitchLineRefiner().refine(blank, _perturbed_h(20.0))
    assert not res.success
    assert res.reason == "no_line_pixels"
    assert res.homography_matrix is None


def test_pitch_line_samples_tangents_are_unit():
    pts, tan = pitch_line_samples(1.0, with_tangents=True)
    assert pts.shape == tan.shape
    assert np.allclose(np.linalg.norm(tan, axis=1), 1.0)


def _noisy_keypoints():
    names = list(config.PITCH_KEYPOINT_NAMES_32)
    meters = np.asarray(config.PITCH_KEYPOINTS_32_METERS, dtype=np.float64)
    px = _project(G_TRUE, meters)
    vis = [i for i, p in enumerate(px) if 0 <= p[0] < W and 60 <= p[1] < H]
    rng = np.random.default_rng(1)
    noisy = px[vis] + rng.uniform(-15, 15, (len(vis), 2))  # model sapması
    assert len(vis) >= 4, [names[i] for i in vis]
    return noisy, meters[vis], np.asarray(vis)


def test_calibrator_applies_line_refinement(pitch_img):
    cal = AutoPitchCalibrator(model=object(), line_refine=True)
    cal.detect_keypoints = lambda frame: _noisy_keypoints()
    res = cal.calibrate_frame(pitch_img)
    assert res.success, res.reason
    assert res.reason == "ok_line_refined"
    assert res.line_inlier_ratio is not None and res.line_inlier_ratio >= 0.5
    assert np.max(_err_m(res.homography_matrix, CHECK_PTS)) < 0.3


def test_calibrator_line_refinement_can_be_disabled(pitch_img):
    cal = AutoPitchCalibrator(model=object(), line_refine=False)
    assert cal.line_refiner is None
    cal.detect_keypoints = lambda frame: _noisy_keypoints()
    res = cal.calibrate_frame(pitch_img)
    assert res.reason != "ok_line_refined"


def test_calibrator_rejects_table_that_does_not_fit_lines():
    """Model sahayı yanlış yorumlarsa (ör. ceza sahasını yanlış uca koymak) tablo reddedilmeli."""
    # Saha yalnızca orta yuvarlak çevresini gösteriyor; model ise noktaları 25 m kaymış bir sahaya göre veriyor
    img = _render_pitch(G_TRUE)
    names = list(config.PITCH_KEYPOINT_NAMES_32)
    meters = np.asarray(config.PITCH_KEYPOINTS_32_METERS, dtype=np.float64)
    shifted = meters.copy()
    shifted[:, 0] += 25.0  # yanlış eşleşme: her nokta 25 m ötedeki noktanın piksel konumunda görülüyor
    px = _project(G_TRUE, shifted)
    vis = [i for i, p in enumerate(px) if 0 <= p[0] < W and 60 <= p[1] < H]
    assert len(vis) >= 4, [names[i] for i in vis]
    cal = AutoPitchCalibrator(model=object(), line_refine=True)
    cal.detect_keypoints = lambda frame: (px[vis], meters[vis], np.asarray(vis))
    res = cal.calibrate_frame(img)
    assert not res.success
    assert res.reason == "line_check_failed"


def test_calibrator_does_not_refine_from_collinear_keypoints(pitch_img):
    """Doğrusal noktalardan tablo kurulamaz; oturtucu bunlardan 'çizgiye uyan' bir tablo üretmemeli."""
    meters = np.array([[52.5, 0.0], [52.5, 24.85], [52.5, 43.15], [52.5, 68.0]])
    px = _project(G_TRUE, meters)
    cal = AutoPitchCalibrator(model=object(), line_refine=True)
    cal.detect_keypoints = lambda frame: (px, meters, np.arange(len(meters)))
    res = cal.calibrate_frame(pitch_img)
    assert not res.success
    assert res.reason == "collinear_keypoints"


def _cal_with_stub_refine(refine_result):
    from modules.pitch_line_refiner import LineRefineResult  # noqa: F401  (tip açıklığı için)
    cal = AutoPitchCalibrator(model=object(), line_refine=True)
    cal.detect_keypoints = lambda frame: _noisy_keypoints()
    cal.line_refiner.refine = lambda frame, cands, prep=None: refine_result
    return cal


def test_calibrator_rejects_model_table_when_refine_fails_unexpectedly(pitch_img):
    """Oturtma 'görünür örnek yok' gibi bir sebeple başarısız olunca model tablosu sınanmadan kabul edilmemeli."""
    from modules.pitch_line_refiner import LineRefineResult
    cal = _cal_with_stub_refine(LineRefineResult(False, inlier_ratio_before=0.05, reason="too_few_visible_samples"))
    res = cal.calibrate_frame(pitch_img)
    assert not res.success
    assert res.reason == "line_check_failed"


def test_calibrator_rejects_model_table_when_rejected_refine_fit_well(pitch_img):
    """TV kare 1430: oturtma shift_too_large ile reddedildi (sonra 0,73) ama model tablosu çizgilerin
    yalnızca %7'sine uyuyor; reddedilen oturtmanın oranı model tablosunu geçirmemeli."""
    from modules.pitch_line_refiner import LineRefineResult
    cal = _cal_with_stub_refine(LineRefineResult(False, inlier_ratio_before=0.071, inlier_ratio_after=0.732,
                                                 reason="shift_too_large"))
    res = cal.calibrate_frame(pitch_img)
    assert not res.success
    assert res.reason == "line_check_failed"


def test_calibrator_keeps_model_table_when_no_lines_visible(pitch_img):
    """Görüntüde çizgi hiç yoksa uyum yargılanamaz; model tablosu korunur."""
    from modules.pitch_line_refiner import LineRefineResult
    cal = _cal_with_stub_refine(LineRefineResult(False, reason="no_line_pixels"))
    res = cal.calibrate_frame(pitch_img)
    assert res.success
    assert res.reason == "ok"


def test_calibrator_rejects_model_table_when_refined_table_fails_checks(pitch_img):
    """Oturtulan tablo doğrulamadan geçmezse, çizgilere uymayan model tablosuna geri dönülmemeli."""
    from modules.pitch_line_refiner import LineRefineResult
    bad = LineRefineResult(
        True, homography_matrix=np.eye(3), inlier_ratio_before=0.1, inlier_ratio_after=0.9,
        meter_pts=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        pixel_pts=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),  # doğrusal: doğrulamadan geçmez
    )
    res = _cal_with_stub_refine(bad).calibrate_frame(pitch_img)
    assert not res.success
    assert res.reason == "line_check_failed"


def test_refined_table_verification_is_fast_with_many_line_samples(pitch_img):
    """Oturtucu yüzlerce örnek döndürse de doğrulama hızlı bitmeli.

    Veri: gerçek bir TV karesinde (test_match) oturtucunun döndürdüğü 716 çizgi örneği. Hepsi doğrulamaya
    verildiğinde genel konum araması (en kötü durumda N^4) dakikalarca sürüyor ve akış takılıyordu.
    """
    import os
    import time
    from modules.pitch_line_refiner import LineRefineResult
    d = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "line_refine_verify_hang.npz"))
    many = LineRefineResult(True, homography_matrix=H_TRUE, inlier_ratio_before=0.1, inlier_ratio_after=0.9,
                            meter_pts=d["m"], pixel_pts=d["px"])
    cal = _cal_with_stub_refine(many)
    t0 = time.perf_counter()
    res = cal.calibrate_frame(pitch_img)
    assert time.perf_counter() - t0 < 10.0
    assert res.reason in ("line_check_failed", "ok", "ok_line_refined")


def test_refine_rejects_table_whose_inliers_collapse_onto_one_line(monkeypatch):
    """Çizgiye oturan örnekler tek bir görüntü çizgisine yığılırsa tablo reddedilmeli.

    Veri: gerçek bir TV karesi (ceza yayını orta yuvarlak sanan model). Oturtucu sahanın birçok çizgisini
    uzak taç çizgisi / reklam panosu hattına yığıp yüksek oran buluyordu; tablo ~28 m yanlıştı.
    Fikstür: 960x540 karenin çizgi maskesi (bit paketli) ve oturtucuya giden aday tablolar.
    """
    import os
    d = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "line_refine_collapse_tv4475.npz"))
    mask = np.unpackbits(d["line_mask_bits"])[:540 * 960].reshape(540, 960)
    img = np.full((540, 960, 3), (40, 140, 40), np.uint8)
    img[mask > 0] = 255
    cands = list(d["candidates"])

    res = PitchLineRefiner().refine(img, cands)
    assert not res.success
    assert res.reason == "inliers_collinear"
    assert res.inlier_ratio_after >= PitchLineRefiner().min_inlier_ratio  # oran tek başına yakalamıyor

    monkeypatch.setattr(config, "PITCH_LINE_REFINE_MIN_INLIER_SPREAD", 0.0)
    assert PitchLineRefiner().refine(img, cands).success  # reddeden yalnızca yayılım koruması


def test_field_mask_adapts_to_dark_and_pale_grass():
    import cv2 as _cv2
    import numpy as _np

    from modules.pitch_line_refiner import _GREEN_HI, _GREEN_LO, _green_hsv_range, field_mask

    def frame_hsv(h, s, v):
        hsv = _np.zeros((180, 320, 3), _np.uint8)
        hsv[30:, :] = (h, s, v)  # üstte koyu tribün şeridi, altta çim
        return _cv2.cvtColor(hsv, _cv2.COLOR_HSV2BGR)

    # Normal çim: varsayılan aralık aynen kalır
    normal = _cv2.cvtColor(frame_hsv(55, 140, 130), _cv2.COLOR_BGR2HSV)
    assert _green_hsv_range(normal) == (_GREEN_LO, _GREEN_HI)

    # Gece gölgesi (V=24) ve soluk kış çimi (S=28): sabit aralık hiçbir şey bulamazdı
    for h, s, v in ((55, 140, 24), (50, 28, 150)):
        bgr = frame_hsv(h, s, v)
        hsv = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2HSV)
        assert _cv2.inRange(hsv, _GREEN_LO, _GREEN_HI).mean() == 0
        assert (field_mask(bgr) > 0).mean() > 0.7
