"""
pitch_line_refiner.py
---------------------
Faz A2: Saha çizgisine oturtma (line-based homography refinement).

Anahtar nokta modeli tabloyu kabaca bulur (tipik sapma 20-50 px). Bu modül
o kaba tabloyu başlangıç alır, karedeki beyaz saha çizgilerini maskeler ve
saha çiziminin (taç/kale çizgileri, orta çizgi, ceza/kale alanları, orta
yuvarlak, ceza yayları) görüntüdeki çizgilerin üstüne oturması için tabloyu
ince ayarlar.

Yöntem:
  1) Çizgi maskesi: yeşil zeminin (en büyük yeşil bölgenin dışbükey örtüsü)
     içinde, gri tonlu top-hat ile ince parlak yapılar.
  2) Uzaklık haritası: her pikselin en yakın çizgi pikseline uzaklığı.
  3) Saha çiziminden metre cinsinden örnek noktalar tabloyla görüntüye
     izdüşürülür; artık = o noktadaki uzaklık (kırpılmış). Tablo, 4 kontrol
     noktasının piksel konumlarıyla parametrelenir ve kaba-ince aşamalarda
     (bulanık -> keskin uzaklık haritası) gürbüz en küçük karelerle çözülür.
     Hız için iş, küçültülmüş karede yapılır.
  4) Birden çok başlangıç tablosu verilebilir; çizgiye en iyi oturan seçilir.
     Kabul: çizgiye oturan örnek oranı başlangıçtakinden yüksek ve yeterli
     olmalı; aksi halde başarısız döner ve çağıran kaba tabloyu kullanır.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Kaba -> ince aşamalar, tam çözünürlük (1080p) pikseli cinsinden: (bulanıklık sigma, artık tavanı)
_STAGES_PX = ((12.0, 60.0), (4.0, 20.0), (0.0, 8.0))


@dataclass
class LineRefineResult:
    success: bool
    homography_matrix: Optional[np.ndarray] = None  # piksel -> metre (tam çözünürlük)
    inlier_ratio_before: float = 0.0
    inlier_ratio_after: float = 0.0
    num_samples: int = 0
    candidate_index: int = -1
    coarse_score: float = float("inf")  # yalnızca kaba aşama modunda: düşük = daha iyi
    reason: str = ""
    # Doğrulama için: görünür örneklerin metre ve (yeni tabloyla) piksel konumları
    meter_pts: Optional[np.ndarray] = None
    pixel_pts: Optional[np.ndarray] = None


# Yön kutuları: çizgi pikselleri yönlerine göre ayrılır; örnek yalnızca kendi yönündeki piksellere çekilir
_N_BINS = 6


@dataclass
class _Prepared:
    dt: np.ndarray  # yön kutularının uzaklık haritaları alt alta (N_BINS*h, w), float32
    blurred: Tuple[np.ndarray, ...]  # aşama başına kırpılmış (bulanık) haritalar, aynı dizilimde
    caps: Tuple[float, ...]
    scale: float  # iş ölçeği / tam çözünürlük
    w: int
    h: int


def _segment(p0, p1, step: float) -> List[Tuple[float, float, float, float]]:
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(2, int(np.ceil(length / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    tx, ty = (p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length
    return [(p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]), tx, ty) for u in t]


def _arc(cx, cy, r, step: float, keep=None) -> List[Tuple[float, float, float, float]]:
    n = max(8, int(np.ceil(2 * np.pi * r / step)))
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = [(cx + r * np.cos(v), cy + r * np.sin(v), -np.sin(v), np.cos(v)) for v in a]
    return [p for p in pts if keep is None or keep(p)]


def pitch_line_samples(step: float = 0.5, with_tangents: bool = False):
    """IFAB saha çiziminden metre cinsinden örnek noktalar (N, 2); istenirse birim teğetler (N, 2)."""
    L = float(getattr(config, "PITCH_LENGTH_M", 105.0))
    W = float(getattr(config, "PITCH_WIDTH_M", 68.0))
    r = float(getattr(config, "CENTER_CIRCLE_RADIUS_M", 9.15))
    cy = W / 2.0
    pts: List[Tuple[float, float, float, float]] = []
    segs = [
        ((0, 0), (L, 0)), ((0, W), (L, W)),            # taç çizgileri
        ((0, 0), (0, W)), ((L, 0), (L, W)),            # kale çizgileri
        ((L / 2, 0), (L / 2, W)),                      # orta çizgi
    ]
    for x_goal, sgn in ((0.0, 1.0), (L, -1.0)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):  # ceza alanı, kale alanı
            x_in = x_goal + sgn * depth
            segs += [
                ((x_in, cy - half), (x_in, cy + half)),
                ((x_goal, cy - half), (x_in, cy - half)),
                ((x_goal, cy + half), (x_in, cy + half)),
            ]
    for p0, p1 in segs:
        pts += _segment(p0, p1, step)
    pts += _arc(L / 2, cy, r, step)
    pts += _arc(11.0, cy, r, step, keep=lambda p: p[0] > 16.5)
    pts += _arc(L - 11.0, cy, r, step, keep=lambda p: p[0] < L - 16.5)
    arr = np.asarray(pts, dtype=np.float64)
    return (arr[:, :2], arr[:, 2:]) if with_tangents else arr[:, :2]


_GREEN_LO = (30, 40, 30)
_GREEN_HI = (90, 255, 255)


def _green_hsv_range(hsv: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Çim rengi aralığı. Varsayılan (30,40,30)-(90,255,255); karede gece gölgesi (koyu) ya
    da soluk/kış çimi (düşük doygunluk) baskınsa doygunluk/parlaklık alt sınırları,
    baskın ton aralık kenarındaysa ton sınırları kareye göre gevşetilir. Aralık yalnızca
    genişler: normal çimde sonuç varsayılanla aynıdır.
    """
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cand = (h >= 20) & (h <= 100) & (sat >= 15) & (val >= 15)
    if cand.mean() < 0.15:  # karede çim baskın değil (yakın çekim vb.): varsayılan
        return _GREEN_LO, _GREEN_HI
    hue_peak = int(np.argmax(np.bincount(h[cand].ravel(), minlength=181)))
    s_lo = int(min(_GREEN_LO[1], max(15, 0.4 * float(np.median(sat[cand])))))
    v_lo = int(min(_GREEN_LO[2], max(15, 0.4 * float(np.median(val[cand])))))
    h_lo = int(max(20, min(_GREEN_LO[0], hue_peak - 10)))
    h_hi = int(min(100, max(_GREEN_HI[0], hue_peak + 10)))
    return (h_lo, s_lo, v_lo), (h_hi, 255, 255)


def field_mask(frame: np.ndarray) -> np.ndarray:
    """Oyun alanı maskesi: en büyük yeşil bölgenin dışbükey örtüsü (tribün/reklam dışarıda)."""
    H, W = frame.shape[:2]
    s = min(1.0, 180.0 / H)  # örtü kaba ölçekte bulunur, sonra büyütülür (hız)
    small = cv2.resize(frame, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, *_green_hsv_range(hsv))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    out = np.zeros((H, W), np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    if n <= 1:
        return out
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((labels == biggest).astype(np.uint8))
    hull = np.round(cv2.convexHull(pts).astype(np.float64) / s).astype(np.int32)
    cv2.fillConvexPoly(out, hull, 255)
    return out


def _tophat(frame: np.ndarray) -> np.ndarray:
    h = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k_top = max(5, int(round(h / 70)) | 1)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k_top, k_top)))


def line_mask(frame: np.ndarray, tophat: Optional[np.ndarray] = None) -> np.ndarray:
    """Oyun alanı içindeki ince parlak yapıların (saha çizgileri) ikili maskesi."""
    if tophat is None:
        tophat = _tophat(frame)
    thr = float(getattr(config, "PITCH_LINE_REFINE_TOPHAT_THRESHOLD", 18))
    return ((tophat > thr) & (field_mask(frame) > 0)).astype(np.uint8)


def line_orientation(tophat: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Verilen piksellerde çizgi yönü [0, pi) (yapı tensörü; çizgi, gradyana diktir)."""
    t = tophat.astype(np.float32)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, tophat.shape[0] / 360.0)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    grad = 0.5 * np.arctan2(2.0 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs])
    return np.mod(grad + np.pi / 2.0, np.pi)


def _angle_bin(theta: np.ndarray) -> np.ndarray:
    return np.round(np.mod(theta, np.pi) / (np.pi / _N_BINS)).astype(np.int64) % _N_BINS


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = cv2.remap(img, pts[:, 0].astype(np.float32).reshape(-1, 1), pts[:, 1].astype(np.float32).reshape(-1, 1),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.ravel().astype(np.float64)


def _project(G: np.ndarray, pts: np.ndarray) -> np.ndarray:
    hom = np.c_[pts, np.ones(len(pts))] @ G.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


class PitchLineRefiner:
    """Kaba piksel->metre tablosunu saha çizgilerine oturtarak iyileştirir."""

    def __init__(
        self,
        sample_step_m: Optional[float] = None,
        inlier_px: Optional[float] = None,
        min_inlier_ratio: Optional[float] = None,
        work_height: Optional[int] = None,
        min_samples: int = 60,
    ):
        step = float(sample_step_m if sample_step_m is not None else getattr(config, "PITCH_LINE_REFINE_SAMPLE_STEP_M", 1.0))
        self.samples_m, self.tangents_m = pitch_line_samples(step, with_tangents=True)
        self.inlier_px = float(inlier_px if inlier_px is not None
                               else getattr(config, "PITCH_LINE_REFINE_INLIER_PX", 3.0))
        self.min_inlier_ratio = float(min_inlier_ratio if min_inlier_ratio is not None
                                      else getattr(config, "PITCH_LINE_REFINE_MIN_INLIER_RATIO", 0.50))
        self.work_height = int(work_height if work_height is not None
                               else getattr(config, "PITCH_LINE_REFINE_WORK_HEIGHT", 540))
        self.min_samples = int(min_samples)

    # ----------------------------------------------------------------
    def prepare(self, frame: np.ndarray) -> Optional[_Prepared]:
        H, W = frame.shape[:2]
        s = min(1.0, self.work_height / float(H))
        small = cv2.resize(frame, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        tophat = _tophat(small)
        mask = line_mask(small, tophat)
        if int(mask.sum()) < 100:
            return None
        # Her çizgi pikseli yönüne en yakın iki kutuya girer (kutu aralığı = tolerans = 30 derece)
        ys, xs = np.nonzero(mask)
        u = line_orientation(tophat, ys, xs) / (np.pi / _N_BINS)
        b0 = np.floor(u).astype(np.int64) % _N_BINS
        b1 = (b0 + 1) % _N_BINS
        hh, ww = mask.shape
        dts = []
        for b in range(_N_BINS):
            nonline = np.ones((hh, ww), np.uint8)
            sel = (b0 == b) | (b1 == b)
            nonline[ys[sel], xs[sel]] = 0
            dts.append(cv2.distanceTransform(nonline, cv2.DIST_L2, 5))
        px_unit = H / 1080.0 * s  # 1080p pikselinin iş ölçeğindeki karşılığı
        blurred, caps = [], []
        for i, (sigma, cap) in enumerate(_STAGES_PX):
            c = cap * px_unit
            if i == 0:
                # Kaba aşama yönden bağımsız: tek bulanık harita yeterli (hız)
                dt_all = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
                blurred.append(np.minimum(cv2.GaussianBlur(dt_all, (0, 0), sigma * px_unit), c))
            else:
                stack = np.vstack(dts)
                f = cv2.GaussianBlur(stack, (0, 0), sigma * px_unit) if sigma > 0 else stack
                blurred.append(np.minimum(f, c))
            caps.append(c)
        # Kutular alt alta tek görüntüde: örnekleme tek remap çağrısıyla yapılır (hız)
        return _Prepared(np.vstack(dts), tuple(blurred), tuple(caps), s, small.shape[1], small.shape[0])

    # ----------------------------------------------------------------
    def refine(
        self,
        frame: np.ndarray,
        candidates: Union[np.ndarray, Sequence[np.ndarray]],
        prep: Optional[_Prepared] = None,
    ) -> LineRefineResult:
        """Bir ya da birkaç piksel->metre başlangıç tablosunu çizgilere oturtur; en iyisini döndürür."""
        if isinstance(candidates, np.ndarray) and candidates.ndim == 2:
            candidates = [candidates]
        cands = [np.asarray(c, dtype=np.float64) for c in candidates if c is not None]
        if not cands:
            return LineRefineResult(False, reason="no_candidates")
        if prep is None:
            prep = self.prepare(frame)
        if prep is None:
            return LineRefineResult(False, reason="no_line_pixels")

        # Her aday yalnızca kaba aşamada denenir; kaba uyumu en iyi olan tüm aşamalarla sürdürülür (hız)
        i_best, coarse_best, best_before = 0, None, 0.0
        if len(cands) > 1:
            coarse = [self._refine_one(prep, H0, coarse_only=True) for H0 in cands]
            best_before = max(r.inlier_ratio_before for r in coarse)
            ok = [i for i, r in enumerate(coarse) if r.success]
            if ok:
                i_best = min(ok, key=lambda i: coarse[i].coarse_score)
                coarse_best = coarse[i_best].homography_matrix
        best = self._refine_one(prep, coarse_best if coarse_best is not None else cands[i_best])
        best.candidate_index = i_best
        if len(cands) > 1:
            best.inlier_ratio_before = best_before  # kaba aşama öncesi, adayların en iyisi
        if best.success and best.inlier_ratio_after <= best.inlier_ratio_before:
            best.success, best.reason = False, "no_improvement"
            best.homography_matrix = best.meter_pts = best.pixel_pts = None
        return best

    # ----------------------------------------------------------------
    def _refine_one(self, prep: _Prepared, h_px_to_m: np.ndarray, coarse_only: bool = False) -> LineRefineResult:
        w, h, s = prep.w, prep.h, prep.scale
        Sc = np.diag([s, s, 1.0])
        H0 = h_px_to_m @ np.linalg.inv(Sc)  # iş ölçeği pikseli -> metre
        if not np.isfinite(H0).all():
            return LineRefineResult(False, reason="non_finite_input")
        # İşaret: kare merkezinde w > 0 olacak şekilde; böylece görünür metre noktaları G ile w > 0 verir
        if H0[2, 0] * (w / 2.0) + H0[2, 1] * (h / 2.0) + H0[2, 2] < 0:
            H0 = -H0
        try:
            G0 = np.linalg.inv(H0)  # metre -> iş pikseli
        except np.linalg.LinAlgError:
            return LineRefineResult(False, reason="singular_input")
        hom = np.c_[self.samples_m, np.ones(len(self.samples_m))] @ G0.T
        front = hom[:, 2] > 1e-9
        px0 = np.full((len(hom), 2), np.nan)
        px0[front] = hom[front, :2] / hom[front, 2:3]
        vis = front & (px0[:, 0] > 1) & (px0[:, 0] < w - 2) & (px0[:, 1] > 1) & (px0[:, 1] < h - 2)
        n_vis = int(vis.sum())
        if n_vis < self.min_samples:
            return LineRefineResult(False, num_samples=n_vis, reason="too_few_visible_samples")
        S = self.samples_m[vis]
        Tm = self.tangents_m[vis]
        P0 = px0[vis]

        # 4 kontrol noktası: kare köşelerine en yakın (birbirinden farklı) görünür örnekler
        ctrl_idx: List[int] = []
        for corner in ((0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)):
            d = np.hypot(P0[:, 0] - corner[0], P0[:, 1] - corner[1])
            d[ctrl_idx] = np.inf
            ctrl_idx.append(int(np.argmin(d)))
        if abs(cv2.contourArea(P0[ctrl_idx].astype(np.float32))) < 0.05 * w * h:
            return LineRefineResult(False, num_samples=n_vis, reason="degenerate_control_points")
        C_m = S[ctrl_idx].astype(np.float32)
        c0 = P0[ctrl_idx].ravel()

        def G_of(c):
            return cv2.getPerspectiveTransform(C_m, c.reshape(4, 2).astype(np.float32)).astype(np.float64)

        def inside(p):
            return np.isfinite(p).all(axis=1) & (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)

        thr = max(1.0, self.inlier_px * s)

        def bins_of(G):
            p = _project(G, S)
            q = _project(G, S + 0.25 * Tm)
            d = q - p
            b = _angle_bin(np.arctan2(d[:, 1], d[:, 0]))
            b[~np.isfinite(d).all(axis=1)] = 0
            return b

        def sample_maps(maps, p, b, fill):
            ok = inside(p)
            r = np.full(len(p), fill, dtype=np.float64)
            if ok.any():
                q = p[ok].copy()
                if maps.shape[0] > h:
                    q[:, 1] += b[ok] * h  # kutunun alt alta dizilmiş haritadaki satır kayması
                r[ok] = _bilinear(maps, q)
            return r

        def inlier_mask(G):
            return sample_maps(prep.dt, _project(G, S), bins_of(G), np.inf) <= thr

        def inlier_ratio(G):
            return float(np.mean(inlier_mask(G)))

        before = inlier_ratio(G0)
        c = c0.copy()
        prior_w = 0.02
        n_stages = 1 if coarse_only else len(prep.blurred)
        for fieldc, cap in list(zip(prep.blurred, prep.caps))[:n_stages]:
            b_stage = bins_of(G_of(c))  # aşama boyunca sabit: artık fonksiyonu pürüzsüz kalsın

            def resid(cv, fieldc=fieldc, cap=cap, b_stage=b_stage):
                try:
                    p = _project(G_of(cv), S)
                except cv2.error:
                    return np.full(len(S) + 8, cap)
                return np.r_[sample_maps(fieldc, p, b_stage, cap), prior_w * (cv - c0)]

            sol = least_squares(resid, c, method="trf", loss="soft_l1", f_scale=cap / 4.0,
                                diff_step=1e-3, xtol=1e-4, ftol=1e-4, max_nfev=60)
            c = sol.x
        G1 = G_of(c)
        if coarse_only:
            cap0 = prep.caps[0]
            score = float(np.mean(sample_maps(prep.blurred[0], _project(G1, S), bins_of(G1), cap0))) / cap0
            try:
                H1c = np.linalg.inv(np.linalg.inv(Sc) @ G1)
            except np.linalg.LinAlgError:  # numpy 1.x NaN/tekil matriste hata fırlatır (2.x NaN döndürür)
                return LineRefineResult(False, inlier_ratio_before=round(before, 4), num_samples=n_vis,
                                        reason="singular_output")
            return LineRefineResult(bool(np.isfinite(H1c).all()), homography_matrix=H1c,
                                    inlier_ratio_before=round(before, 4), num_samples=n_vis,
                                    coarse_score=score, reason="coarse")
        after = inlier_ratio(G1)
        res = LineRefineResult(False, inlier_ratio_before=round(before, 4), inlier_ratio_after=round(after, 4),
                               num_samples=n_vis)
        # Güvenlik: görünür örneklerin ortanca yer değiştirmesi (uç kontrol noktaları değil) sınırlı olmalı
        shift = np.linalg.norm(_project(G1, S) - P0, axis=1)
        med_shift = float(np.nanmedian(shift)) if np.isfinite(shift).any() else np.inf
        if med_shift > float(getattr(config, "PITCH_LINE_REFINE_MAX_SHIFT_FRAC", 0.10)) * float(np.hypot(w, h)):
            res.reason = "shift_too_large"
            return res
        if after < self.min_inlier_ratio:
            res.reason = "low_inlier_ratio"
            return res
        # Çizgiye oturan örnekler görüntüde tek bir doğru boyunca diziliyorsa tablo belirsizdir: oturtucu
        # sahanın birçok çizgisini tek bir görüntü çizgisine (ör. uzak taç çizgisi / reklam panosu) yığmış
        # olabilir. Oturan noktaların baskın doğrultuya dik yayılımı (küçük ana eksen std) yeterli olmalı.
        Q = _project(G1, S)[inlier_mask(G1)]
        min_spread = float(getattr(config, "PITCH_LINE_REFINE_MIN_INLIER_SPREAD", 0.03))
        if len(Q) < 3 or np.sqrt(max(float(np.linalg.eigvalsh(np.cov(Q.T))[0]), 0.0)) < min_spread * h:
            res.reason = "inliers_collinear"
            return res
        G1_full = np.linalg.inv(Sc) @ G1  # metre -> tam çözünürlük pikseli
        try:
            H1 = np.linalg.inv(G1_full)
        except np.linalg.LinAlgError:
            res.reason = "singular_output"
            return res
        res.success = True
        res.homography_matrix = H1 / H1[2, 2] if abs(H1[2, 2]) > 1e-12 else H1
        res.meter_pts = S
        res.pixel_pts = _project(G1_full, S)
        res.reason = "ok"
        return res
