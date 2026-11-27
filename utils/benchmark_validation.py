"""
benchmark_validation.py
=======================
Bir pipeline çıktısını repodaki sabit doğrulama etiketlerine karşı tek komutta ölçer:
A2 kalibrasyon isabeti + Faz V pas / top kaybı / sahiplik (iki aralık). Model veya
algoritma değişikliklerinde "önce / sonra" karşılaştırması için kullanılır.

Kullanım (video repoda yok; aynı klibi kendiniz verin):
    python main.py --input tactical_10min.mp4 --no-video --export-dir $env:TEMP\\run_x
    python utils/benchmark_validation.py --run-dir $env:TEMP\\run_x
    python utils/benchmark_validation.py --run-dir $env:TEMP\\run_x --baseline $env:TEMP\\run_y\\benchmark.json
"""
import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluate_homography import evaluate as evaluate_homography  # noqa: E402
from utils import validate_events  # noqa: E402

DEFAULT_LABEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "validation", "tactical_10min"
)
# (ad, başlangıç sn, bitiş sn, olay dosyası, sahiplik dosyası)
EVENT_RANGES: List[Tuple[str, float, float, str, str]] = [
    ("0-120", 0.0, 120.0, "gt_events_000_120.csv", "gt_possession_000_120.csv"),
    ("120-240", 120.0, 240.0, "gt_events_120_240.csv", "gt_possession_120_240.csv"),
]
# Kabul hedefleri (geniş/sabit kamera)
A2_MEDIAN_TARGET_M = 0.5
A2_VALID_TARGET = 0.9
A2_STATIONARY_TARGET_KMH = 1.0


def compute_stationary_speed(
    log_df: pd.DataFrame,
    video_path: str,
    chunk_interval_sec: float = 12.0,
) -> Dict[str, Any]:
    """
    Sahadaki sabit çim/çizgi noktalarını 1 sn'lik parçalarda ardışık optik akış
    ve RANSAC arka plan ayıklamasıyla izleyip homografi üzerinden görünür hızlarının
    ortancasını (km/sa) ölçer. `blend_dx_m` / `blend_dy_m` varsa telafili hızı da verir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        chunk_frames = max(2, int(round(fps)))
        interval_frames = max(chunk_frames + 1, int(round(chunk_interval_sec * fps)))

        hcols = [f"h{r}{c}" for r in range(3) for c in range(3)]
        has_blend_cols = {"blend_dx_m", "blend_dy_m"}.issubset(log_df.columns)
        by_f: Dict[int, Tuple[str, np.ndarray, float, float]] = {}
        for row in log_df.itertuples(index=False):
            mat = np.array([float(getattr(row, c)) for c in hcols], dtype=np.float64).reshape(3, 3)
            bdx = float(getattr(row, "blend_dx_m") or 0.0) if has_blend_cols else 0.0
            bdy = float(getattr(row, "blend_dy_m") or 0.0) if has_blend_cols else 0.0
            by_f[int(row.frame_idx)] = (
                str(row.source),
                mat,
                bdx if np.isfinite(bdx) else 0.0,
                bdy if np.isfinite(bdy) else 0.0,
            )

        if not by_f:
            return {"stationary_speed_kmh": None, "stationary_speed_comp_kmh": None, "stationary_chunks": 0}
        max_logged_frame = max(by_f.keys())

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        chunks: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        start_f = int(round(4.0 * fps))
        if start_f + chunk_frames > max_logged_frame:
            start_f = 0
        while start_f + chunk_frames <= max_logged_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ok, f0 = cap.read()
            if not ok:
                break
            g_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
            pts0 = cv2.goodFeaturesToTrack(g_prev, maxCorners=80, qualityLevel=0.05, minDistance=25, mask=mask)
            if pts0 is not None and len(pts0) >= 8:
                p_cur = pts0.copy()
                alive = np.ones(len(pts0), dtype=bool)
                for _ in range(chunk_frames):
                    ok, fk = cap.read()
                    if not ok:
                        alive[:] = False
                        break
                    gk = cv2.cvtColor(fk, cv2.COLOR_BGR2GRAY)
                    p_next, st, _ = cv2.calcOpticalFlowPyrLK(g_prev, gk, p_cur, None, **lk_params)
                    p_back, stb, _ = cv2.calcOpticalFlowPyrLK(gk, g_prev, p_next, None, **lk_params)
                    fb_err = np.linalg.norm((p_cur - p_back).reshape(-1, 2), axis=1)
                    alive &= (st.ravel() == 1) & (stb.ravel() == 1) & (fb_err < 1.0)
                    p_cur = p_next
                    g_prev = gk
                if int(alive.sum()) >= 8:
                    a0 = pts0[alive].reshape(-1, 2)
                    a1 = p_cur[alive].reshape(-1, 2)
                    mat_bg, inl = cv2.findHomography(a0, a1, cv2.RANSAC, 3.0)
                    if mat_bg is not None and inl is not None and int(inl.sum()) >= 6:
                        im = inl.ravel().astype(bool)
                        chunks.append((start_f, start_f + chunk_frames, a0[im], a1[im]))
            start_f += interval_frames
    finally:
        cap.release()

    raw_speeds: List[float] = []
    comp_speeds: List[float] = []
    for f0, f1, p0, p1 in chunks:
        if f0 not in by_f or f1 not in by_f:
            continue
        srcs = [by_f[f][0] for f in range(f0, f1 + 1) if f in by_f]
        if any(s in ("keypoint_lock", "keypoint_reset") for s in srcs[1:]):
            continue
        H0, H1 = by_f[f0][1], by_f[f1][1]
        if not (np.isfinite(H0).all() and np.isfinite(H1).all()):
            continue
        q0 = np.c_[p0, np.ones(len(p0))] @ H0.T
        q1 = np.c_[p1, np.ones(len(p1))] @ H1.T
        if (np.abs(q0[:, 2]) < 1e-9).any() or (np.abs(q1[:, 2]) < 1e-9).any():
            continue
        m0 = q0[:, :2] / q0[:, 2:3]
        m1 = q1[:, :2] / q1[:, 2:3]
        dt = float(f1 - f0) / fps
        raw_sp = float(np.median(np.linalg.norm(m1 - m0, axis=1)) / dt * 3.6)
        sum_bx = sum(by_f[f][2] for f in range(f0 + 1, f1 + 1) if f in by_f)
        sum_by = sum(by_f[f][3] for f in range(f0 + 1, f1 + 1) if f in by_f)
        comp_diff = (m1 - m0) - np.array([sum_bx, sum_by], dtype=np.float64)
        comp_sp = float(np.median(np.linalg.norm(comp_diff, axis=1)) / dt * 3.6)
        if np.isfinite(raw_sp) and np.isfinite(comp_sp):
            raw_speeds.append(raw_sp)
            comp_speeds.append(comp_sp)

    return {
        "stationary_speed_kmh": round(float(np.median(raw_speeds)), 4) if raw_speeds else None,
        "stationary_speed_comp_kmh": round(float(np.median(comp_speeds)), 4) if comp_speeds else None,
        "stationary_chunks": len(raw_speeds),
    }


def _validate_range(run_dir: str, label_dir: str, start: float, end: float, ev: str, pos: str) -> Dict[str, Any]:
    argv = [
        "--summary", os.path.join(run_dir, "match_analytics_summary.json"),
        "--timeline", os.path.join(run_dir, "possession_timeline.csv"),
        "--gt-events", os.path.join(label_dir, ev),
        "--gt-possession", os.path.join(label_dir, pos),
        "--start-sec", str(start), "--end-sec", str(end),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        return validate_events.main(argv)


def run_benchmark(
    run_dir: str,
    label_dir: str = DEFAULT_LABEL_DIR,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_dir": os.path.abspath(run_dir)}
    log_path = os.path.join(run_dir, "homography_log.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        a2 = evaluate_homography(log_df, pd.read_csv(os.path.join(label_dir, "a2_labels.csv")))
        out["a2"] = {k: a2.get(k) for k in ("median_error_m", "p90_error_m", "valid_table_ratio", "points_evaluated")}
        if video_path and os.path.exists(video_path):
            out["a2"].update(compute_stationary_speed(log_df, video_path))
    if os.path.exists(os.path.join(run_dir, "match_analytics_summary.json")):
        out["faz_v"] = {}
        for name, start, end, ev, pos in EVENT_RANGES:
            r = _validate_range(run_dir, label_dir, start, end, ev, pos)
            bt = r["events"]["by_type"]
            out["faz_v"][name] = {
                "pass_f1": bt["completed_pass"]["f1"],
                "pass_precision": bt["completed_pass"]["precision"],
                "pass_recall": bt["completed_pass"]["recall"],
                "turnover_f1": bt["turnover"]["f1"],
                "possession_agreement": (r.get("possession") or {}).get("agreement"),
            }
    return out


def _fmt(v: Optional[float], unit: str = "") -> str:
    return "-" if v is None else f"{v:.2f}{unit}"


def format_table(res: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> str:
    rows: List[Tuple[str, Optional[float], Optional[float]]] = []
    a2, b_a2 = res.get("a2") or {}, (base or {}).get("a2") or {}
    rows.append(("A2 ortanca hata (m)", a2.get("median_error_m"), b_a2.get("median_error_m")))
    rows.append(("A2 %90 hata (m)", a2.get("p90_error_m"), b_a2.get("p90_error_m")))
    rows.append(("A2 geçerli tablo oranı", a2.get("valid_table_ratio"), b_a2.get("valid_table_ratio")))
    if a2.get("stationary_speed_kmh") is not None or b_a2.get("stationary_speed_kmh") is not None:
        rows.append(("A2 duran nokta hızı (km/sa)", a2.get("stationary_speed_kmh"), b_a2.get("stationary_speed_kmh")))
    if a2.get("stationary_speed_comp_kmh") is not None or b_a2.get("stationary_speed_comp_kmh") is not None:
        rows.append(("A2 duran nokta telafili hız (km/sa)", a2.get("stationary_speed_comp_kmh"), b_a2.get("stationary_speed_comp_kmh")))
    for name in (res.get("faz_v") or {}):
        cur, old = res["faz_v"][name], ((base or {}).get("faz_v") or {}).get(name, {})
        rows.append((f"Pas F1 {name}", cur["pass_f1"], old.get("pass_f1")))
        rows.append((f"Top kaybı F1 {name}", cur["turnover_f1"], old.get("turnover_f1")))
        rows.append((f"Sahiplik uyumu {name}", cur["possession_agreement"], old.get("possession_agreement")))
    lines = ["| Ölçüt | Bu çalıştırma |" + (" Karşılaştırma |" if base else ""),
             "|---|---|" + ("---|" if base else "")]
    for label, v, b in rows:
        lines.append(f"| {label} | {_fmt(v)} |" + (f" {_fmt(b)} |" if base else ""))
    med, valid = a2.get("median_error_m"), a2.get("valid_table_ratio")
    if med is not None:
        ok = med < A2_MEDIAN_TARGET_M and (valid or 0) >= A2_VALID_TARGET
        lines.append(f"\nA2 geniş kamera hedefi (< {A2_MEDIAN_TARGET_M} m, ≥ %{A2_VALID_TARGET * 100:.0f} tablo): "
                     + ("GEÇTİ" if ok else "KALDI"))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="A2 + Faz V tek komut ölçümü")
    ap.add_argument("--run-dir", required=True, help="main.py --export-dir çıktısı")
    ap.add_argument("--labels-dir", default=DEFAULT_LABEL_DIR)
    ap.add_argument("--video", help="Duran nokta hızı ölçümü için video dosyası (isteğe bağlı)")
    ap.add_argument("--baseline", help="Karşılaştırma için önceki benchmark.json")
    ap.add_argument("--json-out", help="Varsayılan: <run-dir>/benchmark.json")
    args = ap.parse_args(argv)

    res = run_benchmark(args.run_dir, args.labels_dir, video_path=args.video)
    base = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            base = json.load(f)
    print(format_table(res, base))
    out = args.json_out or os.path.join(args.run_dir, "benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"JSON: {out}")
    return res


if __name__ == "__main__":
    main()
