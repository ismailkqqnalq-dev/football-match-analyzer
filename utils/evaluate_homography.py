"""
evaluate_homography.py
----------------------
Faz A2-6: Kalibrasyon isabetini ölçer.

Girdi:
  --log     main.py'nin yazdığı homography_log.csv (kare başına efektif piksel->metre tablosu)
  --labels  calibrate_homography.py --label-mode ile üretilen labels.csv
            (frame_idx, landmark_name, pixel_x, pixel_y)

Her etiketli karede etiketli pikseller o karenin tablosuyla sahaya izdüşürülür ve
gerçek metre konumuyla farkı alınır. Etiketlenip log'da tablosu olmayan kare
"tablo yok" sayılır (atlanmaz) ve geçerli tablo oranını düşürür.

Kullanım:
    python utils/evaluate_homography.py --log data/exports/homography_log.csv \\
        --labels data/labels/labels.csv [--json-out sonuc.json]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

H_COLS = [f"h{r}{c}" for r in range(3) for c in range(3)]
LABEL_COLUMNS = ("frame_idx", "landmark_name", "pixel_x", "pixel_y")


def label_landmarks_meters() -> Dict[str, tuple]:
    """Etiketlemede kullanılabilen tüm saha noktaları (manuel landmark'lar + 32 keypoint)."""
    out = {k: (float(v[0]), float(v[1])) for k, v in config.PITCH_LANDMARKS_METERS.items()}
    for name, pt in zip(config.PITCH_KEYPOINT_NAMES_32, config.PITCH_KEYPOINTS_32_METERS):
        p = (float(pt[0]), float(pt[1]))
        if name in out and (abs(out[name][0] - p[0]) > 1e-6 or abs(out[name][1] - p[1]) > 1e-6):
            raise ValueError(
                f"Saha noktası '{name}' iki tanımda farklı koordinatlara sahip: {out[name]} / {p}"
            )
        out[name] = p
    return out


def _percentile(vals: np.ndarray, q: float) -> Optional[float]:
    return None if vals.size == 0 else round(float(np.percentile(vals, q)), 4)


def evaluate(log_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, Any]:
    landmarks = label_landmarks_meters()
    missing_cols = [c for c in ("frame_idx", "source", *H_COLS) if c not in log_df.columns]
    if missing_cols:
        raise ValueError(f"homography_log eksik sütunlar: {missing_cols}")
    missing_cols = [c for c in LABEL_COLUMNS if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"labels eksik sütunlar: {missing_cols}")

    log_by_frame = {int(r.frame_idx): r for r in log_df.itertuples(index=False)}
    unknown = sorted(set(labels_df["landmark_name"]) - set(landmarks))

    frames_out = []
    all_err, src_err = [], {}
    invalid_total = 0
    labeled_frames = sorted(int(f) for f in labels_df["frame_idx"].unique())
    for f in labeled_frames:
        rows = labels_df[(labels_df["frame_idx"] == f) & labels_df["landmark_name"].isin(landmarks)]
        rec = log_by_frame.get(f)
        h = None
        source = None
        if rec is not None:
            source = str(rec.source)
            vals = np.array([getattr(rec, c) for c in H_COLS], dtype=np.float64)
            if np.isfinite(vals).all():
                h = vals.reshape(3, 3)
        if h is None:
            frames_out.append({"frame_idx": f, "status": "tablo yok", "source": source,
                               "n_points": int(len(rows))})
            continue
        px = rows[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        gt = np.array([landmarks[n] for n in rows["landmark_name"]], dtype=np.float64)
        homog = np.hstack([px, np.ones((len(px), 1))]) @ h.T
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = homog[:, :2] / homog[:, 2:3]
        err = np.linalg.norm(proj - gt, axis=1)
        n_invalid = int((~np.isfinite(err)).sum())  # ufkun ötesine düşen izdüşüm
        invalid_total += n_invalid
        err = err[np.isfinite(err)]
        all_err.extend(err.tolist())
        src_err.setdefault(source, []).extend(err.tolist())
        frames_out.append({
            "frame_idx": f,
            "status": "ok",
            "source": source,
            "n_points": int(len(err)),
            "invalid_projections": n_invalid,
            "mean_error_m": round(float(np.mean(err)), 4) if len(err) else None,
            "max_error_m": round(float(np.max(err)), 4) if len(err) else None,
        })

    all_arr = np.asarray(all_err, dtype=np.float64)
    n_labeled = len(labeled_frames)
    n_valid = sum(1 for fr in frames_out if fr["status"] == "ok")
    by_source = {}
    for src, errs in sorted(src_err.items()):
        arr = np.asarray(errs, dtype=np.float64)
        by_source[src] = {
            "frames": sum(1 for fr in frames_out if fr["status"] == "ok" and fr["source"] == src),
            "points": int(arr.size),
            "median_error_m": _percentile(arr, 50),
            "p90_error_m": _percentile(arr, 90),
        }
    return {
        "labeled_frames": n_labeled,
        "frames_with_table": n_valid,
        "frames_without_table": n_labeled - n_valid,
        "valid_table_ratio": round(n_valid / n_labeled, 4) if n_labeled else None,
        "points_evaluated": int(all_arr.size),
        "median_error_m": _percentile(all_arr, 50),
        "p90_error_m": _percentile(all_arr, 90),
        "by_source": by_source,
        "invalid_projections": invalid_total,
        "unknown_landmarks": unknown,
        "frames": frames_out,
    }


def format_summary_tr(res: Dict[str, Any]) -> str:
    def fmt(v):
        return "-" if v is None else f"{v:.2f} m"

    ratio = res["valid_table_ratio"]
    lines = [
        "KALİBRASYON İSABETİ",
        f"Etiketli kare: {res['labeled_frames']} | tablo olan: {res['frames_with_table']} | "
        f"tablo yok: {res['frames_without_table']} | geçerli tablo oranı: "
        + ("-" if ratio is None else f"%{ratio * 100:.1f}"),
        f"Ölçülen nokta: {res['points_evaluated']} | ortanca hata: {fmt(res['median_error_m'])} | "
        f"%90'lık dilim: {fmt(res['p90_error_m'])}",
    ]
    for src, s in res["by_source"].items():
        lines.append(
            f"  - {src}: {s['frames']} kare, {s['points']} nokta, ortanca {fmt(s['median_error_m'])}, "
            f"%90 {fmt(s['p90_error_m'])}"
        )
    if res["unknown_landmarks"]:
        lines.append(f"UYARI: tanınmayan nokta adları atlandı: {', '.join(res['unknown_landmarks'])}")
    lines.append(
        "Hedef: TV yayını < 1 m, sabit kamera < 0,5 m ortanca hata; etiketli karelerin >= %90'ında tablo."
    )
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Kalibrasyon isabeti ölçümü (A2-6)")
    p.add_argument("--log", required=True, help="homography_log.csv yolu")
    p.add_argument("--labels", required=True, help="labels.csv yolu")
    p.add_argument("--json-out", default=None, help="JSON sonuç yolu (varsayılan: log'un yanında)")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    res = evaluate(pd.read_csv(args.log), pd.read_csv(args.labels))
    print(format_summary_tr(res))
    out = args.json_out or os.path.join(
        os.path.dirname(os.path.abspath(args.log)), "homography_eval.json"
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"JSON: {out}")
    return res


if __name__ == "__main__":
    main()
