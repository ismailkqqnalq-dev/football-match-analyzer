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


def run_benchmark(run_dir: str, label_dir: str = DEFAULT_LABEL_DIR) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_dir": os.path.abspath(run_dir)}
    log_path = os.path.join(run_dir, "homography_log.csv")
    if os.path.exists(log_path):
        a2 = evaluate_homography(pd.read_csv(log_path), pd.read_csv(os.path.join(label_dir, "a2_labels.csv")))
        out["a2"] = {k: a2.get(k) for k in ("median_error_m", "p90_error_m", "valid_table_ratio", "points_evaluated")}
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
    ap.add_argument("--baseline", help="Karşılaştırma için önceki benchmark.json")
    ap.add_argument("--json-out", help="Varsayılan: <run-dir>/benchmark.json")
    args = ap.parse_args(argv)

    res = run_benchmark(args.run_dir, args.labels_dir)
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
