"""
validate_events.py
------------------
Faz V: Sistemin ürettiği olayları (pas / top kaybı) ve top sahipliğini elle
etiketlenmiş gerçek değerlerle karşılaştırır.

Girdi:
  --summary     main.py'nin yazdığı match_analytics_summary.json (passes.events, fps)
  --timeline    possession_timeline.csv (timestamp_sec, smoothed_team) [isteğe bağlı]
  --gt-events   Etiket CSV'si: time_sec, event_type (completed_pass | turnover), team
                (utils/label_events.html ile tarayıcıda üretilir)
  --gt-possession  Etiket CSV'si: start_sec, end_sec, team (team boş = kimsede değil) [isteğe bağlı]

Eşleştirme: aynı türdeki bir sistem olayı ile bir etiket, zaman farkı
--tolerance (varsayılan 1 sn) içindeyse eşleşir. Her etiket en fazla bir
sistem olayıyla eşleşir; en yakın çiftler önce eşlenir. Sistem olayının zamanı
topun ayaktan çıktığı an (start_frame / fps) kabul edilir.

Sahiplik uyumu: her tam saniye için sistemin o saniyedeki çoğunluk takımı ile
etiketteki takım karşılaştırılır; ikisinde de takım olan saniyeler sayılır.

Kullanım:
    python utils/validate_events.py --summary out/match_analytics_summary.json \\
        --timeline out/possession_timeline.csv --gt-events gt_events.csv \\
        --gt-possession gt_possession.csv [--json-out sonuc.json]
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EVENT_TYPES = ("completed_pass", "turnover")


def load_system_events(summary: Dict[str, Any]) -> pd.DataFrame:
    """match_analytics_summary.json içeriğinden (time_sec, event_type, team) tablosu üretir."""
    fps = float(summary.get("fps") or 0)
    if fps <= 0:
        raise ValueError("summary içinde geçerli fps yok")
    events = (summary.get("passes") or {}).get("events") or []
    rows = [
        {
            "time_sec": float(e["start_frame"]) / fps,
            "event_type": str(e["event_type"]),
            "team": e.get("from_team"),
        }
        for e in events
    ]
    return pd.DataFrame(rows, columns=["time_sec", "event_type", "team"])


def match_events(
    system_times: List[float], gt_times: List[float], tolerance: float
) -> List[Tuple[int, int]]:
    """Zaman farkı tolerans içindeki çiftleri, en yakından başlayarak bire bir eşler."""
    pairs = sorted(
        (abs(s - g), i, j)
        for i, s in enumerate(system_times)
        for j, g in enumerate(gt_times)
        if abs(s - g) <= tolerance
    )
    used_s, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_s and j not in used_g:
            used_s.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def _prf(tp: int, n_sys: int, n_gt: int) -> Dict[str, Any]:
    precision = tp / n_sys if n_sys else None
    recall = tp / n_gt if n_gt else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if n_sys and n_gt else None)
    )
    r = lambda v: round(v, 4) if v is not None else None
    return {"tp": tp, "system": n_sys, "ground_truth": n_gt,
            "precision": r(precision), "recall": r(recall), "f1": r(f1)}


def evaluate_events(
    system: pd.DataFrame,
    gt: pd.DataFrame,
    tolerance: float = 1.0,
    time_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Olay türü başına ve toplamda kesinlik / duyarlılık / F1 hesaplar.

    time_range verilirse yalnızca o aralıktaki sistem olayları sayılır
    (etiketlenen parça videonun bir bölümüyse dışarıdaki olaylar yanlış alarm sayılmasın).
    """
    if time_range is not None:
        lo, hi = time_range
        system = system[(system.time_sec >= lo - tolerance) & (system.time_sec <= hi + tolerance)]
    result: Dict[str, Any] = {"tolerance_sec": tolerance, "by_type": {}}
    tp_all = team_ok = 0
    for et in EVENT_TYPES:
        s = system[system.event_type == et].reset_index(drop=True)
        g = gt[gt.event_type == et].reset_index(drop=True)
        pairs = match_events(s.time_sec.tolist(), g.time_sec.tolist(), tolerance)
        result["by_type"][et] = _prf(len(pairs), len(s), len(g))
        tp_all += len(pairs)
        for i, j in pairs:
            if "team" in g and pd.notna(g.team[j]) and str(s.team[i]) == str(g.team[j]):
                team_ok += 1
    n_sys = int(system.event_type.isin(EVENT_TYPES).sum())
    n_gt = int(gt.event_type.isin(EVENT_TYPES).sum())
    result["overall"] = _prf(tp_all, n_sys, n_gt)
    result["overall"]["team_correct"] = team_ok
    return result


def evaluate_possession(timeline: pd.DataFrame, gt_pos: pd.DataFrame) -> Dict[str, Any]:
    """Saniye bazında sistem çoğunluk takımı ile etiket takımını karşılaştırır."""
    tl = timeline.dropna(subset=["smoothed_team"]).copy()
    tl["sec"] = np.floor(tl.timestamp_sec.astype(float)).astype(int)
    sys_team = tl.groupby("sec").smoothed_team.agg(lambda x: x.value_counts().idxmax())

    gt_sec: Dict[int, str] = {}
    for row in gt_pos.itertuples():
        if pd.isna(row.team) or str(row.team).strip() == "":
            continue
        for sec in range(int(np.floor(row.start_sec)), int(np.ceil(row.end_sec))):
            gt_sec[sec] = str(row.team)

    both = [s for s in gt_sec if s in sys_team.index]
    agree = sum(1 for s in both if str(sys_team[s]) == gt_sec[s])
    return {
        "gt_seconds": len(gt_sec),
        "covered_seconds": len(both),
        "coverage": round(len(both) / len(gt_sec), 4) if gt_sec else None,
        "agreement": round(agree / len(both), 4) if both else None,
    }


def swap_teams(teams: pd.Series) -> pd.Series:
    """Team A <-> Team B yer değiştirir; diğer değerlere dokunmaz."""
    return teams.map(lambda t: {"Team A": "Team B", "Team B": "Team A"}.get(t, t))


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Faz V olay ve sahiplik doğrulaması")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gt-events", required=True)
    ap.add_argument("--timeline")
    ap.add_argument("--gt-possession")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--start-sec", type=float, help="Etiketlenen parçanın başlangıcı")
    ap.add_argument("--end-sec", type=float, help="Etiketlenen parçanın sonu")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)
    system = load_system_events(summary)
    gt = pd.read_csv(args.gt_events)
    gt_pos = pd.read_csv(args.gt_possession) if args.gt_possession else None
    timeline = pd.read_csv(args.timeline) if args.timeline else None
    tr = (args.start_sec, args.end_sec) if args.start_sec is not None and args.end_sec is not None else None

    # Etiketçinin "Takım A"sı sistemin "Team A"sı olmayabilir: iki eşlemeyi de dene, uyanı seç.
    def run(swap: bool) -> Dict[str, Any]:
        g = gt.assign(team=swap_teams(gt.team)) if swap and "team" in gt else gt
        r: Dict[str, Any] = {"events": evaluate_events(system, g, args.tolerance, tr)}
        if timeline is not None and gt_pos is not None:
            p = gt_pos.assign(team=swap_teams(gt_pos.team)) if swap else gt_pos
            r["possession"] = evaluate_possession(timeline, p)
        return r

    def score(r: Dict[str, Any]) -> float:
        if "possession" in r and r["possession"]["agreement"] is not None:
            return r["possession"]["agreement"]
        return r["events"]["overall"]["team_correct"]

    direct, swapped = run(False), run(True)
    res = swapped if score(swapped) > score(direct) else direct
    res["team_mapping"] = "swapped" if res is swapped else "direct"

    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main()
