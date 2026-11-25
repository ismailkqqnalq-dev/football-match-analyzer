import json

import pandas as pd
import pytest

from utils.validate_events import (
    evaluate_events,
    evaluate_possession,
    load_system_events,
    main,
    match_events,
)


def _summary(events, fps=50.0):
    return {"fps": fps, "passes": {"events": events}}


def _ev(event_type, start_frame, team="Team A"):
    return {"event_type": event_type, "start_frame": start_frame, "end_frame": start_frame + 10,
            "from_team": team, "to_team": team}


def test_load_system_events_converts_frames_to_seconds():
    df = load_system_events(_summary([_ev("completed_pass", 100), _ev("turnover", 250, "Team B")]))
    assert df.time_sec.tolist() == [2.0, 5.0]
    assert df.event_type.tolist() == ["completed_pass", "turnover"]
    assert df.team.tolist() == ["Team A", "Team B"]


def test_load_system_events_requires_fps():
    with pytest.raises(ValueError):
        load_system_events({"passes": {"events": []}})


def test_match_events_is_one_to_one_and_prefers_closest():
    # İki sistem olayı aynı etikete yakın: yalnızca en yakını eşleşmeli.
    pairs = match_events([10.0, 10.4], [10.3], tolerance=1.0)
    assert pairs == [(1, 0)]


def test_match_events_respects_tolerance():
    assert match_events([10.0], [11.2], tolerance=1.0) == []
    assert match_events([10.0], [11.0], tolerance=1.0) == [(0, 0)]


def test_evaluate_events_precision_recall_f1():
    system = load_system_events(_summary([
        _ev("completed_pass", 100),   # 2.0 s -> eşleşir
        _ev("completed_pass", 500),   # 10.0 s -> yanlış alarm
        _ev("turnover", 1000, "Team B"),  # 20.0 s -> eşleşir, takım yanlış
    ]))
    gt = pd.DataFrame({
        "time_sec": [2.5, 30.0, 20.3],
        "event_type": ["completed_pass", "completed_pass", "turnover"],
        "team": ["Team A", "Team A", "Team A"],
    })
    res = evaluate_events(system, gt, tolerance=1.0)
    p = res["by_type"]["completed_pass"]
    assert (p["tp"], p["system"], p["ground_truth"]) == (1, 2, 2)
    assert p["precision"] == 0.5 and p["recall"] == 0.5 and p["f1"] == 0.5
    assert res["by_type"]["turnover"]["f1"] == 1.0
    assert res["overall"]["tp"] == 2
    assert res["overall"]["team_correct"] == 1


def test_evaluate_events_does_not_match_across_types():
    system = load_system_events(_summary([_ev("turnover", 100)]))
    gt = pd.DataFrame({"time_sec": [2.0], "event_type": ["completed_pass"], "team": ["Team A"]})
    res = evaluate_events(system, gt)
    assert res["overall"]["tp"] == 0
    assert res["overall"]["f1"] == 0.0


def test_evaluate_events_time_range_ignores_events_outside_labelled_clip():
    system = load_system_events(_summary([_ev("completed_pass", 100), _ev("completed_pass", 5000)]))
    gt = pd.DataFrame({"time_sec": [2.0], "event_type": ["completed_pass"], "team": ["Team A"]})
    res = evaluate_events(system, gt, time_range=(0.0, 30.0))
    assert res["by_type"]["completed_pass"]["precision"] == 1.0


def test_evaluate_possession_agreement_and_coverage():
    timeline = pd.DataFrame({
        "timestamp_sec": [0.1, 0.5, 0.9, 1.2, 1.6, 2.5],
        "smoothed_team": ["Team A", "Team A", "Team B", "Team B", "Team B", None],
    })
    gt_pos = pd.DataFrame({"start_sec": [0.0, 2.0], "end_sec": [2.0, 3.0],
                           "team": ["Team A", "Team B"]})
    res = evaluate_possession(timeline, gt_pos)
    # Etiket: 0,1 -> A ; 2 -> B. Sistem: 0 -> A, 1 -> B, 2 -> yok.
    assert res["gt_seconds"] == 3
    assert res["covered_seconds"] == 2
    assert res["agreement"] == 0.5


def test_main_swaps_team_names_when_labeller_used_opposite_letters(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps(_summary([_ev("completed_pass", 100, "Team A")])), encoding="utf-8")
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"time_sec": [2.0], "event_type": ["completed_pass"], "team": ["Team B"]}).to_csv(gt, index=False)
    timeline = tmp_path / "tl.csv"
    pd.DataFrame({"timestamp_sec": [0.5, 1.5], "smoothed_team": ["Team A", "Team A"]}).to_csv(timeline, index=False)
    gt_pos = tmp_path / "pos.csv"
    pd.DataFrame({"start_sec": [0.0], "end_sec": [2.0], "team": ["Team B"]}).to_csv(gt_pos, index=False)
    res = main(["--summary", str(summary), "--gt-events", str(gt), "--timeline", str(timeline),
                "--gt-possession", str(gt_pos)])
    assert res["team_mapping"] == "swapped"
    assert res["possession"]["agreement"] == 1.0
    assert res["events"]["overall"]["team_correct"] == 1


def test_main_writes_json(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps(_summary([_ev("completed_pass", 100)])), encoding="utf-8")
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"time_sec": [2.2], "event_type": ["completed_pass"], "team": ["Team A"]}).to_csv(gt, index=False)
    out = tmp_path / "res.json"
    res = main(["--summary", str(summary), "--gt-events", str(gt), "--json-out", str(out)])
    assert res["events"]["overall"]["f1"] == 1.0
    assert json.loads(out.read_text(encoding="utf-8"))["events"]["overall"]["tp"] == 1
