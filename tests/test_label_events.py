import pandas as pd

from utils.label_events import LabelSession
from utils.validate_events import evaluate_events, evaluate_possession, load_system_events


def _session():
    s = LabelSession()
    s.set_possession(1.0, "Team A")
    s.add_event(3.0, "completed_pass")
    s.set_possession(5.0, None)
    s.set_possession(6.0, "Team B")
    s.add_event(8.0, "turnover")
    return s


def test_events_take_current_possession_team():
    assert _session().events() == [(3.0, "completed_pass", "Team A"), (8.0, "turnover", "Team B")]


def test_possession_segments_close_at_next_mark_and_clip_end():
    assert _session().possession_segments(10.0) == [
        (1.0, 5.0, "Team A"), (5.0, 6.0, None), (6.0, 10.0, "Team B")]


def test_undo_removes_last_mark():
    s = _session()
    s.undo()
    assert s.events() == [(3.0, "completed_pass", "Team A")]


def test_saved_files_are_readable_by_validator(tmp_path):
    ev_path, pos_path = _session().save(str(tmp_path), 10.0)
    gt = pd.read_csv(ev_path)
    system = load_system_events({"fps": 50, "passes": {"events": [
        {"event_type": "completed_pass", "start_frame": 160, "from_team": "Team A"}]}})
    assert evaluate_events(system, gt)["by_type"]["completed_pass"]["tp"] == 1
    timeline = pd.DataFrame({"timestamp_sec": [2.5, 7.5], "smoothed_team": ["Team A", "Team B"]})
    assert evaluate_possession(timeline, pd.read_csv(pos_path))["agreement"] == 1.0
