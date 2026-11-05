import pandas as pd
import pytest

from modelling.robustness_refit_alarms import evaluate_events, check_reference


def test_fixed_events_use_new_onsets_and_saved_eligibility():
    origin = pd.Timestamp("2024-01-31 22:00", tz="UTC")
    predictions = pd.DataFrame(dict(coin=["A", "A"],
                                   hour=[origin, origin + pd.Timedelta(hours=3)], y_true=[0, 0]))
    episodes = pd.DataFrame(dict(coin=["A"] * 4,
                                episode_id=["same_hour", "first", "second", "ineligible"],
                                start=[origin + pd.Timedelta(hours=h) for h in (0, 2, 3, 10)]))
    events = pd.DataFrame(dict(coin=["A", "A", "B"],
                              alarm_hour=[origin, origin + pd.Timedelta(hours=3), origin]))
    result = evaluate_events(predictions, events, episodes, 3)
    # First onset only is credited; same-hour and unavailable forecast windows do not count.
    assert result["targetable_episodes"] == 2
    assert result["episodes_caught"] == 1
    assert result["true_alarms"] == 1
    assert result["false_alarms"] == 2
    assert result["observed_coin_months"] == 2
    assert result["episode_recall"] == .5
    assert result["median_lead_hours"] == 2


def test_right_endpoint_inclusive_and_empty_episodes():
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    pred = pd.DataFrame(dict(coin=["A"], hour=[origin]))
    events = pred.rename(columns={"hour": "alarm_hour"})
    eps = pd.DataFrame(dict(coin=["A"], episode_id=["A_1"],
                            start=[origin + pd.Timedelta(hours=6)]))
    assert evaluate_events(pred, events, eps, 6)["episodes_caught"] == 1
    assert evaluate_events(pred, events, eps, 5)["false_alarms"] == 1
    result = evaluate_events(pred, events, eps.iloc[:0], 6)
    assert result["targetable_episodes"] == 0
    assert result["false_alarms"] == 1
    assert pd.isna(result["episode_recall"])
    assert evaluate_events(pred, events.iloc[:0], eps, 6)["episodes_caught"] == 0


def test_reference_check_fails_on_any_count_mismatch():
    row = dict(model="logreg", config="market_matched", horizon_hours=6,
               episodes_caught=1, targetable_episodes=2, false_alarms=3,
               observed_coin_months=4, alarm_events=5, true_alarms=2)
    actual = pd.DataFrame([{**row, "is_reference_definition": True}])
    assert len(check_reference(actual, pd.DataFrame([row]))) == 1
    with pytest.raises(AssertionError):
        check_reference(actual, pd.DataFrame([{**row, "false_alarms": 4}]))


def test_matches_original_fixed_alarm_evaluator(monkeypatch):
    from pathlib import Path
    from modelling import robustness_check

    origin = pd.Timestamp("2024-01-01", tz="UTC")
    pred = pd.DataFrame(dict(coin=["A", "A"],
                            hour=[origin, origin + pd.Timedelta(hours=3)], year=[2024, 2024]))
    events = pd.DataFrame(dict(coin=["A", "A"],
                              alarm_hour=pred.hour, year=[2024, 2024]))
    eps = pd.DataFrame(dict(coin=["A", "A"], episode_id=["A_1", "A_2"],
                            start=[origin + pd.Timedelta(hours=h) for h in (2, 3)]))
    monkeypatch.setattr(robustness_check, "make_episodes", lambda *args: eps.copy())
    monkeypatch.setattr(Path, "exists", lambda *args: True)
    monkeypatch.setattr(pd, "read_parquet", lambda *args: pred.copy())
    monkeypatch.setattr(pd, "read_csv", lambda *args: events.copy())
    old = robustness_check.evaluate_definition(
        pd.DataFrame(), [("logreg", 1, 3, "fixed", Path("unused"), Path("unused"))], [.01], [2]).iloc[0]
    new = evaluate_events(pred, events, eps, 3)
    for current, previous in (("episodes_caught", "test_episodes_warned"),
                              ("targetable_episodes", "test_episodes_targetable"),
                              ("false_alarms", "test_false_alarm_events"),
                              ("true_alarms", "test_true_alarm_events"),
                              ("observed_coin_months", "coin_months"),
                              ("median_lead_hours", "median_warning_lead_hours")):
        assert new[current] == old[previous]
