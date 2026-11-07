"""Check per-coin metric denominators and undefined metrics."""

import unittest

import pandas as pd

from modelling.reporting.by_coin import coin_metrics, targetable_ids
from modelling.evaluation.threshold_selection import _targetable_episode_ids


class CoinMetricsTests(unittest.TestCase):
    """Check per-coin denominators and undefined metrics."""
    def test_vectorised_targets_match_core_rule_at_boundaries(self):
        """Check vectorized targetability matches the core endpoint rules."""
        hours = pd.date_range("2021-01-01", periods=12, freq="h", tz="UTC")
        predictions = pd.DataFrame({
            "coin": "A", "hour": hours, "y_true": 1,
        })
        episodes = pd.DataFrame({
            "coin": ["A", "A", "A"], "episode_id": ["a", "b", "c"],
            "start": hours[[0, 6, 9]],
        })
        for horizon in [1, 6, 24]:
            self.assertEqual(
                targetable_ids(predictions, episodes, horizon),
                _targetable_episode_ids(predictions, episodes, horizon),
            )

    def test_metrics_use_episode_counts_and_observed_months(self):
        """Check per-coin episode and observed-month denominators."""
        hours = pd.to_datetime([
            "2021-01-31 23:00Z", "2021-02-01 00:00Z",
            "2021-02-01 01:00Z",
        ])
        pred = pd.DataFrame({
            "coin": "A", "hour": hours,
            "y_true": [1, 0, 0], "y_score": [.9, .2, .1],
        })
        episodes = pd.DataFrame({
            "coin": ["A"], "episode_id": ["a"],
            "start": pd.to_datetime(["2021-01-31 23:30Z"]),
        })
        alarms = pd.DataFrame({
            "is_true_alarm": [True, False],
            "target_episode_id": ["a", None], "lead_hours": [.5, None],
        })
        result = coin_metrics(pred, alarms, episodes, 1)
        self.assertEqual(result["episode_recall"], 1.)
        self.assertEqual(result["coin_months"], 2)
        self.assertEqual(result["false_alarms_per_coin_month"], .5)
        self.assertEqual(result["alarm_precision"], .5)
        self.assertEqual(result["average_precision"], 1.)
        self.assertEqual(result["median_lead_hours"], .5)

    def test_no_targets_or_positive_hours_are_reported_as_undefined(self):
        """Check absent targets or positives produce undefined recall or AP."""
        pred = pd.DataFrame({
            "coin": ["A"],
            "hour": pd.to_datetime(["2021-01-01T00:00Z"]),
            "y_true": [0], "y_score": [.1],
        })
        episodes = pd.DataFrame({
            "coin": pd.Series(dtype=str), "episode_id": pd.Series(dtype=str),
            "start": pd.to_datetime([], utc=True),
        })
        alarms = pd.DataFrame({
            "is_true_alarm": pd.Series(dtype=bool),
            "target_episode_id": pd.Series(dtype=str),
            "lead_hours": pd.Series(dtype=float),
        })
        result = coin_metrics(pred, alarms, episodes, 1)
        self.assertTrue(pd.isna(result["episode_recall"]))
        self.assertTrue(pd.isna(result["average_precision"]))
        self.assertEqual(result["false_alarms_per_coin_month"], 0.)
