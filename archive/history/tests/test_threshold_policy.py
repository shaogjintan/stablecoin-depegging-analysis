import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from modelling.alarm_events import build_alarm_events
from modelling.recalibrate_thresholds import oracle_choice, run
from modelling.rolling_threshold import _choose_threshold, _observed_coin_months


class ThresholdPolicyTests(unittest.TestCase):
    def test_nonmonotone_feasible_set(self):
        hours = pd.date_range("2023-01-01", periods=12, freq="h", tz="UTC")
        pred = pd.DataFrame(dict(coin="A", hour=hours[[0, 4, 11]], horizon=6,
                                 y_true=[1, 0, 0], y_score=[.2, .9, .8]))
        eps = pd.DataFrame(dict(coin=["A"], episode_id=["a"], start=hours[[2]]))
        # .2 and .9 are feasible but the intervening .8 has two false alarms.
        threshold, stats = _choose_threshold(pred, eps, 6, 1, "lowest_feasible")
        self.assertEqual(threshold, .2)
        self.assertEqual(stats["false_alarms"], 1)
        self.assertEqual(oracle_choice(pred, eps, 6, 1, "lowest_feasible")[:3], (.2, 1, 1))

    def test_failed_derivation_is_explicit_and_never_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "derived"
            with self.assertRaises(FileNotFoundError):
                run([root / "missing"], out)
            recorded = json.loads((out / "run.json").read_text())
            self.assertEqual(recorded["status"], "failed")
            self.assertIn("FileNotFoundError", recorded["error"])
            with self.assertRaises(FileExistsError):
                run([], out)
            self.assertEqual(recorded, json.loads((out / "run.json").read_text()))

    def test_selectors_against_brute_force(self):
        rng = np.random.default_rng(42)
        for case in range(24):
            hours = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
            pred = pd.DataFrame(dict(coin=np.repeat(["A", "B"], 48),
                hour=list(hours) * 2, horizon=6, y_true=0,
                y_score=rng.integers(0, 7, 96).astype(float)))
            eps = pd.DataFrame(dict(coin=["A", "B"], episode_id=["a", "b"],
                                    start=[hours[20], hours[36]]))
            if case % 4 == 0:
                eps = eps.iloc[:0]
            for ep in eps.itertuples():
                pred.loc[pred.coin.eq(ep.coin) & pred.hour.lt(ep.start)
                         & pred.hour.ge(ep.start - pd.Timedelta(hours=6)), "y_true"] = 1
            if case % 5 == 0:
                pred["y_score"] = .1
            budget = [0., 1., 5.][case % 3]
            candidates = []
            for cutoff in np.r_[np.unique(pred.y_score), np.inf]:
                _, events = build_alarm_events(pred, eps, cutoff)
                fa = int((~events.is_true_alarm).sum())
                caught = events.target_episode_id.nunique()
                if fa <= budget * _observed_coin_months(pred):
                    candidates.append((cutoff, caught, fa))
            for policy in ("lowest_feasible", "max_validation_recall"):
                best = (min(candidates, key=lambda x: x[0]) if policy == "lowest_feasible"
                        else max(candidates, key=lambda x: (x[1], -x[2], x[0])))
                cutoff, stats = _choose_threshold(pred, eps, 6, budget, policy=policy)
                self.assertEqual((cutoff, stats["episodes_warned"], stats["false_alarms"]), best)
                self.assertEqual(oracle_choice(pred, eps, 6, budget, policy)[:3], best)

    def test_zero_positives_tie_policy_differs(self):
        pred = pd.DataFrame(dict(coin=["A"], hour=pd.to_datetime(["2023-01-01"], utc=True),
                                 horizon=[1], y_true=[0], y_score=[.3]))
        eps = pd.DataFrame(columns=["coin", "episode_id", "start"])
        self.assertEqual(_choose_threshold(pred, eps, 1, 1, "lowest_feasible")[0], .3)
        self.assertEqual(_choose_threshold(pred, eps, 1, 1)[0], np.inf)
        self.assertEqual(_choose_threshold(pred, eps, 1, 0, "lowest_feasible")[0], np.inf)
        with self.assertRaisesRegex(ValueError, "Unknown threshold policy"):
            _choose_threshold(pred, eps, 1, 1, "typo")

    def test_legacy_policy_aliases(self):
        pred = pd.DataFrame(dict(coin=["A"], hour=pd.to_datetime(["2023-01-01"], utc=True),
                                 horizon=[1], y_true=[0], y_score=[.3]))
        eps = pd.DataFrame(columns=["coin", "episode_id", "start"])
        for alias, name in (("exhaustive_lowest", "lowest_feasible"),
                            ("exact_max_recall", "max_validation_recall")):
            threshold, stats = _choose_threshold(pred, eps, 1, 1, alias)
            self.assertEqual(threshold, _choose_threshold(pred, eps, 1, 1, name)[0])
            self.assertEqual(stats["threshold_policy"], name)
            self.assertEqual(oracle_choice(pred, eps, 1, 1, alias)[0], threshold)


if __name__ == "__main__":
    unittest.main()
