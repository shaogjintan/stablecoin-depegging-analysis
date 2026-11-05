import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.false_alarm_framework import evaluate_configuration
from modelling.models.baseline import ConstantPriorModel, PersistenceModel
from modelling.rolling_threshold import _choose_threshold
from modelling.verify_baselines import independent_ap, independent_events, full_scan_candidates, target_codes, verify, HOUR


class BaselineTests(unittest.TestCase):
    def test_prior_uses_training_only_and_pooled_ap_is_not_base_rate(self):
        X = pd.DataFrame({"peg_deviation": [0., -.005]})
        model = ConstantPriorModel().fit(X, [1, 0], X_val=X, y_val=[0, 0])
        np.testing.assert_array_equal(model.predict_proba(X)[:, 1], [.5, .5])
        self.assertEqual(model.positive_rate_, .5)
        y, scores = np.array([1, 0, 1, 1]), np.array([.5, .5, 1., 1.])
        self.assertEqual(average_precision_score(y, scores), independent_ap(y, scores))
        self.assertNotEqual(independent_ap(y, scores), y.mean())
        self.assertEqual(independent_ap(y[:2], scores[:2]), y[:2].mean())

    def test_persistence_formula_and_ignores_onchain(self):
        X = pd.DataFrame({"peg_deviation": [.01, 0., -.003, -.02], "tx_count_logdiff_1h": [1, 2, 3, 4]})
        model = PersistenceModel().fit(X, [0, 1, 1, 0])
        scores = model.predict_proba(X)[:, 1]
        np.testing.assert_allclose(scores, [0., 0., .003, .01], atol=1e-15)
        X.tx_count_logdiff_1h = -100.
        np.testing.assert_array_equal(model.predict_proba(X)[:, 1], scores)

    def test_constant_can_select_finite_threshold_and_framework_does_not_bypass(self):
        hours = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
        p = pd.DataFrame({"coin": "A", "hour": hours[[0]], "horizon": 1, "y_true": 1, "y_score": .2})
        episodes = pd.DataFrame({"coin": ["A"], "start": hours[[1]], "episode_id": ["a1"]})
        threshold, stats = _choose_threshold(p, episodes, 1, 1.)
        self.assertEqual(threshold, .2)
        self.assertEqual(stats["episode_recall"], 1.)
        meta = p[["coin", "hour"]].assign(episode_id=None)
        fold = dict(year=2021, horizon=1, tier=1, features=["peg_deviation"],
                    dataset_version="synthetic-test")
        for part in ("inner_train", "inner_val", "test"):
            fold[f"{part}_meta"] = meta
            fold[f"X_{part}"] = pd.DataFrame({"peg_deviation": [0.]})
            fold[f"y_{part}"] = pd.Series([1])
        with patch("modelling.false_alarm_framework._FOLDS_CACHE", {(1, 1): [fold]}):
            rows, _, _ = evaluate_configuration("baseline", 1, 1, episodes, 1.)
        self.assertEqual(rows[0]["threshold"], 1.)
        self.assertEqual(rows[0]["test_episodes_caught"], 1)

    def test_independent_full_scan_and_inclusive_cooldown(self):
        hours = pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC")
        p = pd.DataFrame({"coin": "A", "hour": hours[[0, 4, 11]], "horizon": 6,
                          "y_true": [1, 0, 0], "y_score": [.2, .9, .8], "threshold": .2})
        eps = pd.DataFrame({"coin": ["A"], "start": hours[[2]], "episode_id": ["a1"]})
        o, times, coins, targets, e = target_codes(p, eps, 6)
        counts = full_scan_candidates(o.y_score.to_numpy(), times, coins, targets,
                                      np.array([.2, .8, .9, np.inf]), 6 * HOUR, len(e))
        np.testing.assert_array_equal(counts, [[1, 1, 2], [0, 2, 2], [0, 1, 1], [0, 0, 0]])
        events, ids = independent_events(p, eps, 6)
        self.assertEqual(ids, {"a1"})
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0].lead_hours, 2)
        p.hour = hours[[0, 6, 11]]
        events, _ = independent_events(p, eps, 6)
        self.assertEqual(len(events), 2)

    def test_independent_ap_ties_and_no_positive(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            y = rng.integers(0, 2, 40)
            scores = rng.integers(0, 5, 40) / 100
            self.assertAlmostEqual(independent_ap(y, scores), average_precision_score(y, scores), places=14)
        self.assertEqual(independent_ap([0, 0], [.1, .1]), 0.)

    def test_verifier_refuses_to_overwrite_completed_evidence(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            evidence = folder / "verification.json"
            evidence.write_text('{"status": "passed"}', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "check-only"):
                verify(folder, [])
            self.assertEqual(evidence.read_text(), '{"status": "passed"}')


if __name__ == "__main__":
    unittest.main()
