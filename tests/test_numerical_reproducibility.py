"""Check window-local volatility and deterministic forest prediction."""

import math
import unittest

import numpy as np
import pandas as pd

from data_processing.build_market_features import window_sample_std
from data_processing.build_dataset import assert_rebuild_matches
from modelling.models.random_forest_model import RFModel


class NumericalReproducibilityTests(unittest.TestCase):
    """Keep numerical history and worker scheduling out of model inputs."""

    def test_window_values_do_not_depend_on_prefix(self):
        """A complete lookback must be independent of preceding returns."""
        window = np.array([1e-8, -2e-8, 3e-8, 0., 1e-8, -1e-8])
        short = pd.Series(window)
        long = pd.Series(np.r_[np.linspace(-.5, .5, 200), window])
        a = short.rolling(6).apply(window_sample_std, raw=True).iloc[-1]
        b = long.rolling(6).apply(window_sample_std, raw=True).iloc[-1]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, np.std(window, ddof=1), places=22)

    def test_constant_windows_are_zero_and_small_variation_is_preserved(self):
        """Return zero without discarding genuine small price variation."""
        self.assertEqual(window_sample_std(np.zeros(24)), 0.)
        self.assertEqual(window_sample_std(np.full(24, .001)), 0.)
        values = np.array([0., 1e-12, 0., 0., 0., 0.])
        self.assertGreater(window_sample_std(values), 0.)
        self.assertTrue(math.isclose(window_sample_std(values),
                                     np.std(values, ddof=1), rel_tol=1e-15))
        series = pd.Series([0.] * 5 + [np.nan] + [0.] * 6)
        result = series.rolling(6).apply(window_sample_std, raw=True)
        self.assertTrue(result.iloc[:11].isna().all())
        self.assertEqual(result.iloc[11], 0.)

    def test_stable_rebuild_rejects_even_small_volatility_changes(self):
        """New datasets require exact features rather than historical tolerance."""
        a = pd.DataFrame({'realised_vol_6h': [0.],
                          'realised_vol_24h': [0.], 'value': [1.]})
        b = a.copy()
        b.loc[0, 'realised_vol_6h'] = 1e-12
        assert_rebuild_matches(a, b)
        with self.assertRaises(AssertionError):
            assert_rebuild_matches(a, b, stable_volatility=True)

    def test_forest_predictions_use_fixed_tree_order(self):
        """Serial prediction is repeatable and preserves training settings."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 3))
        y = (X[:, 0] + X[:, 1] > .5).astype(int)
        model = RFModel(n_estimators=40, min_samples_leaf=2, n_jobs=4)
        model.fit(X, y)
        expected_jobs = model.clf_.n_jobs
        a = model.predict_proba(X)
        b = model.predict_proba(X)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(model.clf_.n_jobs, expected_jobs)
        model.clf_.n_jobs = 1
        np.testing.assert_array_equal(a, model.clf_.predict_proba(X))
