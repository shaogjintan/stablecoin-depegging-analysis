"""Checks for the canonical sample contract and the shared fold entry point."""
import unittest
from unittest.mock import patch

import pandas as pd

from data_processing import sample_preparation


class SamplePreparationTests(unittest.TestCase):
    """Check canonical eligibility requirements and fold delegation."""

    def test_missing_canonical_flag_does_not_use_a_legacy_flag(self):
        """Check old generic flags cannot replace the canonical tier flag."""
        master = pd.DataFrame({'prediction_eligible_h6': [True]})
        with self.assertRaisesRegex(KeyError, 'canonical eligibility'):
            sample_preparation.prediction_eligible_mask(master, 6, 1)

    def test_flags_must_be_non_missing_boolean_values(self):
        """Check integers and missing flags cannot silently include rows."""
        for flag in [pd.Series([1]), pd.Series([True, None], dtype='boolean')]:
            with self.subTest(flag=flag.tolist()):
                master = pd.DataFrame({'prediction_eligible_market_h6': flag})
                with self.assertRaisesRegex(ValueError, 'non-missing boolean'):
                    sample_preparation.prediction_eligible_mask(master, 6, 1)
        flag = pd.Series([True, False], index=[10, 20])
        master = pd.DataFrame({'prediction_eligible_market_h6': flag})
        pd.testing.assert_series_equal(
            sample_preparation.prediction_eligible_mask(master, 6, 1),
            master.prediction_eligible_market_h6,
        )

    def test_invalid_horizons_and_tiers_are_rejected(self):
        """Check unsupported selections fail before constructing folds."""
        for horizon, tier in [(2, 1), (6, 0), (24, 3)]:
            with self.subTest(horizon=horizon, tier=tier):
                with self.assertRaises(ValueError):
                    sample_preparation.prediction_eligible_mask(
                        pd.DataFrame(), horizon, tier
                    )

    def test_loading_delegates_to_the_shared_fold_builder(self):
        """Check loaded inputs use the same preparation as the runner."""
        master, episodes, expected = object(), object(), object()
        with patch.object(sample_preparation, 'load_master',
                          return_value=(master, episodes)), patch.object(
                sample_preparation, 'corrected_folds',
                return_value=expected) as prepare:
            actual = sample_preparation.load_folds(horizon=6, tier=2)
        self.assertIs(actual, expected)
        prepare.assert_called_once_with(master, episodes, 6, 2)
