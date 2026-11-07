"""Checks for canonical naming and safe direct dataset reconstruction."""
import unittest
import pandas as pd
from data_processing.config import ACTIVE_MASTER_PATH, ACTIVE_EPISODES_PATH
from data_processing.splits import resolve_recorded_input
from data_processing.build_dataset import assert_rebuild_matches
from data_processing.dataset_transforms import prepare_corrected


class DatasetBuilderTests(unittest.TestCase):
    """Check safe rebuilds and canonical dataset contracts."""
    def test_canonical_names_and_frozen_aliases(self):
        """Check historical dataset aliases resolve to canonical input paths."""
        self.assertEqual(ACTIVE_MASTER_PATH.name, 'master.parquet')
        self.assertEqual(ACTIVE_EPISODES_PATH.name, 'depeg_episodes.parquet')
        self.assertEqual(resolve_recorded_input('data/processed/master_corrected_v1.parquet'), ACTIVE_MASTER_PATH)
        self.assertEqual(resolve_recorded_input('data/processed/depeg_episodes_corrected_v1.parquet'), ACTIVE_EPISODES_PATH)

    def test_corrected_clock_cannot_be_shifted_twice(self):
        """Check already-corrected timestamps cannot be corrected again."""
        with self.assertRaisesRegex(ValueError, 'already corrected'):
            prepare_corrected(pd.DataFrame({'bar_start': [pd.Timestamp('2021-01-01', tz='UTC')]}))

    def test_volatility_tolerance_cannot_hide_changed_targets(self):
        """Check rebuild tolerance cannot mask changes to target labels."""
        frozen = pd.DataFrame({'label_h6': [0., 1.], 'realised_vol_6h': [0., .01],
                               'realised_vol_24h': [0., .02]})
        rebuilt = frozen.copy()
        rebuilt.loc[0, 'realised_vol_6h'] = 5e-10
        assert_rebuild_matches(rebuilt, frozen)
        rebuilt.loc[0, 'label_h6'] = 1.
        with self.assertRaises(AssertionError):
            assert_rebuild_matches(rebuilt, frozen)

    def test_volatility_missingness_must_match(self):
        """Check volatility tolerance preserves missingness patterns."""
        frozen = pd.DataFrame({'label_h6': [0.], 'realised_vol_6h': [float('nan')],
                               'realised_vol_24h': [0.]})
        rebuilt = frozen.fillna(0.)
        with self.assertRaises(AssertionError):
            assert_rebuild_matches(rebuilt, frozen)
