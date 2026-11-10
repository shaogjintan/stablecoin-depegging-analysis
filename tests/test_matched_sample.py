"""Check fair tier comparisons and invalid population matching."""

import copy
import unittest

import pandas as pd

from data_processing.matched_sample import _common_sample_fold


def make_fold(order, horizon=6):
    """Create a small fold with stable key-specific features and labels."""
    fold = dict(year=2024, horizon=horizon, features=['market'])
    hours = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
    for part in ('train', 'inner_train', 'inner_val', 'test'):
        fold[f'{part}_meta'] = pd.DataFrame({
            'coin': ['USDT'] * len(order), 'hour': hours[order],
        })
        fold[f'X_{part}'] = pd.DataFrame({'market': order})
        fold[f'y_{part}'] = pd.Series([i % 2 for i in order])
    return fold


class MatchedSampleTests(unittest.TestCase):
    """Validate population identity without depending on tier row order."""

    def test_matching_preserves_market_order_and_inputs(self):
        """Match by keys even when the on-chain rows arrive reversed."""
        market, onchain = make_fold([0, 1, 2]), make_fold([2, 0])
        before = copy.deepcopy(market)
        result = _common_sample_fold(market, onchain)
        for part in ('train', 'inner_train', 'inner_val', 'test'):
            self.assertEqual(result[f'X_{part}'].market.tolist(), [0, 2])
            self.assertEqual(result[f'y_{part}'].tolist(), [0, 0])
            pd.testing.assert_frame_equal(
                market[f'{part}_meta'], before[f'{part}_meta']
            )

    def test_duplicate_keys_cannot_disguise_a_missing_key(self):
        """Reject a duplicate that makes matching row counts look correct."""
        with self.assertRaisesRegex(ValueError, 'duplicate keys'):
            _common_sample_fold(make_fold([0, 0]), make_fold([0, 1]))

    def test_fold_identity_and_labels_must_agree(self):
        """Reject a wrong horizon, a conflicting label, or a missing key."""
        with self.assertRaisesRegex(ValueError, 'same horizon'):
            _common_sample_fold(make_fold([0, 1]), make_fold([0], 24))
        onchain = make_fold([0])
        onchain['y_test'].iloc[0] = 1
        with self.assertRaisesRegex(ValueError, 'labels disagree'):
            _common_sample_fold(make_fold([0, 1]), onchain)
        with self.assertRaisesRegex(ValueError, 'not a subset'):
            _common_sample_fold(make_fold([0]), make_fold([1]))
