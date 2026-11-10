"""Checks for targetability independent of reference labels."""
import unittest
import pandas as pd
from modelling.evaluation.targetability import targetable_ids


class TargetabilityTests(unittest.TestCase):
    """Check episode targetability independently of reference labels."""
    def test_strict_future_onset_and_inclusive_horizon(self):
        """Check onset exclusion and the inclusive horizon endpoint."""
        origins = pd.DataFrame({
            'coin': ['A'],
            'hour': pd.to_datetime(['2021-01-01T00:00Z']),
        })
        episodes = pd.DataFrame({
            'coin': ['A'] * 3,
            'episode_id': ['same_hour', 'horizon_end', 'too_late'],
            'start': pd.to_datetime([
                '2021-01-01T00:00Z', '2021-01-01T06:00Z',
                '2021-01-01T07:00Z',
            ]),
        })
        self.assertEqual(targetable_ids(origins, episodes, 6), {'horizon_end'})

    def test_reference_labels_do_not_limit_alternative_targets(self):
        """Check original labels do not filter alternative episode targets."""
        origins = pd.DataFrame({
            'coin': ['A'],
            'hour': pd.to_datetime(['2021-01-01T00:00Z']),
            'y_true': [0],
        })
        episodes = pd.DataFrame({
            'coin': ['A', 'B'],
            'episode_id': ['alternative_A', 'unobserved_B'],
            'start': pd.to_datetime(['2021-01-01T01:00Z'] * 2),
        })
        self.assertEqual(targetable_ids(origins, episodes, 6), {'alternative_A'})
