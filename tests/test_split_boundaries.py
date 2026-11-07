"""Independent saved-partition boundary and exclusion checks."""
import unittest
import pandas as pd
from modelling.validation.verify_model_results import check_split_boundaries


def frame(hours):
    return pd.DataFrame(dict(coin=['A']*len(hours),
                             hour=pd.to_datetime(hours, utc=True),
                             episode_id=[None]*len(hours)))


class SplitBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.parts = {
            'inner_train': frame(['2021-12-30T22:00Z']),
            'inner_val': frame(['2022-01-02T00:00Z', '2022-12-30T22:00Z']),
            'test': frame(['2023-01-02T00:00Z', '2023-12-31T23:00Z']),
        }
        self.parts['train'] = pd.concat([
            self.parts['inner_train'], self.parts['inner_val'],
            frame(['2022-01-01T00:00Z']),
        ], ignore_index=True)
        self.episodes = pd.DataFrame(dict(
            episode_id=pd.Series(dtype=str), start=pd.to_datetime([], utc=True),
            end=pd.to_datetime([], utc=True)))

    def test_full_training_window_can_include_calibration_and_inner_gap(self):
        check_split_boundaries(self.parts, self.episodes, 2023)

    def test_every_partition_rejects_its_boundary_violation(self):
        for name, timestamp in [
            ('inner_train', '2021-12-30T23:00Z'),
            ('inner_val', '2022-01-01T23:00Z'),
            ('inner_val', '2022-12-30T23:00Z'),
            ('train', '2022-12-30T23:00Z'),
            ('test', '2023-01-01T23:00Z'),
            ('test', '2024-01-01T00:00Z'),
        ]:
            with self.subTest(partition=name, timestamp=timestamp):
                altered = {k: v.copy() for k, v in self.parts.items()}
                altered[name].loc[0, 'hour'] = pd.Timestamp(timestamp)
                with self.assertRaisesRegex(ValueError, 'boundary'):
                    check_split_boundaries(altered, self.episodes, 2023)

    def test_episodes_touching_either_exclusion_window_are_removed(self):
        for boundary in ['2022-01-01', '2023-01-01']:
            point = pd.Timestamp(boundary, tz='UTC')
            episodes = pd.DataFrame(dict(episode_id=['crossing'],
                start=[point-pd.Timedelta(days=2)],
                end=[point-pd.Timedelta(hours=25)]))
            altered = {k: v.copy() for k, v in self.parts.items()}
            altered['train'].loc[0, 'episode_id'] = 'crossing'
            with self.assertRaisesRegex(ValueError, 'straddling'):
                check_split_boundaries(altered, episodes, 2023)

    def test_duplicate_origins_and_missing_final_training_rows_are_rejected(self):
        altered = {k: v.copy() for k, v in self.parts.items()}
        altered['inner_train'] = pd.concat([altered['inner_train']]*2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            check_split_boundaries(altered, self.episodes, 2023)
        altered = {k: v.copy() for k, v in self.parts.items()}
        altered['train'] = altered['train'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'outside final training'):
            check_split_boundaries(altered, self.episodes, 2023)
