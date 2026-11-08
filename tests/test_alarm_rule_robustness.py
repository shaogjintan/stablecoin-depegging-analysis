"""Check causal smoothing, paired OR inputs and saved-score CLI reports."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.robustness_checks.alarm_rules import (
    align_pair, assess, consecutive_flags,
)
from modelling.utils.io import sha256


class AlarmRuleTests(unittest.TestCase):
    """Validate alarm-rule changes without fitting primary models."""

    def test_smoothing_breaks_at_gaps_and_coin_boundaries(self):
        """Require earlier calendar hours, not merely adjacent saved rows."""
        pred = pd.DataFrame({
            'coin': ['USDT'] * 4 + ['USDC'] * 2,
            'hour': pd.to_datetime([
                '2024-01-01T00:00Z', '2024-01-01T01:00Z',
                '2024-01-01T03:00Z', '2024-01-01T04:00Z',
                '2024-01-01T00:00Z', '2024-01-01T01:00Z',
            ]), 'y_score': [.8, .9, .9, .8, .9, .1],
        })
        actual = consecutive_flags(pred, .5, 2)
        self.assertEqual(actual.tolist(), [False, True, False, True, False, False])
        self.assertEqual(consecutive_flags(pred, .5, 1).tolist(),
                         pred.y_score.ge(.5).tolist())
        prefix = consecutive_flags(pred.iloc[:3], .5, 2)
        pd.testing.assert_series_equal(prefix, actual.iloc[:3])
        self.assertFalse(consecutive_flags(pred, np.inf, 2).any())

    def test_pairing_rejects_missing_keys_and_conflicting_labels(self):
        """Never silently shrink the population by an inner join."""
        frame = pd.DataFrame({
            'coin': ['USDT', 'USDT'],
            'hour': pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC'),
            'horizon': 6, 'y_true': [0., 1.], 'y_score': [.2, .8],
        })
        a, b = align_pair(frame, frame.iloc[::-1])
        pd.testing.assert_frame_equal(a, b)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            align_pair(frame, frame.iloc[:1])
        changed = frame.copy()
        changed.loc[0, 'y_true'] = 1.
        with self.assertRaisesRegex(ValueError, 'disagree'):
            align_pair(frame, changed)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            align_pair(frame, pd.concat([frame, frame.iloc[:1]]))

    def test_alarm_metrics_do_not_use_binary_flags_as_risk_scores(self):
        """Keep single-model AP unchanged and leave binary OR AP missing."""
        pred = pd.DataFrame({
            'coin': ['USDT'] * 3,
            'hour': pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'),
            'horizon': 6, 'y_true': [0., 1., 1.], 'y_score': [.2, .9, .7],
        })
        episodes = pd.DataFrame({
            'coin': ['USDT'], 'episode_id': ['one'],
            'start': [pd.Timestamp('2024-01-01T04:00Z')],
        })
        flags = np.array([False, True, True])
        values, events = assess(pred, episodes, flags, pred.y_score.to_numpy())
        self.assertEqual(values['average_precision'],
                         average_precision_score(pred.y_true, pred.y_score))
        self.assertEqual(values['targetable_episodes'], 1)
        self.assertEqual(values['episodes_caught'], 1)
        self.assertEqual(len(events), 1)
        self.assertIn('rule_signal', events)
        or_values, _ = assess(pred, episodes, flags)
        self.assertTrue(np.isnan(or_values['average_precision']))

    def test_commands_read_saved_scores_and_reconcile_baselines(self):
        """Exercise both CLIs with a small self-contained saved-run fixture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'runs' / 'primary' / 'all'
            fits = run / 'fits'
            fits.mkdir(parents=True)
            source = root / 'source.bin'
            source.write_bytes(b'fixed source')
            models = ['logreg', 'random_forest']
            info = dict(run_id='fixture', status='complete',
                        refit_after_calibration=True, policy='exhaustive_lowest',
                        models=models, configs=['market_onchain'],
                        horizons=[6], years=[2024],
                        input_sha256={str(source): sha256(source)})
            (run / 'run.json').write_text(json.dumps(info))
            episodes = pd.DataFrame({
                'coin': ['USDT', 'USDT'], 'episode_id': ['cal', 'test'],
                'start': pd.to_datetime(['2023-02-01T08:00Z',
                                         '2024-02-01T08:00Z']),
            })
            episodes.to_parquet(run / 'episodes.parquet', index=False)
            original = {}
            for model in models:
                folder = fits / f'{model}_market_onchain_h6_2024'
                folder.mkdir()
                spec = dict(run_id='fixture', model=model,
                            config='market_onchain', horizon=6, year=2024,
                            refit_after_calibration=True,
                            threshold_policy='exhaustive_lowest', threshold=.5)
                (folder / 'fit.json').write_text(json.dumps(spec))
                for part, year in [('inner_val', 2023), ('test', 2024)]:
                    frame = pd.DataFrame({
                        'coin': ['USDT'] * 8,
                        'hour': pd.date_range(f'{year}-02-01', periods=8,
                                              freq='h', tz='UTC'),
                        'horizon': 6,
                        'y_true': [0., 0., 1., 1., 1., 1., 1., 1.],
                        'y_score': ([.1, .1, .8, .9, .1, .7, .8, .1]
                                    if model == 'logreg' else
                                    [.1, .1, .1, .7, .9, .1, .8, .9]),
                    })
                    path = folder / f'{part}.parquet'
                    frame.to_parquet(path, index=False)
                    original[path] = sha256(path)
            for module in ['alarm_smoothing', 'or_combination']:
                output = root / module
                subprocess.run([
                    sys.executable, '-m', f'modelling.robustness_checks.{module}',
                    '--run-dir', str(run), '--out-dir', str(output),
                    '--horizons', '6', '--years', '2024',
                ], check=True, capture_output=True)
                summary = pd.read_csv(output / 'summary.csv')
                yearly = pd.read_csv(output / 'by_year.csv')
                self.assertEqual(len(summary), len(yearly))
                self.assertTrue(summary.targetable_episodes.eq(1).all())
                self.assertTrue(yearly.filter(like='calibration_').shape[1] > 0)
                if module == 'alarm_smoothing':
                    self.assertEqual(len(summary), 4)
                    baseline = summary.loc[summary.variant.eq('baseline')]
                    self.assertTrue(baseline.episodes_caught.eq(1).all())
                else:
                    self.assertEqual(len(summary), 3)
                    self.assertTrue(summary.loc[
                        summary.variant.eq('or'), 'average_precision'
                    ].isna().all())
            self.assertEqual({path: sha256(path) for path in original}, original)
