"""Negative tests for the verifier used to audit retained research results."""
import unittest
import numpy as np
import pandas as pd
from modelling.validation.verify_model_results import (
    HEADLINE_METRICS, check_fit_identity, check_metrics,
    pooled_metric_values, unique_cell,
)


class SavedRunVerificationTests(unittest.TestCase):
    def test_each_headline_metric_detects_changed_results(self):
        actual = dict(median_lead_hours=4., episode_recall=.5, false_alarms=3,
                      false_alarms_per_coin_month=.75, average_precision=.2,
                      episodes_caught=2, targetable_episodes=4, alarm_events=5)
        check_metrics(actual, actual, 'unchanged')
        for metric in HEADLINE_METRICS:
            with self.subTest(metric=metric):
                altered = dict(actual, **{metric: actual[metric] + 1})
                with self.assertRaises(AssertionError):
                    check_metrics(actual, altered, 'altered')

    def test_result_identity_must_be_present_and_unique(self):
        frame = pd.DataFrame({'model': ['logreg'], 'average_precision': [.2]})
        self.assertEqual(unique_cell(frame, model='logreg').average_precision, .2)
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
            unique_cell(frame, model='xgboost')
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
            unique_cell(pd.concat([frame, frame]), model='logreg')

    def test_each_fit_identity_field_is_checked(self):
        info = dict(run_id='expected', policy='exhaustive_lowest')
        spec = dict(run_id='expected', model='logreg', config='market_matched',
                    horizon=6, year=2024, refit_after_calibration=True,
                    threshold_policy='exhaustive_lowest')
        check_fit_identity(spec, info, 'logreg', 'market_matched', 6, 2024)
        for key in spec:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fit identity'):
                check_fit_identity(dict(spec, **{key: None}), info,
                                   'logreg', 'market_matched', 6, 2024)

    def test_pooling_uses_counts_exposure_and_individual_lead_times(self):
        parts = []
        for y, scores, caught, targets, false, months, leads in [
            ([1, 0], [.9, .8], 1, 1, 1, 1, [1.]),
            ([1, 0], [.7, .6], 3, 9, 1, 3, [9., 10., 11.]),
        ]:
            pred = pd.DataFrame(dict(y_true=y, y_score=scores))
            counts = dict(episodes_caught=caught, targetable_episodes=targets,
                          false_alarms=false, alarm_events=len(leads)+false,
                          observed_coin_months=months)
            events = pd.DataFrame(dict(is_true_alarm=[True]*len(leads)+[False],
                                       lead_hours=leads+[np.nan]))
            parts.append((pred, counts, events))
        actual = pooled_metric_values(parts)
        self.assertEqual(actual['episode_recall'], .4)
        self.assertEqual(actual['false_alarms_per_coin_month'], .5)
        self.assertEqual(actual['median_lead_hours'], 9.5)
        self.assertAlmostEqual(actual['average_precision'], (1 + 2/3) / 2)
