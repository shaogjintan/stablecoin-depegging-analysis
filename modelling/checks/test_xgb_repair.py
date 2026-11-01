from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from modelling.corrected_data import (
    calendar_market_features, causal_risk_mask, episode_labels, prepare_corrected,
)
from modelling.alarm_events import build_alarm_events
from modelling.rolling_threshold import _choose_threshold, _targetable_episode_ids, _observed_coin_months, _false_alarm_stats
from modelling.models.xgboost_model import XGBModel
from modelling.exact_threshold import exhaustive_counts


def frame(n=70):
    return pd.DataFrame({
        "coin": "A", "hour": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
        "price": 1., "trading_volume": np.arange(n) + 1.,
        "hours_since_venue_start": np.arange(n) + 1.,
        **{f"{m}_logdiff_{h}h": 0. for m in
           ("tx_count", "tx_volume", "active_senders", "active_receivers") for h in (1, 6, 24)},
    })


class RepairTests(unittest.TestCase):
    def test_targets_clock_and_causal_prefix(self):
        d = frame()
        d.loc[[2, 3, 10, 11], "price"] = .98
        corrected, eps = prepare_corrected(d)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps.iloc[0].start, d.iloc[2].hour + pd.Timedelta(hours=1))
        self.assertEqual(corrected.loc[1, "label_h1"], 1)
        self.assertEqual(corrected.loc[9, "label_h1"], 0)
        self.assertFalse(corrected.loc[9, "observable_at_risk"])
        risk = causal_risk_mask(d)
        for n in (5, 12, 25, 50):
            pd.testing.assert_series_equal(risk.iloc[:n], causal_risk_mask(d.iloc[:n]))
        # A later re-dip cannot retrospectively alter an earlier risk decision.
        self.assertFalse(risk.iloc[36])
        self.assertTrue(risk.iloc[37])
        self.assertFalse(corrected.iloc[-1].prediction_eligible_market_h6)
        self.assertEqual(corrected.iloc[40].hour - corrected.iloc[40].bar_start,
                         pd.Timedelta(hours=1))

    def test_calendar_missing_and_schema(self):
        d = frame()
        d = d.drop(index=30)
        f = calendar_market_features(d)
        self.assertEqual(len(f), 70)
        self.assertTrue(np.isnan(f.loc[36, "log_return_6h"]))
        self.assertTrue(np.isnan(f.loc[31, "realised_vol_6h"]))
        d = frame()
        d.loc[40, "hours_since_venue_start"] = np.nan
        c, _ = prepare_corrected(d)
        self.assertFalse(c.loc[40, "prediction_eligible_market_h6"])
        self.assertIn("prediction_eligible_market_onchain_h24", c)
        # Operational flag does not require a future outcome.
        self.assertTrue(c.iloc[-1].prediction_eligible_market)

    def test_negative_confirmation_and_censoring(self):
        d = frame(10)
        d.loc[8, "price"] = np.nan
        eps = pd.DataFrame(columns=["coin", "start", "episode_id"])
        eps["start"] = pd.to_datetime(eps.start, utc=True)
        labels = episode_labels(d, eps)
        self.assertTrue(np.isnan(labels.loc[6, "label_h1"]))
        self.assertTrue(np.isnan(labels.loc[9, "label_h1"]))

    def test_h1_strict_start_inclusive_end_and_negative_observation(self):
        d = frame(10)
        starts = pd.DataFrame({
            "coin": ["A", "A"],
            "start": pd.to_datetime([d.hour.iloc[2], d.hour.iloc[5]], utc=True),
        })
        labels = episode_labels(d, starts, horizons=(1,))
        self.assertEqual(labels.loc[0, "label_h1"], 0)
        self.assertEqual(labels.loc[1, "label_h1"], 1)
        self.assertEqual(labels.loc[2, "label_h1"], 0)
        self.assertEqual(labels.loc[4, "label_h1"], 1)
        self.assertEqual(labels.loc[5, "label_h1"], 0)
        self.assertTrue(np.isnan(labels.loc[9, "label_h1"]))

        d.loc[2, "price"] = np.nan
        empty = starts.iloc[:0].copy()
        censored = episode_labels(d, empty, horizons=(1,))
        self.assertTrue(np.isnan(censored.loc[0, "label_h1"]))

    def test_h1_decision_and_onset_use_completed_bar_closes(self):
        d = frame(12)
        d.loc[[4, 5], "price"] = .98
        corrected, episodes = prepare_corrected(d)
        self.assertEqual(episodes.iloc[0].start, d.hour.iloc[4] + pd.Timedelta(hours=1))
        self.assertEqual(corrected.loc[2, "hour"], d.hour.iloc[2] + pd.Timedelta(hours=1))
        self.assertEqual(corrected.loc[2, "label_h1"], 0)
        self.assertEqual(corrected.loc[3, "label_h1"], 1)

    def test_h1_alarm_target_and_cooldown_endpoints_are_inclusive(self):
        hours = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
        predictions = pd.DataFrame({
            "coin": "A", "hour": hours, "horizon": 1, "y_true": [1, 0, 0],
            "y_score": .9,
        })
        episodes = pd.DataFrame({
            "coin": ["A"], "episode_id": ["A_1"], "start": hours[[1]],
        })
        _, events = build_alarm_events(predictions, episodes, .5)
        self.assertEqual(len(events), 2)
        self.assertTrue(events.iloc[0].is_true_alarm)
        self.assertEqual(events.iloc[0].lead_hours, 1)
        self.assertFalse(events.iloc[1].is_true_alarm)
        self.assertEqual(events.iloc[1].alarm_hour, hours[2])

    def test_feature_clock_prefix_invariant(self):
        d = frame()
        baseline = calendar_market_features(d)
        changed = d.copy()
        changed.loc[45:, ["price", "trading_volume"]] = [.97, 100000.]
        revised = calendar_market_features(changed)
        columns = [c for c in baseline if "return" in c or "vol_" in c or "volume_pct" in c]
        pd.testing.assert_frame_equal(baseline.loc[:44, columns], revised.loc[:44, columns])
        with self.assertRaises(ValueError):
            calendar_market_features(pd.concat([d, d.iloc[[0]]]))

    def test_threshold_invalid_inputs(self):
        hours = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
        p = pd.DataFrame({"coin": "A", "hour": hours, "horizon": 6,
                          "y_true": 0, "y_score": [0.2, np.nan]})
        eps = pd.DataFrame({"coin": ["A"], "episode_id": ["a"], "start": hours[[1]]})
        with self.assertRaises(ValueError):
            _choose_threshold(p, eps, 6, 1)
        p.y_score = .2
        with self.assertRaises(ValueError):
            _choose_threshold(p, eps, 6, -1)
        with self.assertRaises(ValueError):
            _choose_threshold(p.iloc[:0], eps, 6, 1)

    def test_empty_alarm_pooling_cost_dtype(self):
        hours = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
        p = pd.DataFrame({"coin": "A", "hour": hours, "horizon": 6,
                          "y_true": 0, "y_score": .2})
        eps = pd.DataFrame({"coin": ["B"], "episode_id": ["b"], "start": hours[[1]]})
        _, empty = build_alarm_events(p, eps, np.inf)
        _, fired = build_alarm_events(p, eps, .2)
        pooled = pd.concat([empty, fired], ignore_index=True)
        self.assertEqual(pooled.is_true_alarm.dtype, bool)
        self.assertEqual(_false_alarm_stats(pooled, p), (1, 1, 1., 1))
        pooled.is_true_alarm = pooled.is_true_alarm.astype(object)
        self.assertEqual(_false_alarm_stats(pooled, p), (1, 1, 1., 1))

    def test_threshold_oracle(self):
        rng = np.random.default_rng(42)
        for rep in range(25):
            n, h = 40, 6
            d = pd.DataFrame({"coin": np.repeat(["A", "B"], n),
                              "hour": list(pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")) * 2,
                              "horizon": h, "y_score": rng.integers(0, 10, n * 2) / 10})
            eps = pd.DataFrame({"coin": ["A", "A", "B"],
                                "episode_id": ["a1", "a2", "b1"],
                                "start": pd.to_datetime(["2020-01-01 02:00Z", "2020-01-02 08:00Z",
                                                         "2020-01-01 20:00Z"])})
            d["y_true"] = [
                int(((eps.coin == r.coin) & (eps.start > r.hour)
                     & (eps.start <= r.hour + pd.Timedelta(hours=h))).any())
                for r in d.itertuples()
            ]
            budget = .5 + rep % 4
            targets = _targetable_episode_ids(d, eps, h)
            expected = []
            ordered = d.sort_values(["coin", "hour"]).reset_index(drop=True)
            target_codes = np.full(len(d), -1, dtype=np.int64)
            for i, r in enumerate(ordered.itertuples()):
                hit = eps[(eps.coin == r.coin) & (eps.start > r.hour)
                          & (eps.start <= r.hour + pd.Timedelta(hours=h))]
                if len(hit):
                    target_codes[i] = int(hit.index[0])
            cutoffs, nc, nf, na = exhaustive_counts(
                ordered.y_score.to_numpy(), ordered.hour.astype(
                    "datetime64[ns, UTC]").astype("int64").to_numpy(),
                np.array([n, 2*n], dtype=np.int64), target_codes, h*3600_000_000_000)
            for threshold in np.r_[np.unique(d.y_score), np.inf]:
                _, events = build_alarm_events(d, eps, threshold)
                false = int((~events.is_true_alarm).sum()) if len(events) else 0
                caught = set(events.target_episode_id.dropna()) & targets
                if np.isfinite(threshold):
                    j = np.flatnonzero(cutoffs == threshold)[0]
                    self.assertEqual((int(nc[j]), int(nf[j]), int(na[j])),
                                     (len(caught), false, len(events)))
                if false <= budget * _observed_coin_months(d):
                    expected.append((len(caught), -false, threshold))
            actual, _ = _choose_threshold(d, eps, h, budget)
            self.assertEqual(actual, max(expected)[2])

    def test_nonmonotone_counterexample(self):
        hours = pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC")
        p = pd.DataFrame({"coin": "A", "hour": hours[[0, 4, 11]],
                          "horizon": 6, "y_true": [1, 0, 0], "y_score": [.2, .9, .8]})
        eps = pd.DataFrame({"coin": ["A"], "episode_id": ["A_1"], "start": hours[[2]]})
        threshold, stats = _choose_threshold(p, eps, 6, 1)
        self.assertEqual(threshold, .2)
        self.assertEqual(stats["episode_recall"], 1)
        self.assertEqual(stats["false_alarms"], 1)

    def test_fit_roundtrip(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
        y = (X.a > 0).astype(int)
        model = XGBModel(max_depth=3, learning_rate=.05, n_estimators=300, n_jobs=4)
        model.fit(X.iloc[:50], y.iloc[:50])
        self.assertEqual(model.clf_.get_booster().num_boosted_rounds(), 300)
        # Write scratch only inside the project; remove it even after failure.
        artifact = Path("repair-test-model.joblib")
        try:
            joblib.dump(model, artifact)
            np.testing.assert_allclose(model.predict_proba(X.iloc[50:]),
                                       joblib.load(artifact).predict_proba(X.iloc[50:]), atol=1e-8)
        finally:
            artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
