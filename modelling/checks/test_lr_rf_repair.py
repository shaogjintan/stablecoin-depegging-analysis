import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modelling.corrected_lr_rf import RF_PARAMS, fit_lr, select_c, score_cell, tune_lr
from modelling.train_logreg_yearcv import _calendar_cv_folds
from modelling.verify_lr_rf_run import reference_ap, reference_events, reference_targets, check_preprocessing
from modelling.alarm_events import build_alarm_events
from modelling.rolling_threshold import _choose_threshold


class LRRFTests(unittest.TestCase):
    def test_ap_ties_and_fold_priors(self):
        for y, s in (([1, 0, 0, 1], [.1, .2, .2, .1]),
                     ([0, 1, 0, 1, 0], [.2] * 5)):
            self.assertAlmostEqual(reference_ap(y, s), average_precision_score(y, s))
        y, s = [1, 1, 0, 0, 0, 0], [.8, .8, .8, .1, .1, .1]
        self.assertNotAlmostEqual(reference_ap(y, s), np.mean(y))

    def test_c_tie_is_smallest_even_unsorted(self):
        self.assertEqual(select_c([{"C": 1., "mean_average_precision": .5},
                                   {"C": .001, "mean_average_precision": .5}]), .001)

    def test_training_preprocessing_only(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"peg_deviation": rng.normal(0, .001, 200),
                          "realised_vol_6h": rng.uniform(.001, .01, 200)})
        X.loc[0, "peg_deviation"] = np.nan
        train = X.assign(y_true=np.arange(200) % 2)
        model, imputer = fit_lr(X, train.y_true, .1)
        check_preprocessing({"model": model, "imputer": imputer}, train, list(X.columns))
        lo, mean, med = model.lo_.copy(), model.scaler_.mean_.copy(), imputer.medians_.copy()
        model.predict_proba(imputer.transform(X * 1000))
        pd.testing.assert_series_equal(model.lo_, lo)
        pd.testing.assert_series_equal(imputer.medians_, med)
        np.testing.assert_array_equal(model.scaler_.mean_, mean)

    def test_cv_skips_and_label_separation(self):
        hours = pd.to_datetime(["2018-01-02", "2018-12-30", "2019-01-03", "2019-07-01",
                                "2020-01-03", "2020-07-01"], utc=True)
        meta = pd.DataFrame({"coin": "A", "hour": hours, "episode_id": None})
        y = pd.Series([1, 0, 1, 0, 0, 0])
        X = pd.DataFrame({"x": np.arange(6)})
        eps = pd.DataFrame({"episode_id": [], "start": pd.to_datetime([], utc=True),
                            "end": pd.to_datetime([], utc=True)})
        folds, skipped = _calendar_cv_folds(X, y, meta, eps)
        self.assertEqual([f["validation_year"] for f in folds], [2019])
        self.assertEqual([r["reason"] for r in skipped],
                         ["training_years_do_not_contain_both_classes", "no_depeg_positives"])
        f = folds[0]
        self.assertGreater(meta.hour.iloc[f["val_idx"]].min() - meta.hour.iloc[f["train_idx"]].max(),
                           pd.Timedelta(hours=25))

    def test_no_cv_fallback(self):
        train = pd.DataFrame({"coin": "A", "hour": pd.date_range("2018-01-01", periods=20, freq="h", tz="UTC"),
                              "episode_id": None, "peg_deviation": 0., "y_true": np.arange(20) % 2})
        eps = pd.DataFrame({"episode_id": [], "start": pd.to_datetime([], utc=True),
                            "end": pd.to_datetime([], utc=True)})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "no usable"):
                tune_lr(train, ["peg_deviation"], eps, Path(d), 24)
            self.assertTrue((Path(d) / "cv_skipped.json").exists())

    def test_rf_fixed_saved_fit_scores_both_partitions(self):
        rng = np.random.default_rng(2)
        cell = "market_original_h6_2021"
        spec = dict(config="market_original", horizon=6, year=2021,
                    features=["peg_deviation", "realised_vol_6h"])
        eps = pd.DataFrame({"coin": ["A"], "episode_id": ["A_1"],
                            "start": pd.to_datetime(["2021-01-03 05:00Z"])})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            p = out / "partitions" / cell
            p.mkdir(parents=True)
            for part, date in (("inner_train", "2019-01-03"), ("inner_val", "2020-01-03"), ("test", "2021-01-03")):
                frame = pd.DataFrame({"coin": "A", "hour": pd.date_range(date, periods=100, freq="h", tz="UTC"),
                                      "episode_id": None, "peg_deviation": rng.normal(0, .001, 100),
                                      "realised_vol_6h": rng.uniform(.001, .01, 100), "y_true": np.arange(100) % 2})
                frame.to_parquet(p / f"{part}.parquet", index=False)
            score_cell(out, cell, spec, "random_forest", eps, "test")
            folder = out / "fits" / f"random_forest_{cell}"
            fit = json.loads((folder / "fit.json").read_text())
            for k, v in RF_PARAMS.items():
                self.assertEqual(fit["params"][k], v)
            self.assertFalse(fit["refit_after_calibration"])
            bundle = joblib.load(folder / "pipeline.joblib")
            for part in ("inner_val", "test"):
                X = pd.read_parquet(p / f"{part}.parquet")[spec["features"]]
                scores = bundle["model"].predict_proba(bundle["imputer"].transform(X))[:, 1]
                np.testing.assert_array_equal(scores, pd.read_parquet(folder / f"{part}.parquet").y_score)

    def test_independent_threshold_alarm_oracle(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            p = pd.DataFrame({"coin": np.repeat(["A", "B"], 24),
                              "hour": list(pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")) * 2,
                              "horizon": 6, "y_true": 0, "y_score": rng.integers(0, 5, 48) / 5})
            eps = pd.DataFrame({"coin": ["A", "B"], "episode_id": ["a", "b"],
                                "start": pd.to_datetime(["2020-01-01 08:00Z", "2020-01-01 20:00Z"])})
            p["y_true"] = [int(((eps.coin.eq(r.coin)) & (eps.start > r.hour) &
                               (eps.start <= r.hour + pd.Timedelta(hours=6))).any())
                           for r in p.itertuples()]
            targets = reference_targets(p, eps, 6)
            candidates = []
            for cutoff in np.r_[np.unique(p.y_score), np.inf]:
                ref = reference_events(p, eps, cutoff, 6)
                _, actual = build_alarm_events(p, eps, cutoff)
                pd.testing.assert_frame_equal(actual[ref.columns].reset_index(drop=True), ref,
                                              check_dtype=False)
                fa = int((~ref.is_true_alarm.astype(bool)).sum())
                caught = set(ref.target_episode_id.dropna()) & targets
                if fa <= 2:
                    candidates.append((len(caught), -fa, cutoff))
            self.assertEqual(_choose_threshold(p, eps, 6, 1.)[0], max(candidates)[2])


if __name__ == "__main__":
    unittest.main()
