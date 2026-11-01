import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from data_processing import splits
from modelling.corrected_lr_rf import prepare_partitions, aggregate
from modelling.corrected_xgb import run_shap
from modelling.train_logreg_yearcv import _calendar_cv_folds
from modelling import corrected_xgb, corrected_baselines, corrected_lr_rf
from modelling.corrected_h1 import run as run_h1
from modelling.evaluate import evaluate_run
from modelling.corrected_data import corrected_folds
from modelling.prepare_data import MARKET_FEATURES
from modelling.verify_corrected_run import read_alarm_csv


class ReadinessTests(unittest.TestCase):
    def test_alarm_csv_reader_preserves_exact_saved_float_scores(self):
        values = [0.9447813630104065, 0.9182649254798889, 0.9264349341392517]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.csv"
            pd.DataFrame({"y_score": values, "lead_hours": [1., 6., 24.]}).to_csv(
                path, index=False)
            np.testing.assert_array_equal(read_alarm_csv(path).y_score, values)

    def test_committed_typed_manifest_is_a_supported_input(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "master.parquet"
            episodes = root / "episodes.parquet"
            typed = root / "typed.parquet"
            hour = pd.Timestamp("2021-01-01", tz="UTC")
            pd.DataFrame({"dataset_version": ["corrected_v1"], "coin": ["A"],
                          "hour": [hour], "bar_start": [hour-pd.Timedelta(hours=1)],
                          "episode_id": ["A_1"], "qualifying_depeg_hour": [True]}).to_parquet(master)
            pd.DataFrame({"dataset_version": ["corrected_v1"], "coin": ["A"],
                          "start": [hour], "end": [hour+pd.Timedelta(hours=1)], "episode_id": ["A_1"],
                          "event_type": ["test"]}).to_parquet(typed)
            with patch.multiple(splits, ACTIVE_MASTER_PATH=master,
                                ACTIVE_EPISODES_PATH=episodes,
                                ACTIVE_TYPED_EPISODES_PATH=typed):
                a, b = splits.load_master()
            self.assertEqual(a.dataset_version.iloc[0], "corrected_v1")
            self.assertEqual(b.event_type.iloc[0], "test")
            bad = pd.read_parquet(typed)
            bad["dataset_version"] = "legacy"
            bad.to_parquet(episodes)
            with patch.multiple(splits, ACTIVE_MASTER_PATH=master,
                                ACTIVE_EPISODES_PATH=episodes,
                                ACTIVE_TYPED_EPISODES_PATH=typed):
                with self.assertRaisesRegex(ValueError, "dataset version"):
                    splits.load_master()
            bad["dataset_version"] = "corrected_v1"
            bad["start"] += pd.Timedelta(hours=1)
            bad.to_parquet(episodes)
            with patch.multiple(splits, ACTIVE_MASTER_PATH=master,
                                ACTIVE_EPISODES_PATH=episodes,
                                ACTIVE_TYPED_EPISODES_PATH=typed):
                with self.assertRaisesRegex(ValueError, "do not match"):
                    splits.load_master()

    def test_outer_split_excludes_confirmation_at_next_year_boundary(self):
        boundary = pd.Timestamp("2021-01-01", tz="UTC")
        master = pd.DataFrame({
            "hour": pd.date_range(boundary - pd.Timedelta(hours=30),
                                 boundary + pd.Timedelta(hours=30), freq="h"),
            "episode_id": None,
        })
        episodes = pd.DataFrame({
            "start": pd.to_datetime([], utc=True),
            "end": pd.to_datetime([], utc=True), "episode_id": [],
        })
        fold = splits.expanding_window_folds(master, episodes, 2021, 2021)[0]
        self.assertLess(fold["train"].hour.max() + pd.Timedelta(hours=25), boundary)

    def test_inner_cv_excludes_confirmation_at_validation_boundary(self):
        hours = pd.to_datetime(["2018-06-01", "2018-06-02",
                                "2018-12-30 23:00", "2019-01-03"], utc=True, format="mixed")
        meta = pd.DataFrame({"hour": hours, "episode_id": None})
        episodes = pd.DataFrame({"start": pd.to_datetime([], utc=True),
                                 "end": pd.to_datetime([], utc=True), "episode_id": []})
        folds, _ = _calendar_cv_folds(pd.DataFrame(index=meta.index),
                                      pd.Series([0, 1, 0, 1]), meta, episodes)
        self.assertEqual(len(folds), 1)
        latest = hours[folds[0]["train_idx"]].max()
        self.assertLess(latest + pd.Timedelta(hours=25),
                        pd.Timestamp("2019-01-01", tz="UTC"))

    def test_standalone_lr_rf_partitions_need_no_frozen_run(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            hours = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
            frame = pd.DataFrame({"coin": "A", "hour": hours, "episode_id": None})
            fold = {"year": 2021, "features": ["peg_deviation"],
                    "dataset_version": "synthetic-test"}
            for part in ("train", "inner_train", "inner_val", "test"):
                fold[f"{part}_meta"] = frame
                fold[f"X_{part}"] = pd.DataFrame({"peg_deviation": [0., 0.]})
                fold[f"y_{part}"] = pd.Series([0, 1])
            with patch("modelling.corrected_lr_rf.load_master",
                       return_value=(pd.DataFrame(), pd.DataFrame())), \
                    patch("modelling.corrected_lr_rf.corrected_folds",
                          return_value=[fold]):
                _, cells = prepare_partitions(out, horizons=(1,))
            self.assertEqual(len(cells), 3)
            self.assertTrue(all(spec["horizon"] == 1 for spec in cells.values()))

    def test_shap_missing_scope_fails_before_writing_figures(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run.json").write_text(json.dumps({
                "horizons": [1], "configs": ["market_original", "market_matched",
                                           "market_onchain"],
            }))
            (out / "fits").mkdir()
            with self.assertRaisesRegex(ValueError, "Missing scoring fits"):
                run_shap(out)
            self.assertFalse((out / "figures").exists())
            (out / "run.json").write_text(json.dumps({
                "horizons": [1], "test_years": [2021], "shap_requested": False,
            }))
            with self.assertRaisesRegex(ValueError, "SHAP as omitted"):
                run_shap(out)
            self.assertFalse((out / "figures").exists())

    def test_scoring_failures_persist_incomplete_status_and_raise(self):
        for module in (corrected_xgb, corrected_baselines, corrected_lr_rf):
            with self.subTest(module=module.__name__), TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"

                def fail(output, *args):
                    output.mkdir()
                    (output / "run.json").write_text(json.dumps({"status": "running"}))
                    raise RuntimeError("test scoring failure")

                with patch.object(module, "_run", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "test scoring failure"):
                        module.run(out)
                info = json.loads((out / "run.json").read_text())
                self.assertEqual(info["status"], "incomplete")
                self.assertIn("test scoring failure", info["error"])
                with self.assertRaises(FileExistsError):
                    module.run(out)
                self.assertEqual(info, json.loads((out / "run.json").read_text()))

    def test_h1_defaults_to_portable_runner_and_independent_verification(self):
        out = Path("unused-h1-test-output")
        with patch("modelling.corrected_h1.run_portable") as scoring, \
                patch("modelling.verify_lr_rf_run.verify") as verifying:
            run_h1(out)
        scoring.assert_called_once_with(out, horizons=(1,))
        verifying.assert_called_once_with(out)

    def test_legacy_evaluator_rejects_unknown_or_mixed_version(self):
        pred = pd.DataFrame({"coin": ["A"], "hour": [pd.Timestamp("2021-01-01", tz="UTC")],
                             "episode_id": [None], "y_true": [0], "y_score": [0.],
                             "horizon": [1], "dataset_version": ["legacy"]})
        with self.assertRaisesRegex(ValueError, "Unknown/mixed"):
            evaluate_run(pred, "unused")
        pred["dataset_version"] = "corrected_v1"
        with self.assertRaisesRegex(ValueError, "versions differ"):
            evaluate_run(pred, "unused", episodes=pd.DataFrame({"dataset_version": ["legacy"]}))

    def test_aggregation_respects_selected_year_and_baseline_only_scope(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = out / "fits" / "baseline_market_original_h1_2021"
            folder.mkdir(parents=True)
            hour = pd.Timestamp("2021-01-03", tz="UTC")
            pred = pd.DataFrame({"coin": ["A"], "hour": [hour], "y_true": [1],
                                 "y_score": [.1], "threshold": [np.inf],
                                 "horizon": [1], "test_year": [2021]})
            pred.to_parquet(folder / "test.parquet", index=False)
            pd.DataFrame(columns=["coin", "alarm_hour", "is_true_alarm",
                                  "target_episode_id", "lead_hours"]).to_csv(
                folder / "alarms.csv", index=False)
            pd.DataFrame({"model": ["baseline"], "test_year": [2021]}).to_csv(
                out / "per_year.csv", index=False)
            episodes = pd.DataFrame({"coin": ["A"], "episode_id": ["A_1"],
                                     "start": [hour+pd.Timedelta(hours=1)]})
            aggregate(out, episodes, "test", horizons=(1,), years=(2021,),
                      models=("baseline",), configs=("market_original",))
            summary = pd.read_csv(out / "summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.targetable_episodes.iloc[0], 1)
            self.assertEqual(summary.episodes_caught.iloc[0], 0)

    def test_corrected_fold_metadata_keeps_dataset_version(self):
        hours = pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"], utc=True)
        master = pd.DataFrame({"coin": "A", "hour": hours, "episode_id": None,
                               "dataset_version": "corrected_v1", "label_h1": 0,
                               "prediction_eligible_market_h1": True,
                               **{f: 0. for f in MARKET_FEATURES}})
        fold = {"year": 2021, "train": master.iloc[:2], "inner_train": master.iloc[:1],
                "inner_val": master.iloc[1:2], "test": master.iloc[2:]}
        with patch("modelling.corrected_data.expanding_window_folds", return_value=[fold]):
            result = corrected_folds(master, pd.DataFrame(), 1, 1)[0]
        for part in ("train", "inner_train", "inner_val", "test"):
            self.assertTrue(result[f"{part}_meta"].dataset_version.eq("corrected_v1").all())

    def test_versioned_evaluator_selects_matching_active_episode_manifest(self):
        pred = pd.DataFrame({"coin": ["A"], "hour": [pd.Timestamp("2021-01-01", tz="UTC")],
                             "episode_id": [None], "y_true": [0], "y_score": [0.],
                             "horizon": [1], "dataset_version": ["corrected_v1"]})
        episodes = pd.DataFrame({"dataset_version": ["corrected_v1"]})
        with TemporaryDirectory() as tmp, \
                patch("data_processing.splits.load_master", return_value=(pd.DataFrame(), episodes)), \
                patch("modelling.evaluate.alarm_curve", return_value=pd.DataFrame()), \
                patch("modelling.evaluate.pooled_metrics", return_value={}), \
                patch("modelling.evaluate.per_coin_table", return_value=pd.DataFrame()), \
                patch("modelling.evaluate.per_episode_table", return_value=pd.DataFrame()) as report:
            evaluate_run(pred, "test", out_dir=tmp)
        self.assertIs(report.call_args.args[1], episodes)


if __name__ == "__main__":
    unittest.main()
