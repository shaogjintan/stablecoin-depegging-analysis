"""Independent checks for a modelling.refit_final run.

python -m modelling.verify_refit_run --run-dir <refit run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_processing.config import PURGE_HOURS
from modelling.refit_final import MODELS, REFIT, digest, save_json
from modelling.verify_lr_rf_run import assert_values, reference_events, reference_metrics


def _keys(frame):
    return set(zip(frame.coin, pd.to_datetime(frame.hour, utc=True)))


def verify(run):
    manifest = json.loads((run / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Run is not complete: {manifest['status']}")
    sources = {k: Path(v) for k, v in manifest["scope"]["sources"].items()}
    episodes = pd.read_parquet(sources["lrrf"] / "episodes.parquet")
    pd.testing.assert_frame_equal(episodes, pd.read_parquet(run / "episodes.parquet"), check_exact=True)
    yearly = pd.read_csv(run / "per_year.csv", float_precision="round_trip")
    pooled = pd.read_csv(run / "pooled.csv", float_precision="round_trip")
    scope = manifest["scope"]
    expected = {(m, c, h, y) for m in scope["models"] for c in scope["configs"]
                for h in scope["horizons"] for y in scope["test_years"]}
    got = set(zip(yearly.model, yearly.config, yearly.horizon_hours, yearly.test_year))
    if got != expected or len(yearly) != len(expected):
        raise AssertionError(f"Cell matrix mismatch: missing={expected - got} extra={got - expected}")
    predictions, alarms, checks = {}, {}, []
    for row in yearly.itertuples():
        model, config, h, year = row.model, row.config, int(row.horizon_hours), int(row.test_year)
        cell = f"{config}_h{h}_{year}"
        folder = run / "fits" / f"{model}_{cell}"
        fit = json.loads((folder / "fit.json").read_text())
        features = fit["features"]
        # Final training set: before test boundary minus purge, contains the practice
        # model's train and validation rows (taken from the source run), no test rows.
        train = pd.read_parquet(run / "partitions" / cell / "train.parquet", columns=["coin", "hour", "y_true"])
        test = pd.read_parquet(folder / "test.parquet")
        boundary = pd.Timestamp(f"{year}-01-01", tz="UTC")
        assert train.hour.max() < boundary - pd.Timedelta(hours=PURGE_HOURS), cell
        assert test.hour.min() >= boundary, cell
        practice = {p: pd.read_parquet(sources["lrrf"] / "partitions" / cell / f"{p}.parquet",
                                       columns=["coin", "hour"]) for p in ("inner_train", "inner_val", "test")}
        tk = _keys(train)
        assert _keys(practice["inner_train"]) <= tk and _keys(practice["inner_val"]) <= tk, cell
        assert not tk & _keys(practice["test"]), cell
        assert _keys(test) == _keys(practice["test"]), cell
        assert len(train) == row.train_rows and int(train.y_true.sum()) == row.train_positive_hours, cell
        # Cutoff copied from the verified practice-model lowest-feasible selection.
        evaluation = json.loads((sources["cutoff"] / "cells" / f"{model}_{cell}" / "evaluation.json").read_text())
        assert evaluation["threshold_policy"] == "lowest_feasible"
        assert float(evaluation["threshold"]) == fit["threshold"] == row.threshold, cell
        assert digest(sources["cutoff"] / "cells" / f"{model}_{cell}" / "evaluation.json") == fit["cutoff_evaluation_sha256"]
        practice_dir = sources["xgb"] / "fits" / cell if model == "xgboost" else sources["lrrf"] / "fits" / f"{model}_{cell}"
        practice_scores = pd.read_parquet(practice_dir / "test.parquet", columns=["y_score"]).y_score.to_numpy()
        if model in REFIT:
            artifact = folder / ("model.joblib" if model == "xgboost" else "pipeline.joblib")
            assert digest(artifact) == fit["model_sha256"], cell
            bundle = joblib.load(artifact)
            if model == "xgboost":
                scores = bundle.predict_proba(test[features])[:, 1]
                assert bundle.clf_.get_booster().num_boosted_rounds() == 300
            else:
                X = test[features] if bundle["imputer"] is None else bundle["imputer"].transform(test[features])
                scores = bundle["model"].predict_proba(X)[:, 1]
            np.testing.assert_allclose(scores, test.y_score, atol=1e-8, rtol=0, err_msg=cell)
            assert row.train_rows > row.practice_train_rows, cell
            assert not np.array_equal(scores, practice_scores), f"{cell}: final model equals practice model"
        else:
            np.testing.assert_array_equal(test.y_score.to_numpy(), practice_scores, err_msg=cell)
        pred = test[["coin", "hour", "y_score", "y_true", "horizon"]].copy()
        pred["threshold"] = fit["threshold"]
        events = reference_events(pred, episodes, fit["threshold"], h)
        saved = pd.read_csv(folder / "alarms.csv", float_precision="round_trip")
        saved["alarm_hour"] = pd.to_datetime(saved.alarm_hour, utc=True)
        pd.testing.assert_frame_equal(saved[events.columns], events, check_dtype=False, check_exact=True)
        assert_values(row._asdict(), reference_metrics(pred, events, episodes, h))
        predictions.setdefault((model, config, h), []).append(pred)
        alarms.setdefault((model, config, h), []).append(events)
        checks.append(dict(model=model, config=config, horizon_hours=h, test_year=year, passed=True))
    for (model, config, h), parts in predictions.items():
        row = pooled.loc[pooled.model.eq(model) & pooled.config.eq(config) & pooled.horizon_hours.eq(h)].iloc[0]
        assert_values(row, reference_metrics(pd.concat(parts, ignore_index=True),
                                             pd.concat(alarms[(model, config, h)], ignore_index=True), episodes, h))
    assert len(pooled) == len(predictions)
    for name, value in manifest["source_code_sha256"].items():
        assert digest(run / "source" / name) == value, name
    result = dict(status="passed", cells=len(yearly), pooled_rows=len(pooled),
                  final_train_excludes_test_and_contains_validation=True, cutoffs_copied_exactly=True,
                  saved_models_reproduce_scores=True, refit_models_differ_from_practice=True,
                  persistence_and_prior_unchanged=True, alarms_and_metrics_regenerated=True)
    pd.DataFrame(checks).to_csv(run / "verification_cells.csv", index=False)
    save_json(run / "verification.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    verify(parser.parse_args().run_dir.resolve())
