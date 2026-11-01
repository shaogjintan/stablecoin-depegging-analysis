"""Helper library: rolling false-alarm-budget threshold protocol.
Pure helpers used by the local drivers in `plans/` (`local_curve.py`,
`ex_post_comparison.py`). For each expanding-window test year Y the caller
fits a practice model once via `_validation_and_test_scores`, then picks
thresholds with `_choose_threshold` on either the prior year (real-time) or
the test year itself (ex post), both subject to an average false-alarm
budget of at most one false alarm per observed coin-month. A false alarm is
counted using the cooldown/event definition in `modelling/alarm_events.py`.
There is deliberately no CLI: import the helpers and drive them from a
script in `plans/` (the group's local-driver pattern).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
from modelling.alarm_events import build_alarm_events  # noqa: E402
from modelling.models import get_model  # noqa: E402
from modelling.models.base import MedianImputer  # noqa: E402
def _info_path(model: str, tier: int, horizon: int, year: int,
               model_dir: Path) -> Path:
    stem = f"{model}_t{tier}_h{horizon}_{year}"
    return model_dir / f"{stem}.json"
def _fold_model_params(run_info: dict, model_name: str) -> dict:
    """Recover the selected constructor parameters from train.py's run log."""
    if not run_info.get("tune"):
        return dict(run_info.get("model_params") or {})
    log = run_info.get("tune_log") or {}
    metric = log.get("tune_metric") or run_info.get("tune_metric")
    combos = log.get("combos", [])
    score_col = f"inner_{metric}"
    scored = [entry for entry in combos if entry.get(score_col) is not None]
    if not scored:
        raise ValueError(
            f"Tuned run log has no candidate scores for {model_name} ({metric})"
        )
    best = max(scored, key=lambda entry: float(entry[score_col]))
    params = dict(best["params"])
    best_iteration = best.get("best_iteration")
    if best_iteration is not None:
        params["n_estimators"] = max(int(best_iteration) + 1, 1)
    return params
def _validation_and_test_scores(
    fold: dict, model_name: str, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on inner_train and score inner_val and test with that model."""
    model = get_model(model_name, **params)
    X_train = fold["X_inner_train"]
    X_val = fold["X_inner_val"]
    X_test = fold["X_test"]
    imp = None
    if model.needs_imputation:
        imp = MedianImputer().fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
    model.fit(
        X_train,
        fold["y_inner_train"],
        X_val=X_val,
        y_val=fold["y_inner_val"],
    )
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return val_scores, test_scores
def _validation_prediction_frame(fold: dict, scores: np.ndarray,
                                 horizon: int) -> pd.DataFrame:
    frame = fold["inner_val_meta"].copy()
    frame["horizon"] = horizon
    frame["y_true"] = fold["y_inner_val"].reset_index(drop=True)
    frame["y_score"] = scores
    return frame
def _targetable_episode_ids(predictions: pd.DataFrame,
                            episodes: pd.DataFrame,
                            horizon: int) -> set[str]:
    """Episodes that at least one prediction origin could forecast in-window."""
    targets: set[str] = set()
    starts = {
        coin: group.sort_values("start")
        for coin, group in episodes.groupby("coin", sort=False)
    }
    positives = predictions.loc[predictions["y_true"].eq(1)]
    delta = pd.Timedelta(hours=horizon)
    for row in positives.itertuples(index=False):
        coin_eps = starts.get(row.coin)
        if coin_eps is None:
            continue
        hit = coin_eps.loc[
            (coin_eps["start"] > row.hour)
            & (coin_eps["start"] <= row.hour + delta)
        ]
        if not hit.empty:
            targets.add(str(hit.iloc[0]["episode_id"]))
    return targets
def _observed_coin_months(predictions: pd.DataFrame) -> int:
    months = predictions[["coin", "hour"]].copy()
    months["coin_month"] = pd.to_datetime(months["hour"], utc=True).dt.strftime("%Y-%m")
    return int(months[["coin", "coin_month"]].drop_duplicates().shape[0])
def _false_alarm_stats(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    n_coin_months: int | None = None,
) -> tuple[int, int, float, int]:
    if n_coin_months is None:
        n_coin_months = _observed_coin_months(predictions)
    false_count = (
        int(events["is_true_alarm"].eq(False).sum()) if not events.empty else 0
    )
    rate = false_count / n_coin_months if n_coin_months else float("inf")
    if false_count:
        false_by_month = (
            events.loc[events["is_true_alarm"].eq(False)]
            .groupby(["coin", "coin_month"])
            .size()
        )
        max_false_by_month = int(false_by_month.max())
    else:
        max_false_by_month = 0
    return false_count, n_coin_months, rate, max_false_by_month
def _choose_threshold(
    validation: pd.DataFrame,
    episodes: pd.DataFrame,
    horizon: int,
    budget: float,
) -> tuple[float, dict]:
    """Max validation episode recall under grouped FA/observed coin-month budget.

    Evaluate every distinct score and +inf; tie: fewer false alarms, then
    higher threshold. Cooldown counts are NOT monotone. Never use test labels.
    """
    scores = validation["y_score"].to_numpy(dtype=float)
    if not len(scores):
        raise ValueError("Validation fold has no scored rows")
    if not np.isfinite(scores).all():
        bad_count = int((~np.isfinite(scores)).sum())
        raise ValueError(
            f"Validation scores contain {bad_count} non-finite value(s); "
            "cannot select an alarm threshold"
        )
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be finite and nonnegative")
    target_ids = _targetable_episode_ids(validation, episodes, horizon)
    n_validation_coin_months = _observed_coin_months(validation)
    def evaluate(threshold: float) -> dict:
        _, events = build_alarm_events(validation, episodes, float(threshold))
        false_count, coin_months, rate, max_false_by_month = _false_alarm_stats(
            events, validation, n_validation_coin_months
        )
        caught = (
            set(events.loc[events["is_true_alarm"], "target_episode_id"].dropna().astype(str))
            if not events.empty else set()
        )
        recall = len(caught & target_ids) / len(target_ids) if target_ids else 0.0
        return {
            "threshold": float(threshold),
            "false_alarms": false_count,
            "coin_months": coin_months,
            "false_alarms_per_coin_month": rate,
            "max_false_alarms_in_coin_month": max_false_by_month,
            "episodes_targetable": len(target_ids),
            "episodes_warned": len(caught & target_ids),
            "episode_recall": recall,
            "alarm_events": len(events),
        }
    from modelling.exact_threshold import exhaustive_counts
    frame = validation.sort_values(["coin", "hour"]).copy()
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True)
    if frame.duplicated(["coin", "hour"]).any():
        raise ValueError("Duplicate validation coin-hour")
    times = frame.hour.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    targets = np.full(len(frame), -1, dtype=np.int64)
    ids = {ep: i for i, ep in enumerate(sorted(target_ids))}
    delta = horizon * 3600_000_000_000
    for coin, group in frame.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        positions = np.flatnonzero(frame.coin.eq(coin))
        loc = np.searchsorted(starts, times[positions], side="right")
        for pos, j in zip(positions, loc):
            if j < len(starts) and starts[j] <= times[pos] + delta:
                targets[pos] = ids.get(str(eps.iloc[j].episode_id), -1)
    ends = frame.groupby("coin", sort=False).size().cumsum().to_numpy(dtype=np.int64)
    cutoffs, caught, false, _ = exhaustive_counts(
        frame.y_score.to_numpy(dtype=float), times, ends, targets, delta)
    cutoffs = np.r_[cutoffs, np.inf]
    caught, false = np.r_[caught, 0], np.r_[false, 0]
    feasible = np.flatnonzero(false <= budget * n_validation_coin_months)
    idx = max(feasible, key=lambda i: (caught[i], -false[i], cutoffs[i]))
    best = evaluate(float(cutoffs[idx]))
    assert best["episodes_warned"] == caught[idx] and best["false_alarms"] == false[idx]
    best["candidate_count"] = len(cutoffs)
    return best["threshold"], best
