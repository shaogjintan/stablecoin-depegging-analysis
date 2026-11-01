"""
Pooled out-of-sample evaluation for expanding-window CV predictions.

All inputs are prediction frames produced by `modelling/train.py` with
columns: year, horizon, coin, hour, episode_id, y_true, y_score -- never raw
CSVs (Rule 3: the only data contract is modelling/prepare_data.py). The
depeg episode manifest (depeg_episodes.parquet) is read only to map each
flagged onset hour to the episode it predicts.

Primary metric is PR-AUC: positives are 0.02% (h=1) to 0.57% (h=24) of rows,
so ROC-AUC is over-optimistic and the report tables pool across fold years
(2024/2025 test sets have ~6 positives each -- see README "Known quirks").

Run the self-test from the repo root:
    python modelling/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))
from config import PROCESSED_DATA_DIR  # noqa: E402

RESULTS_DIR = Path("results/evaluation")


def _valid(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where the label or score is missing."""
    yt = pd.Series(y_true).reset_index(drop=True)
    ys = pd.Series(y_score).reset_index(drop=True)
    keep = yt.notna() & ys.notna()
    return np.asarray(yt[keep], dtype=int), np.asarray(ys[keep], dtype=float)


def pooled_metrics(y_true, y_score) -> dict:
    """PR-AUC, ROC-AUC, Brier score + counts on pooled predictions."""
    yt, ys = _valid(y_true, y_score)
    n = len(ys)
    n_pos = int(yt.sum())
    out = {
        "n": n,
        "n_pos": n_pos,
        "pos_rate": n_pos / n if n else float("nan"),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "brier": float("nan"),
    }
    if n == 0:
        return out
    if n_pos > 0:
        out["pr_auc"] = float(average_precision_score(yt, ys))
    if 0 < n_pos < n:
        try:
            out["roc_auc"] = float(roc_auc_score(yt, ys))
        except ValueError:  # e.g. single score value
            out["roc_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(yt, ys))
    return out


def alarm_sweep(y_true, y_score) -> pd.DataFrame:
    """Exact sweep over every distinct score (descending): alarms fired,
    true positives, hour-level recall and alarm rate at each threshold.

    This is the backbone of the recall-first evaluation: it lets us pick the
    operating point (recall vs alarm rate) instead of a single precision/
    recall balance."""
    yt, ys = _valid(y_true, y_score)
    n = len(ys)
    n_pos = int(yt.sum())
    cols = ["threshold", "n_alarms", "tp", "recall", "alarm_rate"]
    if n == 0 or n_pos == 0:
        return pd.DataFrame(columns=cols)
    order = np.argsort(-ys, kind="stable")
    y_sorted = yt[order]
    s_sorted = ys[order]
    cum_tp = np.cumsum(y_sorted)
    change = np.concatenate([[True], s_sorted[1:] != s_sorted[:-1]])
    idx = np.where(change)[0]          # first occurrence of each distinct score
    ends = np.concatenate([idx[1:], [n]])  # exclusive end of each score block
    return pd.DataFrame({
        "threshold": s_sorted[idx],
        "n_alarms": ends,
        "tp": cum_tp[ends - 1],
        "recall": cum_tp[ends - 1] / n_pos,
        "alarm_rate": ends / n,
    })


def select_threshold(y_inner, p_inner, criterion: str = "f1",
                     target_recall: float = 0.9,
                     max_alarm_rate: float | None = None,
                     beta: float = 5) -> dict:
    """Pick the decision threshold on inner-val predictions.

    Criteria (the recall-first ones are for the early-warning rework):
      - "f1":                  best F1 (original behaviour)
      - "f_beta":              best F-beta (beta weights recall beta^2 times
                               precision; beta=5 => "miss nothing" leaning)
      - "recall_at_alarm_rate": maximize hour-level recall subject to
                               alarm_rate <= max_alarm_rate (requires
                               max_alarm_rate)
      - "rate_at_recall":      minimize alarm rate subject to
                               recall >= target_recall
    Thresholds must be tuned on inner_val only -- never on test.
    """
    yt, ys = _valid(y_inner, p_inner)
    n, n_pos = len(yt), int(yt.sum())
    out = {"threshold": float("nan"), "n": n, "n_pos": n_pos,
           "recall": float("nan"), "alarm_rate": float("nan"),
           "f1": float("nan"), "precision": float("nan")}
    if n == 0 or n_pos == 0:
        return out

    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(yt, ys)
        eps = np.finfo(float).eps
        f1 = 2 * precision * recall / (precision + recall + eps)
        i = int(np.nanargmax(f1[:-1]))
        out.update(threshold=float(thresholds[i]), f1=float(f1[i]),
                   precision=float(precision[i]), recall=float(recall[i]))
        return out

    if criterion == "f_beta":
        precision, recall, thresholds = precision_recall_curve(yt, ys)
        eps = np.finfo(float).eps
        fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
        i = int(np.nanargmax(fb[:-1]))
        out.update(threshold=float(thresholds[i]), f1=float("nan"),
                   precision=float(precision[i]), recall=float(recall[i]))
        return out

    sweep = alarm_sweep(yt, ys)
    if criterion == "recall_at_alarm_rate":
        if max_alarm_rate is None:
            raise ValueError("recall_at_alarm_rate requires max_alarm_rate")
        feasible = sweep[sweep["alarm_rate"] <= max_alarm_rate]
        if feasible.empty:  # even the top score exceeds the budget -> alarm on it anyway
            row = sweep.iloc[0]
        else:
            row = feasible.iloc[feasible["recall"].argmax()]
    elif criterion == "rate_at_recall":
        feasible = sweep[sweep["recall"] >= target_recall]
        if feasible.empty:  # can't hit the target -> the top-score row
            row = sweep.iloc[0]
        else:
            row = feasible.iloc[feasible["alarm_rate"].argmin()]
    else:
        raise ValueError(
            f"Unknown criterion '{criterion}' "
            "(f1, f_beta, recall_at_alarm_rate, rate_at_recall)")
    out.update(threshold=float(row["threshold"]),
               recall=float(row["recall"]),
               alarm_rate=float(row["alarm_rate"]),
               tp=int(row["tp"]), n_alarms=int(row["n_alarms"]))
    return out


def alarm_curve(y_true, y_score, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Alarm-performance sweep over decision thresholds (for the report).

    One row per threshold: how many alarms fire and how many of the positives
    they catch. Saved as a CSV by evaluate_run().
    """
    yt, ys = _valid(y_true, y_score)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        alarms = ys >= t
        tp = int((alarms & (yt == 1)).sum())
        fp = int((alarms & (yt == 0)).sum())
        n_alarms = tp + fp
        rows.append({
            "threshold": float(t),
            "n_alarms": n_alarms,
            "tp": tp,
            "fp": fp,
            "precision": tp / n_alarms if n_alarms else float("nan"),
            "recall": tp / int((yt == 1).sum()) if (yt == 1).sum() else float("nan"),
            "alarm_rate": n_alarms / len(yt) if len(yt) else float("nan"),
        })
    return pd.DataFrame(rows)


def per_coin_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics within each coin (out-of-sample test rows only)."""
    rows = []
    for coin, g in preds.groupby("coin", sort=True):
        rows.append({"coin": coin, **pooled_metrics(g["y_true"], g["y_score"])})
    return pd.DataFrame(rows)


def _episode_for(row: pd.Series, episodes: pd.DataFrame,
                 horizon: int) -> pd.Series | None:
    """The episode a positive row predicts: same coin, start in (t, t+h]."""
    coin, t = row["coin"], row["hour"]
    hits = episodes[
        (episodes["coin"] == coin)
        & (episodes["start"] > t)
        & (episodes["start"] <= t + pd.Timedelta(hours=horizon))
    ]
    return None if hits.empty else hits.iloc[0]


def per_episode_table(preds: pd.DataFrame,
                      episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per depeg episode: the model's strongest flag among the onset
    hours that predict it (label rows carry no episode id, so each positive
    row is mapped to its target episode via the manifest), its lead time and
    its rank within that coin's OOS test hours (rank 1 = highest score)."""
    horizon = int(preds["horizon"].iloc[0])
    pos = preds.loc[preds["y_true"] == 1]
    rows = []
    for _, r in pos.iterrows():
        ep = _episode_for(r, episodes, horizon)
        if ep is None:
            continue  # should not happen for onset-only labels
        coin_scores = preds.loc[preds["coin"] == r["coin"], "y_score"]
        rank = int((coin_scores > r["y_score"]).sum()) + 1
        rows.append({
            "episode_id": ep["episode_id"],
            "coin": r["coin"],
            "flagged_at": r["hour"],
            "episode_start": ep["start"],
            "lead_hours": float((ep["start"] - r["hour"]) / pd.Timedelta(hours=1)),
            "y_score": float(r["y_score"]),
            "rank_in_coin": rank,
            "coin_test_rows": int(coin_scores.notna().sum()),
        })
    if not rows:
        return pd.DataFrame(columns=["episode_id", "coin", "flagged_at",
                                     "episode_start", "lead_hours", "y_score",
                                     "rank_in_coin", "coin_test_rows"])
    df = pd.DataFrame(rows)
    # keep the best-ranked flag per episode
    df = (df.sort_values(["episode_id", "rank_in_coin"])
            .groupby("episode_id", as_index=False).first()
            .sort_values(["coin", "rank_in_coin"], ignore_index=True))
    return df


def episode_alarm_table(preds: pd.DataFrame, episodes: pd.DataFrame,
                        threshold: float) -> pd.DataFrame:
    """Per-episode alarm verdict at one threshold: caught (the episode's
    strongest onset-hour score >= threshold), the catch lead time, and the
    score. 'Caught' is the early-warning headline: an episode is missed only
    if NO hour that predicts it was alarmed."""
    per_ep = per_episode_table(preds, episodes)
    per_ep["caught"] = per_ep["y_score"] >= threshold
    per_ep["alarm_lead_hours"] = per_ep["lead_hours"].where(per_ep["caught"])
    return per_ep


def episode_recall_curve(preds: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Recall-first operating curve: for every hourly threshold, the hourly
    alarm rate (fraction of OOS hours alarmed) vs episode-level recall
    (fraction of OOS depeg episodes with at least one alarmed onset hour)."""
    per_ep = per_episode_table(preds, episodes)
    ep_scores = np.sort(per_ep["y_score"].to_numpy(dtype=float))  # ascending
    sweep = alarm_sweep(preds["y_true"], preds["y_score"])
    n_ep = len(ep_scores)
    if n_ep == 0:
        return sweep.assign(episode_recall=0.0)
    # count of episode scores >= threshold via searchsorted on ascending array
    counts_ge = n_ep - np.searchsorted(ep_scores, sweep["threshold"].to_numpy(), side="left")
    out = sweep.copy()
    out["episode_recall"] = counts_ge / n_ep
    return out[["threshold", "alarm_rate", "recall", "episode_recall"]]


def operating_points(curve: pd.DataFrame,
                     recall_targets=(0.90, 0.95, 0.99),
                     alarm_rates=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Summarize a recall-first curve.

    Returns:
      "recall_at_rate": DataFrame (rows = alarm-rate budgets) with the max
          episode recall achievable at each budget and the threshold;
      "rate_at_recall": DataFrame (rows = recall targets) with the min alarm
          rate needed to reach each target and the threshold."""
    recall_rows = []
    for rate in alarm_rates:
        feasible = curve[curve["alarm_rate"] <= rate]
        if feasible.empty:
            recall_rows.append({"alarm_rate": rate, "episode_recall": float("nan"),
                                "threshold": float("nan")})
        else:
            r = feasible.iloc[feasible["episode_recall"].argmax()]
            recall_rows.append({"alarm_rate": rate,
                                "episode_recall": float(r["episode_recall"]),
                                "threshold": float(r["threshold"])})
    rate_rows = []
    for target in recall_targets:
        feasible = curve[curve["episode_recall"] >= target]
        if feasible.empty:
            rate_rows.append({"target_recall": target, "alarm_rate": float("nan"),
                              "threshold": float("nan")})
        else:
            r = feasible.iloc[feasible["alarm_rate"].argmin()]
            rate_rows.append({"target_recall": target,
                              "alarm_rate": float(r["alarm_rate"]),
                              "threshold": float(r["threshold"])})
    return {"recall_at_rate": pd.DataFrame(recall_rows),
            "rate_at_recall": pd.DataFrame(rate_rows)}


def missed_episodes(alarm_table: pd.DataFrame) -> pd.DataFrame:
    """The episodes the model missed at the chosen threshold (false negatives)."""
    return alarm_table.loc[~alarm_table["caught"],
                           ["episode_id", "coin", "episode_start", "y_score"]]


def evaluate_run(preds: pd.DataFrame, tag: str,
                 out_dir: str | Path = RESULTS_DIR,
                 episodes: pd.DataFrame | None = None) -> dict:
    """Evaluate pooled OOS predictions: pooled + per-coin + per-episode tables
    and the alarm-curve CSV. `preds` needs year, horizon, coin, hour,
    episode_id, y_true, y_score (the pooled test frame saved by
    modelling/train.py)."""
    required = {"coin", "hour", "episode_id", "y_true", "y_score", "horizon"}
    missing = required - set(preds.columns)
    if missing:
        raise KeyError(f"predictions frame is missing columns {sorted(missing)}")

    if episodes is None:
        if "dataset_version" in preds:
            from data_processing.config import MODELLING_DATA_VERSION
            from data_processing.splits import load_master
            if not preds.dataset_version.eq(MODELLING_DATA_VERSION).all():
                raise ValueError("Unknown/mixed prediction dataset version; supply matching episodes explicitly")
            _, episodes = load_master()
        else:
            episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")
    elif "dataset_version" in preds:
        versions = set(preds.dataset_version.dropna())
        if ("dataset_version" not in episodes or preds.dataset_version.isna().any()
                or set(episodes.dataset_version.dropna()) != versions
                or episodes.dataset_version.isna().any()):
            raise ValueError("Prediction and episode dataset versions differ")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = alarm_curve(preds["y_true"], preds["y_score"])
    curve.to_csv(out_dir / f"{tag}_alarm_curve.csv", index=False)

    return {
        "tag": tag,
        "pooled": pooled_metrics(preds["y_true"], preds["y_score"]),
        "per_coin": per_coin_table(preds),
        "per_episode": per_episode_table(preds, episodes),
        "alarm_curve_path": str(out_dir / f"{tag}_alarm_curve.csv"),
    }


def print_report(res: dict, max_episodes: int = 15) -> None:
    print("\nPooled OOS metrics:")
    print(pd.DataFrame([res["pooled"]]).round(4).to_string(index=False))
    print("\nPer-coin (pooled OOS):")
    print(res["per_coin"].round(4).to_string(index=False))
    ep = res["per_episode"]
    print(f"\nPer-episode, best-ranked flag (top {min(max_episodes, len(ep))} of {len(ep)}):")
    print(ep.head(max_episodes).round(3).to_string(index=False))
    print(f"\nAlarm curve CSV: {res['alarm_curve_path']}")


def _self_test() -> int:
    """Synthetic checks: perfect ranking -> PR-AUC 1; threshold hook; alarm curve."""
    y = np.array([0, 0, 0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8])
    s_flat = np.full(6, 0.5)

    m = pooled_metrics(y, s_perfect)
    assert abs(m["pr_auc"] - 1.0) < 1e-9, f"perfect ranking should give PR-AUC 1, got {m['pr_auc']}"
    assert m["n"] == 6 and m["n_pos"] == 2
    assert 0.0 <= m["brier"] <= 1.0

    m_flat = pooled_metrics(y, s_flat)
    assert abs(m_flat["pr_auc"] - 2 / 6) < 1e-9, f"constant score PR-AUC = pos rate, got {m_flat['pr_auc']}"
    assert m_flat["roc_auc"] == 0.5  # sklearn tie-breaking convention on constant scores

    th = select_threshold(y, s_perfect)
    assert 0.4 < th["threshold"] < 0.9 and th["f1"] > 0.9, th

    curve = alarm_curve(y, s_perfect, thresholds=np.array([0.5, 0.85]))
    assert curve.loc[0, "tp"] == 2 and curve.loc[1, "tp"] == 1, curve

    # recall-first criteria: 4 negatives at 0.1..0.4, positives at 0.8/0.9
    y2 = np.array([0, 0, 0, 0, 1, 1])
    s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    sw = alarm_sweep(y2, s2)
    assert sw["alarm_rate"].is_monotonic_increasing, "rates rise as threshold descends"
    # top score 0.9 is a positive -> recall starts at 0.5; lowest threshold -> 1.0
    assert abs(sw["recall"].max() - 1.0) < 1e-12 and abs(sw["recall"].min() - 0.5) < 1e-12
    assert abs(sw["alarm_rate"].iloc[0] - 1 / 6) < 1e-12  # only the top score alarmed

    th_r = select_threshold(y2, s2, criterion="recall_at_alarm_rate", max_alarm_rate=0.5)
    # at <=50% alarm rate: alarm the two positives only -> recall 1.0, rate 2/6
    assert abs(th_r["recall"] - 1.0) < 1e-12 and abs(th_r["alarm_rate"] - 2 / 6) < 1e-12, th_r
    assert th_r["threshold"] > 0.4

    th_t = select_threshold(y2, s2, criterion="rate_at_recall", target_recall=1.0)
    assert abs(th_t["recall"] - 1.0) < 1e-12 and abs(th_t["alarm_rate"] - 2 / 6) < 1e-12, th_t

    th_b = select_threshold(y2, s2, criterion="f_beta", beta=5)
    assert abs(th_b["recall"] - 1.0) < 1e-12, th_b  # beta=5 -> prefer catching both

    print("evaluate.py self-test: PASS "
          f"(PR-AUC perfect={m['pr_auc']}, flat={m_flat['pr_auc']}, "
          f"threshold={th['threshold']:.3f} F1={th['f1']:.3f}, "
          f"recall@rate={th_r['recall']:.2f}@{th_r['alarm_rate']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
