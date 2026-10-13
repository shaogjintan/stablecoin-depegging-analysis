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


def select_threshold(y_inner, p_inner, criterion: str = "f1") -> dict:
    """Pick the decision threshold on inner-val predictions (default: best F1).

    Alyssa extends this hook with the full threshold-sweep / PR-curve
    analysis. Thresholds must be tuned on inner_val only -- never on test.
    """
    yt, ys = _valid(y_inner, p_inner)
    out = {"threshold": float("nan"), "f1": float("nan"),
           "precision": float("nan"), "recall": float("nan"),
           "n": len(yt), "n_pos": int(yt.sum())}
    if len(yt) == 0 or yt.sum() == 0:
        return out
    precision, recall, thresholds = precision_recall_curve(yt, ys)
    eps = np.finfo(float).eps
    f1 = 2 * precision * recall / (precision + recall + eps)  # len = len(thresholds) + 1
    if criterion == "f1":
        i = int(np.nanargmax(f1[:-1]))
    else:
        raise ValueError(f"Unknown criterion '{criterion}' (only 'f1' implemented)")
    out.update(
        threshold=float(thresholds[i]),
        f1=float(f1[i]),
        precision=float(precision[i]),
        recall=float(recall[i]),
    )
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


def evaluate_run(preds: pd.DataFrame, tag: str,
                 out_dir: str | Path = RESULTS_DIR) -> dict:
    """Evaluate pooled OOS predictions: pooled + per-coin + per-episode tables
    and the alarm-curve CSV. `preds` needs year, horizon, coin, hour,
    episode_id, y_true, y_score (the pooled test frame saved by
    modelling/train.py)."""
    required = {"coin", "hour", "episode_id", "y_true", "y_score", "horizon"}
    missing = required - set(preds.columns)
    if missing:
        raise KeyError(f"predictions frame is missing columns {sorted(missing)}")

    episodes = pd.read_parquet(PROCESSED_DATA_DIR / "depeg_episodes.parquet")

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

    print("evaluate.py self-test: PASS "
          f"(PR-AUC perfect={m['pr_auc']}, flat={m_flat['pr_auc']}, "
          f"threshold={th['threshold']:.3f} F1={th['f1']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
