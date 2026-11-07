"""Historical pooled metrics, including unused Brier score and ROC-AUC.

Retained for reference; excluded from the maintained evaluation pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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

