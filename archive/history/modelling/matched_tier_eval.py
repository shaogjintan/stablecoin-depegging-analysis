"""Matched-sample T1 vs T2 metrics from saved pooled-test predictions."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

rows = []

for horizon in (6, 24):
    path_t1 = Path(
        f"results/predictions/logreg_t1_h{horizon}_lr01_pooled_test.parquet"
    )
    path_t2 = Path(
        f"results/predictions/logreg_t2_h{horizon}_lr01_pooled_test.parquet"
    )

    t1 = pd.read_parquet(path_t1)[
        ["coin", "hour", "y_true", "y_score"]
    ].rename(columns={"y_true": "y_t1", "y_score": "score_t1"})

    t2 = pd.read_parquet(path_t2)[
        ["coin", "hour", "y_true", "y_score"]
    ].rename(columns={"y_true": "y_t2", "y_score": "score_t2"})

    matched = t1.merge(
        t2,
        on=["coin", "hour"],
        how="inner",
        validate="one_to_one",
    )

    if not matched["y_t1"].equals(matched["y_t2"]):
        raise ValueError(f"Label mismatch between tiers at horizon {horizon}.")

    y = matched["y_t1"]
    n = len(matched)
    n_positive = int(y.sum())

    if n == 0 or n_positive == 0 or n_positive == n:
        raise ValueError(
            f"Cannot calculate all metrics at horizon {horizon}: "
            f"matched rows={n}, positives={n_positive}."
        )

    for tier, score_col in ((1, "score_t1"), (2, "score_t2")):
        scores = matched[score_col]
        rows.append(
            {
                "horizon": horizon,
                "tier": tier,
                "matched_rows": n,
                "positives": n_positive,
                "average_precision": average_precision_score(y, scores),
                "roc_auc": roc_auc_score(y, scores),
                "brier": brier_score_loss(y, scores),
            }
        )

results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))

Path("reports").mkdir(exist_ok=True)
output_path = Path("reports/logreg_matched_tier_eval.csv")
results.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
