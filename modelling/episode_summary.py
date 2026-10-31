"""
episode_summary.py  (professor feedback #1)
Episode-level false alarm + recall table at operating threshold (<=1 FA/coin-month).
Compares T1 and T2 side-by-side.
"""
import pandas as pd, numpy as np, glob

HORIZON = 24
BUDGET  = 1.0   # max false alarms per coin-month

# Load episodes with event types
ep = pd.read_parquet("data/processed/depeg_episodes_typed.parquet")
ep["start"] = pd.to_datetime(ep["start"], utc=True)

# Load predictions
t1 = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(f"results/predictions/logreg_t1_h{HORIZON}_*_pooled_test.parquet"))])
t2 = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(f"results/predictions/logreg_t2_h{HORIZON}_*_pooled_test.parquet"))])

for df in [t1, t2]:
    df["hour"] = pd.to_datetime(df["hour"], utc=True)

def operating_threshold(pred, budget):
    """Highest threshold where FA rate <= budget per coin-month."""
    pred["ym"] = pred["hour"].dt.to_period("M").astype(str) + "_" + pred["coin"]
    n_cm = pred["ym"].nunique()
    best_t, best_far = 0.01, 999
    for t in np.arange(0.01, 0.995, 0.005):
        fa = ((pred["y_score"] >= t) & (pred["y_true"] == 0)).sum()
        far = fa / n_cm
        if far <= budget and abs(far - budget) < abs(best_far - budget):
            best_t, best_far = t, far
    return round(best_t, 3), round(best_far, 3)

t1_thresh, t1_far = operating_threshold(t1.copy(), BUDGET)
t2_thresh, t2_far = operating_threshold(t2.copy(), BUDGET)
print(f"T1 operating threshold: {t1_thresh}  (FA/coin-month: {t1_far})")
print(f"T2 operating threshold: {t2_thresh}  (FA/coin-month: {t2_far})\n")

def episode_flags(pred, thresh, ep_df):
    rows = []
    for _, ep_row in ep_df.iterrows():
        mask = pred["episode_id"] == ep_row["episode_id"]
        ep_pred = pred[mask]
        if ep_pred.empty:
            rows.append(dict(episode_id=ep_row["episode_id"], caught=False, lead_hours=None))
            continue
        fired = ep_pred[ep_pred["y_score"] >= thresh]
        if not fired.empty:
            first = fired["hour"].min()
            lead  = max((ep_row["start"] - first).total_seconds() / 3600, 0)
            rows.append(dict(episode_id=ep_row["episode_id"], caught=True,
                             lead_hours=round(lead, 1)))
        else:
            rows.append(dict(episode_id=ep_row["episode_id"], caught=False, lead_hours=None))
    return pd.DataFrame(rows).set_index("episode_id")

# Only test-set episodes (2024-25 in expanding window)
test_ep = ep[ep["start"].dt.year >= 2024].copy()
# If few test episodes, also show 2023
if len(test_ep) < 5:
    test_ep = ep[ep["start"].dt.year >= 2023].copy()

flags1 = episode_flags(t1, t1_thresh, test_ep)
flags2 = episode_flags(t2, t2_thresh, test_ep)

summary = test_ep[["episode_id","coin","start","n_hours","min_price","event_type"]].copy()
summary = summary.set_index("episode_id")
summary["caught_T1"]     = flags1["caught"]
summary["lead_T1_h"]     = flags1["lead_hours"]
summary["caught_T2"]     = flags2["caught"]
summary["lead_T2_h"]     = flags2["lead_hours"]
summary["start"] = summary["start"].dt.strftime("%Y-%m-%d")

print("=== Episode-level results (test set, h=24) ===")
print(summary.to_string())
print(f"\nT1 episode recall: {summary['caught_T1'].sum()}/{len(summary)}")
print(f"T2 episode recall: {summary['caught_T2'].sum()}/{len(summary)}")
