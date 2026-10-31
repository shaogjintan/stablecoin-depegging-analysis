"""
annual_calibration.py  (professor feedback #3)
Per test year: calibrate each model's threshold to hit ~budget false alarms
per coin-month, then compare episodes caught and average lead time.

Usage:
    python3 modelling/annual_calibration.py --save reports/annual_calibration.csv
"""
import argparse, glob
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save",    default=None)
parser.add_argument("--budget",  type=float, default=1.0)
parser.add_argument("--horizon", type=int,   default=24)
args = parser.parse_args()

ep = pd.read_parquet("data/processed/depeg_episodes.parquet")
ep["start"] = pd.to_datetime(ep["start"], utc=True)
ep["end"]   = pd.to_datetime(ep["end"],   utc=True)

def compute_year_stats(pred, year, budget):
    df = pred[pred["year"] == year].copy()
    if df.empty:
        return None
    df["hour"] = pd.to_datetime(df["hour"], utc=True)

    # coin-months in this year's test window
    df["ym"] = df["hour"].dt.to_period("M").astype(str) + "_" + df["coin"]
    n_coin_months = df["ym"].nunique()
    budget_alarms = budget * n_coin_months

    rows = []
    for thresh in np.arange(0.01, 0.995, 0.005):
        alarms  = df[df["y_score"] >= thresh]
        fa      = alarms[alarms["y_true"] == 0]
        fa_rate = len(fa) / n_coin_months if n_coin_months > 0 else 0
        if fa_rate <= budget:
            # episode-level recall at this threshold
            ep_year = ep[ep["start"].dt.year == year]
            caught, leads = [], []
            for _, row in ep_year.iterrows():
                mask = (df["episode_id"] == row["episode_id"])
                ep_df = df[mask]
                if ep_df.empty:
                    continue
                fired = ep_df[ep_df["y_score"] >= thresh]
                if not fired.empty:
                    caught.append(1)
                    first_alarm = fired["hour"].min()
                    lead = (row["start"] - first_alarm).total_seconds() / 3600
                    leads.append(max(lead, 0))
                else:
                    caught.append(0)
            n_ep    = len(caught)
            n_caught= sum(caught)
            avg_lead= round(np.mean(leads), 1) if leads else 0
            rows.append(dict(year=year, threshold=round(thresh,3),
                             fa_per_coin_month=round(fa_rate,3),
                             n_episodes=n_ep, episodes_caught=n_caught,
                             recall=round(n_caught/n_ep,3) if n_ep else 0,
                             avg_lead_hours=avg_lead))
    # pick the threshold closest to budget (highest fa_rate <= budget)
    if not rows:
        return None
    best = max(rows, key=lambda r: r["fa_per_coin_month"])
    return best

# Load T1 and T2
t1_files = sorted(glob.glob(f"results/predictions/logreg_t1_h{args.horizon}_*_pooled_test.parquet"))
t2_files = sorted(glob.glob(f"results/predictions/logreg_t2_h{args.horizon}_*_pooled_test.parquet"))

if not t1_files or not t2_files:
    raise FileNotFoundError("Cannot find T1 or T2 prediction parquets. Run train.py first.")

t1 = pd.concat([pd.read_parquet(f) for f in t1_files])
t2 = pd.concat([pd.read_parquet(f) for f in t2_files])

years = sorted(t1["year"].unique())
print(f"\nTest years found: {years}")

results = []
for year in years:
    r1 = compute_year_stats(t1, year, args.budget)
    r2 = compute_year_stats(t2, year, args.budget)
    if r1: results.append({"tier":"T1 (market only)", **r1})
    if r2: results.append({"tier":"T2 (+on-chain)",   **r2})

out = pd.DataFrame(results)
cols = ["year","tier","threshold","fa_per_coin_month","n_episodes",
        "episodes_caught","recall","avg_lead_hours"]
out = out[cols].sort_values(["year","tier"])

print("\n=== Per-year calibrated comparison (h={}) ===".format(args.horizon))
print(out.to_string(index=False))

if args.save:
    out.to_csv(args.save, index=False)
    print(f"\nSaved to {args.save}")
