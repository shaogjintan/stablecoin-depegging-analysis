"""LogReg episode-level operating point (<=1 false alarm per coin-month).
Exploratory: threshold is chosen on the pooled test predictions."""
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, ".")
for p in Path(".").rglob("config.py"):
    if ".venv" not in p.parts and "site-packages" not in p.parts:
        sys.path.insert(0, str(p.parent.resolve()))
        break

from modelling.alarm_events import build_alarm_events
from modelling import rolling_threshold as rt

episodes = pd.read_parquet("data/processed/depeg_episodes.parquet")
BUDGET = 1.0

def call(fn, *args, horizon=None):
    if horizon is not None and "horizon" in inspect.signature(fn).parameters:
        return fn(*args, horizon=horizon)
    return fn(*args)

rows = []
for h in (6, 24):
    for tier in (1, 2):
        pred = pd.read_parquet(f"results/predictions/logreg_t{tier}_h{h}_lr01_pooled_test.parquet")
        pred["hour"] = pd.to_datetime(pred["hour"], utc=True)
        n_cm = rt._observed_coin_months(pred)
        target_ids = call(rt._targetable_episode_ids, pred, episodes, h)
        cands = np.unique(np.quantile(pred["y_score"], np.linspace(0.90, 0.9999, 60)))
        best = None
        for t in cands:
            _, ev = call(build_alarm_events, pred, episodes, float(t), horizon=h)
            fa = int((~ev["is_true_alarm"]).sum()) if len(ev) else 0
            if fa / n_cm <= BUDGET:
                caught = set(ev.loc[ev["is_true_alarm"], "target_episode_id"]) & set(target_ids)
                lead = ev.loc[ev["is_true_alarm"], "lead_hours"].mean() if len(ev) else np.nan
                best = dict(horizon=h, tier=tier, threshold=round(float(t), 4),
                            false_alarms=fa, coin_months=n_cm,
                            fa_per_coin_month=round(fa / n_cm, 3),
                            episodes_caught=len(caught), episodes=len(target_ids),
                            recall=round(len(caught) / len(target_ids), 3),
                            avg_lead_h=round(float(lead), 1) if pd.notna(lead) else None)
                break
        rows.append(best or dict(horizon=h, tier=tier, note="no threshold met budget"))

out = pd.DataFrame(rows)
print(out.to_string(index=False))
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/logreg_operating_point.csv", index=False)
