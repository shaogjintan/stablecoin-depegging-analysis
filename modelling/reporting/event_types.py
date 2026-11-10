"""Structural event types, separate from historical context annotations."""
import pandas as pd

def classify(episodes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per event with its type."""
    rows = []
    for _, ev in events.iterrows():
        mem = episodes[episodes["event_id"] == ev["event_id"]]
        n_coins = mem["coin"].nunique()
        spread = (mem["start"].max() - mem["start"].min()) / pd.Timedelta(hours=1)
        if n_coins >= 2:
            typ = ("market-wide, simultaneous" if spread <= 1.0
                   else "market-wide, staggered")
        else:
            r = mem.iloc[0]
            if r["n_hours"] > 24:
                typ = "sustained single-coin"
            elif r["n_hours"] > 3 or r["min_price"] < 0.98:
                typ = "short single-coin"
            else:
                typ = "brief wobble"
        rows.append({"event_id": ev["event_id"], "type": typ,
                     "n_coins": int(n_coins), "onset_spread_hours": float(spread),
                     "n_episodes": int(len(mem))})
    return pd.DataFrame(rows)
