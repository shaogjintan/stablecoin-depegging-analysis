"""Alternative outcome definitions for fixed-alarm sensitivity."""
import pandas as pd
from data_processing.config import DEPEG_GAP_HOURS

def make_episodes(master: pd.DataFrame, band: float,
                  min_duration: int) -> pd.DataFrame:
    """Create episodes using a down-band and consecutive-hour rule.

    Qualifying dips separated by at most DEPEG_GAP_HOURS from the first
    above-band observation are clustered into one episode, matching the main
    data-processing episode definition.
    """
    if not 0 < band < 1:
        raise ValueError("depeg bands must be fractions between 0 and 1")
    if min_duration < 1:
        raise ValueError("persistence durations must be positive")

    rows = []
    for coin, group in master.sort_values(["coin", "hour"]).groupby("coin", sort=False):
        group = group.reset_index(drop=True)
        below = group["price"].lt(1 - band).fillna(False).to_numpy()
        runs: list[tuple[int, int]] = []
        i = 0
        while i < len(below):
            if not below[i]:
                i += 1
                continue
            start = i
            while i < len(below) and below[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_duration:
                runs.append((start, end))

        clusters: list[tuple[int, int]] = []
        for start, end in runs:
            if clusters and start - (clusters[-1][1] + 1) <= DEPEG_GAP_HOURS:
                clusters[-1] = (clusters[-1][0], end)
            else:
                clusters.append((start, end))

        for number, (start, end) in enumerate(clusters, start=1):
            rows.append({
                "coin": coin,
                "episode_id": f"{coin}_{number}",
                "start": group.loc[start, "hour"],
                "end": group.loc[end, "hour"] + pd.Timedelta(hours=1),
                "n_hours": end - start + 1,
                "min_price": group.loc[start:end, "price"].min(),
            })
    return pd.DataFrame(rows, columns=[
        "coin", "episode_id", "start", "end", "n_hours", "min_price"
    ])
