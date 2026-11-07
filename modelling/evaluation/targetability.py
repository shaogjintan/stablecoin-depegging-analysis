"""Episode-level targetability from eligible prediction origins.

Prediction eligibility concerns a single coin-hour: whether that row may be
scored under the data and risk-set rules. Targetability concerns an episode:
at least one eligible origin for the same coin must precede its onset by no
more than the forecast horizon. For onset s and horizon h, origins are in
[s - h, s), equivalently s is in (origin, origin + h]. All clocks are UTC
completed-bar timestamps.

For a noon onset and h6, a 06:00 origin qualifies; a noon origin does not.
Targetable episodes form the episode-recall denominator. Being targetable
does not mean the model raised an alarm or caught the episode.

Callers supply already-eligible origins. This helper neither chooses them
nor requires original y_true labels; that matters when robustness checks
change the outcome definition while keeping the original origins fixed.
Targetability is specific to the supplied sample, coin, horizon and outcome
definition. Keep targetable_episodes columns unchanged in saved reports.
"""
import numpy as np
import pandas as pd


def targetable_ids(pred, episodes, h):
    """Return IDs with an eligible same-coin origin in [onset - h, onset).

    Args:
        pred: Eligible prediction rows containing coin and UTC hour.
        episodes: Episodes containing coin, episode_id and UTC start.
        h: Forecast horizon in hours.

    Returns:
        Set of targetable episode IDs, represented as strings.
    """
    targets = set()
    for coin, group in pred.groupby("coin", sort=False):
        times = np.sort(pd.to_datetime(group.hour, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy())
        eps = episodes.loc[episodes.coin.eq(coin)]
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        positions = np.searchsorted(times, starts, side="left")-1
        valid = positions >= 0
        origins = times[np.maximum(positions, 0)]
        valid &= origins >= starts-h*3600_000_000_000
        targets.update(eps.loc[valid, "episode_id"].astype(str))
    return targets

