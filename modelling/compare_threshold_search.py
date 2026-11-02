"""Compare threshold policies on frozen corrected XGB predictions, without refitting.

python -m modelling.compare_threshold_search --run-dir reports/xgb_corrected_shap_v1
Binary emulates the legacy lowest-feasible search. An exhaustive-lowest control
separates search errors from the new maximum-recall selection objective.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from modelling.alarm_events import build_alarm_events
from modelling.exact_threshold import exhaustive_counts
from modelling.rolling_threshold import _targetable_episode_ids, _false_alarm_stats


def counts(frame, episodes, horizon):
    frame = frame.sort_values(["coin", "hour"]).reset_index(drop=True)
    times = pd.to_datetime(frame.hour, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    targets = _targetable_episode_ids(frame, episodes, horizon)
    codes = {ep: i for i, ep in enumerate(sorted(targets))}
    target = np.full(len(frame), -1, dtype=np.int64)
    delta = horizon * 3600_000_000_000
    for coin, group in frame.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        positions = group.index.to_numpy()
        loc = np.searchsorted(starts, times[positions], side="right")
        for pos, j in zip(positions, loc):
            if j < len(starts) and starts[j] <= times[pos] + delta:
                target[pos] = codes.get(str(eps.iloc[j].episode_id), -1)
    ends = frame.groupby("coin", sort=False).size().cumsum().to_numpy(dtype=np.int64)
    cut, caught, false, alarms = exhaustive_counts(
        frame.y_score.to_numpy(dtype=float), times, ends, target, delta)
    order = np.argsort(cut)
    return (np.r_[cut[order], np.inf], np.r_[caught[order], 0],
            np.r_[false[order], 0], np.r_[alarms[order], 0], len(targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/xgb_threshold_search"))
    parser.add_argument("--budget", type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.budget) or args.budget < 0:
        parser.error("budget must be finite and nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.run_dir / "run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("Source run is incomplete")
    episodes = pd.read_parquet(args.run_dir / "episodes.parquet")
    rows, event_frames = [], []
    for folder in sorted((args.run_dir / "fits").iterdir()):
        info = json.loads((folder / "fit.json").read_text())
        h = int(info["horizon"])
        val = pd.read_parquet(folder / "inner_val.parquet")
        test = pd.read_parquet(folder / "test.parquet")
        cut, caught, false, alarms, n_targets = counts(val, episodes, h)
        months = val.assign(month=pd.to_datetime(val.hour, utc=True).dt.strftime("%Y-%m"))[
            ["coin", "month"]].drop_duplicates().shape[0]
        feasible = false <= args.budget * months
        lo, hi = 0, len(cut) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible[mid]:
                hi = mid
            else:
                lo = mid + 1
        indices = {
            "binary_legacy": lo,
            "exhaustive_lowest": int(np.flatnonzero(feasible)[0]),
            "exact_max_recall": max(np.flatnonzero(feasible),
                                    key=lambda i: (caught[i], -false[i], cut[i])),
        }
        target_ids = _targetable_episode_ids(test, episodes, h)
        for method, idx in indices.items():
            threshold = float(cut[idx])
            # Independently verify accelerated counts at every selected cutoff.
            _, ve = build_alarm_events(val, episodes, threshold)
            vf, _, _, _ = _false_alarm_stats(ve, val)
            vc = set(ve.target_episode_id.dropna().astype(str)) & _targetable_episode_ids(val, episodes, h)
            assert vf == false[idx] and len(vc) == caught[idx]
            _, events = build_alarm_events(test, episodes, threshold)
            nf, nm, rate, _ = _false_alarm_stats(events, test)
            nc = len(set(events.target_episode_id.dropna().astype(str)) & target_ids)
            rows.append(dict(config=info["config"], horizon_hours=h, test_year=info["year"],
                             method=method, threshold=threshold,
                             validation_false_alarms=int(false[idx]),
                             validation_episodes_caught=int(caught[idx]),
                             validation_targetable=n_targets, validation_coin_months=months,
                             false_alarms=nf, coin_months=nm,
                             false_alarms_per_coin_month=rate,
                             episodes_caught=nc, targetable_episodes=len(target_ids),
                             alarm_events=len(events),
                             nonmonotone_false_count=bool((np.diff(false) > 0).any())))
            event_frames.append(events.assign(method=method, config=info["config"],
                                               test_year=info["year"]))
        print(folder.name + ": compared all three policies", flush=True)
        pd.DataFrame(rows).to_csv(args.out_dir / "by_year.csv", index=False)

    years = pd.DataFrame(rows)
    summary = years.groupby(["config", "horizon_hours", "method"], as_index=False).agg(
        false_alarms=("false_alarms", "sum"), coin_months=("coin_months", "sum"),
        episodes_caught=("episodes_caught", "sum"),
        targetable_episodes=("targetable_episodes", "sum"),
        alarm_events=("alarm_events", "sum"))
    summary["false_alarms_per_coin_month"] = summary.false_alarms / summary.coin_months
    summary["episode_recall"] = summary.episodes_caught / summary.targetable_episodes
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(args.out_dir / "alarm_events.csv", index=False)
    (args.out_dir / "protocol.json").write_text(json.dumps({
        "source_run": str(args.run_dir), "run_id": run["run_id"],
        "budget": args.budget, "retrained": False,
        "policies": list(indices), "calibration": "saved inner_val only",
        "test": "same saved test predictions and episode manifest for every policy",
        "implementation": "exhaustive cooldown counts; simulate original binary decisions on those counts; independently verify selected cutoffs with build_alarm_events",
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
