"""Outcome-definition sensitivity for completed matched refit runs.

Freeze prediction hours, models, cutoffs and alarm times; vary episode definitions.
This is not a retraining experiment under alternative labels or eligibility.
"""
import argparse
import json
from pathlib import Path
from modelling.utils.artifacts import evidence_dir

import numpy as np
import pandas as pd

from data_processing.splits import load_master, active_input_paths, canonical_input_hashes
from modelling.utils.io import sha256, save_json
from modelling.robustness_checks.episode_definitions import make_episodes
from modelling.evaluation.targetability import targetable_ids




def evaluate(pred, alarms, episodes, h):
    targets = targetable_ids(pred, episodes, h)
    caught, leads, true_count = set(), [], 0
    for coin, group in alarms.groupby("coin", sort=False):
        eps = episodes.loc[episodes.coin.eq(coin)].sort_values("start")
        starts = pd.to_datetime(eps.start, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        for hour in pd.to_datetime(group.alarm_hour, utc=True):
            j = int(np.searchsorted(starts, hour.value, side="right"))
            if j < len(starts) and starts[j] <= hour.value + h*3600_000_000_000:
                true_count += 1
                caught.add(str(eps.iloc[j].episode_id))
                leads.append((starts[j]-hour.value)/3600_000_000_000)
    hours = pd.to_datetime(pred.hour, utc=True)
    months = len(pd.DataFrame(dict(coin=pred.coin.to_numpy(),
                                   year=hours.dt.year.to_numpy(),
                                   month=hours.dt.month.to_numpy())).drop_duplicates())
    false = len(alarms)-true_count
    years = set(pd.to_datetime(pred.hour, utc=True).dt.year)
    return dict(episodes_defined_total=len(episodes),
                episodes_defined_test_years=int(episodes.start.dt.year.isin(years).sum()),
                targetable_episodes=len(targets), episodes_caught=len(caught & targets),
                episode_recall=len(caught & targets)/len(targets) if targets else np.nan,
                false_alarms=false, coin_months=months, false_alarms_per_coin_month=false/months,
                alarm_events=len(alarms), true_alarms=true_count,
                alarm_precision=true_count/len(alarms) if len(alarms) else np.nan,
                median_lead_hours=float(np.median(leads)) if leads else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/robustness/alt_definitions"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 24], choices=[1, 6, 24])
    parser.add_argument("--bands", type=float, nargs="+", default=[.005, .01, .02])
    parser.add_argument("--durations", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    if .01 not in args.bands or 2 not in args.durations:
        raise ValueError("Include the reference definition: band .01, duration 2")
    master, reference = load_master()
    hashes = {str(p): sha256(p) for p in active_input_paths()}
    prices = []
    for coin, g in master.groupby("coin", sort=False):
        g = g.set_index("hour").sort_index()[["price"]]
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        g.index.name = "hour"
        g["coin"] = coin
        prices.append(g.reset_index())
    prices = pd.concat(prices, ignore_index=True)
    definitions = {(band, duration): make_episodes(prices, band, duration)
                   for band in args.bands for duration in args.durations}
    print(f"Built {len(definitions)} episode definitions", flush=True)
    columns = ["coin", "start", "end"]
    actual = definitions[(.01, 2)][columns].sort_values(columns).reset_index(drop=True)
    expected = reference[columns].sort_values(columns).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Reference episode reconstruction does not match corrected clocks/definitions")
    rows, seen, manifests = [], set(), []
    common_keys = {}
    for run in args.run_dir:
        info = json.loads((run / "run.json").read_text())
        if (info.get("status") != "complete" or not info.get("refit_after_calibration")
                or info.get("policy") != "exhaustive_lowest" or canonical_input_hashes(info.get("input_sha256", {})) != hashes):
            raise ValueError(f"Incompatible or incomplete refit run: {run}")
        manifests.append(dict(path=str(run), run_id=info["run_id"]))
        summary = pd.read_csv(run / "summary.csv")
        events = pd.read_csv(run / "alarm_events.csv", parse_dates=["alarm_hour"])
        for r in summary.loc[summary.horizon_hours.isin(args.horizons)].itertuples():
            if r.config not in ("market_matched", "market_onchain"):
                continue
            key = (r.model, r.config, int(r.horizon_hours))
            if key in seen:
                raise ValueError(f"Duplicate model/config/horizon: {key}")
            seen.add(key)
            h = int(r.horizon_hours)
            pred = pd.concat([pd.read_parquet(evidence_dir(run) / "fits" /
                f"{r.model}_{r.config}_h{h}_{year}" / "test.parquet")
                for year in range(2021, 2026)], ignore_index=True)
            keys = set(zip(pred.coin, pred.hour))
            if h in common_keys and keys != common_keys[h]:
                raise ValueError(f"Models/tiers do not share test rows at h{h}")
            common_keys[h] = keys
            alarms = events.loc[events.model.eq(r.model) & events.config.eq(r.config) &
                                events.horizon_hours.eq(h)]
            for (band, duration), eps in definitions.items():
                result = evaluate(pred, alarms, eps, h)
                is_ref = band == .01 and duration == 2
                if is_ref:
                    for metric in ["episodes_caught", "targetable_episodes", "false_alarms", "alarm_events"]:
                        if result[metric] != getattr(r, metric):
                            raise ValueError(f"Reference mismatch: {key} {metric}")
                rows.append(dict(model=r.model, config=r.config, horizon_hours=h,
                                 band=band, duration_hours=duration, reference=is_ref, **result))
            print(f"Completed {r.model} {r.config} h{h}", flush=True)
    metrics = pd.DataFrame(rows)
    groups = ["horizon_hours", "band", "duration_hours"]
    leaders, effects = [], []
    for group, d in metrics.groupby(groups):
        settings = dict(zip(groups, group))
        for config, tier in d.groupby("config"):
            best = tier.loc[tier.episode_recall.eq(tier.episode_recall.max())]
            leader = "|".join(sorted(best.model)) if len(best) else "undefined_no_targetable_episodes"
            leaders.append(dict(**settings, config=config, recall_leaders=leader))
        for model, pair in d.groupby("model"):
            pair = pair.set_index("config")
            a, b = pair.loc["market_matched"], pair.loc["market_onchain"]
            if not np.isfinite(a.episode_recall) or not np.isfinite(b.episode_recall):
                effect = "undefined_no_targetable_episodes"
            elif b.episodes_caught == a.episodes_caught and b.false_alarms == a.false_alarms:
                effect = "equal"
            elif b.episodes_caught >= a.episodes_caught and b.false_alarms <= a.false_alarms:
                effect = "tier2_dominates"
            elif b.episodes_caught <= a.episodes_caught and b.false_alarms >= a.false_alarms:
                effect = "tier1_dominates"
            else:
                effect = "tradeoff"
            effects.append(dict(**settings, model=model, tier_effect=effect,
                                delta_episode_recall=b.episode_recall-a.episode_recall,
                                delta_false_alarms=int(b.false_alarms-a.false_alarms)))
    leaders, effects = pd.DataFrame(leaders), pd.DataFrame(effects)
    for data, identity, field in [(leaders, ["horizon_hours", "config"], "recall_leaders"),
                                  (effects, ["horizon_hours", "model"], "tier_effect")]:
        ref = data.loc[data.band.eq(.01) & data.duration_hours.eq(2), identity+[field]].rename(
            columns={field: "reference_"+field})
        merged = data.merge(ref, on=identity, how="left")
        merged["changed_from_reference"] = merged[field].ne(merged["reference_"+field])
        if field == "recall_leaders":
            leaders = merged
        else:
            effects = merged
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    leaders.to_csv(args.out_dir / "recall_leaders.csv", index=False)
    effects.to_csv(args.out_dir / "onchain_effects.csv", index=False)
    save_json(args.out_dir / "run.json", dict(source_runs=manifests, input_sha256=hashes,
        targetability="any frozen prediction origin in [episode start - h, episode start); no reference-label filter",
        check="fixed alarm outcome-definition sensitivity", reference_band=.01, reference_duration=2,
        bands=args.bands, durations=args.durations, horizons=args.horizons,
        frozen="prediction hours, models, thresholds, alarm timestamps; standard-definition risk set",
        ranking="highest episode recall; ties retained; not an overall performance ranking"))
    (args.out_dir / "README.md").write_text(
        "# Interpreting this robustness check\n\n"
        "Reference: downward 1% for 2h. Alternatives: the specified bands and durations.\n\n"
        "- metrics.csv: defined and targetable episodes, recall, lead, false alarms and alarm precision.\n"
        "- recall_leaders.csv: highest-recall model(s) by tier/horizon; changed_from_reference marks changes. "
        "A recall leader can raise more false alarms, so inspect metrics.csv alongside it.\n"
        "- onchain_effects.csv: tier2_dominates = at least as many episodes caught and no more false alarms, "
        "with one strict improvement; tier1_dominates is the reverse; tradeoff means neither dominates. "
        "Compare each alternative with the reference using changed_from_reference.\n\n"
        "Stable results across definitions support the conclusion within this sample. Changes indicate "
        "definition sensitivity; do not count them as independent statistical replications.\n\n"
        "Models, prediction eligibility, thresholds and alarm times are frozen. This evaluates how the "
        "existing standard-definition system performs against alternative outcomes. It does not establish "
        "performance for systems retrained and recalibrated for those definitions. AP is not recomputed "
        "against unchanged labels. Median lead is conditional on caught episodes.\n")
    print(f"Saved {args.out_dir}; compare changed_from_reference in both conclusion tables.")


if __name__ == "__main__":
    main()
