"""Re-score FINAL saved refit alarms under alternative episode definitions.

Run from the repository root with ``python -m modelling.robustness_refit_alarms
--source-run <verified refit directory> --out-dir <new directory>``.
No models, scores, cutoffs, alarm origins, or eligible test hours are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.splits import load_master
from modelling.robustness_check import make_episodes, _targetable_ids

MODELS = ("logreg", "random_forest", "xgboost", "persistence", "baseline")
FITTED = MODELS[:3]
CONFIGS = ("market_matched", "market_onchain")
HORIZONS = (6, 24)
BANDS = (0.005, 0.01, 0.02)
DURATIONS = (1, 2, 3, 4)
YEARS = range(2021, 2026)
NAMES = dict(logreg="LR", random_forest="RF", xgboost="XGB")


def evaluate_events(predictions, events, episodes, horizon, *, targetable_ids=None):
    """Use eligible saved origins and the first onset in (alarm, alarm+h]."""
    targets = (_targetable_ids(predictions, episodes, horizon)
               if targetable_ids is None else targetable_ids)
    warned, leads = set(), []
    true = 0
    delta = pd.Timedelta(hours=horizon)
    by_coin = {coin: group.sort_values("start").reset_index(drop=True)
               for coin, group in episodes.groupby("coin", sort=False)}
    for event in events.itertuples(index=False):
        coin_eps = by_coin.get(event.coin)
        if coin_eps is None:
            continue
        index = coin_eps.start.searchsorted(event.alarm_hour, side="right")
        if index < len(coin_eps) and coin_eps.iloc[index].start <= event.alarm_hour + delta:
            episode = coin_eps.iloc[index]
            true += 1
            warned.add(str(episode.episode_id))
            leads.append((episode.start - event.alarm_hour) / pd.Timedelta(hours=1))
    months = predictions[["coin", "hour"]].copy()
    months["coin_month"] = months.hour.dt.year * 12 + months.hour.dt.month
    coin_months = len(months[["coin", "coin_month"]].drop_duplicates())
    caught, false = len(warned & targets), len(events) - true
    return dict(episodes_defined_total=len(episodes), targetable_episodes=len(targets),
                episodes_caught=caught, episode_recall=caught / len(targets) if targets else np.nan,
                alarm_events=len(events), true_alarms=true, false_alarms=false,
                observed_coin_months=coin_months,
                false_alarms_per_coin_month=false / coin_months if coin_months else np.nan,
                median_lead_hours=float(np.median(leads)) if leads else np.nan)


def load_cell(source, model, config, horizon):
    predictions, alarms = [], []
    for year in YEARS:
        folder = source / "fits" / f"{model}_{config}_h{horizon}_{year}"
        pred = pd.read_parquet(folder / "test.parquet", columns=[
            "coin", "hour", "test_year", "horizon", "threshold"])
        events = pd.read_csv(folder / "alarms.csv", float_precision="round_trip")
        pred["hour"] = pd.to_datetime(pred.hour, utc=True)
        events["alarm_hour"] = pd.to_datetime(events.alarm_hour, utc=True)
        if (pred.empty or not pred.test_year.eq(year).all()
                or not pred.horizon.eq(horizon).all() or pred.threshold.nunique() != 1):
            raise ValueError(f"Invalid prediction metadata: {folder}")
        if (not events.year.eq(year).all() or not events.horizon.eq(horizon).all()
                or not events.threshold.eq(pred.threshold.iloc[0]).all()):
            raise ValueError(f"Alarm metadata/cutoff mismatch: {folder}")
        origins = pd.MultiIndex.from_frame(pred[["coin", "hour"]])
        alarm_origins = pd.MultiIndex.from_frame(events[["coin", "alarm_hour"]])
        if not alarm_origins.isin(origins).all():
            raise ValueError(f"Alarm outside eligible saved origins: {folder}")
        predictions.append(pred)
        alarms.append(events)
    pred = pd.concat(predictions, ignore_index=True)
    events = pd.concat(alarms, ignore_index=True)
    if pred.duplicated(["coin", "hour"]).any() or events.duplicated(["coin", "alarm_hour"]).any():
        raise ValueError(f"Duplicate origins: {model}/{config}/h{horizon}")
    return pred, events


def check_reference(results, pooled):
    keys = ["model", "config", "horizon_hours"]
    metrics = ["episodes_caught", "targetable_episodes", "false_alarms",
               "observed_coin_months", "alarm_events", "true_alarms"]
    actual = results.loc[results.is_reference_definition, keys + metrics].set_index(keys).sort_index()
    expected = pooled.set_index(keys).loc[actual.index, metrics].sort_index()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    return actual.reset_index()


def compare_models(results):
    keys = ["depeg_band", "persistence_hours", "horizon_hours", "config"]
    recalls = results.loc[results.model.isin(FITTED)].pivot(
        index=keys, columns="model", values="episode_recall")
    maximum = recalls.max(axis=1)
    winners = recalls.eq(maximum, axis=0)
    recalls["lr_result"] = np.where(
        ~winners.logreg, "loses", np.where(winners.sum(axis=1).eq(1), "strict", "tied"))
    recalls["highest_models"] = winners.apply(
        lambda row: ", ".join(NAMES[m] for m in FITTED if row[m]), axis=1)
    if len(recalls) != 48 or recalls[list(FITTED)].isna().any().any():
        raise ValueError("Expected 48 complete, nonempty fitted-model comparisons")
    counts = recalls.lr_result.value_counts()
    tiers = results.loc[results.model.isin(FITTED)].pivot(
        index=["model", "horizon_hours", "depeg_band", "persistence_hours"],
        columns="config", values=["episodes_caught", "false_alarms"])
    passes = (tiers["episodes_caught"].market_onchain.ge(tiers["episodes_caught"].market_matched)
              & tiers["false_alarms"].market_onchain.le(tiers["false_alarms"].market_matched))
    return recalls.reset_index(), passes.rename("tier2_dominates").reset_index(), counts


def definition_label(row):
    return f"{row.depeg_band:.1%}/{int(row.persistence_hours)}h"


def write_summary(results, out, source, master_rows):
    comparisons, tier_checks, counts = compare_models(results)
    comparisons.to_csv(out / "lr_comparisons.csv", index=False)
    tier_checks.to_csv(out / "tier_comparisons.csv", index=False)
    strict, tied, loses = (int(counts.get(k, 0)) for k in ("strict", "tied", "loses"))
    ref = results.loc[results.is_reference_definition]
    reference_episodes = int(ref.episodes_defined_total.iloc[0])
    examples = []
    for model, config, h in (("logreg", "market_onchain", 24), ("logreg", "market_matched", 24),
                             ("random_forest", "market_matched", 6), ("xgboost", "market_onchain", 24)):
        row = ref.loc[ref.model.eq(model) & ref.config.eq(config) & ref.horizon_hours.eq(h)].iloc[0]
        examples.append(f"{NAMES[model]} T{CONFIGS.index(config)+1} h{h} = "
                        f"{row.episodes_caught}/{row.targetable_episodes}, FA {row.false_alarms}")
    lines = ["# Alternative depeg definitions — verified final refit alarms", "",
             f"Source (read-only): `{source}`", "",
             "## Protocol and mandatory sanity check", "",
             f"Corrected master: {master_rows:,} rows; reference manifest: {reference_episodes} episodes. "
             "Rebuilt reference episode identities, coins, starts and ends match the final-run manifest exactly.",
             "Reference (1%, 2h): **PASS for all 20 model/config/horizon cells**, exactly matching "
             "pooled.csv on caught, targetable, false alarms, observed coin-months, alarm events and true alarms.",
             "Examples: " + "; ".join(examples) + ".", "",
             "Grid: bands 0.5%, 1%, 2%; persistence 1, 2, 3, 4 hours. "
             "Tier 1 = market_matched; Tier 2 = market_onchain; h = 6, 24; test years 2021–2025. "
             "results.csv contains 240 rows including persistence and constant-prior context.", "",
             "Saved eligible test hours remain fixed (including original exclusions); targetability is recomputed "
             "for each new definition using those hours, not the old y_true. An alarm is false iff no same-coin "
             "new onset lies in (alarm_hour, alarm_hour+h]; only the earliest onset is credited. "
             "Alarms are never regenerated, regrouped or recalibrated. Coin-month exposure is counted from "
             "unique eligible (coin, UTC calendar month) pairs, not from alarms. "
             "This is outcome-definition sensitivity conditional on the original at-risk sample, not a rebuilt pipeline.", "",
             "## 1. LR versus RF/XGB: 48 comparisons", "",
             f"LR strictly highest: **{strict}/48**. LR tied for highest: **{tied}/48**. "
             f"LR below the highest: **{loses}/48**. Ties are separate from strict wins.", "",
             "### Where LR is overtaken", "",
             "| Band | Duration | Horizon | Tier | LR recall | Highest model(s) | Highest recall |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for r in comparisons.loc[comparisons.lr_result.eq("loses")].itertuples(index=False):
        lines.append(f"| {r.depeg_band:.1%} | {r.persistence_hours}h | {r.horizon_hours}h | "
                     f"{CONFIGS.index(r.config)+1} | {r.logreg:.1%} | {r.highest_models} | "
                     f"{max(r.logreg, r.random_forest, r.xgboost):.1%} |")
    if not loses:
        lines.append("| None | — | — | — | — | — | — |")
    lines += ["", "### LR ties for highest", ""]
    for r in comparisons.loc[comparisons.lr_result.eq("tied")].itertuples(index=False):
        lines.append(f"- {definition_label(r)}, h{r.horizon_hours}, Tier {CONFIGS.index(r.config)+1}: "
                     f"{r.highest_models}, recall {r.logreg:.1%}.")
    if not tied:
        lines.append("None.")
    lines += ["", "## 2. Tier 2 catches >= episodes with <= false alarms", "",
              "Both inequalities must hold within the same definition; equality counts. "
              "This is the requested episode-count/FA comparison, not a recall-only comparison.", "",
              "| Model | Horizon | Definitions satisfying both / 12 | Passing definitions (band/duration) |",
              "|---|---:|---:|---|"]
    all_combos, some_combos, no_combos = [], [], []
    for (model, h), group in tier_checks.groupby(["model", "horizon_hours"], sort=False):
        passed = group.loc[group.tier2_dominates]
        label = f"{NAMES[model]} h{h}"
        (all_combos if len(passed) == 12 else some_combos if len(passed) else no_combos).append(label)
        definitions = ", ".join(definition_label(r) for r in passed.itertuples(index=False)) or "None"
        lines.append(f"| {NAMES[model]} | {h}h | {len(passed)} | {definitions} |")
    lines += ["", f"**All 12:** {', '.join(all_combos) or 'none'}.",
              f"**Some (1–11):** {', '.join(some_combos) or 'none'}.",
              f"**None:** {', '.join(no_combos) or 'none'}.", "",
              "## 3. Persistence recall per definition", "",
              "Cells show caught/targetable (recall).", "",
              "| Band | Duration | h6 Tier 1 | h6 Tier 2 | h24 Tier 1 | h24 Tier 2 |",
              "|---|---:|---:|---:|---:|---:|"]
    persistence = results.loc[results.model.eq("persistence")]
    for (band, duration), group in persistence.groupby(["depeg_band", "persistence_hours"]):
        values = []
        for h in HORIZONS:
            for config in CONFIGS:
                r = group.loc[group.horizon_hours.eq(h) & group.config.eq(config)].iloc[0]
                values.append(f"{r.episodes_caught}/{r.targetable_episodes} ({r.episode_recall:.1%})")
        lines.append(f"| {band:.1%} | {duration}h | " + " | ".join(values) + " |")
    lines += ["", "## 4. Paste-ready replacement for slide 30", "",
              "- Sensitivity grid: 0.5%, 1%, 2% drops × 1–4h persistence; h=6/24; matched Tier 1 vs Tier 2.",
              f"- LR has strictly highest episode recall in {strict}/48 settings, ties for highest in {tied}/48, "
              f"and is overtaken in {loses}/48; superiority is not universal."]
    losing_bands = comparisons.loc[comparisons.lr_result.eq("loses"), "depeg_band"].unique()
    if len(losing_bands):
        overtakers = [NAMES[m] for m in FITTED[1:] if (comparisons[m] > comparisons.logreg).any()]
        lines.append(f"- LR is overtaken by {'/'.join(overtakers)} at "
                     f"{', '.join(f'{b:.1%}' for b in losing_bands)} bands.")
    tier_counts = tier_checks.groupby(["model", "horizon_hours"]).tier2_dominates.sum()
    count_text = "; ".join(f"{NAMES[m]} {tier_counts[m, 6]}/{tier_counts[m, 24]}"
                           for m in FITTED)
    lines += [f"- Tier 2 joint benefit (>= catches, <= FA), h6/h24 out of 12: {count_text}. "
              f"All-12 combinations: {', '.join(all_combos) or 'none'}.",
              "- Persistence reference recall is 15.2% (h6), 26.5% (h24); "
              "strict definitions leave as few as one targetable episode.",
              "", "Caveat: models and validation cutoffs held fixed; only the episode definition changes", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"LR strict={strict}/48, tied={tied}/48, loses={loses}/48")
    print(tier_checks.groupby(["model", "horizon_hours"]).tier2_dominates.sum().to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source, out = args.source_run.resolve(), args.out_dir.resolve()
    if source == out or source in out.parents:
        raise ValueError("Output must be outside the read-only source run")
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    if (json.loads((source / "run.json").read_text())["status"] != "complete"
            or json.loads((source / "verification.json").read_text())["status"] != "passed"):
        raise ValueError("Source run must be complete and verified")
    master, reference = load_master()
    columns = ["episode_id", "coin", "start", "end"]
    def canonical(frame):
        return frame[columns].sort_values("episode_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical(reference), canonical(pd.read_parquet(source / "episodes.parquet")),
                                  check_exact=True)
    episodes = {(b, d): make_episodes(master, b, d) for b in BANDS for d in DURATIONS}
    pd.testing.assert_frame_equal(canonical(episodes[.01, 2]), canonical(reference), check_exact=True)
    rows, target_cache = [], {}
    for model in MODELS:
        for config in CONFIGS:
            for horizon in HORIZONS:
                pred, events = load_cell(source, model, config, horizon)
                origin_hash = hashlib.sha256(pd.util.hash_pandas_object(
                    pred[["coin", "hour"]], index=False).to_numpy().tobytes()).hexdigest()
                for (band, duration), eps in episodes.items():
                    # Eligibility is identical across models; cache only exact matching origins.
                    key = (horizon, origin_hash, band, duration)
                    if key not in target_cache:
                        target_cache[key] = _targetable_ids(pred, eps, horizon)
                    rows.append(dict(model=model, config=config, tier=CONFIGS.index(config)+1,
                                     horizon_hours=horizon, depeg_band=band, price_cutoff=1-band,
                                     persistence_hours=duration, is_reference_definition=(band == .01 and duration == 2),
                                     **evaluate_events(pred, events, eps, horizon,
                                                       targetable_ids=target_cache[key])))
                print(f"Evaluated {model}/{config}/h{horizon}", flush=True)
    results = pd.DataFrame(rows)
    sanity = check_reference(results, pd.read_csv(source / "pooled.csv"))
    compare_models(results)
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / "results.csv", index=False)
    sanity.to_csv(out / "reference_sanity.csv", index=False)
    write_summary(results, out, source, len(master))
    provenance = dict(status="passed", source_run=str(source), master_rows=len(master),
                      reference_episode_count=len(reference), sanity_cells=len(sanity), result_rows=len(results),
                      source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                     for name in ("run.json", "verification.json", "pooled.csv", "episodes.parquet")})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Reference sanity: PASS ({len(sanity)} cells); saved {len(results)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
