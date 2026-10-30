# Summary: every change made because of Prof Huang's feedback (30 Oct 2026)

> All items verified and committed on branch `cleanup/dead-weight-final-deck`
> (pushed to origin; NOT yet merged into upstream/main).
> Full deck copy: [`plans/final_deck_changes.md`](final_deck_changes.md).

## A. Prof-driven changes (the one clean table)

| # | Prof's comment | Change we made | Artifact |
|---|---|---|---|
| 1 | Main table: PR-AUC **with the positive base rate next to it** | Added base rate for every fixed-config run | `reports/modelling_comparison.csv` (re-run of `compare_runs.py`) — h6 0.13%/0.14%, h24 0.43%/0.46% |
| 2 | Main table: **realised FAs per coin-month in test years, next to the budget** | Computed realised FA/cm vs the 1.0 budget per model × tier × horizon × year, both modes | `reports/ex_post_comparison.csv` + `_per_year.csv` |
| 3 | Main table: **share of OOS episodes with ≥1 alarm + median lead time** | Episode recall (n/34) kept as headline; **median lead time computed** per model/tier/horizon/mode | `reports/ex_post_comparison.csv` (XGB T2 h24 = 9.0h, IQR 4–15) |
| 4 | Keep hourly P/R in the report; drop "hours during the depeg" | Confirmed nothing reports in-depeg hours; hourly P/R stays report-side | — (no slide/metric added) |
| 5 | **Keep the current method; report the threshold-transfer gap as a limitation** | Real-time protocol untouched; budget→realised gap surfaced as a result (2021 XGB T2 = 3.11, 2022 = 0.30) | `reports/ex_post_comparison_per_year.csv` → limitations slide |
| 6 | **Add the ex-post comparison** (test-year thresholds at the same budget; compare catches + lead; label clearly; show both) | New local driver: same `_choose_threshold` applied to the test year's own scores for all 5 models | `plans/ex_post_comparison.py` → `reports/ex_post_comparison.csv` + `ex_post_vs_real_time_h24.png` |
| 7 | Use the **average** budget (≈72/yr), not a per-coin-month cap | Verified code already does the average; deck wording fixed to "on average" | — (no code change; wording) |
| 8 | Event types = context; **add type as a column; state counts, not findings** | Implemented the 5-type classification in code + caught/missed by type as counts only; fixed the wrong "staggered = 10" → **16** (types now sum to 79) | `plans/event_types.py` → `reports/event_tables/types_events.csv`, `types_summary.csv`, `caught_by_type.csv` |
| 9 | Flag **how many independent events** the results rest on | Stated everywhere: 30 events / 34 targetable at h24 (33 h6, 32 h1) | slide copy |
| 10 | Flag **which chains the on-chain data covers** | Confirmed: Dune CSVs have no chain column; verified as **Ethereum-only (ERC-20)** via contract start dates & tx volumes → updated in `README.md` and slide copy | `README.md`, `plans/final_deck_changes.md` |
| 11 | Robustness = "does the conclusion change at 0.5%?" | Computed: at 0.5 FA/cm the leader **flips** (RF T2 6/34 vs XGB T2 3/34; T1 ≥ T2) → budget-dependence stated honestly | `reports/local_budget_curve.csv` (budget-0.5 rows) → P23 copy |
| 12 | **h = 1 one line, labelled near-detection** | Retrained XGB h1 with the fixed config (models were still grid-tuned) | `results/models/xgboost_*_h1_*.json` (now `xgb_d3_lr005_300`) → h1 line: T1 2/32, T2 3/32 |
| 13 | "If the gap is small/inconsistent, say on-chain adds little" | Ex-post numbers now support the honest line: T2−T1 ≈ 1 episode at h24, concentrated in staggered/sustained (SVB-type) events | `reports/ex_post_comparison.csv` + `caught_by_type.csv` |
| 14 | Robustness slide: current-protocol numbers (old OR-union was recall-tuned) | Re-ran OR-combination with fixed tags: union 0.647 vs 0.500 @1% hourly budget, gain 0 @2% | `reports/combination_results.csv` |
| 15 | Regenerate the stale coverage table | New coverage from `master.parquet` with stated definition: all coins ≥91% | `plans/coverage_table.py` → `reports/coverage_table.csv` |
| 16 | Threshold-search details → report, not slides | Real-time vs ex-post protocol documented in code docstring + slide copy keeps only the one-sentence rule | `modelling/rolling_threshold.py` (now a helper library) |

## B. Cleanup changes (your directive, not prof-driven)

| Change | Detail |
|---|---|
| `rolling_threshold.py` CLI removed | Now a pure helper library; analysis drivers live in `plans/` (no flags) |
| Dead scripts archived | `alarm_smoothing.py`, `alarm_cascade.py`, `rolling_threshold_comparison.py`, `partition_report.py` → `archive/legacy_modelling/` |
| Stale figures archived | recall-tuned SHAP outputs → `archive/legacy_figures/` |
| Local data cleanup | ~570 MB superseded intermediates/logs/stale pooled files/pycache deleted |
| Branches | 11 merged local branches deleted |
| Doc fixes | agent.md / shap docstring paths corrected; README chain-scope note |

## C. ⚠️ Open items for the group

1. **Javier pushed 3 new commits to `upstream/logreg-model` today** (`reports/logreg_operating_point.csv`, `logreg_matched_tier_eval.csv`, `annual_calibration.csv`). His numbers use a different protocol (pooled threshold; matched sample; grid-tuned PR-AUC 0.0224/0.0220/0.0325/0.0307; recall 9/34 & 10/34 at h24). **These conflict with the fixed-config protocol the deck uses** (0.0200/0.0197/0.0312/0.0300; 13/34 & 11/34). Reconcile with Javier before pasting P13/P18.
2. Merge PRs: `results/fixed-config-local` → main, then `cleanup/dead-weight-final-deck` (5 commits, 0 behind upstream/main).
3. Verified Dune query's chain scope: **Ethereum mainnet only (ERC-20)** → updated in P4 slide copy.
4. Canva edits from `plans/final_deck_changes.md` (per-slide copy, reorder, delete P26 junk).
5. Confirm with Vienna before archiving `modelling/robustness_check.py` (kept in place).
