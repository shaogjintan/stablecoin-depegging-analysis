# FINAL DECK CHANGES — copy-paste content with verified numbers (30 Oct 2026)

> All numbers verified 2026-10-30 against committed outputs:
> `reports/ex_post_comparison.csv`, `reports/modelling_comparison.csv` (refreshed),
> `reports/local_budget_curve.csv`, `reports/coverage_table.csv`,
> `reports/event_tables/types_summary.csv`, `reports/event_tables/caught_by_type.csv`,
> `reports/combination_results.csv` (fixed tags). Deck reviewed: "DSE4101 Final
> Presentation Slides (3).pdf", 26 pages.
> Fixed-config protocol everywhere; budget = **1 FA per coin-month on average (~72/yr)**.

---

## 1. Per-page changes (old page numbers)

### P4 — Data sources: add the chain flag
Under "On-chain Dune Analytics", add:
> **Chain scope:** Dune exports cover the chains in the dashboard query (no chain
> column in the raw CSVs). Confirm the chain list and state it here — e.g.
> "Ethereum only" or "multi-chain". [GROUP: confirm from the Dune query]

### P5 — Coverage table: REPLACE (stale — old numbers match no file in the repo)
Definition footnote: **Coverage % = hours in the coin's observed span where all
Tier-2 fields (exchange volume + 4 on-chain fields) are present (master.parquet).**

| Coin | Start | End | Expected hrs | Observed hrs | Coverage % |
|---|---|---|---|---|---|
| USDT | 2018-09-01 | 2025-12-31 | 64,295 | 62,027 | 96.5 |
| USDC | 2018-10-05 | 2025-12-31 | 63,479 | 62,071 | 97.8 |
| BUSD | 2019-09-21 | 2024-02-29 | 38,951 | 36,359 | 93.3 |
| DAI  | 2019-11-18 | 2025-12-31 | 53,646 | 53,595 | 99.9 |
| TUSD | 2019-01-04 | 2025-12-31 | 61,295 | 56,991 | 93.0 |
| PAX  | 2018-09-26 | 2025-12-31 | 63,695 | 58,033 | 91.1 |

### P6 + P7 → merge onto ONE slide (definition + 48h event rule)
Text unchanged; drop to one slide.

### P8 — Types: FIX the staggered row (types now sum to 79 episodes)
| Type | Events | Episodes |
|---|---|---|
| Market-wide, simultaneous (≥2 coins, onsets ≤1h) | 7 | 21 |
| Market-wide, staggered (≥2 coins, onsets >1h) | 7 | **16**  ← was 10 (wrong) |
| Sustained single-coin (>24h below $0.99) | 9 | 9 |
| Short single-coin (≤24h, >3h or low <$0.98) | 12 | 12 |
| Brief wobble (≤3h, low ≥$0.98) | 21 | 21 |
| **Total** | **56** | **79** |

Keep: "Only 14 of 56 events are market-wide; 21 are brief wobbles." Add:
"Type becomes a column in the results event table."

### P9 — Evaluation design: align the counts
Add under the fold table:
> **Recall denominators: 34 episodes targetable at h=24 (33 at h=6, 32 at h=1),
> across 30 independent events.** (46 episodes in test years incl. non-targetable.)

### P10 — Benchmarks: wording + keep numbers
"Budget: **1 false alarm per coin-month on average (≈72/yr)**" (was "≤1").
Numbers verified correct: prior 0.006 / 0/34; persistence 0.063 h6 T1, 0.048 h24 T1, recall 0.26 (9/34).

### P12 — Logreg panel: remove the grid-search text (contradicts fixed settings)
Right column becomes:
> **Fixed settings (no grid):** `C=0.001`, L1 penalty, `class_weight="balanced"`,
> train-fold winsorisation (1st/99th pct), StandardScaler fitted on train only,
> same 5 expanding-window folds, `random_state=42`.

Left column (feature engineering) is verified true in code (`logreg.py::_add_feats`:
`abs_peg_dev`, `peg_deviation × realised_vol_6h`).

### P13 — Logreg performance: REPLACE table (was grid-run numbers)
| Horizon | Tier | n (test rows) | Base rate | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| h=6 | Market (T1) | 205,835 | 0.13% | 0.0200 | 0.8885 |
| h=6 | + On-chain (T2)† | 185,460 | 0.14% | 0.0197 | 0.8732 |
| h=24 | Market (T1) | 204,073 | 0.43% | 0.0312 | 0.8344 |
| h=24 | + On-chain (T2)† | 183,895 | 0.46% | 0.0300 | 0.8262 |

**Drop the Brier column** (class-weight-inflated; compare on PR-AUC/ROC, not Brier).
Keep the † different-sample note.

### P14 — Logreg takeaways: update the on-chain line
- "Adding on-chain does not improve PR-AUC at h24 (0.0312 → 0.0300) — ranking alone says nothing about alarm value."
- Keep linearity + strong-regularisation points.

### P15 — RF panel: remove TODO, paste verified fixed settings
> `n_estimators=500, max_depth=8, min_samples_leaf=50, max_features="sqrt",
> class_weight="balanced_subsample", random_state=42` (verified in `results/models/*.json`)

Recall @ ≤1 FA/cm (h24) — show BOTH tiers:
> T1 0.21 (7/34) · T2 0.29 (10/34) · realised FA/cm 0.82 / 0.79 (budget 1.0)

### P17 — XGB results: add base rate, median lead, h=1 line, realised FA
- Next to PR-AUC add: "base rate h24: 0.43% (T1) / 0.46% (T2)".
- Add: "**Median lead (h24 T2): 9.0h** (p25 4.0 / p75 15.0)".
- Add: "Realised FA/coin-month: T1 1.10 · T2 1.12 (budget 1.0; 2021 3.11 → 2022 0.30)."
- Add one line: "**h = 1 (near-detection):** T1 2/32, T2 3/32, lead ≈1h — the useful signal starts at h=6."
- Keep per-year chart + "2022: 0/9 for every model".

### P18 — Combined performance: restructure into prof's main table (REAL-TIME)
Label: **"Real-time protocol: threshold set on the year before each test year."**

h=24 (T1 / T2):

| Model | PR-AUC (base rate) | Recall @ ≤1 FA/cm | Realised FA/cm (budget 1.0) | Median lead |
|---|---|---|---|---|
| Constant prior | 0.0063 / 0.0064 (0.43/0.46%) | 0.00 / 0.00 | 0.00 / 0.00 | — |
| Persistence | 0.0479 / 0.0338 | 0.26 (9/34) / 0.26 (9/34) | 0.94 / 0.87 | 7h / 10h |
| Logistic | 0.0312 / 0.0300 | 0.38 (13/34) / 0.32 (11/34) | 1.03 / 0.93 | 10h / 7h |
| Random Forest | 0.0685 / 0.0454 | 0.21 (7/34) / 0.29 (10/34) | 0.82 / 0.79 | 8h / 8h |
| XGBoost | 0.0480 / 0.0412 | 0.38 (13/34) / **0.44 (15/34)** | 1.10 / 1.12 | 14h / **9h** |

h=6 (T1 / T2): XGB 0.0522/0.0598, recall 0.12 (4/33) both, FA/cm 1.02/1.05, lead 2.5h/4h;
RF 0.0593/0.0413, 0.06/0.12; logreg 0.0200/0.0197, 0.15/0.12; persistence 0.0626/0.0490, 0.15/0.15.
Keep the budget-curve figure.

### P18b — NEW SLIDE: ex-post fair comparison
Title: **"Fair comparison (ex post): thresholds set on the test year itself"**
Warning banner: **"Uses test-year information — a real-time system cannot do this."**

h=24, @1 FA/cm:

| Model | Recall T1 / T2 | Realised FA/cm | Median lead T1 / T2 |
|---|---|---|---|
| Persistence | 0.18 (6/34) / 0.18 | 1.0 / 1.0 | 13h / 20h |
| Logistic | 0.18 (6/34) / 0.18 | 1.0 / 1.0 | 16h / 16h |
| Random Forest | 0.24 (8/34) / **0.26 (9/34)** | 1.0 / 1.0 | 8h / 8h |
| XGBoost | 0.12 (4/34) / 0.15 (5/34) | 0.95 / 0.69 | 16.5h / 10h |

Figure: `reports/ex_post_vs_real_time_h24.png`.
Say: "The real-time XGB lead partly reflects a well-fitting prior-year threshold.
With test-year thresholds the models converge — RF catches the most (9/34), and the
on-chain edge shrinks to ≈1 episode at h24 (5 vs 4). That's the honest reading:
**on-chain data add a small edge, mainly on the staggered/sustained (SVB-type)
events, not the brief wobbles.**"

### P19 — Value of on-chain: keep SHAP numbers, add the ex-post line + hedge
Keep: 20.7% on-chain |SHAP| (17.7% h6); top features (21.8/13.3/12.1/9.2; 2.7/2.7);
recall-cost line (real-time 0.44 vs 0.38). Add:
> "Ex post, the T2−T1 gap is ≈1 episode (5 vs 4 at h24). On-chain helps on the
> staggered/sustained events: vs T1, T2 catches +1 staggered and +1 sustained
> episode at h24 (`reports/event_tables/caught_by_type.csv`) — modest, at these horizons."

### P20 + P21 → merge into ONE slide (duplicates today)
Keep: rule in one sentence ("at most 1 false alarm per coin-month on average,
≈72/yr, fixed before seeing results"), why alarms not hours, calibrate→pick→apply.
P22's worked example (0.728 → 2024) can sit at the bottom or move to backup.

### P23 — Robustness: update OR numbers, add the 0.5 FA/cm line
- OR-union (fixed-config): h24 T2 @1% hourly-alarm budget: union 0.647 vs 0.500 best
  single (+0.147); @2%: gain 0. Same conclusion: standalone XGB suffices.
  (Old slide's 0.588/+0.088 was the recall-tuned run — superseded.)
- **Add:** "At the tighter 0.5 FA/cm budget the leader flips — RF T2 6/34 vs XGB T2
  3/34 — and on-chain adds nothing there (T1 ≥ T2). The on-chain edge appears at
  budgets ≥1 FA/cm. State it; the conclusion is budget-dependent."
- Keep smoothing negative result + depeg-definition grid.

### P24 — Limitations: fix item 2, reword item 3, add ex-post caveat
- Item 2 → "**All models 0/9 on the January 2022 market-wide wobble** (2/9 even at
  8× budget). May-2022 Terra-era dips are single-hour wicks that fall outside the
  ≥2h episode definition, so Terra is outside the scored set."
- Item 3 → "Realised FA/cm drifts from the budget across years (XGB T2 2021: 3.11,
  2022: 0.30; pooled ≈1.1) — the gap is itself a result about real-world reliability."
- New item → "The ex-post comparison uses test-year thresholds — not implementable
  live; it exists only as the fair model comparison."

### P25 — Takeaways: add hedge + event basis
Add: "Results rest on **30 independent events** (34 targetable episodes at h24)."
Add: "On-chain's edge is real but modest and budget-dependent — if anything, it
concentrates in staggered/sustained multi-coin shocks."

### P26 — delete template junk → Q&A slide
Replace with the three flags if not already visible:
1. 30 independent events underpin the results.
2. Realised FA/cm vs budget gap + "fair comparison is ex post".
3. On-chain chain scope (Ethereum-only?) — confirm with the Dune query.

---

## 2. Suggested reorder (21 pages)

| New | Content | From |
|---|---|---|
| 1 | Title | P1 |
| 2 | Motivation | P2 |
| 3 | Research question | P2/P3 split |
| 4 | Literature | P3 |
| 5 | Data & coverage | P4 + P5 merged |
| 6 | Episodes & events | P6 + P7 merged |
| 7 | Event types (fixed counts) | P8 |
| 8 | Evaluation design (counts aligned) | P9 |
| 9 | Budget & threshold (merged) | P20 + P21 (+P22 inline) |
| 10 | Benchmarks | P10 |
| 11 | Model lineup (3 fixed-config mini-panels) | P11 + P12 + P15 + P16 |
| 12 | Logreg results | P13 + P14 |
| 13 | RF results | P15 results half |
| 14 | XGB results | P17 |
| 15 | Main comparison h6/h24 — real-time | P18 restructured |
| 16 | Ex-post fair comparison | NEW |
| 17 | Does on-chain help? | P19 |
| 18 | Robustness | P23 |
| 19 | Limitations | P24 |
| 20 | Takeaways | P25 |
| 21 | Q&A | P26 fixed |

---

## 3. Where every number comes from (cite on backup slides)

- Main table + ex-post: `reports/ex_post_comparison.csv` (driver `plans/ex_post_comparison.py`)
- PR-AUC / base rates: `reports/modelling_comparison.csv` (fresh `compare_runs.py` run)
- Coverage: `reports/coverage_table.csv` (driver `plans/coverage_table.py`)
- Types + caught-by-type: `reports/event_tables/types_summary.csv`, `caught_by_type.csv` (driver `plans/event_types.py`)
- OR-union: `reports/combination_results.csv` (fixed-tag re-run)
- 0.5 FA/cm line: `reports/local_budget_curve.csv` (budget-0.5 rows)
- Figures: `reports/ex_post_vs_real_time_h24.png`, `reports/local_budget_curve_t{1,2}_h24.png`,
  `reports/figures/xgb_h24_t2_per_year_caught.png`, `reports/figures/*_fixed.*` (SHAP)

## 4. Remaining group actions (not mine)
1. Confirm the Dune query's chain scope → P4 + README (README note already added).
2. Canva edits for sections 1–2; insert the ex-post slide; delete P26 junk.
3. Merge PRs: `results/fixed-config-local` → main, then `cleanup/dead-weight-final-deck` (pushed to origin).
4. Confirm with Vienna before archiving `modelling/robustness_check.py` (kept for now).
