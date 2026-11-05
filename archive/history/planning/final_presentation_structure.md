# Final Presentation Structure — DSE4101 Group 1 (Stablecoin Depeg Early Warning)

> **Drafted:** 2026-10-23 · **Updated:** 2026-10-30 (fixed-config protocol) · **Slides due:** Sun 1 Nov (Week 12) · **Presentation:** Tue 3 Nov, 6–10 pm · **Worth:** 15%
> **How to use:** each slide has an **Owner** (who fills in the content, based on what they built in the repo) and **Exactly what to pull**. Paste your content on your slides and push the deck to the group by Fri 31 Oct so Shao Gjin can package.
> **Protocol:** all reported numbers are **fixed-config** runs (prof's guidance, 2026-10-26 — no tuning grids). Never paste `--tune` numbers.

---

## 0. Context — what everyone has done (this drives the slide ownership)

| Member | Shipped in the repo | Natural slides |
|---|---|---|
| **Sasi** | Tier 2 on-chain features, modelling infra (`train.py`/`evaluate.py`), fixed-config XGBoost, persistence/baseline, SHAP, alarm evaluation rework, OR-combination test, midterm pack | Motive, RQ, features, headline results, SHAP |
| **Alyssa** | Data pipeline fixes, master/episode dataset, **v5 labels + eligibility columns (PR #13)**, threshold-sweep hooks in `evaluate.py` | Episode definition & labels, threshold methodology |
| **Javier** | Market + on-chain **coverage tables (common sample, PR #8)**; L1 logistic (`logreg_c0001_l1`, trained) | Data coverage, logistic model |
| **Russell** | Random Forest (`rf_500_d8_l50`, trained) | Random Forest model |
| **Vienna** | Verified `--horizon 1/6` + `recovery_hours=0` knobs; robustness checks assigned | Robustness slides |
| **Shao Gjin** | Pipeline consolidation, expanding-window CV/splits, PR review/governance | CV design, conclusion, packaging |

**⚠️ Before slides are final:** every model row must come from a **fixed-config** run (prof's guidance, 2026-10-26 — no tuning grids). Javier's logistic and Russell's RF are already trained; to regenerate, use the commands in [agent.md](../agent.md) §6 or `README.md` §7, e.g. `python modelling/train.py --model logreg --horizon 24 --tier 1 --tag logreg_c0001_l1 --model-params '{"C":0.001,"penalty":"l1"}'` and `--tag rf_500_d8_l50 --model-params '{"n_estimators":500,"max_depth":8,"min_samples_leaf":50,"max_features":"sqrt","class_weight":"balanced_subsample","random_state":42}'`. **Do not use `--tune`** — grid-search numbers are not comparable with the report tables.

---

## The narrative arc (one sentence per act)

1. **Why this matters** — depegs are rare, expensive, and hit the "safe" asset class (15-min recap of midterm).
2. **Our question** — can we *predict* the next depeg, and does on-chain transparency help beyond prices?
3. **How we test it honestly** — 6 coins × 345k hours, leak-free expanding-window CV, episode-level early-warning metric.
4. **What we found** — an early-warning system that beats the naive peg-stress rule at the same false-alarm budget; on-chain data adds a modest edge.
5. **What we learned + what we'd do next.**

---

## Slide-by-slide structure

### S1 · Title
**Owner:** Shao Gjin (packaging) — everyone else: name + one-line role.
**Content:** Project title — *"Early-Warning Forecasting of Fiat-Backed Stablecoin Depegs: Do On-Chain Signals See the Run Before the Price Does?"* · Group 1 · DSE4101 · 3 Nov 2026.

### S2 · Motivation — why predict depegs (recap, 1 min)
**Owner:** Sasi (reuse midterm figures).
**Content:** SVB/USDC $0.865, Terra/UST contagion → depegs concentrate in the asset class sold as "safe". A 1-hour warning lets AMMs pull liquidity; 24h lets treasury managers move fiat. One big stat: 79 validated episodes / 56 events across 7 years, 6 coins, $232B peak market cap.
**Pull:** `archive/midterm_presentation_pack/figures/04_depeg_episode_timeline.png`, `02_fiat_stablecoins_mcap.png`.

### S3 · Research question & the early-warning framing
**Owner:** Sasi.
**Content:** Q1 — can info at time *t* predict a depeg onset in (*t*, *t+h*], h ∈ {1,6,24}? Q2 — do on-chain change features improve forecasts beyond market data alone (Tier 1 vs Tier 2)? **Objective (group decision 13 Oct): catch every episode — tolerate false alarms.** Headline metrics = **episode recall at ≤1 false alarm/coin-month** and **median lead time**; PR-AUC secondary. (This is the 2-line story that justifies every design choice later — say it out loud on this slide.)

### S4 · Data & coverage — 6 coins, 345k coin-hours
**Owner:** Javier.
**Content:** USDT/USDC/BUSD/DAI/TUSD/PAX, Sep 2018 – Dec 2025, hourly price (DefiLlama) + exchange volume + Dune on-chain. Coverage table: price/volume/on-chain present for ~93–100% of hours per coin. Mention the sample note: Tier 2 is a subset of Tier 1 (on-chain coverage), so comparisons must be read carefully.
**Pull:** coverage tables from PR #8 (`master.parquet` coverage) — make one clean slide table: coin × rows × % price / % volume / % on-chain.

### S5 · What counts as a "depeg"? Labels v5
**Owner:** Alyssa.
**Content:** price < $0.99 for ≥ 2 consecutive hours → episode; 24h bridge merges dips; onset labels `label_h{1,6,24}` = episode *starts* in next h hours; in-episode + recovery-window hours excluded (at-risk sample). 79 episodes → 56 events; 32–34 targetable OOS episodes at h=1/6/24. Emphasise: no look-ahead, onset-only labels, eligibility columns (`prediction_eligible_market_h{h}` / `..._onchain_h{h}`).

### S6 · Leak-free evaluation design
**Owner:** Shao Gjin.
**Content:** Expanding-window CV, test years 2021–2025; 24h purge + 24h embargo; episode-boundary-aware drops; nested inner_train/inner_val used for **leak-free threshold selection** (no model tuning). Pooled OOS = **46 episodes / 30 events** across all 6 coins. Thresholds chosen on inner_val **only**, then applied to that fold's test year — never on test rows.
**Pull:** `data/processed/cv_folds_summary.csv` → one slide table (train rows / test rows / episodes / positives per fold year).

### S7 · Features — Tier 1 vs Tier 2
**Owner:** Sasi.
**Content:** Tier 1 = 9 market features (peg deviation, log returns 1/6/24h, realised vol 6/24h, volume changes). Tier 2 = + 12 on-chain logdiff change features (tx count/volume, active senders/receivers × 1/6/24h). Why logdiffs: raw levels leak coin identity (RF probe 0.70 acc on levels vs 0.41 on diffs) and are economically meaningful as *flow shocks*. NaN kept as missing info (XGBoost native).

### S8 · Models & benchmark strategy
**Owners:** Javier (logistic) + Russell (random forest); Sasi provides the shared contract framing.
**Content:** All models registered through the same `BaseModel` contract + `train.py` — identical folds, imputation, **fixed settings every fold/year (no tuning grid)**, same ≤1 FA/coin-month threshold protocol, `random_state=42`. Fixed configs: logistic `C=0.001` L1; RF `500 × depth 8, leaf 50`; XGBoost `depth 3, lr 0.05, 300 trees`. Benchmarks: constant prior (floor) + peg-stress persistence.
**Pull:** each owner's `results/models/*.json` params + one-line model description.

### S9 · Headline results — episode recall at the false-alarm budget
**Owner:** Sasi (this is THE money slide).
**Content:** Table: episode recall at ≤1 FA/coin-month, h=1/6/24, across constant prior / persistence / logistic / RF / XGB T1 / XGB T2. Star facts:
1. At h=24, XGB **Tier 2 warns 15/34 targetable episodes (44%)** vs **9/34 (26%)** for the persistence rule — at the same alarm budget; Tier 1 warns 13/34.
2. Median lead at h=24: XGB Tier 1 **14h**, Tier 2 9h, persistence 7h (note: the real-time lead partly reflects the prior-year threshold).
3. h=6 is weak for every model (best 5/33) — the operational story is the 24h horizon.
**Pull:** `reports/ex_post_comparison.csv` (`mode = real_time`) → clean table; alarm-curve figure from `results/evaluation/{xgb_d3_lr005_300,rf_500_d8_l50,logreg_c0001_l1}_alarm_curve.csv` (do **not** plot the persistence curve — degenerate, zero alarms).

### S10 · Does on-chain data help? Tier 1 vs Tier 2 + SHAP
**Owner:** Sasi.
**Content:** Honest two-part answer: PR-AUC is **mixed** — T2 0.041 vs T1 0.048 at h=24 (slightly worse), T2 0.060 vs 0.052 at h=6 (better) — so ranking alone does not make the case. The case is **episode coverage at a fixed budget** (15/34 vs 13/34 at h=24). SHAP: `realised_vol_24h` 21.8% of |SHAP|, `realised_vol_6h` 13.3%, `peg_deviation` 12.1% (market stress dominates, matches literature); **on-chain share 20.7% (h=24) / 17.7% (h=6)**. Caveat: Tier 2 ⊂ Tier 1 sample — flag the composition effect.
**Pull:** `reports/figures/shap_ranking_xgboost_t2_h24.png`, `group_share_xgboost_t2_h24.png`, `beeswarm_xgboost_t2_h24.png`.

### S11 · Model comparison — logistic vs RF vs XGB
**Owners:** Javier + Russell + Alyssa (threshold side).
**Content:** One table: per model, per horizon — PR-AUC, ROC-AUC, episode recall at ≤1 FA/coin-month, median lead. Honest line: no model dominates everywhere (RF wins PR-AUC at h=24, XGB wins episode recall at h=24); persistence remains the best single-feature rule and the bar to beat. The 90%-recall alarm-rate column from the old deck came from recall-tuned runs and must be recomputed or dropped.
**Pull:** append their rows to `reports/ex_post_comparison.csv` and `reports/modelling_comparison.csv`.

### S12 · Robustness & ablations
**Owner:** Vienna (+ Alyssa on threshold sensitivity).
**Content:** (1) horizons 1/6/24 all shown; (2) `recovery_hours=0` variant; (3) threshold sweep at 0.5/1/2/5% budgets; (4) bundle tests — alarm smoothing (2 consecutive hours) is a **negative result** (kills recall) and OR-combining persistence+XGB adds little beyond the better single system. Present these as honest science, not failures.
**Pull:** `reports/combination_results.csv`, sensitivity rows from `reports/alarm_tables/`.

### S13 · Limitations
**Owner:** everyone contributes one line; Sasi assembles.
**Content:** Extreme imbalance (0.03–0.43% positives pooled OOS) → probabilities need recalibration (Brier caution with `scale_pos_weight`); Tier 2 sample ⊂ Tier 1 sample (composition effect vs feature effect); 2024–25 test years have ~6 positives each → pooled evaluation only; 6 coins / fiat-backed only → no algorithmic-stablecoin generalisation claim; on-chain share is real but modest (17.7–20.7% of |SHAP|).

### S14 · Conclusion & takeaways
**Owner:** Shao Gjin.
**Content:** 3 takeaways: (1) depeg onsets are forecastable out-of-sample at horizons that matter operationally; (2) the right metric for a rare-disaster alarm is episode recall at a fixed false-alarm budget, and under that metric XGBoost Tier 2 is the most complete early-warning model (15/34 episodes, 9h median lead); (3) on-chain transparency contributes a modest but real edge — Tier 2 covers 2 more episodes than Tier 1 at the same budget, with ~21% of SHAP mass. Close with one future-work line (e.g., cross-venue order-book depth, algorithmic stablecoins).

### Backup slides (for Q&A, not the main deck)
- Full `reports/ex_post_comparison.csv` and `modelling_comparison.csv` tables.
- `reports/alarm_tables/` per-episode caught/missed detail (ready if the prof asks "which episodes did you miss?").
- Midterm Q&A cheat sheet still applies (why 6 coins, why 1%/2h definition).

---

## Division of presenting (suggested, 10-min slot ≈ 14 slides)

| Segment | Slides | Speaker | ~Time |
|---|---|---|---|
| Motive + RQ | S1–S3 | Sasi | 1.5 min |
| Data + labels + design | S4–S6 | Javier → Alyssa → Shao Gjin | 2.5 min |
| Features + models | S7–S8 | Sasi + Javier/Russell (one line each) | 2 min |
| Results + SHAP + comparison | S9–S11 | Sasi (+ Alyssa on thresholds) | 2.5 min |
| Robustness + limitations + conclusion | S12–S14 | Vienna → Sasi → Shao Gjin | 1.5 min |

## Checklist before Fri 31 Oct

- [ ] Javier: logistic fixed-config run (`logreg_c0001_l1`) → rows into the two CSV tables (Slides 8, 11).
- [ ] Russell: same for `random_forest` (`rf_500_d8_l50`).
- [ ] Alyssa: threshold-sweep summary + v5 label/eligibility figure (Slide 5, 11).
- [ ] Vienna: robustness run results for `recovery_hours=0` + budget sweep (Slide 12).
- [ ] Sasi: alarm-curve figure + SHAP figure selection + headline table cleaned (Slides 9–10).
- [ ] Shao Gjin: master deck template + CV table + conclusion slide; everyone keys content into their slides.
- [ ] Re-run `python plans/ex_post_comparison.py` and `python modelling/compare_runs.py --save reports/modelling_comparison.csv` on the final numbers so every slide cites one source of truth.
