# Final Presentation Structure — DSE4101 Group 1 (Stablecoin Depeg Early Warning)

> **Drafted:** 2026-10-23 · **Slides due:** Sun 1 Nov (Week 12) · **Presentation:** Tue 3 Nov, 6–10 pm · **Worth:** 15%
> **How to use:** each slide has an **Owner** (who fills in the content, based on what they built in the repo) and **Exactly what to pull**. Paste your content on your slides and push the deck to the group by Fri 31 Oct so Shao Gjin can package.

---

## 0. Context — what everyone has done (this drives the slide ownership)

| Member | Shipped in the repo | Natural slides |
|---|---|---|
| **Sasi** | Tier 2 on-chain features, modelling infra (`train.py`/`evaluate.py`), XGBoost + tuning, persistence/baseline, SHAP, recall-first rework, OR-combination test, midterm pack | Motive, RQ, features, headline results, SHAP |
| **Alyssa** | Data pipeline fixes, master/episode dataset, **v5 labels + eligibility columns (PR #13)**, threshold-sweep hooks in `evaluate.py` | Episode definition & labels, threshold methodology |
| **Javier** | Market + on-chain **coverage tables (common sample, PR #8)**; L1 logistic assigned (not in registry yet) | Data coverage, logistic model |
| **Russell** | Random Forest assigned (not in registry yet) | Random Forest model |
| **Vienna** | Verified `--horizon 1/6` + `recovery_hours=0` knobs; robustness checks assigned | Robustness slides |
| **Shao Gjin** | Pipeline consolidation, expanding-window CV/splits, PR review/governance | CV design, conclusion, packaging |

**⚠️ Before slides are final:** Javier and Russell must run their models through `train.py` (same folds, same recall objective) so Slides 8 & 11 have real numbers. Commands are in [agent.md](../agent.md#L126) — `python modelling/train.py --model logistic --horizon 24 --tier 1 --tag rec --tune --tune-metric recall_at_rate --max-alarm-rate 0.01` (same for `random_forest`).

---

## The narrative arc (one sentence per act)

1. **Why this matters** — depegs are rare, expensive, and hit the "safe" asset class (15-min recap of midterm).
2. **Our question** — can we *predict* the next depeg, and does on-chain transparency help beyond prices?
3. **How we test it honestly** — 6 coins × 345k hours, leak-free expanding-window CV, episode-level early-warning metric.
4. **What we found** — an early-warning system that beats naive rules; on-chain data earns its place at h=24.
5. **What we learned + what we'd do next.**

---

## Slide-by-slide structure

### S1 · Title
**Owner:** Shao Gjin (packaging) — everyone else: name + one-line role.
**Content:** Project title — *"Early-Warning Forecasting of Fiat-Backed Stablecoin Depegs: Do On-Chain Signals See the Run Before the Price Does?"* · Group 1 · DSE4101 · 3 Nov 2026.

### S2 · Motivation — why predict depegs (recap, 1 min)
**Owner:** Sasi (reuse midterm figures).
**Content:** SVB/USDC $0.865, Terra/UST contagion → depegs concentrate in the asset class sold as "safe". A 1-hour warning lets AMMs pull liquidity; 24h lets treasury managers move fiat. One big stat: 83 validated episodes across 7 years, 6 coins, $232B peak liquidity.
**Pull:** `archive/midterm_presentation_pack/figures/04_depeg_episode_timeline.png`, `02_fiat_stablecoins_mcap.png`.

### S3 · Research question & the early-warning framing
**Owner:** Sasi.
**Content:** Q1 — can info at time *t* predict a depeg onset in (*t*, *t+h*], h ∈ {1,6,24}? Q2 — do on-chain change features improve forecasts beyond market data alone (Tier 1 vs Tier 2)? **Objective (group decision 13 Oct): catch every episode — recall-first, tolerate false alarms.** Headline metric = episode-level recall vs hourly alarm rate; PR-AUC secondary. (This is the 2-line story that justifies every design choice later — say it out loud on this slide.)

### S4 · Data & coverage — 6 coins, 345k coin-hours
**Owner:** Javier.
**Content:** USDT/USDC/BUSD/DAI/TUSD/PAX, Sep 2018 – Dec 2025, hourly price (DefiLlama) + exchange volume + Dune on-chain. Coverage table: price/volume/on-chain present for ~93–100% of hours per coin. Mention the sample note: Tier 2 is a subset of Tier 1 (on-chain coverage), so comparisons must be read carefully.
**Pull:** coverage tables from PR #8 (`master.parquet` coverage) — make one clean slide table: coin × rows × % price / % volume / % on-chain.

### S5 · What counts as a "depeg"? Labels v5
**Owner:** Alyssa.
**Content:** price < $0.99 for ≥ 2 consecutive hours → episode; 24h bridge merges dips; onset labels `label_h{1,6,24}` = episode *starts* in next h hours; in-episode + recovery-window hours excluded (at-risk sample). Positives: 208 / 751 / 2,232 (h=1/6/24); 79 episodes, 32–34 OOS. Emphasise: no look-ahead, onset-only labels, eligibility columns (`prediction_eligible_market_h{h}` / `..._onchain_h{h}`).

### S6 · Leak-free evaluation design
**Owner:** Shao Gjin.
**Content:** Expanding-window CV, test years 2021–2025; 24h purge + 24h embargo; episode-boundary-aware drops; nested inner_train/inner_val for tuning. Pooled OOS = 49 episodes across all 6 coins (vs 2 in the old split). Thresholds chosen on inner_val **only**, then applied to that fold's test year — never tuned on test rows.
**Pull:** `data/processed/cv_folds_summary.csv` → one slide table (train rows / test rows / episodes / positives per fold year).

### S7 · Features — Tier 1 vs Tier 2
**Owner:** Sasi.
**Content:** Tier 1 = 9 market features (peg deviation, log returns 1/6/24h, realised vol 6/24h, volume changes). Tier 2 = + 12 on-chain logdiff change features (tx count/volume, active senders/receivers × 1/6/24h). Why logdiffs: raw levels leak coin identity (RF probe 0.70 acc on levels vs 0.41 on diffs) and are economically meaningful as *flow shocks*. NaN kept as missing info (XGBoost native).

### S8 · Models & benchmark strategy
**Owners:** Javier (logistic) + Russell (random forest); Sasi provides the shared contract framing.
**Content:** All models registered through the same `BaseModel` contract + `train.py` — identical folds, imputation, tuning (≤32 combos on inner_val), same recall@1% objective, `random_state=42`. Benchmarks: constant prior (floor) + peg-stress persistence. Model lineup: L1 logistic, Random Forest, XGBoost (tuned).
**Pull:** each owner's `results/models/*.json` params + one-line model description.

### S9 · Headline results — episode recall vs alarm rate
**Owner:** Sasi (this is THE money slide).
**Content:** Table: episode recall at 1% alarm budget, h=1/6/24, across constant prior / persistence / XGB T1 / XGB T2. Star the two facts:
1. XGB T2 h=24 reaches **0.35 episode recall at 1% alarm rate** (matches persistence) **and needs only 1.8% alarm rate for 90% recall** vs 4.7% persistence / 4.5% T1 — the cheapest path to catching ~all episodes.
2. Recall-first tuning (inner-val thresholds) flips the naive conclusion: on-chain data's value shows up in *recall cost*, not PR-AUC.
**Pull:** `reports/recall_operating_points.csv` → clean table; maybe an alarm-curve figure (recall vs rate) from `results/evaluation/`.

### S10 · Does on-chain data help? Tier 1 vs Tier 2 + SHAP
**Owner:** Sasi.
**Content:** Honest two-part answer: PR-AUC delta ≈ 0 (ranking unchanged) — **but** the recall-cost story at h=24 says yes (1.8% vs 4.5%). SHAP: `realised_vol_24h` = 33–45% of |SHAP|, then peg_deviation, volume changes (matches literature); on-chain share = 6–7% (top: `tx_volume_logdiff_24h`, `active_receivers_logdiff_24h`, `active_senders_logdiff_24h`). Interpretation: market features carry the main signal; on-chain flows add a small but *cheap-alarm* early-warning edge.
**Pull:** `reports/figures/shap_ranking_xgboost_t2_h24.png`, `group_share_xgboost_t2_h24.png`, `beeswarm_xgboost_t2_h24.png`.

### S11 · Model comparison — logistic vs RF vs XGB
**Owners:** Javier + Russell + Alyssa (threshold side).
**Content:** One table: per model, per horizon — PR-AUC, ROC-AUC, recall@1%, alarm rate for 90% recall. Honest line: no model dominates everywhere; XGBoost T2 is the recommended alarm system because it wins the recall-cost trade-off; persistence remains the best single-feature rule (nice benchmark for discussion).
**Pull:** append their rows to `reports/recall_operating_points.csv` and `reports/modelling_comparison.csv`.

### S12 · Robustness & ablations
**Owner:** Vienna (+ Alyssa on threshold sensitivity).
**Content:** (1) horizons 1/6/24 all shown; (2) `recovery_hours=0` variant; (3) threshold sweep at 0.5/1/2/5% budgets; (4) bundle tests — alarm smoothing (2 consecutive hours) is a **negative result** (kills recall) and OR-combining persistence+XGB adds little beyond the better single system. Present these as honest science, not failures.
**Pull:** `reports/combination_results.csv`, sensitivity rows from `reports/alarm_tables/`.

### S13 · Limitations
**Owner:** everyone contributes one line; Sasi assembles.
**Content:** Extreme imbalance (0.02–0.6% positives) → probabilities need recalibration (Brier caution with `scale_pos_weight`); Tier 2 sample ⊂ Tier 1 sample (composition effect vs feature effect); 2024–25 test years have ~1 episode each → pooled evaluation only; 6 coins / fiat-backed only → no algorithmic-stablecoin generalisation claim; on-chain share small in absolute terms.

### S14 · Conclusion & takeaways
**Owner:** Shao Gjin.
**Content:** 3 takeaways: (1) depeg onsets are forecastable out-of-sample at horizons that matter operationally; (2) the right metric for a rare-disaster alarm is episode recall vs alarm rate, and under that metric XGBoost Tier 2 is the best early-warning system (1.8% alarm rate for 90% recall at h=24); (3) on-chain transparency contributes a modest but real edge — biggest at the 24h institutional window. Close with one future-work line (e.g., cross-venue order-book depth, algorithmic stablecoins).

### Backup slides (for Q&A, not the main deck)
- Full `reports/recall_operating_points.csv` and `modelling_comparison.csv` tables.
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

- [ ] Javier: logistic registered in `modelling/models/__init__.py` + recall-tuned runs → rows into the two CSV tables (Slides 8, 11).
- [ ] Russell: same for random_forest.
- [ ] Alyssa: threshold-sweep summary + v5 label/eligibility figure (Slide 5, 11).
- [ ] Vienna: robustness run results for `recovery_hours=0` + budget sweep (Slide 12).
- [ ] Sasi: alarm-curve figure + SHAP figure selection + headline table cleaned (Slides 9–10).
- [ ] Shao Gjin: master deck template + CV table + conclusion slide; everyone keys content into their slides.
- [ ] Re-run `python modelling/alarm_report.py --save ...` on final numbers so every slide cites one source of truth.
