# Agent Handoff: DSE4101 Stablecoin Depegging Forecasting

> **Purpose:** This is the master context file for **any human or AI agent** working on this repo. If you (or your AI) are about to do anything here, read this file first — or paste it into your agent and tell it to follow the rules in Section 3.
>
> **Last updated:** 2026-10-30 — fixed-config protocol (prof's guidance, 2026-10-26): no tuning grids in
> final results. Numbers below are from the fixed-config runs; `README.md` §5–§6 is the source of truth.

---

## 1. Project in one paragraph

NUS DSE4101 capstone (Digital Currencies Group 1). **Research question:** can information available at time *t* predict whether a fiat-backed stablecoin will depeg within the next *h* ∈ {1, 6, 24} hours — and does adding **on-chain** data improve forecasts beyond **market** data alone?

- **Tier 1** = market only (9 features). **Tier 2** = market + 12 on-chain change features.
- **Data:** hourly, 6 coins (USDT, USDC, BUSD, DAI, TUSD, PAX), Sep 2018 – Dec 2025. Canonical dataset: `data/processed/master.parquet` (**345,367 coin-hours × 55 columns**).
- **Evaluation:** expanding-window CV, 5 folds (test years 2021–2025), 24h purge + 24h embargo, episode-boundary-aware. **Headline = episode recall @ ≤1 false alarm/coin-month + median lead time; pooled OOS PR-AUC is secondary** (positives are 0.02–0.57% of rows).
- **Status:** pipeline frozen; Tier 2 ready; shared modelling infra + fixed-config XGBoost / Random Forest / logistic + SHAP shipped. Report due ~Week 12 (early Nov 2026).

---

## 2. Hard rules (break these and your PR will be rejected)

| # | Rule |
|---|---|
| R1 | `data_processing/` is **FROZEN**. The only sanctioned change (Tier 2 change features in `02b`) is **already done**. |
| R2 | **NEVER run `fetch_stablecoin_data.py`** (1,200+ HTTP requests). `master.parquet` is committed and current. |
| R3 | All modelling data access goes through `modelling/prepare_data.py` → `load_folds()`. Never read raw CSVs/parquets inside model code. |
| R4 | Raw on-chain levels (`tx_count`, `tx_volume`, `active_senders`, `active_receivers`) are **NOT features** — they identify the coin (RF probe: 0.70 acc). Use the `*_logdiff_*` columns instead. |
| R5 | No personal folders or duplicate pipelines. One feature branch per task (`feature/<name>`), PR into `main`, Shao Gjin reviews. |
| R6 | `results/` is gitignored scratch (predictions, models). Final tables/figures go in `reports/`. |

---

## 3. Repository map (what is where, what is current)

```
data/
  processed/master.parquet          # canonical dataset (345,367 coin-hours, 55 cols; committed)
  processed/depeg_episodes.parquet  # 79-episode manifest (56 events)
  processed/master_backup_32col.parquet  # pre-Tier-2 backup (gitignored, local only)
data_processing/                    # FROZEN pipeline (fetch_stablecoin_data.py, build_market_features.py, build_onchain_features.py, merge_and_label.py, splits.py, config.py, verify_embargo_and_episodes.py)
modelling/                          # ACTIVE workspace
  prepare_data.py                   # THE data contract: load_folds(horizon, tier, recovery_hours)
  models/base.py                    # BaseModel + MedianImputer (model contract)
  models/baseline.py                # "baseline" (constant prior), "persistence" (peg stress)
  models/logreg.py                  # "logreg" (fixed config, L1)
  models/random_forest_model.py     # "random_forest" (fixed config)
  models/xgboost_model.py           # "xgboost" (fixed config, NaN-native)
  models/__init__.py                # MODEL_REGISTRY — register new models here
  train.py                          # orchestrator (folds, fit, results)
  evaluate.py                       # pooled metrics, thresholds, alarm curves
  compare_runs.py                   # Tier 1 vs Tier 2 comparison table
  rolling_threshold.py              # per-fold threshold selection (inner_val -> test year)
  feature_importance/shap_analysis.py  # SHAP values, rankings, beeswarms, group shares
  checks/test_tier2_features.py     # coin-identity leakage probe + hygiene checks
reports/
  modelling_comparison.csv          # pooled OOS PR-AUC / ROC-AUC table
  ex_post_comparison.csv            # episode recall + FA/coin-month + median lead (HEADLINE)
  alarm_tables/                     # per-episode caught/missed tables
  figures/                          # SHAP outputs
results/                            # gitignored: predictions/, models/, evaluation/
agent.md                            # this file
plans/                              # analysis drivers + planning notes
```

---

## 4. Data contract (the ONLY way to get modelling data)

```python
from modelling.prepare_data import load_folds

for fold in load_folds(horizon=24, tier=2):        # tier: 1 or 2
    X_train, y_train = fold["X_train"], fold["y_train"]
    X_test,  y_test  = fold["X_test"],  fold["y_test"]
    test_meta        = fold["test_meta"]           # coin, hour, episode_id
    X_inner_train, y_inner_train = fold["X_inner_train"], fold["y_inner_train"]  # nested selection
    X_inner_val,   y_inner_val   = fold["X_inner_val"],   fold["y_inner_val"]
    inner_val_meta  = fold["inner_val_meta"]
```

- Features per tier: `prepare_data.MARKET_FEATURES` (9) and `prepare_data.ONCHAIN_FEATURES` (12 `*_logdiff_*` columns).
- **At-risk sample:** onset-only labels; hours inside an episode and the 24h recovery window are already dropped.
- **NaN = missing information, kept as NaN.** XGBoost handles it natively; logistic/RF set `needs_imputation=True` (train-only median imputation, done by `train.py`).
- Run `python modelling/prepare_data.py` to print sample stats. It must show the current manifest (79 episodes; 32/33/34 targetable OOS at h=1/6/24) — if it prints 83 episodes, you are on a stale `master.parquet`.

---

## 5. Model contract (how to add YOUR model)

Subclass `modelling/models/base.py::BaseModel`:

```python
class MyModel(BaseModel):
    name = "my_model"                 # CLI name for train.py
    needs_imputation = True           # False if NaN-native (XGBoost)
    param_grid = [{"C": 0.1}, ...]    # optional/legacy --tune combos (unused in final results)
    supports_early_stopping = False   # True if fit() uses X_val for stopping

    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    def predict_proba(self, X) -> np.ndarray: ...   # shape (n, 2), col 1 = P(y=1)
    def get_params(self) -> dict: ...               # for the results log
```

Then **register it** in `modelling/models/__init__.py` (`MODEL_REGISTRY["my_model"] = MyModel`). Done — `train.py` handles folds, imputation, fitting, saving, and evaluation for you. Example: `modelling/models/xgboost_model.py`.

### Model protocol — fixed settings, NO tuning grid (prof's guidance, 2026-10-26)

Final reported numbers use **one fixed configuration per model, identical in every fold/year**.
No per-fold grid search, no `--tune`, no per-person `GridSearchCV` loops.

| Model | Fixed configuration |
|---|---|
| Constant prior (floor) | predicts the historical base rate |
| Peg-stress persistence | score = −`peg_deviation`, clipped 0–0.01 |
| Logistic (L1) | `C=0.001, penalty="l1", class_weight="balanced", winsor=0.01` (+ `abs_peg_dev`, `peg_deviation × realised_vol_6h`) |
| Random Forest | `n_estimators=500, max_depth=8, min_samples_leaf=50, max_features="sqrt", class_weight="balanced_subsample", random_state=42` |
| XGBoost | `max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=1, scale_pos_weight="auto", tree_method="hist", random_state=42` |

Pin the configuration with `--model-params '<json>'` and give the run a stable tag
(`rf_500_d8_l50`, `xgb_d3_lr005_300`, `logreg_c0001_l1`, `pers_fixed`, `base_fixed`) —
the report tables are keyed on those tags. `--model-params` and `--tune` are mutually
exclusive: `train.py` rejects the combination.

**Why no tuning:** with a 24h purge/embargo and only 32–34 targetable OOS episodes,
a per-fold grid selected on inner-val overfits the selection metric and makes runs
non-comparable across members. `--tune` and each model's `param_grid` remain in the
code as optional machinery, but they must not be used for reported numbers. If you
want a robustness line, report it as an explicitly labelled ablation.

For sklearn models (logistic, RF) set `needs_imputation=True` — train-only median
imputation is applied by `train.py`. Decision thresholds are NEVER chosen on test
data: per test year Y, the threshold is selected on `inner_val` (Y−1) at the
≤1 false-alarm/coin-month budget and applied to Y (`modelling/rolling_threshold.py`).
The ex-post variant (threshold on the test year itself) exists only as a
fair-comparison diagnostic in `plans/ex_post_comparison.py` and is not deployable.

---

## 6. Exact commands (run from the repo root)

```bash
python modelling/prepare_data.py                                   # data sanity check
# Fixed-config training (no --tune; tag must match the report tables)
python modelling/train.py --model xgboost --horizon 24 --tier 2 \
       --tag xgb_d3_lr005_300 \
       --model-params '{"max_depth":3,"learning_rate":0.05,"n_estimators":300}'
python modelling/train.py --model random_forest --horizon 24 --tier 1 \
       --tag rf_500_d8_l50 \
       --model-params '{"n_estimators":500,"max_depth":8,"min_samples_leaf":50,"max_features":"sqrt","class_weight":"balanced_subsample","random_state":42}'
python modelling/train.py --model logreg --horizon 24 --tier 1 \
       --tag logreg_c0001_l1 --model-params '{"C":0.001,"penalty":"l1"}'
python modelling/train.py --model baseline --horizon 24 --tier 1 --tag base_fixed     # constant prior
python modelling/train.py --model persistence --horizon 24 --tier 1 --tag pers_fixed  # peg-stress baseline
python modelling/compare_runs.py --save reports/modelling_comparison.csv     # PR-AUC / ROC-AUC table
python plans/ex_post_comparison.py          # real-time vs ex-post + median lead (HEADLINE)
python plans/local_curve.py                 # episode recall vs false-alarm budget sweep
python plans/event_types.py                 # event types + caught/missed by type
python plans/coverage_table.py              # per-coin Tier-2 coverage
python modelling/feature_importance/shap_analysis.py --model xgboost --horizon 24 --tier 2
python modelling/evaluate.py                                       # self-test
python modelling/checks/test_tier2_features.py                     # leakage probe
# If master.parquet ever lacks the *_logdiff_* columns:
python data_processing/build_onchain_features.py --update-master
```

Outputs: per-fold predictions → `results/predictions/`, models + params JSON → `results/models/`, pooled OOS report + alarm curve → `results/evaluation/`, SHAP figures → `reports/figures/`, headline comparison → `reports/ex_post_comparison.csv`, budget sweep → `reports/local_budget_curve.csv`, per-episode caught/missed tables → `reports/alarm_tables/`.

---

## 7. Findings so far (build on these — don't redo the work)

**Objective:** early warning — catch depeg episodes, tolerate false alarms. Headline
metrics: **episode recall at the ≤1 false alarm/coin-month budget** and **median lead
time**; PR-AUC (reported with its base rate) is secondary. Thresholds are never chosen
on test rows (§5). Sources: `reports/ex_post_comparison.csv` (real-time mode),
`reports/modelling_comparison.csv`, `reports/recall_operating_points.csv`.

**Episode recall at ≤1 FA/coin-month** (Tier 1 / Tier 2; pooled OOS 2021–25;
targetable episodes 32 / 33 / 34 at h=1/6/24):

| Model | h=1 | h=6 | h=24 |
|---|---|---|---|
| Constant prior | 0/32 / 0/32 | 0/33 / 0/33 | 0/34 / 0/34 |
| Persistence | 2/32 / — | 5/33 / 5/33 | 9/34 / 9/34 |
| Logistic (L1) | — | 5/33 / 4/33 | 13/34 / 11/34 |
| Random Forest | — | 2/33 / 4/33 | 7/34 / 10/34 |
| XGBoost | 2/32 / 3/32 | 4/33 / 4/33 | 13/34 / **15/34** |

**Median lead time (hours, Tier 1 / Tier 2), h=24 real-time:** persistence 7/10 ·
logistic 10/7 · Random Forest 8/8 · XGBoost **14/9**.

Read: at h=24 XGBoost Tier 2 catches the most episodes (15/34 vs persistence 9/34) and
XGBoost Tier 1 has the longest lead (14h). At h=6 nobody is strong (max 5/33) and
persistence is as good as anything else.

**Pooled OOS PR-AUC (secondary):**

| Model | h=1 T1/T2 | h=6 T1/T2 | h=24 T1/T2 |
|---|---|---|---|
| Constant prior | 0.0003 / — | 0.0020 / 0.0020 | 0.0063 / 0.0064 |
| Persistence | 0.0624 / — | 0.0626 / 0.0490 | 0.0479 / 0.0338 |
| Logistic (L1) | — | 0.0200 / 0.0197 | 0.0312 / 0.0300 |
| Random Forest | — | 0.0593 / 0.0413 | **0.0685** / **0.0454** |
| XGBoost | 0.0276 / 0.0164 | 0.0522 / 0.0598 | 0.0480 / 0.0412 |

- **PR-AUC does not crown the ML models.** Persistence leads at h=1 and h=6 (0.062);
  Random Forest leads at h=24 (0.069). This is expected: persistence scores *current*
  peg stress and depeg labels are sticky. Do not claim PR-AUC superiority — see the
  persistence note below.
- **ROC-AUC (diagnostic only, `reports/modelling_comparison.csv`):** persistence 0.58–0.74
  vs ML 0.77–0.89. ROC-AUC is lenient under class imbalance (its false-positive rate is
  dominated by the huge negative class), so it is reported as a secondary diagnostic,
  never as the headline and never instead of PR-AUC. It does show that persistence ranks
  the rest of the distribution poorly.
- **Tier 2 − Tier 1 is mixed, not clearly positive.** PR-AUC h=6 0.052 → 0.060 but
  h=24 0.048 → 0.041 (worse); episode recall h=24 13/34 → 15/34. Report the delta with
  the sample caveat (Tier 2 ⊂ Tier 1), and don't oversell.
- **SHAP (fixed XGB Tier 2, pooled OOS):** `realised_vol_24h` 21.8% of |SHAP|,
  `realised_vol_6h` 13.3%, `peg_deviation` 12.1%; **on-chain share 20.7% (h=24),
  17.7% (h=6)**. Top on-chain: `tx_volume_logdiff_24h`, `active_senders_logdiff_24h`,
  `active_receivers_logdiff_24h`.
- **Brier caution:** `scale_pos_weight` inflates XGBoost probabilities. Compare models
  on PR-AUC / ROC-AUC or the alarm curve, not Brier, unless probabilities are recalibrated.
- **Alarm-curve figures:** use `results/evaluation/{tag}_alarm_curve.csv`. Caution: the
  persistence curve is degenerate (its scores cap at 0.01 while the threshold grid
  starts at 0.01 → zero alarms at every grid point). Build figures from the fixed-config
  curves or from the operating points in `reports/ex_post_comparison.csv`.
- **Known warts:** 2022 is a universal miss (every model 0/9 — Terra/LUNA had no 24h
  precursor); 2021 false alarms overshoot the budget (budget is enforced at calibration,
  not in test); h=1 is near-detection only (the useful signal starts at h=6).
- **Persistence is a strong but degenerate baseline.** It learns nothing, uses no
  on-chain data, and scores only current `peg_deviation` — so it fires once the peg is
  *already* off (its ex-post thresholds sit 0.5–0.9% below $1). Quote it as the bar to
  beat on PR-AUC and as the rule XGBoost beats on episode recall and lead time; never
  present it as a learned model.

**Data note (2026-10-17, Alyssa PR #13):** `master.parquet` carries tier-specific
eligibility — Tier 1 sample = `prediction_eligible_market_h{h}` (295k rows), Tier 2 =
`prediction_eligible_market_onchain_h{h}` (271k rows). The two samples differ (Tier 2
is a subset), so the T1-vs-T2 delta mixes feature effect and sample composition — flag
it in the report.

---

## 8. Gotchas your AI WILL hit (and how they were solved)

1. **`X_inner_val` is a subset of `X_train`** in these folds. Early stopping on it while fitting on `X_train` evaluates on rows the model already saw (gave inner-val PR-AUC = 1.0). Fixed-config XGBoost still passes `inner_val` as the eval set, so treat inner-val scores as optimistic (test-year metrics are unaffected). If you ever re-enable `--tune`, use its nested pattern: candidates fit on `X_inner_train` → select on `X_inner_val` → refit on full `X_train` with `n_estimators = best_iteration`.
2. **Onset rows (label = 1) have `episode_id = NaN`** — the id marks in-episode hours. To map a flagged hour to its target episode, use `depeg_episodes.parquet` with `start ∈ (t, t+h]` (`evaluate.py` does this).
3. **The zip's Dune CSVs have no `pct_change` columns** — `02b` was fixed to the real schema (`hour, tx_count, total_volume, active_senders, active_receivers`).
4. **`venue_switch` appears twice** (`venue_switch_x` / `venue_switch_y`) — `prepare_data.py` ignores both.
5. **Raw levels leak coin identity** (RF accuracy 0.70 levels vs 0.41 logdiffs) — the reason Tier 2 uses change features.
6. **2024 and 2025 test sets have ~6 positives each** — always evaluate on pooled OOS predictions, never per-year.
7. **Git/GitHub:** `sasssyboujee/DSE4101-Topic-2-Project` is *not* a registered fork — push feature branches **directly to `shaogjintan/stablecoin-depegging-analysis`** and open PRs there. **Create feature branches from `upstream/main`** (`git fetch upstream && git checkout upstream/main -b feature/...`), never from a stale local `main` — branching from local main re-introduced old milestone folders into upstream (2026-10-13 incident, cleaned up right after).
8. **Large file:** `master.parquet` (68 MB) is committed to git despite GitHub's 50 MB recommendation — keep committing it (the team's convention); don't add LFS mid-project.
9. **Recall-first terminology:** we minimize **false negatives** (missed depegs) and accept false positives (false alarms). "True negatives" are correct non-alarms — do NOT reduce those. Headline = episode recall vs the false-alarm budget; PR-AUC is secondary.
10. **Thresholds are selected on inner_val only**, then applied to OOS test. Never tune the threshold on test rows.

---

## 9. Division of labor & what's left (Phase 6)

| Member | Assignment | Status |
|---|---|---|
| Sasi | Tier 2 features, orchestrator, XGBoost, SHAP, recall-first rework | ✅ done (PRs #3–#7) |
| Javier | L1 logistic regression → fixed config `logreg_c0001_l1` | ✅ done |
| Russell | Random forest → fixed config `rf_500_d8_l50` | ✅ done |
| Alyssa | Threshold sweep / PR curves → `evaluate.py::select_threshold` now has f1/f_beta/recall@rate/rate@recall | ✅ hooks ready |
| Vienna | Robustness: `--horizon 1/6`, `load_folds(recovery_hours=0)` | ✅ verified running |
| Shao Gjin | PR review, governance, final slide packaging | ⬜ review |

**Remaining integration:** confirm every reported row is a fixed-config run whose tag
matches the report tables; keep `reports/ex_post_comparison.csv` and
`reports/modelling_comparison.csv` regenerated together; final README/deck polish.
Optional analysis: OR-combination of persistence + XGBoost alarms (already tested —
see `reports/combination_results.csv`).

**PR merge order:** #2–#6 merged → #7 (recall-first rework) next. #7 builds on main.

---

## 10. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `load_folds(tier=2)` says features missing | `python data_processing/build_onchain_features.py --update-master` |
| `compare_runs.py` prints nothing | `results/predictions/` is empty — run `train.py` first |
| `feature_importance/shap_analysis.py` can't find models | models live in `results/models/` per trained config — train that config first |
| `--model <name>` unknown | model not registered — add it to `models/__init__.py` |
| `--model-params` rejected | you passed `--tune` too; the two are mutually exclusive |
| Numbers don't match the report tables | wrong tag: check `results/models/*_<year>.json` for the run tag (fixed tags are listed in §5) |
| Inner-val PR-AUC suspiciously high | early stopping on a subset of training rows — see gotcha #1 |
