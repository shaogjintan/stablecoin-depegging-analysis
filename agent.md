# Agent Handoff: DSE4101 Stablecoin Depegging Forecasting

> **Purpose:** This is the master context file for **any human or AI agent** working on this repo. If you (or your AI) are about to do anything here, read this file first — or paste it into your agent and tell it to follow the rules in Section 3.
>
> **Last updated:** 2026-10-13 — Phases 1–5 complete (Tier 2 features, modelling infrastructure, tuned XGBoost, SHAP). See the roadmap in `plans/2026-10-13-sasi-phase3-roadmap.md`.

---

## 1. Project in one paragraph

NUS DSE4101 capstone (Digital Currencies Group 1). **Research question:** can information available at time *t* predict whether a fiat-backed stablecoin will depeg within the next *h* ∈ {1, 6, 24} hours — and does adding **on-chain** data improve forecasts beyond **market** data alone?

- **Tier 1** = market only (9 features). **Tier 2** = market + 12 on-chain change features.
- **Data:** hourly, 6 coins (USDT, USDC, BUSD, DAI, TUSD, PAX), Sep 2018 – Dec 2025. Canonical dataset: `data/processed/master.parquet` (**345,367 rows × 44 columns**).
- **Evaluation:** expanding-window CV, 5 folds (test years 2021–2025), 24h purge + 24h embargo, episode-boundary-aware. **Pooled out-of-sample PR-AUC is the primary metric** (positives are 0.02–0.57% of rows).
- **Status:** pipeline frozen; Tier 2 ready; shared modelling infra + tuned XGBoost + SHAP shipped. Report due ~Week 12 (early Nov 2026).

---

## 2. Hard rules (break these and your PR will be rejected)

| # | Rule |
|---|---|
| R1 | `data_processing/` is **FROZEN**. The only sanctioned change (Tier 2 change features in `02b`) is **already done**. |
| R2 | **NEVER run `01_fetch_stablecoin_data.py`** (1,200+ HTTP requests). `master.parquet` is committed and current. |
| R3 | All modelling data access goes through `modelling/prepare_data.py` → `load_folds()`. Never read raw CSVs/parquets inside model code. |
| R4 | Raw on-chain levels (`tx_count`, `tx_volume`, `active_senders`, `active_receivers`) are **NOT features** — they identify the coin (RF probe: 0.70 acc). Use the `*_logdiff_*` columns instead. |
| R5 | No personal folders or duplicate pipelines. One feature branch per task (`feature/<name>`), PR into `main`, Shao Gjin reviews. |
| R6 | `results/` is gitignored scratch (predictions, models). Final tables/figures go in `reports/`. |

---

## 3. Repository map (what is where, what is current)

```
data/
  processed/master.parquet          # canonical 44-col dataset (source of truth, committed)
  processed/depeg_episodes.parquet  # 83 validated episodes manifest
  processed/master_backup_32col.parquet  # pre-Tier-2 backup (gitignored, local only)
data_processing/                    # FROZEN pipeline (01 fetch, 02a market, 02b on-chain, 03 merge+label, 04 splits, splits.py, config.py, verify_embargo_and_episodes.py)
modelling/                          # ACTIVE workspace
  prepare_data.py                   # THE data contract: load_folds(horizon, tier, recovery_hours)
  models/base.py                    # BaseModel + MedianImputer (model contract)
  models/baseline.py                # "baseline" (constant prior), "persistence" (peg stress)
  models/xgboost_model.py           # "xgboost" (tuned, NaN-native)
  models/__init__.py                # MODEL_REGISTRY — register new models here
  train.py                          # orchestrator (folds, tuning, results)
  evaluate.py                       # pooled metrics, thresholds, alarm curves
  compare_runs.py                   # Tier 1 vs Tier 2 comparison table
  shap_analysis.py                  # SHAP values, rankings, beeswarms, group shares
  checks/test_tier2_features.py     # coin-identity leakage probe + hygiene checks
reports/
  modelling_comparison.csv          # pooled OOS results table (Phase 4)
  figures/                          # SHAP outputs (Phase 5)
results/                            # gitignored: predictions/, models/, evaluation/
agent.md                            # this file
plans/2026-10-13-sasi-phase3-roadmap.md
```

---

## 4. Data contract (the ONLY way to get modelling data)

```python
from modelling.prepare_data import load_folds

for fold in load_folds(horizon=24, tier=2):        # tier: 1 or 2
    X_train, y_train = fold["X_train"], fold["y_train"]
    X_test,  y_test  = fold["X_test"],  fold["y_test"]
    test_meta        = fold["test_meta"]           # coin, hour, episode_id
    X_inner_train, y_inner_train = fold["X_inner_train"], fold["y_inner_train"]  # nested tuning
    X_inner_val,   y_inner_val   = fold["X_inner_val"],   fold["y_inner_val"]
    inner_val_meta  = fold["inner_val_meta"]
```

- Features per tier: `prepare_data.MARKET_FEATURES` (9) and `prepare_data.ONCHAIN_FEATURES` (12 `*_logdiff_*` columns).
- **At-risk sample:** onset-only labels; hours inside an episode and the 24h recovery window are already dropped.
- **NaN = missing information, kept as NaN.** XGBoost handles it natively; logistic/RF set `needs_imputation=True` (train-only median imputation, done by `train.py`).
- Run `python modelling/prepare_data.py` to print sample stats (must reproduce: 83 / 495 / 1,930 positives at h=1/6/24).

---

## 5. Model contract (how to add YOUR model)

Subclass `modelling/models/base.py::BaseModel`:

```python
class MyModel(BaseModel):
    name = "my_model"                 # CLI name for train.py
    needs_imputation = True           # False if NaN-native (XGBoost)
    param_grid = [{"C": 0.1}, ...]    # combos tried by --tune
    supports_early_stopping = False   # True if fit() uses X_val for stopping

    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    def predict_proba(self, X) -> np.ndarray: ...   # shape (n, 2), col 1 = P(y=1)
    def get_params(self) -> dict: ...               # for the results log
```

Then **register it** in `modelling/models/__init__.py` (`MODEL_REGISTRY["my_model"] = MyModel`). Done — `train.py` handles folds, imputation, tuning, saving, and evaluation for you. Example: `modelling/models/xgboost_model.py`.

---

## 6. Exact commands (run from the repo root)

```bash
python modelling/prepare_data.py                                   # data sanity check
python modelling/train.py --model xgboost --horizon 24 --tier 1 --tag run01 --tune
python modelling/train.py --model xgboost --horizon 24 --tier 1 --tag rec \
       --tune --tune-metric recall_at_rate --max-alarm-rate 0.01   # recall-first tuning
python modelling/train.py --model baseline --horizon 24 --tier 1 --tag base    # constant prior
python modelling/train.py --model persistence --horizon 24 --tier 1 --tag pers # peg-stress baseline
python modelling/compare_runs.py [--save reports/modelling_comparison.csv]    # PR-AUC table (secondary)
python modelling/alarm_report.py --save reports/recall_operating_points.csv \
       --alarm-tables                                               # recall-first table (HEADLINE)
python modelling/shap_analysis.py --model xgboost --horizon 24 --tier 2
python modelling/evaluate.py                                       # self-test
python modelling/checks/test_tier2_features.py                     # leakage probe
# If master.parquet ever lacks the *_logdiff_* columns:
python data_processing/02b_build_onchain_features.py --update-master
```

Outputs: per-fold predictions → `results/predictions/`, models + params JSON → `results/models/`, pooled OOS report + alarm curve → `results/evaluation/`, SHAP figures → `reports/figures/`, recall-first operating table → `reports/recall_operating_points.csv`, per-episode caught/missed tables → `reports/alarm_tables/`.

---

## 7. Findings so far (build on these — don't redo the work)

**Headline objective (group decision 2026-10-13): early-warning — catch every
depeg episode (minimize false negatives / missed depegs), false alarms are
acceptable.** The headline metric is therefore **episode-level recall vs the
hourly alarm rate** (`modelling/alarm_report.py` →
`reports/recall_operating_points.csv`). Pooled PR-AUC is kept only as a
secondary ranking number.

**Episode recall at a 1% alarm-rate budget** (49 OOS episodes, pooled 2021–25):

| Model | h=1 | h=6 | h=24 |
|---|---|---|---|
| Constant prior | — | — | — |
| Persistence (−peg_deviation) | 0.27 | 0.53 | **0.82** |
| XGBoost recall-tuned Tier 1 | 0.14 | 0.16 | 0.39 |
| XGBoost recall-tuned Tier 2 | 0.14 | 0.22 | 0.41 |

Alarm rate needed for 90% episode recall: persistence h=24 needs **2.4%**;
XGBoost Tier 2 h=24 (recall-tuned) needs **23.8%** (down from 71.7% for the
old PR-AUC-tuned model — the recall objective helped a lot).

- Peg-stress persistence is the strongest single early-warning signal at h=6/24.
- Recall-tuning improved XGBoost up to ~19x at h=24 (tier 1: 2% -> 39% @ 1% rate);
  it is a credible complement, not yet a replacement for persistence.
- At h=1 no model is strong (49 events; catching >90% costs ~80-90% alarm rate).
- **OR-combination tested** (`modelling/combine_alarms.py` →
  `reports/combination_results.csv`): unioning persistence + XGBoost alarms
  at a shared budget helps only marginally — +1 episode at h=24 (2% budget:
  0.8367 vs 0.8163) and +1–2 episodes at h=6; at the 1% budget XGBoost adds
  nothing. Persistence already catches most of what XGBoost catches at low
  rates, so the simple peg-stress rule remains the recommended headline alarm;
  XGBoost is a small add-on at 2–5% budgets (and relatively more useful at h=1,
  +0.10 at 5%).

**Pooled OOS PR-AUC** (secondary; full table in `reports/modelling_comparison.csv`):

| Model | h=1 | h=6 | h=24 |
|---|---|---|---|
| Constant prior (floor) | 0.0003 | 0.0018 | 0.0069 |
| Persistence (−peg_deviation) | 0.0025 | 0.0097 | **0.0220** |
| XGBoost Tier 1 | 0.0017 | **0.0124** | 0.0093 |
| XGBoost Tier 2 | 0.0011 | 0.0037 | 0.0102 |

- XGBoost beats both baselines at h=6 on PR-AUC; at h=24 persistence wins.
- **Tier 2 − Tier 1 ≈ 0** on PR-AUC; on-chain change features do not clearly
  improve the ranking metric (see SHAP below).
- **SHAP** (pooled OOS, h=6/24): `realised_vol_24h` is 33–45% of |SHAP|, then
  `peg_deviation`, `realised_vol_6h`, volume changes — matches the literature.
  **On-chain share: 6.1% (h=24), 7.0% (h=6)**; top on-chain:
  `tx_volume_logdiff_24h`, `active_receivers_logdiff_24h`, `active_senders_logdiff_24h`.
- **Brier caution:** `scale_pos_weight` inflates XGBoost probabilities. Compare
  models on PR-AUC / ROC-AUC or the recall-first curve, not Brier, unless
  probabilities are recalibrated.

---

## 8. Gotchas your AI WILL hit (and how they were solved)

1. **`X_inner_val` is a subset of `X_train`** in these folds. Early stopping on it while fitting on `X_train` evaluates on training rows (gave inner-val PR-AUC = 1.0). Correct nested pattern, already implemented in `train.py --tune`: candidates fit on `X_inner_train` → stop/select on `X_inner_val` → winner refit on full `X_train` with `n_estimators = best_iteration`.
2. **Onset rows (label = 1) have `episode_id = NaN`** — the id marks in-episode hours. To map a flagged hour to its target episode, use `depeg_episodes.parquet` with `start ∈ (t, t+h]` (`evaluate.py` does this).
3. **The zip's Dune CSVs have no `pct_change` columns** — `02b` was fixed to the real schema (`hour, tx_count, total_volume, active_senders, active_receivers`).
4. **`venue_switch` appears twice** (`venue_switch_x` / `venue_switch_y`) — `prepare_data.py` ignores both.
5. **Raw levels leak coin identity** (RF accuracy 0.70 levels vs 0.41 logdiffs) — the reason Tier 2 uses change features.
6. **2024 and 2025 test sets have ~6 positives each** — always evaluate on pooled OOS predictions, never per-year.
7. **Git/GitHub:** `sasssyboujee/DSE4101-Topic-2-Project` is *not* a registered fork — push feature branches **directly to `shaogjintan/stablecoin-depegging-analysis`** and open PRs there. **Create feature branches from `upstream/main`** (`git fetch upstream && git checkout upstream/main -b feature/...`), never from a stale local `main` — branching from local main re-introduced old milestone folders into upstream (2026-10-13 incident, cleaned up right after).
8. **Large file:** `master.parquet` (68 MB) is committed to git despite GitHub's 50 MB recommendation — keep committing it (the team's convention); don't add LFS mid-project.
9. **Recall-first terminology:** we minimize **false negatives** (missed depegs) and accept false positives (false alarms). "True negatives" are correct non-alarms — do NOT reduce those. Headline = episode-level recall vs alarm rate; PR-AUC is secondary.
10. **Thresholds are tuned on inner_val only** (`select_threshold` in evaluate.py), then applied to OOS test. Never tune the threshold on test rows.

---

## 9. Division of labor & what's left (Phase 6)

| Member | Assignment | Status |
|---|---|---|
| Sasi | Tier 2 features, orchestrator, XGBoost, SHAP, recall-first rework | ✅ done (PRs #3–#7) |
| Javier | L1 logistic regression → register `"logistic"` (tune with `--tune-metric recall_at_rate`) | ⬜ implement |
| Russell | Random forest → register `"random_forest"` (same) | ⬜ implement |
| Alyssa | Threshold sweep / PR curves → `evaluate.py::select_threshold` now has f1/f_beta/recall@rate/rate@recall | ✅ hooks ready |
| Vienna | Robustness: `--horizon 1/6`, `load_folds(recovery_hours=0)` | ✅ verified running |
| Shao Gjin | PR review, governance, final slide packaging | ⬜ review |

**Remaining integration (Phase 6):** run Javier's and Russell's models through
`train.py` with the recall objective and add their rows to
`reports/recall_operating_points.csv`; optionally test the OR-combination of
persistence + XGBoost alarms; final README/`agent.md` polish.

**PR merge order:** #2–#6 merged → #7 (recall-first rework) next. #7 builds on main.

---

## 10. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `load_folds(tier=2)` says features missing | `python data_processing/02b_build_onchain_features.py --update-master` |
| `compare_runs.py` prints nothing | `results/predictions/` is empty — run `train.py` first |
| `shap_analysis.py` can't find models | models live in `results/models/` per trained config — train that config first |
| `--model <name>` unknown | model not registered — add it to `models/__init__.py` |
| Inner-val PR-AUC suspiciously high | you're early-stopping on training rows — see gotcha #1 |
