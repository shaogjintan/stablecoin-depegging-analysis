# Sasi's Phase 3 Focus Plan — DSE4101 Stablecoin Depegging Forecasting

**Goal**: Complete Sasi's assigned Phase 3 work (Tier 2 on-chain change features, modelling orchestration, XGBoost, SHAP interpretability) so the team can produce the pooled Tier 1 vs Tier 2 comparison for the Week 12 report.
**Architecture / Approach**: Extend the frozen data pipeline with the one agreed change (on-chain rate features in `02b`, applied incrementally to `master.parquet` since the gitignored intermediates are missing locally), then build the shared `modelling/` infrastructure (base model contract → `train.py` orchestrator → `evaluate.py` pooled metrics), implement XGBoost on top of it, and finish with SHAP analysis. Everything imports data through the single contract `modelling/prepare_data.py` per Rule 3 of `agent.md`.
**Affected Components & Tech Stack**: `data_processing/02b_build_onchain_features.py`, `data/processed/master.parquet`, `modelling/prepare_data.py`, new `modelling/models/`, `modelling/train.py`, `modelling/evaluate.py`, `modelling/shap_analysis.py`; Python 3.12, pandas 2.2.3, scikit-learn 1.8, xgboost 3.4.1 (shap not yet installed).

---

## Current State (verified 2026-10-13)

- `master.parquet` (345,367 × 32) and `depeg_episodes.parquet` are tracked in git; `modelling/prepare_data.py` runs cleanly and reproduces the documented fold/positive-rate stats.
- Local `main` is **5 commits ahead** of `origin/main` (fork never synced after consolidation); `agent.md` itself is untracked.
- **Broken rerun path discovered**: `03_merge_and_label.py` reads `hourly_stablecoin.parquet`, `hourly_stablecoin_market_features.parquet`, `hourly_stablecoin_onchain_features.parquet` — none of these gitignored intermediates exist locally. So the "rerun 02b then 03" instruction in `agent.md` §6.3 **cannot work as written**. The Tier 2 fix must instead be applied incrementally to `master.parquet` (details in Phase 3).
- `shap` is not installed (needed for Phase 6). `xgboost==3.4.1` is installed; SHAP's `TreeExplainer` compatibility with xgboost 3.x is a flagged risk with a fallback.

---

## User Review & Decisions Required

1. **Change-feature form** (recommended: **log1p differences** — `log1p(x_t) − log1p(x_{t−k})`): defined at zero (no ±inf like `pct_change` on quiet coins), comparable scale across coins whose levels differ 100×, and matches the "changes, ratios, deviations — no raw levels" rule in `prepare_data.py`. Alternative: plain `pct_change` with `inf → NaN` (same convention as `volume_pct_change_*`), which loses PAX/TUSD quiet-hour rows and can explode when the base is 1–2 txs.
2. **Where the Tier 2 fix lives** (recommended: **edit `02b`**, exactly as `agent.md` §6.3 authorizes, with a new `--update-master` mode that patches `master.parquet` in place; the normal path is also updated so future full rebuilds produce the same columns).
3. **Phase order** (recommended: **git sync → Tier 2 features → modelling infra → XGBoost → SHAP**, matching `agent.md`'s checklist order; infra-first after Tier 2 is an alternative if teammates block on it).
4. **Results convention** (recommended: `results/` gitignored + added to `.gitignore`; final summary CSVs and figures committed under `reports/`).

---

## Proposed Changes & Task Breakdown

### Phase 1: Git sync & repo hygiene (~15 min)
- [ ] **Task 1.1: Merge upstream → push fork main**
  - **Action**: `git merge upstream/main` (pulls in exactly 1 upstream commit: `c52707d`, Alyssa's rewrite of `01_fetch_stablecoin_data.py` — touches only 01, no overlap with any file in this plan, so conflict-free), then `git push origin main` (fork is 12 commits behind local main, incl. the whole consolidated pipeline).
  - **Verification**: `git status -sb` shows `main` even with `origin/main`; fork GitHub page shows `4188d37` + `c52707d`.
- [ ] **Task 1.2: Commit `agent.md`**
  - **Action**: coordinate with Shao Gjin (repo governance) → `git add agent.md plans/` on a `chore/docs` branch → PR to main, or direct commit if he approves. Do **not** commit `archive/`, `discussant/`, `reports/` scratch files without a decision on them.
  - **Verification**: PR merged; `agent.md` visible on main.
- [ ] **Task 1.3: Feature branch for Tier 2 work**
  - **Action**: `git checkout -b feature/onchain-rate-features` (Rule 2 naming).
  - **Verification**: `git branch --show-current`.

### Phase 2: Tier 2 on-chain change features (~half day) — Sasi's blocker assignment
- [ ] **Task 2.1: Add `build_change_features()` to `02b_build_onchain_features.py`**
  - **Files**: Modify `data_processing/02b_build_onchain_features.py`.
  - **Action**: Per coin: set `hour` index, reindex onto the full hourly grid (so shift windows are true calendar hours, not gapped series), then compute `log1p` differences at 1h/6h/24h for `tx_count`, `tx_volume`, `active_senders`, `active_receivers` → 12 columns named `{metric}_logdiff_{1h,6h,24h}`. Keep raw levels in output (for reference), document the decision in the docstring, and delete/repurpose the dead `add_pct_change_features()` helper (its outputs are currently dropped).
- [ ] **Task 2.2: Add `--update-master` mode to `02b`**
  - **Files**: Modify `data_processing/02b_build_onchain_features.py`.
  - **Action**: `python data_processing/02b_build_onchain_features.py --update-master` reads `data/processed/master.parquet` (its on-chain columns are already on the full hourly grid), computes the 12 change columns, and writes master back (32 → 44 columns). Assert row count unchanged (345,367) and all existing columns byte-identical before overwriting. This sidesteps the missing intermediates; `01`/`03` stay untouched.
- [ ] **Task 2.3: Update `modelling/prepare_data.py` Tier 2 features**
  - **Files**: Modify `modelling/prepare_data.py`; Modify `README.md` (Tier 2 table + "Tier 2 is not ready" warning).
  - **Action**: `ONCHAIN_FEATURES` = the 12 `*_logdiff_*` columns (raw levels excluded from features).
  - **Verification**: `python modelling/prepare_data.py` still prints identical at-risk/positive counts; Tier 2 NaN share printed per feature (expect roughly ≤ 6–8%, driven by coin starts + TUSD's late Dune start).
- [ ] **Task 2.4: Verify no coin-identity leakage**
  - **Files**: Create `modelling/checks/test_tier2_features.py` (or a `python -c` one-liner in the PR description).
  - **Action**: Train a small RandomForest predicting `coin` from (a) raw levels, (b) the 12 logdiffs; report both accuracies.
  - **Verification**: (a) ≈ 1.0 (the bug); (b) well below that — the point is a documented before/after comparison, not a hard threshold. Also check: no ±inf in the new columns; per-coin 1st/99th percentiles of each new feature are within a comparable range.
- [ ] **Task 2.5: PR + commit updated `master.parquet`**
  - **Action**: Commit `02b` + `prepare_data.py` + README on `feature/onchain-rate-features`; `git add -f data/processed/master.parquet` (force-add, `data/` is gitignored); PR to main; tell the team to `git pull`.
  - **Verification**: PR merged; teammates' `load_folds(tier=2)` returns the 12 new columns.

### Phase 3: Shared modelling infrastructure (~1–2 days) — Sasi's orchestration assignment
- [ ] **Task 3.1: Model contract — `modelling/models/base.py`**
  - **Files**: Create `modelling/models/__init__.py`, `modelling/models/base.py`.
  - **Action**: `BaseModel` with `fit(X_train, y_train, X_val=None, y_val=None)` and `predict_proba(X) -> np.ndarray (n, 2)`; plus a `NeedsImputationMixin`/helper that median-imputes NaN fitted on train only (for Javier's logistic and Russell's RF; XGBoost skips it). Add `get_params()` for the results log. This is the contract teammates implement against — ship it early.
- [ ] **Task 3.2: `modelling/evaluate.py` — pooled OOS metrics**
  - **Files**: Create `modelling/evaluate.py`.
  - **Action**: `pooled_metrics(y_true, y_score)` → PR-AUC (primary, given 0.02–0.57% positives), ROC-AUC, Brier score; `select_threshold(y_inner, p_inner, criterion="f1")` hook (Alyssa extends the sweep/PR-curve analysis); `evaluate_run(preds)` → pooled table + per-coin/episode breakdown + alarm-curve CSV. All input from `load_folds` predictions, never from raw CSVs.
  - **Verification**: unit check on synthetic scores (`y=1` rows ranked above `y=0` → PR-AUC = 1); runs against a dummy run produced by Task 3.3.
- [ ] **Task 3.3: `modelling/train.py` — expanding-window orchestrator**
  - **Files**: Create `modelling/train.py`.
  - **Action**: CLI: `python modelling/train.py --model xgboost --horizon {1,6,24} --tier {1,2} --tag run01 [--tune]`. Per fold: fit on `X_train` (XGBoost uses `X_inner_val` for early stopping), predict `X_test` + `X_inner_val`, save `results/predictions/{model}_{tier}_{h}_{year}.parquet` + `results/models/{...}.json`/joblib, log per-fold inner-val PR-AUC. End: call `evaluate.py` on pooled predictions. `--tune`: small grid on inner_val (PR-AUC selection), params logged.
  - **Verification**: `python modelling/train.py --model xgboost --horizon 24 --tier 1 --tag smoke` completes all 5 folds and prints the pooled table.
- [ ] **Task 3.4: `.gitignore` `results/`**
  - **Files**: Modify `.gitignore`.
  - **Action**: Add `results/` (scratch predictions/models stay local; final tables/figures go in `reports/`).

### Phase 4: XGBoost implementation & tuning (~1 day)
- [ ] **Task 4.1: `modelling/models/xgboost_model.py`**
  - **Files**: Create `modelling/models/xgboost_model.py`.
  - **Action**: `XGBModel(BaseModel)` wrapping `XGBClassifier(tree_method="hist", early_stopping_rounds=100, eval_metric="aucpr", n_jobs=-1, random_state=42)`; NaN-native (no imputation); `scale_pos_weight` default `neg/pos`; early stopping on `X_inner_val` only.
- [ ] **Task 4.2: Inner-val tuning grid**
  - **Action**: light grid per fold: `learning_rate ∈ {0.03, 0.1}`, `max_depth ∈ {3, 5}`, `min_child_weight ∈ {1, 5}`, `scale_pos_weight ∈ {auto, 10, 100}` → ≤ 24 combos × 5 folds; select by inner-val PR-AUC; refit on full train. Record chosen params per fold.
  - **Verification**: pooled OOS PR-AUC vs a naive positive-rate baseline for h=24 Tier 1; params log shows sensible spread (not all at grid edge).
- [ ] **Task 4.3: Full run matrix**
  - **Action**: Run XGBoost for all {1, 6, 24} × {Tier 1, Tier 2} with `--tune`. Baseline comparison script (persistence / constant-prior) included for the report table.
  - **Verification**: 6 result rows with pooled PR-AUC, ROC-AUC, Brier; Tier 2 vs Tier 1 delta noted.

### Phase 5: SHAP interpretability (~1 day)
- [ ] **Task 5.1: Install shap**
  - **Action**: `pip install shap`; add `shap` to `requirements.txt`. Test `shap.TreeExplainer` on an xgboost 3.4.1 model. **Risk/fallback**: if incompatible, either pin `xgboost==2.1.4` for SHAP runs only, or fall back to xgboost's built-in gain importance + partial-dependence plots (documented as such).
- [ ] **Task 5.2: `modelling/shap_analysis.py`**
  - **Files**: Create `modelling/shap_analysis.py`.
  - **Action**: Load per-fold saved models, compute SHAP on each fold's OOS test rows, pool across folds. Outputs: mean |SHAP| ranking CSV + bar plot; beeswarm summary; dependence plots for top-3 features; for Tier 2, group contributions into market vs on-chain shares (stacked bar) — the key evidence for the research question. Save figures to `reports/figures/`.
  - **Verification**: top features for Tier 1 ≈ `peg_deviation`/volatility family (sanity vs literature); on-chain share > 0 for Tier 2; plots open without error.

### Phase 6: Integration & handoff (Week 10–11, with team)
- [ ] **Task 6.1: Pooled comparison table**
  - **Action**: Merge Sasi's XGBoost rows with Javier's logistic and Russell's RF (same `evaluate.py` output format) into the single Tier 1 vs Tier 2 table for the report; include h=1,6,24 and the positive-rate baseline.
- [ ] **Task 6.2: Support Vienna's robustness (interface only)**
  - **Action**: `train.py --horizon 1 --horizon 6` and `prepare_data.load_folds(recovery_hours=0)` already exist as knobs — confirm they run so Vienna's single-coin stress tests reuse the orchestrator.
- [ ] **Task 6.3: Update `agent.md` + README**
  - **Action**: Mark Tier 2 ready, record SHAP findings, refresh division-of-labor status; PR through Shao Gjin.

---

## Timeline (report due ~Week 12, early Nov 2026)

| Week | Sasi delivers |
|---|---|
| 8 (now) | Phase 1–2: fork synced; Tier 2 features merged to main |
| 8–9 | Phase 3: `base.py`, `evaluate.py`, `train.py` shipped (unblocks Javier/Russell/Alyssa) |
| 9 | Phase 4: XGBoost tuned on Tier 1 & Tier 2, all horizons |
| 10 | Phase 5: SHAP; Phase 6.1 comparison table |
| 10–11 | Report writing buffer; Phase 6.2–6.3 |

## Risks
1. **Missing intermediates** (confirmed): mitigated by `--update-master` mode — never rerun `01` (1,200+ HTTP requests, explicitly forbidden in `agent.md`).
2. **shap vs xgboost 3.4.1** compatibility: fallback = pin xgboost 2.1.4 for SHAP only, or built-in importance + PDP.
3. **Imbalance** (0.02% positives at h=1, 49 pooled OOS events): PR-AUC is primary; thresholds tuned on inner_val only; fold-2024/2025 test sets have ~6 positives each, so pooled (not per-year) evaluation per README.
4. **Coordination drift**: teammates writing their own train/eval loops is the fragmentation the consolidated pipeline was built to prevent — Sasi must ship Phase 3 artifacts early and point everyone at the `BaseModel` + `evaluate.py` contract.

## Verification & Acceptance Criteria
- [ ] `git push origin main` done; `feature/onchain-rate-features` PR merged with updated `master.parquet` (44 cols, 345,367 rows).
- [ ] `python modelling/prepare_data.py` reproduces documented stats for both tiers.
- [ ] Coin-ID probe: RF accuracy on raw levels ≈ 1.0 → logdiffs markedly lower (documented).
- [ ] `python modelling/train.py --model xgboost --horizon 24 --tier 1 --tag smoke` runs end-to-end and prints pooled PR-AUC/ROC-AUC/Brier.
- [ ] Full 3×2 matrix run; Tier 2 vs Tier 1 pooled OOS delta documented; SHAP on-chain contribution share quantified.
- [ ] Rollback: `master.parquet` is re-derivable — a backup copy of the 32-col version is saved before `--update-master` runs; `git revert` available for code.
