# Agent Handoff: DSE4101 Stablecoin Depegging Forecasting

> **Purpose:** This document is the master context transfer for any team member or agent working on this project. It captures the project state, verified architectural decisions, folder structure, known caveats, and immediate modeling roadmap.

---

## 1. Project Identity & Status Summary

| Field | Value |
|---|---|
| **Course** | DSE4101 — Capstone Project in Data Science and Economics I (NUS) |
| **Group** | Digital Currencies Group 1 |
| **Current Stage** | **Week 8 (Phase 3: Model Development & Evaluation)** |
| **Final Report Due** | ~Week 12 (Early November 2026) |
| **Upstream Repo** | `shaogjintan/stablecoin-depegging-analysis` (`main` branch) |
| **Sasi Fork** | `sasssyboujee/DSE4101-Topic-2-Project` |
| **Local Workspace** | `/Users/sasi/Documents/DSE4101 Topic 2 Project/` |

### Core Research Question
> *"Can information available at time $t$ predict whether a fiat-backed stablecoin will depeg within the next $h \in \{1, 6, 24\}$ hours, and does adding on-chain transaction data improve forecasting accuracy beyond market data alone?"*

This is strictly a **predictive forecasting task** evaluated via an **information-set comparison** (Tier 1: Market Only vs. Tier 2: Market + On-Chain).

---

## 2. Recent Major Breakthrough: Pipeline Consolidation & Discussant Fixes

In Week 8, the team successfully consolidated fragmented pipelines from previous milestones into a unified architecture. All discussant critique items have been addressed:

| Item | Old Implementation | New Consolidated Implementation | Status |
|---|---|---|---|
| **Pipeline Location** | Scattered across `proposal_phase_2/`, `midterm_presentation_folder (sasi)/`, and personal folders | Single canonical folder: `data_processing/` | ✅ Complete |
| **Missing Price Handling** | `interpolate(method="time")` (peeking at future prices, lookahead leakage) | `ffill(limit=3)` (strict historical forward-fill up to 3h; longer gaps remain NaN) | ✅ Fixed in `01` |
| **Exchange Features** | Included `trades` and `taker_buy_base_volume` (NaN for Bitfinex coins USDT/DAI) | Dropped completely; prevents tree models from using missingness as a coin identifier | ✅ Fixed in `02a`/`03` |
| **Validation / Test Split** | Single static split (2018–22 train / 2023 val / 2024–25 test); only 2 test episodes | **Expanding Window Walk-Forward CV (2021–2025)**: 5 folds, 24h purge, 24h embargo, episode-aware boundary drops | ✅ Fixed in `splits.py` & `04` |
| **Out-of-Sample Coverage** | Only 2 test episodes (both TUSD) | **49 out of 83 episodes scored out of sample** across all 6 fiat coins | ✅ Verified |
| **At-Risk Sample** | Trained on all rows including active depegs | **At-Risk Filtering**: drops ongoing episodes (`in_episode`) + 24h post-episode recovery window; onset-only labeling | ✅ Fixed in `modelling/prepare_data.py` |
| **Master Dataset** | Multiple conflicting CSVs/parquets | Single source of truth: `data/processed/master.parquet` (345,367 rows × 32 cols) | ✅ Generated & Committed |

---

## 3. Current Repository Architecture

```
/Users/sasi/Documents/DSE4101 Topic 2 Project/
├── README.md                           # Master repo guide, schema, and usage instructions
├── requirements.txt                    # Project dependencies
├── dune_onchain.zip                    # Per-coin Dune Analytics raw exports
│
├── data/
│   ├── raw/
│   │   └── dune_onchain/               # Unzipped per-coin CSVs (USDT, USDC, DAI, BUSD, TUSD, PAX)
│   └── processed/
│       ├── master.parquet              # Canonical dataset (345,367 coin-hours, 32 columns)
│       ├── depeg_episodes.parquet      # 83 validated depeg episodes manifest
│       └── cv_folds_summary.csv        # Expanding window summary across all 5 folds
│
├── data_processing/                    # CANONICAL DATA PIPELINE (FROZEN)
│   ├── config.py                       # Global parameters (thresholds, horizons, coins, purge/embargo)
│   ├── 01_fetch_stablecoin_data.py     # DefiLlama prices + Binance/Bitfinex OHLCV (ffill limit=3)
│   ├── 02a_build_market_features.py    # Log returns, peg deviation, realised vol, volume pct changes
│   ├── 02b_build_onchain_features.py   # Dune on-chain aggregations
│   ├── 03_merge_and_label.py           # Hourly grid, depeg episode clustering, onset targets (h=1,6,24)
│   ├── 04_split_train_test.py          # CV fold summary generator & statistics
│   ├── splits.py                       # Expanding window generator (imported by modeling)
│   └── verify_embargo_and_episodes.py  # ACF decay verification & leakage audit
│
├── modelling/                          # ACTIVE DEVELOPMENT WORKSPACE (PHASE 3)
│   ├── __init__.py
│   ├── prepare_data.py                 # In-memory fold builder, at-risk filtering & Tier feature selection
│   ├── models/                         # [TO BUILD] Model implementations (Logistic, RF, XGBoost)
│   ├── train.py                        # [TO BUILD] Expanding window training orchestrator
│   └── evaluate.py                     # [TO BUILD] Pooled OOS PR-AUC, ROC-AUC, Brier score, alarm curves
│
└── archive/                            # Archived historical code & presentation decks
```

---

## 4. Verification & Audit Findings

The consolidated code and dataset were audited thoroughly. Key findings:

### 1. Leakage & Boundary Integrity (PASSED)
Running `python data_processing/verify_embargo_and_episodes.py` confirms:
- **Autocorrelation Decay:** Autocorrelations for 24h rolling price returns drop to $\le 0.03$ by lag 18h and turn negative by lag 24h across USDT, USDC, and DAI. This empirically validates the **24-hour embargo**.
- **Train $\to$ Test Gap:** Every fold enforces a gap of at least 49 hours between training observations and test observations ($24\text{h purge} + 24\text{h embargo} + 1\text{h step}$).
- **Shared Episodes:** Exactly **0** episodes cross train and test boundaries in any fold. Boundary-straddling episodes (e.g. `DAI_3` at the 2021 boundary) are explicitly dropped.

### 2. At-Risk Sample & Class Balance (PASSED)
Running `python modelling/prepare_data.py` confirms:
- **Total Master Rows:** 345,367 coin-hours.
- **In-Episode Hours Dropped:** 3,312 hours.
- **Recovery Window Hours Dropped:** 1,992 hours (the 24 hours post-depeg where prices bounce before true stabilization).
- **At-Risk Retention:** ~98.2% (339,276 rows for $h=24$).
- **Positive Rates:**
  - $h=1$: 83 positives total (49 pooled OOS) $\approx 0.02\%$
  - $h=6$: 495 positives total (294 pooled OOS) $\approx 0.15\%$
  - $h=24$: 1,930 positives total (1,153 pooled OOS) $\approx 0.57\%$

### 3. Open Issues & Caveats (ACTION REQUIRED)
- ⚠️ **Tier 2 On-Chain Features Need Change Rates:** In `02b_build_onchain_features.py`, only raw hourly levels (`tx_count`, `tx_volume`, `active_senders`, `active_receivers`) are saved. As noted in `modelling/prepare_data.py`, median hourly `tx_count` for PAX is ~12 vs USDT ~5,400. Using raw levels will allow tree models to memorize coin identities. **Fix:** Update `02b` to compute 1h, 6h, and 24h percentage or log differences before running Tier 2 models.
- ℹ️ **Intermediate Files Not in Git:** Intermediate files (`hourly_stablecoin.parquet`, etc.) are ignored by `.gitignore`. `master.parquet` is already built and committed. **Team members DO NOT need to rerun `01_fetch_stablecoin_data.py` unless they wish to re-fetch all raw exchange data from scratch.**
- ℹ️ **Column Quirk:** `master.parquet` contains both `venue_switch_x` and `venue_switch_y`. `modelling/prepare_data.py` ignores both, so downstream models are unaffected.

---

## 5. Prevention Strategy: How to Prevent Future Pipeline Fragmentation

The team encountered parallel, duplicate pipelines because multiple members wrote standalone data scripts in separate folders for past milestones. To ensure smooth execution for Phase 3:

### Rule 1: Strict Architectural Separation
- **`data_processing/` is FROZEN.** No one edits data generation scripts unless implementing the agreed Tier 2 rate changes.
- **`modelling/` is the SINGLE active folder.** All model training, hyperparameter tuning, cross-validation, and evaluation code must be placed here.

### Rule 2: Git Feature Branches (No More Personal Scratch Folders)
- Never create personal folders in the repository (e.g. `sasi_scripts/`, `alyssa_models/`).
- Use dedicated git branches:
  - `git checkout -b feature/logistic-baseline`
  - `git checkout -b feature/tree-models`
  - `git checkout -b feature/onchain-rate-features`
- Merge back to `main` via GitHub Pull Requests after testing locally.

### Rule 3: Single Import Contract for Modeling
Every modeling script must import data directly via `modelling/prepare_data.py`:
```python
from modelling.prepare_data import load_folds

for fold in load_folds(horizon=24, tier=1):
    X_train, y_train = fold["X_train"], fold["y_train"]
    X_test, y_test = fold["X_test"], fold["y_test"]
    X_val, y_val = fold["X_inner_val"], fold["y_inner_val"]
    test_meta = fold["test_meta"]
```
This guarantees identical sample definitions, features, and leakage protections across all models.

---

## 6. Actionable Checklist for Sasi

### Immediate Technical Actions
1. **Sync Git Fork:**
   ```bash
   cd "/Users/sasi/Documents/DSE4101 Topic 2 Project"
   git push origin main
   ```
2. **Do NOT Rerun Step 01:**
   - `master.parquet` is already updated and verified.
   - Do not waste time making 1,200+ HTTP requests to DefiLlama/Binance/Bitfinex.
3. **Enhance `02b_build_onchain_features.py` for Tier 2:**
   - Add rolling 1h, 6h, 24h percentage changes for on-chain metrics.
   - Run `python data_processing/02b_build_onchain_features.py` and `python data_processing/03_merge_and_label.py` (takes <15 seconds locally).
4. **Implement Modeling Pipeline in `modelling/`:**
   - Build `modelling/models/baseline.py`: Baseline persistence / L1 Regularized Logistic Regression.
   - Build `modelling/models/trees.py`: Random Forest & XGBoost.
   - Build `modelling/evaluate.py`: Evaluate pooled out-of-sample PR-AUC, ROC-AUC, Brier score, and threshold alarm performance.

---

## 7. Division of Labor for Phase 3 (Modeling)

| Team Member | Assigned Responsibility | Target Script / Output |
|---|---|---|
| **Sasi** | Tier 2 feature diffs, modeling pipeline orchestration, XGBoost implementation, SHAP interpretability | `02b`, `modelling/models/xgboost_model.py`, `modelling/shap_analysis.py` |
| **Javier** | Baseline models: L1 Logistic Regression, class-weight tuning, coefficients | `modelling/models/logistic_model.py` |
| **Russell** | Random Forest implementation, max depth/min samples tuning | `modelling/models/random_forest.py` |
| **Alyssa** | Threshold tuning on `inner_val`, Precision-Recall curve analysis | `modelling/evaluate.py` |
| **Vienna** | Robustness checks ($h=1, 6$, recovery window variations, single-coin stress test) | `modelling/robustness.py` |
| **Shao Gjin** | Git review, repo governance, pooled OOS results compilation | PR reviews & slide packaging |
