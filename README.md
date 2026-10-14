# stablecoin-depegging-analysis

Team repository for the DSE4101 Topic 2 project — **stablecoin depegging risk
analysis and early-warning prediction**.

## Workstreams

| Workstream | Folder | Status |
|---|---|---|
| WS1 — Literature replication & empirical proofs | [`workstream-1-literature-replication/`](workstream-1-literature-replication/README.md) | ✅ Complete |
| WS2 — Data merge & depeg target engineering | [`data_processing/`](#data-pipeline-data_processing) | ✅ Complete |
| WS3 — Model development | *(to be added)* | — |

## Quick start (WS1)

```bash
pip install -r requirements.txt
cd workstream-1-literature-replication
python3 scripts/02_replicate_lee.py
```

See `workstream-1-literature-replication/README.md` for the full write-up,
slide assembly guide, and handoff notes.

## Data pipeline (`data_processing/`)

Run every script from the repo root (paths in `config.py` are relative to it).
The final output is the master dataset; see [Master dataset](#master-dataset-dataprocessedmasterparquet) below.

### Pipeline at a glance

**`03_merge_and_label.py` is the script that creates `master.parquet`.**
Every output goes to `data/processed/`, except the unzipped Dune CSVs (`data/raw/dune_onchain/`).

| Step | Script | Reads | Writes |
|---|---|---|---|
| 01 | `01_fetch_stablecoin_data.py` | Internet (DefiLlama, Binance, Bitfinex) | `data/processed/hourly_stablecoin.parquet` |
| 02a | `02a_build_market_features.py` | `hourly_stablecoin.parquet` | `data/processed/hourly_stablecoin_market_features.parquet` |
| 02b | `02b_build_onchain_features.py` | `dune_onchain.zip` (repo root) | `data/raw/dune_onchain/*.csv` (unzipped) and `data/processed/hourly_stablecoin_onchain_features.parquet` |
| **03** | **`03_merge_and_label.py`** | **outputs of 01, 02a, 02b** | **`data/processed/master.parquet`** (+ `.csv`) and `data/processed/depeg_episodes.parquet` (+ `.csv`) |
| 04 | `04_split_train_test.py` | `master.parquet`, `depeg_episodes.parquet` | `data/processed/cv_folds_summary.csv` (fold summary only; no split datasets) |
| — | `verify_embargo_and_episodes.py` | `master.parquet`, `depeg_episodes.parquet` | Nothing (prints checks) |

Other files in `data_processing/`:

- `config.py`: all settings (coins, dates, depeg thresholds, horizons, purge/embargo, CV years).
- `splits.py`: the expanding-window fold function that modelling code imports (see [Cross-validation folds](#cross-validation-folds)).
- `PROPOSAL_PHASE_2_NOTES.md`, `PROPOSAL_UPDATES_WEEK_7.md`: Phase 2 proposal notes. They describe the **old** 3-way split.

### `01_fetch_stablecoin_data.py` — market data

**This step fetches market (price/exchange) data only; no on-chain data.**
On-chain data comes from `dune_onchain.zip` in step `02b`.

Downloads hourly data for the six fiat-backed stablecoins (USDT, USDC, BUSD,
DAI, TUSD, PAX). Price comes from DefiLlama, and trading volume from Binance
(USDC, PAX, TUSD, BUSD) or Bitfinex (USDT, DAI). The two sources are joined
on coin and hour, keeping every price hour.

**Missing prices:** gaps of up to 3 hours are forward-filled with the last
observed price (past information only, no look-ahead); longer gaps stay missing.

**Output:** `data/processed/hourly_stablecoin.parquet`, one row per coin-hour
(~345k rows, Sep 2018 – Dec 2025; BUSD ends Feb 2024 when it was discontinued).

| Column | Description |
|---|---|
| `coin` | Stablecoin ticker |
| `hour` | Hour timestamp (UTC) |
| `price` | USD price (DefiLlama) |
| `volume` | Exchange trading volume for the hour; 0 if no trades, NaN before the exchange's data starts |
| `venue` | Source exchange for the volume (`binance` / `bitfinex`); NaN if no exchange coverage |
| `venue_switch` | True on the first row of each new venue segment |
| `hours_since_venue_start` | Hours since the current venue segment began, so rolling volume changes can skip a data-source switch |

### `02a_build_market_features.py` — market features

Turns step 01's price and volume into per-coin features. All features look
backward only (no look-ahead).

**Input:** `data/processed/hourly_stablecoin.parquet` (from step 01)  
**Output:** `data/processed/hourly_stablecoin_market_features.parquet`, one row per coin-hour

| Column | Description |
|---|---|
| `coin`, `hour` | Keys |
| `peg_deviation`, `peg_deviation_bps` | Price minus $1, in dollars and basis points |
| `return_1h/6h/24h`, `log_return_1h/6h/24h` | Price change over the past 1/6/24 hours |
| `realised_vol_6h`, `realised_vol_24h` | Rolling std of 1h returns (volatility) |
| `trading_volume` | Exchange volume for the hour |
| `volume_pct_change_1h/6h/24h` | Volume change over the past 1/6/24 hours; NaN right after an exchange switch |
| `venue_switch` | Carried over from step 01 |

### `02b_build_onchain_features.py` — on-chain data

Unzips the per-coin Dune Analytics CSVs from `dune_onchain.zip` into
`data/raw/dune_onchain/`, infers the coin from each filename (e.g.
`usdt.csv` → USDT), parses timestamps, and stacks all coins into one table.

**Input:** `dune_onchain.zip` (repo root)  
**Output:** `data/processed/hourly_stablecoin_onchain_features.parquet`, one row per coin-hour

| Column | Description |
|---|---|
| `coin`, `hour` | Keys |
| `tx_count` | Number of on-chain transfers in the hour |
| `tx_volume` | Total tokens transferred in the hour |
| `active_senders`, `active_receivers` | Distinct wallets sending / receiving |
| `{metric}_logdiff_{1h,6h,24h}` | On-chain **change** features: `log1p(x_t) − log1p(x_{t−k})` for `tx_count`, `tx_volume`, `active_senders`, `active_receivers` |
| `total_volume_usd`, `whale_tx_count`, `median_transfer_usd`, `transfer_gini` | Optional; included only if present in the CSV |

> **Note:** the output holds raw hourly on-chain counts plus the 12
> log1p-difference change features. Change features are computed per coin on
> the full hourly grid, so each lag is a true calendar window; the diff is NaN
> whenever either endpoint is unobserved. Raw levels are for reference only —
> the Tier 2 model uses the change features, never the levels.

### `03_merge_and_label.py` — merge, depeg episodes, labels

Combines everything into one master table, finds depeg episodes, and creates
the prediction targets. Thresholds are set in `config.py`.

1. **Merge:** builds a complete hourly grid per coin (no missing hours), then
   left-joins step 01's price data and the outputs of 02a and 02b onto it.
2. **Flag depegged hours:** an hour is depegged if `price < 0.99`
   (`DEPEG_BAND = 0.01`). Downward only — trading above $1 doesn't count.
3. **Group into episodes:** an episode starts after at least 2 consecutive
   depegged hours (`DEPEG_MIN_DURATION_HOURS`). Any further dip within 24 hours
   of the last depegged hour (`DEPEG_GAP_HOURS`) is merged into the same
   episode, so one crash isn't split into many small episodes. 83 episodes
   in total.
4. **Labels:** for each horizon h in {1, 6, 24}, at a normal (non-depegged) hour t,
   `label_h{h} = 1` if a new episode **starts** in the next h hours, else 0.
   The label is NaN when t is already inside an episode, has no price, or the
   next h hours run past the coin's last observation.

**Inputs:** outputs of 01, 02a and 02b  
**Outputs** (in `data/processed/`, each also saved as `.csv`):

- `master.parquet`: one row per coin-hour with every column from 01/02a/02b, plus:

| Column | Description |
|---|---|
| `is_stablecoin` | True for the six fiat stablecoins |
| `outside_band` | True if the hour is depegged (`price < 0.99`) |
| `episode_id` | Episode this hour belongs to (e.g. `USDC_3`); empty if none |
| `in_episode` | True if the hour is inside an episode (including bridged gap hours) |
| `label_h1`, `label_h6`, `label_h24` | Target: 1 if a new episode starts within the next 1/6/24 hours |

- `depeg_episodes.parquet`: one row per episode.

| Column | Description |
|---|---|
| `coin`, `episode_id` | Coin and episode ID |
| `start`, `end` | First and last hour of the episode |
| `n_hours` | Hours actually below $0.99 (bridged gap hours not counted) |
| `min_price`, `max_price` | Lowest and highest price during the episode |

> **Notes:**
> - `venue_switch` appears twice in `master` (`venue_switch_x` / `venue_switch_y`),
>   because both 01 and 02a carry it. They hold the same values.
> - Positive labels are rare: about 0.02% (h=1), 0.15% (h=6) and 0.6% (h=24)
>   of rows.

### `04_split_train_test.py` — cross-validation folds (summary only)

Shows how `master.parquet` will be split for modelling. **It does not create
any training or test datasets.** The split logic lives in `splits.py`, and
modelling code calls it directly to build the folds in memory.

What it does:

1. Loads `master.parquet` and `depeg_episodes.parquet`.
2. Builds one fold per test year Y = 2021 … 2025 (expanding window):
   - **train:** every hour before Y, minus the last 24h (purge)
   - **test:** year Y, minus the first 24h (embargo)
   - **inner_train / inner_val:** the same split one year earlier, for tuning
   - Episodes overlapping a fold boundary are dropped from that fold.
3. Prints rows, episodes and positive labels (h1/h6/h24) for every set in every
   fold, plus the pooled out-of-sample total (49 of 83 episodes).

**Input:** `data/processed/master.parquet`, `data/processed/depeg_episodes.parquet`  
**Output:** `data/processed/cv_folds_summary.csv`, one row per fold × set × horizon

| Column | Description |
|---|---|
| `fold_year` | Test year of the fold |
| `set` | `train`, `test`, `inner_train` or `inner_val` |
| `horizon_h` | 1, 6 or 24 |
| `start`, `end` | First and last hour in the set |
| `rows`, `labelled`, `positives` | Row count, rows with a label, rows with label = 1 |
| `episodes` | Number of depeg episodes with rows in the set |

Use it to check the split; it isn't needed to train models. Fold-by-fold numbers
and example code are in [Cross-validation folds](#cross-validation-folds).

## Master dataset (`data/processed/master.parquet`)

**This is the single source of truth for modelling.** It is created by
`data_processing/03_merge_and_label.py`.

**How to rebuild it** (from the repo root):

```powershell
python data_processing/01_fetch_stablecoin_data.py
python data_processing/02a_build_market_features.py
python data_processing/02b_build_onchain_features.py
python data_processing/03_merge_and_label.py
python data_processing/04_split_train_test.py          # prints CV folds, writes cv_folds_summary.csv
python data_processing/verify_embargo_and_episodes.py  # optional: leakage checks on the folds
```

> If the gitignored intermediates are missing locally, **do not rerun 01**
> (1,200+ HTTP requests). Apply the Tier 2 change features directly:
> `python data_processing/02b_build_onchain_features.py --update-master`

### Overview

- **One row = one coin-hour.**
- **345,367 rows × 44 columns** (32 base columns + 12 on-chain change features).
- **Coins:** USDT, USDC, BUSD, DAI, TUSD, PAX. Sep 2018 – Dec 2025; BUSD ends Feb 2024.
- **Coverage:** price, trading volume and on-chain data are present for ~93–100% of hours for every coin.

| Coin | Start | End | Rows |
|---|---|---|---|
| USDT | 2018-09-01 | 2025-12-31 | 64,296 |
| PAX | 2018-09-26 | 2025-12-31 | 63,696 |
| USDC | 2018-10-05 | 2025-12-31 | 63,480 |
| TUSD | 2019-01-04 | 2025-12-31 | 61,296 |
| DAI | 2019-11-18 | 2025-12-31 | 53,647 |
| BUSD | 2019-09-21 | 2024-02-29 | 38,952 |

### Columns

| Group | Columns | Source |
|---|---|---|
| Keys | `coin`, `hour` (UTC) | — |
| Raw market data | `price`, `venue`, `venue_switch_x`, `hours_since_venue_start` | 01 |
| Market features | `peg_deviation`, `peg_deviation_bps`, `return_1h/6h/24h`, `log_return_1h/6h/24h`, `realised_vol_6h/24h`, `trading_volume`, `volume_pct_change_1h/6h/24h`, `venue_switch_y` | 02a |
| On-chain data | `tx_count`, `tx_volume`, `active_senders`, `active_receivers`, plus `{metric}_logdiff_{1h,6h,24h}` change features | 02b |
| Depeg flags | `is_stablecoin`, `outside_band`, `episode_id`, `in_episode` | 03 |
| **Targets** | `label_h1`, `label_h6`, `label_h24` | 03 |

See the step 01 / 02a / 02b / 03 sections above for what each column means.

### Targets

`label_h{h} = 1` if a new depeg episode (price < $0.99 for ≥ 2 consecutive
hours) **starts** within the next h hours, else 0. NaN for hours already
inside an episode, with no price, or too close to the end of the data.

| Label | Labelled rows | Positives | % positive |
|---|---|---|---|
| `label_h1` | 341,404 | 83 | 0.02% |
| `label_h6` | 341,374 | 495 | 0.15% |
| `label_h24` | 341,266 | 1,964 | 0.58% |

There are 83 depeg episodes in total (`depeg_episodes.parquet`): USDT 13,
USDC 13, BUSD 13, DAI 10, TUSD 16, PAX 18.

### Cross-validation folds

Expanding window by year (replaces the old single 2018–22 / 2023 / 2024–25 split,
which left only 2 episodes in test). For each year Y from 2021 to 2025:

- **train** = every hour before Y, minus a 24h purge at the end (labels look ahead up to 24h)
- **test** = year Y, minus a 24h embargo at the start (features look back up to 24h)
- **inner_train / inner_val** = the same split one year earlier (train before Y−1,
  validate on Y−1), for tuning hyperparameters and the decision threshold
  without touching year Y
- Episodes overlapping a boundary window are dropped from every set in that fold.

Pool the test predictions across folds: **49 of 83 episodes are scored out of
sample** (vs 2 before), covering all six coins.

The folds are **not saved as files**. Modelling code builds them in memory from
`master.parquet` with `splits.py` (run from the repo root):

```python
import sys; sys.path.insert(0, "data_processing")
from splits import load_master, expanding_window_folds

master, episodes = load_master()
for fold in expanding_window_folds(master, episodes):
    year = fold["year"]
    train, test = fold["train"], fold["test"]                 # fit on train, predict test
    inner_train, inner_val = fold["inner_train"], fold["inner_val"]  # for tuning
    # for model-ready X / y with the at-risk filter applied, use modelling/prepare_data.py
```

| Test year | Train rows | Test rows | Test episodes | Test positives (h1 / h6 / h24) |
|---|---|---|---|---|
| 2021 | 97,962 | 52,416 | 23 | 23 / 138 / 552 |
| 2022 | 150,943 | 52,416 | 11 | 11 / 66 / 264 |
| 2023 | 203,503 | 52,416 | 13 | 13 / 78 / 312 |
| 2024 | 256,063 | 45,216 | 1 | 1 / 6 / 24 |
| 2025 | 301,447 | 43,680 | 1 | 1 / 6 / 24 |
| **Pooled** | — | **246,144** | **49** | **49 / 294 / 1,176** |

The 2021 fold trains on only 2018–2020, so early folds have the least training data.
Settings: `OOS_START_YEAR`, `OOS_END_YEAR`, `PURGE_HOURS`, `EMBARGO_HOURS` in `config.py`.

### Known quirks

- `venue_switch` appears twice (`venue_switch_x` / `venue_switch_y`) with the same values.
- No trade-count or taker-buy features: Bitfinex (USDT, DAI) doesn't report them, so they'd be 100% NaN for those coins and a pooled tree model would learn the missingness as a coin identifier. Dropped in 02a/03.
- On-chain **raw levels** are kept in `master.parquet` for reference only; the Tier 2 model uses the 12 `*_logdiff_*` change features instead, because levels differ by orders of magnitude between coins and would act as coin identifiers (see [Modelling](#modelling-modelling)).
- 2024 and 2025 each contain a single TUSD episode, so evaluate on the pooled out-of-sample predictions rather than per year.

## Modelling (`modelling/`)

### `prepare_data.py` — model-ready data per fold

Builds the CV folds (via `data_processing/splits.py`), removes rows outside the
**at-risk sample**, and selects the feature columns. It does not change `master.parquet`.

**At-risk sample.** The label is the onset of a *new* episode, and a coin can
only start one while it is on peg. An hour stays in the sample only if:

1. it is **not inside an ongoing episode** (`in_episode == False`);
2. it is **not in the recovery window**: the 24h after an episode's last hour
   below $0.99 (`RECOVERY_HOURS = DEPEG_GAP_HOURS = 24`, as on slide 10). Any dip
   in that window merges into the same episode, so the coin re-enters the risk
   set only after 24h back above $0.99;
3. it **has a label** (`label_h{h}` not NaN).

This removes about 1.8% of rows (3,312 in-episode hours, 1,992 recovery-window
hours, plus unlabelled hours). Set `RECOVERY_HOURS = 0` to keep recovery hours
as a robustness check.

**Features.** Only changes, ratios and deviations — no raw levels, which
differ by orders of magnitude between coins and would act as coin identifiers.

| Tier | Features |
|---|---|
| 1 — market only | `peg_deviation`, `log_return_1h/6h/24h`, `realised_vol_6h/24h`, `volume_pct_change_1h/6h/24h` |
| 2 — market + on-chain | Tier 1 + `{metric}_logdiff_{1h,6h,24h}` for `tx_count`, `tx_volume`, `active_senders`, `active_receivers` (12 columns) |

Tier 2 uses **change** features (log1p differences over true calendar hours),
not raw on-chain levels. If `prepare_data.py` reports them missing from
`master.parquet`, apply the patch:

```bash
python data_processing/02b_build_onchain_features.py --update-master
```

Missing feature values stay NaN (volume changes ~11%, on-chain changes ~5%).
XGBoost handles NaN natively; for Lasso and Random Forest, impute inside the
model code, fitted on the training set only.

**Usage** (from the repo root):

```python
from modelling.prepare_data import load_folds

for fold in load_folds(horizon=24, tier=1):
    model.fit(fold["X_train"], fold["y_train"])
    p = model.predict_proba(fold["X_test"])[:, 1]
    # fold["X_inner_train"], fold["y_inner_train"], fold["X_inner_val"], fold["y_inner_val"]: for tuning
    # fold["test_meta"]: coin, hour, episode_id of each test row, to pool predictions
```

Run `python modelling/prepare_data.py` to print the at-risk sample size and
positives per fold for each horizon.

| Horizon | At-risk rows | Positives | Pooled OOS positives (2021–25) |
|---|---|---|---|
| h=1 | 339,414 | 83 | 49 |
| h=6 | 339,384 | 495 | 294 |
| h=24 | 339,276 | 1,930 | 1,153 |

### `models/`, `train.py`, `evaluate.py` — model contract & orchestrator

**Rule: every model implements `modelling/models/base.py`'s `BaseModel` and is
registered in `modelling/models/__init__.py`.** Teammates implement against
this contract — no personal training loops.

`BaseModel` requires:

```python
class MyModel(BaseModel):
    name = "my_model"                # CLI name in train.py
    needs_imputation = True          # False if NaN-native (e.g. XGBoost)
    param_grid = [{"C": 0.1}, ...]   # combos tried by --tune

    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    def predict_proba(self, X) -> np.ndarray: ...   # shape (n, 2), col 1 = P(y=1)
    def get_params(self) -> dict: ...               # for the results log
```

If `needs_imputation` is `True`, `train.py` median-imputes each column using
the training fold only (`MedianImputer`). Registered so far: `baseline`
(constant positive-rate prior; the report-table floor).

**Train** (from the repo root):

```bash
python modelling/train.py --model baseline --horizon 24 --tier 1 --tag smoke
python modelling/train.py --model xgboost   --horizon 24 --tier 2 --tag run01 --tune
```

Per fold it fits on `X_train` (early stopping on `X_inner_val` if supported),
saves predictions to `results/predictions/` and model artifacts
(`joblib` + params JSON) to `results/models/`, logs inner-val PR-AUC, then
pools the OOS test predictions and runs `evaluate.py`: pooled PR-AUC /
ROC-AUC / Brier, per-coin and per-episode tables, and the alarm-curve CSV
(`results/evaluation/`). `results/` is gitignored — final tables and figures
go under `reports/`.

**Evaluate any pooled run** with `python modelling/evaluate.py` (self-test)
or import `evaluate_run` / `pooled_metrics` / `select_threshold` (thresholds
tuned on `inner_val` only — Alyssa extends the sweep here).

### Recall-first early-warning evaluation (headline metric)

Per the group decision (2026-10-13), the goal is to **catch every depeg
episode** (minimize false negatives) and tolerate false alarms. The headline
metric is therefore **episode-level recall vs the hourly alarm rate**, not
PR-AUC (which stays secondary).

```bash
# tune for recall at a 1% alarm-rate budget (instead of PR-AUC):
python modelling/train.py --model xgboost --horizon 24 --tier 2 --tag rec \
       --tune --tune-metric recall_at_rate --max-alarm-rate 0.01

# operating table: episode recall at 0.5/1/2/5% alarm budgets and the alarm
# rate needed for 90/95/99% episode recall, plus per-episode caught/missed CSVs:
python modelling/alarm_report.py --save reports/recall_operating_points.csv --alarm-tables
```

Key result so far: peg-stress persistence catches **81.6% of the 49 OOS
episodes at a 1% alarm rate (h=24)**; recall-tuned XGBoost reaches 38.8–40.8%
(same budget) and needs a 23.8% alarm rate for 90% episode recall. Full table:
`reports/recall_operating_points.csv`.
