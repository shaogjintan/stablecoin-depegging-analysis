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

### `01_fetch_stablecoin_data.py` — market data

**This step fetches market (price/exchange) data only; no on-chain data.**
On-chain data comes from `dune_onchain.zip` in step `02b`.

Downloads hourly data for the six fiat-backed stablecoins (USDT, USDC, BUSD,
DAI, TUSD, PAX). Price comes from DefiLlama, and trading volume from Binance
(USDC, PAX, TUSD, BUSD) or Bitfinex (USDT, DAI). The two sources are joined
on coin and hour, keeping every price hour.

**Output:** `data/processed/hourly_stablecoin.parquet`, one row per coin-hour
(~345k rows, Sep 2018 – Dec 2025; BUSD ends Feb 2024 when it was discontinued).

| Column | Description |
|---|---|
| `coin` | Stablecoin ticker |
| `hour` | Hour timestamp (UTC) |
| `price` | USD price (DefiLlama) |
| `volume` | Exchange trading volume for the hour; 0 if no trades, NaN before the exchange's data starts |
| `trades` | Number of trades (Binance only; NaN for Bitfinex coins) |
| `taker_buy_base_volume` | Volume where the buyer was the taker (Binance only; NaN for Bitfinex coins) |
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
| `trades_pct_change_1h` | Change in trade count vs previous hour |
| `taker_buy_ratio` | Share of volume from aggressive buys; NaN for USDT/DAI (Bitfinex doesn't report it) |
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
| `total_volume_usd`, `whale_tx_count`, `median_transfer_usd`, `transfer_gini` | Optional; included only if present in the CSV |

> **Note:** the output holds raw hourly on-chain counts only. The script
> computes `tx_count_pct_change` / `tx_volume_pct_change` but doesn't save
> them, and no 1h/6h/24h change features are built yet.

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
   episode, so one crash isn't split into many small episodes. 75 episodes
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
> - Positive labels are rare: about 0.02% (h=1), 0.13% (h=6) and 0.5% (h=24)
>   of rows.

## Master dataset (`data/processed/master.parquet`)

**This is the single source of truth for modelling.** It is created by
`data_processing/03_merge_and_label.py`. Any other "master" file
(e.g. `master_hourly_dataset.*`) is from an old pipeline and should be ignored.

**How to rebuild it** (from the repo root):

```powershell
python data_processing/01_fetch_stablecoin_data.py
python data_processing/02a_build_market_features.py
python data_processing/02b_build_onchain_features.py
python data_processing/03_merge_and_label.py
python data_processing/04_split_train_test.py          # train/val/test split
python data_processing/verify_embargo_and_episodes.py  # optional: checks the split
```

### Overview

- **One row = one coin-hour.**
- **345,367 rows × 36 columns.**
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
| Raw market data | `price`, `trades`, `taker_buy_base_volume`, `venue`, `venue_switch_x`, `hours_since_venue_start` | 01 |
| Market features | `peg_deviation`, `peg_deviation_bps`, `return_1h/6h/24h`, `log_return_1h/6h/24h`, `realised_vol_6h/24h`, `trading_volume`, `volume_pct_change_1h/6h/24h`, `taker_buy_ratio`, `trades_pct_change_1h`, `venue_switch_y` | 02a |
| On-chain data | `tx_count`, `tx_volume`, `active_senders`, `active_receivers` | 02b |
| Depeg flags | `is_stablecoin`, `outside_band`, `episode_id`, `in_episode` | 03 |
| **Targets** | `label_h1`, `label_h6`, `label_h24` | 03 |

See the step 01 / 02a / 02b / 03 sections above for what each column means.

### Targets

`label_h{h} = 1` if a new depeg episode (price < $0.99 for ≥ 2 consecutive
hours) **starts** within the next h hours, else 0. NaN for hours already
inside an episode, with no price, or too close to the end of the data.

| Label | Labelled rows | Positives | % positive |
|---|---|---|---|
| `label_h1` | 341,480 | 75 | 0.02% |
| `label_h6` | 341,450 | 447 | 0.13% |
| `label_h24` | 341,342 | 1,772 | 0.52% |

There are 75 depeg episodes in total (`depeg_episodes.parquet`): USDT 12,
USDC 14, BUSD 9, DAI 9, TUSD 14, PAX 17.

### Train / validation / test split

`data_processing/04_split_train_test.py` splits the master by date, with a
24h purge and embargo at each boundary, and writes
`train_h*.parquet`, `val_h*.parquet` and `test_h*.parquet`.
The h1/h6/h24 files hold the same rows; use the matching `label_h*` column.

| Split | Period | Rows | Positives (h=24) |
|---|---|---|---|
| Train | Sep 2018 – Dec 2022 | 203,503 | 1,412 |
| Validation | 2023 | 52,272 | 312 |
| Test | 2024 – 2025 | 89,016 | 48 |

### Known quirks

- `venue_switch` appears twice (`venue_switch_x` / `venue_switch_y`) with the same values.
- `taker_buy_ratio`, `trades` and `taker_buy_base_volume` are always NaN for USDT and DAI (Bitfinex doesn't report them).
- On-chain columns are raw hourly counts; no on-chain change features yet.
- The test set has very few positives, so test metrics will be noisy.
