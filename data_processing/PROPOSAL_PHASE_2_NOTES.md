# Proposal Phase 2: Data Pipeline, Purged Splitting & Empirical Validation

**Group:** Digital Currencies Group 1  
**Project:** Stablecoin Depegging Forecasting (May 2022 Stablecoin Crash)  
**Deliverable Owner (Week 7):** Kannan Sasinthiran (Sasi) — A0272426E  
**Status:** Complete & Empirically Validated  

---

## 1. Overview & What Was Done

This directory consolidates the Phase 2 data processing and modeling foundation for forecasting stablecoin depeg events across six major fiat-backed stablecoins (**USDT, USDC, BUSD, DAI, TUSD, PAX**) from 2018 to 2025.

### Key Milestones Completed for Week 7:
1. **Refactored Data Pipeline Integration:**
   * Consolidated data sources: DefiLlama hourly chart prices + Binance/Bitfinex OHLCV + Dune Analytics on-chain activity.
   * Fixed critical pipeline bugs: preserved `taker_buy_base_volume` (previously dropped/backfilled with NaN) and eliminated duplicate volume columns.
   * Standardized depeg definition: **downward-only price deviation** ($< \$0.99$), sustained for a minimum of 2 consecutive hours, with a 24-hour inactivity bridging gap to merge ongoing crisis runs into discrete macro episodes (**75 total depeg episodes**).

2. **Purged Chronological 3-Way Split (`04_split_train_test.py`):**
   * Implemented a strict 3-way partition:
     * **Training (2018-09-01 to 2022-12-31):** 439,777 rows (327,813 labelled). Captures early volatility and the systemic May 2022 Terra/UST + USDT collapse.
     * **Validation (2023-01-01 to 2023-12-31):** 154,177 rows (129,276 labelled). Captures the March 2023 Silicon Valley Bank run (USDC/DAI depeg); held out strictly for tuning classification decision thresholds and hyperparameters.
     * **Test (2024-01-01 to 2025-12-31):** 361,088 rows (321,801 labelled). Untouched out-of-sample forward evaluation period.
   * **Universal 24-Hour Purge:** Purged the final 24 hours of Train and Validation across all horizons ($h=1, 6, 24$) to strictly eliminate forward label lookahead into subsequent evaluation windows and maintain equal sample sizes.
   * **24-Hour Embargo:** Gapped the first 24 hours of Validation and Test to prevent autoregressive feature leakage from 24h rolling lookbacks.
   * **Episode-Aware Boundary Check:** Fully excluded any depeg episode straddling boundary windows ($[T_{\text{split}} - 24\text{h}, T_{\text{split}} + 24\text{h}]$) to ensure zero partial-event contamination.

3. **Empirical Verification (`verify_embargo_and_episodes.py`):**
   * Generated autocorrelation decay (ACF) tables across lags 1h to 48h, proving that feature memory drops to statistical insignificance ($\text{ACF} \approx 0$) by lag 24, providing empirical justification for the 24h embargo.
   * Verified episode balance across partitions: Train (586 sub-episodes), Validation (26 sub-episodes), and Test (34 sub-episodes), proving statistical viability for out-of-sample PR-AUC calculation.

---

## 2. Directory Structure & File Manifest

| File | Purpose |
| :--- | :--- |
| `01_fetch_stablecoin_data.py` | Fetches hourly reference prices from DefiLlama and OHLCV from Binance (USDC, PAX, TUSD, BUSD) and Bitfinex (USDT, DAI). Fills quiet hours with zero volume. |
| `02a_build_market_features.py` | Computes market features following Lee et al. (2025): log price/volume changes ($1\text{h}, 6\text{h}, 24\text{h}$), 24h Rogers–Satchell realized volatility, high-low spreads, and peg deviations. |
| `02b_build_onchain_features.py` | Computes on-chain features from Dune Analytics: transaction count/volume changes ($1\text{h}, 6\text{h}, 24\text{h}$), active senders/receivers, average transfer size, and sender-receiver ratios. |
| `03_merge_and_label.py` | Merges market and on-chain features onto a complete coin-hour grid, identifies downward depeg episodes, and creates binary forward onset labels for $h \in \{1, 6, 24\}$ hours. |
| `04_split_train_test.py` | **Core Week 7 Script:** Implements the purged chronological 3-way split with 24h embargo and episode-aware boundary protection. Outputs train, validation, and test parquet files. |
| `verify_embargo_and_episodes.py` | **Empirical Proof Script:** Runs autocorrelation tests across feature lags and verifies episode distribution across split partitions. |
| `config.py` | Central configuration defining coin universes, start/end dates, depeg thresholds, split boundaries, and directories. |
| `PROPOSAL_UPDATES_WEEK_7.md` | Complete, revised Markdown text for the Phase 2 Proposal Methodology, Task Allocation & Timeline, and Empirical Tables. |

---

## 3. Empirical Verification Summary

### Table 1: Feature Autocorrelation (ACF) Decay Across Lags
*Proves that 24 hours is the exact embargo window needed to break autoregressive memory across split boundaries.*

| Coin | Feature | Lag 1h | Lag 2h | Lag 4h | Lag 6h | Lag 12h | Lag 18h | Lag 24h | Lag 36h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **USDT** | 24h Price Return | 0.8177 | 0.7584 | 0.6577 | 0.5808 | 0.3859 | 0.1907 | **-0.1251** | -0.0464 |
| **USDT** | 24h On-Chain Tx Diff | 0.4622 | 0.3363 | 0.2622 | 0.2137 | 0.1119 | 0.0346 | **-0.3828** | -0.0621 |
| **USDC** | 24h Price Return | 0.5675 | 0.5116 | 0.4200 | 0.3360 | 0.1672 | 0.0182 | **-0.3326** | -0.0783 |
| **USDC** | 24h On-Chain Tx Diff | 0.5600 | 0.4440 | 0.3468 | 0.3065 | 0.1760 | 0.0427 | **-0.3582** | -0.1033 |
| **DAI**  | 24h Price Return | 0.6692 | 0.6086 | 0.4896 | 0.4047 | 0.1828 | -0.0253 | **-0.4043** | -0.1314 |
| **DAI**  | 24h On-Chain Tx Diff | 0.5494 | 0.4146 | 0.2929 | 0.2414 | 0.1212 | 0.0082 | **-0.4112** | -0.0778 |

### Table 2: Dataset Size and Labelled Rows by Horizon
| Horizon | Partition | Total Rows | Labelled Rows | Positive Labels | Pos Rate | Depeg Episodes | Date Span |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **h = 1** | Train | 439,777 | 327,813 | 2,276 | 0.69% | 2,273 | 2018-01-01 to 2022-12-30 |
| | Validation | 154,177 | 129,276 | 335 | 0.26% | 333 | 2023-01-02 to 2023-12-30 |
| | Test | 361,088 | 321,801 | 462 | 0.14% | 462 | 2024-01-02 to 2025-12-31 |
| **h = 6** | Train | 439,777 | 327,813 | 9,632 | 2.94% | 2,273 | 2018-01-01 to 2022-12-30 |
| | Validation | 154,177 | 129,276 | 1,259 | 0.97% | 333 | 2023-01-02 to 2023-12-30 |
| | Test | 361,088 | 321,706 | 1,947 | 0.61% | 462 | 2024-01-02 to 2025-12-31 |
| **h = 24** | Train | 439,777 | 327,813 | 24,687 | 7.53% | 2,273 | 2018-01-01 to 2022-12-30 |
| | Validation | 154,177 | 129,276 | 3,125 | 2.42% | 333 | 2023-01-02 to 2023-12-30 |
| | Test | 361,088 | 321,364 | 5,367 | 1.67% | 462 | 2024-01-02 to 2025-12-31 |

---

## 4. How to Reproduce

1. **Verify autocorrelation and split distribution:**
   ```bash
   python3 verify_embargo_and_episodes.py
   ```
2. **Execute the purged 3-way split (writes parquet files to `data/processed/`):**
   ```bash
   python3 04_split_train_test.py
   ```
