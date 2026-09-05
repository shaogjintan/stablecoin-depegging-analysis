# Replication: Lee et al. (2025), "Stablecoin depegging risk prediction"

**Source paper**: Lee, Y.-H., Chiu, Y.-F., & Hsieh, M.-H. (2025). *Stablecoin depegging risk prediction.* Pacific-Basin Finance Journal, 90, 102640.

**Script**: [`lee_et_al_replication.py`](lee_et_al_replication.py) — run with `python models/lee_et_al_replication.py` (from repo root, inside the project `.venv`). Outputs: [`lee_replication_label_summary.csv`](lee_replication_label_summary.csv), [`lee_replication_results.csv`](lee_replication_results.csv).

**Scope** (Workstream 1 brief): *"Run a simple Logistic Regression and Random Forest using daily price/volatility indicators. Produce a table showing their baseline accuracy/ROC-AUC."*

---

## 1. What the paper did

- **Coins**: USDT, USDC, BUSD, DAI (top 4 by 24h volume, CoinGecko, as of Dec 2023) + BTC/ETH as spillover regressors.
- **Period**: Jan 1, 2022 – Dec 31, 2023, daily frequency (730 rows/coin).
- **Label** — a *dynamic*, volume-scaled depeg threshold (their headline contribution):

  ```
  ThreshD_t = 1 − 10 / V_monthly_t^(1/3)
  ThreshU_t = 1 + 10 / V_monthly_t^(1/3)
  Y_t = 1  if  P_low_t ≤ ThreshD_t  or  P_high_t ≥ ThreshU_t
  ```
  where `V_monthly` is the trailing 30-day sum of trading volume — high-volume coins get a tighter band, low-volume coins a looser one. Bilateral (catches both up- and down-depegs), extending Carey (2023)/Kaiko's downward-only version.
- **Features**: 66 variables lagged by 1 day, across 4 blocks — (1) price/volume % change (1h/24h/7d/30d), (2) market-cap/supply % change (24h/7d/30d), (3) sentiment (Fear & Greed, Sentiment, Awareness indices), (4) volatility (Realized Daily Volatility, Price Deviation, Downward Price Deviation) — each block repeated for the coin itself, BTC, and ETH.
- **Models**: Logistic Regression, Random Forest, XGBoost; 2-fold CV tuned on F1; SMOTE applied only to low-depeg-rate coins (USDC, DAI) at a tuned 0.6 ratio.
- **Split**: **stratified random sampling on the label** — explicitly *not* time-based ("the division of the train dataset and test dataset does not employ a time-based split").
- **Metrics**: Accuracy, Precision, Recall, F1 (primary), Specificity.

### Their results (Table 4 + Table 5)

| Coin | Depeg rate | LR F1 | RF F1 | XGB F1 |
|---|---|---|---|---|
| USDT | 23.0% | 0.531 | 0.744 | 0.780 |
| BUSD | 26.4% | 0.758 | 0.812 | 0.784 |
| USDC | 2.2% | 0.250* | 0.857* | 0.667* |
| DAI | 1.9% | 0.333* | 0.571* | 0.571* |

\* SMOTE applied (ratio 0.6). Headline finding: LR generally performs poorly; RF/XGBoost perform well. Sentiment indicators did **not** appear in the top-15 predictive features for any coin — their own negative finding on sentiment.

---

## 2. What we replicated, and why not everything

| Component | Replicated? | Notes |
|---|---|---|
| Dynamic threshold label (Y equation) | **Yes, exactly** | Same formula, same constants (α=1/3, K=10). |
| Price/volume change-rate features | **Yes** (block 1) | 1d/7d/30d instead of 1h/24h/7d/30d — we're daily-only, no 1h analogue. |
| Volatility features | **Yes** (block 4) | Realized Daily Volatility (Rogers-Satchell), Price Deviation & Downward Price Deviation (5d/30d). |
| **BTC/ETH spillover features** | **Yes** (partial) | Price/volume change-rate + Realized Daily Volatility for BTC and ETH, merged onto every coin's row. Peg-deviation features ("distance from $1") are *not* computed for BTC/ETH — that formula is meaningless for a non-pegged, non-$1 asset (see script docstring), so we deliberately diverge from a literal reading here. |
| Market-cap/supply features (block 2) | No | Not obtainable from Yahoo Finance's free OHLCV endpoint. |
| Sentiment features (block 3) | No | No equivalent free data source; also wasn't predictive in their own results. |
| Logistic Regression, Random Forest | **Yes** | XGBoost dropped — out of scope per brief. |
| Train/test split: stratified random | **Yes, deliberately** | Copied *as-is*, flaws included — see §4. |
| SMOTE | **No, deliberately** | Brief asked for "simple" LR/RF; watching them fail on imbalanced coins without it is itself evidence for the shortcomings slide (see §4). |
| **Sample period** | **Yes — Jan 2022 to Dec 2023** | Matched exactly (see §3: 729 rows vs. their 730). |
| Coin roster | **Yes, plus extras** | USDT/USDC/BUSD/DAI all included. Added PAX, USTC, WLUNA on top — PAX matches our project's other raw dataset; USTC/WLUNA are the two algorithmic coins Lee et al. excluded entirely (see §4). |

**Data source note**: everything (price *and* volume, for every asset including BTC/ETH) now comes from **Yahoo Finance** (`data/ohlcv/*_ohlcv.csv`, gitignored, 2021-11-15 to 2023-12-31 — a ~45-day lookback margin before the analysis window so 30-day rolling features are already "full" from day one). We initially tried to reuse this project's other raw dataset (`ERC20-stablecoins/price_data/`), but it has **no volume column at all** (the label formula needs it) and only spans Apr–Nov 2022 (vs. the paper's full 2 years) — a genuine data gap, not a design choice. CoinGecko's free API — the more obvious fix — now refuses to serve data older than 365 days, which is why Yahoo Finance stands in for the paper's actual source (CoinMarketCap).

---

## 3. Our results

**Label summary** (dynamic threshold, full 2022-01-01 → 2023-12-30 window — matches Lee et al.'s period almost exactly, 729 rows vs. their 730):

| coin | n_obs | date range | n_depeg_days | depeg_rate | Lee et al.'s rate |
|---|---|---|---|---|---|
| usdt | 729 | 2022-01-01 → 2023-12-30 | 179 | 24.6% | 23.0% |
| usdc | 729 | 2022-01-01 → 2023-12-30 | 49 | 6.7% | 2.2% |
| dai | 729 | 2022-01-01 → 2023-12-30 | 14 | 1.9% | 1.9% |
| pax | 729 | 2022-01-01 → 2023-12-30 | 17 | 2.3% | n/a (not in paper) |
| busd | 729 | 2022-01-01 → 2023-12-30 | 208 | 28.5% | 26.4% |
| ustc | 729 | 2022-01-01 → 2023-12-30 | 658 | 90.3% | n/a (excluded by paper) |
| wluna | 281 | 2022-01-01 → 2022-10-08 | 281 | 100.0% | n/a (excluded by paper; Yahoo's price history for this ticker ends Oct 2022) |

**Model performance** (stratified random split, faithful to their Table 5 method):

| coin | model | accuracy | precision | recall | F1 | specificity | ROC-AUC |
|---|---|---|---|---|---|---|---|
| usdt | Logistic Regression | 0.781 | 0.875 | 0.130 | 0.226 | 0.994 | 0.612 |
| usdt | Random Forest | 0.909 | 0.854 | 0.759 | **0.804** | 0.958 | 0.943 |
| usdc | Logistic Regression | 0.936 | 0.667 | 0.133 | 0.222 | 0.995 | 0.595 |
| usdc | Random Forest | 0.945 | 1.000 | 0.200 | **0.333** | 1.000 | 0.947 |
| dai | Logistic Regression | 0.968 | 0.000 | 0.000 | 0.000 | 0.986 | 0.690 |
| dai | Random Forest | 0.977 | 0.000 | 0.000 | 0.000 | 0.995 | 0.820 |
| pax | Logistic Regression | 0.977 | 0.000 | 0.000 | 0.000 | 1.000 | 0.615 |
| pax | Random Forest | 0.977 | 0.000 | 0.000 | 0.000 | 1.000 | 0.612 |
| busd | Logistic Regression | 0.831 | 0.931 | 0.435 | 0.593 | 0.987 | 0.767 |
| busd | Random Forest | 0.918 | 0.844 | 0.871 | **0.857** | 0.936 | 0.967 |
| ustc | Logistic Regression | 0.936 | 0.965 | 0.965 | 0.965 | 0.667 | 0.973 |
| ustc | Random Forest | 0.982 | 0.985 | 0.995 | **0.990** | 0.857 | 0.997 |
| wluna | — | — | — | — | — | — | single-class target, no model fit possible |

---

## 4. Interpretation — did it replicate?

**Yes — on both the mechanism and the core empirical finding, now with the same sample period and much closer depeg rates.**

- **Depeg rates now closely track theirs**, using the same ~2-year window: DAI 1.9% (theirs 1.9%, exact), USDT 24.6% (theirs 23.0%), BUSD 28.5% (theirs 26.4%). USDC comes out higher (6.7% vs. their 2.2%) — plausibly a genuine data-source difference (Yahoo's USDC-USD feed vs. their CoinMarketCap feed can disagree on intraday high/low, which is exactly what the threshold checks against), not a formula error, since every other coin lines up closely.
- **RF beats LR on every well-populated coin**, reproducing their headline claim directly: USDT (F1 0.804 vs 0.226), BUSD (0.857 vs 0.593), USDC (0.333 vs 0.222). The gap is consistent across all three, not a one-off — a stronger replication of the pattern than the previous (shorter-window) version of this script managed.
- **DAI and PAX still collapse to F1 = 0 for both models** — their depeg rates (1.9%, 2.3%) are low enough that "always predict no depeg" remains the easy way out without SMOTE. This is *expected*, not a failure: it's the exact problem Lee et al. built SMOTE to solve, and we deliberately left SMOTE out (see §2). Note this is a smaller, more contained failure than in the previous 7-month-window version of this script (which also failed on USDC) — a longer, more representative window helps, but doesn't fix a fundamentally rare-event class on its own.
- **USTC/WLUNA remain impossible to benchmark against the paper** (both excluded from their study) but are the most useful result of all for our own purposes: USTC now shows a 90.3% depeg rate over the full window, and WLUNA 100% — not because either coin generated hundreds of independent crisis days, but because most of their remaining trading history sits *after* the May 9, 2022 collapse, so nearly every subsequent day is trivially still "depegged." This is the pseudo-replication trap named explicitly in our professor's feedback, visible directly in our own numbers.

### Two concrete shortcomings this replication surfaces (for the presentation)
1. **No episode structure**: a per-day label treats one multi-month collapse as hundreds of "independent" positive observations (starkest in ustc/wluna, but the same logic inflates every coin's counted "depeg days").
2. **Random split, not chronological**: stratified-random sampling can place days from the same crash on both sides of the train/test split, leaking information across an episode boundary — the opposite of the chronological/purged split this project uses.

---

## 5. Reproducing this

```bash
cd stablecoin-depegging-analysis
.venv\Scripts\Activate.ps1        # or activate.bat / source .venv/bin/activate
python models/lee_et_al_replication.py
```

Requires `data/ohlcv/*_ohlcv.csv` to exist — daily OHLCV pulled from Yahoo Finance (via `yfinance`) for usdt, usdc, dai, pax, busd, ustc, wluna, btc, eth, covering 2021-11-15 to 2023-12-31 (gitignored; re-run the fetch to regenerate). This project's other raw dataset, `ERC20-stablecoins/`, is **not** used by this script — see §2 for why.
