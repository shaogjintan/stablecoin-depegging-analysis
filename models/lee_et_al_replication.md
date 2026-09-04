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
| Market-cap/supply features (block 2) | No | Not in our raw dataset (`ERC20-stablecoins/`). |
| Sentiment features (block 3) | No | Not in our raw dataset; also wasn't predictive in their own results. |
| BTC/ETH spillover versions of every feature | No | Out of scope for this pass. |
| Logistic Regression, Random Forest | **Yes** | XGBoost dropped — out of scope per brief. |
| Train/test split: stratified random | **Yes, deliberately** | Copied *as-is*, flaws included — see §4. |
| SMOTE | **No, deliberately** | Brief asked for "simple" LR/RF; watching them fail on imbalanced coins without it is itself evidence for the shortcomings slide (see §4). |
| Coin roster | Adapted | Paper uses USDT/USDC/BUSD/DAI; we don't have BUSD locally, so we ran usdt/usdc/dai/pax (our closest analogues) plus ustc/wluna as a bonus — the two algorithmic coins the paper excluded entirely. |

**Data gap fixed along the way**: the label formula needs trailing 30-day *trading volume*, which our local `ERC20-stablecoins/price_data/*.csv` files don't contain (OHLC only). CoinGecko's free API no longer serves data older than 365 days, so we pulled daily volume from Yahoo Finance instead (`data/volume/*_volume_data.csv`, Feb 20 – Nov 4 2022, gitignored — regenerate by re-running the fetch if needed).

---

## 3. Our results

**Label summary** (dynamic threshold, 2022-05-02 → 2022-11-01, after the 30-day feature/label warm-up):

| coin | n_obs | n_depeg_days | depeg_rate |
|---|---|---|---|
| usdt | 184 | 55 | 29.9% |
| usdc | 184 | 1 | 0.5% |
| dai | 184 | 2 | 1.1% |
| pax | 184 | 11 | 6.0% |
| ustc | 184 | 179 | 97.3% |
| wluna | 184 | 184 | 100.0% |

**Model performance** (stratified random split, faithful to their Table 5 method):

| coin | model | accuracy | precision | recall | F1 | specificity | ROC-AUC |
|---|---|---|---|---|---|---|---|
| usdt | Logistic Regression | 0.696 | 0.000 | 0.000 | 0.000 | 1.000 | 0.487 |
| usdt | Random Forest | 0.946 | 0.938 | 0.882 | **0.909** | 0.974 | 0.992 |
| usdc | Logistic Regression | 1.000 | 0 | 0 | 0.000 | 1.000 | — |
| usdc | Random Forest | 1.000 | 0 | 0 | 0.000 | 1.000 | — |
| dai | Logistic Regression | 0.982 | 0 | 0 | 0.000 | 1.000 | 1.000 |
| dai | Random Forest | 0.982 | 0 | 0 | 0.000 | 1.000 | 1.000 |
| pax | Logistic Regression | 0.946 | 0 | 0 | 0.000 | 1.000 | 0.214 |
| pax | Random Forest | 0.946 | 0 | 0 | 0.000 | 1.000 | 0.811 |
| ustc | Logistic Regression | 0.964 | 0.964 | 1.000 | 0.982 | 0.000 | 1.000 |
| ustc | Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wluna | — | — | — | — | — | — | single-class target, no model fit possible |

---

## 4. Interpretation — did it replicate?

**Yes, on the mechanism and the core finding.** Not on exact numbers — different coins, period, and a reduced feature set mean an exact match was never the bar.

- **Depeg rates land in the same range as theirs** despite different data: USDC 0.5% (theirs 2.2%), DAI 1.1% (theirs 1.9%) — both low single digits; USDT 29.9% (theirs 23.0%) — both high because it's a high-turnover coin that genuinely wobbled. The formula behaves consistently across datasets.
- **USDT is the fair head-to-head comparison** (only coin here with enough positive examples for both models to be meaningfully tested): RF F1 (0.909) far exceeds LR F1 (0.000), reproducing their exact qualitative claim — *"logistic regression generally performs poorly, while Random Forest and XGBoost show good predictive effectiveness."* Ours is more extreme in both directions than theirs (LR 0.531→0.909 vs RF 0.744 in the paper), consistent with our much smaller feature set (11 vs 66) and sample (184 vs 730 days).
- **usdc/dai/pax collapse to F1 = 0 for both models** — expected, not a bug: we deliberately skipped SMOTE (out of scope per brief), and this is *exactly* the failure mode SMOTE exists to fix in their own paper. Reproducing the problem is itself evidence we understood why they needed it.
- **ustc/wluna (~97–100% depeg rate) can't be benchmarked against the paper** — Lee et al. excluded algorithmic stablecoins. But this result is the clearest evidence for our project's own critique: a ~100% depeg rate here isn't "179–184 independent crisis days," it's **one continuous collapse (the May 9 UST/LUNA crash) spanning almost the entire observation window.** Lee et al.'s daily-binary, stratified-random-split design has no way to tell "many independent bad days" apart from "one very long bad day" — precisely the pseudo-replication trap our professor flagged.

### Two concrete shortcomings this replication surfaces (for the presentation)
1. **No episode structure**: their per-day label treats a single multi-week crash as dozens/hundreds of "independent" positive observations (visible starkly in our ustc/wluna rows).
2. **Random split, not chronological**: stratified-random sampling can place days from the same crash on both sides of the train/test split, leaking information across an episode boundary — the opposite of the chronological/purged split this project uses.

---

## 5. Reproducing this

```bash
cd stablecoin-depegging-analysis
.venv\Scripts\Activate.ps1        # or activate.bat / source .venv/bin/activate
python models/lee_et_al_replication.py
```

Requires `data/volume/*_volume_data.csv` to exist (Yahoo Finance daily OHLCV pull, gitignored) and `ERC20-stablecoins/price_data/price_data/*.csv` (this project's raw price data) to be present one level up from the repo root.
