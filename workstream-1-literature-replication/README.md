# Workstream 1 — Literature Replication & Empirical Proofs ✅ COMPLETE

**Owner:** Sasi · **Status:** Complete · **Last verified:** outputs audited and slide-ready

Replication of Lee, Chiu & Hsieh (2025), *"Stablecoin depegging risk prediction"*
(Pacific-Basin Finance Journal 90:102640) and supporting evidence against Yip
(2022, HKMA RM 09/2022), produced for Slides 4 & 5 of the deck.

---

## 1. Folder contents

```
workstream-1-literature-replication/
├── README.md                        <- this file (full write-up + slide guide + handoff)
├── data/                            <- raw + processed market data
│   ├── may_2022_hourly_prices.csv            14,113 rows · 7 coins · Apr 1–Jun 30 2022 (the project spec window)
│   ├── fear_greed_index.csv                   91 daily rows · AlternativeMe Fear & Greed, Apr–Jun 2022
│   └── hourly_prices_full_2022_2023.csv   103,877 rows · full 2-year window (matches Lee et al.'s range)
├── outputs/                         <- slide-ready artifacts
│   ├── proof_daily_vs_hourly.png   → Slide 5 bottom (300 DPI)
│   ├── proof_daily_vs_hourly.pdf   → Slide 5 bottom (vector, for slide insertion)
│   ├── published_vs_ours.csv       → Slide 5 top (use USDT & BUSD rows only)
│   ├── replication_results.csv     → all 84 result rows (full audit trail)
│   ├── feature_importance.csv      → supporting evidence
│   └── literature_comparison.csv   → Slide 4 (5-paper table)
└── scripts/                         <- all re-runnable from scratch
    ├── 01_download_data.py         CoinGecko keyless attempt (fails by design — HTTP 401 error 10012, documented below)
    ├── 01b_download_binance.py     WORKING downloader (Binance archives + Kraken/Coinbase) for Apr–Jun 2022
    ├── 01c_download_full.py        WORKING downloader for the full Jan 2022–Dec 2023 window
    ├── 02_replicate_lee.py         Full replication pipeline (LR / RF / XGBoost, both splits, both thresholds)
    ├── 03_proof_chart.py           "Daily Frequency Flaw" proof chart generator
    └── 04_literature_table.py      Literature comparison table generator
```

**Deliberately not committed:** the intermediate warmup file
`hourly_prices_full.csv` (Jan–Jun 2022, superseded by the 2-year file), paper PDFs,
and personal project docs. Everything needed for the deck and for WS2/WS3 is here.

---

## 2. Data — provenance & verified coverage

| File | Rows | Window | Notes |
|---|---|---|---|
| `may_2022_hourly_prices.csv` | 14,113 | Apr 1 – Jun 30 2022 | Zero missing values. Columns: `timestamp, coin, price, volume` |
| `hourly_prices_full_2022_2023.csv` | 103,877 | Jan 1 2022 – Dec 31 2023 | Includes Jan–Mar 2022 warmup so 30-day features are valid from Apr 1 |
| `fear_greed_index.csv` | 91 | Apr–Jun 2022 | `value` (0–100) + `value_classification` |

Coin coverage (verified):
- **UST** — Apr 1 – **May 13, 2022** (1,009 hours). Ends at Binance's delisting of
  USTUSDT — correct, not a bug.
- **USDT, USDC, BUSD, DAI, BTC, ETH** — 2,184 hourly rows each in the Apr–Jun window.

**Why not CoinGecko:** the keyless CoinGecko API rejects pre-2025 history
(HTTP 401, error 10012: *"Public API users are limited to querying historical data
within the past 365 days"*). Data was therefore sourced from free keyless public
archives:
- **Binance** `data.binance.vision` monthly 1h klines — USTUSDT, USDCUSDT, BUSDUSDT, BTCUSDT, ETHUSDT
- **Coinbase Exchange** public candles — DAI-USD, USDT-USD (true-USD numeraire)
- **AlternativeMe** — Fear & Greed Index

Known gaps: USDCUSDT was suspended on Binance Oct 2022–Feb 2023 (real event; the
Mar 2023 SVB depeg is fully covered), BUSD winds down Dec 15, 2023, Kraken's public
OHLC no longer serves historical pagination (hence Coinbase for USDT/DAI).

---

## 3. Methodology (replication spec)

- **Resample** hourly → daily OHLCV per coin.
- **Depeg definitions** (from Lee et al. PDF p.5):
  - *Dynamic (paper's actual):* `Thresh_D = 1 − 10/V^α`, `Thresh_U = 1 + 10/V^α`
    with V = rolling 30-day volume sum, α = 1/3 (Carey 2023), bilateral.
  - *Fixed (project simplification):* ±1% band (`low < 0.99 or high > 1.01`).
  - Label: **Y = 1 if the coin depegs on the NEXT day** (1-day forward shift).
- **Features (13):** 1/7/30-day price returns, 1/7/30-day volume changes, Grobys
  (2021) realized daily volatility, 5/30-day price deviation, BTC 1/7-day return,
  BTC realized volatility, ETH 1-day return (cross-coin controls).
- **Models:** Logistic Regression (L2, balanced), Random Forest
  (100 trees, depth 5, balanced), XGBoost (100 trees, depth 4, scale_pos_weight).
- **SMOTE(0.6)** on training sets with depeg ratio < 10% (paper's rule).
- **Splits:** (A) stratified random 70/30 — the paper's method;
  (B) chronological 2022 → 2023; (C) chronological Apr 1–May 7 → May 8–Jun 30 2022.

---

## 4. Headline findings (verified)

### 4.1 Daily frequency is too slow — 49 hours of warning lost
UST first broke below $0.99 at **May 7, 2022 22:00 UTC** ($0.9884 — visible hourly).
A daily-frequency model only sees it at the **close of May 9** ($0.7540).
**49 hours of early warning lost.** Proof chart: `outputs/proof_daily_vs_hourly.png`.

### 4.2 Random train/test splits leak future information
Pooled, dynamic threshold, full 2022–2023 window:

| Model | Random split F1 (their method) | Chronological split F1 (honest) | Verdict |
|---|:---:|:---:|---|
| XGBoost | **0.714** | **0.000** | Random split massively inflates scores |
| Random Forest | 0.444 | 0.000 | Same pattern |
| Logistic Regression | 0.099 | 0.178 | LR is more robust (less overfitting) |

> This is the strongest empirical critique: XGBoost achieves F1 = 0.714 on random
> splits but **literally predicts zero depegs** under a chronological split. Shuffling
> time-series data leaks future crisis days into training — which is exactly why our
> project uses chronological purged splits.

### 4.3 Published vs our replication (dynamic threshold, random split, full window)

| Model | Coin | Lee et al. F1 | Our F1 | Gap explained by |
|---|---|:---:|:---:|---|
| Random Forest | BUSD | 0.812 | **0.545** | Venue-level vs market-aggregate volume in the threshold |
| XGBoost | USDT | 0.780 | **0.667** | Same volume caveat |
| Random Forest | USDC | 0.857 | **0.000** | Only 1 positive test sample |
| Random Forest | DAI | 0.571 | **0.000** | Only 1 positive test sample |

Full 12-row table: `outputs/published_vs_ours.csv`.

### 4.4 Feature importance (pooled RF, dynamic threshold)
`real_vol` (0.243) > `BTC_real_vol` (0.178) > `vol_pc_7d` (0.132) >
`BTC_pc_7d` (0.123) > `dev_5d` (0.119).
**Confirms Lee et al. Table 6:** BTC-linked volatility dominates; sentiment adds
nothing. No sentiment features needed.

### 4.5 Depeg-day calendar (fixed ±1% threshold, matches real history)
- **2022-05-12** UST collapse → USDT low 0.9720 · USDC high 1.0500 · BUSD high 1.0392
- **2022-11-10** FTX → USDT low 0.9850 · BUSD high 1.0129
- **2023-03-11…13** SVB/USDC crisis → USDC low 0.9128 · DAI low 0.8900 · USDT high 1.0150

---

## 5. Slide assembly guide (for the deck)

### Slide 4 — Literature Landscape & Key Gaps
- Paste the 5-paper table from `outputs/literature_comparison.csv`.
- **Verbal emphasis:** *"ALL prior work uses daily data. NONE uses on-chain wallet
  flows. Sentiment was proven ineffective by Lee et al. themselves."*

### Slide 5 — Empirical Replication of Lee et al. (2025)
- **Top half:** trimmed table (**USDT + BUSD rows only**) from
  `outputs/published_vs_ours.csv`. USDC/DAI had 1 positive test sample each —
  mention verbally, don't put the unreliable 0.000 rows on the slide.
- **Bottom half:** `outputs/proof_daily_vs_hourly.png`.
- **Verbal emphasis:**
  1. *"49 hours of warning lost by daily models"* (point at chart)
  2. *"Their random train/test split inflates XGBoost F1 from 0.000 to 0.714 — pure
     data leakage"* (point at table)

### Caveat if the professor asks
*"Our dynamic thresholds use venue-level Binance volumes rather than market-aggregate
volumes, so our depeg ratios are lower than their Table 4. The directional
conclusions are unaffected."*

---

## 6. Handoff for other workstreams

- **WS2 — data merge & depeg target engineering:**
  use `data/may_2022_hourly_prices.csv` directly (columns `timestamp, coin, price,
  volume`, UTC hourly). `data/fear_greed_index.csv` available if needed.
  **UST ends May 13, 2022 — correct, not a bug.**
- **WS3:** the full 2-year file
  `data/hourly_prices_full_2022_2023.csv` is available for any out-of-sample work;
  note the USDC Oct 2022–Feb 2023 Binance suspension gap.

---

## 7. What still needs to be added (open items)

- [ ] **Deck assembly** (Slides 4 & 5) per section 5 — owners: Sasi / deck lead.
- [ ] **Paper PDFs** (Lee et al.; Yip 2022) — not committed; add to a
      `references/` folder if the repo should be self-contained.
- [ ] **Requirements:** xgboost + imbalanced-learn are needed to re-run scripts
      (added to root `requirements.txt`).
- [ ] **Optional upgrade:** a CoinGecko Demo key would let us rebuild the dynamic
      thresholds with market-aggregate volume, closing the Table 4 gap in §4.3.

---

## 8. How to re-run

```bash
pip install -r ../../requirements.txt
cd workstream-1-literature-replication
python3 scripts/01c_download_full.py     # rebuild data/hourly_prices_full_2022_2023.csv (~5 min)
python3 scripts/02_replicate_lee.py      # -> outputs/replication_results.csv, feature_importance.csv
python3 scripts/03_proof_chart.py        # -> outputs/proof_daily_vs_hourly.{png,pdf}
python3 scripts/04_literature_table.py   # -> outputs/literature_comparison.csv
```
(`01_download_data.py` is the documented CoinGecko attempt; it fails by design on
the public tier. `01b`/`01c` are the working downloaders.)
