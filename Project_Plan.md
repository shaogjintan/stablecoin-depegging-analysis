# DSE4101 Group Project Plan — Post-Feedback Revision

**Project:** Stablecoin Depegging Forecasting  
**Group:** Digital Currencies Group 1  
**Date:** 31 August 2026  
**Status:** Revised direction after Phase 1 faculty feedback  

---

## 1. What the Prof Told Us (and What It Means)

The feedback boils down to five things we need to change:

| Problem the Prof Identified | What We Do About It |
| :--- | :--- |
| Too many methods (PCA, UMAP, TDA, NLP, RF, XGB, LSTM, SHAP). We're demonstrating algorithms, not answering a question. | **Strip down to a clean core.** Logistic Regression + one tree model (XGBoost or RF). TDA and NLP are optional extensions that must prove their value empirically. |
| "Drivers of crashes" is a causal claim we can't support. Tree models + SHAP show predictive association, not causation. | **Reframe as a forecasting project.** We predict depegging, we don't explain what causes it. |
| Nothing is defined — what is a depeg? what is the forecasting horizon? what's one data point? | **Define everything precisely before touching any model code.** |
| 70M transactions ≠ 70M independent observations. Our real sample is the number of independent depeg episodes, which is tiny. | **Count depeg episodes first. This is probably our biggest constraint.** Design the evaluation around this reality. |
| Random train/test split + accuracy = useless. | **Chronological splits with episode-aware purging. PR-AUC as the primary metric.** |

---

## 2. Our Revised Research Question

> **"Can information available at time *t* predict whether a stablecoin will depeg within the next *h* hours, and does adding on-chain transaction data improve forecasting beyond market data alone?"**

This is a forecasting question with a testable information-set comparison. Not a "why do stablecoins crash" project.

---

## 3. Definitions We Must Lock Down (Week 3–4)

These need to be agreed on by the group before any coding starts. Bring to group meeting for a decision:

### A. What Counts as a "Depeg"?

**Proposal:** Price falls below $0.99 or rises above $1.01 (i.e., deviation > 1 cent from peg).

**Open questions to decide together:**
- Do we require the breach to persist for a minimum duration (e.g., ≥ 2 consecutive hours) to filter out random noise/liquidity blips?
- Do we use a tighter band (e.g., ±0.5%) or the ±1% band?
- Should we use different thresholds for different stablecoins? (Algorithmic coins like UST deviate more than USDT/USDC.)

### B. Forecasting Horizon (*h*)

**Proposal:** Test three horizons — **1 hour, 6 hours, 24 hours**.
- 1h = near-term early warning
- 6h = actionable intervention window
- 24h = strategic risk alert

### C. Unit of Observation

**Proposal:** One observation = one **coin-hour** (stablecoin *i* at hour *t*).

This means: for each stablecoin, for each hour, we have one feature vector *X_{i,t}* and one binary label *Y_{i,t}* (will this coin depeg in the next *h* hours?).

### D. What Counts as an Independent "Depeg Episode"?

This is critical. If UST falls from $0.98 to $0.50 over 72 hours, that is **one episode**, not 72 separate events. We need a rule to cluster consecutive depeg windows.

**Proposal:** Two depeg windows belong to the same episode if they are separated by fewer than 24 hours of peg-stable pricing.

---

## 4. The Empirical Strategy: Information-Set Comparison

The prof was clear — **don't run a model horse race, run an information-set comparison.** Hold the model constant, change the input data, and ask whether more information helps.

### The Three Tiers

```
Tier 1 (Market Only)          → Baseline
Tier 2 (Market + On-Chain)    → Core hypothesis test
Tier 3 (Market + On-Chain + Extension)  → Only if Tier 2 works and we have time
```

| Tier | Feature Set | Purpose |
| :--- | :--- | :--- |
| **Tier 1: Market Only** | Peg deviation, rolling returns (1h/6h/24h), realized volatility, trading volume, BTC/ETH returns as controls | What can exchange data alone tell us? This is the baseline to beat. |
| **Tier 2: Market + On-Chain** | Everything in Tier 1 **+** net wallet outflows, whale transaction count/volume, token velocity, active addresses | Does blockchain transparency give advance warning that exchange prices haven't priced in yet? |
| **Tier 3: Extension (Optional)** | Everything in Tier 2 **+** TDA persistence features OR sentiment scores | TDA/NLP must demonstrably improve OOS PR-AUC. If they don't, we report that finding honestly and drop them. |

### Models (Keep It Simple)

| Model | Role |
| :--- | :--- |
| **Logistic Regression** | Interpretable linear baseline. If this already works well, fancy models aren't needed. |
| **XGBoost** | Non-linear benchmark. Handles interactions and non-linearities in feature space. |
| **Random Forest** | Include only if trivial to implement alongside XGBoost (same sklearn/xgboost pipeline). |

**Dropped from core:** LSTM, PCA, UMAP. These can return as optional extras if the core pipeline is done early and we have spare capacity.

---

## 5. Validation & Evaluation (Non-Negotiable Rules)

### A. No Random Splits

Use **chronological train/test splits** only:
- Train on earlier periods → test on later periods
- Insert an **embargo gap** of *h* hours between train and test to prevent label leakage
- Consider **walk-forward (expanding window)** cross-validation

### B. Episode-Aware Purging

- If any observation from a depeg episode appears in train, **no observation from the same episode** can appear in test
- This is the hardest part of the pipeline to get right — assign it to someone careful

### C. Metrics

| Metric | Why |
| :--- | :--- |
| **PR-AUC** (Primary) | Imbalanced data → precision-recall curve is far more informative than ROC |
| **Precision & Recall** | At what threshold do we get acceptable trade-off? |
| **False Positive Rate** | How many false alarms does the model generate? |
| **Brier Score** | Are the predicted probabilities well-calibrated? |
| **ROC-AUC** | Supplementary only |
| ~~Accuracy~~ | **Do not report this as a main metric.** A model predicting "no depeg" every time gets 99%+ accuracy. |

---

## 6. Data Work — The Real Bottleneck

The prof emphasized: **data is the bigger problem, not ML.** Here's what needs to happen:

### Step 1: Download & Inspect (Week 3)
- [ ] Download Stanford SNAP ERC-20 dataset → identify which stablecoins are covered
- [ ] Download Mendeley "Anatomy of a Run" replication data (Liu, Makarov, Schoar 2023) → check granularity (15-min? hourly? daily?)
- [ ] Pull OHLCV from CoinGecko API for UST, USDT, USDC, DAI, BUSD (Apr–Jun 2022)
- [ ] Document the exact date ranges, frequency, and stablecoin coverage of each source

### Step 2: Find the Overlap (Week 3–4)
- [ ] Create a **coverage matrix**: rows = stablecoins, columns = datasets. Which coins appear in which datasets?
- [ ] Identify the **common date range** where all needed data sources overlap
- [ ] Check timestamp formats and timezones — everything must align to UTC

### Step 3: Count Depeg Episodes (Week 4)
This is the most important output before midterm:
- [ ] Apply our depeg definition to the price series
- [ ] Cluster consecutive depeg hours into episodes
- [ ] **Report the count of independent depeg episodes per stablecoin.** If it's < 5, we need to be honest about what our models can realistically learn.
- [ ] Report class balance: how many coin-hours are Y=1 vs Y=0 for each horizon *h*?

### Step 4: Feature Engineering & Merge (Week 4–5)
- [ ] Aggregate SNAP transaction data to hourly on-chain features (net flows, whale counts, velocity)
- [ ] Merge market OHLCV features + on-chain features on (stablecoin, hour) keys
- [ ] Compute rolling window features: lagged returns, realized vol, momentum
- [ ] Quality checks: missing values, outliers, stationarity of features

---

## 7. Workstream Assignments

We have 6 people. Here's how to divide the work so nobody is blocked and everyone has a clear deliverable:

### Workstream A: Market Data Pipeline
**Lead: Shao Gjin** | Support: Alyssa

- Download and clean OHLCV price data (CoinGecko + Mendeley replication set)
- Resample to hourly grid, compute peg deviation, returns, volatility
- Create the depeg label *Y_{i,t}* for each horizon *h*
- Count and catalogue depeg episodes
- **Deliverable:** Clean `market_features.csv` with label column

### Workstream B: On-Chain Data Pipeline
**Lead: Javier** | Support: Russell

- Download and explore SNAP ERC-20 dataset
- Aggregate raw transactions to hourly on-chain features:
  - Net transfer volume (inflows − outflows)
  - Whale transaction count (transfers > some threshold, e.g., > $100K)
  - Active unique addresses per hour
  - Token velocity (volume / supply)
- **Deliverable:** Clean `onchain_features.csv` keyed on (stablecoin, hour)

### Workstream C: Modeling & Evaluation Pipeline
**Lead: Sasinthiran** | Support: Vienna

- Build the chronological train/test split with episode-aware purging
- Implement Logistic Regression + XGBoost baseline
- Set up evaluation: PR-AUC, precision-recall curves, Brier score, calibration plots
- Run the information-set comparison (Tier 1 vs Tier 2)
- **Deliverable:** Reproducible training/evaluation script + results table

### Workstream D: TDA Extension (Optional — Tier 3)
**Lead: Alyssa** | Starts only after Tier 1 + 2 pipeline is working

- Construct sliding-window persistence diagrams over price + volume series
- Extract persistence landscape norms as features
- Test: does adding these features improve OOS PR-AUC over Tier 2?
- **Decision gate:** If no improvement, document the negative result and drop TDA

### Workstream E: NLP / Sentiment Extension (Optional — Tier 3)
**Lead: Vienna** | Starts only after Tier 1 + 2 pipeline is working

- Source news/social data (CoinDesk, Reddit archives, or Kaggle sentiment datasets)
- Extract daily sentiment scores using a pretrained transformer (e.g., FinBERT)
- Test: does adding sentiment improve OOS PR-AUC over Tier 2?
- **Decision gate:** Same as TDA — earn its place or drop it

### Cross-Cutting: Presentation & Writing
**Lead: Russell** | All contribute

- Midterm slides (Week 6 deadline: **Sunday 13 Sept**)
- Phase 2 Proposal (Week 7 deadline: **29 Sept**)
- Final report & slides (Week 12)

---

## 8. Week-by-Week Timeline

Aligned to the course syllabus deadlines.

| Week | Dates (approx.) | Milestone | Owner |
| :---: | :--- | :--- | :--- |
| **3** | 25 Aug – 31 Aug | Phase 1 Proposal submitted. **Now:** Download all datasets. Inspect coverage, date ranges, frequencies. Lock down depeg definition in group meeting. | All |
| **4** | 1 Sep – 7 Sep | **Data merge sprint.** Market pipeline complete (Shao Gjin). On-chain exploration started (Javier). Depeg episodes counted and class balance documented. Initial coverage matrix showing stablecoin overlap. | Shao Gjin, Javier, Alyssa |
| **5** | 8 Sep – 13 Sep | **Feature engineering + first model.** On-chain features aggregated and merged. Logistic Regression + XGBoost baseline trained on Tier 1 (market only). First PR-AUC numbers. Start slides. | Sasinthiran, Javier, Russell |
| **6** | 13 Sep (slides due) / 15 Sep (presentation) | **Midterm Presentation.** Show: data summary (overlap, date range, frequency, depeg episode count, class balance), Tier 1 baseline results, plan for Tier 2 comparison. | All |
| **R** | 22 Sep | Peer review report due (individual). | Individual |
| **7** | 29 Sep | **Phase 2 Proposal due.** Refined research question, finalized data sources & methodology, updated timeline. Tier 2 (market + on-chain) results should be in progress. | All |
| **8–9** | Oct | **Core analysis.** Complete Tier 1 vs. Tier 2 comparison. Full evaluation suite (PR-AUC, calibration, false alarms). SHAP feature importance for interpretation. | Sasinthiran, Shao Gjin |
| **10–11** | Oct–Nov | **Extensions (if earned).** TDA experiment (Alyssa). NLP experiment (Vienna). Robustness checks (different horizons, different depeg thresholds). Write-up started. | Alyssa, Vienna, Russell |
| **12** | 1 Nov (slides) / 3 Nov (presentation) / 6 Nov (report) | **Final Presentation & Report.** Complete results, write report, submit. | All |
| **13** | 13 Nov | Final peer review report due (individual). | Individual |

---

## 9. Midterm Presentation — What to Show (Week 6)

The prof was specific about what they want to see by midterm. **This is our checklist:**

- [ ] **Stablecoin coverage table:** Which coins appear in SNAP? Which in CoinGecko/Mendeley? What overlaps?
- [ ] **Common date range:** Exact start and end dates where all data sources align.
- [ ] **Timestamp frequency:** Are we working at 15-minute, hourly, or daily granularity? What did we choose and why?
- [ ] **Depeg episode count:** How many distinct depeg episodes exist in our sample, per stablecoin? (This is probably a small number — be honest about it.)
- [ ] **Class balance:** For each horizon (1h, 6h, 24h), how many coin-hours are Y=1 (depeg incoming) vs Y=0 (stable)?
- [ ] **Baseline Tier 1 results:** At minimum a Logistic Regression PR-AUC on market-only features with chronological split.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :---: | :---: | :--- |
| Very few independent depeg episodes → model can't generalise | **High** | **Critical** | Be transparent about effective sample size. Consider expanding to additional stablecoins or time periods. Discuss limitations honestly in report. |
| SNAP dataset doesn't overlap cleanly with price data (different coins, different dates) | Medium | High | Check coverage matrix in Week 3. Have CoinGecko + DeFiLlama + Binance as backup price sources. |
| TDA / NLP adds complexity but no OOS improvement | Medium | Low | This is actually fine — reporting a null result ("TDA didn't help") is a valid research finding. |
| Data merging takes longer than expected (timezone mismatches, missing hours, etc.) | **High** | Medium | Start data work immediately (Week 3). Don't wait until Week 5. |
| Class imbalance causes all models to predict Y=0 constantly | Medium | High | Use class weighting, SMOTE on training data only, or threshold tuning. Evaluate with PR-AUC (not accuracy). |
| Group coordination / uneven workload | Medium | Medium | Clear workstream ownership. Weekly 30-min sync. Peer evaluation at end of semester matters. |

---

## 11. Group Norms & Communication

- **Weekly sync:** 30 minutes, same day each week. Quick standup: what did you do, what's blocking you, what's next.
- **Shared repo:** All code and data in a shared GitHub/Google Drive folder. No one works on local-only files.
- **Naming convention:** `market_features.csv`, `onchain_features.csv`, `merged_panel.csv`. Agree on column names early.
- **Notebook vs. scripts:** Exploratory work in notebooks. Final pipeline in `.py` scripts that can be re-run end-to-end.
- **Decisions log:** Any decision about definitions (depeg threshold, horizon, etc.) gets documented in this plan or a shared decisions doc. No verbal-only agreements.

---

## 12. Summary: The One Slide Version

```
QUESTION:  Can we predict stablecoin depegging, and does on-chain data help?

DATA:      OHLCV prices (CoinGecko/Mendeley) + ERC-20 transactions (SNAP)
           → merged into a coin-hour panel

DESIGN:    Information-set comparison
           Tier 1: Market only  vs.  Tier 2: Market + On-Chain

MODELS:    Logistic Regression (baseline)  +  XGBoost (non-linear)

EVAL:      Chronological split, episode-aware purging
           PR-AUC, precision/recall, calibration

EXTRAS:    TDA & NLP only if they improve OOS performance (Tier 3)
```
