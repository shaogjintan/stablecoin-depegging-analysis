# DSE4101 Phase 1 Proposal: Faculty Feedback & Project Revisions

**Course:** DSE4101 — Digital Currencies Project (Topic 2)  
**Project:** Stablecoin Depegging Forecasting (May 2022 Crash)  
**Status:** Post-Phase 1 Feedback & Midterm Action Plan  

---

## 1. Raw Faculty Feedback

> **Summary of Feedback:**
> 1. **Cut the methods list:** Avoid algorithm overload (PCA, UMAP, TDA, NLP, RF, XGB, SHAP, LSTM). Focus on answering a rigorous empirical research question rather than demonstrating algorithms.
> 2. **Refocus as a Forecasting Project:** Frame as depegging prediction, not causal "drivers of crashes" (tree models + SHAP show predictive association, not causality).
> 3. **Adopt a Clean Research Question:**  
>    *"Can information available at time $t$ predict whether a stablecoin will depeg within the next $h$ hours, and does adding on-chain information improve prediction beyond market information alone?"*
> 4. **Rigorous Definitions Required:**
>    - Precise definition of a "Depeg" event (e.g., $< \$0.99$, outside $[0.99, 1.01]$, sustained duration).
>    - Clear forecasting horizon $h$ (e.g., $1\text{h}, 6\text{h}, 24\text{h}$).
>    - Explicit unit of observation (e.g., coin-hour or coin-15-minute).
> 5. **Data Bottleneck & Effective Sample Size:**
>    - 70M+ SNAP transactions do not mean a huge sample size for classification; count **independent depeg episodes**.
>    - Prevent pseudo-replication from dense observations around a single crash.
> 6. **Midterm Deliverables (Data Merge & Summary):**
>    - Overlapping stablecoins across datasets (market + on-chain).
>    - Common date range and alignment.
>    - Timestamp frequency / aggregation level.
>    - Count of distinct depeg episodes.
>    - Class balance: Positive vs. negative prediction windows.
> 7. **Model Set & Architecture:**
>    - Core models: **Logistic Regression** (baseline) + **Random Forest / XGBoost**.
>    - Drop **LSTM** from the core pipeline (sample size / rare event bottleneck).
>    - **TDA / Persistent Homology:** Must earn its place by demonstrating incremental out-of-sample (OOS) value over standard volatility/liquidity/flow metrics.
>    - **NLP / Sentiment:** Secondary extension only after the market + on-chain pipeline is robust.
> 8. **Experimental Design:**
>    - Shift from an "algorithm horse race" to an **information-set comparison**:
>      $$\text{Model 1: Market Only} \quad \text{vs.} \quad \text{Model 2: Market + On-Chain} \quad \text{vs.} \quad \text{Model 3: Market + On-Chain + Extension (TDA/NLP)}$$
> 9. **Validation & Evaluation Metrics:**
>    - **Strict chronological split / purged rolling-origin CV** (no random train/test splits; prevent leakages across identical depeg episodes).
>    - **Metrics:** **PR-AUC**, Precision/Recall, False Alarm Rate, Probability Calibration (Brier score); ROC-AUC as supplementary. Accuracy is uninformative due to extreme class imbalance.

---

## 2. Core Takeaways & Structural Pivots

```mermaid
graph TD
    subgraph Original Proposal (Algorithm-Centric)
        A1[70M SNAP Txs + Prices + News] --> B1[PCA + UMAP + TDA + NLP]
        B1 --> C1[RF + XGBoost + LSTM + SHAP]
        C1 --> D1[Causal Driver Claims]
    end

    subgraph Revised Strategy (Question-Centric & Information-Set Design)
        A2["Clean Market (OHLCV) + On-Chain (Flows)"] --> B2["Observation Unit: Coin-Hour (t)"]
        B2 --> C2["Info-Set 1: Market Features"]
        B2 --> D2["Info-Set 2: Market + On-Chain Flow Features"]
        B2 --> E2["Info-Set 3: Market + On-Chain + TDA (Optional)"]
        C2 & D2 & E2 --> F2["Models: Logistic Regression vs. XGBoost/RF"]
        F2 --> G2["Chronological Purged Split -> PR-AUC & Calibration"]
    end
```

---

## 3. Revised Project Scope & Operational Definitions

### A. Primary Research Question
> **"Can information available at time $t$ predict whether a stablecoin will experience a depeg event within the next $h$ hours, and does augmenting market data with on-chain transaction metrics provide statistically significant incremental predictive power?"**

### B. Target Variable ($Y_{i, t+h}$) & Event Definition
* **Depeg Event Condition:**  
  A stablecoin $i$ is defined as depegged at time $\tau$ if its price $P_{i, \tau}$ deviates outside the peg tolerance band:
  $$|P_{i, \tau} - 1.00| > 0.01 \quad (\text{i.e., } P_{i, \tau} < \$0.99 \text{ or } P_{i, \tau} > \$1.01)$$
  *(Optionally require persistence for at least $k$ consecutive intervals, e.g., $\ge 30\text{ mins}$, to filter micro-liquidity blips).*
* **Forward-Looking Target Window ($h$):**
  $$Y_{i, t}^{(h)} = \begin{cases} 1 & \text{if } \exists \tau \in (t, t + h] \text{ such that } P_{i, \tau} \text{ depegs} \\ 0 & \text{otherwise} \end{cases}$$
  *Evaluated for horizons: $h \in \{1\text{ hour}, 6\text{ hours}, 24\text{ hours}\}$.*

### C. Unit of Observation & Sampling Frequency
* **Observation Granularity:** **Coin-Hour** (or **Coin-15-minute**).
* **Feature Vector ($\mathbf{X}_{i, t}$):** Constructed strictly using information available at or before timestamp $t$ (zero lookahead bias).

---

## 4. Empirical Framework: Information-Set Comparison

Instead of running an algorithm horse race across many complex models, we compare performance across nested information sets using consistent, interpretable estimators (**Logistic Regression**, **Random Forest**, and **XGBoost**):

| Information Set | Included Features | Hypothesis / Objective |
| :--- | :--- | :--- |
| **$\mathcal{I}_{\text{Market}}$ (Baseline)** | • Rolling price returns ($1\text{h}, 6\text{h}, 24\text{h}$)<br>• Peg deviation ($|P_t - 1|$)<br>• Realized price volatility<br>• Off-chain trading volume & bid-ask spreads<br>• Macro/crypto controls (BTC/ETH returns & volatility) | Establishes the baseline predictive power of publicly available exchange market data. |
| **$\mathcal{I}_{\text{Market+OnChain}}$** | $\mathcal{I}_{\text{Market}}$ **+**<br>• Net wallet outflows / capital flight volume<br>• Large "whale" transaction count & volume share<br>• Token velocity & active address count<br>• DEX pool imbalance (e.g., Curve 3pool / UST pool reserves) | Tests whether on-chain transaction transparency provides leading warning signals before exchange prices reflect stress. |
| **$\mathcal{I}_{\text{Extended}}$ (TDA / NLP)** | $\mathcal{I}_{\text{Market+OnChain}}$ **+**<br>• Persistence landscape summary stats ($L^1, L^2$ norms)<br>• News/social sentiment & volume intensity (if feasible) | **Hypothesis test:** Does topological feature engineering or textual sentiment earn its place by yielding statistically significant OOS gains? |

---

## 5. Validation Protocol & Evaluation Hygiene

1. **Chronological Splitting & Walk-Forward Validation:**
   * **No Random Shuffling:** Random K-fold cross-validation causes massive data leakage in time-series and crisis data.
   * **Purged & Embargoed Rolling-Origin Splits:**
     * Train on pre-crisis / early observations ($t_0 \to t_{\text{train}}$).
     * Embargo window of length $h$ between train and test sets to prevent overlapping prediction labels.
     * Evaluate on subsequent out-of-sample periods containing crisis and stable regimes.
2. **Episode-Aware Accounting:**
   * Distinguish between true independent crisis episodes (e.g., UST failure, USDT May 12 stress, USDC March 2023 depeg) vs. consecutive 15-minute points within the same downward cascade.
3. **Evaluation Metrics:**
   * **Primary:** **PR-AUC (Precision-Recall Area Under Curve)** — sensitive to the rare positive class (depeg events).
   * **Secondary:** **Precision @ Top $K\%$ Risk**, **Recall**, **False Positive Rate (FPR)**.
   * **Calibration:** **Brier Score** and **Calibration Curves** (ensuring predicted probabilities represent real empirical risk).
   * **Supplementary:** **ROC-AUC**.

---

## 6. Midterm Deliverables Checklist

- [ ] **Data Intersection & Coverage Table:**
  - Map which stablecoins are covered in both SNAP ERC-20 and OHLCV market datasets (e.g., UST, USDT, USDC, DAI, BUSD).
- [ ] **Temporal Alignment:**
  - Establish exact common date range (e.g., April 1, 2022 – June 30, 2022).
  - Resample and align timestamps to regular intervals (1-hour or 15-minute grids).
- [ ] **Event & Class Distribution Statistics:**
  - Total coin-intervals ($N$).
  - Count of distinct, clustered depeg episodes.
  - Class balance table ($Y=1$ vs $Y=0$) across $h=1\text{h}, 6\text{h}, 24\text{h}$.
- [ ] **Baseline Implementation:**
  - Feature extraction for $\mathcal{I}_{\text{Market}}$ and $\mathcal{I}_{\text{Market+OnChain}}$.
  - Baseline Logistic Regression and tuned XGBoost/Random Forest benchmarks.
  - Out-of-sample PR-AUC and feature importance comparison.
