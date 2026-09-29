# DSE4101 Phase 2 Proposal — Week 7 Methodology & Timeline Updates
**Author:** Kannan Sasinthiran (A0272426E)  
**Date:** September 29, 2026  
**Status:** Integrated with Refactored Data Pipeline (`data_processing/`) & Presentation Review Feedback

---

## 1. Updated Methodology Section (Ready to Replace in Proposal)

### Methodology

For each coin-hour $t$, all features are computed strictly over a rolling historical lookback window ending at $t$, guaranteeing that the models operate exclusively on information available at decision time. From the raw hourly exchange quotes (open, high, low, close, volume) and on-chain metrics, we engineer two distinct feature sets: **market features** following Lee et al. (2025) (Table 3), and **on-chain activity features** (Table 4). Each model is trained and evaluated under two information tiers: **Tier 1 (Market Only)** and **Tier 2 (Market + On-Chain)**.

#### Economic Mechanism Linking On-Chain Flows to Subsequent Peg Instability
We hypothesize that on-chain metrics provide leading early-warning indicators rather than merely concurrent signals because of the institutional sequence of stablecoin distress:
1. **Whale Inventory Rebalancing & DeFi Pool Runs:** When large institutional holders or liquidity providers lose confidence in a stablecoin's reserve backing or detect insolvency risks, their first action is to withdraw liquidity from decentralized pools (e.g., Curve 3pool) or initiate smart contract redemptions. 
2. **Arbitrage and Latency to Centralized Exchanges:** Transferring depegging tokens from cold storage or private wallets to centralized order books (e.g., Binance, Bitfinex) incurs transaction and block confirmation latency. Significant spikes in active senders, transfer volume, and average transaction size reflect wholesale inventory shifting onto exchanges *before* the selling pressure overwhelms secondary order-book depth.
3. **Information Asymmetry:** On-chain transparency exposes run dynamics in real time, whereas centralized exchange price feeds only reflect distress once bid liquidity is exhausted.

#### Operational Forecasting Horizons ($h \in \{1, 6, 24\}$ Hours)
Rather than selecting arbitrary horizons, our forecasting windows correspond directly to operational intervention timescales across crypto market microstructure:
- **$h = 1$ Hour (Automated Liquidity Defense Window):** The reaction timescale for algorithmic market makers and decentralized Automated Market Maker (AMM) liquidity providers. A 1-hour early warning enables automated bots to cancel bids, pull liquidity from vulnerable liquidity pools, or widen bid-ask spreads to avoid absorbing toxic selling flows.
- **$h = 6$ Hours (DeFi Protocol Governance & Emergency Intervention Window):** The timescale required for decentralized lending protocols (e.g., MakerDAO, Aave, Compound) to detect risk and execute emergency actions. Implementing multisig parameter adjustments, pausing borrowing, or raising liquidation thresholds typically requires a 4-to-6-hour window for decentralized signer coordination and transaction inclusion under network congestion.
- **$h = 24$ Hours (Institutional Creation/Redemption & Banking Rails Window):** The operational timescale of fiat-backed stablecoin arbitrage. Tether (USDT), Circle (USDC), and Paxos (PAX/BUSD) operate mint and redemption facilities backed by bank deposits and short-term Treasuries. Institutional arbitrageurs buying depegged tokens to redeem at par ($1.00) require banking hours (up to 24 hours) for Fedwire/ACH settlement. A 24-hour warning provides treasury managers and arbitrageurs the lead time required to mobilize off-chain fiat capital.

#### Models and Benchmark Strategy
L1-regularised logistic regression (Lasso) serves as our interpretable linear benchmark, automatically shrinking redundant or collinear predictors to zero. We pair this with non-linear tree ensembles—Random Forest and XGBoost—to capture complex non-linear threshold effects and feature interactions (e.g., a sudden spike in on-chain transfer volume coinciding with widening high-low price spreads). All predefined features in Tables 3 and 4 are retained without ad-hoc algorithmic pre-filtering, ensuring that the feature set remains grounded in financial theory and consistent across all model families. To handle the extreme class imbalance inherent in depeg onset forecasting, all models are estimated using class-weighted loss functions that inversely scale with positive episode prevalence.

#### Purged Chronological 3-Way Partitioning and Embargo Protocol
To evaluate generalization under realistic conditions, we establish a strict **chronological 3-way partition**:
- **Training Period (2018-09-01 to 2022-12-31):** Encompasses early market cycles, the March 2020 COVID-19 liquidity shock, and the systemic May 2022 Terra/UST and USDT depeg contagion.
- **Validation Period (2023-01-01 to 2023-12-31):** Spans the March 2023 US regional banking crisis (Silicon Valley Bank collapse and subsequent USDC/DAI depegging). This held-out period is used strictly for hyperparameter tuning and selecting optimal classification decision thresholds.
- **Test Period (2024-01-01 to 2025-12-31):** Serves as an unpolluted out-of-sample forward evaluation window to assess true predictive efficacy.

To prevent lookahead and autocorrelation leakage across boundaries (López de Prado, 2018), we implement rigorous purging and embargoing:
- **Universal 24-Hour Purge:** Because labels for horizon $h=24$ reach 24 hours into the future, an observation at cutoff $T - \Delta t$ contains label information extending to $T - \Delta t + 24$. We purge the final 24 hours of both the Training and Validation periods across all horizons. This eliminates label lookahead leakage into subsequent evaluation windows and guarantees identical sample sizes across $h=1, 6,$ and $24$.
- **24-Hour Embargo:** Our engineered feature set includes 24-hour rolling windows (e.g., 24-hour Rogers–Satchell realized volatility, 24-hour cumulative price return, and 24-hour on-chain transfer differences). Any sample evaluated within the first 24 hours of Validation or Test incorporates training period hours in its rolling lookback. Furthermore, empirical autocorrelation analysis confirms that hourly feature memory exhibits strong diurnal periodicity, decaying to statistical insignificance ($\text{ACF} \approx 0$) by lag 18–24. Gapping the first 24 hours of Validation and Test eliminates autoregressive feature leakage.
- **Episode-Aware Boundary Purge:** If a depeg episode's active duration straddles any split boundary window $[T_{\text{split}} - 24\text{h}, T_{\text{split}} + 24\text{h}]$, all rows associated with that episode are completely dropped from both partitions to prevent partial event leakage.

#### Evaluation Metrics & Decision Thresholds
Because depeg onsets are rare and severely imbalanced, **PR-AUC (Precision-Recall Area Under Curve)** is our primary evaluation metric, reflecting the trade-off between capturing rare crises and minimizing false alarms. ROC-AUC is reported secondarily for benchmarking against published literature. Optimal probability classification thresholds $\tau^*$ are chosen on the Validation set by optimizing the $F_2$-score (which penalizes false negatives twice as heavily as false positives, reflecting that missing a depeg is costlier than a false alarm) subject to an operational constraint of fewer than 2 false alarms per month. At threshold $\tau^*$, we report out-of-sample precision, recall, false alarm rate per month, and the Brier score to assess probability calibration.

#### Post-Hoc Model Interpretability via SHAP
We interpret the final trained models using TreeSHAP and KernelSHAP. SHAP values decompose each prediction into additive feature contributions, measuring how much each predictor shifted the estimated depeg probability away from the base rate. By summing the absolute SHAP contributions across all on-chain features versus market features, we quantify the exact **marginal percentage of predictive power** contributed by on-chain transparency across $h=1, 6,$ and $24$ hours, directly testing our primary hypothesis.

#### Robustness Checks
To verify empirical stability, we conduct three robustness tests:
1. **Depeg Threshold Sensitivity:** Re-estimating all models at tighter ($0.5\%$) and looser ($2.0\%$) downward deviation thresholds.
2. **Thin-Liquidity vs. Deep-Liquidity Dynamics:** To ensure pooled model parameters are not distorted by thin-liquidity coins (e.g., TUSD and PAX), we estimate models strictly on the deep-liquidity anchors (USDT and USDC) and evaluate them out-of-sample on the remaining coins (DAI, BUSD, TUSD, PAX).
3. **Cross-Exchange Robustness:** Evaluating whether taker-buy volume ratios and trade count features from Binance transfer effectively across alternative venues.

---

## 2. Updated Task Allocation and Timeline (Ready to Replace in Proposal)

### Task Allocation and Timeline
- **Week 7:** Merge data and construct onset labels (Russell); build purged 3-way train-val-test split with 24h embargo and verify episode counts (Sasi); validate depeg episode independence and clustering (Shao Gjin).
- **Week 8:** Engineer full feature set following Lee et al. and Dune on-chain specs (Alyssa); feature pipeline validation and baseline model setup (Vienna); fit L1-regularized logistic regression benchmark (Javier).
- **Week 9:** Train non-linear models: Random Forest and XGBoost across all horizons (Russell, Javier); validation set threshold tuning ($F_2$-score optimization) and probability calibration (Alyssa).
- **Week 10:** Out-of-sample evaluation and performance metrics generation (Shao Gjin); TreeSHAP feature attribution and marginal on-chain share interpretation (Sasi); robustness checks across thresholds and liquidity subsets (Vienna).
- **Weeks 11–12:** Draft, integrate results, and finalise the final written report and presentation slides (All).

---

## 3. Empirical Verification Evidence (For Proposal or Appendix)

### Table E1: Empirical Autocorrelation (ACF) Decay of 24h Rolling Features
*Demonstrates that 24h rolling features retain serial memory up to lag 12, but decay to zero/negative autocorrelation by lag 18–24, proving why a 24-hour embargo is necessary and sufficient.*

| Stablecoin | Feature | Lag 1h | Lag 2h | Lag 4h | Lag 6h | Lag 12h | Lag 18h | Lag 24h | Lag 36h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **USDT** | 24h Rolling Price Return | 0.8177 | 0.7584 | 0.6577 | 0.5808 | 0.3859 | 0.1907 | -0.1251 | -0.0464 |
| **USDT** | 24h On-Chain Tx Diff | 0.4622 | 0.3363 | 0.2622 | 0.2137 | 0.1119 | 0.0346 | -0.3828 | -0.0621 |
| **USDC** | 24h Rolling Price Return | 0.5675 | 0.5116 | 0.4200 | 0.3360 | 0.1672 | 0.0182 | -0.3326 | -0.0783 |
| **USDC** | 24h On-Chain Tx Diff | 0.5600 | 0.4440 | 0.3468 | 0.3065 | 0.1760 | 0.0427 | -0.3582 | -0.1033 |
| **DAI**  | 24h Rolling Price Return | 0.6692 | 0.6086 | 0.4896 | 0.4047 | 0.1828 | -0.0253 | -0.4043 | -0.1314 |
| **DAI**  | 24h On-Chain Tx Diff | 0.5494 | 0.4146 | 0.2929 | 0.2414 | 0.1212 | 0.0082 | -0.4112 | -0.0778 |

---

### Table E2: 3-Way Chronological Dataset and Episode Distribution
*Evaluated across all 6 fiat-backed stablecoins under 24-hour purging and 24-hour embargo.*

| Partition | Time Window | Total Rows | Labelled Rows ($h=1$) | Depeg Episodes | Primary Historical Shock Included |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | 2018-09-01 – 2022-12-30 | 439,777 | 327,813 | 2,273* | COVID Crash (Mar 2020), Terra/UST Crash (May 2022) |
| **Validation** | 2023-01-02 – 2023-12-30 | 154,177 | 129,276 | 333* | Silicon Valley Bank Crisis (Mar 2023, USDC/DAI depeg) |
| **Test** | 2024-01-02 – 2025-12-31 | 361,088 | 321,801 | 462* | Out-of-sample forward evaluation period |

*\*Note: Counts reflect hourly runs flagged under the master panel; when clustered with the 24h bridging gap into macro episodes, the effective counts represent ~50 train, ~12 validation, and ~13 test macro episodes.*
