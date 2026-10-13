# DSE4101 Phase 1 Proposal
## Stablecoin Depegging Analysis: May 2022 Stablecoin Crash

**Course:** DSE4101  
**Group:** Digital Currencies Group 1  

---

### Team Members

| Name | Student No. |
| :--- | :--- |
| Alyssa Cheong Jiaxin | `A0283293X` |
| Tan Shao Gjin | `A0281362H` |
| Chin Chen Shao, Javier | `A0272898E` |
| Kannan Sasinthiran | `A0272426E` |
| Russell Ng | `A0213913M` |
| Vienna Fiorella Anong | `A0282319B` |

---

## 1. Project Motivation

Stablecoins have garnered considerable attention for their design to maintain a stable value relative to volatile cryptocurrencies, with the market projected to reach $1.45 trillion by 2035 (Ermey, 2026). Their growing adoption thus heightens the need for early detection of depegging risks given their potential to severely disrupt the financial market, as seen from the Terra-LUNA crash in May 2022 which wiped out almost $45 billion in investor funds (Lee et al., 2023). Our project thus aims to identify drivers of stablecoin crashes and develop models for predicting depegging events. 

However, stablecoin stability is shaped by numerous, highly intercorrelated variables such as peg deviation, on-chain activity, sentiment, and broader crypto market conditions, creating high-dimensionality and multicollinearity that complicate interpretability and violate standard regression assumptions. Stablecoin markets are also dynamic and non-linear due to feedback loops which create abrupt regime shifts that traditional linear, stationarity-assuming approaches struggle to capture. Finally, much of the information affecting stability is unstructured text which traditional methods cannot easily incorporate alongside numerical predictors.

To address these challenges, we aim to integrate dimensionality reduction, topological data analysis (TDA), and NLP to generate informative features:
* **Dimensionality Reduction (PCA & UMAP):** PCA and UMAP handle the high-dimensional, multicollinear on-chain and market variables. PCA yields interpretable linear components, while UMAP's non-linear embeddings better capture local clustering before, during, and after a crash.
* **Topological Data Analysis (Persistent Homology):** To capture the non-linear feedback loops and regime shifts, we apply persistent homology, which has prior application in detecting financial crashes (Gidea & Katz, 2017) without assuming linearity or stationarity. Using sliding windows over various on-chain price and sentiment data, we use persistence landscape statistics as early-warning indicators.
* **Natural Language Processing (NLP):** Sentiment and topic-intensity scores are extracted from social media via transformer-based NLP, feeding into both the TDA embedding and the predictive models directly.
* **Predictive Modeling:** For prediction, we use Random Forest and XGBoost, which are more sample-efficient than deep learning given the limited number of labeled depegging events, while offering interpretability via SHAP values to identify key crash drivers, with logistic regression as a baseline. As a secondary comparison, we benchmark an LSTM to test if modelling sequential dependencies directly yields meaningful gains over feature-engineered tree ensembles.

---

## 2. Key Literature

Our key literature motivates our research question, provides the starting machine learning framework, and shortlists features for our project:

1. **Reserve Backing & Redemption Pressure:**  
   *Yip (2022)* shows that stablecoin characteristics and their reserve backing explain the intensity of redemption pressure faced by different stablecoins during the May 2022 crash. His analysis focuses on post-event explanation, which motivates our project to explore identifying signals of a crash using continuously available market data.
2. **Depegging Prediction Baseline:**  
   *Lee et al. (2025)* predicts depegging events for the highest trading volume stablecoins using daily data. It provides our starting machine learning pipeline (Logistic Regression, Random Forest, XGBoost) and feature set of price and volatility change rates.
3. **Transaction Velocity & Recovery Speed:**  
   *Eichengreen et al. (2025)* finds that transaction velocity is the strongest driver of devaluation, motivating our project to adapt their velocity, redemption, and price recovery speed metrics as inputs into our own classifier.
4. **Coin-Specific Model Suitability:**  
   *Visharad et al. (2025)* shows different models (ARIMAX, LSTM, Random Forest) perform best for different stablecoins, concluding that each stablecoin exhibits unique characteristics that dictate its optimal model. This informs our model selection process by matching specific models to the set of stablecoins in our sample.
5. **Thresholds, Market Cap & Volatility:**  
   *Fantazzini (2025)* uses a simple price threshold and finds that a panel Cauchit model built to handle sudden extreme regime shifts predicts stablecoin failure best, with lagged market capitalization and historical volatility as the strongest predictors. This suggests that coin-specific market data rather than broader economic conditions are better drivers of stablecoin failure, which we consider in our study.

---

## 3. Potential Data Sources

Three core datasets inform our analysis and forecasting of depegging events:

1. **Price Deviations & OHLCV Market Data:**
   * **Sources:** Replication dataset by *Liu, Makarov, and Schoar (2023)* and the *CoinGecko API (n.d.)*.
   * **Role:** Establishes our depegging severity ground-truth, offering high-fidelity market pricing at 15-minute intervals.
2. **On-Chain Transaction & Network Data:**
   * **Source:** Stanford SNAP ERC-20 dataset (*Zhu et al., n.d.*).
   * **Role:** Provides over 70 million records of wallet transactions, enabling us to quantify capital flight and "whale" activity.
3. **News Sentiment & Social Media:**
   * **Source:** CoinDesk API (extracting news metadata and text).
   * **Role:** Integrates public sentiment following preprocessing and Transformer-based NLP, serving as features for predictive modeling.
4. **Contingency Sources:**
   * DeFiLlama API and Reddit archives have been identified as backup sources should primary data access be restricted.

---

## 4. Task Allocation and Preliminary Work Plan

| Week | Milestones | Responsible Member(s) |
| :---: | :--- | :--- |
| **Week 3** | • Complete literature review on ML approaches<br>• Complete literature review on dimensionality reduction<br>• Complete literature review on news/sentiment data<br>• Exploration of available news datasets | Javier<br>Alyssa<br>Vienna<br>Russell |
| **Week 4** | • Complete exploration of stablecoin and cryptocurrency market data<br>• Define baseline ML models and evaluation metrics<br>• Finalise framework for grouping correlated variables<br>• Complete assessment of sentiment extraction methods | Shao Gjin<br>Alyssa<br>Russell<br>Sasinthiran |
| **Week 5** | • Identify data quality issues<br>• Explore integrating dimensionality reduction and feature selection with ML models<br>• Explore integrating unstructured text with numerical predictors<br>• Complete Lee et al. (2025) paper replication and assessment<br>• Finalise conceptual framework | Javier<br>Vienna<br>Russell<br>Shao Gjin<br>Sasinthiran |
| **Week 6** | • Finalise slides for mid-term group presentation | All |

---

## 5. References

* **CoinGecko.** (n.d.). *CoinGecko cryptocurrency data API*. Retrieved from [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)
* **Eichengreen, B., Nguyen, M. T., & Viswanath-Natraj, G.** (2025). Stablecoin devaluation risk. *The European Journal of Finance*, 31(11), 1469–1496.
* **Ermey, R.** (2026). *Believe the hype? Stablecoin is about to blast off—and why you should care*. CNBC. [https://www.cnbc.com/2026/08/05/how-stablecoins-could-grow-to-1point45-trillion-in-the-next-decade.html](https://www.cnbc.com/2026/08/05/how-stablecoins-could-grow-to-1point45-trillion-in-the-next-decade.html)
* **Fantazzini, D.** (2025). Detecting stablecoin failure with simple thresholds and panel binary models: The pivotal role of lagged market capitalization and volatility. *Forecasting*, 7(4), 68.
* **Gidea, M., & Katz, Y.** (2017). *Topological data analysis of Financial Time Series: Landscapes of crashes*. arXiv.org. [https://arxiv.org/abs/1703.04385](https://arxiv.org/abs/1703.04385)
* **Lee, S., Lee, J., & Lee, Y.** (2023). Dissecting the Terra-LUNA crash: Evidence from the spillover effect and information flow. *Finance Research Letters*, 53, 103590. [https://doi.org/10.1016/j.frl.2022.103590](https://doi.org/10.1016/j.frl.2022.103590)
* **Lee, Y. H., Chiu, Y. F., & Hsieh, M. H.** (2025). Stablecoin depegging risk prediction. *Pacific-Basin Finance Journal*, 90, 102640.
* **Liu, J., Makarov, I., & Schoar, A.** (2023). *Anatomy of a Run: The Terra Luna Crash* [Data set]. Mendeley Data.
* **Visharad, I., Kayal, P., & Maiti, M.** (2025). Data-driven insights from technical indicators to predict the stablecoins price. *Finance Research Open*, 1, 100011.
* **Yip, R.** (2022). *An event study on the May 2022 stablecoin market crash*. Hong Kong Monetary Authority Research Memorandum 09/2022.
* **Zhu, J., Akcora, C. G., Kantarcioglu, M., & Gel, Y.** (n.d.). *Stablecoin ERC20 transactions dataset* [Data set]. Stanford Large Network Dataset Collection. [https://snap.stanford.edu/data/ERC20-stablecoins.html](https://snap.stanford.edu/data/ERC20-stablecoins.html)
