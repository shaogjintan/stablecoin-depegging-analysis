# Data Sources Checklist: Stablecoin Depegging (May 2022)

## 1. Official Course Dataset
- [ ] **Stanford ERC20 Dataset**
  - **Link:** [SNAP ERC20-stablecoins](https://snap.stanford.edu/data/ERC20-stablecoins.html)
  - **Task:** Download the `.zip` file, extract it, and review the transaction structure. (Check if `token_transfers.csv` is needed).

## 2. Price Data
- [ ] **Kaggle**
  - **Task:** Search for "UST crash May 2022 dataset" or "Terra Luna historical data".
  - **Action:** Download a daily or hourly dataset covering April - June 2022.
- [ ] **CoinGecko API (Alternative)**
  - **Task:** Check CoinGecko's documentation for fetching historical prices of UST, USDT, and USDC.
- [ ] **Binance Historical Data (Alternative)**
  - **Task:** Check Binance data archives for tick-level data if high granularity is needed.

## 3. News & Sentiment Data
- [ ] **Kaggle (Social Media / News)**
  - **Task:** Search for "Crypto Twitter Sentiment 2022" or "Reddit Crypto 2022".
- [ ] **Academic Sources**
  - **Task:** Look up the "Lee et al. 2025" paper mentioned in the syllabus and check if they linked a public GitHub or Mendeley Data repository for their sentiment dataset.
