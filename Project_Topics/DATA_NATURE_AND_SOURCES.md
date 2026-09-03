# Project Data Overview: Stablecoin Depegging Forecasting

This document outlines the data sources, formats, and the exact nature of the data we will be using to build our forecasting model for the May 2022 stablecoin depegging event.

## 1. Nature of the Data

To build a forecasting model (e.g., predicting the probability of a depeg event), we will primarily work with **numerical time-series data**. 

### A. Market Data (Core Forecasting Data)
This is the primary data that will feed our machine learning algorithms. It consists of numbers tracked sequentially over time.
*   **Timestamps:** When the market activity occurred (e.g., `2022-05-09 14:00:00 UTC`).
*   **OHLCV Prices (Floats):** Open, High, Low, and Close prices for assets like UST, LUNA, BTC, ETH, USDC, and USDT.
*   **Volume & Market Cap (Integers/Floats):** The total value and amount of the asset traded.
*   **Why it matters:** Algorithms like XGBoost, Random Forest, or LSTMs require structured numerical matrices to find historical patterns that precede a price collapse.

### B. On-Chain Transaction Data (Network Data)
This data tracks the actual movement of funds between wallets on the blockchain.
*   **Format:** A mix of strings (wallet addresses like `0x3a4...`) and numbers (transaction amounts and timestamps).
*   **Why it matters:** Useful for identifying "whale" behavior (large institutional dumps) that often triggers retail panic. We can extract numerical features from this (e.g., "total volume moved by top 10 wallets in the last hour") to feed the forecasting model.

### C. Sentiment Data (NLP / Textual)
If we incorporate public sentiment (Twitter/Reddit):
*   **Format:** Raw text strings.
*   **Processing required:** We must use Natural Language Processing (NLP) to convert raw text into numerical sentiment scores (e.g., `-1.0` for extreme panic, `1.0` for extreme greed) before it can be used in forecasting.

---

## 2. Recommended Data Sources & Formats

Based on academic literature (including the Lee et al. 2025 paper), here are the best sources to obtain this data:

### A. For Model Training & Backtesting (Static Data)
To train the model on the historical May 2022 crash, we need bulk historical data. **CSV files** are the standard format here as they are easily loaded into Python (Pandas).

1.  **Mendeley Data ("Anatomy of a Run" Package)**
    *   **What it is:** The official replication data for a major academic paper on the crash.
    *   **Format:** Cleaned `.csv` and Parquet files, plus Python scripts.
    *   **Status:** Exists. Highly recommended for pre-processed baseline data.
2.  **Stanford SNAP ERC20-stablecoins**
    *   **What it is:** Over 70 million on-chain transaction records for major stablecoins.
    *   **Format:** Massive `.csv` files.
    *   **Status:** Exists. Best for analyzing network topology and whale movements.
3.  **Binance Data Vision (Historical)**
    *   **What it is:** Binance's official archive of minute-by-minute tick data.
    *   **Format:** `.csv` files bundled in `.zip` archives.
    *   **Status:** Exists. Best for raw, high-frequency OHLCV market data.

### B. For Live Forecasting (Real-Time Data)
Once the model is trained on historical CSVs, it needs a live data feed to make real-time predictions.

1.  **CoinGecko API**
    *   **What it is:** A REST API providing live cryptocurrency market data.
    *   **Format:** JSON responses (converted to numbers in Python).
    *   **Status:** Exists. Easy to use with a generous free tier.
2.  **Binance API**
    *   **What it is:** WebSocket and REST APIs for real-time exchange data.
    *   **Format:** JSON.
    *   **Status:** Exists. Industry standard for high-frequency live data ingestion.

## 3. Team Next Steps
1.  **Data Acquisition:** Download the Mendeley Data package and Binance historical CSVs for May 2022.
2.  **Feature Engineering:** Decide how to structure the CSV data (e.g., 15-minute intervals) and what features (like rolling averages of volume) to feed the model.
3.  **API Setup:** Register for a free CoinGecko API key to test live data ingestion for the final forecasting pipeline.
