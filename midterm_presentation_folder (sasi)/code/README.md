# Midterm Presentation Code Package

This folder contains all the reproducible Python scripts, configuration, and execution tools required to regenerate the empirical analysis, figures, tables, and presentation artifacts from scratch.

---

## Quick Start

To regenerate all 4 presentation figures and 5 analytical tables:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
bash run_pipeline.sh
# or directly:
python3 07_coin_filter_and_sensitivity.py
```

All figures will be populated into `../figures/` (300 DPI, white background) and all tables into `../tables/`.

---

## Pipeline Scripts

| File | Purpose | Key Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- |
| `07_coin_filter_and_sensitivity.py` | **Core Presentation Engine**: 2-stage filtration funnel, market cap ranking, 10x9 pooled sensitivity matrix, sweet spot detection (1.0%, 2h), 71-episode Gantt timeline, class balance computation, and defense notes. | `../data/hourly_price.parquet` | Figures 01-04 (`../figures/`), Tables (`../tables/`), Notes (`../notes/`) |
| `01_fetch_all_data.py` | **Data Ingestion Pipeline**: Pulls historical hourly OHLCV from Binance, CoinGecko, and DefiLlama across all 24 stablecoins (2020–2024). | `config.py`, Exchange APIs | `../data/hourly_price.parquet` (857,221 rows) |
| `02_build_onchain_features.py` | **On-chain Liquidity Extractor**: Extracts hourly pool liquidity distributions and imbalances (Curve 3pool, stETH). | DefiLlama / Curve APIs | `../data/hourly_onchain_features.parquet` |
| `04_merge_and_label.py` | **Labeling & Feature Merging**: Merges price series with on-chain metrics and computes forward depeg labels ($h=1, 6, 24$). | Hourly prices + On-chain features | `../data/master_hourly_dataset.parquet` |
| `config.py` | **Universe Registry**: Coin metadata, Coingecko IDs, Binance symbol mappings, and date boundaries. | — | Configuration dictionary |
| `run_pipeline.sh` | **Automated Bash Runner**: One-click shell script to execute the presentation pipeline cleanly. | — | Console logs & artifacts |
| `requirements.txt` | **Dependencies**: `pandas`, `numpy`, `matplotlib`, `pyarrow`, `requests`, `scipy`. | — | Python environment |

---

## Data Specifications

- **Universe**: 6 Major Fiat-Backed Stablecoins (`USDT`, `USDC`, `BUSD`, `DAI`, `TUSD`, `PAX`) representing **$232.5B** in cumulative peak market capitalization and **340,378** hourly observations.
- **Academic Sweet Spot**: Drop $\ge 1.0\%$ ($\le \$0.99$), Sustained $\ge 2\text{ hours}$ ($N = 71$ episodes).
- **Format**: Parquet with zstd compression for ultra-fast load times.
