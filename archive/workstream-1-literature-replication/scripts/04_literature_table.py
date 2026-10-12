#!/usr/bin/env python3
"""
TASK 4: Literature comparison table -> outputs/literature_comparison.csv + stdout.
"""
import csv

COLS = ["Paper", "Year", "Method", "Data_Frequency", "Stablecoins", "Key_Finding",
        "Critical_Limitation", "Our_Project_Addresses"]

ROWS = [
    ["Lee, Chiu & Hsieh", "2025",
     "Logistic Regression, Random Forest, XGBoost with SMOTE",
     "Daily",
     "USDT, USDC, BUSD, DAI",
     "BTC volatility is the #1 predictor of stablecoin depegging. Sentiment indicators showed NO significant predictive power.",
     "Daily frequency misses intraday crashes. Random train/test split causes data leakage. No on-chain transaction data. Excluded algorithmic stablecoins (UST).",
     "Hourly prediction granularity. Chronological purged splits. On-chain wallet flow features from SNAP dataset. Includes UST."],
    ["Yip (HKMA)", "2022",
     "Cross-sectional OLS regression (Event Study) with ELV framework",
     "Single cross-section (N=18 stablecoins, 1 observation each)",
     "18 stablecoins including UST, USDC, DAI, USDT",
     "Fair value (ELV) based on reserve quality explains 94% of run pressure variation (R²=0.939). Crypto-collateralized coins faced more run pressure.",
     "Pure post-mortem analysis. Cannot predict in real-time. N=18 cross-sectional observations, not time-series. No deployable prediction model.",
     "Dynamic time-series prediction at hourly intervals. Forward-looking target variable Y(t+h). Deployable ML pipeline."],
    ["Eichengreen, Nguyen & Viswanath-Natraj", "2025",
     "Panel regression with transaction velocity metrics",
     "Daily/Weekly",
     "Multiple stablecoins",
     "Transaction velocity is the strongest driver of devaluation severity.",
     "Explains devaluation severity, not onset timing. No classification model for early warning.",
     "Adapted velocity metrics as input features. Classification framework for onset prediction."],
    ["Fantazzini", "2025",
     "Panel Cauchit model with $0.80 price threshold",
     "Daily",
     "Large panel of stablecoins",
     "Lagged market cap and historical volatility are strongest predictors of stablecoin failure.",
     "$0.80 threshold too extreme for early warning — coin is already dead at 80 cents. Daily frequency.",
     "1% threshold captures early stress. Hourly granularity for actionable alerts."],
    ["Visharad, Kayal & Maiti", "2025",
     "ARIMAX, LSTM, Random Forest comparison",
     "Daily",
     "Multiple stablecoins",
     "Different models perform best for different stablecoins — no single best model.",
     "No on-chain data. Coin-specific model selection not generalizable. No information-set comparison framework.",
     "Unified information-set framework. On-chain features. Consistent estimators across all coins."],
]


def main():
    with open("outputs/literature_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(ROWS)
    print("saved outputs/literature_comparison.csv\n")

    widths = [max(len(str(r[i])) for r in ROWS + [COLS]) for i in range(len(COLS))]
    for i, c in enumerate(COLS):
        widths[i] = max(widths[i], len(c))
    widths = [min(wd, 42) for wd in widths]

    def wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for word in words:
            candidate = word if not cur else f"{cur} {word}"
            if len(candidate) <= width:
                cur = candidate
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(COLS)))
    print("-+-".join("-" * w for w in widths))
    for r in ROWS:
        wrapped = [wrap(str(v), widths[i]) for i, v in enumerate(r)]
        height = max(len(wl) for wl in wrapped)
        for line in range(height):
            cells = [wl[line] if line < len(wl) else "" for wl in wrapped]
            print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)))
        print("-+-".join("-" * w for w in widths))


if __name__ == "__main__":
    main()
