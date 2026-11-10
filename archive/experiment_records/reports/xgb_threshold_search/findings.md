# Threshold-policy experiment on corrected XGBoost predictions

Frozen source: reports/xgb_corrected_shap_v1. No models refitted; all policies use the same validation scores, test scores, corrected labels, eligible samples and episode manifest. Budget: 1 FA per observed coin-month, chosen on validation only. Horizons available in this run: 6 and 24 hours; test years 2021-2025.

| Configuration | Horizon | Binary caught | Binary FA | New policy caught | New policy FA |
|---|---|---|---|---|---|
| market_matched | 6 | 3/33 | 212 | 0/33 | 30 |
| market_matched | 24 | 5/34 | 195 | 3/34 | 79 |
| market_onchain | 6 | 2/33 | 170 | 0/33 | 20 |
| market_onchain | 24 | 4/34 | 185 | 3/34 | 54 |
| market_original | 6 | 3/33 | 236 | 0/33 | 29 |
| market_original | 24 | 4/34 | 242 | 2/34 | 133 |

The legacy binary search matched exhaustive lowest-feasible search in all 30 folds. Thus the search algorithm did not change the chosen threshold in these folds, despite monotonicity being unjustified in general. The new policy achieved the same validation episode count as the legacy policy in all 30 folds, but selected fewer validation false alarms (then higher thresholds). That objective/tie-break change produced the observed test trade-off.

The new policy reproduces all six existing corrected summary totals. Binary gives more held-out catches and more false alarms in every configuration. All pooled test false-alarm rates are below 1 per coin-month; pooled compliance does not guarantee each coin or year meets the budget. AP is unchanged because the scores are unchanged.

An exhaustive lowest-feasible policy is an exact way to preserve the old objective without relying on binary-search monotonicity. Any production objective should be predeclared rather than selected using these inspected test outcomes. Existing final-results files and production threshold code were not changed by this experiment.
