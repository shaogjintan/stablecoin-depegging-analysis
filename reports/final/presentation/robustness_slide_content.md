# Slide content: alternative depeg definitions

Use the corrected results in `reports/final/robustness/`.
The earlier `REFIT-CONCLUSIONS/` tables have a superseded recall denominator.

## Slide 1 — Do the conclusions depend on the depeg definition?

**Reference:** a downward deviation of more than 1% lasting at least 2 consecutive hours.

| Parameter | Definitions tested |
| --- | --- |
| Price band below $1 | 0.5%, 1%, 2% |
| Minimum consecutive duration | 1, 2, 3, 4 hours |
| Total combinations | 12, including the reference |

- Compare LogReg, Random Forest and XGBoost across both matched tiers at h6 and h24.
- Retain the trained models, eligible prediction hours, thresholds and alarm timestamps.
- Redefine episodes and recalculate episode recall and false alarms per coin-month. Also report alarm precision and median warning lead.

**Question:** Does the highest-recall model change, and does adding on-chain data still help?

**Visual:** a 3-by-4 definition grid, with the 1%, 2h reference cell outlined.

**Footnote:** Pooled OOS 2021–2025. This evaluates the existing standard-definition system against alternative outcomes. Model training and prediction eligibility retain the reference definition.

## Slide 2 — LogReg usually leads recall across definitions

| Horizon | Tier | LogReg leads alone | LogReg ties for lead | Other models lead |
| --- | --- | --- | --- | --- |
| 6h | Tier 1 | 10/12 | 0/12 | 2/12 |
| 6h | Tier 2 | 9/12 | 2/12 | 1/12 |
| 24h | Tier 1 | 8/12 | 1/12 | 3/12 |
| 24h | Tier 2 | 9/12 | 1/12 | 2/12 |

- LogReg leads alone in **36 of 48** comparisons and shares the lead in **4 more**.
- Most cases where LogReg loses the recall lead occur at the **0.5% band**.
- False-alarm costs vary across models, so a recall leader does not establish an overall winner.

**Finding:** LogReg generally delivers the highest episode recall, although model rankings change under some alternative definitions.

**Visual:** four compact stacked bars showing outright lead, joint lead and another model leading. Each bar represents 12 definitions.

**Footnote:** Leaders refer to episode recall among the three fitted models. Keep each model's original alarms fixed. The 48 comparisons share data and are not independent statistical replications. This check does not evaluate AP sensitivity to alternative hourly labels.

## Slide 3 — On-chain benefits depend on the model and horizon

| Horizon | Model | Tier 2 improves | Tier 1 improves | Recall/false-alarm trade-off |
| --- | --- | --- | --- | --- |
| 6h | LogReg | 0/12 | 11/12 | 1/12 |
| 6h | Random Forest | 8/12 | 0/12 | 4/12 |
| 6h | XGBoost | 0/12 | 9/12 | 3/12 |
| 24h | LogReg | 7/12 | 0/12 | 5/12 |
| 24h | Random Forest | 2/12 | 0/12 | 10/12 |
| 24h | XGBoost | 6/12 | 0/12 | 6/12 |

**Improves:** catches at least as many episodes with no more false alarms, with at least one strict improvement. A trade-off means one tier catches more episodes but raises more false alarms.

- At 6h, market-only features usually give the better recall/false-alarm combination for LogReg and XGBoost.
- At 24h, on-chain features often reduce false alarms, but some definitions also reduce episodes caught.

**Finding:** The broader conclusion holds: on-chain data does not consistently improve performance across models and definitions.

**Visual:** a stacked bar for each model/horizon showing the three outcome counts, with consistent colours for Tier 2 improvement, Tier 1 improvement and trade-off.

**Footnote:** The comparison uses episode recall and false-alarm counts. It does not incorporate AP or conditional lead times into the improvement classification. At the strict 2%, 4h definition, only one episode is targetable at either horizon. Equal recall can therefore reflect very few catches.

## Closing sentence

**LogReg's recall advantage generally persists, while the value of on-chain data remains conditional on the model, horizon and episode definition.**

## Sources

- `reports/final/robustness/run.json`
- `reports/final/robustness/recall_leaders.csv`
- `reports/final/robustness/onchain_effects.csv`
- `reports/final/robustness/metrics.csv`

Reproduce with:

```bash
python -m modelling.robustness_refit \
  --run-dir artifacts/runs/primary/REFIT-LOWEST-NEW \
  --run-dir artifacts/runs/primary/LR-REFIT-LOWEST-NEW \
  --horizons 6 24 \
  --bands 0.005 0.01 0.02 \
  --durations 1 2 3 4 \
  --out-dir /private/tmp/stablecoin-robustness-next
```
