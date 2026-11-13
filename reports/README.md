# Report outputs

| Folder | Contents |
|---|---|
| `data_coverage/` | Source and feature availability by coin. |
| `depeg_episodes/` | Episode counts and visualisations. |
| `depeg_events/` | Event membership, structural types and historical context. |
| `model_results/` | Performance, benchmarks and caught-episode case tables. |
| `model_interpretation/` | Logistic coefficients, forest importance and SHAP. |
| `robustness/` | Alternative definitions, warning confirmation and OR rules. |
| `validation_checks/` | Calibration cutoff audit. |

`baseline.json` selects the saved run; `verification.json` records consistency
checks of the model evidence. `reproduce.py` is the execution entry point.
Regenerate with `python -m reports.reproduce` from the repository root.

The calibration audit shows the cutoff chosen in each prior-year calibration
period, its false alarms and observed coin-months. It checks how cutoffs relate
to the calibration budget; it does not promise the same rate on test data.

`*_diagnostics.csv` preserves extra fields behind the five headline metrics,
such as thresholds, exposure and base rates. These support interpretation and
are generated alongside the main comparison tables.
