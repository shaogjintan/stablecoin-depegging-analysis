# Appendix: Python file inventory

This inventory covers all 76 maintained Python files. Paths are relative
to the repository root. Historical code in archive/, quarantined code in
delete/, virtual environments, caches and immutable artifact source snapshots
are excluded from the maintained inventory.

Files named __init__.py let Python import code from a folder. Most do not
perform analysis themselves. Test files check that the code behaves correctly;
they do not produce the study results.

## Data storage

| Subfolder | Data file name | What it contains |
|---|---|---|
| `raw/` | `hourly_stablecoin.parquet` | Retained hourly reference prices, exchange trading volumes and venue information used to rebuild market features. |
| `raw/` | `dune_onchain.zip`, six coin-specific CSVs | Original on-chain export bundle and the extracted study-coin inputs. |
| `interim/` | `hourly_stablecoin_market_features.parquet` | Derived market features. |
| `interim/` | `hourly_stablecoin_onchain_features.parquet` | Derived on-chain features. |
| `processed/` | `master.parquet`, `depeg_episodes.parquet` | Final hourly modelling dataset and episode records. |

## data_processing

| Python file | Purpose |
|---|---|
| `data_processing/__init__.py` | Allows Python to import this folder as a package. |
| `data_processing/build_dataset.py` | Builds the final modelling dataset and episode list, or checks that a rebuild matches the retained data. |
| `data_processing/build_market_features.py` | Calculates price changes, volatility, peg deviations and trading-volume changes for each coin. |
| `data_processing/build_onchain_features.py` | Reads and checks the Dune exports, then calculates changes in on-chain activity. |
| `data_processing/config.py` | Stores the study coins, file locations, date ranges, depeg rules and training-period settings. |
| `data_processing/dataset_transformation.py` | Creates the prediction labels, determines when predictions are allowed and records when each hourly bar becomes available. |
| `data_processing/feature_schema.py` | Lists the inputs used by the market-only and market-plus-on-chain models. |
| `data_processing/fetch_stablecoin_data.py` | Downloads prices and exchange trading volume and combines them into an hourly market table. |
| `data_processing/matched_sample.py` | Ensures the two feature sets are compared using the same coin-hour observations and labels. |
| `data_processing/merge_and_label.py` | Joins the data sources, identifies depeg episodes and checks feature availability; its command starts the final dataset builder. |
| `data_processing/sample_preparation.py` | Selects usable observations and arranges model inputs and labels for each training and test period. |
| `data_processing/splits.py` | Checks the retained datasets and divides observations into chronological training, calibration and test periods, with gaps to reduce leakage. |

## modelling

| Python file | Purpose |
|---|---|
| `modelling/__init__.py` | Allows Python to import this folder as a package. |

## modelling/evaluation

| Python file | Purpose |
|---|---|
| `modelling/evaluation/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/evaluation/alarm_events.py` | Turns scores above the cutoff into grouped alarms and checks whether a depeg follows each alarm. |
| `modelling/evaluation/exact_threshold.py` | Works out how many alarms and false alarms each possible calibration cutoff would produce. |
| `modelling/evaluation/metrics.py` | Calculates the five performance measures and the counts used to explain them. |
| `modelling/evaluation/run_summary.py` | Combines the separate test-year results into overall performance summaries. |
| `modelling/evaluation/targetability.py` | Finds episodes for which the available observations gave the model a chance to issue an advance warning. |
| `modelling/evaluation/threshold_selection.py` | Selects the lowest alarm cutoff that stays within the calibration false-alarm budget. |

## modelling/models

| Python file | Purpose |
|---|---|
| `modelling/models/__init__.py` | Selects and creates the requested model from the available model types. |
| `modelling/models/base.py` | Provides common model methods and fills missing inputs using medians learned from training data. |
| `modelling/models/baseline.py` | Defines simple constant-probability and persistence models for comparison. |
| `modelling/models/logreg.py` | Defines logistic regression and its extra features, extreme-value clipping and feature scaling. |
| `modelling/models/random_forest_model.py` | Trains a random forest and produces risk scores and feature-importance values. |
| `modelling/models/xgboost_model.py` | Trains XGBoost and produces risk scores, with class weights calculated from training labels. |

## modelling/reporting

| Python file | Purpose |
|---|---|
| `modelling/reporting/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/reporting/benchmarks.py` | Trains and evaluates the simple comparison models using the saved data splits. |
| `modelling/reporting/by_coin.py` | Produces a performance table for each coin using the saved predictions and alarms. |
| `modelling/reporting/calibration_audit.py` | Produces a table showing the chosen cutoffs, calibration false alarms and training sample sizes. |
| `modelling/reporting/coverage_table.py` | Shows how much source data and how many complete feature observations are available for each coin. |
| `modelling/reporting/detection_by_type.py` | Summarises which types of depeg events the models could warn about and actually detected. |
| `modelling/reporting/event_annotations.py` | Adds historical context based on event dates; these annotations do not establish what caused a depeg. |
| `modelling/reporting/event_table.py` | Groups related depegs across coins into events and creates event tables and charts. |
| `modelling/reporting/event_types.py` | Classifies grouped events using their timing, duration and severity. |
| `modelling/reporting/model_interpretation.py` | Creates tables and charts showing model coefficients, feature importance and SHAP values. |
| `modelling/reporting/model_results.py` | Combines the saved runs into overall and yearly model-comparison tables. |
| `modelling/reporting/schema.py` | Keeps the column names and ordering consistent across comparison tables. |

## modelling/robustness_checks

| Python file | Purpose |
|---|---|
| `modelling/robustness_checks/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/robustness_checks/alarm_rules.py` | Loads saved predictions and cutoffs, checks that models use the same observations, and calculates alarm performance. |
| `modelling/robustness_checks/alarm_smoothing.py` | Checks how requiring consecutive hours above the original cutoff changes alarm performance. |
| `modelling/robustness_checks/or_combination.py` | Checks how raising an alarm when either of two models exceeds its original cutoff changes performance. |
| `modelling/robustness_checks/alternative_definitions.py` | Checks how results change under different depeg definitions while keeping the original models and alarm times fixed. |
| `modelling/robustness_checks/episode_definitions.py` | Identifies episodes using alternative price thresholds and minimum durations. |

These alarm-rule checks use the saved final models and original cutoffs; they do not retrain models or choose cutoffs using test results. Their outputs are written to `reports/robustness/alarm_smoothing/` and `reports/robustness/or_combination/`.

## modelling/training

| Python file | Purpose |
|---|---|
| `modelling/training/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/training/calendar_cv.py` | Creates tuning splits in which each validation year is predicted using earlier years only. |
| `modelling/training/logistic_tuning.py` | Trains logistic regression and chooses its regularisation setting using calendar-year validation scores. |
| `modelling/training/parameters.py` | Stores the fixed random-forest and XGBoost settings and the candidate logistic-regression settings. |
| `modelling/training/train_models.py` | Runs practice training, cutoff selection, final refitting and testing, then saves the models and results. |

## modelling/utils

| Python file | Purpose |
|---|---|
| `modelling/utils/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/utils/artifacts.py` | Finds the saved files belonging to each final model run. |
| `modelling/utils/io.py` | Writes JSON records and calculates file checksums to detect changes. |

## modelling/validation

| Python file | Purpose |
|---|---|
| `modelling/validation/__init__.py` | Allows Python to import this folder as a package. |
| `modelling/validation/reference_checks.py` | Checks saved results by recalculating alarms and metrics and examining the fitted preprocessing. |
| `modelling/validation/verify_model_results.py` | Checks that saved models, data splits, cutoffs and performance results are consistent; can also reproduce saved scores. |

## reports

| Python file | Purpose |
|---|---|
| `reports/__init__.py` | Allows Python to import this folder as a package. |
| `reports/reproduce.py` | Regenerates reports from saved models or runs the complete data, training and reporting workflow. |

## tests

| Python file | Purpose |
|---|---|
| `tests/__init__.py` | Allows Python to import this folder as a package. |
| `tests/diagnose_onchain_features.py` | Manually explores on-chain feature values, missing data and differences between coins; not part of model training. |
| `tests/test_alarm_rule_robustness.py` | Checks smoothing, paired populations, AP interpretation and saved-score command outputs. |
| `tests/test_acquisition_retries.py` | Checks how data-download functions handle connection failures without making real downloads. |
| `tests/test_artifact_utilities.py` | Checks that saved run files are located correctly and that file checksums are calculated correctly. |
| `tests/test_coverage_and_event_grouping.py` | Checks data-coverage calculations and the rules for grouping and describing events. |
| `tests/test_data_processing.py` | Checks data-processing commands, feature-file writing and the handling of missing hours. |
| `tests/test_dataset_builder.py` | Checks dataset filenames, timestamp handling and comparisons between rebuilt and retained data. |
| `tests/test_interpretation_shap.py` | Checks that newly calculated SHAP values come from the selected saved model and test observations. |
| `tests/test_labels_and_evaluation.py` | Checks prediction labels, information timing, alarm grouping and cutoff selection. |
| `tests/test_matched_sample.py` | Checks that feature tiers use matching observations and rejects duplicate or inconsistent records. |
| `tests/test_onchain_validation.py` | Checks that invalid Dune exports and conflicting ZIP contents are rejected. |
| `tests/test_per_coin_metrics.py` | Checks performance calculations for individual coins, including cases with no targetable episodes or positive labels. |
| `tests/test_reproduction_workflow.py` | Checks workflow order, output locations and failure handling without running full model retraining. |
| `tests/test_sample_preparation.py` | Checks that only eligible observations are selected and prepared consistently for modelling. |
| `tests/test_saved_run_verification.py` | Checks that the saved-result verifier rejects deliberately altered or inconsistent evidence. |
| `tests/test_split_boundaries.py` | Checks that data splits respect the required gaps and exclude episodes crossing the boundaries. |
| `tests/test_targetability.py` | Checks whether an episode has a valid advance-warning opportunity, including under changed outcome definitions. |
| `tests/test_training_and_calibration.py` | Checks logistic-regression training, calendar-year tuning and alarm-calibration rules. |


