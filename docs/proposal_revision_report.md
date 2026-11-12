# Proposal revision report

Reviewed 12 November 2026. Page references count the proposal cover as PDF page 1.

## What needs to change

Proposal revision report

Review date: 12 November 2026 | Scope: the supplied 37-page proposal and the published final-run baseline. Page references below are PDF page numbers, counting the cover as page 1.

**The principal conclusions survive.** LR performance, episode recall, detected episode identities, sample sizes and benchmark results are unchanged. Update the RF/XGBoost numbers, selected interpretation percentages, sensitivity wording and the reproduction description.

| Location in proposal | Required action |
| --- | --- |
| 4.1 (p.10) | Change the h=6 AP/prior range from 3.5-5.5 to approximately 3.4-5.5. |
| 4.3 (p.11) | Change the maximum false-alarm difference from 118 to 123; distinguish detection counts from episode identities. |
| 4.4 (p.12) | Update XGBoost SHAP shares and the largest on-chain contribution bound. RF percentages stay the same at one decimal. |
| 4.6 (p.13) | State that XGBoost Tier 2 improves in 4/12 h=24 definitions; flag the historical no-refit comparison for revalidation. |
| 5.1 / 4.7 (pp.13-14) | Keep the LR headline result, but replace claims that on-chain data never identifies an additional individual episode. |
| 5.2 (p.15) | Use 1.22-1.34 for the relevant pooled h=6 over-budget rates and identify the affected tiers. |
| 6.1-6.3 / 6.5 (pp.16-19) | Correct paths and the market-script name; describe stable volatility and the verified published baseline. |
| Appendix B (pp.25-27) | Replace B2, update B3 coordinates/cells, and regenerate B6 percentages. B4 and B7 can stay. |
| Appendix C (pp.29-36) | Update active artifacts, baseline selection, numerical tests and the archive snapshot; fix misplaced script rows. |

Classification: numerical updates are caused by the new baseline. The episode-identity wording and certain path/script mistakes already existed; they are identified separately rather than attributed to the numerical correction.

## Replacement for Table B2

Appendix B, PDF p.25. Use the table below as the new h=6/h=24 performance table. Counts are exact; AP uses four decimals, rates two, and median lead one. Alarm precision is recomputed from true alarms / all alarms, not rounded episode recall.

| h | Model | Tier | AP | Caught | False alarms | FA / coin-month | Median lead (h) | Alarm precision (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | Logistic regression | 1 | 0.0185 | 12/34 | 331 | 1.05 | 6.5 | 3.50 |
| 24 | Logistic regression | 2 | 0.0183 | 12/34 | 316 | 1.00 | 6.0 | 3.66 |
| 24 | Random forest | 1 | 0.0180 | 8/34 | 267 | 0.84 | 8.0 | 2.91 |
| 24 | Random forest | 2 | 0.0173 | 4/34 | 183 | 0.58 | 12.5 | 2.14 |
| 24 | XGBoost | 1 | 0.0144 | 6/34 | 301 | 0.95 | 6.5 | 1.95 |
| 24 | XGBoost | 2 | 0.0164 | 6/34 | 238 | 0.75 | 9.0 | 2.46 |
| 24 | Persistence | - | 0.0149 | 9/34 | 273 | 0.86 | 10.0 | 3.19 |
| 24 | Constant prior | - | 0.0050 | 0/34 | 0 | 0.00 | - | - |
| 6 | Logistic regression | 1 | 0.0068 | 9/33 | 387 | 1.22 | 4.0 | 2.27 |
| 6 | Logistic regression | 2 | 0.0069 | 7/33 | 398 | 1.26 | 4.0 | 1.73 |
| 6 | Random forest | 1 | 0.0052 | 4/33 | 230 | 0.73 | 3.5 | 1.71 |
| 6 | Random forest | 2 | 0.0049 | 2/33 | 217 | 0.69 | 2.5 | 0.91 |
| 6 | XGBoost | 1 | 0.0056 | 6/33 | 300 | 0.95 | 3.0 | 1.96 |
| 6 | XGBoost | 2 | 0.0043 | 5/33 | 423 | 1.34 | 3.0 | 1.17 |
| 6 | Persistence | - | 0.0103 | 5/33 | 289 | 0.91 | 4.0 | 1.70 |
| 6 | Constant prior | - | 0.0013 | 0/33 | 0 | 0.00 | - | - |

### What actually changes at this displayed precision

h=24: RF Tier 2 has 183 false alarms and 2.14% alarm precision. XGBoost Tier 1 has AP 0.0144, 301 false alarms, rate 0.95, lead 6.5 h and precision 1.95%; Tier 2 has AP 0.0164, 238 false alarms, rate 0.75 and precision 2.46%.

h=6: XGBoost Tier 1 has AP 0.0056, 300 false alarms, rate 0.95, lead 3.0 h and precision 1.96%; Tier 2 has AP 0.0043, 423 false alarms and rate 1.34. Other displayed h=6 values stay the same.

Existing typo: the RF Tier 2 h=6 precision printed as "0/91" should be 0.91%. This is a formatting correction, not a new empirical change. Base rates remain 0.41% at h=24 and 0.10% at h=6.

Sources: reports/model_results/model_comparison_summary.csv; benchmark.csv. Supporting precision fields are also available in model_comparison_summary_diagnostics.csv.

## On-chain comparisons and the warning trade-off

Sections 4.1 and 4.3; Figures B3.1 and B3.2 (PDF pp.10-11 and 26). Refresh the arrow endpoints in B3.1 and replace the relevant false-alarm cells in B3.2. The qualitative classifications stay the same.

| Model / horizon | Caught: Tier 1 to 2 | New FA: Tier 1 to 2 | Conclusion |
| --- | --- | --- | --- |
| LR, h=6 | 9 to 7 | 387 to 398 | Worse |
| RF, h=6 | 4 to 2 | 230 to 217 | Trade-off |
| XGBoost, h=6 | 6 to 5 | 300 to 423 | Worse |
| LR, h=24 | 12 to 12 | 331 to 316 | Better |
| RF, h=24 | 8 to 4 | 267 to 183 | Trade-off |
| XGBoost, h=24 | 6 to 6 | 301 to 238 | Better |

### Replacement wording for Section 4.3

"Adding on-chain features does not consistently improve forecasts. It reduces false alarms at unchanged episode recall only for logistic regression and XGBoost at h=24. For XGBoost, false alarms fall from 301 to 238 at h=24, but rise from 300 to 423 while detections fall from six to five at h=6. Across the six primary comparisons, absolute differences range from zero to four episodes caught and from 11 to 123 false alarms. Pooled AP falls in four of six comparisons."

### Important existing wording error: counts are not identities

The sentence "Neither reveals an additional episode" is too strong. Although LR and XGBoost have unchanged catch counts between tiers at h=24, they do not always catch the same episodes. This was already true in the previous results; the new baseline did not create it.

Example: at h=24, LR Tier 2 detects BUSD on 19 April 2021 but loses the USDC episode on 5 January 2021 relative to Tier 1. Its total stays 12. XGBoost Tier 2 adds TUSD on 19 May 2021 and BUSD on 21 January 2022, while losing two other detections; its total stays six.

Use: "Neither improvement increases the total number of episodes caught, although the identities of the detected episodes can differ." Apply the same distinction in Sections 4.7 and 5.1.

### Small change in Section 4.1

At h=6, fitted-model AP is 3.433-5.484 times constant-prior AP. Change "3.5 to 5.5 times" to "approximately 3.4 to 5.5 times". The broader h=24 description of approximately 3-4 times remains reasonable.

## Feature attribution: what to refresh

Section 4.4 and Figures B5-B6 (PDF pp.12 and 27). RF shares change numerically but all six group labels in Figure B5 stay the same at one decimal. The LR h=24 coefficient exhibit B4 is unchanged. XGBoost B6 does need replacement.

| Attribution statistic | Proposal value | Revised value |
| --- | --- | --- |
| RF on-chain share, h=6 | 22.7% | 22.7% |
| RF on-chain share, h=24 | 20.2% | 20.2% |
| XGB on-chain share, h=6 | 24.68% (pie note) | 24.76% |
| XGB on-chain share, h=24 | 26.01% (pie) | 25.71% |
| XGB market share, h=24 | 73.99% | 74.29% |
| XGB 24h volatility share, h=6 | 34.3% | 34.2% |
| XGB 24h volatility share, h=24 | 19.8% | 19.3% |
| XGB 6h volatility share, h=24 | 13.7% | 14.0% |
| XGB peg-deviation share, h=24 | 12.0% | 12.2% |
| XGB 24h transfer-volume share, h=24 | 3.4% | 3.5% |
| XGB 24h active-receiver share, h=24 | 3.3% | 3.1% |
| XGB 1h transfer-volume share, h=24 | 3.1% | 3.1% |

### Replacement for the XGBoost paragraph in Section 4.4

"For XGBoost, realised volatility remains the leading contributor. At h=6, 24-hour volatility accounts for 34.2% of absolute SHAP magnitude. At h=24, the top three contributors are 24-hour volatility (19.3%), 6-hour volatility (14.0%) and peg deviation (12.2%). On-chain features together account for 24.8% at h=6 and 25.7% at h=24."

Also change "no single feature above 3.5%" to "no single on-chain feature above 3.6%". The largest revised on-chain share is 3.5309%. The wider summary that on-chain features receive roughly 20-26% remains valid.

To redraw B6, normalise each feature's mean_abs_shap by the sum over all 22 features. B5 uses mean RF importance across five Tier 2 forests. Do not compare raw importance magnitudes across model families.

Sources: reports/model_interpretation/random_forest/rf_importance_by_year.csv and xgboost/shap_ranking_xgboost_t2_h6.csv / shap_ranking_xgboost_t2_h24.csv.

## Sensitivity, interpretation and unchanged claims

### Section 4.6: revise the definition-sensitivity paragraph (p.13)

The sentence that LR and XGBoost Tier 2 are both better in "roughly half" of the h=24 definitions is no longer accurate. LR improves in 7 of 12 definitions; XGBoost improves in 4 of 12, with trade-offs in the remaining eight. RF still improves in 8 of 12 h=6 definitions and has trade-offs in 10 of 12 at h=24.

Suggested wording: "The benefit of on-chain data remains definition-sensitive. At h=24, Tier 2 dominates Tier 1 in seven of twelve definitions for logistic regression and four of twelve for XGBoost. At h=6 it never dominates for either model, while RF Tier 2 dominates in eight definitions."

LR remains the sole recall leader in 36/48 comparisons; it is a leader including ties in 40/48. Three detailed leader labels change, even though the sole-leader total does not. Refresh any detailed robustness exhibit from recall_leaders.csv.

| Horizon / band / duration / tier | Old leader | New leader |
| --- | --- | --- |
| 6h / 0.5% / 3h / Tier 2 | LR + RF | LR + RF + XGB |
| 6h / 0.5% / 4h / Tier 1 | LR | LR + XGB |
| 24h / 0.5% / 3h / Tier 2 | LR + XGB | LR |

### No-refit statement: historical evidence, not newly reproduced

The archived no-refit experiments were not rerun under the revised volatility method. Do not present the final sentence of Section 4.6 as a newly verified robustness result. Either label it explicitly as a historical comparison or rerun it under the same current inputs before retaining an unqualified claim.

### Sections 5.1 and 5.2 (pp.14-15)

Keep the LR headline: 12/34 episodes, 6.5-hour lead and about 1.05 false alarms per coin-month for Tier 1 at h=24. Replace "those features never uncover an additional depeg episode" with "they never increase the pooled number of episodes caught in the six primary comparisons, although the caught episode sets can differ."

In Section 5.2, write "both LR tiers and XGBoost Tier 2 exceed the pooled h=6 budget, with rates of 1.22-1.34". XGBoost Tier 1 remains below the budget (0.95). This corrects an existing overgeneralisation as well as updating the upper bound.

## Code, data and reproduction wording

### Section 6.1: distinguish saved evidence from publication outputs (p.16)

Use artifacts/final_run/ for the one retained fitted run, including models, saved partitions, prediction scores and alarms. Use reports/ for published tables and figures, and data/processed/ for its exact input dataset. Fresh reproduction workspaces are created under output/reproduction/. See docs/reviewer_code_organisation.md for replacement prose and the complete appendix table.

### Section 6.2: storage location (p.17)

Change the market-source location from interim/ to **data/raw/hourly_stablecoin.parquet**. Feature tables remain under data/interim/. This corrects an existing path error, rather than a change in the acquisition source.

### Section 6.3: feature builder and numerical method (pp.17-18)

Replace the sentence naming build_onchain_features.py as the constructor of the ten market features with **data_processing/build_market_features.py**. Keep build_onchain_features.py for the twelve transfer-activity features.

Add: "Realised volatility is calculated as the sample standard deviation of hourly returns over complete 6- and 24-hour windows. Each window is evaluated independently using centered, compensated summation. This avoids carrying numerical residuals from earlier windows, returns exactly zero for constant returns and does not round away genuine small variations. RF prediction sums trees in a fixed order."

Clarify the final-data paragraph: "A full reproduction writes its master dataset, episode records and dataset metadata under the chosen workspace's data/processed/ directory. The published baseline uses artifacts/final_run/ with its exact inputs under data/processed/, selected by reports/baseline.json. Historical pre-consolidation inputs are archived."

The retained depeg_episodes_w_event_type.parquet is not regenerated in either stable workspace. Current event annotations/classifications are available in reports/depeg_events/. Do not imply that a fresh full run necessarily creates that extra annotated Parquet.

### Section 6.5: replace the obsolete final paragraph (p.19)

"The published results use artifacts/final_run/ and the canonical data/processed/ inputs selected in reports/baseline.json. Two independent full reproductions in the pinned software environment passed their 90-cell model-score audits and matched the scientific outputs across 490 comparison checks, after accounting for expected differences in run identifiers and workspace paths. Report reproduction regenerates these published tables from the selected saved run; full reproduction rebuilds and retrains into output/reproduction/ without replacing the final run. Duplicate validation workspaces were removed; small comparison records remain under reports/validation_checks/reproducibility/. Superseded report snapshots were removed; promotion and relocation checksum records remain under reports/validation_checks/reproducibility/."

Retain the existing caveat about fresh API acquisitions and Dune access. It concerns reacquisition and is separate from repeatability using retained inputs. Do not claim that every operating system or package version has been validated.

## Appendix C: repository inventory corrections

These are structural/provenance updates, not extra model results. Keep h=1 supplementary in the main analysis even though the newly published audits and coefficient tables now include its rows.

| Appendix location | Update required |
| --- | --- |
| C1, p.29 | Move hourly_stablecoin.parquet from interim/ to raw/. The final processed inputs are now canonical under data/processed/; the artifact folder contains one saved run. |
| C2, pp.29-30 | Describe build_market_features.py as using window-local sample volatility; dataset_metadata.json records the method. Label dune_hourly_transfers.sql as an example, not verified per-coin export SQL. |
| C3, p.31 | List alarm_rules.py, alarm_smoothing.py and or_combination.py under modelling/robustness_checks/. They are currently misplaced under the Reports table on pp.32-33. artifacts.py also reads the selected published baseline. |
| C4, pp.31-33 | Add reports/baseline.json: selects the active run, data workspace and run-manifest hash. Verification refers to the final 90-cell audit. Add validation_checks/reproducibility/ for the retained comparison records. |
| C4, p.33 | Add robustness/alarm_smoothing/ and robustness/or_combination/: each contains summary.csv, by_year.csv, alarm_events.csv and manifest.json. These are generated outputs, unlike the Python scripts. |
| C4, pp.32-33 | Note scope expansions: calibration audit 60 to 90 rows; LR coefficients 360 to 540; RF importance 320 to 480. Added rows are supplementary h1, not additional primary tests. |
| C5, pp.33-34 | Replace the claim that final evidence lives only in h1/ and h6_and_h24/. The active run is artifacts/final_run/. The temporary validation workspaces and superseded old run folders were removed. Give root-level index.json and README.md separate rows. |
| C5, pp.33-34 | index.json and README.md are artifacts-root files, not repeated inside every run. Current SHAP rankings belong under reports/model_interpretation/xgboost/. Add final_run/relocation.json to document historical paths without rewriting trained evidence. |
| C6, p.34 | Add test_numerical_reproducibility.py: independent-window volatility, zero/near-zero behavior, exact new rebuild checks and fixed-order RF prediction. Update workflow-test purpose to include published-baseline selection. |
| C7 / C8, pp.35-36 | Add the reproducibility investigation and stable comparison documents. List reports/validation_checks/reproducibility/: comparison audits, promotion checksums and relocation records. Omit superseded report snapshots. |

## What can stay, and the final editing checklist

### No change required because of this baseline update

Sections 1-2; model architectures and seed 42; the expanding folds, 25-hour purge, 24-hour embargo, labels and eligibility; LR selected C values; the LR performance headline; constant-prior and persistence benchmarks; Section 4.2's main precision/2021 figures; and Section 4.5's episode-concentration conclusions.

Appendix A's sample/episode/targetability exhibits are not affected by this update. B1 settings remain valid, with an optional fixed-order RF prediction note. B4's displayed h=24 LR coefficients and B7's episode-catch matrix can remain: detected episode identities were checked and are unchanged in all 18 model/tier/horizon cells. B5's labels remain correct at one decimal.

Section 4.7's and 5.1's broad conclusions remain valid after correcting the distinction between catch counts and episode identities. Sections 5.3 and 6.4 need no result-driven revision. Section 7 could optionally mention the independent full-run check; do not say the historical execution detail was fully recovered.

### Editing order

1. Replace B2 and update B3/B6. Keep h=6 and h=24 as primary horizons.
2. Apply the numerical changes to 4.1, 4.3, 4.4, 4.6 and 5.2.
3. Correct episode-identity wording in 4.3, 4.7 and 5.1.
4. Replace 6.5's obsolete reproducibility caveat; correct 6.2/6.3 paths and script names.
5. Update Appendix C and mark the old no-refit comparison as historical.
6. Search the edited proposal for stale values: 118, 421, 303, 296, 245, 184, 24.68%, 26.01%, 19.8%, 13.7% and "never uncover". Review matches in context rather than replacing globally.

### Evidence and scope

Reviewed source: dse4101 phase 2 proposal.pdf, 37 PDF pages. The proposal was not edited. Numerical evidence comes from the current reports/ tables, the archived pre-stable tables, final-run saved alarm records and the completed stable-a/stable-b comparison. No external empirical claims or bibliographic references were fact-checked.

The unchanged LR/recall results and the stability check do not by themselves prove identical episode identities; those sets were checked separately. The "no new individual episode" error already existed, as old and new caught sets agree. The no-refit experiment was not rerun.

Useful repository references: docs/stable_reproduction_comparison.md; docs/reproduction.md; reports/baseline.json; reports/model_results/; reports/model_interpretation/; reports/robustness/alt_definitions/. Full numerical precision remains in the CSV files.
