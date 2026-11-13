# Methodology and interpretation

The study asks whether hourly market information anticipates new depeg onsets
and whether Ethereum transfer activity adds predictive value. The primary
comparison is market-only versus market-plus-on-chain models on the same
eligible observations. Forecasts at 6 and 24 hours are primary; 1-hour results
provide a supplementary short-horizon comparison.

Warnings are assessed by the episodes they catch, their lead time and the false
alarms they impose, alongside average precision. A feature can contribute to
a fitted model without improving its operating performance; coefficient, forest
importance and SHAP outputs should therefore be read alongside the detection
and false-alarm results.

Dataset timestamps represent completed hourly bars. Downward depegs cross $0.99 for at least two consecutive hours, with the configured 24h merge gap. Targets are merged episode starts in (origin, origin+h]. Prediction eligibility uses the causal risk mask: it excludes missing/below-band prices and the recovery window following a confirmed qualifying dip, using only information available at the origin. It is not selected simply from retrospective episode IDs.

Outer test years are 2021–2025. Practice fitting precedes the calibration year. The lowest feasible cutoff is chosen by exhaustive distinct-score search, including no-alarm, under grouped validation false alarms / observed eligible coin-months <= 1. Cooldown costs are non-monotone. The final model fits the complete outer training partition, with 25h purge and 24h embargo boundaries, and applies the copied practice cutoff. XGB/RF parameters are fixed; LR calendar CV selects C inside practice training and keeps C for refit.

The primary comparison matches market-only and market+on-chain train, validation and test populations. The five headline metrics are median lead time conditional on true alarms, episode recall (distinct caught/targetable episodes), number of false alarms, false alarms per observed coin-month, and average precision. Calibration exposure counts observed coin-months including partial months. Test false alarms may exceed the calibration budget. Cutoff transfer between practice and final models is a limitation.

Alternative-definition robustness freezes prediction origins, thresholds, models and alarm times. Targetability uses any saved eligible origin in the alternative onset window, not the original labels. These are sensitivity analyses, not independent replications.

Events group different-coin episodes within 48h of the first onset, without chaining. Structural types and date-based historical context are distinct. Context annotations are heuristic and do not establish a depeg's cause; they should be checked during report writing. Coverage tables use each coin's full first-to-last corrected master timestamp span and explicitly name the required source fields. Coverage is not engineered-feature eligibility.

Saved-model coefficients, impurity importance and SHAP describe fitted associations. LR coefficients follow winsorisation and standardisation. RF impurity importance has known model-specific selection biases; neither measure proves incremental causal value. Original source gaps/filling and publication delays cannot be fully reconstructed from the retained master.

## Prediction eligibility and episode targetability

Prediction eligibility is a property of a coin-hour: it satisfies the sample's feature, risk-set, and observed-label rules. Episode targetability is a property of an episode relative to the retained eligible origins and forecast horizon. An episode starting at time s is targetable if the same coin has an eligible origin t with s - h <= t < s. The h-hour endpoint is inclusive; an origin at onset is excluded.

For an episode starting at noon with h6, 06:00 is a qualifying origin and noon is not. A targetable episode need not trigger any alarm. Episode recall is the number of distinct targetable episodes caught divided by the number of targetable episodes; recall is undefined when that denominator is zero. Excluding episodes without an eligible warning opportunity from the recall denominator avoids treating them as model misses. Denominators can differ by horizon or sample, so retain the supporting counts.

For alternative outcome definitions, `modelling/evaluation/targetability.py` uses all retained eligible origins and the alternative onset times, without filtering by the reference y_true labels. It does not regenerate prediction eligibility. Core reference-definition evaluation also has a positive-label-based implementation; their applicable scopes are intentionally distinct.
