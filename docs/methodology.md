# Methodology and interpretation

Corrected timestamps represent completed hourly bars. Downward depegs cross $0.99 for at least two consecutive hours, with the configured 24h merge gap. Targets are merged episode starts in (origin, origin+h]. Prediction eligibility excludes in-episode/recovery hours; features use information available at the origin.

Outer test years are 2021–2025. Practice fitting precedes the calibration year. The lowest feasible cutoff is chosen by exhaustive distinct-score search, including no-alarm, under grouped validation false alarms / observed eligible coin-months <= 1. Cooldown costs are non-monotone. The final model fits the complete outer training partition, with 25h purge and 24h embargo boundaries, and applies the copied practice cutoff. XGB/RF parameters are fixed; LR calendar CV selects C inside practice training and keeps C for refit.

The primary comparison matches market-only and market+on-chain train, validation and test populations. Headline metrics are episodes caught/targetable, observed false alarms per coin-month and lead time conditional on catches. Average precision is secondary. Calibration exposure counts observed coin-months including partial months. Test false alarms may exceed the calibration budget. Cutoff transfer between practice and final models is a limitation.

Alternative-definition robustness freezes prediction origins, thresholds, models and alarm times. Targetability uses any saved eligible origin in the alternative onset window, not the original labels. These are sensitivity analyses, not independent replications. The earlier superseded robustness denominator must not be used.

Events group different-coin episodes within 48h of the first onset, without chaining. Structural types and date-based historical context are distinct. Context annotations are heuristic and do not establish a depeg's cause; they should be checked during report writing. Coverage tables use each coin's full first-to-last corrected master timestamp span and explicitly name the required source fields. Coverage is not engineered-feature eligibility.

Saved-model coefficients, impurity importance and SHAP describe fitted associations. LR coefficients follow winsorisation and standardisation. RF impurity importance has known model-specific selection biases; neither measure proves incremental causal value. Original source gaps/filling and publication delays cannot be fully reconstructed from the retained master.
