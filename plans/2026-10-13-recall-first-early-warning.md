# Recall-First Early-Warning Rework — Implementation Plan

**Goal**: Turn the depeg models from a *ranking* system (PR-AUC) into an *early-warning alarm* system: **catch as many depeg episodes as possible (minimize false negatives / missed depegs), accepting false alarms** — per the group's decision on 2026-10-13.

**Architecture / Approach**: Keep the frozen data pipeline and the `load_folds` contract untouched. Change only (a) *how we evaluate* (episode-level recall vs alarm-rate curves instead of pooled PR-AUC), (b) *how we tune* (recall-at-alarm-budget objective instead of inner-val PR-AUC), and (c) *how we pick operating points* (thresholds chosen on inner_val to hit a recall target). Existing tuned models are re-scored first so we know what we already have before retraining anything.

**Affected Components & Tech Stack**: `modelling/evaluate.py`, `modelling/train.py`, `modelling/models/xgboost_model.py` (grid only), `modelling/compare_runs.py`, `reports/`; Python 3.12, scikit-learn 1.8, xgboost 3.4.1. No data changes.

---

## What we have already (verified, on `main`)

| Asset | State |
|---|---|
| `master.parquet` 44 cols (Tier 1: 9 market, Tier 2: +12 on-chain logdiff) | ✅ committed |
| `load_folds()` — 5 expanding-window folds, at-risk sample, onset labels | ✅ |
| Models: `baseline` (constant prior), `persistence` (peg stress), `xgboost` (nested-CV tuned) | ✅ trained, saved in `results/models/` |
| Pooled OOS PR-AUC: floor 0.0069 (h=24) · persistence 0.022 · XGBoost 0.0093–0.0102 | ✅ `reports/modelling_comparison.csv` |
| SHAP: volatility + peg deviation = 33–45% of \|SHAP\|; on-chain 6–7% | ✅ 28 figures |
| `evaluate.py`: `select_threshold(criterion="f1")`, `alarm_curve`, per-episode mapping | ✅ (needs recall extensions) |

**Why the h=1 numbers looked like noise (and what to do about it):**
1. Only **49 out-of-sample positive hours** at h=1 (0.02% of 242k test rows). Any ranking metric computed on 49 events has enormous variance — the 0.0011–0.0025 differences are statistically meaningless. It is not that h=1 is unpredictable; it is that **hour-level PR-AUC cannot measure skill on 49 events**.
2. The h=1 label fires at the exact hour price first breaches $0.99 (for ≥2h). The informative build-up signals (volatility, deviation, volume changes) operate over 6–24h windows, so the 1h-ahead signal is intrinsically weak — but an early-warning system does not need hourly precision; it needs to **flag the episode at all**.
3. **Fix under the new objective:** stop scoring h=1 by PR-AUC. Score it by **episode-level recall** — "of the 49 OOS episodes, how many get an alarm within their final 1h window, at what hourly alarm rate?" That is meaningful even with 49 events, and it directly matches "don't miss any depeg."

---

## User Review & Decisions Required

- **D1 — Target operating point.** What recall must we promise, and what alarm rate is acceptable? Options for the team table:
  - (a) ≥90% of episodes caught at ≤1% of hours alarmed (≈2.4 alarms/day/coin)
  - (b) ≥95% caught at ≤2% alarm rate
  - (c) ≥99% caught, alarm rate whatever it costs (report it)
  *(Recommended: present the curve for all three; the team picks per horizon.)*
- **D2 — Tuning objective.** Replace inner-val PR-AUC with **recall@alarm-budget** (recommended) or Fβ with β=5? Recall@budget matches the group's wording exactly; Fβ is the fallback.
- **D3 — Episode-level evaluation.** Approve switching the headline metric from pooled hour-level PR-AUC to **episode-level recall + alarm-rate curve** (recommended; PR-AUC stays as a secondary number).
- **D4 — Terminology.** The report should say **minimize false negatives** (missed depegs), not "reduce true negatives" — true negatives are correct non-alarms, which we *want*. (Flagging so the report uses correct terms.)

---

## Proposed Changes & Task Breakdown

### Phase 1: Recall-first evaluation (no retraining) — `feature/recall-first-eval`
- [ ] **Task 1.1: `select_threshold` recall criteria**
  - **Files**: Modify `modelling/evaluate.py`
  - **Action**: Add `criterion="recall_at_alarm_rate"` (args: `target_recall=0.9`, `max_alarm_rate=None`) and `criterion="f_beta"` (`beta=5`). Selection always on inner-val predictions only.
  - **Verification**: extend `evaluate.py` self-test: synthetic scores must return the threshold that hits the target recall with minimal alarm rate.
- [ ] **Task 1.2: Episode-level alarm analysis**
  - **Files**: Modify `modelling/evaluate.py`
  - **Action**: `episode_alarm_table(preds, episodes, threshold)` → per episode: caught (y/n), earliest catch lead-time, hours alarmed; `episode_recall_curve(preds, episodes)` → recall vs alarm-rate sweep (thresholds 0.001…0.99); missed-episode list.
  - **Verification**: all 49 OOS episodes appear exactly once; recall ∈ [0,1] and non-decreasing as alarm rate grows.
- [ ] **Task 1.3: Re-score existing models under the new lens**
  - **Files**: none (reads `results/predictions/*_pooled_test.parquet`)
  - **Action**: for each existing run (baseline, persistence, xgboost t1/t2 × h=1/6/24) compute the recall vs alarm-rate curve and the operating table (recall at 0.5% / 1% / 2% alarm rate).
  - **Verification**: prints a comparison table; committed as `reports/recall_operating_points.csv`.

### Phase 2: Recall-focused retraining
- [ ] **Task 2.1: Tuning objective flag in `train.py`**
  - **Files**: Modify `modelling/train.py`
  - **Action**: `--tune-metric {pr_auc,f_beta,recall_at_rate}` (+ `--beta`, `--target-recall`, `--max-alarm-rate`), applied to inner-val candidate scoring in `_tune_fit`.
  - **Verification**: smoke run with `--tune-metric recall_at_rate --target-recall 0.9` selects combos by the new metric and logs it.
- [ ] **Task 2.2: XGBoost grid for recall**
  - **Files**: Modify `modelling/models/xgboost_model.py`
  - **Action**: extend `scale_pos_weight` options to `{auto, 10, 100, 500}` (32 combos; still ~15 min/matrix). Optionally add `max_delta_step` to tame probability inflation.
  - **Verification**: grid size printed = 32; smoke fit completes.
- [ ] **Task 2.3: Retrain the 3×2 matrix with the recall objective (background)**
  - **Action**: rerun `train.py --model xgboost --tune --tune-metric recall_at_rate` for all {1,6,24} × {tier 1,2}; baselines re-scored under the same metric.
  - **Verification**: pooled OOS episode-recall table per run; params log shows spread.

### Phase 3: Operating point & comparison
- [ ] **Task 3.1: Threshold application to OOS test**
  - **Files**: Modify `modelling/train.py` (or a new `modelling/alarm_report.py`)
  - **Action**: after each run, choose threshold on inner_val via Task 1.1 and emit the OOS **episode alarm table** + missed episodes to `reports/`.
  - **Verification**: every run produces `alarm_table_{tag}.csv` with all 49 episodes.
- [ ] **Task 3.2: Comparison table upgrade**
  - **Files**: Modify `modelling/compare_runs.py`
  - **Action**: add columns: recall@0.5%/1%/2% alarm rate, episode recall at chosen threshold, alarm rate at target recall.
  - **Verification**: `reports/modelling_comparison_recall.csv` generated from the matrix.

### Phase 4: Documentation & handoff
- [ ] **Task 4.1**: Update `agent.md` + README: recall-first methodology, operating-point table, terminology (false negatives, not true negatives).
- [ ] **Task 4.2 (optional, after Phase 1.3 numbers)**: h=1 rescue — multi-horizon alarm: fire an h=1 alarm when the h=6/24 score passes its threshold (decision after seeing the operating curves).

---

## Verification & Acceptance Criteria
- [ ] `evaluate.py` self-test passes with the new criteria.
- [ ] Episode recall curve covers all 49 OOS episodes; missed-episode list is empty at the team's chosen operating point (or the trade-off is explicitly reported).
- [ ] Retrained matrix: 6 XGBoost runs + 2 baselines re-scored; `reports/modelling_comparison_recall.csv` shows recall and alarm rate per run.
- [ ] No data or `prepare_data.py` changes; existing PR-AUC results remain reproducible.
- [ ] Rollback: all changes on `feature/recall-first-eval` (+ a retrain branch); `results/` and `reports/` regenerate trivially; models re-trainable in ~15 min.

## Risks
1. **h=1 high recall may cost a huge alarm rate** — the curve may show catching 90% requires alarming 5–10% of hours. Honest reporting beats promising; the team picks the operating point (D1).
2. **Threshold overfitting on inner_val** (few positives in later folds) — thresholds chosen on inner_val, all headline numbers OOS-only.
3. **Precision will look terrible by design** — the group must stop quoting PR-AUC as the headline (D3).
4. **Uncalibrated probabilities from scale_pos_weight** — thresholds are relative scores, not probabilities; document, don't fix now.
