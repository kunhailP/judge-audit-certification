<!-- English translation (2026-09-18) of the lock document committed on 2026-09-10; wording of the criteria translated literally. -->
# PROSPECTIVE LOCK v0.5 — 2026-09-10 (document-level methods, pre-registration)

This document is committed before the target's labels are used for any purpose other than pool construction and judging. Prior access is limited to the aggregate counts of judged turns (173; 159 when restricted to MARCO) and
of pairs (15,337). The primary run is executed once; the result is reported regardless of pass or fail.

## 1. Target
TREC CAsT 2019 evaluation topics, **restricted to the portion judged on MS MARCO v1 passages** (CAR and WaPo documents excluded; the range accessible
through the local corpus). query = manually resolved utterance, one judged turn = one query, the 159 turns that have ≥1 MARCO passage with grade≥2.
relevant = grade ≥ 2 (CAsT convention). pool = union of the top-30 of the 4 systems using the turn's judged MARCO passages as the corpus (same rule as 69/75).
Never used as development data (the distribution shift of conversational queries is also an intended test).
Training (classifier, τ, trunc): BEIR legacy 4 pools (unchanged). Judges: Qwen3-8B UMBRELA (position_ids explicit, 1500 characters), Qwen3-Reranker, inverted reranker.

## 2. Fixed methods and execution
- Document level (precision@cutoff, menu 3): pilot 20 turns (cost included), budget {30,45,60,90} in fully judged query equivalents, ε ∈ {0.01, 0.02}, 300 draws,
  documents/query 4 (including the budget exhaustion rule). arms: uniform, weighted (|w|), strat_pilot, active_judge (calibration assumption)+CV, active_cv,
  weighted_cv (8B / reranker / inverted), ai_resid (residual pilot estimate)+CV, ai_robust_0.5.
- Boundary stress: candidate = 2nd place, ε = 0.9×regret, budget {60, 90}, 300 draws, all arms of 81 and 82.
- Auxiliary (continuity): query-level split_t / split_ppi / split_auto (set-F1, look ≤ n/2, 500 repetitions), 3 judges.
- Metrics: J50 (defined in §6, actual judged pairs), wrong (CP 95%), ρ, document variance ratio. Seeds are the fixed values in the scripts.

## 3. Pre-registered criteria
- **P1 validity**: under boundary stress, the type-I error point estimate of every document-level arm ≤ 0.10, CP upper bound ≤ 0.15.
- **P2 unconditional gain**: at ε=0.02, J50(weighted) ≤ 0.6 × J50(uniform) (pass if uniform does not reach and weighted reaches).
- **P3 judge harmlessness**: for all 3 judges, J50(weighted_cv) ≤ 1.05 × J50(weighted) (if not reached, compare ACT at the maximum budget).
- **P4 active inference equivalence**: with 8B, |J50(ai_resid) − J50(weighted_cv)| ≤ 0.10 × J50(weighted_cv).
- **S1 (auxiliary, conditional)**: if ρ of 8B ≥ 0.55, J50(weighted_cv) ≤ 0.9 × J50(weighted) is expected; if ρ < 0.55, no gain is expected. Reported together with ρ.
- **S2 (auxiliary)**: J50 of active sampling under the calibration assumption (active_judge) ≥ 1.2 × J50(ai_resid).
In case of failure, cause analysis in a separate section; the primary numbers remain unchanged.
