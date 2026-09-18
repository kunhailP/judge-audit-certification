<!-- English translation (2026-09-18) of the lock document committed on 2026-09-10; wording of the criteria translated literally. -->
# PROSPECTIVE LOCK v0.6 — 2026-09-10 (confirmation of the λ-calibrated document-level control variate, pre-registration)

Committed before the labels are used for any purpose other than pool construction and judging. Prior access: only the aggregates of the number of queries 200, the number of judged pairs 6,589, and the grade distribution (1:1642, 2:2417, 3:1196, 4:1334).
The primary run is executed once; the result is reported as is. Reflecting the lessons of v0.5, **criteria that are always defined** are placed as the primary criteria.

## 1. Target
ANTIQUE non-factoid QA test set (Hashemi et al. 2020): 200 queries, about 33 judged answers per query (pooled, 4 grades), relevant = grade ≥ 3
(ANTIQUE convention). Never used in development; since the retrieval targets are community QA answers rather than passages, the domain shift is also tested.
pool = union of the top-30 of the 4 systems using the query's judged answers as the corpus (same rule as 69/75/87/89). Training: BEIR legacy 4 pools.
Judges: Qwen3-8B UMBRELA (primary), Qwen3-Reranker, inverted reranker.

## 2. Fixed methods and execution
Document level (precision@cutoff, menu 3): pilot 20 queries (cost included), budget {30,60,90,120,150} in fully judged query equivalents, ε ∈ {0.01, 0.02},
300 draws, documents/query 4 (budget exhaustion rule). arms (81): uniform, weighted, strat_pilot, active_judge(+CV), weighted_cv (coefficient 1),
**weighted_cvl (pilot λ)**, 3 judges; (82): ai_calib, ai_resid, ai_robust_0.5. Boundary stress: 8B and inverted, budget {60,90}, ε 0.01.
Auxiliary: query-level split_t/split_ppi/split_auto (set-F1, 8B, look ≤ n/2=100, 500 repetitions), ρ (67). Seeds are the fixed values in the scripts.

## 3. Pre-registered criteria (at ε = 0.02; "maximum budget" = 150 query equivalents)
- **P1 validity**: under boundary stress (8B and inverted judges), the type-I error point estimate of every arm ≤ 0.10, CP upper bound ≤ 0.15.
- **P2 gain of weighted sampling (always defined)**: at the maximum budget, ACT(weighted) ≥ 1.5 × ACT(uniform). If both reach 50%, J50(weighted) ≤ 0.6 × J50(uniform) is also reported.
- **P3 λ-CV harmlessness**: for all 3 judges, in every cell with budget ≥ 60, ACT(weighted_cvl) ≥ ACT(weighted) − 0.03. If both reach, J50(cvl) ≤ 1.05 × J50(weighted).
- **P4 verification of the defect fix**: with the inverted judge, in every cell with budget ≥ 60, ACT(weighted_cvl) ≥ ACT(weighted_cv) − 0.01 (λ-CV is no worse than the coefficient-1 CV),
  and if, at the maximum budget, ACT(weighted_cv, inverted) < ACT(weighted) − 0.03, record it as "coefficient-1 defect reproduced".
- **P5 active inference equivalence (always defined)**: with 8B, at the maximum budget |ACT(ai_resid) − ACT(weighted_cv)| ≤ 0.05.
- **S1 (conditional, reported)**: report ACT(weighted_cvl) − ACT(weighted) together with the mean ρ of 8B; if ρ ≥ 0.55, a positive increment is expected.
- **S2 (reported)**: the ratio of cumulative ACT of query-level split_ppi (8B) to split_t (look 90), and wrong.
In case of failure, cause analysis in a separate section; the primary numbers remain unchanged.
