# Tables v0.1 — unified efficiency metric J50 (2026-09-10)

**J50** = 같은 오류 통제(α = 0.10, look × 순서쌍 Bonferroni) 아래 누적 ACT율이 처음 50%에 이르는 시점의 **실제 인간 판정 (query, document) 쌍 수**
(pilot·학습 query 포함, 실행한 예산 사이는 선형 보간, 최대 예산에서도 50% 미만이면 '미도달'). 잘못된 인증률은 같은 실행의 P(ACT ∧ regret > ε).
블록 A(query 단위, set-F1, 한 query 감사 = pool 전체 판정)와 블록 B·C(문서 단위, precision@cutoff)는 utility가 달라 블록 간 J50을 직접 비교하지 않는다.

## A query-level, set-F1, eps=0.01

**dbpedia-entity, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| split_auto | Mistral-7B | 미도달 | 9,797 | 0.43 | 0.002 |
| split_auto | Qwen3-8B | 미도달 | 9,797 | 0.43 | 0.002 |
| split_auto | Qwen3-Reranker | 미도달 | 9,797 | 0.36 | 0.004 |
| split_auto | inverted | 미도달 | 9,797 | 0.32 | 0.002 |
| split_ppi | Mistral-7B | 미도달 | 9,797 | 0.47 | 0.002 |
| split_ppi | Qwen3-8B | 미도달 | 9,797 | 0.44 | 0.000 |
| split_ppi | Qwen3-Reranker | 미도달 | 9,797 | 0.43 | 0.002 |
| split_ppi | inverted | 미도달 | 9,797 | 0.31 | 0.004 |
| split_t | — | 미도달 | 9,797 | 0.32 | 0.002 |

**dl212223, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| split_auto | Mistral-7B | 5,531 | 13,180 | 0.62 | 0.000 |
| split_auto | Qwen3-8B | 5,477 | 13,180 | 0.62 | 0.000 |
| split_auto | Qwen3-Reranker | 5,531 | 13,180 | 0.62 | 0.000 |
| split_ppi | Mistral-7B | 5,509 | 13,180 | 0.61 | 0.000 |
| split_ppi | Qwen3-8B | 5,281 | 13,180 | 0.64 | 0.000 |
| split_ppi | Qwen3-Reranker | 6,174 | 13,180 | 0.51 | 0.000 |
| split_t | — | 5,483 | 13,180 | 0.63 | 0.000 |

## B doc-level, precision

**dbpedia-entity, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| ai_calib | Qwen3-8B | 미도달 | 2,573 | 0.42 | 0.000 |
| ai_calib | Qwen3-Reranker | 미도달 | 2,573 | 0.39 | 0.003 |
| ai_resid | Qwen3-8B | 미도달 | 2,573 | 0.49 | 0.000 |
| ai_resid | Qwen3-Reranker | 미도달 | 2,573 | 0.45 | 0.000 |
| ai_robust_0.5 | Qwen3-8B | 미도달 | 2,573 | 0.46 | 0.000 |
| ai_robust_0.5 | Qwen3-Reranker | 미도달 | 2,573 | 0.44 | 0.000 |
| strat_pilot | — | 미도달 | 2,573 | 0.37 | 0.000 |
| uniform | — | 미도달 | 2,573 | 0.15 | 0.000 |
| weighted | — | 미도달 | 2,573 | 0.36 | 0.000 |
| weighted_cv | Qwen3-8B | 미도달 | 2,573 | 0.44 | 0.000 |
| weighted_cv | Qwen3-Reranker | 미도달 | 2,573 | 0.41 | 0.000 |

**dbpedia-entity, ε = 0.0138365041741949**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| strat_pilot | — | 미도달 | 1,715 | 0.00 | 0.003 |
| uniform | — | 미도달 | 1,715 | 0.00 | 0.000 |
| weighted | — | 미도달 | 1,715 | 0.01 | 0.010 |
| weighted_cv | Qwen3-8B | 미도달 | 1,715 | 0.00 | 0.000 |

**dbpedia-entity, ε = 0.0169009546669507**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| strat_pilot | — | 미도달 | 2,573 | 0.00 | 0.000 |
| uniform | — | 미도달 | 2,573 | 0.02 | 0.020 |
| weighted | — | 미도달 | 2,573 | 0.00 | 0.000 |
| weighted_cv | Qwen3-8B | 미도달 | 2,573 | 0.00 | 0.000 |

**dbpedia-entity, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| ai_calib | Qwen3-8B | 2,020 | 2,573 | 0.61 | 0.000 |
| ai_calib | Qwen3-Reranker | 2,054 | 2,573 | 0.60 | 0.000 |
| ai_resid | Qwen3-8B | 1,610 | 2,573 | 0.71 | 0.000 |
| ai_resid | Qwen3-Reranker | 1,715 | 2,573 | 0.65 | 0.000 |
| ai_robust_0.5 | Qwen3-8B | 1,556 | 2,573 | 0.69 | 0.000 |
| ai_robust_0.5 | Qwen3-Reranker | 1,628 | 2,573 | 0.64 | 0.000 |
| strat_pilot | — | 2,144 | 2,573 | 0.61 | 0.000 |
| uniform | — | 미도달 | 2,573 | 0.30 | 0.000 |
| weighted | — | 1,990 | 2,573 | 0.62 | 0.000 |
| weighted_cv | Qwen3-8B | 1,569 | 2,573 | 0.67 | 0.000 |
| weighted_cv | Qwen3-Reranker | 1,705 | 2,573 | 0.63 | 0.000 |

**dl212223, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| ai_calib | Qwen3-8B | 1,786 | 3,631 | 0.87 | 0.000 |
| ai_resid | Qwen3-8B | 1,532 | 3,631 | 0.88 | 0.000 |
| ai_robust_0.5 | Qwen3-8B | 1,569 | 3,631 | 0.88 | 0.000 |
| strat_pilot | — | 1,522 | 3,631 | 0.85 | 0.000 |
| uniform | — | 미도달 | 3,631 | 0.45 | 0.000 |
| weighted | — | 1,598 | 3,631 | 0.85 | 0.000 |
| weighted_cv | Qwen3-8B | 1,592 | 3,631 | 0.85 | 0.000 |

**dl212223, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| ai_calib | Qwen3-8B | 1,562 | 3,631 | 0.84 | 0.000 |
| ai_resid | Qwen3-8B | 1,340 | 3,631 | 0.86 | 0.000 |
| ai_robust_0.5 | Qwen3-8B | 1,311 | 3,631 | 0.87 | 0.000 |
| strat_pilot | — | 1,388 | 3,631 | 0.89 | 0.003 |
| uniform | — | 3,207 | 3,631 | 0.61 | 0.003 |
| weighted | — | 1,273 | 3,631 | 0.89 | 0.000 |
| weighted_cv | Qwen3-8B | 1,256 | 3,631 | 0.89 | 0.000 |

**dl212223, ε = 0.0265861701992643**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| strat_pilot | — | 미도달 | 2,421 | 0.00 | 0.003 |
| uniform | — | 미도달 | 2,421 | 0.03 | 0.030 |
| weighted | — | 미도달 | 2,421 | 0.00 | 0.000 |
| weighted_cv | Qwen3-8B | 미도달 | 2,421 | 0.00 | 0.000 |

**dl212223, ε = 0.0390712013741052**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| strat_pilot | — | 미도달 | 3,631 | 0.00 | 0.000 |
| uniform | — | 미도달 | 3,631 | 0.01 | 0.013 |
| weighted | — | 미도달 | 3,631 | 0.00 | 0.000 |
| weighted_cv | Qwen3-8B | 미도달 | 3,631 | 0.00 | 0.000 |

## C 4-policy menu, precision

**dbpedia-entity, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| adaptive | Qwen3-8B | 2,522 | 2,573 | 0.50 | 0.000 |
| adaptive | Qwen3-Reranker | 미도달 | 2,573 | 0.46 | 0.000 |
| oracle | Qwen3-8B | 2,316 | 2,573 | 0.52 | 0.000 |
| oracle | Qwen3-Reranker | 미도달 | 2,573 | 0.48 | 0.000 |
| per_pair | Qwen3-8B | 미도달 | 2,573 | 0.42 | 0.000 |
| per_pair | Qwen3-Reranker | 미도달 | 2,573 | 0.41 | 0.000 |
| static_sum | Qwen3-8B | 2,519 | 2,573 | 0.50 | 0.000 |
| static_sum | Qwen3-Reranker | 미도달 | 2,573 | 0.46 | 0.000 |

**dbpedia-entity, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| adaptive | Qwen3-8B | 1,463 | 2,573 | 0.64 | 0.000 |
| adaptive | Qwen3-Reranker | 1,810 | 2,573 | 0.61 | 0.000 |
| oracle | Qwen3-8B | 1,245 | 2,573 | 0.63 | 0.000 |
| oracle | Qwen3-Reranker | 1,740 | 2,573 | 0.61 | 0.000 |
| per_pair | Qwen3-8B | 1,871 | 2,573 | 0.56 | 0.000 |
| per_pair | Qwen3-Reranker | 1,944 | 2,573 | 0.57 | 0.000 |
| static_sum | Qwen3-8B | 1,525 | 2,573 | 0.64 | 0.000 |
| static_sum | Qwen3-Reranker | 1,865 | 2,573 | 0.61 | 0.000 |

**dl212223, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| adaptive | Qwen3-8B | 1,874 | 3,631 | 0.84 | 0.000 |
| oracle | Qwen3-8B | 1,762 | 3,631 | 0.85 | 0.000 |
| per_pair | Qwen3-8B | 2,789 | 3,631 | 0.55 | 0.000 |
| static_sum | Qwen3-8B | 1,927 | 3,631 | 0.84 | 0.000 |

**dl212223, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| adaptive | Qwen3-8B | 1,560 | 3,631 | 0.85 | 0.000 |
| oracle | Qwen3-8B | 1,540 | 3,631 | 0.84 | 0.000 |
| per_pair | Qwen3-8B | 1,633 | 3,631 | 0.73 | 0.000 |
| static_sum | Qwen3-8B | 1,562 | 3,631 | 0.84 | 0.000 |

## D doc-level, set-F1 (linearised; see caveat)

**dbpedia-entity, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| human_full | — | 4,881 | 6,171 | 0.67 | 0.000 |
| judge_ppi | Qwen3-8B | 4,163 | 6,171 | 0.75 | 0.000 |
| judge_ppi | Qwen3-Reranker | 5,044 | 6,171 | 0.65 | 0.000 |
| weighted_lin_cv | Qwen3-8B | 2,927 | 6,171 | 0.81 | 0.000 |
| weighted_lin_cv | Qwen3-Reranker | 2,233 | 6,171 | 0.82 | 0.000 |
| weighted_lin_cv_bc | Qwen3-8B | 미도달 | 6,171 | 0.50 | 0.000 |
| weighted_lin_cv_bc | Qwen3-Reranker | 5,899 | 6,171 | 0.51 | 0.000 |
| weighted_plugin | — | 1,675 | 6,171 | 0.81 | 0.000 |

**dbpedia-entity, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| human_full | — | 3,680 | 6,171 | 0.81 | 0.000 |
| judge_ppi | Qwen3-8B | 2,962 | 6,171 | 0.82 | 0.000 |
| judge_ppi | Qwen3-Reranker | 3,471 | 6,171 | 0.81 | 0.000 |
| weighted_lin_cv | Qwen3-8B | 2,456 | 6,171 | 0.87 | 0.000 |
| weighted_lin_cv | Qwen3-Reranker | 1,633 | 6,171 | 0.87 | 0.000 |
| weighted_lin_cv_bc | Qwen3-8B | 3,144 | 6,171 | 0.68 | 0.000 |
| weighted_lin_cv_bc | Qwen3-Reranker | 2,876 | 6,171 | 0.67 | 0.000 |
| weighted_plugin | — | 1,543 | 6,171 | 0.87 | 0.000 |

**dl212223, ε = 0.01**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| human_full | — | 3,465 | 8,324 | 0.94 | 0.000 |
| judge_ppi | Qwen3-8B | 3,555 | 8,324 | 0.93 | 0.000 |
| weighted_lin_cv | Qwen3-8B | 2,081 | 8,324 | 0.97 | 0.000 |
| weighted_lin_cv_bc | Qwen3-8B | 2,520 | 8,324 | 0.83 | 0.000 |
| weighted_plugin | — | 2,081 | 8,324 | 0.97 | 0.000 |

**dl212223, ε = 0.02**

| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |
|---|---|---:|---:|---:|---:|
| human_full | — | 3,167 | 8,324 | 0.97 | 0.000 |
| judge_ppi | Qwen3-8B | 3,137 | 8,324 | 0.96 | 0.000 |
| weighted_lin_cv | Qwen3-8B | 2,081 | 8,324 | 0.97 | 0.000 |
| weighted_lin_cv_bc | Qwen3-8B | 2,081 | 8,324 | 0.91 | 0.000 |
| weighted_plugin | — | 2,081 | 8,324 | 0.97 | 0.000 |
