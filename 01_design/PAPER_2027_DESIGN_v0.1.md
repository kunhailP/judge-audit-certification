# 2027 Paper Design v0.1 — Auditing Deployment Decisions Under Non-Neutral AI Judges

Status: **design draft (2026-09-10), one prototype simulation attached**
Purpose: reframe the current planner as a deployment-certification problem for the era of
"human audit + possibly biased AI judge", and turn it into a TMLR paper that is ahead of its
time by 2027 standards.

## 0. One-sentence thesis

> When an AI judge is not neutral with respect to the candidate policies, the amount of human
> auditing required to safely certify a deployment decision is determined not by the judge's
> overall accuracy but by its **policy-correlated error in the decision-relevant region**. We
> propose a sequential audit planner that measures and controls this quantity.

The "recalibration vs selection" comparison from the earlier draft is demoted to a secondary result; the thesis above is placed at the center.

## 1. Why this is ahead of its time in 2027

- The 2024–2026 PPI family (Oosterhuis KDD'24, Chatzi NeurIPS'24, Gligorić–Zrnic–Candès'24,
  Kilian NeurIPS'25, Sfyraki–Wang ICLR'26) treats the judge as a **fixed predictor** and
  corrects its bias with a rectifier. It does not address the cost and risk when the judge's
  error *depends on which candidate is being evaluated*.
- The IR evaluation literature (Balog et al. 2025 "Rankers, Judges, and Assistants") showed
  empirically that LLM judges systematically favor LLM-based rankers. In 2027 search and RAG
  pipelines, the candidates (reranker, LLM cutoff) and the judge come from the same model
  family. That is, **a non-neutral judge becomes the default**.
- The structure unique to this repository: candidate policies share the same ranked pool and
  differ only in the cutoff position. Hence the judgments that affect the decision are localized
  to the **disagreement region** between the two cutoffs. We can formalize what happens when
  non-neutrality is concentrated exactly in this region.

## 2. Problem setting

- Query `q` from the target distribution `P_T`, a document ranking over the shared pool, and policy `m ∈ M` with cutoff `k_m(q)`.
- True relevance `r(q,d)`, true utility `u_m(q)`, AI judgment `r̂(q,d)`, predicted utility `û_m(q)`.
- Human audit: full judgment at the query level (current) or at the pair level (extension).
- Decision: after selecting `m̂`, certify `L = max_j μ_j − μ_m̂ ≤ ε` / MORE / ABSTAIN.
- Risk: `P(ACT ∧ L > ε) ≤ α` (at every look, anytime-valid).

### Definition of judge non-neutrality

For a policy pair `(j, m)`, the paired prediction error is `η_{jm}(q) = (u_j − u_m)(q) − (û_j − û_m)(q)`.

- **Neutral**: `η_{jm}` is independent of membership in the disagreement region `Δ_{jm}(q) = R_j(q) △ R_m(q)`.
- **Non-neutral (policy-correlated)**: `E[η_{jm} | d ∈ Δ_{jm}] ≠ 0`, or its variance is concentrated in the region.

Key quantity: **decision-relevant agreement** `ρ_{jm} = corr(u_j − u_m, û_j − û_m)`.
The effective-sample-size gain of the PPI++ paired certificate is `1/(1 − ρ_{jm}²)`, and it is
decoupled from overall judge accuracy. (Prototype: accuracy 0.95→0.94 while ρ 0.82→0.52.)

## 3. Method: three layers of certificates

1. **Paired PPI++ selection certificate.** Replace the currently bootstrapped `T×M` utility
   matrix with the PPI++ estimate of paired differences (with λ tuning). Since the rectifier
   corrects judge bias, validity is preserved, and the cost is determined by `ρ`. The
   anytime-valid version applies the confidence sequence of Kilian et al. to the paired differences.
2. **Judge neutrality diagnostic (neutrality audit).** From a small human audit, estimate the
   region-conditional mean and variance of `η_{jm}` and compute a lower bound on `ρ̂_{jm}`. If this
   value is low, the planner reports **in advance** that "with this judge, certification cannot be
   cheaper than human-only". Whether to adopt the judge at all becomes itself a certified decision.
3. **Region-local audit (pair-level).** Allocate human judgments not to entire queries but to the
   documents in the disagreement region. When utility is determined by the region alone (the
   precision family, and recall under a fixed `nG` assumption), the pair cost is computed exactly;
   when `nG` is required, as with set-F1, the PPI correction cost for `nG` is added. This
   difference is itself a result (connects to the earlier "difficulty reversal in recall" observation).

## 4. Candidate theorems

- **Theorem A (cost separation).** Under a neutral judge, the expected stopping time of the paired
  PPI certificate is a function only of `ρ_{jm}` and the gap `μ_j − μ_m`; overall accuracy `acc`
  enters only through `ρ`. Under a non-neutral judge, `ρ` can be made arbitrarily low at the same
  `acc` (constructive counterexample).
- **Theorem B (validity preservation).** As long as the rectifier is estimated from the human audit,
  non-neutrality does not increase `P(ACT ∧ L>ε)`. In contrast, a judge-only decision without a
  rectifier certifies wrongly with probability 1 when the bias direction is opposite to the gap
  (observed in the prototype).
- **Theorem C (localization).** If utility is additive over the region, pair-level auditing yields the
  same certificate as query-level auditing at a judgment cost of the ratio `|Δ_{jm}|/|pool|`.

## 5. Prototype results (04_code/proto/80_judge_bias_sim.py, 2026-09-10)

Policy A = fixed cutoff 8, B = a mixture of the F1-optimal cutoff and a noisy cutoff. On top of a
random error rate `e`, the judge misjudges "documents retrieved only by B" as relevant with
probability `bias` (policy-correlated bias).
α=0.1, ε=0.01, looks {10,…,250}, 200 replications.

### Gap +0.011 (B marginally better, at the ε boundary)

| e | bias | overall accuracy | ρ | human-only T | PPI T | PPI wrong certificate |
|---|---|---|---|---|---|---|
| 0.05 | 0.0 | 0.950 | 0.82 | 144 | **53** | 0.09 |
| 0.05 | 0.3 | 0.938 | 0.52 | 146 | 106 | 0.06 |
| 0.05 | 0.6 | 0.925 | 0.28 | 141 | 123 | 0.08 |
| 0.15 | 0.0 | 0.850 | 0.43 | 131 | 108 | 0.10 |

A policy-correlated bias that lowers overall accuracy by only 1–2.5 points almost entirely erases the PPI savings (63%).

### Gap −0.017 (A better, judge favors B)

| e | bias | judge-only wrong certificate | human-only | PPI |
|---|---|---|---|---|
| 0.05 | 0.0 | 0.00 | 0.04 (T 89) | 0.03 (T 30) |
| 0.05 | 0.3 | **1.00** | 0.03 (T 81) | 0.06 (T 71) |
| 0.15 | 0.3 | **1.00** | 0.03 (T 90) | 0.05 (T 81) |

The judge-only method deploys wrongly 100% of the time even at accuracy 0.94. Both methods with
a rectifier stay within α. (Because of the normal approximation at the T=10 look, some cells sit
at the boundary at 0.10–0.125 — in the main experiment, replace with a t-distribution or a
confidence sequence.)

## 6. Experimental program

1. **Modernize the retrieval stack (mandatory prerequisite).** Rebuild the candidate pool with
   Qwen3-Embedding 0.6B (+4B Pod) and a reranker. Keep the existing pool. Re-measure the gap
   structure on the 13 collections + android. Report the result whichever way it goes.
2. **Judge-error measurement testbed.** BEIR sparse qrels cannot distinguish judge "errors" from
   qrels holes. Use deeply judged TREC DL 2019–2021 (+ the parallel human/synthetic qrels of
   LLMJudge / TREC RAG 2024) as primary. Prepare 2–3 LLM judges (from different model families).
3. **Measuring non-neutrality.** Measure the judge's region-conditional bias of `η` when a reranker
   policy is on the menu. Compare a judge from the same model family vs a judge from a different model family.
4. **Certificate comparison.** Human-only (current planner, defect-corrected version) / paired PPI++ /
   anytime PPI / Oosterhuis PPI·CRC fixed-budget / Chatzi rank-set / confidence-driven sampling /
   SELECT-LLM heuristic. Metrics: wrong-certificate rate, ACT rate, number of human queries, number
   of pairs, judge inference cost.
5. **Local audit.** Cost and validity of pair localization under 3 utilities (precision@cut, recall,
   set-F1). Whether the conditions of Theorem C actually separate in practice.
6. **Prospective lock.** Lock 1 new collection + 1 new judge and run a single primary run.

## 7. Kill / downgrade criteria

- If all gaps between policies on the modern pool are `≫ ε` (selection is trivial), extend the
  menu with reranker variants; if it is still trivial, report "selection certification is
  unnecessary in modern retrieval" as the result.
- If the measured `ρ` is high regardless of the judge model family, scale down the non-neutrality
  thesis and demote the paper to a PPI application paper.
- If the local audit reduces cost under no utility, delete Theorem C.

## 8. Fixing existing defects (prerequisite, unchanged)

The endpoint check in the 40_planner_replay.py recalibration certificate, the validity of the LOO
bootstrap, the per-look resampling in 50_simulation.py, and the interval reporting in
70_prospective_run.py. These remain as they are even after the judge is introduced.

## 9. Kickoff record (2026-09-10, branch `2027-nonneutral-judge`)

### 9.1 Confirmed facts (synthetic data, `04_code/lib/certificates.py` self-test)

- **Endpoint-check defect reproduced**: in a single-query example, the endpoint spread is 0.000 while
  the exhaustive breakpoint check gives spread 0.167.
  The v0.4 recalibration certificate checks every gate breakpoint inside the interval (`recal_spread`).
- **Winner's curse in candidate selection**: if the candidate is chosen on the same data and only the
  M−1 comparisons are corrected, the per-look risk is 0.049 (budget 0.020) even with an exact
  t-interval. Correcting all ordered pairs M(M−1) simultaneously gives 0.014. Proposition 1 of v0.3
  held only when "the candidate is fixed".
- The PPI++ paired UCB is also valid at 0.006 under the same simultaneous correction.

### 9.2 Intervals for the P4 prospective result (`71_prospective_report.py`, no re-run)

| decision | ACT | wrong | wrong rate [CP 95%] | wrong given ACT [CP 95%] |
|---|---|---|---|---|
| recalibration | 22/50 | 0 | 0.00 [0.000, 0.071] | 0.00 [0.000, 0.154] |
| selection | 40/50 | 4 | 0.08 [0.022, 0.192] | 0.10 [0.028, 0.237] |

### 9.3 Experiments in progress (this environment: RTX 3090, 256 CPUs)

- `61_build_pool.py`: candidate pools for both the legacy and modern stacks on nfcorpus / scifact /
  arguana / cqadupstack-android + Qwen3-Reranker P(yes) judge proxy.
- `62_pool_compare.py`: go/no-go #1 — whether the gap structure survives on the modern stack.
- `63_planner_v2.py`: comparison of loo_boot(v0.3) / split_t / split_ppi certificates + judge diagnostics
  (overall accuracy, ρ, error rates inside and outside the disagreement region).
- `64_trecdl_pool.py`: fully judged TREC DL 2019/2020 pool (go/no-go #2 testbed).

### 9.4 Go/no-go #1 result — decision structure survives on the modern stack (`05_results/pool_compare/`)

LODO (train on 3, evaluate on 1), 50 replications, set-F1, ε_cal=0.005, ε_sel=0.01.
modern = legacy with msmarco-MiniLM replaced by Qwen3-Embedding-0.6B (the 4-feature classifier is unchanged).

| collection | stack | pool recall | best F1 | best policy | gap 1st–2nd | cal ok@10 | sel ok@10 | sel ok@50 |
|---|---|---|---|---|---|---|---|---|
| android | legacy | 0.852 | 0.315 | ad_probe | 0.008 | 0.52 | 0.88 | 0.94 |
| android | modern | 0.865 | 0.356 | glob_probe | 0.023 | 0.26 | 0.86 | 0.84 |
| scifact | legacy | 0.963 | 0.497 | glob_probe | 0.027 | 0.30 | 0.80 | 0.84 |
| scifact | modern | 0.957 | 0.525 | glob_probe | 0.029 | 0.32 | 0.82 | 0.96 |
| nfcorpus | legacy | 0.308 | 0.148 | ad_probe | 0.001 | 1.00 | 0.76 | 0.98 |
| nfcorpus | modern | 0.321 | 0.159 | ad_probe | 0.003 | 0.88 | 0.84 | 0.88 |
| arguana | legacy | 0.986 | 0.163 | glob_probe | 0.041 | 0.30 | 0.90 | 1.00 |
| arguana | modern | 0.992 | 0.166 | glob_probe | 0.043 | 0.30 | 0.84 | 1.00 |

Verdict: **GO.** Retrieval performance rises (android F1 0.315→0.356), but the gaps between policies
remain on the scale of ε (0.003–0.043), and on android the optimal policy itself changes from
ad_probe to glob_probe. That is, even with a strong retriever, "which cutoff rule to deploy" remains
a decision that requires auditing.
The recalibration decision tends to become harder on modern (android cal ok@10 0.52→0.26).
Limitation: since the 4-feature classifier is unchanged, the Qwen3 signal entered only the pool
construction. A "fully modern" variant with qwen3e_cos as a feature is follow-up work.

### 9.5 Planner v2 results — comparison of certificate variants (`05_results/planner_v2/`, 50 replications, α=0.1)

Methods: `loo_boot` (v0.3 as is), `loo_sim` (LOO + simultaneous correction over all ordered pairs), `split_t` (train/validation split + t),
`split_ppi` (split_t + Qwen3-Reranker judge PPI++); recalibration: `recal_ep` (v0.3 endpoints), `recal_bp` (exhaustive breakpoints),
`recal_bpx` (breakpoints + exact median interval), `recal_bpxu` (+ uniform bootstrap upper bound).

**Selection decision (ACT rate / wrong-certificate rate)**

| stack | collection | loo_boot | loo_sim | split_t | split_ppi |
|---|---|---|---|---|---|
| legacy | android | 0.54 / 0.08 | 0.46 / 0.04 | 0.10 / 0.00 | 0.14 / 0.00 |
| legacy | scifact | 0.74 / 0.08 | 0.70 / 0.06 | 0.12 / 0.00 | 0.14 / 0.02 |
| legacy | nfcorpus | 0.78 / 0.00 | 0.62 / 0.00 | 0.24 / 0.02 | 0.38 / 0.02 |
| legacy | arguana | 0.96 / 0.04 | 0.94 / 0.02 | 0.34 / 0.02 | 0.28 / 0.02 |
| modern | android | 0.72 / **0.12** | 0.62 / 0.08 | 0.12 / 0.02 | 0.12 / 0.02 |
| modern | scifact | 0.88 / **0.12** | 0.80 / 0.10 | 0.24 / 0.00 | 0.24 / 0.04 |
| modern | nfcorpus | 0.88 / 0.04 | 0.72 / 0.02 | 0.26 / 0.02 | 0.30 / 0.02 |
| modern | arguana | 0.88 / 0.04 | 0.82 / 0.04 | 0.28 / 0.00 | 0.32 / 0.00 |

- The v0.3 certificate (`loo_boot`) exceeds α on two collections of the modern stack (0.12, MC SE 0.04).
  Simultaneous correction (`loo_sim`) lowers this, but the LOO dependence remains.
- The valid split certificates drop the ACT rate to 0.10–0.38. **Part of v0.3's efficiency came from validity violation.**
  This is the empirical reason a "valid certificate that permits data reuse" is needed.
- The gain from PPI is marginal. The paired-difference ρ of the judge (reranker≥0.5) is only 0.04–0.27.

**Recalibration decision (ACT rate / wrong-certificate rate)**

| stack | collection | recal_ep (v0.3) | recal_bp | recal_bpx | recal_bpxu |
|---|---|---|---|---|---|
| legacy | android | 0.30 / 0.00 | 0.12 / 0.00 | 0.06 / 0.00 | 0.00 / – |
| legacy | nfcorpus | 0.92 / 0.00 | 0.70 / 0.00 | 0.62 / 0.00 | 0.00 / – |
| modern | arguana | 0.32 / **0.14** | 0.30 / **0.14** | 0.20 / 0.08 | 0.14 / 0.08 |
| modern | nfcorpus | 0.80 / 0.00 | 0.48 / 0.00 | 0.44 / 0.00 | 0.00 / – |

- The endpoint check inflates the ACT rate by 2–5x (android 0.30 vs 0.06). The recal ACT of 0.44 in v0.3 P4 is overstated for the same reason.
- The bootstrap median interval fails at T=10 (arguana modern error 0.14 → 0.08 with the exact interval).
- If the uncertainty of the sample utility curve is also folded into the upper bound (`bpxu`), ε_cal=0.005 is almost never certifiable within 90 cases.
  → A decision is needed: either reset ε_cal or demote recalibration to a secondary result.

**Judge diagnostics (pair accuracy / error rate inside the decision region / outside)**

| collection | legacy | modern |
|---|---|---|
| scifact | 0.91 / 0.27 / 0.09 | 0.90 / 0.28 / 0.10 |
| arguana | 0.82 / 0.30 / 0.17 | 0.82 / 0.30 / 0.17 |
| android | 0.61 / 0.70 / 0.38 | 0.56 / 0.75 / 0.43 |
| nfcorpus | 0.87 / 0.15 / 0.13 | 0.86 / 0.15 / 0.14 |

**Judge errors are concentrated 2–3x in the policy disagreement region.** This is the first real-data evidence for the §2 thesis.
However, since BEIR qrels holes are mixed into the "errors", this must be re-measured on the fully judged TREC DL pool (in progress).
On android the reranker judge is effectively random (0.56–0.61): duplicate-question relevance is a different task from passage relevance.

Caution: adding method variants changes the rng consumption order, so the `loo_boot` numbers fluctuate by ±0.04 between runs.
The final experiment must use an independent rng stream per method.

### 9.6 Go/no-go #2 — judge diagnostics on the fully judged TREC DL pool (`05_results/planner_v2/trecdl`, `05_results/ppi_gain`)

pool = all NIST-judged passages (dl2019 43 queries · 9,260 pairs, dl2020 54 queries · 11,386 pairs), relevant = grade≥2,
judge = Qwen3-Reranker-0.6B P(yes)≥0.5 (pair accuracy 0.70 / 0.66). Policy parameters are trained on the other year (LODO).

| collection | policy pair | ρ | error inside region | error outside region | ESS gain (T=10) | theory 1/(1−ρ²) |
|---|---|---|---|---|---|---|
| dl2019 | glob vs trunc | 0.70 | 0.33 | 0.29 | 1.76 | 1.94 |
| dl2019 | glob vs ad | 0.44 | 0.32 | 0.30 | 1.31 | 1.23 |
| dl2019 | trunc vs ad | 0.68 | 0.35 | 0.29 | 1.68 | 1.85 |
| dl2020 | glob vs trunc | 0.27 | 0.52 | 0.24 | 1.18 | 1.08 |
| dl2020 | glob vs ad | **0.01** | **0.62** | 0.27 | **1.08** | 1.00 |
| dl2020 | trunc vs ad | 0.63 | 0.42 | 0.32 | 1.75 | 1.66 |

- **With the same judge, within the same collection**, the PPI gain ranges from 1.0x to 1.75x depending on the policy pair.
  What separates them is not overall accuracy but whether the errors are concentrated in that pair's disagreement region
  (dl2020 glob vs ad: inside 0.62 vs outside 0.27 → ρ≈0 → no gain).
- On dl2019 the errors inside and outside the region are similar (0.32 vs 0.30) and ρ is 0.44–0.70, so there is a gain.
  Accuracy is nearly identical: dl2019 0.70, dl2020 0.66. → Real-data support for Theorem A.
- On BEIR (qrels holes) the same judge had ρ ≤0.27. This is consistent with the interpretation that holes were counted as
  "errors" and suppressed ρ, and shows why a fully judged pool is needed.
- Limitation: with 43 and 54 queries, the SE of ρ is ≈ 0.12–0.15. A sequential-certificate comparison is possible at only the single T=10 look and hence meaningless.
  The ESS gain shrinks as T increases because N is small and the Var(d̂)/N term remains (in real deployment with N≫T it approaches the theoretical value).
- In progress: recompute the same table with Qwen3-8B LLM judgments using the UMBRELA prompt (`66_llm_judge.py`).

### 9.7 F4 — the PPI gain is determined by ρ, and accuracy is irrelevant (`05_results/ppi_gain/F4_gain_vs_rho_rr.png`)

Legacy and modern BEIR pools (4 collections × 3 pairs each) + TREC DL (2 × 3 pairs) = 30 (pool, policy pair) points,
same judge (Qwen3-Reranker), T=10, 300–500 random audit draws.

| point set | n | corr(gain, ρ) | corr(gain, overall accuracy) |
|---|---|---|---|
| legacy | 12 | 0.85 | 0.74 |
| modern | 12 | 0.90 | 0.50 |
| trecdl | 6 | 0.95 | 0.44 |
| all | 30 | **0.94** | **−0.07** |

The points follow the theoretical curve 1/(1−ρ²). Once pools are mixed, overall accuracy becomes unrelated to the gain (−0.07).
This is the candidate for the paper's first figure. Caution: at T=10, λ is estimated in-sample, so points near ρ≈0 float
slightly above the curve (mild optimism). In the main experiment, cross-fit λ or use a fixed λ schedule.

### 9.8 LLM judge (Qwen3-8B, UMBRELA prompt) vs reranker judge — TREC DL

| collection | policy pair | reranker ρ → LLM ρ | reranker inside/outside → LLM inside/outside | ESS gain T=10 (rr → LLM) |
|---|---|---|---|---|
| dl2019 | glob vs trunc | 0.70 → 0.88 | 0.33/0.29 → 0.27/0.25 | 1.76 → 2.64 |
| dl2019 | glob vs ad | 0.44 → 0.57 | 0.32/0.30 → 0.28/0.25 | 1.31 → 1.47 |
| dl2019 | trunc vs ad | 0.68 → 0.87 | 0.35/0.29 → 0.28/0.25 | 1.68 → 2.45 |
| dl2020 | glob vs trunc | 0.27 → 0.56 | 0.52/0.24 → 0.31/0.19 | 1.18 → 1.50 |
| dl2020 | glob vs ad | 0.01 → 0.48 | 0.62/0.27 → 0.35/0.20 | 1.08 → 1.37 |
| dl2020 | trunc vs ad | 0.63 → 0.76 | 0.42/0.32 → 0.27/0.22 | 1.75 → 2.19 |

Pair accuracy rises by 5–10 points, 0.70/0.66 → 0.75/0.77, yet the change in gain differs by pair.
The route by which the LLM judge produces gain is **dissolving the concentration of errors inside the region**
(dl2020 glob vs ad: inside/outside ratio 2.3x → 1.75x, ρ 0.01 → 0.48).
That is, it is not "a more accurate judge" but "a judge less biased in the decision region" that reduces audit cost.
Updated F4 (`F4_gain_vs_rho.png`): over 36 points, corr(gain, ρ)=0.91, corr(gain, accuracy)=−0.07.
The ESS gain falls short of the theoretical value (max 4.5) because of the Var(d̂)/N term with N=43 and 54.

### 9.9 Overall verdict for the kickoff phase

- Go/no-go #1 (structure survives on the modern stack): **passed**.
- Go/no-go #2 (judge errors concentrated in the decision region and dependence on ρ): **passed** — confirmed on the fully judged pool,
  consistent across two judge types (reranker, 8B LLM).
- Go/no-go #3 (the difference in audit savings from measured errors is significant): **partially passed** — the ESS gain of 1.0–2.6x is explained by ρ.
  Savings at the level of a sequential certificate require fully judged collections with more queries (TREC DL 2021–2023, LLMJudge).
→ The evidence needed for the decision to proceed in earnest is in place. Next: the theory and implementation of the
  neutrality diagnostic (§3.2) that estimates a lower bound on ρ from a small audit, fixing the reproducibility
  discipline with per-method independent rng and cross-fit λ, and securing fully judged collections with many queries.
## 10. Stage 2 Kickoff Record (2026-09-10, continued)

### 10.1 Honest F4 after applying the reproducibility discipline (cross-fit λ, independent rng per method)

The F4 in §9.7 used a λ estimated in-sample. Recomputing with a cross-fit λ (two halves crossed):

| T | points | corr(gain, ρ) | corr(gain, accuracy) | mean gain | max gain |
|---|---|---|---|---|---|
| 10 | 36 | 0.73 | −0.02 | 1.01 | 2.17 |
| 30 | 36 | 0.70 | 0.03 | 0.99 | 1.28 |
| 90 | 24 | 0.43 | 0.12 | 0.99 | 1.03 |

- **The mechanism holds** (only ρ explains the gain; accuracy is irrelevant), but **the magnitude of the realized gain shrinks.**
  Only LLM judge pairs with ρ ≥ 0.75 reach 1.3–2.2x (dl2019 glob vs trunc, trunc vs ad); the rest are ≈1.0.
- The gain decreases with T because the Var(d̂)/N term dominates at N=43–54. In real deployment (N ≫ T) it
  approaches the theoretical value 1/(1−ρ²); this is confirmed on the completely judged dbpedia-entity pool with 400 queries (in progress).
- Conclusion: reranker judges (ρ ≤ 0.3) offer no practical gain. The 8B LLM judge offers a gain only for some decisions,
  and ρ tells us which ones. → The judge neutrality diagnostic becomes a core component of the method.

### 10.2 Neutrality diagnostic (`lib/neutrality.py`, `70_neutrality_eval.py`)

- The Fisher-z lower bound on ρ has nominal coverage on normal data (miss 0.10–0.11), but on skewed data with many zeros,
  such as set-F1 differences, it falls short with miss 0.12–0.15. The bootstrap percentile lower bound is valid at n0=30 with 0.07–0.10,
  but at n0=10 it falls short with 0.14 when ρ=0 → **the pilot is recommended to be at least 20–30 queries.**
- dl1920 (97 queries, BEIR training): for reranker judges the diagnostic recommends "use judge" only 3–8% of the time (ground truth: no gain);
  for LLM judges (ρ 0.41–0.48, theoretical gain 1.2–1.3) it recommends 25–41%. The budget projection at N=97 shows PPI
  actually taking longer than human-only (ratio 1.04–1.46), which is due to the Var(d̂)/N term and the noise of the cross-fit λ; re-evaluate on a pool with large N.
- Applying the bootstrap lower bound to real dl1920 data: coverage of gain_lcb 0.78–0.99 (reranker 0.89–0.99, LLM 0.78–0.89).
  At N=97, n0=20–30 is 20–30% of the population, so the finite-population effect is large. Re-evaluate on a pool with large N and,
  if necessary, replace with a BCa or permutation-based lower bound. The 0–3% "use recommendation" for reranker judges matches the ground truth.

### 10.3 Completely judged BEIR pools (trec-covid 50q, webis-touche2020 49q, dbpedia-entity 399q) — reranker judge

pool = legacy pooling with the judged documents as corpus (union of top-30 of 4 systems); training uses the 4 BEIR legacy pools.

**PPI gain (cross-fit λ)** — the prediction that with large N the gain approaches the theoretical value as T grows is confirmed:

| collection | N | pair | ρ | gain T=10 | T=30 | T=90 | theory |
|---|---|---|---|---|---|---|---|
| dbpedia-entity | 399 | trunc vs ad | 0.57 | 1.17 | 1.28 | 1.27 | 1.48 |
| dbpedia-entity | 399 | glob vs trunc | 0.54 | 1.14 | 1.21 | 1.25 | 1.41 |
| trec-covid | 50 | trunc vs ad | 0.82 | 1.49 | 1.21 | – | 3.12 |
| webis-touche | 49 | glob vs ad | 0.24 | 0.90 | 0.89 | – | 1.06 |

**Sequential certificates (dbpedia, looks 10–90, 50 replications)** — first comparison on a completely judged pool with many queries:

| method | ACT | mean T | wrong |
|---|---|---|---|
| loo_boot (v0.3) | 0.90 | 57 | **0.16** |
| loo_sim | 0.74 | 70 | 0.10 |
| split_t | 0.14 | 88 | 0.00 |
| split_ppi (reranker) | 0.18 | 86 | 0.00 |
| recal_ep (v0.3) | 0.14 | 85 | 0.10 |
| recal_bpx | 0.02 | 90 | 0.00 |

- The v0.3 selection certificate clearly exceeds α on dbpedia with an error rate of 0.16 (MC SE 0.04). With simultaneous correction, 0.10.
- Adding PPI to the valid split certificate keeps the error at 0 while raising ACT from 0.14 to 0.18. The judge is weak (ρ≈0.5), so the gain is small.
  To be recomputed with the LLM judge.
- Concentration of judge errors in the decision interval is weak on these three pools (dbpedia 0.29–0.34 vs 0.27). Whether errors concentrate depends on the collection
  and judge, which is why a prior diagnostic is needed.

### 10.4 dbpedia-entity (399 queries, completely judged) + Qwen3-8B LLM judge — results at large N

LLM judge pair accuracy 0.76 (reranker 0.72). ρ is 0.61–0.66 (reranker 0.44–0.57).

**PPI gain (cross-fit λ) is maintained up to T=90:**

| pair | ρ | T=10 | T=30 | T=90 | theory |
|---|---|---|---|---|---|
| glob vs trunc | 0.66 | 1.55 | 1.54 | 1.48 | 1.78 |
| glob vs ad | 0.61 | 1.29 | 1.42 | 1.37 | 1.58 |
| trunc vs ad | 0.66 | 1.34 | 1.49 | 1.42 | 1.75 |

**Sequential certificates (looks 10–90, 50 replications, α=0.1):**

| method | ACT | mean T | wrong |
|---|---|---|---|
| split_t (valid, human only) | 0.22 | 87 | 0.00 |
| **split_ppi (valid, LLM judgments + human rectifier)** | **0.42** | 82 | 0.00 |
| loo_boot (v0.3) | 0.80 | 64 | 0.00 |
| loo_sim | 0.72 | 72 | 0.00 |

- **The ACT rate of the valid certificate nearly doubles with PPI** (0.22 → 0.42), with 0 wrong certificates. This is the paper's practical result:
  "using an AI judge safely increases the number of decisions certifiable by human audit."
- Neutrality diagnostic (pilot 20–30, bootstrap lower bound): "use judge" recommendation rate 0.49–0.82 (ground truth: gain 1.6–1.8x),
  gain_lcb coverage 0.83–0.94. For the reranker it correctly identifies unusability at 0–8%.
- (Caution) This run excluded trec-covid from the training pools, so its policy parameters differ from §10.3. The final table will be produced by
  re-running the three collections under identical conditions.

**Implementation defect found (reproducibility discipline):** in left-padded batches, if position_ids are not passed, the RoPE positions of rows with
large padding are shifted and the output collapses. This is why the trec-covid LLM judgments were all 0 on 49/50 queries. Re-judging after the fix is in progress, and
we are checking whether the same problem affected the Qwen3-Reranker scores (batch 128).

### 10.5 Final re-run — three completely judged collections under identical conditions (training: BEIR legacy 4 pools), two judges

**Confirmed that reranker scores are unaffected by the padding issue** (batch 128 vs 1, mean score difference 0.001, identical accuracy).
For the LLM judge, trec-covid accuracy after the position_ids fix went 0.43 → 0.72.

**Judge diagnostics (pair accuracy / ρ range)**

| collection | N | reranker | LLM(8B) |
|---|---|---|---|
| trec-covid | 50 | 0.67 / 0.74–0.82 | 0.76 / 0.73–0.84 |
| webis-touche | 49 | 0.57 / 0.24–0.48 | 0.79 / 0.69–0.81 |
| dbpedia-entity | 399 | 0.72 / 0.44–0.57 | 0.76 / 0.53–0.63 |

**PPI gain (cross-fit λ, T=10 / T=90)**: with the LLM judge, trec-covid 1.13–1.51, touche 1.12–1.46, dbpedia 1.25–1.42
(1.25–1.42 maintained even at T=90). reranker: touche 0.90–0.97 (no gain), dbpedia 1.06–1.27.

**Sequential certificates (dbpedia, looks 10–90, 50 replications, α=0.1)**

| method | ACT | wrong |
|---|---|---|
| loo_boot (v0.3) | 0.90 | **0.16** |
| loo_sim | 0.74 | 0.10 |
| split_t | 0.14 | 0.00 |
| split_ppi (reranker) | 0.18 | 0.00 |
| split_ppi (LLM) | 0.20 | 0.00 |
| recal_ep (v0.3) | 0.14 | 0.10 |
| recal_bpx | 0.02 | 0.00 |

- In this configuration the ACT increase of the valid certificate is 0.14 → 0.20 (in the different training configuration of §10.4 it was 0.22 → 0.42).
  A fixed-budget ESS gain of 1.3–1.4x shows up as a 1.4–2x ACT rate depending on the configuration. Both configurations have 0 wrong certificates.
- Final F4 (54 points, cross-fit λ): T=10 corr(gain, ρ)=0.78, corr(gain, accuracy)=−0.01; T=30 0.68 / 0.04.

### 10.6 Stage 2 summary

1. **Mechanism confirmed**: the PPI audit saving is determined by the decision-relevant agreement ρ, and overall accuracy is irrelevant (54 points, 2 judges, 4 pools).
2. **Practical result**: on a completely judged pool with large N, LLM judge + valid certificate gives a 1.4–2x ACT rate with 0 errors.
3. **v0.3 defect reconfirmed**: on dbpedia, selection error rate 0.16, recalibrated endpoint-check error rate 0.10.
4. **Neutrality diagnostic**: the bootstrap lower bound on ρ is largely valid at pilot ≥ 20 (coverage 0.83–0.99), correctly identifies the reranker as unusable,
   and recommends LLM use 49–82% of the time. Finite-population and skew corrections are follow-up work.
5. **Remaining work**: (a) proofs of Theorems A, B, C and anytime-valid versions, (b) empirical measurement when the judge's bias is correlated with a specific policy
   (reranker policy from the same model family + LLM judge) — unverified since the current menu has no LLM-based policy,
   (c) adding completely judged collections with ≥ 400 queries (TREC DL 2021–23 v2, LLMJudge), (d) prospective lock.

## 11. Stage 3 — Response to External Review Comments (2026-09-10)

### 11.1 Non-independence of the 54 points → does the relationship hold within collection and judge?

- Within the 6 pool×judge groups, demeaned corr(gain, ρ) = 0.64; within the 14 collection×judge groups, 0.66.
- In **all** 14 groups (3 policy pairs each), the within-group correlation is positive (0.48–1.00).
- When the judge is switched from reranker→LLM on the same collection and policy pair, the gain also rose in **all** 13 cases where ρ rose.
→ The relationship is not an artifact of pooled correlation.

### 11.2 Denominators and intervals (500 replications, Clopper–Pearson 95%)

**Cells where the v0.3 selection certificate (loo_boot) exceeds α with the whole interval:** modern android 73/500 = 0.146 [0.116, 0.180],
modern scifact 90/500 = 0.180 [0.147, 0.217]. Simultaneous correction (loo_sim) also on modern scifact 0.156 [0.125, 0.191].
dbpedia 0.116 [0.089, 0.147] and legacy scifact 0.110 [0.084, 0.141] are borderline.
**v0.3 recalibrated endpoint check (recal_ep):** arguana modern 0.176 [0.144, 0.212], legacy 0.136 [0.107, 0.169] — confirmed exceedance.
breakpoint+exact interval (recal_bpx) is 0.096 [0.072, 0.125] / 0.038.
**Valid split certificates:** wrong ≤ 4/500 in all 24 cells (≤ 0.008, upper bound ≤ 0.020).

**How much PPI increased the ACT of the valid certificate (ACT count/500, wrong 0–1):**

| collection | judge | split_t | split_ppi | multiple |
|---|---|---|---|---|
| dbpedia-entity | LLM 8B | 48 | 86 | 1.8 |
| dbpedia-entity | reranker | 48 | 74 | 1.5 |
| trec-covid | reranker (ρ 0.74–0.82) | 42 | 105 | 2.5 |
| trec-covid | LLM 8B | 42 | 64 | 1.5 |
| BEIR modern 4 | reranker (qrels holes, ρ ≤ 0.3) | 43–140 | 43–139 | 1.0 |

The previously reported "0.22→0.42" (50 replications) was a different training configuration; with the same configuration at 50 replications it was 7→10, not significant.
The 500-replication results are the reference. Absolute ACT rates are low at 10–21%: ε_sel=0.01 remains a strict criterion on this menu.

### 11.3 Target value for coverage

Since α=0.1, the target coverage of gain_lcb is 0.90. At 0.78–0.99 on dl1920 and 0.83–0.94 on dbpedia,
some cells fall short of the target. The causes are the finite population (pilot 30 at N=97) and skew; in the paper we
either state n0 ≥ 30 and N ≫ n0 explicitly as conditions or replace with a BCa/permutation lower bound.

### 11.4 Scope of impact of the padding fix

- Qwen3-Reranker scores: mean difference between batch 128 and 1 is 0.001, identical accuracy → existing rr results are valid.
- LLM judgments: only trec-covid collapsed (re-judging complete, 0.43→0.72). touche, dbpedia and TREC DL were judged before the fix,
  so re-judging is in progress and the differences are recorded as a diff (§11.6).

### 11.5 When a policy from the same model family as the judge is on the menu (self-preference stress test, `72_selfpref.py`)

Added `rr_thresh` (return documents with Qwen3-Reranker score ≥ θ) to the menu. Judge = identical model (reranker) or same family (Qwen3-8B).
5 completely judged pools × 6 policy pairs, T=30, cross-fit λ.

| judge | pair type | n | ρ | error inside/outside interval | judge's paired bias | PPI gain |
|---|---|---|---|---|---|---|
| reranker (identical model) | includes rr_thresh | 15 | **0.25** | 0.41 / 0.31 | **−0.14** (overrates its own policy) | 0.95 |
| reranker | excludes | 15 | 0.57 | 0.36 / 0.32 | −0.03 | 1.07 |
| LLM 8B (same family) | includes rr_thresh | 15 | **0.50** | 0.28 / 0.21 | −0.03 | 1.05 |
| LLM 8B | excludes | 15 | 0.73 | 0.26 / 0.22 | −0.02 | 1.22 |

- When the judge is the very model that defines a policy, the ρ of comparisons involving that policy collapses from 0.57→0.25, and the judge
  overrates its own policy by 0.14 in paired F1. The PPI gain disappears (0.95).
- The same-family 8B judge has a small bias of −0.03, but ρ still drops from 0.73→0.50
  (Δρ per collection: dl2019 −0.47, touche −0.44, dl2020 −0.23, dbpedia −0.04, trec-covid 0.00).
- **Validity is preserved** (Theorem B): sequential certificates on the 4-policy menu, 500 replications, on dbpedia both split_t and split_ppi have wrong 0/500.
  On the same menu, split_ppi ACT: identical-model judge 246/500 (= split_t 245, no gain) vs 8B judge 328/500 (1.34x).
- Conclusion: "judge neutrality" is a quantity measurable at the model-family level, and it is exactly the situation the neutrality diagnostic (§10.2) must catch.

## 12. Stage 4 — Separating Final-Certification Validity from the ρ Diagnostic, Judge Comparison, Budget Curves (2026-09-10)

### 12.1 Dependency structure of the algorithm (external review item 5)

The validity of the final certification (split_ppi) is **independent** of the ρ diagnostic. The PPI++ rectifier removes bias on the human audit sample
whatever the judge is (including biased or inverted ones), and λ is cross-fit and clipped to [0,1], so in the worst case it
degenerates to the human-only certificate. The ρ diagnostic only determines "how much gain there is from using this judge" and plays no role in the error rate.

**Empirical measurement (dbpedia-entity, looks 10–190, 500 replications, ε=0.01, α=0.1):**

| judge | split_ppi cumulative ACT (T=90 / 150 / 190) | wrong | vs human only (T=190) |
|---|---|---|---|
| human only (split_t) | 30 / 103 / 158 | 1/500 | 1.00 |
| Qwen3-8B LLM | 56 / 172 / **244** | 0/500 | **1.54** |
| Qwen3-Reranker | 56 / 149 / 216 | 1/500 | 1.37 |
| MPNet rank (different family, accuracy 0.64) | 32 / 100 / 160 | 2/500 | 1.01 |
| **inverted reranker (adversarial, accuracy 0.34)** | 30 / 103 / 156 | 2/500 | 0.99 |

- Even with the adversarial judge the error rate is maintained and ACT equals human-only (λ→0). That is, **the coverage shortfall of the ρ lower bound
  is a reliability problem for the judge recommendation, not a problem for the final certification guarantee.**
- Figure F5 (`05_results/planner_v2/F5_act_vs_budget_dbpedia-entity.png`): the gap widens as the budget grows.
  ε=0.01 is strict on this menu, so even at T=190 human-only certifies only 32%, versus 49% with the 8B judge.
- v0.3 (loo_boot) certifies 74% at T=90 but with errors 58/500 [0.089, 0.147]. In the paper it now remains only as a comparison
  method that exhibits the problem.

### 12.2 Fixed policies, comparison of 4 judges (external review item 4; `72_selfpref.py`, 5 pools × 6 pairs)

The 4 policies (glob, trunc, ad, rr_thresh) are fixed as the full target. Judges: identical model (reranker), same family (8B LLM),
different family (MPNet rank rule), adversarial (inverted reranker). Pairs are split into those including/excluding rr_thresh.

| judge | pair accuracy | ρ (excl. → incl.) | judge's paired bias (excl. → incl.) | error inside interval (excl. → incl.) | PPI gain (excl. → incl.) |
|---|---|---|---|---|---|
| identical model (reranker) | 0.67 | 0.57 → 0.25 | −0.03 → **−0.14** | 0.36 → **0.41** | 1.07 → 0.95 |
| same family (8B LLM) | 0.76 | 0.73 → 0.50 | −0.02 → −0.03 | 0.26 → 0.28 | 1.22 → 1.05 |
| different family (MPNet) | 0.63 | 0.39 → 0.21 | −0.05 → +0.03 | 0.39 → 0.40 | 1.03 → 0.94 |
| adversarial (inverted) | 0.34 | −0.14 → −0.04 | −0.02 → +0.24 | 0.64 → 0.59 | 0.91 → 0.93 |

Honest interpretation:
- The ρ drop on pairs involving rr_thresh appears **for every judge** (MPNet too, 0.39→0.21). Hence part of the ρ drop is a
  "policy difficulty" effect — that policy's paired differences are inherently hard to predict — and not evidence of self-preference alone.
  The external review's concern is correct.
- There are two signatures **unique** to self-preference: (i) only the identical-model judge systematically overrates its own policy by 0.14 in paired F1
  (other judges −0.03 to +0.03), (ii) only the identical-model judge shows an increase in error inside the interval (0.36→0.41).
  The same-family 8B has almost no bias.
- Even with bias, validity is preserved and only efficiency drops (§12.1, 4-policy menu, 500 replications, wrong 0/500).
→ The paper's claim is narrowed not to "a demonstration of neutrality" but to **"when the judge is the model that defines a policy, bias and decision-interval error
  arise, and they eliminate the audit saving. Our certificate remains valid even then, and the ρ diagnostic warns of that situation in advance."**

### 12.3 Positioning of the contribution (external review item 1)

"Good predictions raise PPI efficiency" is the principle of PPI++. This paper's additions are (a) empirical evidence that, **for policy-selection decisions, that efficiency
is determined by the ρ of paired differences** and is separate from judgment accuracy (54 points, within-group 14/14), (b) empirical measurement of the conditions under which
the relationship between the model defining the judge and a policy collapses ρ, (c) a procedure that estimates a lower bound on ρ from a small pilot to decide judge adoption and
project the budget, and (d) a separation design in which the final certification remains valid even when that procedure is wrong. The final evaluation on new data (§10.6 (d))
remains.

### 12.4 Re-run of LLM judgments after the position_ids fix — final confirmation (2026-09-10 09:38)

Difference in Qwen3-8B judgments before and after the fix (mean |difference| in p_rel / label-flip rate): touche 0.005 / 0.43%,
dbpedia 0.004 / 0.46%, dl2019 0.004 / 0.42%, dl2020 0.004 / 0.34%. Only trec-covid had collapsed (0.43→0.72); for the
rest, pair accuracy is identical to the third decimal place. Final values after re-running all LLM-dependent experiments with post-fix judgments:

| result | before fix | final |
|---|---|---|
| dbpedia split_ppi(8B) ACT/500, look ≤90 | 86 | 87 (wrong 1/500) |
| dbpedia split_ppi(8B) cumulative ACT, look 190 | 244 | 234 (wrong 0/500); human only 158 |
| dbpedia 4-policy menu split_ppi(8B) ACT/500 | 328 | 327 (wrong 0); split_t 245 |
| trec-covid split_ppi(8B) ACT/500 | 64 | 64 (wrong 0) |
| F4 (54 points) corr(gain, ρ) / corr(gain, accuracy), T=10 | 0.78 / −0.01 | 0.78 / −0.02 |
| self-preference: 8B judge ρ (excl. → incl. rr) | 0.73 → 0.50 | 0.73 → 0.49 |

→ All figures in §10–12 are valid on the basis of the final values, and the differences between pre-fix and final values in the tables of this document lie within the ranges above.

### 12.5 Next steps (in the order of the external review)

1. ~~Confirm post-fix re-judging~~ (done)
2. ~~Separate the roles of the final certification procedure and the ρ diagnostic~~ (§12.1, wrong ≤ 2/500 even with the adversarial judge)
3. **Fix the judge selection rule**: pilot n0 ≥ 30, bootstrap lower bound on ρ → PPI if gain_lcb > 1.1, otherwise human only.
   Fix the rule, ε, α, looks and menu in a lock document.
4. **Evaluate on new data**: a completely judged collection with ≥ 100 queries is needed. TREC DL 2021–23 (v2 corpus, access to be confirmed),
   TREC Robust04 (license), LLMJudge/TREC RAG 2024. After confirming which are accessible, lock → single primary run.
5. Proofs of Theorems A, B, C and anytime-valid extension.

## 13. Stage 5 — Aggregation Criterion, Conditional Validity, Scope Definition, Lock Draft (2026-09-10)

### 13.1 The "cumulative error" aggregation criterion (external review item 1)

The two rows of the §12.4 table are **separate runs**: the look grids {10,…,90} (L=5) and {10,…,190} (L=8) have different Bonferroni splits
α/(L·M(M−1)), so decisions at the same time point also differ. Hence 87/500 (wrong 1) and 234/500 (wrong 0) are not nested.
Counting per time point within the same run (L=8), monotonicity holds:

| judge | look ≤90 ACT / wrong | look ≤190 ACT / wrong |
|---|---|---|
| 8B LLM | 56 / 0 | 234 / 0 |
| reranker | 56 / 0 | 216 / 1 |
| inverted (adversarial) | 30 / 1 | 156 / 2 |

Definition of the error event: the planner stops at the first ACT, and if the true regret of the policy certified at that point (computed on the unaudited complement) exceeds ε,
that counts as 1 wrong. Since there is at most 1 per replication, wrong/500 is an estimate of P(ACT ∧ regret > ε). Paper tables always state the look grid and
L alongside.

### 13.2 Conditional validity — why the certification is valid even when judge adoption is wrong (external review item 2)

Audit sample S_T = training half A_T ∪ validation half B_T (first ⌊T/2⌋ / the rest).

- **Decided on A_T**: policy parameters (ĉ, τ̂, θ̂), candidate policy m̂ (LCB rule), whether to adopt the judge (ρ lower-bound rule).
- **Computed on B_T**: the certificate. Paired differences D_j = u_j − u_m̂ between candidate m̂ and competitor j (human labels on B_T),
  judge predictions D̂_j (the output of a fixed function f over all target queries), λ_j cross-fit by splitting B_T into two folds.

**Proposition B′.** If the judge f and all choices made on A_T (policy parameters, candidate, adoption) are independent of B_T, then for each (look, j)
the PPI++ estimator μ̂_j = λ̄·mean_N(D̂_j) + mean_B(D_j − λ_j D̂_j) is an unbiased estimator of μ_j − μ_m̂ regardless of f and λ, and the
t-based UCB is (approximately) valid. The case where the adoption rule chooses "t instead of PPI" has the same property.
Therefore, by a union bound over looks × all ordered pairs, P(∃t: ACT_t ∧ regret > ε) ≤ α holds **conditionally on the outcomes of the choices on A_T**.
The coverage of the ρ lower bound affects only the quality (efficiency) of the adoption decision and does not enter this bound.
The proof is the unbiasedness of PPI++ (Angelopoulos et al. 2023) + conditional independence from sample splitting + Bonferroni. The remaining approximations are the t-distribution approximation and
the slight dependence from the N-sample of D̂ containing B_T (N ≫ |B_T|); the latter is removed by taking the N-mean of D̂ only outside B_T (to be implemented).

**Cost accounting**: pilot = A_T, so judge selection needs no additional human labels. The additional cost is judge inference (all pool pairs, e.g.
20,573 pairs for dbpedia), and the practical value of the adoption rule lies in not wasting that inference. Since some cells have coverage below the
target 0.90, the "gain lower bound" is called an **adoption score, not a guaranteed lower bound**.

### 13.3 Definition of "completely judged pool" and the scope of the conclusions (external review item 3)

The pools in this study consist **only of (query, doc) pairs actually judged by NIST/BEIR assessors** (TREC DL: the full qrels;
BEIR Tier-A: the union of 4-system pooling with the judged documents as corpus). Unjudged documents are not in the pool.
Therefore (i) the policies are policies that choose a cutoff position within the judged candidate set, (ii) the "human ground truth" is the judgment over that candidate set, and
(iii) the conclusions are limited to "deployment decisions over the judged candidate set." In real deployment, if unjudged documents are mixed into the pool,
judge errors and judgment holes become entangled; that effect is shown in §9.5 (BEIR sparse qrels: ρ ≤ 0.3).
Prior work on the bias of the practice of treating unjudged documents as 'non-relevant' (including arXiv:2405.04727 cited by the external review) will be cited after verification.

### 13.4 LOCK v0.4 draft (items to fix before evaluation on new data)

- Decision: ε_sel = 0.01, α = 0.1, look grid {10,30,50,70,90,120,150,190} ∩ [≤ n/2], Bonferroni over looks × M(M−1).
- Menu: {glob_probe, trunc, ad_probe} (+ rr_thresh only for the self-preference analysis).
- Certificate: split (first half training / second half validation), candidate = LCB maximum, t-UCB or PPI++ (cross-fit λ).
- Judge adoption rule: when |A_T| ≥ 30, PPI if the gain score 1/(1−lcb²) from the bootstrap lower bound on ρ (level 1−α) for the candidate–runner-up pair is > 1.1,
  otherwise t. These two values are development-stage choices with no optimality claim. If not adopted, fall back to the human-only certificate.
- Judge: Qwen3-8B UMBRELA (non-thinking, explicit position_ids), threshold 0.5. Qwen3-Reranker-0.6B for comparison.
- Evaluation: single primary run of 500 replications on the new collection. Report: ACT/500, wrong/500 + Clopper–Pearson, judge adoption rate,
  number of judge-inference pairs. Three-way comparison of human only, always-PPI and rule-based.
- New collection requirements: ≥ 200 queries in terms of the judged candidate set (to support look 190 and the n/2 cap), documented judgment depth, unused in development.

### 13.5 Rule-based judge adoption (split_auto) vs human only vs always PPI — dbpedia, 500 replications, 8 looks

Rule: when the training half |A_T| ≥ 30, PPI if the gain score from the bootstrap lower bound on ρ for the candidate–runner-up pair is > 1.1, otherwise t.

| judge | human only (T=190 cumulative ACT) | always PPI | rule-based | rule's judge adoption rate | wrong (all three) |
|---|---|---|---|---|---|
| Qwen3-8B LLM | 158 | 234 | **223** | 78% | 1 / 0 / 1 |
| Qwen3-Reranker | 158 | 216 | 179 | 44% | 1 / 1 / 2 |
| MPNet rank (different family) | 158 | 160 | 158 | 0% | 1 / 2 / 1 |
| inverted (adversarial) | 158 | 156 | 158 | 0% | 1 / 2 / 1 |

- The rule rejects useless judges (MPNet, inverted) 100% of the time, and with the 8B judge recovers 95% (65/76) of the always-PPI gain.
  With the reranker it is conservative (27%) — because the ρ lower bound, near 0.5–0.6, often fails to clear the 1.1 threshold.
- **Always-PPI harms neither validity nor efficiency with any judge** (degenerates via λ→0, wrong ≤ 2/500). Hence, looking at ACT alone,
  always-PPI is better than or equal to the rule. The practical value of the rule is (i) not wasting judge inference cost (all pool pairs),
  (ii) projecting the budget in advance. The paper states this as is: "the adoption rule is not a safety device but a cost-saving device."
- Limitation: |A_T| ≥ 30 requires total T ≥ 60, so on 50-query collections (trec-covid, touche) the rule does not operate and
  reduces to human only. This is the basis for the new collection requirement (§13.4, ≥ 200 queries).

### 13.6 Stage 5 summary
External review items 1 (aggregation), 2 (conditional validity) and 3 (scope) are answered in §13.1–13.3, and of the two decisions, the lock draft is in §13.4.
What remains is (a) the formal proof of Proposition B′ and the implementation excluding B_T from the N-mean of D̂, (b) securing an accessible new completely-judged
collection and the primary run, (c) improving the coverage of the ρ lower bound (BCa/permutation) — (c) is an efficiency issue unrelated to validity.

### 13.7 Improving the coverage of the ρ lower bound — BCa (dbpedia-entity N=399, 8B judge, 400 draws, target 0.90)

| pair (true gain) | n0 | Fisher-z | percentile bootstrap | **BCa** |
|---|---|---|---|---|
| glob vs ad (1.55) | 20 / 30 / 50 | 0.87 / 0.89 / 0.91 | 0.82 / 0.88 / 0.89 | 0.86 / **0.90** / **0.90** |
| glob vs trunc (1.67) | 20 / 30 / 50 | 0.80 / 0.84 / 0.83 | 0.89 / 0.90 / 0.91 | 0.90 / **0.90** / **0.92** |
| trunc vs ad (1.92) | 20 / 30 / 50 | 0.75 / 0.77 / 0.80 | 0.87 / 0.88 / 0.91 | 0.86 / **0.88** / **0.91** |

- Fisher-z fails at 0.75–0.84 on pairs with large skew. BCa is within the 0.90 target at 0.88–0.92 (MC SE 0.015) for n0 ≥ 30.
- On a synthetic zero-inflated DGP as well, BCa at n0=30 has miss 0.09–0.10 (nominal 0.10). At n0=10 no method suffices (0.12–0.13).
- Judge-use recommendation rate of the adoption rule (BCa, n0=30): 0.65–0.78 (true gain 1.55–1.92) — conservative but directionally consistent.
→ The adoption rule in LOCK v0.4 is fixed as **BCa lower bound, n0 ≥ 30**. It is still an "adoption score," not a "guarantee" (§13.2), and unrelated to certification validity.

### 13.8 Implementation of Proposition B′ — excluding the validation half from the N-mean of D̂ (dbpedia, 8B, 500 replications, 8 looks)

| method | cumulative ACT before exclusion (T=90/150/190) | after exclusion | wrong |
|---|---|---|---|
| split_t | 30 / 103 / 158 | same | 1/500 |
| split_ppi | 56 / 168 / 234 | 61 / 168 / 222 | 0/500 |
| split_auto | 46 / 150 / 223 | 50 / 153 / 217 | 1/500 |

The exclusion reduces N by T, so ACT at T=190 decreases by 5%, but the error rate is unchanged. All subsequent runs (including primary) use the exclusion version.
The definition of "always PPI" in LOCK v0.4 is also this version.
## 14. PRIMARY RUN — TREC DL 2021–2023 (MS MARCO v2), LOCK v0.4 (2026-09-10)

Target `dl212223`: 211 queries (53+76+82), 122,240 judged passages of which the pool is 14,637 pairs (65–76 per query), relevant = grade ≥ 2.
Since cap = n/2 = 105, looks are run only up to {10,30,50,70,90} (LOCK §2 grid ∩ [≤ n/2]). 500 draws.
Training uses the 4 BEIR legacy pools (no contact with the target). Only pool and query counts were checked, without any judgments, before the lock commit (cb03b2b).

### 14.1 reranker · inverted judge (locked protocol as is)

| judge | method | cumulative ACT (T=30/50/70/90) | wrong/500 [CP 95%] |
|---|---|---|---|
| — | split_t (human only) | 18 / 84 / 198 / **313** | 0 [0, 0.007] |
| Qwen3-Reranker (accuracy 0.43) | split_ppi (always PPI) | 15 / 52 / 136 / **256** | 0 [0, 0.007] |
| Qwen3-Reranker | split_auto (rule) | 18 / 84 / 194 / 309 | 0 [0, 0.007]; adoption 1.3% |
| inverted | split_ppi | 12 / 70 / 182 / 304 | 0 |
| inverted | split_auto | 18 / 84 / 198 / 313 | 0; adoption 0% |
| — | loo_boot (v0.3) | 297 / 434 / 494 / 498 | 7 = 0.014 [0.006, 0.029] |
| — | recal_ep (v0.3) | 5 / 9 / 10 / 12 | 10 = 0.020 |

**Pre-registered criteria (inverted-judge part)**: validity passed (upper bound 0.007 ≤ 0.15 for all three methods); the rule's adoption of the inverted judge 0% ≤ 5% passed.

**New finding — always-PPI can be worse than human only.** With the reranker judge, split_ppi 256 < split_t 313 (−18%).
Two causes: (i) the PPI++ λ formula ignores the number of unlabeled samples N (assumes N→∞), but in this collection N−T = 121 and validation is 45, so
n/N = 0.37; (ii) the judge is essentially uninformative on this collection (accuracy 0.43, ρ 0.29–0.45), so the noise of λ estimated from 22 folds
adds variance. **Rule-based adoption accepted the reranker only 1.3% of the time and avoided the loss (309 ≈ 313).** In §13.5 we wrote that "the rule is a cost-saving
device"; the new data shows it is also an **efficiency-protection device**. Validity holds in every case.

**Post-hoc deviation (outside the lock)**: replacing λ with the finite-N optimal formula Cov/Var/(1+n/N) moves split_ppi 256 → 282 (inverted 304 → 306).
The remaining gap (282 < 313) is λ estimation noise, and a safeguard such as "λ = 0 if it does not reduce variance on the fitting folds" is needed.
This change does not replace the primary numbers and is reported only as a deviation.

- loo_boot (v0.3) is within α on this collection with error 0.014 (the policy gap is large, so certification is easy: ACT 498/500).
  The v0.3 excess appears on collections where the gap is near ε (modern android·scifact, dbpedia).

### 14.2 Qwen3-8B LLM judge — pre-registered criteria verdict (locked protocol, 500 draws, look ≤ 90)

Judge pair accuracy 0.674 (grade ≥ 2 criterion), ρ = 0.40 / 0.47 / 0.52, error inside/outside the band 0.35–0.38 / 0.30–0.32.

| method | cumulative ACT (T=10/30/50/70/90) | wrong/500 [CP 95%] | adoption rate |
|---|---|---|---|
| split_t (human only) | 0 / 18 / 84 / 198 / **313** | 0 [0, 0.007] | — |
| split_ppi (always PPI, 8B) | 11 / 38 / 105 / 220 / **318** | 0 [0, 0.007] | — |
| split_auto (rule, 8B) | 0 / 18 / 84 / 203 / **308** | 0 [0, 0.007] | 26% |
| loo_boot (v0.3) | 95 / 297 / 434 / 494 / 498 | 7 = 0.014 [0.006, 0.029] | — |
| recal_ep (v0.3) | 4 / 5 / 9 / 10 / 12 | 10 = 0.020 [0.010, 0.036] | — |

**Pre-registered criteria results**

| criterion (LOCK v0.4 §3) | result |
|---|---|
| Primary — wrong upper bound ≤ 0.15 and point estimate ≤ 0.10 for all three methods | **passed** (all 0/500, upper bound 0.007) |
| Secondary — split_ppi ≥ 1.2 × split_t (T=90) | **not met** (318 vs 313 = 1.02×) |
| Secondary — split_auto ≥ split_t | **not met** (308 vs 313, −1.6%) |
| Rule's adoption of the inverted judge ≤ 5% | **passed** (0%) |

**Interpretation (numbers reported as is, per the pre-registration principle)**
- The validity claim replicated on new data: whether the judge is 8B, reranker, or inverted, and whether or not the rule adopts the judge,
  wrong certificates are 0/500. Empirical confirmation of Proposition B′.
- The efficiency claim did not replicate on this collection. PPI leads at intermediate looks (T=50: 105 vs 84, T=70: 220 vs 198) but
  human only catches up at T=90. There are three causes: (i) the policy gap in this collection is large, so human only already certifies 63% at T=90
  (dbpedia is 32% even at T=190), (ii) judge ρ is 0.40–0.52, lower than dbpedia (0.53–0.66), and (iii) N−T = 121 means the
  unlabeled sample is small, so the Var(D̂)/(N−T) term cancels the theoretical gain (1.19–1.38) (measured gain 0.97–1.00 at T=90).
- The rule-based method adopted the judge only 26% of the time and the result is at the same level as human only (−5 cases, within noise). It prevented the loss with the reranker
  (§14.1) and gave neither gain nor loss with 8B.
- Combined with the "efficiency loss of always-PPI" confirmed for the inverted judge and reranker (§14.1), the message of the new data is clear:
  **the audit savings from an AI judge are conditional (ρ ≥ 0.6, N ≫ T, policy gap near ε), while validity is unconditional.**
  The paper's efficiency claim is narrowed to a form that states these conditions explicitly, and the unconditional claims are restricted to validity and "the rule filters out bad judges and
  prevents loss".

### 14.3 Paper status update
- Secured: the central mechanism (ρ), valid certificates (5 development pools + 1 new pool, total wrong ≤ 0.008), conditional validity theory, the judge adoption rule and
  its safety role, measured self-preference, one pre-registered external evaluation (validity passed, efficiency not met, reported as is).
- Remaining weaknesses: an experimental design that confronts head-on the fact that the efficiency gain is conditional (a mechanism map manipulating gap·N·ρ on real-data subsamples),
  judge diversity (a strong LLM from a different family), anytime-valid extension, finite-sample bounds instead of the t-approximation in Proposition B′.

## 15. Stage 6 — ρ-condition experiment (mechanism map) and a judge from a different family (2026-09-10)

### 15.1 Conditions under which PPI gains — real-data mechanism map (`77_rho_map.py`, F6)

In fixed-budget certification (T, training half / validation half, ordered-pair Bonferroni), three axes are manipulated: judge label noise p ∈ {0, .3, .6, .9}
(→ ρ decreases continuously), unlabeled samples N_unlab ∈ {50,100,200,300}, ε ∈ {.005,.01,.02,.05}. 300 draws/cell, 8B judge.
Metric: ACT(PPI)/ACT(human only). wrong ≤ 0.01 in every cell.

**dbpedia-entity (N=399), T=90:**

| ε | N_unlab | ρ=0.26 | ρ=0.39 | ρ=0.57 | ρ=0.64 | human-only ACT |
|---|---|---|---|---|---|---|
| 0.005 | 50 / 300 | 0.76 / 0.97 | 1.02 / 1.04 | 1.03 / **1.41** | 0.97 / **1.52** | 0.31 / 0.27 |
| 0.01 | 50 / 300 | 0.97 / 0.98 | 0.97 / 1.06 | 1.17 / 1.22 | 0.98 / **1.39** | 0.35 / 0.33 |
| 0.02 | 50 / 300 | 0.95 / 0.99 | 0.89 / 0.99 | 0.94 / 1.11 | 0.97 / **1.31** | 0.53 / 0.57 |
| 0.05 | 50 / 300 | 0.97 / 0.98 | 0.95 / 1.01 | 0.95 / 1.00 | 0.95 / 1.01 | 0.92 / 0.92 |

**dl212223 (N=211), T=90:** ρ is 0.46 even at zero noise and N_unlab ≤ 100, so the ratio is 0.88–1.04 in every cell (no gain).

All three conditions must hold **simultaneously** for a gain:
1. **ρ ≥ ~0.55** — at ρ ≤ 0.4 there is no gain regardless of N·ε (0.76–1.06).
2. **N_unlab ≥ ~200 (≈ 2T or more)** — even at ρ=0.64, N_unlab=50 gives 0.95–0.98 while 300 gives 1.31–1.52.
3. **A hard-to-certify decision (ε ≤ 0.02, human-only ACT ≤ 0.6)** — when human only certifies 92%, as at ε=0.05, the gain disappears.

At T=60 the same structure appears more strongly (ρ=0.64, N_unlab=300, ε=0.005: 1.67×).
→ The efficiency shortfall of the primary run (§14) is exactly explained by the failure of two of the three conditions (ρ 0.46, N_unlab 121).
→ Paper claim: "The audit savings from an AI judge are predicted by the product of ρ, N/T, and decision difficulty (F6); outside those conditions they are equal to or slightly worse than human only.
Validity is independent of the conditions (wrong ≤ 0.01 in every cell)."

## 16. Stage 7 — Correction of band-local audit and decision-weighted audit (2026-09-10)

### 16.1 Correction: the "band sufficiency" of Proposition C holds only when the denominators are equal
The external reviewer's counterexample (R_A={a}, R_B={a,b}, b non-relevant: the precision difference is 0.5 or 0 depending on whether a is relevant) is correct.
When the cutoffs differ, a common document retains weight 1/k_a − 1/k_b. The correct statement is the **per-document decision weight** (THEORY §C, corrected version):
band documents ±1/Z, common documents |1/Z_a − 1/Z_b|, outside 0 (when the denominator is independent of labels). set-F1 is nonlinear because nG enters the denominator.
→ The rule is not "judge only the band" but **"allocate the judgment budget in order of weight magnitude"**, and this is implemented naturally by Horvitz–Thompson sampling and
a judge control variate (document-level PPI) (unbiasedness is independent of the judge).

### 16.2 Band-local hybrid predictor (`79_band_audit.py`) — negative result
- set-F1: the hybrid raises ρ to 0.87–0.90 (judge 0.60–0.69), but at the same pair budget the certification rate is equal to or lower than always-PPI
  (nA=90, ε=0.01: human full 0.32 / hybrid optimal allocation 0.48 / always-PPI 0.47). The fully judged queries needed for the rectifier are expensive, so
  the ρ increase is cancelled by cost.
- precision: the hybrid ρ does not rise (0.69 vs 0.73) — because, as in the counterexample, the common-document weights remain.
→ The "it suffices to judge only the band" method is discarded. The point that a ρ increase ≠ a cost saving is confirmed experimentally.

### 16.3 Decision-weighted document-level audit (`80_weighted_audit.py`) — four methods, same document judgment budget, precision utility
After 20 training queries for candidate selection (cost included), each method spends the remaining budget B (number of documents). dbpedia, 8B judge, preliminary 100 draws:

| budget (fully judged query equivalents) | ε | human full | **human weighted sampling (HT)** | existing PPI | **weighted sampling + judge CV** |
|---|---|---|---|---|---|
| 60 | 0.01 | 0.17 | 0.25 | 0.23 | **0.34** |
| 60 | 0.02 | 0.26 | 0.46 | 0.36 | **0.53** |
| 30 | 0.02 | 0.06 | 0.04 | 0.06 | 0.11 |

wrong 0 in every cell. The decomposition the external reviewer asked for is visible directly: the larger share of the gain comes from **concentrating the human budget on the important documents
without any judge** (0.26→0.46), and the judge adds a further gain on top (0.46→0.53). The existing query-level PPI (0.36) is
worse than weighted sampling alone. The full grid (budget 3 × ε 3 × document/query 2 × judge 2 × collection 2, 300 draws) is running.
Caution: this result is on development data (dbpedia), and dl212223 is already exposed as well, so a confirmatory evaluation of this method needs a separate unused
collection (§13.4 requirement).

### 16.4 Decision-weighted document-level audit — full grid (after the budget-exhaustion fix, 300 draws, precision utility, wrong ≤ 0.003 in every cell)

ACT rate (human full / human weighted HT / existing query-PPI / weighted + judge CV), multiplier relative to human full in parentheses:

| collection · judge | budget | ε=0.01 | ε=0.02 | ε=0.05 |
|---|---|---|---|---|
| dbpedia · 8B | 60 | 0.14 / 0.30 / 0.26 / **0.41 (2.8×)** | 0.37 / 0.44 / 0.43 / **0.58 (1.6×)** | 0.73 / 0.90 / 0.87 / **0.95** |
| dbpedia · 8B | 90 | 0.30 / 0.37 / 0.38 / **0.46 (1.5×)** | 0.43 / 0.62 / 0.58 / **0.67 (1.6×)** | 0.81 / 0.93 / 0.90 / **0.95** |
| dbpedia · reranker (weak judge) | 60 | 0.14 / 0.30 / 0.16 / **0.35 (2.4×)** | 0.37 / 0.44 / 0.38 / **0.50** | 0.73 / 0.90 / 0.75 / **0.90** |
| dl212223 · 8B (ρ≈0.46) | 30 | 0.08 / 0.22 / 0.14 / **0.24 (3.1×)** | 0.14 / 0.34 / 0.24 / **0.42 (3.0×)** | 0.37 / 0.73 / 0.51 / **0.77 (2.1×)** |
| dl212223 · 8B | 60 | 0.44 / 0.79 / 0.50 / **0.81 (1.8×)** | 0.58 / 0.77 / 0.61 / 0.77 | 0.89 / 0.87 / 0.88 / 0.87 |
| dl212223 · 8B | 90 | 0.70 / 0.86 / 0.71 / 0.86 | 0.81 / 0.88 / 0.81 / 0.88 | 0.88 / 0.89 / 0.87 / 0.90 |

How to read it (the external reviewer's 4-method decomposition):
1. **The unconditional gain arises without a judge.** Decision-weighted human sampling (HT) alone gives 1.2–2.9× ACT at the same document budget.
   The multiplier is ≈ 1 only in cells where the full human audit already certifies ~90%.
2. **The judge adds on top and does no harm.** With the 8B judge (ρ 0.6), HT 0.30→CV 0.41; with the weak reranker 0.30→0.35;
   on dl212223 with ρ≈0.46, 0.79→0.81. The control variate is unbiased, so even a bad judge does not fall below HT (confirmed in every cell).
3. **Existing query-level PPI almost always loses to HT alone** — the interpretation is that the conditional gain of F6 was "because the judge was used at the wrong unit (query)".
4. Validity is independent of judge and allocation (wrong ≤ 0.003 in every cell; the estimator is unbiased via HT, Proposition B′).

Limitations·next: (a) results are for precision utility (linear). set-F1 is nonlinear in nG and needs a separate estimator (Proposition C(c)).
(b) Both collections are exposed to development → confirmation of this method requires a new lock + an unused collection. (c) The document/query sample count b and
the number of training queries for candidate selection (20) are development choices. (d) The sequential (look) version and the anytime-valid extension are not implemented.

## 17. A judge from a different family — Mistral-7B-Instruct-v0.3 (same UMBRELA prompt, position_ids explicit)

### 17.1 Judge diagnostics (pair accuracy / policy-pair ρ)

| collection | Qwen3-8B | Mistral-7B |
|---|---|---|
| dbpedia-entity | 0.76 / 0.64, 0.53, 0.57 | 0.75 / 0.66, 0.55, 0.63 |
| trec-covid | 0.76 / 0.83, 0.73, 0.84 | 0.74 / 0.65, 0.78, 0.66 |
| webis-touche2020 | 0.79 / 0.81, 0.69, 0.73 | **0.51 / −0.53, 0.25, −0.45** |
| dl212223 | 0.67 / 0.47, 0.40, 0.52 | 0.61 / 0.44, 0.42, 0.47 |

- On dbpedia the two judges are equivalent (Mistral marginally ahead), on trec-covid Qwen is ahead, and **on touche (argument retrieval) Mistral
  collapses** (accuracy 0.51, negative ρ). Which judge family is favorable differs by collection, and ρ diverges widely even when overall accuracy is similar.
  → This strengthens the argument of §13.5 that judge adoption must be decided by a per-collection pilot diagnostic.

### 17.2 Certificates (500 draws)

| setting | split_t | split_ppi (Qwen / Mistral) | split_auto (Qwen / Mistral) |
|---|---|---|---|
| dbpedia, look 190 cumulative ACT | 158 | 222 / **235** | 217 / 216 |
| dl212223, look 90 cumulative ACT (primary protocol) | 313 | 318 / 304 | 308 / 308 (adoption 26% / 24%) |

wrong ≤ 1/500 in every cell. For Mistral too, always-PPI on dl212223 is slightly below human only (304 < 313), and the rule prevents the loss.
On touche, thanks to the λ clip, the gain is 0.98 even with negative ρ, so there is no harm.

### 17.3 Self-preference re-examination (policies fixed, 6 judges)

| judge | ρ (rr_thresh excluded → included) | judge bias (excluded → included) |
|---|---|---|
| same model (Qwen3-Reranker) | 0.57 → 0.25 | −0.03 → **−0.14** |
| same family (Qwen3-8B) | 0.69 → 0.48 | 0.00 → −0.04 |
| **different family (Mistral-7B)** | 0.38 → 0.30 | +0.02 → −0.04 |
| different family (MPNet) | 0.39 → 0.21 | −0.05 → +0.03 |

Since ρ for the rr pair drops even for judges from a different family, the ρ drop itself is a policy-difficulty effect, and the conclusion of §12.2 that **systematic bias (−0.14) appears only for the same-model
judge** holds when the judge families are broadened.

### 17.4 Stage 7 summary
- Judge diversity secured: Qwen3-Reranker, Qwen3-8B, Mistral-7B, MPNet, inverted. The paper's "judge family" axis is filled.
- The new method (decision-weighted document-level audit, §16.4) separately shows an unconditional gain independent of judge quality (HT) and an additional judge-dependent gain (CV).
  This is the candidate methodological contribution of the paper.
- There is no confirmatory evaluation yet: both dbpedia and dl212223 are exposed. A new lock (v0.5: decision-weighted audit + adoption rule + whether to extend to set-F1) and
  an unused completely-judged collection are needed.

## 18. Stage 8 — Strong document-level baselines (requested by the external reviewer, `81_sampling_baselines.py`)

Same expected number of document judgments, same 20 training queries (cost included), same t certificate. precision utility, 300 draws, wrong ≤ 0.007 in every cell.
Six sampling schemes: uniform / proportional to decision weight |w| (importance sampling) / |w|·√(p̂(1−p̂)) (Active-Inference style: influence × judge uncertainty) /
stratified (σ of band and common strata estimated from a 20-query pilot, |w|·σ̂) / weighted + judge control variate / active + CV.

**Budget needed for ACT ≥ 50% (fully judged query equivalents, ε=0.02) and estimator variance ratio (weighted sampling = 1)**

| collection · judge | uniform | **weighted |w|** | active (uncertainty) | stratified (pilot) | **weighted + CV** | active + CV |
|---|---|---|---|---|---|---|
| dbpedia · 8B | >90 (variance 2.9) | 70 (1.0) | >90 (3.2) | 75 (1.0) | **55 (0.73)** | >90 (1.8) |
| dbpedia · reranker | >90 (2.9) | 70 (1.0) | >90 (2.2) | 75 (1.0) | **60 (0.88)** | 75 (1.1) |
| dl212223 · 8B (ρ 0.46) | 80 (4.4) | 32 (1.0) | 50 (2.9) | 34 (1.0) | 31 (0.97) | 47 (2.4) |

How to read it:
1. **The large gain comes from sampling proportional to decision weight** (variance reduced 2.9–4.4× relative to uniform, budget less than half). This part is standard statistics,
   namely importance sampling; what is new is the **problem formulation in which the weights come out in closed form at the cutoffs of the policy pair**.
2. Pilot stratification is identical to weighted sampling (variance 1.0). Estimating label variance gives no information beyond |w|.
3. **Active sampling based on judge uncertainty (Active-Inference style) is harmful here** (variance 2.2–3.2×). This is because it undersamples decision documents on which the judge is confidently wrong,
   consistent with the warning in Robust Sampling (NeurIPS 2025). Adding CV does not recover it.
4. **The judge gives a gain only when used as a control variate, and its size depends on ρ**: on 8B·dbpedia, budget −21% (variance 0.73),
   reranker −14%, and 0 on dl212223 with ρ 0.46. In no cell is it worse than weighted sampling (unbiased estimation).

→ Contributions remaining after the strong baselines are included: (a) the closed form of the decision weights for policy-pair certification and the resulting budget allocation,
(b) the empirical rule "the judge as a control variate, not as a sampling guide" (active sampling harmful, CV harmless with conditional gain),
(c) the combination of the two components in a valid certificate (B′). The definition of the savings multiplier is fixed as **the number of human judgments needed for the same error control and the same ACT 50%**.
The paper states explicitly that the novelty lies not in "new estimation theory" but in "the exact mapping to the decision problem and its consequences".

## 19. Stage 9 — Faithful Active Inference comparison (final key check, `82_active_inference.py`)

Active sampling matched to the estimand D(q) = Σ_d w_d y_d, π_d ∝ |w_d|·√E[(y_d−ĵ_d)²] (the optimal rule of Zrnic & Candès applied to the weighted estimator),
with the residual E[(y−ĵ)²] estimated from the labels of a 20-query pilot in 10 bins of judge probability. The robust mixture is π = (1−γ)π_active + γπ_weighted.
Same pilot (cost included), same actual number of judged documents, same certificate. 300 draws, wrong ≤ 0.003 in every cell.

**Budget to reach ACT ≥ 50% (fully judged query equivalents, ε=0.02) / actual number of judged documents (at B=60) / variance ratio (weighted+CV = 1)**

| collection · judge | weighted+CV | active (calibration assumption) | **active (residual estimate)** | robust 0.5 | robust 0.3 | actual document count |
|---|---|---|---|---|---|---|
| dbpedia · 8B | 54.8 (1.00) | 70.6 (1.48) | 56.3 (1.00) | 54.4 (0.99) | 55.0 (1.01) | 1,710 |
| dbpedia · reranker | 58.7 (1.00) | 71.8 (1.28) | 60.0 (0.97) | 57.0 (0.98) | 57.4 (0.96) | 1,710 |
| dl212223 · 8B | 33.8 (1.00) | 38.7 (1.30) | 33.2 (0.95) | 32.5 (0.95) | 31.9 (0.96) | 2,431 |

**Conclusion.** Faithful Active Inference with residuals estimated from the pilot, and the robust mixture, are **identical** to decision-weighted+CV (variance ratio 0.95–1.01,
budget to reach within ±2 query equivalents). The earlier conclusion of §18 that "active sampling is harmful" is **limited to the calibration-assumption version** and is withdrawn.
That is, our method is the same thing as active/prediction-powered inference on the correct estimand.

**Final position of the contribution (honestly):**
1. There is no new estimation or sampling theory. Existing principles (importance sampling, PPI/active inference, sample splitting + Bonferroni) apply directly to the policy certification problem.
2. The remaining contributions are (a) the **problem mapping** in which the decision weights of policy-pair certification come out in closed form at the cutoffs, and the budget allocation that follows from it,
   (b) how the calibration assumption fails when the judge is used in the sampling rule, and the harmlessness and conditional gain (ρ·N/T·difficulty, F6) when it is used as a control variate,
   (c) measured self-preference of the same-model judge, (d) certification validity independent of the judge (B′) and one pre-registered external evaluation.
3. Therefore the paper is written as "a study that connects existing inference and sampling principles precisely to policy deployment certification in the era of AI judges, and shows the conditions of application and the failure modes".
   This is material for a solid TMLR paper; the original methodology needed for Featured is not secured on current evidence.

**Next decision (user):** (i) complete the paper at this position (F1 extension in the appendix, confirmation of the mapping-based method on 1 new collection) vs
(ii) find a new question for Featured — candidates: optimal allocation in a structure that reuses the same human labels across comparisons of multiple policies (combining decision weights over the whole
menu), budget stopping rules in the sequential/anytime version.

## 20. End of exploration and paper path fixed (2026-09-10)

The scope-limited exploration (menu-level audit allocation, `EXPLORATION_menu_allocation.md`) did not meet the pre-specified criteria: adaptive allocation is identical to static summed allocation
with label sharing allowed, and even an oracle that knows the true gaps leads by only 5–18%. It is preserved in the appendix as a negative result and closed.

**Fixed paper claim (as proposed by the external reviewer):**
> If the estimand needed to certify a policy is defined precisely, existing sampling and prediction-correction methods can be applied efficiently.
> The judge's overall accuracy alone is a poor predictor of that efficiency, and a wrong uncertainty assumption can reduce sampling efficiency.

**Remaining work to complete the manuscript (path 1):**
1. State the relation between the estimator and active inference in formulas (last section of THEORY, done) — no "superiority" claim.
2. Limit the set-F1 extension to the scope that supports the paper's claim (PPI correction of nG in the appendix + precision as the main result).
3. Unify the gain metric as "the actual number of human judgments needed for ACT 50% under the same error control", and do not compute a multiplier for cells that do not reach it.
4. Report the pre-registered dl212223 result (validity passed, efficiency not met) as is, and separate subsequent method changes as follow-up exploration.
5. Confirmatory evaluation of the mapping-based method (decision weighting + CV + adoption rule) on 1 unused completely-judged collection (LOCK v0.5).
6. Position precisely, in the related-work section, the degree of new insight relative to the existing literature (self-preference, calibration-assumption failure, condition map).

## 21. LOCK v0.5 confirmatory evaluation — TREC CAsT 2019 (MARCO-restricted, 159 turns, pool/turn 48.7) — 2026-09-10

Judge Qwen3-8B: pair accuracy 0.79, grade correlation 0.66, mean policy-pair ρ 0.62 (the best judge conditions so far).

### 21.1 Pre-registered criteria verdict (locked budget grid {30,45,60,90} query equivalents, ε ∈ {0.01,0.02}, 300 draws)

| criterion | result |
|---|---|
| P1 validity (boundary stress, 20 arm×budget cells) | **passed** — type I error at most 0.013 (uniform), the rest ≤ 0.003 |
| P2 unconditional gain J50(weighted) ≤ 0.6·J50(uniform) | **not evaluable → not met** — no arm reached ACT 50% at the locked maximum budget |
| P3 judge harmlessness (8B·reranker·inverted) | **passed** (maximum-budget ACT comparison rule: 0.42/0.39/0.36 vs weighted 0.387, within the −0.05 tolerance) |
| P4 active inference equivalence | **not evaluable → not met** (J50 undefined) |
| S1 conditional judge gain (expected if ρ ≥ 0.55) | ρ = 0.62 → gain expected; ratio not computable because J50 is undefined |
| S2 calibration-assumption rule inferiority | J50 undefined |

**Record of a lock design flaw.** This collection consists of conversational queries (hard for BM25·bi-encoder), so the policy gap is small and certification is hard.
At the locked maximum budget of 90 query equivalents (≈2,100 judged pairs, pilot included) the best ACT was 0.43. To use the J50 criterion, the budget grid should have been set to the collection's
difficulty (e.g., a prior estimate of the budget to reach it). This is the second lesson of §14: **pre-registered criteria must include a metric that is always defined, such as "ACT at maximum budget".**
P2·P4 remain not met, as the rules require.

### 21.2 What was observed within the locked grid (not pre-registered metrics, but numbers from the same run)
Maximum locked budget (90 query equivalents ≈ 2,100 judged pairs), ε = 0.02, ACT rate:

| uniform | weighted |w| | stratified (pilot) | active (calibration assumption) | active (residual estimate) | robust 0.5 | **weighted + 8B CV** | weighted + reranker CV | weighted + inverted CV |
|---|---|---|---|---|---|---|---|---|
| 0.09 | 0.39 | 0.37 | 0.22 (+CV 0.39) | 0.43 | 0.42 | **0.42** | 0.39 | 0.36 |

- Decision-weighted sampling vs uniform: 4.4× (direction and size both consistent with §18).
- Increment from the judge CV: +9% (0.387 → 0.42), within the range of the "small positive gain" expected at ρ 0.62.
- Residual-estimate active inference ≈ weighted+CV (0.43 vs 0.42); the calibration-assumption rule is lower (0.39; 0.22 with sampling only).
- With the inverted judge, CV is 0.025 below weighted sampling: a document-level CV used with coefficient 1 and no λ can add a little variance when the judge is adversarial
  (unbiasedness is preserved). → Adding a fold-crossed λ to the document-level CV as well is the next fix.
- Query level (set-F1, auxiliary): at T=70, split_t 48 / split_ppi (8B) **100** / reranker 57 / inverted 42, wrong all 0/500.
  Consistent with the F6 conditions (ρ 0.62, hard decision, N_unlab 89), the query-level gain of the 8B judge came out at 2.1×.

### 21.3 Post-hoc analysis (explicitly post-hoc)
Budgets {120, 150} query equivalents are run additionally to compute J50 (recorded in §21.4). The pre-registered verdict of §21.1 is final.

### 21.4 Post-hoc analysis: J50 after adding budgets {120, 150} (ε = 0.02, judged pairs; 23.4 pairs / query equivalent; wrong ≤ 0.003)

| uniform | weighted |w| | stratified | active (calibration assumption)+CV | active (residual)+CV | robust+CV | weighted+8B CV | weighted+reranker CV | **weighted+inverted CV** |
|---|---|---|---|---|---|---|---|---|
| not reached (ACT 0.23 @150) | 2,750 | 2,993 | 2,732 | 2,808 | 2,780 | 2,632 | 2,787 | **3,194** |

- The intent of P2 (weighted ≫ uniform) holds on the extended grid (uniform is not reached even at 150). The intent of P4 (residual active ≈ weighted+CV) also holds (±5%;
  the rng difference within the same arm, 2,632 vs 2,878, carries about that much noise).
- The judge increment at ρ 0.62 is −4% (8B), +1% (reranker): a "small gain", as predicted by F6·§18.
- **New finding: with the inverted judge, the document-level CV is 16% worse than weighted sampling.** A control variate with coefficient 1 adds variance when the judge is adversarial
  (unbiasedness is preserved). The role that λ played in query-level PPI was absent at the document level. → `weighted_cvl` is added (81, post-hoc), which uses a λ estimated from the pilot (regression of the true paired difference
  on the judged paired difference, clipped to [0,1]) as the CV coefficient. With λ=0 it degenerates to HT.
  The sentence "the judge does no harm" in §16–19 must be restricted to **when λ is present**.
- This fix is post LOCK v0.5, so it is not a confirmation; it is an item to verify after a lock on a third unused collection.

### 21.5 λ-corrected document-level CV (`weighted_cvl`, post-hoc) — J50 (ε=0.02, judged pairs; comparison within the same run)

| collection | judge | weighted HT | CV (coefficient 1) | **CV (pilot λ)** |
|---|---|---|---|---|
| CAsT 2019 | 8B | 2,948 | 2,625 | 2,611 |
| CAsT 2019 | reranker | 2,948 | 3,089 | 2,687 |
| CAsT 2019 | inverted | 2,948 | **3,362** | **2,720** |
| dbpedia | 8B | 2,094 | 1,628 | 1,586 |
| dbpedia | reranker | 2,094 | 1,710 | 1,637 |
| dbpedia | inverted | 2,094 | **2,388** | **2,024** |

Using the pilot-regression λ (∈[0,1]) as the CV coefficient removes the loss from the adversarial judge (+14–16%) and degenerates to the HT level, while with good judges it is
equal to or slightly better than coefficient 1. wrong 0 in every cell. Between-run rng noise is ±7% (weighted HT 2,750 vs 2,948). This result is post-lock, so
it is a confirmation item for the third lock. The document-level claim "the judge does no harm" is retained **only for the λ-corrected CV**.

## 22. LOCK v0.6 confirmatory evaluation — ANTIQUE (200 queries, pool/query 31.6, relevant = grade ≥ 3) — 2026-09-10

Judge Qwen3-8B: pair accuracy **0.47** (threshold mismatch between the UMBRELA 0–3 scale and the ANTIQUE 1–4 scale), yet grade correlation 0.63 and mean policy-pair ρ **0.59**.
The most extreme case of "accuracy does not tell you the judge's value".

### 22.1 Pre-registered criteria verdict (ε = 0.02, maximum budget 150 query equivalents ≈ 1,900 judged pairs, 300 draws)

| criterion | result |
|---|---|
| P1 validity (boundary stress, 8B·inverted) | **passed** — type I error at most 0.010 |
| P2 weighted ≥ 1.5 × uniform (maximum-budget ACT) | **not met** — 0.96 vs 0.92: **ceiling effect** (all arms saturate at the maximum budget). The co-registered criterion J50 is 776 vs 1,555 (0.50 ≤ 0.6) **passed** |
| P3 λ-CV harmlessness (3 judges, all cells with budget ≥ 60) | **passed** — J50: 8B 576, reranker 627, inverted 733 vs weighted 776 |
| P4 λ-CV ≥ coefficient-1 CV with the inverted judge | **passed** (all cells). The coefficient-1 flaw does not reproduce under the "maximum budget" definition (ceiling), but reproduces at budget 60 (0.867 vs weighted 0.913, λ-CV 0.930) |
| P5 residual active ≈ weighted+CV (maximum budget) | **passed** — 0.94 vs 0.94 |
| S1 | ρ 0.59; maximum-budget increment 0 (ceiling); at budget 30, weighted 0.28 → λ-CV 0.50 (+0.22), J50 −26% |
| S2 query level (8B, look 90) | split_t 111 / split_ppi **146** / split_auto 116 (/500), wrong 5–6/500 (≤ 0.012 < α) |

**Second lock lesson.** In v0.5 the budget grid was too small and J50 was undefined; in v0.6 the grid was large and the maximum-budget ACT saturated.
An always-defined criterion must be evaluated **at a budget where the human-only arm is in the informative range (ACT 0.2–0.8)**. From the next lock on, we use the rule
"P2: compare at the smallest executed budget at which the ACT of the uniform arm falls in [0.2, 0.8]".

### 22.2 Interpretation
- Across the 3 judges, λ-corrected CV was nowhere worse than weighted sampling (P3), and with the inverted judge, λ-CV removed the loss of coefficient-1 CV (budget 60, −0.05)
  (P4). The flaw fix of §21 is confirmed on new data.
- The gain of decision-weighted sampling (J50 halved) and the equivalence with residual active inference (P5) also replicated.
- A judge with accuracy 0.47 reduced J50 by 26% with ρ 0.59: CV works even without judge threshold calibration (unbiasedness and λ handle it).
- Validity has now passed 3/3 across the three pre-registered evaluations; the pre-specified efficiency criteria are 3/3 not met (causes: conditions unmet, grid too small, ceiling), but
  the co-registered J50 criterion passed in v0.6. The manuscript reports all of this as is.
## 15. Consistency check and preparation for strengthening the baselines (2026-09-17, featured-prep, after e6bbbad)

Goal restated: this paper is completed for **TMLR (Featured target)**; the ICML follow-up study is prepared separately after acceptance is confirmed. The question pushed as the core contribution is
"when does an improvement in judge-assisted estimation translate into an actual reduction in human verification cost, when does it not, and can this be decided before the audit".

### 15.1 Mismatches between stored results and the manuscript — findings (all recomputed from the committed CSVs; regenerated by `93_plot_F4.py`)
| Item | Manuscript (e6bbbad) | Current CSV | Action |
|---|---|---|---|
| F4 54 points corr(gain, ρ) / corr(gain, acc), T=10 | 0.78 / −0.02 | **0.7549 / +0.0261** | Update abstract, contributions, §3, F4 caption. Cause: between d678bc7→e6bbbad the legacy/modern/judged_rr CSVs were re-run (MC-MSE column added) but the aggregate values in the text were from the previous results (0.7818/−0.0154) |
| Within-group (14 collection×judge groups) demeaned corr / number of positive groups | 0.66 / all 14 | 0.659 / **13** (cqadupstack-android·rr −0.09: ρ −0.17~0.07, all gains <1) | Correct to "13 of 14", state the exception |
| Judge swap (rr→llm): gain↑ among the 13 cases with ρ↑ | 13/13 | 13/13 | Keep |
| 33 points MC-MSE gain corr / estimate-SE corr / median difference | 0.78 / 0.83 / 0.03 | 0.7847 / 0.8306 / 0.034 | Keep |
| F6 ">1.2 only when ρ≥0.55·N_unlab≥2T·ε≤0.02 all hold, otherwise 0.9–1.05" | — | **Counterexamples**: ρ=0.641, N_unlab=100, ε=0.01, T=90 → 1.376; ρ=0.57·N=300·ε=0.02 → 1.11(<1.2); ρ≤0.39 range 0.76–1.06 | Describe paragraph and caption exactly as the stored values (§15.2) |
| DL 21–23 rho_map maximum ratio | "no cell shows a gain" | 1.045 | "no cell above 1.05" |

### 15.2 Manuscript revisions (main.tex, recompilation verified)
1. Abstract compressed from about 620 words → about 420 words; the numbers above synchronized.
2. **Prop. A**: restated in finite-N form — λ* = ρσ/(σ̂(1+n/N)), V_min = σ²/n·(1−ρ²/(1+n/N)), G = 1/(1−ρ²/(1+n/N)) → 1/(1−ρ²) as N/n→∞.
   Positioned as "the baseline formula obtained by applying PPI++ to this problem". Three caveats stated (λ estimation error; with the λ∈[0,1] clip, negative ρ gives λ=0; variance ≠ certification event). Appendix proof synchronized.
3. **Correction of the bias→ρ explanation**: a constant differential bias b_j−b_m does not change ρ (if D̂=D+0.1 then ρ=1) and the CV removes it exactly. What lowers ρ is the per-query varying error and its covariance.
   Separated "mean bias (which ruins judge-only comparison)" from "loss of predictive correlation (which lowers efficiency after correction)". Same fix in THEORY_v0.1.md.
4. **Prop. B**: unbiasedness is claimed only for a *fixed* ordered pair (j,m); the estimator of the selected contrast (j, m̂_t) is not claimed to be unbiased; candidate selection is handled by simultaneous coverage over the fixed set.
   Added a third remark separating the three claims (unbiasedness for fixed contrasts / safe certification after selection / finite-sample level) and the assumptions of each.
5. **Fixed budget vs sequential certification**: the efficiency metric of §2 is defined separately for the two designs (sequential, per query: cumulative ACT; fixed budget, per document: L=1, α'=α/(M(M−1)), certification probability per budget, J50 = budget at which 50% is reached).
   Table 2 caption, §5.3 "Design" paragraph, and Limitations state "this is the cost of a pre-fixed budget, not the cost of sequential stopping".
6. **Mixed effect of the uniform baseline**: uniform spends labels even on documents with decision weight 0 → the −40~−56% is the sum of "excluding unnecessary documents + weighted sampling". Stated in §5.3, and
   the separating baseline (`uniform_nz`) row is added after it is run (`% TODO(2026-09-17)` in main.tex).
7. **Description of the exact-bound range**: Appendix E's "the paired precision difference has no a priori range" → the experiments use set-F1 and range [−1,1]; the precision difference for nested cutoffs is
   known in advance as ±(1−k_a/k_b) (k_a=10,k_b=20 → [−0.5,0.5]); set-F1 depends on n_G and is therefore not label-free. Table 6 is the cost of "this bound with the loose range".
8. **Pre-registration statement**: since the initial release commit of the public history contains the lock and the results together, the timing evidence is the dated lock document and the development log, and the reproducibility note states that this is self-reported.
9. F4 figure regenerated from the current CSVs (`93_plot_F4.py`, `05_results/ppi_gain/F4_summary.json`).

### 15.3 Code (runs on the GPU machine that has the pools bundle)
- `81_sampling_baselines.py`: arm **`uniform_nz`** added — uniform sampling within the documents whose decision weight is nonzero in any comparison (for shared sampling, the union of the per-comparison supports).
  `--dump_draws` records the unique label cost and certification outcome per (draw, arm) in `*_draws.csv`. Synthetic-pool smoke test passed (arm order: uniform, uniform_nz, weighted, …).
- `86_table2_v2.py`: `uniform_nz` row ("Uniform over decision-relevant documents (humans)"); missing rows are skipped. Also added to `84_unify_metrics.py`.
- `94_j50_ci.py`: resamples draw indices (per budget) from `*_draws.csv` but applies the same indices to all arms to keep the pairing; outputs bootstrap 95% intervals and MC SE of J50 and of the savings
  (uniform→uniform_nz, uniform_nz→weighted, uniform→weighted, weighted→weighted_cvl, etc.) → `05_results/unified/J50_CI_eps*.csv`.
- Re-run command (same as the v3 grid, `_v3b{B}` tag convention): add `--dump_draws` to the 81 call in `RUN_REVISED_ACCOUNTING.sh`, run per budget, then
  `python3 86_table2_v2.py v3 && python3 94_j50_ci.py --eps 0.02 && python3 94_j50_ci.py --eps 0.01`.
  Reading criterion: if uniform→uniform_nz accounts for most of the savings, the center of the contribution is "identifying decision-relevant documents" rather than "weights"; if uniform_nz→weighted is large, that is the value of |w|-proportional sampling itself.

### 15.4 Next steps (the part that decides Featured)
- Stage 3: bundle pilot, candidate selection and slack analysis into a "pre-audit decision rule" (inputs: pilot ρ estimate, N_unlab, pilot cost, estimated margin, bound type → output: choice among judge CV / human-only weighted sampling / complex allocation), and
  validate the locked predictions on a separate collection. Contrasting with the decisions that the existing PPI and active-inference literature already provides comes first.

## 16. Track C — A rule that explains and predicts certification cost in advance (2026-09-17, performed with stored results only)

Question: "When a judge (or any sampling design) reduces estimation variance, how much of that translates into actual certification cost savings? Can it be known before the audit?"

### 16.1 Variance-dilution cost model — `95_cost_model.py`, `05_results/unified/COST_MODEL_*.csv`, F8
Model. Under the population estimand, if all non-pilot queries are audited (f=1 at the budget where J50 is reached), the variance of the bound is mean_q(v_q)/n, and under Poisson sampling with π ∝ base,
v_q ∝ 1/b (documents per query). Hence variance ∝ v_arm / L_post (L_post = number of labels after the pilot) and certification occurs when z·sqrt(v_arm/L_post) ≤ slack.
For two arms sharing the same pilot, candidates and slack,
    (J50_a − P)/(J50_b − P) ≈ v_a/v_b,   total saving = 1 − J50_a/J50_b = (1 − v_a/v_b) · (1 − P/J50_b)  [variance reduction × post-pilot share].
v is read from the stored `var_est` (per-query estimator variance) averaged over the 3 smallest budgets (b=4, dominated by within-query sampling variance).

Validation (175 (collection, judge, ε, arm) cells of Tables 2 and 4, reference arm = weighted):
- corr(log cost ratio, log variance ratio) = **0.947**, median multiplicative error 1.10×, 90th percentile 1.35×.
- Total saving of the judge arms: corr(predicted, observed) = 0.83, MAE 0.06 (in total-J50 units). 9 of the 18 λ-CV arm cells are within ±0.02; all within ±0.05 except DBpedia ε=0.01 (predicted 0.19–0.21, observed 0.07–0.10).
- Median |log error| per arm: weighted_cv 0.06, weighted_cvl 0.08, ai_resid 0.06, robust 0.08–0.10, strat 0.05, uniform 0.09, ai_calib 0.11, **active_judge/active_cv 0.22–0.26**
  (under |w|·sqrt(ĵ(1−ĵ)) sampling, π saturates at 1 so the variance falls faster than 1/b, and the small-budget variance ratio over-predicts the cost ratio).
- Per collection: ANTIQUE 0.08, DBpedia 0.09, DL 0.08, CAsT 0.17.
- Post-pilot share (1 − P/J50_w) 0.22–0.71 → this product is exactly why the same variance reduction (≈30%) shows up as a 3% total saving on DL but 19–20% on CAsT and DBpedia.

Reading: **the judge increment in Table 2 is fully explained by "variance reduction produced by the judge × post-pilot label share".** Both factors are computable from the pilot alone
(all labels of the pilot queries are known, so the within-query HT variance of each design is computed directly; the post-pilot share comes from the pilot slack via L_post = z²·b_ref·v_w/ŝ²).

For menu allocation (83, 29 cells) the same model **fails**: corr between the ratio of realized design objective (max_j V_j/s_j²) and the post-pilot cost ratio = 0.53.
The oracle's objective is 1.5–2.2× better than static (13× for CAsT ε=0.01) but its cost is 0.79–1.00; the plug-in's objective is 2–14× *worse* in 6 of 7 cells but its cost is 0.83–1.24.
Cause: the draw-averaged objective is dominated by the draws with the smallest slack (which no design certifies), while J50 is determined by the median draw. This quantifies the
"threshold event" explanation of manuscript §7(2) and Appendix D. → **The exact location of "good estimation ≠ good certification" is not the variance-to-cost translation but (a) pilot dilution and (b) the objective being
dominated by uncertifiable draws**.

Manuscript: §5.3 paragraph "A cost model that reproduces the table", one sentence in §7(2), Appendix F (`app:costmodel`) + F8. (22 pages)

### 16.2 Literature comparison (is this result absent from prior work)
- Mani et al. 2025 (No Free Lunch): finite-sample condition under which PPI++ beats human-only, |ρ| ≳ 1/√(n/2−2) (cross-fit). With an n=45 validation half this is ≈0.22.
  Our map shows no certification gain even at ρ=0.39 → **the threshold of the certification event, not the threshold of estimation efficiency**, dominates. The calculator of 16.3 must quantify this difference.
- Zrnic & Candès 2024, Li et al. 2025, Sfyraki & Wang 2026: variance-optimal sampling, safe mixing, competitiveness of simple sampling. Results up to the variance. The "variance → certification cost" translation,
  pilot dilution and a pre-audit prediction rule are not treated.
- Ochoa Rivera & Tewari 2024 (thresholding linear bandit), Fiez et al. 2019: menu-certification allocation reduces to this framework, but since that literature's objective is also
  slack-weighted variance, the phenomenon of 16.1 "the objective is dominated by uncertifiable draws" has practical implications there too (claim restricted to the range of our data).
→ Candidate contribution sentence: "The value of a judge or sampling design is pre-computed from the pilot as (variance reduction) × (post-pilot share), and this rule correctly predicts [validation results]."

### 16.3 Query-level mechanism-map calculator — `96_ppi_calculator.py`, `05_results/rho_map/CALCULATOR_vs_map.csv`, F9
Setup: the repository's certification code as is (pick_candidate, ucb_t, ucb_ppi, cross-fit λ∈[0,1], Bonferroni over 6 ordered pairs); the 3 policy utilities generated as Gaussians
(pairwise difference variance = ppi_gain's se_human·√10, mean = population gap, judge pairwise ρ = population ρ × attenuation per noise level). 1,000 runs per cell, T=90.
| | DBpedia (64 cells) | DL 21–23 (32 cells) |
|---|---|---|
| Human-only ACT corr(sim, map) / MAE | 0.994 / 0.087 (Gaussian underestimates at small ε) | 0.981 / 0.024 |
| ACT ratio, cross-fit λ: corr / MAE / agreement rate on >1.2 | **0.854 / 0.055 / 0.92** (sim 7 cells > 1.2 vs map 12 cells: slightly conservative) | 0.735 / 0.028 / 1.00 (losses 0.88–0.97 reproduced) |
| Population-optimal λ (no estimation noise) | 0.871 / 0.060 / 0.94 | **0.12 / 0.050** — the loss disappears (all ≥0.99) |
| Clip removed | 0.840 / 0.067 | 0.09 / 0.052 |
| finite-N λ (÷(1+n/N)) | 0.811 / 0.073 / 0.89 — predicts 1.2 at N_unlab=50 (observed 0.95–0.98) | 0.05 / 0.055 |
Reading: (1) F6 is computable from (ρ, σ, gap, n, N_unlab, ε). (2) The always-PPI loss under a bad judge is the cost of **λ estimation noise** (22-query fold), not a property of the judge
itself. (3) The finite-N λ formula assumes the coefficients are known, so it is the wrong correction when they are estimated. (4) The reason a much higher ρ than the No Free Lunch estimation threshold (≈0.22) is
needed is that certification is a threshold event — at ρ=0.39 the variance gain ≤1.18 barely moves ACT after n/N dilution.
Manuscript: end of the §3 map paragraph + Appendix G (`app:calculator`) + F9. Remaining: accuracy when pilot estimates (ρ̂ BCa, σ̂, ĝap) are given as input — per-draw predicted vs observed in the same way as 97
(63 planner needs a --dump option; pools required).

### 16.4 Validation procedure for the pre-audit decision rule — `97_pilot_rule.py` (pools required)
Uses v_pilot (each arm's pilot-based within-query variance, b_ref=4), slack_pilot and slack_true, recorded per draw by `81 --dump_draws`.
- Prediction 1 (arm ranking): v_pilot ratio → post-pilot cost ratio. Prediction 2 (total saving): (1 − v ratio) × predicted share, with share from L_post_pred = z²·b_ref·v_w/ŝ².
- Decision: "recommend judge CV iff predicted saving > τ(=0.05)" made per draw and compared with the cell's observed J50 (accuracy, wrong recommendations, misses).
- Validation design: fix the rule on the 4 development collections → on 1 new collection (candidates: TREC DL 2019/2020 fully judged pool, or Touché/COVID at the document level)
  **record the predictions before running** (using only the 20 pilot queries) → compare with the observed values. This is the central experiment of the Featured claim.

### 16.5 To run on the GPU machine — `04_code/RUN_TRACK_C.sh`
Re-run 81 with `--dump_draws` on the v3 grid as is (`uniform_nz` uses a separate rng stream, so the existing arm numbers are reproduced exactly) → `86 v3`, `95`, `94`, `97`.
Reading order: (1) uniform→uniform_nz→weighted decomposition, (2) bootstrap intervals of the savings, (3) accuracy of the pilot rule (18 cells). If the rule holds, proceed to the held-out validation of §16.4.


### 16.6 Track C execution results (2026-09-17, this machine: RTX 3090, cgroup 31 cores; `RUN_TRACK_C.sh`, 84 processes in 8 minutes)
Execution note: launching all 84 at once initially gave 158 BLAS threads per process × 84 = 13k threads tangled under the 31-core quota, and not a single cell finished in 1.7 hours.
Restarting with `OMP_NUM_THREADS=1` + 31 concurrent took 8 minutes in total. Reflected in the script. The 1,176 rows of the existing arms are **exactly identical** to the committed version (uniform_nz uses a separate rng).

**(1) Savings decomposition (ε=0.02, paired bootstrap 95%, 300 draws)** — `05_results/unified/J50_CI_eps0.02.csv`
| collection | uniform→uniform_nz | uniform_nz→weighted | uniform→weighted | J50: uniform / nz / weighted |
|---|---|---|---|---|
| ANTIQUE | 21% [16,26] | 25% [21,28] | 40% [37,44] | 1,405 / 1,107 / 836 |
| CAsT | 41% [36,45] | 25% [20,30] | 56% [52,59] | 4,778 / 2,823 / 2,127 |
| DBpedia | 39% [22,45] | 25% [18,33] | 54% [42,58] | 5,563 / 3,372 / 2,543 |
| DL 21–23 | 36% [31,39] | 32% [29,36] | 56% [53,58] | 4,092 / 2,613 / 1,783 |
Reading: both are needed. "Excluding decision-irrelevant documents" is half (ANTIQUE) to 2/3–3/4 (the others) of the total saving; |w| weighting is the rest. ε=0.01: nz→weighted 20–39%, uniform→nz 23–30% (only cells where uniform reaches).

**(2) Intervals of the judge increment (weighted→weighted_cvl, ε=0.02)**: ANTIQUE Qwen 7.8 [5.1,10.5], rr 7.6 [5.1,10.3], inv 1.6 [−1.8,4.1]; CAsT Qwen 19.1 [13.5,23.8], rr 13.6 [8.5,18.6], inv 4.8 [−0.7,8.7];
DBpedia Qwen 19.5 [12.6,26.9], rr 19.4 [12.3,26.7]; DL Qwen 3.4 [1.0,6.3]. All real judges exclude 0 (DL barely); the inverted judge includes 0.
coef-1 CV + inv on ANTIQUE: −5.1 [−8.8,−1.6] → a genuine loss; the λ-fit removes it (1.6 [−1.8,4.1]).

**(3) Pre-audit decision rule (`97_pilot_rule.py`, 18 judge-arm cells)**
| | ε=0.02 | ε=0.01 |
|---|---|---|
| corr(pilot variance ratio, observed post-pilot cost ratio) | 0.95 | 0.90 |
| Fully pilot-based total saving prediction: corr / MAE | 0.88 / **0.034** | 0.89 / 0.057 |
| Decision "saving > 5%" (pilot majority) accuracy | 15/18 (2 wrong recommendations, 1 miss) | 14/18 (3 wrong recommendations, 1 miss) |
| λ-CV vs coef-1 CV ranking | 9/9 | 8/9 |
Except for one wrong cell, dbpedia/rr/coef-1 CV (ε=0.01, predicted 0.06, observed −0.01), all wrong cells have observed savings of 0.03–0.06, near the 5% threshold. The inverted judge is never recommended.
The weak factor is the post-pilot share: predicted L_post (z²·b·v_w/ŝ²) medians 186/828/793/659 vs observed 205/1152/1514/397 — within a factor of 2. → The rule is reliable for "whether to use the judge and which design" and rough for "how much".

Manuscript: uniform_nz rows in Tables 2 and 4, §5.3 decomposition paragraph, intervals, and "predictable from the pilot" paragraph; abstract, contribution 3, Limitations, Appendix F updated. 0 TODOs. F7 v3 regenerated.

### 16.7 Next: held-out validation (the central experiment for Featured)
Since the rule (τ=5%, b_ref=4, pilot majority) has been fixed on the 4 development collections, record the predictions on a new collection **before running** and compare with the observed values.
Candidates: TREC DL 2019/2020 fully judged pool (need to check whether the bundle's trecdl stack has _llm judgments), Touché 2020·TREC-COVID (judged stack; no _llm file → need to generate Qwen3-8B judgments with 66, GPU available).
Procedure: (a) compute v_pilot and slack from the 20 pilot queries only → commit the prediction file (hash fixed), (b) run 81 in full, (c) score with 97. Since the prediction itself is determined by code, "pre-registration" is proven by the commit time of the prediction file.

### 16.8 Held-out validation round 1 — TREC-COVID (50 queries)·Touché 2020 (49 queries), reranker judge (2026-09-17)
Protocol: generate only the pilot record with `81 --predict_only` → fix and commit the predictions with `98_heldout_predict.py` (54deba0; corrected version 9ca0f6b) → audit (300 draws, budgets 2–45) → score with `97`.
The pilot columns of the prediction record and the audit match exactly (difference 0). The first lock's 98 had a bug omitting the square of the slack (affects L_post only), disclosed in LOCK_NOTE.md.
Result files: `05_results/heldout/PILOT_RULE_rr_{ho,ho10}_eps*.csv`, `PREDICTIONS_rr*.md`.

**(1) Pre-fixed predictions that were correct (pilot 20)**: per-arm post-pilot cost ratio corr(predicted, observed) = 0.957 (ε=0.02) / 0.971 (ε=0.01), over all arms.
D3 (λ-CV vs coef-1) 2/2 — on COVID, coef-1 was predicted to be better and it was (cost ratio 0.47 vs 0.75). Decomposition prediction uniform→nz 0.50/0.28 vs observed 0.46/0.28.
**(2) What was wrong**: v1 formula's post-pilot label count 592 vs observed 203 (COVID), 246 vs 58 (Touché) — 3–4× overestimated. Two causes confirmed:
  (a) under the population estimand the pilot portion (20 of N=50) is known exactly, so the precision required on the rest is N/N_R (=1.67)× looser → labels (N_R/N)²=0.36×;
  (b) v1 assumes f=1 (ignores the effect of remaining between-query variance at small budgets).
**(3) Improved predictors (built after seeing the held-out, so explicitly not pre-registered predictions)** — `lib/costpredict.py`
| Predictor | Content | dev 18-cell decision accuracy ε=.02/.01 | dev L_post (CAsT, DBpedia) | held-out pilot20 decision | held-out pilot10 decision | held-out L_post (COVID, Touché) |
|---|---|---|---|---|---|---|
| v1 | z²·b·v/s² | 14/18, 14/18 | 828, 793 (observed 1152, 1514) | 3/4, 1/4 | 2/4, 2/4 | 592, 246 (observed 203, 58) |
| v2 | two-stage variance + known pilot portion | 14/18, 14/18 | 633, 751 | 4/4, 4/4 | 2/4, 2/4 | 213, 88 |
| v3 | v2 computed per comparison, take max (binding comparison = max v/s²) | 15/18, 14/18 | 636, 642 | 4/4, 4/4 | 4/4, 3/4 | 215, 46 |
| v4 | v2 + cross-fitted slack (candidate-selection half / slack-measurement half) | 16/18 (0 wrong recommendations), 14/18 (0 wrong recommendations) | 1541, 1684 (overestimated); ε=.01 CAsT diverges | 4/4, 4/4 | 3/4, 3/4 | 215, 89 |
Reading: **the cost ratio (which arm, and how much relatively) is predicted stably from the pilot alone (correlation 0.95–0.97, on both dev and held-out).** The absolute label count is correct on held-out with v2/v3 but
still ~2× underestimated on dev CAsT and DBpedia — in those two collections the policies are close (true slack≈ε) and the slack estimate from the 20 pilot queries is optimistic due to the winner's curse
(plain 0.037/0.039 → cross-fitted 0.023/0.025). Cross-fitting (v4) removes the optimism and gives 0 wrong recommendations, but is unstable when the slack is small. In the paper, v3 as the "improved version" and v4 as the "conservative variant".
Summary of the decision rule: to avoid wrong recommendations use v4, to avoid misses use v3. On dev the two rules agree in 17/18 cells at ε=.02 (15 of them correct) and 12/18 at ε=.01 (11 of them correct); the disagreeing cells all have observed savings near the 5% threshold (0.03–0.06) or, as with CAsT ε=.01, v4's slack diverges.

### 16.9 Held-out validation round 2 — Qwen3-8B judge, **v1–v4 all fixed before the audit** (694e00b) (2026-09-17)
Judgment generation: `66_llm_judge.py` (this machine, RTX 3090, COVID 4,165 pairs in 13 minutes·Touché 1,858 pairs in 6 minutes), accuracy 0.764/0.804 (consistent with the manuscript's 0.76/0.79).
| | pilot 20, ε=.02 | pilot 20, ε=.01 | pilot 10, ε=.02 | pilot 10, ε=.01 |
|---|---|---|---|---|
| corr(pilot cost-ratio prediction, observed), all arms | 0.918 | 0.939 | 0.932 | 0.939 |
| Judge CV observed total saving (COVID cv/cvl; Touché cv/cvl) | .058/.058; .007/.007 | .070/.067; .024/.024 | .172/.148; .089/.074 | .207/.157; .112/.088 |
| Decision accuracy v1 / v2 / v3 / v4 (4 cells) | 2/4, 2/4, **4/4**, 2/4 | 2/4, 2/4, **4/4**, 2/4 | all 4/4 | all 4/4 |
| Total saving MAE v2 / v3 / v4 | .074/.053/.075 | .098/.067/.097 | .162/.113/.166 | .238/.183/.221 |
| L_post observed vs v3 (COVID; Touché) | 203 vs 215; 58 vs 46 | 229 vs 264; 72 vs 62 | 304 vs 369; 97 vs 96 | 376 vs 491; 122 vs 143 |
Reading: (1) The cost ratio is as predicted in advance (0.92–0.94). (2) Among the pre-fixed decisions, v3 is 16/16 correct; the 4 errors of v1, v2 and v4 are all cells in Touché pilot 20 whose predicted saving
of 0.044–0.055 was just above the 5% threshold (observed 0.007–0.024). (3) The absolute size is overestimated at pilot 10 (v3 MAE 0.11–0.18) — the prediction is right in direction and on the upper side in magnitude.
(4) The observed cost ratio 0.47 for COVID pilot 20 is a censored upper bound, because ACT ≥ 0.5 already at the smallest budget of the grid (B=2), so the true ratio is at or below it (not inconsistent with the predicted 0.25–0.38).
(5) D3 (coef-1 vs λ-CV): COVID predicted "coef-1 wins" → observed tie at pilot 20 (1760.8 vs 1760.2, censored), coef-1 wins at pilot 10 (940 vs 968) ✓; Touché predicted "λ-CV wins" → observed tie / coef-1 marginally ahead (434 vs 441) ✗ (difference 1–2%, within MC error).
Conclusion: **"which design is relatively cheaper and by how much" and "does the judge save 5% or more" can be predicted in advance from a 20-query pilot (v3 16/16 + reranker 8/8 at pilot 20).
"How many labels it takes" is within 1.5× (pilot 20) and within 2× (pilot 10).** Limitation: the two held-out collections have 49–50 queries, so the pilot share is large and the total saving is small (≤ 21%);
pre-registered validation at large N requires a new collection.

### 16.10 Correction: un-censoring the pilot-20 held-out cells (budget B=1 added, 2026-09-17)
In the pilot-20 tables of §16.8–16.9, cells where the judge arm's ACT was already ≥0.5 at the smallest budget B=2 (COVID cv, Touché cv/cvl; llm all four cells) were censored and their J50 was an upper bound.
Final numbers with B=1 added (4 processes) to remove the censoring (`PILOT_RULE_{rr,llm}_ho_eps*.csv`, ACT at B=1 is < 0.5 for all arms):
| pilot 20 | reranker ε=.02 / .01 | Qwen3-8B ε=.02 / .01 |
|---|---|---|
| corr(pilot cost-ratio prediction, observed), all arms | 0.958 / 0.969 | 0.944 / 0.946 |
| Judge CV observed total saving COVID cv, cvl; Touché cv, cvl | .062,.028; .008,.010 / .065,.026; .009,.011 | .072,.064; .028,.027 / .081,.067; .040,.035 |
| Decision accuracy v1 / v2 / v3 / v4 | 3/4, 4/4, 4/4, 4/4 / 1/4, 4/4, 4/4, 4/4 | 2/4, 2/4, **4/4**, 2/4 / 2/4, 2/4, **4/4**, 2/4 |
| Total saving MAE v2 / v3 / v4 | .050/.035/.051 / .057/.039/.057 | .075/.052/.075 / .095/.066/.095 |
| L_post observed vs v2 / v3 (COVID; Touché) | 203 vs 203/203; 58 vs 88/47 / 228 vs 248/248; 72 vs 125/63 | (judge-independent, identical) |
Final verdict of the pre-registration (llm, v1–v4 all fixed before the audit): **v3 8/8 correct, v2·v4 4/8 (on Touché predicted saving 0.044–0.055 vs observed 0.027–0.040: just on either side of the 5% threshold), v1 4/8.**
On Touché, even with a strong judge (cost ratio 0.38), the pilot is 93% of the total cost so the total saving stays at 3–4% — a held-out example of why "how good the judge is" and "how much is saved" differ.
reranker (only v1 fixed in advance; v2–v4 post hoc): v2–v4 8/8, v1 4/8.

### 16.11 Manuscript compression (2026-09-17)
25 pages → 20 pages (11 pages of main text + references + 8 pages of appendix). Abstract 450 → 370 words. The 4 contribution items reduced to 2–4 sentences each; §3 MSE re-measurement details, §4 exactness, §5.3 estimand-attribution numbers, §7 CAsT/ANTIQUE narrative, and §8 (2) menu allocation cut to less than half, with details moved to the appendix (estimand attribution to F; D and E compressed). Cost model, pilot rule and held-out merged into two paragraphs of §5.3. All numbers carried over unchanged from the previous text (no new numbers). Remaining edits: polishing sentences in Appendices F/H, unifying table captions.

### 16.12 Title, figures, style (2026-09-17)
- Title: "Certifying Retrieval Policies with Non-Neutral AI Judges: What a Judge Saves, and How a Pilot Predicts It".
- Figures: F4–F9 regenerated from the stored CSV/parquet with `99_figures.py` + `lib/figstyle.py` in one style (STIX typeface, no in-figure titles, Okabe–Ito palette, readable collection and judge names, legend outside). F5 recomputes cumulative ACT from `planner_v2/judged_*_ext/planner_v2.parquet`.
- Style: removed or split "Three things follow", "It is tempting to read", self-congratulatory phrasing, and mid-sentence em-dash parentheticals. Numbers unchanged.
- The co-author trailer in commit messages removed per the repository convention (§12.9); local commit-msg hook installed.

### 16.13 Response to review (5210aee) (2026-09-17)
1. **Reproduction error recovery**: fixed the bug where `81 --predict_only` overwrote the audit summary CSV with an empty file (moved the prediction branch before the summary save). The 84 dev summaries restored from the Track C commit (3b4ebd5); the reranker held-out re-audited (per-draw recorded values identical, 11 columns added). Confirmed that `86 v3` regenerates TABLE2_v3 exactly. Name-collision remnants (`_heldout_b*`) deleted.
2. **Single-pilot execution rule** (`97`, `PILOT_RULE_single_eps*.csv`): per draw, if that pilot's v3 predicted saving > 5% then λ-CV, otherwise weighted. ε=0.02 savings (always λ-CV / rule): ANTIQUE 7.8/6.0, CAsT Qwen 19.1/19.1, CAsT rr 13.6/10.7, DBpedia 19.5/18.5·19.4/19.2, DL 3.4/2.9, inverted judge 1.6·4.8 (noise)/0 (not adopted); held-out Qwen pilot 20: COVID 6.4/5.7, Touché 2.7/0.8 (per-draw pilot cost used, per the review); pilot 10: 14.8/14.8, 7.4/7.5. Wrong-cert of the rule: 0. Per-pilot agreement rate dev 45–98% (DL 21–23 is near the threshold, hence 45%), held-out Qwen 52–99%.
3. **Correction of the held-out description**: not "a collection unused in development" but "v2–v4 were built after seeing the reranker results, and v1–v4 were pre-fixed on the same collections with a new judge (Qwen3-8B)"; the time at which v3 was designated the main predictor (after the results) is stated. The 8/8 majority-vote result and the single-pilot performance are described separately.
4. Manuscript: Prop. A proof in finite-N form, F8 caption 175→193, F8 moved to main text §5.3, F5 moved to Appendix E.
5. Remaining: independent validation with a single fixed rule on a large collection never used in development (the final experiment of §16.7).

### 16.14 LOCK v0.7 results — NeuCLIRBench English mono, a collection never used in development (2026-09-17, rule unchanged)
Data: 105 topics, pool 70.7 documents/query, relevant (gain≥1) fraction 0.27, mean nG 18.9. Judge accuracy: Qwen3-8B **0.437** (relevance-definition mismatch, same situation as ANTIQUE), reranker 0.586. Wrong-cert at most 0.013 (all cells).
| Qwen3-8B | Predicted λ-CV cost ratio | Observed | Predicted total saving (v3) | Observed | v3 decision (pilot majority) | Correct? | Single-pilot rule saving / always-CV / adoption rate |
|---|---|---|---|---|---|---|---|
| pilot 20, ε=.02 | 0.71 | 0.90 | 4.5% | 2.1% | not recommended (0.47) | ✓ | 1.6 / 2.1 / 47% |
| pilot 20, ε=.01 | 0.72 | 0.82 | 5.7% | 5.3% | recommended (0.52) | ✓ | 2.3 / 5.3 / 55% |
| pilot 10, ε=.02 | 0.76 | 0.87 | 7.0% | 6.0% | recommended (0.60) | ✓ | 4.6 / 6.0 / 57% |
| pilot 10, ε=.01 | 0.76 | 0.93 | 8.2% | 3.8% | recommended (0.61) | ✗ | 5.0 / 3.8 / 60% |
reranker and inverted: predicted "not recommended" 8/8 correct (observed savings −0.8~+2.8%, noise); coef-1 CV predicted as a loss (1.21/1.34) → observed loss (1.03–1.13 / 1.14–1.29), direction agrees.
All-arm cost-ratio prediction correlation: Qwen 0.86–0.92, rr 0.89–0.97, inv 0.89–0.96. Decomposition prediction 0.66/0.47/0.82 vs observed 0.62/0.52/0.82.
L_post (pilot 20): observed 401/592 vs v3 324/440 (underestimated 20–25%), v2 351/484, v1 536/738 (overestimated). pilot 10: observed 578/798 vs v3 349/484 (underestimated 40%).
**Pre-specified criteria (re-tallied literally as written, incorporating external re-review)**: (a) the rule is within 2 percentage points of the better of the two fixed policies — **9** of 12 cells satisfied (not satisfied: Qwen p20 ε=.01 2.35 vs 5.26; inverted p20 ε=.01 0.09 vs 2.73; reranker p10 ε=.01 0.56 vs 2.79 — the latter two are sub-5% gains the rule is designed to ignore, but since the lock has no exclusion clause they are counted as not satisfied); (b) inverted-judge adoption ≤10% — 1 of 4 cells satisfied (6.9/11.6/11.9/14.7%). Both criteria satisfied in all settings: **not satisfied**. The Qwen correlation range is 0.825–0.923 (the manuscript's 0.86–0.92 is a typo → corrected to 0.83–0.92). At pilot 20, individual pilots' recommendations are 47%/52%, nearly half and half.
Reading: relative cost, decomposition and "do not use a weak judge" held on the new collection too. The Qwen λ-CV cost ratio is optimistic (0.71 predicted vs 0.82–0.93 observed): with a judge of accuracy 0.44, the λ fitted on 20 pilot queries and the residual variance appear to be overfitted within the pilot (λ is fitted on the same pilot and the variance is evaluated on that pilot). Improvement candidate (v5): compute v_pilot with a cross-fitted λ within the pilot — post hoc on this collection, so separate validation is needed. The decision is 3/4 of 4/4 and all four cells are near the 5% threshold (observed 2.1–6.0%). The total saving is small because the pilot is 70–78% of the total cost (p20).

### 16.15 Final style pass (2026-09-17, in response to the external review's "impression that an AI keeps repeating summaries")
- Last sentence of the abstract: "We propose no new estimator" removed, ending with the result (the same sentence in the introduction is kept).
- Table 2 commentary: result-restating sentences such as "neither part is dispensable" deleted.
- The COVID/Touché validation paragraph (~550 words) split into three paragraphs: design / results / limitations. The rule saving-rate numbers present in Table 3 removed from the text and replaced with a table reference. The pilot-10 "0.11–0.18" figure had an unclear source and was replaced with a table reference.
- NeuCLIR results paragraph (~400 words): split into two paragraphs, Qwen predicted vs realized / rule and criteria; the four cells' saving-rate numbers refer to the table; the final "The independent test thus supports…" summary sentence deleted (only the one near-threshold sentence kept).
- Negative results: "Four things did not work as hoped" → "Four results bound the contribution"; the retraction history in (4) moved to Appendix I (reproducibility note), the main text keeps only the current results.
- Pre-fixing times, the failed criteria (9/12, 1/4), and the order of predictor revisions all unchanged. Compiles to 22 pages, no warnings. Abstract 303 words.
- (Addendum, same day) Review follow-up: last sentence of the abstract deleted (273 words), the Negative results lead-in "Four results bound the contribution." deleted. The `TMLR_submission_anonymous.zip` in the whole repository contained the pre-style-fix manuscript, so it was re-bundled with the latest tex/pdf/figures (identifying-string check passed). 22 pages, no warnings.
