# Exploration (scoped, upper bound 3–5 days): document audit allocation for certifying the whole menu — 2026-09-10

## 1. Problem definition (1 page)

**Setting.** Policy menu M = {1..M}; each policy m returns a set R_m(q) from the shared pool of query q (prefix truncation or threshold rule).
The utility has a form whose denominator is independent of the labels, u_m(q) = Σ_{d∈R_m(q)} y_d / Z_m(q) (precision@cutoff: Z_m = |R_m|).
After the candidate m̂ is determined from the pilot, certification returns ACT when UCB_j ≤ ε for every competitor j ≠ m̂ (Bonferroni over ordered pairs).

**Estimand.** The per-query difference of the pair (j, m̂), D_j(q) = Σ_d w_j(d) y_d, w_j(d) = 1[d∈R_j]/Z_j − 1[d∈R_m̂]/Z_m̂ (corrected version of Proposition C).
Key structure: **the label y_d of a single document is used simultaneously in every comparison j with w_j(d) ≠ 0.** Only the weight differs from comparison to comparison.

**Allocation problem.** Choose the document inclusion probabilities π(d) (budget b per query) so that the certification condition max_j UCB_j ≤ ε is satisfied with the fewest total labels.
UCB_j = μ̂_j + z·se_j, se_j² ≈ (1/n_q) Σ_q Var_π[D̂_j(q)], Var_π[D̂_j(q)] = Σ_d w_j(d)² v_d (1−π_d)/π_d (v_d = residual variance;
with the judge CV, v_d = E[(y_d − ĵ_d)²]). That is, the objective has a **max_j** form, and the constraint has a label-sharing structure.

**What if existing methods are plugged in as they are?**
- (a) Applying pairwise Active Inference independently: π_j ∝ |w_j|√v per comparison. To exploit label sharing, one samples with the union rule π ∝ Σ_j |w_j|√v or
  max_j |w_j|√v (static, treating all comparisons equally). → **static summed weighted sampling**. This is the strong baseline.
- (b) ε-best-arm identification (LUCB family, including ε-best-answer identification): in each round, samples are concentrated on the comparison that "decides the outcome" (the pair whose UCB is
  closest to ε). However, the sampling unit is an arm pull (= judging an entire query), and the structure in which a document label is shared across several arms
  is not modeled.
- (c) Multi-task PPI (several estimands share a predictor): the sharing across estimands is at the predictor level; there is no weight-sharing structure of the labels.

**Remaining problem (hypothesis).** Because the objective is max_j, the optimal π is the λ-mixture π ∝ Σ_j λ_j |w_j| √v that **does not treat all comparisons equally but gives larger weight to the
binding comparisons that decide the outcome**, where λ is updated adaptively from the current UCB slack. This puts the adaptivity of (b) on top of the
document-level sharing structure of (a). **For there to be novelty, this adaptive allocation must certify earlier, at the same total number of labels, than a static summed allocation that
allows label sharing in the same way.** Structural conditions under which a gain arises: the menu contains (i) policies that already lose by a wide margin (those comparisons need
no samples), and (ii) the documents with large weight in the close comparisons and the documents with large weight in the losing comparisons are **different**. If the two sets coincide,
there is no difference from the static sum.

**Prior work to contrast.** ε-best-answer identification (Jourdan et al. 2022), multi-task PPI (2026), correlated/structured
best-arm identification, top-two Thompson sampling (adaptivity in which pair to compare). This exploration differs from these only in that "the sampling unit is the document and
the labels are shared across comparisons".

## 2. Constructed example

Menu of 4, candidate m̂ = A. Competitor B is close to A (μ_B − μ_A = 0.005, ε = 0.01); C and D are far behind (−0.10).
Each query has a pool of 30 documents. The disagreement region between A and B is rank 8–12 (5 documents); C and D return only rank 1–3 (disagreement with A at rank 4–8).
- Static summed allocation: π ∝ |w_B| + |w_C| + |w_D| → about 2/3 of the budget goes to rank 4–8 (the C and D comparisons). But the C and D comparisons
  already have UCB below ε (−0.10 + z·se ≪ 0.01), so they need almost no samples.
- Adaptive allocation: once UCB_C, UCB_D ≪ ε is confirmed after round 1, λ_C = λ_D → 0.1, and the remaining budget is concentrated on rank 8–12 (the B comparison).
  se_B shrinks by a factor of √(1/3) → at the same total number of labels, UCB_B ≤ ε is reached about 3 times faster.
- If the decision documents of the two comparisons overlap (e.g. if C and D also differ only at rank 8–12), static and adaptive are the same. **The gain is proportional to the degree of
  non-overlap of the decision document sets** — this is the quantity the new method must explain, and it is measurable on real data.

## 3. Minimal experimental design (`83_menu_allocation.py`)
- Data: dbpedia-entity (development), menu 4 = {glob, trunc, ad, rr_thresh}, precision@cutoff, 8B judge CV, ε ∈ {0.01, 0.02}.
- Common: pilot 20 queries (candidate selection and cost included), total document budget B, R = 3 rounds, new queries in each round (document reuse only across comparisons
  within a round — keeps the unbiasedness of the adaptive inclusion probabilities simple); the certificate is the t-UCB of the per-query estimates pooled over rounds.
- Comparison groups (all allow label sharing): (A) static sum π ∝ Σ_j |w_j|; (B) pairwise split (the round's queries divided into 3 equal parts, a dedicated π ∝ |w_j| for each pair,
  no sharing); (C) **adaptive binding** λ_j^{(r)} ∝ 1/max(ε − UCB_j^{(r−1)}, δ) with π ∝ Σ_j λ_j |w_j| (floor λ ≥ 0.1);
  (D) oracle λ (uses the true gaps, upper bound).
- Metrics: ACT and wrong per budget, total number of labels to reach ACT 50%, non-overlap index (Jaccard) of the decision document sets.
- Decision: continue if (C) certifies significantly earlier than (A) at the same number of labels and the difference is explained by the non-overlap index; otherwise the exploration ends.

## 4. Results (2026-09-10, `05_results/menu_allocation/`, 300 repetitions, wrong = 0 in all cells)

Budget at which ACT ≥ 50% is reached (in fully judged query equivalents), ε=0.02 / ε=0.01:

| collection · judge | non-overlap index | static sum (shared) | pairwise split (unshared) | **adaptive binding** | oracle λ |
|---|---|---|---|---|---|
| dbpedia · 8B | 0.55 | 53.3 / 88.1 | 65.5 / >90 | 51.2 / 88.2 | 43.5 / 81.0 |
| dbpedia · reranker | 0.55 | 65.2 / >90 | 68.0 / >90 | 63.3 / >90 | 60.9 / >90 |
| dl212223 · 8B | 0.83 | 38.7 / 47.8 | 40.5 / 69.1 | 38.7 / 46.4 | 38.2 / 43.7 |

- Label sharing itself matters (the pairwise split is clearly inferior at ε=0.01). However, **the static summed allocation already captures all of the sharing.**
- The adaptive allocation shows no difference from the static sum (±2 query equivalents, within the noise range). **Even the oracle that knows the true gaps** gains only 5–18%
  — that is, on this menu and data, the upper bound on what adaptive allocation can gain is itself small. This is because the UCB of the losing comparisons already drops below ε
  with a small number of labels, so the static sum wastes little budget.
- Even on dl212223, whose non-overlap index is 0.83, the oracle gain is only 9% at ε=0.01. The 3-fold gain structure of the constructed example (§2)
  did not appear on real data.

**Decision: the pre-specified criterion ("adaptive certifies significantly earlier than the static sum") is not met → exploration ends.** This direction cannot become
a core contribution of the paper. It is kept in the appendix as a negative result: "static label-sharing allocation suffices, and the upper bound of adaptive allocation is small".
Time spent: about 2 hours of coding and running (within the 3–5 day upper bound).
