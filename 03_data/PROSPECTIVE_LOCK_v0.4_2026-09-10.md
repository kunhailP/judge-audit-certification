<!-- English translation (2026-09-18) of the lock document committed on 2026-09-10; wording of the criteria translated literally. -->
# PROSPECTIVE LOCK v0.4 — 2026-09-10 (2027 design, pre-registration of the primary run on a new collection)

This document is written and committed **before the qrels and judgment results of the target collection (TREC DL 2021/2022/2023 passage, MS MARCO v2)
are used in any analysis**. No item may be changed after the commit. The primary run is executed exactly once, and the result is reported regardless of pass or fail.

## 1. Target
- TREC Deep Learning 2021, 2022, 2023 passage ranking, MS MARCO v2 passage corpus.
- Reason for selection (pre-specified): never used during the development stage (§9–13); deep NIST judgments; total number of queries ≥ 200 (supports look 190);
  the corpus is accessible in this environment (28 MB/s measured).
- Pool construction: for each query, the union of the top-30 of the 4 systems (BM25, MiniLM, MPNet, Qwen3-Embedding-0.6B) **using the set of NIST-judged passages as the corpus**
  (the same `69_judged_pool.py` rule as for the development collections). relevant = grade ≥ 2. The three years are merged into a single collection
  `dl212223` for evaluation; per-year results are reported as a side result.
- Training (classifier, τ, trunc, transfer of policy parameters): BEIR legacy 4 pools (same as in development, no contact with the target).

## 2. Fixed methods
- ε_sel = 0.01, α = 0.1, look {10,30,50,70,90,120,150,190} ∩ [≤ n/2], Bonferroni over looks × M(M−1), M = 3
  (glob_probe, trunc, ad_probe).
- Certificate: split (first half for training / second half for validation). Candidate = LCB maximum. Three comparison methods:
  (a) split_t human-only, (b) split_ppi always PPI++ (cross-fit λ), (c) split_auto rule-based adoption.
- Adoption rule: |A_T| ≥ 30, bootstrap lower bound of ρ for the candidate–runner-up pair (level 1−α, B=400) → PPI if the gain score 1/(1−lcb²) > 1.1.
- Judge: Qwen3-8B UMBRELA (non-thinking, position_ids explicit, passage 1500 characters, batch ≤ 32), threshold 0.5.
  Auxiliary comparisons: Qwen3-Reranker-0.6B, inverted reranker (adversarial contrast).
- 500 repetitions, seed rule `5_000_000 + 1000*rep + crc32("planner|dl212223") % 100000`, independent rng stream per method.
- Judge diagnostics (ρ, errors inside/outside the interval, PPI gain) are reported as descriptive statistics.

## 3. Pre-registered decision criteria
- **Primary**: for all three methods, the Clopper–Pearson 95% upper bound of the wrong-certificate rate ≤ 0.15 and the point estimate ≤ α = 0.1.
- **Secondary (efficiency)**: in cumulative ACT at T=190, split_ppi ≥ split_t × 1.2, split_auto ≥ split_t. Falling short is a result, not a failure.
- **Practicality of the judge adoption rule**: the fraction of cases in which split_auto adopts the inverted judge ≤ 5%.
- In case of failure, cause analysis is performed only in a separate section and does not overwrite the primary numbers.

## 4. Code
- Analysis code commit: the parent of this lock commit (`git log`). The pool builder is `75_dlv2_pool.py` for the v2 corpus (same rule as 69, only the corpus
  reader replaced); it is added after the lock, but the pool rule is fixed above.
