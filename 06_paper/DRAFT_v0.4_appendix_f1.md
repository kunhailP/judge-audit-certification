# Draft v0.4 — Appendix A: document-level auditing for set-F1, and validity stress tests (2026-09-10)

## A.1 Why set-F1 is different
set-F1 normalises by k_m + nG(q) with nG the number of relevant documents in the pool, so D(q) is a ratio of label sums and the
per-document decision weights of Proposition C are not exact. We therefore evaluate two approximations, both computable
without unobserved labels: (i) **linearisation at the judge labels** (weights w_d = ∂D/∂y_d at y = ĵ; every pool document
gets a non-zero weight through nG) with the judge as control variate, optionally **bias-corrected** by the mean linearisation
remainder measured on the fully labelled pilot; (ii) **plug-in Horvitz–Thompson** (humans only; tp_a, tp_b, nG estimated by HT
under sampling ∝ |w_d| at the pilot base rate, then plugged into F1).

## A.2 Measured linearisation remainder
Mean of D̂_lin − D over fully labelled queries: dbpedia +0.002 (mean |·| 0.011), dbpedia with the reranker judge −0.012 (0.018),
dl212223 −0.029 (0.030). On dl212223 the remainder is three times ε = 0.01, so the un-corrected linearised certificate is not
trustworthy there; the bias-corrected variant is the one we report, and the plain variant is shown only to document the effect.

## A.3 Results (300 draws; ACT rate; wrong-certificate rate ≤ 0.003 in every cell)
| collection · judge | budget (full-query eq.) | human_full | plug-in HT (humans) | query-PPI | lin+CV | lin+CV, bias-corrected |
|---|---|---|---|---|---|---|
| dbpedia · Qwen3-8B, ε=0.01 | 60 / 90 | 0.23 / 0.47 | 0.78 / 0.76 | 0.33 / 0.57 | 0.55 / 0.73 | 0.32 / 0.45 |
| dbpedia · Qwen3-8B, ε=0.02 | 60 / 90 | 0.39 / 0.67 | 0.83 / 0.87 | 0.54 / 0.80 | 0.73 / 0.87 | 0.49 / 0.67 |
| dbpedia · reranker, ε=0.01 | 60 / 90 | 0.23 / 0.47 | 0.78 / 0.76 | 0.27 / 0.44 | 0.78 / 0.76 | 0.41 / 0.45 |
| dl212223 · Qwen3-8B, ε=0.01 | 30 / 60 | 0.10 / 0.70 | 0.72 / 0.97 | 0.10 / 0.66 | 0.86 / 0.97 | 0.42 / 0.79 |

Reading: the plug-in HT estimator (no judge) certifies far more often than full-query auditing at equal document budget
(dbpedia ε=0.01, B=60: 0.78 vs 0.23) — set-F1's cost structure (a whole pool per query) makes document-level sampling
especially valuable. The un-corrected linearised certificate is inflated exactly where the remainder is large (dl212223);
after bias correction it sits between full-query auditing and the plug-in. J50 values for this block are in TABLES v0.1
(Block D) and are **not** compared with the precision blocks.

## A.4 Validity stress tests (all document-level arms, both utilities)
The wrong-certificate rate is uninformative when the candidate's true regret is ≤ ε (an always-ACT method would score 0).
Two stress designs, 300 draws each, α = 0.10:
- **Forced-worst** (candidate := true worst policy): type-I error ≤ 0.007 for every arm — but the gaps are then far above ε,
  so this is an easy test.
- **Boundary** (candidate := runner-up; ε := 0.9 × its true regret, so any ACT is a type-I error; realised ε ≈ 0.015–0.034):
  type-I error ≤ 0.03 for every arm on both collections (uniform sampling 0.013–0.030, all others ≤ 0.013).
Both tests are below α by a margin because the certificate is Bonferroni-corrected over |M|(|M|−1) ordered pairs. They do
not probe ε = 0.01 directly on dl212223, where the linearisation remainder is largest; there the bias-corrected variant is
mandatory and the plain variant is reported as invalid-by-construction.
