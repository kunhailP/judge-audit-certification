# Draft v0.4 — Section "Evaluation protocol and metric" (2026-09-10)

## Error control (identical for every method)
Every method emits ACT only when, for the candidate m̂ chosen on a held-out pilot, all competitors j ≠ m̂ satisfy
UCB_{α'}(μ_j − μ_m̂) ≤ ε with α' = α / (L · |M|(|M|−1)), α = 0.10, L = number of looks (1 for fixed-budget designs).
A *wrong certificate* is an ACT whose true regret on the population exceeds ε. We report the wrong-certificate rate with a
Clopper–Pearson 95% interval next to every efficiency number; no efficiency comparison is made between methods whose
wrong-certificate intervals differ in whether they contain α.

## Primary efficiency metric: J50
J50 is the number of human-judged (query, document) pairs at which the cumulative ACT rate first reaches 50%. It includes
every human label the method consumed: the pilot/training queries used to fit policy parameters, choose the candidate,
select the judge or fit a residual model, and the labels used by the certificate itself. Between budgets actually run we
interpolate linearly; if the largest budget run stays below 50% we report *not reached* and never form a ratio from that
cell. J50 is preferred to "ACT rate at a fixed budget" because it is directly a cost, and to "cost among successful runs"
because it does not condition on success.

## Cost accounting by audit unit
- Query-level audits (Block A; set-F1 utility, which needs nG(q)): one audited query costs its whole pool, so J50 = T_50 ×
  mean pool size (dbpedia-entity 51.6, dl212223 69.4 judged pairs per query).
- Document-level audits (Blocks B, C; precision@cutoff, label-independent normaliser): J50 counts sampled documents plus the
  pilot's fully labelled prefixes (up to the largest cutoff). The two blocks use different utilities and are not compared
  to each other; within a block all methods share the same pilot, budget and certificate.
- Judge inference cost (LLM calls over the pool) is reported separately and never converted into human-label units.

## What the unified table shows (06_paper/TABLES_v0.1.md, F7)
- Block A, dbpedia (set-F1, ε = 0.01): no query-level method reaches 50% ACT within 9,800 judged pairs (max ACT 0.32
  humans-only, 0.44 with the Qwen3-8B judge). On dl212223 the humans-only J50 is 5,480 pairs and the Qwen3-8B judge lowers
  it to 5,280 (−4%), the reranker judge raises it to 6,170; the adoption rule stays at 5,480–5,530.
- Block B, precision, ε = 0.02: uniform document sampling does not reach 50% on dbpedia (3,210 pairs on dl212223);
  decision-weight importance sampling reaches it at 1,990 (dbpedia) / 1,270 (dl212223); adding the Qwen3-8B judge as a
  control variate gives 1,570 / 1,260; the faithful active-inference rules give 1,560–1,610 / 1,310–1,340 (same within
  noise); the calibrated-judge rule 2,020 / 1,560. All wrong-certificate rates ≤ 0.003.
- Block C (4-policy menu, ε = 0.02): static shared allocation 1,520 / 1,560, adaptive 1,460 / 1,560, oracle 1,240 / 1,540.
