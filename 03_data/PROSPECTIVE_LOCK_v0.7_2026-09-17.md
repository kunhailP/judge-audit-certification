# PROSPECTIVE LOCK v0.7 — independent validation of the pre-audit rule on NeuCLIRBench (English monolingual setting)

Date: 2026-09-17. Written before any NeuCLIRBench pool was built, any judgment generated, or any audit run on it.
Purpose: test, without changing anything, whether the pilot-based cost predictor and the single-pilot rule fixed on the four
development collections (and adjusted after the TREC-COVID / Touché reranker audit) predict relative costs and the judge's
value on a collection that was never used in development.

## Target
- NeuCLIRBench (Lawrie et al. 2025), **monolingual English** setting: `eng` queries of `neuclir/bench`, `mlir` qrels, documents =
  English machine translations (`neuclir/neuclir1` `*.mt.eng` splits). 105 topics carry mlir qrels; only those are used.
- Query text = the `news.eng.tsv` string as released (title + description concatenated, as distributed).
- Relevance: binary, **relevant = gain ≥ 1** (the collection's own gains are {0, 1, 3}); nG = number of relevant judged documents.
- Pool rule identical to the fully judged pools of the paper (64/75/87/89): judged documents of the topic are the corpus;
  union of the top-30 of BM25, MiniLM, MPNet, Qwen3-Embedding-0.6B; features score_norm/rank/consensus/lexoverlap;
  relevance model = LODO logistic regression trained on the four BEIR development pools (never on the target).
- Menu: the paper's three cutoff policies (glob_probe, trunc, ad_probe); utility = precision at cutoff (document-level block).
- Judges: Qwen3-8B (UMBRELA prompt, `66_llm_judge.py`, unchanged) as the primary judge; Qwen3-Reranker-0.6B (`rr_yes`) as
  secondary; inverted reranker as adversarial control.

## Procedure (fixed)
1. Build the pool (`100_neuclir_pool.py`), generate judgments. Nothing is looked at except pool size and judge accuracy
   against the qrels (needed to confirm the judge ran; they are reported as-is).
2. `81 --predict_only` with pilot 20 (primary) and pilot 10 (secondary) on the fixed-budget grid below; `98_heldout_predict.py`
   turns the pilot records into the predictions; both are committed **before** any audit.
3. Audit (`81 --dump_draws`, 300 draws, ε ∈ {0.01, 0.02}, budgets {2,3,5,8,10,15,20,30,45,60,90} full-query equivalents,
   population estimand, t bound, α = 0.10), then `97_pilot_rule.py`.

## The rule under test (fixed; nothing here may be changed after step 2)
- Predictor: **v3** (`lib/costpredict.l_post_v3`): per-comparison two-stage variance with the known pilot part, maximum over
  comparisons; b_ref = 4 documents per query; reference arm = decision-weight sampling (`weighted`).
- Decision rule (single pilot, executable): use the λ-fitted judge control variate (`weighted_cvl`) iff the predicted total
  saving from *this* pilot exceeds **τ = 5%** of the total cost, else humans-only `weighted`. Each draw is charged its own pilot.
- v1, v2, v4 are also recorded, as secondary predictors only.

## Primary outcomes (reported whatever they are)
1. Predicted vs realised **post-pilot cost ratio** of every arm to `weighted` (correlation over arms; per-arm error).
2. Predicted vs realised **total saving** of `weighted_cvl` over `weighted` (v3, median over pilots): absolute error.
3. **Single-pilot rule**: its J50, its saving over `weighted`, compared with "always `weighted_cvl`" and "always `weighted`";
   its wrong-certificate rate; the fraction of pilots that adopt the judge; per-pilot agreement with the cell outcome.
4. Value over the trivial policies: the rule is useful only if it (a) never loses more than 2 points against the better of the
   two always-policies, and (b) does not adopt the inverted judge in more than 10% of pilots.
No saving threshold is a success criterion: a correctly predicted small saving counts as success for the predictor.

## What is not allowed
- Changing the predictor, τ, b_ref, the reference arm, the menu, the judge or the pool rule after step 2.
- Reporting a version other than v3 as the primary result. If v3 fails, that is the result.
- Calling any later re-analysis of this collection "independent".
