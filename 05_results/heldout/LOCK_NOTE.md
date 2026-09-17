# Held-out protocol log (2026-09-17)

- 54deba0 — first lock of the reranker / pilot-20 predictions for trec-covid and webis-touche2020, committed before the audit
  was run. **Formula bug**: 98_heldout_predict.py divided by the pilot slack instead of its square when predicting the post-pilot
  labels L_post = z^2 b v / s^2. The predicted cost ratios (per arm), D3 and the decomposition in that lock are unaffected; the
  predicted L_post, the post-pilot share, the predicted total savings and the D1 decisions were wrong (all "do not recommend").
- The audit (300 draws, budgets 2-45) was run after 54deba0. The bug was found while scoring it (97_pilot_rule.py implements the
  formula correctly and disagreed with the lock). The corrected predictions (this commit) use the same pilot records, whose
  equality with the audit's pilot columns is verified (max difference 0), so nothing in them depends on the audit; but they were
  regenerated after the audit had been run and are reported as such.
- The pilot-10 variant (PREDICTIONS_rr_pilot10) was generated with the corrected formula before its audit was run.
- Scoring: PILOT_RULE_rr_pilot20_eps*.csv (97_pilot_rule.py on the *_ho_*_draws.csv records).
- 694e00b — lock of the Qwen3-8B (llm) predictions for both pools, pilot 20 and pilot 10, predictors v1-v4, committed before the
  audit; judgments generated with 66_llm_judge.py on this machine (accuracy 0.764 / 0.804, matching the manuscript's 0.76 / 0.79).
  Audit run afterwards; pilot columns of the audit records equal the locked records (max difference 0). Scoring:
  PILOT_RULE_llm_{ho,ho10}_eps*.csv.
- 2026-09-17 (review of 5210aee): `81 --predict_only` had overwritten the audit summary CSVs (*_v2_shared_full_t.csv) with empty files
  when it ran after the audits; the 84 development summaries were restored from the Track C commit (3b4ebd5) and the reranker held-out
  audits were re-run (draw records identical to the committed ones, max difference 0; 11 new pilot columns added). The bug is fixed
  (predict-only never writes the summary). Order of events for the held-out test: reranker audit scored -> v2-v4 written -> Qwen3-8B
  judgments generated -> v1-v4 locked -> Qwen3-8B audit -> v3 designated primary. 97_pilot_rule.py now also scores the executable
  single-pilot rule (PILOT_RULE_single_eps*.csv).
- LOCK v0.7 (NeuCLIRBench, English mono, 105 topics; commits 1750865 lock text -> b833065 rr/inv predictions -> ad91bfd Qwen3-8B
  predictions -> audit). Nothing in the rule was changed. Scores: PILOT_RULE_neuclir_*_eps*.csv, PILOT_RULE_single_neuclir_*.csv,
  neuclir_scores/. Pre-registered criteria, tallied exactly as written (corrected after an external re-count, 2026-09-17): (a) rule never more than
  2 points below the better of always-CV / always-weighted: met in 9 of 12 cells (misses: Qwen3-8B pilot 20 eps=0.01, 2.35% vs 5.26%;
  inverted pilot 20 eps=0.01, 0.09% vs 2.73%; reranker pilot 10 eps=0.01, 0.56% vs 2.79%); (b) inverted judge adopted by <=10% of
  pilots: met in 1 of 4 cells (6.9%, 11.6%, 11.9%, 14.7%). The lock required both in every setting: NOT MET. An earlier note here
  said 11 of 12 for (a); that count had excluded two sub-5% gains of judges the rule ignores, which the lock did not allow.
