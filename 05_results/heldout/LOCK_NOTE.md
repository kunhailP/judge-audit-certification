# Held-out protocol log (2026-09-17)

- 7deb082 — first lock of the reranker / pilot-20 predictions for trec-covid and webis-touche2020, committed before the audit
  was run. **Formula bug**: 98_heldout_predict.py divided by the pilot slack instead of its square when predicting the post-pilot
  labels L_post = z^2 b v / s^2. The predicted cost ratios (per arm), D3 and the decomposition in that lock are unaffected; the
  predicted L_post, the post-pilot share, the predicted total savings and the D1 decisions were wrong (all "do not recommend").
- The audit (300 draws, budgets 2-45) was run after 7deb082. The bug was found while scoring it (97_pilot_rule.py implements the
  formula correctly and disagreed with the lock). The corrected predictions (this commit) use the same pilot records, whose
  equality with the audit's pilot columns is verified (max difference 0), so nothing in them depends on the audit; but they were
  regenerated after the audit had been run and are reported as such.
- The pilot-10 variant (PREDICTIONS_rr_pilot10) was generated with the corrected formula before its audit was run.
- Scoring: PILOT_RULE_rr_pilot20_eps*.csv (97_pilot_rule.py on the *_ho_*_draws.csv records).
