# Mock double-blind review of `main.tex` (commit 4386e1a) — and a prose read-through

Written as a TMLR reviewer would, from the submitted PDF/tex only. Part A is the review; Part B lists the sentences a copy-editor
would flag, with line numbers in `06_paper/tmlr_submission/main.tex` and a suggested rewrite.

---

## Part A — Review

**Summary.** The paper studies certifying that a retrieval policy (a cutoff rule over a shared candidate pool) is within ε of the best
alternative from a limited human audit, optionally assisted by an LLM relevance judge that may be biased toward some policies. It
(1) argues, with a restatement of PPI++ and 54 empirical settings, that a judge's value is governed by the correlation ρ of paired policy
differences rather than by label accuracy; (2) uses a sample-split certificate in which the judge enters only as a mean-zero control
variate, so fixed contrasts are unbiased for any judge; (3) derives per-document decision weights and shows that importance sampling by
them removes 40–56% of the human labels relative to uniform sampling, that a λ-fitted judge control variate adds 3–20%, and that an oracle
allocation across a menu adds little; (4) proposes a cost model (saving = variance reduction × post-pilot share) that fits 193 cells, and
tests whether a 20-query pilot can predict relative costs and the judge's value, including one locked test on NeuCLIRBench. Three earlier
pre-registered evaluations replicated validity and missed their efficiency criteria.

**Claims and evidence.** The central empirical claims are supported by the released row-level results and the numbers I could check
against them (Table 2, the bootstrap intervals, the cost-model correlation, the NeuCLIRBench lock record). The paper is unusually explicit
about which analyses were pre-registered and which were post hoc, and reports failed criteria as failures. The strongest claims are
appropriately hedged; a few sentences still read stronger than the evidence (Part B, items 3, 18, 24).

**Strengths.**
1. A strong human-only baseline. The decomposition of the 40–56% saving into "skip zero-weight documents" (21–41%) and "weight the rest"
   (25–32%) is the kind of control that judge-assisted evaluation papers usually lack, and it changes how the judge's 3–20% should be read.
2. The cost model is simple and it works: post-pilot cost ratio ≈ small-budget variance ratio with correlation 0.95 over 193 cells, and
   its failure on the menu-allocation designs (0.53) is diagnosed rather than hidden. The "pilot dilution" explanation of why a good judge
   does not translate into a large total saving is the paper's most useful lesson for practitioners.
3. Validity is handled with care: unbiasedness of fixed contrasts vs. simultaneous coverage vs. finite-sample level are kept apart; the
   t-bound's asymptotic nature is stated with a concrete counterexample; exact bounds are priced rather than assumed.
4. The pre-registration record is exemplary, including the disclosure of a formula bug in the first held-out lock and the post-hoc
   status of predictors v2–v4 on TREC-COVID/Touché.

**Weaknesses / questions.**
1. *Independent evidence for the pre-audit rule is thin in the regime that matters.* The locked NeuCLIRBench test involved only weak
   judges (accuracy 0.44/0.59, λ-CV cost ratio realised 0.82–0.93) with the pilot at 70–78% of the cost; both usefulness criteria of the
   lock were missed (9/12 and 1/4). The development collections where the judge saves 19–20% (CAsT, DBpedia-Entity) were the collections
   the rule was fit on. The paper says this plainly, but the contribution sentence "how a pilot predicts it" (title) should be read as
   "predicts the relative cost of designs"; the judge's own increment is over-estimated out of sample. Please make the title/abstract
   consistent with that reading (Part B, items 1–3).
2. *Monte-Carlo resolution of the mechanism map.* With 300 repeats and ACT rates of 0.2–0.5, the standard error of an ACT ratio is
   roughly 0.08–0.12, so several individual cells quoted around the 1.2 threshold (1.17, 1.21, 1.22) are within noise. The regularities
   (a)–(c) in §3 hold as a pattern, but individual-cell statements such as "11 of 12 cells" should carry this caveat or a pooled test.
3. *Fixed pilot of 20 queries and 8–16% infeasible candidates.* Every fixed-budget result conditions on a 20-query pilot choosing the
   candidate; Appendix D shows the total cost is minimised there, but the rule's failures cluster where the pilot's slack estimate is
   optimistic (CAsT, DBpedia-Entity). A short discussion of how an auditor should size the pilot when ε is close to the policy gaps would
   help.
4. *Two utilities, two designs.* Set-F1 (query-level, sequential) and precision-at-cutoff (document-level, fixed-budget) are analysed
   with different machinery, and the paper is careful never to compare them. Still, a reader may ask which block the "40–56%" applies to;
   say "document-level, precision at cutoff" wherever the headline numbers appear (abstract, contribution 3).
5. *Reproducibility of the pools.* The candidate pools depend on retrievers, embeddings and LLM judgments generated by the authors; the
   paper says row-level results are released, but the pools themselves (document texts) inherit collection licences. State what exactly
   will be public (pool CSVs with features and judge outputs? the texts?) so that Table 2 can be regenerated by others.
6. *The finite-sample failure of the t bound* (55% of samples at n = 30 for a 0.02-probability event) is a real risk for small ε and
   small validation halves; the paper acknowledges it and reports observed rates, but a sentence on how the auditor would detect being
   in that regime (e.g. zero observed successes) would be practical.
7. *Length and voice.* 22 pages with 8 appendices is acceptable for TMLR, but the main text still contains changelog-style sentences
   ("an earlier draft claimed…", "an earlier version of this paper reported…", "we retract"). One place for this history (§8) is enough;
   elsewhere state the correct result directly (Part B, items 11, 17).

**Requested changes (minor).** Items 1, 2, 4, 5 above; the prose items in Part B marked ★.

**Assessment.** Claims are supported by the evidence with the hedges the paper already carries; the audience (IR evaluation, LLM-as-judge,
prediction-powered inference practitioners) will learn something they cannot get from the cited work, chiefly the decomposition of where
savings come from and the pilot-dilution cost model. I would recommend acceptance after minor revisions. I would not currently recommend a
Featured certification: the general lesson ("a good judge alone does not cut certification cost; the baseline, the pilot and the
threshold nature of certification decide") is valuable, but the out-of-sample evidence for the predictive rule is limited to weak-judge
regimes and one collection.

---

## Part B — Prose read-through (line numbers in main.tex; ★ = recommend fixing before submission)

1. ★ l.26 (abstract) — one sentence of ~95 words: "The model fits 193 cells … set in the lock." Split into three: the fit; the
   TREC-COVID/Touché test; the NeuCLIRBench test.
2. ★ l.26 (abstract) — "and, by a majority of pilots, whether the judge pays (8 of 8 cells)": a reader does not yet know what a "majority
   of pilots" is. Suggest: "and whether the judge pays (8 of 8 cells, by the majority of 300 pilots)". Same construction at l.37.
3. ★ l.26 and l.37 — "the measured and predictable conditions under which judges pay": after NeuCLIRBench, "predictable" is too strong.
   Suggest: "the conditions under which judges pay, measured on eight collections and partly predictable from a pilot".
4. l.30 (intro) — "or else collect more judgments or abstain": "or else … or" is awkward. Suggest: "and otherwise collect more judgments
   or abstain".
5. l.37 (contribution 3) — ~170 words, six results in one item. Cut the NeuCLIRBench clause to one short sentence and move the
   "21–41% alone" detail to §5.3.
6. l.58 — "Figure~4: corr(gain, ρ)=0.75 and corr(gain, pair accuracy)=0.03." Telegraphic. Suggest: "Figure 4 shows corr(gain, ρ)=0.75
   against corr(gain, pair accuracy)=0.03."
7. l.65 — "The map can be reproduced without the data." Misleading (the reproduction uses population gaps and σ). Suggest: "The map can
   be reproduced from a few population quantities."
8. l.65 — "so those losses are the price of estimating λ on 22-query folds": "price" appears four times in the paper in this sense;
   vary ("are due to").
9. l.87 (§4 opening) — first sentence carries a parenthetical list plus two clauses. Split: "Our earlier certificate chose the candidate on
   the data used for the bound (…). On synthetic data …; on two development pools …".
10. l.94 — "Safe certification of a candidate chosen anywhere needs simultaneous coverage of the fixed family." "chosen anywhere" →
    "chosen on any part of the data".
11. ★ l.108 (§5) — "An earlier draft claimed band sufficiency for precision. A reviewer's counterexample (…) refuted it, and a band-only
    audit built on the claim gave no gain." Changelog voice in a proposition's commentary. Suggest: "Band sufficiency does not hold for
    precision: with R_A={a}, R_B={a,b} and b irrelevant, the difference is 0.5 or 0 depending on a alone. A band-only audit built on the
    opposite assumption gave no gain in our runs (§8)."
12. l.118 — "We read Table 2 in four parts (…)". Fine, but the four bold lead-ins are full sentences with a period inside bold; TMLR
    style prefers \paragraph or plain run-in headings. Cosmetic.
13. l.120 (cost model) — "Recorded in every draw and scored against the realised costs of the same runs, the pilot-predicted variance
    ratio correlates 0.95 …": the participle attaches to the ratio, which is what is recorded, so it is grammatical, but the sentence runs
    ~90 words with five semicolon-joined results. Split into two or three sentences.
14. ★ l.122 — "The cost ratios were predicted as promised": "as promised" is colloquial. Suggest: "as the pilot had predicted".
15. ★ l.124 (NeuCLIRBench paragraph) — ~480 words in one paragraph. Split into three: (i) what was locked and what the data are; (ii)
    what was predicted correctly (relative costs, decomposition, weak judges rejected); (iii) what failed (judge's own increment,
    single-pilot criteria, post-pilot labels). Also "we fixed everything---the predictor (the corrected one), …" → "we fixed every
    element of the procedure: the corrected predictor, …".
16. l.124 — "Judge accuracy is low here (…), so the locked predictions were themselves modest": the "so" implies accuracy explains the
    prediction, which contradicts §3. Suggest: "Judge accuracy against the gain-based labels is low here (…). The locked predictions were
    modest: …". [applied in this pass]
17. l.172 (§8 item 4) — "An earlier version of this paper reported … we retract the general claim": keep, but this and l.108 and the
    CAsT "exposed a defect" (l.165) are the three places with version history; consider keeping only this one and stating the others as
    results.
18. ★ l.179 (last sentence) — "and a 20-query pilot can tell the auditor both before the audit": over-claims after §5.3's independent
    test. Suggest: "a 20-query pilot can tell the auditor the first factor, and the relative cost of designs, before the audit; the
    judge's own increment it predicts less reliably." [applied in this pass]
19. Repetition — "exactly" (7 times: "removed exactly", "exactly as PPI without λ would not", "exactly where", …); "never used in
    development" (3 times within two paragraphs); "as is"/"reported as is". Trim.
20. l.160 — "the best in this paper" → "the most favourable judge conditions in this paper".
21. l.165 — "exactly as PPI without λ would not" → "as PPI without a tuned λ would not".
22. l.147 (App. D) and l.170 — "n.r." is used before it is defined in the main text (defined in a table caption in the appendix). Define
    "not reached" once in §2 (it is) and use the words in the main text.
23. Notation — $T$ is both the number of audited queries (looks) and the number of looks in Prop. 2's $\alpha'=\alpha/(T|\mathcal M|(|\mathcal M|-1))$;
    §2 uses $L$ for the number of looks. Make Prop. 2 use $L$.
24. ★ Figure captions — F4's caption still says "Gains track ρ (0.75) and not accuracy (0.03)" (fine) but F7's caption mentions the
    calibrated-judge rule being "the only arm badly hurt", while the text says "+29 to +40%"; keep one phrasing. F5 is referenced in §4
    but now lives in Appendix E; say "(Figure 5, Appendix E)" at the reference.
25. Consistency of "λ-fitted" / "λ-corrected" / "coefficient-λ" for the same estimator; choose one ("λ-fitted") outside §7 where the
    lock's wording is quoted.


---

**Status (2026-09-17, later the same day):** all ★ items and items 4, 6, 7, 8, 9, 10, 13, 17, 19, 20, 21, 23 applied in commit after 9b3bf5a; items 12, 22, 25 left as cosmetic (bold run-in headings are TMLR-conformant; "n.r." only in appendix table captions where it is defined; the λ-corrected wording is kept only where the ANTIQUE lock is quoted).
