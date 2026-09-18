# Verified references — 2026-08-25 (web-verified entries only)

Discipline: only entries whose title, authors, year, venue, and DOI have been confirmed on the publisher/arXiv page are listed.
Unverified citations are not used in the paper (inheriting the _legacy citation-verification discipline).

## Batch 1 — IR evaluation (topic-set / low-cost evaluation) — 4/4 VERIFIED

1. **Sakai, T. (2016). "Topic set size design." Information Retrieval Journal
   19(3):256–283.** DOI 10.1007/s10791-015-9273-z.
   A-priori design of the number of topics that satisfies statistical requirements (power / CI width). — Difference:
   a fixed-budget a-priori design, not sequential; no per-decision certificate, abstain, or shift.
   (Related: Sakai 2018 Springer monograph, DOI 10.1007/978-981-13-1199-4)

2. **Guiver, J., Mizzaro, S., Robertson, S. (2009). "A few good topics:
   Experiments in topic set reduction for retrieval evaluation." ACM TOIS
   27(4):21.** DOI 10.1145/1629096.1629099.
   Retrospectively demonstrates that some subsets of topics predict overall performance well. — Difference:
   retrospective/oracle analysis; no online procedure, certificate, or abstain.

3. **Li, D., Kanoulas, E. (2017). "Active Sampling for Large-scale Information
   Retrieval Evaluation." CIKM '17.** DOI 10.1145/3132847.3133015
   (arXiv:1709.01709). Unbiased estimation via active sampling that reduces the number of judgments. — Difference:
   label-efficient metric estimation, not budget planning against a per-decision tolerance.

4. **Oosterhuis, H., Jagerman, R., Qin, Z., Wang, X., Bendersky, M. (2024).
   "Reliable Confidence Intervals for Information Retrieval Evaluation Using
   Generative A.I." KDD '24.** DOI 10.1145/3637528.3671883 (arXiv:2407.02464).
   CIs based on PPI / conformal risk control from LLM annotations plus a small number of human annotations. —
   **Closest neighbor**: the CI is decision-agnostic and computed at a fixed budget;
   no sequential budget planning, act/collect-more/abstain certificate, or shift setting.

## Batch 2 — ML selection / sequential testing family — 6/6 VERIFIED

5. **Okanovic, P., Kirsch, A., Kasper, J., Hoefler, T., Krause, A., Gürel,
   N.M. (2025). "All models are wrong, some are useful: Model Selection with
   Limited Labels." AISTATS 2025 (PMLR v258).** arXiv:2410.13609.
   Identifies the optimal classifier by active selection of a small number of target labels. — Difference: classifier selection with i.i.d.
   labels only; no integration of calibration and selection, certificate, pooled judgment cost, or query correlation.
   ⚠️ Cite as **AISTATS 2025**, not 2024.

6. **Khramtsova, E., Zhuang, S., Baktashmotlagh, M., Zuccon, G. (2024).
   "Leveraging LLMs for Unsupervised Dense Retriever Ranking." SIGIR '24.**
   DOI 10.1145/3626772.3657798 (arXiv:2402.04853). LARMOR = abbreviation of the method.
   Ranks retrievers with zero labels via LLM pseudo-judgments. — Difference: a label-free heuristic without
   statistical guarantees; we decide "when to buy real labels" with a certificate.

7. **Maekawa, S., Iso, H., Gurajada, S., Bhutani, N. (2024). "Retrieval Helps
   or Hurts? A Deeper Dive into the Efficacy of Retrieval Augmentation to
   Language Models." NAACL 2024 Long.** aclanthology 2024.naacl-long.308
   (arXiv:2402.13492). — Difference: offline analysis of when RAG helps; no sequential
   budget or certificate.

8. **Best-arm identification (2 canonical papers):**
   Jamieson, K., Nowak, R. (2014). CISS 2014, DOI 10.1109/CISS.2014.6814096;
   Kaufmann, E., Cappé, O., Garivier, A. (2016). JMLR 17(1):1–42.
   — Difference: pure selection with independent rewards; ours has query-cluster correlation + pooled cost +
   the integrated calibration-and-selection objective.

9. **Angelopoulos, A.N., Bates, S., Fannjiang, C., Jordan, M.I., Zrnic, T.
   (2023). "Prediction-powered inference." Science 382(6671):669–674.**
   DOI 10.1126/science.adi6000. — Difference: a fixed-sample estimation tool; we position it as a component
   that can be inserted into a sequential plan.

10. **Risk control / selective prediction (2 canonical papers):**
    Angelopoulos, Bates, Fisch, Lei, Schuster. "Conformal Risk Control."
    **ICLR 2024** (arXiv:2208.02814 — note the venue); Geifman, El-Yaniv.
    "Selective Classification for Deep Neural Networks." NeurIPS 2017.
    — Difference: per-prediction risk/abstention certification of a fixed model; no label-acquisition budget
    planning, decision integration, or pooled cost.

## Batch 3 — added 2026-09-10 (web-verified: arXiv/venue pages checked)

11. Angelopoulos, Duchi, Zrnic (2023). "PPI++: Efficient Prediction-Powered Inference." arXiv:2311.01453.
12. Zrnic, Candès (2024). "Active Statistical Inference." PNAS. arXiv:2403.03208.
13. Li et al. (2025). "Robust Sampling for Active Statistical Inference." NeurIPS 2025. arXiv:2511.08991.
14. Sfyraki, Wang (2026). "Revisiting Active Sequential Prediction-Powered Mean Estimation." ICLR 2026. arXiv:2604.18569.
15. Kilian et al. (2025). "Anytime-valid, Bayes-assisted, Prediction-Powered Inference." NeurIPS 2025.
16. Chatzi, Straitouri, Thejaswi, Gomez-Rodriguez (2024). "Prediction-Powered Ranking of Large Language Models." NeurIPS 2024. arXiv:2402.17826.
17. Gligorić, Zrnic, Lee, Candès, Jurafsky (2024). "Can Unconfident LLM Annotations Be Used for Confident Conclusions?" arXiv:2408.15204.
18. Durmazkeser, Okanovic, Kirsch, Hoefler, Gürel (2026). "Large Language Model Selection with Limited Annotations." arXiv:2605.24981.
19. Balog et al. (2025). "Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation." arXiv:2503.19092.
20. Upadhyay, Pradeep, Thakur, Craswell, Lin (2024). "UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor." arXiv:2406.06519.
21. Sawade, Landwehr, Scheffer (2012). "Active Comparison of Prediction Models." NIPS 2012.
22. Kossen, Farquhar, Gal, Rainforth (2021). "Active Testing: Sample-Efficient Model Evaluation." ICML 2021 (PMLR 139).
23. Jourdan, Degenne (2022). "Choosing Answers in ε-Best-Answer Identification for Linear Bandits." ICML 2022 (PMLR 162). arXiv:2206.04456.
24. Emmenegger, Stahler, Podimata (2026). "Prediction-Powered Inference Across Many Tasks for AI Evaluation & Social Science Research." arXiv:2605.29249.
25. Zecchin, Simeone (2024). "Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection." arXiv:2409.15844 (venue to be confirmed — when mentioned in the draft, cite as arXiv).
26. Qwen Team (2025). "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." Technical report/blog. Mistral AI (2024). Mistral-7B-Instruct-v0.3 (model card).

## Batch 4 — added 2026-09-14 (directly competing work pointed out by the external review; title and authors confirmed on arXiv/ICLR/NeurIPS pages)

27. Mani, P., Xu, P., Lipton, Z.C., Oberst, M. (2025). "No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference." arXiv:2505.20178.
    Conditions under which PPI++ becomes worse than human labels alone in finite samples: a gain requires the correlation |ρ| to exceed 1/√(n−2) (Gaussian; with cross-fitting about 1/√(n/2−2)).
    — Our empirical threshold "ρ ≥ 0.55" in F6 and the always-PPI loss in §14 are instances of this result, so **they cannot be claimed as a new theorem and must be cited and contrasted.**
28. Cowen-Breen, C., Agarwal, A., Bates, S., et al. (including Globerson, A.) (2026). "Multiple-Prediction-Powered Inference." arXiv:2603.27414. (Only part of the author list was confirmed on the citation page — the full list must be confirmed before writing the bib entry)
    Global sampling allocation exploiting the cost and correlation structure of several predictors.
29. Brawand, N., Leclerc, N., Ngo, A., Peterson, M., Vishwanath, S., Alhussein, L., Wellner, B. (2026). "Active Multiple-Prediction-Powered Inference." arXiv:2605.08429.
    Per-instance predictor routing + label sampling proportional to residual uncertainty + weighted least squares; closed-form KKT, asymptotic normality. — The joint design of judge selection, sampling, and coefficient estimation already exists.
30. Feng, C., Shen, M., Balashankar, A., Gerner-Beuerle, C., Rodrigues, M. (2026). "Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges." **ICLR 2026** (arXiv:2601.20913).
    A certification test that estimates the judge's TPR/FPR from a small human calibration set to calibrate the threshold and guarantees **finite-sample type-I error control**; the difference from PPI is stated explicitly. — Must be contrasted, since it directly addresses "valid certification with an imperfect judge".
31. Kilian, V., et al. (2025). "Anytime-valid, Bayes-assisted, Prediction-Powered Inference." NeurIPS 2025, arXiv:2505.18000 (arXiv number confirmed for the existing entry 15).
    Caution: their confidence sequence is **asymptotic (asymptotic CS)** — not an exact finite-sample guarantee. If an exact guarantee is needed, the betting CS for bounded observations (Waudby-Smith & Ramdas 2023, JRSS-B) must be used.
32. (further confirmation needed) Waudby-Smith, I., Ramdas, A. (2023). "Estimating means of bounded random variables by betting." JRSS-B 86(1). — Basis for the exact finite-sample one-sided upper bound. Maurer, A., Pontil, M. (2009). "Empirical Bernstein Bounds and Sample Variance Penalization." COLT 2009.
33. (for reduction review, unverified) Fiez, Jain, Jamieson, Ratliff (2019). "Sequential Experimental Design for Transductive Linear Bandits." NeurIPS 2019; Soare, Lazaric, Munos (2014). "Best-Arm Identification in Linear Bandits." NeurIPS 2014; Katz-Samuels, Jain, Karnin, Jamieson (2020). "An Empirical Process Approach to the Union Bound: Practical Algorithms for Combinatorial and Linear Bandits." NeurIPS 2020.
    — The document allocation problem for whole-menu certification (document = arm, comparison = direction vector w_j, objective max_j ‖w_j‖²_{A(π)⁻¹}/s_j²) has the same form as the G/XY-optimal design of transductive linear BAI. The extent of this reduction must be checked before claiming originality.

## Batch 5 — 2026-09-14 afternoon (novelty bar for directions 1 and 2; confirmed on arXiv/NeurIPS pages)

34. **Ochoa Rivera, E., Tewari, A. (2024). "Optimal Thresholding Linear Bandit." arXiv:2402.09467** (stat.ML, 2024-02-11; U. Michigan). Confirmed.
    fixed-confidence ε-Thresholding Bandit Problem in stochastic linear bandits: instance-specific sample-complexity lower bound + Lazy
    Track-Threshold-and-Stop (extension of the linear BAI algorithm of Jedra & Proutiere 2020), asymptotically optimal (a.s. and in expectation). Concentrates sampling on the
    arms near the threshold ρ. — **Our min_π max_j V_j/s_j² menu certification is at risk of reducing directly to "structured linear threshold certification".**
    The featured contribution must jointly provide (1) the rectifier observation model with gold + cheap judge, (2) finite-sample certification, and (3) recovery of the oracle complexity C* by the adaptive
    procedure (C_alg ≤ c·C*·polylog or empirical recovery).
35. Fiez, T., Jain, L., Jamieson, K., Ratliff, L. (2019). "Sequential Experimental Design for Transductive Linear Bandits." NeurIPS 2019. (promoted from entry 33 after confirmation)
    A sequential design (RAGE) that nearly matches the instance-dependent lower bound of the transductive linear bandit where measurement set ≠ target set.
36. Soare, M., Lazaric, A., Munos, R. (2014). "Best-Arm Identification in Linear Bandits." NeurIPS 2014. (promoted from entry 33 after confirmation) The relation between linear BAI and G/XY-optimal design.
37. Waudby-Smith, I., Ramdas, A. (2020). "Confidence sequences for sampling without replacement." NeurIPS 2020.
    Finite-sample, time-uniform CS for finite-population WoR. — "finite population exact certification" itself is not new. The difficulty we actually encountered
    is the practical width under unequal/adaptive inclusion probability + HT weighting + several shared linear contrasts + prediction-powered residuals.
38. Waudby-Smith, I., Ramdas, A. (2024). "Estimating means of bounded random variables by betting." JRSS-B 86(1), 1–27. (promoted from entry 32 after confirmation; basis for `ucb_bet` in `lib/certificates.py`)
39. Horvitz–Thompson under unequal probability sampling: unbiased, but the variance and range deteriorate as 1/π_i (textbook fact). — Background for the key question of direction 2,
    "Can exact finite-sample validity coexist with aggressive importance sampling?"


## Batch 5 — Datasets, models, and statistical tools (added 2026-09-18; title, authors, year, and venue confirmed on arXiv/DBLP/ACM DL/Springer pages)

- Thakur et al. (2021) BEIR, NeurIPS 2021 Datasets and Benchmarks. arXiv:2104.08663. `thakur2021beir`
- Bajaj et al. (2016) MS MARCO. arXiv:1611.09268 (current arXiv author order). `bajaj2016msmarco`
- Craswell et al. TREC DL overviews 2019 (arXiv:2003.07820), 2020 (arXiv:2102.07662), 2021 (arXiv:2507.08191), 2022 (arXiv:2507.10865), 2023 (arXiv:2507.08890). `craswell20{20,21}dl{19,20}`, `craswell202{1,2,3}dl2{1,2,3}`
- Voorhees et al. (2020) TREC-COVID, ACM SIGIR Forum 54(1). DOI 10.1145/3451964.3451965. `voorhees2020treccovid`
- Bondarenko et al. (2020) Touché 2020, CLEF 2020, LNCS 12260, pp. 384–395. DOI 10.1007/978-3-030-58219-7_26. `bondarenko2020touche`
- Hasibi et al. (2017) DBpedia-Entity v2, SIGIR 2017, pp. 1265–1268. DOI 10.1145/3077136.3080751. `hasibi2017dbpedia`
- Dalton, Xiong, Callan (2020) TREC CAsT 2019 overview. arXiv:2003.13624. `dalton2020cast`
- Zhang et al. (2025) Qwen3 Embedding. arXiv:2506.05176. `zhang2025qwen3embedding` (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B)
- Yang et al. (2025) Qwen3 Technical Report. arXiv:2505.09388. `yang2025qwen3`
- Jiang et al. (2023) Mistral 7B. arXiv:2310.06825. `jiang2023mistral`
- Song et al. (2020) MPNet, NeurIPS 2020. arXiv:2004.09297. `song2020mpnet`
- Reimers, Gurevych (2019) Sentence-BERT, EMNLP-IJCNLP 2019. arXiv:1908.10084. `reimers2019sbert`
- Lù (2024) BM25S. arXiv:2407.03618. `lu2024bm25s`
- Maurer, Pontil (2009) Empirical Bernstein bounds, COLT 2009. arXiv:0907.3740. `maurer2009eb`
- Classics (bibliographic details as in the standard citations, web re-verification omitted): Robertson & Zaragoza (2009) FnTIR 3(4):333–389 `robertson2009bm25`; Horvitz & Thompson (1952) JASA 47(260):663–685 `horvitz1952`; Efron (1987) JASA 82(397):171–185 `efron1987bca`; Clopper & Pearson (1934) Biometrika 26(4):404–413 `clopper1934`; Cochran (1977) Sampling Techniques 3rd ed. `cochran1977`.
