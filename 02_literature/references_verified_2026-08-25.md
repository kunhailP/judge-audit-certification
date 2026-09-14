# Verified references — 2026-08-25 (웹 검증 완료분만)

규율: 제목·저자·연도·venue·DOI가 출판사/arXiv 페이지에서 확인된 항목만 기재.
검증 안 된 인용은 논문에 쓰지 않는다 (_legacy 인용검증 규율 승계).

## Batch 1 — IR 평가 (topic-set / 저비용 평가) — 4/4 VERIFIED

1. **Sakai, T. (2016). "Topic set size design." Information Retrieval Journal
   19(3):256–283.** DOI 10.1007/s10791-015-9273-z.
   통계적 요구(검정력/CI 폭)를 만족하는 topic 수를 사전 설계. — 차별점:
   고정 예산의 사전 설계이며 순차적이지 않고, 결정별 인증서·abstain·shift 없음.
   (관련: Sakai 2018 Springer 단행본, DOI 10.1007/978-981-13-1199-4)

2. **Guiver, J., Mizzaro, S., Robertson, S. (2009). "A few good topics:
   Experiments in topic set reduction for retrieval evaluation." ACM TOIS
   27(4):21.** DOI 10.1145/1629096.1629099.
   일부 topic 부분집합이 전체 성능을 잘 예측함을 회고적으로 입증. — 차별점:
   회고적/oracle 분석; 온라인 절차·인증서·abstain 없음.

3. **Li, D., Kanoulas, E. (2017). "Active Sampling for Large-scale Information
   Retrieval Evaluation." CIKM '17.** DOI 10.1145/3132847.3133015
   (arXiv:1709.01709). 판정 수를 줄이는 능동 표집으로 불편 추정. — 차별점:
   metric 추정의 라벨 효율화이지 결정별 tolerance 대비 예산 계획이 아님.

4. **Oosterhuis, H., Jagerman, R., Qin, Z., Wang, X., Bendersky, M. (2024).
   "Reliable Confidence Intervals for Information Retrieval Evaluation Using
   Generative A.I." KDD '24.** DOI 10.1145/3637528.3671883 (arXiv:2407.02464).
   LLM 주석 + 소량 인간 주석으로 PPI/conformal risk control 기반 CI. —
   **가장 가까운 이웃**: CI는 decision-agnostic이고 고정 예산에서 계산됨;
   순차적 예산 계획·act/collect-more/abstain 인증서·shift 설정 없음.

## Batch 2 — ML 선택/순차검정 계열 — 6/6 VERIFIED

5. **Okanovic, P., Kirsch, A., Kasper, J., Hoefler, T., Krause, A., Gürel,
   N.M. (2025). "All models are wrong, some are useful: Model Selection with
   Limited Labels." AISTATS 2025 (PMLR v258).** arXiv:2410.13609.
   소량 target 라벨의 능동 선택으로 최적 분류기 식별. — 차별점: i.i.d. 라벨의
   분류기 선택만; 보정+선택 통합·인증서·pooled 판정 비용·query 상관 없음.
   ⚠️ 2024가 아니라 **AISTATS 2025**로 인용할 것.

6. **Khramtsova, E., Zhuang, S., Baktashmotlagh, M., Zuccon, G. (2024).
   "Leveraging LLMs for Unsupervised Dense Retriever Ranking." SIGIR '24.**
   DOI 10.1145/3626772.3657798 (arXiv:2402.04853). LARMOR = 방법 약칭.
   라벨 0개 LLM pseudo-judgment로 retriever 순위. — 차별점: 통계 보증 없는
   label-free 휴리스틱; 우리는 "언제 실제 라벨을 살지"를 인증서로 결정.

7. **Maekawa, S., Iso, H., Gurajada, S., Bhutani, N. (2024). "Retrieval Helps
   or Hurts? A Deeper Dive into the Efficacy of Retrieval Augmentation to
   Language Models." NAACL 2024 Long.** aclanthology 2024.naacl-long.308
   (arXiv:2402.13492). — 차별점: RAG가 언제 돕는지의 오프라인 분석; 순차적
   예산·인증서 없음.

8. **Best-arm identification (정준 2편):**
   Jamieson, K., Nowak, R. (2014). CISS 2014, DOI 10.1109/CISS.2014.6814096;
   Kaufmann, E., Cappé, O., Garivier, A. (2016). JMLR 17(1):1–42.
   — 차별점: 독립 보상의 순수 선택; 우리는 query-cluster 상관 + pooled 비용 +
   보정·선택 통합 목표.

9. **Angelopoulos, A.N., Bates, S., Fannjiang, C., Jordan, M.I., Zrnic, T.
   (2023). "Prediction-powered inference." Science 382(6671):669–674.**
   DOI 10.1126/science.adi6000. — 차별점: 고정 표본 추정 도구; 우리는 순차
   계획 안에 삽입 가능한 구성요소로 위치.

10. **Risk control / selective prediction (정준 2편):**
    Angelopoulos, Bates, Fisch, Lei, Schuster. "Conformal Risk Control."
    **ICLR 2024** (arXiv:2208.02814 — venue 주의); Geifman, El-Yaniv.
    "Selective Classification for Deep Neural Networks." NeurIPS 2017.
    — 차별점: 고정 모델의 per-prediction 위험/기권 인증; 라벨 취득 예산
    계획·결정 통합·pooled 비용 없음.

## Batch 3 — 2026-09-10 추가 (웹 검증: arXiv/venue 페이지 확인)

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
25. Zecchin, Simeone (2024). "Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection." arXiv:2409.15844 (venue 확인 필요 — 초안에서 언급 시 arXiv로 인용).
26. Qwen Team (2025). "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." 기술 보고서/블로그. Mistral AI (2024). Mistral-7B-Instruct-v0.3 (모델 카드).

## Batch 4 — 2026-09-14 추가 (외부 리뷰가 지목한 직접 경쟁 연구; arXiv/ICLR/NeurIPS 페이지에서 제목·저자 확인)

27. Mani, P., Xu, P., Lipton, Z.C., Oberst, M. (2025). "No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference." arXiv:2505.20178.
    유한표본에서 PPI++가 인간 라벨 단독보다 나빠지는 조건: 상관 |ρ|가 1/√(n−2)(가우시안; cross-fitting은 약 1/√(n/2−2))를 넘어야 이득.
    — 우리 F6의 "ρ ≥ 0.55" 경험적 문턱과 §14의 always-PPI 손실은 이 결과의 사례이므로 **새 정리로 주장할 수 없고 인용·대조해야 함.**
28. Cowen-Breen, C., Agarwal, A., Bates, S., 외 (Globerson, A. 포함) (2026). "Multiple-Prediction-Powered Inference." arXiv:2603.27414. (저자 목록은 인용 페이지에서 일부만 확인 — bib 작성 전 전체 확인 필요)
    여러 예측원의 비용·상관 구조를 이용한 전역 표집 배분.
29. Brawand, N., Leclerc, N., Ngo, A., Peterson, M., Vishwanath, S., Alhussein, L., Wellner, B. (2026). "Active Multiple-Prediction-Powered Inference." arXiv:2605.08429.
    사례별 예측원 라우팅 + 잔차 불확실성 비례 라벨 표집 + 가중 최소제곱; KKT 닫힌 형태, 점근 정규성. — 판정자 선택·표집·계수 추정의 공동 설계는 이미 존재.
30. Feng, C., Shen, M., Balashankar, A., Gerner-Beuerle, C., Rodrigues, M. (2026). "Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges." **ICLR 2026** (arXiv:2601.20913).
    소량 인간 보정 집합으로 판정자 TPR/FPR을 추정해 임계값을 보정, **유한표본 1종 오류 통제**를 보장하는 인증 검정; PPI와의 차별점을 명시. — "불완전한 판정자로 유효한 인증"을 직접 다루므로 반드시 대조.
31. Kilian, V., 외 (2025). "Anytime-valid, Bayes-assisted, Prediction-Powered Inference." NeurIPS 2025, arXiv:2505.18000 (기존 15번의 arXiv 번호 확인).
    주의: 이들의 confidence sequence는 **점근적(asymptotic CS)** — 유한표본 정확 보장이 아니다. 정확 보장이 필요하면 bounded 관측에 대한 betting CS(Waudby-Smith & Ramdas 2023, JRSS-B)를 써야 한다.
32. (추가 확인 필요) Waudby-Smith, I., Ramdas, A. (2023). "Estimating means of bounded random variables by betting." JRSS-B 86(1). — 유한표본 정확 one-sided 상한의 근거. Maurer, A., Pontil, M. (2009). "Empirical Bernstein Bounds and Sample Variance Penalization." COLT 2009.
33. (환원 검토용, 미검증) Fiez, Jain, Jamieson, Ratliff (2019). "Sequential Experimental Design for Transductive Linear Bandits." NeurIPS 2019; Soare, Lazaric, Munos (2014). "Best-Arm Identification in Linear Bandits." NeurIPS 2014; Katz-Samuels, Jain, Karnin, Jamieson (2020). "An Empirical Process Approach to the Union Bound: Practical Algorithms for Combinatorial and Linear Bandits." NeurIPS 2020.
    — 메뉴 전체 인증의 문서 배분 문제(문서 = arm, 비교 = 방향 벡터 w_j, 목적 max_j ‖w_j‖²_{A(π)⁻¹}/s_j²)는 transductive linear BAI의 G/XY-optimal design과 형식이 같다. 독창성 주장 전에 이 환원의 정도를 확인해야 한다.

## Batch 5 — 2026-09-14 오후 (방향 1·2의 novelty bar; arXiv/NeurIPS 페이지에서 확인)

34. **Ochoa Rivera, E., Tewari, A. (2024). "Optimal Thresholding Linear Bandit." arXiv:2402.09467** (stat.ML, 2024-02-11; U. Michigan). 확인됨.
    fixed-confidence ε-Thresholding Bandit Problem in stochastic linear bandits: instance-specific sample-complexity lower bound + Lazy
    Track-Threshold-and-Stop(Jedra & Proutiere 2020의 linear BAI 알고리즘 확장), asymptotically optimal(a.s.·기대값). 임계값 ρ 근처의
    arm에 표집을 집중. — **우리의 min_π max_j V_j/s_j² 메뉴 인증은 "structured linear threshold certification"으로 직접 환원될 위험.**
    Featured 기여는 (1) gold + cheap judge의 rectifier 관측 모델, (2) finite-sample certification, (3) oracle complexity C*와 adaptive
    procedure의 회수(C_alg ≤ c·C*·polylog 또는 empirical recovery)를 함께 갖춰야 함.
35. Fiez, T., Jain, L., Jamieson, K., Ratliff, L. (2019). "Sequential Experimental Design for Transductive Linear Bandits." NeurIPS 2019. (33번 확인 승격)
    measurement set ≠ target set인 transductive linear bandit의 instance-dependent lower bound와 거의 matching하는 sequential design(RAGE).
36. Soare, M., Lazaric, A., Munos, R. (2014). "Best-Arm Identification in Linear Bandits." NeurIPS 2014. (33번 확인 승격) linear BAI와 G/XY-optimal design의 관계.
37. Waudby-Smith, I., Ramdas, A. (2020). "Confidence sequences for sampling without replacement." NeurIPS 2020.
    finite population WoR에 대한 finite-sample·time-uniform CS. — "finite population exact certification" 자체는 새롭지 않음. 우리가 실제로 만난
    어려움은 unequal/adaptive inclusion probability + HT weighting + 여러 공유 linear contrast + prediction-powered residual에서의 실용적 폭.
38. Waudby-Smith, I., Ramdas, A. (2024). "Estimating means of bounded random variables by betting." JRSS-B 86(1), 1–27. (32번 확인 승격; `lib/certificates.py`의 `ucb_bet` 근거)
39. Horvitz–Thompson under unequal probability sampling: 불편이지만 분산·범위가 1/π_i로 악화(교과서적 사실). — 방향 2의 핵심 질문
    "Can exact finite-sample validity coexist with aggressive importance sampling?"의 배경.
