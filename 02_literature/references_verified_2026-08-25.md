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
