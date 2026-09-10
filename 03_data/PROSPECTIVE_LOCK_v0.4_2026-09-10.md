# PROSPECTIVE LOCK v0.4 — 2026-09-10 (2027 design, 새 collection primary run 사전 등록)

이 문서는 target collection(TREC DL 2021/2022/2023 passage, MS MARCO v2)의 **qrels·판정 결과를 분석에 사용하기 전에**
작성·커밋된다. 커밋 이후 항목 변경 금지. primary run은 단 한 번이며 결과는 통과·실패와 무관하게 보고한다.

## 1. Target
- TREC Deep Learning 2021, 2022, 2023 passage ranking, MS MARCO v2 passage corpus.
- 선택 이유(사전): 개발 단계(§9–13)에서 한 번도 사용하지 않음; NIST 깊은 판정; query 합계 ≥ 200(look 190 지원);
  corpus가 이 환경에서 접근 가능(28 MB/s 측정).
- pool 구성: 각 query의 **NIST 판정 passage 집합을 corpus로** 한 4-시스템(BM25, MiniLM, MPNet, Qwen3-Embedding-0.6B)
  top-30 합집합(개발 collection과 동일한 `69_judged_pool.py` 규칙). relevant = grade ≥ 2. 세 해를 하나의 collection
  `dl212223`로 합쳐 평가하며, 연도별 결과는 부수 보고.
- 학습(분류기·τ·trunc·정책 파라미터 이전): BEIR legacy 4 pool (개발과 동일, target 무접촉).

## 2. 고정된 방법
- ε_sel = 0.01, α = 0.1, look {10,30,50,70,90,120,150,190} ∩ [≤ n/2], Bonferroni over looks × M(M−1), M = 3
  (glob_probe, trunc, ad_probe).
- 인증서: split(앞 반쪽 학습 / 뒤 반쪽 검증). 후보 = LCB 최대. 비교 방법 3종:
  (a) split_t 인간 단독, (b) split_ppi 항상 PPI++(cross-fit λ), (c) split_auto 규칙 기반 채택.
- 채택 규칙: |A_T| ≥ 30, 후보–차점 쌍 bootstrap ρ 하한(level 1−α, B=400) → gain 점수 1/(1−lcb²) > 1.1이면 PPI.
- 판정자: Qwen3-8B UMBRELA(non-thinking, position_ids 명시, passage 1500자, batch ≤ 32), 임계 0.5.
  보조 비교: Qwen3-Reranker-0.6B, 반전 reranker(적대적 대조).
- 반복 500회, seed 규칙 `5_000_000 + 1000*rep + crc32("planner|dl212223") % 100000`, 방법별 독립 rng stream.
- 판정자 진단(ρ, 구간 안/밖 오류, PPI 이득)은 기술 통계로 보고.

## 3. 사전 등록된 판정 기준
- **Primary**: 세 방법 모두 wrong-certificate rate의 Clopper–Pearson 95% 상한 ≤ 0.15 이고 점추정 ≤ α = 0.1.
- **Secondary(효율)**: T=190 누적 ACT에서 split_ppi ≥ split_t × 1.2, split_auto ≥ split_t. 미달은 실패가 아니라 결과.
- **판정자 채택 규칙의 실용성**: split_auto가 반전 판정자를 채택한 비율 ≤ 5%.
- 실패 시 원인 분석은 별도 절에서만 수행하며 primary 수치를 덮어쓰지 않는다.

## 4. 코드
- 분석 코드 commit: 이 lock 커밋의 부모(`git log`). pool 빌더는 v2 corpus용 `75_dlv2_pool.py`(69와 동일 규칙, corpus
  reader만 교체)로, lock 이후 추가되지만 pool 규칙은 위에 고정.
