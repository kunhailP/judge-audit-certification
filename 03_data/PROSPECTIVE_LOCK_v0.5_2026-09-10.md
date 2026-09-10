# PROSPECTIVE LOCK v0.5 — 2026-09-10 (document-level methods, pre-registration)

이 문서는 target의 라벨을 pool 구성·판정 외의 어떤 용도로도 사용하기 전에 커밋된다. 사전 접근은 judged turn 수(173, MARCO 제한 159)와
pair 수(15,337)의 집계뿐이다. primary run은 한 번, 결과는 통과·실패와 무관하게 보고.

## 1. Target
TREC CAsT 2019 evaluation topics, **MS MARCO v1 passage에 판정된 부분으로 제한**(CAR·WaPo 문서 제외; 로컬 corpus로 접근 가능한
범위). query = 수동 해소(resolved) utterance, 하나의 판정 turn = 하나의 query, ≥1개 grade≥2 MARCO passage가 있는 159 turn.
relevant = grade ≥ 2 (CAsT 관례). pool = turn의 판정 MARCO passage를 corpus로 한 4-시스템 top-30 합집합(69/75와 동일 규칙).
개발 데이터로 한 번도 사용되지 않았음(대화형 질의라는 분포 차이도 의도된 시험).
학습(분류기·τ·trunc): BEIR legacy 4 pool (변경 없음). 판정자: Qwen3-8B UMBRELA(position_ids 명시, 1500자), Qwen3-Reranker, 반전 reranker.

## 2. 고정된 방법과 실행
- 문서 단위(precision@cutoff, 메뉴 3): pilot 20 turn(비용 포함), 예산 {30,45,60,90} 전체 판정 query 상당, ε ∈ {0.01, 0.02}, 300 draws,
  문서/query 4(예산 소진 규칙 포함). arm: uniform, weighted(|w|), strat_pilot, active_judge(보정 가정)+CV, active_cv,
  weighted_cv(8B / reranker / 반전), ai_resid(잔차 pilot 추정)+CV, ai_robust_0.5.
- 경계 스트레스: 후보 = 2위, ε = 0.9×regret, 예산 {60, 90}, 300 draws, 81·82 전 arm.
- 보조(연속성): query 단위 split_t / split_ppi / split_auto(set-F1, look ≤ n/2, 500회), 판정자 3종.
- 지표: J50(§6 정의, 실제 판정 쌍), wrong(CP 95%), ρ, 문서 분산 비. seed는 스크립트 고정값.

## 3. 사전 등록 기준
- **P1 유효성**: 경계 스트레스에서 모든 문서 단위 arm의 1종 오류 점추정 ≤ 0.10, CP 상한 ≤ 0.15.
- **P2 무조건적 이득**: ε=0.02에서 J50(weighted) ≤ 0.6 × J50(uniform) (uniform 미도달·weighted 도달이면 통과).
- **P3 판정자 무해성**: 판정자 3종 모두에서 J50(weighted_cv) ≤ 1.05 × J50(weighted) (미도달이면 최대 예산 ACT로 비교).
- **P4 능동 추론 동등성**: 8B에서 |J50(ai_resid) − J50(weighted_cv)| ≤ 0.10 × J50(weighted_cv).
- **S1 (보조, 조건부)**: 8B의 ρ ≥ 0.55이면 J50(weighted_cv) ≤ 0.9 × J50(weighted)를 기대; ρ < 0.55이면 이득 없음을 기대. ρ와 함께 보고.
- **S2 (보조)**: 보정 가정 능동 표집(active_judge)의 J50 ≥ 1.2 × J50(ai_resid).
실패 시 원인 분석은 별도 절, primary 수치 불변.
