# PROSPECTIVE LOCK v0.6 — 2026-09-10 (λ-보정 문서 단위 control variate의 확증, 사전 등록)

라벨을 pool 구성·판정 외의 용도로 사용하기 전에 커밋. 사전 접근: query 수 200, 판정 쌍 6,589, 등급 분포(1:1642, 2:2417, 3:1196, 4:1334)의 집계뿐.
primary run은 한 번, 결과는 그대로 보고. v0.5의 교훈을 반영해 **항상 정의되는 기준**을 주 기준으로 둔다.

## 1. Target
ANTIQUE 비사실형 QA test set (Hashemi et al. 2020): 200 query, query당 판정 답변 약 33개(pooled, 4단계), relevant = grade ≥ 3
(ANTIQUE 관례). 개발에 사용된 적 없음; 검색 대상이 passage가 아니라 커뮤니티 QA 답변이라 도메인 이동도 시험.
pool = query의 판정 답변을 corpus로 한 4-시스템 top-30 합집합(69/75/87/89 동일 규칙). 학습: BEIR legacy 4 pool.
판정자: Qwen3-8B UMBRELA(주), Qwen3-Reranker, 반전 reranker.

## 2. 고정된 방법과 실행
문서 단위(precision@cutoff, 메뉴 3): pilot 20 query(비용 포함), 예산 {30,60,90,120,150} 전체 판정 query 상당, ε ∈ {0.01, 0.02},
300 draws, 문서/query 4(예산 소진 규칙). arm(81): uniform, weighted, strat_pilot, active_judge(+CV), weighted_cv(계수 1),
**weighted_cvl(pilot λ)**, 판정자 3종; (82): ai_calib, ai_resid, ai_robust_0.5. 경계 스트레스: 8B·반전, 예산 {60,90}, ε 0.01.
보조: query 단위 split_t/split_ppi/split_auto(set-F1, 8B, look ≤ n/2=100, 500회), ρ(67). seed는 스크립트 고정값.

## 3. 사전 등록 기준 (ε = 0.02 기준, "최대 예산" = 150 query 상당)
- **P1 유효성**: 경계 스트레스(8B·반전 판정자)에서 모든 arm의 1종 오류 점추정 ≤ 0.10, CP 상한 ≤ 0.15.
- **P2 가중 표집의 이득(항상 정의)**: 최대 예산에서 ACT(weighted) ≥ 1.5 × ACT(uniform). 둘 다 50% 도달 시 J50(weighted) ≤ 0.6 × J50(uniform)도 보고.
- **P3 λ-CV 무해성**: 판정자 3종 모두, 예산 ≥ 60의 모든 셀에서 ACT(weighted_cvl) ≥ ACT(weighted) − 0.03. 둘 다 도달 시 J50(cvl) ≤ 1.05 × J50(weighted).
- **P4 결함 수정 검증**: 반전 판정자에서 예산 ≥ 60의 모든 셀에서 ACT(weighted_cvl) ≥ ACT(weighted_cv) − 0.01 (λ-CV가 계수-1 CV보다 나쁘지 않음),
  그리고 최대 예산에서 ACT(weighted_cv, 반전) < ACT(weighted) − 0.03이면 "계수-1 결함 재현"으로 기록.
- **P5 능동 추론 동등성(항상 정의)**: 8B에서 최대 예산 |ACT(ai_resid) − ACT(weighted_cv)| ≤ 0.05.
- **S1 (조건부, 보고)**: 8B의 ρ 평균과 함께 ACT(weighted_cvl) − ACT(weighted)를 보고; ρ ≥ 0.55이면 양의 증분을 기대.
- **S2 (보고)**: query 단위 split_ppi(8B) 대 split_t의 누적 ACT 비(look 90)와 wrong.
실패 시 원인 분석은 별도 절, primary 수치 불변.
