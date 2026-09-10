# Draft v0.4 — Related work and positioning (2026-09-10)

규율: 이 절의 모든 인용은 `02_literature/references_verified_*.md`에서 제목·저자·연도·venue가 확인된 것만 쓴다.
각 문단은 "그들이 푼 것 / 우리가 그대로 가져다 쓰는 것 / 우리가 추가로 보이는 것"의 세 문장 구조를 지킨다.

## 1. Prediction-powered inference와 능동 추론
Angelopoulos et al. (Science 2023)의 PPI와 PPI++ (Angelopoulos, Duchi, Zrnic 2023)는 AI 예측을 인간 라벨로 보정해 유효한
신뢰구간을 만들고, 조정 계수 λ로 예측이 나쁠 때의 손실을 막는다. Zrnic & Candès (PNAS 2024)의 active statistical inference는
예측 불확실성에 따라 라벨링 대상을 고르면서 같은 보장을 유지하고, Li et al. (NeurIPS 2025)은 불확실성 추정이 틀릴 때 능동 표집이
나빠지는 문제를 안전 분포와의 혼합으로 완화한다. Sfyraki & Wang (ICLR 2026)은 순차 능동 PPI 평균 추정에서 상수 확률 표집이
불확실성 표집과 대등함을 보였고, Kilian et al. (NeurIPS 2025)은 anytime-valid PPI 신뢰열을 준다. 우리는 이 추정·보정 기계를
그대로 쓴다(THEORY §"Relation"): 우리의 결정 가중 control-variate 추정량은 선형 범함수 Σ_d w_d y_d에 대한 능동 추정량의 특수
경우이며, 실험에서도 잔차 모델을 pilot에서 추정한 충실한 능동 추론과 동등하다(§19). 우리가 추가로 보이는 것은 (i) 정책 배포 인증이
요구하는 범함수가 정책 쌍의 절단점에서 닫힌 형태의 문서 가중치로 주어진다는 점, (ii) 그 가중치를 무시한 판정자 불확실성 표집이
정확히 Li et al.이 경고한 방식으로 실패하는 실측(분산 1.3–1.5배), (iii) 판정자를 표집이 아니라 control variate로만 쓸 때 어떤
판정자에서도 인증 유효성이 유지된다는 것(명제 B′, 반전 판정자 포함 wrong ≤ 2/500)이다.

## 2. AI 판정과 IR 평가
Oosterhuis et al. (KDD 2024)은 LLM 관련성 판정과 소량 인간 판정으로 IR 지표의 PPI·conformal 신뢰구간을 만든다. Chatzi et al.
(NeurIPS 2024)은 LLM 쌍대 비교로 모델 순위의 rank-set을, Gligorić et al. (2024)은 LLM confidence로 인간 주석 대상을 고르는
유효 추론을 준다. Balog et al. (2025, "Rankers, Judges, and Assistants")은 LLM judge가 LLM 기반 ranker를 체계적으로 선호함을
보였고, UMBRELA (Upadhyay et al. 2024)는 우리가 쓴 0–3 관련성 프롬프트의 원형이다. Durmazkeser, Okanovic et al. (2026)은 LLM
후보 다수 중 최적 선택을 위한 능동 query 선택을 다룬다. 이들은 판정자를 고정된 예측기로 두고 지표나 순위를 추정한다. 우리가
추가로 보이는 것은 배포 결정(ε-regret 인증)에서 판정자의 쓸모가 전체 정확도가 아니라 paired 차이의 ρ로 정해진다는 것(54점,
집단 내 14/14, F4), 그 ρ가 판정자 계열·collection마다 크게 갈린다는 것(Mistral이 touche에서 붕괴, §17), 정책을 정의한 바로 그
모델이 판정하면 자기 정책을 paired F1 0.14만큼 과대평가하고 결정 구간 오류가 커진다는 실측(§12.2·§17.3)이다.

## 3. 순차 검증·best-arm identification·adaptive learn-then-test
Adaptive Learn-then-Test (Zecchin & Simeone 2024)는 위험 통제 하의 적응적 검증과 조기 중단을, Jamieson & Nowak (2014)와
Kaufmann, Cappé & Garivier (JMLR 2016)는 fixed-confidence BAI의 정지 규칙을, Jourdan & Degenne (ICML 2022)은 ε-best-answer
identification에서 "가장 먼 답"을 고르는 최적 표본복잡도를 준다. 우리는 이들의 정지 논리 대신 look × 순서쌍 Bonferroni의 단순한
union bound를 쓰며(anytime 확장은 미구현, THEORY 끝), 그 대가로 후보가 자료에서 선택될 때의 winner's curse를 명시적으로 처리한다
(§9.1: M−1 보정만 하면 정확한 구간으로도 위험 2.5배). 메뉴 수준의 적응 배분은 라벨 공유를 허용한 정적 배분과 차이가 없었고 oracle도
5–18%만 앞섰다(EXPLORATION, 부정 결과). 여러 과제가 예측기를 공유하는 다중 과제 PPI (Emmenegger, Stahler, Podimata 2026)는
과제 간 재보정을 다루며, 우리의 "한 라벨이 여러 비교에 다른 가중치로 쓰이는" 구조는 그 범위 밖이다.

## 4. IR 평가 설계와 저비용 판정
Sakai (IRJ 2016)의 topic-set size design, Guiver, Mizzaro & Robertson (TOIS 2009)의 소수 topic 예측, Li & Kanoulas (CIKM 2017)의
능동 판정 표집, Sawade, Landwehr & Scheffer (NIPS 2012)의 두 모델 위험 차이의 능동 비교는 모두 평가 비용을 다룬다. 우리는 Sawade
등의 "차이를 추정하는 데 필요한 곳에 라벨을 둔다"는 원리를 문서 단위로 계승한다. 우리가 추가로 보이는 것은 그 원리를 배포 인증
문제에 맞춰 판정 단위(query vs 문서)와 utility(set-F1은 nG 때문에 query 전체 판정을 요구, precision은 절단점까지)의 비용 회계로
연결하고, 통일 지표 J50(같은 오류 통제에서 ACT 50%에 필요한 실제 판정 쌍 수)으로 모든 방법을 같은 자에 올린 것이다.

## 5. 우리가 주장하지 않는 것
새 추정 이론, 새 표집 최적성, 기존 능동 추론에 대한 우월성. 판정자 "중립성"의 일반 이론(§12.2의 서명은 동일 모델 판정자에 한정).
효율 이득의 무조건성(F6: ρ ≥ 0.55·N ≫ T·어려운 결정에서만; 사전 등록 평가에서 효율 기준 미달을 그대로 보고).
