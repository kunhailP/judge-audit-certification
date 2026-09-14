# 외부 리뷰(e0609ea 기준)에 대한 판단, 확인 결과, 오늘 구현, 결정 요청 — 2026-09-14

## 0. 한 줄 판단

리뷰의 사실 지적은 **전부 코드에서 재현·확인됐다** (§2). 방향 제안(§6 메뉴 동시 인증의 최소 감사 비용)은 타당하지만
독창성은 아직 열려 있고, transductive linear bandit의 최적 실험설계로 얼마나 환원되는지가 먼저 판가름 나야 한다.
현재 원고는 "정직한 TMLR 게재 논문"의 재료이며, Featured로 가려면 리뷰가 말한 대로 **비용·인증 정합성 → 정확한 oracle
격차 측정 → 그 격차를 실제 절차로 달성하는 이론·알고리즘** 순서가 맞다. 나는 그 순서에 동의하되, 두 군데에서 판단을
달리한다(§1.4, §1.6).

## 1. 리뷰 항목별 판단

### 1.1 §2 인증 유효성 — 동의 (가장 중요)
- `ucb_t`(certificates.py) 반례를 실행했다: D=1 w.p. 0.02, n=30, ε=0.01 → 잘못된 인증률 **0.549** (이론 0.98³⁰=0.545).
  t 경계는 점근적이며, 원고의 Prop. B가 적는 `Pr(∃t: wrong ACT) ≤ α`는 유한표본 진술로는 성립하지 않는다.
- 추가로 발견: 문서 단위 HT/CV 추정치의 범위는 Σ|w|/π 규모(π ∝ |w|이면 n·Σ|w|/b)라서 bounded-range 경계가 사실상
  작동하지 않는다(§3.2 합성 검증에서 betting 경계 ACT = 0). **문서 단위 유한표본 인증은 미해결 문제**이며, 이것 자체가
  기여 후보다(§5).
- Kilian et al.의 anytime-valid PPI는 **점근 CS**라 정확 보장이 아니다. 정확 보장은 bounded 관측의 betting/EB 경계
  (Waudby-Smith & Ramdas; Maurer & Pontil)로만 얻는다. **유한모집단 estimand**(§1.4)를 택하면 PPI의 unlabeled 평균
  λ·mean_N(D̂)이 확정값이 되고, 나머지는 [−(1+λ), 1+λ]에 갇힌 rectifier의 평균이므로 betting 경계가 그대로 적용된다 —
  PPI 인증서도 유한표본 정확으로 만들 수 있는 깔끔한 길이다.

### 1.2 §3 비용 집계 — 동의, 영향은 리뷰 추정보다 클 수 있음
- `82`의 `nlab[m] += samp.sum()/len(others)`는 비교당 평균(확인). `81`은 비교마다 독립 표본이고 예산 B를 비교 수만큼
  중복 지출한다(확인). `84`의 J50 환산은 `82`의 이 값으로 모든 문서 단위 블록을 환산하므로 **Table 2 전체가 과소 계상**이다.
- pilot: `fit_params()`가 `nG`(pool 전체의 관련 문서 수)를 읽는데 비용은 최대 cutoff까지만 계상(확인). dbpedia는 pool
  평균 51.6이고 최대 cutoff는 십수 개 수준이라 pilot 20 query의 실제 라벨은 약 1,030 ↔ 계상 약 260. 이 차이는 모든
  문서 단위 arm에 공통 가산되므로 **절대 J50은 크게 오르고 상대 절감률은 압축**된다. 균등 vs 가중의 "절반" 주장은 재계산 전까지 보류.

### 1.3 §4 불일치 — 동의
- `63` 317–318행: 후보를 **검증 반쪽 U**에서 고른다(확인). 유효성은 M(M−1) 동시 보정이 지켜 주고 있었고(자체 테스트
  0.014 ≤ 0.020), 원고 Prop. B의 "훈련 반쪽에서 선택" 서술이 구현과 다르다.
- `63` 351행: wrong_cert를 `perm[T:]`(미감사 잔여)로 계산, `80–83`은 전체 N 평균(확인). 잔여 평균은 감사 평균과
  결정적으로 연동돼 분산이 (N/(N−T))² 배 커지는 더 어려운 표적이다(T=90, N=211이면 3배).
- `82`: 잔차 모델은 연속 p̂로, CV는 이진 rj로(확인).
- `67`: 이득 = 추정 SE 평균의 비율(확인). 합성 실행에서 MC-MSE 비(6.4 / 16.1)와 SE 비(5.2 / 4.1)가 실제로 달랐다.

### 1.4 후보 선택·estimand에 대한 내 판단 (리뷰와 다른 부분)
- 후보를 검증 반쪽에서 고르는 것은 **오류가 아니라 더 효율적인 유효한 설계**다: 모든 순서쌍에 대한 union bound는
  후보가 어느 데이터로 골라졌든 덮는다. 따라서 코드가 아니라 **Prop. B의 문장을 고치는 쪽**을 권한다(분할은 정책
  파라미터 c, τ에만 필요). 대신 "훈련 반쪽 선택 + (M−1) 보정"이 더 효율적일 가능성은 실험으로 비교한다(`--cand_on train`).
- estimand는 **유한모집단 평균(전체 N)** 으로 통일하는 것을 권한다. 배포 결정의 대상이 그 collection이고, PPI의
  unlabeled 평균도 그 위에서 정의되며, §1.1의 유한표본 PPI 인증서가 여기서 성립한다. LOCK v0.4는 잔여 평균으로 정의됐으므로
  v0.4 표는 그대로 두고 population 값을 병기한다.

### 1.5 §5 문헌 — 동의, 5편 모두 확인(02_literature Batch 4)
"No Free Lunch"는 우리 F6 문턱·always-PPI 손실을 이론으로 선점했다. AM-PPI는 판정자 라우팅·표집·가중을 공동 최적화했다.
Noisy-but-Valid는 불완전 판정자로 **유한표본** 인증을 한다. 남는 차별점은 (i) 추정 대상이 "정책 쌍의 결정 가중 선형
범함수"라는 문제 사상, (ii) 여러 비교가 한 라벨을 공유하는 **메뉴 동시 인증**, (iii) 문서 단위 유한표본 인증의 비용. (i)만으로는 부족하다.

### 1.6 §6 방향 — 동의하되 환원 검토를 먼저
- `83`의 oracle은 λ_j = 1/max(ε−gap_j, 0.002) 휴리스틱(확인). 정확한 최적해가 아니므로 5–18%는 상한이 아니다.
- 리뷰의 설계 문제 min_π max_j V_j(π)/s_j²는 **볼록**이고 라그랑주 쌍대로 정확히 풀린다(`lib/design.py`, §3.4).
  합성 검증: 결정 문서가 비겹칠 때 정적 합산 대비 68배, 겹칠 때 1.00배 — 이득이 "비겹침"으로 설명된다는 설계 문서 §2 가설과 일치.
- 그러나 이 문제는 문서 = arm, 비교 = 방향 w_j인 **transductive linear bandit의 XY-optimal design**과 형식이 같다
  (Fiez et al. 2019; Soare et al. 2014). 우리 특수성은 (a) 응답이 Bernoulli이고 잔차 분산이 판정자에 의존, (b) 라벨 비용
  이질성, (c) 정지 규칙이 ε-인증. 독창성은 "이 구조에서 실제 절차가 oracle 격차의 얼마를 회수하는가"에서 나와야 한다.
- **판단을 가르는 실험**: 실데이터에서 정확 oracle vs 정적 합산(+CV)의 J50 비. 내 go/no-go 제안은 §6.

### 1.7 §7 실험 재구성 — 동의; 단 비용이 큼
비중첩 정책(서로 다른 검색기·reranker), 새 collection, 대화 turn 의존성은 pool 재구축 + LLM 판정(GPU)이 필요하다.
우선순위는 1) 기존 pool에서 비용·인증 재계산, 2) oracle 격차 측정, 3) 그 결과가 좋을 때만 새 pool.

## 2. 코드에서 확인한 사실 (파일:행, e0609ea)

| 지적 | 위치 | 확인 |
|---|---|---|
| t 경계 sd=0 → UCB=평균 | `lib/certificates.py` `ucb_t` | 재현(0.549) |
| 비교당 평균 비용 | `82_active_inference.py:92` `nlab[m] += samp.sum()/len(others)` | 확인 |
| 비교마다 독립 표본, 예산 중복 지출 | `81_sampling_baselines.py:106–124` (`for j in others: … sample_pi(bases[key], b, n)`) | 확인 |
| pilot 비용 = 최대 cutoff, 실제는 nG 사용 | `80/81/82` `cost_full[tr].sum()` vs `63:158–162 fit_params` | 확인 |
| 후보를 검증 반쪽에서 선택 | `63_planner_v2.py:317–318` | 확인 |
| 잔여 표본 estimand | `63_planner_v2.py:351` `ev = H[perm[T:]]` | 확인 |
| 잔차 모델 p̂ vs CV rj | `82:66–74` vs `82:91` | 확인 |
| oracle = 휴리스틱 λ | `83_menu_allocation.py:92` | 확인 |
| 이득 = SE 비율 | `67_ppi_gain.py:66` | 확인 |

### 2.1 추가 발견: Table 2가 결정적으로 재생성되지 않는다
커밋된 `05_results/*` 위에서 `84_unify_metrics.py`를 다시 돌리면(저장소 밖 사본에서 확인) 커밋된 `J50.csv`와 **14개 셀이
다르다** — 예: dbpedia `weighted` 1,990 → 2,100(+5.5%), `ai_resid` 1,610 → 1,602. 원인 둘: (i) 문서 단위 J50의 라벨 환산
계수 `conv`가 `active_*.csv` 전체 파일의 평균이라 파일이 추가될 때마다 바뀜(비교당 평균 비용 문제의 파생), (ii) 같은
(collection, judge, method, eps)가 여러 baseline 파일(`_cvl`, `_posthoc` 등)에 있을 때 `drop_duplicates`가 glob 순서에
의존. README의 "모든 수치 재생성" 주장은 현재 형태로는 성립하지 않는다. Ledger 기반 행별 `docs_labelled`와 파일 명시적
지정으로 고친다(§3.2에 반영, 파일 지정은 재실행 시).

## 3. 오늘 구현·검증한 것 (실데이터 없이 가능한 범위)

### 3.1 유한표본 유효 경계 — `lib/certificates.py`
`ucb_eb`(Maurer–Pontil), `ucb_bet`(betting, predictable plug-in, 고정 n), `ucb_mean_upper(x, δ, lo, hi, method)`,
`ucb_pairs`, `ht_range`. 전역 `cert.BOUND ∈ {t, eb, bet}`. 자체 테스트에 반례와 coverage 검사 추가(통과).

**검정력 비용** (ε=0.01, δ=0.10/30, D∈[−1,1], P(U≤ε)≥0.5가 되는 검증 표본 n):

| sd | gap | t | EB | betting |
|---|---|---|---|---|
| 0.10 | 0.02 | 100 | >600 | 300 |
| 0.10 | 0.04 | 45 | >600 | 170 |
| 0.10 | 0.06 | 20 | 600 | 130 |
| 0.15 | 0.02 | 220 | >600 | 400 |
| 0.15 | 0.04 | 80 | >600 | 220 |
| 0.15 | 0.06 | 45 | >600 | 130 |
| 0.25 | 0.04 | 220 | >600 | 300 |
| 0.25 | 0.06 | 100 | >600 | 220 |

→ query 단위에서 정확 보장의 값은 **약 2.5–3배의 검증 query**. 문서 단위(HT 범위)에서는 현재 형태로 인증 불가(§3.2).

### 3.2 공통 라벨 기록부와 비용 재집계 — `lib/ledger.py`, `81`, `82`, `84`
- `Ledger`: (query, doc) 고유 라벨만 과금, 중복 비율 진단. 리뷰 예제(20문서, π=0.2 두 비교) 재현: 4 / 8 / **7.2**.
- `81`, `82` 재작성: 기본값 `--sampling shared`(비교 전체가 한 포함확률 π ∝ Σ_j base_j를 공유), `--pilot_cost full`,
  `--bound {t,eb,bet}`, `82`는 `--cv_pred {binary,prob}`로 잔차 모델과 CV 예측을 일치. `--legacy`로 구설계 재현.
  출력 파일에 `_v2_<sampling>_<pilot>_<bound>` 접미사 → lock 결과 파일은 덮어쓰지 않음. 행에 `docs_labelled`(고유 라벨), `docs_pilot`, `dup_frac` 기록.
- `84`: v2 파일은 행별 `docs_labelled`로 J50 계산, 블록 이름에 `[v2 …]` 표기.
- 합성 pool(`/tmp/synpools`, 저장소 밖)로 `81`/`82` shared·per_pair·legacy·bet 모두 end-to-end 실행 확인.
  합성 데이터에서도 pilot(full) ≈ 740 vs 표집 ≈ 400–800 → **pilot이 비용의 절반 이상**.

### 3.3 `63`, `67`
- `63`: `--cand_on {val,train}`, `--truth {remainder,population}`, `--bound` 추가(기본값은 사전등록 그대로; 출력 디렉터리에 접미사).
- `67`: `mc_var_*`, `mc_mse_*`, `mc_gain`(MSE 비), `cover90_*` 열 추가.

### 3.4 정확 oracle 배분 — `lib/design.py`
min_π max_j V_j(π)/s_j² s.t. Σcπ ≤ B 를 쌍대(θ ∈ simplex, water-filling 닫힌 해, exponentiated gradient)로 풂.
자체 테스트: J=1이면 π ∝ |a|√v로 환원; 구성 예제에서 정적 합산 대비 68×, 겹침 예제에서 1.00×. `83`에 `oracle_exact` arm으로
연결하는 것은 데이터 실행과 함께 진행.

### 3.5 문헌
리뷰가 든 5편을 확인해 `02_literature` Batch 4에 추가(저자·venue·arXiv 번호), 환원 검토용 linear-BAI 3편은 미검증으로 표기.

## 4. 데이터가 있어야 하는 것 (재실행 목록)

저장소에는 `05_results`의 집계 CSV만 있고 pool(`<pools>/<stack>/runs/candidates/*.csv`, `*_meta.csv`, `*_llm.csv`,
`*_mistral.csv`, `*_mpnet.csv`)이 없다. 아래는 pool이 오면 바로 돌릴 수 있다(GPU 불필요, 판정은 CSV에 캐시됨):

1. `81`/`82` v2 (shared·full·t) × {dbpedia, dl212223, cast, antique} × {llm, rr, inv} → Table 2 재생성, 절감률 재계산.
2. 같은 것을 `--bound bet`로 → "문서 단위 유한표본 인증의 현재 비용"(예상: 대부분 미도달) 기록.
3. `63 --cand_on train`, `--truth population`, `--bound bet` × 사전등록 3 collection → Prop. B 재작성 근거·비용.
4. `67` 재실행 → MC-MSE 이득·coverage로 F4 재작성.
5. `83` + `oracle_exact` → **go/no-go 실험**(§6).

## 5. Featured로 가는 기여 후보 (내 순위)

1. **메뉴 동시 인증의 최적 설계와 그것을 회수하는 절차** — 정확 oracle 격차가 크면(§6 기준) 주 기여. 이론은 linear-BAI
   XY-design의 Bernoulli·이질 비용·ε-인증 특수화 + 유한표본 정지 규칙.
2. **문서 단위 유한표본 인증** — HT 범위 문제를 피하는 추정량(예: 문서별 bounded 증분에 대한 betting martingale, 또는
   π 하한 + 절단 + 편향 보정)과 그 비용. 지금은 아무도 못 하는 것을 하는 결과가 되므로 1과 결합 시 강함.
3. 판정자 가치 = ρ (No Free Lunch 이후에는 보조 결과로 강등), 자기선호 실측(보조).

## 6. 결정 요청 (한 번에)

1. **데이터 전달**: pool 디렉터리(candidates CSV + 판정자 side file) 전체를 어떻게 줄지 — private repo(토큰), 압축 파일, 클라우드 경로? 대략 용량은?
   GPU는 재실행에 불필요. 새 pool(§1.7)을 만들 때만 필요 — GPU 접근 가능한가?
2. **인증 보장 노선**: (a) 유한표본 정확(betting)을 주 결과로 하고 t는 "점근 참고"로 강등 / (b) 점근 보장을 명시하고
   "asymptotic certificate"로 용어를 낮춤 / (c) 둘 다 보고. **내 추천: (c) + 유한모집단 estimand**(§1.1의 PPI 정확화 포함).
3. **estimand**: 전체 N 유한모집단 평균으로 통일하고 LOCK v0.4 표에 잔여 평균 값을 병기 — 동의하는가?
4. **후보 선택**: 코드 유지 + Prop. B 재작성(내 추천) vs 코드를 훈련 반쪽 선택으로 변경. 둘 다 실행해 효율 비교는 어차피 함.
5. **pilot 비용**: (a) pool 전체를 정직하게 과금 / (b) precision 메뉴에서 nG가 필요 없도록 pilot을 재설계(예: gate 정책의
   c를 HT 부분표본으로 추정, 또는 precision 실험에서 gate 정책 제외). (a)는 즉시 가능, (b)는 방법 변경. **추천: (a)로 먼저 재계산하고 (b)는 결과 보고 결정.**
6. **go/no-go 기준**: 정확 oracle(참 격차·참 잔차 사용)이 정적 합산+CV 대비 J50을 **≥ 30% 줄이는 collection이 2개 이상**이면
   방향 1에 투자, 아니면 방향 2(문서 단위 유한표본 인증)로 전환 — 이 문턱에 동의하는가? 다른 숫자를 원하는가?
7. **사전등록 결과 처리**: v0.4–v0.6 수치는 그대로 두고 "revised accounting" 절을 따로 두는 것 — 확인.
8. **원고 문구**: 재계산 전까지 "halves the human judgments", "certificate valid for any judge"를 초록·서론에서 내리고
   조건부 표현으로 바꾸는 작업을 지금 할지, 재계산 후 한 번에 할지.
9. **GitHub**: push 대상 브랜치(제안: `featured-prep`, main은 e0609ea 그대로 보존), 그리고 lock 문서의 커밋 타임스탬프 증거(§SUBMISSION_CHECKLIST 4)를 어떻게 남길지.

## 7. 오늘의 변경 파일

- 수정: `04_code/lib/certificates.py`, `04_code/63_planner_v2.py`, `04_code/67_ppi_gain.py`, `04_code/81_sampling_baselines.py`,
  `04_code/82_active_inference.py`, `04_code/84_unify_metrics.py`, `02_literature/references_verified_2026-08-25.md`
- 신규: `04_code/lib/ledger.py`, `04_code/lib/design.py`, 이 문서
- 기존 `05_results/*` 파일은 변경하지 않았음(합성 테스트 출력은 삭제). 이전 세션의 `main.tex` 소폭 수정과 `SUBMISSION_CHECKLIST.md`는 그대로.

---

## 8. 결정 (2026-09-14 오후, 승인됨) — §6의 9개 항목에 대한 확정

| # | 결정 | 구현 상태 |
|---|---|---|
| 1 | pool은 압축 bundle(`pools_bundle/MANIFEST.csv`, `README_DATA.md`, `<stack>/runs/candidates/…`)로 Drive/채팅 전달. PAT 전달 없음. | `04_code/92_pools_manifest.py build|verify` 작성. **bundle에 학습 pool `legacy/{nfcorpus,scifact,arguana,cqadupstack-android}`도 반드시 포함**(LODO 관련성 모델이 읽음). |
| 2 | (c) 둘 다 보고하되 **finite-sample exact(betting/EB)를 main claim**, t는 "asymptotic/diagnostic benchmark"로 강등. | 원고 문구 반영(§9). 결과는 `--bound bet`·`--bound t` 병행 실행. |
| 3 | estimand = 전체 N finite-population mean. remainder는 historical reproduction으로만. | `63 --truth population` 준비됨. |
| 4 | `cand_on=val` 유지 + Prop. B를 "simultaneous coverage of the fixed M(M−1) family" 형태로 재작성. **family는 validation을 보고 확장되지 않는다**는 조건 명시. `train` 선택은 효율 ablation. | `main.tex` Prop. B·부록 증명 재작성 완료. |
| 5 | pilot은 실제 사용한 전체 라벨 100% 과금(a). 재설계는 결과를 본 뒤. | `81/82/83 --pilot_cost full` 기본값. `85_gates.py`가 `pilot_share` 보고. |
| 6 | Gate A + Gate B 2단계(아래). | `85_gates.py`. |
| 7 | v0.4–v0.6 수치 immutable, "legacy/preregistered; superseded for cost conclusions" 명시. | Table 2 caption·Limitations에 반영. v2 결과는 `_v2_*` 파일로 분리. |
| 8 | 원고 문구 즉시 수정. | 완료(§9). |
| 9 | `main`=e0609ea 보존, `featured-prep` branch, annotated tag `lock-e0609ea-2026-09-14`, Release 첨부(SHA256SUMS 등). | 로컬 tag·branch 생성. `03_data/LOCK_ARTIFACTS_e0609ea/`(MANIFEST 244 files, SHA256SUMS, archive sha). Release 첨부물은 `/workspace/release_staging/lock-e0609ea-2026-09-14/`에 준비 — push·Release 생성은 계정 권한 필요. Zenodo/OSF snapshot 권장. |

### 8.1 Go/No-Go — 두 단계

모든 accounting 수정(shared ledger, pilot full cost) + 모든 arm에 동일 certificate를 적용한 상태에서, `85_gates.py`가 계산:

- **Gate A (구조적 head-room)**: `saving_oracle = 1 − J50(oracle_exact)/J50(static_sum)` 이
  **≥ 30%인 collection ≥ 2개** *그리고* **4 collection median ≥ 20%**.
  median 조건은 non-overlap이 극단적인 두 collection만으로 통과하는 것을 막기 위함.
- **Gate B (회수 가능성)**: oracle을 모르는 절차(`plugin_exact`: pilot·누적 추정치에서 s_j, pilot 잔차 모델에서 v̂)가
  `recovery = (J50_static − J50_plugin)/(J50_static − J50_oracle)` **≥ 60%**, 실제 절감 `saving_plugin` **≥ 15–20%가 2 collection 이상**,
  그리고 plugin의 wrong-certificate rate ≤ α 유지.
- 판정: A pass + B pass → 방향 1 강하게 진행. A fail → 방향 1 주 기여 접음. A pass + B fail → algorithmic gap이 연구문제, Featured 주장은 보류.

`85_gates.py --baseline static_sum_v`도 함께 본다: `static_sum_v`(π ∝ Σ|w_j|·√v̂)는 분산 인지 정적 규칙이라, 그것과 oracle의 차이가
**메뉴 설계 자체**의 head-room이고, `static_sum`과의 차이는 분산 인지 + 메뉴 설계의 합이다. Gate A의 공식 baseline은 결정대로 `static_sum`.

### 8.2 `83_menu_allocation.py` v2 (오늘 구현)

- arm 추가: `static_sum_v`, `oracle_exact`, `plugin_exact`. oracle은 **참 slack s_j와 모집단 잔차분산 함수 v(p)=E[(y−ĵ)²|judge-prob bin]** 을
  알되 라벨은 모른다(문서별 참 잔차를 쓰면 라벨을 아는 oracle이 되어 메뉴 설계의 이득과 분리되지 않음). plugin은 같은 설계를
  pilot(1라운드)·누적 추정치(이후)에서 s_j, pilot 잔차 모델에서 v̂(floor 0.02, 모든 결정 문서에 π>0 보장)로 푼다.
- 설계는 라운드의 모든 문서에 대해 한 번에(문서 수천, J=3, 150 EG 반복) 풂; 라운드 예산은 정적 arm과 동일(Σπ 일치).
- Ledger 과금, `--pilot_cost`, `--bound`, `design_obj`(참 v·s에서 각 arm의 realised max_j V_j/s_j² — bound 무관한 구조적 head-room), `--legacy`.
- 발견: 구 `per_pair`는 query당 3b 문서를 뽑아 실제 고유 라벨이 공유 arm의 2배 이상이었음(ledger로 드러남). v2는 b로 균등화.
- 합성 pool smoke test 통과: 비용 균등(≈1,195 vs 1,195), `design_obj` static/oracle_exact 1.7–2.2×, plugin이 그 사이. 실데이터 수치는 아님.

### 8.3 방향 1의 novelty bar (사용자 경고 반영)

min_π max_j V_j(π)/s_j² 는 transductive linear-bandit design(Soare et al. 2014; Fiez et al. 2019)과 닮았고, 더 직접적으로
**Ochoa Rivera & Tewari (2024) "Optimal Thresholding Linear Bandit"(arXiv:2402.09467)** 이 fixed-confidence ε-thresholding linear
bandit의 sample-complexity lower bound와 asymptotically optimal algorithm(Lazy Track-Threshold-and-Stop)을 이미 제시한다 — 확인함.
따라서 "menu certification을 optimal design으로 푼다"만으로는 "thresholding linear pure-exploration with a specialised design matrix"로
환원된다. Featured 기여가 되려면 세 층이 모두 필요:
1. 관측 모델이 다름 — expensive gold + cheap judge prediction, rectifier Y_i − λŶ_i 가 여러 contrast에 동시에 들어감(MultiPPI의
   multi-source 자원배분과도 구별되어야 함).
2. finite-sample certification이 핵심(asymptotic normal stopping rule 이식은 약함).
3. oracle complexity C* 정의 + adaptive procedure의 C_alg ≤ c·C*·polylog 또는 empirical recovery(Gate B).
잠정 제목: **Exact, Cost-Aware Certification of Decision Menus with Imperfect Automated Judges**. 핵심 theorem 사슬:
oracle design → adaptive realisable design → finite-sample-valid stopping.

### 8.4 방향 2의 표현 수정

"문서 단위 유한표본 인증 — 아무도 못 하는 것"은 틀림: Waudby-Smith & Ramdas (NeurIPS 2020)가 finite population WoR에 대한
time-uniform CS를 이미 제공. 실제 발견한 어려움은 **unequal/adaptive inclusion probabilities + HT weighting + multiple shared linear
contrasts + prediction-powered residuals에서 실용적 폭의 exact certificate**. 질문: *Can exact finite-sample validity coexist with
aggressive importance sampling?* Query 단위 단순 WoR + finite-population decomposition(μ_N = 알려진 λ·mean(Ŷ) + bounded rectifier 평균)은
betting/CS와 자연스럽게 결합하지만, document-level adaptive unequal sampling으로의 확장은 자동이 아니며 거기부터가 연구문제.

### 8.5 pilot 비용의 세 경우 (재계산 후 판독 기준)

(i) 정직 과금 후에도 saving 큼 → pilot 재설계 불필요. (ii) pilot이 비용의 50%+이지만 saving 잔존 → "how much pilot information is
actually required?"가 두 번째 기여. (iii) pilot이 saving을 거의 전부 소모 → 현재 절차의 practical claim 붕괴, n_G를 full pool에서
읽는 구조를 바꿔야 함. 합성 데이터에서 pilot_share ≈ 0.53이므로 (ii)/(iii) 가능성을 열어둠.

## 9. 원고 즉시 수정 (완료, `main.tex` 재컴파일 14쪽, 경고 없음)

- 초록 (ii): "remains valid for any judge" → 판정자가 bound를 편향시킬 수 없음 + "asymptotically calibrated under the pre-registered
  t-based implementation; finite-sample-valid certificates are developed separately".
- 초록 (iii)·마지막 문장: "cuts … by more than half"/"halved" → "under the original per-comparison cost accounting, substantially reduced …;
  a revised accounting that charges every unique human label once, including the full pilot, is evaluated separately and supersedes
  the original for cost conclusions".
- 기여 2 제목 "A certificate that is valid for any judge" → "A certificate whose bias does not depend on the judge"(§4 제목도 동일).
- Prop. B: simultaneous coverage 형태로 재작성(fixed family 조건, t의 유한표본 반례 D=1 w.p. 0.02·n=30·55%, EB/betting이 의도된 대체).
  부록 증명도 event E 기반으로 재작성.
- Table 2 caption·§5 첫 reading·Related work·Limitations·결론에 legacy accounting/asymptotic bound 명시.
- `refs.bib`에 waudby2024estimating, waudby2020wor, ochoa2024thresholding, fiez2019transductive, soare2014linear 추가.

## 10. 실행 환경 점검 (RunPod, 2026-09-14)

- GPU: RTX 3090 24 GB(CUDA 13.0 driver, torch 2.4.1+cu124 CUDA 사용 가능), CPU 256 threads, RAM 1 TB.
- 디스크: 컨테이너 overlay 200 GB(199 GB 여유, 재시작 시 비영속 가능) / `/workspace` 네트워크 볼륨(영속). pool bundle·HF cache·로그는 `/workspace`에.
- 설치: transformers(≥4.51), sentence-transformers, bm25s, accelerate 추가 설치 완료(새 pool/판정 생성 시 필요; 81–85 재실행은 GPU 불필요).
- 실행 스크립트: `04_code/RUN_REVISED_ACCOUNTING.sh` (`POOLS`, `TRAIN_DIR` 환경변수) — 81/82/83 v2 × 4 collection × judge × {t, bet}을 병렬 실행 후 84·85.
  예상 시간: 83 v2 300 draws 기준 collection당 약 1시간(단일 프로세스), 전체는 병렬로 수 시간.
