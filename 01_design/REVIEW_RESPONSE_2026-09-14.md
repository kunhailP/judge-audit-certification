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

## 11. 재계산 결과 — Gate A/B 판정 (2026-09-14 21:15 KST, 재생성 pool, 300 draws)

실행: `81/82/83 --pilot_cost full --sampling shared` × {dbpedia-entity, dl212223, cast19, antique} × {llm, rr, inv} × {t, bet}.
83은 `OMP_NUM_THREADS=1` + `lib/design.py`의 닫힌 형태 water-filling(정렬 O(n log n), 이분탐색 구현과 200 인스턴스에서 1e-6 일치)으로
collection당 약 25분. 출력: `05_results/menu_allocation/menu_alloc_*_v2_full_{t,bet}.csv`, `05_results/unified/gates_*.csv`.

### 11.1 Gate A — FAIL (t bound, baseline `static_sum`)

| collection | eps | J50 static | J50 oracle_exact | J50 plugin | saving_oracle | saving_plugin | recovery | obj ratio static/oracle | nonoverlap | pilot_share |
|---|---|---|---|---|---|---|---|---|---|---|
| dl212223 | 0.01 | 2930 | 2543 | 2663 | **13.2%** | 9.1% | 0.69 | 2.47 | 0.82 | 0.38 |
| dbpedia-entity | 0.01 | 3100 | 2962 | 3191 | 4.4% | −2.9% | — | 1.55 | 0.55 | 0.40 |
| dbpedia-entity | 0.02 | 2420 | 2351 | 2352 | 2.9% | 2.8% | 0.98 | 1.56 | 0.55 | 0.40 |
| antique | 0.01/0.02 | 최소 예산(30q)에서 이미 인증률 > 0.5 → J50 left-censored, 절감 측정 불가(≈0%) | | | | | | 1.94 | 0.58 | 0.27 |
| dl212223 | 0.02 | 최소 예산(20q)에서 0.56 → left-censored | | | | | | 2.44 | 0.82 | 0.38 |
| cast19 | 둘 다 | 모든 arm 인증률 ≤ 0.11 (예산 90q까지) → J50 미도달 | | | | | | 1.9–2.1 | 0.60 | 0.38 |

- 유일하게 검열되지 않은 "유리한" 사례(dl212223, nonoverlap 0.82, eps=0.01)에서도 oracle 절감 13%(pilot 제외 25%), 문턱 30%에 크게 못 미침.
  두 번째 collection은 없다. median도 20% 미만. **Gate A 실패 — 방향 1(메뉴 최적 설계)을 Featured 주 기여로 삼지 않는다.**
- 구조적 신호와 인증률 신호의 괴리: 설계 목적함수 max_j V_j/s_j² 기준으로는 oracle이 static_sum보다 1.5–2.5배 좋지만, 인증률 곡선은
  낮은 예산에서만 갈라진다(dl212223 B=20q: 0.39→0.51, B=30q: 0.55→0.71)가 예산이 커지면 모든 arm이 같은 plateau(0.66–0.85)에 붙는다.
  plateau는 인증 불가능한 메뉴(참 regret이 eps 근처 또는 초과)의 비율이 정하므로 설계로 넘을 수 없다. 즉 head-room은 "저예산 구간"에 국한된다.
- pilot이 고유 라벨의 27–40%(§8.5의 경우 (ii)): 절감을 희석하지만 그것을 빼도 30%는 안 된다.
- `static_sum_v`(분산 인지 정적) 대비로도 oracle 절감은 dbpedia 6.6%, 나머지 ≈0 — 메뉴 설계 자체의 head-room이 작다.
- 정합성: `plugin_exact`의 wrong-certificate rate는 모든 블록에서 0.000; `per_pair`(예산 균등화 후)는 모든 arm 중 최하.

### 11.2 `--bound bet` — 문서 단위 유한표본 인증은 현재 형태로는 도달 불가

모든 collection·예산·arm에서 betting 상한의 인증률 0.00–0.013. HT 가중 증분의 범위가 커서 상한이 eps 안으로 들어오지 않음 —
§8.4에서 예상한 그대로. 이 사실 자체가 방향 2의 문제 정의(aggressive importance sampling + exact validity 공존)를 실증한다.

### 11.3 판정에 따른 다음 단계

A fail이므로 결정 6의 규칙대로 **방향 2로 전환**: (a) 우선 query 단위 WoR + finite-population decomposition(μ_N = λ·mean(Ŷ) + bounded rectifier)에
betting CS를 붙여 "exact certificate가 실제로 몇 라벨에서 닫히는지"를 측정(63 계열), (b) 문서 단위로는 π 하한·절단·편향 보정 추정량의 폭을
측정. 방향 1의 결과(oracle 13%/plateau 분석)는 "design head-room이 작다"는 부정적 결과로 Limitations 또는 부록에 남긴다.
81/82 v2 결과(Table 2 재생성, 절감률 재계산)는 84 출력(`TABLES_v0.1.md`, `J50.csv`)에 들어갔고 별도 판독이 필요.

## 12. 목표 전환 (2026-09-14 21:30 KST, 승인): Featured 확장 중단 → TMLR 게재용 원고 완성

사용자 결정: Gate A 실패는 내부 투자 기준 미달이지 연구 실패가 아님. 논문은 **"AI 판정자를 이용한 검색 정책 감사에서 비용 절감은 어디서
발생하며 어떤 조건에서 사라지는가"**에 답하는 실증·방법론 분석 논문으로 정리. 방향 2는 원고의 인증 주장을 정리하는 데 필요한 만큼만
(범위 제한). wrong-certificate 0.000은 관측 오류율로만 보고. 절차: Table 2 판독 → 인증 방식의 제한된 비교 → 주장·원고 재작성 → 제출 전 검증.

### 12.1 Table 2 재판독 — 정직 accounting(shared ledger + pilot 100%), t bound, 300 draws, J50 = 고유 인간 라벨 수(pilot 포함)

예산 격자를 보강해(antique 5–150q, dl212223 5–90q, cast19 30–200q, dbpedia 20–150q) 검열을 제거함. 파일: `*_v2_shared_full_t.csv` + `*_grid2_*`.

**eps = 0.02** (원고 Table 2와 같은 설정):

| method | antique/Qwen | antique/rr | antique/inv | cast19/Qwen | cast19/rr | cast19/inv | dbpedia/Qwen | dbpedia/rr | dl212223/Qwen |
|---|---|---|---|---|---|---|---|---|---|
| uniform (humans) | 1637 | 1637 | 1637 | n.r.(>5710, .39) | n.r. | n.r. | n.r.(>5461, .41) | n.r. | 4341 |
| decision-weight IS (humans) | 877 | 877 | 877 | 3894 | 3894 | 3894 | 2688 | 2688 | 2053 |
| stratified by pilot var (humans) | 869 | 869 | 869 | 3891 | 3891 | 3891 | 2805 | 2805 | 1967 |
| AI calibrated-judge rule + CV | 842 | 913 | **1391** | 3352 | 3504 | 3746 | 2654 | 2765 | **2329** |
| AI residual-estimated rule + CV | 787 | 825 | 915 | 3338 | 3469 | 3518 | 2160 | 2474 | 1953 |
| AI robust mixture (0.3) + CV | 780 | 815 | 949 | 3306 | 3324 | 3460 | 2125 | 2391 | 1943 |
| decision-weight IS + judge CV (λ=1) | 797 | 822 | 947 | 3456 | 3555 | 3702 | 2271 | 2453 | 1976 |
| decision-weight IS + judge CV (λ fitted on pilot) | 815 | 812 | 888 | 3139 | 3709 | 3766 | 2170 | 2272 | 1883 |
| pilot labels (모든 arm 공통) | 631 | 631 | 631 | 975 | 975 | 975 | 1028 | 1028 | 1387 |
| max wrong-certificate rate (관측) | .003 | .003 | .003 | .003 | .003 | .003 | .003 | .003 | .007 |

판독(원고에 그대로 들어갈 결론):
1. **가중 표집의 절감은 정직 accounting에서도 유지된다** — uniform 대비 decision-weight IS: antique −46%, dl212223 −53%, dbpedia > −51%
   (uniform은 5,461 라벨에서도 미도달), cast19 > −32%. pilot은 모든 arm에 공통이므로 이 비교를 희석하지 않는다. "halves"는 **이 행(humans-only,
   uniform 대비)에서만** 쓸 수 있다. 분산비(accounting 무관) uniform/weighted = 2.6–4.3×, 원고의 2.9–4.4×와 일치.
2. **판정자의 추가 이득은 작아졌고 조건부다** — decision-weight IS 대비 최선의 judge arm: antique −11%(Qwen), −7%(rr); cast19 −19%(Qwen, λ-fitted)
   / −15%(rr, robust); dbpedia −19%(Qwen) / −11%(rr); dl212223 −8%(Qwen, ρ≈0.46). 원고의 "−21% / −14%"는 legacy accounting(pilot 일부 과금)의
   수치이며 정직 과금에서는 **−7 ~ −19%**로 낮춰야 한다. pilot을 뺀 표집 라벨 기준으로는 dbpedia/Qwen −25%.
3. **판정자가 손해를 주는 조건이 실측된다** — 반전(adversarial) 판정자 + λ=1 CV: antique +8%, cast19 −5%(잡음 수준), dbpedia는 rr/inv 미실행.
   calibrated-judge 규칙은 최악(antique/inv +59%, dl212223 +13%) — "판정자가 확신하며 틀리는 문서"에 라벨을 안 주는 실패 모드. λ를 pilot에서
   맞추면(cvl) 반전 판정자 손해가 +1%로 사라진다(antique 888 vs 877).
4. **강한 baseline(stratified, residual-estimated active) 대비 우월성 주장은 없다** — humans-only stratified ≈ decision-weight(±5%),
   judge arms 간 차이는 ±5% 안(300 draws에서 J50 잡음 ≈ ±3–5%). 원고는 "방법 우월성"이 아니라 "이득이 생기는 조건(ρ, 비용 구조, pilot 비중)"을
   설명하는 방향으로 다시 쓴다.
5. eps=0.01에서는 cast19·dbpedia가 5,400 라벨까지도 50% 미도달(ACT 0.29 / 0.49) — 표에는 eps=0.02를 주 결과로, eps=0.01은 도달한 셀만.

### 12.2 Gate A 부록 정리 — 분산 목적함수 개선 ≠ 인증 비용 개선

§11의 결과는 "현재 메뉴·데이터에서는 정교한 배분 최적화가 전체 인증 비용을 크게 줄이지 못한다"는 부정적 결과로 부록에 남긴다.
설계 목적함수 max_j V_j/s_j²는 oracle이 1.5–2.5× 좋지만 J50 절감은 0–13%. plateau 분해(§12.3)로 원인을 나눔.

### 12.3 Plateau 분해 (83 `_diag`: static_sum / oracle_exact / plugin_exact, 300 draws)

draw를 **feasible**(pilot 20 query가 고른 후보의 참 regret ≤ eps — 올바른 인증서가 존재할 수 있는 경우)와 infeasible로 나눔.

| collection | feasible 비율 | ACT(전체) 최대 예산 | ACT\|feasible 최대 예산 | 저예산 oracle 이득 (ACT\|feas, static→oracle) |
|---|---|---|---|---|
| antique | 0.80–0.89 | 0.74–0.76 | 0.88–0.90 | B=20q eps=.02: 0.62→0.71 |
| dl212223 | 0.83–0.89 | 0.80–0.85 | 0.96 | B=20q eps=.01: 0.43→0.60; B=30q: 0.68→0.82 |
| dbpedia-entity | 0.83–0.93 | 0.53–0.70 | 0.63–0.75 | ≈0 (±0.03) |
| cast19 | 0.83–0.94 | 0.04–0.10 | 0.04–0.12 | ≈0 |

- plateau의 대부분은 **infeasible draw**(7–20%: 20-query pilot이 잘못된 후보를 고름)다. feasible 조건부 ACT는 예산이 크면 0.88–0.97에
  닿는다. 남은 미인증(3–12%)은 slack이 작은 경우(eps/2 미만 slack이 0–9%).
- 배분 최적화의 이득은 **feasible한 실행에서, 저예산 구간에서만** 실재한다(dl212223 +0.14–0.17). 예산이 커지면 모든 arm이 feasible
  ceiling에 닿아 이득이 사라진다. 인증 비용(J50)으로 환산하면 0–13%. 원고에는 "조건부 이득(진단)"과 "전체 비용(주 결과)"을 함께 보고.
- cast19는 다른 종류의 실패: feasible 84–94%인데 ACT ≤ 0.12 — 정책 격차가 ≈0(slack ≈ eps)이라 폭 < 0.01의 인증서가 필요, 표본 부족.
  어떤 배분도 이를 못 넘는다.

### 12.4 인증 방식의 제한된 비교 (63, query 단위, population estimand, 300 repeats)

구현: `lib/certificates.py` `ucb_bet(..., N=)` = Waudby-Smith & Ramdas (2020) WoR betting CS(가설 m의 조건부 평균 m_i=(Nm−S_{i−1})/(N−i+1),
불가능 가설 즉시 기각); 합성 검증 400회 miscoverage 0.000(목표 ≤ 0.1), n→N에서 정확 평균으로 수축. `ucb_finite_population`:
μ_N = (Σ_train D + (N−ntr)·UCB_rest)/N — 라벨된 train half는 정확히 알고, val은 나머지 N−ntr의 WoR 표본. `63 --bound bet_wor`,
arm `split_fp`(humans) / `split_fp_ppi`(rectifier D−λD̂, λ는 train에서만, 범위 [−1−λ, 1+λ]), `--ntr_fixed 20`(val이 모집단 전체까지 자라도록).

| collection (N) | split_t (t, 사전등록 설계) | split_fp (t + FPC + known train) | split_fp (**bet_wor**, val ≤ N−20) | split_fp_ppi (bet_wor) |
|---|---|---|---|---|
| antique (200) | ACT 0.60, T̄ 126 | 0.82, T̄ 100 | **0.000** (T=150) | 0.000 |
| cast19 (159) | 0.76, T̄ 116 | 0.98, T̄ 86 | 0.28 (T=150 → val 130 = 82% of N) | 0.04 |
| dbpedia-entity (399) | 0.94, T̄ 137 | 0.99, T̄ 111 | 0.10 (T=350 → val 330 = 83%) | 0.007 |
| dl212223 (211) | 0.99, T̄ 88 | 1.00, T̄ 68 | 0.07 (T=200 → val 180 = 85%) | 0.04 |

관측 wrong-certificate rate: 모든 arm ≤ 0.007(t), 0(bet_wor).

결론(원고에 들어갈 문장):
- **현재 구현한 exact certificate(betting, i.i.d. 또는 WoR)는 eps=0.01, N ≤ 400에서 실용적이지 않다**: 모집단의 80% 이상을 라벨해도
  ACT ≤ 0.28. 원인은 범위 항 R·log(1/δ)/n (R=2, δ=α/(looks·M(M−1)) ≈ 1e−3 → n=100에서 ≈0.14 ≫ eps)이지 분산이 아니다(합성:
  σ=0.05, n=100에서 bet 폭 0.06 vs t 0.015; 격자 해상도 무관). 시간-균일 CS로 looks 보정을 없애도 log(1/δ) 7.1→4.8, 폭 15% 감소 — 결론 불변.
  이는 **이 설정에서 이 bound가 비실용적**이라는 진술이며, 모든 exact certificate가 비효율적이라는 주장이나 새 연구 공백 입증은 아님.
- PPI 판정자 arm은 exact bound에서 더 나쁘다(rectifier 범위 1+λ 페널티) — "판정자 이득은 점근 인증에서만 실현된다"고 명시.
- **유한모집단 t 인증서(FPC + known train half)**는 사전등록 split_t보다 일관되게 좋다(T̄ 20–26% 감소, ACT 0.82–1.00). 단 이것은
  **고정 평가 집합 N에 대한 보장**이며 미래 query 분포 평균 성능의 보장이 아님을 명시. 사후(post-hoc) 변형으로 보고.
- 원고의 인증 주장: "asymptotically calibrated (t) certificate; observed wrong-certificate rate ≤ 0.007 over all runs" — 관측 오류율이며
  보장의 증거로 쓰지 않는다. Finite-sample-exact 변형은 부록에 비용과 함께.

### 12.5 원고 재작성 계획 (다음 단계)
1. Table 2 → §12.1 정직 accounting 표(eps=0.02)로 교체; 사전등록 표는 부록 "pre-registered record"로 이동.
2. §5 "Three readings" 수치 교체(−46~−53% humans-only; 판정자 −7~−19% 조건부; 반전 판정자 +8%, λ-fit으로 제거; calibrated rule 최악).
3. 초록·서론·기여·결론에서 "halves" → humans-only 행에만, 판정자 이득은 크기 그대로.
4. 부록: 메뉴 배분 부정적 결과(§11, §12.3), exact certificate 비용(§12.4), 유한모집단 t 변형.
5. Limitations: 관측 오류율, 고정 평가 집합, pilot 27–40% 비중, N ≤ 400.

### 12.6 원고 수정 완료 (main.tex, 17쪽, 경고 0, `pdflatex` 3회 + bibtex)

- 초록 (ii)(iii)·기여 2·3·결론: "halves" → humans-only 가중 표집 행에만(−46~−53%, 4 collection); 판정자 −7~−19% 조건부; 반전 판정자 +8%→λ-fit +1%;
  calibrated rule +59%; oracle 메뉴 설계 0–13%; exact bound는 모집단 85%를 라벨해도 ≤ 28%.
- §5.3 Table 2를 정직 accounting 표(9열: ANTIQUE Q/R/I, CAsT Q/R/I, DBpedia Q/R, DL Q + pilot 행)로 교체, Figure F7_v2, "Four readings" 재작성
  (MC 표준오차 6–9% 명시, 판정자 증분은 λ-fit CV 기준으로 통일).
- §4: Prop. B 뒤 "What exactness costs" 단락 신설(유한모집단 estimand, FPC-t 20–26% 빠름, betting WoR 비실용, 범위 항 진단).
- §7 (2) 메뉴 배분 부정적 결과를 oracle design + plateau 분해로 재작성.
- §6.2/6.3의 legacy J50 인용에 "per-comparison accounting" 표기 + ANTIQUE 정직 accounting 수치 병기.
- Limitations: 관측 오류율 ≠ 보장, 고정 평가 집합, pilot 27–40%, N=159–399.
- 부록 신설: C "Pre-registered cost table (original accounting)"(구 Table 2 이동), D "Allocation across the comparisons of a menu"(Table 5),
  E "Exact and finite-population certificates"(Table 6 + 합성 진단). 재현 노트에 `86_table2_v2.py` 명시.
- 새 스크립트: `04_code/86_table2_v2.py`(TABLE2_v2_eps*.csv, TABLE2_v2.tex, F7_J50_v2.png). `82`에 `--tag`, `83`에 feasible/act_feasible/slack 열,
  `63`에 `--bound bet_wor`, `--ntr_fixed`, arm `split_fp`/`split_fp_ppi`; `lib/certificates.py`에 `ucb_bet(N=)`, `ucb_finite_population`,
  t의 FPC; `lib/design.py` water-filling 닫힌 해.

### 12.7 데이터 아카이브
- `pools_bundle` MANIFEST.csv(53 files, 68.5 MB, sha256) + README_DATA.md(모델·버전·출처·라이선스 주의) 생성, verify 0 problems. `emb/`(2.2 GB 임베딩 캐시) 제외.
- Hugging Face **private** dataset `kunhail/nonneutral-judge-audit-pools` 업로드 완료(56 files). `_texts.tsv`는 원 컬렉션 라이선스를 상속하므로 private 유지.

### 12.8 남은 일(제출 전)
1. 원고 전체 통독(숫자 ↔ CSV 대조는 §12.1·부록 표에 대해 완료; §3·§6 legacy 수치는 불변).
2. `git commit` (featured-prep) — 작업 트리 114개 변경 파일; 커밋 메시지에 §11–12 참조.
3. 남은 판단: eps=0.01 셀(cast/dbpedia n.r.)을 표에 넣을지(현재 본문에서 문장으로만 언급).

### 12.9 저장소 이력 정리 (사용자 요청, 2026-09-14 23:05 KST)
자동 삽입된 `Co-authored-by` trailer(Cursor / Claude)를 세 커밋 메시지에서 제거해 단일 저자로 정리. 트리·저자·날짜 불변.
해시 변경: e0609ea→6aa813a, 978856e→d9aeef3, 9aa624e→d678bc7 (`03_data/LOCK_ARTIFACTS_e0609ea/HASH_MAP.txt`). 태그 `lock-e0609ea-2026-09-14`는
6aa813a로 이동. 로컬 `commit-msg` hook이 이후 커밋에서도 trailer를 제거. 백업 브랜치 `backup/main-e0609ea`, `backup/featured-prep-9aa624e`는 로컬에만.

## 13. 2차 리뷰(2026-09-14 23:18 KST) 대응 — 실행 오류·estimand 정합성 수정과 재생성

### 13.1 수정
| 지적 | 수정 | 파일 |
|---|---|---|
| `63` finite-population arm 평가가 split_t/ppi/auto 종료 시 중단 → 미평가를 `abstain`으로 기록 | 5개 split arm 중 하나라도 미종료면 계속 평가 | `63_planner_v2.py` |
| `81/82/83`가 μ_R(pilot 제외 query 평균)을 추정하면서 μ_N(전체 모집단)으로 판정 | pilot의 D_j는 정확히 알고, 나머지 R에서 WoR로 뽑힌 query의 문서 단위 추정치에 2단계 분산 V̂=(1−f)s_b²/n + f·mean(v̂_within)/n (f=n/|R|, v̂=HT 문서표집 분산)을 씌워 UCB_N=(Σ_A D+|R|·UCB_R)/N. 합성 MC 2000회 miscoverage 0.039 (f=0.3), 0.052 (f=1), 목표 0.05 | `lib/certificates.py` (`ht_var_hat`, `ucb_two_stage`, `ucb_population_two_stage`), `81`, `82`, `83` (`--bound t`, 비-legacy에서만) |
| `86` grid2 파일 중복 읽기 | `sorted(set(files))`; v2 수치 불변 확인. `v3` 세대(`*_v3b{budget}_*`) 지원, eps=0.01 tex도 생성 | `86_table2_v2.py` |
| 부록 E 모집단 라벨 비율에 pilot 20 누락 | T는 pilot 포함 전체 감사 query 수로 정의, T/N 보고. 새 표 생성기 | `87_exact_table.py` |
| 후보를 training half에서 고른다는 본문 서술 (실행은 validation) | §4 split certificate 문단·Prop. B 증명 "(as implemented)" 정정 | `main.tex` |
| "cannot bias the bound" | 추정량의 불편성(모든 judge에서 성립)과 bound의 coverage(bound의 가정에 의존)를 분리해 서술 (초록·기여 2·결론) | `main.tex` |
| `67` ess_gain = 추정 SE 비율 | 명칭을 "ratio of squared estimated standard errors"로 통일. MC-MSE 열은 pool이 번들에 있는 설정만 재생성 가능(legacy/modern/judged reranker); TREC DL 2019/20 pool과 covid/touché의 Qwen3-8B 판정은 번들에 없음 | `main.tex`, `05_results/ppi_gain/` |

### 13.2 재실행 (모두 300 draws, 예산별 1 프로세스, tag `_v3b{B}`)
- `81/82`: antique {5..150}×{llm,rr,inv}, cast19 {15..200}×3, dbpedia {5..200}×{llm,rr}, dl212223 {5..90}×llm → 모든 셀 비검열(eps=0.02).
- `83`: 4 collection, 전 arm, 동일 격자. `85_gates.py --gen v3`.
- `63`: `--ntr_fixed 20 --truth population --looks 30 50 70 90 120 150 158 199 200 210 250 300 350 398` (마지막 look = N−1), bound t / bet_wor, 300 repeats → `planner_v2/{stack}_llm_pop_{bound}_ntr20_v3`.

### 13.3 결과가 바뀐 곳
**부록 E.** 게이트 수정 후 betting WoR arm은 결국 88–100% 인증하지만, ANTIQUE·CAsT(median slack≈ε)는 T/N≤0.75에서 0–2%, ≤0.9에서 3–5%, 마지막 look(T=N−1, 전수조사)에서만 발화. DBpedia(N=399)·DL(margin 큼)은 75%에서 90%/73%. 결론을 "비실용적"에서 "margin 또는 N이 크면 감당 가능, margin≈ε이면 전수조사"로 수정. FPC-t는 4 collection 모두 ACT 1.00, T̄ 20–27% 단축.

**Table 2 (v3).** estimand 정합화로 모든 비용이 내려갔고 CAsT가 가장 크게 움직임(weighted 3,894→2,127). humans-only 절감 40/56/54/56% (ANTIQUE/CAsT/DBpedia/DL). λ-fit Qwen3-8B 증분 8/19/20/3%; reranker 8–19%; 반전 judge 2–5%(잡음). pilot 비중(weighted J50 기준) 76/46/41/78%. pilot 제외 증분 32–35% (DL 16%). judge-assisted 4 arm 간 분산 3–11%. calibrated-judge rule: 반전 judge +40%(ANTIQUE), +29%(CAsT). eps=0.01도 uniform(CAsT·DBpedia) 외 전 셀 도달 → 부록 표 `tab:J50b`.

**메뉴 배분 (Gate A 재평가).** oracle 절감 2/14/0/7% (eps 0.02), 1/17/1/6% (0.01); plug-in 2/12/−13/2.5%. Gate A FAIL 유지. plateau 분해: infeasible 8–16%; ACT|feasible 최대 예산에서 ANTIQUE·CAsT·DL 0.97–1.00 (→ plateau = pilot cap), DBpedia 0.83–0.89 (잔여 분산). **CAsT는 estimand 수정 전 "인증 불가"였으나 이제 인증되며, oracle 이득이 유일하게 의미 있는(14–17%) collection.** 원고 §7(2)·부록 D를 collection별로 다시 씀.

### 13.4 남은 것
- `67` judged stack: 저장된 CSV의 ρ(covid 0.794, dbpedia 0.54)는 judged 3종 LODO(0.50/0.12)로도, `--train_dir legacy`(0.759/…)로도 정확히 재현되지 않음 → 원 실행 구성 미상. F4의 MC-MSE 검증은 정확히 재현되는 legacy/modern 24점 + 재실행 구성의 judged 9점으로 보고하고 구성 차이를 명시.
- pilot × ε sweep(후보 공유, 세 측정값)은 이 커밋 이후.

### 13.5 3차 리뷰(2026-09-15 00:45 KST) 대응
- **같은 추출 기록으로 두 수정의 기여 분리** (`81 --tag _attr_b*`, 열 `act_iid`/`act_pilot_iid`/`act`): CAsT weighted J50 3,907 (i.i.d. t) → 2,600 (pilot 정확 반영, −33%) → 2,127 (2단계 분산+FPC, 추가 −18%); ANTIQUE 881 → 851 → 836; DL은 격자 최소 예산에서 이미 ACT≥0.5(검열). 원고 §5.3 estimand 문단에 기재.
- **부록 E 표현 수정**: T=N−1은 전수조사가 아니라 "전수조사에 가까운 예산"; "정확한 평균을 안다" 삭제; ANTIQUE·CAsT에서 exact 인증은 실질적 라벨 절감이 거의 없다고 서술. margin 설명은 slack s=ε−Regret_N(m̂)·N·잔차분산·범위를 분리하는 고정 후보 실험이 필요한 가설로 격하.
- **공유 구조의 두 이득 분리**: `83`의 `per_pair` arm(비교별 독립 표집, 동일 예산·동일 ledger) vs `static_sum`(재사용, 고정 배분) → 재사용 이득 ANTIQUE 43%, DBpedia 59%, DL 35%, CAsT ≥29%(per_pair n.r.). 배분 최적화 이득(static_sum vs oracle) 0–14%. 원고 §7(2)·부록 D에 "sharing pays, optimising the shared allocation pays little"로 기재. `nonoverlap`은 binding 비교(|gap| 최소)의 band와 나머지 band 합집합의 1−Jaccard이므로 인증을 막는 비교의 공유 구조를 측정함(확인).
- **F4**: MC-MSE 검증을 재현 가능한 33 설정(BEIR 24 + judged reranker 9, 후자는 `--train_dir legacy` 구성에서 저장된 ρ와 일치)에서 수행: corr(MC gain, ρ)=0.78, 추정-SE 비율과 중앙값 차 0.03, TREC-COVID에서 추정-SE 비율이 이득을 과소평가(1.1–1.2 vs 1.5–1.7). 90% 구간 coverage 평균 0.90, 최소 0.81. 나머지 21 설정은 번들 부재로 추정-SE 비율만 보고.
- ICML 트랙 B의 첫 실험은 위 per_pair/static/oracle 세 절차 비교의 합성 버전으로 확정(공유 정도만 조절). 실제 데이터에서 이미 첫 이득이 크고 둘째가 작다는 결과가 있으므로, 합성 실험의 역할은 "공유 구조(band overlap)가 첫 이득의 크기를 결정하는가"의 검증.

## 14. Track A -- pilot sweep and fixed-candidate decomposition (2026-09-15)

Runs: `83_menu_allocation.py` with the population estimand (v3b tags), 300 draws, arms `per_pair static_sum
oracle_exact plugin_exact`, pilot fully charged, t bound. (a) `--n_train 10/40/80` (20 = existing v3b) with the
candidate chosen on the pilot and shared by all arms inside a draw; (b) `--fixed_cand best` at pilot 20 (every draw
certifies the true best policy: regret 0, so the candidate-selection effect is removed). Aggregated by
`88_pilot_sweep.py` -> `05_results/unified/PILOT_SWEEP_v3.{csv,md}`.

Findings (eps = 0.02 unless stated; eps = 0.01 is qualitatively the same):
1. Candidate quality rises with the pilot: P(regret <= eps) = 0.71-0.87 (pilot 10), 0.85-0.92 (20), 0.93-0.98 (40),
   0.99-1.00 (80). The pilot is the only place where a *wrong* candidate can enter.
2. Given a feasible candidate, the shared arms certify essentially always on ANTIQUE, CAsT and DL (>= 0.95 at every
   pilot size). DBpedia plateaus at 0.85-0.95 even when the candidate is feasible, but reaches 0.97-0.99 with the
   true best candidate -> the DBpedia residual is a *slack* problem (feasible candidates with regret close to eps),
   not a variance problem. `per_pair` (no label reuse) stays far below on every collection (CAsT <= 0.12).
3. Total cost (pilot included) is *minimised by small pilots*: pilot 10 or 20 is cheapest for every shared arm on
   every collection; pilot 40 costs 1.3-1.6x and pilot 80 costs 2.5-3.0x pilot 20. The pilot's own labels dominate
   the gain from a better candidate. Sampling labels alone (pilot excluded) do fall monotonically with pilot size.
4. Removing the candidate-selection effect (fixed best) changes J50_total by -2% (ANTIQUE, DL), -18% (DBpedia) and
   -21% (CAsT): candidate selection is a second-order cost on ANTIQUE/DL and a first-order one on CAsT/DBpedia.
5. The allocation-optimisation gain (oracle vs static_sum) stays small at every pilot size and with the fixed
   candidate (<= 14%, CAsT); a better candidate does not unmask a hidden design gain.

Pre-registered prediction for a condition not used so far (eps = 0.03, all four collections, pilot 10/20/40/80,
launched before looking at any eps = 0.03 result):
  P1  For every shared arm and collection, J50_total(pilot 40) > J50_total(pilot 20) and J50_total(pilot 80) >
      J50_total(pilot 40); pilot 10 is within +-15% of pilot 20.
  P2  ACT | feasible >= 0.95 for static_sum/oracle_exact on ANTIQUE, CAsT, DL at every pilot size.
  P3  oracle_exact saves <= 15% of J50_total relative to static_sum on every collection.
Falsification: any collection where pilot 40 beats pilot 20 in total cost (P1), or where oracle saves > 15% (P3).

Outcome of the held-out eps = 0.03 run (`88_pilot_sweep.py`, section "Held-out check"):
  P1 monotone part  : held on 12/12 (arm, collection) cells -- J50_total(40) > J50_total(20), J50_total(80) > J50_total(40).
  P1 "pilot 10 within +-15% of pilot 20": FAILED. Pilot 10 is 3-13% cheaper on CAsT and 18-30% cheaper on ANTIQUE,
      DBpedia, DL. Direction: the looser eps, the smaller the cost-minimising pilot (wrong candidates get rarer,
      the pilot's price does not). Reported as a failed prediction in Appendix D and Limitations.
  P2 held (ACT|feasible >= 0.95 on ANTIQUE, CAsT, DL for static_sum / oracle_exact at every pilot size).
  P3 held (oracle saving 0-4% of J50_total vs static_sum).
Manuscript: Appendix D gains a "Pilot size and the candidate-selection effect" paragraph + Table (tab:pilot);
Sec. 7(2)(a) and App. D(i) now attribute the DBpedia residual to small slack (fixed best -> 0.99 at eps 0.02, 0.90 at
eps 0.01) rather than to "residual variance of the hardest comparison"; Limitations records the failed sub-prediction;
App. D's closing sentence replaces "governed first by the pilot's candidate choice" with the split by collection and
the reading "diagnose the candidate's failure rate and slack before optimising where the remaining labels go".
