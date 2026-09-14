# TMLR 투고 준비 체크리스트 — 2026-09-14

원고: `06_paper/tmlr_submission/main.tex` (13쪽, pdflatex + bibtex, 경고 0). 스타일 파일(`tmlr.sty`, `tmlr.bst`)은
공식 저장소 [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file)의 현재 버전과 바이트 단위로 동일.
TMLR 저자 지침: https://jmlr.org/tmlr/author-guide.html

## 0. 오늘 확인·검증한 것

| 항목 | 결과 |
|---|---|
| 빌드 | pdflatex×3 + bibtex 정상, undefined reference/citation 없음, overfull hbox 0 (수정 후) |
| Table 2 (J50) | `05_results/unified/J50.csv`와 전 셀 일치 |
| Table 3 (DL 21–23 사전등록) | `05_results/primary_dlv2/primary_report.json`과 전 셀 일치 (84/198/313, 105/220/318, 26%, 1.3%, 434/494/498, wrong 7) |
| §3 ρ–gain 상관 | row-level `ppi_gain_*.csv`에서 재계산: 54 설정, corr(gain, ρ)=0.782, corr(gain, acc)=−0.015, 14 그룹 모두 양, 판정자 교체 13/13 ↑ — 본문과 일치 |
| ANTIQUE / CAsT 보고 수치 | `antique_report.json`, `cast_report.json`과 일치 (J50 776 / 1,555, ρ 0.588, S2 111/146/116, wrong 5/6/4) |
| PDF 메타데이터 | Author/Title 필드 비어 있음 (익명성 OK) |
| 코드·문서 내 개인 경로/이름 | 없음 |

## 1. 오늘 수정한 것 (main.tex, 검토 요망)

1. **Table 1·2 여백 넘침 수정** — 각각 60pt, 37pt 초과 → `\footnotesize`, 열 제목 축약, Table 1의 중복 열("Pairs") 제거(캡션에 "without / with" 명시).
2. **§7 DL 21–23 절 끝의 낡은 문장** — "document-level methods … have not yet been tested on data unseen" → 이후 두 lock에서 시험했다는 문장으로 교체 (같은 절 아래에 CAsT·ANTIQUE가 있어 자기모순이었음).
3. **§9 Limitations의 낡은 문장** — 같은 취지의 "a second pre-registered evaluation is the obvious next step" 제거, λ-보정 CV가 lock 1회만 거쳤음을 남은 한계로 기술. "Two judge families" → Qwen3·Mistral·MPNet 세 계열로 정정.
4. `main.pdf` 재컴파일본으로 교체.

## 2. 투고 전 반드시 처리 (내용)

- [ ] **데이터·모델·통계 도구 인용 누락.** 현재 refs.bib은 24개 항목뿐이고 다음이 인용되지 않음:
  BEIR (Thakur et al. 2021), MS MARCO (Nguyen et al. 2016), TREC DL 2019/2020/2021/2022/2023 overview (Craswell et al.),
  TREC-COVID (Voorhees et al. 2021), Touché 2020 (Bondarenko et al.), DBpedia-Entity v2 (Hasibi et al. 2017),
  TREC CAsT 2019 (Dalton et al. 2020), Qwen3 Embedding/Reranker (Qwen 2025), Mistral-7B (Jiang et al. 2023),
  MPNet / Sentence-Transformers (Song et al. 2020; Reimers & Gurevych 2019), BM25/bm25s, Horvitz–Thompson (1952),
  BCa 부트스트랩 (Efron 1987), Clopper–Pearson (1934). `02_literature/references_verified_*.md`의 검증 규율대로 각 항목 출처 확인 후 추가.
- [ ] **bib 항목 불완전.** `li2025robust`, `kilian2025anytime`, `balog2025rankers`의 "and others" → 전체 저자;
  `zrnic2024active` PNAS 권·호·논문번호; `upadhyay2024umbrela` 제목 표기 정리; `jourdan2022choosing` 제목의 `$\varepsilon$`가 bst에서 깨지지 않는지 확인.
- [ ] **Table 1 vs 본문 불일치.** §3.1 본문은 "five judges"를 비교했다고 하나 표에는 4개(반전 판정자 없음). 행 추가 또는 "four judges (the adversarial control is omitted)"로 정정.
- [ ] **수치 반올림.** §3 "demeaned within the 14 groups it is 0.66" — row-level 재계산 0.669 → **0.67**로 고치거나 계산 방식 확인.
- [ ] **초록 길이.** 약 450단어. TMLR에 상한은 없으나 OpenReview 초록란·가독성 위해 250–300단어 권장. (iii)의 사전등록 실패 사유 나열은 본문으로 내려도 됨.
- [ ] **Broader Impact Statement (선택).** 인증서가 "안전하다는 착시"로 오용될 위험, 판정자 편향의 사회적 영향 등 짧은 문단 고려. 필수는 아님.

## 3. 투고 전 반드시 처리 (형식)

- [ ] **그림 내부 제목 제거.** F4/F5/F6/F7 PNG 안에 matplotlib 제목("PPI gain follows decision-relevant agreement…", "F6 = ACT(PPI)/ACT(humans only)…", "F7 — Cost to certify…")이 박혀 있어 캡션과 중복. `78_plot_rho_map.py`, `85_plot_J50.py`, 67의 plot에서 `title` 제거 후 재생성. 가능하면 PDF(벡터)로 export.
- [ ] **F6 가독성.** 12패널 히트맵이 한 폭에 들어가 셀 숫자가 거의 읽히지 않음. 2행×6열 → 상하 두 그림으로 나누거나 부록으로 이동 + 본문엔 대표 패널만.
- [ ] **제목 줄바꿈.** `\\` 위치 때문에 "Judges:"만 한 줄에 남음. `\\`를 빼고 자동 줄바꿈에 맡기거나 부제를 짧게.
- [ ] **hyperref 링크 박스.** 인용·참조에 기본 색 테두리 박스가 그려짐 → `\usepackage[hidelinks]{hyperref}` 또는 `colorlinks` 권장.
- [ ] **PDF 내 폰트 임베딩·타입** 최종 확인 (`pdffonts main.pdf`).

## 4. 익명성·보조자료

- [ ] 공개 GitHub 저장소(`kunhailP/…`)는 개인 계정 → **논문·보조자료 어디에도 링크하지 말 것.** 리뷰용 코드는 익명 zip(≤100MB, `.git` 제거)으로 OpenReview 보조자료에 첨부. 현재 크기: `05_results` 50MB, `04_code` 2.2MB, `06_paper` 6.3MB → 합계 약 60MB로 한도 안.
- [ ] 보조자료 zip 생성 전 `04_code/90_make_release.py` 갱신: 현재 목록에 `LOCK_v0.6`, `89_antique_pool.py`, `91_antique_report.py`, `05_results/primary_antique`가 빠져 있음(저장소에는 수동으로 들어간 상태). README의 "`88` — pre-registered report"도 `88/91`로.
- [ ] 보조자료 zip 안의 `01_design/PAPER_2027_DESIGN_v0.1.md` 등 한국어 개발 로그를 포함할지 결정(익명성 문제는 없으나 리뷰어 접근성).
- [ ] 사전등록 lock 문서의 "커밋 시각" 증거: 공개 저장소는 1커밋뿐이라 lock이 데이터 분석 전에 커밋됐음을 저장소 이력으로 보일 수 없음. 원 저장소의 커밋 해시/시각을 lock 문서에 적거나(익명), OSF 등 타임스탬프 등록 고려.

## 5. OpenReview 제출 절차

- [ ] 전 저자 OpenReview 프로필 완비(소속·이해충돌·출판 이력).
- [ ] Action Editor 후보 제안(PPI/active inference, IR 평가 쪽).
- [ ] 제출 양식: 인간 피험자(IRB) 해당 없음, 자금·이해충돌 기재.
- [ ] 라이선스 CC BY 4.0 동의; arXiv 선공개 시 저자명 버전과 링크하지 않기.

## 6. 있으면 좋은 것

- [ ] 부록에 각 lock 문서(v0.4/v0.5/v0.6) 요약 표(대상·기준·결과) — 본문 §7이 산문이라 리뷰어가 기준 통과/미달을 한눈에 보기 어려움.
- [ ] Prop. C의 반례("a reviewer's counterexample")를 부록 그림 한 줄로.
- [ ] `04_code/84_unify_metrics.py`의 출력 표(`TABLES_v0.1.md`)가 한국어 → 보조자료용 영어 버전.

## 재현 명령 (이 환경에서 확인됨)

```bash
cd 06_paper/tmlr_submission && pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 04_code/84_unify_metrics.py      # J50.csv / TABLES_v0.1.md 재생성 (pandas, numpy 필요)
```
