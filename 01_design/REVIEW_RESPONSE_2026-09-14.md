# Assessment of the external review (against e0609ea), verified findings, today's implementation, and decisions requested — 2026-09-14

## 0. One-line assessment

The review's factual points were **all reproduced and confirmed in the code** (§2). The directional proposal (§6, the minimum audit cost of simultaneous menu certification) is sound, but
its originality is still open, and the first thing to settle is how far it reduces to optimal experimental design for transductive linear bandits.
The current manuscript is the material for an "honest TMLR paper"; to go for Featured, the order the review gives is right: **cost/certificate consistency → exact measurement of the oracle
gap → theory and an algorithm that attain that gap with a real procedure**. I agree with that order, but differ in judgment at two
points (§1.4, §1.6).

## 1. Assessment per review item

### 1.1 §2 Certificate validity — agree (most important)
- Ran the `ucb_t` (certificates.py) counterexample: D=1 w.p. 0.02, n=30, ε=0.01 → wrong-certificate rate **0.549** (theory 0.98³⁰=0.545).
  The t bound is asymptotic, and the manuscript's Prop. B statement `Pr(∃t: wrong ACT) ≤ α` does not hold as a finite-sample statement.
- Additional finding: the range of document-level HT/CV estimates is of order Σ|w|/π (n·Σ|w|/b when π ∝ |w|), so bounded-range bounds effectively
  do not work (betting-bound ACT = 0 in the §3.2 synthetic validation). **Document-level finite-sample certification is an open problem**, and that is itself
  a candidate contribution (§5).
- Kilian et al.'s anytime-valid PPI is an **asymptotic CS**, not an exact guarantee. Exact guarantees come only from betting/EB bounds on bounded observations
  (Waudby-Smith & Ramdas; Maurer & Pontil). If we adopt the **finite-population estimand** (§1.4), the PPI unlabeled mean
  λ·mean_N(D̂) becomes a fixed quantity, and the remainder is the mean of a rectifier confined to [−(1+λ), 1+λ], so the betting bound applies directly —
  a clean route to making the PPI certificate finite-sample exact as well.

### 1.2 §3 Cost accounting — agree; the impact may be larger than the review estimates
- `82`'s `nlab[m] += samp.sum()/len(others)` is a per-comparison average (confirmed). `81` draws an independent sample per comparison and spends budget B
  once per comparison (confirmed). `84`'s J50 conversion uses this value from `82` to convert every document-level block, so **all of Table 2 is under-counted**.
- pilot: `fit_params()` reads `nG` (the number of relevant documents in the whole pool), but cost is charged only up to the maximum cutoff (confirmed). dbpedia has a pool
  mean of 51.6 and the maximum cutoff is in the low tens, so the actual labels for the 20 pilot queries are about 1,030 vs. about 260 charged. This difference is added uniformly to every
  document-level arm, so **absolute J50 rises substantially and relative savings are compressed**. The "half" claim for uniform vs. weighted is on hold until recomputation.

### 1.3 §4 Inconsistencies — agree
- `63` lines 317–318: candidates are chosen from the **validation half U** (confirmed). Validity was being preserved by the M(M−1) simultaneous correction (self-test
  0.014 ≤ 0.020), and the manuscript's Prop. B statement "selected on the training half" differs from the implementation.
- `63` line 351: wrong_cert is computed on `perm[T:]` (the unaudited remainder), whereas `80–83` use the full-N mean (confirmed). The remainder mean is deterministically coupled to the
  audited mean, making it a harder target with variance inflated by (N/(N−T))² (3× for T=90, N=211).
- `82`: the residual model uses continuous p̂, the CV uses binary rj (confirmed).
- `67`: gain = ratio of mean estimated SEs (confirmed). In the synthetic run the MC-MSE ratios (6.4 / 16.1) and SE ratios (5.2 / 4.1) actually differed.

### 1.4 My judgment on candidate selection and the estimand (where I differ from the review)
- Choosing candidates on the validation half is **not an error but a more efficient valid design**: the union bound over all ordered pairs
  covers the candidate regardless of which data it was selected on. I therefore recommend **fixing the wording of Prop. B** rather than the code (the split is needed only for the policy
  parameters c, τ). Whether "training-half selection + (M−1) correction" is more efficient is instead compared experimentally (`--cand_on train`).
- I recommend unifying the estimand as the **finite-population mean (full N)**. The deployment decision concerns that collection, PPI's
  unlabeled mean is defined on it, and the finite-sample PPI certificate of §1.1 holds there. LOCK v0.4 was defined with the remainder mean, so
  the v0.4 tables stay as they are and the population values are reported alongside.

### 1.5 §5 Literature — agree, all 5 papers verified (02_literature Batch 4)
"No Free Lunch" theoretically pre-empts our F6 threshold and the always-PPI loss. AM-PPI jointly optimises judge routing, sampling and weighting.
Noisy-but-Valid does **finite-sample** certification with imperfect judges. The remaining points of difference are (i) the problem mapping in which the estimation target is a "decision-weighted linear
functional of a policy pair", (ii) **simultaneous menu certification** where multiple comparisons share one label, (iii) the cost of document-level finite-sample certification. (i) alone is not enough.

### 1.6 §6 Direction — agree, but check the reduction first
- `83`'s oracle is the heuristic λ_j = 1/max(ε−gap_j, 0.002) (confirmed). It is not the exact optimum, so 5–18% is not an upper bound.
- The review's design problem min_π max_j V_j(π)/s_j² is **convex** and solved exactly via the Lagrange dual (`lib/design.py`, §3.4).
  Synthetic validation: 68× over static summation when decision documents do not overlap, 1.00× when they overlap — consistent with the design-document §2 hypothesis that the gain is explained by "non-overlap".
- However, this problem has the same form as the **XY-optimal design of a transductive linear bandit** with document = arm, comparison = direction w_j
  (Fiez et al. 2019; Soare et al. 2014). Our specifics are (a) Bernoulli responses with residual variance depending on the judge, (b) heterogeneous label
  costs, (c) an ε-certification stopping rule. Originality has to come from "how much of the oracle gap does a real procedure recover in this structure".
- **The deciding experiment**: on real data, the J50 ratio of exact oracle vs. static summation (+CV). My go/no-go proposal is in §6.

### 1.7 §7 Experiment redesign — agree; but the cost is high
Non-nested policies (different retrievers/rerankers), new collections, and conversational-turn dependence require pool rebuilding + LLM judging (GPU).
Priorities: 1) recompute cost and certification on the existing pools, 2) measure the oracle gap, 3) build new pools only if those results are good.

## 2. Facts confirmed in the code (file:line, e0609ea)

| Point | Location | Confirmed |
|---|---|---|
| t bound sd=0 → UCB=mean | `lib/certificates.py` `ucb_t` | Reproduced (0.549) |
| Per-comparison averaged cost | `82_active_inference.py:92` `nlab[m] += samp.sum()/len(others)` | Confirmed |
| Independent sample per comparison, budget spent repeatedly | `81_sampling_baselines.py:106–124` (`for j in others: … sample_pi(bases[key], b, n)`) | Confirmed |
| pilot cost = max cutoff, actual use is nG | `80/81/82` `cost_full[tr].sum()` vs `63:158–162 fit_params` | Confirmed |
| Candidates selected on validation half | `63_planner_v2.py:317–318` | Confirmed |
| Remainder-sample estimand | `63_planner_v2.py:351` `ev = H[perm[T:]]` | Confirmed |
| Residual model p̂ vs CV rj | `82:66–74` vs `82:91` | Confirmed |
| oracle = heuristic λ | `83_menu_allocation.py:92` | Confirmed |
| Gain = SE ratio | `67_ppi_gain.py:66` | Confirmed |

### 2.1 Additional finding: Table 2 is not regenerated deterministically
Rerunning `84_unify_metrics.py` on top of the committed `05_results/*` (verified on a copy outside the repository) gives **14 cells that differ** from the committed `J50.csv`
— e.g. dbpedia `weighted` 1,990 → 2,100 (+5.5%), `ai_resid` 1,610 → 1,602. Two causes: (i) the label conversion
factor `conv` for document-level J50 is the mean over all `active_*.csv` files, so it changes whenever a file is added (a derivative of the per-comparison averaged cost problem), (ii) when the same
(collection, judge, method, eps) exists in several baseline files (`_cvl`, `_posthoc`, etc.), `drop_duplicates` depends on the glob order.
The README's "all numbers regenerate" claim does not hold in its current form. Fixed via ledger-based per-row `docs_labelled` and explicit file
specification (reflected in §3.2; file specification at re-run time).

## 3. Implemented and validated today (what is possible without real data)

### 3.1 Finite-sample valid bounds — `lib/certificates.py`
`ucb_eb` (Maurer–Pontil), `ucb_bet` (betting, predictable plug-in, fixed n), `ucb_mean_upper(x, δ, lo, hi, method)`,
`ucb_pairs`, `ht_range`. Global `cert.BOUND ∈ {t, eb, bet}`. Counterexample and coverage checks added to the self-test (pass).

**Power cost** (ε=0.01, δ=0.10/30, D∈[−1,1], validation sample size n at which P(U≤ε)≥0.5):

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

→ At the query level, the price of an exact guarantee is **about 2.5–3× the validation queries**. At the document level (HT range), certification is impossible in the current form (§3.2).

### 3.2 Shared label ledger and cost re-accounting — `lib/ledger.py`, `81`, `82`, `84`
- `Ledger`: charges only unique (query, doc) labels, reports the duplicate fraction as a diagnostic. The review's example (20 documents, two comparisons with π=0.2) reproduced: 4 / 8 / **7.2**.
- `81`, `82` rewritten: default `--sampling shared` (all comparisons share one inclusion probability π ∝ Σ_j base_j), `--pilot_cost full`,
  `--bound {t,eb,bet}`; `82` aligns the residual model and the CV prediction via `--cv_pred {binary,prob}`. `--legacy` reproduces the old design.
  Output files carry the suffix `_v2_<sampling>_<pilot>_<bound>` → lock result files are not overwritten. Rows record `docs_labelled` (unique labels), `docs_pilot`, `dup_frac`.
- `84`: for v2 files, J50 is computed from per-row `docs_labelled`, and block names are marked `[v2 …]`.
- Verified end-to-end runs of `81`/`82` in shared, per_pair, legacy and bet modes on synthetic pools (`/tmp/synpools`, outside the repository).
  Even on synthetic data, pilot(full) ≈ 740 vs. sampling ≈ 400–800 → **the pilot is more than half of the cost**.

### 3.3 `63`, `67`
- `63`: added `--cand_on {val,train}`, `--truth {remainder,population}`, `--bound` (defaults unchanged from pre-registration; suffix on the output directory).
- `67`: added columns `mc_var_*`, `mc_mse_*`, `mc_gain` (MSE ratio), `cover90_*`.

### 3.4 Exact oracle allocation — `lib/design.py`
Solves min_π max_j V_j(π)/s_j² s.t. Σcπ ≤ B via the dual (θ ∈ simplex, closed-form water-filling, exponentiated gradient).
Self-test: for J=1 it reduces to π ∝ |a|√v; 68× over static summation on the constructed example, 1.00× on the overlapping example. Wiring it into `83` as the `oracle_exact` arm
proceeds together with the data runs.

### 3.5 Literature
Verified the 5 papers the review cited and added them to `02_literature` Batch 4 (authors, venue, arXiv numbers); the 3 linear-BAI papers for the reduction check are marked unverified.

## 4. What requires data (re-run list)

The repository has only the aggregate CSVs in `05_results`; the pools (`<pools>/<stack>/runs/candidates/*.csv`, `*_meta.csv`, `*_llm.csv`,
`*_mistral.csv`, `*_mpnet.csv`) are absent. The following can run as soon as the pools arrive (no GPU needed; judgments are cached in the CSVs):

1. `81`/`82` v2 (shared, full, t) × {dbpedia, dl212223, cast, antique} × {llm, rr, inv} → regenerate Table 2, recompute savings.
2. The same with `--bound bet` → record "the current cost of document-level finite-sample certification" (expected: mostly not reached).
3. `63 --cand_on train`, `--truth population`, `--bound bet` × the 3 pre-registered collections → basis and cost for rewriting Prop. B.
4. Re-run `67` → rewrite F4 with MC-MSE gain and coverage.
5. `83` + `oracle_exact` → **go/no-go experiment** (§6).

## 5. Candidate contributions for Featured (my ranking)

1. **Optimal design for simultaneous menu certification and a procedure that recovers it** — the main contribution if the exact oracle gap is large (by the §6 criterion). The theory is the
   Bernoulli / heterogeneous-cost / ε-certification specialisation of linear-BAI XY-design + a finite-sample stopping rule.
2. **Document-level finite-sample certification** — an estimator that avoids the HT-range problem (e.g. a betting martingale over bounded per-document increments, or
   a lower bound on π + truncation + bias correction) and its cost. This would be a result nobody can currently achieve, so it is strong when combined with 1.
3. Judge value = ρ (demoted to an auxiliary result after No Free Lunch), empirical measurement of self-preference (auxiliary).

## 6. Decisions requested (all at once)

1. **Data delivery**: how to deliver the entire pool directory (candidates CSVs + judge side files) — private repo (token), archive, cloud path? Approximate size?
   No GPU is needed for the re-runs. It is needed only to build new pools (§1.7) — is GPU access available?
2. **Certificate guarantee track**: (a) finite-sample exact (betting) as the main result, with t demoted to an "asymptotic reference" / (b) state the asymptotic guarantee explicitly and
   lower the terminology to "asymptotic certificate" / (c) report both. **My recommendation: (c) + finite-population estimand** (including the exact PPI of §1.1).
3. **Estimand**: unify as the full-N finite-population mean and report the remainder-mean values alongside in the LOCK v0.4 tables — agreed?
4. **Candidate selection**: keep the code + rewrite Prop. B (my recommendation) vs. change the code to training-half selection. Both will be run for the efficiency comparison anyway.
5. **Pilot cost**: (a) honestly charge the whole pool / (b) redesign the pilot so that nG is not needed in the precision menu (e.g. estimate the gate policy's
   c from an HT subsample, or exclude gate policies from the precision experiment). (a) is possible immediately, (b) is a change of method. **Recommendation: recompute with (a) first and decide (b) after seeing the results.**
6. **Go/no-go criterion**: invest in direction 1 if the exact oracle (using true gaps and true residuals) reduces J50 **by ≥ 30% relative to static summation+CV on 2 or more collections**,
   otherwise switch to direction 2 (document-level finite-sample certification) — do you agree with this threshold? Do you want a different number?
7. **Handling pre-registered results**: keep the v0.4–v0.6 numbers as they are and add a separate "revised accounting" section — confirm.
8. **Manuscript wording**: whether to remove "halves the human judgments" and "certificate valid for any judge" from the abstract and introduction now and replace them with
   conditional wording, or do it all at once after recomputation.
9. **GitHub**: target branch for the push (proposal: `featured-prep`, main kept exactly at e0609ea), and how to leave the commit-timestamp evidence for the lock documents (§SUBMISSION_CHECKLIST 4).

## 7. Files changed today

- Modified: `04_code/lib/certificates.py`, `04_code/63_planner_v2.py`, `04_code/67_ppi_gain.py`, `04_code/81_sampling_baselines.py`,
  `04_code/82_active_inference.py`, `04_code/84_unify_metrics.py`, `02_literature/references_verified_2026-08-25.md`
- New: `04_code/lib/ledger.py`, `04_code/lib/design.py`, this document
- The existing `05_results/*` files were not changed (synthetic test outputs deleted). The previous session's minor `main.tex` edits and `SUBMISSION_CHECKLIST.md` are unchanged.

---

## 8. Decisions (2026-09-14 afternoon, approved) — final on the 9 items of §6

| # | Decision | Implementation status |
|---|---|---|
| 1 | Pools delivered as a compressed bundle (`pools_bundle/MANIFEST.csv`, `README_DATA.md`, `<stack>/runs/candidates/…`) via Drive/chat. No PAT handover. | `04_code/92_pools_manifest.py build|verify` written. **The bundle must also include the training pools `legacy/{nfcorpus,scifact,arguana,cqadupstack-android}`** (read by the LODO relevance model). |
| 2 | (c) report both, but **finite-sample exact (betting/EB) is the main claim**; t demoted to an "asymptotic/diagnostic benchmark". | Manuscript wording updated (§9). Results run with both `--bound bet` and `--bound t`. |
| 3 | estimand = full-N finite-population mean. Remainder only as a historical reproduction. | `63 --truth population` ready. |
| 4 | Keep `cand_on=val` + rewrite Prop. B in the form "simultaneous coverage of the fixed M(M−1) family". State explicitly the condition that **the family is not expanded after looking at the validation data**. `train` selection is an efficiency ablation. | `main.tex` Prop. B and appendix proof rewritten. |
| 5 | The pilot is charged 100% of the labels actually used (a). Redesign after seeing results. | `81/82/83 --pilot_cost full` is the default. `85_gates.py` reports `pilot_share`. |
| 6 | Two-stage Gate A + Gate B (below). | `85_gates.py`. |
| 7 | v0.4–v0.6 numbers immutable, marked "legacy/preregistered; superseded for cost conclusions". | Reflected in the Table 2 caption and Limitations. v2 results separated into `_v2_*` files. |
| 8 | Manuscript wording fixed immediately. | Done (§9). |
| 9 | `main`=e0609ea preserved, `featured-prep` branch, annotated tag `lock-e0609ea-2026-09-14`, Release attachments (SHA256SUMS etc.). | Local tag and branch created. `03_data/LOCK_ARTIFACTS_e0609ea/` (MANIFEST 244 files, SHA256SUMS, archive sha). Release attachments staged in `/workspace/release_staging/lock-e0609ea-2026-09-14/` — push and Release creation require account permissions. Zenodo/OSF snapshot recommended. |

### 8.1 Go/No-Go — two stages

With all accounting fixes applied (shared ledger, pilot full cost) and the same certificate applied to every arm, `85_gates.py` computes:

- **Gate A (structural head-room)**: `saving_oracle = 1 − J50(oracle_exact)/J50(static_sum)` is
  **≥ 30% on ≥ 2 collections** *and* **the median over the 4 collections is ≥ 20%**.
  The median condition prevents passing on only the two collections with extreme non-overlap.
- **Gate B (recoverability)**: a procedure that does not know the oracle (`plugin_exact`: s_j from the pilot and cumulative estimates, v̂ from the pilot residual model) achieves
  `recovery = (J50_static − J50_plugin)/(J50_static − J50_oracle)` **≥ 60%**, actual saving `saving_plugin` **≥ 15–20% on 2 or more collections**,
  and the plugin's wrong-certificate rate stays ≤ α.
- Verdict: A pass + B pass → proceed strongly with direction 1. A fail → drop direction 1 as the main contribution. A pass + B fail → the algorithmic gap is the research problem; the Featured claim is on hold.

`85_gates.py --baseline static_sum_v` is also examined: `static_sum_v` (π ∝ Σ|w_j|·√v̂) is a variance-aware static rule, so its difference from the oracle is
the head-room of **menu design itself**, while the difference from `static_sum` is the sum of variance awareness + menu design. Gate A's official baseline is `static_sum`, as decided.

### 8.2 `83_menu_allocation.py` v2 (implemented today)

- Arms added: `static_sum_v`, `oracle_exact`, `plugin_exact`. The oracle **knows the true slack s_j and the population residual-variance function v(p)=E[(y−ĵ)²|judge-prob bin]**
  but not the labels (using the true per-document residuals would make it a label-knowing oracle, inseparable from the gain of menu design). The plugin solves the same design with
  s_j from the pilot (round 1) and cumulative estimates (thereafter), and v̂ from the pilot residual model (floor 0.02, guaranteeing π>0 for every decision document).
- The design is solved at once for all documents of a round (thousands of documents, J=3, 150 EG iterations); the round budget equals that of the static arms (Σπ matched).
- Ledger charging, `--pilot_cost`, `--bound`, `design_obj` (each arm's realised max_j V_j/s_j² under the true v and s — structural head-room independent of the bound), `--legacy`.
- Finding: the old `per_pair` drew 3b documents per query, so its actual unique labels were more than twice those of the shared arm (revealed by the ledger). v2 equalises to b.
- Synthetic pool smoke test passed: costs equalised (≈1,195 vs 1,195), `design_obj` static/oracle_exact 1.7–2.2×, plugin in between. These are not real-data numbers.

### 8.3 The novelty bar for direction 1 (reflecting the user's warning)

min_π max_j V_j(π)/s_j² resembles transductive linear-bandit design (Soare et al. 2014; Fiez et al. 2019), and more directly
**Ochoa Rivera & Tewari (2024) "Optimal Thresholding Linear Bandit" (arXiv:2402.09467)** already gives a sample-complexity lower bound and an asymptotically optimal algorithm
(Lazy Track-Threshold-and-Stop) for the fixed-confidence ε-thresholding linear bandit — verified.
Therefore "solving menu certification as an optimal design" alone reduces to "thresholding linear pure-exploration with a specialised design matrix".
For a Featured contribution, all three layers are needed:
1. A different observation model — expensive gold + cheap judge prediction, with the rectifier Y_i − λŶ_i entering several contrasts simultaneously (must also be distinguished from MultiPPI's
   multi-source resource allocation).
2. Finite-sample certification as the core (porting an asymptotic normal stopping rule is weak).
3. A definition of the oracle complexity C* + C_alg ≤ c·C*·polylog for the adaptive procedure, or empirical recovery (Gate B).
Tentative title: **Exact, Cost-Aware Certification of Decision Menus with Imperfect Automated Judges**. Core theorem chain:
oracle design → adaptive realisable design → finite-sample-valid stopping.

### 8.4 Corrected framing of direction 2

"Document-level finite-sample certification — something nobody can do" is wrong: Waudby-Smith & Ramdas (NeurIPS 2020) already provide
time-uniform CSs for finite-population WoR sampling. The difficulty actually found is **an exact certificate of practical width under unequal/adaptive inclusion probabilities + HT weighting + multiple shared linear
contrasts + prediction-powered residuals**. The question: *Can exact finite-sample validity coexist with
aggressive importance sampling?* Query-level simple WoR + finite-population decomposition (μ_N = known λ·mean(Ŷ) + bounded rectifier mean)
combines naturally with betting/CS, but the extension to document-level adaptive unequal sampling is not automatic, and that is where the research problem begins.

### 8.5 Three cases for pilot cost (criteria for reading the results after recomputation)

(i) Saving remains large even after honest charging → no pilot redesign needed. (ii) The pilot is 50%+ of the cost but saving remains → "how much pilot information is
actually required?" becomes the second contribution. (iii) The pilot consumes almost all of the saving → the practical claim of the current procedure collapses, and the structure that reads n_G from the full pool
must change. On synthetic data pilot_share ≈ 0.53, so (ii)/(iii) are left open as possibilities.

## 9. Immediate manuscript fixes (done, `main.tex` recompiled to 14 pages, no warnings)

- Abstract (ii): "remains valid for any judge" → the judge cannot bias the bound + "asymptotically calibrated under the pre-registered
  t-based implementation; finite-sample-valid certificates are developed separately".
- Abstract (iii) and last sentence: "cuts … by more than half"/"halved" → "under the original per-comparison cost accounting, substantially reduced …;
  a revised accounting that charges every unique human label once, including the full pilot, is evaluated separately and supersedes
  the original for cost conclusions".
- Contribution 2 title "A certificate that is valid for any judge" → "A certificate whose bias does not depend on the judge" (§4 title likewise).
- Prop. B: rewritten in simultaneous-coverage form (fixed-family condition, the finite-sample counterexample for t D=1 w.p. 0.02, n=30, 55%, EB/betting as the intended replacement).
  The appendix proof also rewritten on the basis of event E.
- Legacy accounting/asymptotic bound stated explicitly in the Table 2 caption, the first reading of §5, Related work, Limitations and the conclusion.
- Added waudby2024estimating, waudby2020wor, ochoa2024thresholding, fiez2019transductive, soare2014linear to `refs.bib`.

## 10. Execution environment check (RunPod, 2026-09-14)

- GPU: RTX 3090 24 GB (CUDA 13.0 driver, torch 2.4.1+cu124 with CUDA available), CPU 256 threads, RAM 1 TB.
- Disk: container overlay 200 GB (199 GB free, possibly non-persistent across restarts) / `/workspace` network volume (persistent). Pool bundle, HF cache and logs go in `/workspace`.
- Installed: transformers (≥4.51), sentence-transformers, bm25s, accelerate additionally installed (needed for generating new pools/judgments; re-running 81–85 needs no GPU).
- Run script: `04_code/RUN_REVISED_ACCOUNTING.sh` (`POOLS`, `TRAIN_DIR` environment variables) — runs 81/82/83 v2 × 4 collections × judge × {t, bet} in parallel, then 84 and 85.
  Expected time: about 1 hour per collection for 83 v2 at 300 draws (single process); the whole run takes several hours in parallel.
## 11. Recomputed results — Gate A/B verdict (2026-09-14 21:15 KST, regenerated pools, 300 draws)

Runs: `81/82/83 --pilot_cost full --sampling shared` × {dbpedia-entity, dl212223, cast19, antique} × {llm, rr, inv} × {t, bet}.
83 takes about 25 minutes per collection with `OMP_NUM_THREADS=1` + the closed-form water-filling in `lib/design.py` (sort, O(n log n); agrees with the bisection implementation to 1e-6 on 200 instances).
Outputs: `05_results/menu_allocation/menu_alloc_*_v2_full_{t,bet}.csv`, `05_results/unified/gates_*.csv`.

### 11.1 Gate A — FAIL (t bound, baseline `static_sum`)

| collection | eps | J50 static | J50 oracle_exact | J50 plugin | saving_oracle | saving_plugin | recovery | obj ratio static/oracle | nonoverlap | pilot_share |
|---|---|---|---|---|---|---|---|---|---|---|
| dl212223 | 0.01 | 2930 | 2543 | 2663 | **13.2%** | 9.1% | 0.69 | 2.47 | 0.82 | 0.38 |
| dbpedia-entity | 0.01 | 3100 | 2962 | 3191 | 4.4% | −2.9% | — | 1.55 | 0.55 | 0.40 |
| dbpedia-entity | 0.02 | 2420 | 2351 | 2352 | 2.9% | 2.8% | 0.98 | 1.56 | 0.55 | 0.40 |
| antique | 0.01/0.02 | certification rate already > 0.5 at the minimum budget (30q) → J50 left-censored, saving not measurable (≈0%) | | | | | | 1.94 | 0.58 | 0.27 |
| dl212223 | 0.02 | 0.56 at the minimum budget (20q) → left-censored | | | | | | 2.44 | 0.82 | 0.38 |
| cast19 | both | certification rate ≤ 0.11 for every arm (up to budget 90q) → J50 not reached | | | | | | 1.9–2.1 | 0.60 | 0.38 |

- Even in the only uncensored "favourable" case (dl212223, nonoverlap 0.82, eps=0.01), the oracle saving is 13% (25% excluding the pilot), far short of the 30% threshold.
  There is no second collection. The median is also below 20%. **Gate A fails — direction 1 (optimal menu design) is not adopted as the main Featured contribution.**
- Gap between the structural signal and the certification-rate signal: on the design objective max_j V_j/s_j² the oracle is 1.5–2.5× better than static_sum, but the certification-rate curves
  separate only at low budgets (dl212223 B=20q: 0.39→0.51, B=30q: 0.55→0.71); as the budget grows, every arm attaches to the same plateau (0.66–0.85).
  The plateau is set by the fraction of uncertifiable menus (true regret near or above eps), so no design can cross it. That is, the head-room is confined to the "low-budget regime".
- The pilot is 27–40% of the unique labels (case (ii) of §8.5): it dilutes the saving, but even after removing it the saving does not reach 30%.
- Relative to `static_sum_v` (variance-aware static), the oracle saving is 6.6% on dbpedia and ≈0 elsewhere — the head-room of menu design itself is small.
- Consistency: the wrong-certificate rate of `plugin_exact` is 0.000 in every block; `per_pair` (after budget equalisation) is the lowest of all arms.

### 11.2 `--bound bet` — document-level finite-sample certification is not reachable in its current form

Across every collection, budget and arm, the certification rate of the betting upper bound is 0.00–0.013. The range of the HT-weighted increments is so large that the bound does not enter eps —
exactly as anticipated in §8.4. This fact itself gives empirical evidence for the problem statement of direction 2 (coexistence of aggressive importance sampling and exact validity).

### 11.3 Next steps following the verdict

Since A fails, by the rule of decision 6 we **switch to direction 2**: (a) first, attach a betting CS to the query-level WoR + finite-population decomposition (μ_N = λ·mean(Ŷ) + bounded rectifier)
and measure "at how many labels the exact certificate actually closes" (63 series); (b) at the document level, measure the width of the π-lower-bounded, truncated and bias-corrected estimators.
The results of direction 1 (oracle 13% / plateau analysis) remain in Limitations or an appendix as the negative result "design head-room is small".
The 81/82 v2 results (Table 2 regenerated, savings recomputed) went into the 84 outputs (`TABLES_v0.1.md`, `J50.csv`) and need a separate reading.

## 12. Change of goal (2026-09-14 21:30 KST, approved): stop the Featured extension → complete the manuscript for TMLR publication

User decision: the Gate A failure is a miss on the internal investment criterion, not a research failure. The paper is organised as an empirical and methodological analysis answering
**"in auditing retrieval policies with AI judges, where do the cost savings arise and under what conditions do they disappear"**. Direction 2 only to the extent needed to settle the manuscript's certification claims
(scope limited). Wrong-certificate 0.000 is reported only as an observed error rate. Procedure: read Table 2 → limited comparison of certification methods → rewrite claims and manuscript → verification before submission.

### 12.1 Re-reading of Table 2 — honest accounting (shared ledger + pilot 100%), t bound, 300 draws, J50 = number of unique human labels (pilot included)

The budget grid was extended (antique 5–150q, dl212223 5–90q, cast19 30–200q, dbpedia 20–150q) to remove censoring. Files: `*_v2_shared_full_t.csv` + `*_grid2_*`.

**eps = 0.02** (same setting as manuscript Table 2):

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
| pilot labels (common to all arms) | 631 | 631 | 631 | 975 | 975 | 975 | 1028 | 1028 | 1387 |
| max wrong-certificate rate (observed) | .003 | .003 | .003 | .003 | .003 | .003 | .003 | .003 | .007 |

Reading (conclusions that go into the manuscript as is):
1. **The saving from weighted sampling survives under honest accounting** — decision-weight IS vs uniform: antique −46%, dl212223 −53%, dbpedia > −51%
   (uniform not reached even at 5,461 labels), cast19 > −32%. The pilot is common to all arms, so it does not dilute this comparison. "halves" may be used **only for this row (humans-only,
   vs uniform)**. Variance ratio (independent of accounting) uniform/weighted = 2.6–4.3×, consistent with the manuscript's 2.9–4.4×.
2. **The judge's additional gain has become smaller and is conditional** — best judge arm vs decision-weight IS: antique −11% (Qwen), −7% (rr); cast19 −19% (Qwen, λ-fitted)
   / −15% (rr, robust); dbpedia −19% (Qwen) / −11% (rr); dl212223 −8% (Qwen, ρ≈0.46). The manuscript's "−21% / −14%" are figures under legacy accounting (pilot partially charged)
   and must be lowered to **−7 to −19%** under honest charging. On sampled labels excluding the pilot, dbpedia/Qwen −25%.
3. **The conditions under which the judge hurts are measured** — inverted (adversarial) judge + λ=1 CV: antique +8%, cast19 −5% (noise level); dbpedia rr/inv not run.
   The calibrated-judge rule is the worst (antique/inv +59%, dl212223 +13%) — the failure mode of not labelling "documents where the judge is confidently wrong". Fitting λ on the pilot
   (cvl) removes the inverted-judge loss to +1% (antique 888 vs 877).
4. **No claim of superiority over strong baselines (stratified, residual-estimated active)** — humans-only stratified ≈ decision-weight (±5%),
   differences among judge arms are within ±5% (J50 noise at 300 draws ≈ ±3–5%). The manuscript is rewritten to explain "the conditions under which the gain arises (ρ, cost structure, pilot share)"
   rather than "method superiority".
5. At eps=0.01, cast19 and dbpedia do not reach 50% even at 5,400 labels (ACT 0.29 / 0.49) — the table uses eps=0.02 as the main result, and eps=0.01 only for the cells that are reached.

### 12.2 Gate A appendix write-up — improving the variance objective ≠ improving certification cost

The results of §11 remain in an appendix as the negative result "with the current menus and data, sophisticated allocation optimisation does not substantially reduce the total certification cost".
On the design objective max_j V_j/s_j² the oracle is 1.5–2.5× better, but the J50 saving is 0–13%. The cause is split by the plateau decomposition (§12.3).

### 12.3 Plateau decomposition (83 `_diag`: static_sum / oracle_exact / plugin_exact, 300 draws)

Draws are split into **feasible** (true regret of the candidate chosen by the 20-query pilot ≤ eps — the cases where a correct certificate can exist) and infeasible.

| collection | feasible fraction | ACT (overall) at max budget | ACT\|feasible at max budget | low-budget oracle gain (ACT\|feas, static→oracle) |
|---|---|---|---|---|
| antique | 0.80–0.89 | 0.74–0.76 | 0.88–0.90 | B=20q eps=.02: 0.62→0.71 |
| dl212223 | 0.83–0.89 | 0.80–0.85 | 0.96 | B=20q eps=.01: 0.43→0.60; B=30q: 0.68→0.82 |
| dbpedia-entity | 0.83–0.93 | 0.53–0.70 | 0.63–0.75 | ≈0 (±0.03) |
| cast19 | 0.83–0.94 | 0.04–0.10 | 0.04–0.12 | ≈0 |

- Most of the plateau is **infeasible draws** (7–20%: the 20-query pilot picks a wrong candidate). The feasible-conditional ACT reaches 0.88–0.97 at large budgets.
  The remaining uncertified cases (3–12%) are those with small slack (slack below eps/2 in 0–9%).
- The gain from allocation optimisation is real **only in feasible runs and only in the low-budget regime** (dl212223 +0.14–0.17). As the budget grows, every arm reaches the feasible
  ceiling and the gain disappears. Converted to certification cost (J50) it is 0–13%. The manuscript reports both the "conditional gain (diagnostic)" and the "total cost (main result)".
- cast19 is a different kind of failure: feasible 84–94% yet ACT ≤ 0.12 — the policy gap is ≈0 (slack ≈ eps), so a certificate of width < 0.01 is needed and the sample is insufficient.
  No allocation can overcome this.

### 12.4 Limited comparison of certification methods (63, query level, population estimand, 300 repeats)

Implementation: `lib/certificates.py` `ucb_bet(..., N=)` = Waudby-Smith & Ramdas (2020) WoR betting CS (conditional mean of hypothesis m, m_i=(Nm−S_{i−1})/(N−i+1),
impossible hypotheses rejected immediately); synthetic validation over 400 runs, miscoverage 0.000 (target ≤ 0.1), shrinks to the exact mean as n→N. `ucb_finite_population`:
μ_N = (Σ_train D + (N−ntr)·UCB_rest)/N — the labelled train half is known exactly, and val is a WoR sample of the remaining N−ntr. `63 --bound bet_wor`,
arms `split_fp` (humans) / `split_fp_ppi` (rectifier D−λD̂, λ from train only, range [−1−λ, 1+λ]), `--ntr_fixed 20` (so that val can grow to the whole population).

| collection (N) | split_t (t, pre-registered design) | split_fp (t + FPC + known train) | split_fp (**bet_wor**, val ≤ N−20) | split_fp_ppi (bet_wor) |
|---|---|---|---|---|
| antique (200) | ACT 0.60, T̄ 126 | 0.82, T̄ 100 | **0.000** (T=150) | 0.000 |
| cast19 (159) | 0.76, T̄ 116 | 0.98, T̄ 86 | 0.28 (T=150 → val 130 = 82% of N) | 0.04 |
| dbpedia-entity (399) | 0.94, T̄ 137 | 0.99, T̄ 111 | 0.10 (T=350 → val 330 = 83%) | 0.007 |
| dl212223 (211) | 0.99, T̄ 88 | 1.00, T̄ 68 | 0.07 (T=200 → val 180 = 85%) | 0.04 |

Observed wrong-certificate rate: every arm ≤ 0.007 (t), 0 (bet_wor).

Conclusions (sentences that go into the manuscript):
- **The exact certificates implemented so far (betting, i.i.d. or WoR) are not practical at eps=0.01, N ≤ 400**: even after labelling more than 80% of the population,
  ACT ≤ 0.28. The cause is the range term R·log(1/δ)/n (R=2, δ=α/(looks·M(M−1)) ≈ 1e−3 → ≈0.14 ≫ eps at n=100), not the variance (synthetic:
  σ=0.05, n=100, bet width 0.06 vs t 0.015; independent of grid resolution). Removing the looks correction with a time-uniform CS reduces log(1/δ) 7.1→4.8, width by 15% — conclusion unchanged.
  This is a statement that **this bound is impractical in this setting**, not a claim that every exact certificate is inefficient, nor a demonstration of a new research gap.
- The PPI judge arm is worse under the exact bound (penalty from the rectifier range 1+λ) — state explicitly that "the judge gain is realised only under asymptotic certification".
- **The finite-population t certificate (FPC + known train half)** is consistently better than the pre-registered split_t (T̄ reduced 20–26%, ACT 0.82–1.00). But state explicitly that this is
  **a guarantee for the fixed evaluation set N**, not a guarantee of average performance over the future query distribution. Reported as a post-hoc variant.
- The manuscript's certification claim: "asymptotically calibrated (t) certificate; observed wrong-certificate rate ≤ 0.007 over all runs" — an observed error rate,
  not used as evidence of a guarantee. The finite-sample-exact variant goes in the appendix together with its cost.

### 12.5 Manuscript rewrite plan (next steps)
1. Table 2 → replace with the honest-accounting table of §12.1 (eps=0.02); move the pre-registered table to the appendix as the "pre-registered record".
2. Replace the figures in §5 "Three readings" (−46 to −53% humans-only; judge −7 to −19% conditional; inverted judge +8%, removed by λ-fit; calibrated rule worst).
3. In the abstract, introduction, contributions and conclusion, "halves" → only for the humans-only row; judge gain with its magnitude as is.
4. Appendix: menu-allocation negative result (§11, §12.3), exact-certificate cost (§12.4), finite-population t variant.
5. Limitations: observed error rate, fixed evaluation set, pilot share 27–40%, N ≤ 400.

### 12.6 Manuscript revision complete (main.tex, 17 pages, 0 warnings, `pdflatex` 3 times + bibtex)

- Abstract (ii)(iii), contributions 2 and 3, conclusion: "halves" → only for the humans-only weighted-sampling row (−46 to −53%, 4 collections); judge −7 to −19% conditional; inverted judge +8% → λ-fit +1%;
  calibrated rule +59%; oracle menu design 0–13%; the exact bound ≤ 28% even after labelling 85% of the population.
- §5.3 Table 2 replaced with the honest-accounting table (9 columns: ANTIQUE Q/R/I, CAsT Q/R/I, DBpedia Q/R, DL Q + pilot row), Figure F7_v2, "Four readings" rewritten
  (MC standard error 6–9% stated; judge increments unified on the λ-fit CV basis).
- §4: new "What exactness costs" paragraph after Prop. B (finite-population estimand, FPC-t 20–26% faster, betting WoR impractical, range-term diagnosis).
- §7 (2) menu-allocation negative result rewritten with the oracle design + plateau decomposition.
- Legacy J50 citations in §6.2/6.3 marked "per-comparison accounting" + ANTIQUE honest-accounting figures given alongside.
- Limitations: observed error rate ≠ guarantee, fixed evaluation set, pilot 27–40%, N=159–399.
- New appendices: C "Pre-registered cost table (original accounting)" (old Table 2 moved), D "Allocation across the comparisons of a menu" (Table 5),
  E "Exact and finite-population certificates" (Table 6 + synthetic diagnostics). `86_table2_v2.py` stated in the reproducibility note.
- New script: `04_code/86_table2_v2.py` (TABLE2_v2_eps*.csv, TABLE2_v2.tex, F7_J50_v2.png). `--tag` in `82`; feasible/act_feasible/slack columns in `83`;
  `--bound bet_wor`, `--ntr_fixed`, arms `split_fp`/`split_fp_ppi` in `63`; `ucb_bet(N=)`, `ucb_finite_population` and the FPC for t in `lib/certificates.py`;
  closed-form water-filling in `lib/design.py`.

### 12.7 Data archive
- `pools_bundle` MANIFEST.csv (53 files, 68.5 MB, sha256) + README_DATA.md (models, versions, sources, licence caveats) created, verify 0 problems. `emb/` (2.2 GB embedding cache) excluded.
- Upload to the Hugging Face **private** dataset `<private Hugging Face dataset>` complete (56 files). `_texts.tsv` inherits the original collections' licences and therefore stays private.

### 12.8 Remaining work (before submission)
1. Full read-through of the manuscript (numbers ↔ CSV cross-check completed for §12.1 and the appendix tables; the legacy figures in §3 and §6 are unchanged).
2. `git commit` (featured-prep) — 114 changed files in the working tree; reference §11–12 in the commit message.
3. Remaining decision: whether to put the eps=0.01 cells (cast/dbpedia n.r.) into the table (currently mentioned only in a sentence in the body).

### 12.9 Repository history clean-up (user request, 2026-09-14 23:05 KST)
The auto-inserted `Co-authored-by` trailers (Cursor / Claude) were removed from three commit messages to leave a single author. Trees, authors and dates unchanged.
Hash changes: e0609ea→6aa813a, 978856e→d9aeef3, 9aa624e→d678bc7 (`03_data/LOCK_ARTIFACTS_e0609ea/HASH_MAP.txt`). Tag `lock-e0609ea-2026-09-14` moved
to 6aa813a. A local `commit-msg` hook strips the trailer from subsequent commits as well. Backup branches `backup/main-e0609ea`, `backup/featured-prep-9aa624e` exist only locally.

## 13. Response to the second review (2026-09-14 23:18 KST) — fixes to execution errors and estimand consistency, and regeneration

### 13.1 Fixes
| Issue | Fix | File |
|---|---|---|
| `63` finite-population arm evaluation stops when split_t/ppi/auto terminate → unevaluated recorded as `abstain` | Continue evaluating while any of the 5 split arms has not terminated | `63_planner_v2.py` |
| `81/82/83` estimate μ_R (mean over queries excluding the pilot) but decide on μ_N (whole population) | The pilot's D_j are known exactly; the document-level estimates of the queries drawn WoR from the remaining R get the two-stage variance V̂=(1−f)s_b²/n + f·mean(v̂_within)/n (f=n/|R|, v̂=HT document-sampling variance), giving UCB_N=(Σ_A D+|R|·UCB_R)/N. Synthetic MC, 2000 runs: miscoverage 0.039 (f=0.3), 0.052 (f=1), target 0.05 | `lib/certificates.py` (`ht_var_hat`, `ucb_two_stage`, `ucb_population_two_stage`), `81`, `82`, `83` (`--bound t`, non-legacy only) |
| `86` reads grid2 files twice | `sorted(set(files))`; v2 figures confirmed unchanged. `v3` generation (`*_v3b{budget}_*`) supported; eps=0.01 tex also generated | `86_table2_v2.py` |
| Appendix E population label fraction omits the 20 pilot queries | T defined as the total number of audited queries including the pilot; T/N reported. New table generator | `87_exact_table.py` |
| Body text says the candidate is chosen on the training half (the runs use validation) | §4 split-certificate paragraph and Prop. B proof corrected to "(as implemented)" | `main.tex` |
| "cannot bias the bound" | Unbiasedness of the estimator (holds for every judge) and coverage of the bound (depends on the bound's assumptions) stated separately (abstract, contribution 2, conclusion) | `main.tex` |
| `67` ess_gain = ratio of estimated SEs | Name unified to "ratio of squared estimated standard errors". The MC-MSE column can be regenerated only for the settings whose pool is in the bundle (legacy/modern/judged reranker); the TREC DL 2019/20 pools and the Qwen3-8B judgments for covid/touché are not in the bundle | `main.tex`, `05_results/ppi_gain/` |

### 13.2 Re-runs (all 300 draws, one process per budget, tag `_v3b{B}`)
- `81/82`: antique {5..150}×{llm,rr,inv}, cast19 {15..200}×3, dbpedia {5..200}×{llm,rr}, dl212223 {5..90}×llm → every cell uncensored (eps=0.02).
- `83`: 4 collections, all arms, same grid. `85_gates.py --gen v3`.
- `63`: `--ntr_fixed 20 --truth population --looks 30 50 70 90 120 150 158 199 200 210 250 300 350 398` (last look = N−1), bound t / bet_wor, 300 repeats → `planner_v2/{stack}_llm_pop_{bound}_ntr20_v3`.

### 13.3 Where the results changed
**Appendix E.** After the gate fix, the betting WoR arm eventually certifies 88–100%, but on ANTIQUE and CAsT (median slack≈ε) it fires 0–2% at T/N≤0.75, 3–5% at ≤0.9, and only at the last look (T=N−1, full census). DBpedia (N=399) and DL (large margin): 90%/73% at 75%. Conclusion revised from "impractical" to "affordable when the margin or N is large; a full census when margin≈ε". FPC-t has ACT 1.00 on all 4 collections, T̄ shortened 20–27%.

**Table 2 (v3).** With the estimand made consistent, every cost went down, and CAsT moved the most (weighted 3,894→2,127). Humans-only saving 40/56/54/56% (ANTIQUE/CAsT/DBpedia/DL). λ-fit Qwen3-8B increment 8/19/20/3%; reranker 8–19%; inverted judge 2–5% (noise). Pilot share (on weighted J50) 76/46/41/78%. Increment excluding the pilot 32–35% (DL 16%). Spread among the 4 judge-assisted arms 3–11%. Calibrated-judge rule: inverted judge +40% (ANTIQUE), +29% (CAsT). At eps=0.01 every cell except uniform (CAsT, DBpedia) is also reached → appendix table `tab:J50b`.

**Menu allocation (Gate A re-evaluated).** Oracle saving 2/14/0/7% (eps 0.02), 1/17/1/6% (0.01); plug-in 2/12/−13/2.5%. Gate A FAIL stands. Plateau decomposition: infeasible 8–16%; ACT|feasible at the maximum budget 0.97–1.00 on ANTIQUE, CAsT, DL (→ plateau = pilot cap), 0.83–0.89 on DBpedia (residual variance). **CAsT was "uncertifiable" before the estimand fix and now certifies, and it is the only collection where the oracle gain is meaningful (14–17%).** Manuscript §7(2) and Appendix D rewritten collection by collection.

### 13.4 Remaining
- `67` judged stack: the ρ in the stored CSV (covid 0.794, dbpedia 0.54) is reproduced exactly neither by the 3-judged LODO (0.50/0.12) nor by `--train_dir legacy` (0.759/…) → original run configuration unknown. The MC-MSE validation of F4 is reported on the 24 legacy/modern points that reproduce exactly + the 9 judged points from the re-run configuration, with the configuration difference stated.
- The pilot × ε sweep (shared candidate, three measurements) comes after this commit.

### 13.5 Response to the third review (2026-09-15 00:45 KST)
- **Attribution of the two fixes separated on the same draw record** (`81 --tag _attr_b*`, columns `act_iid`/`act_pilot_iid`/`act`): CAsT weighted J50 3,907 (i.i.d. t) → 2,600 (pilot accounted exactly, −33%) → 2,127 (two-stage variance + FPC, further −18%); ANTIQUE 881 → 851 → 836; DL already has ACT≥0.5 at the smallest grid budget (censored). Recorded in the estimand paragraph of manuscript §5.3.
- **Appendix E wording fixed**: T=N−1 is not a full census but "a budget close to a full census"; "the exact mean is known" deleted; stated that on ANTIQUE and CAsT exact certification yields almost no real label saving. The margin explanation is downgraded to a hypothesis requiring a fixed-candidate experiment that separates slack s=ε−Regret_N(m̂), N, residual variance and range.
- **The two gains of the shared structure separated**: the `per_pair` arm of `83` (independent sampling per comparison, same budget, same ledger) vs `static_sum` (reuse, fixed allocation) → reuse gain ANTIQUE 43%, DBpedia 59%, DL 35%, CAsT ≥29% (per_pair n.r.). Allocation-optimisation gain (static_sum vs oracle) 0–14%. Recorded in manuscript §7(2) and Appendix D as "sharing pays, optimising the shared allocation pays little". `nonoverlap` is 1−Jaccard between the band of the binding comparison (minimum |gap|) and the union of the remaining bands, so it measures the shared structure of the comparison that blocks certification (confirmed).
- **F4**: MC-MSE validation performed on the 33 reproducible settings (BEIR 24 + judged reranker 9, the latter matching the stored ρ under the `--train_dir legacy` configuration): corr(MC gain, ρ)=0.78, median difference from the estimated-SE ratio 0.03, and on TREC-COVID the estimated-SE ratio underestimates the gain (1.1–1.2 vs 1.5–1.7). 90% interval coverage mean 0.90, minimum 0.81. The remaining 21 settings report only the estimated-SE ratio because the bundle is absent.
- The first experiment of ICML track B is fixed as the synthetic version of the three-procedure comparison per_pair/static/oracle above (only the degree of sharing is varied). Since the real data already show that the first gain is large and the second small, the role of the synthetic experiment is to test "does the shared structure (band overlap) determine the size of the first gain".

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
