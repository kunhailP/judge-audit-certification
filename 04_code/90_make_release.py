#!/usr/bin/env python3
"""Build a clean, anonymised release directory for the paper (no git history): curated code, row-level results,
figures, docs, and a README with reproduction commands. The user initialises the git repository themselves."""
import os, shutil, sys
HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HUB), "release_nonneutral_judge")
if os.path.exists(OUT): shutil.rmtree(OUT)
def cp(src, dst=None):
    s = os.path.join(HUB, src); d = os.path.join(OUT, dst or src); os.makedirs(os.path.dirname(d), exist_ok=True)
    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
for f in ["04_code/lib/certificates.py", "04_code/lib/neutrality.py", "04_code/20_budget_curves.py", "04_code/61_build_pool.py", "04_code/63_planner_v2.py",
          "04_code/64_trecdl_pool.py", "04_code/66_llm_judge.py", "04_code/67_ppi_gain.py", "04_code/69_judged_pool.py", "04_code/70_neutrality_eval.py",
          "04_code/72_selfpref.py", "04_code/73_mpnet_judge.py", "04_code/75_dlv2_pool.py", "04_code/76_primary_report.py", "04_code/77_rho_map.py",
          "04_code/78_plot_rho_map.py", "04_code/80_weighted_audit.py", "04_code/81_sampling_baselines.py", "04_code/82_active_inference.py",
          "04_code/83_menu_allocation.py", "04_code/84_unify_metrics.py", "04_code/85_plot_J50.py", "04_code/86_f1_weighted.py", "04_code/87_cast_pool.py",
          "04_code/88_cast_report.py", "04_code/89_antique_pool.py",
          "06_paper/THEORY_v0.1.md", "06_paper/TABLES_v0.1.md", "06_paper/DRAFT_v0.4_metrics_section.md", "06_paper/DRAFT_v0.4_related_work.md",
          "06_paper/DRAFT_v0.4_appendix_f1.md", "01_design/PAPER_2027_DESIGN_v0.1.md", "01_design/EXPLORATION_menu_allocation.md",
          "03_data/PROSPECTIVE_LOCK_v0.4_2026-09-10.md", "03_data/PROSPECTIVE_LOCK_v0.5_2026-09-10.md", "02_literature/references_verified_2026-08-25.md"]:
    if os.path.exists(os.path.join(HUB, f)): cp(f)
for d in ["05_results/planner_v2", "05_results/primary_dlv2", "05_results/primary_cast", "05_results/ppi_gain", "05_results/rho_map", "05_results/selfpref",
          "05_results/sampling_baselines", "05_results/active_inference", "05_results/weighted_audit", "05_results/menu_allocation", "05_results/f1_weighted",
          "05_results/unified", "05_results/neutrality", "06_paper/tmlr_submission"]:
    if os.path.exists(os.path.join(HUB, d)): cp(d)
for junk in ["06_paper/tmlr_submission/preview"] + [f"06_paper/tmlr_submission/{x}" for x in ["b1.log", "b2.log", "b3.log", "bib.log", "build1.log", "build2.log", "build3.log", "main.aux", "main.log", "main.out", "main.blg"]]:
    p = os.path.join(OUT, junk)
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)
open(os.path.join(OUT, "README.md"), "w").write("""# Certifying retrieval-policy deployment with non-neutral AI judges — code and results

Anonymous research artifact accompanying a TMLR submission. Every number in the paper is regenerated from the row-level
result files under `05_results/` by the scripts under `04_code/`.

## Layout
- `04_code/lib/certificates.py` — split / PPI++ certificates (cross-fitted lambda, simultaneous corrections), recalibration bounds
- `04_code/lib/neutrality.py` — rho lower bounds (Fisher-z, percentile, BCa) and the judge-adoption score
- `04_code/6x_*.py` — candidate-pool builders (BEIR legacy/modern stacks, fully judged pools for TREC DL v1/v2, CAsT 2019, ANTIQUE), LLM judges
- `04_code/63_planner_v2.py` — sequential certificates (humans only / always-PPI / rule-based adoption) with per-method random streams
- `04_code/67,70,72,77` — PPI gain, neutrality diagnostic, self-preference test, mechanism map (F6)
- `04_code/80–83` — document-level auditing: decision-weighted sampling, judge control variate (coefficient 1 and pilot-lambda), sampling baselines, faithful active inference, menu-level allocation
- `04_code/84–85` — unified metric J50 and Figure F7; `86` — set-F1 linearised audit; `88` — pre-registered report
- `03_data/PROSPECTIVE_LOCK_v0.4/v0.5/v0.6` — pre-registration documents (committed before the target data were analysed)
- `06_paper/tmlr_submission` — LaTeX source and compiled PDF; `06_paper/THEORY_v0.1.md` — propositions and proofs
- `01_design/PAPER_2027_DESIGN_v0.1.md` — complete development log including negative results and retractions

## Reproduction
Pools: `python3 04_code/61_build_pool.py --beir <BEIR dir> --out <pools>` (BEIR), `64/75/87/89` for the fully judged pools.
Judges: `python3 04_code/66_llm_judge.py --cand <pool dir> --texts ... --queries ... --model Qwen/Qwen3-8B --tag llm <name>`.
Experiments: run the scripts with `--pools <pools> --stack <stack> --names <collection> --judge <llm|rr|mistral|inv> --train_dir <BEIR legacy pools>`.
Compile the paper: `cd 06_paper/tmlr_submission && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Requirements: Python 3.11, numpy, pandas, scipy, scikit-learn, torch, transformers>=4.51, sentence-transformers, bm25s; one 24 GB GPU for the judges.
""")
print("release written to", OUT)
