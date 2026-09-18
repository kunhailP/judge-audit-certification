# OpenReview submission form — TMLR (prepared 2026-09-18; excluded from the supplementary zip)

## Title
Certifying Retrieval Policies with Non-Neutral AI Judges: What a Judge Saves, and How a Pilot Predicts It

## Abstract (plain text with $...$ TeX; 279 words)
Before a retrieval policy is deployed in a new domain, an auditor must certify from a limited human audit that it is within a tolerance $\varepsilon$ of the best alternative. LLM relevance judges can label every candidate document, but may be biased toward the policies they judge. We carry prediction-powered and active inference to this certification problem and measure what they deliver on nine fully judged retrieval collections, five AI judges, three pre-registered evaluations and one locked independent test. The value of a judge is set by the correlation $\rho$ between true and judge-predicted paired policy differences, not by its label accuracy: over 54 settings the realised gain correlates $0.75$ with $\rho$ and $0.03$ with accuracy. A sample-split certificate uses the judge only as a mean-zero control variate, so fixed policy contrasts are estimated without bias for any judge; coverage rests on an asymptotically calibrated $t$ bound whose observed wrong-certificate rate is at most $0.013$. With every human label charged once, pilot included, importance sampling by closed-form per-document decision weights needs $40–56%$ fewer labels than uniform sampling; a $\lambda$-fitted judge control variate saves a further $3–20%$ where $\rho$ is high and nothing otherwise; an oracle allocation across the comparisons of a menu saves $0–14%$ at $\varepsilon=0.02$. A single cost model, saving $=$ variance reduction $\times$ post-pilot share of the cost, fits 193 cells with correlation $0.95$, and both factors can be computed from a 20-query pilot. In tests with predictions fixed before the audit, including one collection never used in development, the pilot predicted the relative cost of designs (correlation $0.83–0.97$) but over-estimated the individual judge's gain, and a single-pilot adoption rule missed the usefulness criteria set for it.

## Authors
Kunwoo Park (OpenReview profile required; the PDF and the supplementary material carry no author information).

## PDF
`06_paper/tmlr_submission/main.pdf` (24 pages: 13 pages of main content, references on pages 14–15, appendices A–I on pages 16–24; `\usepackage{tmlr}` without the `[accepted]` option; PDF Author/Title metadata empty).

## Submission type
The main content (everything before the references) is 13 pages. TMLR defines a *regular* submission as at most 12 pages of main content; the current PDF is therefore a **long submission** unless one page is cut. Candidates for the cut, in order of least damage: (1) move Figure 3 (cost of every arm relative to decision-weight sampling, which duplicates Table 2) to Appendix G; (2) move Table 3 (TREC DL 2021–23 pre-registered counts) to Appendix I and keep the one-sentence summary; (3) shorten Section 4's "What exactness costs" paragraph, which Appendix E repeats. Moving one figure alone does not reach 12 pages (checked); (1)+(2) or (1)+(3) is needed.

## Supplementary material
Build with `SUPP_IDENT_REGEX='<account>|<surname>|<given name>|<e-mail>' 06_paper/make_supplementary.sh <out.zip>` from the submission commit. The zip contains the tracked tree without `.git`, without the two release-only record sets (`*_draws.csv`, `*_predict.csv`) and without this file; every table and figure can be regenerated from it (README, "Reproduction"). The script prints any file that still contains one of the identifying strings; the expected answer is `none`. Size must stay below 100 MB.

## Previous TMLR submission / changes since last submission
None (first submission). Leave both fields empty.

## Competing interests
N/A (no financial relationship with an entity that could be perceived to influence the work in the last 36 months; no hardware or cloud donations). Confirm before submitting.

## Human subjects reporting
N/A (no experiments with human subjects; all human relevance labels are the released NIST / BEIR / ANTIQUE / NeuCLIRBench judgments).

## License
CC BY 4.0.

## Anonymity checklist (double-blind)
- No GitHub or Hugging Face link in the PDF or in the supplementary zip; the code is attached as the zip. The public repository (personal account) must not be linked anywhere in the submission; make it private again for the review period or use an anonymous mirror.
- `git grep -i` for the account name, surname, given name, e-mail and the former repository name returns nothing in the tree.
- Commit history is not part of the zip; the lock ordering is documented in `05_results/heldout/LOCK_NOTE.md` and Appendix I says it is self-reported.
- The candidate pools (document texts) are not attached; the README states why and what the pool CSVs contain.

## Suggested Action Editor expertise
Prediction-powered / active inference, sequential testing, IR evaluation with LLM judges.
