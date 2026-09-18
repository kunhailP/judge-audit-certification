# OpenReview submission form — TMLR (prepared 2026-09-18; excluded from the supplementary zip)

## Title
Certifying Retrieval Policies with Non-Neutral AI Judges: What a Judge Saves, and How a Pilot Predicts It

## Abstract (plain text; formulas in $...$; 288 words)
Before a retrieval policy is deployed in a new domain, an auditor must certify from a limited human audit that it is within a tolerance $\varepsilon$ of the best alternative. LLM relevance judges can label every candidate document, but may be biased toward the policies they judge. We carry prediction-powered and active inference to this certification problem and measure what they deliver on nine fully judged retrieval collections, five AI judges, three pre-registered evaluations and one locked independent test. The value of a judge is set by the correlation $\rho$ between true and judge-predicted paired policy differences, not by its label accuracy: over 54 settings the realised gain correlates 0.75 with $\rho$ and 0.03 with accuracy. A sample-split certificate uses the judge only as a mean-zero control variate, so fixed policy contrasts are estimated without bias for any judge; coverage rests on an asymptotically calibrated $t$ bound whose observed wrong-certificate rate is at most 0.013 (type-I error at most 0.03 under boundary stress designs). With every human label charged once, pilot included, importance sampling by closed-form per-document decision weights needs 40–56% fewer labels than uniform sampling; a $\lambda$-fitted judge control variate saves a further 3–20% where $\rho$ is high and nothing otherwise; an oracle allocation across the comparisons of a menu saves 0–14% at $\varepsilon=0.02$. A single cost model, saving = variance reduction $\times$ post-pilot share of the cost, fits 193 cells with correlation 0.95, and both factors can be computed from a 20-query pilot. In tests with predictions fixed before the audit, including one collection never used in development, the pilot predicted the relative cost of designs (correlation 0.83–0.97) but over-estimated the individual judge's gain, and a single-pilot adoption rule missed the usefulness criteria set for it.

## Authors
The author (OpenReview profile required; the PDF and the supplementary material carry no author information).

## PDF
`05_paper/tmlr_submission/main.pdf` (24 pages: main content on pages 1–12, references from page 12, appendices A–J from page 15; `\usepackage{tmlr}` without the `[accepted]` option; PDF Author/Title metadata empty).

## Submission type
**Regular submission** (no more than 12 pages of main content): the References heading falls on page 12. Re-check after any further edit (the page on which the References heading appears must be at most 13 with the heading not at its top, or at most 12).

## Supplementary material
Build with `SUPP_IDENT_REGEX='<account>|<surname>|<given name>|<e-mail>' RELEASE_DIR=<dir holding the two record archives> 05_paper/make_supplementary.sh <out.zip>` from the submission commit. The zip contains the tracked tree without `.git` and without this file, plus the two record archives under `release/` (`draw_records.tar.gz` 42 MB, `locked_predict_records.tar.gz` 48 MB, both checked to contain no identifying strings), so that every table and figure and the draw-level bootstrap intervals and pilot-rule scores can be regenerated (README, "Reproduction"). The script prints any file that still contains one of the identifying strings; the expected answer is `none`. Total size about 96 MB, under the 100 MB limit; check the printed size.

## Previous TMLR submission / changes since last submission
None (first submission). Leave both fields empty.

## Competing interests
N/A (no financial relationship with an entity that could be perceived to influence the work in the last 36 months; no hardware or cloud donations). Confirm before submitting.

## Human subjects reporting
N/A (no experiments with human subjects; all human relevance labels are the released NIST / BEIR / ANTIQUE / NeuCLIRBench judgments).

## License
CC BY 4.0.

## Anonymity checklist (double-blind)
- No GitHub or Hugging Face link in the PDF or in the supplementary zip; the code is attached as the zip. TMLR permits earlier public versions of a manuscript; what matters is that the submission itself is anonymous, so the public repository (personal account) must not be linked or named anywhere in the submission. Making it private during review is optional.
- `git grep -i` for the account name, surname, given name, e-mail and the former repository name returns nothing in the tree.
- Commit history is not part of the zip; the lock ordering is documented in `04_results/heldout/LOCK_NOTE.md` and Appendix I says it is self-reported.
- The candidate pools (document texts) are not attached; the README states why and what the pool CSVs contain.

## Suggested Action Editor expertise
Prediction-powered / active inference, sequential testing, IR evaluation with LLM judges.
