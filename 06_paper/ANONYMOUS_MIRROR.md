# Anonymous mirror for double-blind review (how to set it up, 5 minutes)

The submission repository `judge-audit-certification` is private and its tree contains no author names, e-mail addresses or
account names (checked with `git grep -i` for the account, the e-mail and the old repository name; the only remaining personal
information is the commit author field, which the services below hide).

## Option 1 (recommended): Anonymous GitHub — https://anonymous.4open.science
1. Open the site, click "Anonymize", sign in with the GitHub account that owns the repository (the repository can stay private;
   the service asks for read access once).
2. Repository: `judge-audit-certification`, branch `main`.
3. Terms to anonymize (each becomes `XXXX` in the mirror): the account name, the account name in lower case, the old repository
   name `nonneutral-judge-audit`, the private dataset name `nonneutral-judge-audit-pools`, and the author's surname and given name.
4. Options: hide the commit history? **Yes** for review (the history carries the author field). Keep "expiration" at the end of
   the review period plus one month. Leave file filters empty (the tree is 10 MB; the two large record sets are release assets and
   are not needed to regenerate any table or figure).
5. Copy the `https://anonymous.4open.science/r/...` link into the OpenReview form's "code" field (not into the manuscript, which
   currently carries no link). Mention in the form that the commit history (lock timestamps) will be made public on acceptance.

## Option 2: supplementary zip on OpenReview (no link at all)
`SUPP_IDENT_REGEX='<account>|<e-mail>|<surname>' 06_paper/make_supplementary.sh` builds `judge-audit-supplementary.zip` and reports any file that still contains one of the terms (the terms are passed in, never stored in the tree) from the tree without the release assets and without
`.git`; it is about 10 MB and contains everything needed to regenerate every table and figure from the committed result files.
Upload it as supplementary material. State that the full history with lock timestamps is released on acceptance.

## What to check before either option
- `git grep -il "<account name>\|<surname>\|<e-mail>"` on `main` returns nothing.
- The manuscript `06_paper/tmlr_submission/main.tex` has `\usepackage{tmlr}` without the `[accepted]` option.
- The design log `01_design/` is in Korean and names no person; it can stay.
