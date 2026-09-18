#!/usr/bin/env bash
# Build an anonymous supplementary zip for OpenReview: the tracked tree without .git, without the release-only record sets
# and without the submission-form notes.
# Usage: SUPP_IDENT_REGEX='account|e-mail|surname' 06_paper/make_supplementary.sh [out.zip]   (the regex is deliberately not stored here)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/root/judge-audit-supplementary.zip}; rm -f "$OUT"
git ls-files | grep -v "_draws.csv$\|_predict.csv$" | grep -v "^06_paper/OPENREVIEW_FORM.md$" > /tmp/supp_files.txt
python3 - "$OUT" <<'PY'
import sys, zipfile, re
out=sys.argv[1]; files=[l.strip() for l in open('/tmp/supp_files.txt') if l.strip()]
import os
pat=os.environ.get('SUPP_IDENT_REGEX','')  # account name, e-mail, surname, ... ; kept out of the tracked script
bad=re.compile(pat, re.I) if pat else None; hits=[]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:
        try:
            if bad and bad.search(open(f,'r',errors='ignore').read()): hits.append(f)
        except Exception: pass
        z.write(f, 'judge-audit-supplementary/'+f)
print('files', len(files), '| identifying strings found in:', (hits or 'none') if bad else 'not checked (set SUPP_IDENT_REGEX)')
PY
ls -la "$OUT" | awk '{print $5/1e6" MB"}'
