#!/usr/bin/env bash
# Build an anonymous supplementary zip for OpenReview: the tracked tree without .git and without the release-only record sets.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/root/judge-audit-supplementary.zip}; rm -f "$OUT"
git ls-files | grep -v "_draws.csv$\|_predict.csv$" | grep -v "^06_paper/SUBMISSION_CHECKLIST.md$\|^06_paper/ANONYMOUS_MIRROR.md$\|^06_paper/reviews/" > /tmp/supp_files.txt
python3 - "$OUT" <<'PY'
import sys, zipfile, re
out=sys.argv[1]; files=[l.strip() for l in open('/tmp/supp_files.txt') if l.strip()]
bad=re.compile(r'kunhail|pkw31386094|@gmail', re.I); hits=[]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:
        try:
            if bad.search(open(f,'r',errors='ignore').read()): hits.append(f)
        except Exception: pass
        z.write(f, 'judge-audit-supplementary/'+f)
print('files', len(files), '| identifying strings found in:', hits or 'none')
PY
ls -la "$OUT" | awk '{print $5/1e6" MB"}'
