#!/usr/bin/env bash
# Build an anonymous supplementary zip for OpenReview: the tracked tree without .git, without the release-only record sets
# and without the submission-form notes.
# Usage: SUPP_IDENT_REGEX='account|e-mail|surname' RELEASE_DIR=<dir with draw_records.tar.gz and locked_predict_records.tar.gz> 05_paper/make_supplementary.sh [out.zip]
#        (the regex is deliberately not stored here; the two archives are added under release/ when RELEASE_DIR is given)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/root/judge-audit-supplementary.zip}; rm -f "$OUT"
git ls-files | grep -v "_draws.csv$\|_predict.csv$" | grep -v "^05_paper/OPENREVIEW_FORM.md$" > /tmp/supp_files.txt
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
    rd=os.environ.get('RELEASE_DIR',''); added=[]
    for name in ['draw_records.tar.gz','locked_predict_records.tar.gz']:
        fp=os.path.join(rd,name) if rd else ''
        if fp and os.path.exists(fp):
            z.write(fp, 'judge-audit-supplementary/release/'+name, compress_type=zipfile.ZIP_STORED); added.append(name)
print('files', len(files), '| identifying strings found in:', (hits or 'none') if bad else 'not checked (set SUPP_IDENT_REGEX)', '| release archives added:', added or 'none (set RELEASE_DIR)')
PY
ls -la "$OUT" | awk '{print $5/1e6" MB"}'
