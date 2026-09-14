#!/usr/bin/env python3
"""Build or verify MANIFEST.csv (relative_path,bytes,sha256) for a pools bundle (review 2026-09-14, item 1).

  build : python3 92_pools_manifest.py build  <pools_bundle_dir>  [--readme code_commit=e0609ea judge_model=... ]
          writes <dir>/MANIFEST.csv over every regular file below <dir> (except MANIFEST.csv itself) and, if
          key=value pairs are given, <dir>/README_DATA.md.
  verify: python3 92_pools_manifest.py verify <pools_bundle_dir>
          recomputes sizes and hashes, reports missing / extra / changed files, exit code 1 on any mismatch.

Only the files the analysis scripts read need to be in the bundle: <stack>/runs/candidates/<name>.csv,
<name>_meta.csv and the judge side files <name>_llm.csv, _mistral.csv, _mpnet.csv (relative paths preserved).
"""
import csv, hashlib, os, sys


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def walk(root):
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn); rel = os.path.relpath(p, root)
            if rel in ("MANIFEST.csv",):
                continue
            yield rel, p


def build(root, readme_kv):
    rows = [(rel, os.path.getsize(p), sha256(p)) for rel, p in sorted(walk(root))]
    with open(os.path.join(root, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["relative_path", "bytes", "sha256"]); w.writerows(rows)
    print(f"MANIFEST.csv: {len(rows)} files, {sum(r[1] for r in rows)/1e6:.1f} MB")
    if readme_kv:
        with open(os.path.join(root, "README_DATA.md"), "w") as f:
            f.write("# Pools bundle\n\n")
            for kv in readme_kv:
                f.write(f"{kv}\n")
            f.write(f"\nfiles={len(rows)}\nmanifest=MANIFEST.csv (relative_path,bytes,sha256)\n")
        print("README_DATA.md written")


def verify(root):
    man = {}
    with open(os.path.join(root, "MANIFEST.csv"), newline="") as f:
        for r in csv.DictReader(f):
            man[r["relative_path"]] = (int(r["bytes"]), r["sha256"])
    seen = set(); bad = 0
    for rel, p in walk(root):
        seen.add(rel)
        if rel == "README_DATA.md" and rel not in man:
            continue
        if rel not in man:
            print(f"EXTRA    {rel}"); bad += 1; continue
        b, h = man[rel]
        if os.path.getsize(p) != b:
            print(f"SIZE     {rel}: {os.path.getsize(p)} != {b}"); bad += 1; continue
        if sha256(p) != h:
            print(f"SHA256   {rel}"); bad += 1
    for rel in man:
        if rel not in seen:
            print(f"MISSING  {rel}"); bad += 1
    print(f"verified {len(man)} files, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "verify"):
        print(__doc__); sys.exit(2)
    mode, root = sys.argv[1], sys.argv[2]
    if mode == "build":
        kv = sys.argv[4:] if len(sys.argv) > 3 and sys.argv[3] == "--readme" else []
        build(root, kv)
    else:
        sys.exit(verify(root))
