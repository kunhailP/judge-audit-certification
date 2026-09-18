#!/usr/bin/env python3
"""Different-family control judge: all-mpnet-base-v2 cosine, label-free rule
(top 30% of each query's pool by cosine -> relevant; p_rel = within-pool rank
percentile).  Writes <name>_mpnet.csv next to the pool."""
import argparse, csv, json, os, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--emb", required=True)
ap.add_argument("--kind", choices=["judged", "trecdl"], required=True); ap.add_argument("names", nargs="+")
a = ap.parse_args()
for name in a.names:
    rows = list(csv.DictReader(open(os.path.join(a.cand, f"{name}.csv"))))
    if a.kind == "judged":
        de = np.load(os.path.join(a.emb, f"{name}__mpnet_jdocs.npy")); qe = np.load(os.path.join(a.emb, f"{name}__mpnet_q.npy"))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib.util
        spec = importlib.util.spec_from_file_location("bp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "61_build_pool.py"))
        bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
        beir = os.path.join(os.environ["BEIR"], name)
        texts, queries, qrels = bp.load_beir(beir)
        qids = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values())]
        judged = sorted({d for q in qids for d in qrels[q] if d in texts})
        di = {d: i for i, d in enumerate(judged)}; qi = {q: i for i, q in enumerate(qids)}
    else:
        de = np.load(os.path.join(a.emb, f"{name}__mpnet_docs.npy")); qe = np.load(os.path.join(a.emb, f"{name}__mpnet_q.npy"))
        # dl pools: dids sorted judged passages, qids in qrels order with >=1 grade>=2 (as in 64)
        year = name[2:]
        qrels = {}
        for line in open(os.path.join(os.environ["TRECDL"], f"{year}qrels-pass.txt")):
            q, _, p, g = line.split(); qrels.setdefault(q, {})[p] = int(g)
        queries = {l.split("\t")[0]: 1 for l in open(os.path.join(os.environ["TRECDL"], f"msmarco-test{year}-queries.tsv"))}
        qids = [q for q in qrels if q in queries and any(g >= 2 for g in qrels[q].values())]
        texts = {l.split("\t")[0]: 1 for l in open(os.path.join(os.environ["TRECDL"], f"judged_passages_{year}.tsv"))}
        needed = {p for q in qids for p in qrels[q]}; judged = sorted(needed & set(texts))
        di = {d: i for i, d in enumerate(judged)}; qi = {q: i for i, q in enumerate(qids)}
    cos = np.array([float(de[di[r["docid"]]] @ qe[qi[r["qid"]]]) for r in rows])
    # within-query rank percentile
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], []).append(i)
    p_rel = np.zeros(len(rows))
    for q, idx in by_q.items():
        c = cos[idx]; order = np.argsort(-c); ranks = np.empty(len(idx)); ranks[order] = np.arange(len(idx))
        p_rel[idx] = np.where(ranks < 0.3 * len(idx), 0.5 + 0.5 * (1 - ranks / max(len(idx), 1)), 0.49 * (1 - ranks / max(len(idx), 1)))
    with open(os.path.join(a.cand, f"{name}_mpnet.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "docid", "mpnet_cos", "mpnet_p_rel"])
        for r, c, p in zip(rows, cos, p_rel):
            w.writerow([r["qid"], r["docid"], f"{c:.5f}", f"{p:.4f}"])
    g = np.array([int(r["grade"]) for r in rows]); rel = (g >= (2 if a.kind == "trecdl" else 1))
    print(name, "mpnet judge acc", round(float(((p_rel >= 0.5) == rel).mean()), 3))
