#!/usr/bin/env python3
"""LOCK v0.5 target: TREC CAsT 2019, restricted to MS MARCO v1 passages (local corpus), one query per judged turn.
Queries = manually resolved utterances (evaluation_topics_annotated_resolved_v1.0.tsv). relevant = grade >= 2.
Pool rule identical to 69/75: judged MARCO passages of the turn are the corpus; BM25 + MiniLM + MPNet + Qwen3-Embedding
top-30 union. Judge proxies qwen3e_cos / rr_yes attached. Writes <out>/cast/runs/candidates/cast19.csv (+meta, texts, queries).
"""
import argparse, csv, importlib.util, os, tarfile, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bp", os.path.join(HERE, "61_build_pool.py")); bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
spec = importlib.util.spec_from_file_location("p64", os.path.join(HERE, "64_trecdl_pool.py")); p64 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p64)
M = 30; SYSTEMS = ["bm25", "minilm", "mpnet", "qwen3e"]
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cast", required=True); ap.add_argument("--marco_tar", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qrels = {}
    for l in open(os.path.join(a.cast, "2019qrels.txt")):
        p = l.split()
        if len(p) >= 4 and p[2].startswith("MARCO_"):
            qrels.setdefault(p[0], {})[p[2][6:]] = int(p[3])
    queries = {}
    for l in open(os.path.join(a.cast, "evaluation_topics_annotated_resolved_v1.0.tsv")):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2: queries[p[0]] = p[1]
    qids = [q for q in qrels if q in queries and any(g >= 2 for g in qrels[q].values())]
    needed = {d for q in qids for d in qrels[q]}
    texts = p64.load_passages(a.marco_tar, needed, os.path.join(a.cast, "judged_passages_cast19.tsv"))
    judged = sorted(needed & set(texts)); di = {d: i for i, d in enumerate(judged)}
    doc_texts = [texts[d] for d in judged]; q_texts = [queries[q] for q in qids]; q_tok = [bp.tset(t) for t in q_texts]
    log(f"=== cast19: turns={len(qids)} judged MARCO passages={len(judged)}")
    emb_dir = os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True); embs = {}
    for key, mname in [("minilm", bp.LEGACY_DENSE["minilm"]), ("mpnet", bp.LEGACY_DENSE["mpnet"])]:
        embs[key] = (bp.embed(mname, q_texts, os.path.join(emb_dir, f"cast19__{key}_q.npy"), batch=256), bp.embed(mname, doc_texts, os.path.join(emb_dir, f"cast19__{key}_jdocs.npy"), batch=256))
    embs["qwen3e"] = (bp.embed(bp.QWEN_E, q_texts, os.path.join(emb_dir, "cast19__qwen3e_q.npy"), prompt=bp.Q_INSTR, batch=32, fp16=True),
                      bp.embed(bp.QWEN_E, doc_texts, os.path.join(emb_dir, "cast19__qwen3e_jdocs.npy"), batch=32, fp16=True))
    import bm25s; ret = bm25s.BM25(); ret.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False), show_progress=False)
    rows, meta, pairs = [], [], []
    for qi, qid in enumerate(qids):
        rel = qrels[qid]; pool_all = [d for d in rel if d in di]; pidx = np.array([di[d] for d in pool_all])
        bi, bsc = ret.retrieve(bm25s.tokenize([q_texts[qi]], stopwords="en", show_progress=False), k=len(judged), show_progress=False)
        bm = np.zeros(len(judged)); bm[bi[0]] = bsc[0]; scores = {"bm25": bm[pidx]}
        for key, (qe, de) in embs.items(): scores[key] = de[pidx] @ qe[qi]
        agg = {}
        for key in SYSTEMS:
            sc = scores[key]; order = np.argsort(-sc)[:M]; top1 = float(sc[order[0]]) if len(order) else 1.0
            for r, j in enumerate(order):
                d = pool_all[j]; sn = float(sc[j]) / (abs(top1) + 1e-9)
                if d not in agg: agg[d] = dict(score_norm=sn, rank=r, consensus=1)
                else: agg[d]["consensus"] += 1; agg[d]["score_norm"] = max(agg[d]["score_norm"], sn); agg[d]["rank"] = min(agg[d]["rank"], r)
        bm25_top10 = set(pool_all[j] for j in np.argsort(-scores["bm25"])[:bp.K0]); G = {d for d in agg if rel.get(d, 0) >= 2}
        meta.append([qid, len(G), len(agg)])
        for d, info in agg.items():
            grade = rel[d]; lex = len(q_tok[qi] & bp.tset(texts[d])) / max(len(q_tok[qi]), 1); cos = float(embs["qwen3e"][1][di[d]] @ embs["qwen3e"][0][qi])
            rows.append([qid, d, f"{info['score_norm']:.6f}", info["rank"], info["consensus"], f"{lex:.6f}", int(d in bm25_top10), 1, grade, int(grade >= 2), f"{cos:.6f}", None]); pairs.append((q_texts[qi], texts[d]))
    log(f"  reranker scoring {len(pairs)} pairs"); rr = bp.Reranker().score(pairs)
    for row, s in zip(rows, rr): row[-1] = f"{s:.6f}"
    out = os.path.join(a.out, "cast", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cast19.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10", "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"]); w.writerows(rows)
    with open(os.path.join(out, "cast19_meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
    with open(os.path.join(out, "cast19_texts.tsv"), "w") as f:
        for d in {r[1] for r in rows}: f.write(f"{d}\t{' '.join(texts[d].splitlines()).replace(chr(9), ' ')}\n")
    with open(os.path.join(out, "cast19_queries.tsv"), "w") as f:
        for q in qids: f.write(f"{q}\t{queries[q]}\n")
    log(f"  saved cast19: {len(rows)} rows, pool/turn={len(rows)/len(qids):.1f}")

if __name__ == "__main__":
    main()
