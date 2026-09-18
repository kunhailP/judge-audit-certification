#!/usr/bin/env python3
"""Fully-judged candidate pools from deeply judged BEIR collections
(trec-covid, webis-touche2020, dbpedia-entity): the legacy pooling protocol
(BM25 + MiniLM + MPNet + Qwen3-Embedding, per-system top M=30, union) is run
with the *judged documents of each query as the corpus*, so every pooled pair
carries a human grade and judge error can be measured without qrels holes.
relevant = grade >= 1 (BEIR convention).  Also writes <name>_texts.tsv and
<name>_queries.tsv for 66_llm_judge.py.

Usage: python3 69_judged_pool.py --beir <dir> --out <pools dir> trec-covid webis-touche2020 dbpedia-entity
"""
import argparse, csv, importlib.util, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bp", os.path.join(HERE, "61_build_pool.py"))
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
M = 30
SYSTEMS = ["bm25", "minilm", "mpnet", "qwen3e"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build(name, beir_root, out_root, emb_dir, reranker):
    texts, queries, qrels = bp.load_beir(os.path.join(beir_root, name))
    qids = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values())]
    judged = sorted({d for q in qids for d in qrels[q] if d in texts})
    did_index = {d: i for i, d in enumerate(judged)}
    doc_texts = [texts[d] for d in judged]; q_texts = [queries[q] for q in qids]
    q_tok = [bp.tset(t) for t in q_texts]
    log(f"=== {name}: queries={len(qids)} judged docs={len(judged)}")
    embs = {}
    for key, mname in [("minilm", bp.LEGACY_DENSE["minilm"]), ("mpnet", bp.LEGACY_DENSE["mpnet"])]:
        de = bp.embed(mname, doc_texts, os.path.join(emb_dir, f"{name}__{key}_jdocs.npy"), batch=256)
        qe = bp.embed(mname, q_texts, os.path.join(emb_dir, f"{name}__{key}_q.npy"), batch=256)
        embs[key] = (qe, de)
    de = bp.embed(bp.QWEN_E, doc_texts, os.path.join(emb_dir, f"{name}__qwen3e_jdocs.npy"), batch=32, fp16=True)
    qe = bp.embed(bp.QWEN_E, q_texts, os.path.join(emb_dir, f"{name}__qwen3e_q.npy"), prompt=bp.Q_INSTR, batch=32, fp16=True)
    embs["qwen3e"] = (qe, de)
    import bm25s
    ret = bm25s.BM25()
    ret.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False), show_progress=False)
    rows, meta, pairs = [], [], []
    for qi, qid in enumerate(qids):
        rel = qrels[qid]
        pool_all = [d for d in rel if d in did_index]
        pidx = np.array([did_index[d] for d in pool_all])
        scores = {}
        bi, bsc = ret.retrieve(bm25s.tokenize([q_texts[qi]], stopwords="en", show_progress=False),
                               k=len(judged), show_progress=False)
        bm = np.zeros(len(judged)); bm[bi[0]] = bsc[0]; scores["bm25"] = bm[pidx]
        for key, (qe_, de_) in embs.items():
            scores[key] = de_[pidx] @ qe_[qi]
        agg = {}
        for key in SYSTEMS:
            sc = scores[key]; order = np.argsort(-sc)[:M]; top1 = float(sc[order[0]]) if len(order) else 1.0
            for r, j in enumerate(order):
                d = pool_all[j]; sn = float(sc[j]) / (abs(top1) + 1e-9)
                if d not in agg:
                    agg[d] = dict(score_norm=sn, rank=r, consensus=1)
                else:
                    agg[d]["consensus"] += 1; agg[d]["score_norm"] = max(agg[d]["score_norm"], sn)
                    agg[d]["rank"] = min(agg[d]["rank"], r)
        bm25_top10 = set(pool_all[j] for j in np.argsort(-scores["bm25"])[:bp.K0])
        G = {d for d in agg if rel.get(d, 0) > 0}          # nG = relevant docs *within the pooled candidates*
        meta.append([qid, len(G), len(agg)])
        for d, info in agg.items():
            grade = rel[d]
            lex = len(q_tok[qi] & bp.tset(texts[d])) / max(len(q_tok[qi]), 1)
            cos = float(embs["qwen3e"][1][did_index[d]] @ embs["qwen3e"][0][qi])
            rows.append([qid, d, f"{info['score_norm']:.6f}", info["rank"], info["consensus"], f"{lex:.6f}",
                         int(d in bm25_top10), 1, grade, int(grade > 0), f"{cos:.6f}", None])
            pairs.append((q_texts[qi], texts[d]))
    log(f"  reranker scoring {len(pairs)} pairs")
    rr = reranker.score(pairs)
    for row, s in zip(rows, rr):
        row[-1] = f"{s:.6f}"
    out = os.path.join(out_root, "judged", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.csv"), "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10",
                    "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"]); w.writerows(rows)
    with open(os.path.join(out, f"{name}_meta.csv"), "w", newline="") as fm:
        w = csv.writer(fm); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
    pooled_docs = {row[1] for row in rows}
    with open(os.path.join(out, f"{name}_texts.tsv"), "w") as f:
        for d in pooled_docs:
            f.write(f"{d}\t{texts[d].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    with open(os.path.join(out, f"{name}_queries.tsv"), "w") as f:
        for q in qids:
            f.write(f"{q}\t{queries[q].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    log(f"  saved {name}: {len(rows)} rows, pool/query={len(rows)/len(qids):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", required=True); ap.add_argument("--out", required=True); ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    emb_dir = os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True)
    rr = bp.Reranker()
    for n in a.names:
        build(n, a.beir, a.out, emb_dir, rr)
    log("DONE")
