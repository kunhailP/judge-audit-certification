#!/usr/bin/env python3
"""Fully-judged candidate pools from TREC Deep Learning 2019/2020 passage qrels
(2027 design §6.2: the testbed where AI-judge error can be measured without
qrels holes).

Pool(q) = every passage NIST judged for q (depth-10 pooling of TREC runs,
~200 passages/query).  Systems (BM25 over the union of judged passages,
MiniLM, MPNet, msmarco-MiniLM, Qwen3-Embedding-0.6B) score every pooled
passage; features follow the legacy schema (score_norm = score / top-1 in
pool, rank within pool, consensus = #systems ranking it in their top-30,
lexoverlap).  relevant = grade >= 2 (TREC DL passage convention).  Judge
proxies qwen3e_cos and rr_yes are attached to every pair.

Usage: python3 64_trecdl_pool.py --trecdl <dir with qrels/queries/collection.tar.gz> --out <pools dir>
Writes <out>/trecdl/runs/candidates/dl2019.csv, dl2020.csv (+ _meta.csv).
"""
import argparse, csv, importlib.util, os, tarfile, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bp", os.path.join(HERE, "61_build_pool.py"))
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
M = 30


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_qrels(fp):
    q = {}
    for line in open(fp):
        qid, _, pid, g = line.split()
        q.setdefault(qid, {})[pid] = int(g)
    return q


def read_queries(fp):
    return {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open(fp)}


def load_passages(tar_path, needed, cache):
    if os.path.exists(cache):
        return {l.split("\t")[0]: l.rstrip("\n").split("\t", 1)[1] for l in open(cache)}
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.endswith("collection.tsv"):
                continue
            f = tf.extractfile(m)
            for line in f:
                pid, text = line.decode("utf-8").rstrip("\n").split("\t", 1)
                if pid in needed:
                    out[pid] = text
    with open(cache, "w") as f:
        for pid, t in out.items():
            f.write(f"{pid}\t{t}\n")
    return out


def build_year(year, trecdl, out_root, emb_dir, reranker):
    qrels = read_qrels(os.path.join(trecdl, f"{year}qrels-pass.txt"))
    queries = read_queries(os.path.join(trecdl, f"msmarco-test{year}-queries.tsv"))
    qids = [q for q in qrels if q in queries and any(g >= 2 for g in qrels[q].values())]
    needed = {p for q in qids for p in qrels[q]}
    texts = load_passages(os.path.join(trecdl, "collection.tar.gz"), needed,
                          os.path.join(trecdl, f"judged_passages_{year}.tsv"))
    dids = sorted(needed & set(texts)); did_index = {d: i for i, d in enumerate(dids)}
    doc_texts = [texts[d] for d in dids]; q_texts = [queries[q] for q in qids]
    q_tok = [bp.tset(t) for t in q_texts]
    log(f"=== dl{year}: queries={len(qids)} judged passages={len(dids)}")
    tag = f"dl{year}"
    # dense embeddings over the judged union
    embs = {}
    for key, mname in bp.LEGACY_DENSE.items():
        de = bp.embed(mname, doc_texts, os.path.join(emb_dir, f"{tag}__{key}_docs.npy"), batch=256)
        qe = bp.embed(mname, q_texts, os.path.join(emb_dir, f"{tag}__{key}_q.npy"), batch=256)
        embs[key] = (qe, de)
    de = bp.embed(bp.QWEN_E, doc_texts, os.path.join(emb_dir, f"{tag}__qwen3e_docs.npy"), batch=32, fp16=True)
    qe = bp.embed(bp.QWEN_E, q_texts, os.path.join(emb_dir, f"{tag}__qwen3e_q.npy"), prompt=bp.Q_INSTR, batch=32, fp16=True)
    embs["qwen3e"] = (qe, de)
    import bm25s
    ret = bm25s.BM25()
    ret.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False), show_progress=False)
    rows, meta, pairs = [], [], []
    for qi, qid in enumerate(qids):
        pool = [p for p in qrels[qid] if p in did_index]
        pidx = np.array([did_index[p] for p in pool])
        rel = qrels[qid]; G = {p for p, g in rel.items() if g >= 2}
        meta.append([qid, len(G), len(rel)])
        scores = {}
        bi, bsc = ret.retrieve(bm25s.tokenize([q_texts[qi]], stopwords="en", show_progress=False),
                               k=len(dids), show_progress=False)
        bm = np.zeros(len(dids)); bm[bi[0]] = bsc[0]
        scores["bm25"] = bm[pidx]
        for key, (qe_, de_) in embs.items():
            scores[key] = de_[pidx] @ qe_[qi]
        agg = {p: dict(score_norm=-1e9, rank=10 ** 6, consensus=0) for p in pool}
        for key, sc in scores.items():
            order = np.argsort(-sc); top1 = float(sc[order[0]]) if len(order) else 1.0
            for r, j in enumerate(order):
                p = pool[j]; sn = float(sc[j]) / (abs(top1) + 1e-9)
                agg[p]["score_norm"] = max(agg[p]["score_norm"], sn)
                agg[p]["rank"] = min(agg[p]["rank"], r)
                if r < M:
                    agg[p]["consensus"] += 1
        bm25_top10 = set(pool[j] for j in np.argsort(-scores["bm25"])[:bp.K0])
        for p in pool:
            info = agg[p]; grade = rel[p]
            lex = len(q_tok[qi] & bp.tset(texts[p])) / max(len(q_tok[qi]), 1)
            cos = float(embs["qwen3e"][1][did_index[p]] @ embs["qwen3e"][0][qi])
            rows.append([qid, p, f"{info['score_norm']:.6f}", info["rank"], max(info["consensus"], 1),
                         f"{lex:.6f}", int(p in bm25_top10), 1, grade, int(grade >= 2), f"{cos:.6f}", None])
            pairs.append((q_texts[qi], texts[p]))
    log(f"  reranker scoring {len(pairs)} pairs")
    rr = reranker.score(pairs)
    for row, s in zip(rows, rr):
        row[-1] = f"{s:.6f}"
    out = os.path.join(out_root, "trecdl", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{tag}.csv"), "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10",
                    "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"])
        w.writerows(rows)
    with open(os.path.join(out, f"{tag}_meta.csv"), "w", newline="") as fm:
        w = csv.writer(fm); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
    log(f"  saved {tag}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trecdl", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="+", default=["2019", "2020"])
    a = ap.parse_args()
    emb_dir = os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True)
    rr = bp.Reranker()
    for y in a.years:
        build_year(y, a.trecdl, a.out, emb_dir, rr)
    log("DONE")
