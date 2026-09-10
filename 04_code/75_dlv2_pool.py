#!/usr/bin/env python3
"""Fully-judged pools for TREC DL 2021/2022/2023 (MS MARCO v2 passage) — LOCK v0.4 §1.

Same pooling rule as 69_judged_pool.py: for each query the NIST-judged passages
form the corpus; BM25 + MiniLM + MPNet + Qwen3-Embedding-0.6B each take top-30;
the union is the pool.  relevant = grade >= 2.  Judge proxies (qwen3e_cos,
rr_yes) attached.  Also writes the merged collection dl212223 and the text /
query side files for 66_llm_judge.py.

Usage: python3 75_dlv2_pool.py --v2 <dir with parts or tar> --out <pools dir>
"""
import argparse, csv, glob, gzip, importlib.util, io, json, os, tarfile, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bp", os.path.join(HERE, "61_build_pool.py"))
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
M = 30
SYSTEMS = ["bm25", "minilm", "mpnet", "qwen3e"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_qrels(fp):
    q = {}
    for line in open(fp):
        p = line.split()
        if len(p) >= 4:
            q.setdefault(p[0], {})[p[2]] = int(p[3])
    return q


def read_queries(fp):
    out = {}
    for l in open(fp):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2:
            out[p[0]] = p[1]
    return out


def extract_passages(tar_path, needed, cache):
    """Scan the v2 tar (jsonl.gz shards) once; keep only needed pids."""
    if os.path.exists(cache):
        return {l.split("\t")[0]: l.rstrip("\n").split("\t", 1)[1] for l in open(cache)}
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.endswith(".gz"):
                continue
            f = tf.extractfile(m)
            with gzip.open(f, "rt", encoding="utf-8") as g:
                for line in g:
                    d = json.loads(line)
                    pid = d["pid"]
                    if pid in needed:
                        out[pid] = d["passage"]
            log(f"  scanned {m.name}: have {len(out)}/{len(needed)}")
            if len(out) == len(needed):
                break
    with open(cache, "w") as f:
        for pid, t in out.items():
            f.write(f"{pid}\t{t.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    return out


def build_year(year, v2, out_root, emb_dir, reranker, texts):
    qrels = read_qrels(os.path.join(v2, f"{year}qrels-pass.txt"))
    queries = read_queries(os.path.join(v2, f"{year}_queries.tsv"))
    qids = [q for q in qrels if q in queries and any(g >= 2 for g in qrels[q].values())]
    judged = sorted({d for q in qids for d in qrels[q] if d in texts})
    di = {d: i for i, d in enumerate(judged)}
    doc_texts = [texts[d] for d in judged]; q_texts = [queries[q] for q in qids]; q_tok = [bp.tset(t) for t in q_texts]
    tag = f"dl{year}"
    log(f"=== {tag}: queries={len(qids)} judged passages={len(judged)}")
    embs = {}
    for key, mname in [("minilm", bp.LEGACY_DENSE["minilm"]), ("mpnet", bp.LEGACY_DENSE["mpnet"])]:
        de = bp.embed(mname, doc_texts, os.path.join(emb_dir, f"{tag}__{key}_jdocs.npy"), batch=256)
        qe = bp.embed(mname, q_texts, os.path.join(emb_dir, f"{tag}__{key}_q.npy"), batch=256)
        embs[key] = (qe, de)
    de = bp.embed(bp.QWEN_E, doc_texts, os.path.join(emb_dir, f"{tag}__qwen3e_jdocs.npy"), batch=32, fp16=True)
    qe = bp.embed(bp.QWEN_E, q_texts, os.path.join(emb_dir, f"{tag}__qwen3e_q.npy"), prompt=bp.Q_INSTR, batch=32, fp16=True)
    embs["qwen3e"] = (qe, de)
    import bm25s
    ret = bm25s.BM25(); ret.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False), show_progress=False)
    rows, meta, pairs = [], [], []
    for qi, qid in enumerate(qids):
        rel = qrels[qid]; pool_all = [d for d in rel if d in di]; pidx = np.array([di[d] for d in pool_all])
        bi, bsc = ret.retrieve(bm25s.tokenize([q_texts[qi]], stopwords="en", show_progress=False), k=len(judged), show_progress=False)
        bm = np.zeros(len(judged)); bm[bi[0]] = bsc[0]
        scores = {"bm25": bm[pidx]}
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
                    agg[d]["consensus"] += 1; agg[d]["score_norm"] = max(agg[d]["score_norm"], sn); agg[d]["rank"] = min(agg[d]["rank"], r)
        bm25_top10 = set(pool_all[j] for j in np.argsort(-scores["bm25"])[:bp.K0])
        G = {d for d in agg if rel.get(d, 0) >= 2}
        meta.append([qid, len(G), len(agg)])
        for d, info in agg.items():
            grade = rel[d]; lex = len(q_tok[qi] & bp.tset(texts[d])) / max(len(q_tok[qi]), 1)
            cos = float(embs["qwen3e"][1][di[d]] @ embs["qwen3e"][0][qi])
            rows.append([qid, d, f"{info['score_norm']:.6f}", info["rank"], info["consensus"], f"{lex:.6f}", int(d in bm25_top10), 1, grade, int(grade >= 2), f"{cos:.6f}", None])
            pairs.append((q_texts[qi], texts[d]))
    log(f"  reranker scoring {len(pairs)} pairs")
    rr = reranker.score(pairs)
    for row, s in zip(rows, rr):
        row[-1] = f"{s:.6f}"
    out = os.path.join(out_root, "dlv2", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{tag}.csv"), "w", newline="") as fc:
        w = csv.writer(fc); w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10", "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"]); w.writerows(rows)
    with open(os.path.join(out, f"{tag}_meta.csv"), "w", newline="") as fm:
        w = csv.writer(fm); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
    with open(os.path.join(out, f"{tag}_queries.tsv"), "w") as f:
        for q in qids:
            f.write(f"{q}\t{queries[q]}\n")
    log(f"  saved {tag}: {len(rows)} rows, pool/query={len(rows)/len(qids):.1f}")
    return rows, meta, qids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="+", default=["2021", "2022", "2023"])
    a = ap.parse_args()
    emb_dir = os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True)
    needed = set()
    for y in a.years:
        qr = read_qrels(os.path.join(a.v2, f"{y}qrels-pass.txt")); needed |= {d for q in qr for d in qr[q]}
    log(f"needed passages: {len(needed)}")
    texts = extract_passages(os.path.join(a.v2, "msmarco_v2_passage.tar"), needed, os.path.join(a.v2, "judged_passages_v2.tsv"))
    rr = bp.Reranker()
    out = os.path.join(a.out, "dlv2", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    all_rows, all_meta = [], []
    for y in a.years:
        rows, meta, _ = build_year(y, a.v2, a.out, emb_dir, rr, texts); all_rows += rows; all_meta += meta
    with open(os.path.join(out, "dl212223.csv"), "w", newline="") as fc:
        w = csv.writer(fc); w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10", "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"]); w.writerows(all_rows)
    with open(os.path.join(out, "dl212223_meta.csv"), "w", newline="") as fm:
        w = csv.writer(fm); w.writerow(["qid", "nG", "n_judged"]); w.writerows(all_meta)
    with open(os.path.join(out, "dl212223_texts.tsv"), "w") as f:
        for pid in {r[1] for r in all_rows}:
            f.write(f"{pid}\t{texts[pid].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    with open(os.path.join(out, "dl212223_queries.tsv"), "w") as f:
        for y in a.years:
            for line in open(os.path.join(out, f"dl{y}_queries.tsv")):
                f.write(line)
    log("DONE")
