#!/usr/bin/env python3
"""LOCK v0.7 target: NeuCLIRBench, monolingual English setting (105 topics with mlir qrels; documents = English machine
translations of the Persian/Russian/Chinese collections). relevant = gain >= 1 (the collection's gains are {0, 1, 3}).
Pool rule identical to 64/75/87/89: the judged documents of the topic are the corpus; BM25 + MiniLM + MPNet + Qwen3-Embedding
top-30 union; features score_norm/rank/consensus/lexoverlap; Qwen3-Reranker P(yes) as `rr_yes`.
Inputs: --bench <dir with data/news.eng.tsv, data/qrels.mlir.gains.txt>  --texts <judged_texts.tsv: docid \\t text>
Writes <out>/neuclir/runs/candidates/neuclir_eng.csv (+_meta, _texts.tsv, _queries.tsv), like the other fully judged pools."""
import argparse, csv, importlib.util, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bp", os.path.join(HERE, "61_build_pool.py")); bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
M = 30; SYSTEMS = ["bm25", "minilm", "mpnet", "qwen3e"]; REL_MIN = 1; NAME = "neuclir_eng"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); ap.add_argument("--texts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=4000, help="truncate document text for embedding/BM25 (news articles are long)")
    a = ap.parse_args()
    qrels = {}
    for l in open(os.path.join(a.bench, "data", "qrels.mlir.gains.txt")):
        p = l.split()
        if len(p) >= 4: qrels.setdefault(p[0], {})[p[2]] = int(p[3])
    queries = {}
    for l in open(os.path.join(a.bench, "data", "news.eng.tsv"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t", 1)
        if len(p) >= 2: queries[p[0]] = p[1]
    qids = sorted([q for q in qrels if q in queries and any(g >= REL_MIN for g in qrels[q].values())], key=int)
    needed = {d for q in qids for d in qrels[q]}; texts = {}
    for l in open(a.texts, encoding="utf-8", errors="ignore"):
        p = l.rstrip("\n").split("\t", 1)
        if len(p) == 2 and p[0] in needed: texts[p[0]] = p[1][:a.max_chars]
    judged = sorted(needed & set(texts)); di = {d: i for i, d in enumerate(judged)}
    doc_texts = [texts[d] for d in judged]; q_texts = [queries[q] for q in qids]; q_tok = [bp.tset(t) for t in q_texts]
    log(f"=== {NAME}: queries={len(qids)} judged docs with text={len(judged)} of {len(needed)}")
    emb_dir = os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True); embs = {}
    for key, mname in [("minilm", bp.LEGACY_DENSE["minilm"]), ("mpnet", bp.LEGACY_DENSE["mpnet"])]:
        embs[key] = (bp.embed(mname, q_texts, os.path.join(emb_dir, f"{NAME}__{key}_q.npy"), batch=256), bp.embed(mname, doc_texts, os.path.join(emb_dir, f"{NAME}__{key}_jdocs.npy"), batch=256))
        log(f"  embedded {key}")
    embs["qwen3e"] = (bp.embed(bp.QWEN_E, q_texts, os.path.join(emb_dir, f"{NAME}__qwen3e_q.npy"), prompt=bp.Q_INSTR, batch=32, fp16=True),
                      bp.embed(bp.QWEN_E, doc_texts, os.path.join(emb_dir, f"{NAME}__qwen3e_jdocs.npy"), batch=32, fp16=True))
    log("  embedded qwen3e")
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
        bm25_top10 = set(pool_all[j] for j in np.argsort(-scores["bm25"])[:bp.K0]); G = {d for d in agg if rel.get(d, 0) >= REL_MIN}
        meta.append([qid, len(G), len(agg)])
        for d, info in agg.items():
            grade = rel[d]; lex = len(q_tok[qi] & bp.tset(texts[d])) / max(len(q_tok[qi]), 1); cos = float(embs["qwen3e"][1][di[d]] @ embs["qwen3e"][0][qi])
            rows.append([qid, d, f"{info['score_norm']:.6f}", info["rank"], info["consensus"], f"{lex:.6f}", int(d in bm25_top10), 1, grade, int(grade >= REL_MIN), f"{cos:.6f}", None]); pairs.append((q_texts[qi], texts[d]))
    log(f"  reranker scoring {len(pairs)} pairs"); rr = bp.Reranker().score(pairs)
    for row, s in zip(rows, rr): row[-1] = f"{s:.6f}"
    out = os.path.join(a.out, "neuclir", "runs", "candidates"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{NAME}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10", "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"]); w.writerows(rows)
    with open(os.path.join(out, f"{NAME}_meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
    with open(os.path.join(out, f"{NAME}_texts.tsv"), "w", encoding="utf-8") as f:
        for d in {r[1] for r in rows}: f.write(f"{d}\t{texts[d].replace(chr(9), ' ')}\n")
    with open(os.path.join(out, f"{NAME}_queries.tsv"), "w", encoding="utf-8") as f:
        for q in qids: f.write(f"{q}\t{queries[q]}\n")
    log(f"  saved {NAME}: {len(rows)} rows, pool/query={len(rows)/len(qids):.1f}, relevant/query={np.mean([m[1] for m in meta]):.1f}")


if __name__ == "__main__":
    main()
