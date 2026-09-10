#!/usr/bin/env python3
"""Candidate-pool builder for the legacy and modern retrieval stacks (2027 design §6.1).

Legacy stack = build_candidates.py (shift-study ef825e1d) logic: BM25 + 3 dense
(all-MiniLM-L6-v2, all-mpnet-base-v2, msmarco-MiniLM-L6-cos-v5), per-system
top M=30, features score_norm/rank/consensus/lexoverlap, K0=10 bm25 flag.
Modern stack = the weakest legacy dense system (msmarco-MiniLM) replaced by
Qwen3-Embedding-0.6B (instruction-tuned queries), everything else identical so
the 4-feature semantics (consensus in 1..4) are preserved.

Both stacks additionally carry two *judge-proxy* columns for every pooled pair:
  qwen3e_cos  cosine of Qwen3-Embedding-0.6B (query, doc)
  rr_yes      P(yes) of Qwen3-Reranker-0.6B
These are never used by the legacy 4-feature classifier; they feed the
non-neutral-judge experiments.

Output: <out>/<stack>/runs/candidates/<name>.csv and <name>_meta.csv in the
legacy schema (+ extra columns), so 20_budget_curves.py runs unchanged with
--source <out>/<stack>.

Usage: python3 61_build_pool.py --beir <dir> --out <dir> nfcorpus scifact arguana cqadupstack/android
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch

K0, M = 10, 30
LEGACY_DENSE = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "msmarco": "sentence-transformers/msmarco-MiniLM-L6-cos-v5"}
QWEN_E = "Qwen/Qwen3-Embedding-0.6B"
QWEN_RR = "Qwen/Qwen3-Reranker-0.6B"
Q_INSTR = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
RR_INSTR = "Given a web search query, retrieve relevant passages that answer the query"
STACKS = {"legacy": ["bm25", "minilm", "mpnet", "msmarco"],
          "modern": ["bm25", "minilm", "mpnet", "qwen3e"]}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_beir(path):
    texts, queries, qrels = {}, {}, {}
    for line in open(os.path.join(path, "corpus.jsonl")):
        d = json.loads(line)
        texts[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    for line in open(os.path.join(path, "queries.jsonl")):
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    with open(os.path.join(path, "qrels", "test.tsv")) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for q, d, g in rd:
            qrels.setdefault(q, {})[d] = int(g)
    return texts, queries, qrels


def tset(s):
    return set(s.lower().split())


def dense_topk(qemb, demb, k):
    dt = torch.from_numpy(np.ascontiguousarray(demb)).to(DEVICE)
    qt = torch.from_numpy(qemb).to(DEVICE)
    ii, ss = [], []
    for s0 in range(0, len(qt), 256):
        sims = qt[s0:s0 + 256] @ dt.T
        v, ix = torch.topk(sims, min(k, dt.shape[0]), dim=1)
        ii.extend(ix.cpu().numpy()); ss.extend(v.cpu().numpy())
    return ii, ss


def embed(model_name, texts, cache, prompt="", batch=128, fp16=False):
    if os.path.exists(cache):
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    kw = {"torch_dtype": torch.float16} if fp16 else {}
    m = SentenceTransformer(model_name, device=DEVICE, model_kwargs=kw)
    m.max_seq_length = min(m.max_seq_length or 512, 512)
    if prompt:
        texts = [prompt + t for t in texts]
    e = m.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    np.save(cache, e)
    del m; torch.cuda.empty_cache()
    return e


class Reranker:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(QWEN_RR, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_RR, torch_dtype=torch.float16).to(DEVICE).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes"); self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
                       "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                       "<|im_end|>\n<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    @torch.no_grad()
    def score(self, pairs, batch=128, max_len=384):
        out = np.zeros(len(pairs), np.float32)
        for s0 in range(0, len(pairs), batch):
            chunk = pairs[s0:s0 + batch]
            texts = [self.prefix + f"<Instruct>: {RR_INSTR}\n<Query>: {q}\n<Document>: {d[:2000]}" + self.suffix
                     for q, d in chunk]
            enc = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out[s0:s0 + len(chunk)] = torch.softmax(two, dim=1)[:, 1].cpu().numpy()
            if (s0 // batch) % 100 == 0:
                log(f"    rr {s0}/{len(pairs)}")
        return out


def build(name, beir_root, out_root, emb_dir, reranker):
    tag = name.replace("/", "-")
    outs = {st: os.path.join(out_root, st, "runs", "candidates") for st in STACKS}
    for o in outs.values():
        os.makedirs(o, exist_ok=True)
    if all(os.path.exists(os.path.join(o, f"{tag}.csv")) for o in outs.values()):
        log(f"skip {tag}"); return
    texts, queries, qrels = load_beir(os.path.join(beir_root, name))
    qids = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values())]
    dids = list(texts); doc_texts = [texts[d] for d in dids]; q_texts = [queries[q] for q in qids]
    q_tok = [tset(t) for t in q_texts]
    log(f"=== {tag}: docs={len(dids)} queries={len(qids)}")

    import bm25s
    ret = bm25s.BM25()
    ret.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False), show_progress=False)
    bi, bsc = ret.retrieve(bm25s.tokenize(q_texts, stopwords="en", show_progress=False),
                           k=min(M, len(dids)), show_progress=False)
    sysret = {"bm25": (bi, bsc)}
    for key, mname in LEGACY_DENSE.items():
        demb = embed(mname, doc_texts, os.path.join(emb_dir, f"{tag}__{key}_docs.npy"), batch=256)
        qemb = embed(mname, q_texts, os.path.join(emb_dir, f"{tag}__{key}_q.npy"), batch=256)
        sysret[key] = dense_topk(qemb, demb, M)
        log(f"  {key} done")
    demb_q3 = embed(QWEN_E, doc_texts, os.path.join(emb_dir, f"{tag}__qwen3e_docs.npy"), batch=32, fp16=True)
    qemb_q3 = embed(QWEN_E, q_texts, os.path.join(emb_dir, f"{tag}__qwen3e_q.npy"), prompt=Q_INSTR, batch=32, fp16=True)
    sysret["qwen3e"] = dense_topk(qemb_q3, demb_q3, M)
    log("  qwen3e done")
    did_index = {d: i for i, d in enumerate(dids)}

    # ---- aggregate both stacks, score the union of pairs once with the reranker ----
    per_stack = {}
    for st, systems in STACKS.items():
        rows, meta = [], []
        for qi, qid in enumerate(qids):
            rel = qrels[qid]; G = {d for d, g in rel.items() if g > 0}
            meta.append([qid, len(G), len(rel)])
            bm25_top10 = set(dids[j] for j in sysret["bm25"][0][qi][:K0])
            agg = {}
            for skey in systems:
                idxs, scs = sysret[skey]
                row_i, row_s = idxs[qi], scs[qi]
                top1 = float(row_s[0]) if len(row_s) else 1.0
                for r, (j, sc) in enumerate(zip(row_i, row_s)):
                    d = dids[j]; sn = float(sc) / (abs(top1) + 1e-9)
                    if d not in agg:
                        agg[d] = dict(score_norm=sn, rank=r, consensus=1)
                    else:
                        agg[d]["consensus"] += 1
                        agg[d]["score_norm"] = max(agg[d]["score_norm"], sn)
                        agg[d]["rank"] = min(agg[d]["rank"], r)
            for d, info in agg.items():
                lex = len(q_tok[qi] & tset(texts[d])) / max(len(q_tok[qi]), 1)
                grade = rel.get(d, -1)
                cos = float(qemb_q3[qi] @ demb_q3[did_index[d]])
                rows.append([qid, d, f"{info['score_norm']:.6f}", info["rank"], info["consensus"],
                             f"{lex:.6f}", int(d in bm25_top10), int(d in rel), grade, int(grade > 0),
                             f"{cos:.6f}", None, qi])
        per_stack[st] = (rows, meta)
    union = sorted({(row[-1], row[1]) for rows, _ in per_stack.values() for row in rows})
    log(f"  reranker scoring {len(union)} unique pairs (union of stacks)")
    rr = reranker.score([(q_texts[qi], texts[d]) for qi, d in union])
    rr_map = {k: float(v) for k, v in zip(union, rr)}
    for st, (rows, meta) in per_stack.items():
        for row in rows:
            row[-2] = f"{rr_map[(row[-1], row[1])]:.6f}"
        with open(os.path.join(outs[st], f"{tag}.csv"), "w", newline="") as fc:
            w = csv.writer(fc)
            w.writerow(["qid", "docid", "score_norm", "rank", "consensus", "lexoverlap", "in_bm25top10",
                        "judged", "grade", "relevant", "qwen3e_cos", "rr_yes"])
            w.writerows([row[:-1] for row in rows])
        with open(os.path.join(outs[st], f"{tag}_meta.csv"), "w", newline="") as fm:
            w = csv.writer(fm); w.writerow(["qid", "nG", "n_judged"]); w.writerows(meta)
        log(f"  {st} saved ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default=None); ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    emb_dir = a.emb or os.path.join(a.out, "emb"); os.makedirs(emb_dir, exist_ok=True)
    rr = Reranker()
    for n in a.names:
        build(n, a.beir, a.out, emb_dir, rr)
    log("DONE")
