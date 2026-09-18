#!/usr/bin/env python3
"""LLM relevance judge (Qwen3-8B, non-thinking) over a candidate pool.

UMBRELA-style 0-3 graded prompt (Upadhyay et al. 2024).  For every pooled
(query, passage) pair we read the next-token logits over {"0","1","2","3"} and
store the expected grade and P(grade >= 2).  Output side file:
<cand_dir>/<name>_llm.csv with qid, docid, llm_grade, llm_p_rel.

Usage: python3 66_llm_judge.py --cand <dir> --texts <judged_passages.tsv> --queries <tsv> dl2019 dl2020
"""
import argparse, csv, os, time
import numpy as np
import torch

MODEL = "Qwen/Qwen3-8B"          # overridden by --model; side-file prefix by --tag
PROMPT = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but does not answer it,
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--texts", required=True)
    ap.add_argument("--queries", nargs="+", required=True); ap.add_argument("names", nargs="+")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max_chars", type=int, default=2500)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--tag", default="llm")
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    # every vocabulary token that decodes (after stripping) to a single digit 0-3, so that
    # tokenizers with space-prefixed digits (Llama/Mistral) are handled like Qwen's
    grade_ids = [[] for _ in range(4)]
    for tid in range(len(tok)):
        t = tok.decode([tid]).strip()
        if t in ("0", "1", "2", "3"):
            grade_ids[int(t)].append(tid)
    print("grade token ids:", [len(g) for g in grade_ids], flush=True)
    texts = {}
    for fp in a.texts.split(","):
        for l in open(fp):
            pid, t = l.rstrip("\n").split("\t", 1); texts[pid] = t
    queries = {}
    for fp in a.queries:
        for l in open(fp):
            q, t = l.rstrip("\n").split("\t", 1); queries[q] = t
    for name in a.names:
        fp = os.path.join(a.cand, f"{name}.csv")
        rows = list(csv.DictReader(open(fp)))
        prompts = []
        for r in rows:
            msg = PROMPT.format(query=queries[r["qid"]], passage=texts[r["docid"]][:a.max_chars])
            try:
                chat = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                chat = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
            prompts.append(chat + "##final score: ")
        log(f"{name}: {len(prompts)} pairs")
        out = np.zeros((len(prompts), 4), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(prompts), a.batch):
                enc = tok(prompts[s0:s0 + a.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")
                # left padding: give real tokens positions 0..L-1 (a plain forward would use
                # arange over the padded width and shift RoPE for heavily padded rows)
                pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
                lg = torch.stack([torch.logsumexp(logits[:, ids], dim=1) for ids in grade_ids], dim=1)
                out[s0:s0 + len(enc["input_ids"])] = torch.softmax(lg, dim=1).cpu().numpy()
                if (s0 // a.batch) % 50 == 0:
                    log(f"  {s0}/{len(prompts)}")
        exp_grade = out @ np.arange(4); p_rel = out[:, 2:].sum(1)
        with open(os.path.join(a.cand, f"{name}_{a.tag}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["qid", "docid", f"{a.tag}_grade", f"{a.tag}_p_rel", f"{a.tag}_argmax"])
            for r, eg, pr, am in zip(rows, exp_grade, p_rel, out.argmax(1)):
                w.writerow([r["qid"], r["docid"], f"{eg:.4f}", f"{pr:.4f}", int(am)])
        g = np.array([int(r["grade"]) for r in rows]); rel = g >= 2; pred = p_rel >= 0.5
        log(f"{name}: acc={(pred == rel).mean():.3f} corr(exp_grade, grade)={np.corrcoef(exp_grade, g)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
