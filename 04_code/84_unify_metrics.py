#!/usr/bin/env python3
"""Unified efficiency metric for the manuscript.

  J50 := number of human-judged (query, document) pairs — pilot / training queries included — at which the
         cumulative ACT rate first reaches 50%, under the same error control (alpha = 0.10, Bonferroni over
         looks x ordered pairs; wrong-certificate rate reported alongside).  Linear interpolation between the
         budgets actually run; "not reached" if the largest budget run stays below 50%.  No ratios are formed
         from "not reached" cells.

Block A (query-level audits, set-F1, one audited query = its whole pool): planner_v2 runs (split_t / split_ppi /
  split_auto), docs = T x mean pool size of the collection.
Block B (document-level audits, precision@cutoff): sampling baselines (81), active inference (82), weighted audit (80),
  docs = budget_full_eq x mean(max cutoff) of the collection (identical accounting inside those scripts; the
  docs_labelled column of 82 gives the per-collection conversion).
Block C (4-policy menu, precision): menu allocation (83).
Writes 06_paper/TABLES_v0.1.md and 05_results/unified/J50.csv.
"""
import glob, json, os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "05_results"); OUT = os.path.join(R, "unified"); os.makedirs(OUT, exist_ok=True)
LOOKS = [10, 30, 50, 70, 90, 120, 150, 190]
POOL = {}   # mean pool size per collection (set-F1 audit cost per query)
for f in glob.glob(os.path.join(R, "..", "..", "..", "pools", "*", "runs", "candidates", "*_meta.csv")):
    pass


def interp_first(xs, ys, target=0.5):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs)):
        if ys[i] >= target:
            if i == 0:
                return float(xs[0])
            return float(xs[i - 1] + (target - ys[i - 1]) * (xs[i] - xs[i - 1]) / max(ys[i] - ys[i - 1], 1e-9))
    return float("nan")


rows = []
# ---------- Block A: query-level (set-F1) ----------
pool_mean = {"dbpedia-entity": 20573 / 399, "dl212223": 14637 / 211, "trec-covid": 4165 / 50, "webis-touche2020": 1858 / 49}
blockA = [("dbpedia-entity", "Qwen3-8B", os.path.join(R, "planner_v2", "judged_llm_auto_excl", "planner_v2.parquet")),
          ("dbpedia-entity", "Mistral-7B", os.path.join(R, "planner_v2", "judged_mistral_ext", "planner_v2.parquet")),
          ("dbpedia-entity", "Qwen3-Reranker", os.path.join(R, "planner_v2", "judged_rr_auto", "planner_v2.parquet")),
          ("dbpedia-entity", "inverted", os.path.join(R, "planner_v2", "judged_inv_auto", "planner_v2.parquet")),
          ("dl212223", "Qwen3-8B", os.path.join(R, "primary_dlv2", "planner_llm", "planner_v2.parquet")),
          ("dl212223", "Qwen3-Reranker", os.path.join(R, "primary_dlv2", "planner_rr", "planner_v2.parquet")),
          ("dl212223", "Mistral-7B", os.path.join(R, "primary_dlv2", "planner_mistral", "planner_v2.parquet"))]
for coll, judge, fp in blockA:
    if not os.path.exists(fp):
        continue
    df = pd.read_parquet(fp); df = df[df.collection == coll]
    for m in ["split_t", "split_ppi", "split_auto"]:
        s = df[df.method == m]; a = s[s.action == "act"]; n = len(s)
        looks = [T for T in LOOKS if (s.final_T <= T).any() or T <= s.final_T.max()]
        curve = [(a.final_T <= T).sum() / n for T in looks]
        j50_T = interp_first(looks, curve); docs = j50_T * pool_mean[coll] if j50_T == j50_T else float("nan")
        rows.append(dict(block="A query-level, set-F1, eps=0.01", collection=coll, judge=judge if m != "split_t" else "—", method=m,
                         eps=0.01, J50_docs=docs, J50_queries=j50_T, wrong_rate=float(s.wrong_cert.mean()), n=n,
                         max_budget_docs=float(max(looks) * pool_mean[coll]), max_act=float(max(curve))))
# ---------- Block B: document-level (precision) ----------
conv = {}
for f in glob.glob(os.path.join(R, "active_inference", "active_*.csv")):
    d = pd.read_csv(f)
    for c, g in d.groupby("collection"):
        conv[c] = float((g.docs_labelled / g.budget_full_eq).mean())
def add_block(pattern, block, arms, judge_of):
    for f in glob.glob(pattern):
        d = pd.read_csv(f)
        for (c, j, e), g in d.groupby(["collection", "judge", "eps"]):
            for m in arms:
                s = g[g.method == m].sort_values("budget_full_eq")
                if s.empty:
                    continue
                docs = s.budget_full_eq.values * conv.get(c, np.nan)
                j50 = interp_first(docs, s.act.values)
                rows.append(dict(block=block, collection=c, judge=judge_of(j, m), method=m, eps=e, J50_docs=j50, J50_queries=np.nan,
                                 wrong_rate=float(s.wrong.max()), n=300, max_budget_docs=float(docs.max()), max_act=float(s.act.max())))
jname = {"llm": "Qwen3-8B", "rr": "Qwen3-Reranker", "mistral": "Mistral-7B", "inv": "inverted"}
add_block(os.path.join(R, "sampling_baselines", "baselines_*.csv"), "B doc-level, precision", ["uniform", "weighted", "strat_pilot", "weighted_cv"],
          lambda j, m: jname.get(j, j) if m.endswith("_cv") else "—")
add_block(os.path.join(R, "active_inference", "active_*.csv"), "B doc-level, precision", ["ai_calib", "ai_resid", "ai_robust_0.5"], lambda j, m: jname.get(j, j))
add_block(os.path.join(R, "menu_allocation", "menu_alloc_*.csv"), "C 4-policy menu, precision", ["static_sum", "per_pair", "adaptive", "oracle"], lambda j, m: jname.get(j, j))
# ---------- Block D: document-level, set-F1 (linearised / plug-in) ----------
for f in glob.glob(os.path.join(R, "f1_weighted", "f1_weighted_*.csv")):
    if "forceworst" in f or "boundary" in f:
        continue
    d = pd.read_csv(f)
    for (c, j, e), g in d.groupby(["collection", "judge", "eps"]):
        for m in ["human_full", "weighted_plugin", "judge_ppi", "weighted_lin_cv", "weighted_lin_cv_bc"]:
            s_ = g[g.method == m].sort_values("budget_full_eq")
            if s_.empty:
                continue
            docs = s_.docs.values; j50 = interp_first(docs, s_.act.values)
            rows.append(dict(block="D doc-level, set-F1 (linearised; see caveat)", collection=c, judge=jname.get(j, j) if m not in ("human_full", "weighted_plugin") else "—",
                             method=m, eps=e, J50_docs=j50, J50_queries=np.nan, wrong_rate=float(s_.wrong.max()), n=300,
                             max_budget_docs=float(docs.max()), max_act=float(s_.act.max())))
df = pd.DataFrame(rows).drop_duplicates(subset=["block", "collection", "judge", "method", "eps"]); df.to_csv(os.path.join(OUT, "J50.csv"), index=False)

# ---------- manuscript table ----------
def fmt(x):
    return "미도달" if x != x else f"{x:,.0f}"
lines = ["# Tables v0.1 — unified efficiency metric J50 (2026-09-10)", "",
         "**J50** = 같은 오류 통제(α = 0.10, look × 순서쌍 Bonferroni) 아래 누적 ACT율이 처음 50%에 이르는 시점의 **실제 인간 판정 (query, document) 쌍 수**",
         "(pilot·학습 query 포함, 실행한 예산 사이는 선형 보간, 최대 예산에서도 50% 미만이면 '미도달'). 잘못된 인증률은 같은 실행의 P(ACT ∧ regret > ε).",
         "블록 A(query 단위, set-F1, 한 query 감사 = pool 전체 판정)와 블록 B·C(문서 단위, precision@cutoff)는 utility가 달라 블록 간 J50을 직접 비교하지 않는다.", ""]
for block, g in df.groupby("block", sort=True):
    lines += [f"## {block}", ""]
    for (c, e), gg in g.groupby(["collection", "eps"]):
        lines += [f"**{c}, ε = {e}**", "", "| method | 판정자 | J50 (판정 쌍) | 최대 예산 (판정 쌍) | 최대 ACT | wrong |", "|---|---|---:|---:|---:|---:|"]
        for _, r in gg.sort_values(["method", "judge"]).iterrows():
            lines.append(f"| {r.method} | {r.judge} | {fmt(r.J50_docs)} | {r.max_budget_docs:,.0f} | {r.max_act:.2f} | {r.wrong_rate:.3f} |")
        lines.append("")
open(os.path.join(HUB, "06_paper", "TABLES_v0.1.md"), "w").write("\n".join(lines))
pd.set_option("display.width", 220)
print(df[["block", "collection", "judge", "method", "eps", "J50_docs", "max_act", "wrong_rate"]].round(3).to_string(index=False))
