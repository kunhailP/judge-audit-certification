#!/usr/bin/env python3
"""Pre-registered report for LOCK v0.6 (ANTIQUE). Always-defined criteria on ACT at the largest budget (150), eps=0.02."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import beta
HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HUB, "05_results"); B_MAX = 150
def cp_hi(k, n): return beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
def j50(d, conv):
    d = d.sort_values("budget_full_eq"); x = d.budget_full_eq.values * conv; y = d.act.values
    for i in range(len(x)):
        if y[i] >= 0.5: return float(x[0]) if i == 0 else float(x[i-1] + (0.5 - y[i-1]) * (x[i]-x[i-1]) / max(y[i]-y[i-1], 1e-9))
    return float("nan")
sb = {j: pd.read_csv(os.path.join(R, "sampling_baselines", f"baselines_antique_{j}.csv")) for j in ["llm", "rr", "inv"]}
ai = pd.read_csv(os.path.join(R, "active_inference", "active_antique_llm.csv")); conv = float((ai.docs_labelled / ai.budget_full_eq).mean())
A = lambda df, m, e, b: float(df[(df.method == m) & (df.eps == e) & (df.budget_full_eq == b)].act.iloc[0])
out = {}
# P1
p1 = {}
for f in glob.glob(os.path.join(R, "sampling_baselines", "baselines_antique_*boundary*.csv")):
    d = pd.read_csv(f); tag = os.path.basename(f)
    for (m, b), g in d.groupby(["method", "budget_full_eq"]):
        k = int(round(g.act.iloc[0] * 300)); p1[f"{tag}:{m}@{b}"] = dict(type1=float(g.act.iloc[0]), cp_hi=float(cp_hi(k, 300)))
out["P1"] = dict(pass_=all(v["type1"] <= 0.10 and v["cp_hi"] <= 0.15 for v in p1.values()), worst=max(v["type1"] for v in p1.values()))
# P2
aw, au = A(sb["llm"], "weighted", 0.02, B_MAX), A(sb["llm"], "uniform", 0.02, B_MAX)
out["P2"] = dict(pass_=bool(aw >= 1.5 * au), act_weighted=aw, act_uniform=au, J50_weighted=j50(sb["llm"][(sb["llm"].method == "weighted") & (sb["llm"].eps == 0.02)], conv),
                 J50_uniform=j50(sb["llm"][(sb["llm"].method == "uniform") & (sb["llm"].eps == 0.02)], conv))
# P3
p3 = {}
for j, df in sb.items():
    cells = [(b, A(df, "weighted_cvl", 0.02, b), A(df, "weighted", 0.02, b)) for b in [60, 90, 120, 150]]
    p3[j] = dict(pass_=all(c >= w - 0.03 for _, c, w in cells), cells={b: (round(c, 3), round(w, 3)) for b, c, w in cells},
                 J50_cvl=j50(df[(df.method == "weighted_cvl") & (df.eps == 0.02)], conv), J50_weighted=j50(df[(df.method == "weighted") & (df.eps == 0.02)], conv))
out["P3"] = dict(pass_=all(v["pass_"] for v in p3.values()), by_judge=p3)
# P4
inv = sb["inv"]; cells = [(b, A(inv, "weighted_cvl", 0.02, b), A(inv, "weighted_cv", 0.02, b)) for b in [60, 90, 120, 150]]
out["P4"] = dict(pass_=all(l >= c - 0.01 for _, l, c in cells), cells={b: (round(l, 3), round(c, 3)) for b, l, c in cells},
                 coefficient1_defect_reproduced=bool(A(inv, "weighted_cv", 0.02, B_MAX) < A(inv, "weighted", 0.02, B_MAX) - 0.03))
# P5
ar, ac = A(ai, "ai_resid", 0.02, B_MAX), A(ai, "weighted_cv", 0.02, B_MAX)
out["P5"] = dict(pass_=bool(abs(ar - ac) <= 0.05), act_ai_resid=ar, act_weighted_cv=ac)
# S1, S2
g = pd.read_csv(os.path.join(R, "ppi_gain", "ppi_gain_antique_llm.csv")); rho = float(g[g["T"] == 10].rho.mean())
out["S1"] = dict(rho_mean=rho, increment_cvl_minus_weighted=float(A(sb["llm"], "weighted_cvl", 0.02, B_MAX) - aw), increment_cv=float(A(sb["llm"], "weighted_cv", 0.02, B_MAX) - aw))
fp = os.path.join(R, "primary_antique", "planner_llm", "planner_v2.parquet")
if os.path.exists(fp):
    df = pd.read_parquet(fp); s2 = {}
    for m in ["split_t", "split_ppi", "split_auto"]:
        s = df[df.method == m]; s2[m] = dict(act90=int(((s.action == "act") & (s.final_T <= 90)).sum()), wrong=int(s.wrong_cert.sum()))
    out["S2"] = s2
os.makedirs(os.path.join(R, "primary_antique"), exist_ok=True); json.dump(out, open(os.path.join(R, "primary_antique", "antique_report.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
pd.set_option("display.width", 220)
for j, df in sb.items():
    print(f"\n[{j}] ACT by budget (eps=0.02):"); print(df[df.eps == 0.02].pivot_table(index="budget_full_eq", columns="method", values="act").round(3).to_string())
print("\n[ai] ACT by budget (eps=0.02):"); print(ai[ai.eps == 0.02].pivot_table(index="budget_full_eq", columns="method", values="act").round(3).to_string())
