#!/usr/bin/env python3
"""Pre-registered report for LOCK v0.5 (TREC CAsT 2019, MARCO-restricted). Criteria P1-P4, S1-S2. Reports everything."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import beta
HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HUB, "05_results")
def cp_hi(k, n): return beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
def j50(d, conv):
    d = d.sort_values("budget_full_eq"); x = d.budget_full_eq.values * conv; y = d.act.values
    for i in range(len(x)):
        if y[i] >= 0.5: return float(x[0]) if i == 0 else float(x[i-1] + (0.5 - y[i-1]) * (x[i]-x[i-1]) / max(y[i]-y[i-1], 1e-9))
    return float("nan")
out = {}
ai = pd.read_csv(os.path.join(R, "active_inference", "active_cast_llm.csv")); conv = float((ai.docs_labelled / ai.budget_full_eq).mean())
sb = {j: pd.read_csv(os.path.join(R, "sampling_baselines", f"baselines_cast_{j}.csv")) for j in ["llm", "rr", "inv"] if os.path.exists(os.path.join(R, "sampling_baselines", f"baselines_cast_{j}.csv"))}
def J(df, m, e): return j50(df[(df.method == m) & (df.eps == e)], conv)
def maxact(df, m, e): return float(df[(df.method == m) & (df.eps == e)].act.max())
rows = []
for j, df in sb.items():
    for m in ["uniform", "weighted", "strat_pilot", "active_judge", "active_cv", "weighted_cv"]:
        for e in [0.01, 0.02]:
            rows.append(dict(judge=j, method=m, eps=e, J50=J(df, m, e), max_act=maxact(df, m, e), wrong=float(df[(df.method == m) & (df.eps == e)].wrong.max())))
for m in ["ai_calib", "ai_resid", "ai_robust_0.5", "ai_robust_0.3"]:
    for e in [0.01, 0.02]:
        rows.append(dict(judge="llm", method=m, eps=e, J50=J(ai, m, e), max_act=maxact(ai, m, e), wrong=float(ai[(ai.method == m) & (ai.eps == e)].wrong.max())))
tab = pd.DataFrame(rows); pd.set_option("display.width", 200); print(tab.round(3).to_string(index=False))
# P1 validity from boundary runs
p1 = {}
for f in glob.glob(os.path.join(R, "*", "*cast*boundary*.csv")):
    d = pd.read_csv(f)
    for (m, b), g in d.groupby(["method", "budget_full_eq"]):
        k = int(round(g.act.iloc[0] * 300)); p1[f"{m}@{b}"] = dict(type1=float(g.act.iloc[0]), cp_hi=float(cp_hi(k, 300)))
out["P1_validity"] = dict(pass_=all(v["type1"] <= 0.10 and v["cp_hi"] <= 0.15 for v in p1.values()), cells=p1)
w2, u2 = J(sb["llm"], "weighted", 0.02), J(sb["llm"], "uniform", 0.02)
out["P2_unconditional_gain"] = dict(pass_=bool((w2 == w2) and ((u2 != u2) or w2 <= 0.6 * u2)), J50_weighted=w2, J50_uniform=u2)
p3 = {}
for j, df in sb.items():
    a, b = J(df, "weighted_cv", 0.02), J(df, "weighted", 0.02)
    ok = (a <= 1.05 * b) if (a == a and b == b) else (maxact(df, "weighted_cv", 0.02) >= maxact(df, "weighted", 0.02) - 0.05)
    p3[j] = dict(J50_cv=a, J50_weighted=b, pass_=bool(ok))
out["P3_judge_never_hurts"] = dict(pass_=all(v["pass_"] for v in p3.values()), by_judge=p3)
a, b = J(ai, "ai_resid", 0.02), J(sb["llm"], "weighted_cv", 0.02)
out["P4_active_equivalence"] = dict(pass_=bool(a == a and b == b and abs(a - b) <= 0.10 * b), J50_ai_resid=a, J50_weighted_cv=b)
g = pd.read_csv(os.path.join(R, "ppi_gain", "ppi_gain_cast_llm.csv")); rho = float(g[g["T"] == 10].rho.mean())
out["S1_conditional_judge_gain"] = dict(rho_mean=rho, expect_gain=bool(rho >= 0.55), J50_ratio_cv_over_weighted=float(b / w2) if (b == b and w2 == w2) else float("nan"))
c, r = J(ai, "ai_calib", 0.02), a
out["S2_calibrated_rule_worse"] = dict(pass_=bool(c == c and r == r and c >= 1.2 * r) if (c == c and r == r) else None, J50_calib=c, J50_resid=r)
json.dump(out, open(os.path.join(R, "primary_cast", "cast_report.json"), "w"), indent=1, default=str)
print("\nCRITERIA:", json.dumps({k: (v.get("pass_") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1, default=str))
print("P1 cells:", {k: round(v["type1"], 3) for k, v in p1.items()})
print("S1:", out["S1_conditional_judge_gain"])
