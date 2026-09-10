#!/usr/bin/env python3
"""Pre-registered primary-run report for LOCK v0.4 (TREC DL 2021-2023, dl212223).

Criteria (LOCK v0.4 §3):
  Primary : for split_t, split_ppi, split_auto — Clopper–Pearson 95% upper bound of wrong-cert rate <= 0.15
            and point estimate <= alpha = 0.10.
  Secondary: cumulative ACT at T=190: split_ppi >= 1.2 x split_t ; split_auto >= split_t.
  Rule     : split_auto adopts the inverted (adversarial) judge in <= 5% of ACTs.
Reports every number regardless of outcome.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import beta

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HUB, "05_results", "primary_dlv2")
LOOKS = [10, 30, 50, 70, 90, 120, 150, 190]; ALPHA = 0.10


def cp(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


out = {"criteria": {}, "tables": {}}
for judge in ["llm", "rr", "inv"]:
    fp = os.path.join(ROOT, f"planner_{judge}", "planner_v2.parquet")
    if not os.path.exists(fp):
        print(f"[{judge}] missing"); continue
    df = pd.read_parquet(fp); rows = []
    for m in ["split_t", "split_ppi", "split_auto", "loo_boot", "loo_sim", "recal_ep", "recal_bpx"]:
        s = df[df.method == m]; a = s[s.action == "act"]; w = int(s.wrong_cert.sum()); n = len(s); lo, hi = cp(w, n)
        used = float(s.used_judge.dropna().astype(bool).mean()) if m == "split_auto" and s.used_judge.notna().any() else float("nan")
        rows.append(dict(judge=judge, method=m, cum_act=[int((a.final_T <= T).sum()) for T in LOOKS], act_190=int((a.final_T <= 190).sum()),
                         wrong=w, n=n, wrong_rate=w / n, cp_lo=lo, cp_hi=hi, judge_used=used))
    t = pd.DataFrame(rows); out["tables"][judge] = t.to_dict("records")
    pd.set_option("display.width", 220); print(f"\n=== PRIMARY dl212223, judge={judge}, {n} repeats, looks {LOOKS}")
    print(t[["method", "cum_act", "wrong", "wrong_rate", "cp_lo", "cp_hi", "judge_used"]].round(3).to_string(index=False))
    r = {x["method"]: x for x in rows}
    if judge == "llm":
        prim = all(r[m]["cp_hi"] <= 0.15 and r[m]["wrong_rate"] <= ALPHA for m in ["split_t", "split_ppi", "split_auto"])
        sec_ppi = r["split_ppi"]["act_190"] >= 1.2 * r["split_t"]["act_190"]; sec_auto = r["split_auto"]["act_190"] >= r["split_t"]["act_190"]
        out["criteria"].update(primary_validity=bool(prim), secondary_ppi_gain=bool(sec_ppi), secondary_auto=bool(sec_auto),
                               act_190=dict(split_t=r["split_t"]["act_190"], split_ppi=r["split_ppi"]["act_190"], split_auto=r["split_auto"]["act_190"]))
    if judge == "inv":
        out["criteria"]["rule_rejects_adversarial"] = bool((r["split_auto"]["judge_used"] if r["split_auto"]["judge_used"] == r["split_auto"]["judge_used"] else 0.0) <= 0.05)
        out["criteria"]["inv_judge_used"] = r["split_auto"]["judge_used"]
        out["criteria"]["primary_validity_inv"] = bool(all(r[m]["cp_hi"] <= 0.15 and r[m]["wrong_rate"] <= ALPHA for m in ["split_t", "split_ppi", "split_auto"]))
json.dump(out, open(os.path.join(ROOT, "primary_report.json"), "w"), indent=2, default=str)
print("\nCRITERIA:", json.dumps(out["criteria"], indent=1, default=str))
