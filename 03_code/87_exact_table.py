#!/usr/bin/env python3
"""Appendix E table (exact / finite-population certificates) from the corrected 63 runs
(04_results/planner_v2/{stack}_llm_pop_{t,bet_wor}_ntr20_v3): every split arm is evaluated until IT terminates.

For each collection: ACT and mean stopping look T̄ (T counts ALL audited queries, the 20-query pilot included) for
  split_t          pre-registered t bound on the validation half (superpopulation estimand)
  split_fp (t)     finite-population t: known pilot + FPC on the validation half
  split_fp (bet)   exact WoR betting bound (Waudby-Smith & Ramdas 2020) in the same decomposition
  split_fp_ppi(bet) judge-assisted exact bound (rectifier D - lam*Dhat)
and, for the exact arms, the cumulative ACT by the fraction of the population audited (T/N), so that "fires only once
the whole population is labelled" is visible. Writes 04_results/unified/TABLE_exact_v3.csv and TABLE_exact_v3.tex.
"""
import os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HUB, "04_results", "planner_v2")
COLL = [("antique", "antique", "ANTIQUE"), ("cast", "cast19", "CAsT 2019"), ("judged", "dbpedia-entity", "DBpedia-Entity"), ("dlv2", "dl212223", "TREC DL 21--23")]
FRACS = [0.5, 0.75, 0.9, 1.0]


def arm(d, method):
    g = d[d.method == method]; N = int(g.N.iloc[0])
    act = (g.action == "act").mean(); T = g.final_T.mean(); wrong = g.wrong_cert.mean()
    cum = {f: ((g.action == "act") & (g.final_T <= f * N + 1e-9)).mean() for f in FRACS}
    tmax = int(g.final_T.max())
    return dict(act=act, T=T, wrong=wrong, cum=cum, tmax=tmax, N=N)


def main():
    rows = []; tex = []
    for stack, coll, label in COLL:
        dt = pd.read_parquet(os.path.join(R, f"{stack}_llm_pop_t_ntr20_v3", "planner_v2.parquet"))
        db = pd.read_parquet(os.path.join(R, f"{stack}_llm_pop_bet_wor_ntr20_v3", "planner_v2.parquet"))
        N = {"antique": 200, "cast19": 159, "dbpedia-entity": 399, "dl212223": 211}[coll]
        dt["N"] = N; db["N"] = N
        a_t = arm(dt, "split_t"); a_fp = arm(dt, "split_fp"); a_fpp = arm(dt, "split_fp_ppi")
        b_fp = arm(db, "split_fp"); b_fpp = arm(db, "split_fp_ppi")
        for name, a in [("split_t", a_t), ("split_fp_t", a_fp), ("split_fp_ppi_t", a_fpp), ("split_fp_bet_wor", b_fp), ("split_fp_ppi_bet_wor", b_fpp)]:
            rows.append(dict(collection=coll, N=N, arm=name, act=a["act"], mean_T=a["T"], wrong=a["wrong"], T_max=a["tmax"], frac_max=a["tmax"] / N,
                             **{f"act_by_{int(f*100)}pct": a["cum"][f] for f in FRACS}))
        tex.append(f"{label} ({N}) & ${a_t['act']:.2f}$ ($\\bar T{{=}}{a_t['T']:.0f}$) & ${a_fp['act']:.2f}$ ($\\bar T{{=}}{a_fp['T']:.0f}$) & "
                   f"${a_fpp['act']:.2f}$ ($\\bar T{{=}}{a_fpp['T']:.0f}$) & "
                   f"${b_fp['cum'][0.75]:.2f}$ / ${b_fp['cum'][0.9]:.2f}$ / ${b_fp['act']:.2f}$ ($\\bar T{{=}}{b_fp['T']:.0f}$) & "
                   f"${b_fpp['cum'][0.75]:.2f}$ / ${b_fpp['cum'][0.9]:.2f}$ / ${b_fpp['act']:.2f}$ & {100*b_fp['tmax']/N:.0f}\\%\\\\")
    t = pd.DataFrame(rows); out = os.path.join(HUB, "04_results", "unified"); os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "TABLE_exact_v3.csv"), index=False)
    open(os.path.join(out, "TABLE_exact_v3.tex"), "w").write("\n".join(tex) + "\n")
    pd.set_option("display.width", 250); print(t.round(3).to_string(index=False)); print("\n".join(tex))


if __name__ == "__main__":
    main()
