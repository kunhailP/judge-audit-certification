"""
88_pilot_sweep.py -- Track A: separate the candidate-selection effect from the sampling-design effect.

Reads the 83 outputs produced with the population estimand (tag _v3b*):
  * pilot sweep  n_train in {10, 20, 40, 80}, candidate chosen on the pilot (fixed_cand=pilot)
  * fixed candidate (fixed_cand=best): every draw certifies the true best policy, so regret = 0 and the only
    remaining obstacles are the decision margin, residual variance and the pilot cost.

Three measurements per (collection, eps, n_train, arm):
  1. candidate quality      : feasible_frac = P(regret(m_hat) <= eps),  median slack eps - regret
  2. certification given a feasible candidate : max over budgets of act_feasible
  3. total-cost efficiency  : J50_total = unique labelled documents (pilot included) needed to reach ACT 0.5
     (linear interpolation on the budget grid; 'n.r.' if not reached)
Writes 04_results/unified/PILOT_SWEEP_v3.csv and PILOT_SWEEP_v3.md (+ the fixed-candidate comparison).
"""
import glob, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "04_results", "menu_allocation"); OUT = os.path.join(ROOT, "04_results", "unified")
ARMS = ["per_pair", "static_sum", "oracle_exact", "plugin_exact"]
NAME = {"antique": "ANTIQUE", "cast19": "CAsT", "dbpedia-entity": "DBpedia", "dl212223": "DL 21-23"}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "menu_alloc_*_llm_v2_full_t_v3[bc]*.csv"))):
        if "_diag" in f or "_smoke" in f:
            continue
        d = pd.read_csv(f)
        if "n_train" not in d.columns:            # the original v3b runs (pilot 20, candidate from the pilot)
            d["n_train"] = 20; d["fixed_cand"] = "pilot"
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return d[d.method.isin(ARMS)].drop_duplicates(["collection", "eps", "n_train", "fixed_cand", "method", "budget_full_eq"])


def j50(g, xcol="docs_labelled", target=0.5):
    g = g.sort_values(xcol); x = g[xcol].to_numpy(float); y = g.act.to_numpy(float)
    if y.max() < target:
        return np.nan
    i = int(np.argmax(y >= target))
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / max(y[i] - y[i - 1], 1e-12))


def main():
    d = load(); recs = []
    for (c, eps, ntr, fc, m), g in d.groupby(["collection", "eps", "n_train", "fixed_cand", "method"]):
        recs.append(dict(collection=c, eps=eps, n_train=ntr, fixed_cand=fc, method=m,
                         feasible_frac=g.feasible_frac.mean(), slack_med=g.slack_med.mean(), slack_q25=g.slack_q25.mean(),
                         act_feasible_max=g.act_feasible.max(), act_max=g.act.max(), wrong_max=g.wrong.max(),
                         docs_pilot=g.docs_pilot.mean(), J50_total=j50(g), J50_sampling=j50(g, "docs_sampling"),
                         budgets=len(g)))
    R = pd.DataFrame(recs).sort_values(["collection", "eps", "fixed_cand", "method", "n_train"])
    os.makedirs(OUT, exist_ok=True); R.to_csv(os.path.join(OUT, "PILOT_SWEEP_v3.csv"), index=False)

    L = ["# Pilot sweep and fixed-candidate decomposition (population estimand, pilot charged in full, t bound)\n",
         "Source: 04_results/menu_allocation/menu_alloc_*_v3b*[_ntr*|_candbest].csv (300 draws per budget).\n"]
    fmt = lambda v: "n.r." if pd.isna(v) else f"{v:,.0f}"
    for eps in sorted(R.eps.unique()):
        L.append(f"\n## eps = {eps}\n")
        L.append("### 1. Candidate quality vs pilot size (candidate chosen on the pilot; identical across arms)\n")
        L.append("| Collection | n_train | P(regret <= eps) | median slack | slack q25 | pilot docs |\n|---|---:|---:|---:|---:|---:|")
        for c in NAME:
            for ntr in [10, 20, 40, 80]:
                s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == ntr) & (R.fixed_cand == "pilot") & (R.method == "static_sum")]
                if len(s):
                    s = s.iloc[0]
                    L.append(f"| {NAME[c]} | {ntr} | {s.feasible_frac:.2f} | {s.slack_med:.3f} | {s.slack_q25:.3f} | {s.docs_pilot:,.0f} |")
        L.append("\n### 2. Certification given a feasible candidate (max over budgets of ACT | regret <= eps)\n")
        L.append("| Collection | arm | " + " | ".join(f"pilot {n}" for n in [10, 20, 40, 80]) + " | fixed best (pilot 20) |\n|---|---|" + "---:|" * 5)
        for c in NAME:
            for m in ARMS:
                cells = []
                for ntr in [10, 20, 40, 80]:
                    s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == ntr) & (R.fixed_cand == "pilot") & (R.method == m)]
                    cells.append(f"{s.iloc[0].act_feasible_max:.2f}" if len(s) else "-")
                s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == 20) & (R.fixed_cand == "best") & (R.method == m)]
                cells.append(f"{s.iloc[0].act_max:.2f}" if len(s) else "-")
                L.append(f"| {NAME[c]} | {m} | " + " | ".join(cells) + " |")
        L.append("\n### 3. Total unique documents (pilot included) to reach ACT 0.5\n")
        L.append("| Collection | arm | " + " | ".join(f"pilot {n}" for n in [10, 20, 40, 80]) + " | fixed best (pilot 20) |\n|---|---|" + "---:|" * 5)
        for c in NAME:
            for m in ARMS:
                cells = []
                for ntr in [10, 20, 40, 80]:
                    s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == ntr) & (R.fixed_cand == "pilot") & (R.method == m)]
                    cells.append(fmt(s.iloc[0].J50_total) if len(s) else "-")
                s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == 20) & (R.fixed_cand == "best") & (R.method == m)]
                cells.append(fmt(s.iloc[0].J50_total) if len(s) else "-")
                L.append(f"| {NAME[c]} | {m} | " + " | ".join(cells) + " |")
        L.append("\n### 3b. Sampling documents only (pilot excluded) to reach ACT 0.5\n")
        L.append("| Collection | arm | " + " | ".join(f"pilot {n}" for n in [10, 20, 40, 80]) + " | fixed best (pilot 20) |\n|---|---|" + "---:|" * 5)
        for c in NAME:
            for m in ARMS:
                cells = []
                for ntr in [10, 20, 40, 80]:
                    s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == ntr) & (R.fixed_cand == "pilot") & (R.method == m)]
                    cells.append(fmt(s.iloc[0].J50_sampling) if len(s) else "-")
                s = R[(R.collection == c) & (R.eps == eps) & (R.n_train == 20) & (R.fixed_cand == "best") & (R.method == m)]
                cells.append(fmt(s.iloc[0].J50_sampling) if len(s) else "-")
                L.append(f"| {NAME[c]} | {m} | " + " | ".join(cells) + " |")
        w = R[(R.eps == eps)].wrong_max.max()
        L.append(f"\nMax wrong-certificate rate over all cells at eps={eps}: {w:.3f}")
    # ---- pre-registered predictions (02_data/PREREG_pilot_sweep_eps003_2026-09-15.md) on the held-out eps = 0.03 ----
    H = R[(R.eps == 0.03) & (R.fixed_cand == "pilot")]
    if len(H):
        L.append("\n## Held-out check (eps = 0.03), predictions P1-P3 of 02_data/PREREG_pilot_sweep_eps003_2026-09-15.md\n")
        L.append("| Collection | arm | J50 p10/p20 | J50 p40/p20 | J50 p80/p40 | ACT|feasible min over pilots | oracle saving vs static_sum (pilot 20) | P1 | P2 | P3 |\n|---|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|")
        for c in NAME:
            for m in ["static_sum", "oracle_exact", "plugin_exact"]:
                g = H[(H.collection == c) & (H.method == m)].set_index("n_train")
                if not {10, 20, 40, 80} <= set(g.index):
                    continue
                r10, r40, r80 = g.J50_total[10] / g.J50_total[20], g.J50_total[40] / g.J50_total[20], g.J50_total[80] / g.J50_total[40]
                p1 = (r40 > 1) and (r80 > 1) and (0.85 <= r10 <= 1.15)
                af = g.act_feasible_max.min(); p2 = (af >= 0.95) if c != "dbpedia-entity" else None
                st = H[(H.collection == c) & (H.method == "static_sum") & (H.n_train == 20)].J50_total.iloc[0]
                orc = H[(H.collection == c) & (H.method == "oracle_exact") & (H.n_train == 20)].J50_total.iloc[0]
                sav = 1 - orc / st; p3 = sav <= 0.15
                yn = lambda v: "-" if v is None else ("yes" if v else "NO")
                L.append(f"| {NAME[c]} | {m} | {r10:.2f} | {r40:.2f} | {r80:.2f} | {af:.2f} | {sav:+.1%} | {yn(p1)} | {yn(p2)} | {yn(p3)} |")
    open(os.path.join(OUT, "PILOT_SWEEP_v3.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
