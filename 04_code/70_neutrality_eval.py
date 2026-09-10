#!/usr/bin/env python3
"""Evaluate the judge-neutrality diagnostic on real pools (2027 design §3.2).

For each collection and policy pair: draw pilots of n0 audited queries (300
draws), run lib/neutrality.diagnose, and report
  P(use_judge)        how often the diagnostic recommends PPI
  cover_gain          P(gain_lcb <= true population gain 1/(1-rho^2))
  T_ppi/T_human       median projected budget ratio
  actual ratio        (SE_human/SE_ppi)^-2 measured at T=30 (from 67 logic)
so the diagnostic can be judged on validity (coverage >= 1-alpha) and on
whether its recommendation matches the realised gain.

Usage: python3 70_neutrality_eval.py --pools <dir> --stack trecdl --names dl1920 --judge llm
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("nt", os.path.join(HERE, "lib", "neutrality.py"))
nt = importlib.util.module_from_spec(spec); spec.loader.exec_module(nt)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA, EPS = 0.10, 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="rr")
    ap.add_argument("--pilots", nargs="+", type=int, default=[10, 20, 30]); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--method", default="bca", choices=["fisher", "boot", "bca"])
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(7)
    rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg)
        _, tau_i, _ = pv.fit_params(H)
        U = pv.utils(H, c_star, tau_i); Uh = pv.utils(HJ, c_star, tau_i); N = len(U)
        for ia in range(len(MENU)):
            for ib in range(ia + 1, len(MENU)):
                d, dh = U[:, ia] - U[:, ib], Uh[:, ia] - Uh[:, ib]
                rho = float(np.corrcoef(d, dh)[0, 1]); true_gain = 1 / (1 - rho ** 2)
                for n0 in a.pilots:
                    use, cover, ratio, tp, th = [], [], [], [], []
                    for _ in range(a.draws):
                        idx = rng.choice(N, n0, replace=False)
                        r = nt.diagnose(d[idx], dh[idx], EPS, ALPHA, N_unlabeled=N, rng=rng, method=a.method)
                        use.append(r["use_judge"]); cover.append(r["gain_lcb"] <= true_gain + 1e-9)
                        if np.isfinite(r["T_ppi"]) and r["T_human"] > 0:
                            ratio.append(r["T_ppi"] / r["T_human"]); tp.append(r["T_ppi"]); th.append(r["T_human"])
                    rows.append(dict(collection=held, judge=a.judge, pair=f"{MENU[ia]}_vs_{MENU[ib]}", n0=n0, n_queries=N,
                                     rho=rho, true_gain=true_gain, p_use_judge=float(np.mean(use)),
                                     cover_gain_lcb=float(np.mean(cover)), med_T_ratio=float(np.median(ratio)) if ratio else float("nan"),
                                     med_T_human=float(np.median(th)) if th else float("nan"),
                                     med_T_ppi=float(np.median(tp)) if tp else float("nan")))
    import pandas as pd
    df = pd.DataFrame(rows); out = os.path.join(HUB, "05_results", "neutrality"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"neutrality_{a.stack}_{a.judge}_{a.method}.csv"), index=False)
    pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
