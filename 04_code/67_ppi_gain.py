#!/usr/bin/env python3
"""Fixed-budget measurement of the prediction-powered gain for *paired policy
differences* on fully judged pools (2027 design §3.1 / Theorem A check).

For each collection, policy pair and human budget T: draw T audited queries
at random (500 draws), compute the standard error of the paired mean
difference from humans only vs. PPI++ (judge utilities on all queries, human
rectifier on the T).  Report effective-sample-size gain (SE_h / SE_ppi)^2,
alongside the population rho between true and judge paired differences and
the judge's global accuracy.  Theorem A predicts gain -> 1/(1 - rho^2) as
T grows, independent of global accuracy.

Usage: python3 67_ppi_gain.py --pools <dir> --stack trecdl --names dl2019 dl2020 [--judge rr|llm]
"""
import argparse, importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="rr")
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--lambda_finite_n", action="store_true")
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(1)
    rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg)
        _, tau_i, _ = pv.fit_params(H)
        U = pv.utils(H, c_star, tau_i); Uh = pv.utils(HJ, c_star, tau_i)
        N = len(U)
        acc = float(((data[held]["rr"] >= pv.JUDGE_THR).astype(int) == data[held]["rel"]).mean())
        for ia in range(len(MENU)):
            for ib in range(ia + 1, len(MENU)):
                d, dh = U[:, ia] - U[:, ib], Uh[:, ia] - Uh[:, ib]
                rho = float(np.corrcoef(d, dh)[0, 1]) if d.std() > 0 and dh.std() > 0 else float("nan")
                for T in a.budgets:
                    if T >= N:
                        continue
                    se_h, se_p, lam_s = [], [], []
                    for _ in range(a.draws):
                        idx = rng.choice(N, T, replace=False)
                        dS, dhS = d[idx], dh[idx]
                        se_h.append(dS.std(ddof=1) / math.sqrt(T))
                        mask = np.ones(N, bool); mask[idx] = False; dh_out = dh[mask]      # unlabeled = outside the audited draw
                        lam_vec = cert.crossfit_lambda(dS, dhS, n_over_N=(T / len(dh_out) if a.lambda_finite_n else 0.0)); lam = float(lam_vec.mean())
                        resid = dS - lam_vec * dhS
                        se_p.append(math.sqrt(lam ** 2 * dh_out.var(ddof=1) / len(dh_out) + resid.var(ddof=1) / T))
                        lam_s.append(lam)
                    rows.append(dict(collection=held, judge=a.judge, pair=f"{MENU[ia]}_vs_{MENU[ib]}", T=T,
                                     n_queries=N, judge_acc=acc, rho=rho, true_gap=float(d.mean()),
                                     se_human=float(np.mean(se_h)), se_ppi=float(np.mean(se_p)),
                                     ess_gain=float((np.mean(se_h) / np.mean(se_p)) ** 2),
                                     theory_gain=float(1 / (1 - rho ** 2)) if rho == rho else float("nan"),
                                     lam_mean=float(np.mean(lam_s))))
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(HUB, "05_results", "ppi_gain"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"ppi_gain_{a.stack}_{a.judge}{'_lamN' if a.lambda_finite_n else ''}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
