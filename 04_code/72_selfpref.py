#!/usr/bin/env python3
"""Self-preference stress test (2027 design §6.3 / assessment priority 2).

Adds a 4th policy to the menu: rr_thresh = return every pooled doc whose
Qwen3-Reranker P(yes) >= theta (theta tuned on the audit sample).  The policy
is built from the *same model family* as the AI judges (Qwen3-Reranker-0.6B
judge = identical model; Qwen3-8B judge = same family).  For every policy
pair we report rho, in/out-band judge error, and the fixed-budget PPI gain,
split by whether the pair involves rr_thresh.  If judge bias is policy-
correlated, pairs involving rr_thresh should show lower rho / larger in-band
error than pairs among the three classifier-based policies, at equal judge
accuracy.

Usage: python3 72_selfpref.py --pools <dir> --stack judged --names ... --judge rr|llm --train_dir <dir>
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
bc, cert = pv.bc, pv.cert
MENU4 = pv.MENU4


def cut_set(s, name, c, tau_i, th_i):
    """Boolean mask over the struct's docs (descending-P order) selected by the policy."""
    n = len(s["sp"]); m = np.zeros(n, bool)
    if name == "rr_thresh":
        return s["rr_sorted"] >= pv.RR_GRID[th_i]
    k = pv.cutoffs(s, name, c, tau_i); m[:k] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="rr")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--T", type=int, default=30); ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(3)
    rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); pv.attach_rr(H, data[held], P[held])
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg); pv.attach_rr(HJ, data[held]["judge"], P[held])
        _, tau_i, _ = pv.fit_params(H); th_i = pv.fit_theta(H)
        U = pv.utils(H, c_star, tau_i, th_i); Uh = pv.utils(HJ, c_star, tau_i, th_i); N = len(U)
        acc = float(((data[held]["rr"] >= pv.JUDGE_THR).astype(int) == data[held]["rel"]).mean())
        mu = U.mean(0)
        for ia in range(4):
            for ib in range(ia + 1, 4):
                d, dh = U[:, ia] - U[:, ib], Uh[:, ia] - Uh[:, ib]
                rho = float(np.corrcoef(d, dh)[0, 1]) if d.std() > 0 and dh.std() > 0 else float("nan")
                inb, outb = [], []
                for s, sj in zip(H, HJ):
                    ma, mb = cut_set(s, MENU4[ia], c_star, tau_i, th_i), cut_set(s, MENU4[ib], c_star, tau_i, th_i)
                    band = ma ^ mb
                    err = np.diff(np.r_[0, s["cumtp"]]) != np.diff(np.r_[0, sj["cumtp"]])
                    inb.extend(err[band].tolist()); outb.extend(err[~band].tolist())
                # fixed-budget PPI gain with cross-fit lambda
                se_h, se_p = [], []
                for _ in range(a.draws):
                    idx = rng.choice(N, a.T, replace=False); dS, dhS = d[idx], dh[idx]
                    se_h.append(dS.std(ddof=1) / math.sqrt(a.T))
                    lam = cert.crossfit_lambda(dS, dhS); resid = dS - lam * dhS
                    se_p.append(math.sqrt(lam.mean() ** 2 * dh.var(ddof=1) / N + resid.var(ddof=1) / a.T))
                # judge bias on the *level* of each policy: E[u_hat - u]
                rows.append(dict(collection=held, judge=a.judge, judge_acc=acc, pair=f"{MENU4[ia]}_vs_{MENU4[ib]}",
                                 involves_rr=("rr_thresh" in (MENU4[ia], MENU4[ib])), rho=rho,
                                 err_in=float(np.mean(inb)) if inb else float("nan"), err_out=float(np.mean(outb)) if outb else float("nan"),
                                 band_frac=len(inb) / max(len(inb) + len(outb), 1),
                                 bias_diff=float((dh - d).mean()), true_gap=float(d.mean()),
                                 ess_gain=float((np.mean(se_h) / np.mean(se_p)) ** 2), N=N,
                                 mu_a=float(mu[ia]), mu_b=float(mu[ib])))
    import pandas as pd
    df = pd.DataFrame(rows); out = os.path.join(HUB, "05_results", "selfpref"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"selfpref_{a.stack}_{a.judge}.csv"), index=False)
    pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))
    g = df.groupby("involves_rr")[["rho", "err_in", "err_out", "ess_gain"]].mean()
    print("\nmean by involves_rr:\n", g.round(3))


if __name__ == "__main__":
    main()
