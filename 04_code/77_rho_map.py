#!/usr/bin/env python3
"""Mechanism map on real data: when does PPI pay?  (design §14.3)

Fixed-budget certificate at T audited queries (train half fits policy params,
validation half certifies; Bonferroni over ordered pairs), on a target
subpopulation of size N_pop = T + N_unlab drawn from the collection.  Three
manipulations: judge noise p (a fraction p of judge relevance labels replaced
by coin flips at the pool base rate -> lowers rho continuously), unlabeled
size N_unlab, and tolerance eps.  Reports rho, ACT rate for humans-only vs
PPI, their ratio, and wrong-certificate rates (truth = collection population).

Usage: python3 77_rho_map.py --pools <dir> --stack judged --names dbpedia-entity --judge llm --train_dir <dir>
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--T", nargs="+", type=int, default=[60, 90]); ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--nunlab", nargs="+", type=int, default=[50, 100, 200, 300]); ap.add_argument("--eps", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(11)
    M = len(MENU); a_sim = ALPHA / (M * (M - 1))
    rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); N = len(H)
        _, tau_star, _ = pv.fit_params(H)
        U_true_pop = pv.utils(H, c_star, tau_star)          # for reference rho only
        base_rate = float(data[held]["judge"]["rel"].mean())
        jd0 = data[held]["judge"]
        for p in a.noise:
            jrel = jd0["rel"].copy()
            flip = rng.random(len(jrel)) < p
            jrel[flip] = (rng.random(flip.sum()) < base_rate).astype(np.int8)
            jnG = {q: int(jrel[s0:s1].sum()) for q, s0, s1 in jd0["slices"]}
            HJ = bc.build_structs({held: dict(X=jd0["X"], rel=jrel, slices=jd0["slices"], nG=jnG)}, held, P[held]); pv.finalize(HJ, reg)
            Uh_pop = pv.utils(HJ, c_star, tau_star)
            rhos = []
            for i in range(M):
                for j in range(i + 1, M):
                    d, dh = U_true_pop[:, i] - U_true_pop[:, j], Uh_pop[:, i] - Uh_pop[:, j]
                    rhos.append(float(np.corrcoef(d, dh)[0, 1]) if d.std() > 0 and dh.std() > 0 else 0.0)
            rho = float(np.mean(rhos))
            for T in a.T:
                for nun in a.nunlab:
                    if T + nun > N:
                        continue
                    for eps in a.eps:
                        act_t = act_p = wr_t = wr_p = 0
                        for _ in range(a.draws):
                            pop = rng.choice(N, T + nun, replace=False); aud = pop[:T]; unl = pop[T:]
                            ntr = T // 2; tr = [H[k] for k in aud[:ntr]]; va = [H[k] for k in aud[ntr:]]
                            c_tr, tau_tr, _ = pv.fit_params(tr)
                            U = pv.utils(va, c_tr, tau_tr); cand = cert.pick_candidate(U)
                            Uh_all = pv.utils([HJ[k] for k in unl], c_tr, tau_tr); Uh_val = pv.utils([HJ[k] for k in aud[ntr:]], c_tr, tau_tr)
                            mu = pv.utils(H, c_tr, tau_tr).mean(axis=0); regret = float(mu.max() - mu[cand])
                            ut = cert.ucb_t(U, cand, a_sim); up = cert.ucb_ppi(U, Uh_val, Uh_all, cand, a_sim)
                            if ut.max() <= eps:
                                act_t += 1; wr_t += regret > eps
                            if up.max() <= eps:
                                act_p += 1; wr_p += regret > eps
                        rows.append(dict(collection=held, judge=a.judge, noise=p, rho=rho, T=T, N_unlab=nun, eps=eps,
                                         act_t=act_t / a.draws, act_ppi=act_p / a.draws, ratio=(act_p / act_t) if act_t else float("nan"),
                                         wrong_t=wr_t / a.draws, wrong_ppi=wr_p / a.draws))
                print(f"{held} noise={p} rho={rho:.2f} T={T} done", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); out = os.path.join(HUB, "05_results", "rho_map"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"rho_map_{a.stack}_{a.judge}.csv"), index=False)
    pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
