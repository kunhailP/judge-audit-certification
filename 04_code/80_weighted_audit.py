#!/usr/bin/env python3
"""Decision-weighted document-level auditing — four arms at equal document-label budget.

Utility: precision at the policy's cutoff (label-independent normaliser, so Prop. C(b) weights are exact).
For a query q and pair (j, cand): D(q) = Σ_d w(d) r(d) with w from Prop. C.  Arms (all human budget = B docs):
  human_full     : label every doc up to max cutoff on n_full queries; query-level t certificate
  human_weighted : on n_w queries, label b docs per query sampled ∝ |w(d)| (Poisson sampling, HT estimate D̂_HT)
  judge_ppi      : human_full labels on n_full queries + judge everywhere else; query-level PPI (Prop. B′)
  weighted_cv    : as human_weighted but with the judge as document-level control variate (D̂_CV)
Candidate is chosen on a small fully labelled train set (shared by all arms, cost counted).  Validity check: wrong
certificate iff population regret > eps.  Reports ACT and wrong per arm, per (B, eps).

Usage: python3 80_weighted_audit.py --pools <dir> --stack judged --names dbpedia-entity --judge llm --train_dir <dir>
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10


def arrays(H, HJ, c, tau_i):
    out = []
    for s, sj in zip(H, HJ):
        r = np.diff(np.r_[0, s["cumtp"]]).astype(float); rj = np.diff(np.r_[0, sj["cumtp"]]).astype(float)
        ks = [max(pv.cutoffs(s, m, c, tau_i), 1) for m in MENU]
        out.append((r, rj, ks))
    return out


def weights(ks, ia, ib, n):
    """w(d) for D = u_a - u_b with precision at cutoff."""
    ka, kb = ks[ia], ks[ib]; w = np.zeros(n)
    w[:min(ka, kb)] = 1.0 / ka - 1.0 / kb
    if ka > kb:
        w[kb:ka] = 1.0 / ka
    elif kb > ka:
        w[ka:kb] = -1.0 / kb
    return w


def prec_utils(arr):
    return np.array([[r[:k].sum() / k for k in ks] for r, _, ks in arr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[30, 60, 90], help="budget expressed in fully-labelled-query equivalents")
    ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02, 0.05]); ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--docs_per_query", type=int, default=4); ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(29)
    M = len(MENU); a_sim = ALPHA / (M * (M - 1)); rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); N = len(H)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg)
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in ["human_full", "human_weighted", "judge_ppi", "weighted_cv"]}
                for _ in range(a.draws):
                    perm = rng.permutation(N)
                    tr = perm[:a.n_train]; c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr])
                    arr = arrays(H, HJ, c_tr, tau_tr)
                    cost_full = np.array([max(ks) for _, _, ks in arr], float)          # docs to label for exact D on that query
                    U_tr = prec_utils([arr[k] for k in tr]); cand = cert.pick_candidate(U_tr)
                    mu = prec_utils(arr).mean(axis=0); regret = float(mu.max() - mu[cand])
                    others = [j for j in range(M) if j != cand]
                    B = Bq * float(cost_full.mean()) - float(cost_full[tr].sum())        # remaining doc budget after the train set
                    if B <= 0:
                        continue
                    rest = perm[a.n_train:]
                    # ---- human_full: fully label queries in order until budget exhausted ----
                    cum = np.cumsum(cost_full[rest]); n_full = int((cum <= B).sum())
                    q_full = rest[:n_full]
                    if n_full >= 5:
                        U = prec_utils([arr[k] for k in q_full]); ut = cert.ucb_t(U, cand, a_sim)
                        if ut.max() <= eps:
                            cnt["human_full"][0] += 1; cnt["human_full"][1] += regret > eps
                        # ---- judge_ppi: same labels + judge on remaining queries (query-level PPI, Prop. B') ----
                        q_unl = rest[n_full:]
                        Uj = prec_utils([(rj, rj, ks) for r, rj, ks in [arr[k] for k in q_unl]])
                        Ujl = prec_utils([(rj, rj, ks) for r, rj, ks in [arr[k] for k in q_full]])
                        up = cert.ucb_ppi(U, Ujl, Uj, cand, a_sim)
                        if up.max() <= eps:
                            cnt["judge_ppi"][0] += 1; cnt["judge_ppi"][1] += regret > eps
                    # ---- weighted arms: b docs per query sampled ∝ |w| (Poisson), across as many queries as the budget allows ----
                    # spend the whole budget: if b docs/query would run out of queries, raise b so that B is used
                    b = max(a.docs_per_query, int(math.ceil(B / len(rest)))); n_w = int(B // b); q_w = rest[:min(n_w, len(rest))]
                    if len(q_w) >= 5:
                        ok_ht = ok_cv = True
                        for j in others:
                            Dht, Dcv = [], []
                            for k in q_w:
                                r, rj, ks = arr[k]; w = weights(ks, j, cand, len(r)); aw = np.abs(w)
                                if aw.sum() == 0:
                                    Dht.append(0.0); Dcv.append(float((w * rj).sum())); continue
                                pi = np.minimum(1.0, b * aw / aw.sum())
                                # redistribute mass capped at 1 so that E[#labels] stays ~b (two passes suffice in practice)
                                for _ in range(2):
                                    free = pi < 1
                                    if free.any() and pi.sum() < min(b, len(r)):
                                        pi[free] = np.minimum(1.0, pi[free] * (min(b, len(r)) - pi[~free].sum()) / max(pi[free].sum(), 1e-9))
                                samp = rng.random(len(r)) < pi
                                Dht.append(float((w[samp] * r[samp] / pi[samp]).sum()))
                                Dcv.append(float((w * rj).sum() + (w[samp] * (r[samp] - rj[samp]) / pi[samp]).sum()))
                            for name, D in [("human_weighted", np.array(Dht)), ("weighted_cv", np.array(Dcv))]:
                                n = len(D); ucb = D.mean() + cert.t_quantile(1 - a_sim, n - 1) * D.std(ddof=1) / math.sqrt(n)
                                if ucb > eps:
                                    if name == "human_weighted": ok_ht = False
                                    else: ok_cv = False
                        if ok_ht:
                            cnt["human_weighted"][0] += 1; cnt["human_weighted"][1] += regret > eps
                        if ok_cv:
                            cnt["weighted_cv"][0] += 1; cnt["weighted_cv"][1] += regret > eps
                for m, (act, wr) in cnt.items():
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=act / a.draws, wrong=wr / a.draws))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={v[0]/a.draws:.2f}/{v[1]/a.draws:.3f}" for m, v in cnt.items()), flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "weighted_audit"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"weighted_audit_{a.stack}_{a.judge}_b{a.docs_per_query}.csv"), index=False)


if __name__ == "__main__":
    main()
