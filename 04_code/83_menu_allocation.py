#!/usr/bin/env python3
"""Bounded exploration: document-label allocation for certifying a whole menu (EXPLORATION_menu_allocation.md §3).

Menu of 4 policies (glob, trunc, ad, rr_thresh), precision@cutoff, judge control variate in every arm.
Candidate m̂ from a 20-query pilot. R rounds on fresh queries; within a round every pair shares the labels.
Arms (equal total human-labelled documents, same certificate):
  static_sum   : π ∝ Σ_j |w_j|                         (label sharing, no adaptivity)   — strong baseline
  per_pair     : round queries split 3 ways, each with π ∝ |w_j|, no sharing         — weak baseline
  adaptive     : π ∝ Σ_j λ_j |w_j|, λ_j ∝ 1/max(ε − UCB_j, δ) from previous rounds (floor 0.1)
  oracle       : λ_j from true gaps (upper bound on what adaptivity can give)
Also reports the non-overlap index between decision-document sets of the binding pair and the other pairs.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py")); pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("sb", os.path.join(HERE, "81_sampling_baselines.py")); sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
bc, cert = pv.bc, pv.cert
MENU4 = pv.MENU4; M = 4
ALPHA = 0.10; ARMS = ["static_sum", "per_pair", "adaptive", "oracle"]


def masks_for(s, c, tau_i, th_i):
    n = len(s["sp"]); out = []
    for m in MENU4:
        mk = np.zeros(n, bool)
        if m == "rr_thresh":
            mk = s["rr_sorted"] >= pv.RR_GRID[th_i]
        else:
            mk[:max(pv.cutoffs(s, m, c, tau_i), 1)] = True
        if not mk.any():
            mk[0] = True
        out.append(mk)
    return out


def prec(r, mk):
    return float(r[mk].sum() / mk.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[30, 45, 60, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge; pv.MENU = MENU4
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(41)
    a_sim = ALPHA / (M * (M - 1)); rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); pv.attach_rr(H, data[held], P[held]); N = len(H)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg); pv.attach_rr(HJ, data[held]["judge"], P[held])
        R_true = [np.diff(np.r_[0, s["cumtp"]]).astype(float) for s in H]; R_j = [np.diff(np.r_[0, s["cumtp"]]).astype(float) for s in HJ]
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in ARMS}; overlap = []
                for _ in range(a.draws):
                    perm = rng.permutation(N); tr = perm[:a.n_train]
                    c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr]); th_tr = pv.fit_theta([H[k] for k in tr])
                    MK = [masks_for(H[k], c_tr, tau_tr, th_tr) for k in range(N)]
                    U_tr = np.array([[prec(R_true[k], MK[k][m]) for m in range(M)] for k in tr]); cand = cert.pick_candidate(U_tr)
                    mu = np.array([[prec(R_true[k], MK[k][m]) for m in range(M)] for k in range(N)]).mean(axis=0); regret = float(mu.max() - mu[cand])
                    others = [j for j in range(M) if j != cand]; gaps = {j: float(mu[j] - mu[cand]) for j in others}
                    cost_full = np.array([max(mk.sum() for mk in MK[k]) for k in range(N)], float)
                    B = Bq * float(cost_full.mean()) - float(cost_full[tr].sum())
                    if B <= 0:
                        continue
                    rest = perm[a.n_train:]; chunks = np.array_split(rest, a.rounds)
                    # decision-document non-overlap: binding pair = smallest |gap| among others
                    jb = min(others, key=lambda j: abs(gaps[j]))
                    ov = []
                    for k in rest[:60]:
                        wb = MK[k][jb] ^ MK[k][cand]; wo = np.zeros_like(wb)
                        for j in others:
                            if j != jb:
                                wo |= (MK[k][j] ^ MK[k][cand])
                        ov.append(1 - (wb & wo).sum() / max((wb | wo).sum(), 1))
                    overlap.append(float(np.mean(ov)))
                    W = {}
                    def weights(k, j):
                        if (k, j) not in W:
                            W[(k, j)] = MK[k][j] / MK[k][j].sum() - MK[k][cand] / MK[k][cand].sum()
                        return W[(k, j)]
                    est = {m: {j: [] for j in others} for m in ARMS}; lam = {m: {j: 1.0 for j in others} for m in ARMS}
                    lam["oracle"] = {j: 1.0 / max(eps - gaps[j], 0.002) for j in others}
                    for r_idx, chunk in enumerate(chunks):
                        if len(chunk) == 0:
                            continue
                        b = max(2, int(math.ceil((B / a.rounds) / len(chunk))))
                        groups = np.array_split(chunk, len(others))
                        for m in ARMS:
                            for gi, k in enumerate(chunk):
                                r, rj = R_true[k], R_j[k]; n = len(r)
                                if m == "per_pair":
                                    # this query serves only one pair; b*3 docs so the total label count matches
                                    j_only = others[[i for i, g in enumerate(groups) if k in g][0]]
                                    base = np.abs(weights(k, j_only)); pi = sb.sample_pi(base, min(b * len(others), n), n)
                                    samp = (rng.random(n) < pi) & (pi > 0); w = weights(k, j_only)
                                    est[m][j_only].append(float((w * rj).sum() + (w[samp] * (r[samp] - rj[samp]) / pi[samp]).sum()))
                                    continue
                                lm = lam[m]; base = sum(lm[j] * np.abs(weights(k, j)) for j in others)
                                pi = sb.sample_pi(base, b, n); samp = (rng.random(n) < pi) & (pi > 0)
                                for j in others:
                                    w = weights(k, j)
                                    est[m][j].append(float((w * rj).sum() + (w[samp] * (r[samp] - rj[samp]) / pi[samp]).sum()))
                        # adaptive λ from current UCBs
                        if r_idx < a.rounds - 1:
                            lm = {}
                            for j in others:
                                D = np.array(est["adaptive"][j]); nq = len(D)
                                ucb = D.mean() + cert.t_quantile(1 - a_sim, max(nq - 1, 2)) * D.std(ddof=1) / math.sqrt(max(nq, 2)) if nq > 2 else eps
                                lm[j] = 1.0 / max(eps - ucb, 0.002) if ucb < eps else 1.0 / 0.002
                            mx = max(lm.values()); lam["adaptive"] = {j: max(v / mx, 0.1) for j, v in lm.items()}
                    for m in ARMS:
                        ok = True
                        for j in others:
                            D = np.array(est[m][j]); nq = len(D)
                            if nq < 5 or D.mean() + cert.t_quantile(1 - a_sim, nq - 1) * D.std(ddof=1) / math.sqrt(nq) > eps:
                                ok = False; break
                        if ok:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                for m in ARMS:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws, wrong=cnt[m][1] / a.draws,
                                     nonoverlap=float(np.mean(overlap)) if overlap else float("nan")))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in ARMS) + f" | nonoverlap={np.mean(overlap):.2f} | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "menu_allocation"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"menu_alloc_{a.stack}_{a.judge}.csv"), index=False)


if __name__ == "__main__":
    main()
