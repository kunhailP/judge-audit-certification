#!/usr/bin/env python3
"""Faithful Active Statistical Inference comparison for the estimand D(q) = Σ_d w_d y_d (§18 follow-up).

For the prediction-powered / active estimator  D̂ = Σ_d w_d ĵ_d + Σ_{d sampled} w_d (y_d − ĵ_d)/π_d ,
Var(D̂) = Σ_d w_d² E[(y_d − ĵ_d)²] (1 − π_d)/π_d, so the budget-optimal rule is
      π_d ∝ |w_d| · sqrt( E[(y_d − ĵ_d)²] )        (Zrnic & Candès 2024, adapted to the weighted estimand).
Arms (all: same pilot of 20 fully labelled queries — used for candidate choice AND for the residual model —,
same expected number of human-labelled documents B, same t certificate over ordered pairs):
  weighted_cv   : π ∝ |w|                                         + judge control variate      (current best)
  ai_calib      : π ∝ |w|·sqrt(ĵ(1−ĵ))                             + CV   (assumes a calibrated judge; previous 'active_cv')
  ai_resid      : π ∝ |w|·sqrt(r̂(ĵ)), r̂ = pilot-estimated E[(y−ĵ)²] in 10 judge-probability bins   + CV
  ai_robust_0.5 : π = 0.5·π_ai_resid + 0.5·π_weighted            + CV   (Robust Sampling-style mixture)
  ai_robust_0.3 : π = 0.7·π_ai_resid + 0.3·π_weighted            + CV
Reports ACT, wrong, mean number of human-labelled docs (pilot + sampled), per-query estimator variance.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py")); pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("wa", os.path.join(HERE, "80_weighted_audit.py")); wa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wa)
spec = importlib.util.spec_from_file_location("sb", os.path.join(HERE, "81_sampling_baselines.py")); sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10
ARMS = ["weighted_cv", "ai_calib", "ai_resid", "ai_robust_0.5", "ai_robust_0.3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[30, 45, 60, 75, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--docs_per_query", type=int, default=4); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--boundary", action="store_true", help="validity stress: candidate := runner-up, eps := 0.9 x its true regret")
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(37)
    M = len(MENU); a_sim = ALPHA / (M * (M - 1)); rows = []
    bins = np.linspace(0, 1, 11)
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); N = len(H)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg)
        pj_all = data[held]["rr"]; prob = {}
        for s, (q, s0, s1) in zip(H, data[held]["slices"]):
            p = P[held][s0:s1]; o = np.argsort(-p); prob[s["qid"]] = np.clip(pj_all[s0:s1][o], 1e-3, 1 - 1e-3)
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in ARMS}; varr = {m: [] for m in ARMS}; labels = {m: [] for m in ARMS}
                for _ in range(a.draws):
                    perm = rng.permutation(N); tr = perm[:a.n_train]
                    c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr]); arr = wa.arrays(H, HJ, c_tr, tau_tr)
                    cost_full = np.array([max(ks) for _, _, ks in arr], float)
                    U_tr = wa.prec_utils([arr[k] for k in tr]); cand = cert.pick_candidate(U_tr)
                    mu = wa.prec_utils(arr).mean(axis=0)
                    if a.boundary:
                        cand = int(np.argsort(-mu)[1]); eps = 0.9 * float(mu.max() - mu[cand])
                    regret = float(mu.max() - mu[cand]); others = [j for j in range(M) if j != cand]
                    pilot_docs = float(cost_full[tr].sum()); B = Bq * float(cost_full.mean()) - pilot_docs
                    if B <= 0:
                        continue
                    # residual model from the pilot (labels on docs up to max cutoff): E[(y - j)^2] per judge-probability bin
                    py, pjv = [], []
                    for k in tr:
                        r, rj, ks = arr[k]; kmax = max(ks); pj = prob[H[k]["qid"]][:kmax]; py += list(r[:kmax]); pjv += list(pj)
                    py, pjv = np.array(py), np.array(pjv); res = (py - pjv) ** 2
                    idx = np.clip(np.digitize(pjv, bins) - 1, 0, 9); rhat = np.full(10, float(res.mean()) if len(res) else 0.25)
                    for bi in range(10):
                        if (idx == bi).sum() >= 5:
                            rhat[bi] = float(res[idx == bi].mean())
                    rest = perm[a.n_train:]
                    b = max(a.docs_per_query, int(math.ceil(B / len(rest)))); n_w = int(B // b); q_w = rest[:min(n_w, len(rest))]
                    if len(q_w) < 5:
                        continue
                    ok = {m: True for m in ARMS}; nlab = {m: 0.0 for m in ARMS}
                    for j in others:
                        est = {m: [] for m in ARMS}
                        for k in q_w:
                            r, rj, ks = arr[k]; n = len(r); w = wa.weights(ks, j, cand, n); aw = np.abs(w); pj = prob[H[k]["qid"]]
                            rres = rhat[np.clip(np.digitize(pj, bins) - 1, 0, 9)]
                            base = {"weighted_cv": aw, "ai_calib": aw * np.sqrt(pj * (1 - pj)), "ai_resid": aw * np.sqrt(np.clip(rres, 1e-4, None))}
                            pi_w = sb.sample_pi(base["weighted_cv"], b, n); pi_r = sb.sample_pi(base["ai_resid"], b, n)
                            pis = {"weighted_cv": pi_w, "ai_calib": sb.sample_pi(base["ai_calib"], b, n), "ai_resid": pi_r,
                                   "ai_robust_0.5": 0.5 * pi_r + 0.5 * pi_w, "ai_robust_0.3": 0.7 * pi_r + 0.3 * pi_w}
                            for m in ARMS:
                                pi = pis[m]; samp = (rng.random(n) < pi) & (pi > 0)
                                est[m].append(float((w * rj).sum() + (w[samp] * (r[samp] - rj[samp]) / pi[samp]).sum()))
                                nlab[m] += float(samp.sum()) / len(others)
                        for m in ARMS:
                            D = np.array(est[m]); nq = len(D); varr[m].append(float(D.var(ddof=1)))
                            if D.mean() + cert.t_quantile(1 - a_sim, nq - 1) * D.std(ddof=1) / math.sqrt(nq) > eps:
                                ok[m] = False
                    for m in ARMS:
                        labels[m].append(pilot_docs + nlab[m])
                        if ok[m]:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                for m in ARMS:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws, wrong=cnt[m][1] / a.draws,
                                     docs_labelled=float(np.mean(labels[m])) if labels[m] else float("nan"), var_est=float(np.mean(varr[m])) if varr[m] else float("nan")))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in ARMS) + f" | docs≈{np.mean(labels['weighted_cv']):.0f} | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "active_inference"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"active_{a.stack}_{a.judge}{'_boundary' if a.boundary else ''}.csv"), index=False)


if __name__ == "__main__":
    main()
