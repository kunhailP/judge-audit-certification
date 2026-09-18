#!/usr/bin/env python3
"""Faithful Active Statistical Inference comparison for the estimand D(q) = Σ_d w_d y_d (§18 follow-up; cost accounting
and residual model revised 2026-09-14).

For the prediction-powered / active estimator  D̂ = Σ_d w_d ĵ_d + Σ_{d sampled} w_d (y_d − ĵ_d)/π_d ,
Var(D̂) = Σ_d w_d² E[(y_d − ĵ_d)²] (1 − π_d)/π_d, so the budget-optimal rule is
      π_d ∝ |w_d| · sqrt( E[(y_d − ĵ_d)²] )        (Zrnic & Candès 2024, adapted to the weighted estimand).
Arms (same pilot — used for candidate choice AND for the residual model —, same sampling budget, same bound):
  weighted_cv   : π ∝ |w|                                         + judge control variate
  ai_calib      : π ∝ |w|·sqrt(p̂(1−p̂))                             + CV   (assumes a calibrated judge)
  ai_resid      : π ∝ |w|·sqrt(r̂(p̂)), r̂ = pilot-estimated E[(y−ĵ)²] in 10 judge-probability bins   + CV
  ai_robust_0.5 : π = 0.5·π_ai_resid + 0.5·π_weighted            + CV   (Robust Sampling-style mixture)
  ai_robust_0.3 : π = 0.7·π_ai_resid + 0.3·π_weighted            + CV

Revisions (review 2026-09-14, items 3 and 4):
  * the control-variate prediction ĵ and the residual model use the SAME quantity: --cv_pred binary (default,
    ĵ = 1[p̂ ≥ 0.5] as in the manuscript; residual (y − 1[p̂≥0.5])²) or prob (ĵ = p̂; residual (y − p̂)²). Before,
    the residual was fitted on p̂ while the CV used the binarised label.
  * one shared inclusion vector per query across all comparisons (π ∝ Σ_j base_j), one Ledger per arm, unique-label
    cost (pilot charged for the whole pool by default: fit_params reads nG). `docs_labelled` is the realised unique
    cost; the old `nlab += samp.sum()/len(others)` (a per-comparison average) is gone.
  * --bound t|eb|bet as in 81. --legacy restores the old per-pair draws / cutoff pilot cost / mismatched residual.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py")); pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("wa", os.path.join(HERE, "80_weighted_audit.py")); wa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wa)
spec = importlib.util.spec_from_file_location("sb", os.path.join(HERE, "81_sampling_baselines.py")); sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
spec = importlib.util.spec_from_file_location("ledger", os.path.join(HERE, "lib", "ledger.py")); ledger = importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10
ARMS = ["weighted_cv", "ai_calib", "ai_resid", "ai_robust_0.5", "ai_robust_0.3"]


def cv_estimate(w, r, jhat, pi, samp):
    """CV estimate of Σ w r, its deterministic range, and the HT estimate of its document-sampling variance."""
    ok = samp & (pi > 0); pos = pi > 0
    base = float((w * jhat).sum()); span = float((np.abs(w[pos]) / pi[pos]).sum()); y = r - jhat
    return base + float((w[ok] * y[ok] / pi[ok]).sum()), base - span, base + span, cert.ht_var_hat(w, y, pi, samp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[30, 45, 60, 75, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--docs_per_query", type=int, default=4); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--boundary", action="store_true", help="validity stress: candidate := runner-up, eps := 0.9 x its true regret")
    ap.add_argument("--sampling", choices=["shared", "per_pair"], default="shared")
    ap.add_argument("--pilot_cost", choices=["full", "cutoff"], default="full")
    ap.add_argument("--cv_pred", choices=["binary", "prob"], default="binary")
    ap.add_argument("--bound", choices=["t", "eb", "bet"], default="t")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _grid2)")
    a = ap.parse_args()
    if a.legacy:
        a.sampling, a.pilot_cost, a.bound = "per_pair", "cutoff", "t"
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
                cnt = {m: [0, 0] for m in ARMS}; varr = {m: [] for m in ARMS}; labels = {m: [] for m in ARMS}; dup = {m: [] for m in ARMS}; pil = []
                for _ in range(a.draws):
                    perm = rng.permutation(N); tr = perm[:a.n_train]
                    c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr]); arr = wa.arrays(H, HJ, c_tr, tau_tr)
                    cost_full = np.array([max(ks) for _, _, ks in arr], float)
                    U_tr = wa.prec_utils([arr[k] for k in tr]); cand = cert.pick_candidate(U_tr)
                    mu = wa.prec_utils(arr).mean(axis=0)
                    if a.boundary:
                        cand = int(np.argsort(-mu)[1]); eps = 0.9 * float(mu.max() - mu[cand])
                    regret = float(mu.max() - mu[cand]); others = [j for j in range(M) if j != cand]
                    L = {m: ledger.Ledger() for m in ARMS}
                    for k in tr:
                        r, _, ks = arr[k]
                        for m in ARMS:
                            if a.pilot_cost == "full":
                                L[m].request_full(H[k]["qid"], len(r))
                            else:
                                L[m].request(H[k]["qid"], np.arange(int(max(ks))), pool_size=len(r))
                    pilot_docs = L[ARMS[0]].cost; pil.append(pilot_docs)
                    B = Bq * float(cost_full.mean()) - (float(cost_full[tr].sum()) if a.legacy else 0.0)
                    if B <= 0:
                        continue
                    # residual model from the pilot: E[(y - ĵ)^2] per judge-probability bin, with ĵ the SAME prediction the CV uses
                    py, pjv, jv = [], [], []
                    for k in tr:
                        r, rj, ks = arr[k]; kmax = len(r) if a.pilot_cost == "full" else int(max(ks)); pj = prob[H[k]["qid"]][:kmax]
                        py += list(r[:kmax]); pjv += list(pj); jv += list(rj[:kmax] if a.cv_pred == "binary" else pj)
                    py, pjv, jv = np.array(py), np.array(pjv), np.array(jv)
                    res = (py - (pjv if a.legacy else jv)) ** 2
                    idx = np.clip(np.digitize(pjv, bins) - 1, 0, 9); rhat = np.full(10, float(res.mean()) if len(res) else 0.25)
                    for bi in range(10):
                        if (idx == bi).sum() >= 5:
                            rhat[bi] = float(res[idx == bi].mean())
                    rest = perm[a.n_train:]
                    b = max(a.docs_per_query, int(math.ceil(B / len(rest)))); n_w = int(B // b); q_w = rest[:min(n_w, len(rest))]
                    if len(q_w) < 5:
                        continue
                    b_pair = b if a.legacy else max(1, b / len(others))
                    est = {m: {j: [] for j in others} for m in ARMS}; lo_ = {m: {j: [] for j in others} for m in ARMS}; hi_ = {m: {j: [] for j in others} for m in ARMS}
                    vh = {m: {j: [] for j in others} for m in ARMS}
                    D_known = {j: np.array([(wa.weights(arr[k][2], j, cand, len(arr[k][0])) * arr[k][0]).sum() for k in tr]) for j in others}
                    for k in q_w:
                        r, rj, ks = arr[k]; n = len(r); pj = prob[H[k]["qid"]]; qid = H[k]["qid"]
                        jhat = rj if a.cv_pred == "binary" else pj
                        rres = rhat[np.clip(np.digitize(pj, bins) - 1, 0, 9)]
                        W = {j: wa.weights(ks, j, cand, n) for j in others}

                        def pis_for(aw):
                            pi_w = sb.sample_pi(aw, bb, n); pi_r = sb.sample_pi(aw * np.sqrt(np.clip(rres, 1e-4, None)), bb, n)
                            return {"weighted_cv": pi_w, "ai_calib": sb.sample_pi(aw * np.sqrt(pj * (1 - pj)), bb, n), "ai_resid": pi_r,
                                    "ai_robust_0.5": 0.5 * pi_r + 0.5 * pi_w, "ai_robust_0.3": 0.7 * pi_r + 0.3 * pi_w}
                        if a.sampling == "shared":
                            bb = b; pis = pis_for(sum(np.abs(W[j]) for j in others))
                            for m in ARMS:
                                pi = pis[m]; samp = (rng.random(n) < pi) & (pi > 0); L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                for j in others:
                                    v, lo, hi, vv = cv_estimate(W[j], r, jhat, pi, samp); est[m][j].append(v); lo_[m][j].append(lo); hi_[m][j].append(hi); vh[m][j].append(vv)
                        else:
                            bb = b_pair
                            for j in others:
                                pis = pis_for(np.abs(W[j]))
                                for m in ARMS:
                                    pi = pis[m]; samp = (rng.random(n) < pi) & (pi > 0); L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                    v, lo, hi, vv = cv_estimate(W[j], r, jhat, pi, samp); est[m][j].append(v); lo_[m][j].append(lo); hi_[m][j].append(hi); vh[m][j].append(vv)
                    for m in ARMS:
                        ok = True
                        for j in others:
                            D = np.array(est[m][j]); varr[m].append(float(D.var(ddof=1)))
                            if a.bound == "t" and not a.legacy:   # population estimand, two-stage bound (see 81)
                                ucb = cert.ucb_population_two_stage(D_known[j], D, np.array(vh[m][j]), N, a_sim)
                            else:
                                ucb = cert.ucb_mean_upper(D, a_sim, min(lo_[m][j]), max(hi_[m][j]), a.bound)
                            if ucb > eps:
                                ok = False
                        labels[m].append(L[m].cost); dup[m].append(L[m].duplicate_fraction())
                        if ok:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                for m in ARMS:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws, wrong=cnt[m][1] / a.draws,
                                     docs_labelled=float(np.mean(labels[m])) if labels[m] else float("nan"), docs_pilot=float(np.mean(pil)) if pil else float("nan"),
                                     dup_frac=float(np.mean(dup[m])) if dup[m] else float("nan"), var_est=float(np.mean(varr[m])) if varr[m] else float("nan"),
                                     sampling=a.sampling, pilot_cost=a.pilot_cost, cv_pred=a.cv_pred, bound=a.bound))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in ARMS) + f" | docs≈{np.mean(labels['weighted_cv']) if labels['weighted_cv'] else float('nan'):.0f} | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "04_results", "active_inference"); os.makedirs(out, exist_ok=True)
    suffix = "" if a.legacy else f"_v2_{a.sampling}_{a.pilot_cost}_{a.cv_pred}_{a.bound}"
    pd.DataFrame(rows).to_csv(os.path.join(out, f"active_{a.stack}_{a.judge}{'_boundary' if a.boundary else ''}{a.tag}{suffix}.csv"), index=False)


if __name__ == "__main__":
    main()
