#!/usr/bin/env python3
"""Document-level sampling baselines for pair certification (reviewer request, §18).

All arms label the SAME expected number of documents (budget B), choose the
candidate on the same 20 fully-labelled train queries, and certify with the
same t bound over ordered pairs.  Per-query estimator of D = Σ_d w(d) r(d):
  uniform      : π_d = b/|pool|                       (HT)
  weighted     : π_d ∝ |w_d|                          (HT)   — importance sampling by decision weight
  active_judge : π_d ∝ |w_d| · sqrt(p̂_d(1−p̂_d))       (HT)   — Active-Inference-style: influence × judge uncertainty
  strat_pilot  : π_d ∝ |w_d| · σ̂_stratum, σ̂ from a 20-query pilot (labels counted), strata = {band, common}
  weighted_cv  : weighted sampling + judge control variate (D̂_CV)
  active_cv    : active_judge sampling + judge control variate
Metrics: ACT rate and wrong rate at each budget; budget needed to reach ACT ≥ 50% (interpolated); also the
variance of the per-query estimator relative to full labelling.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("wa", os.path.join(HERE, "80_weighted_audit.py"))
wa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wa)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10
ARMS = ["uniform", "weighted", "active_judge", "strat_pilot", "weighted_cv", "active_cv", "weighted_cvl"]
# weighted_cvl: control variate with a coefficient lambda estimated on the pilot (query-level regression of true D on
# judge D over the pilot's fully labelled queries, clipped to [0,1]); lambda=0 recovers HT, so an adversarial judge
# can no longer add variance (post-hoc addition after LOCK v0.5, see design doc §21.4).


def sample_pi(base, b, n):
    if base.sum() <= 0:
        return np.zeros(n)
    pi = np.minimum(1.0, b * base / base.sum())
    for _ in range(2):
        free = pi < 1
        if free.any() and pi.sum() < min(b, n):
            pi[free] = np.minimum(1.0, pi[free] * (min(b, n) - pi[~free].sum()) / max(pi[free].sum(), 1e-9))
    return pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[20, 30, 45, 60, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--docs_per_query", type=int, default=4); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--force_worst", action="store_true", help="validity stress: candidate := true worst policy")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _cvl)")
    ap.add_argument("--boundary", action="store_true", help="sharp validity stress: candidate := runner-up, eps := 0.9 x its true regret (every ACT is a type-I error)")
    a = ap.parse_args()
    pv.NAMES = a.names; pv.JUDGE = a.judge
    data = pv.load_with_judge(os.path.join(a.pools, a.stack, "runs", "candidates"))
    if a.train_dir:
        pv.JUDGE = "rr"; data.update(pv.load_train_only(a.train_dir, a.train_names)); pv.JUDGE = a.judge
    rng_global = np.random.default_rng(0); rng = np.random.default_rng(31)
    M = len(MENU); a_sim = ALPHA / (M * (M - 1)); rows = []
    for held in [d for d in data if d not in pv.TRAIN_ONLY]:
        P, structs, reg = pv.fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = pv.finalize(H, reg); N = len(H)
        HJ = bc.build_structs({held: data[held]["judge"]}, held, P[held]); pv.finalize(HJ, reg)
        pj_all = data[held]["rr"]                                # judge probability per pooled doc (loader order)
        # per-doc judge probability in struct (descending-P) order
        prob = {}
        for s, (q, s0, s1) in zip(H, data[held]["slices"]):
            p = P[held][s0:s1]; o = np.argsort(-p); prob[s["qid"]] = pj_all[s0:s1][o]
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in ARMS}; varr = {m: [] for m in ARMS}
                for _ in range(a.draws):
                    perm = rng.permutation(N); tr = perm[:a.n_train]
                    c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr]); arr = wa.arrays(H, HJ, c_tr, tau_tr)
                    cost_full = np.array([max(ks) for _, _, ks in arr], float)
                    U_tr = wa.prec_utils([arr[k] for k in tr]); cand = cert.pick_candidate(U_tr)
                    if a.force_worst:
                        cand = int(np.argmin(wa.prec_utils(arr).mean(axis=0)))
                    if a.boundary:
                        mu_all = wa.prec_utils(arr).mean(axis=0); cand = int(np.argsort(-mu_all)[1]); eps = 0.9 * float(mu_all.max() - mu_all[cand])
                    mu = wa.prec_utils(arr).mean(axis=0); regret = float(mu.max() - mu[cand]); others = [j for j in range(M) if j != cand]
                    B = Bq * float(cost_full.mean()) - float(cost_full[tr].sum())
                    if B <= 0:
                        continue
                    rest = perm[a.n_train:]
                    # pilot stratum sds for strat_pilot from the train queries (already labelled): band vs common
                    sd = {}
                    for j in others:
                        vals = {"band": [], "common": []}
                        for k in tr:
                            r, rj, ks = arr[k]; w = wa.weights(ks, j, cand, len(r)); lo, hi = min(ks[j], ks[cand]), max(ks[j], ks[cand])
                            vals["common"] += list(r[:lo]); vals["band"] += list(r[lo:hi])
                        sd[j] = {s: (np.std(v) if len(v) > 1 else 0.5) + 1e-3 for s, v in vals.items()}
                    b = max(a.docs_per_query, int(math.ceil(B / len(rest)))); n_w = int(B // b); q_w = rest[:min(n_w, len(rest))]
                    if len(q_w) < 5:
                        continue
                    ok = {m: True for m in ARMS}
                    # pilot lambda per pair: regress true paired difference on judge paired difference over the 20 pilot queries
                    lam_pair = {}
                    for j in others:
                        dt, dj = [], []
                        for k in tr:
                            r, rj, ks = arr[k]; w = wa.weights(ks, j, cand, len(r)); dt.append(float((w * r).sum())); dj.append(float((w * rj).sum()))
                        dt, dj = np.array(dt), np.array(dj); v = dj.var(ddof=1) if len(dj) > 1 else 0.0
                        lam_pair[j] = float(np.clip(np.cov(dt, dj)[0, 1] / v, 0, 1)) if v > 0 else 0.0
                    for j in others:
                        est = {m: [] for m in ARMS}
                        for k in q_w:
                            r, rj, ks = arr[k]; n = len(r); w = wa.weights(ks, j, cand, n); aw = np.abs(w); pj = prob[H[k]["qid"]]
                            lo, hi = min(ks[j], ks[cand]), max(ks[j], ks[cand])
                            strat = np.array([sd[j]["band"] if lo <= i < hi else sd[j]["common"] for i in range(n)])
                            bases = {"uniform": np.ones(n), "weighted": aw, "active_judge": aw * np.sqrt(np.clip(pj * (1 - pj), 1e-4, None)),
                                     "strat_pilot": aw * strat}
                            for m in ARMS:
                                key = {"weighted_cv": "weighted", "active_cv": "active_judge", "weighted_cvl": "weighted"}.get(m, m)
                                pi = sample_pi(bases[key], b, n); samp = (rng.random(n) < pi) & (pi > 0)
                                if m == "weighted_cvl":
                                    lm = lam_pair[j]
                                    v = float(lm * (w * rj).sum() + (w[samp] * (r[samp] - lm * rj[samp]) / pi[samp]).sum())
                                elif m.endswith("_cv"):
                                    v = float((w * rj).sum() + (w[samp] * (r[samp] - rj[samp]) / pi[samp]).sum())
                                else:
                                    v = float((w[samp] * r[samp] / pi[samp]).sum())
                                est[m].append(v)
                        for m in ARMS:
                            D = np.array(est[m]); nq = len(D)
                            ucb = D.mean() + cert.t_quantile(1 - a_sim, nq - 1) * D.std(ddof=1) / math.sqrt(nq)
                            varr[m].append(float(D.var(ddof=1)))
                            if ucb > eps:
                                ok[m] = False
                    for m in ARMS:
                        if ok[m]:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                for m in ARMS:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws,
                                     wrong=cnt[m][1] / a.draws, var_est=float(np.mean(varr[m])) if varr[m] else float("nan")))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in ARMS) + f" | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "sampling_baselines"); os.makedirs(out, exist_ok=True)
    ap_tag = a.tag if hasattr(a, "tag") and a.tag else ""
    pd.DataFrame(rows).to_csv(os.path.join(out, f"baselines_{a.stack}_{a.judge}{'_forceworst' if a.force_worst else ''}{'_boundary' if a.boundary else ''}{ap_tag}.csv"), index=False)


if __name__ == "__main__":
    main()
