#!/usr/bin/env python3
"""Document-level sampling baselines for pair certification (reviewer request, §18; cost accounting revised 2026-09-14).

All arms choose the candidate on the same fully-labelled pilot queries, and certify with the same bound over ordered
pairs. Per-query estimator of D_j = Σ_d w_j(d) r(d) for every competitor j of the candidate:
  uniform      : π_d = b/|pool|                       (HT)
  weighted     : π_d ∝ |w_d|                          (HT)   — importance sampling by decision weight
  active_judge : π_d ∝ |w_d| · sqrt(p̂_d(1−p̂_d))       (HT)   — Active-Inference-style: influence × judge uncertainty
  strat_pilot  : π_d ∝ |w_d| · σ̂_stratum, σ̂ from the pilot (labels counted), strata = {band, common}
  weighted_cv  : weighted sampling + judge control variate, coefficient 1 (D̂_CV)
  active_cv    : active_judge sampling + judge control variate
  weighted_cvl : weighted sampling + judge control variate with a pilot-estimated coefficient λ ∈ [0,1]

Cost accounting (v2, default) — review 2026-09-14 item 3:
  * one human label = one unique (query, document) pair, recorded in a shared Ledger per arm; labels reused by
    several comparisons or already present from the pilot are never charged twice;
  * --sampling shared (default): ONE inclusion vector per query, π ∝ Σ_j base_j(d) over all competitors j, and the
    same labelled set serves every comparison (this is the design the manuscript describes);
    --sampling per_pair: independent draws per comparison with b/(M−1) docs each (the pre-2026-09-14 design, now
    charged at its true unique cost);
  * --pilot_cost full (default): the pilot queries are charged for their whole pool, because fit_params() reads nG
    (all relevant documents) to set the gate scalar; --pilot_cost cutoff charges only up to the largest cutoff (old);
  * the per-query sampling budget is B = budget_full_eq × mean(max cutoff) documents (the pilot is charged on top
    and reported separately), so the x-axis of the ACT curve is the realised `docs_labelled`.
Bound: --bound t (asymptotic, as pre-registered) | eb | bet (finite-sample valid for bounded observations; the range of
the per-query HT / CV estimate is computed from (w, π) and is sample-independent given the design).
--legacy reproduces the old per-comparison design and cutoff pilot cost with the t bound (file name without _v2).
Metrics: ACT rate, wrong rate, mean unique human labels (docs_labelled, docs_pilot), duplicate fraction, per-query
estimator variance.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py"))
pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("wa", os.path.join(HERE, "80_weighted_audit.py"))
wa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wa)
spec = importlib.util.spec_from_file_location("ledger", os.path.join(HERE, "lib", "ledger.py"))
ledger = importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
bc, cert, MENU = pv.bc, pv.cert, pv.MENU
ALPHA = 0.10
ARMS = ["uniform", "weighted", "active_judge", "strat_pilot", "weighted_cv", "active_cv", "weighted_cvl"]
SAMPLER_OF = {"weighted_cv": "weighted", "active_cv": "active_judge", "weighted_cvl": "weighted"}


def sample_pi(base, b, n):
    if base.sum() <= 0:
        return np.zeros(n)
    pi = np.minimum(1.0, b * base / base.sum())
    for _ in range(2):
        free = pi < 1
        if free.any() and pi.sum() < min(b, n):
            pi[free] = np.minimum(1.0, pi[free] * (min(b, n) - pi[~free].sum()) / max(pi[free].sum(), 1e-9))
    return pi


def estimate(m, w, r, rj, pi, samp, lam):
    """Per-query estimate of D = Σ w r for arm m and its deterministic range [lo, hi] given (w, rj, pi)."""
    ok = samp & (pi > 0); pos = pi > 0
    inv = np.zeros_like(pi); inv[pos] = np.abs(w[pos]) / pi[pos]; span = float(inv.sum())
    if m == "weighted_cvl":
        base = float(lam * (w * rj).sum())
        return base + float((w[ok] * (r[ok] - lam * rj[ok]) / pi[ok]).sum()), base - span, base + span
    if m.endswith("_cv"):
        base = float((w * rj).sum())
        return base + float((w[ok] * (r[ok] - rj[ok]) / pi[ok]).sum()), base - span, base + span
    lo, hi = cert.ht_range(w, pi)
    return float((w[ok] * r[ok] / pi[ok]).sum()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[20, 30, 45, 60, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--docs_per_query", type=int, default=4); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--force_worst", action="store_true", help="validity stress: candidate := true worst policy")
    ap.add_argument("--boundary", action="store_true", help="sharp validity stress: candidate := runner-up, eps := 0.9 x its true regret (every ACT is a type-I error)")
    ap.add_argument("--sampling", choices=["shared", "per_pair"], default="shared")
    ap.add_argument("--pilot_cost", choices=["full", "cutoff"], default="full")
    ap.add_argument("--bound", choices=["t", "eb", "bet"], default="t")
    ap.add_argument("--legacy", action="store_true", help="old design: per-pair draws with b docs each, cutoff pilot cost, t bound, no _v2 suffix")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _cvl)")
    a = ap.parse_args()
    if a.legacy:
        a.sampling, a.pilot_cost, a.bound = "per_pair", "cutoff", "t"
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
        pj_all = data[held]["rr"]                                # probability of the *current* judge (loader order)
        prob = {}
        for s, (q, s0, s1) in zip(H, data[held]["slices"]):
            p = P[held][s0:s1]; o = np.argsort(-p); prob[s["qid"]] = pj_all[s0:s1][o]
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in ARMS}; varr = {m: [] for m in ARMS}; docs = {m: [] for m in ARMS}; dup = {m: [] for m in ARMS}; pil = []
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
                    # ---- ledgers: one per arm; pilot charged first ----
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
                    rest = perm[a.n_train:]
                    sd = {}
                    for j in others:
                        vals = {"band": [], "common": []}
                        for k in tr:
                            r, rj, ks = arr[k]; lo, hi = min(ks[j], ks[cand]), max(ks[j], ks[cand])
                            vals["common"] += list(r[:lo]); vals["band"] += list(r[lo:hi])
                        sd[j] = {s: (np.std(v) if len(v) > 1 else 0.5) + 1e-3 for s, v in vals.items()}
                    b = max(a.docs_per_query, int(math.ceil(B / len(rest)))); n_w = int(B // b); q_w = rest[:min(n_w, len(rest))]
                    if len(q_w) < 5:
                        continue
                    lam_pair = {}
                    for j in others:
                        dt, dj = [], []
                        for k in tr:
                            r, rj, ks = arr[k]; w = wa.weights(ks, j, cand, len(r)); dt.append(float((w * r).sum())); dj.append(float((w * rj).sum()))
                        dt, dj = np.array(dt), np.array(dj); v = dj.var(ddof=1) if len(dj) > 1 else 0.0
                        lam_pair[j] = float(np.clip(np.cov(dt, dj)[0, 1] / v, 0, 1)) if v > 0 else 0.0
                    est = {m: {j: [] for j in others} for m in ARMS}; rng_lo = {m: {j: [] for j in others} for m in ARMS}; rng_hi = {m: {j: [] for j in others} for m in ARMS}
                    b_pair = b if a.legacy else max(1, b / len(others))
                    for k in q_w:
                        r, rj, ks = arr[k]; n = len(r); pj = prob[H[k]["qid"]]; qid = H[k]["qid"]
                        W = {j: wa.weights(ks, j, cand, n) for j in others}
                        bases = {}
                        for j in others:
                            aw = np.abs(W[j]); lo, hi = min(ks[j], ks[cand]), max(ks[j], ks[cand])
                            strat = np.array([sd[j]["band"] if lo <= i < hi else sd[j]["common"] for i in range(n)])
                            bases[j] = {"uniform": np.ones(n), "weighted": aw, "active_judge": aw * np.sqrt(np.clip(pj * (1 - pj), 1e-4, None)), "strat_pilot": aw * strat}
                        for m in ARMS:
                            key = SAMPLER_OF.get(m, m)
                            if a.sampling == "shared":
                                pi = sample_pi(sum(bases[j][key] for j in others), b, n); samp = (rng.random(n) < pi) & (pi > 0)
                                L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                for j in others:
                                    v, lo, hi = estimate(m, W[j], r, rj, pi, samp, lam_pair[j]); est[m][j].append(v); rng_lo[m][j].append(lo); rng_hi[m][j].append(hi)
                            else:
                                for j in others:
                                    pi = sample_pi(bases[j][key], b_pair, n); samp = (rng.random(n) < pi) & (pi > 0)
                                    L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                    v, lo, hi = estimate(m, W[j], r, rj, pi, samp, lam_pair[j]); est[m][j].append(v); rng_lo[m][j].append(lo); rng_hi[m][j].append(hi)
                    for m in ARMS:
                        ok = True
                        for j in others:
                            D = np.array(est[m][j]); varr[m].append(float(D.var(ddof=1)))
                            ucb = cert.ucb_mean_upper(D, a_sim, min(rng_lo[m][j]), max(rng_hi[m][j]), a.bound)
                            if ucb > eps:
                                ok = False
                        docs[m].append(L[m].cost); dup[m].append(L[m].duplicate_fraction())
                        if ok:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                for m in ARMS:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws,
                                     wrong=cnt[m][1] / a.draws, var_est=float(np.mean(varr[m])) if varr[m] else float("nan"),
                                     docs_labelled=float(np.mean(docs[m])) if docs[m] else float("nan"), docs_pilot=float(np.mean(pil)) if pil else float("nan"),
                                     dup_frac=float(np.mean(dup[m])) if dup[m] else float("nan"),
                                     sampling=a.sampling, pilot_cost=a.pilot_cost, bound=a.bound))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in ARMS)
                      + f" | docs≈{np.mean(docs['weighted']) if docs['weighted'] else float('nan'):.0f} (pilot {np.mean(pil) if pil else float('nan'):.0f})"
                      + f" | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "sampling_baselines"); os.makedirs(out, exist_ok=True)
    suffix = "" if a.legacy else f"_v2_{a.sampling}_{a.pilot_cost}_{a.bound}"
    pd.DataFrame(rows).to_csv(os.path.join(out, f"baselines_{a.stack}_{a.judge}{'_forceworst' if a.force_worst else ''}{'_boundary' if a.boundary else ''}{a.tag}{suffix}.csv"), index=False)


if __name__ == "__main__":
    main()
