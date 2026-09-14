#!/usr/bin/env python3
"""Document-label allocation for certifying a whole menu (EXPLORATION_menu_allocation.md §3; revised 2026-09-14).

Menu of 4 policies (glob, trunc, ad, rr_thresh), precision@cutoff, judge control variate in every arm.
Candidate m̂ from a fully-labelled pilot of n_train queries. R rounds on fresh queries; within a round every
pair shares the labels of a query. All arms get the same expected number of sampled documents and the same
certificate (one-sided bound over the M−1 competitors at level α/(M(M−1)), --bound t | eb | bet).

Arms
  static_sum   : π ∝ Σ_j |w_j|                                  (label sharing, no adaptivity)   — strong baseline
  per_pair     : round queries split 3 ways, each with π ∝ |w_j|, no sharing, b docs per query  — weak baseline
                 (the legacy run gave it 3b docs per query, i.e. ~2-3x the unique labels of the shared arms; the
                 ledger makes that visible, so the v2 default equalises the budget)
  adaptive     : π ∝ Σ_j λ_j |w_j|, λ_j ∝ 1/max(ε − UCB_j, δ) from previous rounds (floor 0.1)  — legacy heuristic
  oracle       : λ_j from true gaps                                                             — legacy heuristic
  static_sum_v : π ∝ Σ_j |w_j| · sqrt(v̂_i), v̂ from the pilot residual model (variance-aware static rule, the
                 active-inference analogue for a shared menu; the fair baseline for the menu-design head-room)
  oracle_exact : exact minimax design (lib/design.py)  min_π max_j V_j(π)/s_j²  with the TRUE slacks
                 s_j = ε − (μ_j − μ_m̂) and the POPULATION residual-variance function v(p) = E[(y−ĵ)² | judge-prob
                 bin] fitted on the whole collection. The oracle knows the gaps and the variance law, not the
                 labels. Upper bound on what any allocation can achieve with this certificate (Gate A numerator).
  plugin_exact : the same design with everything estimated from data the auditor has: s_j from the pilot
                 (round 1) and then from the running estimates, v̂ from the pilot residual model on the judge
                 probability (5 quantile bins, floor --v_floor so that every decision document keeps π > 0).
                 Realistic procedure without oracle knowledge (Gate B numerator).

Cost accounting (v2, default): a shared Ledger per arm charges each unique (query, document) human label once;
--pilot_cost full charges the pilot queries for their whole pool (fit_params reads nG), cutoff only up to the
largest cutoff (old). Reported: docs_labelled (unique, incl. pilot), docs_pilot, docs_sampling, dup_frac.
--legacy reproduces the pre-2026-09-14 run (cutoff pilot cost, t bound, budget net of the pilot, four legacy arms,
file name without _v2).
Also reports the non-overlap index between decision-document sets of the binding pair and the other pairs, and
the design objective max_j V_j/s_j² (true v, true s) of every arm's realised π averaged over rounds — the quantity
the oracle minimises, so the ratio static_sum/oracle_exact is the structural head-room independent of the bound.
"""
import argparse, importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("pv", os.path.join(HERE, "63_planner_v2.py")); pv = importlib.util.module_from_spec(spec); spec.loader.exec_module(pv)
spec = importlib.util.spec_from_file_location("sb", os.path.join(HERE, "81_sampling_baselines.py")); sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
spec = importlib.util.spec_from_file_location("ledger", os.path.join(HERE, "lib", "ledger.py")); ledger = importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
spec = importlib.util.spec_from_file_location("design", os.path.join(HERE, "lib", "design.py")); design = importlib.util.module_from_spec(spec); spec.loader.exec_module(design)
bc, cert = pv.bc, pv.cert
MENU4 = pv.MENU4; M = 4
ALPHA = 0.10
LEGACY_ARMS = ["static_sum", "per_pair", "adaptive", "oracle"]
ARMS = LEGACY_ARMS + ["static_sum_v", "oracle_exact", "plugin_exact"]
S_FLOOR = 0.002


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


def cv_estimate(w, r, rj, pi, samp):
    """Control-variate estimate of Σ w r and its deterministic range given (w, rj, π)."""
    ok = samp & (pi > 0); pos = pi > 0
    base = float((w * rj).sum()); span = float((np.abs(w[pos]) / pi[pos]).sum())
    return base + float((w[ok] * (r[ok] - rj[ok]) / pi[ok]).sum()), base - span, base + span


def residual_model(pj_bins, resid_sq, edges, v_floor):
    """v̂(p) = mean squared residual in the quantile bin of the judge probability p (pilot-estimated)."""
    idx = np.clip(np.searchsorted(edges, pj_bins, side="right") - 1, 0, len(edges) - 2)
    vb = np.array([resid_sq[idx == b].mean() if (idx == b).any() else resid_sq.mean() for b in range(len(edges) - 1)])
    return np.maximum(vb, v_floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--judge", default="llm")
    ap.add_argument("--train_dir", default=None); ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    ap.add_argument("--budget_full", nargs="+", type=int, default=[30, 45, 60, 90]); ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--n_train", type=int, default=20); ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--pilot_cost", choices=["full", "cutoff"], default="full")
    ap.add_argument("--bound", choices=["t", "eb", "bet"], default="t")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--design_iters", type=int, default=150)
    ap.add_argument("--v_floor", type=float, default=0.02, help="floor of the plug-in residual variance (keeps π > 0 on every decision document)")
    ap.add_argument("--legacy", action="store_true", help="pre-2026-09-14 design: cutoff pilot cost, budget net of pilot, t bound, legacy arms, no _v2 suffix")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.legacy:
        a.pilot_cost, a.bound, a.arms = "cutoff", "t", LEGACY_ARMS
    arms = a.arms
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
        pj_all = data[held]["rr"]; prob = {}
        for s, (q, s0, s1) in zip(H, data[held]["slices"]):
            p = P[held][s0:s1]; o = np.argsort(-p); prob[s["qid"]] = pj_all[s0:s1][o]
        PJ = [np.asarray(prob[s["qid"]], float) for s in H]
        # population residual-variance function v(p) = E[(y - ĵ)^2 | judge prob bin], known to the oracle only
        pop_p = np.concatenate(PJ); pop_res = np.concatenate([(R_true[k] - R_j[k]) ** 2 for k in range(N)])
        edges_pop = np.unique(np.quantile(pop_p, np.linspace(0, 1, 6))); edges_pop[0] = -np.inf; edges_pop[-1] = np.inf
        vbin_pop = residual_model(pop_p, pop_res, edges_pop, a.v_floor)
        for Bq in a.budget_full:
            for eps in a.eps:
                cnt = {m: [0, 0] for m in arms}; overlap = []; docs = {m: [] for m in arms}; dup = {m: [] for m in arms}; pil = []
                obj = {m: [] for m in arms}
                for _ in range(a.draws):
                    perm = rng.permutation(N); tr = perm[:a.n_train]
                    c_tr, tau_tr, _ = pv.fit_params([H[k] for k in tr]); th_tr = pv.fit_theta([H[k] for k in tr])
                    MK = [masks_for(H[k], c_tr, tau_tr, th_tr) for k in range(N)]
                    U_tr = np.array([[prec(R_true[k], MK[k][m]) for m in range(M)] for k in tr]); cand = cert.pick_candidate(U_tr)
                    mu = np.array([[prec(R_true[k], MK[k][m]) for m in range(M)] for k in range(N)]).mean(axis=0); regret = float(mu.max() - mu[cand])
                    others = [j for j in range(M) if j != cand]; gaps = {j: float(mu[j] - mu[cand]) for j in others}
                    cost_full = np.array([max(mk.sum() for mk in MK[k]) for k in range(N)], float)
                    # ---- ledgers, pilot charged first ----
                    L = {m: ledger.Ledger() for m in arms}
                    for k in tr:
                        for m in arms:
                            if a.pilot_cost == "full":
                                L[m].request_full(H[k]["qid"], len(R_true[k]))
                            else:
                                L[m].request(H[k]["qid"], np.arange(int(cost_full[k])), pool_size=len(R_true[k]))
                    pil.append(L[arms[0]].cost)
                    B = Bq * float(cost_full.mean()) - (float(cost_full[tr].sum()) if a.legacy else 0.0)
                    if B <= 0:
                        continue
                    rest = perm[a.n_train:]; chunks = np.array_split(rest, a.rounds)
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
                    # ---- pilot-side quantities the plug-in arm may use ----
                    s_true = np.array([max(eps - gaps[j], S_FLOOR) for j in others])
                    D_pilot = {j: np.array([(weights(k, j) * R_true[k]).sum() for k in tr]) for j in others}
                    pil_p = np.concatenate([PJ[k] for k in tr]); pil_res = np.concatenate([(R_true[k] - R_j[k]) ** 2 for k in tr])
                    edges = np.unique(np.quantile(pil_p, np.linspace(0, 1, 6))); edges[0] = -np.inf; edges[-1] = np.inf
                    vbin = residual_model(pil_p, pil_res, edges, a.v_floor)
                    def v_hat(k):
                        idx = np.clip(np.searchsorted(edges, PJ[k], side="right") - 1, 0, len(edges) - 2); return vbin[idx]
                    def v_pop(k):
                        idx = np.clip(np.searchsorted(edges_pop, PJ[k], side="right") - 1, 0, len(edges_pop) - 2); return vbin_pop[idx]
                    est = {m: {j: [] for j in others} for m in arms}; rlo = {m: {j: [] for j in others} for m in arms}; rhi = {m: {j: [] for j in others} for m in arms}
                    lam = {m: {j: 1.0 for j in others} for m in arms}
                    if "oracle" in arms:
                        lam["oracle"] = {j: 1.0 / max(eps - gaps[j], S_FLOOR) for j in others}
                    for r_idx, chunk in enumerate(chunks):
                        if len(chunk) == 0:
                            continue
                        b = max(2, int(math.ceil((B / a.rounds) / len(chunk))))
                        budget_chunk = float(sum(min(b, len(R_true[k])) for k in chunk))
                        groups = np.array_split(chunk, len(others))
                        # exact designs over all documents of the round (one π per document, shared by every pair)
                        PI_exact = {}
                        if "oracle_exact" in arms or "plugin_exact" in arms:
                            A = np.concatenate([np.stack([weights(k, j) for j in others]) for k in chunk], axis=1)   # (J, n_docs)
                            offs = np.cumsum([0] + [len(R_true[k]) for k in chunk])
                        if "oracle_exact" in arms:
                            v_o = np.concatenate([v_pop(k) for k in chunk])
                            pi_o, _ = design.oracle_design(A, v_o, s_true, budget_chunk, iters=a.design_iters); PI_exact["oracle_exact"] = pi_o
                        if "plugin_exact" in arms:
                            s_hat = []
                            for j in others:
                                D = np.concatenate([D_pilot[j], np.array(est["plugin_exact"][j])])
                                s_hat.append(max(eps - float(D.mean()), S_FLOOR))
                            v_pl = np.concatenate([v_hat(k) for k in chunk])
                            pi_p, _ = design.oracle_design(A, v_pl, np.asarray(s_hat), budget_chunk, iters=a.design_iters); PI_exact["plugin_exact"] = pi_p
                        for m in arms:
                            A_round = []; PI_round = []
                            for gi, k in enumerate(chunk):
                                r, rj = R_true[k], R_j[k]; n = len(r); qid = H[k]["qid"]
                                if m == "per_pair":
                                    j_only = others[[i for i, g in enumerate(groups) if k in g][0]]
                                    # equal expected label count per query (b) as the shared arms; each query serves one pair only
                                    base = np.abs(weights(k, j_only)); pi = sb.sample_pi(base, min(b * (len(others) if a.legacy else 1), n), n)
                                    samp = (rng.random(n) < pi) & (pi > 0); L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                    v, lo, hi = cv_estimate(weights(k, j_only), r, rj, pi, samp); est[m][j_only].append(v); rlo[m][j_only].append(lo); rhi[m][j_only].append(hi)
                                    continue
                                if m in PI_exact:
                                    pi = PI_exact[m][offs[gi]:offs[gi + 1]]
                                elif m == "static_sum_v":
                                    base = sum(np.abs(weights(k, j)) for j in others) * np.sqrt(v_hat(k)); pi = sb.sample_pi(base, b, n)
                                else:
                                    lm = lam[m]; base = sum(lm[j] * np.abs(weights(k, j)) for j in others); pi = sb.sample_pi(base, b, n)
                                samp = (rng.random(n) < pi) & (pi > 0); L[m].request(qid, np.flatnonzero(samp), pool_size=n)
                                for j in others:
                                    v, lo, hi = cv_estimate(weights(k, j), r, rj, pi, samp); est[m][j].append(v); rlo[m][j].append(lo); rhi[m][j].append(hi)
                                PI_round.append(pi)
                            if m != "per_pair" and PI_round:
                                A_r = np.concatenate([np.stack([weights(k, j) for j in others]) for k in chunk], axis=1)
                                v_r = np.concatenate([v_pop(k) for k in chunk])
                                val, _ = design.evaluate(np.concatenate(PI_round), A_r, v_r, s_true); obj[m].append(val if np.isfinite(val) else np.nan)
                        if "adaptive" in arms and r_idx < a.rounds - 1:
                            lm = {}
                            for j in others:
                                D = np.array(est["adaptive"][j]); nq = len(D)
                                ucb = D.mean() + cert.t_quantile(1 - a_sim, max(nq - 1, 2)) * D.std(ddof=1) / math.sqrt(max(nq, 2)) if nq > 2 else eps
                                lm[j] = 1.0 / max(eps - ucb, S_FLOOR) if ucb < eps else 1.0 / S_FLOOR
                            mx = max(lm.values()); lam["adaptive"] = {j: max(v / mx, 0.1) for j, v in lm.items()}
                    for m in arms:
                        ok = True
                        for j in others:
                            D = np.array(est[m][j]); nq = len(D)
                            if nq < 5:
                                ok = False; break
                            ucb = cert.ucb_mean_upper(D, a_sim, min(rlo[m][j]), max(rhi[m][j]), a.bound)
                            if ucb > eps:
                                ok = False; break
                        if ok:
                            cnt[m][0] += 1; cnt[m][1] += regret > eps
                        docs[m].append(L[m].cost); dup[m].append(L[m].duplicate_fraction())
                for m in arms:
                    rows.append(dict(collection=held, judge=a.judge, budget_full_eq=Bq, eps=eps, method=m, act=cnt[m][0] / a.draws, wrong=cnt[m][1] / a.draws,
                                     docs_labelled=float(np.mean(docs[m])) if docs[m] else float("nan"), docs_pilot=float(np.mean(pil)) if pil else float("nan"),
                                     docs_sampling=float(np.mean(docs[m]) - np.mean(pil)) if docs[m] else float("nan"),
                                     dup_frac=float(np.mean(dup[m])) if dup[m] else float("nan"),
                                     design_obj=float(np.nanmean(obj[m])) if obj[m] else float("nan"),
                                     nonoverlap=float(np.mean(overlap)) if overlap else float("nan"),
                                     pilot_cost=a.pilot_cost, bound=a.bound, legacy=a.legacy))
                print(f"{held} B={Bq}q eps={eps}: " + " ".join(f"{m}={cnt[m][0]/a.draws:.2f}" for m in arms)
                      + f" | docs static={np.mean(docs.get('static_sum', [np.nan])):.0f} pilot={np.mean(pil) if pil else float('nan'):.0f}"
                      + f" | obj static/oracle_exact={np.nanmean(obj.get('static_sum', [np.nan]))/max(np.nanmean(obj.get('oracle_exact', [np.nan])), 1e-12):.2f}"
                      + f" | nonoverlap={np.mean(overlap) if overlap else float('nan'):.2f} | wrong max {max(v[1] for v in cnt.values())/a.draws:.3f}", flush=True)
    import pandas as pd
    out = os.path.join(HUB, "05_results", "menu_allocation"); os.makedirs(out, exist_ok=True)
    suffix = "" if a.legacy else f"_v2_{a.pilot_cost}_{a.bound}"
    pd.DataFrame(rows).to_csv(os.path.join(out, f"menu_alloc_{a.stack}_{a.judge}{suffix}{a.tag}.csv"), index=False)


if __name__ == "__main__":
    main()
