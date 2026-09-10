#!/usr/bin/env python3
"""Planner replay v0.4 — certificate variants on the two-stack pools.

Methods (selection decision):
  loo_boot  : v0.3 legacy — LOO cross-fitted utilities, paired bootstrap,
              Bonferroni over looks x (M-1)            [baseline, empirical validity]
  split_t   : train half fits (c_hat, tau); validation half gives paired
              query differences; one-sided t UCBs, Bonferroni over looks x M(M-1)
  split_ppi : split_t + PPI++ using AI-judge utilities (Qwen3-Reranker P(yes)
              thresholded at 0.5) on every target query
Recalibration decision: breakpoint-scan certificate (v0.4) vs endpoint (v0.3).

True losses are computed on the never-audited complement of the audit set.
Judge diagnostics per collection: pair-level accuracy vs qrels, rho between
true and judge paired policy differences on the population, and
error rates inside vs outside the policy-disagreement band.

Usage: python3 63_planner_v2.py --pools <dir> --stack legacy|modern [--repeats 50]
"""
import argparse, csv, importlib.util, json, os, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("bc", os.path.join(HERE, "20_budget_curves.py"))
bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
spec = importlib.util.spec_from_file_location("cert", os.path.join(HERE, "lib", "certificates.py"))
cert = importlib.util.module_from_spec(spec); spec.loader.exec_module(cert)
spec = importlib.util.spec_from_file_location("nt", os.path.join(HERE, "lib", "neutrality.py"))
nt = importlib.util.module_from_spec(spec); spec.loader.exec_module(nt)
PILOT_MIN, GAIN_MIN = 30, 1.1        # judge-adoption rule, fixed at development time (LOCK v0.4 draft)

NAMES = ["nfcorpus", "scifact", "arguana", "cqadupstack-android"]
ALPHA, EPS_CAL, EPS_SEL = 0.10, 0.005, 0.01
LOOKS = [10, 30, 50, 70, 90]
BOOT = 400
MENU = ["glob_probe", "trunc", "ad_probe"]
JUDGE_THR = 0.5


JUDGE = "rr"   # "rr" = Qwen3-Reranker P(yes) column; "llm" = <name>_llm.csv side file (llm_p_rel)


def load_with_judge(cand_dir):
    """bc.load ordering (stable sort by qid) plus aligned judge column."""
    data = {}
    for name in NAMES:
        fp = os.path.join(cand_dir, f"{name}.csv")
        if not os.path.exists(fp):
            continue
        side = {}
        if JUDGE not in ("rr", "inv"):
            for r in csv.DictReader(open(os.path.join(cand_dir, f"{name}_{JUDGE}.csv"))):
                side[(r["qid"], r["docid"])] = float(r[f"{JUDGE}_p_rel"])
        qids, X, rel, rr = [], [], [], []
        with open(fp) as f:
            for r in csv.DictReader(f):
                qids.append(r["qid"]); X.append([float(r[k]) for k in bc.FEATS])
                rel.append(int(float(r["relevant"])))
                rr.append(side[(r["qid"], r["docid"])] if JUDGE not in ("rr", "inv")
                          else (1.0 - float(r["rr_yes"]) if JUDGE == "inv" else float(r["rr_yes"])))
        X = np.asarray(X, np.float32); rel = np.asarray(rel, np.int8); rr = np.asarray(rr, np.float32)
        qids = np.asarray(qids)
        nG = {r["qid"]: int(r["nG"]) for r in csv.DictReader(open(os.path.join(cand_dir, f"{name}_meta.csv")))}
        order = np.argsort(qids, kind="stable")
        X, rel, rr, qids = X[order], rel[order], rr[order], qids[order]
        bounds = np.flatnonzero(np.r_[1, qids[1:] != qids[:-1], 1])
        slices = [(qids[a], a, b) for a, b in zip(bounds[:-1], bounds[1:]) if nG.get(qids[a], 0) > 0]
        jrel = (rr >= JUDGE_THR).astype(np.int8)
        jnG = {q: int(jrel[a:b].sum()) for q, a, b in slices}
        rr_raw = np.array([float(r["rr_yes"]) for r in csv.DictReader(open(fp))], np.float32)[order]  # always the reranker
        data[name] = dict(X=X, rel=rel, slices=slices, nG=nG, rr=rr, rr_raw=rr_raw,
                          judge=dict(X=X, rel=jrel, slices=slices, nG=jnG, rr_raw=rr_raw))
    return data


TRAIN_ONLY = set()


def load_train_only(train_dir, train_names):
    """Extra collections used only to fit the classifier / tau / trunc regressor."""
    global NAMES, TRAIN_ONLY
    saved = NAMES; NAMES = train_names
    extra = load_with_judge(train_dir)
    NAMES = saved; TRAIN_ONLY = set(extra)
    return extra


RR_GRID = np.linspace(0.05, 0.95, 19)
MENU4 = ["glob_probe", "trunc", "ad_probe", "rr_thresh"]


def attach_rr(structs, d, P):
    """Per-doc Qwen3-Reranker score in the struct's (descending P) order, plus set-F1 of the
    reranker-threshold policy on the RR_GRID (true relevance from cumtp)."""
    idx = 0
    slices = [(q, a, b) for q, a, b in d["slices"]]
    by_q = {q: (a, b) for q, a, b in slices}
    for s in structs:
        a, b = by_q[s["qid"]]
        p = P[a:b]; o = np.argsort(-p)
        s["rr_sorted"] = d["rr_raw"][a:b][o]
        rel = np.diff(np.r_[0, s["cumtp"]])
        f = []
        for th in RR_GRID:
            m = s["rr_sorted"] >= th; k = int(m.sum()); tp = int((rel * m).sum())
            f.append(2.0 * tp / (k + s["nG"]) if tp > 0 else 0.0)
        s["f_rr"] = np.array(f)
    return structs


def fit_lodo(data, held, rng_global):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    avail = list(data); tr_idx = [d for d in avail if d != held]
    Xtr = np.concatenate([data[d]["X"] for d in tr_idx]); ytr = np.concatenate([data[d]["rel"] for d in tr_idx])
    if len(Xtr) > bc.MAX_TRAIN:
        idx = rng_global.choice(len(Xtr), bc.MAX_TRAIN, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler().fit(Xtr)
    clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="sigmoid", cv=3).fit(sc.transform(Xtr), ytr)
    P = {d: clf.predict_proba(sc.transform(data[d]["X"]))[:, 1] for d in avail}
    structs = {d: bc.build_structs(data, d, P[d]) for d in avail}
    tau_scores = np.zeros(len(bc.TAUS)); n_tr_q = 0; Xk, yk = [], []
    for d in tr_idx:
        for s in structs[d]:
            for ti in range(len(bc.TAUS)):
                tau_scores[ti] += bc.f1_at_k(s["cumtp"], int(s["tau_ks"][ti]), s["nG"])
            n_tr_q += 1; Xk.append(s["feat"]); yk.append(s["k_star"])
    reg = HistGradientBoostingRegressor(random_state=0).fit(np.asarray(Xk), np.asarray(yk))
    return P, structs, reg


def finalize(H, reg):
    k_trunc = np.clip(np.round(reg.predict(np.asarray([s["feat"] for s in H]))), 1,
                      [len(s["sp"]) for s in H]).astype(int)
    for s, kt in zip(H, k_trunc):
        s["k_trunc"] = int(kt)
        s["f_trunc"] = bc.f1_at_k(s["cumtp"], int(kt), s["nG"])
        s["f_tau"] = np.array([bc.f1_at_k(s["cumtp"], int(s["tau_ks"][ti]), s["nG"]) for ti in range(len(bc.TAUS))])
    c_star = float(np.median([s["nG"] / s["Sp"] for s in H if s["Sp"] > 0]))
    for s in H:
        s["f_adc"] = bc.f1_at_k(s["cumtp"], bc.gate_k(s["sp"], s["cump"], c_star * s["Sp"]), s["nG"])
    return c_star


def utils(structs, c_hat, tau_i, th_i=None):
    ad = np.array([bc.f1_at_k(s["cumtp"], bc.gate_k(s["sp"], s["cump"], c_hat * s["Sp"]), s["nG"]) for s in structs])
    gl = np.array([float(s["f_tau"][tau_i]) for s in structs])
    tr = np.array([s["f_trunc"] for s in structs])
    cols = [gl, tr, ad]
    if th_i is not None:
        cols.append(np.array([float(s["f_rr"][th_i]) for s in structs]))
    return np.stack(cols, axis=1)          # MENU order (+ rr_thresh)


def fit_params(probe):
    ratios = np.array([s["nG"] / s["Sp"] for s in probe if s["Sp"] > 0])
    c_hat = float(np.median(ratios)) if len(ratios) else 1.0
    tau_i = int(np.argmax(np.mean([s["f_tau"] for s in probe], axis=0)))
    return c_hat, tau_i, ratios


def fit_theta(probe):
    return int(np.argmax(np.mean([s["f_rr"] for s in probe], axis=0)))


def menu_utils(structs, c_hat, tau_i, th_i):
    return utils(structs, c_hat, tau_i, th_i if len(MENU) == 4 else None)


def cutoffs(s, name, c, tau_i):
    if name == "rr_thresh":
        return int((s["rr_sorted"] >= RR_GRID[fit_theta([s])]).sum())   # diagnostic only
    if name == "ad_probe":
        return bc.gate_k(s["sp"], s["cump"], c * s["Sp"])
    if name == "glob_probe":
        return int(s["tau_ks"][tau_i])
    return s["k_trunc"]


def judge_diagnostics(H, HJ, rr_pairs, rel_pairs, c_star, tau_i):
    """Population-level judge diagnostics for the frozen menu at (c_star, tau_i):
    pair-level accuracy, and for every policy pair the paired-difference
    correlation rho and the judge error rate inside vs outside the
    disagreement band (docs ranked between the two cutoffs)."""
    th_i = fit_theta(H) if len(MENU) == 4 else None
    U = utils(H, c_star, tau_i, th_i); Uh = utils(HJ, c_star, tau_i, th_i)
    out = dict(judge_acc=float(((rr_pairs >= JUDGE_THR).astype(int) == rel_pairs).mean()))
    for a in range(len(MENU)):
        for b in range(a + 1, len(MENU)):
            d, dh = U[:, a] - U[:, b], Uh[:, a] - Uh[:, b]
            key = f"{MENU[a]}_vs_{MENU[b]}"
            out[f"rho_{key}"] = float(np.corrcoef(d, dh)[0, 1]) if d.std() > 0 and dh.std() > 0 else float("nan")
            inb, outb = [], []
            for s, sj in zip(H, HJ):
                ka, kb = cutoffs(s, MENU[a], c_star, tau_i), cutoffs(s, MENU[b], c_star, tau_i)
                lo, hi = min(ka, kb), max(ka, kb)
                r_true = np.diff(np.r_[0, s["cumtp"]]); r_j = np.diff(np.r_[0, sj["cumtp"]])
                err = r_true != r_j
                inb.extend(err[lo:hi].tolist()); outb.extend(np.r_[err[:lo], err[hi:]].tolist())
            out[f"err_in_{key}"] = float(np.mean(inb)) if inb else float("nan")
            out[f"err_out_{key}"] = float(np.mean(outb)) if outb else float("nan")
            out[f"band_frac_{key}"] = len(inb) / max(len(inb) + len(outb), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True); ap.add_argument("--stack", required=True)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", nargs="+", default=None, help="collections (default: BEIR four)")
    ap.add_argument("--judge", default="rr", help="rr | inv (1-rr, adversarial control) | llm | mpnet | any <name>_<judge>.csv")
    ap.add_argument("--looks", nargs="+", type=int, default=None)
    ap.add_argument("--lambda_finite_n", action="store_true", help="post-hoc deviation: PPI++ lambda shrunk by 1/(1+n/N)")
    ap.add_argument("--lambda_guard", action="store_true", help="post-hoc deviation: lambda=0 unless fitting-fold variance drops >= 5%")
    ap.add_argument("--menu4", action="store_true", help="add rr_thresh (Qwen3-Reranker threshold policy) to the menu")
    ap.add_argument("--train_dir", default=None, help="candidates dir of training-only collections")
    ap.add_argument("--train_names", nargs="+", default=["nfcorpus", "scifact", "arguana", "cqadupstack-android"])
    a = ap.parse_args()
    global NAMES, JUDGE, MENU, LOOKS
    JUDGE = a.judge
    if a.looks:
        LOOKS = a.looks
    cert.LAMBDA_FINITE_N = bool(a.lambda_finite_n); cert.LAMBDA_GUARD = bool(a.lambda_guard)
    if a.menu4:
        MENU = MENU4
    if a.names:
        NAMES = a.names
    out = a.out or os.path.join(HUB, "05_results", "planner_v2", a.stack + ("" if a.judge == "rr" else "_" + a.judge) + ("_menu4" if a.menu4 else "")); os.makedirs(out, exist_ok=True)
    cand_dir = os.path.join(a.pools, a.stack, "runs", "candidates")
    data = load_with_judge(cand_dir)
    if a.train_dir:
        saved_judge = JUDGE; JUDGE = "rr"          # training pools need no LLM side file
        data.update(load_train_only(a.train_dir, a.train_names)); JUDGE = saved_judge
    rng_global = np.random.default_rng(0)
    rows, diags = [], {}
    a_look = ALPHA / len(LOOKS)
    a_sel_legacy = cert.bonferroni_alpha(ALPHA, len(LOOKS), len(MENU), simultaneous=False)
    a_sel_sim = cert.bonferroni_alpha(ALPHA, len(LOOKS), len(MENU), simultaneous=True)

    for held in [d for d in data if d not in TRAIN_ONLY]:
        P, structs, reg = fit_lodo(data, held, rng_global)
        H = structs[held]; c_star = finalize(H, reg); attach_rr(H, data[held], P[held])
        # judge structs: same P ordering, judge relevance and judge nG
        jd = {held: data[held]["judge"]}
        HJ = bc.build_structs(jd, held, P[held]); finalize(HJ, reg); attach_rr(HJ, data[held]["judge"], P[held])
        assert [s["qid"] for s in H] == [s["qid"] for s in HJ]
        cert.prepare_gate(H)
        n = len(H); cap = n // 2
        looks = [t for t in LOOKS if t <= cap] or [cap]
        _, tau_star, _ = fit_params(H)
        diags[held] = judge_diagnostics(H, HJ, data[held]["rr"], data[held]["rel"], c_star, tau_star)
        diags[held].update(n_queries=n, c_star=c_star)
        print(held, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in diags[held].items()}), flush=True)
        salt = zlib.crc32(f"planner|{held}".encode()) % 100_000
        # population judge utilities need (c_hat, tau_i) -> computed per look (cheap: n x M)
        for rep in range(a.repeats):
            rng = np.random.default_rng(5_000_000 + 1000 * rep + salt)
            perm = rng.permutation(n)
            # independent bootstrap streams per method so adding a method never perturbs another's numbers
            rng_m = {m: np.random.default_rng([5_000_000 + 1000 * rep + salt, zlib.crc32(m.encode())])
                     for m in ["recal", "loo_boot", "loo_sim"]}
            state = {m: None for m in ["recal_bp", "recal_ep", "recal_bpx", "recal_bpu", "recal_bpxu", "loo_boot", "loo_sim", "split_t", "split_ppi", "split_auto"]}
            auto_used = None
            for T in looks:
                probe = [H[i] for i in perm[:T]]
                c_hat, tau_i, ratios = fit_params(probe); th_i = fit_theta(probe)
                # ---- recalibration (full probe) ----
                if any(state[k] is None for k in ["recal_bp", "recal_ep", "recal_bpx", "recal_bpu", "recal_bpxu"]):
                    bs = rng_m["recal"].integers(0, len(ratios), (BOOT, len(ratios)))
                    bmed = np.median(ratios[bs], axis=1)
                    c_lo, c_hi = np.quantile(bmed, [a_look / 2, 1 - a_look / 2])      # v0.3 bootstrap CI
                    x_lo, x_hi = cert.median_ci_exact(ratios, a_look)                 # exact order-stat CI
                    for key, mode, lo, hi in [("recal_ep", "endpoints", c_lo, c_hi),
                                              ("recal_bp", "breakpoints", c_lo, c_hi),
                                              ("recal_bpx", "breakpoints", x_lo, x_hi)]:
                        if state[key] is None and np.isfinite(lo) and np.isfinite(hi):
                            spread, npts = cert.recal_spread(probe, c_hat, lo, hi, mode=mode)
                            if spread <= EPS_CAL:
                                state[key] = ("act", T, dict(c_hat=c_hat, spread=spread, npts=npts))
                    for key, lo, hi in [("recal_bpu", c_lo, c_hi), ("recal_bpxu", x_lo, x_hi)]:
                        if state[key] is None and np.isfinite(lo) and np.isfinite(hi):
                            spread, npts = cert.recal_spread_ucb(probe, c_hat, lo, hi, a_look, rng_m["recal"])
                            if spread <= EPS_CAL:
                                state[key] = ("act", T, dict(c_hat=c_hat, spread=spread, npts=npts))
                # ---- legacy LOO bootstrap ----
                if state["loo_boot"] is None or state["loo_sim"] is None:
                    ratios_all = np.array([s["nG"] / s["Sp"] if s["Sp"] > 0 else np.nan for s in probe])
                    F = np.stack([s["f_tau"] for s in probe]); colsum = F.sum(axis=0)
                    FR = np.stack([s["f_rr"] for s in probe]); colsum_r = FR.sum(axis=0)
                    U = np.empty((T, len(MENU)))
                    for i, s in enumerate(probe):
                        others = np.delete(ratios_all, i); others = others[~np.isnan(others)]
                        c_loo = float(np.median(others)) if len(others) else 1.0
                        U[i, 2] = bc.f1_at_k(s["cumtp"], bc.gate_k(s["sp"], s["cump"], c_loo * s["Sp"]), s["nG"])
                        U[i, 0] = float(s["f_tau"][int(np.argmax(colsum - F[i]))])
                        U[i, 1] = s["f_trunc"]
                        if len(MENU) == 4:
                            U[i, 3] = float(s["f_rr"][int(np.argmax(colsum_r - FR[i]))])
                    cand = cert.pick_candidate(U)
                    if state["loo_boot"] is None:
                        ucb = cert.ucb_bootstrap(U, cand, a_sel_legacy, rng_m["loo_boot"], BOOT)
                        if ucb.max() <= EPS_SEL:
                            state["loo_boot"] = ("act", T, dict(pick=cand, c_hat=c_hat, tau_i=tau_i, th_i=th_i))
                    if state["loo_sim"] is None:
                        ucb = cert.ucb_bootstrap(U, cand, a_sel_sim, rng_m["loo_sim"], BOOT)
                        if ucb.max() <= EPS_SEL:
                            state["loo_sim"] = ("act", T, dict(pick=cand, c_hat=c_hat, tau_i=tau_i, th_i=th_i))
                # ---- split designs ----
                if state["split_t"] is None or state["split_ppi"] is None or state["split_auto"] is None:
                    ntr = T // 2
                    train = [H[i] for i in perm[:ntr]]; val = [H[i] for i in perm[ntr:T]]
                    c_tr, tau_tr, _ = fit_params(train); th_tr = fit_theta(train)
                    U = menu_utils(val, c_tr, tau_tr, th_tr)
                    cand = cert.pick_candidate(U)
                    if state["split_t"] is None:
                        ucb = cert.ucb_t(U, cand, a_sel_sim)
                        if ucb.max() <= EPS_SEL:
                            state["split_t"] = ("act", T, dict(pick=cand, c_hat=c_tr, tau_i=tau_tr, th_i=th_tr))
                    if state["split_ppi"] is None or state["split_auto"] is None:
                        Uh_all = menu_utils(HJ, c_tr, tau_tr, th_tr)
                        Uh_val = Uh_all[perm[ntr:T]]
                        ucb_p = cert.ucb_ppi(U, Uh_val, Uh_all, cand, a_sel_sim, exclude_idx=perm[:T])
                        if state["split_ppi"] is None and ucb_p.max() <= EPS_SEL:
                            state["split_ppi"] = ("act", T, dict(pick=cand, c_hat=c_tr, tau_i=tau_tr, th_i=th_tr))
                        if state["split_auto"] is None:
                            # judge-adoption rule evaluated on the TRAIN half only (pilot): candidate-vs-runner-up pair
                            Ut = menu_utils(train, c_tr, tau_tr, th_tr); Uh_tr = Uh_all[perm[:ntr]]
                            ct = cert.pick_candidate(Ut) if ntr >= 3 else 0
                            others = [j for j in range(Ut.shape[1]) if j != ct]
                            ru = max(others, key=lambda j: Ut[:, j].mean())
                            use = False
                            if ntr >= PILOT_MIN:
                                dtr, dhtr = Ut[:, ru] - Ut[:, ct], Uh_tr[:, ru] - Uh_tr[:, ct]
                                _, lcb = nt.rho_lcb_boot(dtr, dhtr, ALPHA, rng_m["recal"], boot=400)
                                use = (1.0 / (1 - lcb ** 2) if lcb > 0 else 1.0) > GAIN_MIN
                            ucb_a = ucb_p if use else cert.ucb_t(U, cand, a_sel_sim)
                            if auto_used is None:
                                auto_used = use
                            if ucb_a.max() <= EPS_SEL:
                                state["split_auto"] = ("act", T, dict(pick=cand, c_hat=c_tr, tau_i=tau_tr, th_i=th_tr, used_judge=use))
                if all(v is not None for v in state.values()):
                    break
            for key in state:
                if state[key] is None:
                    state[key] = ("abstain", looks[-1], dict(c_hat=c_hat, tau_i=tau_i, th_i=th_i))
            for key, (action, T, pl) in state.items():
                ev = [H[i] for i in perm[T:]]
                if key.startswith("recal"):
                    loss = abs(cert.u_gate_at(ev, pl["c_hat"])[0] - float(np.mean([s["f_adc"] for s in ev])))
                    eps = EPS_CAL; picked = "ad_probe"
                else:
                    Uev = menu_utils(ev, pl["c_hat"], pl["tau_i"], pl.get("th_i", 0)).mean(axis=0)
                    picked = MENU[pl["pick"]] if "pick" in pl else None
                    loss = float(Uev.max() - Uev[pl["pick"]]) if "pick" in pl else None
                    eps = EPS_SEL
                rows.append(dict(stack=a.stack, collection=held, repeat_id=rep, method=key, action=action,
                                 final_T=T, selected_policy=picked, loss=loss,
                                 wrong_cert=(action == "act" and loss is not None and loss > eps),
                                 used_judge=pl.get("used_judge", None)))
        sub = [r for r in rows if r["collection"] == held]
        for key in state:
            d = [r for r in sub if r["method"] == key]
            print(f"  {held:20} {key:10} act={np.mean([r['action']=='act' for r in d]):.2f} "
                  f"meanT={np.mean([r['final_T'] for r in d]):.1f} wrong={np.mean([r['wrong_cert'] for r in d]):.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows); df.to_parquet(os.path.join(out, "planner_v2.parquet"), index=False)
    g = df.groupby(["collection", "method"], as_index=False).agg(
        act_rate=("action", lambda s: float((s == "act").mean())), mean_T=("final_T", "mean"),
        wrong_cert_rate=("wrong_cert", "mean"))
    g.to_csv(os.path.join(out, "summary.csv"), index=False)
    json.dump(diags, open(os.path.join(out, "judge_diagnostics.json"), "w"), indent=2)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
