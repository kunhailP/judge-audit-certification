#!/usr/bin/env python3
"""Fixed-budget retrospective study (Phase 2, C1 evidence).

Re-runs the Block 1 probe protocol over a budget grid k in {5,10,20,50,100}
and preserves repeat-level rows per CONTRACTS.md (per_repeat + decision means,
probe/evaluation query ids, seeds). The LODO training, policy definitions and
selection rule are byte-for-byte the same math as shift-study
src/block1_confirm.py at commit ef825e1d (verified byte-identical reproduction
on 2026-08-25); only the probe size k varies and outputs are row-level.

Seeds: for k=10 the legacy formula (1000*rep + crc32(dataset)%1000) is reused so
the first 20 repeats reproduce Block 1 draws exactly; other k use a disjoint
crc32-derived range. Python hash() is never used.

Provisional development thresholds (config v0.2 not yet frozen; inherited from
Block 1 pre-registration, recorded in RUN_NOTES.md):
  recalibration success: |mean_ev(ad_probe) - mean_ev(ad_c)| <= 0.005
  selection success:     regret(select3) <= 0.01
"""
import argparse, csv, json, os, zlib
import numpy as np
from collections import defaultdict

FEATS = ["score_norm", "rank", "consensus", "lexoverlap"]
ALL_DATASETS = ["trec-covid", "webis-touche2020", "dbpedia-entity", "nfcorpus",
                "scifact", "arguana", "scidocs", "fiqa", "quora", "nq",
                "hotpotqa", "fever", "climate-fever"]
UNTOUCHED = {"nq", "hotpotqa", "fever", "climate-fever"}
K_GRID = [5, 10, 20, 50, 100]
CORE_GRID = [5, 10, 20]
REPEATS = 50
MAX_TRAIN = 400_000
TOPP = 30
TAUS = np.linspace(0.02, 0.6, 30)
EPS_CAL = 0.005
EPS_SEL = 0.01

# Utility selected via --utility (M7 sensitivity). "set_f1" reproduces the
# primary study; recall/ndcg rerun the identical protocol under a different
# per-query utility. Thresholds keep the same numeric value on each utility's
# own scale (documented caveat in RUN_NOTES).
UTILITY = "set_f1"
IDEAL_DCG = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(2, 2002)))]


def util_at_k(s, k):
    if k <= 0:
        return 0.0
    if UTILITY == "set_f1":
        tp = s["cumtp"][k - 1]
        return 2.0 * tp / (k + s["nG"]) if tp > 0 else 0.0
    if UTILITY == "recall":
        return float(s["cumtp"][k - 1]) / s["nG"]
    if UTILITY == "ndcg":
        ideal = IDEAL_DCG[min(s["nG"], k)]
        return float(s["cumdcg"][k - 1]) / ideal if ideal > 0 else 0.0
    raise ValueError(UTILITY)


def f1_at_k(cumtp, k, nG):
    if k <= 0:
        return 0.0
    tp = cumtp[k - 1]
    return 2.0 * tp / (k + nG) if tp > 0 else 0.0


def gate_k(sp, cump, R):
    if R <= 0:
        return 0
    bad = np.flatnonzero(sp * (np.arange(len(sp)) + R + 1.0) <= cump)
    return int(bad[0]) if len(bad) else len(sp)


def load(cand_dir):
    data = {}
    for name in ALL_DATASETS:
        fp = os.path.join(cand_dir, f"{name}.csv")
        if not os.path.exists(fp):
            continue
        qids, X, rel = [], [], []
        with open(fp) as f:
            for r in csv.DictReader(f):
                qids.append(r["qid"])
                X.append([float(r[k]) for k in FEATS])
                rel.append(int(float(r["relevant"])))
        X = np.asarray(X, np.float32)
        rel = np.asarray(rel, np.int8)
        qids = np.asarray(qids)
        nG = {r["qid"]: int(r["nG"]) for r in
              csv.DictReader(open(os.path.join(cand_dir, f"{name}_meta.csv")))}
        order = np.argsort(qids, kind="stable")
        X, rel, qids = X[order], rel[order], qids[order]
        bounds = np.flatnonzero(np.r_[1, qids[1:] != qids[:-1], 1])
        slices = [(qids[a], a, b) for a, b in zip(bounds[:-1], bounds[1:])
                  if nG.get(qids[a], 0) > 0]
        data[name] = dict(X=X, rel=rel, slices=slices, nG=nG)
    return data


def build_structs(data, name, P):
    d = data[name]
    out = []
    for qid, a, b in d["slices"]:
        p = P[a:b]
        o = np.argsort(-p)
        sp = p[o].astype(np.float64)
        srel = d["rel"][a:b][o]
        cumtp = np.cumsum(srel)
        cump = np.cumsum(sp)
        cumdcg = np.cumsum(srel / np.log2(np.arange(len(sp)) + 2.0))
        ks = np.searchsorted(-sp, -TAUS, side="right")
        nG = d["nG"][qid]
        s = dict(
            qid=qid, sp=sp, cumtp=cumtp, cump=cump, cumdcg=cumdcg,
            Sp=float(cump[-1]),
            nG=nG, tau_ks=ks, pool=len(sp),
            feat=list(sp[:TOPP]) + [0.0] * max(0, TOPP - len(sp)) +
                 [float(cump[-1]), float(np.log1p(len(sp))),
                  float(sp.mean()), float(sp.std()), float(sp.max())])
        s["k_star"] = int(np.argmax([util_at_k(s, k + 1)
                                     for k in range(len(sp))])) + 1
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/root/shift-study",
                    help="shift-study checkout containing runs/candidates/")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <hub>/05_results/budget_curves)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--utility", choices=["set_f1", "recall", "ndcg"],
                    default="set_f1")
    args = ap.parse_args()
    global UTILITY
    UTILITY = args.utility

    hub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    suffix = "" if args.utility == "set_f1" else f"_{args.utility}"
    out_dir = args.out or os.path.join(hub, "05_results", f"budget_curves{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingRegressor

    cand_dir = os.path.join(args.source, "runs", "candidates")
    data = load(cand_dir)
    avail = list(data)
    print(f"datasets: {avail}", flush=True)
    rng_global = np.random.default_rng(0)

    rows = []          # per (dataset, k, repeat): all policy eval means
    decisions = []     # per (dataset, k, repeat, decision): contract-style row

    for held in avail:
        tr_idx = [d for d in avail if d != held]
        Xtr = np.concatenate([data[d]["X"] for d in tr_idx])
        ytr = np.concatenate([data[d]["rel"] for d in tr_idx])
        if len(Xtr) > MAX_TRAIN:
            idx = rng_global.choice(len(Xtr), MAX_TRAIN, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        sc = StandardScaler().fit(Xtr)
        clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000),
                                     method="sigmoid", cv=3
                                     ).fit(sc.transform(Xtr), ytr)
        structs = {d: build_structs(data, d,
                                    clf.predict_proba(sc.transform(data[d]["X"]))[:, 1])
                   for d in avail}
        # tau transfer + trunc training on LODO datasets (identical to legacy)
        tau_scores = np.zeros(len(TAUS))
        n_tr_q = 0
        Xk, yk = [], []
        for d in tr_idx:
            for s in structs[d]:
                for ti in range(len(TAUS)):
                    tau_scores[ti] += util_at_k(s, int(s["tau_ks"][ti]))
                n_tr_q += 1
                Xk.append(s["feat"])
                yk.append(s["k_star"])
        tau_transfer_i = int(np.argmax(tau_scores / n_tr_q))
        reg = HistGradientBoostingRegressor(random_state=0).fit(
            np.asarray(Xk), np.asarray(yk))
        H = structs[held]
        k_trunc = np.clip(np.round(reg.predict(np.asarray([s["feat"] for s in H]))),
                          1, [len(s["sp"]) for s in H]).astype(int)
        c_star = float(np.median([s["nG"] / s["Sp"] for s in H if s["Sp"] > 0]))
        for s, kt in zip(H, k_trunc):
            s["f_ad"] = util_at_k(s, gate_k(s["sp"], s["cump"], s["Sp"]))
            s["f_adc"] = util_at_k(s, gate_k(s["sp"], s["cump"], c_star * s["Sp"]))
            s["f_glob"] = util_at_k(s, int(s["tau_ks"][tau_transfer_i]))
            s["f_trunc"] = util_at_k(s, int(kt))
            s["f_tau"] = np.array([util_at_k(s, int(s["tau_ks"][ti]))
                                   for ti in range(len(TAUS))])
        salt_legacy = zlib.crc32(held.encode()) % 1000

        for k_req in K_GRID:
            for rep in range(args.repeats):
                if k_req == 10:
                    seed = 1000 * rep + salt_legacy
                else:
                    seed = (2_000_000 + 100_000 * K_GRID.index(k_req)
                            + 1000 * rep + zlib.crc32(f"{held}|{k_req}".encode()) % 1000)
                rng = np.random.default_rng(seed)
                n_pr = min(k_req, len(H) // 2)
                pidx = set(rng.choice(len(H), n_pr, replace=False).tolist())
                probe = [H[i] for i in sorted(pidx)]
                ev = [s for i, s in enumerate(H) if i not in pidx]
                c_hat = float(np.median([s["nG"] / s["Sp"] for s in probe if s["Sp"] > 0]))
                tau_probe_i = int(np.argmax(
                    np.mean([s["f_tau"] for s in probe], axis=0)))

                def adp(s):
                    return util_at_k(s, gate_k(s["sp"], s["cump"], c_hat * s["Sp"]))

                def glp(s):
                    return float(s["f_tau"][tau_probe_i])

                menu = [("glob_probe", glp), ("trunc", lambda s: s["f_trunc"]),
                        ("ad_probe", adp)]
                pv = {n: np.array([f(s) for s in probe]) for n, f in menu}
                probe_means = {n: float(v.mean()) for n, v in pv.items()}
                probe_lcb = {n: float(v.mean() - v.std(ddof=1) / np.sqrt(len(v)))
                             for n, v in pv.items()}
                pick3 = max(menu, key=lambda nf: probe_lcb[nf[0]])[0]
                pick2 = ("ad_probe" if probe_means["ad_probe"] >= probe_means["glob_probe"]
                         else "glob_probe")
                fns = dict(menu)
                ev_means = {nm: float(np.mean([fns[nm](s) for s in ev]))
                            for nm in ["ad_probe", "glob_probe", "trunc"]}
                ev_static = dict(
                    ad=float(np.mean([s["f_ad"] for s in ev])),
                    ad_c=float(np.mean([s["f_adc"] for s in ev])),
                    glob=float(np.mean([s["f_glob"] for s in ev])))
                regret3 = max(ev_means.values()) - ev_means[pick3]
                cal_loss = abs(ev_means["ad_probe"] - ev_static["ad_c"])
                pair_pool = int(sum(s["pool"] for s in probe))
                probe_qids = [s["qid"] for s in probe]

                rows.append(dict(
                    collection=held, k_requested=k_req, k_effective=n_pr,
                    clipped=n_pr < k_req, repeat_id=rep, seed=seed,
                    untouched=held in UNTOUCHED,
                    c_hat=c_hat, c_star=c_star,
                    **{f"ev_{n}": v for n, v in ev_means.items()},
                    **{f"ev_{n}": v for n, v in ev_static.items()},
                    select3=ev_means[pick3], select2=ev_means[pick2],
                    pick3=pick3, pick2=pick2, regret3=regret3,
                    cal_loss=cal_loss, pair_pool_budget=pair_pool,
                    n_eval=len(ev),
                    probe_query_ids=json.dumps(probe_qids)))
                decisions.append(dict(
                    collection=held, repeat_id=rep, decision="recalibration",
                    final_T=n_pr, final_B_pool=pair_pool, k_requested=k_req,
                    selected_policy="ad_probe", reference_policy="ad_c(full_target_scalar_reference)",
                    loss=cal_loss, success=cal_loss <= EPS_CAL,
                    param_error=abs(c_hat - c_star), seed=seed))
                decisions.append(dict(
                    collection=held, repeat_id=rep, decision="selection",
                    final_T=n_pr, final_B_pool=pair_pool, k_requested=k_req,
                    selected_policy=pick3, reference_policy="best_in_menu_on_eval",
                    loss=regret3, success=regret3 <= EPS_SEL,
                    param_error=None, seed=seed))
            done = [r for r in rows if r["collection"] == held and r["k_requested"] == k_req]
            print(f"{held:16} k={k_req:3d} (eff={done[-1]['k_effective']:3d}) "
                  f"cal_loss={np.mean([r['cal_loss'] for r in done]):.4f} "
                  f"regret={np.mean([r['regret3'] for r in done]):.4f} "
                  f"sel_ok={np.mean([r['regret3'] <= EPS_SEL for r in done]):.2f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    dd = pd.DataFrame(decisions)
    df.to_parquet(os.path.join(out_dir, "per_repeat.parquet"), index=False)
    df.drop(columns=["probe_query_ids"]).to_csv(
        os.path.join(out_dir, "per_repeat.csv"), index=False)
    dd.to_parquet(os.path.join(out_dir, "decision_trace.parquet"), index=False)
    dd.to_csv(os.path.join(out_dir, "decision_trace.csv"), index=False)
    cfg = dict(utility=args.utility,
               k_grid=K_GRID, core_grid=CORE_GRID, repeats=args.repeats,
               eps_cal=EPS_CAL, eps_sel=EPS_SEL, max_train=MAX_TRAIN,
               taus=[float(t) for t in TAUS], feats=FEATS,
               source=args.source,
               source_commit="ef825e1d35d58d178c50054122ccc9e9ab54012c",
               legacy_seed_compat="k=10 uses legacy seed formula (Block 1 draws)",
               thresholds_provenance="Block 1 pre-registered SUFFICIENCY/SAFETY thresholds; provisional until v0.2 freeze")
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print(f"[saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
