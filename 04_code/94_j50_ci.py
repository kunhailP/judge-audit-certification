#!/usr/bin/env python3
"""Bootstrap uncertainty for J50 and for the savings ratios of Table 2 (2026-09-17).

Input: the per-draw files written by `81_sampling_baselines.py --dump_draws` (and, when present, by 82 with the same
flag): one row per (collection, judge, budget_full_eq, eps, method, draw) with the unique-label cost `docs`, the
certificate outcome `act` and `wrong`. Draws are paired across arms (same permutation, pilot and candidate inside a
draw), so a bootstrap that resamples *draw indices* and applies the same indices to every arm keeps that pairing and
gives an honest interval for a ratio of two J50 values.

For every bootstrap replicate: resample draws with replacement inside each (budget, eps) cell, recompute the ACT
curve and the mean cost per budget, interpolate J50 as 86_table2_v2.py does, and form the ratios
    uniform -> uniform_nz        (skipping documents that cannot move any comparison)
    uniform_nz -> weighted       (sampling the relevant documents in proportion to |w|)
    uniform -> weighted          (both together; the "-40 to -56%" of the manuscript)
    weighted -> weighted_cvl     (the judge as a lambda-fitted control variate)
    weighted -> <any other arm>  (on request, --pairs)
Reported: point estimate on the original draws, bootstrap percentile 95% interval, and the Monte-Carlo standard error.

Usage: python3 94_j50_ci.py [--eps 0.02] [--boot 1000] [--pairs uniform:weighted weighted:weighted_cvl ...]
Output: 05_results/unified/J50_CI_eps{eps}.csv and a printed table.
"""
import argparse, glob, os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "05_results")
DEFAULT_PAIRS = ["uniform:uniform_nz", "uniform_nz:weighted", "uniform:weighted", "weighted:weighted_cvl", "weighted:weighted_cv"]


def j50(x, y, target=0.5):
    """Same interpolation as 86_table2_v2.py: first budget at which the ACT curve crosses `target`; nan if never."""
    x = np.asarray(x, float); y = np.asarray(y, float); o = np.argsort(x); x, y = x[o], y[o]
    if y[0] >= target:
        return float(x[0])
    for i in range(1, len(x)):
        if y[i] >= target:
            return float(x[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return float("nan")


def curves(cell, idx_by_budget):
    """cell: rows of one (collection, judge, eps, method) pivoted to arrays per budget. Returns J50 for resampled draws."""
    xs, ys = [], []
    for b, (docs, act) in cell.items():
        idx = idx_by_budget[b]
        xs.append(docs[idx].mean()); ys.append(act[idx].mean())
    return j50(xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="from:to arm pairs; saving = 1 - J50(to)/J50(from)")
    ap.add_argument("--glob", default=os.path.join(R, "sampling_baselines", "baselines_*_draws.csv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob)) + sorted(glob.glob(os.path.join(R, "active_inference", "active_*_draws.csv")))
    if not files:
        raise SystemExit("no *_draws.csv files: re-run 81/82 with --dump_draws")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.eps == a.eps]
    rng = np.random.default_rng(a.seed); rows = []
    for (c, j), g in d.groupby(["collection", "judge"]):
        # arrays per (method, budget); every arm of a draw shares the draw index, so a common index set keeps pairing
        cells = {}
        for m, gm in g.groupby("method"):
            cells[m] = {b: (gb.sort_values("draw").docs.values.astype(float), gb.sort_values("draw").act.values.astype(float))
                        for b, gb in gm.groupby("budget_full_eq")}
        budgets = sorted(set(b for m in cells for b in cells[m]))
        n_by_b = {b: min(len(cells[m][b][0]) for m in cells if b in cells[m]) for b in budgets}
        point = {m: curves(cells[m], {b: np.arange(n_by_b[b]) for b in cells[m]}) for m in cells}
        boots = {m: [] for m in cells}
        for _ in range(a.boot):
            idx = {b: rng.integers(0, n_by_b[b], n_by_b[b]) for b in budgets}
            for m in cells:
                boots[m].append(curves(cells[m], {b: idx[b] for b in cells[m]}))
        boots = {m: np.array(v) for m, v in boots.items()}
        for m in cells:
            v = boots[m]; ok = np.isfinite(v)
            rows.append(dict(collection=c, judge=j, eps=a.eps, quantity=f"J50[{m}]", point=point[m],
                             lo=np.nanpercentile(v, 2.5) if ok.any() else np.nan, hi=np.nanpercentile(v, 97.5) if ok.any() else np.nan,
                             mc_se=float(np.nanstd(v)), n_reached=int(ok.sum()), boot=a.boot))
        for pair in a.pairs:
            f, t = pair.split(":")
            if f not in cells or t not in cells:
                continue
            s = 1.0 - boots[t] / boots[f]; ok = np.isfinite(s)
            rows.append(dict(collection=c, judge=j, eps=a.eps, quantity=f"saving {f}->{t}", point=1.0 - point[t] / point[f],
                             lo=np.nanpercentile(s, 2.5) if ok.any() else np.nan, hi=np.nanpercentile(s, 97.5) if ok.any() else np.nan,
                             mc_se=float(np.nanstd(s)), n_reached=int(ok.sum()), boot=a.boot))
    out = pd.DataFrame(rows); os.makedirs(os.path.join(R, "unified"), exist_ok=True)
    out.to_csv(os.path.join(R, "unified", f"J50_CI_eps{a.eps}.csv"), index=False)
    pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
