#!/usr/bin/env python3
"""Gate A / Gate B for the Featured direction (decision 2026-09-14, item 6), from the v2 menu-allocation results.

Per (collection, judge, eps, bound, pilot_cost) the script interpolates J50 (unique human labels, pilot included,
at which the ACT rate first reaches 0.5) from `menu_alloc_*_v2_*.csv` and computes

  saving_oracle  = 1 − J50(oracle_exact) / J50(baseline)                 baseline = static_sum (default) or static_sum_v
  saving_plugin  = 1 − J50(plugin_exact) / J50(baseline)
  recovery       = (J50(baseline) − J50(plugin_exact)) / (J50(baseline) − J50(oracle_exact))

Gate A (structural head-room): saving_oracle ≥ 0.30 on ≥ 2 collections AND median over collections ≥ 0.20.
Gate B (realisable):           recovery ≥ 0.60 on the Gate-A collections AND saving_plugin ≥ 0.15 on ≥ 2 collections
                               AND wrong-certificate rate of plugin_exact ≤ α on every cell.
Also reports the design-objective ratio static/oracle (bound-free structural head-room) as a diagnostic.
Usage: python3 85_gates.py [--eps 0.02] [--bound t|eb|bet] [--baseline static_sum|static_sum_v] [--judge llm]
"""
import argparse, glob, os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "05_results", "menu_allocation")
ALPHA = 0.10


def interp_first(xs, ys, target=0.5):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs)):
        if ys[i] >= target:
            return float(xs[0]) if i == 0 else float(xs[i - 1] + (target - ys[i - 1]) * (xs[i] - xs[i - 1]) / max(ys[i] - ys[i - 1], 1e-9))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--bound", default="t"); ap.add_argument("--pilot_cost", default="full")
    ap.add_argument("--baseline", default="static_sum", choices=["static_sum", "static_sum_v"]); ap.add_argument("--judge", default="llm")
    ap.add_argument("--min_saving", type=float, default=0.30); ap.add_argument("--min_median", type=float, default=0.20)
    ap.add_argument("--min_recovery", type=float, default=0.60); ap.add_argument("--min_actual", type=float, default=0.15)
    ap.add_argument("--gen", default="v3", choices=["v2", "v3"], help="v3 = population estimand runs (menu_alloc_*_v3b*.csv); v2 = superseded")
    a = ap.parse_args()
    if a.gen == "v3":
        files = sorted(glob.glob(os.path.join(R, f"menu_alloc_*_{a.judge}_v2_{a.pilot_cost}_{a.bound}_v3b*.csv")))
    else:
        files = sorted(glob.glob(os.path.join(R, f"menu_alloc_*_{a.judge}_v2_{a.pilot_cost}_{a.bound}.csv")))
    if not files:
        print("no v2 menu results found for", a.judge, a.pilot_cost, a.bound); return
    d = pd.concat([pd.read_csv(f) for f in files]); d = d[(d.eps == a.eps)]
    out = []
    for c, g in d.groupby("collection"):
        j50 = {}; j50x = {}; wrong = {}; obj = {}
        for m in ["static_sum", "static_sum_v", "oracle_exact", "plugin_exact", "per_pair", "adaptive", "oracle"]:
            s = g[g.method == m].sort_values("budget_full_eq")
            if s.empty:
                continue
            j50[m] = interp_first(s.docs_labelled.values, s.act.values); wrong[m] = float(s.wrong.max()); obj[m] = float(s.design_obj.mean())
            j50x[m] = interp_first((s.docs_labelled - s.docs_pilot).values, s.act.values)   # sampling labels only (diagnostic)
        base = j50.get(a.baseline, np.nan)
        so = 1 - j50.get("oracle_exact", np.nan) / base; sp = 1 - j50.get("plugin_exact", np.nan) / base
        rec = (base - j50.get("plugin_exact", np.nan)) / (base - j50.get("oracle_exact", np.nan)) if base - j50.get("oracle_exact", np.nan) > 0 else np.nan
        out.append(dict(collection=c, J50_base=base, J50_oracle=j50.get("oracle_exact"), J50_plugin=j50.get("plugin_exact"), J50_static_v=j50.get("static_sum_v"),
                        saving_oracle=so, saving_plugin=sp, recovery=rec, wrong_plugin=wrong.get("plugin_exact"),
                        saving_oracle_ex_pilot=1 - j50x.get("oracle_exact", np.nan) / j50x.get(a.baseline, np.nan),
                        obj_ratio_static_over_oracle=obj.get("static_sum", np.nan) / obj.get("oracle_exact", np.nan),
                        nonoverlap=float(g.nonoverlap.mean()), pilot_share=float((g.docs_pilot / g.docs_labelled).mean())))
    t = pd.DataFrame(out); pd.set_option("display.width", 200)
    print(f"eps={a.eps} bound={a.bound} pilot_cost={a.pilot_cost} baseline={a.baseline} judge={a.judge}")
    print(t.round(3).to_string(index=False))
    so = t.saving_oracle.dropna()
    gateA = (so >= a.min_saving).sum() >= 2 and (so.median() >= a.min_median if len(so) else False)
    print(f"\nGate A: collections with saving_oracle >= {a.min_saving:.0%}: {(so >= a.min_saving).sum()} (need >= 2); median saving {so.median() if len(so) else float('nan'):.1%} (need >= {a.min_median:.0%}) -> {'PASS' if gateA else 'FAIL'}")
    if gateA:
        ga = t[t.saving_oracle >= a.min_saving]
        gateB = (ga.recovery >= a.min_recovery).all() and (t.saving_plugin >= a.min_actual).sum() >= 2 and (t.wrong_plugin.fillna(0) <= ALPHA).all()
        print(f"Gate B: recovery on Gate-A collections {ga.recovery.round(2).tolist()} (need all >= {a.min_recovery:.0%}); collections with saving_plugin >= {a.min_actual:.0%}: {(t.saving_plugin >= a.min_actual).sum()} (need >= 2); max wrong(plugin) {t.wrong_plugin.max():.3f} (need <= {ALPHA}) -> {'PASS' if gateB else 'FAIL'}")
    os.makedirs(os.path.join(HUB, "05_results", "unified"), exist_ok=True)
    t.to_csv(os.path.join(HUB, "05_results", "unified", f"gates_eps{a.eps}_{a.bound}_{a.pilot_cost}_{a.baseline}_{a.judge}{'_v3' if a.gen == 'v3' else ''}.csv"), index=False)


if __name__ == "__main__":
    main()
