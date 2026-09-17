#!/usr/bin/env python3
"""Variance-dilution cost model for the fixed-budget document-level certificates (Track C, 2026-09-17).

Model. With the population estimand and every non-pilot query audited (f = 1 at the budgets where J50 is reached), the
bound's variance is mean_q(v_q)/n with v_q the within-query document-sampling variance of the arm's estimator, which for
Poisson sampling with pi ∝ base_d is proportional to 1/b (b = documents per query). Hence Var ∝ v1_arm / L_post with
L_post = n·b the labels drawn after the pilot, and the certificate crosses eps when z·sqrt(v1_arm / L_post) ≈ s (slack).
So, for any two arms sharing pilot P, candidate and slack,
        (J50_a − P) / (J50_b − P)  ≈  v1_a / v1_b,
i.e. the post-pilot cost ratio is the within-query variance ratio, and the *total* saving of arm a over b is
        1 − J50_a/J50_b = (1 − v1_a/v1_b) · (1 − P/J50_b):                      (variance reduction) × (post-pilot share).
The per-query variance ratio is read from the stored `var_est` at the smallest budgets (b = 4 documents per query, where
the within-query part dominates the between-query variance s_b^2 that every arm shares).

Tests, all from committed result files:
  (1) Table 2 arms (81/82 v3): post-pilot cost ratio vs small-budget variance ratio, every (collection, judge, eps, arm).
  (2) Predicted vs realised total saving for the judge arms and for uniform sampling.
  (3) Menu allocation (83 v3): the same model with the realised design objective max_j V_j/s_j^2 in place of the variance —
      where it fails, the certificate is not a variance-limited event.
Outputs: 05_results/unified/COST_MODEL_table2.csv, COST_MODEL_menu.csv, F8_cost_model.png; a printed summary.
"""
import glob, os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "05_results"); OUT = os.path.join(R, "unified")
REF = "weighted"
N_SMALL = 3          # budgets used for the within-dominated variance ratio


def j50(x, y, target=0.5):
    x = np.asarray(x, float); y = np.asarray(y, float); o = np.argsort(x); x, y = x[o], y[o]
    if y[0] >= target:
        return float(x[0]), "censored"
    for i in range(1, len(x)):
        if y[i] >= target:
            return float(x[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])), "ok"
    return float("nan"), "n.r."


def load_table2():
    fs = glob.glob(os.path.join(R, "sampling_baselines", "baselines_*_v3b*_v2_shared_full_t.csv")) + \
         glob.glob(os.path.join(R, "active_inference", "active_*_v3b*_v2_shared_full_binary_t.csv"))
    return pd.concat([pd.read_csv(f) for f in sorted(set(fs))], ignore_index=True)


def table2_block():
    d = load_table2(); rows = []
    for (c, j, e), g in d.groupby(["collection", "judge", "eps"]):
        gm = g.groupby(["method", "budget_full_eq"], as_index=False).agg(docs=("docs_labelled", "mean"), act=("act", "mean"),
                                                                        var=("var_est", "mean"), pilot=("docs_pilot", "mean"))
        P = float(gm.pilot.mean())
        J = {m: j50(x.docs, x.act) for m, x in gm.groupby("method")}
        if REF not in J or J[REF][1] != "ok":
            continue
        small = sorted(gm.budget_full_eq.unique())[:N_SMALL]
        vref = gm[(gm.method == REF) & gm.budget_full_eq.isin(small)].set_index("budget_full_eq")["var"]
        for m, x in gm.groupby("method"):
            if m == REF or J[m][1] != "ok":
                continue
            v = x[x.budget_full_eq.isin(small)].set_index("budget_full_eq")["var"]
            r_var = float((v / vref).mean())
            r_cost = (J[m][0] - P) / (J[REF][0] - P)
            share = 1 - P / J[REF][0]
            rows.append(dict(collection=c, judge=j, eps=e, method=m, pilot=P, J50_ref=J[REF][0], J50=J[m][0],
                             post_pilot_share=share, var_ratio=r_var, cost_ratio=r_cost,
                             saving_pred=(1 - r_var) * share, saving_obs=1 - J[m][0] / J[REF][0]))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "COST_MODEL_table2.csv"), index=False)
    return t


def menu_block():
    fs = [f for f in glob.glob(os.path.join(R, "menu_allocation", "menu_alloc_*_v2_full_t_v3b*.csv")) if "candbest" not in f and "ntr" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); rows = []
    for (c, e), g in d.groupby(["collection", "eps"]):
        gm = g.groupby(["method", "budget_full_eq"], as_index=False).agg(docs=("docs_labelled", "mean"), act=("act", "mean"),
                                                                        obj=("design_obj", "mean"), pilot=("docs_pilot", "mean"),
                                                                        feas=("feasible_frac", "mean"), actf=("act_feasible", "mean"))
        P = float(gm.pilot.mean()); ref = "static_sum"
        J = {m: j50(x.docs, x.act) for m, x in gm.groupby("method")}
        if J[ref][1] != "ok":
            continue
        # design objective ratio at the budget nearest the reference J50
        bref = gm[gm.method == ref].iloc[(gm[gm.method == ref].docs - J[ref][0]).abs().argsort().values[0]].budget_full_eq
        oref = float(gm[(gm.method == ref) & (gm.budget_full_eq == bref)].obj.iloc[0])
        for m in ["oracle_exact", "plugin_exact", "static_sum_v", "adaptive", "per_pair"]:
            if m not in J or J[m][1] != "ok":
                continue
            om = gm[(gm.method == m) & (gm.budget_full_eq == bref)].obj
            r_obj = float(om.iloc[0] / oref) if len(om) else float("nan")
            rows.append(dict(collection=c, eps=e, method=m, pilot=P, J50_ref=J[ref][0], J50=J[m][0], budget_ref=int(bref),
                             obj_ratio=r_obj, cost_ratio=(J[m][0] - P) / (J[ref][0] - P),
                             feasible=float(gm.feas.mean()), act_max_ref=float(gm[gm.method == ref].act.max())))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "COST_MODEL_menu.csv"), index=False)
    return t


def main():
    pd.set_option("display.width", 220)
    t = table2_block()
    lt = np.log(t.cost_ratio / t.var_ratio)
    print("== Table 2 arms: post-pilot cost ratio vs small-budget variance ratio (reference = weighted)")
    print(t.round(3).to_string(index=False))
    print(f"\nn={len(t)}  corr(log cost ratio, log var ratio)={np.corrcoef(np.log(t.cost_ratio), np.log(t.var_ratio))[0,1]:.3f}"
          f"  median |log(cost/var)|={lt.abs().median():.3f}  (i.e. median multiplicative error {np.exp(lt.abs().median()):.2f}x)"
          f"  90th pct {np.exp(lt.abs().quantile(0.9)):.2f}x")
    jud = t[t.method.isin(["weighted_cvl", "weighted_cv", "ai_resid", "ai_robust_0.3", "ai_calib"])]
    print(f"judge arms: corr(pred saving, obs saving)={np.corrcoef(jud.saving_pred, jud.saving_obs)[0,1]:.3f}"
          f"  mean abs error {np.abs(jud.saving_pred - jud.saving_obs).mean():.3f} (in units of total J50)")
    uni = t[t.method == "uniform"]
    print(f"uniform: pred saving of weighted over uniform = 1 - 1/(1/(1-var_red)...): obs {(1-1/(uni.cost_ratio*(1-uni.post_pilot_share)+uni.post_pilot_share)).round(3).tolist()}")
    m = menu_block()
    print("\n== Menu allocation: design-objective ratio vs post-pilot cost ratio (reference = static_sum)")
    print(m.round(3).to_string(index=False))
    if len(m):
        mm = m.dropna(subset=["obj_ratio"])
        print(f"corr(log cost ratio, log obj ratio)={np.corrcoef(np.log(mm.cost_ratio), np.log(mm.obj_ratio))[0,1]:.3f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for mth, g in t.groupby("method"):
        ax[0].scatter(g.var_ratio, g.cost_ratio, s=22, label=mth, alpha=0.8, edgecolor="k", linewidth=0.3)
    lim = [0.4, 8]; ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("within-query variance ratio (arm / weighted), smallest budgets"); ax[0].set_ylabel("post-pilot cost ratio (J50 − pilot)")
    ax[0].set_title("Table 2 arms, ε ∈ {0.01, 0.02}: cost ∝ variance", fontsize=9); ax[0].legend(fontsize=6, frameon=False)
    ax[1].scatter(jud.saving_pred, jud.saving_obs, s=22, c="C3", edgecolor="k", linewidth=0.3)
    ax[1].plot([-0.1, 0.35], [-0.1, 0.35], "k--", lw=0.8); ax[1].set_xlabel("predicted total saving = (1 − var ratio) × post-pilot share"); ax[1].set_ylabel("observed total saving 1 − J50/J50(weighted)")
    ax[1].set_title("judge arms: the judge's increment is variance reduction diluted by the pilot", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "F8_cost_model.png"), dpi=200)


if __name__ == "__main__":
    main()
