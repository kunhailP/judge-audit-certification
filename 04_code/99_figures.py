#!/usr/bin/env python3
"""Regenerate every figure of the manuscript from the committed result files, in one style (lib/figstyle.py).
F4 gain vs rho (ppi_gain/*.csv) | F5 cumulative ACT vs budget (planner_v2/judged_*_ext/planner_v2.parquet) |
F6 mechanism map (rho_map/*.csv) | F7 J50 by arm (unified/TABLE2_v3_eps0.02.csv) | F8 cost model (unified/COST_MODEL_table2.csv) |
F9 calculator (rho_map/CALCULATOR_vs_map.csv). Output: 06_paper/tmlr_submission/figures/F{4..9}*.png (and copies next to the data).
"""
import glob, importlib.util, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
R = os.path.join(HUB, "05_results"); FIG = os.path.join(HUB, "06_paper", "tmlr_submission", "figures")
_s = importlib.util.spec_from_file_location("figstyle", os.path.join(HERE, "lib", "figstyle.py")); fs = importlib.util.module_from_spec(_s); _s.loader.exec_module(fs)
fs.use(); import matplotlib.pyplot as plt; from matplotlib.colors import TwoSlopeNorm
PAL = fs.PALETTE; W1, W2 = 3.3, 6.75   # single- and double-column widths (inches)


def save(fig, name, also=None):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    if also:
        fig.savefig(os.path.join(also, name), bbox_inches="tight")
    plt.close(fig); print("saved", name)


def f4():
    files = ["legacy_rr", "modern_rr", "judged_rr", "judged_llm", "trecdl_rr", "trecdl_llm"]
    t = pd.concat([pd.read_csv(os.path.join(R, "ppi_gain", f"ppi_gain_{f}.csv")).assign(file=f) for f in files]); t = t[t["T"] == 10]
    lab = {"legacy_rr": ("o", PAL[0], "BEIR pools, legacy stack, reranker"), "modern_rr": ("s", PAL[5], "BEIR pools, modern stack, reranker"),
           "judged_rr": ("^", PAL[1], "judged pools, reranker"), "judged_llm": ("v", PAL[2], "judged pools, Qwen3-8B"),
           "trecdl_rr": ("D", PAL[3], "TREC DL 2019/20, reranker"), "trecdl_llm": ("P", PAL[4], "TREC DL 2019/20, Qwen3-8B")}
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 2.9))
    for f, (mk, c, l) in lab.items():
        x = t[t.file == f]; ax.scatter(x.rho, x.ess_gain, marker=mk, s=16, color=c, label=l, edgecolor="k", linewidth=0.3, zorder=3)
    r = np.linspace(-0.2, 0.92, 200); ax.plot(r, 1 / (1 - r ** 2), color="k", ls="--", lw=0.8, label=r"$1/(1-\rho^2)$")
    ax.axhline(1.0, color="0.6", lw=0.5); ax.set_xlabel(r"$\rho$ (true vs. judge-predicted paired difference)"); ax.set_ylabel("realised PPI gain")
    ax.set_ylim(0.6, 2.4); ax.set_xlim(-0.25, 0.95); ax.legend(loc="upper left", ncol=1)
    save(fig, "F4_gain_vs_rho.png", os.path.join(R, "ppi_gain"))


def f5():
    looks = [10, 30, 50, 70, 90, 120, 150, 190]
    series = [("judged_llm_ext", "split_t", "humans only", "k", "--"), ("judged_llm_ext", "split_ppi", "PPI, Qwen3-8B", PAL[0], "-"),
              ("judged_rr_ext", "split_ppi", "PPI, Qwen3-Reranker", PAL[1], "-"), ("judged_mpnet_ext", "split_ppi", "PPI, MPNet rule", PAL[2], "-"),
              ("judged_inv_ext", "split_ppi", "PPI, inverted reranker", PAL[3], "-")]
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.7))
    for d, m, l, c, ls in series:
        p = os.path.join(R, "planner_v2", d, "planner_v2.parquet")
        if not os.path.exists(p):
            continue
        x = pd.read_parquet(p); x = x[(x.collection == "dbpedia-entity") & (x.method == m)]; n = x.repeat_id.nunique()
        act = x[x.action == "act"]; y = [(act.final_T <= T).sum() / n for T in looks]; wrong = int(act.wrong_cert.sum())
        ax.plot(looks, y, marker="o", color=c, ls=ls, label=f"{l} (wrong {wrong}/{n})")
    ax.set_xlabel("audited queries $T$"); ax.set_ylabel("cumulative certification rate"); ax.set_ylim(0, 0.5); ax.legend(loc="upper left")
    save(fig, "F5_act_vs_budget.png", os.path.join(R, "planner_v2"))


def f6():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "rho_map", "rho_map_*_llm.csv"))]); d = d[d["T"] == 90]
    colls = ["dbpedia-entity", "dl212223"]; epss = sorted(d.eps.unique())
    fig, axes = plt.subplots(2, len(epss), figsize=(W2, 3.6), squeeze=False)
    norm = TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=1.55)
    for i, c in enumerate(colls):
        for j, e in enumerate(epss):
            ax = axes[i, j]; g = d[(d.collection == c) & (d.eps == e)]
            piv = g.pivot_table(index="N_unlab", columns="rho", values="ratio").sort_index(ascending=False)
            im = ax.imshow(piv.values, cmap="RdBu", norm=norm, aspect="auto")
            for (yi, xi), v in np.ndenumerate(piv.values):
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=6, color="k")
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{r:.2f}" for r in piv.columns]); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(v) for v in piv.index])
            ax.set_title(f"{fs.name(c)}, $\\varepsilon={e}$", fontsize=7); ax.tick_params(length=0)
            if j == 0: ax.set_ylabel("$N_{\\mathrm{unlab}}$")
            if i == 1: ax.set_xlabel(r"$\rho$")
            for s in ax.spines.values(): s.set_visible(False)
    fig.subplots_adjust(hspace=0.6, wspace=0.35)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02); cb.set_label("ACT(PPI) / ACT(humans)")
    save(fig, "F6_rho_map.png", os.path.join(R, "rho_map"))


def f7():
    t = pd.read_csv(os.path.join(R, "unified", "TABLE2_v3_eps0.02.csv"))
    cols = [("antique", "llm"), ("antique", "rr"), ("antique", "inv"), ("cast19", "llm"), ("cast19", "rr"), ("cast19", "inv"), ("dbpedia-entity", "llm"), ("dbpedia-entity", "rr"), ("dl212223", "llm")]
    arms = ["uniform", "uniform_nz", "weighted", "strat_pilot", "ai_calib", "ai_resid", "ai_robust_0.3", "weighted_cv", "weighted_cvl"]
    piv = t.pivot_table(index="method", columns=["collection", "judge"], values="J50")
    fig, ax = plt.subplots(figsize=(W2, 3.0)); x = np.arange(len(cols)); w = 0.09
    for i, a in enumerate(arms):
        vals = [piv.loc[a, c] if c in piv.columns and a in piv.index else np.nan for c in cols]
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, vals, w, color=PAL[i % len(PAL)], label=fs.ARMS[a], edgecolor="k", linewidth=0.2)
    short = {"llm": "Qwen3-8B", "rr": "reranker", "inv": "inverted"}
    cname = {"antique": "ANTIQUE", "cast19": "CAsT 2019", "dbpedia-entity": "DBpedia", "dl212223": "DL 21\u201323"}
    ax.set_xticks(x); ax.set_xticklabels([f"{cname[c]}\n{short[j]}" for c, j in cols], fontsize=6)
    ax.set_ylabel("$J_{50}$ (unique human labels)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5)
    save(fig, "F7_J50_v3.png", os.path.join(R, "unified"))


def f8():
    t = pd.read_csv(os.path.join(R, "unified", "COST_MODEL_table2.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    order = ["uniform", "uniform_nz", "strat_pilot", "ai_calib", "ai_resid", "ai_robust_0.3", "ai_robust_0.5", "weighted_cv", "weighted_cvl", "active_judge", "active_cv"]
    for i, m in enumerate([m for m in order if m in set(t.method)]):
        g = t[t.method == m]; ax[0].scatter(g.var_ratio, g.cost_ratio, s=12, color=PAL[i % len(PAL)], marker="os^v<>DPXd*"[i % 11], label=fs.ARMS.get(m, m), edgecolor="k", linewidth=0.25, zorder=3)
    lim = [0.4, 8]; ax[0].plot(lim, lim, "k--", lw=0.7); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("within-query variance ratio (arm / decision-weight sampling)"); ax[0].set_ylabel("post-pilot cost ratio")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    for a_ in (ax[0].xaxis, ax[0].yaxis):
        a_.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8])); a_.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4", "8"])); a_.set_minor_formatter(FixedFormatter([]))
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6)
    j = t[t.method.isin(["weighted_cvl", "weighted_cv", "ai_resid", "ai_robust_0.3", "ai_calib"])]
    ax[1].scatter(j.saving_pred, j.saving_obs, s=12, color=PAL[1], edgecolor="k", linewidth=0.25, zorder=3)
    ax[1].plot([-0.15, 0.35], [-0.15, 0.35], "k--", lw=0.7); ax[1].set_xlim(-0.15, 0.35); ax[1].set_ylim(-0.15, 0.35)
    ax[1].set_xlabel("predicted saving: $(1-v_a/v_w)\\,(1-P/J_{50}^{w})$"); ax[1].set_ylabel("realised saving $1-J_{50}^{a}/J_{50}^{w}$")
    fig.tight_layout(); save(fig, "F8_cost_model.png", os.path.join(R, "unified"))


def f9():
    t = pd.read_csv(os.path.join(R, "rho_map", "CALCULATOR_vs_map.csv")); t = t[t["T"] == 90]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    for i, (c, g) in enumerate(t.groupby("collection")):
        ax[i].scatter(g.ratio_map, g.ratio_crossfit, marker="o", s=12, color=PAL[0], label=r"cross-fitted $\lambda$", edgecolor="k", linewidth=0.25, zorder=3)
        ax[i].scatter(g.ratio_map, g.ratio_oracle, marker="^", s=12, color=PAL[1], label=r"population-optimal $\lambda$", edgecolor="k", linewidth=0.25, zorder=3)
        lo, hi = (0.7, 1.7) if c == "dbpedia-entity" else (0.85, 1.12)
        ax[i].plot([lo, hi], [lo, hi], "k--", lw=0.7); ax[i].set_xlim(lo, hi); ax[i].set_ylim(lo, hi)
        ax[i].set_xlabel("measured ACT ratio"); ax[i].set_ylabel("simulated ACT ratio"); ax[i].set_title(fs.name(c), fontsize=8); ax[i].legend(loc="upper left")
    fig.tight_layout(); save(fig, "F9_calculator.png", os.path.join(R, "rho_map"))


if __name__ == "__main__":
    for f in [f4, f5, f6, f7, f8, f9]:
        f()
