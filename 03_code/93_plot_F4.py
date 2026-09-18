#!/usr/bin/env python3
"""Figure F4 and the numbers quoted with it, regenerated from the committed ppi_gain CSVs (2026-09-17).

The 54 (collection, pair, judge) settings are the T=10 rows of
  legacy_rr (12), modern_rr (12), judged_rr (9), judged_llm (9), trecdl_rr (6), trecdl_llm (6).
Prints: corr(gain, rho), corr(gain, pair accuracy), the demeaned within-(collection, judge) correlation and how many of the
14 groups have a positive within-group correlation, the judge-swap count, and on the 33 settings with Monte-Carlo MSE
columns the MSE-ratio correlation. Writes 05_paper/tmlr_submission/figures/F4_gain_vs_rho.png and
04_results/ppi_gain/F4_summary.json. Run it after any ppi_gain CSV changes so that the text and the figure agree.
"""
import json, os
import numpy as np, pandas as pd

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "04_results", "ppi_gain")
FILES = ["legacy_rr", "modern_rr", "judged_rr", "judged_llm", "trecdl_rr", "trecdl_llm"]


def main():
    d = pd.concat([pd.read_csv(os.path.join(R, f"ppi_gain_{f}.csv")).assign(file=f) for f in FILES], ignore_index=True)
    t = d[d["T"] == 10].copy()
    out = dict(n_settings=int(len(t)), corr_gain_rho=float(t.ess_gain.corr(t.rho)), corr_gain_acc=float(t.ess_gain.corr(t.judge_acc)))
    g = t.groupby(["collection", "judge"])
    dm = t.assign(g_=t.ess_gain - g.ess_gain.transform("mean"), r_=t.rho - g.rho.transform("mean"))
    within = {f"{c}/{j}": float(x.ess_gain.corr(x.rho)) for (c, j), x in g}
    out.update(n_groups=int(g.ngroups), corr_within_demeaned=float(np.corrcoef(dm.g_, dm.r_)[0, 1]),
               n_groups_positive=int(sum(v > 0 for v in within.values())), within=within)
    piv = t.pivot_table(index=["collection", "pair"], columns="judge", values=["rho", "ess_gain"]).dropna()
    up = piv[("rho", "llm")] > piv[("rho", "rr")]
    out.update(judge_swap_rho_up=int(up.sum()), judge_swap_gain_up=int(((piv[("ess_gain", "llm")] > piv[("ess_gain", "rr")]) & up).sum()))
    if "mc_mse_human" in t:
        m = t.dropna(subset=["mc_mse_human"]); mg = m.mc_mse_human / m.mc_mse_ppi
        out.update(n_mc=int(len(m)), corr_mc_gain_rho=float(mg.corr(m.rho)), corr_se_gain_rho_on_mc=float(m.ess_gain.corr(m.rho)),
                   median_abs_diff=float((mg - m.ess_gain).abs().median()))
    for T in sorted(d["T"].unique()):
        tt = d[d["T"] == T]
        out[f"T{int(T)}"] = dict(n=int(len(tt)), corr_gain_rho=float(tt.ess_gain.corr(tt.rho)), corr_gain_acc=float(tt.ess_gain.corr(tt.judge_acc)))
    json.dump(out, open(os.path.join(R, "F4_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "within"}, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = {"legacy_rr": ("o", "BEIR legacy, reranker"), "modern_rr": ("s", "BEIR modern, reranker"), "judged_rr": ("^", "judged pools, reranker"),
               "judged_llm": ("v", "judged pools, Qwen3-8B"), "trecdl_rr": ("D", "TREC DL 19/20, reranker"), "trecdl_llm": ("P", "TREC DL 19/20, Qwen3-8B")}
    for f, (mk, lab) in markers.items():
        x = t[t.file == f]
        ax.scatter(x.rho, x.ess_gain, marker=mk, s=34, label=lab, alpha=0.85, edgecolor="k", linewidth=0.4)
    r = np.linspace(-0.2, 0.92, 200); ax.plot(r, 1 / (1 - r ** 2), "k--", lw=1, label=r"$1/(1-\rho^2)$ ($N/n\to\infty$)")
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_xlabel(r"$\rho$ = corr(true, judge-predicted paired difference)"); ax.set_ylabel("realised gain (ratio of squared estimated SEs), $T=10$")
    ax.set_title(f"{out['n_settings']} settings: corr(gain, ρ) = {out['corr_gain_rho']:.2f}, corr(gain, accuracy) = {out['corr_gain_acc']:+.2f}", fontsize=9)
    ax.set_ylim(0.6, 2.4); ax.legend(fontsize=7, frameon=False, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(HUB, "05_paper", "tmlr_submission", "figures", "F4_gain_vs_rho.png"), dpi=200)
    fig.savefig(os.path.join(R, "F4_gain_vs_rho.png"), dpi=200)


if __name__ == "__main__":
    main()
