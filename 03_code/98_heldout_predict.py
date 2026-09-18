#!/usr/bin/env python3
"""Held-out protocol, step (a): turn the pilot-only records of `81 --predict_only` into the locked predictions
(design log §16.7). Nothing here uses a label outside the 20-query pilot of each draw.

For every (collection, judge, eps) and arm a, with w = decision-weight sampling as reference:
  * predicted post-pilot cost ratio  r_a = median over draws of v_pilot[a] / v_pilot[w]
  * predicted post-pilot labels of w  L_w = z^2 * b_ref * v_pilot[w] / slack_pilot^2   (median over draws)
  * predicted total saving of a over w:  (1 - r_a) * L_w / (L_w + P)
  * decision D1: recommend a judge arm iff its predicted saving > tau in the majority of draws
  * decision D3: lambda-fitted CV preferred to coefficient-1 CV iff r_cvl < r_cv
  * decomposition: predicted savings uniform -> uniform_nz -> weighted from the same ratios.
The rule (tau, b_ref, the majority vote, the reference arm) is the one fixed on the four development collections.
Output: 04_results/heldout/PREDICTIONS_<tag>.csv and .md  (commit these before running the audit).
Usage: python3 98_heldout_predict.py --glob "04_results/sampling_baselines/baselines_judged_rr_heldout_b*_predict.csv" --tag rr
"""
import argparse, glob, importlib.util, os
import numpy as np, pandas as pd
from scipy.stats import t as tdist
_spec = importlib.util.spec_from_file_location("costpredict", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "costpredict.py"))
cp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cp)

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "04_results")
REF = "weighted"; JUDGE_ARMS = ["weighted_cvl", "weighted_cv", "active_cv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--b_ref", type=int, default=4)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.glob))], ignore_index=True)
    rows = []
    for (c, j, e), g in d.groupby(["collection", "judge", "eps"]):
        g0 = g[g.budget_full_eq == g.budget_full_eq.min()]          # pilot quantities do not depend on the budget
        ref = g0[g0.method == REF].set_index("draw"); P = float(g0.pilot.mean()); N = int(g0.n_queries.iloc[0])
        z = float(tdist.ppf(1 - 0.10 / 6, N - 21))
        L = z ** 2 * a.b_ref * ref.v_pilot / ref.slack_pilot.clip(lower=1e-4) ** 2; share = L / (L + P)   # v1; bug fixed 2026-09-17 (slack was not squared in the first lock, commit 7deb082)
        n_tr = int(ref.n_train.iloc[0]) if "n_train" in ref else 20
        sb2 = ref.sb2_pilot if "sb2_pilot" in ref else pd.Series(np.nan, index=ref.index)
        L2 = pd.Series([cp.l_post_v2(v, sb, sl, z, N, n_tr, a.b_ref) for v, sb, sl in zip(ref.v_pilot, sb2, ref.slack_pilot)], index=ref.index); share2 = L2 / (L2 + P)
        L3 = pd.Series([cp.l_post_v3(r, z, N, n_tr, a.b_ref) for r in ref.to_dict("records")], index=ref.index) if "slack_c0" in ref else L2 * np.nan; share3 = L3 / (L3 + P)
        L4 = pd.Series([cp.l_post_v2(v, sb, sl, z, N, n_tr, a.b_ref) for v, sb, sl in zip(ref.v_pilot, sb2, ref.slack_xfit)], index=ref.index) if "slack_xfit" in ref else L2 * np.nan; share4 = L4 / (L4 + P)
        for m, gm in g0.groupby("method"):
            if m == REF:
                continue
            gm = gm.set_index("draw"); vr = (gm.v_pilot / ref.v_pilot).dropna(); sav = (1 - vr) * share.reindex(vr.index); sav2 = (1 - vr) * share2.reindex(vr.index); sav3 = (1 - vr) * share3.reindex(vr.index); sav4 = (1 - vr) * share4.reindex(vr.index)
            rows.append(dict(collection=c, judge=j, eps=e, method=m, n_draws=len(vr), pilot=P, N=N,
                             vr_med=float(vr.median()), vr_q25=float(vr.quantile(.25)), vr_q75=float(vr.quantile(.75)),
                             L_post_pred_med=float(L.median()), share_pred_med=float(share.median()),
                             saving_pred_med=float(sav.median()), frac_recommend=float((sav > a.tau).mean()),
                             decision=int((sav > a.tau).mean() > 0.5) if m in JUDGE_ARMS else -1,
                             L_post_pred_v2_med=float(L2.median()), saving_pred_v2_med=float(sav2.median()), frac_recommend_v2=float((sav2 > a.tau).mean()),
                             decision_v2=int((sav2 > a.tau).mean() > 0.5) if m in JUDGE_ARMS else -1,
                             L_post_pred_v3_med=float(L3.median()), saving_pred_v3_med=float(sav3.median()), frac_recommend_v3=float((sav3 > a.tau).mean()),
                             decision_v3=int((sav3 > a.tau).mean() > 0.5) if m in JUDGE_ARMS else -1,
                             L_post_pred_v4_med=float(L4.median()), saving_pred_v4_med=float(sav4.median()), frac_recommend_v4=float((sav4 > a.tau).mean()),
                             decision_v4=int((sav4 > a.tau).mean() > 0.5) if m in JUDGE_ARMS else -1,
                             slack_pilot_med=float(ref.slack_pilot.median())))
    t = pd.DataFrame(rows); out = os.path.join(R, "heldout"); os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, f"PREDICTIONS_{a.tag}.csv"), index=False)
    lines = [f"# Locked pre-audit predictions ({a.tag}; tau={a.tau}, b_ref={a.b_ref}, reference={REF}); generated by 98_heldout_predict.py", ""]
    for (c, j, e), g in t.groupby(["collection", "judge", "eps"]):
        g = g.set_index("method")
        lines.append(f"## {c} / judge={j} / eps={e}  (N={int(g.N.iloc[0])}, pilot={g.pilot.iloc[0]:.0f} labels, predicted post-pilot labels of weighted: v1 {g.L_post_pred_med.iloc[0]:.0f}, v2 {g.L_post_pred_v2_med.iloc[0]:.0f}, v3 {g.L_post_pred_v3_med.iloc[0]:.0f}, v4 {g.L_post_pred_v4_med.iloc[0]:.0f})")
        for m in ["uniform", "uniform_nz", "strat_pilot", "weighted_cv", "weighted_cvl", "active_cv", "active_judge", "ai_calib", "ai_resid", "ai_robust_0.3"]:
            if m in g.index:
                r = g.loc[m]; dec = {1: "RECOMMEND", 0: "do not recommend", -1: ""}[int(r.decision)]
                dec2 = {1: "RECOMMEND", 0: "do not recommend", -1: ""}[int(r.decision_v2)]; dec3 = {1: "RECOMMEND", 0: "do not recommend", -1: ""}[int(r.decision_v3)]
                lines.append(f"- {m:14s} predicted cost ratio {r.vr_med:.2f} [{r.vr_q25:.2f}, {r.vr_q75:.2f}]; total saving v1 {r.saving_pred_med:+.3f} ({r.frac_recommend:.2f}) {dec}; v2 {r.saving_pred_v2_med:+.3f} ({r.frac_recommend_v2:.2f}) {dec2}; v3 {r.saving_pred_v3_med:+.3f} ({r.frac_recommend_v3:.2f}) {dec3}; v4 {r.saving_pred_v4_med:+.3f} ({r.frac_recommend_v4:.2f}) {'RECOMMEND' if r.decision_v4 == 1 else ('do not recommend' if r.decision_v4 == 0 else '')}")
        if "weighted_cvl" in g.index and "weighted_cv" in g.index:
            lines.append(f"- D3: lambda-fitted CV {'preferred' if g.loc['weighted_cvl'].vr_med < g.loc['weighted_cv'].vr_med else 'NOT preferred'} to coefficient-1 CV")
        if "uniform" in g.index and "uniform_nz" in g.index:
            u, nz = g.loc["uniform"].vr_med, g.loc["uniform_nz"].vr_med
            lines.append(f"- decomposition (post-pilot labels): uniform -> uniform_nz saves {1 - nz / u:.2f}, uniform_nz -> weighted saves {1 - 1 / nz:.2f}, uniform -> weighted saves {1 - 1 / u:.2f}")
        lines.append("")
    open(os.path.join(out, f"PREDICTIONS_{a.tag}.md"), "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
