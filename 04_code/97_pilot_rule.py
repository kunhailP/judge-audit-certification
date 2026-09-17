#!/usr/bin/env python3
"""Pre-audit rule: can the pilot alone predict which arm to use and how much it saves? (Track C, 2026-09-17)

Input: *_draws.csv from `81_sampling_baselines.py --dump_draws` (one row per draw and arm, with the pilot-only within-query
variance prediction v_pilot, the pilot slack estimate, the realised unique-label cost and the certificate outcome).

The variance-dilution model (95_cost_model.py) says
    post-pilot cost ratio (arm / weighted) ≈ v_arm / v_weighted,   total saving ≈ (1 − v ratio) × post-pilot share.
Both factors can be estimated before the audit: the variance ratio from the pilot (v_pilot), and the post-pilot share from
the predicted post-pilot labels of the reference arm, L_post ≈ z² · v_w / ŝ², with ŝ the pilot slack of the binding
comparison (noisy at 20 queries: reported with its draw-level spread).

For every (collection, judge, eps) this script reports, per arm:
  * the pilot-predicted variance ratio (median and IQR over draws) against the realised post-pilot cost ratio (J50 − P
    of the arm over that of `weighted`, from the same draws);
  * the pilot-predicted total saving against the realised one;
  * the accuracy of three pre-audit decisions taken draw by draw:
      D1  "use the judge control variate rather than humans-only weighted sampling"   (predicted saving > tau)
      D2  "the judge will save at least tau of the total"                              (predicted saving > tau)
      D3  "prefer the lambda-fitted CV to the coefficient-1 CV"                         (v_cvl < v_cv)
    scored against the realised J50s of the cell (the cell-level truth) and, for D1/D2, against the realised outcome of
    the same draw at the budget nearest the reference J50 (draw-level truth, noisy).
Output: 05_results/unified/PILOT_RULE_eps{eps}.csv and a printed table.
"""
import argparse, glob, importlib.util, math, os
import numpy as np, pandas as pd
from scipy.stats import t as tdist
_spec = importlib.util.spec_from_file_location("costpredict", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "costpredict.py"))
cp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cp)

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HUB, "05_results")
REF = "weighted"


def j50(x, y, target=0.5):
    x = np.asarray(x, float); y = np.asarray(y, float); o = np.argsort(x); x, y = x[o], y[o]
    if y[0] >= target:
        return float(x[0])
    for i in range(1, len(x)):
        if y[i] >= target:
            return float(x[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--glob", default=os.path.join(R, "sampling_baselines", "baselines_*_draws.csv"))
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit("no *_draws.csv files: re-run 81 with --dump_draws")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True); d = d[d.eps == a.eps]
    if "v_pilot" not in d:
        raise SystemExit("draw files predate the pilot prediction columns; re-run 81 with --dump_draws")
    if "sb2_pilot" not in d or d.sb2_pilot.isna().all():
        # older draw records: take sb2_pilot / n_train from the matching predict-only records (same permutations)
        pf = sorted(glob.glob(a.glob.replace("_draws.csv", "_predict.csv")))
        if pf:
            pr = pd.concat([pd.read_csv(f) for f in pf], ignore_index=True)
            extra = [c for c in pr.columns if c in ("sb2_pilot", "n_train") or c.startswith(("slack_c", "sb2_c", "v_pilot_c", "v_pilot_ref_c"))]
            d = d.drop(columns=[c for c in extra if c in d]).merge(pr[["collection", "judge", "budget_full_eq", "eps", "method", "draw"] + extra], on=["collection", "judge", "budget_full_eq", "eps", "method", "draw"], how="left")
        else:
            d["sb2_pilot"] = np.nan; d["n_train"] = 20
    rows = []
    for (c, j), g in d.groupby(["collection", "judge"]):
        curves = g.groupby(["method", "budget_full_eq"], as_index=False).agg(docs=("docs", "mean"), act=("act", "mean"), pilot=("pilot", "mean"))
        J = {m: j50(x.docs, x.act) for m, x in curves.groupby("method")}
        P = float(curves.pilot.mean())
        if not np.isfinite(J.get(REF, np.nan)):
            continue
        share = 1 - P / J[REF]
        # pilot-only prediction of the reference arm's post-pilot labels: the two-stage bound with every remaining query
        # audited needs mean(v_q(b))/n <= (s/z)^2 with v_q(b) ~ v_pilot(b_ref) * b_ref / b, so L_post = n b = z^2 b_ref v_pilot / s^2
        z = float(tdist.ppf(1 - 0.10 / 6, int(g.n_queries.iloc[0]) - 21)); b_ref = 4
        # pilot quantities are identical across budgets within a draw (same permutation) -> take the smallest budget's rows
        b0 = g.budget_full_eq.min(); g0 = g[g.budget_full_eq == b0]
        ref = g0[g0.method == REF].set_index("draw")
        for m, gm in g0.groupby("method"):
            if m == REF or not np.isfinite(J.get(m, np.nan)):
                continue
            gm = gm.set_index("draw"); vr = (gm.v_pilot / ref.v_pilot).dropna()
            pred_saving = (1 - vr) * share
            s_hat = ref.slack_pilot.clip(lower=1e-4); L_pred = z ** 2 * b_ref * ref.v_pilot / s_hat ** 2
            share_pred = (L_pred / (L_pred + P)).reindex(vr.index); pred_saving_pilot = (1 - vr) * share_pred
            n_tr = int(ref.n_train.iloc[0]) if "n_train" in ref and np.isfinite(ref.n_train.iloc[0]) else 20
            L2 = pd.Series([cp.l_post_v2(v, sb, sl, z, int(g.n_queries.iloc[0]), n_tr, b_ref) for v, sb, sl in zip(ref.v_pilot, ref.sb2_pilot if "sb2_pilot" in ref else [np.nan] * len(ref), ref.slack_pilot)], index=ref.index)
            share2 = (L2 / (L2 + P)).reindex(vr.index); pred_saving_v2 = (1 - vr) * share2
            L3 = pd.Series([cp.l_post_v3(r, z, int(g.n_queries.iloc[0]), n_tr, b_ref) for r in ref.to_dict("records")], index=ref.index) if "slack_c0" in ref else L2 * np.nan
            share3 = (L3 / (L3 + P)).reindex(vr.index); pred_saving_v3 = (1 - vr) * share3
            obs_cost_ratio = (J[m] - P) / (J[REF] - P); obs_saving = 1 - J[m] / J[REF]
            rows.append(dict(collection=c, judge=j, eps=a.eps, method=m, pilot=P, J50_ref=J[REF], J50=J[m], post_pilot_share=share,
                             vr_pred_med=float(vr.median()), vr_pred_q25=float(vr.quantile(0.25)), vr_pred_q75=float(vr.quantile(0.75)),
                             cost_ratio_obs=obs_cost_ratio, saving_pred_med=float(pred_saving.median()), saving_obs=obs_saving,
                             L_post_obs=J[REF] - P, L_post_pred_med=float(L_pred.median()), L_post_pred_q25=float(L_pred.quantile(0.25)), L_post_pred_q75=float(L_pred.quantile(0.75)),
                             share_pred_med=float(share_pred.median()), saving_pred_pilot_med=float(pred_saving_pilot.median()),
                             frac_draws_recommend_pilot=float((pred_saving_pilot > a.tau).mean()),
                             L_post_pred_v2_med=float(L2.median()), saving_pred_v2_med=float(pred_saving_v2.median()),
                             frac_draws_recommend_v2=float((pred_saving_v2 > a.tau).mean()),
                             L_post_pred_v3_med=float(L3.median()), saving_pred_v3_med=float(pred_saving_v3.median()),
                             frac_draws_recommend_v3=float((pred_saving_v3 > a.tau).mean()),
                             frac_draws_recommend=float((pred_saving > a.tau).mean()),
                             cell_truth=int(obs_saving > a.tau),
                             slack_pilot_med=float(ref.slack_pilot.median()), slack_true_med=float(ref.slack_true.median())))
    t = pd.DataFrame(rows); os.makedirs(os.path.join(R, "unified"), exist_ok=True)
    t.to_csv(os.path.join(R, "unified", f"PILOT_RULE_eps{a.eps}.csv"), index=False)
    pd.set_option("display.width", 250); print(t.round(3).to_string(index=False))
    if len(t):
        print(f"\ncorr(pilot-predicted variance ratio, realised post-pilot cost ratio) = {np.corrcoef(np.log(t.vr_pred_med), np.log(t.cost_ratio_obs))[0,1]:.3f}")
        print(f"corr(pilot-predicted total saving, realised) = {np.corrcoef(t.saving_pred_med, t.saving_obs)[0,1]:.3f}; MAE = {np.abs(t.saving_pred_med - t.saving_obs).mean():.3f}  (post-pilot share taken from the realised reference J50)")
        print(f"fully pilot-based (share predicted from pilot slack): corr = {np.corrcoef(t.saving_pred_pilot_med, t.saving_obs)[0,1]:.3f}; MAE = {np.abs(t.saving_pred_pilot_med - t.saving_obs).mean():.3f}")
        print(f"fully pilot-based, v2 predictor (two-stage variance, known pilot part): corr = {np.corrcoef(t.saving_pred_v2_med, t.saving_obs)[0,1]:.3f}; MAE = {np.abs(t.saving_pred_v2_med - t.saving_obs).mean():.3f}")
        u = t.drop_duplicates(["collection", "judge"])
        print(f"fully pilot-based, v3 predictor (max over comparisons): corr = {np.corrcoef(t.saving_pred_v3_med, t.saving_obs)[0,1]:.3f}; MAE = {np.abs(t.saving_pred_v3_med - t.saving_obs).mean():.3f}")
        print("post-pilot labels of weighted: realised vs v1 vs v2 vs v3 (medians): " + "; ".join(f"{r.collection}/{r.judge} {r.L_post_obs:.0f} vs {r.L_post_pred_med:.0f} vs {r.L_post_pred_v2_med:.0f} vs {r.L_post_pred_v3_med:.0f}" for r in u.itertuples()))
        u = t.drop_duplicates(["collection", "judge"])
        print(f"post-pilot labels of weighted: realised vs pilot-predicted median: " + "; ".join(f"{r.collection}/{r.judge} {r.L_post_obs:.0f} vs {r.L_post_pred_med:.0f} [{r.L_post_pred_q25:.0f}, {r.L_post_pred_q75:.0f}]" for r in u.itertuples()))
        jd = t[t.method.isin(["weighted_cvl", "weighted_cv", "ai_resid", "ai_robust_0.3", "ai_calib"])]
        for col, lab in [("frac_draws_recommend", "share from realised J50"), ("frac_draws_recommend_pilot", "fully pilot-based v1"), ("frac_draws_recommend_v2", "fully pilot-based v2"), ("frac_draws_recommend_v3", "fully pilot-based v3")]:
            dec = (jd[col] > 0.5).astype(int)
            print(f"decision 'saving > {a.tau}' ({lab}) vs cell truth: accuracy {(dec == jd.cell_truth).mean():.2f} on {len(jd)} judge cells; "
                  f"false recommendations {int(((dec == 1) & (jd.cell_truth == 0)).sum())}, missed {int(((dec == 0) & (jd.cell_truth == 1)).sum())}")


if __name__ == "__main__":
    main()
