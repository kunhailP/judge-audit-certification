#!/usr/bin/env python3
"""Query-level certificate calculator: does (rho, sigma, gap, n, N_unlab, eps) predict the mechanism map? (2026-09-17)

The map (77_rho_map.py, F6) measured on real data the ACT ratio PPI/humans of the split certificate at T audited queries
(T/2 validation) with lambda cross-fitted and clipped to [0,1]. Proposition A gives the *variance* gain at the oracle
lambda; the certificate is a threshold event on a t bound with an *estimated* lambda. This script runs the identical
certificate code (lib/certificates.py: pick_candidate, ucb_t, ucb_ppi) on Gaussian paired differences whose parameters are
read from the released ppi_gain files (per-pair population gap, sigma = se_human·sqrt(T), rho), and compares the simulated
ACT ratio with the stored map cell by cell. Variants isolate the mechanisms: oracle lambda (no estimation noise), no
clipping, finite-N lambda. If the simulation reproduces the map, the map is predictable before the audit from quantities a
pilot can estimate (rho, sigma, gap) plus design quantities (n, N_unlab, eps).

Usage: python3 96_ppi_calculator.py [--draws 2000]
Outputs: 04_results/rho_map/CALCULATOR_vs_map.csv, F9_calculator.png
"""
import argparse, importlib.util, math, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); HUB = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("cert", os.path.join(HERE, "lib", "certificates.py"))
cert = importlib.util.module_from_spec(spec); spec.loader.exec_module(cert)
ALPHA = 0.10; M = 3; A_SIM = ALPHA / (M * (M - 1))
R = os.path.join(HUB, "04_results")
MENU = ["glob_probe", "trunc", "ad_probe"]


def pair_params(collection, judge_file):
    """Population gaps mu_m (glob_probe = 0), per-pair sigma (from se_human at T=10) and rho, from ppi_gain files."""
    d = pd.read_csv(os.path.join(R, "ppi_gain", f"ppi_gain_{judge_file}.csv"))
    d = d[(d.collection == collection) & (d["T"] == 10)]
    g = {r.pair: r for r in d.itertuples()}
    mu = {"glob_probe": 0.0}
    mu["trunc"] = -g["glob_probe_vs_trunc"].true_gap; mu["ad_probe"] = -g["glob_probe_vs_ad_probe"].true_gap
    sig = {("glob_probe", "trunc"): g["glob_probe_vs_trunc"].se_human * math.sqrt(10),
           ("glob_probe", "ad_probe"): g["glob_probe_vs_ad_probe"].se_human * math.sqrt(10),
           ("trunc", "ad_probe"): g["trunc_vs_ad_probe"].se_human * math.sqrt(10)}
    rho = {("glob_probe", "trunc"): g["glob_probe_vs_trunc"].rho, ("glob_probe", "ad_probe"): g["glob_probe_vs_ad_probe"].rho,
           ("trunc", "ad_probe"): g["trunc_vs_ad_probe"].rho}
    return mu, sig, rho


def utility_cov(sig):
    """Per-policy covariance of utilities consistent with the three paired-difference variances (plus a common query effect)."""
    s_gt, s_ga, s_ta = sig[("glob_probe", "trunc")] ** 2, sig[("glob_probe", "ad_probe")] ** 2, sig[("trunc", "ad_probe")] ** 2
    # Var(u_i - u_j) = C_ii + C_jj - 2 C_ij. Choose C = c0 * 11' + diag(v) + off-diagonals so that all three match:
    # with a common component c0 (cancels in differences) and residual covariance K, K_ii + K_jj - 2K_ij = s_ij.
    # Take K = diag(v) + w * (11' - I) style: use the general solution with K_ij chosen to make K PSD.
    # Simplest PSD construction: embed as distances -> Gram matrix (classical MDS on 3 points with squared distances s).
    D2 = np.array([[0, s_gt, s_ga], [s_gt, 0, s_ta], [s_ga, s_ta, 0]])
    Jc = np.eye(3) - np.ones((3, 3)) / 3
    K = -0.5 * Jc @ D2 @ Jc                      # Gram matrix of the three points (PSD iff distances are Euclidean)
    w, V = np.linalg.eigh(K); w = np.clip(w, 0, None); K = (V * w) @ V.T
    return K + 0.04 * np.ones((3, 3))            # common query effect (sd 0.2) that cancels in every pair


def simulate(mu, sig, rho, T, N_unlab, eps, draws, rng, lam_mode="crossfit", sig_hat_scale=1.0):
    K = utility_cov(sig); L = np.linalg.cholesky(K + 1e-9 * np.eye(3))
    mvec = np.array([mu[m] for m in MENU])
    # judge: u_hat_m = u_m * a + e_m with the pair correlations reproduced approximately by a common attenuation:
    # D_hat_ij = rho_ij * D_ij * (sig_hat/sig) + sqrt(1 - rho_ij^2) * sig_hat_ij * xi. We generate it pairwise against the
    # candidate later, so we only need a per-pair construction; a per-policy judge with matching pair rho is built by
    # scaling a shared noise: u_hat = r * u + sqrt(1 - r^2) * noise_with_same_cov, r = mean pair rho.
    r = float(np.mean(list(rho.values())))
    act_t = act_p = 0
    n_val = T - T // 2
    for _ in range(draws):
        Z = rng.standard_normal((T + N_unlab, 3)); U = mvec + Z @ L.T
        Zh = rng.standard_normal((T + N_unlab, 3)); Uh = sig_hat_scale * (r * (U - mvec) + math.sqrt(max(1 - r * r, 0)) * (Zh @ L.T)) + mvec
        va = slice(T // 2, T); unl = slice(T, None)
        Uv = U[va]; cand = cert.pick_candidate(Uv)
        ut = cert.ucb_t(Uv, cand, A_SIM)
        if lam_mode == "crossfit":
            up = cert.ucb_ppi(Uv, Uh[va], Uh[unl], cand, A_SIM)
        else:
            up = ucb_ppi_variant(Uv, Uh[va], Uh[unl], cand, A_SIM, lam_mode, K, r, sig_hat_scale)
        act_t += ut.max() <= eps; act_p += up.max() <= eps
    return act_t / draws, act_p / draws


def ucb_ppi_variant(U, Uhat_lab, Uhat_all, cand, alpha_prime, mode, K, r, s):
    """oracle: lambda = population Cov/Var (no estimation noise), clipped to [0,1]; oracle_noclip: unclipped;
    finiteN: cross-fitted lambda divided by (1 + n/N)."""
    n, N = len(U), len(Uhat_all); D = U - U[:, [cand]]; Dh = Uhat_lab - Uhat_lab[:, [cand]]; Da = Uhat_all - Uhat_all[:, [cand]]
    ucb = np.full(U.shape[1], -np.inf); q = cert.t_quantile(1 - alpha_prime, n - 1)
    for j in range(U.shape[1]):
        if j == cand:
            continue
        if mode.startswith("oracle"):
            vD = K[j, j] + K[cand, cand] - 2 * K[j, cand]; cov = r * s * vD; vDh = s * s * vD
            lam = cov / vDh
            lam = lam if mode == "oracle_noclip" else float(np.clip(lam, 0, 1))
            lam_vec = np.full(n, lam)
        else:
            lam_vec = cert.crossfit_lambda(D[:, j], Dh[:, j], n_over_N=n / N)
        lam = float(lam_vec.mean()); resid = D[:, j] - lam_vec * Dh[:, j]
        est = lam * Da[:, j].mean() + resid.mean(); se = math.sqrt(lam ** 2 * Da[:, j].var(ddof=1) / N + resid.var(ddof=1) / n)
        ucb[j] = est + q * se
    return ucb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); rows = []
    for coll, jf, mapfile in [("dbpedia-entity", "judged_llm", "rho_map_judged_llm.csv"), ("dl212223", "dlv2_llm", "rho_map_dlv2_llm.csv")]:
        mu, sig, rho0 = pair_params(coll, jf)
        mp = pd.read_csv(os.path.join(R, "rho_map", mapfile))
        rho_levels = mp.groupby("noise").rho.first().to_dict(); r0 = rho_levels[0.0]
        for noise, rmean in rho_levels.items():
            rho = {k: v * rmean / r0 for k, v in rho0.items()}     # noise attenuates every pair's rho proportionally
            for T in sorted(mp["T"].unique()):
                for N_unlab in sorted(mp.N_unlab.unique()):
                    for eps in sorted(mp.eps.unique()):
                        cell = mp[(mp.noise == noise) & (mp["T"] == T) & (mp.N_unlab == N_unlab) & (mp.eps == eps)]
                        if cell.empty:
                            continue
                        out = dict(collection=coll, noise=noise, rho=rmean, T=int(T), N_unlab=int(N_unlab), eps=eps,
                                   act_t_map=float(cell.act_t.iloc[0]), act_ppi_map=float(cell.act_ppi.iloc[0]), ratio_map=float(cell.ratio.iloc[0]))
                        for mode in ["crossfit", "oracle", "oracle_noclip", "finiteN"]:
                            at, ap_ = simulate(mu, sig, rho, int(T), int(N_unlab), eps, a.draws, rng, lam_mode=mode)
                            out[f"act_t_sim"] = at; out[f"act_ppi_{mode}"] = ap_; out[f"ratio_{mode}"] = ap_ / at if at > 0 else float("nan")
                        rows.append(out)
                print(f"{coll} noise={noise} rho={rmean:.2f} T={T} done", flush=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(R, "rho_map", "CALCULATOR_vs_map.csv"), index=False)
    pd.set_option("display.width", 250)
    for coll, g in t.groupby("collection"):
        g9 = g[g["T"] == g["T"].max()]
        print(f"\n== {coll}: T={g['T'].max()} cells n={len(g9)}")
        print(f"  humans-only ACT: corr(sim, map)={np.corrcoef(g9.act_t_sim, g9.act_t_map)[0,1]:.3f}  MAE={np.abs(g9.act_t_sim-g9.act_t_map).mean():.3f}")
        for mode in ["crossfit", "oracle", "oracle_noclip", "finiteN"]:
            ok = np.isfinite(g9[f"ratio_{mode}"]) & np.isfinite(g9.ratio_map)
            print(f"  ratio {mode:14s}: corr(sim, map)={np.corrcoef(g9[f'ratio_{mode}'][ok], g9.ratio_map[ok])[0,1]:.3f}  MAE={np.abs(g9[f'ratio_{mode}']-g9.ratio_map)[ok].mean():.3f}"
                  f"  cells>1.2: sim {(g9[f'ratio_{mode}']>1.2).sum()} vs map {(g9.ratio_map>1.2).sum()}  agreement on >1.2: {((g9[f'ratio_{mode}']>1.2)==(g9.ratio_map>1.2)).mean():.2f}")
        print(g9[["noise", "rho", "N_unlab", "eps", "act_t_map", "act_t_sim", "ratio_map", "ratio_crossfit", "ratio_oracle", "ratio_oracle_noclip", "ratio_finiteN"]].round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i, (coll, g) in enumerate(t[t["T"] == 90].groupby("collection")):
        for mode, mk in [("crossfit", "o"), ("oracle", "^")]:
            ax[i].scatter(g.ratio_map, g[f"ratio_{mode}"], marker=mk, s=20, alpha=0.8, label=f"λ {mode}", edgecolor="k", linewidth=0.3)
        ax[i].plot([0.7, 1.8], [0.7, 1.8], "k--", lw=0.8); ax[i].set_title(f"{coll}, T=90: simulated vs measured ACT ratio", fontsize=9)
        ax[i].set_xlabel("measured ACT(PPI)/ACT(humans) (F6)"); ax[i].set_ylabel("simulated from (ρ, σ, gap, n, N, ε)"); ax[i].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "rho_map", "F9_calculator.png"), dpi=200)


if __name__ == "__main__":
    main()
