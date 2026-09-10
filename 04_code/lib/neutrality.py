"""Judge-neutrality diagnostic (2027 design §3.2).

From a pilot of n0 human-audited queries with true paired differences d and
judge paired differences dh for one policy pair, produce:

  rho_hat, rho_lcb        correlation and its Fisher-z lower confidence bound
  gain_lcb                lower confidence bound on the PPI effective-sample-
                          size gain 1/(1-rho^2)  (from rho_lcb; 1 if rho_lcb<=0)
  T_human, T_ppi          projected human audits needed to certify a gap of
                          eps at one-sided level alpha (normal approx.),
                          humans-only vs PPI with cross-fitted lambda
  band_test               two-proportion z-test of judge error inside vs
                          outside the disagreement band (p-value, one-sided:
                          in-band error larger)

A planner uses gain_lcb to decide whether the judge is worth using for this
decision, and T_ppi/T_human to plan the budget before spending it.
"""
import math
import numpy as np


def _z(p):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rho_lcb(d, dh, alpha):
    n = len(d)
    if n < 4 or d.std() == 0 or dh.std() == 0:
        return float("nan"), -1.0
    r = float(np.clip(np.corrcoef(d, dh)[0, 1], -0.999999, 0.999999))
    z = math.atanh(r) - _z(1 - alpha) / math.sqrt(n - 3)
    return r, math.tanh(z)


def rho_lcb_boot(d, dh, alpha, rng=None, boot=1000):
    """Bootstrap percentile lower bound on rho (robust to non-normal, zero-inflated
    differences where Fisher-z under-covers)."""
    n = len(d)
    if n < 4 or d.std() == 0 or dh.std() == 0:
        return float("nan"), -1.0
    rng = np.random.default_rng(0) if rng is None else rng
    r = float(np.corrcoef(d, dh)[0, 1])
    idx = rng.integers(0, n, (boot, n))
    D, Dh = d[idx], dh[idx]
    Dc, Dhc = D - D.mean(1, keepdims=True), Dh - Dh.mean(1, keepdims=True)
    den = np.sqrt((Dc ** 2).sum(1) * (Dhc ** 2).sum(1))
    rb = np.where(den > 0, (Dc * Dhc).sum(1) / np.maximum(den, 1e-12), 0.0)
    return r, float(np.quantile(rb, alpha))


def _corr_rows(D, Dh):
    Dc, Dhc = D - D.mean(1, keepdims=True), Dh - Dh.mean(1, keepdims=True)
    den = np.sqrt((Dc ** 2).sum(1) * (Dhc ** 2).sum(1))
    return np.where(den > 0, (Dc * Dhc).sum(1) / np.maximum(den, 1e-12), 0.0)


def rho_lcb_bca(d, dh, alpha, rng=None, boot=1000):
    """BCa bootstrap lower bound on rho (bias-corrected, jackknife-accelerated)."""
    n = len(d)
    if n < 6 or d.std() == 0 or dh.std() == 0:
        return float("nan"), -1.0
    rng = np.random.default_rng(0) if rng is None else rng
    r = float(np.corrcoef(d, dh)[0, 1])
    idx = rng.integers(0, n, (boot, n))
    rb = _corr_rows(d[idx], dh[idx])
    # bias correction
    p0 = float(np.clip((rb < r).mean(), 1e-6, 1 - 1e-6))
    z0 = _z(p0)
    # jackknife acceleration
    jk = np.empty(n)
    mask = ~np.eye(n, dtype=bool)
    for i in range(n):
        jk[i] = np.corrcoef(d[mask[i]], dh[mask[i]])[0, 1] if d[mask[i]].std() > 0 and dh[mask[i]].std() > 0 else r
    jm = jk.mean(); num = ((jm - jk) ** 3).sum(); den = 6.0 * (((jm - jk) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    za = _z(alpha)
    adj = z0 + (z0 + za) / max(1 - a * (z0 + za), 1e-6)
    q = 0.5 * (1 + math.erf(adj / math.sqrt(2)))
    q = float(np.clip(q, 1e-4, 1 - 1e-4))
    return r, float(np.quantile(rb, q))


def crossfit_lambda(d, dh, rng=None):
    """Cross-fitted PPI++ tuning: lambda for each half estimated on the other."""
    n = len(d)
    idx = np.arange(n) if rng is None else rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    lam = np.zeros(n)
    for fit, apply in [(a, b), (b, a)]:
        v = dh[fit].var(ddof=1) if len(fit) > 1 else 0.0
        l = float(np.clip(np.cov(d[fit], dh[fit])[0, 1] / v, 0, 1)) if v > 0 and len(fit) > 1 else 0.0
        lam[apply] = l
    return lam


def budget_projection(d, dh, eps, alpha, N_unlabeled=np.inf, rng=None):
    """Projected T so that z_{1-alpha} * SE(T) <= eps for the paired mean."""
    z = _z(1 - alpha)
    sd_h = d.std(ddof=1)
    lam = crossfit_lambda(d, dh, rng)
    resid = d - lam * dh
    sd_p = resid.std(ddof=1)
    T_h = (z * sd_h / eps) ** 2
    extra = (lam.mean() ** 2) * dh.var(ddof=1) / N_unlabeled if np.isfinite(N_unlabeled) else 0.0
    # solve z^2 (sd_p^2 / T + extra) = eps^2
    denom = eps ** 2 / z ** 2 - extra
    T_p = (sd_p ** 2 / denom) if denom > 0 else float("inf")
    return float(T_h), float(T_p), float(lam.mean())


def band_test(err_in, n_in, err_out, n_out):
    """One-sided two-proportion z-test H1: error inside band > outside."""
    if n_in == 0 or n_out == 0:
        return float("nan"), float("nan")
    p = (err_in * n_in + err_out * n_out) / (n_in + n_out)
    se = math.sqrt(p * (1 - p) * (1 / n_in + 1 / n_out)) if 0 < p < 1 else float("nan")
    if not se or se != se:
        return float("nan"), float("nan")
    zst = (err_in - err_out) / se
    pval = 1 - 0.5 * (1 + math.erf(zst / math.sqrt(2)))
    return float(zst), float(pval)


def diagnose(d, dh, eps, alpha, N_unlabeled=np.inf, band=None, rng=None, method="bca"):
    if method == "bca":
        r, lcb = rho_lcb_bca(d, dh, alpha, rng)
    elif method == "boot":
        r, lcb = rho_lcb_boot(d, dh, alpha, rng)
    else:
        r, lcb = rho_lcb(d, dh, alpha)
    gain_lcb = 1.0 / (1 - lcb ** 2) if lcb > 0 else 1.0
    T_h, T_p, lam = budget_projection(d, dh, eps, alpha, N_unlabeled, rng)
    out = dict(n_pilot=len(d), rho_hat=r, rho_lcb=lcb, gain_lcb=gain_lcb, T_human=T_h, T_ppi=T_p, lam=lam,
               use_judge=bool(gain_lcb > 1.1))
    if band is not None:
        out["band_z"], out["band_p"] = band_test(*band)
    return out


def _selftest():
    rng = np.random.default_rng(0)
    # coverage of rho_lcb and of gain_lcb at alpha=0.1, n0 in {10, 20, 40}
    N = 200000
    for rho_true in [0.0, 0.5, 0.8]:
        x = rng.normal(size=N); e = rng.normal(size=N)
        d = x; dh = rho_true * x + math.sqrt(1 - rho_true ** 2) * e
        for n0 in [10, 20, 40]:
            miss = 0
            for _ in range(2000):
                idx = rng.choice(N, n0, replace=False)
                _, lcb = rho_lcb(d[idx], dh[idx], 0.1)
                miss += lcb > rho_true
            print(f"[INFO] rho={rho_true} n0={n0}: P(lcb > rho) = {miss/2000:.3f} (nominal 0.10)")
            assert miss / 2000 <= 0.13
    # zero-inflated, skewed paired differences (set-F1 like): Fisher-z vs bootstrap LCB coverage
    for rho_target in [0.0, 0.5, 0.9]:
        x = rng.normal(size=N); e = rng.normal(size=N)
        d = np.where(rng.random(N) < 0.6, 0.0, 0.2 * np.exp(0.5 * x))
        dh = np.where(rng.random(N) < 0.6, 0.0, 0.2 * np.exp(0.5 * (rho_target * x + math.sqrt(1 - rho_target ** 2) * e)))
        rho_pop = float(np.corrcoef(d, dh)[0, 1])
        for n0 in [10, 30]:
            mf = mb = mc = 0
            for _ in range(1000):
                idx = rng.choice(N, n0, replace=False)
                mf += rho_lcb(d[idx], dh[idx], 0.1)[1] > rho_pop
                mb += rho_lcb_boot(d[idx], dh[idx], 0.1, rng, boot=400)[1] > rho_pop
                mc += rho_lcb_bca(d[idx], dh[idx], 0.1, rng, boot=400)[1] > rho_pop
            print(f"[INFO] zero-inflated rho_pop={rho_pop:.2f} n0={n0}: miss fisher={mf/1000:.3f} boot={mb/1000:.3f} bca={mc/1000:.3f} (nominal 0.10)")
    # budget projection sanity: PPI projection <= human projection, ratio ~ 1-rho^2
    x = rng.normal(size=N); e = rng.normal(size=N); d = 0.02 + 0.1 * x; dh = 0.1 * (0.8 * x + 0.6 * e)
    idx = rng.choice(N, 40, replace=False)
    T_h, T_p, lam = budget_projection(d[idx], dh[idx], 0.01, 0.1)
    print(f"[INFO] projected T human={T_h:.0f} ppi={T_p:.0f} ratio={T_p/T_h:.2f} (1-rho^2=0.36) lam={lam:.2f}")
    assert T_p < T_h
    print("[PASS] neutrality diagnostic self-test")


if __name__ == "__main__":
    _selftest()
