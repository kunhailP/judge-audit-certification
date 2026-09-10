"""Certificate constructions for the Decision-Specific Audit Planner (v0.4 work).

Fixes and additions relative to 40_planner_replay.py (frozen v0.3):

1. Recalibration certificate scans *every breakpoint* of the gate inside the
   confidence interval for c, instead of the two endpoints.  Proposition 2 of
   DRAFT_v0.3 needs sup_{c in C_t} |u(c) - u(c_hat)|; the gate is piecewise
   constant in c so the sup is attained on the finite breakpoint set.

2. Selection certificates:
     * "loo"   — legacy leave-one-out cross-fitted paired bootstrap (kept as a
                 baseline; validity is empirical, see DRAFT §4.1).
     * "split" — train/validation split of the audit sample: policy
                 parameters (c_hat, tau) are fitted on the first half, paired
                 differences are evaluated on the second half, one-sided
                 t-based UCBs with Bonferroni over looks x (M-1).  Validity
                 follows from Proposition 1 with exact (t) bounds under
                 approximate normality of query-level differences.
     * "ppi"   — prediction-powered (PPI++) paired certificate: AI-judge
                 utilities for *all* target queries plus human-audited
                 differences on the validation half.  Same Bonferroni scheme.

All functions are pure numpy; structs follow 20_budget_curves.build_structs.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# gate breakpoints
# ----------------------------------------------------------------------------
def gate_prefix_max(s):
    """Thresholds c_i with gate(c) = min{i : c <= c_i}; returned as prefix-max
    so that gate(c) = searchsorted(P, c, 'left')."""
    sp, cump, Sp = s["sp"], s["cump"], s["Sp"]
    i = np.arange(len(sp))
    with np.errstate(divide="ignore", invalid="ignore"):
        c_i = (cump / np.maximum(sp, 1e-300) - (i + 1.0)) / Sp
    return np.maximum.accumulate(c_i)


def f1_curve(s):
    """F1 at k = 0..pool (index k)."""
    k = np.arange(1, len(s["sp"]) + 1)
    tp = s["cumtp"]
    f = np.where(tp > 0, 2.0 * tp / (k + s["nG"]), 0.0)
    return np.r_[0.0, f]


def prepare_gate(structs):
    for s in structs:
        if "_P" not in s:
            s["_P"] = gate_prefix_max(s)
            s["_F"] = f1_curve(s)
    return structs


def u_gate_at(structs, cs):
    """Mean set-F1 of the gate policy for an array of scalars cs. gate(c)=0 if c<=0."""
    cs = np.atleast_1d(np.asarray(cs, dtype=float))
    acc = np.zeros(len(cs))
    for s in structs:
        k = np.searchsorted(s["_P"], cs, side="left")
        k = np.where(cs <= 0, 0, k)
        acc += s["_F"][k]
    return acc / len(structs)


def recal_spread(structs, c_hat, c_lo, c_hi, mode="breakpoints"):
    """sup_{c in [c_lo,c_hi]} |u(c) - u(c_hat)| on the audited sample.

    mode="endpoints" reproduces the v0.3 computation (ablation only)."""
    prepare_gate(structs)
    u_hat = u_gate_at(structs, c_hat)[0]
    if mode == "endpoints":
        grid = np.array([c_lo, c_hi])
    else:
        bps = np.concatenate([s["_P"] for s in structs])
        bps = bps[(bps >= c_lo) & (bps <= c_hi) & np.isfinite(bps)]
        grid = np.unique(np.r_[c_lo, c_hi, bps])
    u = u_gate_at(structs, grid)
    return float(np.max(np.abs(u - u_hat))), len(grid)


def recal_spread_ucb(structs, c_hat, c_lo, c_hi, alpha, rng, boot=200):
    """Bootstrap upper confidence bound (level 1-alpha) on the *population*
    sup_{c in [c_lo,c_hi]} |u(c) - u(c_hat)|: resample audited queries, take the
    sup of the resampled mean-difference curve over the breakpoint grid, and
    return its (1-alpha) quantile.  Addresses DRAFT v0.3 §4.1 step (ii)."""
    prepare_gate(structs)
    bps = np.concatenate([s["_P"] for s in structs])
    bps = bps[(bps >= c_lo) & (bps <= c_hi) & np.isfinite(bps)]
    grid = np.unique(np.r_[c_lo, c_hi, bps])
    T = len(structs)
    Uq = np.empty((T, len(grid)))
    for i, s in enumerate(structs):
        k = np.searchsorted(s["_P"], grid, side="left"); k = np.where(grid <= 0, 0, k)
        Uq[i] = s["_F"][k]
        kh = int(np.searchsorted(s["_P"], c_hat, side="left")) if c_hat > 0 else 0
        Uq[i] -= s["_F"][kh]
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=boot) / T          # (boot, T)
    sup = np.abs(W @ Uq).max(axis=1)
    return float(np.quantile(sup, 1 - alpha)), len(grid)


def median_ci_exact(x, alpha):
    """Distribution-free two-sided (1-alpha) CI for the population median from
    order statistics (sign-test inversion).  Returns (-inf, inf) when n is too
    small for the requested level — the planner then cannot certify."""
    from scipy.stats import binom
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if n == 0:
        return -np.inf, np.inf
    k = int(binom.ppf(alpha / 2, n, 0.5))          # P(Bin <= k) >= alpha/2 ...
    while k > 0 and binom.cdf(k - 1, n, 0.5) > alpha / 2:
        k -= 1
    # need P(Bin <= k-1) <= alpha/2 with k >= 1
    while k >= 1 and binom.cdf(k - 1, n, 0.5) > alpha / 2:
        k -= 1
    if k < 1:
        return -np.inf, np.inf
    return float(x[k - 1]), float(x[n - k])


LAMBDA_GUARD = False      # post-hoc deviation: lambda=0 unless it reduces the fitting fold's PPI variance by >= 5%


def crossfit_lambda(d, dh, n_over_N=0.0):
    """Cross-fitted PPI++ tuning (deterministic halves): lambda for each half
    is estimated on the other half, removing the in-sample optimism of a
    lambda tuned and applied on the same audited queries.

    n_over_N: ratio of labeled to unlabeled sample sizes. The variance-optimal
    PPI++ tuning with a finite unlabeled set is lambda* = Cov(d,dh)/Var(dh) /
    (1 + n/N) (Angelopoulos et al. 2023, PPI++); the default 0.0 is the
    N -> infinity formula used in the LOCK v0.4 primary run."""
    n = len(d); a, b = np.arange(n // 2), np.arange(n // 2, n)
    lam = np.zeros(n)
    for fit, apply in [(a, b), (b, a)]:
        v = dh[fit].var(ddof=1) if len(fit) > 1 else 0.0
        raw = np.cov(d[fit], dh[fit])[0, 1] / v if v > 0 and len(fit) > 1 else 0.0
        l = float(np.clip(raw / (1.0 + n_over_N), 0, 1))
        if LAMBDA_GUARD and len(fit) > 1:
            base = d[fit].var(ddof=1)
            with_l = (d[fit] - l * dh[fit]).var(ddof=1) + (l ** 2) * v * n_over_N
            if base <= 0 or with_l > 0.95 * base:
                l = 0.0
        lam[apply] = l
    return lam


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def t_quantile(p, df):
    """Upper-tail t quantile via scipy if available, else normal fallback."""
    try:
        from scipy.stats import t
        return float(t.ppf(p, df))
    except Exception:
        lo, hi = 0.0, 20.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def bonferroni_alpha(alpha, n_looks, n_menu, simultaneous=False):
    """Per-comparison level.  simultaneous=True corrects over all M(M-1)
    ordered pairs so that a candidate chosen *from the same data* is covered
    (the (M-1)-only correction assumes the candidate was fixed in advance and
    is anticonservative under data-driven candidate choice; see self-test)."""
    n_cmp = n_menu * (n_menu - 1) if simultaneous else max(n_menu - 1, 1)
    return alpha / (n_looks * n_cmp)


# ----------------------------------------------------------------------------
# selection certificates.  U: (n, M) human-audited utilities.
# ----------------------------------------------------------------------------
def pick_candidate(U):
    means = U.mean(axis=0)
    se = U.std(axis=0, ddof=1) / math.sqrt(len(U))
    return int(np.argmax(means - se))


def ucb_bootstrap(U, cand, alpha_prime, rng, boot=400):
    bs = rng.integers(0, len(U), (boot, len(U)))
    bmeans = U[bs].mean(axis=1)
    diffs = bmeans - bmeans[:, [cand]]
    ucb = np.quantile(diffs, 1 - alpha_prime, axis=0)
    ucb[cand] = -np.inf
    return ucb


def ucb_t(U, cand, alpha_prime):
    """One-sided t UCB for mu_j - mu_cand from paired query differences."""
    n = len(U)
    D = U - U[:, [cand]]
    m = D.mean(axis=0)
    sd = D.std(axis=0, ddof=1)
    q = t_quantile(1 - alpha_prime, n - 1)
    ucb = m + q * sd / math.sqrt(n)
    ucb[cand] = -np.inf
    return ucb


LAMBDA_FINITE_N = False   # post-hoc deviation switch (see design doc §14); LOCK v0.4 runs use False


def ucb_ppi(U, Uhat_lab, Uhat_all, cand, alpha_prime, crossfit=True, exclude_idx=None):
    """PPI++ one-sided UCB for mu_j - mu_cand.

    U        (n, M) human utilities on audited queries
    Uhat_lab (n, M) judge utilities on the same audited queries
    Uhat_all (N, M) judge utilities on all target queries (incl. audited)
    """
    if exclude_idx is not None and len(exclude_idx) < len(Uhat_all) - 2:
        keep = np.ones(len(Uhat_all), bool); keep[np.asarray(exclude_idx)] = False
        Uhat_all = Uhat_all[keep]           # Proposition B': N-mean taken outside the validation half
    n, N = len(U), len(Uhat_all)
    D = U - U[:, [cand]]
    Dh_lab = Uhat_lab - Uhat_lab[:, [cand]]
    Dh_all = Uhat_all - Uhat_all[:, [cand]]
    M = U.shape[1]
    ucb = np.full(M, -np.inf)
    q = t_quantile(1 - alpha_prime, n - 1)
    for j in range(M):
        if j == cand:
            continue
        if crossfit and n >= 6:
            lam_vec = crossfit_lambda(D[:, j], Dh_lab[:, j], n_over_N=(n / N if LAMBDA_FINITE_N else 0.0))
        else:
            v = Dh_lab[:, j].var(ddof=1)
            lam_vec = np.full(n, float(np.clip(np.cov(D[:, j], Dh_lab[:, j])[0, 1] / v, 0.0, 1.0)) if v > 0 else 0.0)
        lam = float(lam_vec.mean())
        resid = D[:, j] - lam_vec * Dh_lab[:, j]
        est = lam * Dh_all[:, j].mean() + resid.mean()
        se = math.sqrt(lam ** 2 * Dh_all[:, j].var(ddof=1) / N + resid.var(ddof=1) / n)
        ucb[j] = est + q * se
    return ucb


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    # 1. breakpoint scan finds interior utility change that endpoints miss
    # struct: probabilities 0.9,0.6,0.5,0.4,0.1 ; relevance 1,0,1,0,0 ; nG=2
    sp = np.array([0.9, 0.6, 0.5, 0.4, 0.1]); rel = np.array([1, 0, 1, 0, 0])
    s = dict(sp=sp, cump=np.cumsum(sp), Sp=float(sp.sum()), cumtp=np.cumsum(rel), nG=2)
    prepare_gate([s])
    from importlib import util as _u
    import os
    spec = _u.spec_from_file_location("bc", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "20_budget_curves.py"))
    bc = _u.module_from_spec(spec); spec.loader.exec_module(bc)
    cs = np.linspace(0.01, 3.0, 300)
    mine = u_gate_at([s], cs)
    ref = np.array([bc.f1_at_k(s["cumtp"], bc.gate_k(sp, s["cump"], c * s["Sp"]), 2) for c in cs])
    assert np.allclose(mine, ref), "u_gate_at disagrees with gate_k/f1_at_k"
    # find an interval whose endpoints have equal utility but interior differs
    found = False
    for lo in cs[::5]:
        for hi in cs[::5]:
            if hi <= lo:
                continue
            e, _ = recal_spread([s], lo, lo, hi, mode="endpoints")
            b, _ = recal_spread([s], lo, lo, hi, mode="breakpoints")
            if e == 0.0 and b > 0.1:
                found = True; break
        if found:
            break
    assert found, "expected an interval where endpoints miss interior change"
    print(f"[PASS] breakpoint scan: endpoint spread 0.000 vs breakpoint spread {b:.3f} on [{lo:.2f},{hi:.2f}]")
    # 2. t / ppi UCB coverage on synthetic paired data (no judge signal -> ppi ~ t)
    N, n, M = 5000, 40, 3
    base = rng.beta(2, 4, N)
    true = np.array([0.0, -0.02, -0.05])
    Uall = np.clip(base[:, None] + true[None, :] + rng.normal(0, 0.15, (N, M)), 0, 1)
    mu = Uall.mean(axis=0)
    Uhat = np.clip(Uall + rng.normal(0, 0.05, Uall.shape), 0, 1)   # good judge
    R = 1000
    for simul in [False, True]:
        ap = bonferroni_alpha(0.1, 5, M, simultaneous=simul)
        budget = 0.1 / 5                      # per-look family-wise budget
        miss_t = miss_p = 0
        for rep in range(R):
            idx = rng.choice(N, n, replace=False)
            U = Uall[idx]
            cand = pick_candidate(U)          # data-driven candidate
            ut = ucb_t(U, cand, ap)
            up = ucb_ppi(U, Uhat[idx], Uhat, cand, ap)
            gap = mu - mu[cand]; others = np.arange(M) != cand
            miss_t += bool((ut[others] < gap[others]).any())
            miss_p += bool((up[others] < gap[others]).any())
        tag = "simultaneous M(M-1)" if simul else "(M-1)-only, candidate from same data"
        print(f"[INFO] per-look miss rate, {tag}: t={miss_t/R:.3f} ppi={miss_p/R:.3f} (budget {budget:.3f})")
        if simul:
            assert miss_t / R <= budget + 0.01 and miss_p / R <= budget + 0.01
    print("[PASS] simultaneous t and PPI UCBs cover under data-driven candidate choice")


if __name__ == "__main__":
    _selftest()
