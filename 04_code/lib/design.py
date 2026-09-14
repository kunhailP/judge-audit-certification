"""Exact oracle allocation for certifying a whole menu (review 2026-09-14, item 6).

Setting. Candidate m̂ fixed. For each competitor j the certificate needs an upper bound on μ_j − μ_m̂ ≤ ε; with
slack s_j = ε − (μ_j − μ_m̂) > 0 the comparison certifies when z·se_j ≤ s_j, i.e. when V_j/s_j² is small enough,
V_j(π) = Σ_i a_{j,i}² v_i (1 − π_i)/π_i  (Poisson sampling with inclusion probabilities π_i over all decision
documents i of the audited queries; a_{j,i} = w_j(q_i, d_i)/n_q; v_i = E[(y_i − ĵ_i)²] the residual variance).
The oracle design problem is

        minimise_π  max_j  V_j(π)/s_j²     subject to  Σ_i c_i π_i ≤ B,  0 < π_i ≤ 1.

This is a convex program (each V_j is convex in π). We solve it through its Lagrangian dual: for θ in the simplex
the inner problem min_π Σ_j θ_j V_j/s_j² s.t. the budget has the closed form π_i = min(1, sqrt(g_i(θ)/(μ c_i)))
with g_i(θ) = Σ_j θ_j a_{j,i}² v_i/s_j² (water-filling on μ), and the outer max over θ is concave; we run
exponentiated-gradient ascent on θ and return the best primal iterate. J is small (|M|−1), so this is cheap.

Static heuristics for comparison: π ∝ Σ_j |w_j| (label-sharing static sum, the strong baseline of 83) and the
per-pair rule π ∝ |w_j| √v_i. `evaluate()` returns max_j V_j/s_j² and each V_j for any π.
"""
import numpy as np


def waterfill(g, c, B):
    """argmin_π Σ_i g_i (1-π_i)/π_i  s.t. Σ c_i π_i <= B, 0 < π_i <= 1;  π_i = min(1, sqrt(g_i/(μ c_i)))."""
    g = np.asarray(g, float); c = np.asarray(c, float); n = len(g)
    pos = g > 0
    if not pos.any():
        return np.full(n, min(1.0, B / c.sum()))
    lo, hi = 0.0, 1.0
    f = lambda mu: np.minimum(1.0, np.sqrt(g / (mu * c)))
    while (c * f(hi)).sum() > B:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (c * f(mid)).sum() > B:
            lo = mid
        else:
            hi = mid
    pi = f(hi)
    pi[~pos] = 0.0
    return pi


def evaluate(pi, A, v, s):
    """V_j(π) = Σ_i A[j,i]^2 v_i (1-π_i)/π_i (documents with π=0 and A≠0 give inf); returns (max_j V_j/s_j^2, V)."""
    pi = np.asarray(pi, float); A2 = np.asarray(A, float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pi > 0, (1 - pi) / pi, np.inf)
    V = np.array([(A2[j] * v * np.where((A2[j] > 0) | (pi > 0), ratio, 0.0)).sum() for j in range(A2.shape[0])])
    return float(np.max(V / np.asarray(s, float) ** 2)), V


def oracle_design(A, v, s, B, c=None, iters=300, eta=None):
    """Exact minimax design. A: (J, n) weights a_{j,i}; v: (n,) residual variances; s: (J,) slacks; B: budget."""
    A = np.asarray(A, float); v = np.asarray(v, float); s = np.asarray(s, float); J, n = A.shape
    c = np.ones(n) if c is None else np.asarray(c, float)
    A2s = A ** 2 * v[None, :] / (s[:, None] ** 2)          # (J, n): per-doc contribution to V_j/s_j^2 at unit (1-π)/π
    theta = np.full(J, 1.0 / J); best = (np.inf, None)
    scale = None
    for t in range(iters):
        g = theta @ A2s
        pi = waterfill(g, c, B)
        val, V = evaluate(pi, A, v, s); obj = V / s ** 2
        if val < best[0]:
            best = (val, pi.copy())
        if scale is None:
            scale = max(obj.max(), 1e-12)
        step = (eta or 0.5) / scale / np.sqrt(t + 1)
        theta = theta * np.exp(step * obj); theta /= theta.sum()
    return best[1], best[0]


def static_sum(W, B, c=None):
    """π ∝ Σ_j |w_j| (label-sharing static rule), capped at 1 and rescaled to the budget."""
    W = np.asarray(W, float); base = np.abs(W).sum(axis=0); n = len(base)
    c = np.ones(n) if c is None else np.asarray(c, float)
    return waterfill(base ** 2 / c, c, B) if base.sum() > 0 else np.zeros(n)   # sqrt(base^2)=base -> π ∝ base, budget-exact


def _selftest():
    rng = np.random.default_rng(0)
    # (1) single comparison: optimum is π ∝ |a| sqrt(v)
    n = 40; a = rng.normal(0, 1, n); v = rng.uniform(0.05, 0.25, n); B = 8.0
    pi, val = oracle_design(a[None, :], v, np.array([0.01]), B)
    ref = waterfill(a ** 2 * v, np.ones(n), B)
    assert np.allclose(pi, ref, atol=1e-6), "J=1 optimum must reduce to π ∝ |a|√v"
    print("[PASS] J=1 reduces to π ∝ |a|√v")
    # (2) constructed menu (EXPLORATION §2): binding pair B (slack small), dominated pairs C, D (slack large),
    #     decision documents of B (rank 8-12) disjoint from those of C, D (rank 4-8), 30 docs, budget 6 labels
    n = 30; W = np.zeros((3, n)); W[0, 8:12] = 1 / 12; W[1, 4:8] = -1 / 8; W[2, 4:8] = -1 / 8; W[:, :4] = 0.01
    v = np.full(n, 0.25); s = np.array([0.005, 0.11, 0.11]); B = 6.0
    pi_o, val_o = oracle_design(W, v, s, B); val_s, _ = evaluate(static_sum(W, B), W, v, s)
    print(f"[INFO] constructed menu: max_j V_j/s_j^2 static-sum {val_s:.1f} vs exact oracle {val_o:.1f} (ratio {val_s/val_o:.2f}x)")
    assert val_o < val_s / 1.5
    print("[PASS] exact oracle beats static sum when the binding pair's documents are disjoint from the dominated pairs'")
    # (3) same menu but the dominated pairs share B's documents -> little to gain
    W2 = W.copy(); W2[1] = 0; W2[2] = 0; W2[1, 8:12] = -1 / 12; W2[2, 8:12] = -1 / 12
    pi2, val2 = oracle_design(W2, v, s, B); val_s2, _ = evaluate(static_sum(W2, B), W2, v, s)
    print(f"[INFO] overlapping menu: static-sum {val_s2:.1f} vs oracle {val2:.1f} (ratio {val_s2/val2:.2f}x)")
    assert val_s2 / val2 < 1.15
    print("[PASS] no gain when decision documents overlap (as the design doc predicts)")


if __name__ == "__main__":
    _selftest()
