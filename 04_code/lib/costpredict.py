"""Pilot-only predictors of the post-pilot labels a fixed-budget certificate needs (Track C, 2026-09-17).

v1 (the formula fixed on the development collections and used in the first held-out lock):
    L_post = z^2 * b_ref * v_pilot / s^2
  assumes every remaining query is audited (f = 1) and ignores that the pilot's part of the population mean is known.

v2 (after the held-out run showed v1 over-predicting by 3-4x on 49-50-query pools and under-predicting by ~2x on the
development pools): the certificate uses the population estimand mu_N = (sum_pilot D + N_R * mu_rest) / N, so the bound on
mu_rest only needs precision s * N / N_R; and the two-stage variance of the mean over the rest is
    V(L) = (1 - f) sb2 / n + v(b) / n,   n = min(N_R, L / b_min),  f = n / N_R,  b = max(b_min, L / N_R),  v(b) = v_pilot * b_ref / b,
(the sample variance of the per-query estimates that the bound uses is about sb2 + v(b), which gives this form)
with sb2 the between-query variance of the paired difference (estimated on the pilot queries) and v_pilot the within-query
document-sampling variance of the design at b_ref documents per query (also from the pilot). L_post is the smallest L with
z * sqrt(V(L)) <= s * N / N_R. Everything is a pilot quantity except N (known) and the design constants.
"""
import math
import numpy as np


def l_post_v1(v_pilot, slack, z, b_ref=4):
    s = max(float(slack), 1e-4)
    return z ** 2 * b_ref * float(v_pilot) / s ** 2


def l_post_v2(v_pilot, sb2, slack, z, N, n_pilot, b_ref=4, b_min=4, l_max=2_000_000):
    """Smallest post-pilot label count L whose two-stage bound width z*sqrt(V(L)) is below the slack the rest must meet."""
    N_R = max(int(N) - int(n_pilot), 1); s = max(float(slack), 1e-4) * N / N_R
    v_pilot = float(v_pilot); sb2 = 0.0 if sb2 is None or not np.isfinite(sb2) else max(float(sb2), 0.0)

    def width(L):
        n = min(N_R, max(L / b_min, 1.0)); f = n / N_R; b = max(b_min, L / N_R)
        V = (1 - f) * sb2 / n + (v_pilot * b_ref / b) / n
        return z * math.sqrt(max(V, 0.0))

    if width(l_max) > s:
        return float("inf")
    lo, hi = 1.0, float(l_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if width(mid) <= s:
            hi = mid
        else:
            lo = mid
    return hi


def l_post_v3(row, z, N, n_pilot, b_ref=4, b_min=4, ref="ref"):
    """v3: the certificate must pass every comparison, so L_post = max over comparisons j of the v2 label count computed
    from that comparison's own (v_j, sb2_j, slack_j). `row` is a per-draw record with columns v_pilot_{ref}_c{k}, sb2_c{k},
    slack_c{k} (k = 0, 1, ...); comparisons whose columns are missing are skipped."""
    best = 0.0; k = 0
    while f"slack_c{k}" in row and f"sb2_c{k}" in row:
        vcol = f"v_pilot_{ref}_c{k}" if ref else f"v_pilot_c{k}"
        if vcol not in row:
            break
        v, sb, sl = row[vcol], row[f"sb2_c{k}"], row[f"slack_c{k}"]
        if np.isfinite(v) and np.isfinite(sl):
            best = max(best, l_post_v2(v, sb, sl, z, N, n_pilot, b_ref, b_min))
        k += 1
    return best if k else float("nan")
