#!/usr/bin/env python3
"""
Larger-M tests of the bracket  M/lambda_max(C) <= G^2 <= 1'C^-1 1.

Two regimes, with different logical standing, kept separate on purpose:

  EXHAUSTIVE  (M <= 12):  every one of the 2^(M-1) sign orthants is solved,
                          so the reported G^2 is the GLOBAL minimum.
  SAMPLED     (M > 12):   a random subset of orthants is solved.  Each solve
                          returns a FEASIBLE point, so the result is a
                          certified UPPER bound on G^2.  That is one-sided but
                          it is the side that matters: an upper bound below
                          M_eff proves M_eff overestimates the true threshold.

Each orthant subproblem is strictly convex, so one start per orthant suffices.
"""
import itertools, time, numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

rng = np.random.default_rng(31415)


def _orthant_min(A, signs):
    D = np.diag(signs)
    As = D @ A @ D
    M = A.shape[0]
    r = minimize(lambda z: z @ As @ z, np.ones(M), jac=lambda z: 2 * As @ z,
                 method="L-BFGS-B", bounds=[(1.0, None)] * M,
                 options=dict(maxiter=4000, ftol=1e-16))
    return r.fun


def G2_global(C):
    """Exhaustive over all 2^(M-1) orthants -> global minimum."""
    A = np.linalg.inv(C)
    M = C.shape[0]
    return min(_orthant_min(A, np.array((1,) + s, float))
               for s in itertools.product([1, -1], repeat=M - 1))


def G2_upper(C, n_orthants=4000):
    """Random orthants -> certified upper bound on G^2 (feasible points only)."""
    A = np.linalg.inv(C)
    M = C.shape[0]
    best = _orthant_min(A, np.ones(M))            # all-plus orthant always tried
    for _ in range(n_orthants):
        s = np.ones(M)
        s[1:] = rng.choice([-1.0, 1.0], M - 1)
        best = min(best, _orthant_min(A, s))
    return best


def kernel(pts, ell, kind):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    C = np.exp(-d / ell) if kind == "exp" else np.exp(-0.5 * (d / ell) ** 2)
    return C + 1e-11 * np.eye(len(pts))


def meff(C):
    M = C.shape[0]
    return float(np.ones(M) @ cho_solve(cho_factor(C), np.ones(M)))


def screening_ok(C):
    A = np.linalg.inv(C)
    off = A - np.diag(np.diag(A))
    return bool(off.max() <= 1e-9 and A.sum(1).min() >= -1e-9)


def draw(M, kind, spread):
    pts = rng.random((M, 2)) * spread
    return kernel(pts, 1.0, kind)


# =====================================================================
print("=" * 78)
print("EXHAUSTIVE (global minima), 2^(M-1) orthants")
print("=" * 78)
print(f"{'kernel':6s} {'M':>3s} {'n':>4s} {'Meff/G2-1 med':>14s} {'max':>9s} "
      f"{'G2/(M/lmax)-1 med':>18s} {'screen%':>8s}")
rows = {}
for kind in ("exp", "gauss"):
    for M in (6, 8, 10, 12):
        n = {6: 40, 8: 30, 10: 16, 12: 8}[M]
        up, dn, scr = [], [], []
        t0 = time.time()
        for _ in range(n):
            C = draw(M, kind, rng.uniform(1.0, 6.0))
            if np.linalg.cond(C) > 1e11:
                continue
            g2 = G2_global(C)
            up.append(meff(C) / g2 - 1)
            dn.append(g2 / (M / np.linalg.eigvalsh(C).max()) - 1)
            scr.append(screening_ok(C))
        up, dn = np.array(up), np.array(dn)
        rows[(kind, M)] = (np.median(up), up.max(), np.median(dn))
        print(f"{kind:6s} {M:3d} {len(up):4d} {np.median(up):13.4%} {up.max():8.3%} "
              f"{np.median(dn):17.3%} {100*np.mean(scr):7.0f}%   "
              f"[{time.time()-t0:.0f}s]")

print()
print("=" * 78)
print("SAMPLED (certified upper bounds on G^2), 4000 random orthants")
print("=" * 78)
print(f"{'kernel':6s} {'M':>3s} {'n':>3s} {'Meff/G2ub-1 med':>16s} {'max':>9s}"
      f"   (a positive value PROVES Meff overestimates)")
for kind in ("exp", "gauss"):
    for M in (16, 24, 32):
        n = 6 if M < 32 else 4
        up = []
        t0 = time.time()
        for _ in range(n):
            C = draw(M, kind, rng.uniform(2.0, 8.0))
            if np.linalg.cond(C) > 1e11:
                continue
            ub = G2_upper(C, n_orthants=4000)
            up.append(meff(C) / ub - 1)
        up = np.array(up)
        print(f"{kind:6s} {M:3d} {len(up):3d} {np.median(up):15.3%} {up.max():8.3%}"
              f"      [{time.time()-t0:.0f}s]")

print()
print("=" * 78)
print("BALANCED SETS AT LARGE M  (Theorem 5: exact, no enumeration needed)")
print("=" * 78)
for M in (8, 16, 32, 64, 128):
    th = 2 * np.pi * np.arange(M) / M
    pts = np.c_[np.cos(th), np.sin(th)] * 3.0
    for kind in ("exp", "gauss"):
        C = kernel(pts, 1.0, kind)
        lo = M / np.linalg.eigvalsh(C).max()
        hi = meff(C)
        print(f"  M={M:4d} {kind:5s}  M/lmax={lo:10.5f}  Meff={hi:10.5f}  "
              f"rel.gap={abs(hi-lo)/hi:.2e}")
