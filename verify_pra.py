#!/usr/bin/env python3
"""
Numerical verification of the new results added for the PRA version of
"Path diversity and mode cancellation share one protection budget ...".

Claims checked
  V1  G^2 = min{ y^T C^{-1} y : |y_j| >= 1 }   (global, sign-orthant enumeration)
  V2  sandwich      M/lambda_max(C)  <=  G^2  <=  1^T C^{-1} 1
  V3  equality throughout iff C has constant row sums (1 is an eigenvector)
  V4  rank-r cancellation bound   G_tot^2(r) >= M/lambda_{r+1}(C)
  V5  exchangeable complementarity  G^2 * (lambda_1/lambda_2) = G_tot^2(1) = M/(1-rho)
"""
import itertools, numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- global QP
def qp_orthant(A, signs, y_lo=1.0):
    """min z^T A_s z over z >= 1 (convex, A_s = D_s A D_s)."""
    D = np.diag(signs)
    As = D @ A @ D
    M = A.shape[0]
    f = lambda z: z @ As @ z
    g = lambda z: 2 * As @ z
    best = np.inf
    for z0 in (np.ones(M), np.ones(M) * 1.5, 1 + rng.random(M)):
        r = minimize(f, z0, jac=g, method="L-BFGS-B",
                     bounds=[(y_lo, None)] * M, options=dict(maxiter=2000, ftol=1e-15))
        best = min(best, r.fun)
    return best

def G2_exact(C):
    """Global minimum of y^T C^{-1} y over |y_j| >= 1, by orthant enumeration."""
    A = np.linalg.inv(C)
    M = C.shape[0]
    best = np.inf
    for s in itertools.product([1, -1], repeat=M - 1):   # y_1 > 0 wlog
        best = min(best, qp_orthant(A, np.array((1,) + s, float)))
    return best

def G2_cancel(C, r):
    """Global min of y^T (C^(r))^+ y over |y_j|>=1, y in range(C^(r));
       C^(r) = C with its top r eigenmodes removed."""
    w, U = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w, U = w[idx], U[:, idx]
    keep = slice(r, len(w))
    Up, wp = U[:, keep], w[keep]
    P = Up @ Up.T                      # projector onto the surviving subspace
    Apinv = Up @ np.diag(1 / wp) @ Up.T
    M = C.shape[0]
    best = np.inf
    for s in itertools.product([1, -1], repeat=M - 1):
        sv = np.array((1,) + s, float)
        D = np.diag(sv)
        As = D @ Apinv @ D
        Ps = D @ (np.eye(M) - P) @ D   # penalty enforcing y in range
        f = lambda z: z @ As @ z + 1e6 * (z @ Ps @ z)
        g = lambda z: 2 * As @ z + 2e6 * (Ps @ z)
        for z0 in (np.ones(M), 1 + rng.random(M)):
            res = minimize(f, z0, jac=g, method="L-BFGS-B",
                           bounds=[(1.0, None)] * M, options=dict(maxiter=4000, ftol=1e-15))
            if res.fun < best:
                best = res.fun
    return best

# ------------------------------------------------------------ test families
def equicorr(M, rho):
    return (1 - rho) * np.eye(M) + rho * np.ones((M, M))

def kernel_matrix(pts, ell, kind="exp"):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return np.exp(-d / ell) if kind == "exp" else np.exp(-0.5 * (d / ell) ** 2)

def circulant_ring(M, ell):
    """Balanced (constant row sum) set: M points equispaced on a circle."""
    th = 2 * np.pi * np.arange(M) / M
    pts = np.c_[np.cos(th), np.sin(th)]
    return kernel_matrix(pts, ell, "exp")

# =============================================================== V2, V3
print("=" * 74)
print("V2/V3  sandwich  M/lmax <= G^2 <= 1'C^-1 1   and the balanced-set equality")
print("=" * 74)
print(f"{'family':28s} {'M':>2s} {'M/lmax':>9s} {'G^2':>9s} {'Meff':>9s} {'bal?':>5s}")
cases = []
for M in (3, 4, 5, 6):
    cases.append((f"equicorrelated rho=0.3", equicorr(M, 0.3)))
    cases.append((f"ring (balanced) l=1.0", circulant_ring(M, 1.0)))
for M in (3, 4, 5):
    for t in range(3):
        pts = rng.random((M, 2)) * 3
        cases.append((f"random 2D exp kernel #{t}", kernel_matrix(pts, 1.0, "exp")))
        cases.append((f"random 2D gauss kernel #{t}", kernel_matrix(pts, 1.0, "gauss")))
viol = 0
for name, C in cases:
    M = C.shape[0]
    lo = M / np.linalg.eigvalsh(C).max()
    g2 = G2_exact(C)
    hi = np.ones(M) @ np.linalg.inv(C) @ np.ones(M)
    bal = np.allclose(C.sum(1), C.sum(1)[0], atol=1e-9)
    ok = lo <= g2 + 1e-7 and g2 <= hi + 1e-7
    viol += (not ok)
    eq = abs(lo - hi) < 1e-7
    assert eq == bal or not bal, (name, lo, hi)
    print(f"{name:28s} {M:2d} {lo:9.4f} {g2:9.4f} {hi:9.4f} {str(bal):>5s}"
          + ("" if ok else "   <-- SANDWICH VIOLATED"))
print(f"\nsandwich violations: {viol}")

# relative slack of Meff when screening may fail
slack = [(np.ones(C.shape[0]) @ np.linalg.inv(C) @ np.ones(C.shape[0]) - G2_exact(C))
         / G2_exact(C) for _, C in cases]
print(f"max relative slack of Meff over all cases: {max(slack):.3%}")

# =============================================================== V4, V5
print()
print("=" * 74)
print("V4/V5  rank-1 cancellation:  G^2 * (l1/l2)  vs  G_tot^2  vs  M/(1-rho)")
print("=" * 74)
print(f"{'M':>2s} {'rho':>5s} {'G^2':>8s} {'l1/l2':>8s} {'product':>9s} {'G_tot^2':>9s} {'M/(1-rho)':>10s} {'M/l2':>8s}")
for M in (4, 6, 8):
    for rho in (0.2, 0.5, 0.8):
        C = equicorr(M, rho)
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        g2 = G2_exact(C)
        gtot = G2_cancel(C, 1)
        print(f"{M:2d} {rho:5.2f} {g2:8.4f} {w[0]/w[1]:8.4f} {g2*w[0]/w[1]:9.4f} "
              f"{gtot:9.4f} {M/(1-rho):10.4f} {M/w[1]:8.4f}")

print()
print("rank-r lower bound  G_tot^2(r) >= M/lambda_{r+1}  on random balanced sets:")
for M in (4, 6):
    for r in (1, 2):
        C = circulant_ring(M, 1.0)
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        gt = G2_cancel(C, r)
        print(f"  M={M} r={r}:  G_tot^2={gt:8.4f}   M/lambda_{{r+1}}={M/w[r]:8.4f}   "
              f"{'OK' if gt >= M/w[r] - 1e-6 else 'VIOLATED'}")

# =============================================================== V6
print()
print("=" * 74)
print("V6  how far can Meff overestimate G^2 ?  (screening condition failing)")
print("=" * 74)
import collections
stat = collections.defaultdict(list)
for kind in ("exp", "gauss"):
    for M in (3, 4, 5, 6):
        for t in range(60):
            pts = rng.random((M, 2)) * rng.uniform(1.0, 6.0)
            C = kernel_matrix(pts, 1.0, kind)
            if np.linalg.cond(C) > 1e10:
                continue
            A = np.linalg.inv(C)
            off = A - np.diag(np.diag(A))
            screen = (off.max() <= 1e-9) and (A.sum(1).min() >= -1e-9)
            g2 = G2_exact(C)
            meff = np.ones(M) @ A @ np.ones(M)
            lo = M / np.linalg.eigvalsh(C).max()
            stat[(kind, screen)].append(((meff - g2) / g2, (g2 - lo) / g2))
for (kind, screen), v in sorted(stat.items()):
    up = np.array([a for a, _ in v]); dn = np.array([b for _, b in v])
    print(f"{kind:6s} screening={str(screen):5s} n={len(v):3d}  "
          f"Meff overestimate: median {np.median(up):7.3%}  max {up.max():7.3%}   |   "
          f"M/lmax underestimate: median {np.median(dn):7.3%}  max {dn.max():7.3%}")
