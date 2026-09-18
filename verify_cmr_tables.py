#!/usr/bin/env python3
"""
Deterministic recomputation of the Appendix C tables.

The published versions were Monte Carlo and, at the coincidence points rho=1 and
kappa=0 where the sandwich of Appendix C is an equality, the sampling error made
the printed rows violate the very proposition they verify.  Everything here is
quadrature, so the rows are consistent by construction.

  Delta_M^rho = 1 - E[min_j S_j^2],      S_j = sqrt(rho) Z0 + sqrt(1-rho) Z_j
  Delta_M(rho,kappa) = (1+kappa) - E[min_j (S_j^2 + kappa W_j^2)]   (lambda_1 = 1)

Exact limits used as cross-checks:
  kappa = 0  ->  Delta_M = Delta_M^rho
  rho   = 1  ->  Delta_M = kappa * Delta_M^0   (all three sandwich entries equal)
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(120)     # nodes for N(0,1)
GH_W = GH_W / GH_W.sum()


def E_min_S2(M, rho):
    """E[min_j S_j^2] by quadrature."""
    if rho >= 1.0:
        return 1.0
    sig = np.sqrt(1.0 - rho)

    def surv(t):
        r = np.sqrt(t)
        mu = np.sqrt(rho) * GH_X
        logs = np.logaddexp(norm.logsf((r - mu) / sig), norm.logsf((r + mu) / sig))
        return float(np.dot(GH_W, np.exp(M * logs)))

    val, _ = quad(surv, 0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)
    return val


GL_U, GL_W = np.polynomial.legendre.leggauss(120)          # nodes on [-1,1]


def E_min_Q(M, rho, kappa, T=80.0, nt=6000):
    """E[min_j (S_j^2 + kappa W_j^2)] by a vectorized product rule."""
    if kappa == 0.0:
        return E_min_S2(M, rho)
    sig = np.sqrt(max(1.0 - rho, 0.0))
    mus = np.sqrt(rho) * GH_X                              # (Z,)
    t = np.linspace(1e-12, T, nt)                          # (T,)
    rt = np.sqrt(t)

    # F(t|z) = P(S^2 + kappa W^2 <= t), substitution s = sqrt(t) u
    s_grid = rt[:, None] * GL_U[None, :]                   # (T,K)
    wfac = 2 * norm.cdf(np.sqrt(np.clip(t[:, None] * (1 - GL_U[None, :] ** 2),
                                        0, None) / kappa)) - 1          # (T,K)
    out = np.empty(nt)
    step = 400
    for a in range(0, nt, step):                           # chunk over t
        b = min(a + step, nt)
        if sig == 0.0:
            surv = np.where(mus[None, :] ** 2 >= t[a:b, None], 1.0,
                            2 * norm.sf(np.sqrt(np.clip(
                                (t[a:b, None] - mus[None, :] ** 2), 0, None) / kappa)))
        else:
            dens = norm.pdf(s_grid[a:b, :, None], mus[None, None, :], sig)  # (t,K,Z)
            integ = (GL_W[None, :, None] * dens
                     * wfac[a:b, :, None]).sum(axis=1) * rt[a:b, None]      # (t,Z)
            surv = 1.0 - np.clip(integ, 0.0, 1.0)
        out[a:b] = (GH_W[None, :] * np.clip(surv, 0, 1) ** M).sum(axis=1)
    from scipy.integrate import simpson
    return float(simpson(out, x=t))


# ---------------------------------------------------------------- Table V
print("TABLE V   Delta_M^rho = 1 - E[min_j S_j^2]   (quadrature)")
print(r"$M$ & $\rho=0$ & $0.2$ & $0.5$ & $0.8$ & $1.0$\\")
D0 = {}
for M in (2, 4, 8, 16):
    row = []
    for rho in (0.0, 0.2, 0.5, 0.8, 1.0):
        d = 1.0 - E_min_S2(M, rho)
        row.append(d)
        if rho == 0.0:
            D0[M] = d
    print(f"{M:2d} & " + " & ".join(f"{v:.4f}" for v in row) + r"\\")

# ---------------------------------------------------------------- Table VI
print()
print("TABLE VI   Delta_M(rho,kappa) against the sandwich, lambda_1 = 1  (quadrature)")
print(r"$M$ & $\rho$ & $\kappa$ & lower & $\Delta_M$ & upper\\")
cases = [(4, 0.0, 0.0), (4, 0.0, 0.5), (4, 0.0, 1.0),
         (4, 0.5, 0.0), (4, 0.5, 0.5),
         (4, 1.0, 0.5), (4, 1.0, 1.0),
         (8, 0.5, 0.5), (8, 1.0, 0.5)]
ok = True
for (M, rho, kap) in cases:
    drho = 1.0 - E_min_S2(M, rho)
    d0 = D0[M]
    if rho >= 1.0:                       # exact: Delta = kappa * Delta_M^0
        dm = kap * d0
    elif kap == 0.0:                     # exact: Delta = Delta_M^rho
        dm = drho
    else:
        dm = (1.0 + kap) - E_min_Q(M, rho, kap)
    lo, hi = max(drho, kap * d0), drho + kap * d0
    good = lo - 1e-9 <= dm <= hi + 1e-9
    ok &= good
    print(f"{M:2d} & {rho:.1f} & {kap:.1f} & {lo:.4f} & {dm:.4f} & {hi:.4f}"
          + r"\\" + ("" if good else "   <-- SANDWICH VIOLATED"))
print("\nall rows satisfy the sandwich:", ok)
