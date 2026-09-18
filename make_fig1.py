#!/usr/bin/env python3
"""
Figure 1 of the PRA manuscript -- caption-faithful, three panels.

(a) threshold gain G(M,rho) vs candidate path count, ceilings and M_star
(b) ceiling vs footprint: 1D chain, planar fit, and the unrealizable
    equicorrelated form, with computed markers
(c) overhead penalty vs theta/theta_c at M=4, rho=0.3, kappa=0: selection by
    quadrature, the Theorem-2 band, coherent superposition and uniform mixture
    by Gauss-Hermite-stratified Monte Carlo, and the static single path

Every curve is computed here; nothing is sketched.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve

BLUE, ORANGE, PURPLE, GREEN = "#1B5EA8", "#C2571A", "#6A3D9A", "#0B7D5E"
GREY = "#8A8A8A"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.3, "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

rng = np.random.default_rng(20260918)

# =============================================================== panel (a)
def G(M, rho):
    return np.sqrt(M / (M * rho + 1.0 - rho))

# =============================================================== panel (b)
def meff_line(Lratio, M=512, ell=1.0):
    """1^T C^-1 1 for M equispaced points on a line, exponential kernel."""
    x = np.linspace(0.0, Lratio * ell, M)
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / ell) + 1e-12 * np.eye(M)
    return float(np.ones(M) @ cho_solve(cho_factor(C), np.ones(M)))

def meff_square(Lratio, spacing=0.5, ell=1.0):
    """1^T C^-1 1 for a square grid footprint, exponential kernel."""
    n = int(round(Lratio / spacing)) + 1
    g = np.linspace(0.0, Lratio * ell, n)
    X, Y = np.meshgrid(g, g)
    p = np.c_[X.ravel(), Y.ravel()]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    C = np.exp(-d / ell) + 1e-10 * np.eye(len(p))
    return float(np.ones(len(p)) @ cho_solve(cho_factor(C), np.ones(len(p))))

# =============================================================== panel (c)
M_C, RHO_C = 4, 0.3
D_C = M_C * RHO_C + 1 - RHO_C
THETA_C = M_C / (2.0 * D_C)            # = I, the adaptive pole
THETA_STATIC = 0.5                     # pole of E[exp(theta S^2)]

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)   # nodes for N(0,1)
GH_W = GH_W / GH_W.sum()

def selection_overhead(theta):
    """E[exp(theta min_j S_j^2)] by log-domain quadrature over the survival fn."""
    sig = np.sqrt(1 - RHO_C)
    tot = 0.0
    for z, w in zip(GH_X, GH_W):
        mu = np.sqrt(RHO_C) * z

        def integrand(t):
            # log domain: exp(theta t) * S(t)^M with S the two-sided survival
            r = np.sqrt(t)
            logs = np.logaddexp(norm.logsf((r - mu) / sig),
                                norm.logsf((r + mu) / sig))
            return np.exp(theta * t + M_C * logs)

        val, _ = quad(integrand, 0, np.inf, limit=600, epsabs=1e-13, epsrel=1e-10)
        tot += w * (1.0 + theta * val)
    return tot

def policy_overheads(theta, n=400_000):
    """Coherent superposition and uniform mixture, stratified on the shared mode."""
    sig = np.sqrt(1 - RHO_C)
    coh = mix = 0.0
    for z, w in zip(GH_X, GH_W):
        S = np.sqrt(RHO_C) * z + sig * rng.standard_normal((n, M_C))
        eta = np.exp(-theta * S ** 2)
        ssum = eta.sum(axis=1)
        coh += w * np.mean(np.maximum(1.0, 1.0 / ssum))
        mix += w * np.mean(M_C / ssum)
    return coh, mix

# =============================================================== figure
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.5))

# ---- (a) -------------------------------------------------------------
Ms = np.arange(1, 201)
for rho, col in zip((0.05, 0.1, 0.3, 0.6), (BLUE, ORANGE, GREEN, PURPLE)):
    axA.semilogx(Ms, G(Ms, rho), color=col, label=rf"$\rho={rho}$")
    axA.axhline(rho ** -0.5, color=col, ls=":", lw=0.9)
    ms = 9.256 * (1 - rho) / rho
    axA.plot(ms, G(ms, rho), "*", ms=7, color=col, mec="white", mew=0.4)
axA.set_xlabel(r"candidate paths $M$")
axA.set_ylabel(r"threshold gain $G(M,\rho)$")
axA.set_xlim(1, 200)
axA.set_ylim(1, 5.2)
axA.legend(frameon=False, loc="upper left", handlelength=1.5, labelspacing=0.25)
axA.set_title(r"(a) saturation at $G_\infty=\rho^{-1/2}$", fontsize=8, loc="left")

# ---- (b) -------------------------------------------------------------
LR = np.linspace(0.5, 10.0, 200)
axB.semilogy(LR, np.exp(LR / 2), color=ORANGE, ls="--",
             label=r"equicorrelated $e^{L/2\ell}$ (unrealizable)")
axB.semilogy(LR, np.sqrt(1 + LR / 2), color=BLUE,
             label=r"line, $\sqrt{1+\Lambda/\ell_c}$")
axB.semilogy(LR, np.sqrt(0.156 * LR ** 2 + 1.00 * LR + 0.74), color=PURPLE,
             label=r"planar fit, Eq. (15)")
mk = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
axB.plot(mk, [np.sqrt(meff_line(L)) for L in mk], "o", ms=3.4,
         mfc="none", mec=BLUE, mew=0.9)
mk2 = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
axB.plot(mk2, [np.sqrt(meff_square(L)) for L in mk2], "s", ms=3.4,
         mfc="none", mec=PURPLE, mew=0.9)
axB.set_xlabel(r"footprint $\Lambda/\ell$")
axB.set_ylabel(r"ceiling $G_\infty$")
axB.set_xlim(0.5, 10)
axB.set_ylim(0.9, 200)
axB.legend(frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25)
axB.set_title(r"(b) the ceiling counts cells", fontsize=8, loc="left")

# ---- (c) -------------------------------------------------------------
tr = np.linspace(0.02, 0.985, 45)
sel = np.array([selection_overhead(t * THETA_C) for t in tr])
axC.fill_between(tr, sel / M_C, sel * M_C, color=BLUE, alpha=0.10, lw=0,
                 label="Thm. 2 band (Assumption 1)")
axC.semilogy(tr, sel, color=BLUE, label="adaptive selection (quadrature)")

tr_mc = np.linspace(0.05, 0.93, 16)
coh, mix = np.array([policy_overheads(t * THETA_C) for t in tr_mc]).T
axC.semilogy(tr_mc, coh, color=PURPLE, ls="--", label="coherent superposition")
axC.semilogy(tr_mc, mix, color=GREEN, ls="-.", label="uniform mixture")

ts = np.linspace(0.02, 0.4735, 200) * THETA_C
axC.semilogy(ts / THETA_C, (1 - 2 * ts) ** -0.5, color=GREY,
             label="static single path")
axC.axvline(THETA_STATIC / THETA_C, color=GREY, ls=":", lw=0.9)
axC.text(THETA_STATIC / THETA_C - 0.02, 1.35, "static pole", rotation=90,
         ha="right", va="bottom", fontsize=6, color=GREY)

axC.set_xlabel(r"$\theta/\theta_c$")
axC.set_ylabel(r"overhead penalty $\mathcal{R}/\mathcal{R}_{\rm ideal}$")
axC.set_xlim(0, 1)
axC.set_ylim(0.25, 60)
axC.legend(frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25)
axC.set_title(r"(c) one pole for the whole class", fontsize=8, loc="left")

for a in (axA, axB, axC):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.16, lw=0.4)
    a.tick_params(width=0.6, length=3)

fig.tight_layout(pad=0.5)
fig.savefig("/mnt/user-data/outputs/f1.pdf")
fig.savefig("/mnt/user-data/outputs/f1.png", dpi=240)

print(f"theta_c = {THETA_C:.4f}, static pole ratio = {THETA_STATIC/THETA_C:.4f}")
print(f"selection overhead at 0.93 theta_c = {sel[np.argmin(abs(tr-0.93))]:.3f}")
print("coherent/mixture inside band:",
      bool(np.all(coh >= np.interp(tr_mc, tr, sel) / M_C - 1e-9) and
           np.all(mix <= np.interp(tr_mc, tr, sel) * M_C + 1e-9)))
print("line markers  :", [round(np.sqrt(meff_line(L)), 3) for L in mk])
print("planar markers:", [round(np.sqrt(meff_square(L)), 3) for L in mk2])
print("written f1.pdf / f1.png")
