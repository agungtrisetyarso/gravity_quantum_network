#!/usr/bin/env python3
"""Figure 2 of the PRA version: the protection budget and the bound bracket."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/mnt/user-data/outputs")
from verify_pra import G2_exact, G2_cancel, equicorr, kernel_matrix   # noqa

BLUE, ORANGE, PURPLE = "#1B5EA8", "#C2571A", "#6A3D9A"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

# ---------------------------------------------------------------- panel (a)
M = 8
rho = np.linspace(0.02, 0.97, 400)
D = M * rho + 1 - rho
G2 = M / D                      # diversity alone
CAN = D / (1 - rho)             # rank-1 cancellation alone
TOT = M / (1 - rho)             # product, invariant

ax1.semilogy(rho, G2, color=BLUE, label=r"diversity  $G^{2}=M/D$")
ax1.semilogy(rho, CAN, color=ORANGE, ls="--", label=r"cancellation  $\lambda_1/\lambda_2$")
ax1.semilogy(rho, TOT, color=PURPLE, ls=":", lw=1.8, label=r"budget  $G^{2}_{\rm tot}=M/(1-\rho)$")

for r in (0.2, 0.5, 0.8):                       # exact numerics
    C = equicorr(M, r)
    ax1.plot(r, G2_exact(C), "o", ms=4, mfc="none", mec=BLUE, mew=1.0)
    ax1.plot(r, G2_cancel(C, 1), "s", ms=4, mfc="none", mec=PURPLE, mew=1.0)

ax1.annotate("diversity", xy=(0.80, M / (M * 0.8 + 0.2)), xytext=(0.60, 1.35),
             color=BLUE, fontsize=7)
ax1.annotate("cancellation", xy=(0.80, (M * .8 + .2) / .2), xytext=(0.30, 26),
             color=ORANGE, fontsize=7)
ax1.annotate("total budget", xy=(0.5, 16), xytext=(0.09, 33), color=PURPLE, fontsize=7)
ax1.set_xlabel(r"common-mode correlation $\rho$")
ax1.set_ylabel(r"threshold gain in $\delta_c^{2}$")
ax1.set_xlim(0, 1); ax1.set_ylim(0.9, 400)
ax1.legend(frameon=False, loc="upper left", handlelength=1.8)
ax1.set_title(r"(a) $M=8$: the split moves, the product does not", fontsize=8, loc="left")
ax1.grid(alpha=0.18, lw=0.4)

# ---------------------------------------------------------------- panel (b)
Mb = 5
ratio = np.linspace(0.4, 8.0, 22)          # footprint / correlation length
ex, up, lo = [], [], []
for R in ratio:
    pts = np.c_[np.linspace(0, R, Mb), np.zeros(Mb)]
    C = kernel_matrix(pts, 1.0, "gauss")
    A = np.linalg.inv(C)
    ex.append(G2_exact(C))
    up.append(np.ones(Mb) @ A @ np.ones(Mb))
    lo.append(Mb / np.linalg.eigvalsh(C).max())

ax2.plot(ratio, up, color=ORANGE, ls="--", label=r"$M_{\rm eff}=\mathbf{1}^{T}C^{-1}\mathbf{1}$ (upper)")
ax2.plot(ratio, ex, color=BLUE, label=r"exact $G^{2}$ (global QP)")
ax2.plot(ratio, lo, color=PURPLE, ls=":", lw=1.8, label=r"$M/\lambda_{\max}(C)$ (lower)")
ax2.fill_between(ratio, lo, up, color=BLUE, alpha=0.06, lw=0)
ax2.set_xlabel(r"footprint / correlation length $\Lambda/\ell$")
ax2.set_ylabel(r"$G^{2}$ and its bounds")
ax2.set_xlim(0.4, 8.0); ax2.set_ylim(0.9, 5.4)
ax2.legend(frameon=False, loc="upper left", handlelength=1.8)
ax2.set_title(r"(b) $M=5$ collinear paths, Gaussian kernel", fontsize=8, loc="left")
ax2.grid(alpha=0.18, lw=0.4)

for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(width=0.6, length=3)

fig.tight_layout(pad=0.6)
fig.savefig("/mnt/user-data/outputs/fig2.pdf")
fig.savefig("/mnt/user-data/outputs/fig2.png", dpi=220)
print("max Meff overestimate in panel b: %.1f%%" %
      (100 * max((u - e) / e for u, e in zip(up, ex))))
print("written")
