"""Figures for the QIP manuscript (fig1.pdf, fig2.pdf, fig3.pdf).  Fixed seeds."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import integrate, stats
from qp import G2, G2_cancel, kernel_matrix, m_M
from mgf import MinMGF
import warnings
warnings.filterwarnings("ignore")

BLUE, ORANGE, GREEN, PURPLE, GREY = "#1f5fa8", "#c8541e", "#0f8a6c", "#6b3fa0", "#7a7a7a"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "lines.linewidth": 1.6, "legend.frameon": False,
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})

# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0))
    Ms = np.geomspace(1, 200, 300)
    for rho, c in zip((0.05, 0.1, 0.3, 0.6), (BLUE, ORANGE, GREEN, PURPLE)):
        G = np.sqrt(Ms / (Ms * rho + 1 - rho))
        ax[0].plot(Ms, G, color=c, label=fr"$\rho={rho}$")
        ax[0].axhline(rho ** -0.5, color=c, ls=":", lw=1)
        Ms_ = 9.256 * (1 - rho) / rho
        if Ms_ <= 200:
            ax[0].plot(Ms_, np.sqrt(Ms_ / (Ms_ * rho + 1 - rho)), "*", color=c, ms=7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("candidate paths $M$"); ax[0].set_ylabel(r"threshold gain $G(M,\rho)$")
    ax[0].set_title(r"(a) saturation at $G_\infty=\rho^{-1/2}$", fontsize=9); ax[0].legend(fontsize=6.5, loc="upper left")
    ax[0].set_ylim(1, 5.2)
    # (b)
    L = np.linspace(0.5, 10, 200)
    ax[1].plot(L, np.exp(L / 2), "--", color=ORANGE, label=r"equicorrelated $e^{L/2\ell}$")
    ax[1].plot(L, np.sqrt(1 + L / 2), color=BLUE, label=r"line, $\sqrt{1+\Lambda/\ell_c}$")
    ax[1].plot(L, np.sqrt(0.156 * L ** 2 + 1.00 * L + 0.74), color=PURPLE, label="planar fit")
    mk = [1, 2, 4, 6, 8, 10]
    line_pts = []
    for Lam in mk:
        x = np.linspace(0, Lam, 512); r = np.exp(-(x[1] - x[0]))
        line_pts.append(np.sqrt((2 + 510 * (1 - r)) / (1 + r)))
    ax[1].plot(mk, line_pts, "o", mfc="white", color=BLUE, ms=5)
    plan = []
    for Lam in mk:
        n = int(np.ceil(Lam / 0.5)) + 1
        g = np.linspace(0, Lam, n); X, Y = np.meshgrid(g, g)
        C = kernel_matrix(np.c_[X.ravel(), Y.ravel()], 1.0, "exp")
        plan.append(np.sqrt(np.ones(n * n) @ np.linalg.solve(C, np.ones(n * n))))
    ax[1].plot(mk, plan, "s", mfc="white", color=PURPLE, ms=5)
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"footprint $\Lambda/\ell$"); ax[1].set_ylabel(r"ceiling $G_\infty$")
    ax[1].set_title("(b) the ceiling counts cells", fontsize=9); ax[1].legend(fontsize=6.5, loc="upper left")
    # (c)
    M, rho = 4, 0.3
    thc = M / (2 * (M * rho + 1 - rho))
    x = np.linspace(0, 0.995, 60)
    mgf = MinMGF(M, rho)
    sel = np.array([mgf(xx * thc) for xx in x])
    rng = np.random.default_rng(7)
    zh, wh = np.polynomial.hermite_e.hermegauss(60); wh = wh / wh.sum()
    Zp = rng.normal(size=(40000, M))
    xc = np.linspace(0, 0.93, 32)
    coh, mix = [], []
    for xx in xc:
        th = xx * thc; ec = em = 0.0
        for z, w in zip(zh, wh):
            S2 = (np.sqrt(rho) * z + np.sqrt(1 - rho) * Zp) ** 2
            eta = np.exp(-th * S2)
            ec += w * np.mean(1 / np.minimum(1, eta.sum(1)))
            em += w * np.mean(1 / eta.mean(1))
        coh.append(ec); mix.append(em)
    ax[2].fill_between(x, sel / M, sel * M, color=BLUE, alpha=0.1, lw=0, label=r"band, $c_-{=}1/M$, $c_+{=}1$")
    ax[2].plot(x, sel, color=BLUE, label="adaptive selection")
    ax[2].plot(xc, coh, "--", color=PURPLE, label="coherent superposition")
    ax[2].plot(xc, mix, "-.", color=GREEN, label="uniform mixture")
    xs = np.linspace(0, 0.4745, 200)
    ax[2].plot(xs, (1 - 2 * xs * thc) ** -0.5, color=GREY, label="static single path")
    ax[2].axvline(0.475, color=GREY, ls=":", lw=1)
    ax[2].set_yscale("log"); ax[2].set_ylim(0.2, 3000); ax[2].set_xlim(0, 1)
    ax[2].set_xlabel(r"$\theta/\theta_c$"); ax[2].set_ylabel(r"overhead penalty $\mathcal{R}/\mathcal{R}_{\rm ideal}$")
    ax[2].set_title("(c) one pole for the whole class", fontsize=9); ax[2].legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(); fig.savefig("../fig1.pdf")
    i93 = np.argmin(abs(x - 0.93))
    return dict(sel_at_093=float(sel[i93]), x_at=float(x[i93]), coh_ratio=[float(c / s) for c, s in zip(coh, np.interp(xc, x, sel))],
                line_pts=line_pts, plan=[float(p) ** 2 for p in plan])

# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    M = 8; r = np.linspace(0.005, 0.97, 300); D = M * r + 1 - r
    ax[0].plot(r, M / D, color=BLUE, label=r"diversity $G^2=M/D$")
    ax[0].plot(r, D / (1 - r), "--", color=ORANGE, label=r"cancellation $\lambda_1/\lambda_2$")
    ax[0].plot(r, M / (1 - r), ":", color=PURPLE, lw=2, label=r"budget $G^2_{\rm tot}=M/(1-\rho)$")
    chk = []
    for rho in (0.2, 0.5, 0.8):
        C = (1 - rho) * np.eye(M) + rho * np.ones((M, M))
        g = G2(C); gt = G2_cancel(C, np.ones(M) / np.sqrt(M)); chk.append((rho, g, gt))
        ax[0].plot(rho, g, "o", mfc="white", color=BLUE, ms=5); ax[0].plot(rho, gt, "s", mfc="white", color=PURPLE, ms=5)
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"common-mode correlation $\rho$"); ax[0].set_ylabel(r"squared threshold gain")
    ax[0].set_title(r"(a) $M=8$: the split moves, the product does not", fontsize=9); ax[0].legend(fontsize=7.5)
    Ls = np.linspace(0.4, 8, 40); ex, me, lo = [], [], []
    for Lam in Ls:
        C = kernel_matrix(np.c_[np.linspace(0, Lam, 5), np.zeros(5)], 1.0, "gauss")
        ex.append(G2(C)); me.append(np.ones(5) @ np.linalg.solve(C, np.ones(5))); lo.append(5 / np.linalg.eigvalsh(C)[-1])
    ax[1].plot(Ls, me, "--", color=ORANGE, label=r"$M_{\rm eff}=\mathbf{1}^{\top}C^{-1}\mathbf{1}$ (upper)")
    ax[1].plot(Ls, ex, color=BLUE, label=r"exact $G^2$ (global)")
    ax[1].plot(Ls, lo, ":", color=PURPLE, lw=2, label=r"$M/\lambda_{\max}(C)$ (lower)")
    ax[1].fill_between(Ls, lo, me, color=BLUE, alpha=0.07, lw=0)
    ax[1].set_xlabel(r"footprint / correlation length $\Lambda/\ell$"); ax[1].set_ylabel(r"$G^2$ and its bounds")
    ax[1].set_title(r"(b) $M=5$ collinear paths, Gaussian kernel", fontsize=9); ax[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig("../fig2.pdf")
    return dict(chk=chk, max_excess=float(max(np.array(me) / np.array(ex) - 1)))

# ------------------------------------------------------------------ Fig. 3 (new)
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
    M = 6; Ls = np.linspace(1.0, 8, 22)
    pts0 = np.random.default_rng(5).uniform(0, 1, size=(M, 2))
    rec = {k: [] for k in ("none", "cm", "top", "cm_lo", "top_lo", "mid", "none_lo")}
    for Lam in Ls:
        C = kernel_matrix(Lam * pts0, 1.0, "gauss")
        w, V = np.linalg.eigh(C); one = np.ones(M) / np.sqrt(M); P = np.eye(M) - np.outer(one, one)
        rec["none"].append(G2(C)); rec["cm"].append(G2_cancel(C, one)); rec["top"].append(G2_cancel(C, V[:, -1]))
        rec["cm_lo"].append(M / np.linalg.eigvalsh(P @ C @ P)[-1]); rec["top_lo"].append(M / w[-2])
        rec["none_lo"].append(M / w[-1])
        rec["mid"].append(M / ((M * M - np.ones(M) @ C @ np.ones(M)) / (M * (M - 1))))
    ax[0].plot(Ls, rec["top"], color=ORANGE, label="top-mode canceller (exact)")
    ax[0].plot(Ls, rec["top_lo"], ":", color=ORANGE, lw=1.4, label=r"$M/\lambda_2(C)$")
    ax[0].plot(Ls, rec["cm"], color=GREEN, label="common-mode canceller (exact)")
    ax[0].plot(Ls, rec["cm_lo"], ":", color=GREEN, lw=1.4, label=r"$M/\lambda_{\max}(PCP)$")
    ax[0].plot(Ls, rec["mid"], "-.", color=GREY, lw=1.2, label=r"$M/\bar\gamma$ (structure function)")
    ax[0].plot(Ls, rec["none"], color=BLUE, label="no cancellation (exact)")
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"footprint $\Lambda/\ell$"); ax[0].set_ylabel(r"squared threshold gain")
    ax[0].set_title(r"(a) $M=6$ irregular planar set, Gaussian kernel", fontsize=9); ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    # (b) estimation-limited canceller, planar Gaussian-kernel configuration
    rng = np.random.default_rng(11)
    M = 6; pts = rng.uniform(0, 3.0, size=(M, 2)); C = kernel_matrix(pts, 1.0, "gauss")
    w, V = np.linalg.eigh(C); l1, l2, u1 = w[-1], w[-2], V[:, -1]
    g_ideal = G2_cancel(C, u1); g_none = G2(C)
    scales = np.geomspace(3e-4, 0.12, 12)
    E_, gh, pop, cert, cert0 = [], [], [], [], []
    for s in scales:
        es, gs, ps, cs, c0 = [], [], [], [], []
        for _ in range(8):
            while True:
                E = rng.normal(scale=s, size=(M, M)); E = (E + E.T) / 2; np.fill_diagonal(E, 0)
                if np.linalg.eigvalsh(C + E)[0] > 1e-6: break
            e = np.linalg.norm(E, 2); wh, Vh = np.linalg.eigh(C + E); uh = Vh[:, -1]
            eps = 1 - (uh @ u1) ** 2
            es.append(e); gs.append(G2_cancel(C, uh)); ps.append(M / (l2 + eps * (l1 - l2)))
            cs.append(M / (wh[-2] + e)); c0.append(M / (wh[-1] + e))
        E_.append(np.median(es)); gh.append(np.median(gs)); pop.append(np.median(ps)); cert.append(np.median(cs)); cert0.append(np.median(c0))
    ax[1].axhline(g_ideal, color=ORANGE, lw=1, ls="--", label=r"ideal canceller $G^2_{\rm tot}(1)$")
    ax[1].plot(E_, gh, "o-", color=ORANGE, ms=4, label=r"estimated mode $\hat u_1$ (exact)")
    ax[1].plot(E_, pop, ":", color=PURPLE, lw=2, label=r"$M/[\lambda_2+\epsilon(\lambda_1-\lambda_2)]$")
    ax[1].plot(E_, cert, "s-", color=GREEN, ms=4, mfc="white", label=r"certificate $M/(\hat\lambda_2+\|E\|)$")
    ax[1].plot(E_, cert0, "^-", color=BLUE, ms=4, mfc="white", label=r"no-cancel certificate $M/(\hat\lambda_1+\|E\|)$")
    ax[1].axhline(g_none, color=BLUE, lw=1, ls="--", label=r"no cancellation $G^2$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"estimation error $\|\hat C-C\|$"); ax[1].set_ylabel(r"squared threshold gain")
    ax[1].set_title(r"(b) estimation-limited canceller, $M=6$ planar", fontsize=9); ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for a in ax:
        a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_yticks([2, 3, 5, 10, 20, 30]); ax[1].set_yticks([2, 3, 4, 6, 8, 10])
    fig.tight_layout(); fig.savefig("../fig3.pdf")
    return dict(Ls=Ls.tolist(), rec=rec, pts0=pts0.tolist(), l1=l1, l2=l2, gap=l1 - l2, g_ideal=g_ideal, g_none=g_none,
                E=E_, gh=gh, pop=pop, cert=cert, cert0=cert0, pts=pts.tolist())

if __name__ == "__main__":
    import json, sys
    which = sys.argv[1:] or ["1", "2", "3"]
    res = {}
    if "1" in which: res["fig1"] = fig1()
    if "2" in which: res["fig2"] = fig2()
    if "3" in which: res["fig3"] = fig3()
    json.dump(res, open("figures_results_%s.json" % "".join(which), "w"), indent=1, default=float)
    print(json.dumps(res, default=float)[:3000])
