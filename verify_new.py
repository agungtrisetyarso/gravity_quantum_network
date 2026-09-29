"""Numerical verification of the results added for the QIP version (checks V1-V5).

V1  misaligned canceller: lambda_max(P C P) <= lambda_2 + eps (lambda_1 - lambda_2), tight
    when u_hat lies in span(u_1, u_2); exact G^2(u_hat) >= M / lambda_max(P C P).
V2  Davis-Kahan-type control: eps <= 4 ||E||^2 / (lambda_1 - lambda_2)^2.
V3  data-driven certificates: G^2 >= M/(lambda_1(C_hat)+||E||),
    G^2(u_hat) >= M/(lambda_2(C_hat)+||E||),  G^2_cm >= M/(lambda_max(P C_hat P)+||E||).
V4  field-agnostic common-mode canceller: M/lambda_max(PCP) <= G^2_cm <= m_M tr((PCP)^+)/(M-1),
    with M/gamma_bar inside; equality on the equicorrelated model.
V5  comparison of no cancellation / common-mode / top-mode cancellation by kernel.
"""
import sys, json
import numpy as np
from qp import G2, G2_cancel, kernel_matrix, m_M
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(20260929)
out = {"V1": [], "V2": [], "V3": [], "V4": [], "V5": []}
viol = {k: 0 for k in out}
tol = 1e-7

# ---- V1 tightness on span(u1,u2), no QP needed
tight_err = 0.0
for _ in range(2000):
    M = rng.integers(3, 12)
    X = rng.normal(size=(M, M)); C = X @ X.T + 0.1 * np.eye(M)
    dg = np.sqrt(np.diag(C)); C = C / dg[:, None] / dg[None, :]
    w, V = np.linalg.eigh(C); l1, l2 = w[-1], w[-2]; u1, u2 = V[:, -1], V[:, -2]
    eps = rng.uniform(0, 1)
    uh = np.sqrt(1 - eps) * u1 + np.sqrt(eps) * u2
    P = np.eye(M) - np.outer(uh, uh)
    lm = np.linalg.eigvalsh(P @ C @ P)[-1]
    tight_err = max(tight_err, abs(lm - (l2 + eps * (l1 - l2))))
    # generic direction
    uh = rng.normal(size=M); uh /= np.linalg.norm(uh)
    eps = 1 - (uh @ u1) ** 2
    P = np.eye(M) - np.outer(uh, uh)
    if np.linalg.eigvalsh(P @ C @ P)[-1] > l2 + eps * (l1 - l2) + 1e-12:
        viol["V1"] += 1
print("V1 tightness max abs error on span(u1,u2):", tight_err)

# ---- main survey with global QPs
configs = []
for kind in ("exp", "gauss"):
    for M in (4, 5, 6, 7):
        for _ in range(10):
            Lam = rng.uniform(1.0, 6.0)
            configs.append((kind, M, Lam, rng.uniform(0, Lam, size=(M, 2))))

rows = []
for ci, (kind, M, Lam, pts) in enumerate(configs):
    C = kernel_matrix(pts, 1.0, kind)
    w, V = np.linalg.eigh(C); l1, l2 = w[-1], w[-2]; u1 = V[:, -1]
    one = np.ones(M) / np.sqrt(M)
    Pu = np.eye(M) - np.outer(one, one)
    PCP = Pu @ C @ Pu
    g_none = G2(C)
    g_top = G2_cancel(C, u1)
    g_cm = G2_cancel(C, one)
    # V4 bracket
    lam_cm = np.linalg.eigvalsh(PCP)[-1]
    lo = M / lam_cm
    up = m_M(M) * np.trace(np.linalg.pinv(PCP, hermitian=True)) / (M - 1)
    gbar = (M * M - np.ones(M) @ C @ np.ones(M)) / (M * (M - 1))
    mid = M / gbar
    ok4 = (lo <= g_cm * (1 + tol)) and (g_cm <= up * (1 + tol)) and (lo <= mid * (1 + tol)) and (mid <= up * (1 + tol))
    viol["V4"] += (not ok4)
    # V1 on the common-mode canceller and the top-mode lower bound
    eps_cm = 1 - (one @ u1) ** 2
    ok1 = (lam_cm <= l2 + eps_cm * (l1 - l2) + 1e-12) and (g_top >= M / l2 * (1 - tol)) and (g_cm >= lo * (1 - tol))
    viol["V1"] += (not ok1)
    # V2/V3 with a noisy estimate
    for s in (0.01, 0.03, 0.1):
        while True:
            E = rng.normal(scale=s, size=(M, M)); E = (E + E.T) / 2; np.fill_diagonal(E, 0)
            Ch = C + E
            if np.linalg.eigvalsh(Ch)[0] > 1e-6:
                break
        e = np.linalg.norm(E, 2)
        wh, Vh = np.linalg.eigh(Ch); uh = Vh[:, -1]
        eps = 1 - (uh @ u1) ** 2
        viol["V2"] += (eps > 4 * e ** 2 / (l1 - l2) ** 2 + 1e-12)
        g_hat = G2_cancel(C, uh)
        Ph = np.eye(M) - np.outer(uh, uh)
        lam_h = np.linalg.eigvalsh(Ph @ C @ Ph)[-1]
        c_none = M / (wh[-1] + e)
        c_top = M / (wh[-2] + e)
        c_cm = M / (np.linalg.eigvalsh(Pu @ Ch @ Pu)[-1] + e)
        ok3 = (g_none >= c_none * (1 - tol)) and (g_hat >= c_top * (1 - tol)) and (g_cm >= c_cm * (1 - tol))
        ok1b = (g_hat >= M / lam_h * (1 - tol)) and (lam_h <= l2 + eps * (l1 - l2) + 1e-12)
        viol["V3"] += (not ok3); viol["V1"] += (not ok1b)
        rows.append(dict(kind=kind, M=M, s=s, e=e, gap=l1 - l2, eps=eps, dk=4 * e ** 2 / (l1 - l2) ** 2,
                         g_hat=g_hat, g_top=g_top, cert_top=c_top, cert_none=c_none, g_none=g_none))
    out["V5"].append(dict(kind=kind, M=M, Lam=Lam, g_none=g_none, g_cm=g_cm, g_top=g_top,
                          lo=lo, up=up, mid=mid, ml1=M / l1, ml2=M / l2, eps_cm=eps_cm))
    print(ci, kind, M, round(Lam, 2), f"none {g_none:.3f} cm {g_cm:.3f} top {g_top:.3f} | cm-bracket [{lo:.3f},{up:.3f}] mid {mid:.3f}", flush=True)

# equicorrelated equality case of V4
eq_err = 0.0
for M in (3, 4, 5, 6, 7, 8):
    for rho in (0.2, 0.5, 0.8):
        C = (1 - rho) * np.eye(M) + rho * np.ones((M, M))
        one = np.ones(M) / np.sqrt(M)
        g = G2_cancel(C, one)
        eq_err = max(eq_err, abs(g / (m_M(M) / (1 - rho)) - 1))
print("V4 equicorrelated: max rel err of G2_cm vs m_M/(1-rho):", eq_err)
print("violations:", viol)

import statistics as st
for kind in ("exp", "gauss"):
    R = [r for r in out["V5"] if r["kind"] == kind]
    f = lambda key: st.median([r[key] for r in R])
    print(kind, "median G2 none/cm/top:", round(f("g_none"), 3), round(f("g_cm"), 3), round(f("g_top"), 3),
          "median ratio cm/none", round(st.median([r["g_cm"] / r["g_none"] for r in R]), 2),
          "median ratio top/cm", round(st.median([r["g_top"] / r["g_cm"] for r in R]), 3),
          "max ratio top/cm", round(max(r["g_top"] / r["g_cm"] for r in R), 3),
          "median eps_cm", round(f("eps_cm"), 4), "max eps_cm", round(max(r["eps_cm"] for r in R), 4),
          "median lo/g_cm", round(st.median([r["lo"] / r["g_cm"] for r in R]), 3),
          "min lo/g_cm", round(min(r["lo"] / r["g_cm"] for r in R), 3),
          "median up/g_cm", round(st.median([r["up"] / r["g_cm"] for r in R]), 3),
          "max up/g_cm", round(max(r["up"] / r["g_cm"] for r in R), 3),
          "median mid/g_cm", round(st.median([r["mid"] / r["g_cm"] for r in R]), 3))
for s in (0.01, 0.03, 0.1):
    R = [r for r in rows if r["s"] == s]
    print("noise", s, "median ||E||", round(st.median([r["e"] for r in R]), 4),
          "median eps", f"{st.median([r['eps'] for r in R]):.2e}",
          "median g_hat/g_top", round(st.median([r["g_hat"] / r["g_top"] for r in R]), 4),
          "min g_hat/g_top", round(min(r["g_hat"] / r["g_top"] for r in R), 4),
          "median cert_top/g_hat", round(st.median([r["cert_top"] / r["g_hat"] for r in R]), 3),
          "median cert_top/cert_none", round(st.median([r["cert_top"] / r["cert_none"] for r in R]), 2))
json.dump(dict(V5=out["V5"], noisy=rows, viol=viol, eq_err=eq_err, tight_err=tight_err),
          open("verify_new_results.json", "w"), indent=1, default=float)
for kind in ("exp", "gauss"):
    R = [r for r in out["V5"] if r["kind"] == kind]
    print(kind, "min ratio cm/none", round(min(r["g_cm"] / r["g_none"] for r in R), 3),
          "min ratio top/none", round(min(r["g_top"] / r["g_none"] for r in R), 3),
          "min top/(M/l2)", round(min(r["g_top"] / r["ml2"] for r in R), 4),
          "median top/(M/l2)", round(st.median([r["g_top"] / r["ml2"] for r in R]), 4))
