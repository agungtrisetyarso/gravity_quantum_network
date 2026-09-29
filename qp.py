"""Global solution of the nonconvex threshold programs by sign-orthant enumeration.

    G^2      = min { y^T C^{-1} y : |y_j| >= 1 }                       (no cancellation)
    G^2(u)   = min { y^T (P C P)^+ y : |y_j| >= 1, U^T y = 0 }          (cancel span U)

On each sign orthant y = diag(s) z, z >= 1, the problem is a convex QP.  One sign is
fixed by the symmetry y -> -y, so 2^(M-1) orthants are solved.
"""
import itertools
import numpy as np
import cvxpy as cp


def kernel_matrix(pts, ell, kind):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    if kind == "exp":
        return np.exp(-d / ell)
    if kind == "gauss":
        return np.exp(-(d / ell) ** 2 / 2)
    raise ValueError(kind)


def _solve_orthants(A, M, U=None, max_orthants=None, rng=None):
    best = np.inf
    signs_iter = itertools.product([1.0, -1.0], repeat=M - 1)
    if max_orthants is not None and 2 ** (M - 1) > max_orthants:
        rng = rng or np.random.default_rng(0)
        signs_iter = (rng.choice([1.0, -1.0], size=M - 1) for _ in range(max_orthants))
    z = cp.Variable(M)
    for tail in signs_iter:
        s = np.concatenate([[1.0], np.asarray(tail)])
        B = (s[:, None] * A) * s[None, :]
        B = (B + B.T) / 2
        cons = [z >= 1]
        if U is not None:
            cons.append((U.T * s[None, :]) @ z == 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(z, cp.psd_wrap(B))), cons)
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                       tol_feas=1e-11)
        except cp.error.SolverError:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and prob.value < best:
            best = prob.value
    return best


def _sqrt(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return V * np.sqrt(np.clip(w, 0, None))


def _solve_primal(B, max_orthants=None, rng=None):
    """min |x|^2 s.t. |(Bx)_j| >= 1 for all j  (no matrix inversion; well conditioned)."""
    M, n = B.shape
    best = np.inf
    signs_iter = itertools.product([1.0, -1.0], repeat=M - 1)
    if max_orthants is not None and 2 ** (M - 1) > max_orthants:
        rng = rng or np.random.default_rng(0)
        signs_iter = (rng.choice([1.0, -1.0], size=M - 1) for _ in range(max_orthants))
    x = cp.Variable(n)
    for tail in signs_iter:
        s = np.concatenate([[1.0], np.asarray(tail)])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.multiply(s, B @ x) >= 1])
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        except cp.error.SolverError:
            continue
        if prob.status == "optimal" and prob.value < best:
            best = prob.value
    return best


def G2(C, **kw):
    """Squared threshold gain, no cancellation: 2 x (LDP rate) with unit static reference."""
    return _solve_primal(_sqrt(C), **kw)


def G2_cancel(C, U, **kw):
    """Threshold (squared gain vs. the uncancelled static path) after cancelling span(U)."""
    M = C.shape[0]
    U = U.reshape(M, -1)
    U, _ = np.linalg.qr(U)
    P = np.eye(M) - U @ U.T
    return _solve_primal(P @ _sqrt(C), **kw)


def G2_cancel_dual(C, U, **kw):
    """Same quantity via the pseudoinverse (dual) form; kept as a cross-check."""
    M = C.shape[0]
    U = U.reshape(M, -1)
    U, _ = np.linalg.qr(U)
    P = np.eye(M) - U @ U.T
    A = np.linalg.pinv(P @ C @ P, rcond=1e-12, hermitian=True)
    return _solve_orthants(A, M, U=U, **kw)


def m_M(M):
    return M if M % 2 == 0 else M * (M + 1) / (M - 1)
