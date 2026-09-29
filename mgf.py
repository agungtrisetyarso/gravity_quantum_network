"""Selection overhead E[exp(theta min_j S_j^2)] for the exchangeable model (kappa=0, lambda_1=1).

The survival function P(min_j S_j^2 > t) does not depend on theta, so it is tabulated once,
in the log domain and multiplied by e^{I t} (I = M/2D the exact rate), which removes the
underflow that otherwise occurs at large t.
"""
import numpy as np
from scipy import integrate, stats


def log_surv_scaled(t, M, rho, I):
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    st_ = np.sqrt(t)

    def logf(z):
        lp = np.logaddexp(stats.norm.logcdf((-st_ - a * z) / b), stats.norm.logsf((st_ - a * z) / b))
        return stats.norm.logpdf(z) + M * lp + I * t

    zc = st_ / a if a > 0 else 0.0
    zs = np.linspace(-zc - 12, zc + 12, 4001)
    shift = np.max(logf(zs))
    f = lambda z: np.exp(logf(z) - shift)
    val = integrate.quad(f, zs[0] - 20, zs[-1] + 20, points=sorted({-zc, zc}),
                         epsabs=0, epsrel=1e-11, limit=500)[0]
    return np.log(val) + shift


class MinMGF:
    def __init__(self, M, rho, T=4000.0, n=500):
        self.I = M / (2 * (M * rho + 1 - rho))
        self.t = np.concatenate([[0.0], np.geomspace(1e-5, T, n)])
        self.ls = np.array([log_surv_scaled(tt, M, rho, self.I) for tt in self.t])
        self.slope = np.polyfit(np.log(self.t[-60:]), self.ls[-60:], 1)

    def logS(self, t):
        t = np.asarray(t, float)
        inside = np.interp(t, self.t, self.ls)
        tail = np.polyval(self.slope, np.log(np.maximum(t, 1e-300)))
        return np.where(t <= self.t[-1], inside, tail)

    def __call__(self, theta):
        eps = self.I - theta
        g = lambda t: np.exp(-eps * t + self.logS(t))
        edges = np.concatenate([[0], self.t[1::10], [self.t[-1]]])
        s = sum(integrate.quad(g, edges[k], edges[k + 1], epsabs=0, epsrel=1e-10, limit=200)[0]
                for k in range(len(edges) - 1) if edges[k + 1] > edges[k])
        s += integrate.quad(g, self.t[-1], np.inf, epsabs=0, epsrel=1e-8, limit=400)[0]
        return 1 + theta * s
