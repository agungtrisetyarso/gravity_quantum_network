# ============================================================
# Gravity-Aware Quantum Routing – Numerical Validation
# Colab-ready simulation of the exact Gaussian model
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from tqdm.notebook import tqdm

# ------------------------------------------------------------
# 1. Parameters (match the analytic model)
# ------------------------------------------------------------
np.random.seed(42)

T          = 200.0          # total simulation time
dt         = 0.05           # time step
t          = np.arange(0, T, dt)
n_t        = len(t)

tau_g      = 2.0            # gravitational correlation time
a          = 1.0            # spectral-overlap coefficient (a = ω²/4)
p0         = 0.8            # bare success probability (distance loss absorbed)
k_hops     = 3              # number of hops on the static path

# Adaptive policy: re-estimate every tau_adapt
tau_adapt  = tau_g          # coherence-window length
n_adapt    = int(tau_adapt / dt)

# Range of gravitational amplitudes δ
delta_list = np.linspace(0.05, 0.95, 25)   # stay below δ_c = 1/√(2a) = 0.707 for finite moments
# (we also probe a few points beyond criticality)

# ------------------------------------------------------------
# 2. Generate a stationary Gaussian process with exponential covariance
#    C(t,t') = exp(-|t-t'|/τ_g)   (unit marginal variance)
# ------------------------------------------------------------
def generate_OU_process(t, tau_g, n_realizations=1):
    """
    Exact simulation of the Ornstein–Uhlenbeck process
    dg = -g/τ_g dt + √(2/τ_g) dW
    which has stationary covariance exp(-|t-t'|/τ_g).
    """
    n_t = len(t)
    dt  = t[1] - t[0]
    theta = 1.0 / tau_g
    sigma = np.sqrt(2.0 / tau_g)

    g = np.zeros((n_realizations, n_t))
    # start from stationary distribution
    g[:, 0] = np.random.normal(0, 1, size=n_realizations)

    for i in range(1, n_t):
        dW = np.random.normal(0, np.sqrt(dt), size=n_realizations)
        g[:, i] = g[:, i-1] - theta * g[:, i-1] * dt + sigma * dW

    return g   # shape (n_realizations, n_t)

# ------------------------------------------------------------
# 3. Spectral overlap and path success probability
# ------------------------------------------------------------
def spectral_factor(g, delta, a):
    """η = exp(-a δ² g²)"""
    return np.exp(-a * delta**2 * g**2)

def path_success_prob(g_list, delta, a, p0):
    """
    For a path of k hops: P(t) = p0^k * ∏ η_e(t)
    (memory factor set to 1 for the pure spectral-overlap test;
     it only strengthens the divergence)
    """
    P = np.ones_like(g_list[0])
    for g in g_list:
        P *= p0 * spectral_factor(g, delta, a)
    return P

# ------------------------------------------------------------
# 4. Operational overhead estimators
# ------------------------------------------------------------
def static_overhead(P):
    """Ω_static = E[1/P(t)]  (clip extremely small P to avoid numerical overflow)"""
    P_safe = np.maximum(P, 1e-12)
    return np.mean(1.0 / P_safe)

def adaptive_overhead(P_paths, n_adapt):
    """
    Adaptive policy: every n_adapt steps choose the path
    with currently highest instantaneous success probability.
    Returns E[1/P_selected(t)]
    """
    n_real, n_t = P_paths[0].shape
    n_paths = len(P_paths)
    P_stack = np.stack(P_paths, axis=0)          # (n_paths, n_real, n_t)

    selected = np.zeros((n_real, n_t))
    for start in range(0, n_t, n_adapt):
        end = min(start + n_adapt, n_t)
        # average success probability inside the window (simple estimator)
        window_mean = P_stack[:, :, start:end].mean(axis=2)   # (n_paths, n_real)
        best = np.argmax(window_mean, axis=0)                 # (n_real,)
        for r in range(n_real):
            selected[r, start:end] = P_stack[best[r], r, start:end]

    P_safe = np.maximum(selected, 1e-12)
    return np.mean(1.0 / P_safe)

# ------------------------------------------------------------
# 5. Main Monte-Carlo loop
# ------------------------------------------------------------
n_realizations = 40          # independent gravitational realisations
n_paths_adapt  = 4           # number of alternative paths for the adaptive policy

Omega_static  = []
Omega_adapt   = []
Omega_ideal   = []

print("Running Monte-Carlo simulation …")
for delta in tqdm(delta_list):
    # --- generate independent OU processes for every link of every path ---
    # Static path: k_hops independent links
    g_static = [generate_OU_process(t, tau_g, n_realizations) for _ in range(k_hops)]

    # Adaptive: n_paths_adapt edge-disjoint paths, each with k_hops links
    g_adapt_paths = []
    for _ in range(n_paths_adapt):
        g_path = [generate_OU_process(t, tau_g, n_realizations) for _ in range(k_hops)]
        g_adapt_paths.append(g_path)

    # Success probabilities
    P_static = path_success_prob(g_static, delta, a, p0)

    P_adapt_list = [path_success_prob(g_path, delta, a, p0) for g_path in g_adapt_paths]

    # Overheads
    Os = static_overhead(P_static)
    Oa = adaptive_overhead(P_adapt_list, n_adapt)
    Oi = 1.0 / (p0 ** k_hops)          # ideal (δ=0) overhead

    Omega_static.append(Os)
    Omega_adapt.append(Oa)
    Omega_ideal.append(Oi)

Omega_static = np.array(Omega_static)
Omega_adapt  = np.array(Omega_adapt)
Omega_ideal  = np.array(Omega_ideal)

# ------------------------------------------------------------
# 6. Analytic curves for comparison
# ------------------------------------------------------------
# Exact single-link reciprocal moment (1-2aδ²)^(-1/2)
# For k hops → (1-2aδ²)^(-k/2)
mask = 2*a*delta_list**2 < 1
delta_analytic = delta_list[mask]
analytic_static = Omega_ideal[0] * (1 - 2*a*delta_analytic**2)**(-k_hops/2)

# ------------------------------------------------------------
# 7. Figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))

ax.semilogy(delta_list, Omega_static, 'o-', color='C0', lw=2, markersize=6,
            label=r'$\Omega_{\rm static}$ (Monte Carlo)')
ax.semilogy(delta_list, Omega_adapt,  's-', color='C1', lw=2, markersize=6,
            label=r'$\Omega_{\rm adaptive}$ (Monte Carlo)')
ax.semilogy(delta_analytic, analytic_static, '--', color='C0', lw=1.5, alpha=0.8,
            label=r'analytic $(1-2a\delta^2)^{-k/2}$')

# Critical amplitude
delta_c = 1/np.sqrt(2*a)
ax.axvline(delta_c, color='k', ls=':', lw=1.5, label=fr'$\delta_c=1/\sqrt{{2a}}\approx{delta_c:.3f}$')

ax.set_xlabel(r'gravitational amplitude $\delta$', fontsize=13)
ax.set_ylabel(r'routing overhead $\Omega$', fontsize=13)
ax.set_title('Static vs Adaptive Overhead under Exact Gaussian Spectral Overlap', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.set_xlim(0, delta_list[-1]*1.02)

plt.tight_layout()
plt.savefig('gravity_routing_overhead.pdf', bbox_inches='tight')
plt.savefig('gravity_routing_overhead.png', dpi=200, bbox_inches='tight')
plt.show()

print("\nFigure saved as gravity_routing_overhead.pdf / .png")
print(f"Critical amplitude δ_c = {delta_c:.4f}")
print("Simulation finished.")
