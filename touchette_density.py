"""
Exact white-noise velocity densities of Touchette, Van der Straeten &
Just (2010), converted from their nondimensional variables (x, tau)
back to physical variables (v, t) for GENERAL parameters.

Physical model:
    dV = (-gamma V + ell) dt - Delta sign(V) dt + sigma dW
Fokker--Planck:
    d_t p = d_v[ (gamma v + Delta sign(v) - ell) p ] + (sigma^2/2) d_vv p
Touchette's Eq. (1.4) is the same with Gamma = sigma^2, a = ell.
Dictionary:  Gamma = sigma^2,  a = ell,  Delta = Delta,  gamma = gamma.

Scalings.
  Regime I   (gamma = ell = 0), Touchette Eq. (2.1):
        x = (2 Delta / Gamma) v,        tau = (2 Delta^2 / Gamma) t
  Regimes II, III (gamma > 0), Touchette Eq. (3.1):
        x = (2 gamma / Gamma)^{1/2} v,  tau = gamma t
  with, from Eqs. (3.2) and (4.1),
        delta = Delta (2/(gamma Gamma))^{1/2},
        b     = ell   (2/(gamma Gamma))^{1/2}.
A density is not invariant under a change of variables: it carries the
Jacobian J = dx/dv,
    p_v(v, t | v_0, 0) = J * p_x( J v, tau(t) | J v_0, 0 ).
All numerical defaults are imported from `params.py`.

Requires: numpy, scipy, params.py.
"""
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from scipy.special import pbdv, erf
from scipy.optimize import brentq
from scipy.integrate import quad



# Default values, pulled from params.py
from params import (
    # physical model parameters
    DELTA, GAMMA, SIGMA, ELL,
    # initial condition and time
    V0, T,
    # numerical parameters
    N_MODES, STEP, LAM_START,
    CUTOFF, QUAD_LIM,                       # generic (fall-back)
    CUTOFF_II, QUAD_LIM_II,                 # regime II
    CUTOFF_III, QUAD_LIM_III,               # regime III
)


# Parabolic cylinder function and eigenvalue solvers

def Dv(nu, x):
    """Parabolic cylinder function D_nu(x)."""
    return pbdv(nu, float(x))[0]


def odd_eigenvalues(delta, n_modes=N_MODES, step=STEP, lam_start=LAM_START):
    """Regime II: odd-mode eigenvalues, roots of D_Lambda(delta) = 0
    (Touchette et al. 2010, Eq. (3.3)).  Located by scanning for sign
    changes and refined with Brent's method."""
    roots, l, v0 = [], lam_start, Dv(lam_start, delta)
    while len(roots) < n_modes:
        l_next, v_next = l + step, Dv(l + step, delta)
        if v0 == 0.0:
            roots.append(l)
        elif v0 * v_next < 0:
            roots.append(brentq(lambda x: Dv(x, delta), l, l_next, xtol=1e-11))
        l, v0 = l_next, v_next
    return np.array(roots)


def char_eq3(nu, delta, beta):
    """Regime III characteristic function, Touchette et al. 2010 Eq. (4.2)."""
    return (Dv(nu, delta + beta) * Dv(nu - 1.0, delta - beta)
            + Dv(nu, delta - beta) * Dv(nu - 1.0, delta + beta))


def eigenvalues_case3(delta, beta, n_modes=N_MODES,
                      step=STEP, lam_start=LAM_START):
    """Regime III eigenvalues, roots of char_eq3."""
    roots, l, v0 = [], lam_start, char_eq3(lam_start, delta, beta)
    while len(roots) < n_modes:
        l_next = l + step
        v_next = char_eq3(l_next, delta, beta)
        if v0 == 0.0:
            roots.append(l)
        elif v0 * v_next < 0:
            roots.append(brentq(lambda x: char_eq3(x, delta, beta),
                                l, l_next, xtol=1e-11))
        l, v0 = l_next, v_next
    return np.array(roots)


# Regime I: closed-form propagator (dimensionless, Eq. (2.10))

def density_regime1_nd(x, tau, xp):
    """Touchette et al. (2010), Eq. (2.10): the exact propagator for
    pure dry friction in nondimensional variables (x, tau)."""
    x = np.asarray(x, dtype=float)
    term1 = (np.exp(-tau / 4.0) / (2 * np.sqrt(np.pi * tau))
             * np.exp(-(np.abs(x) - abs(xp)) / 2.0)
             * np.exp(-(x - xp) ** 2 / (4 * tau)))
    term2 = (np.exp(-np.abs(x)) / 4.0) * (
        1.0 + erf((tau - (np.abs(x) + abs(xp))) / (2 * np.sqrt(tau))))
    return term1 + term2


# Regime II: symmetric piecewise-parabolic potential (Eq. (3.12))

def _Z_case2(lam, delta, cutoff=CUTOFF_II):
    val, _ = quad(lambda y: Dv(lam, y) ** 2, delta, cutoff,
                  limit=150, epsabs=1e-10, epsrel=1e-8)
    return 2.0 * val


def make_density_regime2_nd(tau, xp, delta,
                            n_modes=N_MODES,
                            cutoff=CUTOFF_II,
                            quad_lim=QUAD_LIM_II):
    """Return callable x -> p_x(x, tau | xp, 0) for regime II,
    assembled from the truncated spectral sum (Eq. (3.12)) plus the
    stationary part, in nondimensional variables."""
    odd = odd_eigenvalues(delta, n_modes)
    even = odd + 1.0
    lams = np.concatenate([odd, even])
    parities = ['odd'] * len(odd) + ['even'] * len(even)
    Zs = np.array([_Z_case2(l, delta, cutoff) for l in lams])

    Phi_xp = (abs(xp) + delta) ** 2 / 2.0
    y0 = abs(xp) + delta
    un_xp = np.exp(-y0 ** 2 / 4.0) * np.array([Dv(l, y0) for l in lams])
    un_xp = np.array([u if p == 'even' else u * np.sign(xp)
                      for u, p in zip(un_xp, parities)])
    coeff = np.exp(-lams * tau) * un_xp * np.exp(Phi_xp) / Zs

    Zstar, _ = quad(lambda x: np.exp(-(abs(x) + delta) ** 2 / 2.0),
                    -quad_lim, quad_lim, limit=200)

    def p(x):
        y = abs(x) + delta
        un_x = np.exp(-y ** 2 / 4.0) * np.array([Dv(l, y) for l in lams])
        un_x = np.array([u if par == 'even' else u * np.sign(x)
                         for u, par in zip(un_x, parities)])
        return (np.exp(-(abs(x) + delta) ** 2 / 2.0) / Zstar
                + float(np.sum(coeff * un_x)))
    return p

# Regime III: asymmetric potential with constant force (Eq. (4.7))
def _Phi3_nd(x, delta, beta):
    return (abs(x) + delta) ** 2 / 2.0 - beta * x

def _chi_case3(nu, delta, beta):
    num = np.exp(-(delta - beta) ** 2 / 4.0) * Dv(nu, delta - beta)
    den = np.exp(-(delta + beta) ** 2 / 4.0) * Dv(nu, delta + beta)
    return num / den

def _u_case3(nu, delta, beta, x, chi):
    if x >= 0:
        return np.exp(-(x + delta - beta) ** 2 / 4.0) * Dv(nu, x + delta - beta)
    return chi * np.exp(-(-x + delta + beta) ** 2 / 4.0) * Dv(nu, -x + delta + beta)

def _Z_case3(nu, delta, beta, chi, cutoff=CUTOFF_III):
    # Gaussian prefactors cancelled analytically to avoid overflow.
    c_pos = np.exp(beta * delta - beta ** 2 / 2.0)
    c_neg = chi ** 2 * np.exp(-beta * delta - beta ** 2 / 2.0)
    Ipos, _ = quad(lambda x: Dv(nu, x + delta - beta) ** 2, 0.0, cutoff,
                   limit=200, epsabs=1e-10, epsrel=1e-8)
    Ineg, _ = quad(lambda y: Dv(nu, y + delta + beta) ** 2, 0.0, cutoff,
                   limit=200, epsabs=1e-10, epsrel=1e-8)
    return c_pos * Ipos + c_neg * Ineg

def make_density_regime3_nd(tau, xp, delta, beta,
                            n_modes=N_MODES,
                            cutoff=CUTOFF_III,
                            quad_lim=QUAD_LIM_III):
    """Return callable x -> p_x(x, tau | xp, 0) for regime III,
    assembled from the truncated spectral sum (Eq. (4.7)), in
    nondimensional variables."""
    nus = eigenvalues_case3(delta, beta, n_modes)
    chis = np.array([_chi_case3(nu, delta, beta) for nu in nus])
    Zs = np.array([_Z_case3(nu, delta, beta, chi, cutoff)
                   for nu, chi in zip(nus, chis)])
    Phi_xp = _Phi3_nd(xp, delta, beta)
    un_xp = np.array([_u_case3(nu, delta, beta, xp, chi)
                      for nu, chi in zip(nus, chis)])
    coeff = np.exp(-nus * tau) * un_xp * np.exp(Phi_xp) / Zs

    Zstar, _ = quad(lambda x: np.exp(-_Phi3_nd(x, delta, beta)),
                    -quad_lim, quad_lim, limit=300, points=[0.0])

    def p(x):
        s = 0.0
        for c, nu, chi in zip(coeff, nus, chis):
            s += c * _u_case3(nu, delta, beta, x, chi)
        return np.exp(-_Phi3_nd(x, delta, beta)) / Zstar + s
    return p

# Change of variables: nondimensional (x, tau) -> physical (v, t)
def scaling(regime, Delta, gamma, sigma, ell):
    """Return (J, tau_of_t, delta, b) for the requested regime.

    J = dx/dv is the Jacobian the physical density must be multiplied by;
    tau_of_t(t) = tau; delta, b are the (dimensionless) Touchette
    parameters.  For regime I, delta and b are None.
    """
    Gam = sigma ** 2
    if regime == "I":
        J = 2.0 * Delta / Gam
        return J, (lambda t: 2.0 * Delta ** 2 / Gam * t), None, None
    J = np.sqrt(2.0 * gamma / Gam)
    s = np.sqrt(2.0 / (gamma * Gam))
    return J, (lambda t: gamma * t), Delta * s, ell * s


# Physical density for general parameters

def make_density(regime, T=T, v0=V0,
                 Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL,
                 n_modes=N_MODES, cutoff=None, quad_lim=None):
    """Return callable v -> p_v(v, T | v0, 0), for arbitrary parameters.

    `cutoff` and `quad_lim` fall back to the regime-specific defaults
    CUTOFF_II / QUAD_LIM_II (regime II) and CUTOFF_III / QUAD_LIM_III
    (regime III) defined in `params.py`.
    """
    J, tau_of, delta, b = scaling(regime, Delta, gamma, sigma, ell)
    tau = tau_of(T)
    xp = J * v0
    if regime == "I":
        return lambda v: J * float(density_regime1_nd(J * v, tau, xp))
    if regime == "II":
        cut = CUTOFF_II   if cutoff   is None else cutoff
        ql  = QUAD_LIM_II if quad_lim is None else quad_lim
        p_nd = make_density_regime2_nd(tau, xp, delta,
                                       n_modes=n_modes,
                                       cutoff=cut,
                                       quad_lim=ql)
        return lambda v: J * p_nd(J * v)
    if regime == "III":
        cut = CUTOFF_III   if cutoff   is None else cutoff
        ql  = QUAD_LIM_III if quad_lim is None else quad_lim
        p_nd = make_density_regime3_nd(tau, xp, delta, b,
                                       n_modes=n_modes,
                                       cutoff=cut,
                                       quad_lim=ql)
        return lambda v: J * p_nd(J * v)
    raise ValueError(regime)


# Expectation of an arbitrary observable
def expectation(g, regime, T=T, v0=V0,
                Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL,
                n_modes=N_MODES, cutoff=None, quad_lim=None, breakpoints=()):
    """E[g(V^0(T)) | V^0(0)=v0], by quadrature of g against the exact
    physical Touchette density.

    breakpoints : interior points where g is non-smooth (e.g. +/- v_f for
                  threshold observables); splitting the quadrature there
                  keeps the integration accurate for kinked or
                  discontinuous g.
    """
    p = make_density(regime, T, v0, Delta, gamma, sigma, ell,
                     n_modes=n_modes, cutoff=cutoff, quad_lim=quad_lim)
    # Outermost quadrature cut-off: use the regime-specific default if
    # none was supplied (regime I has no regime-specific value, so it
    # falls back to the generic QUAD_LIM).
    if quad_lim is not None:
        ql = quad_lim
    elif regime == "II":
        ql = QUAD_LIM_II
    elif regime == "III":
        ql = QUAD_LIM_III
    else:
        ql = QUAD_LIM
    pts = sorted(set([-ql, 0.0] + list(breakpoints) + [ql]))
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val, _ = quad(lambda v: g(v) * p(v), a, b,
                      limit=200, epsabs=1e-11, epsrel=1e-9)
        total += val
    return total

def normalisation_check(regime, T=T, v0=V0,
                        Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL,
                        n_modes=N_MODES, cutoff=None, quad_lim=None):
    """Integral of the density over R; should be 1 to quadrature accuracy."""
    return expectation(lambda v: 1.0, regime, T, v0,
                       Delta, gamma, sigma, ell,
                       n_modes=n_modes, cutoff=cutoff, quad_lim=quad_lim)

