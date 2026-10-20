"""
touchette_density.py -- the law of the white-noise terminal velocity V^0(T).

Exact white-noise velocity densities of Touchette, Van der Straeten & Just (2010)
[TVJ10], converted from their nondimensional variables (x, tau) to physical
variables (v, t), together with the explicit density bound of the manuscript.

Physical model (white noise, intensity s):
    dV = (-gamma V + ell) dt - Delta sgn(V) dt + s dB,
Fokker--Planck equation (manuscript (FP)):
    d_t p = d_v[ (gamma v + Delta sgn(v) - ell) p ] + (s^2/2) d_vv p.
[TVJ10, Eq. (1.4)] is the same with  Gamma = s^2,  a = ell.

IMPORTANT: the argument `sigma` of every function below is the intensity s of the
white noise driving V^0, i.e. s = sigma |c K| in the notation of the manuscript
(params.S_WN).  With A = K = r = 1 (Section 5.1), s = sigma.

Scalings.
  Regime I   (gamma = ell = 0), [TVJ10, Eq. (2.1)]:
        x = (2 Delta / Gamma) v,        tau = (2 Delta^2 / Gamma) t
  Regimes II, III (gamma > 0), [TVJ10, Eq. (3.1)]:
        x = (2 gamma / Gamma)^{1/2} v,  tau = gamma t,
  with, from [TVJ10, Eqs. (3.2), (4.1)],
        delta = Delta (2/(gamma Gamma))^{1/2},   b = ell (2/(gamma Gamma))^{1/2}.
A density carries the Jacobian J = dx/dv:
    p_v(v, t | v_0) = J * p_x(J v, tau(t) | J v_0).

Representations used.
  Regime I   : closed form [TVJ10, Eq. (2.10)]  (= manuscript (pI));
  Regime II  : spectral series [TVJ10, Eq. (3.12)] (odd modes D_L(delta) = 0,
               even modes L + 1);
  Regime III : spectral series [TVJ10, Eq. (4.7)] (characteristic equation (4.2)).
The truncated series are cross-validated in validate_reference.py against the
Laplace-transform solution of Touchette, Prellberg & Just (2012) and against a
finite-volume solution of (FP).

Requires: numpy, scipy, params.py.
"""
import numpy as np
from scipy.special import pbdv, erf, erfc
from scipy.optimize import brentq
from scipy.integrate import quad

from params import (
    DELTA, GAMMA, S_WN, ELL, V0, T,
    N_MODES, STEP, LAM_START,
    QUAD_LIM, CUTOFF_II, QUAD_LIM_II, CUTOFF_III, QUAD_LIM_III,
)

# quadrature tolerances for the normalisation integrals of the eigenfunctions
_ZQ = dict(limit=400, epsabs=0.0, epsrel=1e-12)

# scipy.special.pbdv is accurate at integer orders but not just above a positive
# integer n: at nu = n + 1e-9, n + 1e-12 and n + 1e-15 its absolute error is about
# 1e-7, 1e-5 and 1e-1 (checked against mpmath.pcfd).  This matters here: for
# delta = 1 (the test problem) the first odd eigenvalue of regime II is the integer
# 2 (D_2(1) = 0), and the root finder returns 2 +- 1e-13; without the correction
# the normalisation of the regime-II density is off by 5e-4.  Within _NEAR_INT of
# an integer, D_nu is therefore evaluated by 5-point Lagrange interpolation in nu
# through the nodes n + k h, k = -2..2, at which pbdv is accurate (error 1e-14).
_NEAR_INT = 1e-4
_H_INT = 2e-3
_NODES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * _H_INT


# ----------------------------------------------------------------------------
# Parabolic cylinder function and eigenvalue solvers
# ----------------------------------------------------------------------------
def Dv(nu, x):
    """Parabolic cylinder function D_nu(x) (Whittaker), scalar or array,
    accurate also for orders close to an integer (see _NEAR_INT)."""
    nu_b, x_b = np.broadcast_arrays(np.asarray(nu, dtype=float),
                                    np.asarray(x, dtype=float))
    out = np.array(pbdv(nu_b, x_b)[0], dtype=float, copy=True)
    n = np.rint(nu_b)
    eta = nu_b - n
    bad = (eta != 0.0) & (np.abs(eta) < _NEAR_INT)
    if np.any(bad):
        nb, eb, xb = n[bad], eta[bad], x_b[bad]
        vals = np.stack([pbdv(nb + t, xb)[0] for t in _NODES])        # (5, m)
        w = np.ones((_NODES.size, eb.size))
        for k, tk in enumerate(_NODES):
            for j, tj in enumerate(_NODES):
                if j != k:
                    w[k] *= (eb - tj) / (tk - tj)
        out[bad] = np.sum(w * vals, axis=0)
    return out if out.ndim else float(out)


def odd_eigenvalues(delta, n_modes=N_MODES, step=STEP, lam_start=LAM_START):
    """Regime II: odd-mode eigenvalues, the roots of D_L(delta) = 0
    [TVJ10, Eq. (3.3)], located by a sign scan and refined by Brent's method."""
    roots, l, v0 = [], lam_start, Dv(lam_start, delta)
    while len(roots) < n_modes:
        l_next = l + step
        v_next = Dv(l_next, delta)
        if v0 == 0.0:
            roots.append(l)
        elif v0 * v_next < 0:
            roots.append(brentq(lambda z: Dv(z, delta), l, l_next, xtol=1e-13))
        l, v0 = l_next, v_next
    return np.array(roots)


def char_eq3(nu, delta, beta):
    """Regime III characteristic function [TVJ10, Eq. (4.2)] without the factor L."""
    return (Dv(nu, delta + beta) * Dv(nu - 1.0, delta - beta)
            + Dv(nu, delta - beta) * Dv(nu - 1.0, delta + beta))


def eigenvalues_case3(delta, beta, n_modes=N_MODES, step=STEP, lam_start=LAM_START):
    """Regime III eigenvalues (L > 0), roots of char_eq3."""
    roots, l, v0 = [], lam_start, char_eq3(lam_start, delta, beta)
    while len(roots) < n_modes:
        l_next = l + step
        v_next = char_eq3(l_next, delta, beta)
        if v0 == 0.0:
            roots.append(l)
        elif v0 * v_next < 0:
            roots.append(brentq(lambda z: char_eq3(z, delta, beta),
                                l, l_next, xtol=1e-13))
        l, v0 = l_next, v_next
    return np.array(roots)


# ----------------------------------------------------------------------------
# Regime I: closed form, [TVJ10, Eq. (2.10)], nondimensional
# ----------------------------------------------------------------------------
def density_regime1_nd(x, tau, xp):
    """Exact propagator of pure dry friction in the variables (x, tau)."""
    x = np.asarray(x, dtype=float)
    term1 = (np.exp(-tau / 4.0) / (2.0 * np.sqrt(np.pi * tau))
             * np.exp(-(np.abs(x) - abs(xp)) / 2.0)
             * np.exp(-(x - xp) ** 2 / (4.0 * tau)))
    term2 = (np.exp(-np.abs(x)) / 4.0) * (
        1.0 + erf((tau - (np.abs(x) + abs(xp))) / (2.0 * np.sqrt(tau))))
    return term1 + term2


def density_pure_dry(v, T=T, v0=V0, Delta=DELTA, sigma=S_WN):
    """Closed-form density of pure dry friction in PHYSICAL variables,
    manuscript (pI):
        e^{-Delta^2 T/(2 s^2) - Delta(|v|-|v0|)/s^2} e^{-(v-v0)^2/(2 s^2 T)} / sqrt(2 pi s^2 T)
        + (Delta/s^2) e^{-2 Delta |v| / s^2} Phibar((|v|+|v0|-Delta T)/(s sqrt T))."""
    v = np.asarray(v, dtype=float)
    s2 = sigma ** 2
    a = np.abs(v) + abs(v0)
    t1 = (np.exp(-Delta ** 2 * T / (2 * s2) - Delta * (np.abs(v) - abs(v0)) / s2
                 - (v - v0) ** 2 / (2 * s2 * T)) / np.sqrt(2 * np.pi * s2 * T))
    t2 = (Delta / s2) * np.exp(-2 * Delta * np.abs(v) / s2) * 0.5 * erfc(
        (a - Delta * T) / (sigma * np.sqrt(2 * T)))
    return t1 + t2


# ----------------------------------------------------------------------------
# Regime II: symmetric piecewise-parabolic potential, [TVJ10, Eq. (3.12)]
# ----------------------------------------------------------------------------
def _Z_case2(lam, delta, cutoff=CUTOFF_II):
    """Z = int v u dx = 2 int_delta^cutoff D_lam(y)^2 dy (both parities)."""
    val, _ = quad(lambda y: Dv(lam, y) ** 2, delta, cutoff, **_ZQ)
    return 2.0 * val


def make_density_regime2_nd(tau, xp, delta, n_modes=N_MODES,
                            cutoff=CUTOFF_II, quad_lim=QUAD_LIM_II):
    """Callable x -> p_x(x, tau | xp) for regime II (vectorised in x)."""
    odd = odd_eigenvalues(delta, n_modes)
    lams = np.concatenate([odd, odd + 1.0])                 # odd, then even modes
    is_odd = np.r_[np.ones(odd.size, bool), np.zeros(odd.size, bool)]
    Zs = np.array([_Z_case2(l, delta, cutoff) for l in lams])

    y0 = abs(xp) + delta
    u_xp = np.exp(-y0 ** 2 / 4.0) * Dv(lams, y0)
    u_xp = np.where(is_odd, u_xp * np.sign(xp), u_xp)
    coeff = np.exp(-lams * tau) * u_xp * np.exp(y0 ** 2 / 2.0) / Zs   # v(xp) = u(xp) e^{Phi(xp)}

    Zstar, _ = quad(lambda x: np.exp(-(abs(x) + delta) ** 2 / 2.0),
                    -quad_lim, quad_lim, points=[0.0], limit=200)

    def p(x):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.abs(xa) + delta
        u = np.exp(-y[None, :] ** 2 / 4.0) * Dv(lams[:, None], y[None, :])
        u = np.where(is_odd[:, None], u * np.sign(xa)[None, :], u)
        out = np.exp(-y ** 2 / 2.0) / Zstar + coeff @ u
        return out if np.ndim(x) else float(out[0])
    return p


# ----------------------------------------------------------------------------
# Regime III: asymmetric potential with constant force, [TVJ10, Eq. (4.7)]
# ----------------------------------------------------------------------------
def _Phi3_nd(x, delta, beta):
    return (np.abs(x) + delta) ** 2 / 2.0 - beta * x


def _chi_case3(nu, delta, beta):
    """Amplitude ratio, [TVJ10, Eq. (4.5)], times the Gaussian factors of _u_case3."""
    num = np.exp(-(delta - beta) ** 2 / 4.0) * Dv(nu, delta - beta)
    den = np.exp(-(delta + beta) ** 2 / 4.0) * Dv(nu, delta + beta)
    return num / den


def _u_case3(nu, delta, beta, x, chi):
    """Eigenfunction [TVJ10, Eq. (4.4)] up to the constant factor e^{b delta/2 - b^2/4},
    which cancels in u(x) v(xp) / Z.  Vectorised in x (and in nu, chi if arrays)."""
    x = np.asarray(x, dtype=float)
    pos = np.exp(-(x + delta - beta) ** 2 / 4.0) * Dv(nu, np.abs(x) + delta - beta)
    neg = chi * np.exp(-(-x + delta + beta) ** 2 / 4.0) * Dv(nu, np.abs(x) + delta + beta)
    return np.where(x >= 0, pos, neg)


def _Z_case3(nu, delta, beta, chi, cutoff=CUTOFF_III):
    """Z = int u^2 e^{Phi} dx; the Gaussian prefactors are cancelled analytically."""
    c_pos = np.exp(beta * delta - beta ** 2 / 2.0)
    c_neg = chi ** 2 * np.exp(-beta * delta - beta ** 2 / 2.0)
    Ipos, _ = quad(lambda x: Dv(nu, x + delta - beta) ** 2, 0.0, cutoff, **_ZQ)
    Ineg, _ = quad(lambda y: Dv(nu, y + delta + beta) ** 2, 0.0, cutoff, **_ZQ)
    return c_pos * Ipos + c_neg * Ineg


def make_density_regime3_nd(tau, xp, delta, beta, n_modes=N_MODES,
                            cutoff=CUTOFF_III, quad_lim=QUAD_LIM_III):
    """Callable x -> p_x(x, tau | xp) for regime III (vectorised in x)."""
    nus = eigenvalues_case3(delta, beta, n_modes)
    chis = np.array([_chi_case3(nu, delta, beta) for nu in nus])
    Zs = np.array([_Z_case3(nu, delta, beta, chi, cutoff) for nu, chi in zip(nus, chis)])
    u_xp = np.array([float(_u_case3(nu, delta, beta, xp, chi)) for nu, chi in zip(nus, chis)])
    coeff = np.exp(-nus * tau) * u_xp * np.exp(_Phi3_nd(xp, delta, beta)) / Zs

    Zstar, _ = quad(lambda x: np.exp(-_Phi3_nd(x, delta, beta)),
                    -quad_lim, quad_lim, limit=300, points=[0.0])

    def p(x):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        u = _u_case3(nus[:, None], delta, beta, xa[None, :], chis[:, None])
        out = np.exp(-_Phi3_nd(xa, delta, beta)) / Zstar + coeff @ u
        return out if np.ndim(x) else float(out[0])
    return p


# ----------------------------------------------------------------------------
# Change of variables: nondimensional (x, tau) -> physical (v, t)
# ----------------------------------------------------------------------------
def scaling(regime, Delta, gamma, sigma, ell):
    """Return (J, tau_of_t, delta, b) for the requested regime.

    J = dx/dv; tau_of_t(t) = tau; delta, b are the dimensionless parameters of
    [TVJ10] (None in regime I).  In regime II the force ell is ignored (b = 0)."""
    Gam = sigma ** 2
    if regime == "I":
        J = 2.0 * Delta / Gam
        return J, (lambda t: 2.0 * Delta ** 2 / Gam * t), None, None
    J = np.sqrt(2.0 * gamma / Gam)
    sc = np.sqrt(2.0 / (gamma * Gam))
    b = 0.0 if regime == "II" else ell * sc
    return J, (lambda t: gamma * t), Delta * sc, b


_CACHE = {}


def make_density(regime, T=T, v0=V0, Delta=DELTA, gamma=GAMMA, sigma=S_WN, ell=ELL,
                 n_modes=N_MODES, cutoff=None, quad_lim=None):
    """Return the callable v -> p_v(v, T | v0) (scalar or array argument).

    `sigma` is the white-noise intensity s.  `cutoff` and `quad_lim` default to
    the regime-specific values CUTOFF_II/QUAD_LIM_II and CUTOFF_III/QUAD_LIM_III.
    Results are cached, since the eigen-decomposition is the expensive part."""
    key = (regime, float(T), float(v0), float(Delta), float(gamma), float(sigma),
           float(ell), int(n_modes), cutoff, quad_lim)
    if key in _CACHE:
        return _CACHE[key]
    J, tau_of, delta, b = scaling(regime, Delta, gamma, sigma, ell)
    tau, xp = tau_of(T), J * v0
    if regime == "I":
        def p(v):
            out = J * density_regime1_nd(J * np.asarray(v, dtype=float), tau, xp)
            return out if np.ndim(v) else float(out)
    elif regime == "II":
        p_nd = make_density_regime2_nd(
            tau, xp, delta, n_modes=n_modes,
            cutoff=CUTOFF_II if cutoff is None else cutoff,
            quad_lim=QUAD_LIM_II if quad_lim is None else quad_lim)
        def p(v):
            out = J * p_nd(J * np.asarray(v, dtype=float))
            return out if np.ndim(v) else float(out)
    elif regime == "III":
        p_nd = make_density_regime3_nd(
            tau, xp, delta, b, n_modes=n_modes,
            cutoff=CUTOFF_III if cutoff is None else cutoff,
            quad_lim=QUAD_LIM_III if quad_lim is None else quad_lim)
        def p(v):
            out = J * p_nd(J * np.asarray(v, dtype=float))
            return out if np.ndim(v) else float(out)
    else:
        raise ValueError(regime)
    _CACHE[key] = p
    return p


# ----------------------------------------------------------------------------
# Expectations
# ----------------------------------------------------------------------------
def _quad_limit(regime, quad_lim):
    if quad_lim is not None:
        return quad_lim
    return {"II": QUAD_LIM_II, "III": QUAD_LIM_III}.get(regime, QUAD_LIM)


def expectation(g, regime, T=T, v0=V0, Delta=DELTA, gamma=GAMMA, sigma=S_WN, ell=ELL,
                n_modes=N_MODES, cutoff=None, quad_lim=None, breakpoints=()):
    """E[g(V^0(T)) | V^0(0) = v0] by quadrature of g against the density.

    breakpoints: interior points where g is not smooth (0 is always added, since
    the density has a kink there); the quadrature is split at all of them."""
    p = make_density(regime, T, v0, Delta, gamma, sigma, ell,
                     n_modes=n_modes, cutoff=cutoff, quad_lim=quad_lim)
    ql = _quad_limit(regime, quad_lim)
    pts = sorted(set([-ql, 0.0, ql] + [float(b) for b in breakpoints]))
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        # g is evaluated strictly inside (a, b): one-sided limits at the ends
        val, _ = quad(lambda v: g(v) * p(v), a, b, limit=200, epsabs=1e-13, epsrel=1e-11)
        total += val
    return total


def normalisation_check(regime, **kw):
    """Integral of the density over R (should be 1 to quadrature accuracy)."""
    return expectation(lambda v: 1.0, regime, **kw)


# ----------------------------------------------------------------------------
# The explicit density bound of the manuscript (Proposition on the density of V^0(T))
# ----------------------------------------------------------------------------
def potential(v, gamma, ell, Delta):
    """U(v) = gamma v^2/2 - ell v + Delta |v|  (manuscript (potential))."""
    v = np.asarray(v, dtype=float)
    return 0.5 * gamma * v ** 2 - ell * v + Delta * np.abs(v)


def potential_min(gamma, ell, Delta):
    """U_* = inf U (manuscript (potential))."""
    if gamma == 0.0:
        return 0.0
    return -max(abs(ell) - Delta, 0.0) ** 2 / (2.0 * gamma)


def pbar(v, regime, T=T, v0=V0, Delta=DELTA, sigma=S_WN, gamma=GAMMA, ell=ELL):
    """The explicit bound bar p_T of the manuscript, eq. (pbar):
    p_T <= bar p_T = e^{gamma T/2 + Delta^2 T/(2 s^2) + (U_0(v0)-U_0(v))/s^2} p^(I)_T(v)."""
    g, l = {"I": (0.0, 0.0), "II": (gamma, 0.0), "III": (gamma, ell)}[regime]
    s2 = sigma ** 2
    U0 = lambda x: 0.5 * g * np.asarray(x, float) ** 2 - l * np.asarray(x, float)
    return (np.exp(g * T / 2.0 + Delta ** 2 * T / (2 * s2) + (U0(v0) - U0(v)) / s2)
            * density_pure_dry(v, T, v0, Delta, sigma))


def M_bound(regime, T=T, v0=V0, Delta=DELTA, sigma=S_WN, gamma=GAMMA, ell=ELL):
    """Right-hand side of the manuscript bound (M) on M = ess sup p_T."""
    g, l = {"I": (0.0, 0.0), "II": (gamma, 0.0), "III": (gamma, ell)}[regime]
    s2 = sigma ** 2
    U = float(potential(v0, g, l, Delta))
    return (np.exp(g * T / 2.0 + (U - potential_min(g, l, Delta)) / s2)
            * (1.0 / np.sqrt(2 * np.pi * s2 * T) + Delta / s2 * np.exp(Delta ** 2 * T / (2 * s2))))


if __name__ == "__main__":
    for r in ("I", "II", "III"):
        print(f"regime {r}: normalisation = {normalisation_check(r):.12f}")
    # closed form (pI) against the nondimensional formula [TVJ10, (2.10)]
    vv = np.linspace(-4, 4, 9)
    print("regime I, (pI) vs [TVJ10,(2.10)] max diff:",
          float(np.max(np.abs(density_pure_dry(vv) - make_density("I")(vv)))))
