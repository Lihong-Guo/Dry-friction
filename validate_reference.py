"""
validate_reference.py -- cross-validation of the white-noise reference values
(Appendix C, "Reference values"; Table "tab:ref").

Checks performed (results written to validation.json):

 1. Normalisation of the densities of touchette_density.py (regimes I-III), and
    the closed form (pI) of the manuscript against [TVJ10, Eq. (2.10)].
 2. Convergence in the number of spectral modes (regimes II-III): reference
    values with 20, 30 and 40 modes.
 3. Spectral densities [TVJ10, Eqs. (3.12), (4.7)] against the numerical
    inversion (Talbot's method) of the closed-form Laplace transform of the
    propagator of Touchette, Prellberg & Just (2012) [TPJ12, Eqs. (7)-(9),
    (A.21), (A.23)]  (requires mpmath; skipped otherwise).
 4. A finite-volume solution of the Fokker--Planck equation (FP) of the
    manuscript (Scharfetter--Gummel fluxes, implicit Euler in time with
    Richardson extrapolation): densities and reference values of all observables.
 5. The explicit density bound of the manuscript: p_T <= bar p_T on a grid, and
    sup p_T against the bound (M).
The Monte Carlo cross-check (white-noise means of f(V^0_n) along the coupled
paths) is part of Production_Run.py (field "I0_mc_z" of every cell).

Usage:   python validate_reference.py            (about one minute)
Requires: numpy, scipy, mpmath, params.py, touchette_density.py, observables.py
"""


import json
import time
import numpy as np
from scipy.linalg import solve_banded

from params import (DELTA, GAMMA, S_WN, ELL, V0, T, V_F, N_MODES, REGIMES,
                    VALIDATION_FILE, regime_params)
from touchette_density import (make_density, normalisation_check, density_pure_dry,
                               pbar, M_bound, scaling)
from observables import observable_registry, reference_values

X_TEST = (-2.0, -0.5, 0.25, 1.0, 2.5)        # physical test points of checks 3-4
_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy >= 2.0 / < 2.0


# ----------------------------------------------------------------------------
# 3. Laplace-transform solution of Touchette, Prellberg & Just (2012)
# ----------------------------------------------------------------------------
def tpj12_density(v_points, regime, T=T, v0=V0, Delta=DELTA, gamma=GAMMA,
                  sigma=S_WN, ell=ELL, dps=30):
    """Physical density at v_points by Talbot inversion of [TPJ12, (7)-(9)].

    Nondimensional variables as in [TPJ12]: x = (2 gamma/Gamma)^{1/2} v,
    t -> gamma t, delta = Delta (2/(gamma Gamma))^{1/2}, f = ell (2/(gamma Gamma))^{1/2}.
    The formulas hold for x' > 0; x' < 0 is reduced to it by (x, x', f) -> (-x, -x', -f)."""
    import mpmath as mp
    mp.mp.dps = dps
    if regime == "I":
        raise ValueError("[TPJ12] requires gamma > 0 (regimes II-III)")
    g, l = (gamma, 0.0) if regime == "II" else (gamma, ell)
    J, tau_of, d, f = scaling(regime, Delta, g, sigma, l)
    tau, xp = tau_of(T), J * v0
    sgn = 1.0
    if xp < 0:
        sgn, xp, f = -1.0, -xp, -f
    if xp == 0:
        raise ValueError("v0 = 0 is not covered by [TPJ12, (7)-(9)]")
    d, f, xp = mp.mpf(d), mp.mpf(f), mp.mpf(xp)

    def D(nu, z):
        return mp.pcfd(nu, z)

    def g_less(s):                                    # [TPJ12, (A.21)]
        num = mp.e ** (-(d - f) ** 2 / 4) * (D(-s, -d + f) / D(-s, d - f)
                                              + D(-s - 1, -d + f) / D(-s - 1, d - f))
        den = mp.e ** (-(d + f) ** 2 / 4) * (D(-s, d + f) / D(-s, d - f)
                                              + D(-s - 1, d + f) / D(-s - 1, d - f))
        return num / den

    def g_great(s):                                   # [TPJ12, (A.23)]
        num = D(-s - 1, -d + f) / D(-s - 1, d + f) - D(-s, -d + f) / D(-s, d + f)
        den = D(-s, d - f) / D(-s, d + f) + D(-s - 1, d - f) / D(-s - 1, d + f)
        return num / den

    def Ptilde(x, s):                                 # [TPJ12, (7)-(9)]
        c = mp.gamma(s) / mp.sqrt(2 * mp.pi)
        A = mp.e ** ((xp + d - f) ** 2 / 4) * D(-s, xp + d - f)
        if x < 0:
            return g_less(s) * c * A * mp.e ** (-(x - d - f) ** 2 / 4) * D(-s, -x + d + f)
        common = g_great(s) * c * A * mp.e ** (-(x + d - f) ** 2 / 4) * D(-s, x + d - f)
        if x < xp:
            return common + c * A * mp.e ** (-(x + d - f) ** 2 / 4) * D(-s, -x - d + f)
        return common + (c * mp.e ** ((xp + d - f) ** 2 / 4) * D(-s, -xp - d + f)
                         * mp.e ** (-(x + d - f) ** 2 / 4) * D(-s, x + d - f))

    out = []
    for v in v_points:
        x = mp.mpf(sgn * J * v)
        val = mp.invertlaplace(lambda s: Ptilde(x, s), tau, method="talbot")
        out.append(float(J * val))
    return np.array(out)


# ----------------------------------------------------------------------------
# 4. Finite-volume solution of (FP)
# ----------------------------------------------------------------------------
def _bernoulli(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    out[small] = 1.0 - 0.5 * x[small]
    out[~small] = x[~small] / np.expm1(x[~small])
    return out


def fv_density(regime, T=T, v0=V0, Delta=DELTA, sigma=S_WN, L=20.0, h=0.0025,
               dt=4e-4):
    """Cell values of the density of V^0(T) on the grid v_i = i h, |v_i| <= L.

    Fluxes J_{i+1/2} = (D/h)[B(-P) p_i - B(P) p_{i+1}], P = a h / D, where
    a = -gamma v - Delta sgn(v) + ell is the drift at the cell interface and
    B(x) = x/(e^x - 1) (Scharfetter--Gummel).  v = 0 is a cell centre, so the
    drift is continuous at every interface.  Zero-flux boundaries at +-L.
    Implicit Euler with steps dt and dt/2, combined by Richardson extrapolation.
    Initial condition: unit mass in the cell(s) containing v0 (linear split)."""
    gamma, ell = regime_params(regime)
    n_half = int(round(L / h))
    v = h * np.arange(-n_half, n_half + 1)
    M = v.size
    Dc = 0.5 * sigma ** 2
    vf = v[:-1] + 0.5 * h
    a = -gamma * vf - Delta * np.sign(vf) + ell
    P = a * h / Dc
    Bm, Bp = _bernoulli(-P), _bernoulli(P)
    c = Dc / h ** 2
    lo = np.zeros(M); di = np.zeros(M); up = np.zeros(M)
    di[:-1] -= c * Bm;  up[1:] += c * Bp
    di[1:] -= c * Bp;   lo[:-1] += c * Bm

    def run(step):
        ab = np.zeros((3, M))
        ab[0, 1:] = -step * up[1:]
        ab[1] = 1.0 - step * di
        ab[2, :-1] = -step * lo[:-1]
        p = np.zeros(M)
        k = (v0 + L) / h
        k0 = int(np.floor(k)); w = k - k0
        p[k0] += (1.0 - w) / h
        if w > 0:
            p[k0 + 1] += w / h
        for _ in range(int(round(T / step))):
            p = solve_banded((1, 1), ab, p)
        return p

    return v, 2.0 * run(0.5 * dt) - run(dt)


def fv_expectation(g, v, p, breakpoints):
    """int g p dv, with p the cubic-spline interpolant of the finite-volume values
    (built separately on v <= 0 and v >= 0, where the density has a kink), by
    adaptive quadrature split at 0 and at the breakpoints of g.  The adaptive
    rule resolves the cusps (|v| - v_f)^alpha of the severity family."""
    from scipy.interpolate import CubicSpline
    from scipy.integrate import quad
    i0 = int(np.argmin(np.abs(v)))
    left = CubicSpline(v[:i0 + 1], p[:i0 + 1])
    right = CubicSpline(v[i0:], p[i0:])
    pf = lambda x: float(left(x)) if x <= 0.0 else float(right(x))
    pts = sorted(set([v[0], 0.0, v[-1]] + [float(b) for b in breakpoints]))
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val, _ = quad(lambda x: float(np.asarray(g(x))) * pf(x), a, b,
                      limit=400, epsabs=1e-13, epsrel=1e-11)
        total += val
    return total


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(out_file=VALIDATION_FILE, skip_tpj12=False):
    t00 = time.time()
    reg = observable_registry(V_F)
    res = {"x_test": list(X_TEST)}

    # 1. normalisation and closed form
    res["normalisation"] = {r: float(normalisation_check(r)) for r in REGIMES}
    vv = np.linspace(-6.0, 6.0, 241)
    res["regime_I_closed_form_max_abs_diff"] = float(
        np.max(np.abs(density_pure_dry(vv) - make_density("I")(vv))))
    print("[1] normalisation:", res["normalisation"],
          " (pI) vs [TVJ10,(2.10)]:", res["regime_I_closed_form_max_abs_diff"])

    # 2. number of modes
    ref = reference_values()
    res["modes"] = {}
    for r in ("II", "III"):
        rows = {}
        for nm in (20, 30, 40):
            rows[nm] = reference_values(regimes=(r,), n_modes=nm)[r]
        res["modes"][r] = dict(
            n_modes=[20, 30, N_MODES],
            max_change_20_30=max(abs(rows[30][k] - rows[20][k]) for k in reg),
            max_change_30_40=max(abs(rows[40][k] - rows[30][k]) for k in reg))
    print("[2] modes:", res["modes"])

    # 3. TPJ12 Laplace inversion
    res["tpj12"] = {}
    if not skip_tpj12:
        try:
            import mpmath  # noqa: F401
            for r in ("II", "III"):
                p = make_density(r)
                spec = p(np.array(X_TEST))
                lap = tpj12_density(X_TEST, r)
                res["tpj12"][r] = dict(spectral=spec.tolist(), laplace=lap.tolist(),
                                       max_abs_diff=float(np.max(np.abs(spec - lap))))
                print(f"[3] TPJ12 regime {r}: max |spectral - Laplace| = "
                      f"{res['tpj12'][r]['max_abs_diff']:.2e}")
        except ImportError:
            print("[3] mpmath not available: TPJ12 check skipped")

    # 4. finite volumes
    res["fv"] = {}
    for r in REGIMES:
        v, pfv = fv_density(r)
        pex = make_density(r)
        dens_diff = float(np.max(np.abs(np.interp(X_TEST, v, pfv) - pex(np.array(X_TEST)))))
        i0_fv = {k: float(fv_expectation(e["g"], v, pfv, e["breakpoints"]))
                 for k, e in reg.items()}
        i0_diff = max(abs(i0_fv[k] - ref[r][k]) for k in reg)
        res["fv"][r] = dict(mass=float(_trapezoid(pfv, v)),
                            max_abs_density_diff=dens_diff,
                            reference_values=i0_fv, max_abs_I0_diff=float(i0_diff))
        print(f"[4] FV regime {r}: mass={res['fv'][r]['mass']:.10f}  "
              f"max density diff={dens_diff:.2e}  max |I0_spec - I0_FV|={i0_diff:.2e}")

    # 5. density bound
    res["density_bound"] = {}
    vv = np.linspace(-10.0, 10.0, 4001)
    for r in REGIMES:
        pv = make_density(r)(vv)
        pb = pbar(vv, r)
        res["density_bound"][r] = dict(sup_p=float(pv.max()), sup_pbar=float(pb.max()),
                                       M_bound=float(M_bound(r)),
                                       min_pbar_minus_p=float((pb - pv).min()))
        print(f"[5] regime {r}: sup p_T={pv.max():.4f}  sup bar p_T={pb.max():.4f}  "
              f"bound (M)={M_bound(r):.4f}  min(bar p_T - p_T)={(pb - pv).min():.2e}")

    res["reference_values"] = ref
    res["elapsed_s"] = time.time() - t00
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {out_file}  [{res['elapsed_s']:.0f}s]")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--skip-tpj12", action="store_true", help="skip the mpmath check")
    args = ap.parse_args()
    main(skip_tpj12=args.skip_tpj12)
