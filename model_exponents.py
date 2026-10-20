"""
model_exponents.py -- pre-asymptotic exponents of D_2 in a model computation
(Section 5.4, "Exponents of D_2").

Model: V = V^0(T) has the exact white-noise density p_T of the regime, and the
coupling error is replaced by Z = eps * s_Z * xi with xi ~ N(0, 1) independent of V,
where s_Z^2 = E|Z^eps(T)|^2 / eps^2 is taken from production_results.json (smallest
eps of the fitting window; default values if the file is missing).  Then

    D_2^model(eps) = E[(f(V + Z) - f(V))^2]

is computed by Gauss--Hermite quadrature in xi and adaptive quadrature in v (split
at the non-smooth points of f and their shifts; p_T is replaced by a cubic-spline
interpolant on a grid of step 0.002), on the grid values eps <= FIT_SMALL_EPS_MAX
used for the fitted exponents.  The least-squares slope of log D_2^model against log eps
shows the pre-asymptotic deviation from the exponent of Theorem 3.1 that is to be
expected on this window, without statistical error.

Output: model_exponents.json (read by Make_tables.py) and a printed table.
Usage:  python model_exponents.py [production_results.json]    (a few minutes)
Requires: numpy, scipy, params.py, touchette_density.py, observables.py.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from params import RATE_GRID, FIT_SMALL_EPS_MAX, REGIMES, OUTPUT_FILE
from touchette_density import make_density
from observables import observable_registry

MODEL_FILE = "model_exponents.json"
S_Z2_DEFAULT = {"I": 1.3, "II": 1.0, "III": 1.0}     # E|Z(T)|^2 / eps^2 (approximate)
N_HERMITE = 40
V_MAX = 12.0
H_GRID = 0.002          # grid of the spline interpolant of p_T (error ~1e-11)


def tabulated_density(regime):
    """Cubic-spline interpolant of p_T, built separately on v <= 0 and v >= 0 (p_T has
    a kink at 0), so that the quadratures below do not re-evaluate the spectral series."""
    p = make_density(regime)
    n = int(round(V_MAX / H_GRID))
    vl = np.linspace(-V_MAX, 0.0, n + 1)
    vr = np.linspace(0.0, V_MAX, n + 1)
    left, right = CubicSpline(vl, p(vl)), CubicSpline(vr, p(vr))
    return lambda v: float(left(v)) if v <= 0.0 else float(right(v))


def s_z2_from_results(path):
    if not os.path.exists(path):
        return dict(S_Z2_DEFAULT), "default"
    with open(path) as f:
        D = json.load(f)
    out = {}
    for r in REGIMES:
        cells = D.get("rate", {}).get(r, {})
        eps = sorted(float(e) for e in cells if float(e) <= FIT_SMALL_EPS_MAX + 1e-12)
        if not eps:
            out[r] = S_Z2_DEFAULT[r]
            continue
        c = cells[f"{eps[0]}"]["_coupling"]
        out[r] = c["strong_T"] / eps[0] ** 2
    return out, os.path.basename(path)


def model_D2(f, p, eps, s, bps, xs, ws):
    tot = 0.0
    for x, w in zip(xs, ws):
        h = eps * s * x
        pts = sorted(set([-V_MAX, 0.0, V_MAX] + list(bps) + [b - h for b in bps]))
        val = sum(quad(lambda v: (f(v + h) - f(v)) ** 2 * p(v), a, b, limit=200,
                       epsabs=1e-15, epsrel=1e-10)[0] for a, b in zip(pts[:-1], pts[1:]))
        tot += w * val
    return tot


def main(results=OUTPUT_FILE):
    reg = observable_registry()
    s2, src = s_z2_from_results(results)
    eps = np.array(sorted([e for e in RATE_GRID if e <= FIT_SMALL_EPS_MAX + 1e-12], reverse=True))
    xs, ws = np.polynomial.hermite_e.hermegauss(N_HERMITE)
    ws = ws / ws.sum()
    out = {"eps": eps.tolist(), "s_Z2": s2, "s_Z2_source": src, "exponents": {}}
    for r in REGIMES:
        p = tabulated_density(r)
        out["exponents"][r] = {}
        print(f"regime {r} (s_Z^2 = {s2[r]:.3f}):")
        for k, e in reg.items():
            if not e["in_table"]:
                continue
            y = np.array([model_D2(e["g"], p, ee, np.sqrt(s2[r]), e["breakpoints"], xs, ws)
                          for ee in eps])
            slope = float(np.polyfit(np.log(eps), np.log(y), 1)[0])
            out["exponents"][r][k] = dict(slope=slope, predicted=e["predicted"],
                                          deficit=e["predicted"] - slope)
            print(f"   {k:14s} model exponent {slope:.3f}   predicted {e['predicted']:.3f}"
                  f"   deficit {e['predicted'] - slope:+.3f}", flush=True)
    with open(MODEL_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {MODEL_FILE}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
