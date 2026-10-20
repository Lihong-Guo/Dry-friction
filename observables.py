"""
observables.py -- the test observables of Example 2.1, the exponents predicted by
Theorem 3.1, and the exact white-noise reference values I^0 = E[f(V^0(T))].

Observables (threshold v_f > 0)
  f_ind(v)   = 1{|v| >= v_f}                       BV_J, TV = 2, not Holder
  f_a(v)     = min{ ((|v| - v_f)_+)^a , 1 },  a in (0,1]
               a-Holder ([f_a]_a <= 1), TV = 2, cusp class K^a_{{+-v_f}} with
               [f_a]_{a,S} = a; kinks (Lipschitz) at |v| = v_f + 1
  f_sm(v)    = tanh((v / v_f)^2)                   Lipschitz, C_b^infinity, TV = 2

Exponents of D_2(eps) = E[(f(V^eps(T)) - f(V^0(T)))^2] (Theorem 3.1, up to
logarithmic factors in (b)-(d)):
  f_ind              : 1                   (part (b))
  f_a, 0 < a < 1     : min{2, 1 + 2a}      (part (d); f_a is also in C^{0,a} cap BV_J,
                                            part (c) gives 1 + a, which is weaker)
  f_1, f_sm          : 2                   (part (a) with alpha = 1)
Second-moment benchmark (mean-square bound (C1) + Avikainen's inequality):
  f_ind : 2/3,   f_a : max{2a, 2/3},   smooth/Lipschitz : 2.

Reference values are computed from the white-noise densities of
touchette_density.py (closed form in regime I, spectral series of Touchette et
al. (2010) in regimes II-III), by quadrature split at 0 and at the non-smooth
points of the observable (+-v_f, and +-(v_f + 1) for the severity family).

Requires: numpy, scipy, params.py, touchette_density.py.
"""
import json
import time
import numpy as np

from params import (
    DELTA, GAMMA, S_WN, ELL, V0, T, N_MODES,
    V_F, ALPHAS, ALPHAS_TABLE, REFERENCE_FILE,
)
from touchette_density import expectation, scaling


# ----------------------------------------------------------------------------
# Observable definitions
# ----------------------------------------------------------------------------
def make_indicator(vf=V_F):
    """f_ind(v) = 1{|v| >= v_f}; satisfies condition (J)."""
    def g(v):
        return (np.abs(v) >= vf) * 1.0
    return g


def make_holder(alpha, vf=V_F):
    """f_alpha(v) = min{((|v| - v_f)_+)^alpha, 1}."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    def g(v):
        over = np.maximum(np.abs(v) - vf, 0.0)
        return np.minimum(over ** alpha, 1.0)
    return g


def make_smooth(vf=V_F):
    """f_sm(v) = tanh((v / v_f)^2)."""
    def g(v):
        return np.tanh((np.asarray(v, dtype=float) / vf) ** 2)
    return g


def holder_key(alpha):
    """Registry key of f_alpha, e.g. 'holder0.5000'."""
    return f"holder{alpha:.4f}"


# ----------------------------------------------------------------------------
# Predicted exponents
# ----------------------------------------------------------------------------
def predicted_exponent(entry):
    """Exponent of D_2 predicted by Theorem 3.1 (up to logarithms)."""
    if entry["kind"] == "smooth":
        return 2.0
    if entry["kind"] == "BV":
        return 1.0
    return min(2.0, 1.0 + 2.0 * entry["alpha"])


def benchmark_exponent(entry):
    """Exponent given by (C1) and Avikainen's inequality only: max{2a, 2/3}."""
    if entry["kind"] == "smooth":
        return 2.0
    if entry["kind"] == "BV":
        return 2.0 / 3.0
    return max(2.0 * entry["alpha"], 2.0 / 3.0)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def observable_registry(vf=V_F, alphas=ALPHAS):
    """Ordered dict  name -> dict(g, alpha, kind, breakpoints, label, in_table,
    predicted, benchmark).  `in_table` marks the seven test observables reported
    in the tables (f_ind, f_alpha for alpha in ALPHAS_TABLE, f_sm)."""
    table_keys = {holder_key(a) for a in ALPHAS_TABLE}
    reg = {}
    reg["indicator"] = dict(
        g=make_indicator(vf), alpha=0.0, kind="BV",
        breakpoints=(-vf, vf),
        label=r"$f_{\rm ind}$", in_table=True)
    for a in alphas:
        k = holder_key(a)
        reg[k] = dict(
            g=make_holder(a, vf), alpha=float(a), kind="Holder",
            breakpoints=(-vf - 1.0, -vf, vf, vf + 1.0),
            label=rf"$f_{{{_alpha_tex(a)}}}$", in_table=(k in table_keys))
    reg["smooth"] = dict(
        g=make_smooth(vf), alpha=None, kind="smooth",
        breakpoints=(), label=r"$f_{\rm sm}$", in_table=True)
    for e in reg.values():
        e["predicted"] = predicted_exponent(e)
        e["benchmark"] = benchmark_exponent(e)
    return reg


def _alpha_tex(a):
    """TeX string of a Holder exponent: 1/3 and 1/4, 3/4, 1/2 as fractions."""
    for num, den in ((1, 3), (1, 4), (1, 2), (3, 4)):
        if abs(a - num / den) < 1e-9:
            return rf"{num}/{den}"
    return f"{a:g}"


# ----------------------------------------------------------------------------
# Exact white-noise reference values
# ----------------------------------------------------------------------------
def reference_values(regimes=("I", "II", "III"), vf=V_F, T=T, v0=V0,
                     Delta=DELTA, gamma=GAMMA, sigma=S_WN, ell=ELL,
                     n_modes=N_MODES, cutoff=None, quad_lim=None):
    """E[f(V^0(T)) | V^0(0) = v0] for every observable and regime.

    `sigma` is the white-noise intensity s; cutoff/quad_lim = None selects the
    regime-specific values of params.py."""
    reg = observable_registry(vf)
    out = {}
    for regime in regimes:
        out[regime] = {}
        for name, entry in reg.items():
            out[regime][name] = float(expectation(
                entry["g"], regime, T=T, v0=v0, Delta=Delta, gamma=gamma,
                sigma=sigma, ell=ell, n_modes=n_modes, cutoff=cutoff,
                quad_lim=quad_lim, breakpoints=entry["breakpoints"]))
    return out


def scaling_table(regimes=("I", "II", "III"), Delta=DELTA, gamma=GAMMA,
                  sigma=S_WN, ell=ELL, T=T):
    """Jacobian J, rescaled time tau(T) and the parameters (delta, b) of
    Touchette et al. (2010) for every regime."""
    rows = {}
    for regime in regimes:
        J, tau_of, delta, b = scaling(regime, Delta, gamma, sigma, ell)
        rows[regime] = dict(J=float(J), tau_T=float(tau_of(T)),
                            delta=None if delta is None else float(delta),
                            b=None if b is None else float(b))
    return rows


def reference_payload():
    """Everything written to REFERENCE_FILE (also used by Production_Run.py)."""
    return {
        "parameters": dict(Delta=DELTA, gamma=GAMMA, s=float(S_WN), ell=ELL,
                           v0=V0, T=T, v_f=V_F, n_modes=N_MODES,
                           alphas=[float(a) for a in ALPHAS]),
        "scaling": scaling_table(),
        "reference_values": reference_values(),
    }


if __name__ == "__main__":
    t0 = time.time()
    reg = observable_registry()
    print("Observables (predicted exponent of D_2 by Theorem 3.1 / benchmark):")
    for name, e in reg.items():
        a = "  -   " if e["alpha"] is None else f"{e['alpha']:.4f}"
        print(f"  {name:<14} kind={e['kind']:<7} alpha={a}  "
              f"pred={e['predicted']:.3f}  bench={e['benchmark']:.3f}"
              f"{'  [table]' if e['in_table'] else ''}")
    payload = reference_payload()
    print("\nScaling diagnostics:", json.dumps(payload["scaling"]))
    print("\nWhite-noise reference values E[f(V^0(T))]:")
    for regime, d in payload["reference_values"].items():
        print(f"  regime {regime}:")
        for name, v in d.items():
            print(f"     {name:<14} {v:.10f}")
    with open(REFERENCE_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {REFERENCE_FILE}   [{time.time() - t0:.0f}s]")
