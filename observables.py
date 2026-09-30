"""
The family of observables spanning the regularity ladder, and their
exact white-noise reference values E[f(V^0(T))] computed from the
Touchette et al. (2010) densities (through the fixed
`touchette_density.py`, i.e. in PHYSICAL variables).

The ladder.  For an observable f we prove (see the manuscript) two
competing bounds on the coupled second moment

    E[(f(V^eps(T)) - f(V^0(T)))^2] :

  * Holder route   (uses regularity of f):    <= [f]_a^2 C1^a eps^{2a}
  * BV/density route (uses the law of V):     <= V_f^2 C' eps^{2/3}

so the effective exponent is  r(a) = max(2a, 2/3),  with a crossover
at a = 1/3.  Above it the smoothness of the observable controls the
rate; below it the smoothness of the law of V^0(T) takes over, and a
plain indicator (no Holder regularity at all) sits at the floor 2/3.

The family
    f_a(v) = min{ ((|v| - v_f)^+)^a , 1 },     a in (0,1]

The smooth anchor
    f_sm(v) = tanh((v/v_f)^2)

Requires: numpy, touchette_density.py, params.py.
"""
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from params import (
    # physical model parameters
    DELTA, GAMMA, SIGMA, ELL,
    # initial condition and time
    V0, T,
    # numerical parameters
    N_MODES, CUTOFF, QUAD_LIM,
    # observable parameters
    V_F, ALPHAS,
)
from touchette_density import expectation, scaling


# Observable definitions

def make_indicator(vf=V_F):
    """f_ind(v) = 1{|v| >= v_f}.  BV (total variation 2); not Holder."""
    def g(v):
        return (np.abs(v) >= vf) * 1.0
    return g

def make_holder(alpha, vf=V_F):
    """f_a(v) = min{ ((|v| - v_f)^+)^a , 1 }.

    a-Holder with seminorm <= 1, bounded in [0,1], total variation 2.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    def g(v):
        over = np.maximum(np.abs(v) - vf, 0.0)
        return np.minimum(over ** alpha, 1.0)
    return g


def make_smooth(vf=V_F):
    """f_sm(v) = tanh((v / v_f)^2).

    C_b^infinity, bounded in [0,1), the smooth anchor of the ladder.
    """
    def g(v):
        return np.tanh((v / vf) ** 2)
    return g


# Registry
def observable_registry(vf=V_F, alphas=ALPHAS):
    """Return an ordered dict

        name -> dict(g, alpha, kind, breakpoints, label)

    with

      * indicator        f_ind       (BV, no Holder regularity)
      * holder{a}        f_a         (a-Holder, a in (0,1])
      * smooth           f_sm        (C_b^infinity anchor)

    `breakpoints` lists the interior points where the observable is not
    smooth (here +/- v_f), for use in the quadrature.
    """
    reg = {}
    reg["indicator"] = dict(
        g=make_indicator(vf),
        alpha=0.0,
        kind="BV",
        breakpoints=(-vf, vf),
        label=r"$\mathbf{1}_{\{|v|\ge v_f\}}$",
    )
    for a in alphas:
        reg[f"holder{a:.4f}"] = dict(
            g=make_holder(a, vf),
            alpha=a,
            kind="Holder",
            breakpoints=(-vf, vf),
            label=rf"$\alpha={a:.2f}$",
        )
    reg["smooth"] = dict(
        g=make_smooth(vf),
        alpha=None,
        kind="smooth",
        breakpoints=(),
        label=r"$\tanh((v/v_f)^2)$",
    )
    return reg


def predicted_exponent(entry):
    """Theoretical second-moment exponent r = max(2a, 2/3).

    The smooth anchor and the Lipschitz endpoint (a = 1) both give 2;
    the indicator saturates the BV floor 2/3.
    """
    if entry["kind"] == "smooth":
        return 2.0
    if entry["kind"] == "BV":
        return 2.0 / 3.0
    return max(2.0 * entry["alpha"], 2.0 / 3.0)

# Exact white-noise reference values
def reference_values(regimes=("I", "II", "III"),
                     vf=V_F, T=T, v0=V0,
                     Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL,
                     n_modes=N_MODES, cutoff=CUTOFF, quad_lim=QUAD_LIM):
    """E[f(V^0(T)) | V^0(0) = v0] for every observable and regime, by
    quadrature against the exact physical Touchette densities."""
    reg = observable_registry(vf)
    out = {}
    for regime in regimes:
        out[regime] = {}
        for name, entry in reg.items():
            val = expectation(entry["g"], regime,
                              T=T, v0=v0,
                              Delta=Delta, gamma=gamma, sigma=sigma, ell=ell,
                              n_modes=n_modes, cutoff=cutoff,
                              quad_lim=quad_lim,
                              breakpoints=entry["breakpoints"])
            out[regime][name] = float(val)
    return out


def scaling_table(regimes=("I", "II", "III"),
                  Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL, T=T):
    """Diagnostic table: Jacobian J, rescaled time tau(T), and the
    Touchette parameters (delta, b) for every regime."""
    rows = {}
    for regime in regimes:
        J, tau_of, delta, b = scaling(regime, Delta, gamma, sigma, ell)
        rows[regime] = dict(J=J, tau_T=tau_of(T), delta=delta, b=b)
    return rows

# Self-test / reference-table dump
if __name__ == "__main__":
    import json
    import time

    t0 = time.time()

    reg = observable_registry()
    print("Observable ladder (predicted second-moment exponents):")
    for name, e in reg.items():
        a_repr = "None" if e["alpha"] is None else f"{e['alpha']:.6f}"
        print(f"  {name:<16} kind={e['kind']:<7} "
              f"alpha={a_repr:<10} r_pred={predicted_exponent(e):.4f}")
    print()

    print("Scaling diagnostics at current parameters:")
    for r, row in scaling_table().items():
        extra = "" if row["delta"] is None else \
            f"  delta={row['delta']:.6f}  b={row['b']:.6f}"
        print(f"  regime {r}: J={row['J']:.6f}  "
              f"tau(T)={row['tau_T']:.6f}{extra}")
    print()

    vals = reference_values()
    print("Exact white-noise reference values E[f(V^0(T))]:")
    for regime, d in vals.items():
        print(f"  regime {regime}:")
        for name, v in d.items():
            print(f"     {name:<16} {v:.6e}")

    payload = {
        "parameters": dict(Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL,
                           v0=V0, T=T, v_f=V_F, alphas=list(ALPHAS)),
        "scaling": scaling_table(),
        "reference_values": vals,
    }
    with open("reference_values.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved reference_values.json   [{time.time()-t0:.0f}s]")
