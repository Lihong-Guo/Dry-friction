"""
Per cell, on the SAME simulated paths, we record:
  strong    E|V^eps(T) - V^0(T)|^2                    (coupling error)
  M2        E[(f(V^eps(T)) - f(V^0(T)))^2]            (upper bound on VarJ)
  VarI      Var(f(V^eps)) = sigma^2_{I^eps}
  VarJ      Var(f(V^eps) - f(V^0)) = sigma^2_{J^eps}  (what Theorem bounds)
  VarK      Var(f(V^eps) - lambda_hat f(V^0)) = sigma^2_{K^eps}
  bias      E[f(V^eps(T)) - f(V^0(T))]
  lambda_hat, Ihat, Khat

Time step dt = min(DT_CAP, DT_SAFETY * eps^2).  The OU contraction factor
is then 1 - dt*A/eps^2, i.e., 0.98 whenever the second branch is active.

All parameters live in `params.py`.
Requires: numpy, params.py, observables.py, reference_values.json.
"""

#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
import os
import sys
import time
import numpy as np

# Make the imports work regardless of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from params import (  # noqa: E402
    # physical parameters
    DELTA, GAMMA, SIGMA, ELL,
    # OU parameters
    OU_A, OU_K,
    # initial condition / time
    V0, T,
    # observable family
    V_F, ALPHAS,
    # experiment design
    RATE_GRID, DOMAIN_GRID, REGIMES, RIDX,
    N_BATCHES, DT_CAP, DT_SAFETY, MASTER,
    N_PATHS_LARGE, N_PATHS_SMALL, N_PATHS_EPS_THRESHOLD,
    REFERENCE_FILE, OUTPUT_FILE,
)
from observables import observable_registry  # noqa: E402


# Sampling and time-step rules

def n_paths(eps):
    """Sample size per cell.  Smaller eps -> larger N to keep the
    log-log fit well resolved (M2 ~ eps^beta)."""
    return N_PATHS_LARGE if eps >= N_PATHS_EPS_THRESHOLD else N_PATHS_SMALL

def time_step(eps):
    """dt = min(DT_CAP, DT_SAFETY * eps^2)."""
    return min(DT_CAP, DT_SAFETY * eps ** 2)

def regime_params(regime):
    """(gamma, ell, Delta) for the requested regime."""
    return {
        "I":   (0.0,   0.0, DELTA),
        "II":  (GAMMA, 0.0, DELTA),
        "III": (GAMMA, ELL, DELTA),
    }[regime]


# Coupled simulation: (V^eps, V^0) share a single Wiener path

def simulate(regime, eps, dt, N, rng):
    """Coupled Euler--Maruyama for (V^eps, V^0) with a common Wiener path Z.  
    The multivalued term uses the exact backward-Euler resolvent of 
    d(phi): v = soft_threshold(v_pred, dt*Delta)."""
    gamma, ell, Delta = regime_params(regime)
    n = int(round(T / dt))

    Ve = np.full(N, V0)
    Vz = np.full(N, V0)

    # OU factor eta^eps, initialized in its stationary law N(0, C(A,K)).
    # For scalar A, K:  C(A,K) = K^2 / (2 A).
    var_eta = OU_K ** 2 / (2.0 * OU_A)
    eta = rng.normal(0.0, np.sqrt(var_eta), size=N)

    # AR(1) coefficients for eta^eps with step dt:
    #   eta_{n+1} = a_c eta_n + b_c Z_n
    a_c = 1.0 - dt * OU_A / eps ** 2
    b_c = (OU_K / eps) * np.sqrt(dt)

    for _ in range(n):
        Z = rng.normal(0.0, 1.0, size=N)

        # White-noise process V^0: sigma dW  (K = A => unit intensity)
        Vt = Vz + dt * (-gamma * Vz + ell) + SIGMA * np.sqrt(dt) * Z
        Vz = np.sign(Vt) * np.maximum(np.abs(Vt) - dt * Delta, 0.0)

        # Colored-noise process V^eps: sigma (eta^eps / eps) dt
        Vt = Ve + dt * (-gamma * Ve + ell) + SIGMA * dt * eta / eps
        Ve = np.sign(Vt) * np.maximum(np.abs(Vt) - dt * Delta, 0.0)

        # Advance the OU factor with the SAME Z (this is the coupling).
        eta = a_c * eta + b_c * Z
    return Ve, Vz, n, a_c


# One (regime, eps) cell
def cell(regime, eps, reg, refs):
    dt, N = time_step(eps), n_paths(eps)
    Nb = N // N_BATCHES

    per = {k: [] for k in reg}
    strong = []

    t0 = time.time()
    steps = 0
    a_c = None

    for b in range(N_BATCHES):
        seed = (MASTER
                + RIDX[regime] * 1_000_003
                + int(round(eps * 1e4)) * 101
                + b * 7919)
        rng = np.random.default_rng(seed)

        Ve, Vz, steps, a_c = simulate(regime, eps, dt, Nb, rng)
        strong.append(float(np.mean((Ve - Vz) ** 2)))

        for k, e in reg.items():
            Fe = np.asarray(e["g"](Ve), float)
            Fz = np.asarray(e["g"](Vz), float)
            D = Fe - Fz

            den = float(np.sum((Fz - Fz.mean()) ** 2))
            lam = (float(np.sum((Fe - Fe.mean()) * (Fz - Fz.mean()))) / den
                   if den > 0 else 0.0)

            per[k].append(dict(
                M2=float(np.mean(D ** 2)),
                VarJ=float(D.var()),                      # = sigma^2_{J^eps}
                VarK=float((Fe - lam * Fz).var()),        # = sigma^2_{K^eps}
                VarI=float(Fe.var()),                     # = sigma^2_{I^eps}
                bias=float(np.mean(D)),
                lam=lam,
                Ihat=float(Fe.mean()),
                Khat=float(lam * refs[regime][k]
                           + (Fe - lam * Fz).mean()),
            ))

    elapsed = time.time() - t0

    def agg(rows, key):
        a = np.array([r[key] for r in rows])
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    sa = np.array(strong)
    out = {"_strong": dict(m2=float(sa.mean()),
                           se=float(sa.std(ddof=1) / np.sqrt(len(sa))))}

    for k in reg:
        rows = per[k]
        M2, M2se = agg(rows, "M2")
        vj, vjse = agg(rows, "VarJ")
        vk, vkse = agg(rows, "VarK")
        vi, vise = agg(rows, "VarI")
        bi, bise = agg(rows, "bias")
        lm, _    = agg(rows, "lam")
        ih, ihse = agg(rows, "Ihat")
        kh, khse = agg(rows, "Khat")

        out[k] = dict(
            eps=eps, dt=dt, n_steps=steps, a_coef=a_c, N=N,
            # upper bound and the two theorem quantities
            M2=M2, M2_se=M2se,
            rel_se=M2se / M2 if M2 > 0 else np.nan,
            VarJ=vj, VarJ_se=vjse,
            VarK=vk, VarK_se=vkse,
            VarI=vi, VarI_se=vise,
            # bias
            bias=bi, bias_se=bise, abs_bias=abs(bi),
            # diagnostics
            lam=lm,
            reduction=100 * (1 - vk / vi) if vi > 0 else np.nan,
            ratio=vi / vk if vk > 0 else np.nan,
            # point estimates of I^eps
            Ihat=ih, Ihat_se=ihse,
            Khat=kh, Khat_se=khse,
            # exact white-noise reference E[f(V^0)]
            P0=refs[regime][k],
            elapsed=elapsed,
        )
    return out


# Reference values (exact E[f(V^0(T))] from Touchette densities)

def load_references():
    """Load reference_values.json produced by observables.py.

    If missing, regenerate it and warn the user.  Also sanity-check that
    the stored physical parameters match the current ones.
    """
    path = os.path.join(_HERE, REFERENCE_FILE)
    if not os.path.exists(path):
        print(f"[warn] {REFERENCE_FILE} not found -> regenerating ...")
        from observables import reference_values, scaling_table
        payload = {
            "parameters": dict(Delta=DELTA, gamma=GAMMA,
                               sigma=float(SIGMA), ell=ELL,
                               v0=V0, T=T, v_f=V_F,
                               alphas=list(ALPHAS)),
            "scaling": scaling_table(),
            "reference_values": reference_values(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    with open(path) as f:
        payload = json.load(f)

    refs = payload.get("reference_values", payload)
    stored = payload.get("parameters", {})
    for key, val in (("Delta", DELTA), ("gamma", GAMMA),
                     ("sigma", float(SIGMA)), ("ell", ELL),
                     ("v0", V0), ("T", T), ("v_f", V_F)):
        if key in stored and abs(float(stored[key]) - float(val)) > 1e-12:
            raise RuntimeError(
                f"[params mismatch] {REFERENCE_FILE} has {key}={stored[key]}, "
                f"current params.py has {key}={val}.  "
                "Regenerate reference_values.json (delete it and rerun)."
            )
    return refs


# Log-log fits of the rate

def fit_one(e, y):
    """Return (global slope, local slopes) of log y vs log e."""
    if y.min() <= 0:
        return None, []
    slope = float(np.polyfit(np.log(e), np.log(y), 1)[0])
    local = [float(np.log(y[i + 1] / y[i]) / np.log(e[i + 1] / e[i]))
             for i in range(len(e) - 1)]
    return slope, local


def fit_regime(rate_r, reg):
    """Fits on the RATE grid only."""
    e = np.array(sorted((float(x) for x in rate_r), reverse=True))
    out = {}

    sm = np.array([rate_r[f"{x}"]["_strong"]["m2"] for x in e])
    s, loc = fit_one(e, sm)
    out["_strong"] = dict(slope=s, local=loc)

    for k in reg:
        M2 = np.array([rate_r[f"{x}"][k]["M2"]    for x in e])
        vj = np.array([rate_r[f"{x}"][k]["VarJ"]  for x in e])
        vk = np.array([rate_r[f"{x}"][k]["VarK"]  for x in e])
        bb = np.array([rate_r[f"{x}"][k]["abs_bias"] for x in e])
        bs = np.array([rate_r[f"{x}"][k]["bias_se"]  for x in e])

        ok = bb > 3.0 * bs                          # bias resolved over MC noise
        bias_slope = (float(np.polyfit(np.log(e[ok]), np.log(bb[ok]), 1)[0])
                      if ok.sum() >= 3 else None)

        sM2, lM2 = fit_one(e, M2)
        sJ,  lJ  = fit_one(e, vj)
        sK,  lK  = fit_one(e, vk)

        a = reg[k]["alpha"]
        predicted = (2.0 if a is None else max(2.0 * a, 2.0 / 3.0))
        out[k] = dict(
            slope_M2=sM2, local_M2=lM2,
            slope_VarJ=sJ, local_VarJ=lJ,
            slope_VarK=sK, local_VarK=lK,
            slope_bias=bias_slope, n_bias_pts=int(ok.sum()),
            alpha=a, predicted=predicted,
        )
    return out


# Driver

def main():
    reg = observable_registry(V_F)
    refs = load_references()

    rate, domain = {}, {}
    t00 = time.time()

    for grid, store, tag in ((RATE_GRID, rate, "rate"),
                             (DOMAIN_GRID, domain, "domain")):
        for eps in grid:
            for r in REGIMES:
                o = cell(r, eps, reg, refs)
                store.setdefault(r, {})[f"{eps}"] = o
                i = o["indicator"]
                print(f"  [{tag}] eps={eps:<5.2f} {r:<3} dt={i['dt']:.1e} "
                      f"steps={i['n_steps']:>7,} N={i['N']:>7,}  "
                      f"M2={i['M2']:.4e} (relSE {100*i['rel_se']:4.1f}%)  "
                      f"VarJ={i['VarJ']:.4e}  VarK={i['VarK']:.4e}  "
                      f"|Z|^2={o['_strong']['m2']:.4e}  "
                      f"[{i['elapsed']:5.0f}s] [{time.time()-t00:6.0f}s]")

    fits = {r: fit_regime(rate[r], reg) for r in REGIMES}

    out = dict(
        rate=rate, domain=domain, fits=fits,
        meta=dict(
            rate_grid=RATE_GRID, domain_grid=DOMAIN_GRID, regimes=REGIMES,
            n_batches=N_BATCHES, master_seed=MASTER,
            sigma=float(SIGMA), delta=DELTA, ell=ELL, gamma=GAMMA,
            ou_A=OU_A, ou_K=OU_K, v0=V0, T=T, vf=V_F,
            dt_rule=f"min({DT_CAP}, {DT_SAFETY}*eps^2)",
            n_paths_rule=(f"N={N_PATHS_LARGE} for eps>={N_PATHS_EPS_THRESHOLD}, "
                          f"N={N_PATHS_SMALL} otherwise"),
            elapsed_s=time.time() - t00,
        ),
    )
    with open(os.path.join(_HERE, OUTPUT_FILE), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}  [{time.time()-t00:.0f}s]")

    print("\nFitted exponents on the rate grid [0.02, 0.40]:")
    for r in REGIMES:
        print(f"  regime {r}:  strong {fits[r]['_strong']['slope']:.2f}")
        for k in reg:
            f = fits[r][k]
            print(f"    {k:<16}  M2:{f['slope_M2']:+.2f}  "
                  f"VarJ:{f['slope_VarJ']:+.2f}  "
                  f"VarK:{f['slope_VarK']:+.2f}  "
                  f"pred={f['predicted']:.2f}")

if __name__ == "__main__":
    main()