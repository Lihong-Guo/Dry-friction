"""
Production_Run.py -- the coupled Monte Carlo experiment of Section 5.

For every cell (regime, eps), N coupled paths (V^eps_m, V^0_m) are simulated with
the scheme of Section 5.2 (manuscript (OU-discrete), (BE)); both forcing
increments are built from the SAME Gaussian variables, which is the coupling of
Garnier and Mertz (2022).  On the same paths the script records, for every observable f:

  M2      D_2(eps) = E[(f(V^eps(T)) - f(V^0(T)))^2]   (bounds sigma^2_J; Theorem 3.1)
  VarI    sigma^2_{I^eps} = Var f(V^eps(T))             (plain Monte Carlo)
  VarJ    sigma^2_{J^eps} = Var(f(V^eps(T)) - f(V^0(T)))
  VarK    sigma^2_{K^eps} = Var(f(V^eps) - lambda_hat f(V^0))  (empirical optimum)
  lam     lambda_hat, the empirical regression coefficient
  bias    E[f(V^eps(T)) - f(V^0(T))] = I^eps - I^0      (Corollary 3.1)
  Ihat, Jhat, Khat   the three estimators of (estimators), with standard errors
  I0_mc   mean of f(V^0_n): Monte Carlo check of the reference value I^0

and, independently of f, the coupling errors (Proposition B.2, (C1))

  E|Z^eps(T)|^2,   E max_m |Z^eps_m|^2,   Z^eps_m = V^eps_m - V^0_m,

together with a pathwise check of the discrete comparison principle
(Proposition 5.1): max_m |Z_m| <= 2 max_m |h_m|.

Time step: dt = T/n, n = ceil(T / min(DT_CAP, DT_SAFETY eps^2)), so that
dt A / eps^2 <= DT_SAFETY A; the OU contraction factor is 1 - dt A/eps^2.

Stages
  rate    : RATE_GRID, exponents of all quantities are fitted on it
  domain  : DOMAIN_GRID (large eps), range in which the method is useful
  refine  : REFINE_GRID with DT_SAFETY_FINE (time-step refinement test)

Statistics.  Every cell is split into N_BATCHES independent batches with
independent random streams numpy.random.SeedSequence(MASTER, spawn_key=(stage,
regime, round(1e6 eps), batch)).  Means and D_2 use pooled per-path standard
errors; variances, lambda_hat and their ratios use batch-means standard errors.

Usage
  python Production_Run.py                      # all stages, all regimes
  python Production_Run.py --workers 8          # parallel batches
  python Production_Run.py --quick              # smoke test (minutes)
  python Production_Run.py --stages rate --regimes III

Every finished batch is appended to CACHE_FILE (production_batches.jsonl); a rerun
with the same parameters reuses the cached batches, so an interrupted run resumes
where it stopped, and stages run separately can be assembled by a final call with
all stages (--no-cache disables this).  The full design needs about 4.8e11
coupled path-steps (about 3 h on one core at 22 ns per path-step).

All parameters live in params.py.
Requires: numpy, params.py, observables.py (and reference_values.json, which is
regenerated automatically if missing or inconsistent).
"""


import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from params import (  # noqa: E402
    DELTA, SIGMA, OU_A, OU_K, OU_R, C_OU, S_WN, KAPPA0, V0, T, V_F, ALPHAS,
    RATE_GRID, DOMAIN_GRID, REFINE_GRID, REGIMES, RIDX, STAGE_ID,
    FIT_SMALL_EPS_MAX, BIAS_FIT_EPS_MAX, BIAS_RESOLVED, BIAS_MIN_POINTS,
    DT_CAP, DT_SAFETY, DT_SAFETY_FINE, SCHEME,
    N_PATHS_LARGE, N_PATHS_SMALL, N_PATHS_EPS_THRESHOLD, N_PATHS_REFINE,
    N_BATCHES, MASTER, REFERENCE_FILE, OUTPUT_FILE, CACHE_FILE, regime_params,
    GAMMA, ELL,
)
from observables import observable_registry  # noqa: E402


# ----------------------------------------------------------------------------
# Sampling and time-step rules
# ----------------------------------------------------------------------------
def n_paths(eps, stage):
    """Paths per cell."""
    if stage == "refine":
        return N_PATHS_REFINE
    return N_PATHS_LARGE if eps >= N_PATHS_EPS_THRESHOLD else N_PATHS_SMALL


def time_grid(eps, dt_safety):
    """(dt, n) with n = ceil(T / min(DT_CAP, dt_safety eps^2)) and dt = T/n."""
    target = min(DT_CAP, dt_safety * eps ** 2)
    n = int(np.ceil(T / target - 1e-9))
    return T / n, n


def batch_seed(stage, regime, eps, b):
    return np.random.SeedSequence(
        MASTER, spawn_key=(STAGE_ID[stage], RIDX[regime], int(round(eps * 1e6)), b))


# ----------------------------------------------------------------------------
# Coupled simulation (manuscript (OU-discrete) and (BE); Algorithm 1)
# ----------------------------------------------------------------------------
def simulate(regime, eps, dt, n, N, rng, scheme=SCHEME):
    """Simulate N coupled paths on [0, T] with n steps of size dt.

    Returns (V^eps_n, V^0_n, max_m |Z_m|, max_m |h_m|) as arrays of length N, with
    h_m = sum_{j<m} (dN^eps_j - dN^0_j) the forcing difference of Prop. 5.1."""
    gamma, ell = regime_params(regime)
    sq = np.sqrt(dt)
    a_c = 1.0 - dt * OU_A / eps ** 2          # OU contraction factor
    b_c = OU_K * sq / eps                     # OU noise coefficient
    ce = SIGMA * OU_R * dt / eps              # dN^eps_m = ce * eta_m
    c0 = SIGMA * C_OU * OU_K * sq             # dN^0_m   = c0 * zeta_m
    thr = DELTA * dt                          # soft-threshold level Delta dt
    shift = ell * dt
    if scheme == "backward":                  # V <- soft(V + ell dt + dN)/(1 + gamma dt)
        lin, inv = 1.0, 1.0 / (1.0 + gamma * dt)
    elif scheme == "semi-implicit":           # V <- soft((1 - gamma dt) V + ell dt + dN)
        lin, inv = 1.0 - gamma * dt, 1.0
    else:
        raise ValueError(scheme)

    Ve = np.full(N, V0)
    Vz = np.full(N, V0)
    eta = rng.standard_normal(N) * np.sqrt(OU_K ** 2 / (2.0 * OU_A))   # N(0, C)
    h = np.zeros(N)
    hsup = np.zeros(N)
    zsup = np.zeros(N)
    zeta = np.empty(N)
    dz = np.empty(N)
    de = np.empty(N)
    buf = np.empty(N)

    for _ in range(n):
        rng.standard_normal(out=zeta)                     # one Gaussian drives both
        np.multiply(zeta, c0, out=dz)                     # dN^0_m
        np.multiply(eta, ce, out=de)                      # dN^eps_m
        h += de
        h -= dz
        np.abs(h, out=buf)
        np.maximum(hsup, buf, out=hsup)
        # white-noise velocity V^0
        if lin != 1.0:
            Vz *= lin
        Vz += shift
        Vz += dz
        np.clip(Vz, -thr, thr, out=buf)
        Vz -= buf                                         # soft-threshold
        if inv != 1.0:
            Vz *= inv
        # colored-noise velocity V^eps
        if lin != 1.0:
            Ve *= lin
        Ve += shift
        Ve += de
        np.clip(Ve, -thr, thr, out=buf)
        Ve -= buf
        if inv != 1.0:
            Ve *= inv
        # OU factor, with the SAME zeta (this is the coupling)
        eta *= a_c
        np.multiply(zeta, b_c, out=buf)
        eta += buf
        # running maximum of the coupling error
        np.subtract(Ve, Vz, out=buf)
        np.abs(buf, out=buf)
        np.maximum(zsup, buf, out=zsup)
    return Ve, Vz, zsup, hsup


# ----------------------------------------------------------------------------
# One batch: sufficient statistics
# ----------------------------------------------------------------------------
_REG = None


def _registry():
    global _REG
    if _REG is None:
        _REG = observable_registry(V_F)
    return _REG


def run_batch(task):
    """task = (stage, regime, eps, dt_safety, Nb, b, scheme).  Top-level function,
    so that it can be dispatched to worker processes."""
    stage, regime, eps, dt_safety, Nb, b, scheme = task
    reg = _registry()
    dt, n = time_grid(eps, dt_safety)
    rng = np.random.default_rng(batch_seed(stage, regime, eps, b))
    t0 = time.time()
    Ve, Vz, zsup, hsup = simulate(regime, eps, dt, n, Nb, rng, scheme)
    zT = Ve - Vz
    ratio = np.divide(zsup, 2.0 * hsup, out=np.zeros_like(zsup), where=hsup > 0)
    out = {"_coupling": dict(
        n=Nb, s_z2=float(np.sum(zT ** 2)), s_z4=float(np.sum(zT ** 4)),
        s_s2=float(np.sum(zsup ** 2)), s_s4=float(np.sum(zsup ** 4)),
        dcp_max=float(ratio.max()),
        dcp_viol=int(np.sum(zsup > 2.0 * hsup * (1.0 + 1e-10) + 1e-14)),
        dt=dt, n_steps=n, elapsed=time.time() - t0)}
    for k, e in reg.items():
        Fe = np.asarray(e["g"](Ve), dtype=float)
        Fz = np.asarray(e["g"](Vz), dtype=float)
        D = Fe - Fz
        D2 = D * D
        out[k] = dict(n=Nb, s_fe=float(Fe.sum()), s_fe2=float((Fe * Fe).sum()),
                      s_fz=float(Fz.sum()), s_fz2=float((Fz * Fz).sum()),
                      s_fefz=float((Fe * Fz).sum()), s_d=float(D.sum()),
                      s_d2=float(D2.sum()), s_d4=float((D2 * D2).sum()))
    return out


# ----------------------------------------------------------------------------
# Estimates from sufficient statistics
# ----------------------------------------------------------------------------
def _moments(s, I0):
    """Estimates of a set of sums (one batch, or the pooled cell)."""
    n = s["n"]
    mfe, mfz, md = s["s_fe"] / n, s["s_fz"] / n, s["s_d"] / n
    vfe = max(s["s_fe2"] - n * mfe ** 2, 0.0) / (n - 1)
    vfz = max(s["s_fz2"] - n * mfz ** 2, 0.0) / (n - 1)
    cov = (s["s_fefz"] - n * mfe * mfz) / (n - 1)
    vd = max(s["s_d2"] - n * md ** 2, 0.0) / (n - 1)
    M2 = s["s_d2"] / n
    vd2 = max(s["s_d4"] - n * M2 ** 2, 0.0) / (n - 1)
    lam = cov / vfz if vfz > 0 else 0.0
    vk = max(vfe - lam * cov, 0.0)
    return dict(n=n, M2=M2, M2_se=np.sqrt(vd2 / n), VarI=vfe, VarJ=vd, VarK=vk,
                lam=lam, bias=md, bias_se=np.sqrt(vd / n),
                Ihat=mfe, Ihat_se=np.sqrt(vfe / n),
                Jhat=I0 + md, Jhat_se=np.sqrt(vd / n),
                Khat=lam * I0 + mfe - lam * mfz, Khat_se=np.sqrt(vk / n),
                I0_mc=mfz, I0_mc_se=np.sqrt(vfz / n))


def _pool(rows, keys):
    out = dict(rows[0])
    for r in rows[1:]:
        for k in keys:
            out[k] += r[k]
    return out


def _bm_se(vals):
    """Batch-means standard error; batches with an undefined value (e.g. a variance
    ratio in a batch where f(V^eps) is constant) are left out."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")


def summarize_cell(stage, regime, eps, dt_safety, batches, refs):
    reg = _registry()
    cpl = [b["_coupling"] for b in batches]
    sc = _pool(cpl, ("n", "s_z2", "s_z4", "s_s2", "s_s4", "dcp_viol"))
    N = sc["n"]
    zT2 = sc["s_z2"] / N
    zs2 = sc["s_s2"] / N
    out = {"_coupling": dict(
        eps=eps, dt=cpl[0]["dt"], n_steps=cpl[0]["n_steps"], N=N,
        strong_T=zT2, strong_T_se=np.sqrt(max(sc["s_z4"] / N - zT2 ** 2, 0.0) / (N - 1)),
        strong_sup=zs2, strong_sup_se=np.sqrt(max(sc["s_s4"] / N - zs2 ** 2, 0.0) / (N - 1)),
        dcp_max_ratio=max(c["dcp_max"] for c in cpl), dcp_violations=int(sc["dcp_viol"]),
        elapsed=sum(c["elapsed"] for c in cpl), stage=stage, dt_safety=dt_safety)}
    keys = ("n", "s_fe", "s_fe2", "s_fz", "s_fz2", "s_fefz", "s_d", "s_d2", "s_d4")
    for k in reg:
        I0 = refs[regime][k]
        rows = [b[k] for b in batches]
        pooled = _moments(_pool(rows, keys), I0)
        per = [_moments(r, I0) for r in rows]
        d = dict(eps=eps, N=N, I0=I0, **{q: float(pooled[q]) for q in pooled if q != "n"})
        for q in ("VarI", "VarJ", "VarK", "lam"):
            d[q + "_se"] = _bm_se([p[q] for p in per])
        for q, num in (("var_ratio_J", "VarJ"), ("var_ratio_K", "VarK")):
            d[q + "_se"] = _bm_se([p[num] / p["VarI"] if p["VarI"] > 0 else np.nan
                                   for p in per])
        d["rel_se"] = d["M2_se"] / d["M2"] if d["M2"] > 0 else float("nan")
        d["abs_bias"] = abs(d["bias"])
        d["I0_mc_z"] = (d["I0_mc"] - I0) / d["I0_mc_se"] if d["I0_mc_se"] > 0 else 0.0
        vi = d["VarI"]
        d["var_ratio_J"] = d["VarJ"] / vi if vi > 0 else float("nan")
        d["var_ratio_K"] = d["VarK"] / vi if vi > 0 else float("nan")
        d["vrf_K"] = vi / d["VarK"] if d["VarK"] > 0 else float("nan")
        d["reduction_pct"] = 100.0 * (1.0 - d["var_ratio_K"]) if vi > 0 else float("nan")
        out[k] = d
    return out


# ----------------------------------------------------------------------------
# Log-log fits on the rate grid
# ----------------------------------------------------------------------------
def local_slopes(e, y, rel):
    """Local slopes r_i between consecutive grid values and their standard errors
    (e_i^2 + e_{i+1}^2)^{1/2} / |log(eps_{i+1}/eps_i)|  (Appendix C)."""
    if np.any(y <= 0):
        return [], []
    loc = [float(np.log(y[i + 1] / y[i]) / np.log(e[i + 1] / e[i])) for i in range(len(e) - 1)]
    se = [float(np.hypot(rel[i], rel[i + 1]) / abs(np.log(e[i + 1] / e[i])))
          for i in range(len(e) - 1)]
    return loc, se


def wls_slope(e, y, rel):
    """Weighted least-squares slope of log y against log e, weights 1/rel^2.
    The standard error is inflated by sqrt(chi^2/dof) when the latter exceeds 1."""
    if len(e) < 2 or np.any(y <= 0):
        return None, None
    x, z = np.log(e), np.log(y)
    w = 1.0 / np.maximum(np.asarray(rel, dtype=float), 1e-12) ** 2
    xb, zb = np.sum(w * x) / w.sum(), np.sum(w * z) / w.sum()
    sxx = np.sum(w * (x - xb) ** 2)
    slope = float(np.sum(w * (x - xb) * (z - zb)) / sxx)
    se = 1.0 / np.sqrt(sxx)
    if len(e) > 2:
        chi2 = np.sum(w * (z - zb - slope * (x - xb)) ** 2) / (len(e) - 2)
        se *= np.sqrt(max(1.0, chi2))
    return slope, float(se)


def ols_slope(e, y):
    if len(e) < 2 or np.any(y <= 0):
        return None
    return float(np.polyfit(np.log(e), np.log(y), 1)[0])


def fit_regime(rate_r, fit_max=FIT_SMALL_EPS_MAX, bias_max=BIAS_FIT_EPS_MAX):
    """Exponents on the rate grid: local slopes, global OLS slope, and the WLS
    slope over eps <= fit_max ("slope_small"), for D_2 and the variances;
    WLS slope of the resolved bias over eps <= bias_max; coupling-error slopes."""
    reg = _registry()
    e = np.array(sorted((float(x) for x in rate_r), reverse=True))
    cells = [rate_r[f"{x}"] for x in e]
    small = e <= fit_max + 1e-12
    out = {"eps": e.tolist(), "eps_small": e[small].tolist(), "fit_max": fit_max,
           "bias_max": bias_max}

    c = {}
    for q in ("strong_T", "strong_sup"):
        y = np.array([cl["_coupling"][q] for cl in cells])
        rel = np.array([cl["_coupling"][q + "_se"] for cl in cells]) / y
        loc, lse = local_slopes(e, y, rel)
        s, sse = wls_slope(e[small], y[small], rel[small])
        c[q] = dict(slope_all=ols_slope(e, y), slope_small=s, slope_small_se=sse,
                    local=loc, local_se=lse, predicted=2.0)
    out["_coupling"] = c

    for k, entry in reg.items():
        rec = dict(alpha=entry["alpha"], kind=entry["kind"],
                   predicted=entry["predicted"], benchmark=entry["benchmark"],
                   in_table=entry["in_table"])
        for q in ("M2", "VarJ", "VarK"):
            y = np.array([cl[k][q] for cl in cells])
            if q == "M2":
                rel = np.array([cl[k]["M2_se"] for cl in cells]) / np.where(y > 0, y, 1.0)
            else:
                rel = np.array([cl[k][q + "_se"] for cl in cells]) / np.where(y > 0, y, 1.0)
            loc, lse = local_slopes(e, y, rel)
            s, sse = wls_slope(e[small], y[small], rel[small])
            rec.update({f"slope_all_{q}": ols_slope(e, y),
                        f"slope_small_{q}": s, f"slope_small_{q}_se": sse,
                        f"local_{q}": loc, f"local_{q}_se": lse})
        bb = np.array([cl[k]["abs_bias"] for cl in cells])
        bs = np.array([cl[k]["bias_se"] for cl in cells])
        ok = (bb > BIAS_RESOLVED * bs) & (e <= bias_max + 1e-12)
        s, sse = (wls_slope(e[ok], bb[ok], bs[ok] / bb[ok]) if ok.sum() >= BIAS_MIN_POINTS
                  else (None, None))
        rec.update(slope_bias=s, slope_bias_se=sse, n_bias_pts=int(ok.sum()),
                   eps_bias=e[ok].tolist())
        out[k] = rec
    return out


def refine_summary(rate, refine):
    """Time-step refinement test: differences between dt = DT_SAFETY_FINE eps^2
    and the production dt, in units of their combined standard error."""
    reg = _registry()
    res = {}
    for r in refine:
        res[r] = {}
        zmax_M2 = zmax_K = 0.0
        for ek, cf in refine[r].items():
            if r not in rate or ek not in rate[r]:
                continue
            cp = rate[r][ek]
            row = {}
            for k in reg:
                dM2 = cf[k]["M2"] - cp[k]["M2"]
                zM2 = dM2 / np.hypot(cf[k]["M2_se"], cp[k]["M2_se"])
                dK = cf[k]["Khat"] - cp[k]["Khat"]
                zK = dK / np.hypot(cf[k]["Khat_se"], cp[k]["Khat_se"])
                row[k] = dict(dM2=dM2, zM2=float(zM2), dKhat=dK, zKhat=float(zK))
                zmax_M2, zmax_K = max(zmax_M2, abs(zM2)), max(zmax_K, abs(zK))
            a, b = cf["_coupling"], cp["_coupling"]
            row["_coupling"] = dict(
                d_strong_T=a["strong_T"] - b["strong_T"],
                z_strong_T=float((a["strong_T"] - b["strong_T"])
                                 / np.hypot(a["strong_T_se"], b["strong_T_se"])))
            res[r][ek] = row
        res[r]["max_abs_zM2"], res[r]["max_abs_zKhat"] = float(zmax_M2), float(zmax_K)
    return res


# ----------------------------------------------------------------------------
# Reference values
# ----------------------------------------------------------------------------
def load_references():
    """Load reference_values.json (observables.py); regenerate it if it is missing,
    if its parameters differ from params.py, or if an observable is missing."""
    from observables import reference_payload
    path = os.path.join(_HERE, REFERENCE_FILE)
    want = dict(Delta=DELTA, s=float(S_WN), v0=V0, T=T, v_f=V_F)
    regenerate = not os.path.exists(path)
    if not regenerate:
        with open(path) as f:
            payload = json.load(f)
        stored = payload.get("parameters", {})
        regenerate = any(abs(float(stored.get(k, np.nan)) - v) > 1e-12 for k, v in want.items())
        refs = payload.get("reference_values", {})
        regenerate |= any(k not in refs.get(r, {}) for r in REGIMES for k in _registry())
    if regenerate:
        print(f"[info] (re)generating {REFERENCE_FILE} ...")
        payload = reference_payload()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    return payload["reference_values"]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_tasks(stage, grid, regimes, scale, scheme):
    dt_safety = DT_SAFETY_FINE if stage == "refine" else DT_SAFETY
    tasks = []
    for eps in grid:
        for r in regimes:
            N = max(N_BATCHES * 2, int(round(n_paths(eps, stage) * scale)))
            Nb = N // N_BATCHES
            for b in range(N_BATCHES):
                tasks.append((stage, r, eps, dt_safety, Nb, b, scheme))
    return tasks


# ----------------------------------------------------------------------------
# Batch cache: every finished batch is appended to a JSON-lines file, so that an
# interrupted run can be resumed.  A cached batch is reused only if its task and
# the fingerprint of the model/sampling parameters coincide with the current ones.
# ----------------------------------------------------------------------------
def fingerprint():
    return json.dumps(dict(
        master=MASTER, n_batches=N_BATCHES, dt_cap=DT_CAP, T=T, v0=V0, Delta=DELTA,
        gamma=GAMMA, ell=ELL, sigma=float(SIGMA), A=OU_A, K=OU_K, r=OU_R, v_f=V_F,
        observables=list(_registry())), sort_keys=True)


def task_key(t):
    stage, regime, eps, dt_safety, Nb, b, scheme = t
    return json.dumps([stage, regime, float(eps), float(dt_safety), int(Nb), int(b), scheme])


def load_cache(path, fp):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:          # e.g. a line cut by an interruption
                continue
            if rec.get("fp") == fp:
                done[rec["task"]] = rec["result"]
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Coupled Monte Carlo experiment of Section 5.")
    ap.add_argument("--stages", nargs="+", default=["rate", "domain", "refine"],
                    choices=["rate", "domain", "refine"])
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--workers", type=int, default=1, help="worker processes")
    ap.add_argument("--scheme", default=SCHEME, choices=["backward", "semi-implicit"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 1/40 of the paths, eps in {0.4, 0.2, 0.1, 0.05}")
    ap.add_argument("--fit-max", type=float, default=FIT_SMALL_EPS_MAX,
                    help="exponents of D_2 are fitted over eps <= FIT_MAX")
    ap.add_argument("--out", default=None, help="output file")
    ap.add_argument("--cache", default=None, help="batch cache (JSON lines)")
    ap.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    args = ap.parse_args(argv)

    grids = {"rate": RATE_GRID, "domain": DOMAIN_GRID, "refine": REFINE_GRID}
    scale = 1.0
    out_file = args.out or OUTPUT_FILE
    cache_file = args.cache or CACHE_FILE
    if args.quick:
        grids = {"rate": [0.4, 0.2, 0.1, 0.05], "domain": [1.0], "refine": [0.2]}
        scale = 1.0 / 40.0
        args.fit_max = max(args.fit_max, 0.2)
        out_file = args.out or OUTPUT_FILE.replace(".json", "_quick.json")
        cache_file = args.cache or CACHE_FILE.replace(".jsonl", "_quick.jsonl")
    cache_path = os.path.join(_HERE, cache_file)

    refs = load_references()
    fp = fingerprint()
    cache = {} if args.no_cache else load_cache(cache_path, fp)
    t00 = time.time()
    results = {s: {} for s in ("rate", "domain", "refine")}
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    cf = None if args.no_cache else open(cache_path, "a")
    try:
        for stage in args.stages:
            tasks = build_tasks(stage, grids[stage], args.regimes, scale, args.scheme)
            todo = [t for t in tasks if task_key(t) not in cache]
            work = sum(t[4] * time_grid(t[2], t[3])[1] for t in todo)
            print(f"[{stage}] {len(tasks)} batches ({len(tasks) - len(todo)} cached), "
                  f"{work:.2e} path-steps to simulate")
            it = iter(pool.map(run_batch, todo) if pool else map(run_batch, todo))
            groups = {}
            for t in tasks:
                key = task_key(t)
                if key in cache:
                    res = cache[key]
                else:
                    res = next(it)
                    if cf is not None:
                        cf.write(json.dumps(dict(fp=fp, task=key, result=res)) + "\n")
                        cf.flush()
                gk = (t[1], t[2])
                groups.setdefault(gk, []).append(res)
                if len(groups[gk]) == N_BATCHES:
                    r, eps = gk
                    cell = summarize_cell(stage, r, eps, t[3], groups.pop(gk), refs)
                    results[stage].setdefault(r, {})[f"{eps}"] = cell
                    c, i = cell["_coupling"], cell["indicator"]
                    print(f"  [{stage}] eps={eps:<5g} {r:<3} dt={c['dt']:.2e} "
                          f"n={c['n_steps']:>7,} N={c['N']:>7,} "
                          f"E|Z_T|^2={c['strong_T']:.3e} M2(ind)={i['M2']:.3e} "
                          f"(rel.SE {100 * i['rel_se']:4.1f}%) VarK/VarI(ind)="
                          f"{i['var_ratio_K']:.3f} DCP max ratio={c['dcp_max_ratio']:.3f} "
                          f"[{c['elapsed']:6.0f}s cpu, {time.time() - t00:7.0f}s wall]",
                          flush=True)
    finally:
        if pool:
            pool.shutdown()
        if cf is not None:
            cf.close()

    fits = {r: fit_regime(results["rate"][r], args.fit_max) for r in results["rate"]
            if len(results["rate"][r]) >= 2}
    ref_sum = refine_summary(results["rate"], results["refine"])
    cells = [c for st in results.values() for rr in st.values() for c in rr.values()]
    checks = dict(
        dcp_max_ratio=max((c["_coupling"]["dcp_max_ratio"] for c in cells), default=None),
        dcp_violations=sum(c["_coupling"]["dcp_violations"] for c in cells),
        I0_mc_max_abs_z=max((abs(c[k]["I0_mc_z"]) for c in cells for k in _registry()),
                            default=None))
    out = dict(
        rate=results["rate"], domain=results["domain"], refine=results["refine"],
        fits=fits, refine_summary=ref_sum, checks=checks,
        meta=dict(
            regimes=args.regimes, stages=args.stages, scheme=args.scheme,
            grids={k: list(v) for k, v in grids.items()}, quick=args.quick,
            n_batches=N_BATCHES, master_seed=MASTER,
            sigma=float(SIGMA), s=float(S_WN), kappa0=float(KAPPA0), Delta=DELTA,
            ou_A=OU_A, ou_K=OU_K, ou_r=OU_R, v0=V0, T=T, v_f=V_F,
            alphas=[float(a) for a in ALPHAS],
            dt_rule=f"dt = T/ceil(T/min({DT_CAP}, {DT_SAFETY:.6g} eps^2))",
            dt_rule_refine=f"dt = T/ceil(T/min({DT_CAP}, {DT_SAFETY_FINE:.6g} eps^2))",
            n_paths_rule=(f"N={int(N_PATHS_LARGE * scale)} for eps>={N_PATHS_EPS_THRESHOLD}, "
                          f"N={int(N_PATHS_SMALL * scale)} otherwise; "
                          f"refine N={int(N_PATHS_REFINE * scale)}"),
            fit_small_eps_max=args.fit_max, bias_fit_eps_max=BIAS_FIT_EPS_MAX,
            numpy=np.__version__, cache=None if args.no_cache else cache_file,
            cpu_s=sum(c["_coupling"]["elapsed"] for c in cells),
            elapsed_s=time.time() - t00))
    with open(os.path.join(_HERE, out_file), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {out_file}  [{time.time() - t00:.0f}s]")

    for r, F in fits.items():
        c = F["_coupling"]
        print(f"\nregime {r}: exponents over eps <= {args.fit_max} "
              f"(E|Z_T|^2: {c['strong_T']['slope_small']}, "
              f"E max|Z|^2: {c['strong_sup']['slope_small']})")
        for k, f_ in F.items():
            if k.startswith("_") or k in ("eps", "eps_small", "fit_max", "bias_max"):
                continue
            s, se = f_["slope_small_M2"], f_["slope_small_M2_se"]
            txt = "n/a" if s is None else f"{s:+.3f} +- {se:.3f}"
            print(f"   {k:<14} D_2 exponent {txt}   predicted {f_['predicted']:.3f}"
                  f"   benchmark {f_['benchmark']:.3f}")
    print("\nchecks:", checks)


if __name__ == "__main__":
    main()
