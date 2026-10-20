"""
All numerical parameters of Section 5 ("Numerical experiments").

Model (Assumption A1, scalar velocity inclusion)
-----------------------------------------------
    dV + d(phi)(V) dt  contains  (-gamma V + ell) dt + dN,     phi(v) = Delta |v|,

  regime I   : pure dry friction                       (gamma = 0, ell = 0)
  regime II  : dry + viscous friction                  (gamma > 0, ell = 0)
  regime III : dry + viscous friction + constant force (gamma > 0, ell != 0)

Colored noise (Assumption A2, scalar case d = d' = 1)
-----------------------------------------------------
    d eta^eps = -(A / eps^2) eta^eps dt + (K / eps) dW,   eta^eps(0) ~ N(0, K^2/(2A)),
    colored forcing   dN^eps = sigma * r * eta^eps / eps * dt,
    white-noise limit dN^0   = sigma * c K dW,   c = r / A,
so that the white-noise velocity V^0 has noise intensity  s = sigma |r K / A|.

White-noise Fokker--Planck equation
--------------------------------------------------------
    d_t p = d_v[ (gamma v + Delta sgn(v) - ell) p ] + (s^2 / 2) d_vv p,
which is Touchette, Van der Straeten & Just (2010), Eq. (1.4), with
    Gamma = s^2,   a = ell.

With the test values below, A = K = r = 1, hence s = sigma = sqrt(2), and all the
scalings of Touchette et al. (2010, 2012) reduce to the identity.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Physical parameters (Section 5.1)
# ----------------------------------------------------------------------------
DELTA = 1.0          # dry-friction threshold Delta
GAMMA = 1.0          # viscous damping gamma (regimes II and III)
SIGMA = np.sqrt(2.0) # forcing amplitude sigma
ELL   = 0.5          # constant force ell (regime III; |ell| < Delta: stick phase)

# Colored noise (scalar Ornstein--Uhlenbeck factor)
OU_A = 1.0           # A
OU_K = 1.0           # K
OU_R = 1.0           # r   (xi^eps = r eta^eps / eps)

# Derived constants of Section 2.2
C_OU = OU_R / OU_A               # c = A^{-T} r
S_WN = SIGMA * abs(C_OU * OU_K)  # white-noise intensity s (= sigma here)
KAPPA0 = SIGMA * abs(C_OU)       # kappa_0 = sigma |c|

# Initial condition and horizon
V0 = 0.5             # initial velocity v_0
X0 = 0.0             # initial position (not used)
T  = 1.0             # final time

# ----------------------------------------------------------------------------
# White-noise density (touchette_density.py) -- numerical parameters
# ----------------------------------------------------------------------------
N_MODES   = 40       # spectral modes in regimes II and III (convergence: validate_reference.py)
STEP      = 0.05     # scan step of the eigenvalue root finders
LAM_START = 1e-6     # first trial eigenvalue
QUAD_LIM  = 50.0     # generic semi-infinite quadrature cut-off (physical units)
CUTOFF    = 45.0     # generic tail truncation (unused: CUTOFF_II, CUTOFF_III apply)
CUTOFF_II    = 35.0  # regime II : parabolic-cylinder tail truncation
QUAD_LIM_II  = 50.0  # regime II : density quadrature cut-off
CUTOFF_III   = 45.0  # regime III: parabolic-cylinder tail truncation
QUAD_LIM_III = 55.0  # regime III: density quadrature cut-off

# ----------------------------------------------------------------------------
# Observables (Example 2.1)
# ----------------------------------------------------------------------------
V_F = 2.0            # threshold v_f

# Holder exponents of the severity family f_alpha used for Fig. 1(b).
# Theorem 3.1(d) predicts the exponent min{2, 1 + 2 alpha} (kink at alpha = 1/2);
# the second-moment benchmark max{2 alpha, 2/3} has its kink at alpha = 1/3.
ALPHAS = (0.1, 0.2, 0.25, 0.3, 1.0 / 3.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)

# The severity exponents reported in the tables (the "seven test observables" are
# f_ind, f_alpha for these alpha, and f_sm).
ALPHAS_TABLE = (0.25, 1.0 / 3.0, 0.5, 0.75, 1.0)

# ----------------------------------------------------------------------------
# Production experiment design (Section 5.3)
# ----------------------------------------------------------------------------
# rate grid: exponents are fitted on it
RATE_GRID   = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10,
               0.05, 0.04, 0.03, 0.02]
# domain grid: large eps, to delimit the range in which the method is useful
# (0.5 is already in the rate grid and is not recomputed)
DOMAIN_GRID = [1.0, 0.8, 0.6]
# time-step refinement test: dt = eps^2/400 against the production dt = eps^2/50
REFINE_GRID = [0.20, 0.10, 0.05]

REGIMES = ["I", "II", "III"]
RIDX    = {"I": 0, "II": 1, "III": 2}
STAGE_ID = {"rate": 0, "domain": 1, "refine": 2}

# least-squares exponent of log D_2 against log eps over eps <= FIT_SMALL_EPS_MAX
FIT_SMALL_EPS_MAX = 0.05
# exponent of the bias |E D^eps| (Corollary 3.1): least squares over the grid values
# eps <= BIAS_FIT_EPS_MAX at which |bias| > BIAS_RESOLVED * SE(bias); at least
# BIAS_MIN_POINTS such values are required
BIAS_FIT_EPS_MAX = 0.2
BIAS_RESOLVED = 3.0
BIAS_MIN_POINTS = 3

# eps values reported in the variance-reduction table (Table 2)
VR_TABLE_EPS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)

# Time step: dt = T / n,  n = ceil(T / min(DT_CAP, DT_SAFETY * eps^2)),
# so that dt * A / eps^2 <= DT_SAFETY * A  (= 1/50 here) for every eps.
DT_CAP         = 1e-3
DT_SAFETY      = 1.0 / 50.0     # production runs
DT_SAFETY_FINE = 1.0 / 400.0    # refinement test

# Velocity update (manuscript eq. (BE)):
#   "backward"      : V <- soft_{Delta dt}(V + ell dt + dN) / (1 + gamma dt)   [default]
#   "semi-implicit" : V <- soft_{Delta dt}((1 - gamma dt) V + ell dt + dN)
SCHEME = "backward"

# Sample sizes (paths per cell), split into N_BATCHES independent batches
N_PATHS_LARGE = 240_000         # eps >= N_PATHS_EPS_THRESHOLD
N_PATHS_SMALL = 480_000         # eps <  N_PATHS_EPS_THRESHOLD
N_PATHS_EPS_THRESHOLD = 0.05
N_PATHS_REFINE = 240_000        # refinement test
N_BATCHES = 12

# Master seed. Every (stage, regime, eps, batch) gets its own independent stream
# numpy.random.SeedSequence(MASTER, spawn_key=(stage, regime, round(1e6 eps), batch)).
MASTER = 42424242

# ----------------------------------------------------------------------------
# Convenience bundles and file names
# ----------------------------------------------------------------------------
# The density routines take the WHITE-NOISE intensity s as `sigma` (Gamma = s^2).
PHYS  = dict(Delta=DELTA, gamma=GAMMA, sigma=S_WN, ell=ELL)
PHYS0 = dict(v0=V0, T=T)
NUM   = dict(n_modes=N_MODES)

REFERENCE_FILE  = "reference_values.json"   # observables.py
VALIDATION_FILE = "validation.json"         # validate_reference.py
OUTPUT_FILE     = "production_results.json" # Production_Run.py
CACHE_FILE      = "production_batches.jsonl"  # Production_Run.py: finished batches
                                              # (an interrupted run resumes from it)


def regime_params(regime):
    """(gamma, ell) of the requested regime (Assumption A1)."""
    return {"I": (0.0, 0.0), "II": (GAMMA, 0.0), "III": (GAMMA, ELL)}[regime]
