"""
Central place for the concrete numerical parameters
Physical model:
    dV = (-gamma V + ell) dt - Delta sign(V) dt + sigma dW
Fokker--Planck equation:
    d_t p = d_v[ (gamma v + Delta sign(v) - ell) p ] + (sigma^2 / 2) d_vv p
Touchette et al. (2010), Eq. (1.4), is the same with
    Gamma = sigma^2,    a = ell,    Delta = Delta,    gamma = gamma.
Regimes
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  I   pure dry friction                     (gamma = 0, ell = 0)
  II  dry + viscous friction                (gamma > 0, ell = 0)
  III dry + viscous + constant external force (gamma > 0, ell in R)
All quantities below are in physical units.

Colored noise
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The scalar Ornstein--Uhlenbeck factor solves
    d(eta^eps) = -(A / eps^2) eta^eps dt + (K / eps) dW,
with A = OU_A, K = OU_K.  With K = A the limit xi^0 has unit intensity
so that sigma xi^0 dt = sigma dW for the white-noise process V^0.
"""

#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
# Physical parameters:
DELTA = 1.0        # dry-friction threshold
GAMMA = 1.0        # viscous damping coefficient
SIGMA = np.sqrt(2) # noise amplitude  (Gamma = sigma^2 = 2)
ELL   = 0.5        # constant external force  (regime III only)

# Colored-noise (Ornstein--Uhlenbeck) parameters: 
OU_A = 1.0         # A in d(eta^eps) = -(A/eps^2) eta^eps dt + (K/eps) dW
OU_K = 1.0         # K ; equal to OU_A for unit-intensity white-noise limit

# Initial condition and time:
V0 = 0.5           # initial velocity
X0 = 0.0           # initial position (unused by the velocity code)
T  = 1.0           # final time

# Numerical parameters shared by the density / expectation routines: 
N_MODES  = 40          # number of spectral modes (regimes II and III)
STEP      = 0.05       # scan step for the eigenvalue root finders
LAM_START = 1e-6       # first trial eigenvalue
QUAD_LIM = 50.0        # generic semi-infinite quadrature cut-off
CUTOFF   = 45.0        # generic parabolic-cylinder tail truncation
CUTOFF_II    = 35.0    # regime II: parabolic-cylinder tail truncation
QUAD_LIM_II  = 50.0    # regime II: density quadrature cut-off
CUTOFF_III   = 45.0    # regime III: parabolic-cylinder tail truncation
QUAD_LIM_III = 55.0    # regime III: density quadrature cut-off
 
# Observables' extra parameters:
V_F = 2          # threshold used by indicator / Holder observables

# Holder exponents tested in the ladder.  
# The crossover of the theorem is at alpha = 1/3;  we straddle it.
ALPHAS = (0.1, 0.2, 0.3, 1.0 / 3.0, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0)


# Convenience bundle for calls:  make_density(regime, T, V0, **PHYS)
PHYS = dict(Delta=DELTA, gamma=GAMMA, sigma=SIGMA, ell=ELL)
PHYS0 = dict(v0=V0, T=T)                      # initial condition + time
NUM   = dict(n_modes=N_MODES, cutoff=CUTOFF, quad_lim=QUAD_LIM)

# Production experiment design:
# eps in RATE_GRID
RATE_GRID   = [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02]  
DOMAIN_GRID = [1.0, 0.8, 0.6, 0.5]
REGIMES = ["I", "II", "III"]
RIDX    = {"I": 0, "II": 1, "III": 2}

N_BATCHES = 6          # batches per cell (used for the standard errors)

# Time-step rule:  dt = min(DT_CAP, DT_SAFETY * eps^2)
DT_CAP     = 1e-3
DT_SAFETY  = 0.02

N_PATHS_LARGE = 240_000    # for eps >= N_PATHS_EPS_THRESHOLD
N_PATHS_SMALL = 480_000    # for eps <  N_PATHS_EPS_THRESHOLD
N_PATHS_EPS_THRESHOLD = 0.05

# Master seed; per-cell seeds are derived deterministically from it.
MASTER = 42424242

# File names:
REFERENCE_FILE = "reference_values.json"
OUTPUT_FILE    = "production_results.json"