# Code for numerical experiments:  
# Control variates for dry friction under colored noise: variance rates for non-smooth observables  

# Requirements: 
# Python 3.9 or later, NumPy, SciPy, Matplotlib, mpmath
# python3 -m pip install numpy scipy matplotlib mpmath

All parameters are in `params.py`. Run the scripts from this folder, in this order:

| step | command | time | output |
| 1 | observables.py | < 1 min | `reference_values.json` (exact white-noise values I^0) |

| 2 | validate_reference.py | ~1.5 min | `validation.json` (cross-validation of I^0, density bound) |

| 3 | Production_Run.py | ~3 h on 1 core | `production_results.json` |
| 3b | python model_exponents.py | < 1 min | `model_exponents.json` (model computation of Section 5.4) |

| 4 | Make_tables.py | seconds | `tab_fitted.tex`, `tab_vr.tex`, `tab_ref.tex`, `tab_refine.tex`, `results_macros.tex` |

| 5 | Make_figure1.py, python Make_figure2.py | seconds | `fig01a.pdf/png`, `fig01b.pdf/png` |

## Modules

params.py: model, noise, observables, grids, sample sizes, time-step rule and seeds.

touchette_density.py: white-noise densities, namely the closed form (pI) in regime I and the spectral series of Touchette et al. (2010) in regimes II–III. 
It also holds the bound `bar p_T` and the bound (M).

observables.py: the test observables, the predicted and benchmark exponents, and the reference values.

validate_reference.py: checks of the reference values against the Laplace inversion (Touchette et al. 2012), 
a finite-volume solver, mode convergence, normalisation, and the density bound.

Production_Run.py: the coupled Monte Carlo experiment (stages rate, domain and refine) and the exponent fits.

model_exponents.py: the pre-asymptotic exponents of D_2 when the coupling error is replaced by an independent Gaussian with the measured variance. 
There is no statistical error, and the results are quoted in Section 5.4.

Make_tables.py, Make_figure1.py, Make_figure2.py: output.
