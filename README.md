# Code for: Control variates for dry-friction systems under colored noise

This repository contains the numerical code for the paper:

## Contents

- `params.py` — all physical and numerical parameters.
- `touchette_density.py` — exact white-noise densities.
- `observables.py` — observable family and reference values.
- `production_run.py` — production Monte Carlo run.
- `make_figure1.py` — produces fig01a.png/pdf.
- `make_figure2.py` — produces fig01b.png/pdf.

# production_results.json and reference_values.json are included so that
# the figures can be reproduced without rerunning the full Monte Carlo study.


python observables.py          # optional: regenerates reference_values.json
python production_run.py       # regenerates production_results.json
python make_figure1.py
python make_figure2.py
