"""
Model exponents (independent-Gaussian coupling error) on several fitting windows,
compared with the fitted exponents.  
usage: python -I model_windows.py <numerics dir>
"""


import json, sys, os
import numpy as np
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
import model_exponents as me
from params import RATE_GRID
from observables import observable_registry

D = json.load(open("production_results.json"))
reg = observable_registry()
s2, _ = me.s_z2_from_results("production_results.json")
xs, ws = np.polynomial.hermite_e.hermegauss(me.N_HERMITE); ws = ws / ws.sum()
TABLE = ["indicator", "holder0.2500", "holder0.3333", "holder0.5000", "holder0.7500", "holder1.0000", "smooth"]


def wls(e, y, rel):
    x, z = np.log(e), np.log(y)
    w = 1.0 / np.maximum(rel, 1e-12) ** 2
    xb, zb = np.sum(w * x) / w.sum(), np.sum(w * z) / w.sum()
    sxx = np.sum(w * (x - xb) ** 2)
    s = np.sum(w * (x - xb) * (z - zb)) / sxx
    se = 1 / np.sqrt(sxx)
    if len(e) > 2:
        chi2 = np.sum(w * (z - zb - s * (x - xb)) ** 2) / (len(e) - 2)
        se *= np.sqrt(max(1, chi2))
    return s, se


out = {}
for r in ("I", "II", "III"):
    p = me.tabulated_density(r)
    e_all = np.array(sorted((float(k) for k in D["rate"][r]), reverse=True))
    cl = [D["rate"][r][f"{x}"] for x in e_all]
    for wmax in (0.05, 0.1, 0.2):
        eps = e_all[e_all <= wmax + 1e-12]
        for k in TABLE:
            e = reg[k]
            y = np.array([me.model_D2(e["g"], p, ee, np.sqrt(s2[r]), e["breakpoints"], xs, ws) for ee in eps])
            slope = float(np.polyfit(np.log(eps), np.log(y), 1)[0])
            m = e_all <= wmax + 1e-12
            yy = np.array([c[k]["M2"] for c in cl])[m]; rel = (np.array([c[k]["M2_se"] for c in cl]) / np.array([c[k]["M2"] for c in cl]))[m]
            fs, fse = wls(eps, yy, rel)
            out[(r, wmax, k)] = (slope, fs, fse)
            print(f"{r:3s} eps<={wmax:<4g} {k:13s} model {slope:.3f}  fitted {fs:.3f}({fse:.3f})  z={(fs-slope)/fse:+.2f}", flush=True)
for wmax in (0.05, 0.1, 0.2):
    zs = [abs((v[1] - v[0]) / v[2]) for key, v in out.items() if key[1] == wmax]
    print(f"window eps<={wmax:g}: max |fitted-model|/SE = {max(zs):.2f}, mean = {np.mean(zs):.2f}")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_windows.json"), "w"), indent=1)
