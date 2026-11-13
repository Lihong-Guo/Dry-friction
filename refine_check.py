"""
Effect of the time step on the fitted exponents: 
production step (eps^2/50) vs fine step (eps^2/400) on the cells eps in {0.2, 0.1, 0.05}.   
usage: python -I refine_check.py production_results.json
"""


import json, sys, numpy as np
D = json.load(open(sys.argv[1]))
TABLE = ["indicator", "holder0.2500", "holder0.3333", "holder0.5000", "holder0.7500", "holder1.0000", "smooth"]
LAB = {"indicator": "f_ind", "holder0.2500": "f_1/4", "holder0.3333": "f_1/3", "holder0.5000": "f_1/2",
       "holder0.7500": "f_3/4", "holder1.0000": "f_1", "smooth": "f_sm"}
E = [0.2, 0.1, 0.05]

def wls(e, y, rel):
    x, z = np.log(e), np.log(y); w = 1 / np.maximum(rel, 1e-12) ** 2
    xb, zb = np.sum(w * x) / w.sum(), np.sum(w * z) / w.sum(); sxx = np.sum(w * (x - xb) ** 2)
    s = np.sum(w * (x - xb) * (z - zb)) / sxx; se = 1 / np.sqrt(sxx)
    if len(e) > 2:
        chi2 = np.sum(w * (z - zb - s * (x - xb)) ** 2) / (len(e) - 2); se *= np.sqrt(max(1, chi2))
    return s, se

worst = 0; worst_lab = None; maxd = 0
for r in ("I", "II", "III"):
    print(f"regime {r}: exponent over eps in {E}: production step | fine step | difference (SE units)")
    for k in TABLE + ["_coupling"]:
        rows = []
        for stage in ("rate", "refine"):
            cells = [D[stage][r][f"{e}"] for e in E]
            if k == "_coupling":
                y = np.array([c["_coupling"]["strong_T"] for c in cells]); se = np.array([c["_coupling"]["strong_T_se"] for c in cells])
            else:
                y = np.array([c[k]["M2"] for c in cells]); se = np.array([c[k]["M2_se"] for c in cells])
            rows.append(wls(np.array(E), y, se / y))
        (s1, e1), (s2, e2) = rows
        z = (s2 - s1) / np.hypot(e1, e2)
        print(f"   {LAB.get(k, 'E|Z_T|^2'):9s} {s1:.3f}({e1:.3f}) | {s2:.3f}({e2:.3f}) | {s2-s1:+.3f} ({z:+.2f} SE)")
        if abs(z) > abs(worst): worst, worst_lab = z, (r, k)
        maxd = max(maxd, abs(s2 - s1))
print("largest |difference| =", round(maxd, 3), " largest |z| =", round(abs(worst), 2), worst_lab)
# per-cell relative change of E|Z_T|^2 and of M2 for the indicator
for r in ("I", "II", "III"):
    for e in E:
        a, b = D["refine"][r][f"{e}"], D["rate"][r][f"{e}"]
        print(f"  {r:3s} eps={e}: dE|Z_T|^2 = {100*(a['_coupling']['strong_T']/b['_coupling']['strong_T']-1):+.2f}%  "
              f"z={ (a['_coupling']['strong_T']-b['_coupling']['strong_T'])/np.hypot(a['_coupling']['strong_T_se'], b['_coupling']['strong_T_se']):+.2f}   "
              f"M2(ind) z={(a['indicator']['M2']-b['indicator']['M2'])/np.hypot(a['indicator']['M2_se'], b['indicator']['M2_se']):+.2f}  "
              f"Khat(ind) z={(a['indicator']['Khat']-b['indicator']['Khat'])/np.hypot(a['indicator']['Khat_se'], b['indicator']['Khat_se']):+.2f}")
# OU stationary variance factor
for e in E:
    dt1 = min(1e-3, e**2/50); dt2 = min(1e-3, e**2/400)
    f1 = 1/(1-dt1/(2*e**2)); f2 = 1/(1-dt2/(2*e**2))
    print(f"  eps={e}: OU variance factor production {f1:.5f}, fine {f2:.5f}, ratio-1 = {100*(f2/f1-1):+.2f}%")
# slope of log vartheta over the fit window
ee = np.array([0.05, 0.04, 0.03, 0.02]); th = ee*np.sqrt(1+np.log(1/ee))
print("OLS slope of log vartheta over window:", np.polyfit(np.log(ee), np.log(th), 1)[0], " local:", [round(1-1/(2*(1+np.log(1/e))),3) for e in (0.05,0.02)])
for r, (s, se) in {"I": (1.085, 0.043), "II": (0.834, 0.096), "III": (1.023, 0.097)}.items():
    print(f"  f_ind regime {r}: (fit-0.89)/SE = {(s-0.89)/se:+.2f}, (fit-1)/SE = {(s-1)/se:+.2f}")
