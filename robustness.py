"""Robustness checks of the fitted exponents.

usage: python -I robustness.py production_results.json production_batches.jsonl
"""
import json, sys
import numpy as np

RES, BATCH = sys.argv[1], sys.argv[2]
D = json.load(open(RES))
TABLE = ["indicator", "holder0.2500", "holder0.3333", "holder0.5000", "holder0.7500",
         "holder1.0000", "smooth"]
LAB = {"indicator": "f_ind", "holder0.2500": "f_1/4", "holder0.3333": "f_1/3",
       "holder0.5000": "f_1/2", "holder0.7500": "f_3/4", "holder1.0000": "f_1", "smooth": "f_sm"}
REG = ["I", "II", "III"]


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


def ols(e, y):
    return np.polyfit(np.log(e), np.log(y), 1)[0]


def cells(r):
    e = np.array(sorted((float(k) for k in D["rate"][r]), reverse=True))
    return e, [D["rate"][r][f"{x}"] for x in e]


print("=== 1. fitted exponents of D_2 for different windows (WLS; OLS; last local slope)")
windows = [0.05, 0.1, 0.2]
summary = {}
for r in REG:
    e, cl = cells(r)
    print(f"regime {r}")
    for k in TABLE:
        y = np.array([c[k]["M2"] for c in cl]); rel = np.array([c[k]["M2_se"] for c in cl]) / y
        pred, bench = cl[0][k] and D["fits"][r][k]["predicted"], D["fits"][r][k]["benchmark"]
        row = []
        for wmax in windows:
            m = e <= wmax + 1e-12
            s, se = wls(e[m], y[m], rel[m])
            row.append((wmax, s, se, ols(e[m], y[m])))
        loc = D["fits"][r][k]["local_M2"][-1]; locse = D["fits"][r][k]["local_M2_se"][-1]
        summary[(r, k)] = dict(pred=pred, bench=bench, rows=row, last=(loc, locse))
        txt = "  ".join(f"<={w:g}: {s:.3f}({se:.3f}) ols {o:.3f}" for w, s, se, o in row)
        print(f"  {LAB[k]:6s} pred {pred:.3f} bench {bench:.3f}  {txt}  last {loc:.3f}({locse:.3f})")

print("\n=== 2. conclusions per window")
for wmax in windows:
    devA, exc, dev_all = [], [], []
    for r in REG:
        for k in TABLE:
            S = summary[(r, k)]
            s, se = [(s, se) for w, s, se, o in S["rows"] if w == wmax][0]
            dev_all.append(((s - S["pred"]) / se, r, k))
            if k in ("indicator", "holder1.0000", "smooth"):
                devA.append((abs(s - S["pred"]) / se, r, k))
            if S["bench"] < S["pred"] - 1e-9:
                exc.append(((s - S["bench"]) / se, r, k))
    print(f"window eps<={wmax:g}: f_ind,f_1,f_sm max |dev| = {max(devA)[0]:.2f} SE ({max(devA)[1]},{LAB[max(devA)[2]]});"
          f" excess over benchmark min = {min(exc)[0]:.2f} SE ({min(exc)[1]},{LAB[min(exc)[2]]}), "
          f"{sum(z > 3 for z, _, _ in exc)}/{len(exc)} above 3 SE;"
          f" max deficit (pred-fit) over all = {max(-z for z,_,_ in dev_all):.2f} SE")

print("\n=== 3. change of exponent between windows (max over 21 pairs)")
for w2 in (0.1, 0.2):
    d = [abs([s for w, s, se, o in S["rows"] if w == w2][0] - [s for w, s, se, o in S["rows"] if w == 0.05][0])
         for S in summary.values()]
    dz = [abs([s for w, s, se, o in S["rows"] if w == w2][0] - [s for w, s, se, o in S["rows"] if w == 0.05][0])
          / [se for w, s, se, o in S["rows"] if w == 0.05][0] for S in summary.values()]
    print(f"  eps<=0.05 vs eps<={w2:g}: max |change| = {max(d):.3f} ({max(dz):.2f} SE of the eps<=0.05 fit)")
d = [abs(S["last"][0] - [s for w, s, se, o in S["rows"] if w == 0.05][0]) for S in summary.values()]
dz = [abs(S["last"][0] - [s for w, s, se, o in S["rows"] if w == 0.05][0]) / S["last"][1] for S in summary.values()]
print(f"  eps<=0.05 vs last local slope: max |change| = {max(d):.3f} ({max(dz):.2f} SE of the local slope)")
d = [abs([o for w, s, se, o in S["rows"] if w == 0.05][0] - [s for w, s, se, o in S["rows"] if w == 0.05][0]) for S in summary.values()]
print(f"  WLS vs OLS on eps<=0.05: max |change| = {max(d):.3f}")

print("\n=== 4. jackknife over the 12 batches (eps<=0.05 window)")
B = 12
batches = {}
for line in open(BATCH):
    rec = json.loads(line)
    stage, regime, eps, dts, Nb, b, scheme = json.loads(rec["task"])
    if stage != "rate":
        continue
    batches.setdefault((regime, eps), {})[b] = rec["result"]
for r in REG:
    e, cl = cells(r)
    m = e <= 0.05 + 1e-12
    es = e[m]
    for k in TABLE:
        full = []
        reps = []
        for b_out in range(B):
            y = []
            for x in es:
                rows = [batches[(r, x)][b][k] for b in range(B) if b != b_out]
                s_d2 = sum(rw["s_d2"] for rw in rows); n = sum(rw["n"] for rw in rows)
                y.append(s_d2 / n)
            reps.append(ols(es, np.array(y)))
        y = []
        for x in es:
            rows = [batches[(r, x)][b][k] for b in range(B)]
            y.append(sum(rw["s_d2"] for rw in rows) / sum(rw["n"] for rw in rows))
        full = ols(es, np.array(y))
        reps = np.array(reps)
        jk_se = np.sqrt((B - 1) / B * np.sum((reps - reps.mean()) ** 2))
        S = summary[(r, k)]
        s, se = [(s, se) for w, s, se, o in S["rows"] if w == 0.05][0]
        print(f"  {r:3s} {LAB[k]:6s} OLS {full:.3f}  jackknife SE {jk_se:.3f}   quoted WLS SE {se:.3f}  ratio {jk_se/se:.2f}")

print("\n=== 5. numbers for the discussion")
for eps in (0.05, 0.02):
    L = np.log(1 / eps)
    print(f"  eps={eps}: local exponent of vartheta = 1 - 1/(2(1+L)) = {1 - 1/(2*(1+L)):.3f}; of eps^2(1+L): {2 - 1/(1+L):.3f}; of eps^2(1+L)^2: {2-2/(1+L):.3f}")
print("  window ratio 0.05/0.02 =", 0.05 / 0.02)
for r in REG:
    for eps in (0.2, 0.1, 0.05, 0.02):
        c = D["rate"][r][f"{eps}"]["indicator"]
        work = 2 * c["var_ratio_K"]
        print(f"  {r:3s} eps={eps:<5g} indicator: bias/SE = {abs(c['bias'])/c['bias_se']:5.1f}  |bias|/Khat = {100*abs(c['bias'])/c['Khat']:4.1f}%  "
              f"2 sK^2/sI^2 = {work:.3f}  VarJ/VarI = {c['var_ratio_J']:.3f}  lam = {c['lam']:.3f}  rel SE of Khat = {100*c['Khat_se']/c['Khat']:.2f}%")
