"""
Figure 1(a): local slopes r_i of D_2(eps) = E[(f(V^eps(T)) - f(V^0(T)))^2] for the
Lipschitz member f_1 of the severity family (alpha = 1), plotted at the geometric
midpoint of each consecutive pair of the rate grid, for the three regimes, with
standard-error bars (Appendix C).  Dashed guide: the exponent 2 of Theorem 3.1(a).

Produces: fig01a.png, fig01a.pdf
Requires: numpy, matplotlib, production_results.json (Production_Run.py).
Usage:    python Make_figure1.py [production_results.json]
"""
import json
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.4,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 5.0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_REG = {"I": "#2a78d6", "II": "#eb6834", "III": "#1baf7a"}
M_REG = {"I": "o", "II": "s", "III": "^"}
L_REG = {"I": "-", "II": "--", "III": "-."}
LBL_REG = {"I": "regime I (dry)", "II": "regime II (dry+visc.)",
           "III": "regime III (+force)"}
INK, MUTED, GUIDE, GRID = "#2b2b2b", "#8a8a8a", "#b4b4b4", "#e4e4e4"
OBS_KEY = "holder1.0000"          # f_1, alpha = 1
PREDICTED = 2.0                   # Theorem 3.1(a) with alpha = 1


def ax_style(ax):
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.minorticks_off()


def main(results="production_results.json", fn="fig01a.png"):
    with open(results) as f:
        D = json.load(f)
    FIT = D["fits"]
    regimes = [r for r in ("I", "II", "III") if r in FIT]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax_style(ax)
    for j, r in enumerate(regimes):
        eps = np.array(FIT[r]["eps"])
        mid = np.sqrt(eps[:-1] * eps[1:])
        loc = np.array(FIT[r][OBS_KEY]["local_M2"], dtype=float)
        se = np.array(FIT[r][OBS_KEY]["local_M2_se"], dtype=float)
        xs = mid * (1.0 + 0.03 * (j - 1))           # small offset between regimes
        ax.errorbar(xs, loc, yerr=se, color=C_REG[r], marker=M_REG[r], ls=L_REG[r],
                    markeredgecolor="white", markeredgewidth=0.7, elinewidth=0.8,
                    capsize=1.5, zorder=3, label=LBL_REG[r])
    ax.axhline(PREDICTED, color=GUIDE, ls=(0, (6, 3)), lw=1.1, zorder=2)
    ax.annotate(r"Theorem 3.1(a): $2$", xy=(0.03, 0.93), xycoords="axes fraction",
                color=MUTED, fontsize=7.8)

    eps = np.array(FIT[regimes[0]]["eps"])
    mid = np.sqrt(eps[:-1] * eps[1:])
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
             if mid.min() / 1.2 <= t <= mid.max() * 1.2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.5)
    ax.set_xlabel(r"$\varepsilon$  (geometric midpoint of consecutive grid values)")
    ax.set_ylabel(r"local slope of $\mathcal{D}_2$  ($f_1$)")
    ax.set_title(r"(a) Local slopes, $\alpha=1$", loc="left", color=INK)
    ax.legend(loc="lower right", labelcolor=INK, handlelength=2.1,
              borderpad=0.15, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(fn)
    fig.savefig(fn.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {fn} and {fn.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
