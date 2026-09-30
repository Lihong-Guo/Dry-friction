"""
Local slopes of M2 for the alpha = 1 (Lipschitz) member of the severity family, 
at the geometric midpoint of each consecutive eps-pair on the rate grid, 
for all three regimes.  The dashed guide is the predicted value 2.

Produces: fig01a.png, fig01a.pdf

Requires: numpy, matplotlib, production_results.json.
"""
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import json
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
           "III": "regime III (+bias)"}
INK, MUTED, GUIDE, GRID = "#2b2b2b", "#8a8a8a", "#b4b4b4", "#e4e4e4"
REG = ["I", "II", "III"]


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
    D = json.load(open(results))
    FIT = D["fits"]
    EPS = sorted((float(x) for x in D["rate"]["I"]), reverse=True)
    mid = np.sqrt(np.array(EPS[:-1]) * np.array(EPS[1:]))

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax_style(ax)

    for r in REG:
        loc = FIT[r]["holder1.0000"]["local_M2"]
        ax.plot(mid, loc, color=C_REG[r], marker=M_REG[r], ls=L_REG[r],
                markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                label=LBL_REG[r])

    ax.axhline(2.0, color=GUIDE, ls=(0, (6, 3)), lw=1.1, zorder=2)
    ax.annotate(r"theory: $2$", xy=(mid[2], 2.02), color=MUTED, fontsize=7.8)

    ax.set_xticks(mid)
    ax.set_xticklabels([f"{x:.2f}" for x in mid], fontsize=7)
    ax.set_xlabel(r"$\varepsilon$  (geometric midpoint)")
    ax.set_ylabel(r"fitted local slope of $M_2$  ($\alpha=1$)")
    ax.set_title(r"Local slopes  ($\alpha=1$)", loc="left", color=INK)
    ax.legend(loc="lower right", labelcolor=INK, handlelength=2.1,
              borderpad=0.15, labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(fn)
    fig.savefig(fn.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {fn} and {fn.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()