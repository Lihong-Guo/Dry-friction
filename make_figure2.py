"""
Fitted exponent of M2 in the local-slopes sense versus alpha 
for every member of the severity family, including the smooth anchor
for all three regimes.  
Dashed curve: proven bound beta = max(2*alpha, 2/3).  
The crossover alpha = 1/3 is marked.
  
The fitted exponents themselves are read from `production_results.json` 
(produced by `production_run.py`).

The smooth anchor has no Holder exponent, so it is drawn at a separate
x position and labelled $C_b^\infty$ on the axis.

Produces: fig01b.png, fig01b.pdf

Requires: numpy, matplotlib, params.py, observables.py, production_results.json.
"""
#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from params import ALPHAS, V_F, REGIMES
from observables import observable_registry

# x position and axis label of the smooth anchor (no Holder exponent)
X_SMOOTH = 1.22
SMOOTH_LABEL = r"$C_b^\infty$"


# Style
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

# Column layout, derived from params.py + observable_registry
def observable_columns():
    """Return [(name, alpha, x_position), ...] in plotting order.

    alpha is None for the smooth anchor, in which case x_position is
    X_SMOOTH.  Every observable name here is a key of the `fits` dict in
    production_results.json.
    """
    reg = observable_registry(V_F)
    cols = []
    if "indicator" in reg:
        cols.append(("indicator", 0.0, 0.0))
    for a in ALPHAS:
        name = f"holder{a:.4f}"          # matches observable_registry's key
        if name in reg:
            cols.append((name, a, a))
    if "smooth" in reg:
        cols.append(("smooth", None, X_SMOOTH))
    return cols


def fmt_alpha(a):
    """Human-friendly tick label for a Holder exponent."""
    if a is None:
        return SMOOTH_LABEL
    if abs(a - 1.0 / 3.0) < 1e-9:
        return "1/3"
    return f"{a:g}"

def ax_style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.minorticks_off()

# Main
def main(results="production_results.json", fn="fig01b.png"):
    D = json.load(open(results))
    FIT = D["fits"]

    cols = observable_columns()
    names = [c[0] for c in cols]
    xs = np.array([c[2] for c in cols], dtype=float)
    is_smooth = np.array([n == "smooth" for n in names])

    # Which regimes to draw: fall back to whatever `fits` contains if
    # params.REGIMES is not in sync with the data file.
    regimes = [r for r in REGIMES if r in FIT]
    if not regimes:
        regimes = [r for r in ("I", "II", "III") if r in FIT]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax_style(ax)

    #### theory curve 
    aa = np.linspace(0, 1, 400)
    ax.plot(aa, np.maximum(2 * aa, 2 / 3), color=INK, lw=1.3, ls=(0, (6, 3)),
            zorder=2, label=r"$\beta=\max(2\alpha,\,2/3)$")

    #### crossover at alpha = 1/3 
    ax.axvline(1.0 / 3.0, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate(r"$\alpha=1/3$", xy=(1.0 / 3.0 - 0.02, 0.47), color=MUTED,
                fontsize=7.4, ha="right")

    #### separator before the smooth anchor 
    ax.axvline(X_SMOOTH - 0.11, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)

    #### data  
    for r in regimes:
        # (i) indicator + Holder ladder: connected line series
        y_holder = np.array(
            [FIT[r][n]["local_M2"][-1] for n, s in zip(names, is_smooth)
             if not s]
        )
        ax.plot(xs[~is_smooth], y_holder,
                color=C_REG[r], marker=M_REG[r], ls=L_REG[r],
                markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label=LBL_REG[r])

        # (ii) smooth anchor: separate marker, same colour
        y_smooth = np.array(
            [FIT[r][n]["local_M2"][-1] for n, s in zip(names, is_smooth) if s]
        )
        if y_smooth.size:
            ax.plot(xs[is_smooth], y_smooth,
                    color=C_REG[r], marker=M_REG[r], ls="none",
                    markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    #### axes  
    ax.set_xlim(-0.06, 1.32)
    ax.set_ylim(0.4, 2.35)

    # Ticks: only the interesting alpha values, plus the smooth anchor.
    show_alphas = sorted({0.0, 1.0 / 3.0, 0.5, 0.75, 1.0})
    xticks = show_alphas + [X_SMOOTH]
    xlabels = [fmt_alpha(a) for a in show_alphas] + [SMOOTH_LABEL]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)

    ax.set_xlabel(r"Hölder exponent  $\alpha$")
    ax.set_ylabel(r"fitted exponent of $M_2$")
    ax.set_title(r"Fitted exponent vs $\alpha$", loc="left", color=INK)
    ax.legend(loc="upper left", labelcolor=INK, handlelength=2.1,
              borderpad=0.15, labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(fn)
    fig.savefig(fn.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {fn} and {fn.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()