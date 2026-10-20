"""
Figure 1(b): fitted exponent of D_2(eps) against the Holder exponent alpha for the
severity family f_alpha, with the crossing indicator at alpha = 0 and the smooth
anchor f_sm at a separate position, for the three regimes.

Fitted exponent (default): weighted least-squares slope of log D_2 against log eps
over eps <= FIT_SMALL_EPS_MAX (field "slope_small_M2" of production_results.json),
with its standard error.  Alternative (--estimator last_local): the local slope on
the two smallest grid values.

Curves:
  solid  : min{2, 1 + 2 alpha}, Theorem 3.1(d) (kink at alpha = 1/2); at alpha = 0
           it equals the exponent 1 of Theorem 3.1(b) for the indicator;
  dashed : max{2 alpha, 2/3}, the benchmark given by the mean-square coupling
           bound (C1) and Avikainen's inequality (kink at alpha = 1/3).

Produces: fig01b.png, fig01b.pdf
Requires: numpy, matplotlib, params.py, observables.py, production_results.json.
Usage:    python Make_figure2.py [production_results.json] [--estimator last_local]
"""
import argparse
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from params import ALPHAS, V_F, REGIMES
from observables import observable_registry

X_SMOOTH = 1.22                       # x position of the smooth anchor
SMOOTH_LABEL = r"$f_{\rm sm}$"

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


def observable_columns():
    """[(name, alpha, x_position)]: indicator, the severity ladder, smooth anchor."""
    reg = observable_registry(V_F)
    cols = [("indicator", 0.0, 0.0)] if "indicator" in reg else []
    for a in ALPHAS:
        name = f"holder{a:.4f}"
        if name in reg:
            cols.append((name, a, a))
    if "smooth" in reg:
        cols.append(("smooth", None, X_SMOOTH))
    return cols


def exponent(rec, estimator):
    """(value, standard error) of the fitted exponent of D_2."""
    if estimator == "last_local":
        return float(rec["local_M2"][-1]), float(rec["local_M2_se"][-1])
    v, se = rec["slope_small_M2"], rec["slope_small_M2_se"]
    return (np.nan, np.nan) if v is None else (float(v), float(se))


def fmt_alpha(a):
    if a is None:
        return SMOOTH_LABEL
    for num, den in ((1, 3), (1, 2), (1, 4), (3, 4)):
        if abs(a - num / den) < 1e-9:
            return f"{num}/{den}"
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


def main(results="production_results.json", fn="fig01b.png", estimator="ls_small"):
    with open(results) as f:
        D = json.load(f)
    FIT = D["fits"]
    cols = observable_columns()
    names = [c[0] for c in cols]
    xs = np.array([c[2] for c in cols], dtype=float)
    is_smooth = np.array([n == "smooth" for n in names])
    regimes = [r for r in REGIMES if r in FIT] or [r for r in ("I", "II", "III") if r in FIT]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax_style(ax)
    aa = np.linspace(0.0, 1.0, 401)
    ax.plot(aa, np.minimum(2.0, 1.0 + 2.0 * aa), color=INK, lw=1.2, zorder=2,
            label=r"Thm 3.1: $\min\{2,1+2\alpha\}$")
    ax.plot(aa, np.maximum(2.0 * aa, 2.0 / 3.0), color=MUTED, lw=1.1, ls=(0, (6, 3)),
            zorder=2, label=r"benchmark $\max\{2\alpha,2/3\}$")
    ax.plot([X_SMOOTH - 0.06, X_SMOOTH + 0.06], [2.0, 2.0], color=INK, lw=1.2, zorder=2)
    for a0 in (1.0 / 3.0, 0.5):
        ax.axvline(a0, color=GUIDE, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.axvline(X_SMOOTH - 0.11, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)

    for j, r in enumerate(regimes):
        vals = np.array([exponent(FIT[r][n], estimator) for n in names])
        off = 0.012 * (j - 1)
        ax.errorbar(xs[~is_smooth] + off, vals[~is_smooth, 0], yerr=vals[~is_smooth, 1],
                    color=C_REG[r], marker=M_REG[r], ls=L_REG[r], lw=1.0,
                    markeredgecolor="white", markeredgewidth=0.7, elinewidth=0.8,
                    capsize=1.5, zorder=4, label=LBL_REG[r])
        if is_smooth.any():
            ax.errorbar(xs[is_smooth] + off, vals[is_smooth, 0], yerr=vals[is_smooth, 1],
                        color=C_REG[r], marker=M_REG[r], ls="none",
                        markeredgecolor="white", markeredgewidth=0.7, elinewidth=0.8,
                        capsize=1.5, zorder=4)

    ax.set_xlim(-0.06, 1.32)
    ax.set_ylim(0.4, 2.45)
    show = [0.0, 1.0 / 3.0, 0.5, 0.75, 1.0]
    ax.set_xticks(show + [X_SMOOTH])
    ax.set_xticklabels([fmt_alpha(a) for a in show] + [SMOOTH_LABEL], fontsize=7.5)
    ax.set_xlabel(r"Hölder exponent $\alpha$  ($\alpha=0$: $f_{\rm ind}$)")
    ax.set_ylabel(r"fitted exponent of $\mathcal{D}_2$")
    ax.set_title(r"(b) Fitted exponent vs $\alpha$", loc="left", color=INK)
    leg = ax.legend(loc="lower right", labelcolor=INK, handlelength=2.1, frameon=True,
                    borderpad=0.3, labelspacing=0.25, fontsize=6.6)
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=0.9)
    fig.tight_layout()
    fig.savefig(fn)
    fig.savefig(fn.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {fn} and {fn.replace('.png', '.pdf')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="production_results.json")
    ap.add_argument("--estimator", default="ls_small", choices=["ls_small", "last_local"])
    a = ap.parse_args()
    main(a.results, estimator=a.estimator)
