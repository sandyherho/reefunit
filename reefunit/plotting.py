"""Figure style, palette, and multi-format export.

Conventions follow the usual physics-journal layout.  Panel labels sit
outside the axes frame, above its top-left corner.  Legends sit below the
axes, unframed, so that no curve is ever obscured and no explanatory text
competes with the data.  A legend that describes every panel is written
once for the figure; where the panels carry unrelated families of curves,
each panel gets its own legend directly beneath it, all sharing one
baseline.  Figures carry no titles and no
in-panel annotation: numerical values belong in the reports and the manuscript
captions, not inside the axes.

Axis quantities are either dimensionless ratios or SI values with the
unit in the label.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

__all__ = ["OKABE_ITO", "SEQ", "setup", "panel_label", "outside_legend",
           "panel_legends", "handles", "save", "DIVERGING", "sci"]

# Okabe and Ito colorblind-safe qualitative palette
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#8C8C8C",
}

# ordered sequence for families of curves, used consistently across figures
SEQ = [
    OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"],
    OKABE_ITO["orange"], OKABE_ITO["vermil"], OKABE_ITO["purple"],
]

# diverging map for signed quantities
DIVERGING = "RdBu_r"

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(_HERE), "outputs", "figures")


def setup():
    """Apply the figure style used by every script in this repository."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sci(x):
    """Mathtext for a positive number as m x 10^k, dropping m = 1."""
    k = int(np.floor(np.log10(x) + 1e-9))
    m = x / 10.0 ** k
    if abs(m - 1.0) < 1e-9:
        return rf"10^{{{k}}}"
    return rf"{m:g}\times10^{{{k}}}"


def panel_label(ax, text, dx=0.0, dy=1.03):
    """Panel label outside the axes, above its top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def handles(labels, colors, styles=None, markers=None, filled=None):
    """Legend proxies for a family of curves.

    Markers are drawn unfilled, as they are in the figures; pass
    ``filled`` as a per-entry boolean sequence where a solid marker is
    wanted instead.
    """
    styles = styles or ["-"] * len(labels)
    markers = markers or [None] * len(labels)
    filled = filled or [False] * len(labels)
    return ([Line2D([], [], color=c, ls=s, lw=1.3, marker=m, ms=3.8,
                    mfc=(c if f else "none"))
             for c, s, m, f in zip(colors, styles, markers, filled)],
            list(labels))


def outside_legend(fig, handles, labels, ncol=None, y=0.0):
    """Single unframed figure-level legend below the axes."""
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                      borderaxespad=0.0)


def panel_legends(fig, specs, pad=0.02, fontsize=8):
    """One unframed legend under each panel, aligned on a common baseline.

    ``specs`` is a list of ``(where, handles, labels)`` or
    ``(where, handles, labels, ncol)``, in which ``where`` is one axes or
    a sequence of adjacent axes that share the legend.  Each legend is
    centred on its panel or group of panels, and all of them hang from
    the lowest panel, so that a row of panels carries a tidy row of
    legends.  Call after ``tight_layout``.
    Entries stack over several rows when ``ncol`` is smaller than the
    number of labels.  The legends are figure-level artists, so the tight
    bounding box used on saving already includes them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    groups = [spec[0] if np.iterable(spec[0]) else [spec[0]]
              for spec in specs]
    y = min(inv.transform((0, ax.get_tightbbox(r).y0))[1]
            for g in groups for ax in g)
    out = []
    for spec, g in zip(specs, groups):
        hs, labs = spec[1], spec[2]
        ncol = spec[3] if len(spec) > 3 else len(labs)
        x0 = min(ax.get_position().x0 for ax in g)
        x1 = max(ax.get_position().x1 for ax in g)
        out.append(fig.legend(
            hs, labs, loc="upper center",
            bbox_to_anchor=(0.5 * (x0 + x1), y - pad),
            bbox_transform=fig.transFigure, ncol=ncol, frameon=False,
            fontsize=fontsize, handlelength=1.6, columnspacing=1.1,
            handletextpad=0.5, labelspacing=0.35, borderaxespad=0.0))
    return out


def save(fig, stem, dpi=600):
    """Write a vector PDF and a high-resolution PNG.

    The PDF is the submission artifact; fonts are embedded as Type 42 so that
    text remains selectable and editable.  The PNG is rendered at ``dpi`` for
    screen use and for preview in the repository.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths[0]
