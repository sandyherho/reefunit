"""Figure 0. The five reference units and the planar load model.

(a) Edge-on outlines of the block, perforated dome, rebar table,
hexagonal spider and tetrahedral frame, drawn to scale with the centre of
mass (filled circle), the centre of pressure (cross) and the two rocking
pivots (triangles) used by the planar model.
(b) Loads on a unit resting on the bed: Morison drag F_D at the centre of
pressure z_p, Froude-Krylov and added-mass force F_I and lift F_L at the
centre of mass h_c, submerged weight W', and the normal and friction
reactions at the pivots, separated by 2b.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.io_utils import write_csv
from reefunit.plotting import setup, OKABE_ITO as C, panel_label, save
from reefunit.scenario import UNIT_ORDER, UNIT_LABEL
from reefunit.units import catalogue

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.8, 2.7),
                              gridspec_kw={"width_ratios": [2.0, 1.0]})
units = {u.name: u for u in catalogue()}
x0 = 0.0
tab = {k: [] for k in ("x_offset", "b", "h_c", "z_p", "H")}
for name in UNIT_ORDER:
    u = units[name]
    x0 += u.b + 0.12 if name != "block" else u.b
    for line in u.outline:
        xy = np.array(line)
        ax.plot(x0 + xy[:, 0], xy[:, 1], color="k", lw=1.0)
    ax.plot(x0, u.h_c, "o", color=C["vermil"], ms=3.5)
    ax.plot(x0, u.z_p, "x", color=C["blue"], ms=4.5, mew=1.1)
    ax.plot([x0 - u.b, x0 + u.b], [0, 0], "^", color=C["green"], ms=4)
    ax.text(x0, -0.13, UNIT_LABEL[name], ha="center", va="top", fontsize=7)
    for k, v in (("x_offset", x0), ("b", u.b), ("h_c", u.h_c),
                 ("z_p", u.z_p), ("H", u.H)):
        tab[k].append(v)
    x0 += u.b
ax.axhline(0, color=C["grey"], lw=0.8)
ax.set_xlim(-0.3, x0 + 0.1)
ax.set_ylim(-0.3, 0.9)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_ylabel(r"$z$ (m)")
panel_label(ax, "(a)")
write_csv("fig00_outlines", tab)

# ---------------------------------------------------------- (b) load model
b, hc, zp, H = 0.5, 0.45, 0.6, 1.0
axb.plot([-b, b, b, -b, -b], [0, 0, H, H, 0], color="k", lw=1.0)
axb.axhline(0, color=C["grey"], lw=0.8)
arr = dict(arrowstyle="-|>", lw=1.0, mutation_scale=8)
axb.annotate("", xy=(0.95, zp), xytext=(0.25, zp),
             arrowprops=dict(color=C["blue"], **arr))
axb.annotate("", xy=(0.7, hc), xytext=(0.0, hc),
             arrowprops=dict(color=C["orange"], **arr))
axb.annotate("", xy=(0.0, hc + 0.4), xytext=(0.0, hc),
             arrowprops=dict(color=C["purple"], **arr))
axb.annotate("", xy=(0.0, hc - 0.45), xytext=(0.0, hc),
             arrowprops=dict(color="k", **arr))
for sx in (-b, b):
    axb.annotate("", xy=(sx, 0.3), xytext=(sx, 0.0),
                 arrowprops=dict(color=C["green"], **arr))
    axb.annotate("", xy=(sx - 0.25, 0.0), xytext=(sx, 0.0),
                 arrowprops=dict(color=C["green"], **arr))
axb.plot(0, hc, "o", color=C["vermil"], ms=3.5)
axb.plot(0.25, zp, "x", color=C["blue"], ms=4.5, mew=1.1)
axb.text(0.97, zp, r"$F_D$", va="center", fontsize=8)
axb.text(0.72, hc - 0.04, r"$F_I$", va="top", fontsize=8)
axb.text(0.05, hc + 0.38, r"$F_L$", fontsize=8)
axb.text(0.05, hc - 0.42, r"$W'$", fontsize=8)
axb.text(-b - 0.05, 0.33, r"$N$", ha="right", fontsize=8)
axb.text(-b - 0.28, -0.03, r"$\mu N$", va="top", fontsize=8)
axb.annotate("", xy=(-b, -0.12), xytext=(b, -0.12),
             arrowprops=dict(arrowstyle="<->", lw=0.7))
axb.text(0, -0.15, r"$2b$", ha="center", va="top", fontsize=8)
axb.annotate("", xy=(1.18, 0), xytext=(1.18, zp),
             arrowprops=dict(arrowstyle="<->", lw=0.7))
axb.text(1.21, 0.5 * zp, r"$z_p$", va="center", fontsize=8)
axb.annotate("", xy=(-0.25, 0), xytext=(-0.25, hc),
             arrowprops=dict(arrowstyle="<->", lw=0.7))
axb.text(-0.28, 0.5 * hc, r"$h_c$", va="center", ha="right", fontsize=8)
axb.text(-1.05, 1.05, r"$u(t)\;\rightarrow$", fontsize=8)
axb.set_xlim(-1.15, 1.4)
axb.set_ylim(-0.35, 1.2)
axb.set_aspect("equal")
axb.axis("off")
panel_label(axb, "(b)")
fig.tight_layout(w_pad=1.0)
print(save(fig, "fig00_schematic"))
