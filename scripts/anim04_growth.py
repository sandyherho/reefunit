"""Animation 4. Growth eating the stability margin.

The rebar table, staked against sliding, carries twenty colonies on its
top surface that grow from 0.03 m to 0.10 m radius over the first two and
a half years after transplanting, where the model stops following them
because merging colonies and the canopy they form are not represented.

The map is the critical wave height over water depth and wave period, the
smallest regular wave that moves the unit, recomputed at every frame from
the current colony size.  The white contour is the monsoon design wave of
2 m: inside it the unit is adequate, outside it is not, and the contour
sweeps outward as the corals grow.  The marker is the monsoon reference
condition of 8 m depth and 8 s period, which starts inside the contour and
ends outside it.

The inset draws the unit and its colonies at true scale, so that the
geometry driving the map is visible beside it.  Depth and period beyond
the top of the colour scale are shown at its last colour: there no wave
below three times the water depth moves the unit at all.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.anim import dark, fig_to_rgb, write_gif, scale_bar, FG
from reefunit.io_utils import load_cache, save_cache
from reefunit.draw import colony_centers, transform
from reefunit.plotting import OKABE_ITO as C
from reefunit.scenario import COLONIES, R0_COLONY, SITES
from reefunit.statics import crit_height
from reefunit.units import batch, expand, table, with_colonies

dark()
site = SITES["monsoon"]
u0 = table()
N_COL, FTOP, _ = COLONIES["table"]
Z_MOUNT = FTOP * u0.H
MU, RATE, R_CAP = 0.9, 0.03, 0.10

years = np.linspace(0.0, 2.5, 31)
radii = np.minimum(R0_COLONY + RATE * years, R_CAP)
depths = np.linspace(2.0, 20.0, 48)
periods = np.linspace(3.0, 16.0, 42)
DG, TG = np.meshgrid(depths, periods, indexing="ij")

cache = load_cache("anim04")
if cache is not None and cache["radii"].shape == radii.shape:
    maps = list(cache["maps"])
    ratio = cache["ratio"]
else:
    maps, ratio = [], []
    for rad in radii:
        g = with_colonies(u0, N_COL, rad, Z_MOUNT)
        d = expand(batch([g], mu=MU), DG.shape)
        d1 = {k: v[0] for k, v in d.items()}
        maps.append(crit_height(d1, TG, DG, stokes=True, cap=False))
        ratio.append(float(np.ravel(crit_height(
            batch([g], mu=MU), site["T"], site["depth"], stokes=True,
            cap=False))[0]) / site["H"])
    ratio = np.array(ratio)
    save_cache("anim04", maps=np.array(maps), ratio=ratio, radii=radii)
print(f"H_c/H_design at the monsoon reference falls from {ratio[0]:.2f} "
      f"to {ratio[-1]:.2f}; colony radius {radii[0]:.3f} to "
      f"{radii[-1]:.3f} m")

VMAX = 6.0
CMAP = plt.get_cmap("magma").copy()
CMAP.set_over(CMAP(1.0))
frames = []
for j, (Hc, rad) in enumerate(zip(maps, radii)):
    fig = plt.figure(figsize=(6.0, 3.8))
    ax = fig.add_axes([0.095, 0.145, 0.60, 0.78])
    cax = fig.add_axes([0.715, 0.145, 0.018, 0.78])
    axi = fig.add_axes([0.805, 0.145, 0.185, 0.46])

    Hs = np.where(np.isfinite(Hc), Hc, 9.9e3)
    im = ax.pcolormesh(periods, depths, Hs, cmap=CMAP, vmin=0.0,
                       vmax=VMAX, shading="auto", rasterized=True)
    ax.contour(periods, depths, Hs, levels=[site["H"]], colors="white",
               linewidths=1.4)
    ax.plot(site["T"], site["depth"], marker="o", ms=6,
            mfc=C["vermil"] if ratio[j] < 1 else C["green"], mec="white",
            mew=0.9)
    ax.set_xlabel(r"$T$ (s)", labelpad=1)
    ax.set_ylabel(r"depth $h$ (m)")
    ax.text(0.0, 1.03, f"year {years[j]:4.1f}      colony radius "
            f"{100 * rad:4.1f} cm      "
            rf"$H_c/H_{{\rm design}}$ = {ratio[j]:4.2f}",
            transform=ax.transAxes, color=FG, fontsize=8, va="bottom")
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"critical wave height $H_c$ (m)")
    cb.outline.set_edgecolor("#3a4152")
    cb.ax.axhline(site["H"], color="white", lw=1.4)

    g = with_colonies(u0, N_COL, rad, Z_MOUNT)
    axi.axhspan(-0.18, 0.0, color="#141a28")
    axi.axhline(0.0, color="#3a4152", lw=1.0)
    for seg in transform(g, 0.0, 0.0, Z=g.h_c, lines=u0.outline):
        axi.plot(seg[:, 0], seg[:, 1], color=C["vermil"], lw=1.4)
    for cx, cz in colony_centers(g, 0.0, 0.0, N_COL, Z_MOUNT, Z=g.h_c):
        axi.add_patch(plt.Circle((cx, cz + 0.35 * rad), rad,
                                 color=C["skyblue"], alpha=0.8, lw=0))
    scale_bar(axi, 0.5, "0.5 m", -0.25, 0.92, dy=0.025)
    axi.set_xlim(-0.75, 0.75)
    axi.set_ylim(-0.2, 1.08)
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(False)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

frames += [frames[-1]] * 10
print(write_gif("anim04_growth", frames, fps=12, colors=128))
