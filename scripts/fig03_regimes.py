"""Figure 3. Critical wave height across depth and period.

(a-e) Height H_c of the regular wave that first slides, tips or lifts
each reference unit in its native material, against water depth and
period, from the quasi-static criteria with the second-order Stokes
harmonic where it applies.  White marks depths and periods where the unit
survives every wave up to the depth-limited breaking height 0.78 h.
Contours at 1, 2 and 3 m.
(f) H_c at the monsoon reference condition (depth 8 m, period 8 s) for
every structure made of every material, geometry held fixed.  Material
enters the quasi-static criteria only through the submerged weight.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from reefunit.io_utils import write_csv, save_cache
from reefunit.plotting import setup, panel_label, save
from reefunit.scenario import (DEPTHS, PERIODS, UNIT_ORDER, UNIT_LABEL,
                               MATERIAL_ORDER, SITES)
from reefunit.statics import crit_height
from reefunit.units import catalogue, batch, expand, with_material

setup()
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
fig.subplots_adjust(hspace=0.55, wspace=0.35, right=0.86, top=0.92,
                    bottom=0.16)
axs = axs.ravel()
units = {u.name: u for u in catalogue()}
Hg, Tg = np.meshgrid(DEPTHS, PERIODS, indexing="ij")
norm = Normalize(0, 6)
cmap = plt.cm.cividis.copy()
cmap.set_over("white")
stats = {}
for i, name in enumerate(UNIT_ORDER):
    ax = axs[i]
    d = expand(batch([units[name]]), Hg.shape)
    d1 = {k: v[0] for k, v in d.items()}
    Hc = crit_height(d1, Tg, Hg, stokes=True)
    Hs = np.where(np.isfinite(Hc), Hc, 99.0)
    im = ax.pcolormesh(PERIODS, DEPTHS, Hs, cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)
    cs = ax.contour(PERIODS, DEPTHS, Hs, levels=[1, 2, 3], colors="k",
                    linewidths=0.6)
    ax.set_xlabel(r"$T$ (s)")
    if i % 3 == 0:
        ax.set_ylabel(r"depth $h$ (m)")
    panel_label(ax, f"({'abcde'[i]}) {UNIT_LABEL[name]}")
    write_csv(f"fig03_{name}_Hc", {"depth_m": Hg.ravel(), "T_s": Tg.ravel(),
                                   "Hc_m": Hc.ravel()})
    stats[name] = Hc
cax = fig.add_axes([0.89, 0.55, 0.018, 0.34])
cb = fig.colorbar(im, cax=cax, extend="max")
cb.set_label(r"$H_c$ (m)")

ax = axs[5]
s = SITES["monsoon"]
M = np.zeros((len(UNIT_ORDER), len(MATERIAL_ORDER)))
for i, name in enumerate(UNIT_ORDER):
    for j, mat in enumerate(MATERIAL_ORDER):
        d = batch([with_material(units[name], mat)])
        M[i, j] = crit_height(d, s["T"], s["depth"], stokes=True,
                              cap=False)[0]
im2 = ax.imshow(M, cmap="cividis", norm=Normalize(0, 6), aspect="auto")
ax.set_xticks(range(len(MATERIAL_ORDER)))
ax.set_xticklabels([m.replace("coated steel", "coated")
                    for m in MATERIAL_ORDER], rotation=40, ha="right",
                   fontsize=7)
ax.set_yticks(range(len(UNIT_ORDER)))
ax.set_yticklabels([UNIT_LABEL[k] for k in UNIT_ORDER], fontsize=7)
ax.tick_params(top=False, right=False)
panel_label(ax, "(f)", dx=-0.02)
tab = {"unit_index": np.repeat(np.arange(len(UNIT_ORDER)),
                               len(MATERIAL_ORDER)),
       "material_index": np.tile(np.arange(len(MATERIAL_ORDER)),
                                 len(UNIT_ORDER)),
       "Hc_m": M.ravel()}
write_csv("fig03f_material_matrix", tab)
save_cache("fig03", material_matrix=M, **{f"Hc_{k}": v
                                          for k, v in stats.items()})
print(save(fig, "fig03_regimes"))
