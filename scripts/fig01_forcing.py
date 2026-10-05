"""Figure 1. Near-bed forcing of the four site classes and of an internal
solitary wave.

(a) Near-bed velocity over two periods of the design wave of each site
class at its reference depth, on the class current.  The second-order
Stokes harmonic is included where the Ursell number is below 26; the
swell class exceeds it and is shown by its linear component.
(b) Near-bed velocity amplitude per metre of wave height, u_b/H, against
depth for periods of 5, 8 and 14 s.  Long-period swell reaches the bed at
depths where wind sea no longer does.
(c) Lower-layer velocity of a two-layer KdV soliton at a fixed point,
for crest displacements eta0 = 0.3 to 1.5 m of a pycnocline 3 m above a
15 m bed, with time measured from the crest.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import near_bed_amplitude, stokes_harmonic, ursell
from reefunit.forcing import ISW
from reefunit.io_utils import write_csv
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.scenario import SITES, SITE_ORDER, ISW_REF

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
cols = [C["green"], C["blue"], C["vermil"], C["purple"]]

ax = axs[0]
tab = {}
ph = np.linspace(0, 2, 400)
tab["t_over_T"] = ph
for name, col in zip(SITE_ORDER, cols):
    s = SITES[name]
    ub = near_bed_amplitude(s["H"], s["T"], s["depth"])
    u2 = stokes_harmonic(s["H"], s["T"], s["depth"])
    print(f"{name}: u_b = {ub:.3f}, u_2 = {u2:.3f} m/s, Ursell ="
          f" {ursell(s['H'], s['T'], s['depth']):.1f}")
    th = 2 * np.pi * ph
    uu = s["Uc"] + ub * np.cos(th) + u2 * np.cos(2 * th)
    tab[name] = uu
    ax.plot(ph, uu, color=col)
ax.axhline(0, color=C["grey"], lw=0.6)
ax.set_xlim(0, 2)
ax.set_xlabel(r"$t/T$")
ax.set_ylabel(r"$u$ (m s$^{-1}$)")
panel_label(ax, "(a)")
write_csv("fig01a_site_velocity", tab)

ax = axs[1]
h = np.linspace(1, 25, 300)
tab = {"depth_m": h}
for T, col, ls in ((5.0, C["green"], "-"), (8.0, C["blue"], "--"),
                   (14.0, C["vermil"], "-.")):
    r = near_bed_amplitude(1.0, T, h)
    tab[f"ub_per_H_T{T:g}"] = r
    ax.plot(h, r, color=col, ls=ls)
ax.set_xlim(1, 25)
ax.set_yscale("log")
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"depth $h$ (m)")
ax.set_ylabel(r"$u_b/H$ (s$^{-1}$)")
panel_label(ax, "(b)")
write_csv("fig01b_depth_attenuation", tab)

ax = axs[2]
t = np.linspace(-120, 120, 800)
tab = {"t_s": t}
etas = [0.3, 0.6, 0.9, 1.2, 1.5]
ecols = [plt.cm.viridis(x) for x in np.linspace(0.05, 0.85, len(etas))]
for eta, col in zip(etas, ecols):
    w = ISW(ISW_REF["h1"], ISW_REF["h2"], ISW_REF["drho"], eta, 0.0)
    uu, _ = w.at(t, 0)
    tab[f"u_eta{eta:g}"] = uu
    ax.plot(t, uu, color=col)
ax.set_xlim(t[0], t[-1])
ax.set_ylim(0, None)
ax.set_xlabel(r"$t-t_0$ (s)")
ax.set_ylabel(r"$u_2$ (m s$^{-1}$)")
panel_label(ax, "(c)")
write_csv("fig01c_isw_velocity", tab)

fig.tight_layout(w_pad=1.2)
h1, l1 = handles(SITE_ORDER, cols)
h2, l2 = handles(["T = 5 s", "T = 8 s", "T = 14 s"],
                 [C["green"], C["blue"], C["vermil"]], ["-", "--", "-."])
h3, l3 = handles([rf"$\eta_0={e:g}$ m" for e in etas], ecols)
panel_legends(fig, [(axs[0], h1, l1, 2), (axs[1], h2, l2, 2),
                    (axs[2], h3, l3, 3)])
print(save(fig, "fig01_forcing"))
