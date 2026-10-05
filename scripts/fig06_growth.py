"""Figure 6. Coral growth erodes the stability it was built for.

(a) Critical wave height of each unit at the monsoon reference condition
(8 m depth, T = 8 s) as colonies grow at 0.03 m per year of colony
radius, capped at 0.10 m, over the design wave height of that class.
Solid lines are the units as deployed, where sliding governs, and dashed
lines the same units staked against sliding (mu = 0.9), where tipping
governs.  A unit is adequate while the ratio exceeds one.
(b) Ratio after five years for the rebar table against the friction
coefficient, with colonies mounted on top and at half height.  Below the
switch at mu = b/z_p (dotted) sliding governs and mounting height changes
nothing, because sliding depends on the total load and not on where it
acts; above it tipping governs and the two mountings separate.  Colonies
are capped at 0.10 m radius because merging colonies and the canopy they
form are not modelled.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.io_utils import write_csv, save_cache
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.scenario import (COLONIES, R0_COLONY, SITES,
                               UNIT_ORDER, UNIT_LABEL)
from reefunit.statics import crit_height
from reefunit.units import catalogue, batch, with_colonies

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.8, 2.7))
units = {u.name: u for u in catalogue()}
site = SITES["monsoon"]
ucols = [C["black"], C["blue"], C["vermil"], C["green"], C["purple"]]
years = np.linspace(0, 5, 51)
R_CAP = 0.10       # colonies are not followed past this radius


def ratio_curve(name, rate, where="top", mu=0.6, t=None):
    """H_c(t)/H_design for one unit, mounting height and friction."""
    n, ftop, fside = COLONIES[name]
    u0 = units[name]
    z = (ftop if where == "top" else fside) * u0.H
    tt = years if t is None else np.atleast_1d(t)
    r = np.minimum(R0_COLONY + rate * tt, R_CAP)
    d = batch([with_colonies(u0, n, ri, z) for ri in r], mu=mu)
    return crit_height(d, site["T"], site["depth"], stokes=True,
                       cap=False) / site["H"]


tab = {"years": years}
for name, col in zip(UNIT_ORDER, ucols):
    for lab, mu, ls in (("free", 0.6, "-"), ("staked", 0.9, "--")):
        y = ratio_curve(name, 0.03, "top", mu)
        tab[f"{name}_{lab}"] = y
        ax.plot(years, y, color=col, ls=ls)
    tab[f"{name}_staked_side"] = ratio_curve(name, 0.03, "side", 0.9)
ax.axhline(1.0, color=C["grey"], lw=0.8, ls=":")
ax.set_xlim(0, years[-1])
ax.set_yscale("log")
ax.set_xlabel("years after transplanting")
ax.set_ylabel(r"$H_c/H_{\rm design}$")
panel_label(ax, "(a)")
write_csv("fig06a_growth_ratio", tab)

mus = np.linspace(0.35, 1.3, 40)
top5 = np.array([ratio_curve("table", 0.03, "top", m)[-1] for m in mus])
side5 = np.array([ratio_curve("table", 0.03, "side", m)[-1] for m in mus])
tabb = {"mu": mus, "table_top_5yr": top5, "table_side_5yr": side5}
axb.plot(mus, top5, color=C["vermil"])
axb.plot(mus, side5, color=C["skyblue"], ls="--")
nt, ft, fs = COLONIES["table"]
gt = with_colonies(units["table"], nt, R_CAP, ft * units["table"].H)
axb.axvline(gt.b / gt.z_p, color=C["grey"], lw=0.8, ls=":")
axb.axhline(1.0, color=C["grey"], lw=0.6, ls=":")
axb.set_xlim(mus[0], mus[-1])
axb.set_xlabel(r"$\mu$")
axb.set_ylabel(r"$H_c/H_{\rm design}$ at 5 yr")
panel_label(axb, "(b)")
write_csv("fig06b_horizon", tabb)
save_cache("fig06", years=years, mu=mus, table_top_5yr=top5,
           table_side_5yr=side5, mode_switch=np.array([gt.b / gt.z_p]),
           **{f"free_{k}": tab[f"{k}_free"] for k in UNIT_ORDER},
           **{f"staked_{k}": tab[f"{k}_staked"] for k in UNIT_ORDER})
for k in UNIT_ORDER:
    print(f"{k}: free {tab[f'{k}_free'][0]:.2f} -> "
          f"{tab[f'{k}_free'][-1]:.2f}; staked top "
          f"{tab[f'{k}_staked'][-1]:.2f}, staked side "
          f"{tab[f'{k}_staked_side'][-1]:.2f}")

fig.tight_layout(w_pad=1.5)
h1, l1 = handles([UNIT_LABEL[k] for k in UNIT_ORDER] +
                 ["as deployed", "staked"], ucols + ["k", "k"],
                 ["-"] * 5 + ["-", "--"])
h2, l2 = handles(["top mounted", "side mounted", r"$\mu=b/z_p$"],
                 [C["vermil"], C["skyblue"], C["grey"]], ["-", "--", ":"])
panel_legends(fig, [(ax, h1, l1, 3), (axb, h2, l2, 1)])
print(save(fig, "fig06_growth"))
