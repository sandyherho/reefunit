"""Figure 2. Which failure mode comes first.

(a) Plane of pivot half-distance over centre-of-pressure height, b/z_p,
and friction coefficient mu.  In the drag-dominated limit a unit slides
first where b/z_p > mu (shaded) and tips first otherwise, independent of
lift.  Circles are the five reference units as deployed and squares the
same units carrying grown colonies (radius 0.10 m, top mounted); the
vertical bars span the plausible friction range 0.4 to 0.8.
(b) Onset value of Theta = F_D/W' under a slowly ramped current, from the
time-stepper (markers) and from the closed forms Theta_s = mu/(1 + mu
Lambda) and Theta_t = 1/(z_p/b + Lambda) (lines), for blocks of varying
height with Lambda = C_L A_L/(C_D A) = 0 and 0.5.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import PRM
from reefunit.io_utils import write_csv, save_cache
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.scenario import UNIT_ORDER, UNIT_LABEL, R_GROWN
from reefunit.statics import theta_onsets, crit_velocity
from reefunit.units import catalogue, grown, batch, block

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.8, 2.8))
axs_a, axs_b = ax, axb
ucols = [C["black"], C["blue"], C["vermil"], C["green"], C["purple"]]

x = np.logspace(np.log10(0.3), np.log10(6), 200)
ax.fill_between(x, 0.3, np.minimum(x, 1.0), color=C["skyblue"], alpha=0.25,
                lw=0)
ax.plot(x, x, color="k", lw=1.0)
units = {u.name: u for u in catalogue()}
tab = {k: [] for k in ("b_over_zp", "b_over_zp_grown")}
for name, col in zip(UNIT_ORDER, ucols):
    u = units[name]
    g = grown(u, R_GROWN)
    r0, r1 = u.b / u.z_p, g.b / g.z_p
    tab["b_over_zp"].append(r0)
    tab["b_over_zp_grown"].append(r1)
    ax.annotate("", xy=(r1, 0.6), xytext=(r0, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                                mutation_scale=7, shrinkA=3, shrinkB=3))
    ax.plot(r0, 0.6, "o", color=col, ms=4.5)
    ax.plot(r1, 0.6, "s", color=col, ms=4.5, mfc="none")
    ax.plot([r0, r0], [0.4, 0.8], color=col, lw=0.6)
ax.set_xscale("log")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0.3, 1.0)
ax.set_xlabel(r"$b/z_p$")
ax.set_ylabel(r"$\mu$")
panel_label(ax, "(a)")
write_csv("fig02a_units", tab)

# ---------------------------------------------- (b) ramped-current onset
heights = np.geomspace(0.05, 1.6, 14)
lams = [0.0, 0.5]
bl = []
for lam in lams:
    for H in heights:
        u = block(width=0.4, height=H, depth=0.4, C_L=0.0)
        CLA = lam * u.CDA
        bl.append((u, CLA))
d = batch([p[0] for p in bl], CLA=[p[1] for p in bl])
n = len(bl)
ucr = crit_velocity(d, 1.0, mode=None)          # steady: omega irrelevant
Uend = 1.25 * crit_velocity({k: v for k, v in d.items()}, 1e-9)
t_ramp, h = 400.0, 0.004
nst = int(t_ramp / h)
f = forcing_spec(n, ub=0.0, omega=1e-9, Uc=Uend, ramp_time=t_ramp)
every = 10
r = run(d, f, h, nst, record_every=every)
moved = (np.abs(r.rec_X) > 1e-4) | (np.abs(r.rec_th) > 1e-4)
k_on = np.argmax(moved, axis=0)
t_on = (k_on + 1) * every * h
ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(t_on / t_ramp, 0, 1))
u_on = Uend * ramp
theta_sim = 0.5 * PRM.rho * d["CDA"] * u_on ** 2 / d["Wsub"]
ts, tt, tl = theta_onsets(d)
theta_th = np.minimum(np.minimum(ts, tt), tl)
mode_sim = np.where(np.abs(r.rec_th[k_on, np.arange(n)]) > 1e-4, 1, 0)
zb = d["z_p"] / d["b"]
zz = np.geomspace(0.1, 5, 300)
lcols = [C["grey"], C["orange"]]
tab = {"zp_over_b": zb, "Lambda": np.repeat(lams, len(heights)),
       "theta_sim": theta_sim, "theta_closed_form": theta_th,
       "mode_sim_0slide_1tip": mode_sim}
for i, (lam, col) in enumerate(zip(lams, lcols)):
    mu = 0.6
    axb.plot(zz, np.full_like(zz, mu / (1 + mu * lam)), color=col, lw=0.9,
             ls="--")
    axb.plot(zz, 1 / (zz + lam), color=col, lw=0.9)
    sl = slice(i * len(heights), (i + 1) * len(heights))
    mk = np.where(mode_sim[sl] == 1, "^", "o")
    for xx, yy, m in zip(zb[sl], theta_sim[sl], mk):
        axb.plot(xx, yy, m, color=col, ms=4, mfc="none")
axb.set_xscale("log")
axb.set_yscale("log")
axb.set_xlim(zz[0], zz[-1])
axb.set_ylim(0.15, 3)
axb.set_xlabel(r"$z_p/b$")
axb.set_ylabel(r"onset $\Theta=F_D/W'$")
panel_label(axb, "(b)")
write_csv("fig02b_onset", tab)
rel = np.abs(theta_sim / theta_th - 1)
print(f"ramp onset vs closed form: max rel {rel.max():.3e}")
mode_th = np.where(tt < ts, 1, 0)
print(f"mode agreement {np.mean(mode_th == mode_sim):.3f}")
save_cache("fig02", theta_rel_max=np.array([rel.max()]),
           theta_rel_median=np.array([np.median(rel)]),
           mode_agree=np.array([np.mean(mode_th == mode_sim)]),
           b_over_zp=np.array(tab_units := [units[k].b / units[k].z_p
                                            for k in UNIT_ORDER]))

fig.tight_layout(w_pad=1.5)
h1, l1 = handles([UNIT_LABEL[k] for k in UNIT_ORDER] +
                 ["as deployed", "with grown colonies"],
                 ucols + ["k", "k"], ["-"] * 5 + ["", ""],
                 [None] * 5 + ["o", "s"],
                 [False] * 5 + [True, False])
h2, l2 = handles([r"$\Lambda=0$", r"$\Lambda=0.5$", "slide onset",
                  "tip onset", "slid (sim)", "tipped (sim)"],
                 lcols + ["k", "k", "k", "k"],
                 ["-", "-", "--", "-", "", ""],
                 [None, None, None, None, "o", "^"])
panel_legends(fig, [(axs_a, h1, l1, 3), (axs_b, h2, l2, 3)])
print(save(fig, "fig02_statics"))
