"""Figure 4. Waves on a current make a unit walk.

(a) Net drift per wave cycle of the concrete block at T = 8 s and
u_b = 1.8 m/s, against the current ratio U_c/u_b: time-stepper with drag on
the relative velocity, time-stepper with drag on the ambient velocity
alone, the event-driven stick-slip reference (open circles), and the
closed-form near-threshold estimate.  A symmetric wave produces equal and
opposite slips and no net drift; the current breaks that symmetry.
(b) Drift with relative-velocity drag over the ambient-velocity estimate
Delta_0, against the damping number Gamma, with the first-order law
1 - (6/5) Gamma.  Slip against the flow it is sliding through removes a
fixed fraction of the drift.
(c) Practical rate: displacement per hour of each unit under the monsoon
design wave at 8 m depth, against a superimposed current.  Rates above a
few metres per hour mean the unit slides during most of the wave cycle
and is being transported rather than ratcheting.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import near_bed_amplitude
from reefunit.forcing import Harmonic
from reefunit.io_utils import write_csv, save_cache
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.ratchet import (asymptotic_drift, damping_number,
                              drift_per_cycle, peak_excess)
from reefunit.reduced import slide1d
from reefunit.scenario import SITES, UNIT_ORDER, UNIT_LABEL
from reefunit.units import block, catalogue, batch

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
T, om = 8.0, 2 * np.pi / 8.0
blk = batch([block()], CLA=0.0)

# ----------------------------------------------------------- (a) drift map
ub0 = 1.8
ratios = np.linspace(0.0, 0.45, 28)
Uc = ratios * ub0
d_rel, _ = drift_per_cycle(blk, ub0, om, Uc, n_cycles=12, n_skip=6,
                           spc=2000)
d_abs, _ = drift_per_cycle(blk, ub0, om, Uc, n_cycles=12, n_skip=6,
                           spc=2000, relative=False)
d_as = asymptotic_drift(blk, ub0, om, Uc)
d_ev = []
for U in Uc:
    F = Harmonic(ub0, om, Uc=U)
    ts, xs, _ = slide1d(blk, lambda t: (float(F.at(t, 0)[0]),
                                        float(F.at(t, 0)[1])), 12 * T)
    x = np.interp([6 * T, 12 * T], ts, xs)
    d_ev.append((x[1] - x[0]) / 6)
d_ev = np.array(d_ev)
ax = axs[0]
ax.plot(ratios, 100 * d_rel, color=C["blue"])
ax.plot(ratios, 100 * d_abs, color=C["vermil"], ls="--")
ax.plot(ratios, 100 * d_ev, "o", color=C["blue"], ms=3, mfc="none")
ax.plot(ratios, 100 * d_as, color=C["grey"], ls=":")
ax.set_xlabel(r"$U_c/u_b$")
ax.set_ylabel(r"drift per cycle (cm)")
panel_label(ax, "(a)")
write_csv("fig04a_drift", {"Uc_over_ub": ratios, "drift_relative_m": d_rel,
                           "drift_ambient_m": d_abs,
                           "drift_event_driven_m": d_ev,
                           "drift_closed_form_m": d_as})
big = d_rel > 1e-4
ev_rel = np.max(np.abs(d_ev[big] / d_rel[big] - 1))
print(f"event-driven vs time-stepper, max rel diff: {ev_rel:.3e}")

# --------------------------------------------------- (b) damping correction
eps_t = np.geomspace(2e-4, 5e-2, 9)
Ucb = 0.3
ubs = []
for e in eps_t:
    lo, hi = 0.5, 6.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = ((m, hi) if peak_excess(blk, m, om, Ucb)[0][0] < e
                  else (lo, m))
    ubs.append(0.5 * (lo + hi))
ubs = np.array(ubs)
dr, _ = drift_per_cycle(blk, ubs, om, Ucb, n_cycles=10, n_skip=4, spc=4000)
da = asymptotic_drift(blk, ubs, om, Ucb)
G = damping_number(blk, ubs, om, Ucb)
ax = axs[1]
ax.plot(G, dr / da, "o", color=C["blue"], ms=3.5, mfc="none")
gg = np.linspace(0, G.max() * 1.05, 50)
ax.plot(gg, 1 - 1.2 * gg, color=C["grey"], ls="--")
ax.set_xlim(0, G.max() * 1.05)
ax.set_xlabel(r"$\Gamma$")
ax.set_ylabel(r"$\Delta/\Delta_0$")
panel_label(ax, "(b)")
write_csv("fig04b_damping", {"eps": eps_t, "ub": ubs, "Gamma": G,
                             "ratio": dr / da, "drift_m": dr,
                             "drift_closed_form_m": da})

# ------------------------------------------------------- (c) drift per hour
s = SITES["monsoon"]
ubm = float(near_bed_amplitude(s["H"], s["T"], s["depth"]))
Ucs = np.linspace(0.0, 0.6, 25)
units = {u.name: u for u in catalogue()}
ax = axs[2]
ucols = [C["black"], C["blue"], C["vermil"], C["green"], C["purple"]]
tab = {"Uc": Ucs}
for name, col in zip(UNIT_ORDER, ucols):
    dd = batch([units[name]] * len(Ucs))
    dpc, _ = drift_per_cycle(dd, ubm, 2 * np.pi / s["T"], Ucs, n_cycles=12,
                             n_skip=6, spc=800)
    rate = dpc * 3600.0 / s["T"]
    tab[name] = rate
    ax.plot(Ucs, rate, color=col)
ax.set_xlabel(r"$U_c$ (m s$^{-1}$)")
ax.set_ylabel(r"drift (m h$^{-1}$)")
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_ylim(-1e-4, 2e3)
panel_label(ax, "(c)")
write_csv("fig04c_drift_rate", tab)
save_cache("fig04", ev_rel=np.array([ev_rel]), ratios=ratios,
           d_rel=d_rel, d_abs=d_abs, d_ev=d_ev, d_as=d_as, Gamma=G,
           ratio=dr / da, Ucs=Ucs,
           rates=np.array([tab[k] for k in UNIT_ORDER]), ub_monsoon=ubm)

fig.tight_layout(w_pad=1.3)
h1, l1 = handles(["relative velocity", "ambient velocity",
                  "event-driven", "closed form"],
                 [C["blue"], C["vermil"], C["blue"], C["grey"]],
                 ["-", "--", "", ":"], [None, None, "o", None])
h2, l2 = handles(["time-stepper", r"$1-(6/5)\Gamma$"],
                 [C["blue"], C["grey"]], ["", "--"], ["o", None])
h3, l3 = handles([UNIT_LABEL[k] for k in UNIT_ORDER], ucols)
panel_legends(fig, [(axs[0], h1, l1, 2), (axs[1], h2, l2, 1),
                    (axs[2], h3, l3, 2)])
print(save(fig, "fig04_ratchet"))
