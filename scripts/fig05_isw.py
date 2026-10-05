"""Figure 5. Internal solitary waves as a transient current.

(a) Peak near-bed velocity of a two-layer KdV soliton against crest
displacement, for lower layers of 2, 3 and 5 m under a 15 m bed, with the
pulse duration 2 lam/V shown for the 3 m case.  At transplant depths the
pulse is tens of centimetres per second lasting a few wave periods, too
weak to overturn a unit on its own and long enough to bias many waves.
(b) Critical wave height of each unit at 15 m depth and T = 14 s against
the superimposed soliton current, showing how much of the wave margin a
passing soliton consumes.
(c) Net displacement per soliton passage for units held at 95 percent of
their critical wave height, from the time-stepper and from the
quasi-steady estimate that integrates the steady per-cycle drift D(U_c)
over the pulse.  The displacement of a control run without the soliton,
which carries only the start-up transient of the ramp, is subtracted.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import near_bed_amplitude
from reefunit.forcing import ISW, isw_params
from reefunit.io_utils import write_csv, save_cache
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.ratchet import drift_per_cycle, quasi_steady_displacement
from reefunit.scenario import ISW_REF, ETA_GRID, UNIT_ORDER, UNIT_LABEL
from reefunit.statics import crit_height, crit_velocity
from reefunit.units import catalogue, batch

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
units = {u.name: u for u in catalogue()}
ucols = [C["black"], C["blue"], C["vermil"], C["green"], C["purple"]]
DEPTH, TW = 15.0, 14.0
om = 2 * np.pi / TW

# ------------------------------------------------- (a) pulse amplitude
ax = axs[0]
tab = {"eta0_m": ETA_GRID}
for h2, col, ls in ((2.0, C["green"], "-"), (3.0, C["blue"], "--"),
                    (5.0, C["vermil"], "-.")):
    h1 = DEPTH - h2
    p = isw_params(h1, h2, ISW_REF["drho"], ETA_GRID)
    umax = p["c0"] * ETA_GRID / h2
    dur = 2 * p["lam"] / p["V"]
    tab[f"umax_h2_{h2:g}"] = umax
    tab[f"duration_h2_{h2:g}"] = dur
    ax.plot(ETA_GRID, umax, color=col, ls=ls)
ax2 = ax.twinx()
p3 = isw_params(DEPTH - 3.0, 3.0, ISW_REF["drho"], ETA_GRID)
ax2.plot(ETA_GRID, 2 * p3["lam"] / p3["V"], color=C["grey"], lw=0.9,
         ls=":")
ax2.set_ylabel(r"$2\lambda/V$ (s)")
ax2.tick_params(right=True)
ax.set_xlabel(r"$\eta_0$ (m)")
ax.set_ylabel(r"$\max u_2$ (m s$^{-1}$)")
panel_label(ax, "(a)")
write_csv("fig05a_pulse", tab)

# ------------------------------------------- (b) margin consumed by an ISW
ax = axs[1]
Ugrid = np.linspace(0.0, 0.35, 60)
tab = {"Uc": Ugrid}
for name, col in zip(UNIT_ORDER, ucols):
    d = batch([units[name]] * len(Ugrid))
    Hc = crit_height(d, TW, DEPTH, Ugrid, stokes=True, cap=False)
    tab[name] = Hc
    ax.plot(Ugrid, Hc, color=col)
umax3 = float(isw_params(DEPTH - 3.0, 3.0, ISW_REF["drho"], 1.5)["c0"]
              * 1.5 / 3.0)
ax.axvline(umax3, color=C["grey"], lw=0.8, ls=":")
ax.set_xlim(0, Ugrid[-1])
ax.set_ylim(0, None)
ax.set_xlabel(r"$U_c$ (m s$^{-1}$)")
ax.set_ylabel(r"$H_c$ (m)")
panel_label(ax, "(b)")
write_csv("fig05b_margin", tab)

# ----------------------------------------- (c) displacement per passage
ax = axs[2]
sel = ["block", "table", "spider"]
scols = [C["black"], C["vermil"], C["green"]]
etas = ETA_GRID[::2]
tab = {"eta0_m": etas}
h2 = ISW_REF["h2"]
h1 = DEPTH - h2
res = {}
for name, col in zip(sel, scols):
    d1 = batch([units[name]])
    Hc0 = float(np.ravel(crit_height(d1, TW, DEPTH, 0.0, stokes=True,
                                     cap=False))[0])
    ub = float(near_bed_amplitude(0.95 * Hc0, TW, DEPTH))
    # steady drift map for the quasi-steady estimate
    dmap = batch([units[name]] * len(Ugrid))
    D, _ = drift_per_cycle(dmap, ub, om, Ugrid, n_cycles=12, n_skip=6,
                           spc=600)
    sim, qs = [], []
    for eta in etas:
        w = ISW(h1, h2, ISW_REF["drho"], eta, 0.0)
        dur = float(2 * w.p["lam"] / w.p["V"])
        t_end = 6 * dur + 20 * TW
        t0 = 0.5 * t_end
        spc = 500
        h = TW / spc
        n = int(t_end / h)
        dd = batch([units[name]])
        f = forcing_spec(1, ub=ub, omega=om, isw_amp=float(w.amp),
                         isw_s=float(w.s), isw_t0=t0)
        r = run(dd, f, h, n)
        f0 = forcing_spec(1, ub=ub, omega=om)
        r0 = run(dd, f0, h, n)
        sim.append(float(r.X[0] - r0.X[0]))
        tt = np.linspace(0, t_end, 4000)
        uisw = float(w.amp) / np.cosh(float(w.s) * (tt - t0)) ** 2
        qs.append(quasi_steady_displacement(Ugrid, D, uisw, tt, TW))
    res[name] = (np.array(sim), np.array(qs), ub, Hc0)
    tab[f"{name}_sim_m"] = res[name][0]
    tab[f"{name}_quasi_steady_m"] = res[name][1]
    ax.plot(etas, res[name][0], color=col)
    ax.plot(etas, res[name][1], "o", color=col, ms=3, mfc="none")
ax.set_xlabel(r"$\eta_0$ (m)")
ax.set_ylabel(r"displacement per passage (m)")
ax.set_yscale("symlog", linthresh=1e-3)
panel_label(ax, "(c)")
write_csv("fig05c_passage", tab)
save_cache("fig05", etas=etas, Ugrid=Ugrid, umax3=np.array([umax3]),
           **{f"{k}_{n}": v for k, (s, q, ub, Hc) in res.items()
              for n, v in (("sim", s), ("qs", q), ("ub", np.array([ub])),
                           ("Hc", np.array([Hc])))})
for k, (s, q, ub, Hc) in res.items():
    print(f"{k}: Hc = {Hc:.2f} m, ub = {ub:.3f} m/s, "
          f"max displacement {s.max():.3f} m")

fig.tight_layout(w_pad=1.6)
h1l, l1 = handles([r"$h_2=2$ m", r"$h_2=3$ m", r"$h_2=5$ m",
                   r"duration ($h_2=3$ m)"],
                  [C["green"], C["blue"], C["vermil"], C["grey"]],
                  ["-", "--", "-.", ":"])
h2l, l2 = handles([UNIT_LABEL[k] for k in UNIT_ORDER] +
                  [r"soliton peak, $\eta_0=1.5$ m"],
                  ucols + [C["grey"]], ["-"] * 5 + [":"])
h3l, l3 = handles([UNIT_LABEL[k] for k in sel] +
                  ["time-stepper", "quasi-steady"],
                  scols + ["k", "k"], ["-"] * 3 + ["-", ""],
                  [None] * 4 + ["o"])
panel_legends(fig, [(axs[0], h1l, l1, 2), (axs[1], h2l, l2, 2),
                    (axs[2], h3l, l3, 2)])
print(save(fig, "fig05_isw"))
