"""Animation 2. A block walking under waves on a current.

The concrete block sits on the bed under a regular wave of period 8 s and
near-bed amplitude 1.8 m/s, with a steady current of 0.45 m/s running
with the waves.  Each crest pushes it a little further than the following
trough pulls it back, so it migrates in the direction of the current
without ever leaving the bed or overturning.

The upper panel is the near-bed velocity, signed, with amber onshore and
blue offshore; the arrows above the block are coloured by the
instantaneous velocity, and their length is that velocity to scale.  The
marker on the colour bar tracks the current value, and the dashed line is
the velocity at which the quasi-static sliding threshold is reached.

The lower panel records the position, whose staircase is the drift: one
tread per wave, each one in the same direction.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from reefunit.anim import dark, fig_to_rgb, write_gif, scale_bar, SIGNED, FG
from reefunit.draw import transform
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import OKABE_ITO as C
from reefunit.ratchet import asymptotic_drift
from reefunit.statics import crit_velocity
from reefunit.units import batch, block

dark()
u0 = block()
T, UB, UC, NC = 8.0, 1.8, 0.45, 14
SPC = 1200
h = T / SPC
d = batch([u0], CLA=0.0)
om = 2 * np.pi / T
r = run(d, forcing_spec(1, ub=UB, omega=om, Uc=UC), h, NC * SPC,
        record_every=100)
X = r.rec_X[:, 0]
t = np.arange(1, len(X) + 1) * 100 * h
ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / (2 * T), 0, 1))
u_t = ramp * (UC + UB * np.cos(om * t))
u_slide = float(np.ravel(crit_velocity(d, om, Uc=UC, mode=0))[0] + UC)
drift = float(np.ravel(asymptotic_drift(d, UB, om, UC))[0])
print(f"sliding begins at u = {u_slide:.3f} m/s; closed-form drift "
      f"{100 * drift:.2f} cm per cycle; simulated total {X[-1]:.3f} m "
      f"over {NC} cycles")

VMAX = UC + UB
NORM = Normalize(-VMAX, VMAX)
SM = ScalarMappable(norm=NORM, cmap=SIGNED)
xs = (-0.75, 3.1)
zrows = (0.40, 0.50, 0.60)
frames = []
for j in range(len(t)):
    fig = plt.figure(figsize=(5.9, 3.9))
    ax = fig.add_axes([0.085, 0.50, 0.775, 0.45])
    axb = fig.add_axes([0.125, 0.135, 0.735, 0.27])
    cax = fig.add_axes([0.878, 0.50, 0.018, 0.45])

    col = SM.to_rgba(u_t[j])
    nq = 8
    xq = np.linspace(xs[0] + 0.25, xs[1] - 0.35, nq)
    for zr in zrows:
        ax.quiver(xq, np.full(nq, zr), np.full(nq, u_t[j]), np.zeros(nq),
                  color=col, angles="xy", scale_units="xy", scale=5.0,
                  width=0.005, headwidth=4, headlength=5)
    ax.axhspan(-0.16, 0.0, color="#141a28")
    ax.axhline(0.0, color="#3a4152", lw=1.0)
    ax.plot([-u0.b, u0.b], [-0.04, -0.04], color="#4b5468", lw=1.4)
    for seg in transform(u0, X[j], 0.0):
        ax.plot(seg[:, 0], seg[:, 1], color=C["yellow"], lw=1.6)
    scale_bar(ax, 0.5, "0.5 m", xs[1] - 0.70, 0.08, dy=0.02)
    ax.text(0.0, 1.05, f"t = {t[j]:5.1f} s      X = {X[j]:5.3f} m"
            rf"      $u$ = {u_t[j]:+5.2f} m s$^{{-1}}$",
            transform=ax.transAxes, color=FG, fontsize=8, va="bottom")
    ax.set_xlim(xs[0], xs[1])
    ax.set_ylim(-0.2, 0.66)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(SM, cax=cax)
    cb.set_label(r"near-bed velocity $u$ (m s$^{-1}$)")
    cb.outline.set_edgecolor("#3a4152")
    cax.axhline(u_t[j], color=FG, lw=1.4)
    cax.axhline(u_slide, color="#9fe88d", lw=0.9, ls="--")

    axb.plot(t[:j + 1], X[:j + 1], color=C["yellow"], lw=1.3)
    axb.set_xlim(0, t[-1])
    axb.set_ylim(-0.03, max(0.1, X.max() * 1.08))
    axb.set_xlabel(r"$t$ (s)", labelpad=1)
    axb.set_ylabel(r"$X$ (m)")
    axb.tick_params(labelsize=7.5)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

print(write_gif("anim02_ratchet", frames, fps=16, colors=128))
