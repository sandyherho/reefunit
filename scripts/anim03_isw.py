"""Animation 3. An internal solitary wave passing over a unit.

A two-layer Korteweg-de Vries soliton of 1.5 m crest displacement travels
along a pycnocline 3 m above a 15 m bed while a 14 s swell works the bed.
The swell alone holds the hexagonal spider at rest; the soliton adds a
transient current of a little over 0.2 m/s for a couple of minutes, and
in that window the unit steps downstream and then stops.

The upper panel shows the displaced interface and the horizontal velocity
of the two layers, which are opposed: the lower layer runs with the wave
and the upper layer returns.  The lower panel is the unit at true scale,
drawn where the contact solver puts it, with its track beneath.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from reefunit.anim import dark, fig_to_rgb, write_gif, scale_bar, SIGNED, FG
from reefunit.core import near_bed_amplitude
from reefunit.draw import transform
from reefunit.forcing import ISW
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import OKABE_ITO as C
from reefunit.scenario import ISW_REF
from reefunit.statics import crit_height
from reefunit.units import batch, spider

dark()
u0 = spider()
DEPTH, TW, ETA = 15.0, 14.0, 1.5
H2, DRHO = ISW_REF["h2"], ISW_REF["drho"]
H1 = DEPTH - H2
w = ISW(H1, H2, DRHO, ETA, 0.0)
LAM, V = float(w.p["lam"]), float(w.p["V"])
C0 = float(w.p["c0"])

d = batch([u0])
Hc = float(np.ravel(crit_height(d, TW, DEPTH, 0.0, stokes=True,
                                cap=False))[0])
ub = float(near_bed_amplitude(0.95 * Hc, TW, DEPTH))
T_END, T0, SPC = 460.0, 230.0, 500
h = TW / SPC
nst = int(T_END / h)
every = 130
r = run(d, forcing_spec(1, ub=ub, omega=2 * np.pi / TW,
                        isw_amp=float(w.amp), isw_s=float(w.s),
                        isw_t0=T0), h, nst, record_every=every)
r0 = run(d, forcing_spec(1, ub=ub, omega=2 * np.pi / TW), h, nst,
         record_every=every)
X = r.rec_X[:, 0] - r0.rec_X[:, 0]
t = np.arange(1, len(X) + 1) * every * h
print(f"H_c = {Hc:.2f} m, u_b = {ub:.3f} m/s, soliton peak "
      f"{float(w.amp):.3f} m/s, half-width {LAM:.0f} m, speed "
      f"{V:.2f} m/s, net step {X[-1]:.3f} m")

x = np.linspace(-110.0, 110.0, 360)
z = np.linspace(-DEPTH, 0.0, 120)
XG, ZG = np.meshgrid(x, z)
UMAX = float(w.amp)
NORM = Normalize(-UMAX, UMAX)
SM = ScalarMappable(norm=NORM, cmap=SIGNED)

frames = []
for j in range(len(t)):
    xc = V * (t[j] - T0)
    eta = ETA / np.cosh((x - xc) / LAM) ** 2
    # layer velocities: lower layer with the wave, upper layer returning
    u_lo = C0 * eta / H2
    u_hi = -C0 * eta / H1
    iface = -H2 + eta
    fld = np.where(ZG < iface[None, :], u_lo[None, :] * np.ones_like(ZG),
                   u_hi[None, :] * np.ones_like(ZG))

    fig = plt.figure(figsize=(6.2, 4.1))
    ax = fig.add_axes([0.100, 0.615, 0.735, 0.315])
    axb = fig.add_axes([0.100, 0.075, 0.735, 0.355])
    cax = fig.add_axes([0.852, 0.075, 0.017, 0.855])

    im = ax.pcolormesh(x, z, fld, cmap=SIGNED, norm=NORM, shading="auto",
                       rasterized=True)
    ax.plot(x, iface, color="#9fe88d", lw=1.2)
    ax.axhline(0.0, color="#3a4152", lw=0.8)
    ax.axhline(-DEPTH, color="#3a4152", lw=1.0)
    ax.plot([0], [-DEPTH + 0.35], marker="^", color=C["green"], ms=6,
            mec=FG, mew=0.4)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-DEPTH - 0.4, 0.4)
    ax.set_xlabel(r"$x$ (m)", labelpad=1)
    ax.set_ylabel(r"$z$ (m)")
    u2 = float(w.amp) / np.cosh(float(w.s) * (t[j] - T0)) ** 2
    ax.text(0.0, 1.05, f"t = {t[j]:5.0f} s      "
            rf"$u_2$ = {u2:4.2f} m s$^{{-1}}$", transform=ax.transAxes,
            color=FG, fontsize=8, va="bottom")
    cb = fig.colorbar(SM, cax=cax)
    cb.set_label(r"horizontal velocity (m s$^{-1}$)")
    cb.outline.set_edgecolor("#3a4152")

    axb.axhspan(-0.16, 0.0, color="#141a28")
    axb.axhline(0.0, color="#3a4152", lw=1.0)
    axb.plot([-u0.b, u0.b], [-0.05, -0.05], color="#4b5468", lw=1.4)
    axb.plot([0.0, X[j]], [-0.115, -0.115], color=C["green"], lw=2.5,
             solid_capstyle="butt")
    for seg in transform(u0, X[j], r.rec_th[j, 0]):
        axb.plot(seg[:, 0], seg[:, 1], color=C["green"], lw=1.5)
    scale_bar(axb, 0.2, "0.2 m", 0.66, 0.46, dy=0.018)
    axb.text(0.0, 1.06, f"hex spider,  net displacement {X[j]:+5.3f} m",
             transform=axb.transAxes, color=FG, fontsize=8, va="bottom")
    axb.set_xlim(-0.95, 0.95)
    axb.set_ylim(-0.22, 0.62)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_visible(False)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

print(write_gif("anim03_isw", frames, fps=14, colors=96))
