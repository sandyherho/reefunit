"""Animation 1. The wave that reaches the bed, and what it moves.

A regular wave of period 8 s on 8 m of water grows from calm to a height
of 3 m over ten periods while the five reference units sit on the bed in
their native materials.

The upper panel is the linear orbital velocity field over one and a half
wavelengths, coloured by speed; the surface is drawn at its computed
elevation and the box marks the stretch of bed shown below.  Horizontal
and vertical scales differ there, as the wavelength is some seventy times
the water depth.

The lower panel is that stretch at true scale, with the units drawn from
their own geometry and placed where the solver puts them.  Nothing is
exaggerated: the units that move are the ones whose quasi-static
threshold the growing wave has passed, and they move at the speed the
contact solver gives them.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.anim import (dark, fig_to_rgb, write_gif, scale_bar, SPEED,
                           FG)
from reefunit.core import near_bed_amplitude, orbital_field, wavenumber
from reefunit.draw import com_height, transform
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import OKABE_ITO as C
from reefunit.scenario import SPC, UNIT_LABEL, UNIT_ORDER
from reefunit.units import batch, catalogue

dark()
DEPTH, T, H_MAX, NC = 8.0, 8.0, 3.0, 10
TOTAL = NC * T
units = {u.name: u for u in catalogue()}
ucols = [C["yellow"], C["skyblue"], C["vermil"], C["green"], C["purple"]]

# ------------------------------------------------------------- simulation
d = batch([units[k] for k in UNIT_ORDER])
ub_max = float(near_bed_amplitude(H_MAX, T, DEPTH))
h = T / SPC
nst = int(TOTAL / h)
every = 50
r = run(d, forcing_spec(len(UNIT_ORDER), ub=ub_max, omega=2 * np.pi / T,
                        ramp_time=TOTAL), h, nst, record_every=every)
X, TH = r.rec_X, r.rec_th
t = np.arange(1, X.shape[0] + 1) * every * h
ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / TOTAL, 0, 1))
H_t = ramp * H_MAX
print(f"u_b at H = {H_MAX} m is {ub_max:.3f} m/s; "
      f"final displacement (m): "
      + ", ".join(f"{k} {X[-1, i]:+.2f}" for i, k in enumerate(UNIT_ORDER)))

# ---------------------------------------------------------------- geometry
L = float(2 * np.pi / wavenumber(2 * np.pi / T, DEPTH))
xf = np.linspace(-0.75 * L, 0.75 * L, 260)
zf = np.linspace(-DEPTH, 0.0, 90)
XF, ZF = np.meshgrid(xf, zf)
VMAX = float(np.hypot(*orbital_field(H_MAX, T, DEPTH, 0.0, 0.0, 0.0)[:2]))

half = [units[k].b for k in UNIT_ORDER]
offs = np.cumsum([0.0] + [half[i] + half[i + 1] + 0.95
                          for i in range(len(half) - 1)])
offs = offs - offs.mean()
XZOOM = 0.5 * (offs[-1] - offs[0]) + half[-1] + 0.7

frames = []
for j in range(len(t)):
    Hj = max(H_t[j], 1e-3)
    fig = plt.figure(figsize=(6.6, 4.3))
    ax = fig.add_axes([0.105, 0.595, 0.755, 0.335])
    axb = fig.add_axes([0.105, 0.10, 0.755, 0.355])
    cax = fig.add_axes([0.878, 0.10, 0.017, 0.845])

    # ------------------------------------------------- (top) orbital field
    u, w, _ = orbital_field(Hj, T, DEPTH, XF, ZF, t[j])
    _, _, eta = orbital_field(Hj, T, DEPTH, xf, 0.0, t[j])
    spd = np.where(ZF <= eta[None, :], np.hypot(u, w), np.nan)
    im = ax.pcolormesh(xf, zf, spd, cmap=SPEED, vmin=0.0, vmax=VMAX,
                       shading="auto", rasterized=True)
    ax.plot(xf, eta, color="#9fe88d", lw=1.0)
    ax.axhline(-DEPTH, color="#3a4152", lw=1.0)
    ax.add_patch(plt.Rectangle((-XZOOM, -DEPTH), 2 * XZOOM, 1.3,
                               fill=False, ec=FG, lw=0.8, ls="--"))
    ax.set_xlim(xf[0], xf[-1])
    ax.set_ylim(-DEPTH - 0.3, 1.9)
    ax.set_xlabel(r"$x$ (m)", labelpad=1)
    ax.set_ylabel(r"$z$ (m)")
    ax.text(0.0, 1.04, f"t = {t[j]:5.1f} s      H = {Hj:4.2f} m"
            rf"      $u_b$ = {ramp[j] * ub_max:4.2f} m s$^{{-1}}$",
            transform=ax.transAxes, color=FG, fontsize=8, va="bottom")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"orbital speed $\sqrt{u^2+w^2}$ (m s$^{-1}$)")
    cb.outline.set_edgecolor("#3a4152")

    # ------------------------------------------------ (bottom) the units
    ubed, _, _ = orbital_field(Hj, T, DEPTH, np.array([0.0]),
                               np.array([-DEPTH]), t[j])
    axb.axhspan(-0.22, 0.0, color="#141a28")
    axb.axhline(0.0, color="#3a4152", lw=1.0)
    nq = 30
    xq = np.linspace(-XZOOM, XZOOM, nq)
    axb.quiver(xq, np.full(nq, 1.12), np.full(nq, float(ubed[0])),
               np.zeros(nq), color=SPEED(float(abs(ubed[0])) / VMAX),
               scale=9.0, width=0.0032, headwidth=4)
    for i, (k, col) in enumerate(zip(UNIT_ORDER, ucols)):
        u0 = units[k]
        axb.plot([offs[i] - u0.b, offs[i] + u0.b], [-0.055, -0.055],
                 color="#4b5468", lw=1.2)
        for seg in transform(u0, offs[i] + X[j, i], TH[j, i]):
            axb.plot(seg[:, 0], seg[:, 1], color=col, lw=1.4)
        axb.text(offs[i], -0.30, UNIT_LABEL[k], color=col, fontsize=7.5,
                 ha="center", va="top")
        axb.text(offs[i], u0.H + 0.22, f"{X[j, i]:+.2f} m", color=col,
                 fontsize=7, ha="center")
    scale_bar(axb, 1.0, "1 m", -XZOOM + 0.35, 1.34)
    axb.set_xlim(-XZOOM, XZOOM)
    axb.set_ylim(-0.55, 1.6)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_visible(False)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

print(write_gif("anim01_wavefield", frames, fps=16, colors=128))
