"""Figure 8. Verification.

(a) Angular-velocity ratio across an impact at theta = 0 in vacuum, from
the time-stepper, against the closed form (I_G + m(h_c^2 - b^2))/(I_G +
m R^2), which for a solid rectangle is Housner's 1 - (3/2) sin^2 alpha.
(b) Free rocking in vacuum: error in the time to the first impact against
step size, with the exact nonlinear pivot equation integrated by DOP853
as reference.  The error falls with step size and then flattens near
10^-5 s, a floor set by the tolerance that decides when a contact counts
as closed rather than by the step.
(c) Stick-slip under waves on a current: error of the drift per cycle
against step size, with the event-driven reference of the reduced sliding
model.
(d) Rocking under waves with sliding suppressed: error of the peak
rotation against step size, with the event-driven single-pivot model and
its angular-momentum impact map as reference.
Residuals of the dispersion relation, of the KdV soliton, of the contact
solver, and of the two implementations of the time-stepper are in the
verification report.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import PRM, wavenumber
from reefunit.dynamics import simulate
from reefunit.forcing import Harmonic, isw_params
from reefunit.io_utils import write_csv, save_cache
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               outside_legend, handles, save)
from reefunit.reduced import free_rock_time, restitution, rock1p, slide1d
from reefunit.scenario import R_GROWN, SPC
from reefunit.units import batch, block, grown, table

setup()
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3))
V = {}

# --------------------------------------------------------- (a) restitution
blocks = [block(width=w, height=0.5, depth=0.4) for w in
          np.linspace(0.05, 0.6, 12)]
d = batch(blocks, fluid=False, mu=5.0)
alpha = np.arctan2(d["b"], d["h_c"])
th0 = -0.5 * alpha
st = (np.zeros(len(blocks)), np.zeros(len(blocks)), th0,
      np.zeros(len(blocks)), np.zeros(len(blocks)), np.zeros(len(blocks)))
c, s = np.cos(th0), np.sin(th0)
rx = c * d["b"] + s * d["h_c"]
ry = s * d["b"] - c * d["h_c"]
st = (d["b"] - rx, -ry, th0, np.zeros(len(blocks)), np.zeros(len(blocks)),
      np.zeros(len(blocks)))
h = 2e-5
r = run(d, forcing_spec(len(blocks), ub=0.0, omega=1.0), h, 60000,
        state=st, record_every=20)
rat = []
for i in range(len(blocks)):
    th = r.rec_th[:, i]
    w = np.gradient(th, 20 * h)
    k = np.where((th[:-1] < 0) & (th[1:] >= 0))[0][0]
    rat.append(w[k + 3] / w[k - 3])
rat = np.array(rat)
exact = np.array([restitution(d, i) for i in range(len(blocks))])
axs[0].plot(np.degrees(alpha), exact, color=C["grey"], lw=1.0)
axs[0].plot(np.degrees(alpha), rat, "o", color=C["blue"], ms=3.5,
            mfc="none")
axs[0].set_xlabel(r"$\alpha$ (deg)")
axs[0].set_ylabel(r"$\dot\theta^+/\dot\theta^-$")
panel_label(axs[0], "(a)")
V["restitution_max_abs"] = np.max(np.abs(rat - exact))
write_csv("fig08a_restitution", {"alpha_deg": np.degrees(alpha),
                                 "sim": rat, "closed_form": exact})

# ------------------------------------------------------ (b) free rocking
one = batch([block(width=0.3, height=0.5, depth=0.3)], fluid=False, mu=5.0)
a1 = float(np.arctan2(one["b"], one["h_c"])[0])
t_ref, w_ref = free_rock_time(one, 0.5 * a1)
hs = np.array([8e-4, 4e-4, 2e-4, 1e-4, 5e-5])
err_b = []
for hh in hs:
    st1 = None
    c, s = np.cos(-0.5 * a1), np.sin(-0.5 * a1)
    rx = c * one["b"] + s * one["h_c"]
    ry = s * one["b"] - c * one["h_c"]
    st1 = (one["b"] - rx, -ry, np.array([-0.5 * a1]), np.zeros(1),
           np.zeros(1), np.zeros(1))
    n = int(1.5 * t_ref / hh)
    rr = run(one, forcing_spec(1, ub=0.0, omega=1.0), hh, n, state=st1,
             record_every=1)
    th = rr.rec_th[:, 0]
    k = np.where((th[:-1] < 0) & (th[1:] >= 0))[0][0]
    t_hit = np.interp(0.0, [th[k], th[k + 1]], [(k + 1) * hh, (k + 2) * hh])
    err_b.append(abs(t_hit - t_ref))
err_b = np.array(err_b)
axs[1].loglog(hs, err_b, "o", color=C["blue"], ms=3.5, mfc="none")
axs[1].loglog(hs, err_b[0] * hs / hs[0], color=C["grey"], ls="--", lw=0.8)
axs[1].set_xlabel(r"$h$ (s)")
axs[1].set_ylabel(r"error in $t_{\rm impact}$ (s)")
panel_label(axs[1], "(b)")
V["free_rock_t_ref"] = t_ref
V["free_rock_err"] = err_b
write_csv("fig08b_free_rock", {"h_s": hs, "error_s": err_b})

# ---------------------------------------------------------- (c) stick-slip
blk = batch([block()], CLA=0.0)
T, om, ub, Uc = 8.0, 2 * np.pi / 8.0, 1.8, 0.4
F = Harmonic(ub, om, Uc=Uc)
ts, xs, _ = slide1d(blk, lambda t: (float(F.at(t, 0)[0]),
                                    float(F.at(t, 0)[1])), 12 * T)
x_ref = np.diff(np.interp([6 * T, 12 * T], ts, xs))[0] / 6
spcs = np.array([250, 500, 1000, 2000, 4000])
err_c = []
for spc in spcs:
    rr = run(blk, forcing_spec(1, ub=ub, omega=om, Uc=Uc), T / spc,
             12 * spc, record_every=spc)
    err_c.append(abs((rr.rec_X[11, 0] - rr.rec_X[5, 0]) / 6 - x_ref))
err_c = np.array(err_c)
hc = T / spcs
axs[2].loglog(hc, err_c, "o", color=C["blue"], ms=3.5, mfc="none")
axs[2].loglog(hc, err_c[0] * hc / hc[0], color=C["grey"], ls="--", lw=0.8)
axs[2].set_xlabel(r"$h$ (s)")
axs[2].set_ylabel(r"error in drift (m)")
panel_label(axs[2], "(c)")
V["slide_ref"] = x_ref
V["slide_err"] = err_c
write_csv("fig08c_stick_slip", {"h_s": hc, "error_m": err_c})

# ------------------------------------------------------------- (d) rocking
gt = batch([grown(table(), R_GROWN)], mu=3.0)
ubr = 0.90
Fr = Harmonic(ubr, om)
ts, th, _ = rock1p(gt, lambda t: (float(Fr.at(t, 0)[0]),
                                  float(Fr.at(t, 0)[1])), 10 * T)
th_ref = np.abs(th).max()
spcs_d = np.array([500, 1000, 2000, 4000, 8000])
hd = T / spcs_d
err_d = []
for spc in spcs_d:
    rr = run(gt, forcing_spec(1, ub=ubr, omega=om), T / spc, 10 * spc)
    err_d.append(abs(rr.max_th[0] - th_ref))
err_d = np.array(err_d)
axs[3].loglog(hd, err_d, "o", color=C["blue"], ms=3.5, mfc="none")
axs[3].loglog(hd, err_d[0] * hd / hd[0], color=C["grey"], ls="--", lw=0.8)
axs[3].set_xlabel(r"$h$ (s)")
axs[3].set_ylabel(r"error in $\max|\theta|$ (rad)")
panel_label(axs[3], "(d)")
V["rock_ref"] = th_ref
V["rock_err"] = err_d
write_csv("fig08d_rocking", {"h_s": hd, "error_rad": err_d})

# ------------------------------------------- residuals for the report only
om_t = 2 * np.pi / np.array([3.0, 8.0, 14.0, 20.0])
h_t = np.array([2.0, 8.0, 15.0, 40.0])[:, None]
k_t = wavenumber(om_t, h_t)
V["dispersion_residual"] = np.max(np.abs(
    PRM.g * k_t * np.tanh(k_t * h_t) / om_t ** 2 - 1.0))

p = isw_params(12.0, 3.0, 5.0, 1.0)
L = 400.0
x = np.linspace(-L / 2, L / 2, 4096, endpoint=False)
kx = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
eta = 1.0 / np.cosh(x / p["lam"]) ** 2


def dx(f, n=1):
    return np.real(np.fft.ifft((1j * kx) ** n * np.fft.fft(f)))


res = (-p["V"] * dx(eta) + p["c0"] * dx(eta) + p["a1"] * eta * dx(eta)
       + p["beta"] * dx(eta, 3))
scale = max(np.abs(p["c0"] * dx(eta)).max(),
            np.abs(p["beta"] * dx(eta, 3)).max())
V["kdv_residual"] = np.max(np.abs(res)) / scale

d2 = batch([block(), grown(table(), R_GROWN)])
f2 = forcing_spec(2, ub=[1.9, 1.0], omega=om)
r_k = run(d2, f2, T / SPC, 8 * SPC, n_iter=20)
s_n, _ = simulate(d2, Harmonic(np.array([1.9, 1.0]), om), T / SPC, 8 * SPC,
                  n_iter=20)
V["kernel_vs_numpy_X"] = np.max(np.abs(r_k.X - s_n.X))
V["kernel_vs_numpy_theta"] = np.max(np.abs(r_k.th - s_n.th))
V["contact_residual"] = r_k.max_res.max()
save_cache("verification", **{k: np.atleast_1d(v) for k, v in V.items()})
for k, v in V.items():
    print(k, np.round(np.atleast_1d(v), 12))

fig.tight_layout(w_pad=1.4)
h1, l1 = handles(["time-stepper", "closed form or reference", "slope 1"],
                 [C["blue"], C["grey"], C["grey"]], ["", "-", "--"],
                 ["o", None, None])
outside_legend(fig, h1, l1, ncol=3, y=0.02)
print(save(fig, "fig08_verification"))
