"""Figure 7. Sensitivity to the three weakest modelling assumptions.

The model rests on three choices that cannot be checked against the
closed forms, because the reference solvers share them: drag on a frame
taken as the sum of its rods with no sheltering, added mass applied to
the rotation as well as to the translations, and a colony drag
coefficient and bulk density that are assumed rather than measured.

(a) Critical wave height of each unit at the monsoon reference condition
against a sheltering factor applied to the frontal area of the rod
frames, relative to the value at no sheltering.  The block and the dome
are solid and are not affected.
(b) Rank of each unit by critical wave height over 4000 samples in which
the sheltering factor, the colony drag coefficient, the colony bulk
density and the four uncalibrated coefficients are drawn together from
their ranges.  A unit whose bar is concentrated on one rank keeps its
place whatever the assumptions.
(c) Overturning threshold of the staked grown rebar table at T = 8 s
against the fraction of the added mass that is also applied to the
rotation, from 0 (translations only) to 1 (isotropic, the default), on
axes spanning plus or minus ten percent.  The threshold moves by less
than one percent, so the choice barely matters here; the residual
wiggle is the resolution of the bisection.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefunit.core import PRM
from reefunit.io_utils import write_csv, save_cache
from reefunit.kernel import forcing_spec, run
from reefunit.plotting import (setup, OKABE_ITO as C, panel_label,
                               panel_legends, handles, save)
from reefunit.scenario import (COLONIES, R_GROWN, SITES, SPC, UNIT_ORDER,
                               UNIT_LABEL, sample_coefficients)
from reefunit.statics import crit_height
from reefunit.units import catalogue, batch, grown, with_colonies

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
units = {u.name: u for u in catalogue()}
site = SITES["monsoon"]
ucols = [C["black"], C["blue"], C["vermil"], C["green"], C["purple"]]
FRAMES = {"table", "spider", "tetra"}
SHELTER = (0.6, 1.0)
COLONY_CD = (0.6, 1.6)
COLONY_RHO = (1200.0, 1900.0)

# --------------------------------------------------- (a) frame sheltering
sig = np.linspace(SHELTER[0], 1.0, 25)
tab = {"shelter_factor": sig}
for name, col in zip(UNIT_ORDER, ucols):
    d = batch([units[name]] * len(sig))
    if name in FRAMES:
        d["CDA"] = d["CDA"] * sig
    Hc = crit_height(d, site["T"], site["depth"], stokes=True, cap=False)
    tab[name] = Hc / Hc[-1]
    axs[0].plot(sig, Hc / Hc[-1], color=col)
axs[0].set_xlim(sig[0], 1.0)
axs[0].set_xlabel("frame sheltering factor")
axs[0].set_ylabel(r"$H_c/H_c(\sigma=1)$")
panel_label(axs[0], "(a)")
write_csv("fig07a_sheltering", tab)

# ------------------------------------------------------ (b) rank stability
NS = 4000
rng = np.random.default_rng(17)
coef = sample_coefficients(NS, rng)
sh = rng.uniform(*SHELTER, NS)
cd_c = rng.uniform(*COLONY_CD, NS)
rho_c = rng.uniform(*COLONY_RHO, NS)
Hc = np.zeros((len(UNIT_ORDER), NS))
for i, name in enumerate(UNIT_ORDER):
    n, ftop, _ = COLONIES[name]
    z = ftop * units[name].H
    base = units[name]
    # colony drag and density enter through the grown unit, so build the
    # grown unit for each distinct pair by sampling in blocks
    order = np.argsort(cd_c * 1e4 + rho_c)
    us = [with_colonies(base, n, R_GROWN, z, rho_c=rho_c[j],
                        C_Dc=cd_c[j]) for j in order]
    d = batch(us)
    inv = np.empty(NS, int)
    inv[order] = np.arange(NS)
    d = {k: v[inv] for k, v in d.items()}
    d["CDA"] = d["CDA"] * coef["CD_factor"] * (sh if name in FRAMES else 1)
    d["CLA"] = d["CLA"] * coef["CL_factor"]
    d["C_a"] = coef["C_a"]
    d["mu"] = coef["mu"]
    d["mx"] = d["m"] + d["C_a"] * PRM.rho * d["V"]
    d["IG"] = d["mx"] * d["k2"]
    d["FKV"] = PRM.rho * d["V"] * (1 + d["C_a"])
    Hc[i] = crit_height(d, site["T"], site["depth"], stokes=True,
                        cap=False)
ranks = np.argsort(np.argsort(-Hc, axis=0), axis=0) + 1
freq = np.stack([[np.mean(ranks[i] == r) for r in range(1, 6)]
                 for i in range(len(UNIT_ORDER))])
bottom = np.zeros(5)
x = np.arange(1, 6)
for i, (name, col) in enumerate(zip(UNIT_ORDER, ucols)):
    axs[1].bar(x, freq[i], 0.7, bottom=bottom, color=col, lw=0)
    bottom += freq[i]
axs[1].set_xticks(x)
axs[1].set_xlabel("rank by $H_c$ (1 = most stable)")
axs[1].set_ylabel("fraction of samples")
axs[1].tick_params(top=False, right=False)
panel_label(axs[1], "(b)")
write_csv("fig07b_rank_frequency", {
    "rank": x.astype(float),
    **{UNIT_ORDER[i]: freq[i] for i in range(len(UNIT_ORDER))}})
nominal = np.argsort(np.argsort(-np.median(Hc, axis=1))) + 1
exact = np.mean(np.all(ranks == nominal[:, None], axis=0))
print(f"median ranking {dict(zip(UNIT_ORDER, nominal.tolist()))}")
print(f"fraction of samples reproducing it exactly: {exact:.3f}")
print("H_c interquartile spread as a fraction of the median:",
      np.round((np.percentile(Hc, 75, axis=1)
                - np.percentile(Hc, 25, axis=1))
               / np.median(Hc, axis=1), 3).tolist())

# ------------------------------------------- (c) rotational added mass
gt = grown(units["table"], R_GROWN)
T = 8.0
lam = np.linspace(0.0, 1.0, 11)
lo = np.full(len(lam), 0.5)
hi = np.full(len(lam), 2.0)
for _ in range(16):
    mid = 0.5 * (lo + hi)
    d = batch([gt] * len(lam), mu=0.9, rot_am=lam)
    r = run(d, forcing_spec(len(lam), ub=mid, omega=2 * np.pi / T),
            T / SPC, 12 * SPC)
    lo = np.where(r.over, lo, mid)
    hi = np.where(r.over, mid, hi)
ub_c = 0.5 * (lo + hi)
axs[2].plot(lam, ub_c, color=C["vermil"])
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0.9 * ub_c.mean(), 1.1 * ub_c.mean())
axs[2].set_xlabel("rotational added mass")
axs[2].set_ylabel(r"overturning $u_b$ (m s$^{-1}$)")
panel_label(axs[2], "(c)")
write_csv("fig07c_rotational_added_mass", {"fraction": lam,
                                           "ub_overturn": ub_c})
print(f"overturning threshold varies from {ub_c[0]:.3f} to "
      f"{ub_c[-1]:.3f} m/s, a spread of "
      f"{100 * (ub_c.max() / ub_c.min() - 1):.1f} percent")
save_cache("fig07", shelter=sig, ratio=np.array([tab[k] for k in
                                                 UNIT_ORDER]),
           rank_freq=freq, nominal=nominal, exact=np.array([exact]),
           Hc_iqr=(np.percentile(Hc, 75, axis=1)
                   - np.percentile(Hc, 25, axis=1)) / np.median(Hc,
                                                                axis=1),
           lam=lam, ub_c=ub_c)

fig.tight_layout(w_pad=1.4)
h1, l1 = handles([UNIT_LABEL[k] for k in UNIT_ORDER], ucols)
h2, l2 = handles(["staked grown table"], [C["vermil"]])
panel_legends(fig, [((axs[0], axs[1]), h1, l1, 5), (axs[2], h2, l2, 1)])
print(save(fig, "fig07_sensitivity"))
