"""Write the plain-text reports from the cached results of the figures.

Run after the figure scripts.  Every number quoted in the manuscript
should appear in one of these files.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np

from reefunit.core import PRM, near_bed_amplitude, URSELL_MAX
from reefunit.forcing import isw_params
from reefunit.io_utils import fmt, load_cache, write_report
from reefunit.scenario import (COEF_BOUNDS, COLONIES, ISW_REF, R0_COLONY,
                               R_GROWN, SITES, SITE_ORDER, SPC,
                               UNIT_LABEL, UNIT_ORDER, MATERIAL_ORDER,
                               sample_coefficients)
from reefunit.statics import holddown, theta_onsets
from reefunit.units import MATERIALS, batch, catalogue, grown

units = {u.name: u for u in catalogue()}
C = {k: load_cache(k) for k in ("fig02", "fig03", "fig04", "fig06",
                                "fig07", "verification")}
paths = []


def row(*cells):
    return "".join(str(c).ljust(w) for c, w in
                   zip(cells, (20, 14, 13, 13, 13, 13, 13)))


# --------------------------------------------------------------- units.txt
lines = [row("unit", "material", "mass_kg", "W_sub_N", "b_m", "z_p_m",
             "h_c_m")]
for k in UNIT_ORDER:
    u = units[k]
    lines.append(row(k, u.material, f"{u.m:.2f}", f"{u.Wsub:.1f}",
                     f"{u.b:.3f}", f"{u.z_p:.3f}", f"{u.h_c:.3f}"))
lines.append("")
lines.append(row("unit", "C_D A m2", "C_L A m2", "k^2 m2", "alpha_deg",
                 "b/z_p", "H_m"))
for k in UNIT_ORDER:
    u = units[k]
    lines.append(row(k, f"{u.CDA:.4f}", f"{u.CLA:.4f}", f"{u.k2:.4f}",
                     f"{np.degrees(u.alpha):.1f}", f"{u.b / u.z_p:.2f}",
                     f"{u.H:.2f}"))
lines.append("")
lines.append("with reference colonies at radius "
             f"{R_GROWN} m, top mounted:")
lines.append(row("unit", "n_colonies", "C_D A m2", "z_p m", "b/z_p",
                 "W_sub N", "mass kg"))
for k in UNIT_ORDER:
    g = grown(units[k], R_GROWN)
    lines.append(row(k, COLONIES[k][0], f"{g.CDA:.4f}", f"{g.z_p:.3f}",
                     f"{g.b / g.z_p:.2f}", f"{g.Wsub:.1f}", f"{g.m:.1f}"))
mat = [f"{k}: {v:.0f} kg m^-3" for k, v in MATERIALS.items()]
paths.append(write_report(
    "units", "Reference units, geometry and derived mechanical data",
    [("Bare units", lines[:len(UNIT_ORDER) * 2 + 3]),
     ("Grown units", lines[len(UNIT_ORDER) * 2 + 3:]),
     ("Materials", mat),
     ("Provenance and rescaling", [
         "The dimensions below are a parameter set, not measurements of a",
         "particular design, and any statement about a named structure is",
         "a statement about these numbers.  In the drag-dominated limit",
         "the quasi-static thresholds depend on the geometry only through",
         "W'/(C_D A) and z_p/b, and the critical velocity scales as",
         "sqrt(W'/(C_D A)), so a reader with their own unit can rescale",
         "the results rather than rerun them."]),
     ("Notes", [
         "Dimensions and coefficients are illustrative of units deployed",
         "on Indonesian reefs and are not measurements of a particular",
         "design.  Frames are built from rods in three dimensions and",
         "their mass, inertia, frontal area and centre of pressure follow",
         "from that geometry; the block and dome use closed forms.",
         "b is the half-distance between the base contacts that act as",
         "pivots; for the hexagonal spider it is the apothem of the foot",
         "ring and for the tetrahedral frame the inradius of the base",
         "triangle, the conservative choice in each case."])]))

# ---------------------------------------------------------- parameters.txt
site_lines = [row("site class", "H_des m", "T s", "U_c m/s", "depth m")]
for k in SITE_ORDER:
    s = SITES[k]
    site_lines.append(row(k, f"{s['H']:.2f}", f"{s['T']:.1f}",
                          f"{s['Uc']:.2f}", f"{s['depth']:.1f}"))
p_isw = isw_params(ISW_REF["h1"], ISW_REF["h2"], ISW_REF["drho"], 1.0)
isw_lines = [
    f"upper layer h1 = {ISW_REF['h1']:.1f} m, lower layer h2 = "
    f"{ISW_REF['h2']:.1f} m, total depth "
    f"{ISW_REF['h1'] + ISW_REF['h2']:.1f} m",
    f"density step {ISW_REF['drho']:.1f} kg m^-3, reduced gravity "
    f"{p_isw['gp']:.5f} m s^-2",
    f"linear speed c0 = {p_isw['c0']:.4f} m s^-1",
    f"nonlinear coefficient a1 = {p_isw['a1']:.4f} s^-1",
    f"dispersion coefficient beta = {p_isw['beta']:.3f} m^3 s^-1",
    f"at eta0 = 1 m: V = {p_isw['V']:.4f} m s^-1, lam = "
    f"{p_isw['lam']:.2f} m, near-bed peak "
    f"{p_isw['c0'] / ISW_REF['h2']:.4f} m s^-1",
    "the lower layer is thinner than the upper one, so a1 > 0 and the",
    "soliton is one of elevation, as expected for a wave that has",
    "shoaled onto a shelf",
]
coef_lines = [f"{k}: {v[0]} to {v[1]}" for k, v in COEF_BOUNDS.items()]
paths.append(write_report(
    "parameters", "Model parameters and reference conditions",
    [("Fluid", [f"seawater density {PRM.rho:.0f} kg m^-3",
                f"gravity {PRM.g:.2f} m s^-2",
                f"depth-limited breaking index {PRM.gamma_b:.2f}",
                f"Stokes second order used below Ursell {URSELL_MAX:.0f}"]),
     ("Site classes (illustrative design conditions)", site_lines),
     ("Colonies", [
         f"initial radius {R0_COLONY:.3f} m, reference grown radius "
         f"{R_GROWN:.3f} m",
         "bulk density 1500 kg m^-3 over a solid fraction of 0.5",
         "drag coefficient 1.0, frontal area pi r^2/2 per colony",
         "counts and mounting heights (fraction of unit height):",
         *[f"  {k}: n = {v[0]}, top {v[1]:.2f} H, side {v[2]:.2f} H"
           for k, v in COLONIES.items()]]),
     ("Internal solitary wave reference state", isw_lines),
     ("Uncalibrated coefficient ranges", coef_lines),
     ("Numerics", [
         f"time steps per wave period: {SPC} for production runs",
         "Moreau-Jean theta = 1/2, projected Gauss-Seidel on the",
         "Delassus operator, 20 to 30 sweeps per step",
         "contacts count as closed within 1e-8 m",
         "normal restitution e = 0 at the base contacts"])]))

# -------------------------------------------------------- closed_forms.txt
ts, tt, tl = theta_onsets(batch([units[k] for k in UNIT_ORDER]))
on_lines = [row("unit", "Theta_slide", "Theta_tip", "Theta_lift",
                "first mode")]
for i, k in enumerate(UNIT_ORDER):
    on_lines.append(row(k, f"{ts[i]:.3f}", f"{tt[i]:.3f}",
                        f"{tl[i]:.3f}" if np.isfinite(tl[i]) else "inf",
                        "slide" if ts[i] < tt[i] else "tip"))
paths.append(write_report(
    "closed_forms", "Closed-form results used and checked",
    [("Quasi-static onsets, drag-dominated limit", [
        "Theta = F_D/W' and Lambda = C_L A_L/(C_D A):",
        "  Theta_slide = mu/(1 + mu Lambda)",
        "  Theta_tip   = 1/(z_p/b + Lambda)",
        "  Theta_lift  = 1/Lambda",
        "sliding precedes tipping if and only if b/z_p > mu, a",
        "comparison from which lift cancels identically", "",
        *on_lines]),
     ("Ratchet drift near threshold", [
         "with the load peaking at F = mu N (1 + eps) and curvature",
         "F'' = -F c omega^2 there, the slip episode runs from -tau0 to",
         "2 tau0 with tau0 = sqrt(2 eps/kappa), kappa = (1+eps) c omega^2,",
         "and the displacement is",
         "  Delta_0 = (9/2) (mu N/m_x) eps^2 / kappa",
         "for pure drag on u = U_c + u_b cos(omega t), c = 2 u_b/(U_c+u_b)",
         "drag on the relative velocity adds a damping",
         "gamma = rho C_D A |u_p|/m_x and, to first order in",
         "Gamma = gamma tau0,",
         "  Delta = Delta_0 (1 - (6/5) Gamma + O(Gamma^2))"]),
     ("Impact map", [
         "angular velocity across a pivot change at theta = 0:",
         "  ratio = (I_G + m_x (h_c^2 - b^2))/(I_G + m_x R^2)",
         "which for a solid rectangle in vacuum is Housner's",
         "  1 - (3/2) sin^2(alpha)"]),
     ("Two-layer KdV soliton", [
         "eta = eta0 sech^2((x - V t)/lam),  V = c0 + a1 eta0/3,",
         "lam^2 = 12 beta/(a1 eta0), near-bed velocity u2 = c0 eta/h2"])]))

# -------------------------------------------------------- verification.txt
V = C["verification"]
v_lines = []
if V is not None:
    v_lines = [
        f"restitution, max |simulated - closed form|: "
        f"{fmt(float(V['restitution_max_abs'][0]))}",
        f"dispersion relation residual: "
        f"{fmt(float(V['dispersion_residual'][0]))}",
        f"KdV soliton residual (relative): "
        f"{fmt(float(V['kdv_residual'][0]))}",
        f"compiled kernel vs NumPy reference, max |dX|: "
        f"{fmt(float(V['kernel_vs_numpy_X'][0]))}",
        f"compiled kernel vs NumPy reference, max |dtheta|: "
        f"{fmt(float(V['kernel_vs_numpy_theta'][0]))}",
        f"contact complementarity residual (worst step): "
        f"{fmt(float(V['contact_residual'][0]))}",
        "",
        "free rocking, time to first impact "
        f"(reference {float(V['free_rock_t_ref'][0]):.6f} s):",
        "  errors " + ", ".join(fmt(x, 3) for x in V["free_rock_err"]),
        "  the error flattens near 1e-5 s, a floor set by the closed-gap",
        "  tolerance rather than by the step size",
        "stick-slip drift per cycle "
        f"(reference {float(V['slide_ref'][0]):.6f} m):",
        "  errors " + ", ".join(fmt(x, 3) for x in V["slide_err"]),
        "rocking peak rotation "
        f"(reference {float(V['rock_ref'][0]):.6f} rad):",
        "  errors " + ", ".join(fmt(x, 3) for x in V["rock_err"]),
    ]
if C["fig02"] is not None:
    v_lines += [
        "",
        "slow-ramp onset against the closed forms:",
        f"  max relative error {fmt(float(C['fig02']['theta_rel_max'][0]))}",
        f"  median relative error "
        f"{fmt(float(C['fig02']['theta_rel_median'][0]))}",
        f"  agreement on which mode comes first: "
        f"{float(C['fig02']['mode_agree'][0]) * 100:.1f} percent"]
if C["fig04"] is not None:
    v_lines += [
        "event-driven sliding reference against the time-stepper:",
        f"  max relative difference in drift per cycle "
        f"{fmt(float(C['fig04']['ev_rel'][0]))}"]
paths.append(write_report(
    "verification", "Verification of the solvers",
    [("Residuals and cross-checks", v_lines),
     ("What is verified", [
         "the contact solver reproduces the angular-momentum impact map",
         "and, in vacuum for a solid rectangle, Housner's restitution",
         "the time-stepper converges to two independent event-driven",
         "solvers that share no code with it, one for pure sliding and",
         "one for single-pivot rocking",
         "onsets under a slowly ramped current reproduce the closed-form",
         "thresholds and the predicted first mode",
         "the compiled kernel and the NumPy reference agree to round-off",
         "the wave and soliton kinematics satisfy their own equations"]),
     ("What is not verified", [
         "nothing here is validated against field or laboratory data;",
         "the force coefficients, colony properties and site classes are",
         "assumptions, and the results are relative rather than absolute"])]))

# ------------------------------------------------------------ results.txt
res = []
if C["fig03"] is not None:
    M = C["fig03"]["material_matrix"]
    res.append("critical wave height at the monsoon reference condition")
    res.append(row("unit", *[m[:10] for m in MATERIAL_ORDER]))
    for i, k in enumerate(UNIT_ORDER):
        res.append(row(k, *[f"{M[i, j]:.2f}" for j in
                            range(len(MATERIAL_ORDER))]))
    res.append("material enters the quasi-static criteria only through")
    res.append("the submerged weight, so H_c scales as sqrt(rho_s - rho)")
    res.append("at fixed geometry in the drag-dominated limit")
if C["fig04"] is not None:
    f5 = C["fig04"]
    res += ["",
            "ratchet drift, concrete block, T = 8 s, u_b = 1.8 m/s:",
            f"  drift per cycle at U_c/u_b = 0.2: "
            f"{np.interp(0.2, f5['ratios'], f5['d_rel']) * 100:.2f} cm",
            f"  drift per cycle at U_c/u_b = 0.4: "
            f"{np.interp(0.4, f5['ratios'], f5['d_rel']) * 100:.2f} cm",
            "  relative-velocity drag reduces the drift by the factor",
            f"  1 - (6/5) Gamma, verified for Gamma up to "
            f"{f5['Gamma'].max():.2f}"]
if C["fig06"] is not None:
    f9 = C["fig06"]
    res += ["", "H_c over the monsoon design height, as deployed and after",
            "five years of colony growth at 0.03 m per year:"]
    res.append(row("unit", "t = 0", "t = 5 yr", "staked t = 5 yr"))
    for k in UNIT_ORDER:
        res.append(row(k, f"{f9['free_' + k][0]:.2f}",
                       f"{f9['free_' + k][-1]:.2f}",
                       f"{f9['staked_' + k][-1]:.2f}"))
if C["fig07"] is not None:
    f7 = C["fig07"]
    res += ["",
            "sensitivity to the three weakest assumptions:",
            f"  frame sheltering at 0.6 raises H_c of the frames by up to "
            f"{100 * (f7['ratio'].max() - 1):.0f} percent",
            "  interquartile spread of H_c under all assumptions together,",
            "  as a fraction of the median: "
            + ", ".join(f"{k} {v:.2f}" for k, v in
                        zip(UNIT_ORDER, f7["Hc_iqr"])),
            f"  the ranking by H_c is reproduced exactly in "
            f"{100 * float(f7['exact'][0]):.1f} percent of 4000 samples",
            f"  applying added mass to the rotation as well as to the",
            f"  translations moves the overturning threshold by "
            f"{100 * (f7['ub_c'].max() / f7['ub_c'].min() - 1):.1f} "
            f"percent"]
paths.append(write_report("results", "Principal results", [
    ("Numbers quoted in the manuscript", res)]))

# --------------------------------------------------------- field_table.txt
rng = np.random.default_rng(3)
coefs = sample_coefficients(240, rng)
depths = [3.0, 5.0, 8.0, 12.0]
ft = [row("unit", "depth m", "F_req N", "ballast kg", "mode"),
      "(90th percentile over the coefficient ranges, safety factor 1.5)"]
for site_name in SITE_ORDER:
    s = SITES[site_name]
    ft.append("")
    ft.append(f"site class: {site_name}  (H = {s['H']} m, T = {s['T']} s,"
              f" U_c = {s['Uc']} m/s)")
    for k in UNIT_ORDER:
        for h in depths:
            H = min(s["H"], PRM.gamma_b * h)
            ub = near_bed_amplitude(H, s["T"], h)
            d = batch([units[k]] * 240)
            d["CDA"] = d["CDA"] * coefs["CD_factor"]
            d["CLA"] = d["CLA"] * coefs["CL_factor"]
            d["mu"] = coefs["mu"]
            d["C_a"] = coefs["C_a"]
            d["FKV"] = PRM.rho * d["V"] * (1 + d["C_a"])
            F = holddown(d, ub, 2 * np.pi / s["T"], s["Uc"], sf=1.5)
            f90 = float(np.percentile(F, 90))
            ts_, tt_, _ = theta_onsets(d)
            mode = "slide" if np.median(ts_) < np.median(tt_) else "tip"
            ft.append(row(k, f"{h:.0f}", f"{f90:.0f}",
                          f"{f90 / PRM.g:.0f}", mode))
paths.append(write_report(
    "field_table",
    "Hold-down requirement per unit (tabel kebutuhan pemberat)",
    [("Required hold-down force and equivalent submerged ballast mass",
      ft),
     ("How to read it / Cara membaca", [
         "F_req is the extra downward force, applied through the base,",
         "that keeps every quasi-static ratio at or below 1/1.5 under the",
         "design wave of that site class at that depth; the ballast mass",
         "is F_req/g, an effective submerged mass.",
         "",
         "F_req adalah gaya tekan ke bawah tambahan pada dasar unit yang",
         "menjaga rasio kuasi-statik tetap di bawah 1/1,5 pada gelombang",
         "rencana kelas lokasi tersebut; massa pemberat adalah F_req/g,",
         "yaitu massa terendam efektif.",
         "",
         "The table is a screening aid, not a design code.  It assumes a",
         "rigid bed, non-breaking waves, and the stated coefficient",
         "ranges, and it says nothing about coral survival."])]))

# ------------------------------------------------------- figure_notes.txt
notes = [
    "fig00 schematic of the five units and the planar load model",
    "fig01 near-bed forcing of the site classes and of a soliton",
    "fig02 failure-mode plane and onset under a ramped current",
    "fig03 critical wave height over depth and period, and the",
    "      structure by material matrix",
    "fig04 ratchet drift, damping correction and drift rate",
    "fig05 soliton amplitude, margin consumed, displacement per passage",
    "fig06 growth of colonies against the design wave",
    "fig07 sensitivity to the three weakest assumptions",
    "fig08 verification",
    "",
    "anim01 the orbital velocity field and the five units on the bed",
    "anim02 a block walking under waves on a current",
    "anim03 a two-layer soliton passing over a hexagonal spider",
    "anim04 the critical-height map as colonies grow",
]
paths.append(write_report("figure_notes", "Figures and animations", [
    ("Contents", notes),
    ("Reproduction", [
        "each figure script writes its panels' data to outputs/data as",
        "CSV and its derived numbers to outputs/cache; the reports are",
        "written from those caches by scripts/make_reports.py",
        "the checks summarised in verification.txt also run as a pytest",
        "suite in tests/, which takes a few seconds",
        "the animations are rendered from the same solvers and carry a",
        "colour bar in SI units; anim04 caches its maps, so a rerun is",
        "fast once they exist"])]))

# --------------------------------------------------------- open_items.txt
paths.append(write_report("open_items", "Limitations and open items", [
    ("Assumptions that bound the results", [
        "planar motion: oblique and directionally spread waves, rocking",
        "about a corner, and yaw are not represented; head-on loading is",
        "the worst case for a given height in this model but not in three",
        "dimensions",
        "rigid, flat bed: scour, burial, rubble mobility and bearing",
        "failure are absent, and these are common on degraded reefs",
        "non-breaking waves: the Morison description is not used inside",
        "the surf zone, so reef flats and crests are out of scope",
        "force coefficients are ranges, not measurements; the frontal",
        "area is held fixed as the unit rotates",
        "frame drag sums the rods with no sheltering, and added mass is",
        "applied isotropically by default; both are varied in fig07, and",
        "the ranking of the units survives, though the frame thresholds",
        "move by up to a quarter",
        "colonies are hemispheres at a single mounting height; merging",
        "colonies, canopy flow and colony breakage are not modelled, and",
        "colony radius is capped at 0.10 m for that reason",
        "colony weight and colony drag both grow with radius and partly",
        "cancel; which one wins depends on the assumed colony density and",
        "solid fraction, which are uncalibrated"]),
    ("Results that are weaker than they look", [
        "mounting height changes nothing while sliding governs, which is",
        "the case for four of the five reference geometries, so the",
        "top-versus-side comparison is informative only for staked units",
        "the quasi-static criterion is conservative for wave loading: a",
        "wave must exceed it by a margin before the dynamic problem",
        "actually overturns a unit"]),
    ("Natural next steps", [
        "three-dimensional rocking and oblique seas",
        "linked units: webs of spiders as coupled nonsmooth oscillators",
        "design optimisation over a continuous family of geometries",
        "scour and soft-bed settlement",
        "force coefficients for porous frames with grown colonies from",
        "computational fluid dynamics, which would replace the widest",
        "uncertainty range in the model",
        "a field screening tool and a printed chart, checked with two or",
        "three restoration teams before publication"])]))

for p in paths:
    print(p)
