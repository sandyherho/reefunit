"""Fast checks of the physics and the solvers.

Each test is a quick version of a check reported in
outputs/reports/verification.txt.  The suite runs in under a minute after
the kernel has compiled.
"""

import numpy as np
import pytest

from reefunit.core import PRM, near_bed_amplitude, wavenumber
from reefunit.dynamics import simulate
from reefunit.forcing import Harmonic, ISW, isw_params
from reefunit.kernel import forcing_spec, run
from reefunit.ratchet import asymptotic_drift, drift_per_cycle
from reefunit.reduced import free_rock_time, restitution, slide1d
from reefunit.statics import (crit_height, crit_velocity, holddown,
                              mode_first, ratios, theta_onsets)
from reefunit.units import batch, block, catalogue, grown, with_colonies


def test_dispersion_relation():
    """k solves omega^2 = g k tanh(k h) across depths and periods."""
    om = 2 * np.pi / np.array([3.0, 8.0, 14.0, 20.0])
    h = np.array([2.0, 8.0, 15.0, 40.0])[:, None]
    k = wavenumber(om, h)
    assert np.allclose(PRM.g * k * np.tanh(k * h), om ** 2, rtol=1e-12)


def test_kdv_soliton_residual():
    """The sech^2 soliton satisfies the two-layer KdV equation."""
    p = isw_params(12.0, 3.0, 5.0, 1.0)
    x = np.linspace(-200, 200, 4096, endpoint=False)
    kx = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
    eta = 1.0 / np.cosh(x / p["lam"]) ** 2

    def dx(f, n=1):
        return np.real(np.fft.ifft((1j * kx) ** n * np.fft.fft(f)))

    res = ((p["c0"] - p["V"]) * dx(eta) + p["a1"] * eta * dx(eta)
           + p["beta"] * dx(eta, 3))
    scale = np.abs(p["c0"] * dx(eta)).max()
    assert np.max(np.abs(res)) / scale < 1e-6


def test_isw_velocity_sign_and_peak():
    """An elevation soliton drives the lower layer forward."""
    w = ISW(12.0, 3.0, 5.0, 1.0, 0.0)
    u, _ = w.at(np.array([0.0, 1e4]), 0)
    assert u[0] > 0 and u[0] == pytest.approx(float(w.amp))
    assert u[1] == pytest.approx(0.0, abs=1e-9)


def test_mode_criterion_matches_onsets():
    """b/z_p > mu is equivalent to Theta_slide < Theta_tip."""
    d = batch(catalogue() + [grown(u, 0.10) for u in catalogue()])
    for mu in (0.4, 0.6, 0.9, 1.4):
        d["mu"] = np.full(len(d["m"]), mu)
        ts, tt, _ = theta_onsets(d)
        assert np.array_equal(mode_first(d), np.where(ts < tt, 0, 1))


def test_lift_cancels_from_the_mode_comparison():
    """Lift shifts both onsets but not which one comes first."""
    d = batch([block()] * 4, CLA=np.array([0.0, 0.05, 0.2, 0.5]))
    ts, tt, _ = theta_onsets(d)
    assert np.all(np.diff(ts) < 0) and np.all(np.diff(tt) < 0)
    assert len(np.unique(ts < tt)) == 1


def test_restitution_matches_housner():
    """A solid rectangle in vacuum gives 1 - (3/2) sin^2(alpha)."""
    blocks = [block(width=w, height=0.5, depth=0.4)
              for w in (0.1, 0.3, 0.5)]
    d = batch(blocks, fluid=False)
    alpha = np.arctan2(d["b"], d["h_c"])
    got = np.array([restitution(d, i) for i in range(len(blocks))])
    assert np.allclose(got, 1 - 1.5 * np.sin(alpha) ** 2, rtol=1e-12)


def test_kernel_matches_numpy_reference():
    """The compiled kernel reproduces the NumPy implementation."""
    d = batch([block(), grown(catalogue()[2], 0.10)])
    T = 8.0
    om = 2 * np.pi / T
    ub = np.array([1.9, 1.0])
    r = run(d, forcing_spec(2, ub=ub, omega=om), T / 400, 1200, n_iter=20)
    s, _ = simulate(d, Harmonic(ub, om), T / 400, 1200, n_iter=20)
    assert np.allclose(r.X, s.X, atol=1e-12)
    assert np.allclose(r.th, s.th, atol=1e-12)


def test_static_onset_under_a_ramped_current():
    """Onset under a slow ramp reproduces the closed-form threshold."""
    units = [block(width=0.4, height=hh, depth=0.4, C_L=0.0)
             for hh in (0.15, 0.5, 1.2)]
    d = batch(units)
    u_end = 1.25 * crit_velocity(d, 1e-9)
    t_ramp, h = 300.0, 0.01
    r = run(d, forcing_spec(len(units), ub=0.0, omega=1e-9, Uc=u_end,
                            ramp_time=t_ramp), h, int(t_ramp / h),
            record_every=20)
    moved = (np.abs(r.rec_X) > 1e-4) | (np.abs(r.rec_th) > 1e-4)
    k = np.argmax(moved, axis=0)
    t_on = (k + 1) * 20 * h
    u_on = u_end * (0.5 - 0.5 * np.cos(np.pi * t_on / t_ramp))
    theta = 0.5 * PRM.rho * d["CDA"] * u_on ** 2 / d["Wsub"]
    ts, tt, tl = theta_onsets(d)
    assert np.allclose(theta, np.minimum(np.minimum(ts, tt), tl),
                       rtol=5e-3)


def test_free_rocking_converges_to_the_reference():
    """Time to the first impact converges under step refinement."""
    d = batch([block(width=0.3, height=0.5, depth=0.3)], fluid=False,
              mu=5.0)
    a = float(np.arctan2(d["b"], d["h_c"])[0])
    t_ref, _ = free_rock_time(d, 0.5 * a)
    errs = []
    for hh in (4e-4, 1e-4):
        c, s = np.cos(-0.5 * a), np.sin(-0.5 * a)
        st = (d["b"] - (c * d["b"] + s * d["h_c"]),
              -(s * d["b"] - c * d["h_c"]), np.array([-0.5 * a]),
              np.zeros(1), np.zeros(1), np.zeros(1))
        r = run(d, forcing_spec(1, ub=0.0, omega=1.0), hh,
                int(1.5 * t_ref / hh), state=st, record_every=1)
        th = r.rec_th[:, 0]
        k = np.where((th[:-1] < 0) & (th[1:] >= 0))[0][0]
        t_hit = np.interp(0.0, [th[k], th[k + 1]],
                          [(k + 1) * hh, (k + 2) * hh])
        errs.append(abs(t_hit - t_ref))
    assert errs[1] < errs[0] and errs[1] < 1e-4


def test_sliding_matches_the_event_driven_solver():
    """Drift per cycle agrees with the reduced stick-slip model."""
    d = batch([block()], CLA=0.0)
    T, om, ub, Uc = 8.0, 2 * np.pi / 8.0, 1.8, 0.4
    F = Harmonic(ub, om, Uc=Uc)
    ts, xs, _ = slide1d(d, lambda t: (float(F.at(t, 0)[0]),
                                      float(F.at(t, 0)[1])), 10 * T)
    ref = np.diff(np.interp([6 * T, 10 * T], ts, xs))[0] / 4
    r = run(d, forcing_spec(1, ub=ub, omega=om, Uc=Uc), T / 2000,
            10 * 2000, record_every=2000)
    got = (r.rec_X[9, 0] - r.rec_X[5, 0]) / 4
    assert got == pytest.approx(ref, rel=5e-3)


def test_ratchet_asymptote_near_threshold():
    """Drift approaches the closed form as the excess vanishes."""
    d = batch([block()], CLA=0.0)
    om = 2 * np.pi / 8.0
    ub = np.array([1.9, 2.0])
    sim, _ = drift_per_cycle(d, ub, om, 0.3, n_cycles=8, n_skip=4,
                             spc=2000, relative=False)
    closed = asymptotic_drift(d, ub, om, 0.3)
    assert np.allclose(sim / closed, 1.0, rtol=0.05)


def test_no_drift_without_a_current():
    """A symmetric wave moves a unit back and forth but not along."""
    d = batch([block()], CLA=0.0)
    om = 2 * np.pi / 8.0
    drift, _ = drift_per_cycle(d, 2.2, om, 0.0, n_cycles=8, n_skip=4,
                               spc=2000)
    assert abs(float(drift)) < 1e-6


def test_critical_height_is_monotone_in_depth():
    """Deeper water takes a larger wave to move the same unit."""
    d = batch([block()] * 6)
    h = np.linspace(4.0, 20.0, 6)
    Hc = crit_height(d, 8.0, h, cap=False)
    assert np.all(np.diff(Hc) > 0)


def test_holddown_removes_the_exceedance():
    """Applying the required hold-down brings the worst ratio to one."""
    d = batch([catalogue()[2]])
    ub = float(near_bed_amplitude(3.0, 8.0, 8.0))
    F = float(np.ravel(holddown(d, ub, 2 * np.pi / 8.0, sf=1.0))[0])
    assert F > 0
    r = ratios(d, ub, 2 * np.pi / 8.0, anchor=F)
    worst = max(float(np.ravel(x)[0]) for x in r)
    assert worst == pytest.approx(1.0, rel=1e-3)


def test_colonies_add_weight_and_drag():
    """Grown colonies raise the drag area, the mass and the pressure
    height."""
    u = catalogue()[2]
    g = with_colonies(u, 20, 0.10, u.H)
    assert g.CDA > u.CDA and g.m > u.m and g.z_p > u.z_p
    assert g.Wsub > u.Wsub
