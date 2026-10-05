"""Quasi-static stability of a unit resting on the bed.

With the unit held at rest, the loads over one wave phase phi are

    u = U_c + u_b cos(phi) + u_2 cos(2 phi),
    F_D = (1/2) rho C_D A u|u|,  F_I = rho V (1 + C_a) du/dt,
    F_L = (1/2) rho C_L A_L u^2,

and three demand-to-capacity ratios follow:

    sliding  S_s = |F_D + F_I| / (mu (W' + F_a - F_L)),
    tipping  S_t = (|F_D z_p + F_I h_c| + F_L b) / ((W' + F_a) b),
    lift-off S_l = F_L / (W' + F_a),

where W' is the submerged weight and F_a an optional hold-down (ballast
or anchor) force acting through the base centre.  A mode is reached when
its ratio, maximised over phi, reaches one.

In the drag-dominated limit (F_I -> 0) with Theta = F_D/W', the onset
values are Theta_s = mu/(1 + mu Lambda), Theta_t = 1/(z_p/b + Lambda) and
Theta_l = 1/Lambda, with Lambda = C_L A_L/(C_D A).  Sliding precedes
tipping if and only if b/z_p > mu; lift cancels from the comparison.
"""

import numpy as np

from .core import PRM, near_bed_amplitude, stokes2_amplitude, breaking_height

__all__ = ["ratios", "theta_onsets", "mode_first", "crit_velocity",
           "crit_height", "holddown", "MODES"]

MODES = ("slide", "tip", "lift")
_PHI = np.linspace(0.0, 2 * np.pi, 257)[:-1]


def ratios(u, ub, omega, Uc=0.0, u2=0.0, anchor=0.0, phi=_PHI):
    """Phase-maximised (S_slide, S_tip, S_lift) for arrays of units."""
    rho = PRM.rho
    ub, om, Uc, u2 = (np.asarray(x, float)[..., None]
                      for x in np.broadcast_arrays(ub, omega, Uc, u2))
    W = (np.asarray(u["Wsub"], float) + anchor)[..., None]

    def g(k):
        return np.asarray(u[k], float)[..., None]
    uu = Uc + ub * np.cos(phi) + u2 * np.cos(2 * phi)
    du = -om * (ub * np.sin(phi) + 2 * u2 * np.sin(2 * phi))
    FD = 0.5 * rho * g("CDA") * uu * np.abs(uu)
    FI = g("FKV") * du
    FL = 0.5 * rho * g("CLA") * uu ** 2
    N = W - FL
    Fx = np.abs(FD + FI)
    ss = np.where(N > 0, Fx / (g("mu") * np.maximum(N, 1e-300)), np.inf)
    ss = np.where((N <= 0) & (Fx == 0), np.inf, ss)
    st = (np.abs(FD * g("z_p") + FI * g("h_c")) + FL * g("b")) / (W * g("b"))
    sl = FL / W
    return ss.max(-1), st.max(-1), sl.max(-1)


def theta_onsets(u):
    """Drag-only onset values of Theta = F_D/W' for slide, tip, lift."""
    lam = np.asarray(u["CLA"], float) / np.asarray(u["CDA"], float)
    mu, zp, b = (np.asarray(u[k], float) for k in ("mu", "z_p", "b"))
    ts = mu / (1 + mu * lam)
    tt = 1.0 / (zp / b + lam)
    with np.errstate(divide="ignore"):
        tl = np.where(lam > 0, 1.0 / lam, np.inf)
    return ts, tt, tl


def mode_first(u):
    """Index of the drag-only first mode: 0 slide, 1 tip."""
    return np.where(np.asarray(u["b"]) / np.asarray(u["z_p"])
                    > np.asarray(u["mu"]), 0, 1)


def _bisect(fun, lo, hi, n=60):
    """Smallest x in [lo, hi] with fun(x) >= 1 for a monotone fun."""
    lo = np.array(lo, float)
    hi = np.array(hi, float)
    ok = fun(hi) >= 1.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        f = fun(mid) >= 1.0
        hi = np.where(f, mid, hi)
        lo = np.where(f, lo, mid)
    return np.where(ok, hi, np.inf)


def crit_velocity(u, omega, Uc=0.0, mode=None, anchor=0.0, umax=10.0):
    """Critical near-bed amplitude u_b for regular waves (no harmonics).

    ``mode`` None gives the first of the three modes; 0, 1, 2 a single
    mode.
    """
    n = np.broadcast(np.asarray(u["Wsub"]), np.asarray(omega),
                     np.asarray(Uc)).shape

    def fun(x):
        r = ratios(u, x, omega, Uc, anchor=anchor)
        return r[mode] if mode is not None else np.maximum.reduce(r)
    return _bisect(fun, np.zeros(n), np.full(n, umax))


def crit_height(u, T, h, Uc=0.0, stokes=False, mode=None, anchor=0.0,
                cap=True):
    """Critical regular-wave height H_c at depth h and period T.

    Returns inf where no mode is reached below the depth-limited breaking
    height (when ``cap``), since the Morison loading model is not used
    inside the surf zone.
    """
    T = np.asarray(T, float)
    h = np.asarray(h, float)
    om = 2 * np.pi / T
    Hmax = breaking_height(h) if cap else 3.0 * h

    def fun(H):
        ub = near_bed_amplitude(H, T, h)
        u2 = stokes2_amplitude(H, T, h) if stokes else 0.0
        r = ratios(u, ub, om, Uc, u2, anchor)
        return r[mode] if mode is not None else np.maximum.reduce(r)
    n = np.broadcast(np.asarray(u["Wsub"]), T, h, np.asarray(Uc)).shape
    return _bisect(fun, np.zeros(n), np.broadcast_to(Hmax, n))


def holddown(u, ub, omega, Uc=0.0, u2=0.0, sf=1.0, phi=_PHI):
    """Hold-down force (N) that keeps every ratio at or below 1/sf."""
    rho = PRM.rho
    ub, om, Uc, u2 = (np.asarray(x, float)[..., None]
                      for x in np.broadcast_arrays(ub, omega, Uc, u2))

    def g(k):
        return np.asarray(u[k], float)[..., None]
    uu = Uc + ub * np.cos(phi) + u2 * np.cos(2 * phi)
    du = -om * (ub * np.sin(phi) + 2 * u2 * np.sin(2 * phi))
    FD = 0.5 * rho * g("CDA") * uu * np.abs(uu)
    FI = g("FKV") * du
    FL = 0.5 * rho * g("CLA") * uu ** 2
    W = g("Wsub")
    need_s = sf * np.abs(FD + FI) / g("mu") + FL - W
    need_t = sf * (np.abs(FD * g("z_p") + FI * g("h_c")) / g("b") + FL) - W
    need_l = sf * FL - W
    need = np.maximum(np.maximum(need_s, need_t), need_l).max(-1)
    return np.maximum(need, 0.0)
