"""Near-bed velocity histories u(t) and accelerations du/dt.

Every forcing is an object with a method ``at(t, k)`` returning the pair
(u, du) as arrays over the batch, where t is the time and k the step index
(used only by tabulated series).  Forcings add with ``+``.

Internal solitary waves use the first-order two-layer Korteweg-de Vries
(KdV) equation for the interface displacement eta(x, t),

    eta_t + c0 eta_x + a1 eta eta_x + beta eta_xxx = 0,
    c0 = sqrt(g' h1 h2/(h1 + h2)),  a1 = 3 c0 (h1 - h2)/(2 h1 h2),
    beta = c0 h1 h2 / 6,

whose soliton is eta = eta0 sech^2((x - V t)/lam), V = c0 + a1 eta0/3,
lam^2 = 12 beta/(a1 eta0).  With the pycnocline nearer the bed than the
surface (h2 < h1) the wave is one of elevation, and the lower-layer
velocity that loads a unit on the bed is, to the same order,
u2 = c0 eta / h2.
"""

import numpy as np

from .core import PRM

__all__ = ["Forcing", "Harmonic", "ISW", "ramp", "isw_params"]


def ramp(t, t_ramp):
    """Smooth C1 ramp from 0 to 1 over t_ramp, and its derivative."""
    t_ramp = np.asarray(t_ramp, float)
    s = np.clip(t / np.maximum(t_ramp, 1e-300), 0.0, 1.0)
    r = 0.5 - 0.5 * np.cos(np.pi * s)
    dr = np.where((s > 0) & (s < 1),
                  0.5 * np.pi * np.sin(np.pi * s) / np.maximum(t_ramp,
                                                               1e-300), 0.0)
    return r, dr


class Forcing:
    """Base class; subclasses implement ``at``."""

    def __add__(self, other):
        return _Sum(self, other)

    def at(self, t, k):
        """Return (u, du) at time t, step k."""
        raise NotImplementedError


class _Sum(Forcing):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def at(self, t, k):
        ua, da = self.a.at(t, k)
        ub, db = self.b.at(t, k)
        return ua + ub, da + db


class Harmonic(Forcing):
    """u = r(t) [U_c + u_b cos(omega t + phi) + u_2 cos(2 omega t + 2 phi)].

    r is a smooth ramp over ``ramp_cycles`` periods so that the unit starts
    from rest without an artificial jerk.  All parameters broadcast over
    the batch.
    """

    def __init__(self, ub, omega, Uc=0.0, u2=0.0, phi=0.0, ramp_cycles=2.0):
        self.ub = np.asarray(ub, float)
        self.om = np.asarray(omega, float)
        self.Uc = np.asarray(Uc, float)
        self.u2 = np.asarray(u2, float)
        self.phi = np.asarray(phi, float)
        self.tr = ramp_cycles * 2 * np.pi / self.om

    def at(self, t, k):
        """Velocity and acceleration."""
        th = self.om * t + self.phi
        f = self.Uc + self.ub * np.cos(th) + self.u2 * np.cos(2 * th)
        df = -self.om * (self.ub * np.sin(th) + 2 * self.u2 * np.sin(2 * th))
        r, dr = ramp(t, self.tr)
        return r * f, r * df + dr * f


def isw_params(h1, h2, drho, eta0, rho=PRM.rho, g=PRM.g):
    """Two-layer KdV coefficients and soliton speed and width."""
    gp = g * drho / rho
    c0 = np.sqrt(gp * h1 * h2 / (h1 + h2))
    a1 = 1.5 * c0 * (h1 - h2) / (h1 * h2)
    beta = c0 * h1 * h2 / 6.0
    V = c0 + a1 * eta0 / 3.0
    lam = np.sqrt(12.0 * beta / (a1 * eta0))
    return dict(gp=gp, c0=c0, a1=a1, beta=beta, V=V, lam=lam)


class ISW(Forcing):
    """Near-bed lower-layer velocity of a KdV soliton passing x = 0.

    The crest passes at t0; u = c0 eta0 sech^2(V (t - t0)/lam) / h2.
    """

    def __init__(self, h1, h2, drho, eta0, t0):
        p = isw_params(h1, h2, drho, eta0)
        self.p = p
        self.amp = np.asarray(p["c0"] * eta0 / h2, float)
        self.s = np.asarray(p["V"] / p["lam"], float)
        self.t0 = np.asarray(t0, float)

    def at(self, t, k):
        """Velocity and acceleration."""
        x = self.s * (t - self.t0)
        sech2 = 1.0 / np.cosh(x) ** 2
        return self.amp * sech2, -2.0 * self.amp * self.s * sech2 * np.tanh(x)
