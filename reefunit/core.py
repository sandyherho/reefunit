"""Constants and near-bed wave kinematics.

Linear (Airy) theory supplies the near-bed orbital velocity of a regular
wave of height H, period T and water depth h,

    u_b = (H/2) omega / sinh(k h),    omega^2 = g k tanh(k h),

and second-order Stokes theory adds the bound harmonic

    u_2 = (3/4) (H/2)^2 omega k / sinh^4(k h)

at the bed, which makes the onshore half-cycle stronger than the offshore
half-cycle.  Every function is vectorised over its arguments.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Params", "PRM", "wavenumber", "fenton_k", "near_bed_amplitude",
           "stokes2_amplitude", "ursell", "breaking_height",
           "stokes_harmonic"]

URSELL_MAX = 26.0   # conventional upper limit of Stokes second-order theory


@dataclass(frozen=True)
class Params:
    """Seawater and gravity (illustrative, not site-specific)."""

    rho: float = 1025.0        # seawater density, kg m^-3
    g: float = 9.81            # gravity, m s^-2
    gamma_b: float = 0.78      # depth-limited breaking index H/h


PRM = Params()


def fenton_k(omega, h, g=PRM.g):
    """Explicit approximation to the linear dispersion relation.

    Fenton and McKee's form k0 = (omega^2/g) coth^(2/3)((omega^2 h/g)^(3/4))
    is accurate to about 1.5 percent and seeds the Newton iteration in
    :func:`wavenumber`.
    """
    omega, h = np.broadcast_arrays(np.asarray(omega, float),
                                   np.asarray(h, float))
    x = omega ** 2 * h / g
    return omega ** 2 / g / np.tanh(x ** 0.75) ** (2.0 / 3.0)


def wavenumber(omega, h, g=PRM.g, tol=1e-14, itmax=50):
    """Solve omega^2 = g k tanh(k h) for k by Newton iteration."""
    omega, h = np.broadcast_arrays(np.asarray(omega, float),
                                   np.asarray(h, float))
    k = fenton_k(omega, h, g)
    for _ in range(itmax):
        th = np.tanh(k * h)
        f = g * k * th - omega ** 2
        df = g * th + g * k * h * (1.0 - th ** 2)
        dk = f / df
        k = k - dk
        if np.all(np.abs(dk) <= tol * np.abs(k)):
            break
    return k


def near_bed_amplitude(H, T, h, g=PRM.g):
    """Linear near-bed orbital velocity amplitude u_b (m s^-1)."""
    omega = 2 * np.pi / np.asarray(T, float)
    k = wavenumber(omega, h, g)
    return 0.5 * np.asarray(H, float) * omega / np.sinh(k * h)


def stokes2_amplitude(H, T, h, g=PRM.g):
    """Second-harmonic near-bed velocity amplitude of a Stokes wave."""
    omega = 2 * np.pi / np.asarray(T, float)
    k = wavenumber(omega, h, g)
    a = 0.5 * np.asarray(H, float)
    return 0.75 * a ** 2 * omega * k / np.sinh(k * h) ** 4


def ursell(H, T, h, g=PRM.g):
    """Ursell number H L^2 / h^3, a measure of shallow-water nonlinearity."""
    k = wavenumber(2 * np.pi / np.asarray(T, float), h, g)
    L = 2 * np.pi / k
    return np.asarray(H, float) * L ** 2 / np.asarray(h, float) ** 3


def breaking_height(h, prm=PRM):
    """Depth-limited height beyond which the regular wave is taken to break."""
    return prm.gamma_b * np.asarray(h, float)


def stokes_harmonic(H, T, h, g=PRM.g):
    """Second-harmonic amplitude where Stokes theory applies, else zero.

    Beyond an Ursell number of about 26 the second-order series develops
    a spurious secondary crest and cnoidal theory is required; such waves
    are then represented by their linear component only.
    """
    u2 = stokes2_amplitude(H, T, h, g)
    return np.where(ursell(H, T, h, g) < URSELL_MAX, u2, 0.0)


def orbital_field(H, T, h, x, z, t, g=PRM.g):
    """Linear-wave orbital velocity field and surface elevation.

    For a wave of height H and period T on depth h, with z measured
    upward from the still water line and the bed at z = -h,

        eta = (H/2) cos(k x - omega t),
        u   = (H/2) omega cosh(k(z+h))/sinh(k h) cos(k x - omega t),
        w   = (H/2) omega sinh(k(z+h))/sinh(k h) sin(k x - omega t).

    Returns (u, w, eta) in m/s and m; u and w on the broadcast of x and
    z, and eta with the shape of x.
    """
    omega = 2 * np.pi / np.asarray(T, float)
    k = wavenumber(omega, h, g)
    a = 0.5 * np.asarray(H, float)
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    ph = k * x - omega * t
    sh = np.sinh(k * h)
    u = a * omega * np.cosh(k * (z + h)) / sh * np.cos(ph)
    w = a * omega * np.sinh(k * (z + h)) / sh * np.sin(ph)
    eta = a * np.cos(k * x - omega * t)
    return u, w, eta


__all__.append("orbital_field")
