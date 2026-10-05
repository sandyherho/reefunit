"""Net migration of a sliding unit under waves on a current.

Near-threshold asymptote.  Let the horizontal load at rest peak at
F_max = mu N (1 + eps) with time curvature F'' = -F_max c omega^2 at the
peak.  A slip episode starts where F = mu N, the slip velocity grows while
F > mu N and the unit re-sticks when the impulse of the excess vanishes.
Writing the excess as mu N (eps - kappa tau^2/2) with
kappa = (1 + eps) c omega^2, the episode lasts from -tau0 to 2 tau0,
tau0 = sqrt(2 eps/kappa), and the displacement is

    Delta = (9/2) (mu N/m_x) eps^2 / kappa,

exact for this parabolic excess and the leading term for any smooth load
as eps -> 0.  For pure drag under u = U_c + u_b cos(omega t),
c = 2 u_b/(U_c + u_b).  A symmetric wave (U_c = 0) gives equal and
opposite episodes and no net drift; a current breaks the symmetry.

Relative-velocity correction.  Drag acts on u - v, so a slipping unit
feels a restoring load dF/dv = -rho C_D A |u| that is absent from the
estimate above.  Treating it as a weak damping gamma = rho C_D A |u_p|/m_x
and expanding to first order in the damping number Gamma = gamma tau0,

    Delta = Delta_0 (1 - (6/5) Gamma + O(Gamma^2)),

since the damping removes gamma times the integral of the undamped
displacement over the episode, (81/10)/(27/4) = 6/5 of Delta_0 per unit
Gamma.  Gamma grows like eps^(1/2), so the correction vanishes at onset
but slowly.

Quasi-steady integration.  When the current varies slowly compared with
the wave period (a tide, or the passage of an internal solitary wave), the
displacement is approximated by integrating the per-cycle drift D(U_c) of
the steady problem over the slowly varying current,
X = (1/T) integral D(U_c(t)) dt.
"""

import numpy as np

from .core import PRM
from .kernel import forcing_spec, run

__all__ = ["peak_excess", "asymptotic_drift", "damping_number",
           "drift_per_cycle", "quasi_steady_displacement"]


def peak_excess(u, ub, omega, Uc, nphi=20001):
    """eps and c of the rest load F(phi) at its maximum (absolute value).

    Returns (eps, c, N_peak, u_peak) for unit arrays; the load includes drag,
    inertia and the lift reduction of the normal force.
    """
    rho = PRM.rho
    phi = np.linspace(-np.pi, np.pi, nphi)
    ub, om, Uc = (np.asarray(x, float)[..., None]
                  for x in np.broadcast_arrays(ub, omega, Uc))

    def g(k):
        return np.asarray(u[k], float)[..., None]
    uu = Uc + ub * np.cos(phi)
    du = -om * ub * np.sin(phi)
    Fx = 0.5 * rho * g("CDA") * uu * np.abs(uu) + g("FKV") * du
    N = g("Wsub") - 0.5 * rho * g("CLA") * uu ** 2
    r = Fx / (g("mu") * N)
    j = np.argmax(r, axis=-1)
    rp = np.take_along_axis(r, j[..., None], -1)[..., 0]
    jm = np.clip(j - 1, 1, nphi - 2)
    d2 = (np.take_along_axis(r, (jm - 1)[..., None], -1)[..., 0]
          - 2 * np.take_along_axis(r, jm[..., None], -1)[..., 0]
          + np.take_along_axis(r, (jm + 1)[..., None], -1)[..., 0])
    dphi = phi[1] - phi[0]
    c = -d2 / dphi ** 2 / rp
    Np = np.take_along_axis(N, j[..., None], -1)[..., 0]
    up = np.take_along_axis(np.broadcast_to(uu, r.shape), j[..., None],
                            -1)[..., 0]
    return rp - 1.0, c, Np, up


def asymptotic_drift(u, ub, omega, Uc):
    """Leading-order forward slip per cycle near threshold (m)."""
    eps, c, N, _ = peak_excess(u, ub, omega, Uc)
    om = np.asarray(omega, float)
    mu, mx = (np.asarray(u[k], float) for k in ("mu", "mx"))
    kap = (1 + eps) * c * om ** 2
    return np.where(eps > 0, 4.5 * mu * N / mx * eps ** 2 / kap, 0.0)


def damping_number(u, ub, omega, Uc):
    """Gamma = rho C_D A |u_peak| tau0 / m_x at the load peak."""
    eps, c, _, up = peak_excess(u, ub, omega, Uc)
    om = np.asarray(omega, float)
    kap = (1 + eps) * c * om ** 2
    tau0 = np.sqrt(2 * np.maximum(eps, 0) / kap)
    return (PRM.rho * np.asarray(u["CDA"]) * np.abs(up) * tau0
            / np.asarray(u["mx"]))


def drift_per_cycle(u, ub, omega, Uc, n_cycles=10, n_skip=4, spc=800,
                    relative=True):
    """Net drift per wave cycle from the time-stepper.

    Integrates n_cycles periods from rest (ramped over two) and averages
    the displacement over the last n_cycles - n_skip, when the stick-slip
    cycle has become periodic.  Arrays broadcast over units and forcing.
    """
    ub, omega, Uc = np.broadcast_arrays(np.asarray(ub, float),
                                        np.asarray(omega, float),
                                        np.asarray(Uc, float))
    n = ub.size
    uu = {k: np.broadcast_to(np.asarray(v, float), (n,)).copy()
          for k, v in u.items()}
    T = 2 * np.pi / omega.ravel()
    h = T / spc
    f = forcing_spec(n, ub=ub.ravel(), omega=omega.ravel(), Uc=Uc.ravel())
    r = run(uu, f, h, n_cycles * spc, relative=relative, record_every=spc)
    X = r.rec_X
    d = (X[n_cycles - 1] - X[n_skip - 1]) / (n_cycles - n_skip)
    return d.reshape(ub.shape), r


def quasi_steady_displacement(Ugrid, Dgrid, Uc_t, t, T):
    """Integrate the per-cycle drift over a slowly varying current."""
    D = np.interp(Uc_t, Ugrid, Dgrid)
    return np.trapezoid(D, t) / T
