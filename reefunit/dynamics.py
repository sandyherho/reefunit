"""Moreau-Jean time-stepping for a planar rigid unit on a rigid bed.

Coordinates are the centre of mass (X, Z) and the rotation theta
(counter-clockwise positive), with generalised velocity v = (VX, VZ, w)
and mass matrix M = diag(m_x, m_x, I_G), where m_x = m + C_a rho V
includes added mass.  The two base contacts sit at body offsets
(+-b, -h_c) from the centre of mass.  For contact i with rotated offset
(r_x, r_y), the gap is g_i = Z + r_y, and the normal and tangential
contact velocities are U_n = VZ + w r_x and U_t = VX - w r_y.

The applied generalised force collects submerged weight, the
Froude-Krylov and added-mass force rho V (1 + C_a) du/dt at the centre of
mass, Morison drag (1/2) rho C_D A (u - v_P)|u - v_P| at the centre of
pressure P, and lift (1/2) rho C_L A_L (u - VX)^2 at the centre of mass.

One step of size h (theta = 1/2):

    q_m    = q_k + (h/2) v_k
    v_free = v_k + h M^-1 f(t_k + h/2, q_m, v_k)
    v_k+1  = v_free + M^-1 H(q_m)^T P
    q_k+1  = q_k + (h/2)(v_k + v_k+1)

with the impulse P solving, on contacts whose gap at q_m is closed, the
Signorini-Newton condition 0 <= U_n+ + e min(U_n-, 0) _|_ P_n >= 0 and
Coulomb's law |P_t| <= mu P_n with maximal dissipation.  The small
per-step problem is solved by projected Gauss-Seidel on the Delassus
operator W = H M^-1 H^T.  All arrays carry a leading batch dimension, so
thousands of units with different parameters and forcings advance
together.
"""

import numpy as np

from .core import PRM

__all__ = ["State", "rest_state", "step", "simulate", "OUTCOMES"]

GAP_TOL = 1e-8   # m; contacts within this gap are treated as closed
OUTCOMES = ("stable", "displaced", "overturned")


class State:
    """Positions, velocities and running diagnostics of a batch."""

    def __init__(self, n):
        z = np.zeros(n)
        self.X, self.Z, self.th = z.copy(), z.copy(), z.copy()
        self.VX, self.VZ, self.w = z.copy(), z.copy(), z.copy()
        self.P = np.zeros((n, 4))           # warm start: (n_L, t_L, n_R, t_R)
        self.max_th = z.copy()
        self.max_lift = z.copy()
        self.over = np.zeros(n, bool)
        self.t_over = np.full(n, np.nan)
        self.res = z.copy()                  # last complementarity residual


def rest_state(u):
    """All units at rest on the bed."""
    s = State(len(u["m"]))
    s.Z[:] = u["h_c"]
    s.X0 = s.X.copy()
    return s


def _geometry(u, th):
    c, sn = np.cos(th), np.sin(th)
    b, hc = u["b"], u["h_c"]
    # contact offsets: left (-b, -hc), right (+b, -hc)
    rxL, ryL = -b * c + hc * sn, -b * sn - hc * c
    rxR, ryR = b * c + hc * sn, b * sn - hc * c
    dz = u["z_p"] - hc
    rxP, ryP = -dz * sn, dz * c
    return rxL, ryL, rxR, ryR, rxP, ryP


def step(u, s, forcing, t, h, k, relative=True, n_iter=30, anchor=0.0):
    """Advance the batch by one step of size h (array or scalar)."""
    rho = PRM.rho
    live = ~s.over
    hm = 0.5 * h
    Xm, Zm, thm = s.X + hm * s.VX, s.Z + hm * s.VZ, s.th + hm * s.w
    rxL, ryL, rxR, ryR, rxP, ryP = _geometry(u, thm)
    uf, duf = forcing.at(t + hm, k)

    vPx = s.VX - s.w * ryP
    ur = uf - vPx if relative else uf
    FD = 0.5 * rho * u["CDA"] * ur * np.abs(ur)
    urG = uf - s.VX if relative else uf
    FL = 0.5 * rho * u["CLA"] * urG ** 2
    FI = u["FKV"] * duf
    Qx = FD + FI
    Qz = -u["Wsub"] - anchor + FL
    Qt = -ryP * FD
    mx, IG = u["mx"], u["IG"]
    vx = s.VX + h * Qx / mx
    vz = s.VZ + h * Qz / mx
    w = s.w + h * Qt / IG

    # contact rows: normal (0, 1, rx), tangent (1, 0, -ry)
    Hn = [(0.0, 1.0, rxL), (0.0, 1.0, rxR)]
    Ht = [(1.0, 0.0, -ryL), (1.0, 0.0, -ryR)]
    rows = [Hn[0], Ht[0], Hn[1], Ht[1]]
    minv = (1.0 / mx, 1.0 / mx, 1.0 / IG)
    gapL, gapR = Zm + ryL, Zm + ryR
    act = [gapL <= GAP_TOL, gapR <= GAP_TOL]
    Unk = [s.VZ + s.w * rxL, s.VZ + s.w * rxR]
    W = [[sum(ra[j] * rb[j] * minv[j] for j in range(3)) for rb in rows]
         for ra in rows]
    Uf = [vx * r[0] + vz * r[1] + w * r[2] for r in rows]
    P = s.P * np.column_stack([act[0], act[0], act[1], act[1]])
    P = [P[:, i].copy() for i in range(4)]
    mu, e = u["mu"], u["e"]
    for _ in range(n_iter):
        for c in (0, 1):
            i_n, i_t = 2 * c, 2 * c + 1
            a = act[c]
            Un = Uf[i_n] + sum(W[i_n][j] * P[j] for j in range(4))
            target = Un + e * np.minimum(Unk[c], 0.0)
            P[i_n] = np.where(a, np.maximum(0.0, P[i_n] - target
                                            / W[i_n][i_n]), 0.0)
            Ut = Uf[i_t] + sum(W[i_t][j] * P[j] for j in range(4))
            lim = mu * P[i_n]
            P[i_t] = np.where(a, np.clip(P[i_t] - Ut / W[i_t][i_t], -lim,
                                         lim), 0.0)
    # residual of the normal complementarity condition (diagnostic)
    res = np.zeros_like(vx)
    for c in (0, 1):
        i_n = 2 * c
        Un = Uf[i_n] + sum(W[i_n][j] * P[j] for j in range(4))
        tgt = Un + e * np.minimum(Unk[c], 0.0)
        res = np.maximum(res, np.where(act[c], np.abs(np.minimum(
            tgt, P[i_n] * W[i_n][i_n])), 0.0))
    dv = [sum(rows[i][j] * P[i] for i in range(4)) * minv[j]
          for j in range(3)]
    vx, vz, w = vx + dv[0], vz + dv[1], w + dv[2]

    X = s.X + hm * (s.VX + vx)
    Z = s.Z + hm * (s.VZ + vz)
    th = s.th + hm * (s.w + w)
    for name, new in (("X", X), ("Z", Z), ("th", th), ("VX", vx),
                      ("VZ", vz), ("w", w)):
        getattr(s, name)[live] = new[live]
    s.P[live] = np.column_stack(P)[live]
    s.res = res
    s.max_th = np.maximum(s.max_th, np.abs(s.th))
    lift = np.minimum(Z + ryL, Z + ryR)
    s.max_lift = np.maximum(s.max_lift, lift)
    alpha = np.arctan2(u["b"], u["h_c"])
    newly = live & (np.abs(s.th) > alpha + 0.2)
    s.over |= newly
    s.t_over[newly] = t + h if np.ndim(h) == 0 else (t + h)[newly]
    return s


def simulate(u, forcing, h, n_steps, relative=True, record=None, every=1,
             n_iter=30, anchor=0.0, t0=0.0):
    """Integrate from rest; optionally record fields every ``every`` steps.

    ``h`` may be an array (one step size per unit), in which case the
    time of unit i after k steps is k h_i.  Returns the final State and a
    dict of recorded arrays with shape (n_records, n_units).
    """
    s = rest_state(u)
    rec = {name: [] for name in (record or [])}
    if record:
        rec["t"] = []
    h = np.asarray(h, float)
    for k in range(n_steps):
        t = t0 + k * h
        step(u, s, forcing, t, h, k, relative, n_iter, anchor)
        if record and (k % every == every - 1):
            for name in record:
                rec[name].append(getattr(s, name).copy())
            rec["t"].append(np.broadcast_to(t + h, s.X.shape).copy())
    return s, {k: np.array(v) for k, v in rec.items()}


def classify(s, disp_tol=0.05, th_tol=0.02):
    """Outcome index per unit: 0 stable, 1 displaced, 2 overturned.

    Displaced means a net horizontal excursion above ``disp_tol`` metres
    or a rocking amplitude above ``th_tol`` radians without overturning.
    """
    out = np.zeros(len(s.X), int)
    moved = (np.abs(s.X - s.X0) > disp_tol) | (s.max_th > th_tol)
    out[moved] = 1
    out[s.over] = 2
    return out


__all__.append("classify")


def tilted_state(u, th0):
    """Units at rest, tilted by th0 (< 0: on the right corner) on a pivot."""
    s = rest_state(u)
    th0 = np.broadcast_to(np.asarray(th0, float), s.X.shape)
    sgn = np.where(th0 < 0, 1.0, -1.0)          # pivot on the right if th<0
    c, sn = np.cos(th0), np.sin(th0)
    bx, by = sgn * u["b"], -u["h_c"]
    rx, ry = c * bx - sn * by, sn * bx + c * by  # rotated pivot offset
    s.X[:] = sgn * u["b"] - rx
    s.Z[:] = -ry
    s.th[:] = th0
    s.X0 = s.X.copy()
    return s


def run_from(u, s, forcing, h, n_steps, relative=True, record=None, every=1,
             n_iter=30, anchor=0.0, t0=0.0):
    """Like :func:`simulate` but from a given state."""
    rec = {name: [] for name in (record or [])}
    if record:
        rec["t"] = []
    h = np.asarray(h, float)
    for k in range(n_steps):
        t = t0 + k * h
        step(u, s, forcing, t, h, k, relative, n_iter, anchor)
        if record and (k % every == every - 1):
            for name in record:
                rec[name].append(getattr(s, name).copy())
            rec["t"].append(np.broadcast_to(t + h, s.X.shape).copy())
    return s, {k: np.array(v) for k, v in rec.items()}


__all__ += ["tilted_state", "run_from"]
