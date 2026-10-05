"""Event-driven reference solvers that share no code with the time-stepper.

Two reduced models are integrated with SciPy's DOP853 and exact event
location:

``slide1d``
    pure sliding without rotation,
    m_x x'' = F_x(t, x') - s mu N(t, x'),  s = sign(x'),
    with sticking whenever x' = 0 and |F_x| <= mu N.

``rock1p``
    rocking about a stuck base corner, I_O theta'' = tau(t, theta,
    theta'), with I_O = I_G + m_x R^2, and the angular-momentum impact map
    at theta = 0,

        theta'+ = theta'- (I_G + m_x (h_c^2 - b^2)) / (I_G + m_x R^2),

    which for a solid rectangle in vacuum reduces to Housner's
    1 - (3/2) sin^2(alpha).

Both evaluate the same load model as the time-stepper (Morison drag on
relative velocity at the centre of pressure, Froude-Krylov and added-mass
force and lift at the centre of mass).
"""

import numpy as np
from scipy.integrate import solve_ivp

from .core import PRM

__all__ = ["slide1d", "rock1p", "restitution", "free_rock_time"]

_OPT = dict(method="DOP853", rtol=1e-11, atol=1e-13)


def _scalar(u, i=0):
    return {k: float(np.asarray(v).ravel()[i]) for k, v in u.items()}


def slide1d(u, force, t_end, relative=True, i=0, max_events=100000):
    """Stick-slip history of unit i; returns dense (t, x, v) arrays."""
    p = _scalar(u, i)
    rho = PRM.rho

    def loads(t, v):
        uf, du = force(t)
        ur = uf - v if relative else uf
        Fx = 0.5 * rho * p["CDA"] * ur * abs(ur) + p["FKV"] * du
        N = p["Wsub"] - 0.5 * rho * p["CLA"] * ur ** 2
        return Fx, N

    ts, xs, vs = [0.0], [0.0], [0.0]
    t, x = 0.0, 0.0
    mode = 0              # 0 stick, +1 / -1 slip direction
    for _ in range(max_events):
        if t >= t_end:
            break
        if mode == 0:
            def ev(tt, y):
                Fx, N = loads(tt, 0.0)
                return abs(Fx) - p["mu"] * N
            ev.terminal, ev.direction = True, 1
            sol = solve_ivp(lambda tt, y: [0.0], (t, t_end), [x],
                            events=ev, dense_output=True, max_step=0.05,
                            **_OPT)
            tt = np.linspace(t, sol.t[-1], 20)
            ts += list(tt[1:])
            xs += [x] * (len(tt) - 1)
            vs += [0.0] * (len(tt) - 1)
            t = sol.t[-1]
            if sol.status == 1:
                Fx, _ = loads(t, 0.0)
                mode = 1 if Fx > 0 else -1
            continue
        s = mode

        def rhs(tt, y):
            Fx, N = loads(tt, y[1])
            return [y[1], (Fx - s * p["mu"] * N) / p["mx"]]

        def ev(tt, y):
            return y[1]
        ev.terminal, ev.direction = True, -s
        sol = solve_ivp(rhs, (t, t_end), [x, 0.0], events=ev,
                        dense_output=True, max_step=0.05, **_OPT)
        tt = np.linspace(t, sol.t[-1], max(20, int(200 * (sol.t[-1] - t))))
        yy = sol.sol(tt)
        ts += list(tt[1:])
        xs += list(yy[0, 1:])
        vs += list(yy[1, 1:])
        t = sol.t[-1]
        x = float(sol.y[0, -1])
        if sol.status == 1:
            Fx, N = loads(t, 0.0)
            if abs(Fx) > p["mu"] * N:
                mode = 1 if Fx > 0 else -1
            else:
                mode = 0
    return np.array(ts), np.array(xs), np.array(vs)


def restitution(u, i=0):
    """Angular-velocity ratio across a pivot change at theta = 0."""
    p = _scalar(u, i)
    R2 = p["b"] ** 2 + p["h_c"] ** 2
    return ((p["IG"] + p["mx"] * (p["h_c"] ** 2 - p["b"] ** 2))
            / (p["IG"] + p["mx"] * R2))


def _torque(p, force, t, th, w, relative):
    """Torque about the active pivot (right if th < 0 or at rest pushing
    clockwise; left otherwise)."""
    rho = PRM.rho
    sgn = 1.0 if th < 0 else -1.0
    c, s = np.cos(th), np.sin(th)
    bx = -sgn * p["b"]
    dx, dy = c * bx - s * p["h_c"], s * bx + c * p["h_c"]
    px, py = c * bx - s * p["z_p"], s * bx + c * p["z_p"]
    uf, du = force(t)
    vP = -w * py
    vG = -w * dy
    ur = uf - vP if relative else uf
    urG = uf - vG if relative else uf
    FD = 0.5 * rho * p["CDA"] * ur * abs(ur)
    FL = 0.5 * rho * p["CLA"] * urG ** 2
    FI = p["FKV"] * du
    return dx * (-p["Wsub"] + FL) - dy * FI - py * FD


def rock1p(u, force, t_end, relative=True, i=0, th0=0.0, max_events=5000,
           w_stop=1e-7):
    """Rocking history of unit i about stuck corners; returns (t, th, w).

    Integration stops when |theta| exceeds alpha + 0.2 (overturned).
    """
    p = _scalar(u, i)
    R2 = p["b"] ** 2 + p["h_c"] ** 2
    IO = p["IG"] + p["mx"] * R2
    e = restitution(u, i)
    alpha = np.arctan2(p["b"], p["h_c"])
    t, th, w = 0.0, float(th0), 0.0
    ts, ths, ws = [t], [th], [w]
    rest = th0 == 0.0
    for _ in range(max_events):
        if t >= t_end:
            break
        if rest:
            def evR(tt, y):
                return _torque(p, force, tt, -1e-300, 0.0, relative)
            evR.terminal, evR.direction = True, -1

            def evL(tt, y):
                return _torque(p, force, tt, 1e-300, 0.0, relative)
            evL.terminal, evL.direction = True, 1
            sol = solve_ivp(lambda tt, y: [0.0], (t, t_end), [0.0],
                            events=[evR, evL], max_step=0.05, **_OPT)
            tt = np.linspace(t, sol.t[-1], 10)
            ts += list(tt[1:])
            ths += [0.0] * 9
            ws += [0.0] * 9
            t = sol.t[-1]
            if sol.status == 1:
                rest = False
                th = -1e-12 if len(sol.t_events[0]) else 1e-12
                w = 0.0
            continue
        side = -1.0 if th < 0 else 1.0

        def rhs(tt, y):
            return [y[1], _torque(p, force, tt, y[0], y[1], relative) / IO]

        def ev0(tt, y):
            return y[0]
        ev0.terminal, ev0.direction = True, -side

        def evo(tt, y):
            return abs(y[0]) - (alpha + 0.2)
        evo.terminal, evo.direction = True, 1
        sol = solve_ivp(rhs, (t, t_end), [th, w], events=[ev0, evo],
                        dense_output=True, max_step=0.02, **_OPT)
        tt = np.linspace(t, sol.t[-1], max(10, int(400 * (sol.t[-1] - t))))
        yy = sol.sol(tt)
        ts += list(tt[1:])
        ths += list(yy[0, 1:])
        ws += list(yy[1, 1:])
        t = sol.t[-1]
        if sol.status == 1 and len(sol.t_events[1]):
            break
        if sol.status == 1:
            w = float(sol.y[1, -1]) * e
            th = 1e-14 * np.sign(w) if w != 0 else 0.0
            if abs(w) < w_stop:
                rest, th, w = True, 0.0, 0.0
                continue
    return np.array(ts), np.array(ths), np.array(ws)


def free_rock_time(u, th0, i=0):
    """Time for a unit in vacuum released at -|th0| to reach theta = 0.

    Integrates the exact nonlinear pivot equation with DOP853.
    """
    p = _scalar(u, i)
    R2 = p["b"] ** 2 + p["h_c"] ** 2
    IO = p["IG"] + p["mx"] * R2
    R = np.sqrt(R2)
    alpha = np.arctan2(p["b"], p["h_c"])
    Wg = p["Wsub"]

    def rhs(t, y):
        return [y[1], Wg * R * np.sin(alpha + y[0]) / IO]

    def ev(t, y):
        return y[0]
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0, 100.0), [-abs(th0), 0.0], events=ev, **_OPT)
    return float(sol.t_events[0][0]), float(sol.y_events[0][0][1])
