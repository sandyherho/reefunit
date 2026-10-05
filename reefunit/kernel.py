"""Compiled Moreau-Jean kernel.

The algorithm is the one documented in :mod:`reefunit.dynamics`, written as
a per-unit loop and compiled with Numba.  The NumPy implementation in
:mod:`reefunit.dynamics` is kept as the reference and the two agree to
round-off (see the verification report).

The forcing of each unit is the sum of three parts evaluated inside the
kernel:

    u(t) = r(t) [U_c + u_b cos(w t + phi) + u_2 cos(2 w t + 2 phi)]
           + a_s sech^2(s (t - t_0)),

a ramped regular (Stokes) wave on a current, an internal-solitary-wave
pulse, and a tabulated random sea sampled at the step midpoints.
"""

import numpy as np
from numba import njit, prange

from .core import PRM

__all__ = ["run", "forcing_spec"]


def forcing_spec(n, ub=0.0, omega=1.0, Uc=0.0, u2=0.0, phi=0.0,
                 ramp_cycles=2.0, isw_amp=0.0, isw_s=1.0, isw_t0=0.0,
                 ramp_time=None):
    """Broadcast forcing parameters to arrays of length n."""
    def arr(x):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(x, float),
                                                    (n,)))
    om = arr(omega)
    tr = (arr(ramp_time) if ramp_time is not None
          else ramp_cycles * 2 * np.pi / om)
    return dict(ub=arr(ub), om=om, Uc=arr(Uc), u2=arr(u2), phi=arr(phi),
                tr=tr, ia=arr(isw_amp), isn=arr(isw_s), it0=arr(isw_t0))


@njit(cache=True)
def _forcing(i, t, k, ub, om, Uc, u2, phi, tr, ia, isn, it0):
    th = om[i] * t + phi[i]
    f = Uc[i] + ub[i] * np.cos(th) + u2[i] * np.cos(2 * th)
    df = -om[i] * (ub[i] * np.sin(th) + 2 * u2[i] * np.sin(2 * th))
    s = t / tr[i] if tr[i] > 0 else 1.0
    if s >= 1.0:
        r, dr = 1.0, 0.0
    elif s <= 0.0:
        r, dr = 0.0, 0.0
    else:
        r = 0.5 - 0.5 * np.cos(np.pi * s)
        dr = 0.5 * np.pi * np.sin(np.pi * s) / tr[i]
    u = r * f
    du = r * df + dr * f
    if ia[i] != 0.0:
        x = isn[i] * (t - it0[i])
        ch = np.cosh(x)
        se = 1.0 / (ch * ch)
        u += ia[i] * se
        du += -2.0 * ia[i] * isn[i] * se * np.tanh(x)
    return u, du


@njit(parallel=True, cache=True)
def _run(mx, IG, Wsub, FKV, CDA, CLA, zp, hc, b, mu, e, anchor, h, n_steps,
         relative, n_iter, X, Z, th, VX, VZ, W, t0,
         ub, om, Uc, u2, phi, tr, ia, isn, it0,
         rec_every, rec_X, rec_th, rho):
    n = mx.shape[0]
    max_th = np.zeros(n)
    max_lift = np.zeros(n)
    over = np.zeros(n, np.bool_)
    t_over = np.full(n, np.nan)
    max_res = np.zeros(n)
    x0 = X.copy()
    for i in prange(n):
        x, z, a, vx, vz, w = X[i], Z[i], th[i], VX[i], VZ[i], W[i]
        P = np.zeros(4)
        H = np.empty((4, 3))
        Wd = np.empty((4, 4))
        Uf = np.empty(4)
        alpha = np.arctan2(b[i], hc[i])
        for k in range(n_steps):
            if over[i]:
                break
            t = t0 + k * h[i]
            hm = 0.5 * h[i]
            zm = z + hm * vz
            am = a + hm * w
            c, sn = np.cos(am), np.sin(am)
            rxL = -b[i] * c + hc[i] * sn
            ryL = -b[i] * sn - hc[i] * c
            rxR = b[i] * c + hc[i] * sn
            ryR = b[i] * sn - hc[i] * c
            dz = zp[i] - hc[i]
            ryP = dz * c
            uf, duf = _forcing(i, t + hm, k, ub, om, Uc, u2, phi, tr, ia,
                               isn, it0)
            if relative:
                ur = uf - (vx - w * ryP)
                urG = uf - vx
            else:
                ur = uf
                urG = uf
            FD = 0.5 * rho * CDA[i] * ur * abs(ur)
            FL = 0.5 * rho * CLA[i] * urG * urG
            FI = FKV[i] * duf
            ivm = 1.0 / mx[i]
            ivI = 1.0 / IG[i]
            fx = vx + h[i] * (FD + FI) * ivm
            fz = vz + h[i] * (-Wsub[i] - anchor[i] + FL) * ivm
            fw = w + h[i] * (-ryP * FD) * ivI
            actL = zm + ryL <= 1e-8
            actR = zm + ryR <= 1e-8
            # rows: 0 nL, 1 tL, 2 nR, 3 tR ; each row (hx, hz, hth)
            H[0, 0], H[0, 1], H[0, 2] = 0.0, 1.0, rxL
            H[1, 0], H[1, 1], H[1, 2] = 1.0, 0.0, -ryL
            H[2, 0], H[2, 1], H[2, 2] = 0.0, 1.0, rxR
            H[3, 0], H[3, 1], H[3, 2] = 1.0, 0.0, -ryR
            for p in range(4):
                for q in range(4):
                    Wd[p, q] = (H[p, 0] * H[q, 0] * ivm
                                + H[p, 1] * H[q, 1] * ivm
                                + H[p, 2] * H[q, 2] * ivI)
            for p in range(4):
                Uf[p] = H[p, 0] * fx + H[p, 1] * fz + H[p, 2] * fw
            UnkL = vz + w * rxL
            UnkR = vz + w * rxR
            if not actL:
                P[0] = 0.0
                P[1] = 0.0
            if not actR:
                P[2] = 0.0
                P[3] = 0.0
            if actL or actR:
                for _ in range(n_iter):
                    for cc in range(2):
                        act = actL if cc == 0 else actR
                        if not act:
                            continue
                        pn, pt = 2 * cc, 2 * cc + 1
                        unk = UnkL if cc == 0 else UnkR
                        Un = Uf[pn]
                        for q in range(4):
                            Un += Wd[pn, q] * P[q]
                        tgt = Un + e[i] * min(unk, 0.0)
                        P[pn] = max(0.0, P[pn] - tgt / Wd[pn, pn])
                        Ut = Uf[pt]
                        for q in range(4):
                            Ut += Wd[pt, q] * P[q]
                        lim = mu[i] * P[pn]
                        v = P[pt] - Ut / Wd[pt, pt]
                        P[pt] = min(lim, max(-lim, v))
                res = 0.0
                for cc in range(2):
                    act = actL if cc == 0 else actR
                    if not act:
                        continue
                    pn = 2 * cc
                    unk = UnkL if cc == 0 else UnkR
                    Un = Uf[pn]
                    for q in range(4):
                        Un += Wd[pn, q] * P[q]
                    tgt = Un + e[i] * min(unk, 0.0)
                    r = abs(min(tgt, P[pn] * Wd[pn, pn]))
                    if r > res:
                        res = r
                if res > max_res[i]:
                    max_res[i] = res
            nvx = fx + (H[0, 0] * P[0] + H[1, 0] * P[1] + H[2, 0] * P[2]
                        + H[3, 0] * P[3]) * ivm
            nvz = fz + (H[0, 1] * P[0] + H[1, 1] * P[1] + H[2, 1] * P[2]
                        + H[3, 1] * P[3]) * ivm
            nw = fw + (H[0, 2] * P[0] + H[1, 2] * P[1] + H[2, 2] * P[2]
                       + H[3, 2] * P[3]) * ivI
            x += hm * (vx + nvx)
            z += hm * (vz + nvz)
            a += hm * (w + nw)
            vx, vz, w = nvx, nvz, nw
            if abs(a) > max_th[i]:
                max_th[i] = abs(a)
            c2, s2 = np.cos(a), np.sin(a)
            gl = z + (-b[i] * s2 - hc[i] * c2)
            gr = z + (b[i] * s2 - hc[i] * c2)
            lift = min(gl, gr)
            if lift > max_lift[i]:
                max_lift[i] = lift
            if abs(a) > alpha + 0.2:
                over[i] = True
                t_over[i] = t + h[i]
            if rec_every > 0 and (k + 1) % rec_every == 0:
                j = (k + 1) // rec_every - 1
                rec_X[j, i] = x
                rec_th[j, i] = a
        X[i], Z[i], th[i], VX[i], VZ[i], W[i] = x, z, a, vx, vz, w
    return X, Z, th, VX, VZ, W, max_th, max_lift, over, t_over, max_res, x0


class Result:
    """Final state and diagnostics of a batch run."""

    def __init__(self, out, rec_X, rec_th, h, every):
        (self.X, self.Z, self.th, self.VX, self.VZ, self.w, self.max_th,
         self.max_lift, self.over, self.t_over, self.max_res,
         self.X0) = out
        self.rec_X, self.rec_th = rec_X, rec_th
        self.h, self.every = h, every

    def classify(self, disp_tol=0.05, th_tol=0.02):
        """0 stable, 1 displaced (slid or rocked), 2 overturned."""
        out = np.zeros(len(self.X), int)
        moved = ((np.abs(self.X - self.X0) > disp_tol)
                 | (self.max_th > th_tol))
        out[moved] = 1
        out[self.over] = 2
        return out


def run(u, f, h, n_steps, relative=True, n_iter=20, anchor=0.0,
        state=None, record_every=0, t0=0.0):
    """Integrate a batch of units (dict from :func:`units.batch`).

    ``f`` is a dict from :func:`forcing_spec`; ``h`` a scalar or per-unit
    step.  ``state`` optionally gives initial (X, Z, th, VX, VZ, w) arrays;
    by default every unit starts at rest on the bed.
    """
    n = len(u["m"])

    def arr(x):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(x, float),
                                                    (n,))).copy()
    if state is None:
        X, Z, th = np.zeros(n), arr(u["h_c"]), np.zeros(n)
        VX, VZ, W = np.zeros(n), np.zeros(n), np.zeros(n)
    else:
        X, Z, th, VX, VZ, W = (arr(v) for v in state)
    nrec = n_steps // record_every if record_every > 0 else 1
    rec_X = np.zeros((nrec, n))
    rec_th = np.zeros((nrec, n))
    hh = arr(h)
    out = _run(arr(u["mx"]), arr(u["IG"]), arr(u["Wsub"]), arr(u["FKV"]),
               arr(u["CDA"]), arr(u["CLA"]), arr(u["z_p"]), arr(u["h_c"]),
               arr(u["b"]), arr(u["mu"]), arr(u["e"]), arr(anchor), hh,
               int(n_steps), bool(relative), int(n_iter), X, Z, th, VX, VZ,
               W, float(t0), f["ub"], f["om"], f["Uc"], f["u2"], f["phi"],
               f["tr"], f["ia"], f["isn"], f["it0"], int(record_every),
               rec_X, rec_th, float(PRM.rho))
    return Result(out, rec_X if record_every else None,
                  rec_th if record_every else None, hh, record_every)
