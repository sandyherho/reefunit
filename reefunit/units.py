"""Transplant units as planar rigid bodies.

A unit is reduced to the quantities that enter the planar dynamics: the
solid volume V and density rho_s, the height of the centre of mass h_c,
the moment of inertia about the centre of mass for rotation about the
out-of-plane axis, the half-distance b between the two base contacts that
act as rocking pivots, and the hydrodynamic data: drag coefficient times
frontal area C_D A, the height z_p of its centroid (the centre of
pressure), lift coefficient times plan area C_L A_L, and the added-mass
coefficient C_a.  Flow is along x; the out-of-plane direction is y.

Solid bodies (block, dome) use closed-form geometry.  Frames (table,
hexagonal spider, tetrahedral frame) are built from straight rods in
three dimensions, discretised into points, so that mass,
inertia, frontal area and centre of pressure follow from one code path.
Frontal area of a rod is its diameter times the length of its projection
on the plane normal to the flow.  Every dimension and coefficient is
illustrative: typical of units deployed on Indonesian reefs, not taken
from a particular design drawing.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from .core import PRM

__all__ = ["Unit", "MATERIALS", "block", "dome", "table", "spider",
           "tetra", "catalogue", "with_material", "with_colonies",
           "batch", "NATIVE"]

MATERIALS = {
    "concrete": 2300.0,
    "limestone": 2600.0,
    "ceramic": 2000.0,
    "steel": 7850.0,
    "coated steel": 4285.0,   # 10 mm rebar in a 16 mm sand-resin coat
    "polymer": 1380.0,
}


@dataclass(frozen=True)
class Unit:
    """Mechanical data of one unit (SI)."""

    name: str
    material: str
    rho_s: float          # density of the solid, kg m^-3
    V: float              # solid volume, m^3
    h_c: float            # centre-of-mass height, m
    k2: float             # squared radius of gyration about the COM, m^2
    b: float              # half-distance between the rocking pivots, m
    H: float              # overall height, m
    CDA: float            # drag coefficient times frontal area, m^2
    z_p: float            # centre-of-pressure height, m
    CLA: float = 0.0      # lift coefficient times plan area, m^2
    C_a: float = 1.0      # added-mass coefficient
    mu: float = 0.6       # Coulomb friction coefficient on the bed
    e: float = 0.0        # normal restitution at a contact
    top_half: float = 0.0  # half-width of the mounting surface, m
    outline: tuple = field(default=(), repr=False, compare=False)

    @property
    def m(self):
        """Mass, kg."""
        return self.rho_s * self.V

    @property
    def Wsub(self):
        """Submerged weight, N."""
        return (self.rho_s - PRM.rho) * PRM.g * self.V

    @property
    def alpha(self):
        """Slenderness angle atan(b/h_c)."""
        return np.arctan2(self.b, self.h_c)

    @property
    def R(self):
        """Distance from a pivot to the centre of mass, m."""
        return np.hypot(self.b, self.h_c)


# --------------------------------------------------------------- rod frames
def _rod_points(rods, npts=40):
    """Discretise rods into mass points and frontal-area elements.

    ``rods`` is a list of (p0, p1, d) with 3D endpoints (x, y, z) and a
    diameter d.  Returns point arrays (x, z, dm_volume, dA_frontal).
    """
    xs, zs, dv, da = [], [], [], []
    for p0, p1, d in rods:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        s = (np.arange(npts) + 0.5) / npts
        pts = p0 + s[:, None] * (p1 - p0)
        proj = np.hypot(p1[1] - p0[1], p1[2] - p0[2])
        xs.append(pts[:, 0])
        zs.append(pts[:, 2])
        dv.append(np.full(npts, 0.25 * np.pi * d ** 2 * L / npts))
        da.append(np.full(npts, d * proj / npts))
    return (np.concatenate(xs), np.concatenate(zs), np.concatenate(dv),
            np.concatenate(da))


def _frame(name, material, rods, b, H, C_D=1.2, C_a=1.0, top_half=0.0,
           outline=()):
    """Assemble a Unit from rods of one material."""
    x, z, dv, da = _rod_points(rods)
    V = dv.sum()
    xg = (x * dv).sum() / V
    zg = (z * dv).sum() / V
    k2 = (((x - xg) ** 2 + (z - zg) ** 2) * dv).sum() / V
    A = da.sum()
    z_p = (z * da).sum() / A
    return Unit(name=name, material=material, rho_s=MATERIALS[material],
                V=V, h_c=zg, k2=k2, b=b, H=H, CDA=C_D * A, z_p=z_p,
                CLA=0.0, C_a=C_a, top_half=top_half, outline=outline)


# ---------------------------------------------------------------- catalogue
def block(width=0.40, height=0.25, depth=0.40, material="concrete",
          C_D=1.2, C_L=0.4, C_a=1.0):
    """Solid rectangular block."""
    b = 0.5 * width
    V = width * height * depth
    k2 = (width ** 2 + height ** 2) / 12.0
    ol = (((-b, 0), (b, 0), (b, height), (-b, height), (-b, 0)),)
    return Unit("block", material, MATERIALS[material], V, 0.5 * height, k2,
                b, height, C_D * depth * height, 0.5 * height,
                CLA=C_L * width * depth, C_a=C_a, top_half=b, outline=ol)


def dome(R=0.45, t=0.08, open_frac=0.30, material="concrete", C_D=0.9,
         C_L=0.3, C_a=0.6):
    """Perforated hemispherical shell (outer radius R, wall t)."""
    r = R - t
    shell = (2.0 / 3.0) * np.pi * (R ** 3 - r ** 3)
    V = (1.0 - open_frac) * shell
    zc = 3.0 * (R ** 4 - r ** 4) / (8.0 * (R ** 3 - r ** 3))
    k2_base = 0.4 * (R ** 5 - r ** 5) / (R ** 3 - r ** 3)
    k2 = k2_base - zc ** 2
    A = (1.0 - open_frac) * 0.5 * np.pi * R ** 2
    ph = np.linspace(0, np.pi, 60)
    ol = (tuple(zip(R * np.cos(ph), R * np.sin(ph))),
          tuple(zip(r * np.cos(ph), r * np.sin(ph))),
          ((-R, 0), (R, 0)))
    return Unit("dome", material, MATERIALS[material], V, zc, k2, R, R,
                C_D * A, 4 * R / (3 * np.pi),
                CLA=C_L * (1 - open_frac) * np.pi * R ** 2, C_a=C_a,
                top_half=0.5 * R, outline=ol)


def table(span=1.0, depth=0.8, height=0.6, d_leg=0.016, d_bar=0.012,
          n_cross=6, material="steel", C_D=1.2, C_a=1.0):
    """Rebar table: four legs and a grid of cross bars at the top."""
    b = 0.5 * span
    rods = []
    for sx in (-b, b):
        for sy in (-0.5 * depth, 0.5 * depth):
            rods.append(((sx, sy, 0.0), (sx, sy, height), d_leg))
    for xi in np.linspace(-b, b, n_cross):
        rods.append(((xi, -0.5 * depth, height), (xi, 0.5 * depth, height),
                     d_bar))
    for sy in (-0.5 * depth, 0.5 * depth):
        rods.append(((-b, sy, height), (b, sy, height), d_bar))
    ol = (((-b, 0), (-b, height), (b, height), (b, 0)),)
    return _frame("table", material, rods, b, height, C_D, C_a,
                  top_half=b, outline=ol)


def spider(R_f=0.55, height=0.35, d=0.016, material="coated steel",
           C_D=1.2, C_a=1.0):
    """Hexagonal spider: six legs from a raised hub to a ground ring.

    The pivot half-distance is the apothem R_f cos(30 deg): the frame tips
    about the line through two adjacent feet.
    """
    ang = np.deg2rad(np.arange(6) * 60.0)
    feet = [(R_f * np.cos(a), R_f * np.sin(a), 0.0) for a in ang]
    hub = (0.0, 0.0, height)
    rods = [(hub, f, d) for f in feet]
    rods += [(feet[i], feet[(i + 1) % 6], d) for i in range(6)]
    b = R_f * np.cos(np.pi / 6)
    ol = (((-b, 0), (0, height), (b, 0)), ((-R_f, 0), (R_f, 0)))
    return _frame("spider", material, rods, b, height, C_D, C_a,
                  top_half=0.25 * b, outline=ol)


def tetra(side=1.0, height=0.75, d=0.07, material="concrete", C_D=1.2,
          C_a=1.0):
    """Tetrahedral rod frame, one base edge facing the flow.

    The pivot half-distance is the inradius of the base triangle, the
    smaller of the two lever arms and hence the conservative choice.
    """
    rin = side / (2.0 * np.sqrt(3.0))
    rcirc = side / np.sqrt(3.0)
    # base edge normal to the flow at x = +rin, opposite vertex at -rcirc
    v = [(rin, -0.5 * side, 0.0), (rin, 0.5 * side, 0.0), (-rcirc, 0.0, 0.0)]
    apex = (0.0, 0.0, height)
    rods = [(v[i], v[(i + 1) % 3], d) for i in range(3)]
    rods += [(vi, apex, d) for vi in v]
    ol = (((-rin, 0), (0, height), (rin, 0), (-rin, 0)),)
    return _frame("tetra", material, rods, rin, height, C_D, C_a,
                  top_half=0.2 * rin, outline=ol)


def catalogue():
    """The five reference units in their native materials."""
    return [block(), dome(), table(), spider(), tetra()]


NATIVE = {"block": "concrete", "dome": "concrete", "table": "steel",
          "spider": "coated steel", "tetra": "concrete"}


def with_material(u, material):
    """Same geometry, different solid."""
    return replace(u, material=material, rho_s=MATERIALS[material])


def with_colonies(u, n, r, z_mount, rho_c=1500.0, solid=0.5, C_Dc=1.0):
    """Add n hemispherical colonies of radius r mounted at height z_mount.

    Each colony has frontal area pi r^2/2 with centroid 4r/(3 pi) above its
    mount, a solid fraction ``solid`` of bulk density ``rho_c``, and adds its
    submerged weight and inertia (spread across the mounting surface).
    The unit's own density is replaced by an effective density that keeps
    the submerged weight and mass exact.
    """
    if n == 0 or r <= 0:
        return u
    Ac = n * 0.5 * np.pi * r ** 2
    zc_drag = z_mount + 4.0 * r / (3.0 * np.pi)
    CDA = u.CDA + C_Dc * Ac
    z_p = (u.CDA * u.z_p + C_Dc * Ac * zc_drag) / CDA
    Vc = n * solid * (2.0 / 3.0) * np.pi * r ** 3
    mc = rho_c * Vc
    zc_mass = z_mount + 3.0 * r / 8.0
    m = u.m + mc
    V = u.V + Vc
    h_c = (u.m * u.h_c + mc * zc_mass) / m
    spread = u.top_half ** 2 / 3.0
    inertia = (u.m * (u.k2 + (u.h_c - h_c) ** 2)
               + mc * (spread + (zc_mass - h_c) ** 2))
    return replace(u, V=V, rho_s=m / V, h_c=h_c, k2=inertia / m, CDA=CDA,
                   z_p=z_p)


def batch(units, fluid=True, rot_am=1.0, **override):
    """Stack units into a dict of arrays for the vectorised solvers.

    Keyword overrides (arrays or scalars) replace a field for every unit,
    e.g. ``mu=0.4``.  Derived fields are computed after the override.
    With ``fluid=False`` the body is in vacuum: no buoyancy, added mass or
    hydrodynamic load, which is the setting of the classical rocking-block
    results used for verification.  Added mass acts on the translations;
    ``rot_am`` is the fraction of it that also acts on the rotation, 1 by
    default (isotropic) and 0 for translations only.  The sensitivity of
    the results to that choice is reported with the figures.
    """
    keys = ["rho_s", "V", "h_c", "k2", "b", "H", "CDA", "z_p", "CLA",
            "C_a", "mu", "e"]
    d = {k: np.array([getattr(u, k) for u in units], float) for k in keys}
    for k, v in override.items():
        d[k] = np.broadcast_to(np.asarray(v, float), d["V"].shape).copy()
    rho, g = (PRM.rho if fluid else 0.0), PRM.g
    if not fluid:
        d["CDA"][:] = 0.0
        d["CLA"][:] = 0.0
    d["m"] = d["rho_s"] * d["V"]
    d["mx"] = d["m"] + d["C_a"] * rho * d["V"]
    d["IG"] = (d["m"] + rot_am * d["C_a"] * rho * d["V"]) * d["k2"]
    d["Wsub"] = (d["rho_s"] - rho) * g * d["V"]
    d["FKV"] = rho * d["V"] * (1.0 + d["C_a"])
    return d


def expand(d, shape):
    """Broadcast a batch of n units to arrays of shape (n,) + shape."""
    n = len(d["m"])
    return {k: np.broadcast_to(np.asarray(v).reshape((n,) + (1,) * len(
        shape)), (n,) + tuple(shape)) for k, v in d.items()}


def grown(u, r, where="top", colonies=None):
    """Unit with its reference colonies at radius r (top or side mount)."""
    from .scenario import COLONIES
    n, ftop, fside = (colonies or COLONIES)[u.name]
    z = (ftop if where == "top" else fside) * u.H
    return with_colonies(u, n, r, z)


__all__ += ["expand", "grown"]
