"""Place a unit's outline and its colonies in world coordinates.

A unit's outline is stored relative to the centre of its base.  The body
frame is anchored at the centre of mass, so drawing a configuration means
rotating the outline about that point and translating it to where the
solver put it.

While a unit rocks it pivots about one base corner and therefore rises:
with R the distance from a pivot to the centre of mass and alpha the
slenderness angle, the centre of mass sits at R cos(alpha - |theta|),
which is h_c at rest and falls back to it at every impact.  Using that
height rather than h_c is what keeps the drawn corner on the bed.
"""

import numpy as np

from .scenario import COLONIES

__all__ = ["com_height", "transform", "colony_centers"]


def com_height(unit, th):
    """Centre-of-mass height of a unit rocking on a base corner (m)."""
    R = np.hypot(unit.b, unit.h_c)
    alpha = np.arctan2(unit.b, unit.h_c)
    return R * np.cos(alpha - np.abs(np.asarray(th, float)))


def transform(unit, X, th, Z=None, lines=None):
    """Map a unit's outline into world coordinates.

    ``Z`` defaults to :func:`com_height`, the height the body reaches by
    pivoting on a corner.
    """
    Z = com_height(unit, th) if Z is None else Z
    c, s = np.cos(th), np.sin(th)
    out = []
    for line in (lines if lines is not None else unit.outline):
        p = np.asarray(line, float)
        px, pz = p[:, 0], p[:, 1] - unit.h_c
        out.append(np.column_stack([X + c * px - s * pz,
                                    Z + s * px + c * pz]))
    return out


def colony_centers(unit, X, th, n=None, z_mount=None, spread=None, Z=None):
    """World positions of n colonies spread across the mounting surface."""
    if n is None:
        n, ftop, _ = COLONIES[unit.name]
        z_mount = ftop * unit.H
    Z = com_height(unit, th) if Z is None else Z
    spread = unit.top_half if spread is None else spread
    m = max(int(np.ceil(n / 2)), 1)
    xs = np.linspace(-spread, spread, m) if m > 1 else np.array([0.0])
    c, s = np.cos(th), np.sin(th)
    pz = z_mount - unit.h_c
    return np.column_stack([X + c * xs - s * pz, Z + s * xs + c * pz])
