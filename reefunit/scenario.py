"""Reference configuration shared by every figure.

Site classes stand for the settings practitioners recognise: a sheltered
bay, a coast exposed to monsoon wind sea, a south-facing coast exposed to
Indian Ocean swell, and a strait with a strong current.  Design waves are
regular waves meant to represent a storm at the site.  Colony counts,
mounting heights and growth rates are typical of transplant practice.
All values are illustrative inputs, not estimates for a particular site.
"""

import numpy as np

__all__ = ["SITES", "SITE_ORDER", "COLONIES", "R0_COLONY", "GROWTH",
           "R_GROWN", "ISW_REF", "ETA_GRID", "DEPTHS", "PERIODS",
           "SPC", "UNIT_ORDER", "UNIT_LABEL", "MATERIAL_ORDER",
           "COEF_BOUNDS", "sample_coefficients"]

SITES = {
    "sheltered": dict(H=0.8, T=5.0, Uc=0.0, depth=5.0),
    "monsoon": dict(H=2.0, T=8.0, Uc=0.0, depth=8.0),
    "swell": dict(H=3.0, T=14.0, Uc=0.0, depth=10.0),
    "strait": dict(H=1.2, T=6.0, Uc=0.8, depth=8.0),
}
SITE_ORDER = ["sheltered", "monsoon", "swell", "strait"]

UNIT_ORDER = ["block", "dome", "table", "spider", "tetra"]
UNIT_LABEL = {"block": "block", "dome": "dome", "table": "rebar table",
              "spider": "hex spider", "tetra": "tetra frame"}
MATERIAL_ORDER = ["polymer", "ceramic", "concrete", "limestone",
                  "coated steel", "steel"]

# colonies per unit and mounting heights as fractions of unit height:
# (count, top mount, side mount)
COLONIES = {
    "block": (4, 1.00, 0.50),
    "dome": (12, 0.75, 0.35),
    "table": (20, 1.00, 0.50),
    "spider": (12, 0.80, 0.40),
    "tetra": (12, 0.70, 0.35),
}
R0_COLONY = 0.03               # colony radius at transplanting, m
GROWTH = [0.01, 0.03, 0.05]    # radial growth rate, m per year
R_GROWN = 0.10                 # reference grown colony radius, m

# two-layer internal solitary wave on a shallow slope: pycnocline 3 m
# above a 15 m bed, 5 kg m^-3 density step
ISW_REF = dict(h1=12.0, h2=3.0, drho=5.0)
ETA_GRID = np.linspace(0.1, 1.5, 15)

DEPTHS = np.linspace(2.0, 20.0, 91)
PERIODS = np.linspace(3.0, 16.0, 66)
SPC = 400                      # time steps per wave period

# plausible ranges for the uncalibrated coefficients: multiplicative for
# the drag and lift areas, absolute for the added mass and friction
COEF_BOUNDS = {
    "CD_factor": (0.7, 1.5),
    "CL_factor": (0.0, 2.0),
    "C_a": (0.5, 1.5),
    "mu": (0.4, 0.8),
}


def sample_coefficients(n, rng):
    """Independent uniform samples of the uncalibrated coefficients."""
    return {k: rng.uniform(lo, hi, n) for k, (lo, hi) in
            COEF_BOUNDS.items()}
