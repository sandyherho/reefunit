"""CSV writers and the plain-text report writer."""

import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "outputs", "data")
REPDIR = os.path.join(ROOT, "outputs", "reports")

CACHEDIR = os.path.join(ROOT, "outputs", "cache")

__all__ = ["write_csv", "write_report", "fmt", "save_cache", "load_cache"]


def fmt(x, n=12):
    """Format a real or complex value in fixed exponential notation."""
    if isinstance(x, complex):
        return f"{x.real:.{n}e}{x.imag:+.{n}e}j"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{n}e}"
    return str(x)


def write_csv(stem, columns):
    """Write columns to CSV, splitting complex arrays into re and im.

    ``columns`` maps a column name to a one-dimensional array.
    """
    os.makedirs(DATADIR, exist_ok=True)
    names, cols = [], []
    for name, v in columns.items():
        v = np.asarray(v)
        if np.iscomplexobj(v):
            names += [f"{name}_re", f"{name}_im"]
            cols += [v.real, v.imag]
        else:
            names.append(name)
            cols.append(v)
    n = len(cols[0])
    path = os.path.join(DATADIR, f"{stem}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow([f"{c[i]:.12e}" for c in cols])
    return path


def write_report(stem, title, blocks):
    """blocks: list of (heading, list-of-lines)."""
    os.makedirs(REPDIR, exist_ok=True)
    path = os.path.join(REPDIR, f"{stem}.txt")
    with open(path, "w") as fh:
        fh.write(title + "\n" + "=" * len(title) + "\n\n")
        for head, lines in blocks:
            fh.write(head + "\n" + "-" * len(head) + "\n")
            for ln in lines:
                fh.write(ln + "\n")
            fh.write("\n")
    return path


def save_cache(stem, **arrays):
    """Store intermediate arrays shared between scripts (not committed)."""
    os.makedirs(CACHEDIR, exist_ok=True)
    path = os.path.join(CACHEDIR, f"{stem}.npz")
    np.savez_compressed(path, **arrays)
    return path


def load_cache(stem):
    """Load arrays written by :func:`save_cache`, or None if absent."""
    path = os.path.join(CACHEDIR, f"{stem}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
