"""Run every figure script, then the animations, then the reports.

The figures take a few minutes in total on one core; the animations take
longer, because every frame is rendered with Matplotlib.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = [f"fig{i:02d}_{n}.py" for i, n in enumerate(
    ["schematic", "forcing", "statics", "regimes", "ratchet", "isw",
     "growth", "sensitivity", "verification"])]


def run(name):
    """Run one script and report its wall time."""
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, name)],
                       cwd=HERE)
    dt = time.time() - t0
    print(f"{name:28s} {'ok' if r.returncode == 0 else 'FAILED':6s} "
          f"{dt:7.1f} s", flush=True)
    return r.returncode


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = []
    if which in ("all", "figures"):
        todo += FIGS
    if which in ("all", "animations"):
        todo += ANIMS
    if which in ("all", "reports"):
        todo += ["make_reports.py"]
    bad = sum(run(n) != 0 for n in todo)
    print(f"{len(todo) - bad} of {len(todo)} scripts succeeded")
    sys.exit(1 if bad else 0)
