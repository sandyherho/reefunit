"""Dark figure style, colour maps, and looping GIF export for animations.

Frames are rendered one by one with Matplotlib into RGB arrays and written
as palette GIFs with Pillow.  Every frame is a direct render of a computed
field or of a simulated configuration; nothing is interpolated or
exaggerated for visual effect.  Each animation carries a colour bar in SI
units and draws the units to scale, so that what is pleasant to watch is
also readable as a measurement.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

matplotlib.use("Agg")

__all__ = ["dark", "SPEED", "SIGNED", "fig_to_rgb", "write_gif", "ANIMDIR",
           "BG", "FG", "scale_bar"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

BG = "#07090f"
FG = "#d9dee8"

# speed, zero at the background: still water disappears into the page
SPEED = LinearSegmentedColormap.from_list("speed", [
    (0.00, BG), (0.22, "#10364f"), (0.45, "#1f7a8c"),
    (0.70, "#3fc1b0"), (0.88, "#9fe88d"), (1.00, "#fff4d6"),
])

# signed velocity: offshore blue, onshore amber, zero at the background
SIGNED = LinearSegmentedColormap.from_list("signed", [
    (0.00, "#e8f7ff"), (0.18, "#35c3ff"), (0.36, "#1446a0"),
    (0.50, BG), (0.64, "#8a1c0a"), (0.82, "#ff8a1f"), (1.00, "#fff4d6"),
])


def dark():
    """Apply the animation style."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#3a4152",
        "xtick.color": FG, "ytick.color": FG, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.6, "figure.dpi": 110,
    })


def scale_bar(ax, length, label, x0, y0, color=FG, lw=1.6, dy=0.03):
    """Horizontal scale bar of a stated length in data coordinates."""
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=lw,
            solid_capstyle="butt")
    for xx in (x0, x0 + length):
        ax.plot([xx, xx], [y0 - dy, y0 + dy], color=color, lw=lw)
    ax.text(x0 + 0.5 * length, y0 + 1.6 * dy, label, color=color,
            ha="center", va="bottom", fontsize=8)


def fig_to_rgb(fig):
    """Rasterise a figure to an (h, w, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def write_gif(stem, frames, fps=16, colors=192):
    """Write a looping, palette-quantised GIF and return its path."""
    os.makedirs(ANIMDIR, exist_ok=True)
    path = os.path.join(ANIMDIR, f"{stem}.gif")
    imgs = [Image.fromarray(f).quantize(colors=colors, method=Image.MEDIANCUT,
                                        dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(round(1000 / fps)), optimize=True, disposal=2)
    return path
