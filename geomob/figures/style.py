"""Figure style for the paper.

One place to change fonts, sizes and colours so every figure in the manuscript
matches. Defaults target a two-column LaTeX article: `WIDTH_1COL` is a single
column, `WIDTH_2COL` spans both.

Colours are the Okabe-Ito palette, which stays distinguishable under the three
common forms of colour blindness and in greyscale print. Nothing here relies
on colour alone to carry meaning: series are also distinguished by marker and
line style, and bar groups by hatch.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

WIDTH_1COL = 3.4   # inches
WIDTH_2COL = 7.0

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#8C8C8C"

PALETTE = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY, YELLOW, BLACK]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
HATCHES = ["", "///", "...", "\\\\\\", "xxx", "+++"]

# Semantic colours, used consistently across figures: the reader should be
# able to learn "orange is GPE" once and keep it for the whole paper.
METHOD_COLORS = {
    "GPE": VERMILLION,
    "GPE(h=128)": VERMILLION,
    "GPE(h=256)": "#8C3A00",
    "Space2Vec": PURPLE,
    "TorusLattice(i)": BLUE,
    "GEO(i)+disp": BLUE,
    "GEO(i)+disp [d=128]": BLUE,
    "GEO(i)+disp+time": SKY,
    "GEO(iii)+disp+time": GREEN,
    "DisplacementGEO": GREEN,
    "Grid": GREY,
    "Grid(150m)": GREY,
    "RawXY": "#5C5C5C",
    "XY(SS)": "#5C5C5C",
    "Fourier/TriW": ORANGE,
    "TriW": ORANGE,
}


def color_for(name: str, i: int = 0) -> str:
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    for key, c in METHOD_COLORS.items():
        if key.split("(")[0] and name.startswith(key.split("(")[0]):
            return c
    return PALETTE[i % len(PALETTE)]


def use_paper_style(font: str = "serif", base: float = 8.0):
    """Apply the manuscript style. Call once before building figures."""
    mpl.rcParams.update({
        "font.family": font,
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,     # embed TrueType, not Type 3: required by many
        "ps.fonttype": 42,      # publishers and keeps text selectable
        "text.usetex": False,
    })


def save(fig, out_dir, name: str, formats=("pdf", "png")) -> list[Path]:
    """Write one figure in every requested format. PDF is the one to \\includegraphics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths


def annotate_bars(ax, bars, values, fmt="{:.3f}", dy=0.01, fontsize=6,
                  rotation=0):
    """Numeric labels above bars, so the figure doubles as a table."""
    for b, v in zip(bars, values):
        if v is None:
            continue
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2,
                                    b.get_height()),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", va="bottom", fontsize=fontsize,
                    rotation=rotation)


def shorten(name: str, n: int = 18) -> str:
    return name if len(name) <= n else name[:n - 1] + "\u2026"
