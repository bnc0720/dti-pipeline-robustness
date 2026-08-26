# -*- coding: utf-8 -*-
"""
figstyle.py
Shared publication style (Helvetica/Arial look, trimmed spines, clean ticks).
Import and call apply() once, then use despine() / panel() per axes.
"""
import matplotlib as mpl

# Helvetica/Arial-compatible stack. On the authors' machines Arial is picked;
# in this build Nimbus Sans / TeX Gyre Heros (Helvetica clones) render crisply.
FONT_STACK = ["Arial", "Helvetica", "Nimbus Sans", "TeX Gyre Heros",
              "Liberation Sans", "DejaVu Sans"]
INK = "#1a1a1a"
AXIS = "#3a3a3a"
GRID = "#e9e9e9"
GREY = "#b9b9b9"


def apply():
    mpl.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def despine(ax, offset=4, trim=True, which=("left", "bottom")):
    """Hide top/right spines; offset + trim the kept spines to their tick range."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if "left" in which and ax.spines["left"].get_visible():
        sp = ax.spines["left"]
        sp.set_position(("outward", offset))
        if trim:
            yt = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
            if len(yt) >= 2:
                sp.set_bounds(min(yt), max(yt))
    if "bottom" in which and ax.spines["bottom"].get_visible():
        sp = ax.spines["bottom"]
        sp.set_position(("outward", offset))
        if trim:
            xt = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
            if len(xt) >= 2:
                sp.set_bounds(min(xt), max(xt))


def panel(ax, letter, dx=-26, dy=10):
    """Bold panel label just outside the top-left corner of the axes."""
    ax.annotate(letter, xy=(0, 1), xytext=(dx, dy), xycoords="axes fraction",
                textcoords="offset points", fontsize=11, fontweight="bold",
                va="bottom", ha="left", color="#000000")
