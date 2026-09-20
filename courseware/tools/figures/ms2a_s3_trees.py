"""Figures for the `ms2a-machine-learning-practice` lesson
`s3-tabular-models/trees-and-ensembles.md`.

One figure so far: `gini-by-feature.png`, the worked Gini example on the
14-row AllElectronics "buys computer" table (Han, Kamber & Pei, *Data Mining*,
3rd ed., table 8.1). The same slide exists in `python-ai-engineering`
(`s3-models-and-tuning/decision-trees.md`), so the PNG is written to both
courses' asset trees. Re-run with:

    uv run --no-project --with matplotlib --with numpy \
        python courseware/tools/figures/ms2a_s3_trees.py

Every number in the figure is computed from the count table below, not typed
in, and the script asserts the weighted impurities the lesson quotes (Age 0.343,
Student 0.367, Credit Rating 0.429, Income 0.440) and that entropy ranks the four
features in the same order. The drawing has no random element; the seed is set
for the convention only. The previous still drew Credit Rating with Yes and No
swapped (5 yes / 9 no against a parent of 9 yes / 5 no) — Gini is symmetric, so
its 0.429 was right while its counts were not.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

np.random.seed(0)

ROOT = pathlib.Path(__file__).resolve().parents[2] / "content"
TARGETS = (
    ROOT / "ms2a-machine-learning-practice" / "assets" / "tabular",
    ROOT
    / "python-ai-engineering"
    / "assets"
    / "s3-models-and-tuning"
    / "decision-trees",
)

# Parent: 9 yes, 5 no. Each feature: (level, yes, no), in the order drawn.
FEATURES = {
    "Age": [("Youth", 2, 3), ("Middle age", 4, 0), ("Senior", 3, 2)],
    "Income": [("High", 2, 2), ("Medium", 4, 2), ("Low", 3, 1)],
    "Student": [("Yes", 6, 1), ("No", 3, 4)],
    "Credit Rating": [("Fair", 6, 2), ("Excellent", 3, 3)],
}
EXPECTED = {"Age": 0.343, "Income": 0.440, "Student": 0.367,
            "Credit Rating": 0.429}

# Fill colours of the count tables, one per feature, as in the original still.
FILL = {
    "Age": "#c8e6c0",
    "Income": "#9ed9d6",
    "Student": "#bfe3f7",
    "Credit Rating": "#fdeaa8",
}
INK = "#1f2933"
BOX = "#4a4a4a"
RED = "#c8323c"

W_PX, H_PX, DPI = 1200, 747, 100


def gini(yes: int, no: int) -> float:
    n = yes + no
    return 1.0 - (yes / n) ** 2 - (no / n) ** 2


def entropy(yes: int, no: int) -> float:
    n = yes + no
    return -sum((c / n) * math.log2(c / n) for c in (yes, no) if c)


def weighted(measure, levels) -> float:
    n = sum(y + no for _, y, no in levels)
    return sum((y + no) / n * measure(y, no) for _, y, no in levels)


def check() -> None:
    for name, levels in FEATURES.items():
        assert sum(y for _, y, _ in levels) == 9, name
        assert sum(no for _, _, no in levels) == 5, name
        got = round(weighted(gini, levels), 3)
        assert got == EXPECTED[name], (name, got)
    by_gini = sorted(FEATURES, key=lambda f: weighted(gini, FEATURES[f]))
    by_entropy = sorted(FEATURES,
                        key=lambda f: weighted(entropy, FEATURES[f]))
    assert by_gini == by_entropy == ["Age", "Student", "Credit Rating",
                                     "Income"], (by_gini, by_entropy)


def fmt(x: float) -> str:
    """0.48, 0, 0.5, 0.375: as many decimals as the value needs, at most 3."""
    return f"{x:.3f}".rstrip("0").rstrip(".") if x else "0"


def draw_table(ax, x, y, yes, no, fill) -> None:
    """A three-row count table with its top-left corner at (x, y)."""
    w_label, w_value, h = 0.065, 0.06, 0.036
    rows = [("Yes", str(yes), False), ("No", str(no), False),
            ("Gini", fmt(gini(yes, no)), True)]
    for i, (label, value, bold) in enumerate(rows):
        yy = y - (i + 1) * h
        ax.add_patch(Rectangle((x, yy), w_label, h, facecolor=fill,
                               edgecolor=BOX, linewidth=1.0))
        ax.add_patch(Rectangle((x + w_label, yy), w_value, h,
                               facecolor="white", edgecolor=BOX,
                               linewidth=1.0))
        ax.text(x + 0.008, yy + h / 2, label, va="center", ha="left",
                fontsize=12, fontweight="bold" if bold else "normal",
                color=INK)
        ax.text(x + w_label + w_value / 2, yy + h / 2, value, va="center",
                ha="center", fontsize=12, color=INK)


def draw_feature(ax, name, cx, top, table_w=0.125) -> None:
    """A feature node, its branches and one count table per level."""
    levels = FEATURES[name]
    ax.add_patch(FancyBboxPatch((cx - 0.075, top - 0.075), 0.15, 0.075,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor="white", edgecolor=BOX,
                                linewidth=1.8))
    ax.text(cx, top - 0.0375, name, ha="center", va="center", fontsize=14,
            color=INK)
    n = len(levels)
    spread = 0.17
    xs = [cx + (i - (n - 1) / 2) * spread for i in range(n)]
    y_fork = top - 0.075 - 0.03
    y_table = top - 0.075 - 0.145
    for (level, yes, no), x in zip(levels, xs):
        # elbow connector: down from the node, across, down to the table
        ax.plot([cx, cx], [top - 0.075, y_fork], color=BOX, linewidth=1.2)
        ax.plot([cx, x], [y_fork, y_fork], color=BOX, linewidth=1.2)
        ax.annotate("", xy=(x, y_table + 0.004), xytext=(x, y_fork),
                    arrowprops=dict(arrowstyle="-|>", color=BOX,
                                    linewidth=1.2, shrinkA=0, shrinkB=0))
        ax.text(x, y_fork - 0.032, level, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        draw_table(ax, x - table_w / 2, y_table, yes, no, FILL[name])
    caption_y = y_table - 3 * 0.036 - 0.03
    ax.text(cx, caption_y,
            f"Gini impurity for {name} is {weighted(gini, levels):.3f}",
            ha="center", va="center", fontsize=13, color=INK)
    return cx, caption_y


def gini_by_feature() -> None:
    check()
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    age_caption = draw_feature(ax, "Age", 0.26, 0.975)
    draw_feature(ax, "Income", 0.74, 0.975)
    draw_feature(ax, "Student", 0.26, 0.47)
    draw_feature(ax, "Credit Rating", 0.74, 0.47)

    # the winner
    x0, y0 = age_caption
    ax.add_patch(FancyArrowPatch((0.50, 0.40), (x0 + 0.15, y0),
                                 connectionstyle="arc3,rad=0.25",
                                 arrowstyle="-|>", mutation_scale=22,
                                 color=RED, linewidth=3))
    ax.text(0.505, 0.385, "Best", ha="left", va="center", fontsize=17,
            fontweight="bold", color=INK)

    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        out = target / "gini-by-feature.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        print(f"  wrote {out.relative_to(ROOT.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    gini_by_feature()
