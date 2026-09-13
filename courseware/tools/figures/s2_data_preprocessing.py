"""Figures generated for `ms2a-machine-learning-practice` Session 2.

Every image this script writes is committed under
`content/ms2a-machine-learning-practice/assets/preprocessing/`. Re-run with:

    uv run --with matplotlib --with numpy \
        python courseware/tools/figures/s2_data_preprocessing.py

The session's other figures are not produced here: seven are copies of
`python-ai-engineering` data-preparation figures, six are plot outputs lifted
from the module5 example notebooks. This script owns only the figures authored
for this session.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "ms2a-machine-learning-practice"
    / "assets"
    / "preprocessing"
)

INK = "#1f2933"
ACCENT = "#2f6f9f"
WARM = "#c1553b"
TRAIN_FILL = "#dce8f2"
TEST_FILL = "#f6dcd4"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "text.color": INK,
        "font.family": "DejaVu Sans",
    }
)


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ASSETS.parents[3])}")


# --------------------------------------------------------------------------- #
# the-preprocessing-contract — the lesson's own ten-row example, drawn
# --------------------------------------------------------------------------- #
def fit_on_train_vs_leak() -> None:
    """X = 0..9, the last two rows held out (train_test_split(test_size=0.2,
    shuffle=False)). Fitting StandardScaler before the split gives mean_ 4.5;
    fitting it on train gives 3.5 — the numbers the lesson's exercise prints."""
    x = np.arange(10, dtype=float)
    n_train = 8
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    cell_w, cell_h = 0.075, 0.105
    x0, y0 = 0.08, 0.20

    def rows(ax):
        for i, v in enumerate(x):
            train = i < n_train
            ax.add_patch(plt.Rectangle(
                (x0 + i * cell_w, y0), cell_w, cell_h,
                facecolor=TRAIN_FILL if train else TEST_FILL,
                edgecolor=ACCENT if train else WARM, linewidth=1.3, zorder=2))
            ax.text(x0 + (i + 0.5) * cell_w, y0 + cell_h / 2, f"{v:.0f}",
                    ha="center", va="center", fontsize=12, zorder=3)
        ax.text(x0 + n_train * cell_w / 2, y0 - 0.06, "train", ha="center",
                va="top", fontsize=11.5, color=ACCENT, fontweight="bold")
        ax.text(x0 + (n_train + 1) * cell_w, y0 - 0.06, "test", ha="center",
                va="top", fontsize=11.5, color=WARM, fontweight="bold")

    def bracket(ax, i0, i1, color):
        top = y0 + cell_h + 0.04
        xa, xb = x0 + i0 * cell_w + 0.006, x0 + i1 * cell_w - 0.006
        ax.plot([xa, xa, xb, xb], [top - 0.02, top, top, top - 0.02],
                color=color, linewidth=1.8)
        return (xa + xb) / 2, top

    def params_box(ax, cx, text, color):
        ax.add_patch(plt.Rectangle((cx - 0.19, 0.66), 0.38, 0.16,
                                   facecolor="white", edgecolor=color,
                                   linewidth=1.8, zorder=2))
        ax.text(cx, 0.74, text, ha="center", va="center", fontsize=12.5,
                zorder=3, family="DejaVu Sans Mono")

    def arrow(ax, x_from, y_from, x_to, y_to, color, label=None, label_side=1):
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                    shrinkA=0, shrinkB=0))
        if label:
            ax.text((x_from + x_to) / 2 + 0.02 * label_side, (y_from + y_to) / 2,
                    label, ha="left" if label_side > 0 else "right",
                    va="center", fontsize=11, style="italic", color=color)

    # Left: the leak
    ax = axes[0]
    ax.set_title("Fit before the split", fontsize=13.5, fontweight="bold",
                 color=WARM, pad=10)
    rows(ax)
    cx, top = bracket(ax, 0, 10, WARM)
    arrow(ax, cx, top, cx, 0.66, WARM, "fit on all 10 rows")
    params_box(ax, cx, f"mean_ = {x.mean():.1f}", WARM)
    ax.text(cx, 0.93, "the test rows helped produce this number",
            ha="center", va="center", fontsize=11, color=WARM)

    # Right: the contract
    ax = axes[1]
    ax.set_title("Fit on train, transform everywhere", fontsize=13.5,
                 fontweight="bold", color=ACCENT, pad=10)
    rows(ax)
    cx, top = bracket(ax, 0, n_train, ACCENT)
    arrow(ax, cx - 0.10, top, cx - 0.10, 0.66, ACCENT, "fit", label_side=-1)
    params_box(ax, cx, f"mean_ = {x[:n_train].mean():.1f}", ACCENT)
    arrow(ax, cx + 0.10, 0.66, cx + 0.10, top, INK, "transform")
    test_cx = x0 + (n_train + 1) * cell_w
    ax.annotate("", xy=(test_cx, y0 + cell_h + 0.01), xytext=(cx + 0.19, 0.70),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6,
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=0, shrinkB=0))
    ax.text(test_cx + 0.035, 0.50, "transform", ha="left", va="center", fontsize=11,
            style="italic", color=INK)
    ax.text(cx, 0.93, "computed from rows the model may see",
            ha="center", va="center", fontsize=11, color=ACCENT)

    for ax in axes:
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.tight_layout(w_pad=3)
    save(fig, "fit-on-train-vs-leak.png")


def main() -> None:
    print("generating Session 2 figures ->", ASSETS)
    fit_on_train_vs_leak()
    print("done.")


if __name__ == "__main__":
    main()
