"""Figures for the `python-medecine` module `pym-s8-projet-puissance-4`,
lesson `projet-puissance-4.md`.

Three figures:

- `directions-victoire.png`: the four ways to line up four discs (horizontal,
  vertical, diagonal ↗, diagonal ↘) on 6 × 7 boards.
- `observation-canaux.png`: one position as the player to move sees it, and
  the two channels of PettingZoo's `(6, 7, 2)` observation (channel 0 = own
  discs, channel 1 = opponent's; row 0 is the top row).
- `arbre-minimax.png`: a depth-2 game tree, max then min, with the values
  backed up from the leaves (the classic minimax diagram).

`connect4.png` (the project logo) is copied from the source course
(github.com/bgregorutti/python-project), not drawn here.

No random element. Re-run with:

    uv run --no-project --with matplotlib \
        python courseware/tools/figures/pym_s8.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "python-medecine"
       / "assets" / "pym-s8-projet-puissance-4")

BOARD = "#2f5d9e"
EMPTY = "#f4f6fa"
ME = "#e8b923"        # own discs (yellow)
OPP = "#c0392b"       # opponent's discs (red)
WIN_EDGE = "#1b1b1b"
TEXT = "#33415c"


def _board(ax, cells, highlight=(), title=None):
    """cells: dict (row, col) -> 'X' (me) or 'O' (opponent); row 0 = top."""
    ax.add_patch(FancyBboxPatch((-0.1, -0.1), 7.2, 6.2,
                                boxstyle="round,pad=0.05", fc=BOARD, ec=BOARD))
    for r in range(6):
        for c in range(7):
            v = cells.get((r, c))
            fill = ME if v == "X" else OPP if v == "O" else EMPTY
            lw = 3 if (r, c) in highlight else 0
            ax.add_patch(Circle((c + 0.5, 5.5 - r), 0.38, fc=fill,
                                ec=WIN_EDGE, lw=lw))
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-0.3, 6.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, color=TEXT)


def directions() -> None:
    examples = [
        ("Horizontale", [(5, 1), (5, 2), (5, 3), (5, 4)],
         {(5, 0): "O", (4, 2): "O", (4, 3): "O"}),
        ("Verticale", [(5, 3), (4, 3), (3, 3), (2, 3)],
         {(5, 2): "O", (5, 4): "O", (4, 4): "O"}),
        ("Diagonale ↗", [(5, 1), (4, 2), (3, 3), (2, 4)],
         {(5, 2): "O", (5, 3): "O", (4, 3): "O", (5, 4): "O", (4, 4): "O",
          (3, 4): "X", (5, 5): "X"}),
        ("Diagonale ↘", [(2, 2), (3, 3), (4, 4), (5, 5)],
         {(5, 2): "O", (4, 2): "X", (3, 2): "O", (5, 3): "O", (4, 3): "O",
          (5, 4): "O"}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (title, line, others) in zip(axes, examples):
        cells = dict(others)
        cells.update({p: "X" for p in line})
        _board(ax, cells, highlight=set(line), title=title)
    fig.suptitle("Quatre pions alignés : quatre directions à vérifier",
                 fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "directions-victoire.png", dpi=130)
    plt.close(fig)


def observation() -> None:
    cells = {(5, 3): "X", (5, 2): "O", (4, 3): "O", (5, 4): "X", (3, 3): "X",
             (5, 1): "O"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    _board(axes[0], cells, title="Le plateau (jaune = moi)")
    for k, (ax, who, color) in enumerate(
            [(axes[1], "X", ME), (axes[2], "O", OPP)]):
        for r in range(6):
            for c in range(7):
                on = cells.get((r, c)) == who
                ax.add_patch(Rectangle((c, 5 - r), 1, 1,
                                       fc=color if on else "white",
                                       ec="#9aa5b8", lw=1))
                ax.text(c + 0.5, 5.5 - r, "1" if on else "0", ha="center",
                        va="center", fontsize=12,
                        color="black" if on else "#9aa5b8", family="monospace")
        for r in range(6):
            ax.text(-0.35, 5.5 - r, str(r), ha="center", va="center",
                    fontsize=10, color=TEXT)
        for c in range(7):
            ax.text(c + 0.5, 6.3, str(c), ha="center", va="center",
                    fontsize=10, color=TEXT)
        ax.set_xlim(-0.7, 7.1)
        ax.set_ylim(-0.2, 6.7)
        ax.set_aspect("equal")
        ax.axis("off")
        label = "mes pions" if k == 0 else "pions adverses"
        ax.set_title(f"observation[:, :, {k}] — {label}", fontsize=13,
                     color=TEXT, family="monospace")
    fig.tight_layout()
    fig.savefig(OUT / "observation-canaux.png", dpi=130)
    plt.close(fig)


def minimax_tree() -> None:
    # depth 2: root (max) -> 3 moves -> 2 replies each (min); leaf values.
    leaves = [[3, 5], [2, 9], [0, 1]]
    mins = [min(v) for v in leaves]
    root = max(mins)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def node(x, y, value, shape, best=False):
        fc = "#fdf1dc" if best else "#e8eef7"
        if shape == "max":
            ax.add_patch(plt.Polygon([(x - 0.42, y - 0.3), (x + 0.42, y - 0.3),
                                      (x, y + 0.38)], fc=fc, ec=TEXT, lw=1.6, zorder=3))
        elif shape == "min":
            ax.add_patch(plt.Polygon([(x - 0.42, y + 0.3), (x + 0.42, y + 0.3),
                                      (x, y - 0.38)], fc=fc, ec=TEXT, lw=1.6,
                                      zorder=3))
        else:
            ax.add_patch(Rectangle((x - 0.3, y - 0.25), 0.6, 0.5, fc="white",
                                   ec=TEXT, lw=1.2, zorder=3))
        ax.text(x, y - (0.02 if shape != "min" else -0.04), str(value),
                ha="center", va="center", fontsize=13, color=TEXT, zorder=4,
                weight="bold" if best else None)

    xs_mid = [3.0, 5.6, 8.2]
    best_i = mins.index(root)
    for i, (xm, pair) in enumerate(zip(xs_mid, leaves)):
        ax.plot([5.6, xm], [3.65, 2.8], color=TEXT, lw=2.4 if i == best_i else 1)
        ax.text((5.6 + xm) / 2 + (0.15 if xm >= 5.6 else -0.15), 3.3,
                f"col {i}", fontsize=10, color="#666",
                ha="right" if xm < 5.6 else "left")
        for j, v in enumerate(pair):
            xl = xm - 0.55 + 1.1 * j
            ax.plot([xm, xl], [2.12, 1.05], color=TEXT, lw=1)
            node(xl, 0.8, v, "leaf")
        node(xm, 2.5, mins[i], "min", best=(i == best_i))
    node(5.6, 3.95, root, "max", best=True)
    ax.text(0.1, 3.95, "MAX (moi)", fontsize=12, color=TEXT, va="center")
    ax.text(0.1, 2.5, "MIN (adversaire)", fontsize=12, color=TEXT, va="center")
    ax.text(0.1, 0.8, "évaluation", fontsize=12, color=TEXT, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "arbre-minimax.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    directions()
    observation()
    minimax_tree()
    print(f"wrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
