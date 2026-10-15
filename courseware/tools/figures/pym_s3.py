"""Figures for the `python-medecine` module `pym-s3-structures-de-donnees`,
lesson `mutabilite.md`.

Two figures, names on the left, objects on the right, arrows for references
(the classic "names point to objects" diagram, as in the source course's
`part_2/img/mutabliite_int.png`, redrawn with French labels):

- `entier-immuable.png`: `a = 120; b = a; a = a + 10` — `a` is rebound to a
  new int, `b` still points to 120.
- `liste-alias.png`: `a = [120, 135]; b = a; b.append(128)` — both names
  point to the same list, which now holds three values.

No random element. Re-run with:

    uv run --no-project --with matplotlib \
        python courseware/tools/figures/pym_s3.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "python-medecine"
       / "assets" / "pym-s3-structures-de-donnees")

NAME_FILL = "#e8eef7"
OBJ_FILL = "#fdf1dc"
EDGE = "#33415c"
ARROW = "#33415c"
NEW = "#c0392b"


def _box(ax, x, y, text, fill, w=1.0, h=0.55, color=EDGE, mono=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04", fc=fill, ec=color,
                                lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=13,
            family="monospace" if mono else None, color=color)


def _arrow(ax, x0, y0, x1, y1, color=ARROW, style="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color=color,
                                 linestyle=style))


def _panel(ax, title):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title(title, fontsize=13, family="monospace", loc="left")
    ax.text(0.8, 3.0, "noms", ha="center", fontsize=11, color="#666")
    ax.text(3.6, 3.0, "objets en mémoire", ha="center", fontsize=11,
            color="#666")


def entier() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    _panel(ax, "a = 120\nb = a")
    _box(ax, 0.8, 2.2, "a", NAME_FILL, w=0.6)
    _box(ax, 0.8, 1.0, "b", NAME_FILL, w=0.6)
    _box(ax, 3.6, 1.6, "120", OBJ_FILL)
    _arrow(ax, 1.15, 2.15, 3.05, 1.7)
    _arrow(ax, 1.15, 1.05, 3.05, 1.5)
    ax = axes[1]
    _panel(ax, "a = a + 10")
    _box(ax, 0.8, 2.2, "a", NAME_FILL, w=0.6)
    _box(ax, 0.8, 1.0, "b", NAME_FILL, w=0.6)
    _box(ax, 3.6, 2.2, "130", OBJ_FILL, color=NEW)
    _box(ax, 3.6, 1.0, "120", OBJ_FILL)
    _arrow(ax, 1.15, 2.2, 3.05, 2.2, color=NEW)
    _arrow(ax, 1.15, 1.0, 3.05, 1.0)
    ax.text(3.6, 0.3, "un nouvel objet : b n'a pas changé", ha="center",
            fontsize=11, color=NEW)
    fig.tight_layout()
    fig.savefig(OUT / "entier-immuable.png", dpi=150)
    plt.close(fig)


def liste() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    _panel(ax, "a = [120, 135]\nb = a")
    _box(ax, 0.8, 2.2, "a", NAME_FILL, w=0.6)
    _box(ax, 0.8, 1.0, "b", NAME_FILL, w=0.6)
    _box(ax, 3.6, 1.6, "[120, 135]", OBJ_FILL, w=1.9)
    _arrow(ax, 1.15, 2.15, 2.6, 1.7)
    _arrow(ax, 1.15, 1.05, 2.6, 1.5)
    ax = axes[1]
    _panel(ax, "b.append(128)")
    _box(ax, 0.8, 2.2, "a", NAME_FILL, w=0.6)
    _box(ax, 0.8, 1.0, "b", NAME_FILL, w=0.6)
    _box(ax, 3.6, 1.6, "[120, 135, 128]", OBJ_FILL, w=2.5, color=NEW)
    _arrow(ax, 1.15, 2.15, 2.3, 1.7)
    _arrow(ax, 1.15, 1.05, 2.3, 1.5)
    ax.text(3.6, 0.5, "le même objet, modifié sur place :\na voit aussi 128",
            ha="center", fontsize=11, color=NEW)
    fig.tight_layout()
    fig.savefig(OUT / "liste-alias.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entier()
    liste()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
