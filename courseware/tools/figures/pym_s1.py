"""Figures for the `python-medecine` module `pym-s1-premiers-pas`.

Two drawings, written into `content/python-medecine/assets/pym-s1-premiers-pas/`:

* `trois-facons-executer.png` — lesson `colab-jupyter-scripts.md`: the same two
  lines of code run in an interactive shell, as a script from the terminal, and
  as notebook cells (Colab / Jupyter), with what each one shows.
* `indexation-chaine.png` — lesson `chaines-de-caracteres.md`: the string
  "TA 142/91" with its positive and negative indices, and the slice [3:6].

The other images of the folder are copied from the source course
(Python-Master-Actuariat: `part_1/img/abstraction.png`,
`part_1/img/python-versions.png`, `pages/img/type_hierarchy.png`).

    uv run --no-project --with matplotlib python courseware/tools/figures/pym_s1.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "python-medecine"
       / "assets" / "pym-s1-premiers-pas")

INK = "#1f2933"
BLUE = "#2f6db5"
GREEN = "#2e8b57"
GREY = "#eef1f5"
MONO = "DejaVu Sans Mono"


def _panel(ax, x, title, subtitle, lines, out_lines):
    ax.add_patch(FancyBboxPatch((x, 0.05), 0.3, 0.8, boxstyle="round,pad=0.01",
                                fc="white", ec=INK, lw=1.2))
    ax.text(x + 0.15, 0.80, title, ha="center", va="center", fontsize=13,
            weight="bold", color=INK)
    ax.text(x + 0.15, 0.74, subtitle, ha="center", va="center", fontsize=9.5,
            color="#52606d")
    y = 0.64
    for kind, text in lines:
        if kind == "cell":
            ax.add_patch(Rectangle((x + 0.02, y - 0.055), 0.26, 0.085, fc=GREY,
                                   ec=BLUE, lw=1))
            ax.text(x + 0.03, y - 0.012, text, fontsize=9, family=MONO,
                    va="center", color=INK)
            y -= 0.1
        elif kind == "out":
            ax.text(x + 0.03, y, text, fontsize=9, family=MONO, va="center",
                    color=GREEN)
            y -= 0.07
        else:
            ax.text(x + 0.03, y, text, fontsize=9, family=MONO, va="center",
                    color=INK)
            y -= 0.07
    ax.text(x + 0.15, 0.10, out_lines, ha="center", va="center", fontsize=9.5,
            color=INK, style="italic")


def trois_facons() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.axis("off")
    _panel(ax, 0.02, "Shell interactif", "$ python   (ou ipython)", [
        ("txt", ">>> poids = 70"),
        ("txt", ">>> taille = 1.75"),
        ("txt", ">>> poids / taille ** 2"),
        ("out", "22.857142857142858"),
    ], "une ligne à la fois,\nrien n'est enregistré")
    _panel(ax, 0.35, "Script", "fichier imc.py, puis $ python imc.py", [
        ("txt", "poids = 70"),
        ("txt", "taille = 1.75"),
        ("txt", "print(poids / taille ** 2)"),
        ("out", "22.857142857142858"),
    ], "tout le fichier, de haut en bas,\nà chaque exécution")
    _panel(ax, 0.68, "Notebook", "Google Colab, Jupyter", [
        ("cell", "poids = 70"),
        ("cell", "taille = 1.75"),
        ("cell", "poids / taille ** 2"),
        ("out", "22.857142857142858"),
    ], "cellule par cellule ; les variables\nrestent en mémoire entre cellules")
    fig.savefig(OUT / "trois-facons-executer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def indexation() -> None:
    s = "TA 142/91"
    n = len(s)
    w = 1.0
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(-1.8, n + 0.5)
    ax.set_ylim(-2.3, 2.2)
    ax.axis("off")
    for i, ch in enumerate(s):
        sel = 3 <= i < 6
        ax.add_patch(Rectangle((i * w, 0), w, 1, fc="#dbe8f7" if sel else "white",
                               ec=INK, lw=1.3))
        ax.text(i * w + 0.5, 0.5, ch if ch != " " else "␣", ha="center",
                va="center", fontsize=20, family=MONO, color=INK)
        ax.text(i * w + 0.5, 1.35, str(i), ha="center", va="center",
                fontsize=12, color=BLUE)
        ax.text(i * w + 0.5, -0.35, str(i - n), ha="center", va="center",
                fontsize=12, color=GREEN)
    ax.text(-0.3, 1.35, "indice positif", ha="right", va="center", fontsize=11,
            color=BLUE)
    ax.text(-0.3, -0.35, "indice négatif", ha="right", va="center", fontsize=11,
            color=GREEN)
    ax.annotate("", xy=(3, -1.0), xytext=(6, -1.0),
                arrowprops=dict(arrowstyle="|-|", color=INK, lw=1.3))
    ax.text(4.5, -1.45, 's[3:6]  →  "142"   (l\'indice 6 est exclu)',
            ha="center", va="center", fontsize=12, family=MONO, color=INK)
    ax.text(n / 2, 1.95, 's = "TA 142/91"      len(s) = 9', ha="center",
            va="center", fontsize=13, family=MONO, color=INK)
    fig.savefig(OUT / "indexation-chaine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    trois_facons()
    indexation()
    print("written to", OUT)
