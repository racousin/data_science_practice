"""Figures for the `python-medecine` module `pym-s2-controle-et-fonctions`.

Two figures, written to `content/python-medecine/assets/pym-s2-controle-et-fonctions/`:

- `organigramme-imc.png` (lesson `conditions.md`): the flowchart of an
  `if / elif / elif / else` chain that classifies a BMI with the WHO cut-offs
  (18.5, 25, 30 kg/m²). Only the first true condition runs.
- `demi-vie.png` (lesson `boucles.md`): the plasma concentration halving at each
  half-life until it falls under a threshold, i.e. the trace of the `while`
  loop of the lesson (C0 = 80 mg/L, threshold 5 mg/L -> 5 iterations).

No random element. Re-run with:

    uv run --no-project --with matplotlib \
        python courseware/tools/figures/pym_s2.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "python-medecine"
       / "assets" / "pym-s2-controle-et-fonctions")

BLUE = "#2b6cb0"
BLUE_BG = "#e6eef8"
GREEN = "#2f855a"
GREEN_BG = "#e6f4ea"
GREY = "#4a5568"


def _arrow(ax, x0, y0, x1, y1, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=GREY, lw=1.4))
    if label:
        lx, ly = label_xy
        ax.text(lx, ly, label, fontsize=11, color=GREY, ha="center", va="center",
                fontweight="bold")


def _diamond(ax, cx, cy, text, w=3.0, h=1.0):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=BLUE_BG, ec=BLUE, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11,
            family="monospace", color="#1a202c")


def _box(ax, cx, cy, text, w=2.4, h=0.7):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=GREEN_BG, ec=GREEN, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11, color="#1a202c")


def flowchart() -> None:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    conds = ["imc < 18.5", "imc < 25", "imc < 30"]
    labels = ["maigreur", "normal", "surpoids"]
    keywords = ["if", "elif", "elif"]
    ys = [5.4, 3.9, 2.4]
    cx, bx = 2.6, 6.9

    ax.text(cx, 6.35, "imc = poids / taille²", ha="center", va="center",
            fontsize=11, family="monospace", color=GREY)
    _arrow(ax, cx, 6.15, cx, ys[0] + 0.5)
    for kw, cond, lab, y in zip(keywords, conds, labels, ys):
        _diamond(ax, cx, y, cond)
        ax.text(cx - 1.75, y, kw, ha="right", va="center", fontsize=12,
                family="monospace", color=BLUE, fontweight="bold")
        _box(ax, bx, y, f"classe = « {lab} »")
        _arrow(ax, cx + 1.5, y, bx - 1.2, y, "Vrai", ((cx + 1.5 + bx - 1.2) / 2, y + 0.22))
        _arrow(ax, cx, y - 0.5, cx, y - 1.0, "Faux", (cx + 0.38, y - 0.75))
    # else branch
    ax.text(cx - 1.75, 0.9, "else", ha="right", va="center", fontsize=12,
            family="monospace", color=BLUE, fontweight="bold")
    _box(ax, cx, 0.9, "classe = « obésité »")
    # join
    ax.text(bx, 0.9, "une seule branche\ns'exécute", ha="center", va="center",
            fontsize=11, color=GREY, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "organigramme-imc.png", dpi=150)
    plt.close(fig)


def half_life() -> None:
    c0, t_half, threshold = 80.0, 6.0, 5.0
    times, concs = [0.0], [c0]
    c, t = c0, 0.0
    while c >= threshold:
        c = c / 2
        t = t + t_half
        times.append(t)
        concs.append(c)
    assert len(times) - 1 == 5 and concs[-1] == 2.5, (times, concs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(times, concs, where="post", color=BLUE, lw=1.2, alpha=0.35)
    ax.plot(times, concs, "o", color=BLUE, ms=8, zorder=3)
    for i, (tt, cc) in enumerate(zip(times, concs)):
        ax.annotate(f"{cc:g}", (tt, cc), textcoords="offset points", xytext=(8, 6),
                    fontsize=10, color=BLUE)
        if i > 0:
            ax.text(tt - t_half / 2, -6.5, f"tour {i}", ha="center", fontsize=9,
                    color=GREY)
    ax.axhline(threshold, color="#c53030", ls="--", lw=1.4)
    ax.text(28.5, threshold + 6, "seuil = 5 mg/L", color="#c53030",
            fontsize=10, ha="center")
    ax.set_xlabel("temps (h)  —  demi-vie de 6 h")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("while concentration >= seuil :  concentration = concentration / 2",
                 fontsize=11, family="monospace")
    ax.set_xticks(times)
    ax.set_ylim(-9, 90)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "demi-vie.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flowchart()
    half_life()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
