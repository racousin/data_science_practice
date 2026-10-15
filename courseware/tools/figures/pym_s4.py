"""Figures of *Python pour la médecine*, Séance 4 — Objets et modules.

Three class diagrams, drawn in the UML style of the source course's
`part_3/img/heritage.png` but with French, medical content:

- `classe_et_instances.png`: one class (`Patient`), three objects built from it.
- `heritage_examen.png`: `Examen` and its two subclasses.
- `heritage_multiple.png`: a hospital-university doctor (PU-PH) who is both a
  `Soignant` and an `Enseignant`.

    uv run --with matplotlib python courseware/tools/figures/pym_s4.py

Writes into `content/python-medecine/assets/pym-s4-objets-et-modules/`.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ASSETS = (pathlib.Path(__file__).resolve().parents[2] / "content"
          / "python-medecine" / "assets" / "pym-s4-objets-et-modules")

INK = "#1f2933"
MUTED = "#6b7684"
BLUE, BLUE_BG = "#2f6f9f", "#e3eef7"
GOLD, GOLD_BG = "#b5892a", "#faf0d9"
GREEN, GREEN_BG = "#3f8f6f", "#e2f0ea"
PURPLE, PURPLE_BG = "#6b4f9f", "#ece6f5"
GREY_BG = "#f4f5f7"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans", "text.color": INK})

LINE = 0.42  # height of one text line, in data units


def class_box(ax, x, y, w, title, attrs, methods, color, bg, italic=False):
    """UML class box whose top-left corner is (x, y). Returns (top, bottom)
    attachment points (centre of the top edge, centre of the bottom edge)."""
    head = 0.6
    h_attr = LINE * len(attrs) + 0.2
    h_meth = LINE * len(methods) + 0.2 if methods else 0
    total = head + h_attr + h_meth
    ax.add_patch(Rectangle((x, y - head), w, head, fc=bg, ec=color, lw=1.6))
    ax.text(x + w / 2, y - head / 2, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=INK,
            style="italic" if italic else "normal")
    ax.add_patch(Rectangle((x, y - head - h_attr), w, h_attr, fc=GREY_BG,
                           ec=color, lw=1.6))
    for i, a in enumerate(attrs):
        ax.text(x + 0.15, y - head - 0.1 - LINE * (i + 0.5), a, va="center",
                fontsize=9.5, family="DejaVu Sans Mono")
    if methods:
        ax.add_patch(Rectangle((x, y - total), w, h_meth, fc="white",
                               ec=color, lw=1.6))
        for i, m in enumerate(methods):
            ax.text(x + 0.15, y - head - h_attr - 0.1 - LINE * (i + 0.5), m,
                    va="center", fontsize=9.5, family="DejaVu Sans Mono")
    return (x + w / 2, y), (x + w / 2, y - total)


def inherit(ax, child_top, parent_bottom):
    """Hollow-triangle arrow from the child class to its parent (UML)."""
    ax.add_patch(FancyArrowPatch(child_top, parent_bottom,
                                 arrowstyle="-|>,head_length=0.5,head_width=0.3",
                                 mutation_scale=18, fc="white", ec=INK, lw=1.3))


def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 1.6)
    ax.set_ylim(0, h * 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def classe_et_instances():
    fig, ax = setup(9, 4.4)
    _, bottom = class_box(ax, 4.6, 6.8, 5.2, "Patient  (la classe)",
                          ["nom", "date_naissance", "poids_kg", "taille_m"],
                          ["imc()"], BLUE, BLUE_BG)
    objets = [
        ("p1 : Patient", ["nom = 'Durand'", "poids_kg = 72", "taille_m = 1.80"]),
        ("p2 : Patient", ["nom = 'Martin'", "poids_kg = 58", "taille_m = 1.64"]),
        ("p3 : Patient", ["nom = 'Nguyen'", "poids_kg = 90", "taille_m = 1.75"]),
    ]
    for i, (titre, attrs) in enumerate(objets):
        x = 0.3 + i * 4.8
        top, _ = class_box(ax, x, 2.6, 4.0, titre, attrs, [], GOLD, GOLD_BG)
        ax.add_patch(FancyArrowPatch(bottom, top, arrowstyle="-|>",
                                     mutation_scale=14, color=MUTED, lw=1.2,
                                     linestyle="--"))
    ax.text(14.6, 3.5, "instanciation\nPatient(...)", ha="right", va="center",
            fontsize=10, color=MUTED, style="italic")
    ax.text(7.2, 0.0, "trois objets (instances) : même plan, valeurs différentes",
            ha="center", fontsize=10, color=MUTED)
    fig.savefig(ASSETS / "classe_et_instances.png")
    plt.close(fig)


def heritage_examen():
    fig, ax = setup(9, 4.6)
    _, parent_bottom = class_box(ax, 4.6, 7.2, 5.2, "Examen",
                                 ["id_patient", "date"],
                                 ["resume()"], BLUE, BLUE_BG)
    top1, _ = class_box(ax, 0.4, 3.4, 6.0, "ExamenBiologique",
                        ["valeur", "unite", "borne_basse, borne_haute"],
                        ["est_normal()", "resume()   (redéfinie)"],
                        GREEN, GREEN_BG)
    top2, _ = class_box(ax, 8.0, 3.4, 6.0, "ExamenImagerie",
                        ["modalite", "conclusion"],
                        ["resume()   (redéfinie)"], PURPLE, PURPLE_BG)
    inherit(ax, top1, (parent_bottom[0] - 0.8, parent_bottom[1]))
    inherit(ax, top2, (parent_bottom[0] + 0.8, parent_bottom[1]))
    ax.text(14.2, 6.0, "« est un »", ha="right", fontsize=10, color=MUTED,
            style="italic")
    fig.savefig(ASSETS / "heritage_examen.png")
    plt.close(fig)


def heritage_multiple():
    fig, ax = setup(9, 5.4)
    _, b_root = class_box(ax, 4.9, 8.5, 4.6, "Personne", ["nom", "prenom"],
                          ["__str__()"], BLUE, BLUE_BG)
    t_soig, b_soig = class_box(ax, 0.6, 5.0, 5.4, "Soignant",
                               ["service"], ["soigner()"], GREEN, GREEN_BG)
    t_ens, b_ens = class_box(ax, 8.4, 5.0, 5.4, "Enseignant",
                             ["universite"], ["enseigner()"], GOLD, GOLD_BG)
    t_pu, _ = class_box(ax, 4.6, 2.2, 5.2, "PUPH", ["discipline"], [],
                        PURPLE, PURPLE_BG)
    inherit(ax, t_soig, (b_root[0] - 0.8, b_root[1]))
    inherit(ax, t_ens, (b_root[0] + 0.8, b_root[1]))
    inherit(ax, (t_pu[0] - 1.2, t_pu[1]), b_soig)
    inherit(ax, (t_pu[0] + 1.2, t_pu[1]), b_ens)
    ax.text(7.2, 0.15, "class PUPH(Soignant, Enseignant) : deux classes mères",
            ha="center", fontsize=10, color=MUTED, family="DejaVu Sans Mono")
    fig.savefig(ASSETS / "heritage_multiple.png")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    classe_et_instances()
    heritage_examen()
    heritage_multiple()
    for p in sorted(ASSETS.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
