"""Figures for the `python-medecine` module `pym-s5-numpy-pandas`.

Five diagrams drawn as small tables, French labels, medical values:

- `broadcasting.png` (lesson `numpy.md`): a patients x variables matrix minus
  the row of column means — the row is "stretched" over every patient.
- `anatomie-dataframe.png` (lesson `pandas-lire-et-explorer.md`): index,
  columns, one column = one Series, one dtype per column.
- `split-apply-combine.png` (lesson `agreger-et-joindre.md`): `groupby` on
  the service, mean stay length per group.
- `long-large.png` (lesson `agreger-et-joindre.md`): lab results in long
  format and in wide format, `pivot` one way, `melt` the other.
- `merge-how.png` (lesson `agreger-et-joindre.md`): `inner`, `left` and
  `outer` joins of two small tables on `id_patient`.

Two stills of the source course (`part_4/img/pandas_1.png`, `pandas_2.png`)
are copied as is into the same folder.

No random element. Re-run with:

    uv run --no-project --with matplotlib \
        python courseware/tools/figures/pym_s5.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "python-medecine"
       / "assets" / "pym-s5-numpy-pandas")

EDGE = "#33415c"
HEAD = "#dfe6f1"
INDEX = "#eeeeee"
CELL = "#ffffff"
HIGHLIGHT = "#fdf1dc"
ACCENT = "#c0392b"
GROUPS = ["#e3f0e3", "#fdf1dc", "#e8eef7"]
NAN = "#999999"


def _table(ax, x, y, rows, widths, h=0.42, fills=None, header=True,
           index=False, fontsize=11):
    """Draw `rows` (list of lists of str) with top-left corner at (x, y).

    `fills[i][j]` overrides the fill of a cell. Returns the bottom y."""
    for i, row in enumerate(rows):
        cx = x
        for j, txt in enumerate(row):
            w = widths[j]
            if fills is not None and fills[i][j] is not None:
                fc = fills[i][j]
            elif header and i == 0:
                fc = HEAD
            elif index and j == 0:
                fc = INDEX
            else:
                fc = CELL
            ax.add_patch(Rectangle((cx, y - (i + 1) * h), w, h, fc=fc,
                                   ec=EDGE, lw=1.0))
            bold = (header and i == 0) or (index and j == 0)
            color = NAN if txt == "NaN" else EDGE
            ax.text(cx + w / 2, y - (i + 0.5) * h, txt, ha="center",
                    va="center", fontsize=fontsize, color=color,
                    family="monospace",
                    fontweight="bold" if bold else "normal")
            cx += w
    return y - len(rows) * h


def _arrow(ax, x0, y0, x1, y1, color=EDGE, text=None, dy=0.18):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color=color))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center",
                va="bottom", fontsize=12, family="monospace", color=color)


def _canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def broadcasting() -> None:
    fig, ax = _canvas(14, 4.2, (12, 3.6))
    head = ["PAS", "FC", "Gly"]
    x = [["128", "72", "0.95"], ["151", "88", "1.30"], ["117", "64", "0.88"],
         ["140", "80", "1.07"]]
    mu = ["134", "76", "1.05"]
    res = [["-6", "-4", "-0.10"], ["17", "12", "0.25"], ["-17", "-12", "-0.17"],
           ["6", "4", "0.02"]]
    w = [0.9, 0.9, 0.9]
    top = 3.5
    ax.text(1.35, top + 0.25, "X  (4 patients × 3)", ha="center", fontsize=11)
    _table(ax, 0, top, [head] + x, w)
    ax.text(3.4, top - 1.2, "−", ha="center", va="center", fontsize=26)
    ax.text(5.55, top + 0.25, "X.mean(axis=0)  (3,)", ha="center", fontsize=11)
    fills = [[None] * 3] + [[HIGHLIGHT] * 3] + [["#fbf8f2"] * 3] * 3
    _table(ax, 4.2, top, [head] + [mu] + [mu] * 3, w, fills=fills)
    ax.text(5.55, top - 2.45, "la ligne est répétée\npour chaque patient",
            ha="center", va="top", fontsize=10, color=ACCENT)
    ax.text(7.6, top - 1.2, "=", ha="center", va="center", fontsize=26)
    ax.text(9.75, top + 0.25, "X - X.mean(axis=0)", ha="center", fontsize=11)
    _table(ax, 8.4, top, [head] + res, w)
    ax.text(12.0, top - 1.2, "(4, 3) − (3,)\n→ (4, 3)", ha="left",
            va="center", fontsize=12, family="monospace", color=EDGE)
    _save(fig, "broadcasting.png")


def anatomie() -> None:
    fig, ax = _canvas(11, 4.8, (10, 4.4))
    ax.set_xlim(-1, 11)
    rows = [["", "id_patient", "age", "sexe", "deces"],
            ["0", "IC001", "75", "M", "1"],
            ["1", "IC002", "55", "M", "1"],
            ["2", "IC003", "65", "M", "1"],
            ["3", "IC004", "50", "M", "1"]]
    fills = [[None, None, None, None, None]] + [
        [None, None, HIGHLIGHT, None, None] for _ in range(4)]
    w = [0.7, 1.5, 0.9, 0.9, 0.9]
    x0, top = 2.0, 3.9
    bottom = _table(ax, x0, top, rows, w, fills=fills, index=True)
    dtypes = ["", "str", "int64", "str", "int64"]
    cx = x0
    for d, wi in zip(dtypes, w):
        ax.text(cx + wi / 2, bottom - 0.25, d, ha="center", va="top",
                fontsize=10, color=ACCENT, family="monospace")
        cx += wi
    ax.text(x0 - 0.15, bottom - 0.25, "dtypes", ha="right", va="top",
            fontsize=10, color=ACCENT)
    _arrow(ax, 1.2, 2.6, x0 - 0.05, 2.6)
    ax.text(1.1, 2.6, "index\n(étiquettes\ndes lignes)", ha="right",
            va="center", fontsize=11)
    _arrow(ax, x0 + 2.0, top + 0.6, x0 + 2.0, top + 0.05)
    ax.text(x0 + 2.0, top + 0.65, "columns (noms des colonnes)", ha="center",
            va="bottom", fontsize=11)
    xs = x0 + 0.7 + 1.5 + 0.9 + 0.9 + 0.9
    ax.text(xs + 0.4, 2.6, "df[\"age\"]\n= une Series :\nun index\n+ des valeurs\nd'un seul type",
            ha="left", va="center", fontsize=11, color=EDGE)
    _arrow(ax, xs + 0.35, 2.6, x0 + 0.7 + 1.5 + 0.9, 2.6, color=ACCENT)
    _save(fig, "anatomie-dataframe.png")


def split_apply_combine() -> None:
    fig, ax = _canvas(15, 5.4, (12, 4.4))
    data = [("Cardiologie", "4"), ("Gériatrie", "12"), ("Cardiologie", "6"),
            ("Neurologie", "5"), ("Gériatrie", "8"), ("Neurologie", "3")]
    color = {"Cardiologie": GROUPS[0], "Gériatrie": GROUPS[1],
             "Neurologie": GROUPS[2]}
    w = [2.0, 0.9]
    top = 4.6
    rows = [["service", "duree"]] + [list(r) for r in data]
    fills = [[None, None]] + [[color[s], color[s]] for s, _ in data]
    _table(ax, 0, top, rows, w, fills=fills)
    ax.text(1.45, top + 0.2, "df", ha="center", fontsize=12, family="monospace")
    # split
    gx = 4.6
    y = top + 0.1
    means = {}
    centers = []
    for s in ["Cardiologie", "Gériatrie", "Neurologie"]:
        vals = [d for n, d in data if n == s]
        g = [[s, v] for v in vals]
        bottom = _table(ax, gx, y, g, w, header=False,
                        fills=[[color[s]] * 2 for _ in g])
        centers.append((y + bottom) / 2)
        means[s] = sum(int(v) for v in vals) / len(vals)
        y = bottom - 0.35
    for c in centers:
        _arrow(ax, 3.05, 3.1, gx - 0.1, c)
    ax.text(3.75, 4.95, "split", ha="center", fontsize=13, color=ACCENT)
    # apply
    ax.text(8.7, 4.95, "apply : mean()", ha="center", fontsize=13,
            color=ACCENT)
    ox = 9.6
    for s, c in zip(["Cardiologie", "Gériatrie", "Neurologie"], centers):
        _arrow(ax, gx + 2.95, c, ox - 0.1, c)
        _table(ax, ox, c + 0.21, [[f"{means[s]:g}"]], [0.9], header=False,
               fills=[[color[s]]])
    # combine
    ax.text(13.2, 4.95, "combine", ha="center", fontsize=13, color=ACCENT)
    rows = [["service", "duree"]] + [[s, f"{means[s]:g}"] for s in means]
    fills = [[None, None]] + [[color[s]] * 2 for s in means]
    _table(ax, 11.8, 3.9, rows, [2.0, 0.9], fills=fills)
    _arrow(ax, 10.6, 3.1, 11.7, 3.1)
    ax.text(7.5, 0.05, 'df.groupby("service")["duree"].mean()', ha="center",
            fontsize=12, family="monospace", color=EDGE)
    _save(fig, "split-apply-combine.png")


def long_large() -> None:
    fig, ax = _canvas(16, 4.6, (13, 3.9))
    longr = [["id_patient", "examen", "valeur"],
             ["P01", "crp", "5.6"], ["P01", "potassium", "4.1"],
             ["P02", "crp", "21.0"], ["P02", "potassium", "3.8"],
             ["P03", "crp", "8.3"]]
    ax.text(2.15, 4.25, "format long : une ligne par mesure", ha="center",
            fontsize=11)
    _table(ax, 0, 4.0, longr, [1.7, 1.7, 1.1])
    wide = [["id_patient", "crp", "potassium"],
            ["P01", "5.6", "4.1"], ["P02", "21.0", "3.8"], ["P03", "8.3", "NaN"]]
    ax.text(13.6, 4.25, "format large : une ligne par patient", ha="center",
            fontsize=11)
    _table(ax, 11.4, 4.0, wide, [1.7, 0.9, 1.6], index=True)
    _arrow(ax, 5.0, 2.7, 11.3, 2.7, color=EDGE,
           text='pivot(index="id_patient",\n      columns="examen",\n      values="valeur")')
    _arrow(ax, 11.3, 1.2, 5.0, 1.2, color=ACCENT,
           text='melt(id_vars="id_patient")')
    _save(fig, "long-large.png")


def merge_how() -> None:
    fig, ax = _canvas(17, 5.2, (13.5, 4.2))
    left = [["id_patient", "service"], ["P1", "Cardio"], ["P2", "Neuro"],
            ["P3", "Pneumo"]]
    right = [["id_patient", "crp"], ["P2", "21.0"], ["P3", "8.3"],
             ["P4", "4.5"]]
    ax.text(1.4, 4.85, "sejours (gauche)", ha="center", fontsize=11)
    _table(ax, 0, 4.6, left, [1.7, 1.3])
    ax.text(1.2, 2.55, "examens (droite)", ha="center", fontsize=11)
    _table(ax, 0, 2.3, right, [1.7, 0.9])
    w = [1.7, 1.3, 0.9]
    results = {
        'how="inner"': [["P2", "Neuro", "21.0"], ["P3", "Pneumo", "8.3"]],
        'how="left"': [["P1", "Cardio", "NaN"], ["P2", "Neuro", "21.0"],
                       ["P3", "Pneumo", "8.3"]],
        'how="outer"': [["P1", "Cardio", "NaN"], ["P2", "Neuro", "21.0"],
                        ["P3", "Pneumo", "8.3"], ["P4", "NaN", "4.5"]],
    }
    x = 4.0
    for title, rows in results.items():
        ax.text(x + 1.95, 4.85, title, ha="center", fontsize=12,
                family="monospace", color=ACCENT)
        _table(ax, x, 4.6, [["id_patient", "service", "crp"]] + rows, w,
               fontsize=10)
        x += 4.3
    ax.text(10.5, 0.6, 'sejours.merge(examens, on="id_patient", how=...)',
            ha="center", fontsize=12, family="monospace", color=EDGE)
    _save(fig, "merge-how.png")


def main() -> None:
    broadcasting()
    anatomie()
    split_apply_combine()
    long_large()
    merge_how()


if __name__ == "__main__":
    main()
