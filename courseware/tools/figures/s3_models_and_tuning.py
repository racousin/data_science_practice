"""Figures generated for `python-ai-engineering` Session 3.

Every image this script writes is committed under
`content/python-ai-engineering/assets/s3-models-and-tuning/<lesson-slug>/`, so a
figure can always be traced back to the code that drew it. Re-run with:

    uv run --with matplotlib \
        python courseware/tools/figures/s3_models_and_tuning.py

Deck-extracted images (the ones lifted from the taught PPTX) are *not* produced
here — this script owns only the figures authored for the 2026 rewrite.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "python-ai-engineering"
    / "assets"
    / "s3-models-and-tuning"
)

INK = "#1f2933"
EDGE = "#3b3b3b"
# The three candidates keep one colour across all three columns, so a row can be
# followed left to right: a model, the model trained, the score it got.
CANDIDATES = ["#f79e9e", "#93c6f5", "#93e49b"]
DATA = "#ffd21e"      # the two data boxes — the only thing that is not a model
WINNER = "#22b24c"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "text.color": INK,
    }
)


def out(lesson: str, name: str) -> pathlib.Path:
    d = ASSETS / lesson
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def save(fig, lesson: str, name: str) -> None:
    p = out(lesson, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ASSETS.parents[2])}")


# --------------------------------------------------------------------------- #
# validation-and-overfitting — the selection loop the three-way split is for
# --------------------------------------------------------------------------- #
def model_selection_flow() -> None:
    """Candidates -> fit on train -> score on **eval** -> keep the best.

    Redrawn from the taught deck's version, which scored the candidates on the
    test set. That is the lesson's own Leak 4, so the figure that opens the
    selection loop cannot be the one that commits it: the middle data box is the
    eval set — named X_val / y_val, as in the lesson's own split code — and the
    test set does not appear in this picture at all.
    """
    fig, ax = plt.subplots(figsize=(12.4, 4.4))

    def box(x, y, w, h, text, fc, fs=13):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=EDGE,
                          linewidth=2.0, zorder=2)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def header(x, w, text):
        ax.text(x + w / 2, 0.90, text, ha="center", va="center",
                fontsize=15, color=INK)

    # Column geometry: three stacked candidate boxes at x, a single data box
    # between two of them. Rows are at 0.66 / 0.44 / 0.22; a data box spans the
    # middle row's height so it reads as "applied to all three".
    bw, bh, gap = 0.155, 0.145, 0.075
    rows = [0.615, 0.400, 0.185]

    stacks = [
        (0.020, "Model Candidates", ["Model A", "Model B", "Model C"]),
        (0.370, "Trained Models", ["Trained A", "Trained B", "Trained C"]),
        (0.700, "Model Scores", ["Score A", "Score B", "Score C"]),
    ]
    for x, title, labels in stacks:
        header(x, bw, title)
        for y, label, fc in zip(rows, labels, CANDIDATES):
            box(x, y, bw, bh, label, fc)

    # The two data boxes sit in the gaps, vertically centred on the middle row.
    for x, label in ((0.205, "Training Data"), (0.545, "Eval Data")):
        box(x, rows[1] - 0.045, bw - 0.010, bh + 0.090, label, DATA)

    header(0.870, bw, "Selection")
    box(0.870, rows[1] - 0.045, bw, bh + 0.090, "Best Model", WINNER)

    # The timeline, and the one call that happens over each data box.
    ax.annotate("", xy=(1.025, 0.085), xytext=(0.020, 0.085),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, linewidth=2.0,
                                shrinkA=0, shrinkB=0), zorder=1)
    ax.text(0.205 + bw / 2, 0.015, "model.fit(X_tr, y_tr)",
            ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(0.545 + bw / 2, 0.015, "metric(model.predict(X_val), y_val)",
            ha="center", va="center", fontsize=12.5, color=INK)

    ax.set_xlim(0, 1.035)
    ax.set_ylim(-0.03, 1.0)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "validation-and-overfitting", "model-selection-flow.png")


def main() -> None:
    print("generating Session 3 figures ->", ASSETS)
    model_selection_flow()
    print("done.")


if __name__ == "__main__":
    main()
