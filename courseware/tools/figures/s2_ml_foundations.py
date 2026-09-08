"""Figures generated for `python-ai-engineering` Session 2.

Every image this script writes is committed under
`content/python-ai-engineering/assets/s2-ml-foundations/<lesson-slug>/`, so a
figure can always be traced back to the code that drew it. Re-run with:

    uv run --with seaborn --with scikit-learn --with pandas \
        python courseware/tools/figures/s2_ml_foundations.py

Deck-extracted images (the ones lifted from the taught PPTX) are *not* produced
here — this script owns only the figures authored for the 2026 rewrite.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "python-ai-engineering"
    / "assets"
    / "s2-ml-foundations"
)
# The feature-engineering figure is drawn from the session's own challenge data,
# not from a bundled seaborn dataset: the -104 -> -74 number the lesson quotes is
# a claim about *these* CSVs, so the picture has to come from them too.
BIKE = pathlib.Path(__file__).resolve().parents[2] / "competitions" / "s2-bike-demand"

INK = "#1f2933"
ACCENT = "#2f6f9f"
WARM = "#c1553b"
SPECIES = {"Adelie": "#2f6f9f", "Chinstrap": "#c1553b", "Gentoo": "#3f8f6f"}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#dfe3e8",
        "font.family": "DejaVu Sans",
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
# ai-ml-dl-landscape — the one message the retired pipeline lesson carried
# --------------------------------------------------------------------------- #
def rules_vs_learning() -> None:
    """Classical programming and machine learning, as the same box wired two
    different ways. This is the closing message of the landscape lesson."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    def box(ax, xy, w, h, text, fc, ec, fs=12, weight="normal"):
        ax.add_patch(
            plt.Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2)
        )
        ax.text(
            xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, fontweight=weight,
        )

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6, shrinkA=0, shrinkB=0),
            zorder=1,
        )

    for ax, title, inputs, engine, output, ec in (
        (axes[0], "Classical programming", ["Rules", "Data"], "Program", "Answers", ACCENT),
        (axes[1], "Machine learning", ["Data", "Answers"], "Training", "Rules", WARM),
    ):
        ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, pad=14)
        box(ax, (0.02, 0.60), 0.26, 0.24, inputs[0], "#eef3f8", ec)
        box(ax, (0.02, 0.16), 0.26, 0.24, inputs[1], "#eef3f8", ec)
        box(ax, (0.40, 0.38), 0.26, 0.24, engine, "#ffffff", INK, weight="bold")
        box(ax, (0.74, 0.38), 0.24, 0.24, output, "#eef3f8", ec)
        arrow(ax, 0.28, 0.72, 0.40, 0.56)
        arrow(ax, 0.28, 0.28, 0.40, 0.44)
        arrow(ax, 0.66, 0.50, 0.74, 0.50)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    axes[1].text(
        0.5, -0.04, "the rules are the output, not the input",
        ha="center", va="top", fontsize=11, style="italic", color=WARM,
        transform=axes[1].transAxes,
    )
    fig.tight_layout()
    save(fig, "ai-ml-dl-landscape", "rules-vs-learning.png")


# --------------------------------------------------------------------------- #
# the-data — the concrete dataset the lesson is built on
# --------------------------------------------------------------------------- #
def penguins_pairplot(df) -> None:
    """Every numeric pair at once: the first thing to run on a new table."""
    num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    g = sns.pairplot(
        df.dropna(subset=num + ["species"]),
        vars=num,
        hue="species",
        palette=SPECIES,
        diag_kind="hist",
        plot_kws=dict(s=18, alpha=0.75, edgecolor="none"),
        height=1.9,
    )
    g.figure.suptitle(
        "penguins: 4 numeric variables, all 6 pairs, coloured by species",
        y=1.02, fontsize=13, color=INK,
    )
    g.figure.savefig(out("the-data", "penguins-pairplot.png"), facecolor="white", bbox_inches="tight")
    plt.close(g.figure)
    print("  wrote assets/s2-ml-foundations/the-data/penguins-pairplot.png")


def penguins_targets(df) -> None:
    """The same two features, the same 342 rows — and two different targets.
    The task is decided by the type of y, not by the model."""
    d = df.dropna(subset=["flipper_length_mm", "body_mass_g", "species"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    sc = axes[0].scatter(
        d["flipper_length_mm"], d["bill_length_mm"],
        c=d["body_mass_g"], cmap="viridis", s=26, edgecolor="none",
    )
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("body_mass_g", color=INK)
    axes[0].set_title(r"regression:  $y \in \mathbb{R}$", fontsize=13, color=INK)

    for name, sub in d.groupby("species", observed=True):
        axes[1].scatter(
            sub["flipper_length_mm"], sub["bill_length_mm"],
            c=SPECIES[str(name)], label=str(name), s=26, edgecolor="none",
        )
    axes[1].legend(title="species", frameon=False)
    axes[1].set_title(r"classification:  $y \in \{1,\,2,\,3\}$", fontsize=13, color=INK)

    for ax in axes:
        ax.set_xlabel("flipper_length_mm")
        ax.set_ylabel("bill_length_mm")

    fig.suptitle(
        "Same X. Change the column you call y and the task changes.",
        fontsize=13.5, color=INK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "the-data", "penguins-targets.png")


# --------------------------------------------------------------------------- #
# data-preparation — missingness in the dataset they already met
# --------------------------------------------------------------------------- #
def penguins_missing(df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2, 1]})

    sns.heatmap(
        df.isna().T, cbar=False, cmap=["#eef3f8", WARM],
        ax=axes[0], xticklabels=False, yticklabels=True,
    )
    axes[0].set_title("df.isna() — 344 rows across, one line per column", fontsize=12, color=INK)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9)
    axes[0].set_xlabel("observations")

    counts = df.isna().sum().sort_values()
    counts = counts[counts > 0]
    axes[1].barh(counts.index, counts.values, color=WARM)
    for i, v in enumerate(counts.values):
        axes[1].text(v + 0.15, i, str(v), va="center", fontsize=10, color=INK)
    axes[1].set_title("missing per column", fontsize=12, color=INK)
    axes[1].set_xlim(0, counts.max() * 1.25)
    axes[1].tick_params(labelsize=9)

    fig.tight_layout()
    save(fig, "data-preparation", "penguins-missing.png")


def hour_numeric_vs_onehot() -> None:
    """The feature-engineering claim, drawn from the challenge's own CSVs.

    `hour` as a quantity buys a linear model one coefficient, so it can only
    tilt; `hour` as 24 unordered levels buys it 24, so it can trace the commute
    curve. Both panels are the same LinearRegression on the same rows — the only
    difference is the encoding, which is the entire point of the section.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    data = BIKE / "data"
    if not (data / "X_train.csv").exists():           # pragma: no cover
        raise SystemExit(f"{data}/X_train.csv missing — run the package's "
                         f"prepare_data.py first")
    X = pd.read_csv(data / "X_train.csv")
    y = pd.read_csv(data / "y_train.csv")["prediction"]

    hours = np.arange(24)
    observed = y.groupby(X["hour"]).mean().reindex(hours)

    fits = {
        "hour as a number — 1 coefficient":
            (X[["hour"]].to_numpy(float), hours.reshape(-1, 1)),
        "hour as 24 categories — 24 coefficients":
            (np.eye(24)[X["hour"].to_numpy()], np.eye(24)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, (title, (design, grid)) in zip(axes, fits.items()):
        pred = LinearRegression().fit(design, y).predict(grid)
        ax.plot(hours, observed, "o-", color=ACCENT, ms=4, lw=1.6,
                label="mean rentals, observed")
        ax.plot(hours, pred, color=WARM, lw=2.4, label="what the model can say")
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("hour")
        ax.set_xticks(range(0, 24, 4))
    axes[0].set_ylabel("rentals")
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle("Same model, same rows. Only the encoding of one column changed.",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "data-preparation", "hour-numeric-vs-onehot.png")


def main() -> None:
    print("generating Session 2 figures ->", ASSETS)
    df = sns.load_dataset("penguins")
    rules_vs_learning()
    penguins_pairplot(df)
    penguins_targets(df)
    penguins_missing(df)
    hour_numeric_vs_onehot()
    print("done.")


if __name__ == "__main__":
    main()
