"""Figures of *Python pour la médecine*, séance 6, lesson `analyse-exploratoire`.

Reads the course's own copy of the UCI heart failure records
(`ai_for_modecine/python-medecine/donnees/insuffisance_cardiaque.csv`, `aim_repo`),
Chicco & Jurman 2020, CC BY 4.0) and writes into
`courseware/content/python-medecine/assets/pym-s6-visualisation-analyse/`:

- `pym_s6_fe_par_deces.png`: ejection fraction by outcome, box + points;
- `pym_s6_correlations.png`: Pearson and Spearman correlation matrices;
- `pym_s6_suivi_deces.png`: follow-up time by outcome (the bias slide).

    uv run --with pandas --with matplotlib --with seaborn python courseware/tools/figures/pym_s6.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import aim_repo  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
DATA = aim_repo.out_dir("python-medecine", "donnees") / "insuffisance_cardiaque.csv"
OUT = (REPO / "courseware" / "content" / "python-medecine" / "assets"
       / "pym-s6-visualisation-analyse")

SURVIE, DECES = "#2a78d6", "#eb6834"
PALETTE = {"Survivant": SURVIE, "Décès": DECES}
INK, MUTED = "#222222", "#6b6b6b"


def style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "grid.color": "#e6e6e6", "axes.titleweight": "bold",
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def load() -> pd.DataFrame:
    ic = pd.read_csv(DATA)
    ic["issue"] = ic["deces"].map({0: "Survivant", 1: "Décès"})
    return ic


def fe_par_deces(ic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    order = ["Survivant", "Décès"]
    sns.boxplot(data=ic, x="issue", y="fraction_ejection", order=order,
                hue="issue", palette=PALETTE, width=0.45, fliersize=0,
                boxprops={"alpha": 0.35}, ax=ax, legend=False)
    sns.stripplot(data=ic, x="issue", y="fraction_ejection", order=order,
                  hue="issue", palette=PALETTE, size=3.5, jitter=0.18,
                  alpha=0.75, ax=ax, legend=False)
    for i, g in enumerate(order):
        sub = ic[ic["issue"] == g]
        ax.text(i + 0.3, sub["fraction_ejection"].median(),
                f"médiane {sub['fraction_ejection'].median():.0f} %\n(n = {len(sub)})",
                va="center", fontsize=9, color=INK)
    ax.set(xlabel="", ylabel="Fraction d'éjection (%)",
           title="Fraction d'éjection selon l'issue du suivi")
    fig.savefig(OUT / "pym_s6_fe_par_deces.png")
    plt.close(fig)


def correlations(ic: pd.DataFrame) -> None:
    cols = ["age", "fraction_ejection", "creatinine", "sodium", "cpk", "deces"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.2]})
    for ax, method in zip(axes, ["pearson", "spearman"]):
        sns.heatmap(ic[cols].corr(method=method), annot=True, fmt=".2f",
                    cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                    cbar=method == "spearman", annot_kws={"size": 9},
                    linewidths=1, linecolor="white", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.set_title(method.capitalize())
    fig.tight_layout()
    fig.savefig(OUT / "pym_s6_correlations.png")
    plt.close(fig)


def suivi_deces(ic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.histplot(data=ic, x="suivi_jours", hue="issue", palette=PALETTE,
                 hue_order=["Survivant", "Décès"], bins=24, multiple="stack",
                 alpha=0.85, edgecolor="white", ax=ax)
    ax.set(xlabel="Durée de suivi (jours)", ylabel="Patients",
           title="Les décès ont, par construction, un suivi court")
    ax.get_legend().set_title("")
    fig.savefig(OUT / "pym_s6_suivi_deces.png")
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    ic = load()
    fe_par_deces(ic)
    correlations(ic)
    suivi_deces(ic)
    print(f"wrote 3 figures into {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
