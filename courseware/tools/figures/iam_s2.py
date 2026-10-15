"""Figures of *IA pour la médecine*, séance 2 (`iam-s2-preparation-forets`).

Writes into `courseware/content/ia-medecine/assets/iam-s2-preparation-forets/`.

Generated from the course's own data (`ai_for_medicine/ia-medecine/donnees/`,
`aim_repo`) and scikit-learn's bundled breast cancer dataset:

- `manquant-informatif.png`: SUPPORT2 death rate at 2 months when a value is
  missing vs present (lesson `preparer-les-donnees`);
- `ecretage-iqr.png`: `mean area` of breast_cancer_noisy before / after IQR
  clipping (same lesson);
- `fuite-ajustement.png`: fit on train, transform everywhere vs fit before
  the split (same lesson);
- `arbre-profondeur-2.png`: a depth-2 decision tree on breast cancer
  (lesson `arbres-et-forets`);
- `profondeur-surapprentissage.png`: train / test accuracy of one tree vs its
  depth, with the random forest for reference (same lesson);
- `importances-foret.png`: random forest feature importances (same lesson);
- `decoupage-apprentissage-validation-test.png`: the three-way split
  (lesson `hyperparametres-validation`);
- `grille-recherche.png`: 5-fold CV accuracy of a grid of trees (same lesson).

Cropped from the owner's deck "Course 1 DS" (Raphael Cousin, Sorbonne), only
when `--slides <path to Course 1 DS.pdf>` is given (needs `pdftoppm`):

- `sous-et-sur-apprentissage.png` (slide 59, the three plots only);
- `validation-croisee-5-plis.png` (slide 63).

    uv run --with pandas --with matplotlib --with seaborn --with scikit-learn \
        python courseware/tools/figures/iam_s2.py [--slides PDF]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import aim_repo  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
DATA = aim_repo.out_dir("ia-medecine", "donnees")
OUT = (REPO / "courseware" / "content" / "ia-medecine" / "assets"
       / "iam-s2-preparation-forets")

BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#2a9d6f"
INK, MUTED = "#222222", "#6b6b6b"


def style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "grid.color": "#e6e6e6", "axes.titleweight": "bold",
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)


# ------------------------------------------------- preparer-les-donnees
def manquant_informatif() -> None:
    X = pd.read_csv(DATA / "support2" / "X_train.csv", index_col="patient_id")
    y = pd.read_csv(DATA / "support2" / "y_train_survie.csv",
                    index_col="patient_id")["outcome"]
    assert (X.index == y.index).all()
    deces = y == "dead"
    labels = {"adlp": "autonomie\n(questionnaire patient)", "edu": "années\nd'études",
              "income": "revenus", "glucose": "glycémie"}
    rows = []
    for col, lab in labels.items():
        m = X[col].isna()
        rows.append({"variable": f"{lab}\n({m.mean():.0%} manquant)",
                     "valeur": "manquante", "deces": deces[m].mean()})
        rows.append({"variable": f"{lab}\n({m.mean():.0%} manquant)",
                     "valeur": "renseignée", "deces": deces[~m].mean()})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    sns.barplot(data=d, x="variable", y="deces", hue="valeur",
                palette={"manquante": ORANGE, "renseignée": BLEU}, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt=lambda v: f"{v:.0%}", fontsize=9, padding=2)
    ax.axhline(deces.mean(), color=MUTED, ls="--", lw=1)
    ax.text(-0.45, deces.mean() + 0.015, f"tous les patients : {deces.mean():.0%}",
            ha="left", color=MUTED, fontsize=9)
    ax.set(xlabel="", ylabel="Décès à 2 mois", ylim=(0, 0.85),
           title="SUPPORT2 : l'absence d'une valeur est déjà une information")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(title="Valeur", frameon=False)
    save(fig, "manquant-informatif.png")


def ecretage_iqr() -> None:
    df = pd.read_csv(DATA / "breast_cancer_noisy.csv")
    x = df["mean area"].dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    bas, haut = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    for ax, serie, titre in ((axes[0], x, "Avant"),
                             (axes[1], x.clip(bas, haut), "Après écrêtage")):
        sns.boxplot(y=serie, ax=ax, color=BLEU, width=0.4,
                    boxprops={"alpha": 0.4}, flierprops={"markerfacecolor": ORANGE})
        ax.set(title=titre, ylabel="mean area (pixels²)")
    axes[0].axhline(haut, color=ORANGE, ls="--", lw=1)
    axes[0].text(0.45, haut, f" Q3 + 1,5 × IQR = {haut:.0f}", va="bottom",
                 ha="right", color=ORANGE, fontsize=9)
    axes[1].set_ylabel("")
    fig.suptitle("Cancer du sein bruité : aire moyenne du noyau", fontweight="bold")
    fig.tight_layout()
    save(fig, "ecretage-iqr.png")


def _cases(ax, x0: float, n_train: int, n_test: int, y: float = 0.0) -> None:
    for i in range(n_train + n_test):
        test = i >= n_train
        ax.add_patch(Rectangle((x0 + i, y), 1, 1, lw=1.2,
                               ec=ORANGE if test else BLEU,
                               fc="#f6dcd3" if test else "#dce8f5"))
        ax.text(x0 + i + 0.5, y + 0.5, str(i), ha="center", va="center", fontsize=11)
    ax.text(x0 + n_train / 2, y - 0.6, "apprentissage", ha="center", color=BLEU,
            fontweight="bold")
    ax.text(x0 + n_train + n_test / 2, y - 0.6, "test", ha="center", color=ORANGE,
            fontweight="bold")


def fuite_ajustement() -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set(xlim=(-0.5, 23.5), ylim=(-1.4, 5.2))
    ax.axis("off")
    # left: fit on all rows
    _cases(ax, 0, 8, 2)
    ax.add_patch(Rectangle((2.5, 3), 5, 1.2, fc="white", ec=ORANGE, lw=1.6))
    ax.text(5, 3.6, "médiane calculée\nsur les 10 lignes", ha="center", va="center",
            fontsize=10)
    ax.add_patch(FancyArrowPatch((5, 1.1), (5, 2.95), arrowstyle="-|>",
                                 mutation_scale=14, color=ORANGE))
    ax.text(5, 4.9, "Fuite : ajuster avant de séparer", ha="center",
            color=ORANGE, fontweight="bold", fontsize=12)
    # right: fit on train only
    _cases(ax, 13, 8, 2)
    ax.add_patch(Rectangle((14.5, 3), 5, 1.2, fc="white", ec=BLEU, lw=1.6))
    ax.text(17, 3.6, "médiane calculée\nsur l'apprentissage", ha="center",
            va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((16, 1.1), (16, 2.95), arrowstyle="-|>",
                                 mutation_scale=14, color=BLEU))
    ax.text(15.8, 2, "fit", ha="right", color=BLEU, style="italic")
    ax.add_patch(FancyArrowPatch((19.5, 3.0), (22, 1.1), arrowstyle="-|>",
                                 mutation_scale=14, color=INK,
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(21.6, 2.4, "transform", color=INK, style="italic")
    ax.text(18, 4.9, "Correct : fit sur l'apprentissage seul", ha="center",
            color=BLEU, fontweight="bold", fontsize=12)
    save(fig, "fuite-ajustement.png")


# ----------------------------------------------------- arbres-et-forets
def _cancer():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def arbre_profondeur_2() -> None:
    X, y = _cancer()
    arbre = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_tree(arbre, feature_names=list(X.columns),
              class_names=["maligne", "bénigne"], filled=True, rounded=True,
              impurity=True, proportion=False, fontsize=10, ax=ax)
    save(fig, "arbre-profondeur-2.png")


def profondeur_surapprentissage() -> None:
    """Mean train / held-out accuracy over a 5-fold split, so no lucky split."""
    from sklearn.model_selection import cross_validate

    X, y = _cancer()
    depths = list(range(1, 13))
    tr, te = [], []
    for d in depths:
        cv = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=0),
                            X, y, cv=5, return_train_score=True)
        tr.append(cv["train_score"].mean())
        te.append(cv["test_score"].mean())
    foret = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0),
                            X, y, cv=5).mean()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(depths, tr, "o-", color=BLEU, label="arbre : données d'apprentissage")
    ax.plot(depths, te, "o-", color=ORANGE, label="arbre : nouvelles données")
    ax.axhline(foret, color=VERT, ls="--", label="forêt de 300 arbres : nouvelles données")
    ax.set(xlabel="Profondeur maximale de l'arbre (max_depth)",
           ylabel="Exactitude (moyenne sur 5 plis)", ylim=(0.86, 1.005), xticks=depths,
           title="Un arbre profond apprend aussi le bruit")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "profondeur-surapprentissage.png")


def importances_foret() -> None:
    X, y = _cancer()
    foret = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
    imp = pd.Series(foret.feature_importances_, index=X.columns).nlargest(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color=BLEU)
    ax.set(xlabel="Importance (baisse moyenne d'impureté)",
           title="Forêt aléatoire : les 10 variables les plus utiles")
    save(fig, "importances-foret.png")


# ------------------------------------------- hyperparametres-validation
def decoupage() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set(xlim=(0, 100), ylim=(-1.5, 2.2))
    ax.axis("off")
    blocs = [(0, 60, "#dce8f5", BLEU, "apprentissage (60 %)",
              "le modèle ajuste\nses paramètres"),
             (60, 20, "#d9efe5", VERT, "validation (20 %)",
              "on choisit les\nhyperparamètres"),
             (80, 20, "#f6dcd3", ORANGE, "test (20 %)",
              "une seule fois,\nà la fin")]
    for x0, w, fc, ec, titre, role in blocs:
        ax.add_patch(Rectangle((x0, 0), w, 1.2, fc=fc, ec=ec, lw=1.6))
        ax.text(x0 + w / 2, 0.6, titre, ha="center", va="center",
                fontweight="bold", color=ec)
        ax.text(x0 + w / 2, -0.6, role, ha="center", va="center", fontsize=10,
                color=INK)
    ax.text(50, 1.8, "Les patients disponibles", ha="center", fontweight="bold")
    save(fig, "decoupage-apprentissage-validation-test.png")


def grille_recherche() -> None:
    X, y = _cancer()
    depths = [1, 2, 3, 5, 8, None]
    leaves = [1, 5, 20, 50]
    scores = np.zeros((len(leaves), len(depths)))
    for i, f in enumerate(leaves):
        for j, d in enumerate(depths):
            arbre = DecisionTreeClassifier(max_depth=d, min_samples_leaf=f,
                                           random_state=0)
            scores[i, j] = cross_val_score(arbre, X, y, cv=5).mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.heatmap(pd.DataFrame(scores, index=leaves,
                             columns=["aucune" if d is None else d for d in depths]),
                annot=True, fmt=".3f", cmap="Blues", cbar=False, linewidths=1,
                linecolor="white", ax=ax)
    i, j = np.unravel_index(scores.argmax(), scores.shape)
    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, ec=ORANGE, lw=3))
    ax.set(xlabel="max_depth", ylabel="min_samples_leaf",
           title="Exactitude moyenne en validation croisée à 5 plis")
    ax.tick_params(axis="y", rotation=0)
    save(fig, "grille-recherche.png")


# --------------------------------------------------- owner's slide crops
def crops(pdf: pathlib.Path) -> None:
    """Slides 59 and 63 of "Course 1 DS", rendered at 150 dpi (1500 x 844)."""
    from PIL import Image

    boxes = {59: ((70, 330, 1430, 612), "sous-et-sur-apprentissage.png"),
             63: ((380, 160, 1035, 625), "validation-croisee-5-plis.png")}
    with tempfile.TemporaryDirectory() as tmp:
        for page, (box, name) in boxes.items():
            stem = pathlib.Path(tmp) / f"p{page}"
            subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile",
                            "-f", str(page), "-l", str(page), str(pdf), str(stem)],
                           check=True)
            img = Image.open(f"{stem}.png")
            assert img.size == (1500, 844), img.size
            img.crop(box).save(OUT / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=pathlib.Path,
                        help='"Course 1 DS.pdf", to crop slides 59 and 63')
    args = parser.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    manquant_informatif()
    ecretage_iqr()
    fuite_ajustement()
    arbre_profondeur_2()
    profondeur_surapprentissage()
    importances_foret()
    decoupage()
    grille_recherche()
    if args.slides:
        crops(args.slides)
    print(f"wrote figures into {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
