"""Figures for the `ia-medecine` module `iam-s1-classification`.

Writes into `content/ia-medecine/assets/iam-s1-classification/`:

* `cancer-sein-deux-variables.png` — `apprendre-a-partir-de-donnees.md`: the
  569 tumours of Breast Cancer Wisconsin on two features, coloured by diagnosis.
* `regression-temperature.png` — `modele-perte-gradient.md`: the owner's
  worked example (slides 35-36 of "Course 1 DS": mean temperature 1980-2020,
  initial line a=-0.01, b=34.70) trained by gradient descent on the MSE, with
  the loss per iteration.
* `taux-apprentissage.png` — `modele-perte-gradient.md`: gradient descent on
  f(θ) = θ² with a small, a good and a too large learning rate (slide 33).
* `sigmoide.png`, `lineaire-vs-logistique.png`, `entropie-croisee.png`,
  `mise-a-l-echelle.png` — `regression-logistique.md` (slides 49-53 and the
  scaling cell of session1/tp1.ipynb).
* `validation-croisee.png`, `courbe-roc.png` — `evaluer-un-classifieur.md`
  (slides 63 and 76; the ROC is the TP's scaled logistic regression).

`types-apprentissage.png` in the same folder is not drawn here: it is the
figure strip of slide 18 of the owner's deck "Course 1 DS" (github.com/racousin/
DFGSM2-IA-S2, slides/), cropped at 150 dpi.

    uv run --no-project --with matplotlib --with scikit-learn \
        python courseware/tools/figures/iam_s1.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content" / "ia-medecine"
       / "assets" / "iam-s1-classification")

INK = "#1f2933"
BLUE = "#2f6db5"
RED = "#c0392b"
GREEN = "#2e8b57"
ORANGE = "#e08a1e"
GREY = "#9aa5b1"

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def _cancer():
    """Features and target with malignant = 1 (sklearn codes malignant as 0)."""
    data = load_breast_cancer(as_frame=True)
    return data.data, 1 - data.target


def cancer_deux_variables() -> None:
    X, y = _cancer()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for value, label, color, marker in ((0, "bénigne", BLUE, "o"),
                                        (1, "maligne", RED, "^")):
        sel = y == value
        ax.scatter(X.loc[sel, "mean radius"], X.loc[sel, "mean texture"], s=18,
                   alpha=0.6, c=color, marker=marker,
                   label=f"tumeur {label} ({sel.sum()})")
    ax.set_xlabel("rayon moyen des noyaux (variable 1)")
    ax.set_ylabel("texture moyenne (variable 2)")
    ax.set_title("569 tumeurs du sein : une observation = un point", color=INK)
    ax.legend(frameon=False)
    _save(fig, "cancer-sein-deux-variables.png")


def regression_temperature() -> None:
    years = np.arange(1980, 2021, 5)
    temp = np.array([14.18, 14.10, 14.35, 14.38, 14.33, 14.55, 14.58, 14.80,
                     14.92])
    # Train on x = (year - 2000) / 10 so that one learning rate fits both
    # parameters; a (per decade) and b (at 2000) map back to the slide's line.
    x = (years - 2000) / 10
    a, b = -0.01 * 10, -0.01 * 2000 + 34.70
    lr, losses, start = 0.1, [], (a, b)
    for _ in range(60):
        err = a * x + b - temp
        losses.append(np.mean(err ** 2))
        a, b = a - lr * 2 * np.mean(err * x), b - lr * 2 * np.mean(err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    grid = np.linspace(-2.2, 2.2, 50)
    ax1.scatter(years, temp, c=INK, zorder=3, label="mesures")
    ax1.plot(2000 + 10 * grid, start[0] * grid + start[1], c=GREY, ls="--",
             label="modèle initial")
    ax1.plot(2000 + 10 * grid, a * grid + b, c=RED, label="modèle entraîné")
    for xi, yi in zip(x, temp):
        ax1.plot([2000 + 10 * xi] * 2, [yi, start[0] * xi + start[1]],
                 c=GREY, lw=0.8)
    ax1.set_xlabel("année")
    ax1.set_ylabel("température moyenne (°C)")
    ax1.set_title("Ajuster une droite : les écarts initiaux", color=INK)
    ax1.legend(frameon=False, loc="upper left")
    ax2.plot(losses, c=BLUE, lw=2)
    ax2.set_xlabel("itération de la descente de gradient")
    ax2.set_ylabel("perte (MSE)")
    ax2.set_title("La perte diminue à chaque pas", color=INK)
    _save(fig, "regression-temperature.png")


def taux_apprentissage() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    grid = np.linspace(-4.6, 4.6, 200)
    for ax, lr, title in zip(axes, (0.05, 0.3, 1.05),
                             ("trop petit : lent", "adapté : converge",
                              "trop grand : diverge")):
        theta, path = 3.0, [3.0]
        for _ in range(8 if lr < 1 else 4):
            theta = theta - lr * 2 * theta
            path.append(theta)
        path = np.array(path)
        ax.plot(grid, grid ** 2, c=GREY)
        ax.plot(path, path ** 2, "-o", c=RED, ms=5, lw=1.2)
        ax.scatter([0], [0], marker="*", s=160, c=GREEN, zorder=4)
        ax.set_title(f"η = {str(lr).replace('.', ',')} — {title}", color=INK)
        ax.set_xlabel("paramètre θ")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.5, 22)
    axes[0].set_ylabel("perte L(θ) = θ²")
    _save(fig, "taux-apprentissage.png")


def sigmoide() -> None:
    z = np.linspace(-7, 7, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, 1 / (1 + np.exp(-z)), c=BLUE, lw=2.5)
    ax.axhline(0.5, c=GREY, ls="--", lw=1)
    ax.axvline(0, c=GREY, ls="--", lw=1)
    ax.axhline(0, c=GREY, lw=0.6)
    ax.axhline(1, c=GREY, lw=0.6)
    ax.annotate("σ(0) = 0,5", (0, 0.5), (1.2, 0.35), color=INK,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set_xlabel("z = θ₀ + θ₁x₁ + … + θₚxₚ")
    ax.set_ylabel("σ(z)")
    ax.set_title("La sigmoïde envoie ℝ dans ]0, 1[", color=INK)
    _save(fig, "sigmoide.png")


def lineaire_vs_logistique() -> None:
    X, y = _cancer()
    r = X[["mean radius"]].to_numpy()
    grid = np.linspace(5, 30, 300).reshape(-1, 1)
    lin = LinearRegression().fit(r, y)
    log = LogisticRegression().fit(r, y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, len(y))
    for ax, curve, title, color in (
            (axes[0], lin.predict(grid), "Régression linéaire", ORANGE),
            (axes[1], log.predict_proba(grid)[:, 1], "Régression logistique",
             BLUE)):
        ax.scatter(r[:, 0], y + jitter, s=10, alpha=0.35, c=INK)
        ax.plot(grid[:, 0], curve, c=color, lw=2.5)
        ax.axhline(0.5, c=GREY, ls="--", lw=1)
        ax.set_ylim(-0.45, 1.45)
        ax.set_xlabel("rayon moyen des noyaux")
        ax.set_title(title, color=INK)
    axes[0].set_ylabel("maligne (1) / bénigne (0)")
    axes[0].text(6, 1.25, "sort de [0, 1]", color=ORANGE)
    axes[1].text(6, 1.25, "une probabilité", color=BLUE)
    _save(fig, "lineaire-vs-logistique.png")


def entropie_croisee() -> None:
    p = np.linspace(0.005, 0.995, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, -np.log(p), c=RED, lw=2.5, label="si y = 1 : −log(p̂)")
    ax.plot(p, -np.log(1 - p), c=BLUE, lw=2.5, label="si y = 0 : −log(1 − p̂)")
    ax.set_xlabel("probabilité prédite p̂ (classe 1)")
    ax.set_ylabel("perte")
    ax.set_ylim(0, 5)
    ax.set_title("Une erreur sûre d'elle coûte très cher", color=INK)
    ax.legend(frameon=False, loc="upper center")
    _save(fig, "entropie-croisee.png")


def mise_a_l_echelle() -> None:
    def path(scale_x: float) -> np.ndarray:
        # Quadratic loss (scale_x * θ1)² + θ2²; the step is set by the
        # steepest direction, so the flat one crawls when scale_x is large.
        lr = 0.9 / (2 * max(scale_x ** 2, 1))
        theta, pts = np.array([-3.0, 3.0]), []
        for _ in range(25):
            pts.append(theta.copy())
            theta = theta - lr * np.array([2 * scale_x ** 2 * theta[0],
                                           2 * theta[1]])
        return np.array(pts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    g = np.linspace(-3.6, 3.6, 200)
    T1, T2 = np.meshgrid(g, g)
    for ax, s, title in ((axes[0], 4.0, "variables à des échelles différentes"),
                         (axes[1], 1.0, "variables standardisées")):
        ax.contour(T1, T2, (s * T1) ** 2 + T2 ** 2, levels=12, colors=GREY,
                   linewidths=0.8)
        pts = path(s)
        ax.plot(pts[:, 0], pts[:, 1], "-o", c=RED, ms=3, lw=1)
        ax.scatter([0], [0], marker="*", s=160, c=GREEN, zorder=4)
        ax.set_title(f"{title}\n25 pas de gradient", color=INK)
        ax.set_xlabel("θ₁")
        ax.set_ylabel("θ₂")
        ax.set_aspect("equal")
    _save(fig, "mise-a-l-echelle.png")


def validation_croisee() -> None:
    k = 5
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for split in range(k):
        y0 = k - 1 - split
        ax.text(-0.15, y0 + 0.4, f"tour {split + 1}", ha="right", va="center",
                color=INK)
        for fold in range(k):
            test = fold == split
            ax.add_patch(Rectangle((fold, y0 + 0.05), 0.95, 0.7,
                                   fc=RED if test else BLUE,
                                   alpha=0.85 if test else 0.35, ec="white"))
            ax.text(fold + 0.475, y0 + 0.4, "test" if test else "entraînement",
                    ha="center", va="center", fontsize=10,
                    color="white" if test else INK)
        ax.text(k + 0.15, y0 + 0.4, f"score {split + 1}", va="center",
                color=INK)
    ax.set_xlim(-1.2, k + 1.2)
    ax.set_ylim(-0.1, k)
    ax.axis("off")
    ax.set_title("Validation croisée à 5 blocs : chaque patient est testé une "
                 "fois, on moyenne les 5 scores", color=INK)
    _save(fig, "validation-croisee.png")


def courbe_roc() -> None:
    X, y = _cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=5000).fit(scaler.transform(X_tr), y_tr)
    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot(fpr, tpr, c=BLUE, lw=2.5,
            label=f"régression logistique (AUC = {auc:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], c=GREY, ls="--", label="hasard (AUC = 0,5)")
    ax.set_xlabel("taux de faux positifs (1 − spécificité)")
    ax.set_ylabel("rappel (sensibilité)")
    ax.set_title("Courbe ROC, cancer du sein (jeu de test)", color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal")
    _save(fig, "courbe-roc.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cancer_deux_variables()
    regression_temperature()
    taux_apprentissage()
    sigmoide()
    lineaire_vs_logistique()
    entropie_croisee()
    mise_a_l_echelle()
    validation_croisee()
    courbe_roc()


if __name__ == "__main__":
    main()
