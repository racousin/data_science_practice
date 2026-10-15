"""Figures of *IA pour la médecine*, séance 3 (module `iam-s3-reseaux-de-neurones`).

Writes into `courseware/content/ia-medecine/assets/iam-s3-reseaux-de-neurones/`:

Reused classic diagrams (copied, unmodified):
- `neurone-biologique-artificiel.png`: from the owner's website,
  `website/public/assets/python-deep-learning/module1/neuron_bio.webp` (to PNG);
- `perceptron-multicouche.jpeg`: `ms2a-machine-learning-practice/assets/nn/mlp.jpeg`;
- `fonctions-activation.png`: `ms2a-machine-learning-practice/assets/nn/activation-functions.png`.

Crops of the owner's deck "Course 1 DS" (github.com/racousin/DFGSM2-IA-S2,
slides/Course 1 DS.pdf, rendered with `pdftoppm`):
- `taux-apprentissage.png`: slide 33, the three learning-rate panels;
- `boucle-entrainement.png`: slide 35, the training loop (logo masked).

Generated (matplotlib, French labels):
- `frontieres-lr-mlp.png`: decision boundary of a logistic regression and an
  MLP on two interleaved half-moons (`make_moons`);
- `courbe-perte-arret-precoce.png`: training and validation loss of an MLP on
  `maladie_cardiaque.csv`, epoch by epoch, with the early-stopping point;
- `matrices-confusion.png`: LR / random forest / MLP on the TP 3 test split.

    DFGSM2_DIR=<clone of DFGSM2-IA-S2> uv run --with pandas --with scikit-learn --with matplotlib --with pillow \
        python courseware/tools/figures/iam_s3.py
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, log_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import aim_repo  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "courseware" / "content" / "ia-medecine" / "assets" / "iam-s3-reseaux-de-neurones"
DATA = aim_repo.out_dir("ia-medecine", "donnees") / "maladie_cardiaque.csv"
# The owner's deck: DFGSM2_DIR = a local clone of github.com/racousin/DFGSM2-IA-S2
# (required; KeyError if unset).
DECK = pathlib.Path(os.environ["DFGSM2_DIR"]) / "slides" / "Course 1 DS.pdf"

MS2A = REPO / "courseware" / "content" / "ms2a-machine-learning-practice" / "assets"
COPIES = {
    "perceptron-multicouche.jpeg": MS2A / "nn" / "mlp.jpeg",
    "fonctions-activation.png": MS2A / "nn" / "activation-functions.png",
}
NEURON = REPO / "website" / "public" / "assets" / "python-deep-learning" / "module1" / "neuron_bio.webp"

BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#2e9e5b"
INK, MUTED = "#222222", "#6b6b6b"


def style() -> None:
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e6e6",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "font.size": 11,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def copies() -> None:
    for name, src in COPIES.items():
        shutil.copyfile(src, OUT / name)
    Image.open(NEURON).convert("RGB").save(OUT / "neurone-biologique-artificiel.png")


def slide(page: int, box: tuple[float, float, float, float],
          mask: tuple[float, float, float, float] | None, name: str) -> None:
    """Render one deck page; crop `box` (fractions x0, y0, x1, y1) after
    painting `mask` (same fractions) white."""
    with tempfile.TemporaryDirectory() as tmp:
        stem = pathlib.Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-f", str(page), "-l",
                        str(page), "-singlefile", str(DECK), str(stem)], check=True,
                       stderr=subprocess.DEVNULL)
        img = Image.open(f"{stem}.png").convert("RGB")
    w, h = img.size
    if mask is not None:
        x0, y0, x1, y1 = mask
        img.paste((255, 255, 255), (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    x0, y0, x1, y1 = box
    img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(OUT / name)


def frontieres() -> None:
    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    models = [
        ("Régression logistique", LogisticRegression()),
        ("Réseau de neurones (32, 32)",
         MLPClassifier((32, 32), max_iter=2000, random_state=0)),
    ]
    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 300), np.linspace(-1.4, 1.9, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, model) in zip(axes, models):
        model.fit(X, y)
        proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, proba, levels=20, cmap="RdBu_r", alpha=0.35)
        ax.contour(xx, yy, proba, levels=[0.5], colors=INK, linewidths=1.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, c=BLEU, label="classe 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, c=ORANGE, label="classe 1")
        acc = model.score(X, y)
        ax.set_title(f"{title}\nexactitude {acc:.0%}".replace("%", " %"))
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.savefig(OUT / "frontieres-lr-mlp.png")
    plt.close(fig)


def heart() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA)
    return df.drop(columns="maladie_cardiaque"), df["maladie_cardiaque"]


def arret_precoce() -> None:
    X, y = heart()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    sc = StandardScaler().fit(X_fit)
    a, v = sc.transform(X_fit), sc.transform(X_val)
    model = MLPClassifier((100,), batch_size=32, random_state=42)
    train, val = [], []
    for _ in range(200):
        model.partial_fit(a, y_fit, classes=[0, 1])
        train.append(log_loss(y_fit, model.predict_proba(a)))
        val.append(log_loss(y_val, model.predict_proba(v)))
    best = int(np.argmin(val))
    patience = 10
    epochs = np.arange(1, len(val) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(epochs, train, color=BLEU, lw=2, label="perte d'entraînement")
    ax.plot(epochs, val, color=ORANGE, lw=2, label="perte de validation")
    ax.axvline(best + 1, color=INK, ls="--", lw=1)
    ax.axvspan(best + 1, best + 1 + patience, color="#cccccc", alpha=0.4,
               label=f"patience : {patience} époques sans amélioration")
    ax.scatter([best + 1], [val[best]], color=INK, zorder=5, s=40)
    ax.annotate(f"meilleure validation (époque {best + 1}) :\non garde ce modèle",
                xy=(best + 1, val[best]), xytext=(best + 30, val[best] + 0.35),
                arrowprops={"arrowstyle": "->", "color": INK}, fontsize=10)
    ax.annotate("le réseau apprend le bruit\ndes patients d'entraînement",
                xy=(150, val[149]), xytext=(110, 1.1), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set_xlabel("époque (un passage sur les données d'entraînement)")
    ax.set_ylabel("perte (entropie croisée)")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "courbe-perte-arret-precoce.png")
    plt.close(fig)
    print(f"  early stopping: best epoch {best + 1}, val loss {val[best]:.3f}, "
          f"final {val[-1]:.3f}")


def matrices() -> None:
    X, y = heart()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(X_train)
    a, b = sc.transform(X_train), sc.transform(X_test)
    models = {
        "Régression logistique": LogisticRegression(max_iter=5000, random_state=42),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=100, random_state=42),
        "Réseau de neurones": MLPClassifier(max_iter=1000, early_stopping=True,
                                            random_state=42),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, model) in zip(axes, models.items()):
        pred = model.fit(a, y_train).predict(b)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, ax=ax, cmap="Blues", colorbar=False,
            display_labels=["absente", "présente"])
        tp = int(((pred == 1) & (y_test == 1)).sum())
        rappel = tp / int((y_test == 1).sum())
        acc = float((pred == y_test).mean())
        ax.set_title(f"{name}\nexactitude {acc:.2f} — rappel {rappel:.2f}")
        ax.set_xlabel("prédiction")
        ax.set_ylabel("réalité")
        ax.grid(False)
        print(f"  {name}: accuracy {acc:.3f}, recall {rappel:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "matrices-confusion.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    copies()
    slide(33, (0.0, 0.33, 1.0, 0.81), None, "taux-apprentissage.png")
    slide(35, (0.0, 0.0, 1.0, 1.0), (0.9, 0.9, 1.0, 1.0), "boucle-entrainement.png")
    frontieres()
    arret_precoce()
    matrices()
    for p in sorted(OUT.iterdir()):
        print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
