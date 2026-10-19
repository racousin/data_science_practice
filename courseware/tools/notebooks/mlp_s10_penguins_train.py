"""Session 10, Step 1 — train the penguin classifier in Colab and save it.

The lesson `s11-docker-and-deployment/train-and-save-a-model.md` links here.
The notebook is the first step of the deployment workshop: look at the data,
fit a scaler + logistic-regression pipeline, check it, and download the
`model.joblib` file that Step 2's API loads. It ends by printing the Python and
package versions, which the student pins in their uv project: the model file
was written by these versions and is read back by them.

The reference project `labs/s11-docker-and-deployment/penguin-api/` holds the
`model.joblib` this notebook produced, with the same seed and split.

Writes one file and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/mlp_s10_penguins_train.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = (REPO / "website" / "public" / "modules"
       / "ms2a-machine-learning-practice" / "challenges")
NAME = "mlp-s10-penguins-train.ipynb"
COLAB = (f"https://colab.research.google.com/github/{nb.GITHUB}/blob/"
         f"{nb.BRANCH}/website/public/modules/ms2a-machine-learning-practice/"
         f"challenges/{NAME}")

md, code = nb.md, nb.code


def build() -> dict:
    cells = [
        md("# Step 1 — Train and save a penguin classifier",
           "",
           f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
           "",
           "The first step of the deployment workshop. This notebook trains a "
           "model and saves it to **one file**, `model.joblib`. Step 2 loads "
           "that file in an API on your laptop.",
           "",
           "Run every cell from top to bottom (*Runtime → Run all*)."),
        md("## 1. The data",
           "",
           "The **Palmer penguins**: 344 penguins of three species, measured on "
           "three islands of Antarctica between 2007 and 2009 (CC0, "
           "[allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins))."),
        code("import pandas as pd",
             "",
             'URL = ("https://raw.githubusercontent.com/allisonhorst/"',
             '       "palmerpenguins/main/inst/extdata/penguins.csv")',
             "df = pd.read_csv(URL)",
             "df.head()"),
        md("The model uses the four numeric measurements. Two penguins have "
           "none of them: those rows are dropped."),
        code('FEATURES = ["bill_length_mm", "bill_depth_mm",',
             '            "flipper_length_mm", "body_mass_g"]',
             "df = df.dropna(subset=FEATURES)",
             'print(len(df), "penguins")',
             'df["species"].value_counts()'),
        code("df[FEATURES].describe().round(1)"),
        md("Two measurements already separate the three species well:"),
        code("import matplotlib.pyplot as plt",
             "",
             'for species, g in df.groupby("species"):',
             '    plt.scatter(g["flipper_length_mm"], g["bill_length_mm"], label=species, s=15)',
             'plt.xlabel("flipper length (mm)"); plt.ylabel("bill length (mm)")',
             "plt.legend(); plt.show()"),
        md("## 2. Train",
           "",
           "A quarter of the penguins is kept aside to test the model. "
           "`stratify` keeps the same species mix in both parts."),
        code("from sklearn.model_selection import train_test_split",
             "",
             'X, y = df[FEATURES], df["species"]',
             "X_train, X_test, y_train, y_test = train_test_split(",
             "    X, y, test_size=0.25, stratify=y, random_state=0)",
             "len(X_train), len(X_test)"),
        md("The **baseline** always answers the most frequent species. "
           "A model is useful only above it."),
        code('print("baseline:", (y_test == y_train.mode()[0]).mean().round(4))'),
        md("The model is a **pipeline**: a scaler, then a logistic regression. "
           "The scaler is saved inside the model file, so the API can send raw "
           "millimetres and grams."),
        code("from sklearn.linear_model import LogisticRegression",
             "from sklearn.pipeline import make_pipeline",
             "from sklearn.preprocessing import StandardScaler",
             "",
             "model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))",
             "model.fit(X_train, y_train)",
             'print("test accuracy:", round(model.score(X_test, y_test), 4))'),
        code("from sklearn.metrics import ConfusionMatrixDisplay",
             "",
             "ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)",
             "plt.show()"),
        md("## 3. Three penguins to check",
           "",
           "The first penguin of each species in the file. The API of Step 2 "
           "must give the same three answers."),
        code("birds = pd.DataFrame([",
             "    [39.1, 18.7, 181, 3750],   # Adelie",
             "    [46.5, 17.9, 192, 3500],   # Chinstrap",
             "    [46.1, 13.2, 211, 4500],   # Gentoo",
             "], columns=FEATURES)",
             "model.predict(birds)"),
        md("## 4. Save the model",
           "",
           "`joblib.dump` writes the fitted pipeline to a file. The file "
           "contains the scaler's means, the regression's weights and the "
           "class names, but not the code of scikit-learn itself."),
        code("import joblib",
             "",
             'joblib.dump(model, "model.joblib")',
             "",
             "reloaded = joblib.load(\"model.joblib\")",
             "reloaded.predict(birds)"),
        md("## 5. The versions that wrote the file",
           "",
           "`model.joblib` is read back correctly by the **same** scikit-learn "
           "version. Note these numbers: Step 2 pins them in your project."),
        code("import sys, sklearn",
             "",
             'print("python      ", sys.version.split()[0])',
             'print("scikit-learn", sklearn.__version__)',
             'print("joblib      ", joblib.__version__)',
             'print("pandas      ", pd.__version__)'),
        md("## 6. Download the file",
           "",
           "In Colab, the cell below downloads `model.joblib` to your "
           "Downloads folder. Elsewhere, the file is already next to the "
           "notebook."),
        code("try:",
             "    from google.colab import files",
             '    files.download("model.joblib")',
             "except ImportError:",
             '    print("not in Colab: model.joblib is in", __import__("os").getcwd())'),
        md("**Checkpoint:** `model.joblib` (a few kB) is in your Downloads "
           "folder. Next: Step 2, an API that loads it."),
    ]
    return nb.notebook(cells)


def main() -> None:
    nb._COUNTER = itertools.count()     # byte-identical on every regeneration
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}\n      {COLAB}")


if __name__ == "__main__":
    main()
