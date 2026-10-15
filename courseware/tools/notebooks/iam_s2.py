"""IA pour la médecine, séance 2 — Préparer les données, forêts aléatoires.

Writes two notebooks and their corrections from one source each:

    iam-s2-tp.ipynb / iam-s2-tp-correction.ipynb
        French translation of github.com/racousin/DFGSM2-IA-S2 session2/tp1.ipynb
        (noisy breast cancer: imputation, IQR clipping, one-hot, scaling,
        logistic regression baseline, random forest, GridSearchCV).
        Fixes against the source: the confusion-matrix figure has two axes,
        the stray empty-figure cell is gone, exercise 9.0 gets a correction,
        and the final cross-validation scales inside a pipeline instead of
        scaling the whole table first. The table is still imputed and clipped
        before the split, as in the source; a note says so and the défi does
        it in the right order.
    iam-s2-defi.ipynb / iam-s2-defi-correction.ipynb
        Session 2's challenge on SUPPORT2 (alive / dead at 2 months):
        IQR clipping with bounds from the training rows, a ColumnTransformer
        (median imputation + missing indicators, most-frequent + one-hot),
        a random forest, a small GridSearchCV, an honest held-out validation,
        then submission.csv on the 1541 test patients.

Writes four files and nothing else (never run `build_notebooks.py`):

    uv run python courseware/tools/notebooks/iam_s2.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live)

COURSE = "ia-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
TP, TP_CORR = "iam-s2-tp.ipynb", "iam-s2-tp-correction.ipynb"
DEFI, DEFI_CORR = "iam-s2-defi.ipynb", "iam-s2-defi-correction.ipynb"
CHALLENGE = "https://ml-arena.com/viewchallenge/200"

md, code = nb.md, nb.code


def badge(name: str) -> str:
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({aim_repo.colab(COURSE, name)})")


# =========================================================================== TP
def build_tp(solution: bool) -> dict:
    """The TP; `solution` swaps every skeleton for its answer."""

    def ex(todo: list[str], answer: list[str]) -> dict:
        return code(*(answer if solution else todo))

    def reponse(text: list[str]) -> dict:
        return md(*(["**Réponse :** " + text[0], *text[1:]] if solution
                    else ["*Votre réponse ici*"]))

    name = TP_CORR if solution else TP
    suffix = " (correction)" if solution else ""
    url = aim_repo.raw(COURSE, "donnees/breast_cancer_noisy.csv")
    cells: list[dict] = [
        md(
            f"# TP 2 — Préparation des données et forêt aléatoire (cancer du sein bruité){suffix}",
            "",
            badge(name),
            "",
            "## Objectif",
            "",
            "En séance 1, le jeu de données était **propre**. Les données médicales",
            "réelles le sont rarement : elles contiennent des **valeurs manquantes**,",
            "des **valeurs aberrantes** et des **variables catégorielles** qu'il faut",
            "préparer.",
            "",
            "Dans ce TP :",
            "",
            "1. on travaille sur une version **bruitée** du jeu de données du cancer du",
            "   sein (valeurs manquantes, valeurs aberrantes, variables catégorielles) ;",
            "2. on applique la **préparation des données** : imputation → valeurs",
            "   aberrantes → encodage → mise à l'échelle ;",
            "3. on entraîne une **forêt aléatoire** et on la compare à la régression",
            "   logistique.",
            "",
            "La méthode reste celle de la séance 1 : **explorer → visualiser → préparer",
            "→ entraîner → évaluer → valider**. Cette séance porte sur l'étape",
            "**préparer**, sautée en séance 1 parce que les données étaient propres.",
            "",
            "**Comment travailler** : Fichier → *Enregistrer une copie dans Drive* ;",
            "exécutez les cellules dans l'ordre (Maj + Entrée) ; complétez les cellules",
            "marquées `# TODO` en remplaçant les `...`.",
        ),
        md("## 1. Mise en place"),
        code(
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.impute import SimpleImputer",
            "from sklearn.pipeline import make_pipeline",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.ensemble import RandomForestClassifier",
            "from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report",
        ),
    ]

    # -------------------------------------------------------- 2. charger
    cells += [
        md(
            "## 2. Charger et découvrir les données",
            "",
            "En séance 1, le jeu de données *Breast Cancer Wisconsin* était **parfaitement",
            "propre** : aucune valeur manquante, aucune valeur aberrante, que des",
            "variables numériques. C'est rare en pratique.",
            "",
            "Vous recevez cette fois une version plus proche de **vraies données",
            "cliniques** :",
            "",
            "- des valeurs **manquent** (panne d'appareil, dossier incomplet) ;",
            "- des mesures sont **aberrantes** (erreur de saisie, cas extrême) ;",
            "- des **variables catégorielles** ont été ajoutées (tranche d'âge, statut",
            "  ménopausique, localisation de la tumeur).",
            "",
            "Premier travail : **repérer et comprendre** ces défauts avant tout modèle.",
        ),
        md(
            "### Exercice 1.1 — Charger le jeu de données",
            "",
            "Chargez le fichier CSV, puis définissez les noms des classes et les listes",
            "de colonnes numériques et catégorielles.",
            "",
            "**Pourquoi :** on commence toujours par regarder ses données.",
            "",
            "*Indice :*",
            "```python",
            f"df = pd.read_csv('{url}')",
            "noms_classes = np.array(['maligne', 'bénigne'])   # cible : 0 = maligne, 1 = bénigne",
            "colonnes_num = df.select_dtypes(include='number').columns.drop('target').tolist()",
            "colonnes_cat = df.select_dtypes(exclude='number').columns.tolist()",
            "```",
        ),
        ex(
            [
                "# TODO : chargez les données et définissez les trois variables",
                "df = ...",
                "noms_classes = ...",
                "colonnes_num = ...",
                "colonnes_cat = ...",
            ],
            [
                f"df = pd.read_csv('{url}')",
                "",
                "# Même cible qu'en séance 1 : 0 = maligne, 1 = bénigne",
                "noms_classes = np.array(['maligne', 'bénigne'])",
                "",
                "colonnes_num = df.select_dtypes(include='number').columns.drop('target').tolist()",
                "colonnes_cat = df.select_dtypes(exclude='number').columns.tolist()",
                "",
                "print(f\"Dimensions : {df.shape}\")",
                "print(f\"Variables numériques : {len(colonnes_num)}\")",
                "print(f\"Variables catégorielles : {colonnes_cat}\")",
                "df.head()",
            ],
        ),
        md(
            "### Exercice 1.2 — Inspecter la qualité des données",
            "",
            "Utilisez `.info()` pour voir les types et les valeurs manquantes, et",
            "`.describe()` pour vérifier les plages de valeurs.",
            "",
            "**Pourquoi :** repérer les trois types de problèmes : valeurs manquantes",
            "(`NaN`), valeurs aberrantes (extrêmes) et variables non numériques.",
            "",
            "**Question :** combien de valeurs manquent ? Quelles variables ont des",
            "valeurs suspectes ? Quelles colonnes sont catégorielles ?",
        ),
        ex(
            ["# TODO : df.info() puis df.describe()", "..."],
            ["df.info()", "df.describe()"],
        ),
        reponse([
            "866 valeurs manquent, réparties sur les variables numériques (environ",
            "5 % des cellules). `mean radius` (maximum 176 pour une médiane de 13,5) et",
            "`mean area` (maximum 7 400 pour une médiane de 560) ont des valeurs",
            "aberrantes. `age_group`, `menopause_status` et `tumor_location` sont",
            "catégorielles (du texte).",
        ]),
    ]

    # -------------------------------------------------------- 3. visualiser
    cells += [
        md(
            "## 3. Visualiser les défauts",
            "",
            "Avant de corriger, il faut **voir** les problèmes.",
        ),
        md(
            "### Exercice 2.1 — Carte des valeurs manquantes",
            "",
            "Visualisez où se trouvent les valeurs manquantes.",
            "",
            "**Pourquoi :** voir si elles sont dispersées au hasard ou concentrées sur",
            "certaines variables.",
            "",
            "*Indice :* `sns.heatmap(df[colonnes_num].isnull(), cbar=True, yticklabels=False)`",
            "ou `df[colonnes_num].isnull().sum().plot(kind='bar')`",
        ),
        ex(
            ["# TODO : carte des valeurs manquantes et/ou nombre par variable", "..."],
            [
                "fig, axes = plt.subplots(1, 2, figsize=(16, 5))",
                "",
                "sns.heatmap(df[colonnes_num].isnull(), cbar=True, yticklabels=False,",
                "            ax=axes[0], cmap='viridis')",
                "axes[0].set_title('Carte des valeurs manquantes')",
                "",
                "df[colonnes_num].isnull().sum().plot(kind='bar', ax=axes[1], color='coral')",
                "axes[1].set_title('Valeurs manquantes par variable')",
                "axes[1].set_ylabel('Nombre')",
                "",
                "plt.tight_layout()",
                "plt.show()",
                "",
                "print(f\"Total des valeurs manquantes : {df[colonnes_num].isnull().sum().sum()}\")",
            ],
        ),
        md(
            "### Exercice 2.2 — Boîtes à moustaches pour repérer les valeurs aberrantes",
            "",
            "Tracez une boîte à moustaches de `mean radius` et une de `mean area`.",
            "",
            "**Pourquoi :** une boîte à moustaches montre la médiane, les quartiles et",
            "les points au-delà de 1,5 × IQR. Les valeurs extrêmes sautent aux yeux.",
            "",
            "*Indice :* `fig, axes = plt.subplots(1, 2, figsize=(12, 5))`,",
            "`sns.boxplot(y=df['mean radius'], ax=axes[0])`",
        ),
        ex(
            ["# TODO : deux boîtes à moustaches côte à côte", "..."],
            [
                "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
                "sns.boxplot(y=df['mean radius'], ax=axes[0], color='lightblue')",
                "axes[0].set_title('mean radius (avec valeurs aberrantes)')",
                "sns.boxplot(y=df['mean area'], ax=axes[1], color='lightgreen')",
                "axes[1].set_title('mean area (avec valeurs aberrantes)')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        md(
            "### Exercice 2.3 — Répartition des variables catégorielles",
            "",
            "Tracez un diagramme en barres pour `age_group`, `menopause_status` et",
            "`tumor_location`.",
            "",
            "**Pourquoi :** connaître la répartition des modalités et repérer une",
            "modalité très rare.",
            "",
            "*Indice :* `df['age_group'].value_counts().plot(kind='bar')`",
        ),
        ex(
            ["# TODO : un diagramme en barres par variable catégorielle", "..."],
            [
                "fig, axes = plt.subplots(1, 3, figsize=(15, 4))",
                "for ax, col in zip(axes, colonnes_cat):",
                "    df[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')",
                "    ax.set_title(col)",
                "axes[0].set_ylabel('Nombre')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
    ]

    # -------------------------------------------------------- 4. manquantes
    cells += [
        md(
            "## 4. Traiter les valeurs manquantes",
            "",
            "La plupart des modèles refusent les `NaN` : il faut les traiter avant",
            "l'entraînement.",
            "",
            "| Stratégie | Quand l'utiliser |",
            "|---|---|",
            "| **Supprimer les lignes** | très peu de valeurs manquantes, beaucoup de données |",
            "| **Imputer par la moyenne ou la médiane** | variables numériques |",
            "| **Imputer par la modalité la plus fréquente** | variables catégorielles |",
            "| **Imputer par un modèle** | cas complexes (avancé) |",
            "",
            "On utilise `SimpleImputer` de scikit-learn : la **médiane** pour les",
            "variables numériques (robuste aux valeurs aberrantes), la **modalité la",
            "plus fréquente** pour les variables catégorielles.",
        ),
        md(
            "### Exercice 3.1 — Compter les valeurs manquantes",
            "",
            "Comptez les valeurs manquantes par colonne, puis au total.",
            "",
            "*Indice :* `df.isnull().sum()`, `df.isnull().sum().sum()`",
        ),
        ex(
            ["# TODO : nombre de valeurs manquantes par colonne et au total", "..."],
            [
                "print(df.isnull().sum())",
                "print(f\"\\nTotal : {df.isnull().sum().sum()}\")",
            ],
        ),
        md(
            "### Exercice 3.2 — Imputer les valeurs manquantes",
            "",
            "Utilisez `SimpleImputer` avec `strategy='median'` pour les variables",
            "numériques et `strategy='most_frequent'` pour les catégorielles.",
            "",
            "**Pourquoi :** la médiane résiste mieux que la moyenne aux valeurs",
            "aberrantes, et nos données en contiennent.",
            "",
            "*Indice :*",
            "```python",
            "imputeur_num = SimpleImputer(strategy='median')",
            "df[colonnes_num] = imputeur_num.fit_transform(df[colonnes_num])",
            "```",
            "",
            "La cellule suivante vérifie qu'il ne reste aucune valeur manquante.",
        ),
        ex(
            [
                "# TODO : imputez les colonnes numériques (médiane) et catégorielles",
                "# (modalité la plus fréquente)",
                "...",
            ],
            [
                "imputeur_num = SimpleImputer(strategy='median')",
                "df[colonnes_num] = imputeur_num.fit_transform(df[colonnes_num])",
                "",
                "imputeur_cat = SimpleImputer(strategy='most_frequent')",
                "df[colonnes_cat] = imputeur_cat.fit_transform(df[colonnes_cat])",
            ],
        ),
        code(
            "assert df.isnull().sum().sum() == 0, \"il reste des valeurs manquantes\"",
            "print(\"✅ Plus aucune valeur manquante\")",
        ),
    ]

    # -------------------------------------------------------- 5. aberrantes
    cells += [
        md(
            "## 5. Traiter les valeurs aberrantes",
            "",
            "Les valeurs extrêmes peuvent fausser l'entraînement. On les repère et on",
            "les **écrête** avec la **méthode de l'IQR** :",
            "",
            "- **IQR** = Q3 − Q1 (écart interquartile) ;",
            "- **borne basse** = Q1 − 1,5 × IQR ;",
            "- **borne haute** = Q3 + 1,5 × IQR ;",
            "- une valeur hors des bornes est remplacée par la borne la plus proche.",
            "",
            "**Pourquoi écrêter plutôt que supprimer ?** En médecine, on ne veut pas",
            "perdre de patients : l'écrêtage garde le patient tout en réduisant",
            "l'influence de la valeur extrême.",
        ),
        md(
            "### Exercice 4.1 — Repérer et écrêter les valeurs aberrantes",
            "",
            "Pour chaque variable numérique, écrêtez les valeurs hors des bornes IQR.",
            "Affichez le minimum et le maximum de `mean radius` et `mean area` avant et",
            "après.",
            "",
            "*Indice :*",
            "```python",
            "for col in colonnes_num:",
            "    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)",
            "    iqr = q3 - q1",
            "    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)",
            "```",
        ),
        ex(
            [
                "print(f\"Avant : mean radius max {df['mean radius'].max():.1f}, \"",
                "      f\"mean area max {df['mean area'].max():.1f}\")",
                "",
                "# TODO : écrêtez chaque colonne numérique aux bornes IQR",
                "...",
                "",
                "print(f\"Après : mean radius max {df['mean radius'].max():.1f}, \"",
                "      f\"mean area max {df['mean area'].max():.1f}\")",
            ],
            [
                "print(f\"Avant : mean radius max {df['mean radius'].max():.1f}, \"",
                "      f\"mean area max {df['mean area'].max():.1f}\")",
                "",
                "for col in colonnes_num:",
                "    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)",
                "    iqr = q3 - q1",
                "    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)",
                "",
                "print(f\"Après : mean radius max {df['mean radius'].max():.1f}, \"",
                "      f\"mean area max {df['mean area'].max():.1f}\")",
            ],
        ),
        md(
            "### Exercice 4.2 — Visualiser après l'écrêtage",
            "",
            "Refaites les boîtes à moustaches de `mean radius` et `mean area` et",
            "comparez avec l'exercice 2.2.",
        ),
        ex(
            ["# TODO : mêmes boîtes à moustaches qu'à l'exercice 2.2", "..."],
            [
                "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
                "sns.boxplot(y=df['mean radius'], ax=axes[0], color='lightblue')",
                "axes[0].set_title('mean radius (après écrêtage)')",
                "sns.boxplot(y=df['mean area'], ax=axes[1], color='lightgreen')",
                "axes[1].set_title('mean area (après écrêtage)')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
    ]

    # -------------------------------------------------------- 6. encodage
    cells += [
        md(
            "## 6. Encoder les variables catégorielles",
            "",
            "Les modèles calculent avec des **nombres**, pas avec du texte.",
            "",
            "**Encodage one-hot** : pour une variable sans ordre naturel, on crée **une",
            "colonne 0/1 par modalité**.",
            "",
            "| tumor_location | → | upper_inner | central | lower_inner |",
            "|---|---|---|---|---|",
            "| upper_inner | | 1 | 0 | 0 |",
            "| central | | 0 | 1 | 0 |",
            "| lower_inner | | 0 | 0 | 1 |",
            "",
            "**Pourquoi ne pas numéroter (upper_inner = 0, central = 1, lower_inner = 2) ?**",
            "Le modèle y lirait un **ordre** (upper_inner < central < lower_inner) qui",
            "n'a pas de sens ici.",
        ),
        md(
            "### Exercice 5.1 — Encodage one-hot",
            "",
            "Utilisez `pd.get_dummies()` pour encoder les colonnes catégorielles.",
            "",
            "*Indice :* `df = pd.get_dummies(df, columns=colonnes_cat, drop_first=True)`.",
            "`drop_first=True` retire une colonne redondante par variable : si toutes",
            "les autres valent 0, la modalité retirée est connue. (scikit-learn propose",
            "aussi `OneHotEncoder`, utilisé dans le défi.)",
            "",
            "**Question :** combien de variables explicatives a-t-on après l'encodage ?",
        ),
        ex(
            ["# TODO : encodez colonnes_cat", "df = ...", "df.head()"],
            [
                "df = pd.get_dummies(df, columns=colonnes_cat, drop_first=True)",
                "",
                "print(f\"Dimensions après encodage : {df.shape}\")",
                "print([c for c in df.columns if c not in colonnes_num and c != 'target'])",
                "df.head()",
            ],
        ),
        reponse([
            "30 variables numériques + 2 (age_group) + 1 (menopause_status) +",
            "4 (tumor_location) = 37 variables explicatives, plus la cible : 38 colonnes.",
        ]),
    ]

    # -------------------------------------------------------- 7. échelle
    cells += [
        md(
            "## 7. Mise à l'échelle",
            "",
            "Comme en séance 1, on standardise les variables (moyenne 0, écart-type 1),",
            "ce qui aide la régression logistique :",
            "",
            "$$x_{\\text{standardisé}} = \\frac{x - \\mu}{\\sigma}$$",
            "",
            "**Rappel :** `fit` sur l'apprentissage, `transform` sur l'apprentissage et",
            "sur le test.",
            "",
            "> **Remarque.** Pour rester simple, ce TP a imputé et écrêté **tout** le",
            "> tableau avant de séparer apprentissage et test : les médianes et les",
            "> bornes ont donc vu les patients du test, une petite **fuite",
            "> d'information**. La bonne méthode (séparer d'abord, puis ajuster chaque",
            "> étape sur l'apprentissage seul) est celle du défi.",
        ),
        md(
            "### Exercice 6.1 — Séparer et mettre à l'échelle",
            "",
            "Séparez les variables (`X`) de la cible (`y`), coupez en apprentissage et",
            "test (80/20, `random_state=42`), puis standardisez avec `StandardScaler`.",
            "",
            "*Indice :* comme en séance 1 ; nommez les résultats `X_app_s` et `X_test_s`.",
        ),
        ex(
            [
                "X = df.drop(columns='target')",
                "y = df['target']",
                "",
                "# TODO : séparation 80/20 puis standardisation",
                "X_app, X_test, y_app, y_test = ...",
                "X_app_s = ...",
                "X_test_s = ...",
            ],
            [
                "X = df.drop(columns='target')",
                "y = df['target']",
                "",
                "X_app, X_test, y_app, y_test = train_test_split(X, y, test_size=0.2,",
                "                                                random_state=42)",
                "scaler = StandardScaler()",
                "X_app_s = scaler.fit_transform(X_app)",
                "X_test_s = scaler.transform(X_test)",
                "",
                "print(f\"Apprentissage : {X_app.shape[0]} patientes, {X_app.shape[1]} variables\")",
                "print(f\"Test : {X_test.shape[0]} patientes\")",
            ],
        ),
    ]

    # -------------------------------------------------------- 8. baseline
    cells += [
        md(
            "## 8. Modèle de référence : la régression logistique",
            "",
            "Avant d'essayer un nouveau modèle, on établit toujours une **référence**",
            "avec un modèle connu : elle dit si le nouveau modèle fait vraiment mieux.",
        ),
        md(
            "### Exercice 7.1 — Régression logistique de référence",
            "",
            "Entraînez une `LogisticRegression(max_iter=5000)` sur les données",
            "standardisées. Affichez l'exactitude et le `classification_report`.",
        ),
        ex(
            [
                "# TODO : entraînez lr_model, prédisez y_pred_lr sur X_test_s",
                "lr_model = ...",
                "y_pred_lr = ...",
            ],
            [
                "lr_model = LogisticRegression(max_iter=5000, random_state=42)",
                "lr_model.fit(X_app_s, y_app)",
                "y_pred_lr = lr_model.predict(X_test_s)",
                "",
                "print(f\"Exactitude, régression logistique : {accuracy_score(y_test, y_pred_lr):.4f}\")",
                "print(classification_report(y_test, y_pred_lr, target_names=noms_classes))",
            ],
        ),
    ]

    # -------------------------------------------------------- 9. forêt
    cells += [
        md(
            "## 9. La forêt aléatoire",
            "",
            "Une **forêt aléatoire** est un **ensemble** de nombreux **arbres de",
            "décision**, chacun entraîné sur un tirage aléatoire des patients et des",
            "variables. La prédiction finale est le **vote majoritaire** des arbres.",
            "",
            "```",
            "             Données",
            "          /    |     \\",
            "   Arbre 1  Arbre 2  ...  Arbre N",
            "      |        |            |",
            "   préd. 1  préd. 2  ...  préd. N",
            "          \\    |     /",
            "        vote majoritaire",
            "      → prédiction finale",
            "```",
            "",
            "| Propriété | Régression logistique | Forêt aléatoire |",
            "|---|---|---|",
            "| **Frontière de décision** | linéaire | non linéaire |",
            "| **Mise à l'échelle nécessaire** | oui | non |",
            "| **Valeurs aberrantes** | sensible | robuste |",
            "| **Interprétation** | coefficients | importances des variables |",
            "| **Risque de surapprentissage** | faible | moyen (réglable) |",
            "",
            "**Idée clé :** chaque arbre voit un tirage différent des données, donc fait",
            "des **erreurs différentes** ; le vote les compense. Comme plusieurs avis",
            "médicaux : le consensus est souvent plus fiable que chaque avis isolé.",
        ),
        md(
            "### Exercice 8.1 — Entraîner une forêt aléatoire",
            "",
            "Entraînez un `RandomForestClassifier(n_estimators=100, random_state=42)` et",
            "comparez son exactitude à celle de la régression logistique.",
            "",
            "**Remarque :** la forêt n'a pas besoin de données standardisées, mais on",
            "garde les mêmes données pour une comparaison équitable.",
        ),
        ex(
            [
                "# TODO : entraînez rf_model, prédisez y_pred_rf sur X_test_s",
                "rf_model = ...",
                "y_pred_rf = ...",
            ],
            [
                "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)",
                "rf_model.fit(X_app_s, y_app)",
                "y_pred_rf = rf_model.predict(X_test_s)",
                "",
                "print(f\"Régression logistique : {accuracy_score(y_test, y_pred_lr):.4f}\")",
                "print(f\"Forêt aléatoire :       {accuracy_score(y_test, y_pred_rf):.4f}\")",
                "print(classification_report(y_test, y_pred_rf, target_names=noms_classes))",
            ],
        ),
        md(
            "### Exercice 8.2 — Comparer les matrices de confusion",
            "",
            "Affichez côte à côte les matrices de confusion des deux modèles.",
            "",
            "**Pourquoi :** voir si les deux modèles font les mêmes types d'erreurs.",
            "",
            "*Indice :* `fig, axes = plt.subplots(1, 2, figsize=(12, 5))`,",
            "`ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=noms_classes, ax=axes[0])`",
        ),
        ex(
            ["# TODO : deux matrices de confusion côte à côte", "..."],
            [
                "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
                "ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=noms_classes,",
                "                                        ax=axes[0], cmap='Blues')",
                "axes[0].set_title('Régression logistique')",
                "ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=noms_classes,",
                "                                        ax=axes[1], cmap='Greens')",
                "axes[1].set_title('Forêt aléatoire')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        md(
            "### Exercice 8.3 — Importances des variables",
            "",
            "Tracez les 15 variables les plus importantes selon la forêt.",
            "",
            "**Pourquoi :** savoir quelles mesures guident le diagnostic du modèle.",
            "",
            "*Indice :*",
            "```python",
            "importances = rf_model.feature_importances_",
            "indices = np.argsort(importances)[-15:]",
            "plt.barh(range(len(indices)), importances[indices])",
            "plt.yticks(range(len(indices)), X.columns[indices])",
            "```",
            "",
            "**Question :** quelles variables sont les plus importantes ? Est-ce",
            "cohérent avec les corrélations vues en séance 1 ?",
        ),
        ex(
            ["# TODO : diagramme en barres horizontales des 15 plus grandes importances", "..."],
            [
                "importances = rf_model.feature_importances_",
                "indices = np.argsort(importances)[-15:]",
                "",
                "plt.figure(figsize=(10, 7))",
                "plt.barh(range(len(indices)), importances[indices], color='steelblue')",
                "plt.yticks(range(len(indices)), X.columns[indices])",
                "plt.xlabel('Importance')",
                "plt.title('Les 15 variables les plus importantes (forêt aléatoire)')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        reponse([
            "les mesures `worst` et `mean` du rayon, du périmètre, de l'aire et des",
            "points concaves dominent, comme les corrélations de la séance 1. Les",
            "variables catégorielles (âge, ménopause, localisation) ont une importance",
            "faible : elles ont été simulées au hasard.",
        ]),
    ]

    # -------------------------------------------------------- 10. hyperparamètres
    cells += [
        md(
            "## 10. Régler les hyperparamètres",
            "",
            "Les **hyperparamètres** sont des réglages choisis **avant**",
            "l'entraînement : le modèle ne les apprend pas des données.",
            "",
            "| Hyperparamètre | Ce qu'il règle | Trop petit | Trop grand |",
            "|---|---|---|---|",
            "| `n_estimators` | nombre d'arbres | prédictions instables | lent, gain négligeable |",
            "| `max_depth` | profondeur maximale d'un arbre | sous-apprentissage | surapprentissage |",
            "| `min_samples_split` | patients minimum pour couper un nœud | surapprentissage | sous-apprentissage |",
        ),
        md(
            "### Exercice 9.0 — Essayer des hyperparamètres",
            "",
            "Entraînez quelques `RandomForestClassifier(n_estimators=..., max_depth=...,",
            "min_samples_split=..., random_state=42)` avec des valeurs de votre choix et",
            "comparez leur exactitude **sur l'apprentissage** et **sur le test**.",
        ),
        ex(
            [
                "# TODO : essayez plusieurs réglages et affichez les deux exactitudes",
                "...",
            ],
            [
                "for profondeur in [2, 5, None]:",
                "    for min_split in [2, 20]:",
                "        rf = RandomForestClassifier(n_estimators=100, max_depth=profondeur,",
                "                                    min_samples_split=min_split, random_state=42)",
                "        rf.fit(X_app_s, y_app)",
                "        print(f\"max_depth={profondeur}, min_samples_split={min_split} : \"",
                "              f\"apprentissage {rf.score(X_app_s, y_app):.3f}, \"",
                "              f\"test {rf.score(X_test_s, y_test):.3f}\")",
            ],
        ),
        md(
            "Choisir le réglage d'après le score **de test** revient à régler le modèle",
            "sur le test. `GridSearchCV` compare les réglages par **validation croisée**",
            "sur l'apprentissage seul, et laisse le test intact.",
        ),
        md(
            "### Exercice 9.1 — Recherche en grille",
            "",
            "`GridSearchCV` essaie toutes les combinaisons et garde la meilleure en",
            "validation croisée :",
            "",
            "```",
            "n_estimators : [50, 100, 200] × max_depth : [3, 5, 10, None] × min_samples_split : [2, 5]",
            "= 3 × 4 × 2 = 24 combinaisons, chacune en validation croisée à 5 plis = 120 entraînements",
            "```",
            "",
            "Lancez la recherche et affichez les meilleurs hyperparamètres et le meilleur",
            "score.",
            "",
            "*Indice :*",
            "```python",
            "grille = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None],",
            "          'min_samples_split': [2, 5]}",
            "recherche = GridSearchCV(RandomForestClassifier(random_state=42), grille,",
            "                         cv=5, scoring='accuracy', n_jobs=-1)",
            "recherche.fit(X_app_s, y_app)",
            "```",
        ),
        ex(
            [
                "# TODO : définissez grille et recherche, puis lancez la recherche",
                "grille = ...",
                "recherche = ...",
            ],
            [
                "grille = {",
                "    'n_estimators': [50, 100, 200],",
                "    'max_depth': [3, 5, 10, None],",
                "    'min_samples_split': [2, 5],",
                "}",
                "recherche = GridSearchCV(RandomForestClassifier(random_state=42), grille,",
                "                         cv=5, scoring='accuracy', n_jobs=-1)",
                "recherche.fit(X_app_s, y_app)",
                "",
                "print(f\"Meilleurs hyperparamètres : {recherche.best_params_}\")",
                "print(f\"Meilleure exactitude en validation croisée : {recherche.best_score_:.4f}\")",
            ],
        ),
        md(
            "### Exercice 9.2 — Évaluer le meilleur modèle sur le test",
            "",
            "Prédisez le test avec le meilleur modèle et affichez l'exactitude et le",
            "`classification_report`.",
            "",
            "*Indice :* `recherche.best_estimator_` est le meilleur modèle, déjà",
            "réentraîné sur tout l'apprentissage.",
            "",
            "**Question :** le réglage a-t-il amélioré la forêt par défaut ?",
        ),
        ex(
            ["# TODO : prédictions y_pred_best du meilleur modèle", "y_pred_best = ..."],
            [
                "y_pred_best = recherche.best_estimator_.predict(X_test_s)",
                "",
                "print(f\"Forêt par défaut : {accuracy_score(y_test, y_pred_rf):.4f}\")",
                "print(f\"Forêt réglée :     {accuracy_score(y_test, y_pred_best):.4f}\")",
                "print(classification_report(y_test, y_pred_best, target_names=noms_classes))",
            ],
        ),
        reponse([
            "le gain est nul ou faible : la forêt par défaut est déjà bonne sur ce jeu.",
            "Sur 114 patientes de test, une erreur de plus ou de moins change",
            "l'exactitude de 0,009 : l'écart observé est du même ordre. L'intérêt de",
            "`GridSearchCV` est une recherche systématique, évaluée en validation",
            "croisée sans toucher au test.",
        ]),
    ]

    # -------------------------------------------------------- 11. CV
    cells += [
        md("## 11. Validation croisée et comparaison finale"),
        md(
            "### Exercice 10.1 — Comparer les deux modèles en validation croisée",
            "",
            "Lancez une validation croisée à 5 plis pour la régression logistique et",
            "pour la meilleure forêt, puis affichez un tableau comparatif.",
            "",
            "La standardisation est mise dans un **pipeline** avec la régression",
            "logistique : elle est ainsi réajustée sur les 4 plis d'entraînement à",
            "chaque tour, sans voir le pli de validation.",
            "",
            "*Indice :*",
            "```python",
            "lr_scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),",
            "                            X, y, cv=5, scoring='accuracy')",
            "rf_scores = cross_val_score(recherche.best_estimator_, X, y, cv=5, scoring='accuracy')",
            "```",
        ),
        ex(
            [
                "# TODO : lr_scores et rf_scores, puis le tableau",
                "lr_scores = ...",
                "rf_scores = ...",
            ],
            [
                "lr_scores = cross_val_score(make_pipeline(StandardScaler(),",
                "                                          LogisticRegression(max_iter=5000)),",
                "                            X, y, cv=5, scoring='accuracy')",
                "rf_scores = cross_val_score(recherche.best_estimator_, X, y, cv=5,",
                "                            scoring='accuracy')",
                "",
                "print(f\"{'Modèle':<25} {'Exactitude':>12} {'Écart-type':>12}\")",
                "print(f\"{'Régression logistique':<25} {lr_scores.mean():>12.4f} {lr_scores.std():>12.4f}\")",
                "print(f\"{'Forêt aléatoire (réglée)':<25} {rf_scores.mean():>12.4f} {rf_scores.std():>12.4f}\")",
            ],
        ),
        md(
            "## 12. Résumé",
            "",
            "| Étape | Technique | Pourquoi |",
            "|---|---|---|",
            "| **Valeurs manquantes** | `SimpleImputer` (médiane) | combler les `NaN` sans perdre de patients |",
            "| **Valeurs aberrantes** | écrêtage par l'IQR | limiter les extrêmes sans perdre de patients |",
            "| **Variables catégorielles** | encodage one-hot (`get_dummies`) | transformer le texte en nombres |",
            "| **Mise à l'échelle** | `StandardScaler` | mettre les variables sur la même échelle |",
            "| **Référence** | régression logistique | le score à battre |",
            "| **Nouveau modèle** | forêt aléatoire | un ensemble d'arbres de décision |",
            "| **Réglage** | `GridSearchCV` | choisir les hyperparamètres en validation croisée |",
            "",
            "```",
            "Données brutes → manquantes → aberrantes → encodage → mise à l'échelle → modèle",
            "```",
            "",
            "**À retenir :** la préparation des données compte souvent plus que le choix",
            "du modèle. Un modèle simple sur des données bien préparées bat souvent un",
            "modèle complexe sur des données sales.",
            "",
            "**Séance 3 :** les réseaux de neurones (`MLPClassifier`), un nouveau jeu de",
            "données (maladie cardiaque) et la comparaison des trois modèles.",
        ),
    ]
    return nb.notebook(cells)


# ========================================================================= DÉFI
def build_defi(solution: bool) -> dict:
    """The challenge notebook; `solution` swaps every skeleton for its answer."""

    def ex(todo: list[str], answer: list[str]) -> dict:
        return code(*(answer if solution else todo))

    name = DEFI_CORR if solution else DEFI
    suffix = " (correction)" if solution else ""
    base = aim_repo.raw(COURSE, "donnees/support2/")
    cells: list[dict] = [
        md(
            f"# Défi 2 — Survie à 2 mois avec une forêt aléatoire{suffix}",
            "",
            badge(name),
            "",
            "Patients de réanimation de l'étude **SUPPORT2** : prédire s'ils sont",
            "vivants (`alive`) ou décédés (`dead`) **2 mois** après l'admission, à partir",
            "de 28 variables mesurées à l'admission. Le score est l'**exactitude** ;",
            "prédire `alive` pour tout le monde donne déjà environ 0,59.",
            "",
            "Méthode de la séance 2, dans le bon ordre :",
            "",
            "1. séparer apprentissage et **validation** ;",
            "2. écrêter les valeurs aberrantes (bornes IQR de l'apprentissage) ;",
            "3. préparer : imputation (+ indicateurs de valeur manquante), one-hot ;",
            "4. entraîner une **forêt aléatoire**, puis régler quelques",
            "   hyperparamètres par **validation croisée** ;",
            "5. mesurer **une fois** sur la validation, puis entraîner sur tout",
            "   l'apprentissage et écrire `submission.csv`.",
            "",
            f"Soumission : [ml-arena.com]({CHALLENGE}), page du défi 2.",
        ),
        code(
            "import pandas as pd",
            "from sklearn.model_selection import train_test_split, GridSearchCV",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.impute import SimpleImputer",
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
            "from sklearn.pipeline import Pipeline, make_pipeline",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.ensemble import RandomForestClassifier",
            "from sklearn.metrics import accuracy_score",
            "",
            f"URL = '{base}'",
            "X = pd.read_csv(URL + 'X_train.csv', index_col='patient_id')",
            "y = pd.read_csv(URL + 'y_train_survie.csv', index_col='patient_id')['outcome']",
            "X_test = pd.read_csv(URL + 'X_test.csv', index_col='patient_id')",
            "assert (X.index == y.index).all()",
            "",
            "colonnes_num = X.select_dtypes(include='number').columns.tolist()",
            "colonnes_cat = X.select_dtypes(exclude='number').columns.tolist()",
            "print(f\"Apprentissage : {X.shape}, test : {X_test.shape}\")",
            "print(f\"{len(colonnes_num)} variables numériques, catégorielles : {colonnes_cat}\")",
            "print(y.value_counts(normalize=True).round(3))",
        ),
        md(
            "Les valeurs manquantes sont nombreuses : jusqu'à 61 % pour `adlp`",
            "(autonomie déclarée par le patient).",
        ),
        code("X.isna().mean().sort_values(ascending=False).head(10).round(2)"),
        md(
            "## 1. Séparer apprentissage et validation",
            "",
            "On met de côté 20 % des patients pour la validation, **avant** toute",
            "préparation. `stratify=y` garde la même proportion de décès des deux",
            "côtés.",
        ),
        code(
            "X_app, X_val, y_app, y_val = train_test_split(X, y, test_size=0.2,",
            "                                              random_state=42, stratify=y)",
            "print(X_app.shape, X_val.shape)",
        ),
        md(
            "## 2. Écrêter les valeurs aberrantes",
            "",
            "Complétez `bornes_iqr` : elle renvoie, pour chaque colonne numérique, les",
            "bornes $Q_1 - 1.5\\,\\mathrm{IQR}$ et $Q_3 + 1.5\\,\\mathrm{IQR}$ calculées",
            "sur le tableau reçu. `ecreter` applique ces bornes à n'importe quel tableau",
            "(`clip(..., axis=1)` aligne les bornes sur les colonnes). Les bornes se",
            "calculent sur **l'apprentissage seul**.",
        ),
        ex(
            [
                "def bornes_iqr(tableau):",
                "    # TODO : q1, q3 = quantiles 0.25 et 0.75 de tableau[colonnes_num]",
                "    q1 = ...",
                "    q3 = ...",
                "    iqr = q3 - q1",
                "    return q1 - 1.5 * iqr, q3 + 1.5 * iqr",
                "",
                "",
                "def ecreter(tableau, bas, haut):",
                "    tableau = tableau.copy()",
                "    tableau[colonnes_num] = tableau[colonnes_num].clip(bas, haut, axis=1)",
                "    return tableau",
                "",
                "",
                "bas, haut = bornes_iqr(X_app)",
                "X_app_e, X_val_e = ecreter(X_app, bas, haut), ecreter(X_val, bas, haut)",
            ],
            [
                "def bornes_iqr(tableau):",
                "    q1 = tableau[colonnes_num].quantile(0.25)",
                "    q3 = tableau[colonnes_num].quantile(0.75)",
                "    iqr = q3 - q1",
                "    return q1 - 1.5 * iqr, q3 + 1.5 * iqr",
                "",
                "",
                "def ecreter(tableau, bas, haut):",
                "    tableau = tableau.copy()",
                "    tableau[colonnes_num] = tableau[colonnes_num].clip(bas, haut, axis=1)",
                "    return tableau",
                "",
                "",
                "bas, haut = bornes_iqr(X_app)",
                "X_app_e, X_val_e = ecreter(X_app, bas, haut), ecreter(X_val, bas, haut)",
            ],
        ),
        code(
            "assert (X_app_e['crea'].max() <= haut['crea']) and X_app_e['crea'].isna().sum() == X_app['crea'].isna().sum()",
            "print(f\"crea : max {X_app['crea'].max():.1f} → {X_app_e['crea'].max():.1f} mg/dL\")",
            "print(\"✅ Écrêtage correct\")",
        ),
        md(
            "## 3. La préparation : imputation et one-hot",
            "",
            "Un `ColumnTransformer` applique une préparation par groupe de colonnes :",
            "",
            "- numériques : `SimpleImputer(strategy='median', add_indicator=True)` ;",
            "  l'indicateur ajoute une colonne 0/1 par variable incomplète, car",
            "  l'absence d'une valeur est une information ;",
            "- catégorielles : `SimpleImputer(strategy='most_frequent')` puis",
            "  `OneHotEncoder(handle_unknown='ignore')`.",
            "",
            "Il n'est pas encore ajusté : il le sera dans le pipeline, sur",
            "l'apprentissage seul.",
        ),
        ex(
            [
                "# TODO : complétez les deux préparations",
                "preparation = ColumnTransformer([",
                "    ('num', ..., colonnes_num),",
                "    ('cat', make_pipeline(..., ...), colonnes_cat),",
                "])",
            ],
            [
                "preparation = ColumnTransformer([",
                "    ('num', SimpleImputer(strategy='median', add_indicator=True), colonnes_num),",
                "    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),",
                "                          OneHotEncoder(handle_unknown='ignore')), colonnes_cat),",
                "])",
            ],
        ),
        md(
            "## 4. Une forêt aléatoire",
            "",
            "Le `Pipeline` enchaîne la préparation et la forêt : `fit` ajuste les deux",
            "sur l'apprentissage, `predict` applique la même préparation à de nouveaux",
            "patients. Entraînez-le sur `X_app_e` et mesurez l'exactitude sur `X_val_e`.",
        ),
        ex(
            [
                "foret = Pipeline([",
                "    ('prep', preparation),",
                "    ('foret', RandomForestClassifier(n_estimators=200, random_state=42)),",
                "])",
                "# TODO : entraînez foret, puis calculez son exactitude de validation",
                "...",
            ],
            [
                "foret = Pipeline([",
                "    ('prep', preparation),",
                "    ('foret', RandomForestClassifier(n_estimators=200, random_state=42)),",
                "])",
                "foret.fit(X_app_e, y_app)",
                "print(f\"Forêt par défaut, validation : {foret.score(X_val_e, y_val):.3f}\")",
            ],
        ),
        md(
            "Deux points de comparaison : toujours prédire `alive`, et le modèle du",
            "défi 1 (régression logistique standardisée, ici avec la même préparation).",
        ),
        code(
            "print(f\"Toujours 'alive' : {(y_val == 'alive').mean():.3f}\")",
            "logistique = Pipeline([('prep', preparation), ('echelle', StandardScaler()),",
            "                       ('logistique', LogisticRegression(max_iter=2000))])",
            "logistique.fit(X_app_e, y_app)",
            "print(f\"Régression logistique : {logistique.score(X_val_e, y_val):.3f}\")",
        ),
        md(
            "## 5. Régler quelques hyperparamètres",
            "",
            "Une petite grille, évaluée en validation croisée à 5 plis **sur",
            "l'apprentissage seul**. Dans un pipeline, un hyperparamètre se nomme",
            "`<étape>__<paramètre>` : `foret__max_depth`. 3 × 2 × 2 = 12 combinaisons,",
            "60 entraînements (environ une minute sur Colab).",
        ),
        ex(
            [
                "grille = {",
                "    'foret__max_depth': [None, 8, 16],",
                "    'foret__min_samples_leaf': [1, 5],",
                "    'foret__max_features': ['sqrt', 0.3],",
                "}",
                "# TODO : GridSearchCV(foret, grille, cv=5, scoring='accuracy', n_jobs=-1), puis fit",
                "recherche = ...",
            ],
            [
                "grille = {",
                "    'foret__max_depth': [None, 8, 16],",
                "    'foret__min_samples_leaf': [1, 5],",
                "    'foret__max_features': ['sqrt', 0.3],",
                "}",
                "recherche = GridSearchCV(foret, grille, cv=5, scoring='accuracy', n_jobs=-1)",
                "recherche.fit(X_app_e, y_app)",
                "print(f\"Meilleurs hyperparamètres : {recherche.best_params_}\")",
                "print(f\"Validation croisée : {recherche.best_score_:.3f}\")",
            ],
        ),
        md(
            "Mesure **unique** sur la validation, mise de côté depuis le début :",
            "c'est l'estimation honnête du score à attendre sur ML-Arena.",
        ),
        ex(
            [
                "# TODO : exactitude de recherche.best_estimator_ sur X_val_e",
                "...",
            ],
            [
                "acc_val = recherche.best_estimator_.score(X_val_e, y_val)",
                "print(f\"Forêt réglée, validation : {acc_val:.3f}\")",
            ],
        ),
        md(
            "## 6. Le modèle final et la soumission",
            "",
            "Les hyperparamètres sont choisis : on réentraîne le pipeline sur **tous**",
            "les patients d'apprentissage (bornes IQR recalculées sur `X`), on prédit",
            "les 1541 patients du test, et on écrit `submission.csv` avec les colonnes",
            "`patient_id,outcome`.",
        ),
        ex(
            [
                "bas, haut = bornes_iqr(X)",
                "modele_final = recherche.best_estimator_",
                "# TODO : entraînez modele_final sur ecreter(X, bas, haut) et y,",
                "# puis prédisez ecreter(X_test, bas, haut)",
                "predictions = ...",
                "",
                "soumission = pd.DataFrame({'patient_id': X_test.index, 'outcome': predictions})",
                "soumission.to_csv('submission.csv', index=False)",
            ],
            [
                "bas, haut = bornes_iqr(X)",
                "modele_final = recherche.best_estimator_",
                "modele_final.fit(ecreter(X, bas, haut), y)",
                "predictions = modele_final.predict(ecreter(X_test, bas, haut))",
                "",
                "soumission = pd.DataFrame({'patient_id': X_test.index, 'outcome': predictions})",
                "soumission.to_csv('submission.csv', index=False)",
            ],
        ),
        code(
            "verif = pd.read_csv('submission.csv')",
            "assert list(verif.columns) == ['patient_id', 'outcome'], verif.columns",
            "assert len(verif) == 1541 and set(verif['patient_id']) == set(X_test.index)",
            "assert set(verif['outcome']) <= {'alive', 'dead'}",
            "print(verif['outcome'].value_counts())",
            "print(\"✅ submission.csv est prêt\")",
        ),
        md(
            "Téléchargez le fichier (sur Colab, la cellule suivante l'enregistre dans",
            "vos Téléchargements), puis déposez-le sur la page du défi :",
            f"[{CHALLENGE}]({CHALLENGE}).",
        ),
        code(
            "try:",
            "    from google.colab import files",
            "except ImportError:",
            "    print(\"Hors de Colab : submission.csv est dans le dossier du notebook.\")",
            "else:",
            "    files.download('submission.csv')",
        ),
    ]
    return nb.notebook(cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for builder, names in ((build_tp, (TP, TP_CORR)), (build_defi, (DEFI, DEFI_CORR))):
        for solution, name in zip((False, True), names):
            nb._COUNTER = itertools.count()
            with open(OUT / name, "w", encoding="utf-8") as fh:
                json.dump(builder(solution), fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print("wrote", OUT / name)


if __name__ == "__main__":
    main()
