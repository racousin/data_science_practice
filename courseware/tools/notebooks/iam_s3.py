"""Séance 3 of *IA pour la médecine* — réseaux de neurones. The six notebooks.

Builds, from one source each, a student notebook and its correction:

- `iam-s3-tp.ipynb` (+ `-correction`): French translation of session3/tp1.ipynb
  of github.com/racousin/DFGSM2-IA-S2 — heart disease (Statlog), logistic
  regression, random forest, MLPClassifier, loss curve, early stopping,
  architectures, grid search, cross-validated comparison. The data is the
  course's own `donnees/maladie_cardiaque.csv` (written by `iam_s3_donnees.py`,
  French column names) instead of `fetch_openml`.
- `iam-s3-bonus.ipynb` (+ `-correction`): French translation of
  session3/tp_bonus.ipynb — gradient descent step by step, the learning rate,
  linear regression in scikit-learn. The source's helper package (`l2math`,
  pip-installed from GitHub) is replaced by a few plotting functions defined
  in the notebook.
- `iam-s3-defi.ipynb` (+ `-correction`): the Session 3 challenge — SUPPORT2,
  `outcome` alive/dead at 2 months: an MLP with early stopping, compared with a
  logistic regression and a random forest on the same held-out split, then
  `submission.csv` (patient_id,outcome; 1541 rows).

Student cells carry `# TODO` skeletons; the correction has the solution in
their place. Writes six files and nothing else — unlike `build_notebooks.py`,
running this script is safe:

    uv run python courseware/tools/notebooks/iam_s3.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import textwrap
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_medicine)

COURSE = "ia-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
HEART = aim_repo.raw(COURSE, "donnees/maladie_cardiaque.csv")
SUPPORT2 = aim_repo.raw(COURSE, "donnees/support2/")
CHALLENGE = "https://ml-arena.com/viewchallenge/201"


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines()


def m(text: str) -> dict:
    return nb.md(*_lines(text))


def c(text: str) -> dict:
    return nb.code(*_lines(text))


class Builder:
    """Collects the cells of one notebook, student or correction."""

    def __init__(self, correction: bool) -> None:
        self.correction = correction
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append(m(text))

    def code(self, text: str) -> None:
        self.cells.append(c(text))

    def ex(self, todo: str, solution: str) -> None:
        """An exercise cell: the skeleton for students, the solution otherwise."""
        self.cells.append(c(solution) if self.correction else c(todo))

    def answer(self, text: str) -> None:
        """A written answer: the expected answer in the correction."""
        if self.correction:
            self.md("**Réponse.** " + textwrap.dedent(text).strip())
        else:
            self.md("*Votre réponse ici.*")

    def title(self, name: str, heading: str) -> None:
        suffix = " (correction)" if self.correction else ""
        full = name + ("-correction" if self.correction else "") + ".ipynb"
        self.md(f"# {heading}{suffix}\n\n"
                f"[![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, full)})")


# ====================================================================== TP 3
def tp(b: Builder) -> None:
    b.title("iam-s3-tp", "TP 3 — Maladie cardiaque : réseaux de neurones et comparaison de modèles")
    b.md("""
    *IA pour la médecine — séance 3.*

    ## Objectif

    Ce dernier TP réunit tout ce qui a été vu aux séances 1 et 2 :

    1. appliquer la **démarche de préparation des données** (séance 2) à un nouveau jeu de données ;
    2. entraîner et comparer **trois modèles** : régression logistique, forêt aléatoire et
       **réseau de neurones** (perceptron multicouche, *MLP*) ;
    3. comprendre comment un réseau de neurones apprend et comment le régler.

    ### Le jeu de données : maladie cardiaque (Statlog)

    270 patients, 13 variables cliniques. La tâche : prédire la **présence ou l'absence
    d'une maladie cardiaque**. Certaines variables sont vraiment numériques (âge, pression
    artérielle), d'autres sont des entiers qui codent des catégories (type de douleur
    thoracique, thalassémie) : c'est fréquent dans les données médicales réelles.

    ### La démarche

    Toujours la même : **explorer → visualiser → préparer → entraîner → évaluer → comparer**.

    **Comment travailler** : exécutez les cellules dans l'ordre (Maj + Entrée) et complétez
    les cellules marquées `# TODO`.
    """)
    b.md("---\n## 1. Préparation de l'environnement")
    b.code("""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, recall_score, ConfusionMatrixDisplay,
                                 classification_report)

    import warnings
    warnings.filterwarnings("ignore")  # masque les avertissements de convergence du MLP
    """)
    b.md("""
    ---
    ## 2. Charger et découvrir les données

    **13 variables :**

    | Variable | Signification |
    |---|---|
    | `age` | âge (années) |
    | `sexe` | 1 = homme, 0 = femme |
    | `type_douleur_thoracique` | 1 angor typique, 2 angor atypique, 3 douleur non angineuse, 4 asymptomatique |
    | `pression_arterielle_repos` | pression artérielle au repos (mmHg) |
    | `cholesterol` | cholestérol sérique (mg/dL) |
    | `glycemie_jeun_sup_120` | glycémie à jeun > 120 mg/dL (1 = oui) |
    | `ecg_repos` | ECG au repos : 0 normal, 1 anomalie ST-T, 2 hypertrophie ventriculaire gauche |
    | `frequence_cardiaque_max` | fréquence cardiaque maximale atteinte à l'effort |
    | `angor_effort` | angor provoqué par l'effort (1 = oui) |
    | `depression_st` | sous-décalage du segment ST à l'effort (mm) |
    | `pente_st` | pente du segment ST à l'effort : 1 ascendante, 2 plate, 3 descendante |
    | `nb_vaisseaux_colores` | nombre de gros vaisseaux visibles en fluoroscopie (0 à 3) |
    | `thalassemie` | scintigraphie au thallium : 3 normal, 6 défaut fixe, 7 défaut réversible |

    **Cible :** `maladie_cardiaque`, 1 = présente, 0 = absente.

    *Source : Statlog (Heart), UCI Machine Learning Repository, CC BY 4.0.*
    """)
    b.md("""
    ### Exercice 1.1 — Charger les données

    Lisez le fichier avec `pd.read_csv(URL)` dans `df`, affichez sa taille et ses
    premières lignes.
    """)
    b.ex(f"""
    URL = "{HEART}"
    # TODO : df = ...
    """, f"""
    URL = "{HEART}"
    df = pd.read_csv(URL)
    print("Taille :", df.shape)
    df.head()
    """)
    b.md("""
    ### Exercice 1.2 — Statistiques et types

    Affichez les types (`df.info()`) et les statistiques descriptives (`df.describe()`).

    **Question :** quelles variables représentent des catégories, bien qu'elles soient
    stockées sous forme de nombres ?
    """)
    b.ex("# TODO", """
    df.info()
    df.describe()
    """)
    b.answer("""
    `sexe`, `type_douleur_thoracique`, `glycemie_jeun_sup_120`, `ecg_repos`,
    `angor_effort`, `pente_st`, `thalassemie` sont des codes de catégories
    (`nb_vaisseaux_colores` est un comptage, ordonné). Leur moyenne n'a pas de sens
    clinique : « type de douleur moyen = 3,2 » ne veut rien dire.
    """)
    b.md("""
    ### Exercice 1.3 — Équilibre des classes

    Comptez les patients malades et non malades.

    **Pourquoi :** comme à la séance 1, pour savoir si l'exactitude est une métrique fiable.

    *Indice :* `y.value_counts(normalize=True)`
    """)
    b.ex("""
    y = df["maladie_cardiaque"]
    # TODO
    """, """
    y = df["maladie_cardiaque"]
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    """)
    b.md("---\n## 3. Visualiser les données")
    b.md("""
    ### Exercice 2.1 — Distributions selon la classe

    Tracez la distribution de `age` et de `frequence_cardiaque_max`, colorée par classe.

    *Indice :* `sns.histplot(data=df, x="age", hue="maladie_cardiaque", kde=True)`
    """)
    b.ex("# TODO", """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="age", hue="maladie_cardiaque", kde=True, ax=axes[0], bins=20)
    axes[0].set_title("Âge selon la classe")
    sns.histplot(data=df, x="frequence_cardiaque_max", hue="maladie_cardiaque", kde=True,
                 ax=axes[1], bins=20)
    axes[1].set_title("Fréquence cardiaque maximale selon la classe")
    plt.tight_layout()
    plt.show()
    """)
    b.md("""
    ### Exercice 2.2 — Nuage de points

    Tracez `age` en fonction de `frequence_cardiaque_max`, coloré par classe : les deux
    variables ensemble séparent-elles les classes ?

    *Indice :* `sns.scatterplot(data=df, x="age", y="frequence_cardiaque_max", hue="maladie_cardiaque", alpha=0.6)`
    """)
    b.ex("# TODO", """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="age", y="frequence_cardiaque_max", hue="maladie_cardiaque",
                    alpha=0.6)
    plt.title("Âge et fréquence cardiaque maximale")
    plt.show()
    """)
    b.md("""
    ### Exercice 2.3 — Carte des corrélations

    Affichez la matrice des corrélations de toutes les variables.

    *Indice :* `sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)`

    **Question :** quelles variables semblent les plus liées à la maladie ?
    """)
    b.ex("# TODO", """
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Corrélations")
    plt.tight_layout()
    plt.show()

    print(df.corr()["maladie_cardiaque"].drop("maladie_cardiaque").abs()
          .sort_values(ascending=False))
    """)
    b.answer("""
    `thalassemie`, `nb_vaisseaux_colores`, `type_douleur_thoracique`, `angor_effort`,
    `depression_st` et `frequence_cardiaque_max` (corrélation négative : une fréquence
    maximale basse va avec la maladie) sont les plus liées à la cible. Une corrélation ne
    mesure qu'un lien linéaire, et sur des codes de catégories elle n'est qu'indicative.
    """)
    b.md("""
    ---
    ## 4. Préparer les données

    Ce jeu n'a pas de valeurs manquantes et toutes les variables sont déjà numériques :
    il suffit de **standardiser**. Pour les arbres, l'échelle ne compte pas ; pour la
    régression logistique et surtout pour les réseaux de neurones, elle est importante.

    ### Exercice 3.1 — Découper et standardiser

    Séparez `X` (toutes les colonnes sauf la cible) et `y`, découpez 80 % / 20 %
    (`random_state=42`), puis standardisez avec un `StandardScaler` **ajusté sur
    l'entraînement seulement**.
    """)
    b.ex("""
    # TODO : X = ..., y = ...
    # TODO : X_train, X_test, y_train, y_test = train_test_split(...)
    # TODO : scaler = StandardScaler() ; X_train_scaled = ... ; X_test_scaled = ...
    """, """
    X = df.drop(columns="maladie_cardiaque")
    y = df["maladie_cardiaque"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Entraînement : {X_train.shape[0]} patients, {X_train.shape[1]} variables")
    print(f"Test : {X_test.shape[0]} patients")
    """)
    b.md("""
    ---
    ## 5. Modèle de référence : la régression logistique

    On commence toujours par un modèle de référence simple.

    ### Exercice 4.1 — Régression logistique

    Entraînez `LogisticRegression(max_iter=5000, random_state=42)` et affichez
    l'exactitude, le rapport de classification et la matrice de confusion.
    """)
    b.ex("""
    # TODO : lr_model = ... ; lr_model.fit(...)
    # TODO : y_pred_lr = ...
    """, """
    lr_model = LogisticRegression(max_iter=5000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    print(f"Exactitude régression logistique : {accuracy_score(y_test, y_pred_lr):.4f}")
    print(classification_report(y_test, y_pred_lr))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, cmap="Blues")
    plt.title("Régression logistique")
    plt.show()
    """)
    b.md("""
    ---
    ## 6. Forêt aléatoire

    ### Exercice 5.1 — Forêt aléatoire

    Entraînez `RandomForestClassifier(n_estimators=100, random_state=42)` et comparez-la
    à la régression logistique. Calculez aussi son exactitude **sur l'entraînement** :
    que constatez-vous ?
    """)
    b.ex("""
    # TODO : rf_model = ... ; y_pred_rf = ...
    """, """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    print(f"Régression logistique : {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Forêt aléatoire       : {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Forêt, entraînement   : {accuracy_score(y_train, rf_model.predict(X_train_scaled)):.4f}")
    print(classification_report(y_test, y_pred_rf))
    """)
    b.answer("""
    La forêt classe parfaitement les patients d'entraînement (1,00) mais fait moins bien
    que la régression logistique sur le test : elle sur-apprend. Avec 54 patients de
    test, une différence de quelques patients change l'exactitude de plusieurs points :
    on y revient avec la validation croisée (partie 9).
    """)
    b.md("""
    ---
    ## 7. Réseau de neurones (`MLPClassifier`)

    Un **réseau de neurones** transforme l'entrée couche après couche. Chaque neurone
    calcule une somme pondérée de ses entrées plus un biais, puis applique une
    **fonction d'activation** non linéaire (ReLU par défaut). Les poids, comme les
    coefficients θ de la régression logistique, sont **appris** par descente de gradient.

    | | Régression logistique | Réseau de neurones (MLP) |
    |---|---|---|
    | **Couches** | une (entrée → sortie) | plusieurs (entrée → cachées → sortie) |
    | **Frontière de décision** | linéaire | non linéaire |
    | **Paramètres** | peu (nb de variables + 1) | beaucoup (selon l'architecture) |
    | **Entraînement** | rapide, un seul minimum | plus lent, plusieurs minimums possibles |
    | **Standardisation** | importante | **indispensable** |

    Hyperparamètres principaux de `MLPClassifier` (*Multi-Layer Perceptron*) :

    - `hidden_layer_sizes` : `(100,)` = une couche cachée de 100 neurones, `(64, 32)` = deux couches ;
    - `activation` : `"relu"` (défaut), `"tanh"`, `"logistic"` ;
    - `learning_rate_init` : taux d'apprentissage initial (0,001 par défaut) ;
    - `max_iter` : nombre maximal d'époques (passages sur les données).
    """)
    b.md("""
    ### Exercice 6.1 — Un premier réseau

    Entraînez `MLPClassifier(max_iter=1000, random_state=42)` et comparez-le aux deux
    modèles précédents.
    """)
    b.ex("""
    # TODO : mlp_model = ... ; y_pred_mlp = ...
    """, """
    mlp_model = MLPClassifier(max_iter=1000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)

    print(f"Régression logistique : {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Forêt aléatoire       : {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Réseau de neurones    : {accuracy_score(y_test, y_pred_mlp):.4f}")
    print(f"Réseau, entraînement  : {accuracy_score(y_train, mlp_model.predict(X_train_scaled)):.4f}")
    print(classification_report(y_test, y_pred_mlp))
    """)
    b.md("""
    ### Exercice 6.2 — La courbe de perte

    Tracez la perte d'entraînement à chaque époque : elle dit si le réseau a convergé
    (perte qui diminue puis se stabilise).

    *Indice :* `plt.plot(mlp_model.loss_curve_)`, et `mlp_model.n_iter_` donne le nombre d'époques.
    """)
    b.ex("# TODO", """
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_model.loss_curve_)
    plt.xlabel("Époque")
    plt.ylabel("Perte (entraînement)")
    plt.title("Courbe de perte du MLP")
    plt.show()
    print(f"Perte finale : {mlp_model.loss_curve_[-1]:.4f} après {mlp_model.n_iter_} époques")
    """)
    b.md("""
    ### Exercice 6.3 — L'arrêt précoce

    Entraîné trop longtemps, un réseau sur-apprend : la perte d'entraînement continue de
    baisser alors que le modèle se dégrade sur de nouveaux patients. **L'arrêt précoce**
    met de côté une partie de l'entraînement comme validation et arrête l'apprentissage
    quand le score de validation ne s'améliore plus :

    - `early_stopping=True` active l'arrêt précoce ;
    - `validation_fraction=0.1` : 10 % de l'entraînement sert de validation ;
    - `n_iter_no_change=10` : arrêt après 10 époques sans amélioration.

    Entraînez `mlp_es` avec l'arrêt précoce, comparez son nombre d'époques, son
    exactitude et sa courbe de perte à celles de `mlp_model`. `mlp_es.validation_scores_`
    contient l'exactitude de validation à chaque époque.
    """)
    b.ex("""
    # TODO : mlp_es = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
    """, """
    mlp_es = MLPClassifier(max_iter=1000, random_state=42,
                           early_stopping=True, validation_fraction=0.1)
    mlp_es.fit(X_train_scaled, y_train)
    y_pred_es = mlp_es.predict(X_test_scaled)

    print(f"Sans arrêt précoce : {mlp_model.n_iter_} époques, "
          f"test {accuracy_score(y_test, y_pred_mlp):.4f}, "
          f"rappel {recall_score(y_test, y_pred_mlp):.2f}")
    print(f"Avec arrêt précoce : {mlp_es.n_iter_} époques, "
          f"test {accuracy_score(y_test, y_pred_es):.4f}, "
          f"rappel {recall_score(y_test, y_pred_es):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(mlp_model.loss_curve_, label="sans arrêt précoce")
    axes[0].plot(mlp_es.loss_curve_, label="avec arrêt précoce")
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Perte (entraînement)"); axes[0].legend()
    axes[1].plot(mlp_es.validation_scores_)
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Exactitude de validation")
    plt.tight_layout()
    plt.show()
    """)
    b.answer("""
    Sans arrêt précoce, le réseau tourne des centaines d'époques et atteint une
    exactitude d'entraînement proche de 1 : il apprend par cœur. Avec l'arrêt précoce il
    s'arrête après quelques dizaines d'époques. Sur ces 54 patients de test l'exactitude
    est voisine, mais le rappel (malades détectés) est meilleur. Attention : la
    validation ne compte ici qu'une vingtaine de patients, donc le point d'arrêt est
    lui-même bruité.
    """)
    b.md("""
    ---
    ## 8. Régler le réseau

    L'architecture (nombre et taille des couches cachées) change le résultat :

    | Architecture | Description |
    |---|---|
    | `(50,)` | 1 couche, 50 neurones |
    | `(100,)` | 1 couche, 100 neurones (défaut) |
    | `(64, 32)` | 2 couches (64 → 32) |
    | `(128, 64, 32)` | 3 couches : beaucoup de paramètres, il faut beaucoup de données |

    **Règle pratique :** commencer simple, n'ajouter de la complexité que si elle aide.
    Plus de couches ne veut pas dire meilleur, surtout avec 270 patients.

    ### Exercice 7.1 — Comparer des architectures

    Pour chaque architecture, entraînez un MLP, affichez l'exactitude de test et tracez
    les courbes de perte sur un même graphique.
    """)
    b.ex("""
    architectures = [(50,), (100,), (64, 32), (128, 64, 32)]
    for arch in architectures:
        # TODO : model = MLPClassifier(hidden_layer_sizes=arch, max_iter=1000, random_state=42)
        # TODO : entraîner, afficher l'exactitude, plt.plot(model.loss_curve_, label=str(arch))
        pass
    """, """
    architectures = [(50,), (100,), (64, 32), (128, 64, 32)]
    plt.figure(figsize=(10, 6))
    for arch in architectures:
        model = MLPClassifier(hidden_layer_sizes=arch, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{str(arch):<15} exactitude : {acc:.4f}  ({model.n_iter_} époques)")
        plt.plot(model.loss_curve_, label=f"{arch} (exactitude {acc:.3f})")
    plt.xlabel("Époque")
    plt.ylabel("Perte (entraînement)")
    plt.title("Courbes de perte selon l'architecture")
    plt.legend()
    plt.show()
    """)
    b.md("""
    ### Exercice 7.2 — Recherche sur grille

    Avec `GridSearchCV` (validation croisée à 5 plis sur l'entraînement), cherchez la
    meilleure combinaison de `hidden_layer_sizes`, `activation` et `learning_rate_init`,
    puis évaluez le meilleur modèle sur le test.

    ```python
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (64, 32)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
    }
    ```
    """)
    b.ex("""
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (64, 32)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
    }
    # TODO : grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42),
    #                                       param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    # TODO : grid_search_mlp.fit(...) ; afficher best_params_ et best_score_
    """, """
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (64, 32)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
    }
    grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42),
                                   param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search_mlp.fit(X_train_scaled, y_train)

    print("Meilleurs paramètres :", grid_search_mlp.best_params_)
    print(f"Meilleure exactitude (validation croisée) : {grid_search_mlp.best_score_:.4f}")
    y_pred_best_mlp = grid_search_mlp.best_estimator_.predict(X_test_scaled)
    print(f"Exactitude de test : {accuracy_score(y_test, y_pred_best_mlp):.4f}")
    """)
    b.md("""
    ---
    ## 9. Comparaison finale

    ### Exercice 8.1 — Comparaison par validation croisée

    Un seul découpage de 54 patients de test est trop petit pour départager les modèles.
    Comparez-les par validation croisée à 5 plis sur **toutes** les données. Pour que la
    standardisation soit refaite dans chaque pli (sans fuite d'information du pli de
    test), on la met dans un **pipeline** : `make_pipeline(StandardScaler(), modèle)`.
    """)
    b.ex("""
    models = {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=100, random_state=42),
        # TODO : "Réseau de neurones": make_pipeline(StandardScaler(), grid_search_mlp.best_estimator_),
    }
    for name, model in models.items():
        # TODO : scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        # TODO : afficher la moyenne et l'écart type
        pass
    """, """
    models = {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=100, random_state=42),
        "Réseau de neurones": make_pipeline(StandardScaler(), grid_search_mlp.best_estimator_),
    }
    print(f"{'Modèle':<25} {'Exactitude':>11} {'Écart type':>11}")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"{name:<25} {scores.mean():>11.4f} {scores.std():>11.4f}")
    """)
    b.md("""
    ### Exercice 8.2 — Matrices de confusion côte à côte

    Affichez les matrices de confusion des trois modèles sur le jeu de test.

    *Indice :* `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`

    **Question :** pour dépister une maladie cardiaque, quelle case de la matrice
    regardez-vous en premier ?
    """)
    b.ex("# TODO", """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    predictions = {
        "Régression logistique": y_pred_lr,
        "Forêt aléatoire": y_pred_rf,
        "Réseau de neurones": y_pred_best_mlp,
    }
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues",
                                                colorbar=False)
        ax.set_title(f"{name}\\nrappel {recall_score(y_test, y_pred):.2f}")
    plt.tight_layout()
    plt.show()
    """)
    b.answer("""
    La case en bas à gauche : les **faux négatifs**, des malades prédits sains. Un
    dépistage qui les manque est dangereux ; on regarde donc le **rappel** de la classe
    « maladie présente » autant que l'exactitude.
    """)
    b.md("""
    ---
    ## 10. Bilan des trois séances

    | | Régression logistique | Forêt aléatoire | Réseau de neurones (MLP) |
    |---|---|---|---|
    | **Type** | linéaire | ensemble d'arbres | couches de neurones |
    | **Frontière** | linéaire | non linéaire | non linéaire |
    | **Standardisation** | importante | inutile | indispensable |
    | **Interprétabilité** | forte (coefficients) | moyenne (importance des variables) | faible (boîte noire) |
    | **Quand l'utiliser** | référence, petits jeux | données tabulaires, défaut robuste | motifs complexes, beaucoup de données |

    **La démarche (pour tous les modèles)** : explorer, visualiser, préparer, découper,
    entraîner (en commençant par une référence), évaluer avec la bonne métrique,
    régler par validation croisée, comparer.

    > En santé, **le choix de la métrique compte plus que le choix du modèle**. Un
    > modèle exact à 95 % mais qui manque des malades est dangereux : un diagnostic
    > manqué (faux négatif) coûte en général bien plus qu'une fausse alerte.
    """)


# ================================================================ TP bonus
HELPERS = """
# Fonctions d'affichage du TP : exécutez la cellule, inutile de la lire.
def plot_gradient_step(theta, theta_new, grad, learning_rate, f, theta_range):
    t = np.linspace(*theta_range, 200)
    plt.figure(figsize=(8, 4))
    plt.plot(t, f(t), "b-", lw=2, label="f(θ)")
    plt.plot(t, f(theta) + grad * (t - theta), "g--", alpha=0.6, label="tangente (pente = gradient)")
    plt.scatter([theta], [f(theta)], c="green", s=80, zorder=5, label=f"θ = {theta:.2f}")
    plt.scatter([theta_new], [f(theta_new)], c="red", s=80, zorder=5, label=f"θ nouveau = {theta_new:.2f}")
    plt.annotate("", xy=(theta_new, f(theta_new)), xytext=(theta, f(theta)),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    plt.ylim(min(f(t)) - 5, max(f(t)) * 1.05)
    plt.xlabel("θ"); plt.ylabel("f(θ)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Un pas de descente de gradient (taux d'apprentissage = {learning_rate})")


def plot_gradient_descent_1d(f, theta_history, theta_range):
    t = np.linspace(*theta_range, 200)
    h = np.array(theta_history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(t, f(t), "b-", lw=2)
    axes[0].plot(h, f(h), "ro-", ms=4, alpha=0.7)
    axes[0].scatter([h[0]], [f(h[0])], c="green", s=80, zorder=5, label="départ")
    axes[0].scatter([h[-1]], [f(h[-1])], c="red", marker="*", s=200, zorder=5, label="fin")
    axes[0].set_xlabel("θ"); axes[0].set_ylabel("f(θ)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Trajectoire de θ")
    axes[1].plot(f(h), "o-")
    axes[1].set_xlabel("itération"); axes[1].set_ylabel("f(θ)"); axes[1].grid(alpha=0.3)
    axes[1].set_title("Valeur de la fonction à chaque itération")
    plt.tight_layout()


def plot_predictions(X, y, y_pred, title):
    order = np.argsort(X[:, 0])
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], y, alpha=0.6, label="données")
    plt.plot(X[order, 0], y_pred[order], "r-", lw=2, label="prédiction")
    plt.xlabel("X"); plt.ylabel("y"); plt.title(title); plt.legend(); plt.grid(alpha=0.3)


def generate_linear_data(n_samples=100, n_features=1, noise=1.0, bias=5.0,
                         weight_scale=3.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    true_weights = rng.normal(0, weight_scale, size=n_features)
    y = X @ true_weights + bias + rng.normal(0, noise, size=n_samples)
    return X, y, true_weights, bias
"""


def bonus(b: Builder) -> None:
    b.title("iam-s3-bonus", "TP bonus — La descente de gradient pas à pas")
    b.md("""
    *IA pour la médecine — séance 3, TP facultatif.*

    **Objectifs**

    1. comprendre la descente de gradient sur une fonction d'une variable ;
    2. voir l'effet du **taux d'apprentissage** ;
    3. ajuster une régression linéaire avec scikit-learn et retrouver les vrais paramètres.

    C'est l'algorithme qui entraîne la régression logistique et les réseaux de
    neurones : ici on le regarde sur l'exemple le plus simple possible.
    """)
    b.md("## Préparation")
    b.code("""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    """)
    b.code(HELPERS)
    b.md("""
    ---
    # Partie 1 — Comprendre la descente de gradient

    **But :** trouver la valeur de $\\theta$ qui minimise une fonction $f(\\theta)$.

    La descente de gradient est itérative : à chaque pas, on se déplace dans le sens
    opposé à la pente.

    $$\\theta_{\\text{nouveau}} = \\theta_{\\text{ancien}} - \\eta \\cdot \\nabla f(\\theta_{\\text{ancien}})$$

    - $\\theta$ : le paramètre à optimiser ;
    - $\\eta$ (êta) : le **taux d'apprentissage** (la taille du pas) ;
    - $\\nabla f(\\theta)$ : le **gradient**, ici simplement la dérivée de $f$.

    ## Une fonction d'exemple

    On minimise $f(\\theta) = (3\\theta - 7)^2$.

    **Question :** où est le minimum de cette fonction ?
    """)
    b.code("""
    def f(theta):
        \"\"\"Fonction à minimiser : (3θ - 7)²\"\"\"
        return (3 * theta - 7) ** 2

    def gradient_f(theta):
        \"\"\"Dérivée de f : 2 × (3θ - 7) × 3 = 6 × (3θ - 7)\"\"\"
        return 6 * (3 * theta - 7)

    theta_values = np.linspace(-2, 5, 100)
    plt.figure(figsize=(8, 4))
    plt.plot(theta_values, f(theta_values), "b-", linewidth=2)
    plt.xlabel("θ"); plt.ylabel("f(θ)"); plt.title("f(θ) = (3θ - 7)²")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Minimum exact : θ = 7/3 ≈ {7/3:.4f}")
    """)
    b.md("## Un pas de descente de gradient, en détail")
    b.code("""
    theta = 0.0
    learning_rate = 0.05

    grad = gradient_f(theta)
    print(f"θ actuel       = {theta}")
    print(f"f(θ) actuel    = {f(theta)}")
    print(f"gradient en θ  = {grad}")
    print(f"θ nouveau = θ - taux × gradient = {theta} - {learning_rate} × {grad} = {theta - learning_rate * grad}")

    theta_new = theta - learning_rate * grad
    plot_gradient_step(theta, theta_new, grad, learning_rate, f, theta_range=(-2, 5))
    plt.show()
    """)
    b.md("## La descente complète\n\nOn répète le pas un nombre fixé de fois :")
    b.code("""
    def gradient_descent(f, gradient_f, theta_init, learning_rate, n_iterations):
        \"\"\"Descente de gradient ; renvoie la liste des θ successifs.\"\"\"
        theta = theta_init
        theta_history = [theta]
        for i in range(n_iterations):
            grad = gradient_f(theta)
            theta = theta - learning_rate * grad
            theta_history.append(theta)
        return theta_history

    theta_history = gradient_descent(f, gradient_f, theta_init=0.0,
                                     learning_rate=0.05, n_iterations=20)
    print(f"θ de départ : {theta_history[0]:.4f}")
    print(f"θ final     : {theta_history[-1]:.4f}")
    print(f"vrai minimum : {7/3:.4f}")
    plot_gradient_descent_1d(f, theta_history, theta_range=(-2, 5))
    plt.show()
    """)
    b.md("""
    ---
    ## Exercice 1 — Le taux d'apprentissage

    Essayez plusieurs taux d'apprentissage : 0,01 (petit), 0,05 (moyen), 0,1 (grand),
    0,12 (trop grand ?).

    **Questions :**
    1. Que se passe-t-il avec un très petit taux ?
    2. Et avec un très grand taux ?
    3. Quel taux convient à ce problème ?
    """)
    b.ex("""
    learning_rate = 0.05  # TODO : modifiez cette valeur (0.01, 0.05, 0.1, 0.12)

    theta_history = gradient_descent(f, gradient_f, theta_init=0.0,
                                     learning_rate=learning_rate, n_iterations=20)
    print(f"Taux : {learning_rate} → θ final : {theta_history[-1]:.4f} (cible {7/3:.4f})")
    plot_gradient_descent_1d(f, theta_history, theta_range=(-2, 5))
    plt.show()
    """, """
    for learning_rate in [0.01, 0.05, 0.1, 0.12]:
        theta_history = gradient_descent(f, gradient_f, theta_init=0.0,
                                         learning_rate=learning_rate, n_iterations=20)
        print(f"Taux : {learning_rate} → θ final : {theta_history[-1]:.4f} (cible {7/3:.4f})")
        plot_gradient_descent_1d(f, theta_history, theta_range=(-3, 8))
        plt.show()
    """)
    b.answer("""
    1. Avec 0,01, θ avance lentement : après 20 pas il n'a pas atteint 7/3.
    2. Le pas vaut ici θ ← θ − η × 18 × (θ − 7/3) : au-delà de η = 1/9 ≈ 0,11, chaque pas
       dépasse le minimum de plus que la distance d'origine, θ oscille de plus en plus
       loin et la descente **diverge** (0,12). Entre 1/18 et 1/9 (0,1), θ oscille autour
       du minimum mais converge.
    3. Autour de 0,05 (exactement 1/18 ≈ 0,056 atteint le minimum en un pas). Le bon
       taux dépend de la courbure de la fonction : c'est un **hyperparamètre**.
    """)
    b.md("""
    ---
    ## Exercice 2 — Une autre fonction

    Minimisez $f_2(\\theta) = \\theta^2 + 5\\theta + 6$. Écrivez la fonction, sa dérivée,
    et le minimum exact (là où la dérivée s'annule), puis lancez la descente depuis
    $\\theta = 5$.
    """)
    b.ex("""
    def f2(theta):
        return ...  # TODO

    def gradient_f2(theta):
        return ...  # TODO : la dérivée de f2

    analytical_min = ...  # TODO : θ où la dérivée vaut 0

    theta_history = gradient_descent(f2, gradient_f2, theta_init=5.0,
                                     learning_rate=0.1, n_iterations=30)
    print(f"θ final : {theta_history[-1]:.4f} (minimum exact : {analytical_min})")
    plot_gradient_descent_1d(f2, theta_history, theta_range=(-6, 6))
    plt.show()
    """, """
    def f2(theta):
        return theta**2 + 5 * theta + 6

    def gradient_f2(theta):
        return 2 * theta + 5

    analytical_min = -5 / 2  # 2θ + 5 = 0

    theta_history = gradient_descent(f2, gradient_f2, theta_init=5.0,
                                     learning_rate=0.1, n_iterations=30)
    print(f"θ final : {theta_history[-1]:.4f} (minimum exact : {analytical_min})")
    plot_gradient_descent_1d(f2, theta_history, theta_range=(-6, 6))
    plt.show()
    """)
    b.md("""
    ---
    # Partie 2 — La régression linéaire avec scikit-learn

    Problème d'apprentissage réel : trouver un poids $w$ et un biais $b$ tels que
    $\\hat{y} = Xw + b$ minimise l'erreur quadratique moyenne (MSE) :

    $$\\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

    ## Un jeu de données simulé

    On fabrique des données dont on connaît les vrais paramètres, pour vérifier que le
    modèle les retrouve.
    """)
    b.code("""
    X, y, true_weights, true_bias = generate_linear_data(
        n_samples=100, n_features=1, noise=1.0, bias=5.0, weight_scale=3.0, seed=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.6)
    plt.xlabel("X"); plt.ylabel("y"); plt.title("Données simulées")
    plt.grid(True, alpha=0.3)
    plt.show()
    """)
    b.md("""
    ## Ajuster le modèle

    `LinearRegression` n'utilise pas la descente de gradient : pour la MSE, le minimum
    a une formule exacte (moindres carrés ordinaires),

    $$w = (X^T X)^{-1} X^T y$$

    obtenue en une seule fois, sans taux d'apprentissage ni nombre d'époques. La
    descente de gradient sert quand il n'existe pas de telle formule : régression
    logistique, réseaux de neurones.
    """)
    b.code("""
    model = LinearRegression()   # 1. créer : pas encore de paramètres
    model.fit(X, y)              # 2. ajuster : calcule coef_ et intercept_
    y_pred = model.predict(X)    # 3. prédire
    mse = mean_squared_error(y, y_pred)  # 4. évaluer

    print(f"Poids appris : {model.coef_} (vrai : {true_weights})")
    print(f"Biais appris : {model.intercept_:.4f} (vrai : {true_bias})")
    print(f"MSE : {mse:.4f}")
    plot_predictions(X, y, y_pred, title="Prédictions du modèle")
    plt.show()
    """)
    b.md("""
    ---
    ## Exercice 3 — L'effet du bruit

    Changez le niveau de bruit : 0,1 (données très propres), 1,0 (bruit modéré),
    5,0 (très bruitées).

    **Questions :**
    1. Comment le bruit change-t-il la MSE ?
    2. Les paramètres appris restent-ils proches des vrais ?
    """)
    b.ex("""
    noise_level = 1.0  # TODO : essayez 0.1, 1.0, 5.0

    X_exp, y_exp, true_w, true_b = generate_linear_data(n_samples=100, noise=noise_level, seed=42)
    model_exp = LinearRegression().fit(X_exp, y_exp)
    y_pred_exp = model_exp.predict(X_exp)
    print(f"Bruit : {noise_level} → MSE : {mean_squared_error(y_exp, y_pred_exp):.4f}")
    print(f"Poids : {model_exp.coef_} (vrai : {true_w}), biais : {model_exp.intercept_:.3f} (vrai : {true_b})")
    plot_predictions(X_exp, y_exp, y_pred_exp, title=f"Bruit = {noise_level}")
    plt.show()
    """, """
    for noise_level in [0.1, 1.0, 5.0]:
        X_exp, y_exp, true_w, true_b = generate_linear_data(n_samples=100, noise=noise_level, seed=42)
        model_exp = LinearRegression().fit(X_exp, y_exp)
        y_pred_exp = model_exp.predict(X_exp)
        print(f"Bruit : {noise_level} → MSE : {mean_squared_error(y_exp, y_pred_exp):.4f}")
        print(f"  poids : {model_exp.coef_} (vrai : {true_w}), biais : {model_exp.intercept_:.3f} (vrai : {true_b})")
        plot_predictions(X_exp, y_exp, y_pred_exp, title=f"Bruit = {noise_level}")
        plt.show()
    """)
    b.answer("""
    1. La MSE suit le carré du bruit (≈ 0,01, ≈ 1, ≈ 25) : le modèle ne peut pas
       prédire la part aléatoire.
    2. Les paramètres restent proches des vrais, mais s'en écartent d'autant plus que
       le bruit est fort : avec 100 points et un bruit de 5, le biais appris s'écarte
       d'environ un dixième du vrai (4,91 au lieu de 5), contre 0,002 pour un bruit de 0,1.
    """)
    b.md("""
    ---
    ## Exercice 4 — Modifier le jeu de données

    Essayez : plus d'exemples (`n_samples=500`), plus de bruit (`noise=3.0`), plus de
    variables (`n_features=3`).

    **Questions :**
    1. Plus d'exemples change-t-il la MSE ? et la qualité des paramètres appris ?
    2. Que se passe-t-il avec plus de variables ?
    """)
    b.ex("""
    X_new, y_new, true_w, true_b = generate_linear_data(
        n_samples=100,   # TODO : essayez 500
        n_features=1,    # TODO : essayez 3
        noise=1.0,       # TODO : essayez 3.0
        seed=42)
    model_new = LinearRegression().fit(X_new, y_new)
    y_pred_new = model_new.predict(X_new)
    print(f"{X_new.shape[0]} exemples, {X_new.shape[1]} variables → MSE : {mean_squared_error(y_new, y_pred_new):.4f}")
    print(f"Poids : {model_new.coef_} (vrais : {true_w}), biais : {model_new.intercept_:.3f}")
    """, """
    for n_samples, n_features, noise in [(100, 1, 1.0), (500, 1, 1.0), (500, 1, 3.0), (500, 3, 1.0)]:
        X_new, y_new, true_w, true_b = generate_linear_data(
            n_samples=n_samples, n_features=n_features, noise=noise, seed=42)
        model_new = LinearRegression().fit(X_new, y_new)
        y_pred_new = model_new.predict(X_new)
        print(f"{n_samples} exemples, {n_features} variables, bruit {noise} → "
              f"MSE : {mean_squared_error(y_new, y_pred_new):.4f}")
        print(f"  poids : {model_new.coef_.round(3)} (vrais : {true_w.round(3)}), "
              f"biais : {model_new.intercept_:.3f} (vrai : {true_b})")
    """)
    b.answer("""
    1. La MSE reste proche du carré du bruit (elle ne peut pas descendre en dessous),
       mais avec plus d'exemples les paramètres appris se rapprochent des vrais.
    2. Avec 3 variables, le modèle apprend un poids par variable ; il les retrouve tant
       qu'il y a beaucoup plus d'exemples que de variables. Le graphique en une
       dimension n'est plus possible.
    """)


# ==================================================================== Défi 3
def defi(b: Builder) -> None:
    b.title("iam-s3-defi", "Défi 3 — Survie à 2 mois avec un réseau de neurones")
    b.md(f"""
    *IA pour la médecine — séance 3.*

    Même défi qu'aux séances 1 et 2 : prédire si un patient de réanimation de l'étude
    **SUPPORT2** est vivant (`alive`) ou décédé (`dead`) **2 mois** après son
    admission. Métrique : l'**exactitude** ; prédire toujours `alive` donne environ 0,59.

    La méthode du jour : un **réseau de neurones avec arrêt précoce**, comparé
    honnêtement, sur le même jeu de validation, à la régression logistique (séance 1)
    et à la forêt aléatoire (séance 2).

    **Soumission** : le fichier `submission.csv` (colonnes `patient_id,outcome`, une
    ligne par patient de `X_test`) se dépose sur la page du défi : {CHALLENGE}
    """)
    b.md("## 1. Charger les données")
    b.code(f"""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, recall_score

    URL = "{SUPPORT2}"
    X_train = pd.read_csv(URL + "X_train.csv")
    y_train = pd.read_csv(URL + "y_train_survie.csv")
    X_test = pd.read_csv(URL + "X_test.csv")

    assert (X_train["patient_id"] == y_train["patient_id"]).all()
    print(f"Entraînement : {{len(X_train)}} patients ; test : {{len(X_test)}} patients")
    y_train["outcome"].value_counts(normalize=True)
    """)
    b.md("""
    ## 2. Préparer les données

    La préparation de la séance 2, écrite une seule fois dans un `ColumnTransformer` :

    - variables numériques : valeurs manquantes remplacées par la **médiane**, puis
      **standardisation** (indispensable pour le réseau) ;
    - variables catégorielles (texte) : valeur la plus fréquente, puis **encodage
      one-hot**.

    Placé dans un pipeline avec le modèle, le préparateur est ajusté sur les seuls
    patients d'entraînement à chaque `fit` : pas de fuite d'information.
    """)
    b.code("""
    X = X_train.drop(columns="patient_id")
    y = y_train["outcome"]
    categorielles = X.select_dtypes(exclude="number").columns.tolist()
    numeriques = X.select_dtypes(include="number").columns.tolist()
    print("Catégorielles :", categorielles)

    def preparation():
        return ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeriques),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), categorielles),
        ])

    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Ajustement : {len(X_fit)} patients ; validation : {len(X_val)} patients")
    """)
    b.md("""
    ## 3. Les modèles des séances 1 et 2

    Deux références, évaluées sur le **même** jeu de validation. On affiche
    l'exactitude (la métrique du défi) et le rappel des décès.
    """)
    b.code("""
    def evaluer(nom, modele):
        pipe = make_pipeline(preparation(), modele).fit(X_fit, y_fit)
        pred = pipe.predict(X_val)
        print(f"{nom:<32} exactitude {accuracy_score(y_val, pred):.4f}   "
              f"rappel des décès {recall_score(y_val, pred, pos_label='dead'):.3f}")
        return pipe

    evaluer("Régression logistique", LogisticRegression(max_iter=5000))
    evaluer("Forêt aléatoire", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                      random_state=42, n_jobs=-1))
    """)
    b.md("""
    ## 4. Le réseau de neurones

    **À faire** : évaluez deux réseaux de deux couches cachées `(64, 32)`
    (`random_state=42`) :

    1. sans arrêt précoce (`max_iter=500`) ;
    2. avec arrêt précoce : `early_stopping=True`, `validation_fraction=0.1`,
       `n_iter_no_change=10`.

    Comparez-les aux deux références. Que fait l'arrêt précoce ?
    """)
    b.ex("""
    # TODO : evaluer("MLP sans arrêt précoce", MLPClassifier(...))
    # TODO : mlp = evaluer("MLP avec arrêt précoce", MLPClassifier(..., early_stopping=True, ...))
    """, """
    evaluer("MLP sans arrêt précoce", MLPClassifier((64, 32), max_iter=500, random_state=42))
    mlp = evaluer("MLP avec arrêt précoce",
                  MLPClassifier((64, 32), max_iter=500, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=10, random_state=42))
    print("Époques avec arrêt précoce :", mlp[-1].n_iter_)
    """)
    b.answer("""
    Sans arrêt précoce, le réseau s'entraîne des centaines d'époques et sur-apprend :
    son exactitude de validation (environ 0,70) tombe nettement sous celle des
    références (environ 0,76). Avec l'arrêt précoce, il s'arrête après une vingtaine
    d'époques et atteint environ 0,78 : un peu mieux que les références, mais l'écart
    ne représente qu'une vingtaine de patients sur 1 233. Sur des données tabulaires
    de quelques milliers de patients, un réseau n'apporte pas forcément beaucoup plus
    qu'un modèle simple.
    """)
    b.md("""
    ## 5. Soumission

    Le modèle choisi est réentraîné sur **tous** les patients d'entraînement, puis
    prédit ceux de `X_test`.
    """)
    b.ex("""
    # TODO : final = make_pipeline(preparation(), MLPClassifier(... avec arrêt précoce ...))
    # TODO : final.fit(X, y)
    # TODO : submission = pd.DataFrame({"patient_id": X_test["patient_id"],
    #                                   "outcome": final.predict(X_test.drop(columns="patient_id"))})
    # TODO : submission.to_csv("submission.csv", index=False)
    """, """
    final = make_pipeline(preparation(),
                          MLPClassifier((64, 32), max_iter=500, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=10,
                                        random_state=42))
    final.fit(X, y)
    submission = pd.DataFrame({
        "patient_id": X_test["patient_id"],
        "outcome": final.predict(X_test.drop(columns="patient_id")),
    })
    submission.to_csv("submission.csv", index=False)

    assert len(submission) == 1541 and set(submission["outcome"]) <= {"alive", "dead"}
    print(submission["outcome"].value_counts(normalize=True))
    submission.head()
    """)
    b.md(f"""
    Dans Colab, la cellule suivante télécharge `submission.csv` ; ailleurs, le fichier
    est dans le dossier du notebook. Déposez-le sur {CHALLENGE}.
    """)
    b.code("""
    try:
        from google.colab import files
    except ImportError:
        print("Hors de Colab : submission.csv est dans le dossier courant.")
    else:
        files.download("submission.csv")
    """)


NOTEBOOKS: dict[str, Callable[[Builder], None]] = {
    "iam-s3-tp": tp,
    "iam-s3-bonus": bonus,
    "iam-s3-defi": defi,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in NOTEBOOKS.items():
        for correction in (False, True):
            nb._COUNTER = itertools.count()
            b = Builder(correction)
            build(b)
            notebook = nb.notebook(b.cells)
            path = OUT / (name + ("-correction" if correction else "") + ".ipynb")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(notebook, fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print("wrote", path, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
