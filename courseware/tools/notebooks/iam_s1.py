"""IA pour la médecine, séance 1 — classification. The Colab notebooks.

Writes four notebooks from one source each, into the ai_for_medicine checkout:

    iam-s1-tp.ipynb               TP 1, student version (`# TODO` skeletons)
    iam-s1-tp-correction.ipynb    TP 1, the solutions in place of the skeletons
    iam-s1-defi.ipynb             Défi 1, student version
    iam-s1-defi-correction.ipynb  Défi 1, solutions; writes a valid submission.csv

TP 1 is the French translation of session1/tp1.ipynb of
github.com/racousin/DFGSM2-IA-S2 (Breast Cancer Wisconsin, logistic
regression, metrics, scaling, cross-validation), same structure and exercises.
Fixes against the source: the split hint said test_size=0.33 for an 80/20
split; "same as before" pointed at a regression TP this course does not have;
the cross-validation now chains StandardScaler and the model with
make_pipeline, so that the scaler is refitted inside each fold (lesson
`evaluer-un-classifieur`, data leakage).

Défi 1 is the session's challenge (a copy of ML-Arena challenge 172, SUPPORT2,
alive/dead at 2 months, accuracy): numeric features, median imputation,
standardisation, logistic regression, an honest 80/20 validation, then a refit
on all training patients and `submission.csv` (patient_id,outcome; 1541 rows).
The challenge link is the placeholder `199`, which the lead
replaces.

Writes four files and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/iam_s1.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_medicine)

REPO = pathlib.Path(__file__).resolve().parents[3]
COURSE = "ia-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
DATA = aim_repo.raw(COURSE, "donnees/support2/")
CHALLENGE = "https://ml-arena.com/viewchallenge/199"


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines()


def m(text: str) -> dict:
    return nb.md(*_lines(text))


def c(text: str) -> dict:
    return nb.code(*_lines(text))


def header(title: str, name: str, solution: bool) -> str:
    suffix = " (correction)" if solution else ""
    return f"""
    # {title}{suffix}

    [![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, name)})

    *IA pour la médecine — séance 1.*
    """


# --------------------------------------------------------------------------- #
# TP 1 — cancer du sein
# --------------------------------------------------------------------------- #
def tp(solution: bool) -> list[dict]:
    def ex(todo: str, answer: str) -> dict:
        return c(answer if solution else todo)

    def reponse(answer: str) -> dict:
        return m(answer if solution else "*Votre réponse :* ...")

    name = "iam-s1-tp-correction.ipynb" if solution else "iam-s1-tp.ipynb"
    return [
        m(header("TP 1 — Cancer du sein et régression logistique", name,
                 solution) + """
        ## Objectif

        La **classification** est une tâche d'apprentissage supervisé où l'on
        prédit une **catégorie** (et non un nombre).

        Dans ce TP, on prédit si une tumeur du sein est **maligne**
        (cancéreuse) ou **bénigne** (non cancéreuse) à partir de mesures des
        noyaux cellulaires. C'est une **classification binaire** : deux
        sorties possibles.

        | | Régression | Classification (ce TP) |
        |---|---|---|
        | **Sortie** | Un nombre réel | Une catégorie (classe) |
        | **Exemple** | Durée de séjour = 6,5 jours | Tumeur = maligne |
        | **Fonction de perte** | MSE | Entropie croisée |
        | **Métriques** | MSE, MAE, R² | Exactitude, précision, rappel, F1 |

        **Méthode** : explorer → visualiser → découper → entraîner → évaluer
        → valider par validation croisée.

        **Comment travailler** : Fichier → *Enregistrer une copie dans Drive*,
        puis exécutez les cellules dans l'ordre (Maj + Entrée) et complétez
        celles marquées `# TODO` en remplaçant les `...`.
        """),
        m("## 1. Mise en place"),
        c("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, confusion_matrix,
                                     ConfusionMatrixDisplay, classification_report,
                                     precision_score, recall_score, f1_score)
        """),
        m("""
        ## 2. Charger et découvrir les données

        Le jeu **Breast Cancer Wisconsin** contient des mesures de noyaux
        cellulaires issues de biopsies mammaires. Chaque échantillon est
        étiqueté **maligne** (classe 0) ou **bénigne** (classe 1) : c'est le
        codage de scikit-learn, à garder en tête pour la suite.
        """),
        m("""
        ### Exercice 1.1 — Charger le jeu de données

        Chargez le jeu de données et affichez sa forme, le nom des classes et
        les premières lignes.

        **Pourquoi** : savoir combien d'échantillons, combien de variables,
        et ce que désignent les classes.

        *Indice* : `load_breast_cancer(as_frame=True)`, `data.target_names`,
        `df.head()`
        """),
        ex("""
        # TODO : charger les données dans data, puis le tableau dans df
        data = ...
        df = data.frame

        print(f"Forme : {df.shape}")
        print(f"Classes : {data.target_names}")
        df.head()
        """, """
        data = load_breast_cancer(as_frame=True)
        df = data.frame

        print(f"Forme : {df.shape}")
        print(f"Classes : {data.target_names}")
        df.head()
        """),
        m("""
        Les 30 variables sont calculées sur une image numérisée d'une
        **cytoponction à l'aiguille fine** d'une masse mammaire. Elles
        décrivent les noyaux des cellules.

        **10 mesures de base** :

        - **radius** (rayon) — distance moyenne du centre au contour du noyau
        - **texture** — écart-type des niveaux de gris
        - **perimeter** (périmètre) et **area** (aire) du noyau
        - **smoothness** (régularité) — variation locale du rayon
        - **compactness** (compacité) — périmètre² / aire − 1
        - **concavity** (concavité) — profondeur des parties concaves du contour
        - **concave points** — nombre de parties concaves du contour
        - **symmetry** (symétrie) du noyau
        - **fractal dimension** — complexité du contour

        **Trois variantes par mesure** : `mean` (moyenne sur les noyaux de
        l'image), `error` (erreur standard), `worst` (moyenne des trois plus
        grandes valeurs).

        **Cible** : maligne (0) / bénigne (1).
        """),
        m("""
        ### Exercice 1.2 — Statistiques descriptives

        Affichez les statistiques descriptives des variables.

        *Indice* : `df.describe()`

        **Question** : les variables ont-elles des échelles comparables ?
        Qu'est-ce que cela suggère pour la préparation des données ?
        """),
        ex("""
        # TODO : statistiques descriptives de df
        ...
        """, """
        df.describe()
        """),
        reponse("""
        **Réponse** : non. `mean area` va jusqu'à environ 2 500 alors que
        `mean smoothness` reste entre 0,05 et 0,16. Il faudra **mettre les
        variables à l'échelle** avant l'entraînement : la régression
        logistique y est sensible (partie 6).
        """),
        m("""
        ### Exercice 1.3 — Équilibre des classes

        Comptez les tumeurs malignes et bénignes.

        **Pourquoi** : si 95 % des tumeurs étaient bénignes, un modèle qui
        répond toujours « bénigne » aurait 95 % d'exactitude tout en étant
        inutile. Il faut connaître l'équilibre des classes avant de faire
        confiance à l'exactitude.

        *Indice* : `y.value_counts()` et `y.value_counts(normalize=True)`
        """),
        ex("""
        X = df.drop(columns="target")
        y = df["target"]

        # TODO : effectifs, puis proportions de chaque classe
        print(...)
        print(...)
        """, """
        X = df.drop(columns="target")
        y = df["target"]

        print(y.value_counts())
        print(y.value_counts(normalize=True))
        """),
        reponse("""
        **Réponse** : 357 bénignes (63 %) et 212 malignes (37 %). Les classes
        sont raisonnablement équilibrées : l'exactitude a un sens ici, mais on
        regardera aussi la précision et le rappel.
        """),
        m("""
        ## 3. Visualiser les données

        En classification, la visualisation montre si les classes sont
        **séparables** : peut-on distinguer les deux groupes à l'œil avec
        certaines variables ?
        """),
        m("""
        ### Exercice 2.1 — Distribution d'une variable par classe

        Tracez la distribution de `mean radius`, colorée par classe.

        **Pourquoi** : si les deux distributions se chevauchent beaucoup, la
        variable aide peu à distinguer les classes.

        *Indice* : `sns.histplot(data=df, x="mean radius", hue="target", kde=True)`
        """),
        ex("""
        # TODO : histogramme de mean radius coloré par target
        ...
        plt.show()
        """, """
        sns.histplot(data=df, x="mean radius", hue="target", kde=True, bins=30)
        plt.title("Rayon moyen selon la classe (0 = maligne, 1 = bénigne)")
        plt.show()
        """),
        reponse("""
        Les distributions diffèrent nettement : les tumeurs malignes ont des
        noyaux plus grands. Cette variable sera utile au modèle.
        """),
        m("""
        ### Exercice 2.2 — Nuage de points de deux variables

        Tracez `mean radius` en abscisse, `mean texture` en ordonnée, colorés
        par classe.

        *Indice* : `sns.scatterplot(data=df, x="mean radius", y="mean texture", hue="target", alpha=0.6)`
        """),
        ex("""
        # TODO : nuage de points coloré par target
        ...
        plt.show()
        """, """
        sns.scatterplot(data=df, x="mean radius", y="mean texture",
                        hue="target", alpha=0.6)
        plt.title("Rayon moyen et texture moyenne (0 = maligne, 1 = bénigne)")
        plt.show()
        """),
        reponse("""
        Les deux classes forment deux nuages assez distincts, qui se
        chevauchent en partie : elles sont partiellement séparables avec ces
        deux variables seulement.
        """),
        m("""
        ### Exercice 2.3 — Carte des corrélations

        Calculez la corrélation de chaque variable avec la cible, gardez les
        10 plus fortes en valeur absolue et affichez leur carte de
        corrélation (*heatmap*).

        **Question** : quelles variables semblent les plus utiles pour
        distinguer tumeurs malignes et bénignes ?
        """),
        ex("""
        correlations = df.corr()["target"].drop("target").abs()
        # TODO : les 10 noms de variables les plus corrélées à la cible
        top = ...

        sns.heatmap(df[top + ["target"]].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0)
        plt.show()
        """, """
        correlations = df.corr()["target"].drop("target").abs()
        top = correlations.sort_values(ascending=False).head(10).index.tolist()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df[top + ["target"]].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0)
        plt.tight_layout()
        plt.show()
        print(correlations.sort_values(ascending=False).head(10))
        """),
        reponse("""
        **Réponse** : les variantes `worst` et `mean` de *concave points*,
        *perimeter*, *radius* et *area* (corrélations de 0,7 à 0,8 en valeur
        absolue). Ce sont des mesures de taille et de forme des noyaux, des
        critères que l'anatomopathologiste regarde aussi.
        """),
        m("""
        ## 4. La régression logistique

        En régression linéaire, on prédit directement un nombre :
        $\\hat{y} = \\theta_0 + \\theta_1 x_1 + \\dots + \\theta_p x_p$.

        Pour classer, on veut une **probabilité** entre 0 et 1. La régression
        logistique applique la **sigmoïde** à cette combinaison linéaire :

        $$P(y=1 \\mid x) = \\sigma(\\theta_0 + \\theta_1 x_1 + \\dots + \\theta_p x_p),
        \\qquad \\sigma(z) = \\frac{1}{1 + e^{-z}}$$

        **Règle de décision** : si $P(y=1 \\mid x) > 0{,}5$, on prédit la classe 1
        (bénigne), sinon la classe 0 (maligne).

        **Fonction de perte** : l'entropie croisée,

        $$\\mathcal{L} = -\\frac{1}{n}\\sum_{i=1}^{n}\\left[y_i \\log(\\hat{p}_i)
        + (1-y_i)\\log(1-\\hat{p}_i)\\right]$$

        qui pénalise fortement les prédictions fausses faites avec assurance.
        """),
        m("""
        ### Exercice 3.1 — Découper les données

        Découpez en entraînement (80 %) et test (20 %), avec `random_state=42`.

        *Indice* : `train_test_split(X, y, test_size=0.2, random_state=42)`
        """),
        ex("""
        # TODO : découpe 80 / 20
        X_train, X_test, y_train, y_test = ...

        print(f"Entraînement : {X_train.shape[0]} échantillons")
        print(f"Test : {X_test.shape[0]} échantillons")
        """, """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        print(f"Entraînement : {X_train.shape[0]} échantillons")
        print(f"Test : {X_test.shape[0]} échantillons")
        """),
        m("""
        ### Exercice 3.2 — Entraîner une régression logistique

        Entraînez `LogisticRegression(max_iter=5000)` sur le jeu
        d'entraînement et prédisez le jeu de test.

        *Indice* : `.fit(X_train, y_train)`, puis `.predict(X_test)`
        """),
        ex("""
        model = LogisticRegression(max_iter=5000)
        # TODO : entraîner, puis prédire X_test
        ...
        y_pred = ...

        print(f"Prédictions (10 premières) : {y_pred[:10]}")
        print(f"Réalité     (10 premières) : {y_test.values[:10]}")
        """, """
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Prédictions (10 premières) : {y_pred[:10]}")
        print(f"Réalité     (10 premières) : {y_test.values[:10]}")
        """),
        m("""
        ## 5. Évaluer le classifieur

        **Exactitude** (*accuracy*) : la part de prédictions correctes. Simple,
        mais trompeuse quand les classes sont déséquilibrées.

        **Matrice de confusion**, avec la tumeur **maligne comme classe
        positive** (c'est elle qu'on cherche à dépister) :

        | | Prédit maligne | Prédit bénigne |
        |---|---|---|
        | **Maligne** | VP (vrai positif) | FN (faux négatif : cancer manqué) |
        | **Bénigne** | FP (faux positif : fausse alerte) | VN (vrai négatif) |

        | Métrique | Formule | Question |
        |---|---|---|
        | **Précision** | $\\frac{VP}{VP + FP}$ | Parmi les tumeurs prédites malignes, combien le sont ? |
        | **Rappel** (sensibilité) | $\\frac{VP}{VP + FN}$ | Parmi les tumeurs malignes, combien a-t-on trouvées ? |
        | **F1** | $2 \\cdot \\frac{P \\cdot R}{P + R}$ | Moyenne harmonique des deux |

        En diagnostic, **le rappel est crucial** : un cancer manqué (faux
        négatif) est bien plus grave qu'une fausse alerte, qui mène à des
        examens complémentaires.
        """),
        m("""
        ### Exercice 4.1 — Exactitude

        *Indice* : `accuracy_score(y_test, y_pred)`
        """),
        ex("""
        # TODO : exactitude sur le jeu de test
        acc = ...
        print(f"Exactitude : {acc:.4f}")
        """, """
        acc = accuracy_score(y_test, y_pred)
        print(f"Exactitude : {acc:.4f}")
        """),
        m("""
        ### Exercice 4.2 — Matrice de confusion

        Affichez la matrice de confusion avec le nom des classes.

        *Indice* : `ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)`

        **Question** : combien de tumeurs malignes ont été manquées (faux
        négatifs) ? Pourquoi est-ce le chiffre important ?
        """),
        ex("""
        # TODO : afficher la matrice de confusion
        ...
        plt.show()
        """, """
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=data.target_names, cmap="Blues")
        plt.title("Matrice de confusion")
        plt.show()
        """),
        reponse("""
        **Réponse** : la ligne *malignant* (réalité), colonne *benign*
        (prédiction) : 4 des 43 tumeurs malignes du test sont prédites
        bénignes. Ce sont
        les erreurs les plus graves : une patiente atteinte d'un cancer serait
        rassurée à tort.
        """),
        m("""
        ### Exercice 4.3 — Précision, rappel et F1 à la main

        Extrayez VP, VN, FP et FN de la matrice, calculez précision, rappel
        et F1 avec les formules, puis vérifiez avec scikit-learn.

        *Indice* : `confusion_matrix(y_test, y_pred)` a pour lignes la réalité
        et pour colonnes la prédiction, dans l'ordre des classes 0 (maligne)
        puis 1 (bénigne). La classe positive étant la maligne (0), il faut
        `pos_label=0` dans les fonctions de scikit-learn.
        """),
        ex("""
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # TODO : lire les quatre cases (classe positive = maligne = 0)
        VP = ...   # maligne prédite maligne
        FN = ...   # maligne prédite bénigne
        FP = ...   # bénigne prédite maligne
        VN = ...   # bénigne prédite bénigne

        # TODO : les trois formules
        precision = ...
        rappel = ...
        f1 = ...
        print(f"Précision {precision:.4f}  rappel {rappel:.4f}  F1 {f1:.4f}")
        print(f"scikit-learn : {precision_score(y_test, y_pred, pos_label=0):.4f}  "
              f"{recall_score(y_test, y_pred, pos_label=0):.4f}  "
              f"{f1_score(y_test, y_pred, pos_label=0):.4f}")
        """, """
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        VP = cm[0, 0]   # maligne prédite maligne
        FN = cm[0, 1]   # maligne prédite bénigne
        FP = cm[1, 0]   # bénigne prédite maligne
        VN = cm[1, 1]   # bénigne prédite bénigne

        precision = VP / (VP + FP)
        rappel = VP / (VP + FN)
        f1 = 2 * precision * rappel / (precision + rappel)
        print(f"Précision {precision:.4f}  rappel {rappel:.4f}  F1 {f1:.4f}")
        print(f"scikit-learn : {precision_score(y_test, y_pred, pos_label=0):.4f}  "
              f"{recall_score(y_test, y_pred, pos_label=0):.4f}  "
              f"{f1_score(y_test, y_pred, pos_label=0):.4f}")
        """),
        m("""
        ### Exercice 4.4 — Rapport de classification complet

        *Indice* : `print(classification_report(y_test, y_pred, target_names=data.target_names))`
        """),
        ex("""
        # TODO : rapport de classification
        ...
        """, """
        print(classification_report(y_test, y_pred,
                                    target_names=data.target_names))
        """),
        reponse("""
        Le rapport donne précision, rappel et F1 pour **chaque** classe, plus
        l'exactitude et deux moyennes (*macro* : moyenne simple des classes ;
        *weighted* : pondérée par leurs effectifs). La ligne *malignant*
        reprend les valeurs calculées à la main.
        """),
        m("""
        ## 6. Mettre les variables à l'échelle

        La régression logistique minimise l'entropie croisée par un algorithme
        d'optimisation itératif, de la famille de la descente de gradient.
        Quand les variables ont des échelles très différentes, la surface de
        perte est étirée et l'optimisation converge lentement ou mal.

        La **standardisation** ramène chaque variable à une moyenne 0 et un
        écart-type 1 :

        $$x_{\\text{standardisé}} = \\frac{x - \\mu}{\\sigma}$$

        **À retenir** : $\\mu$ et $\\sigma$ sont calculés sur le jeu
        d'**entraînement** seulement (`fit_transform`), puis appliqués au
        test (`transform`).
        """),
        m("""
        ### Exercice 5.1 — Standardiser et réentraîner

        Standardisez avec `StandardScaler`, réentraînez, et comparez
        l'exactitude et la matrice de confusion.

        **Question** : la mise à l'échelle a-t-elle amélioré les résultats ?
        Pourquoi ?
        """),
        ex("""
        scaler = StandardScaler()
        # TODO : apprendre la standardisation sur l'entraînement, l'appliquer aux deux
        X_train_s = ...
        X_test_s = ...

        model_s = LogisticRegression(max_iter=5000)
        # TODO : entraîner sur X_train_s, prédire X_test_s
        ...
        y_pred_s = ...

        print(f"Exactitude sans mise à l'échelle : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Exactitude avec mise à l'échelle : {accuracy_score(y_test, y_pred_s):.4f}")
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_s, display_labels=data.target_names, cmap="Blues")
        plt.show()
        """, """
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model_s = LogisticRegression(max_iter=5000)
        model_s.fit(X_train_s, y_train)
        y_pred_s = model_s.predict(X_test_s)

        print(f"Exactitude sans mise à l'échelle : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Exactitude avec mise à l'échelle : {accuracy_score(y_test, y_pred_s):.4f}")
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_s, display_labels=data.target_names, cmap="Blues")
        plt.show()
        """),
        reponse("""
        **Réponse** : l'exactitude passe de 0,956 à 0,974 (3 erreurs au lieu
        de 5 sur 114 tumeurs). Avec `max_iter=5000`, le modèle non standardisé finit par
        converger, mais lentement ; sur des variables standardisées,
        l'optimisation converge vite et vers un meilleur réglage. Le gain
        dépend du jeu de données.
        """),
        m("""
        ## 7. Validation croisée

        Un seul découpage donne une seule estimation, qui dépend du hasard du
        découpage. La **validation croisée à 5 blocs** découpe les données en
        5 parts, entraîne 5 fois en testant chaque fois sur une part
        différente, et donne la moyenne et l'écart-type des 5 scores.

        Pour standardiser **dans** chaque tour (sans que le bloc de test
        serve à calculer $\\mu$ et $\\sigma$), on enchaîne la standardisation
        et le modèle avec `make_pipeline`.
        """),
        m("""
        ### Exercice 6.1 — Validation croisée à 5 blocs

        *Indice* : `cross_val_score(modele, X, y, cv=5, scoring="accuracy")`

        **Question** : le modèle est-il stable d'un bloc à l'autre ?
        """),
        ex("""
        modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        # TODO : les 5 scores d'exactitude
        scores = ...

        print(f"Scores par bloc : {np.round(scores, 3)}")
        print(f"Exactitude = {scores.mean():.4f} ± {scores.std():.4f}")
        """, """
        modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        scores = cross_val_score(modele, X, y, cv=5, scoring="accuracy")

        print(f"Scores par bloc : {np.round(scores, 3)}")
        print(f"Exactitude = {scores.mean():.4f} ± {scores.std():.4f}")
        """),
        reponse("""
        **Réponse** : l'écart-type est faible (de l'ordre de 0,01 à 0,02) :
        les performances varient peu selon le découpage, le modèle est
        stable. L'exactitude moyenne est une estimation plus fiable que celle
        d'un seul découpage.
        """),
        m("""
        ## 8. Bilan

        | | Régression | Classification (ce TP) |
        |---|---|---|
        | **Tâche** | Prédire une valeur continue | Prédire une catégorie |
        | **Sortie** | Un nombre réel | Une classe (0 ou 1), via une probabilité |
        | **Modèle** | Régression linéaire | Régression logistique |
        | **Perte** | MSE | Entropie croisée |
        | **Métriques** | MSE, MAE, R² | Exactitude, précision, rappel, F1 |

        La **méthode** est la même dans les deux cas :

        1. **Explorer** les données ;
        2. **Visualiser** les relations ;
        3. **Découper** en entraînement et test pour une évaluation honnête ;
        4. **Entraîner** le modèle adapté ;
        5. **Évaluer** avec les bonnes métriques ;
        6. **Valider** par validation croisée.

        Prochaine étape : le **défi 1**, la même méthode sur des patients de
        réanimation.
        """),
    ]


# --------------------------------------------------------------------------- #
# Défi 1 — survie à 2 mois
# --------------------------------------------------------------------------- #
def defi(solution: bool) -> list[dict]:
    def ex(todo: str, answer: str) -> dict:
        return c(answer if solution else todo)

    name = "iam-s1-defi-correction.ipynb" if solution else "iam-s1-defi.ipynb"
    return [
        m(header("Défi 1 — Survie à 2 mois avec une régression logistique",
                 name, solution) + f"""
        On prédit si un patient de réanimation est **vivant ou décédé 2 mois**
        après son admission, avec la méthode du TP 1 : une régression
        logistique sur des variables standardisées.

        **Le défi** : [{CHALLENGE}]({CHALLENGE}). Vous y déposez le fichier
        `submission.csv` produit à la fin de ce notebook ; le score est
        l'**exactitude** sur des patients dont vous ne connaissez pas le
        devenir. Répondre toujours « vivant » donne environ 0,59 : c'est le
        score à battre.

        Complétez les cellules marquées `# TODO` en remplaçant les `...`.
        """),
        m("""
        ## 1. Les données

        Les patients viennent de l'étude **SUPPORT2** (*Study to Understand
        Prognoses and Preferences for Outcomes and Risks of Treatments*),
        une étude multicentrique américaine de patients hospitalisés en
        réanimation. Chaque patient est décrit par 28 variables mesurées à
        l'admission : âge, sexe, groupe diagnostique, signes vitaux (pression
        artérielle moyenne `meanbp`, fréquence cardiaque `hrt`, température
        `temp`…), biologie (créatinine `crea`, sodium `sod`, pH `ph`…) et
        scores de gravité (`aps`, `scoma`…).

        La cible `outcome` vaut `alive` (vivant) ou `dead` (décédé) à 2 mois.
        """),
        c(f"""
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt

        URL = "{DATA}"
        X_train = pd.read_csv(URL + "X_train.csv")
        y_train = pd.read_csv(URL + "y_train_survie.csv")
        X_test = pd.read_csv(URL + "X_test.csv")

        assert (X_train["patient_id"] == y_train["patient_id"]).all()
        print(f"Entraînement : {{len(X_train)}} patients ; test : {{len(X_test)}} patients")
        X_train.head()
        """),
        m("""
        ### Exercice 1 — La cible

        Quelle proportion de patients est vivante à 2 mois ? C'est
        l'exactitude d'un modèle qui répond toujours `alive`.
        """),
        ex("""
        y = y_train["outcome"]
        # TODO : proportion de chaque classe
        ...
        """, """
        y = y_train["outcome"]
        y.value_counts(normalize=True)
        """),
        m("""
        ## 2. Les variables numériques

        La régression logistique ne prend que des nombres. Pour ce premier
        défi, on garde les **variables numériques** (les variables texte comme
        `sex` ou `dzgroup` seront encodées en séance 2). Plusieurs ont des
        **valeurs manquantes** : un examen non fait.
        """),
        ex("""
        # TODO : les colonnes numériques de X_train (select_dtypes)
        X = ...
        print(X.columns.tolist())
        X.isna().mean().sort_values(ascending=False).head(8)
        """, """
        X = X_train.select_dtypes(include="number")
        print(X.columns.tolist())
        X.isna().mean().sort_values(ascending=False).head(8)
        """),
        m("""
        ## 3. Le modèle

        Trois étapes, enchaînées par `make_pipeline` :

        1. `SimpleImputer(strategy="median")` remplace chaque valeur manquante
           par la médiane de la variable ;
        2. `StandardScaler()` standardise (moyenne 0, écart-type 1) ;
        3. `LogisticRegression(max_iter=1000)` classe.

        `fit` apprend les trois étapes sur les seules données qu'on lui donne
        (médianes, moyennes, écarts-types, coefficients) ; `predict` les
        applique telles quelles. Les données de validation ne servent donc
        jamais à l'entraînement : pas de fuite.

        ### Exercice 2 — Validation honnête

        Réservez 20 % des patients pour la validation (`stratify=y`,
        `random_state=42`), entraînez le pipeline sur les 80 % restants et
        mesurez l'exactitude en validation.
        """),
        ex("""
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        modele = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=1000))
        # TODO : entraîner sur (X_tr, y_tr), prédire X_val
        ...
        y_val_pred = ...

        print(f"Exactitude en validation : {accuracy_score(y_val, y_val_pred):.3f}")
        print(f"Toujours 'alive'         : {(y_val == 'alive').mean():.3f}")
        ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, cmap="Blues")
        plt.show()
        """, """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        modele = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=1000))
        modele.fit(X_tr, y_tr)
        y_val_pred = modele.predict(X_val)

        print(f"Exactitude en validation : {accuracy_score(y_val, y_val_pred):.3f}")
        print(f"Toujours 'alive'         : {(y_val == 'alive').mean():.3f}")
        ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, cmap="Blues")
        plt.show()
        """),
        m("""
        Le modèle doit battre nettement la réponse constante. Regardez la
        matrice : combien de décès sont prédits « vivant » ?

        ## 4. La soumission

        Le choix est fait : on réentraîne sur **tous** les patients
        d'entraînement, puis on prédit ceux du test.

        ### Exercice 3 — Réentraîner et prédire le test
        """),
        ex("""
        # TODO : entraîner modele sur (X, y), puis prédire les colonnes de X.columns dans X_test
        ...
        predictions = ...

        submission = pd.DataFrame({"patient_id": X_test["patient_id"],
                                   "outcome": predictions})
        assert len(submission) == 1541 and submission["patient_id"].is_unique
        assert set(submission["outcome"]) <= {"alive", "dead"}
        submission.to_csv("submission.csv", index=False)
        print(f"submission.csv : {len(submission)} lignes")
        submission["outcome"].value_counts()
        """, """
        modele.fit(X, y)
        predictions = modele.predict(X_test[X.columns])

        submission = pd.DataFrame({"patient_id": X_test["patient_id"],
                                   "outcome": predictions})
        assert len(submission) == 1541 and submission["patient_id"].is_unique
        assert set(submission["outcome"]) <= {"alive", "dead"}
        submission.to_csv("submission.csv", index=False)
        print(f"submission.csv : {len(submission)} lignes")
        submission["outcome"].value_counts()
        """),
        m(f"""
        Sur Colab, la cellule suivante télécharge `submission.csv` ; déposez-le
        ensuite sur la page du défi : [{CHALLENGE}]({CHALLENGE}).
        """),
        c("""
        try:
            from google.colab import files
        except ImportError:
            print("Hors de Colab : submission.csv est dans le dossier courant.")
        else:
            files.download("submission.csv")
        """),
        m("""
        ## 5. Pour aller plus loin

        - Le score du classement est-il proche de votre exactitude en
          validation ? Pourquoi peut-il différer un peu ?
        - Remplacez la découpe unique par une validation croisée à 5 blocs
          (`cross_val_score(modele, X, y, cv=5)`) : quelle est la variabilité ?
        - Calculez le **rappel** de la classe `dead` en validation : quelle part
          des décès le modèle annonce-t-il ?
        """),
    ]


NOTEBOOKS = (
    ("iam-s1-tp.ipynb", tp, False),
    ("iam-s1-tp-correction.ipynb", tp, True),
    ("iam-s1-defi.ipynb", defi, False),
    ("iam-s1-defi-correction.ipynb", defi, True),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder, solution in NOTEBOOKS:
        nb._COUNTER = itertools.count()     # byte-identical on every regeneration
        path = OUT / name
        notebook = nb.notebook(builder(solution))
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
