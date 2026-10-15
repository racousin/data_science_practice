"""Projet: the project notebook of `ia-medecine`.

The module `iam-projet` has one exercise page (`projet-statut-fonctionnel.md`)
and this one notebook, `iam-projet.ipynb`, adapted from project/project.ipynb
of github.com/racousin/DFGSM2-IA-S2. By the owner's decision the project runs
on ML-Arena challenge 172 itself ("2-Month Survival Prediction"): SUPPORT2
patients, binary target `outcome` (alive/dead at 2 months, from
`donnees/support2/y_train_survie.csv`), metric accuracy (always "alive"
scores about 0.59). Sessions 1-3 already fit a logistic regression, a random
forest and an MLP on this task, so the project asks the groups to go further
(feature engineering, missingness as a feature, gradient boosting,
calibration, leakage check) and to present their method.

The notebook is a BASELINE, not an exercise: it loads the data, prepares it
(median imputation, one-hot encoding, standardisation, all fitted on the
training rows), trains a logistic regression, measures accuracy, the
confusion matrix and the recall of `dead` on a held-out validation split,
refits on all the training data and writes `submission.csv`
(patient_id,outcome, one row per X_test patient). The groups start from it
and improve it. There is therefore no `-correction` twin.

Writes one notebook and nothing else — unlike `build_notebooks.py`, running
this script is safe:

    uv run python courseware/tools/notebooks/iam_projet.py
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
NAME = "iam-projet"
OUT = aim_repo.out_dir(COURSE, "notebooks")
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
DATA = aim_repo.raw(COURSE, "donnees/support2/")
CHALLENGE_URL = "https://ml-arena.com/viewchallenge/172"


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines()


def m(text: str) -> dict:
    return nb.md(*_lines(text))


def c(text: str) -> dict:
    return nb.code(*_lines(text))


def cells() -> list[dict]:
    return [
        m(f"""
        # Projet — Survie à 2 mois : aller plus loin

        [![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, NAME + '.ipynb')})

        *IA pour la médecine — projet en binôme.*

        Pendant les séances, vous avez prédit la survie à 2 mois avec une
        régression logistique, une forêt aléatoire et un réseau de neurones.
        Le projet consiste à **aller plus loin** que ces modèles et à
        présenter votre méthode.

        Ce notebook est un **point de départ** : il charge les données,
        entraîne une régression logistique, mesure son exactitude sur un jeu
        de validation et écrit `submission.csv`.

        **Soumission** : `submission.csv` se dépose sur le challenge du projet,
        [ml-arena.com/viewchallenge/172]({CHALLENGE_URL}).
        """),
        m("""
        ## 1. Contexte médical

        Les données viennent de l'étude **SUPPORT2** (*Study to Understand
        Prognoses and Preferences for Outcomes and Risks of Treatments*), une
        étude multicentrique américaine de patients gravement malades
        hospitalisés en réanimation ou en soins intensifs.

        On prédit si le patient est **vivant ou décédé 2 mois** après
        l'admission (`outcome` ∈ {`alive`, `dead`}). Un tel pronostic sert à
        informer les familles, à orienter les décisions thérapeutiques et à
        adapter le niveau de soins.
        """),
        m("""
        ## 2. Les données

        28 variables cliniques et démographiques mesurées à l'admission, plus
        l'identifiant `patient_id`.

        **Démographie**

        | Variable | Description |
        |---|---|
        | `age` | Âge (années) |
        | `sex` | Sexe (male/female) |
        | `race` | Origine ethnique |
        | `income` | Tranche de revenus |
        | `edu` | Années d'éducation |

        **Maladie**

        | Variable | Description |
        |---|---|
        | `dzgroup` | Groupe diagnostique (ARF/MOSF w/Sepsis, COPD, Lung Cancer, Coma, ...) |
        | `dzclass` | Classe de maladie (ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, Coma) |
        | `num.co` | Nombre de comorbidités |
        | `ca` | Statut cancérologique (no, yes, metastatic) |

        **Signes vitaux**

        | Variable | Description | Unité |
        |---|---|---|
        | `meanbp` | Pression artérielle moyenne | mmHg |
        | `hrt` | Fréquence cardiaque | bpm |
        | `resp` | Fréquence respiratoire | cycles/min |
        | `temp` | Température | °C |

        **Biologie**

        | Variable | Description | Unité |
        |---|---|---|
        | `wblc` | Leucocytes | ×10³/µL |
        | `pafi` | Rapport PaO₂/FiO₂ (oxygénation) | - |
        | `alb` | Albumine sérique | g/dL |
        | `bili` | Bilirubine | mg/dL |
        | `crea` | Créatinine (fonction rénale) | mg/dL |
        | `sod` | Sodium | mEq/L |
        | `ph` | pH artériel | - |
        | `glucose` | Glycémie | mg/dL |
        | `bun` | Urée sanguine (BUN) | mg/dL |
        | `urine` | Débit urinaire | mL/jour |

        **Scores cliniques**

        | Variable | Description |
        |---|---|
        | `adlp` | Score ADL (*Activities of Daily Living*) rapporté par le patient |
        | `adls` | Score ADL rapporté par un proche |
        | `scoma` | Score de coma (conscience, dérivé du score de Glasgow) |
        | `avtisst` | Score TISS moyen (intensité des soins) |
        | `aps` | Score APACHE III (sévérité physiologique) |

        De nombreuses variables numériques ont des **valeurs manquantes**.
        """),
        m("""
        ## 3. La métrique : l'exactitude

        Le classement du challenge utilise l'**exactitude** (*accuracy*), la
        part des patients bien classés :

        $$\\text{exactitude} = \\frac{VP + VN}{VP + VN + FP + FN}$$

        Prédire `alive` pour tout le monde donne déjà environ **0,59** : c'est
        la référence minimale à battre.

        Cliniquement, l'exactitude ne suffit pas : manquer un décès (faux
        négatif) n'a pas le même coût qu'une fausse alerte. Regardez donc
        aussi la **matrice de confusion** et le **rappel** de la classe `dead`,
        la part des décès que le modèle détecte :

        $$\\text{rappel}_{dead} = \\frac{VP}{VP + FN}$$
        """),
        m("## 4. Baseline\n\n### 4.1 Chargement des données"),
        c(f"""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        URL = "{DATA}"
        X_train = pd.read_csv(URL + "X_train.csv")
        y_train = pd.read_csv(URL + "y_train_survie.csv")
        X_test = pd.read_csv(URL + "X_test.csv")

        assert (X_train["patient_id"] == y_train["patient_id"]).all()
        print(f"Entraînement : {{len(X_train)}} patients, {{X_train.shape[1] - 1}} variables")
        print(f"Test         : {{len(X_test)}} patients")
        y_train["outcome"].value_counts(normalize=True)
        """),
        m("""
        ### 4.2 Prétraitement

        On met de côté les identifiants du test (pour la soumission), puis on
        retire `patient_id`, qui n'est pas une variable prédictive.
        """),
        c("""
        test_ids = X_test["patient_id"]
        X = X_train.drop(columns="patient_id")
        X_t = X_test.drop(columns="patient_id")
        y = y_train["outcome"]

        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = [col for col in X.columns if col not in num_cols]
        print("Catégorielles :", cat_cols)
        print("Numériques    :", num_cols)
        """),
        m("""
        On réserve 20 % des patients d'entraînement pour la **validation** :
        le modèle ne les voit pas, ils mesurent honnêtement sa performance.
        La découpe est stratifiée pour garder la proportion de décès.
        """),
        c("""
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        print(len(X_tr), "patients pour entraîner,", len(X_val), "pour valider")
        """),
        m("""
        La préparation est **apprise sur les seules lignes d'entraînement**
        (médianes, catégories, moyennes et écarts-types), puis appliquée telle
        quelle aux autres : sinon de l'information fuit de la validation ou
        du test vers le modèle.

        - valeurs manquantes numériques → médiane ;
        - variables catégorielles → encodage *one-hot* (une colonne 0/1 par
          catégorie ; une catégorie inconnue donne des zéros) ;
        - standardisation (moyenne 0, écart-type 1).
        """),
        c("""
        def preparer(X_fit, *autres):
            \"\"\"Apprend la préparation sur X_fit et l'applique à X_fit et aux autres.\"\"\"
            medianes = X_fit[num_cols].median()
            colonnes = pd.get_dummies(X_fit, columns=cat_cols).columns

            def transformer(X):
                X = X.copy()
                X[num_cols] = X[num_cols].fillna(medianes)
                X = pd.get_dummies(X, columns=cat_cols)
                return X.reindex(columns=colonnes, fill_value=0).astype(float)

            scaler = StandardScaler().fit(transformer(X_fit))
            return [scaler.transform(transformer(Z)) for Z in (X_fit, *autres)]

        X_tr_p, X_val_p = preparer(X_tr, X_val)
        X_tr_p.shape
        """),
        m("### 4.3 Entraînement et évaluation"),
        c("""
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr_p, y_tr)

        y_val_pred = model.predict(X_val_p)
        print(f"Exactitude en validation : {accuracy_score(y_val, y_val_pred):.3f}")
        print(f"Rappel des décès         : {recall_score(y_val, y_val_pred, pos_label='dead'):.3f}")
        pd.DataFrame(confusion_matrix(y_val, y_val_pred, labels=["alive", "dead"]),
                     index=["vrai alive", "vrai dead"],
                     columns=["prédit alive", "prédit dead"])
        """),
        m("""
        Comparez à la référence « toujours `alive` » (≈ 0,59) et aux scores de
        vos modèles des séances 1 à 3. Les décès manqués (vrai `dead`, prédit
        `alive`) sont les erreurs les plus lourdes cliniquement.
        """),
        m("""
        ### 4.4 Générer la soumission

        Une fois la méthode choisie, on réentraîne sur **tous** les patients
        d'entraînement, puis on prédit ceux du test.
        """),
        c("""
        X_p, X_test_p = preparer(X, X_t)
        model = LogisticRegression(max_iter=1000).fit(X_p, y)

        submission = pd.DataFrame({"patient_id": test_ids,
                                   "outcome": model.predict(X_test_p)})
        assert len(submission) == len(X_test) and submission["patient_id"].is_unique
        submission.to_csv("submission.csv", index=False)
        print(f"submission.csv : {len(submission)} prédictions")
        submission["outcome"].value_counts()
        """),
        m("""
        Le fichier contient exactement deux colonnes, `patient_id` et
        `outcome`, et une ligne par patient du test. Sur Colab, la cellule
        suivante le télécharge.
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
        ## 5. Aller plus loin

        - **Nouvelles variables** : cliniquement pertinentes (rapport urée/créatinine, seuils cliniques, interactions).
        - **La donnée manquante comme information** : un examen non prescrit en dit long ; ajoutez des indicateurs de valeur manquante.
        - **Gradient boosting** : `HistGradientBoostingClassifier`, qui accepte directement les valeurs manquantes ; réglez ses hyperparamètres en validation croisée stratifiée.
        - **Calibration** : les probabilités prédites sont-elles fiables ? `CalibrationDisplay`, `CalibratedClassifierCV`.
        - **Fuite d'information** : chaque variable est-elle connue au moment de la prédiction ? Le challenge a déjà retiré la fuite la plus évidente (le statut DNR, parfois décidé après l'admission) ; interrogez par exemple `avtisst`, moyenne de l'intensité des soins sur plusieurs jours.
        - **Interprétation** : quelles variables comptent le plus ? Les erreurs ont-elles un sens clinique ?
        """),
    ]


def build() -> dict:
    nb._COUNTER = itertools.count()     # byte-identical on every regeneration
    return nb.notebook(cells())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{NAME}.ipynb"
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
