"""Séance 5 of *Python pour la médecine* — NumPy et pandas. The two notebooks.

Builds, from one source, the exercise notebook and its correction:

- `pym-s5-exercices.ipynb`: statements, `# À vous` cells with a skeleton,
  and check cells (`assert` + "✅ Exercice N réussi");
- `pym-s5-exercices-correction.ipynb`: the same notebook with the solution
  in place of each skeleton.

Part A works on small NumPy arrays typed in the notebook; part B reads
`insuffisance_cardiaque.csv` and `covid_hospitalisations.csv.gz`; part C
cleans `sejours_hopital_brut.csv` step by step; part D pivots
`examens_biologiques.csv` and joins it to the cleaned stays. The datasets are
written by `pym_datasets.py` and read from GitHub,
`racousin/ai_for_medicine` (`URL` below).

Every number an `assert` checks was read off an executed run of the
correction on those files. A change to `pym_datasets.py` changes them:
re-execute the correction before publishing.

Writes two files and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/pym_s5.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_medicine)

COURSE = "python-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
NAME = "pym-s5-exercices"
DATA = aim_repo.raw(COURSE, "donnees/")
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

md, code = nb.md, nb.code


def build(correction: bool) -> list[dict]:
    """All cells; `correction` picks the solution or the skeleton."""

    def ex(todo: list[str], solution: list[str]) -> dict:
        if correction:
            return code(*solution)
        return code("# À vous", *todo)

    name = NAME + ("-correction" if correction else "")
    cells: list[dict] = []
    add = cells.append

    # ------------------------------------------------------------------ intro
    add(md(
        "# Séance 5 — NumPy et pandas : un export hospitalier"
        + (" (correction)" if correction else ""),
        "",
        f"[![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, name + '.ipynb')})",
        "",
        "**Objectifs**",
        "",
        "- calculer sur des tableaux NumPy sans boucle : statistiques, masques"
        " booléens, broadcasting ;",
        "- lire, inspecter, filtrer et résumer un fichier de patients avec pandas ;",
        "- nettoyer pas à pas un export hospitalier réaliste (doublons, libellés,"
        " dates, unités, sentinelles) ;",
        "- passer du format long au format large et joindre deux tableaux.",
        "",
        "**Comment travailler**",
        "",
        "- Exécutez les cellules **dans l'ordre** (Maj + Entrée).",
        "- Complétez les cellules marquées `# À vous` : remplacez les `...`.",
        "- Exécutez la cellule de vérification qui suit : elle affiche"
        " « ✅ Exercice N réussi » ou une `AssertionError` qui dit ce qui ne va pas.",
        "- Les cellules **Lire du code** : écrivez votre réponse *avant*"
        " d'exécuter.",
        "",
        "Les seuils cliniques utilisés sont des exemples d'enseignement, pas des"
        " outils de décision clinique. Les séjours et examens biologiques sont"
        " des données synthétiques construites pour le cours.",
    ))
    add(code(
        "import numpy as np",
        "import pandas as pd",
        "",
        'pd.set_option("display.max_columns", 30)',
        f'URL = "{DATA}"',
    ))

    # ------------------------------------------------------------- partie A
    add(md(
        "---",
        "",
        "## Partie A — NumPy",
        "",
        "### Exercice 1 — Pressions systoliques",
        "",
        "Dix patients d'une consultation ont une pression artérielle systolique"
        " (PAS) mesurée en mmHg. Calculez :",
        "",
        "- `pas_moy`, `pas_ecart_type`, `pas_max` : moyenne, écart type et maximum ;",
        "- `pas_kpa` : le tableau des PAS en kPa (1 mmHg = 0,133322 kPa), **sans boucle**.",
    ))
    add(code("pas = np.array([128, 151, 117, 140, 163, 122, 135, 182, 109, 147])"))
    add(ex(
        ["pas_moy = ...", "pas_ecart_type = ...", "pas_max = ...", "pas_kpa = ..."],
        ["pas_moy = pas.mean()",
         "pas_ecart_type = pas.std()",
         "pas_max = pas.max()",
         "pas_kpa = pas * 0.133322",
         "print(pas_moy, round(pas_ecart_type, 1), pas_max)",
         "print(pas_kpa.round(1))"],
    ))
    add(code(
        "assert np.isclose(pas_moy, 139.4), \"pas_moy : utilisez .mean()\"",
        "assert np.isclose(pas_ecart_type, np.std(pas)), \"pas_ecart_type : utilisez .std()\"",
        "assert pas_max == 182",
        "assert isinstance(pas_kpa, np.ndarray) and pas_kpa.shape == (10,), \\",
        "    \"pas_kpa doit être un tableau de 10 valeurs\"",
        "assert np.isclose(pas_kpa[0], 17.065216)",
        'print("✅ Exercice 1 réussi")',
    ))

    add(md(
        "### Exercice 2 — Masques booléens : hypertension",
        "",
        "Classification ESC/ESH 2018 de la pression systolique : normale haute de"
        " 130 à 139 mmHg, hypertension (HTA) à partir de 140 mmHg, grade 3 à partir"
        " de 180 mmHg. Avec le tableau `pas` de l'exercice 1, calculez :",
        "",
        "- `hta` : le masque booléen des patients hypertendus ;",
        "- `n_hta` : leur nombre, et `part_hta` : leur proportion (entre 0 et 1) ;",
        "- `pas_hta` : les valeurs de PAS des patients hypertendus ;",
        "- `n_normale_haute` : le nombre de PAS dans [130, 140[ (deux conditions,"
        " combinées avec `&`) ;",
        "- `statut` : un tableau de chaînes, `\"HTA\"` ou `\"non HTA\"`, avec `np.where`.",
    ))
    add(ex(
        ["hta = ...", "n_hta = ...", "part_hta = ...", "pas_hta = ...",
         "n_normale_haute = ...", "statut = ..."],
        ["hta = pas >= 140",
         "n_hta = hta.sum()",
         "part_hta = hta.mean()",
         "pas_hta = pas[hta]",
         "n_normale_haute = ((pas >= 130) & (pas < 140)).sum()",
         'statut = np.where(hta, "HTA", "non HTA")',
         "print(n_hta, part_hta, pas_hta, n_normale_haute)",
         "print(statut)"],
    ))
    add(code(
        "assert hta.dtype == bool, \"hta doit être un tableau de booléens (une comparaison)\"",
        "assert n_hta == 5 and np.isclose(part_hta, 0.5)",
        "assert sorted(pas_hta.tolist()) == [140, 147, 151, 163, 182]",
        "assert n_normale_haute == 1, \"une seule PAS entre 130 et 139 : 135\"",
        'assert list(statut[:2]) == ["non HTA", "HTA"]',
        'print("✅ Exercice 2 réussi")',
    ))

    add(md(
        "### Lire du code — vues et axes",
        "",
        "**Que vont afficher ces deux cellules ? Répondez avant d'exécuter.**",
    ))
    add(code(
        "a = np.array([120, 130, 140])",
        "b = a[:2]",
        "b[0] = 0",
        "print(a)",
    ))
    add(code(
        "m = np.array([[1, 2, 3],",
        "              [4, 5, 6]])",
        "print(m.shape)",
        "print(m.sum(axis=0))",
        "print(m.sum(axis=1))",
        "print(m[:, 1])",
    ))
    add(md(
        "*Explication* : un découpage NumPy est une **vue** sur les mêmes données,"
        " donc modifier `b` modifie `a` (utilisez `a[:2].copy()` pour une copie"
        " indépendante). `axis=0` somme le long des lignes : un résultat par"
        " colonne, `[5 7 9]` ; `axis=1` donne un résultat par ligne, `[ 6 15]`.",
    ))

    add(md(
        "### Exercice 3 — IMC et scores z",
        "",
        "**a.** Calculez `imc`, le tableau des IMC des cinq patients"
        " (IMC = poids en kg / taille en m²), puis `n_obeses`, le nombre de"
        " patients d'IMC ≥ 30 (seuil OMS de l'obésité).",
        "",
        "**b.** La matrice `mesures` a une ligne par patient et trois colonnes :"
        " PAS (mmHg), fréquence cardiaque (/min), glycémie (g/L). Pour comparer des"
        " variables d'unités différentes, on les centre-réduit :"
        " $z = (x - \\bar{x}) / s$, colonne par colonne. Calculez `moyennes` et"
        " `ecarts` (une valeur par colonne, avec `axis`), puis `z` **sans boucle**,"
        " grâce au broadcasting. Enfin, `plus_atypique` : l'indice (ligne) du"
        " patient dont la valeur absolue du z de PAS est la plus grande"
        " (`np.argmax`).",
    ))
    add(code(
        "poids = np.array([72, 95, 58, 110, 64.5])",
        "taille_cm = np.array([178, 170, 162, 175, 168])",
        "",
        "mesures = np.array([[128, 72, 0.95],",
        "                    [151, 88, 1.30],",
        "                    [117, 64, 0.88],",
        "                    [140, 80, 1.07],",
        "                    [182, 104, 2.10],",
        "                    [122, 70, 0.92]])",
        "mesures.shape",
    ))
    add(ex(
        ["imc = ...", "n_obeses = ...", "", "moyennes = ...", "ecarts = ...",
         "z = ...", "plus_atypique = ..."],
        ["imc = poids / (taille_cm / 100) ** 2",
         "n_obeses = (imc >= 30).sum()",
         "print(imc.round(1), n_obeses)",
         "",
         "moyennes = mesures.mean(axis=0)",
         "ecarts = mesures.std(axis=0)",
         "z = (mesures - moyennes) / ecarts",
         "plus_atypique = np.argmax(np.abs(z[:, 0]))",
         "print(z.round(2))",
         "print(plus_atypique)"],
    ))
    add(code(
        "assert imc.shape == (5,), \"imc : un IMC par patient\"",
        "assert np.isclose(imc[0], 22.7244), \"pensez à convertir la taille en mètres\"",
        "assert n_obeses == 2",
        "assert moyennes.shape == (3,), \"moyennes : une valeur par colonne (axis=0)\"",
        "assert z.shape == (6, 3)",
        "assert np.allclose(z.mean(axis=0), 0) and np.allclose(z.std(axis=0), 1)",
        "assert plus_atypique == 4",
        'print("✅ Exercice 3 réussi")',
    ))

    # ------------------------------------------------------------- partie B
    add(md(
        "---",
        "",
        "## Partie B — pandas : lire et explorer",
        "",
        "Le fichier `insuffisance_cardiaque.csv` contient 299 patients suivis"
        " après une insuffisance cardiaque (UCI *Heart failure clinical records*,"
        " Chicco et Jurman, *BMC Medical Informatics and Decision Making*, 2020,"
        " licence CC BY 4.0). Une ligne par patient ; `deces` vaut 1 si le patient"
        " est décédé pendant le suivi, `sexe_homme` vaut 1 pour un homme, `anemie`"
        " 1 en cas d'anémie, `fraction_ejection` est en %.",
        "",
        "### Exercice 4 — Charger et inspecter",
        "",
        "Lisez le fichier dans `ic` (`URL + \"insuffisance_cardiaque.csv\"`), puis"
        " affichez `head()`, `info()` et `describe()`. Calculez `n_patients`,"
        " `n_variables` (à partir de `shape`) et `taux_deces`, la proportion de"
        " patients décédés.",
    ))
    add(ex(
        ["ic = ...", "", "n_patients = ...", "n_variables = ...", "taux_deces = ..."],
        ['ic = pd.read_csv(URL + "insuffisance_cardiaque.csv")',
         "ic.info()",
         "display(ic.head())",
         "display(ic.describe())",
         "",
         "n_patients, n_variables = ic.shape",
         'taux_deces = ic["deces"].mean()',
         "print(n_patients, n_variables, round(taux_deces, 3))"],
    ))
    add(code(
        "assert isinstance(ic, pd.DataFrame)",
        "assert (n_patients, n_variables) == (299, 14)",
        "assert np.isclose(taux_deces, 96 / 299), \"la moyenne d'une colonne 0/1 est une proportion\"",
        'print("✅ Exercice 4 réussi")',
    ))

    add(md(
        "### Exercice 5 — Mortalité par groupe, filtre, classes d'âge",
        "",
        "1. `mort_sexe` et `mort_anemie` : le taux de décès selon `sexe_homme`,"
        " puis selon `anemie` (`groupby(...)[\"deces\"].mean()`).",
        "2. `fe_basse` : les patients dont la fraction d'éjection est inférieure"
        " à 30 % (dysfonction systolique sévère), et `mort_fe_basse` leur taux de"
        " décès.",
        "3. Ajoutez à `ic` une colonne `classe_age` avec `pd.cut` : classes"
        " `\"<60\"`, `\"60-74\"`, `\"75+\"`, bornes `[0, 60, 75, 120]`,"
        " `right=False`. Puis `mort_age` : le taux de décès par classe"
        " (`groupby(\"classe_age\", observed=True)`).",
        "",
        "Que concluez-vous de la comparaison des sexes ? de l'âge ?",
    ))
    add(ex(
        ["mort_sexe = ...", "mort_anemie = ...", "fe_basse = ...", "mort_fe_basse = ...",
         'ic["classe_age"] = ...', "mort_age = ..."],
        ['mort_sexe = ic.groupby("sexe_homme")["deces"].mean()',
         'mort_anemie = ic.groupby("anemie")["deces"].mean()',
         'fe_basse = ic[ic["fraction_ejection"] < 30]',
         'mort_fe_basse = fe_basse["deces"].mean()',
         'ic["classe_age"] = pd.cut(ic["age"], bins=[0, 60, 75, 120],',
         '                          labels=["<60", "60-74", "75+"], right=False)',
         'mort_age = ic.groupby("classe_age", observed=True)["deces"].mean()',
         "print(mort_sexe.round(3), mort_anemie.round(3), sep='\\n')",
         "print(len(fe_basse), round(mort_fe_basse, 3))",
         "print(mort_age.round(2))"],
    ))
    add(code(
        "assert np.isclose(mort_sexe[0], 34 / 105) and np.isclose(mort_sexe[1], 62 / 194)",
        "assert mort_anemie[1] > mort_anemie[0]",
        "assert len(fe_basse) == 59, \"fraction_ejection strictement inférieure à 30\"",
        "assert np.isclose(mort_fe_basse, 38 / 59)",
        'assert ic["classe_age"].value_counts().sort_index().tolist() == [129, 129, 41], \\',
        '    "vérifiez bins et right=False : 60 ans doit être dans 60-74"',
        'assert mort_age.idxmax() == "75+"',
        'print("✅ Exercice 5 réussi")',
    ))
    add(md(
        "*Lecture* : la mortalité est la même chez les hommes et les femmes"
        " (32 %), un peu plus élevée en cas d'anémie, deux fois plus élevée quand"
        " la fraction d'éjection est inférieure à 30 %, et atteint 61 % après"
        " 75 ans. Ce sont des comparaisons brutes, sans ajustement sur les autres"
        " variables.",
    ))

    add(md(
        "### Lire du code — loc ou iloc ?",
        "",
        "**Que va afficher cette cellule ? Répondez avant d'exécuter.**"
        " (Indice : le patient d'étiquette 0 a 75 ans ; le plus âgé en a 95.)",
    ))
    add(code(
        'tri = ic.sort_values("age", ascending=False)',
        'print(tri.loc[0, "age"])',
        'print(tri.iloc[0]["age"])',
        "print(tri.index[:3].tolist())",
    ))
    add(md(
        "*Explication* : le tri réordonne les lignes **sans changer leur index**."
        " `loc[0]` cherche l'étiquette 0 (le patient de 75 ans, où qu'il soit) ;"
        " `iloc[0]` prend la première ligne du tableau trié, le patient le plus âgé.",
    ))

    add(md(
        "### Exercice 6 — Lire un fichier compressé : options de `read_csv`",
        "",
        "`covid_hospitalisations.csv.gz` (Santé publique France, Licence Ouverte)"
        " donne, par département et par jour, le nombre de patients COVID-19"
        " hospitalisés (`hosp`) et en réanimation (`rea`). Le fichier est compressé"
        " en gzip : `read_csv` le lit directement.",
        "",
        "Lisez-le dans `cov` de sorte que :",
        "",
        "- `dep` reste du **texte** (sinon `\"01\"` devient `1` ; option `dtype`) ;",
        "- `jour` soit lu comme une **date** (option `parse_dates`).",
        "",
        "Puis calculez `pic_rea_paris`, le maximum de `rea` à Paris (`dep == \"75\"`),"
        " et `jour_pic`, le jour de ce maximum (`idxmax` donne l'étiquette de la"
        " ligne du maximum).",
    ))
    add(ex(
        ["cov = ...", "pic_rea_paris = ...", "jour_pic = ..."],
        ['cov = pd.read_csv(URL + "covid_hospitalisations.csv.gz",',
         '                  dtype={"dep": str}, parse_dates=["jour"])',
         'paris = cov[cov["dep"] == "75"]',
         'pic_rea_paris = paris["rea"].max()',
         'jour_pic = paris.loc[paris["rea"].idxmax(), "jour"]',
         "print(cov.dtypes)",
         "print(pic_rea_paris, jour_pic)"],
    ))
    add(code(
        'assert cov["dep"].iloc[0] == "01", "dep doit rester du texte : dtype={\\"dep\\": str}"',
        'assert pd.api.types.is_datetime64_any_dtype(cov["jour"]), "jour doit être une date : parse_dates"',
        "assert pic_rea_paris == 855",
        'assert jour_pic == pd.Timestamp("2020-04-08")',
        'print("✅ Exercice 6 réussi")',
    ))

    # ------------------------------------------------------------- partie C
    add(md(
        "---",
        "",
        "## Partie C — Nettoyer un export hospitalier",
        "",
        "### Exercice 7 — `sejours_hopital_brut.csv`, étape par étape",
        "",
        "Ce fichier contient des séjours hospitaliers exportés d'un logiciel,"
        " avec les défauts habituels. Exécutez la cellule ci-dessous et repérez-les"
        " avant de lire les étapes.",
    ))
    add(code(
        'brut = pd.read_csv(URL + "sejours_hopital_brut.csv")',
        "print(brut.shape)",
        "brut.info()",
        'print(brut["service"].value_counts())',
        'print(brut["sexe"].value_counts())',
        'print(brut["date_entree"].sample(5, random_state=0).tolist())',
        'brut[["age", "taille_cm", "glycemie"]].describe()',
    ))
    add(md(
        "Chaque étape modifie `sejours` et est suivie d'une vérification."
        " Exécutez-les dans l'ordre ; en cas d'erreur, réexécutez à partir de"
        " l'étape 7.1.",
        "",
        "**7.1 — Doublons.** Créez `sejours` : `brut` sans les lignes en double,"
        " puis renumérotez les lignes de 0 à 599 avec `.reset_index(drop=True)`"
        " (sans cela, certaines versions de pandas avertissent à chaque"
        " modification que `sejours` est peut-être une copie de `brut`).",
    ))
    add(ex(
        ["sejours = ..."],
        ["sejours = brut.drop_duplicates().reset_index(drop=True)",
         "print(brut.duplicated().sum(), len(sejours))"],
    ))
    add(code(
        "assert len(sejours) == 600 and sejours.duplicated().sum() == 0",
        "assert sejours.index[-1] == 599, \"renumérotez avec reset_index(drop=True)\"",
        'print("✅ Étape 7.1 réussie")',
    ))
    add(md(
        "**7.2 — Services.** Ramenez `service` aux 6 noms canoniques : Cardiologie,"
        " Pneumologie, Neurologie, Gériatrie, Chirurgie digestive, Endocrinologie."
        " Normalisez d'abord (`.str.strip().str.lower()`), regardez les variantes"
        " restantes avec `unique()`, puis complétez le dictionnaire et appliquez"
        " `.map`.",
    ))
    add(ex(
        ['print(sorted(sejours["service"].str.strip().str.lower().unique()))',
         "canonique = {",
         '    "cardio": "Cardiologie",',
         "    # ...",
         "}",
         'sejours["service"] = ...'],
        ['print(sorted(sejours["service"].str.strip().str.lower().unique()))',
         "canonique = {",
         '    "cardio": "Cardiologie", "cardiologie": "Cardiologie",',
         '    "pneumo": "Pneumologie", "pneumologie": "Pneumologie",',
         '    "neuro": "Neurologie", "neurologie": "Neurologie",',
         '    "geriatrie": "Gériatrie", "gériatrie": "Gériatrie",',
         '    "chir dig": "Chirurgie digestive", "chir. digestive": "Chirurgie digestive",',
         '    "chirurgie digestive": "Chirurgie digestive",',
         '    "endocrino": "Endocrinologie", "endocrinologie": "Endocrinologie",',
         "}",
         'sejours["service"] = sejours["service"].str.strip().str.lower().map(canonique)',
         'print(sejours["service"].value_counts())'],
    ))
    add(code(
        'assert sejours["service"].isna().sum() == 0, "une variante manque dans le dictionnaire (map donne NaN)"',
        'assert set(sejours["service"]) == {"Cardiologie", "Pneumologie", "Neurologie",',
        '                                   "Gériatrie", "Chirurgie digestive", "Endocrinologie"}',
        'assert (sejours["service"] == "Cardiologie").sum() == 108',
        'print("✅ Étape 7.2 réussie")',
    ))
    add(md(
        "**7.3 — Sexe.** Ramenez `sexe` à `\"F\"` ou `\"M\"`. Attention :"
        " `.str.upper().str[0]` est tentant, mais que donne-t-il pour `\"Homme\"` ?"
        " Utilisez un dictionnaire.",
    ))
    add(ex(
        ['sejours["sexe"] = ...'],
        ['sejours["sexe"] = sejours["sexe"].map({"F": "F", "f": "F", "Femme": "F",',
         '                                       "M": "M", "Homme": "M"})',
         'print(sejours["sexe"].value_counts())'],
    ))
    add(code(
        'assert set(sejours["sexe"]) == {"F", "M"}',
        'assert (sejours["sexe"] == "F").sum() == 310',
        'print("✅ Étape 7.3 réussie")',
    ))
    add(md(
        "**7.4 — Dates et durée de séjour.** `date_entree` mélange deux formats,"
        " `AAAA-MM-JJ` et `JJ/MM/AAAA` ; `date_sortie` est toujours en `AAAA-MM-JJ`."
        " Convertissez les deux colonnes en dates avec `pd.to_datetime` et un"
        " `format` explicite (pour `date_entree` : deux passes avec"
        " `errors=\"coerce\"`, combinées par `fillna`). Ajoutez ensuite"
        " `duree_sejour`, la durée en jours (`.dt.days`).",
        "",
        "N'utilisez pas `format=\"mixed\", dayfirst=True` : sur ce fichier, il"
        " inverse silencieusement le jour et le mois de nombreuses dates ISO.",
    ))
    add(ex(
        ['sejours["date_entree"] = ...', 'sejours["date_sortie"] = ...',
         'sejours["duree_sejour"] = ...'],
        ['iso = pd.to_datetime(sejours["date_entree"], format="%Y-%m-%d", errors="coerce")',
         'fr = pd.to_datetime(sejours["date_entree"], format="%d/%m/%Y", errors="coerce")',
         'sejours["date_entree"] = iso.fillna(fr)',
         'sejours["date_sortie"] = pd.to_datetime(sejours["date_sortie"], format="%Y-%m-%d")',
         'sejours["duree_sejour"] = (sejours["date_sortie"] - sejours["date_entree"]).dt.days',
         'print(sejours["duree_sejour"].describe())'],
    ))
    add(code(
        'assert pd.api.types.is_datetime64_any_dtype(sejours["date_entree"])',
        'assert sejours["date_entree"].isna().sum() == 0, "des dates n\'ont pas été converties"',
        'ligne = sejours[sejours["id_sejour"] == "S0525"].iloc[0]',
        'assert ligne["date_entree"] == pd.Timestamp("2025-12-26"), "26/12/2025 est le 26 décembre"',
        'ligne = sejours[sejours["id_sejour"] == "S0458"].iloc[0]',
        'assert ligne["date_entree"] == pd.Timestamp("2025-07-21")',
        'assert sejours["duree_sejour"].min() == 1 and sejours["duree_sejour"].max() == 33',
        'assert sejours["duree_sejour"].sum() == 4700',
        'print("✅ Étape 7.4 réussie")',
    ))
    add(md(
        "**7.5 — Taille en cm.** Quelques tailles ont été saisies en mètres"
        " (1.72 au lieu de 172). Avec un masque et `.loc`, multipliez-les par 100.",
    ))
    add(ex(
        ["en_metres = ...", "..."],
        ['en_metres = sejours["taille_cm"] < 3',
         'sejours.loc[en_metres, "taille_cm"] = sejours.loc[en_metres, "taille_cm"] * 100',
         'print(en_metres.sum(), sejours["taille_cm"].min())'],
    ))
    add(code(
        'assert sejours["taille_cm"].min() == 143 and sejours["taille_cm"].max() == 195',
        'assert sejours["taille_cm"].isna().sum() == 34, "les tailles manquantes doivent le rester"',
        'print("✅ Étape 7.5 réussie")',
    ))
    add(md(
        "**7.6 — Glycémie en g/L.** Certaines glycémies sont en mmol/L. On"
        " considère qu'une valeur supérieure à 4 est en mmol/L, et on la convertit"
        " en g/L (1 g/L de glucose = 5,551 mmol/L). Utilisez `np.where`.",
        "",
        "Ce seuil est un compromis : une valeur de 3,4 peut être 3,4 g/L ou"
        " 3,4 mmol/L. En pratique, on demande l'unité au service qui a fait"
        " l'export.",
    ))
    add(ex(
        ['sejours["glycemie"] = ...'],
        ['g = sejours["glycemie"]',
         'print((g > 4).sum(), "valeurs en mmol/L")',
         'sejours["glycemie"] = np.where(g > 4, g / 5.551, g)',
         'print(sejours["glycemie"].describe())'],
    ))
    add(code(
        'assert sejours["glycemie"].max() <= 4',
        'assert np.isclose(sejours["glycemie"].mean(), 1.18518, atol=1e-4)',
        'assert sejours["glycemie"].isna().sum() == 51, "np.where garde les NaN : 51 glycémies manquantes"',
        'print("✅ Étape 7.6 réussie")',
    ))
    add(md(
        "**7.7 — Âge inconnu.** L'âge 999 code « inconnu ». Remplacez-le par"
        " `np.nan`. Que serait devenu l'âge moyen sans cette correction ?",
    ))
    add(ex(
        ['sejours["age"] = ...'],
        ['print("avant :", round(sejours["age"].mean(), 1))',
         'sejours["age"] = sejours["age"].replace(999, np.nan)',
         'print("après :", round(sejours["age"].mean(), 1))'],
    ))
    add(code(
        'assert sejours["age"].isna().sum() == 6 and sejours["age"].max() == 99',
        'print("✅ Exercice 7 réussi : export nettoyé,", sejours.shape)',
    ))

    # ------------------------------------------------------------- partie D
    add(md(
        "---",
        "",
        "## Partie D — Agréger et joindre",
        "",
        "Cette partie utilise `sejours` nettoyé à l'exercice 7.",
        "",
        "### Exercice 8 — Biologie : du format long au format large, puis jointure",
        "",
        "`examens_biologiques.csv` a une ligne **par mesure** : `id_patient`,"
        " `examen` (hémoglobine, créatinine, potassium, CRP), `valeur`, `unite`.",
        "",
        "1. Lisez-le dans `examens`, puis créez `large` : une ligne par patient,"
        " une colonne par examen (`pivot`).",
        "2. `complet` : `sejours` joint à `large` sur `id_patient`, en gardant"
        " **tous** les séjours (quel `how` ?). Pensez à `large.reset_index()`.",
        "3. `duree_par_service` : la durée moyenne de séjour par service, triée"
        " par ordre décroissant.",
        "4. `resume` : un tableau par service avec trois colonnes nommées, `n`"
        " (nombre de séjours), `duree_moy` et `mortalite` (taux de décès), avec"
        " `agg`.",
    ))
    add(ex(
        ["examens = ...", "large = ...", "complet = ...", "duree_par_service = ...",
         "resume = ..."],
        ['examens = pd.read_csv(URL + "examens_biologiques.csv")',
         'large = examens.pivot(index="id_patient", columns="examen", values="valeur")',
         'complet = sejours.merge(large.reset_index(), on="id_patient", how="left")',
         'duree_par_service = (complet.groupby("service")["duree_sejour"].mean()',
         "                     .sort_values(ascending=False))",
         'resume = sejours.groupby("service").agg(n=("id_sejour", "count"),',
         '                                        duree_moy=("duree_sejour", "mean"),',
         '                                        mortalite=("deces", "mean"))',
         "print(large.shape, complet.shape)",
         "print(duree_par_service.round(2))",
         "resume"],
    ))
    add(code(
        "assert large.shape == (332, 4), \"large : 332 patients x 4 examens\"",
        "assert len(complet) == 600, \"la jointure ne doit ni perdre ni dupliquer de séjour\"",
        'assert "crp" in complet.columns',
        'assert duree_par_service.index[0] == "Gériatrie"',
        'assert np.isclose(duree_par_service["Gériatrie"], 1146 / 94)',
        'assert list(resume.columns) == ["n", "duree_moy", "mortalite"]',
        'assert resume["n"].sum() == 600',
        'print("✅ Exercice 8 réussi")',
    ))
    add(md(
        "Et si l'on joignait les séjours directement au format **long** ?"
        " Exécutez la cellule et expliquez le nombre de lignes.",
    ))
    add(code(
        'sejours.merge(examens, on="id_patient").shape',
    ))

    add(md(
        "### Lire du code — filtres enchaînés",
        "",
        "**Que va afficher cette cellule ? Les deux façons de filtrer sont-elles"
        " équivalentes ? Répondez avant d'exécuter.**",
    ))
    add(code(
        'a = sejours[sejours["age"] >= 80]',
        'b = a[a["service"] == "Gériatrie"]',
        'c = sejours[(sejours["age"] >= 80) & (sejours["service"] == "Gériatrie")]',
        "print(len(a) >= len(b))",
        "print(b.index.equals(c.index))",
        'print(sejours[sejours["age"] >= 80]["service"].value_counts().index[0])',
    ))
    add(md(
        "*Explication* : filtrer deux fois de suite revient à combiner les deux"
        " conditions par `&` ; les lignes gardées (et leur index) sont les mêmes."
        " Chaque filtre ne peut que retirer des lignes. La dernière ligne lit de"
        " gauche à droite : filtrer, prendre la colonne `service`, compter, garder"
        " l'étiquette la plus fréquente.",
    ))

    add(md(
        "### Exercice 9 — Trouvez l'erreur",
        "",
        "**a.** On veut les séjours de patients de 75 ans ou plus dont la glycémie"
        " est ≥ 1,26 g/L (seuil du diabète pour une glycémie à jeun, OMS). Le code"
        " ci-dessous lève une erreur. Lisez le message, puis corrigez.",
    ))
    add(ex(
        ['agees_hyperglycemie = sejours[sejours["age"] >= 75 and sejours["glycemie"] >= 1.26]'],
        ['agees_hyperglycemie = sejours[(sejours["age"] >= 75) & (sejours["glycemie"] >= 1.26)]',
         "print(len(agees_hyperglycemie))"],
    ))
    add(code(
        "assert len(agees_hyperglycemie) == 19",
        'print("✅ Exercice 9a réussi")',
    ))
    add(md(
        "*Pourquoi* : `and` demande **un** booléen ; une Series de 600 booléens"
        " n'est ni vraie ni fausse (« The truth value of a Series is ambiguous »)."
        " Sur des tableaux, on combine avec `&`, `|`, `~`, et chaque condition entre"
        " parenthèses, car `&` est prioritaire sur `>=`.",
        "",
        "**b.** On veut marquer les séjours de patients obèses (IMC ≥ 30)."
        " La cellule s'exécute, avec un avertissement, mais la colonne `obesite`"
        " reste entièrement à `False`. Corrigez la ligne fautive.",
    ))
    add(ex(
        ['sejours["imc"] = sejours["poids_kg"] / (sejours["taille_cm"] / 100) ** 2',
         'sejours["obesite"] = False',
         'sejours[sejours["imc"] >= 30]["obesite"] = True',
         'print(sejours["obesite"].sum())'],
        ['sejours["imc"] = sejours["poids_kg"] / (sejours["taille_cm"] / 100) ** 2',
         'sejours["obesite"] = False',
         'sejours.loc[sejours["imc"] >= 30, "obesite"] = True',
         'print(sejours["obesite"].sum())'],
    ))
    add(code(
        'assert sejours["obesite"].sum() == 135, "la modification n\'a pas atteint sejours"',
        'print("✅ Exercice 9b réussi")',
    ))
    add(md(
        "*Pourquoi* : `sejours[masque]` crée un nouveau DataFrame ;"
        " `[\"obesite\"] = True` modifie ce tableau intermédiaire, qui est aussitôt"
        " jeté. pandas le signale (`SettingWithCopyWarning` ou"
        " `ChainedAssignmentError` selon la version). Une seule opération,"
        " `sejours.loc[masque, \"obesite\"] = True`, modifie `sejours` lui-même.",
    ))

    add(md(
        "---",
        "",
        "## Pour aller plus loin",
        "",
        "### Exercice 10 — Tableau croisé, retour au format long",
        "",
        "1. `croise` : la durée moyenne de séjour par service (lignes) et par sexe"
        " (colonnes), avec les totaux (`pivot_table(..., margins=True)`).",
        "2. `long_retour` : repassez `large` au format long avec `melt`"
        " (colonnes `id_patient`, `examen`, `valeur`), puis retirez les lignes"
        " où `valeur` manque. Combien de lignes retrouvez-vous, comparé à `examens` ?",
        "3. `crp_elevee` : par service, la proportion de séjours dont la CRP"
        " dépasse 10 mg/L, calculée sur `complet`. Attention aux CRP manquantes :"
        " une comparaison avec `NaN` donne `False`, la proportion se calcule donc"
        " ici sur tous les séjours.",
    ))
    add(ex(
        ["croise = ...", "long_retour = ...", "crp_elevee = ..."],
        ['croise = sejours.pivot_table(index="service", columns="sexe",',
         '                              values="duree_sejour", aggfunc="mean", margins=True)',
         'long_retour = (large.reset_index()',
         '               .melt(id_vars="id_patient", value_name="valeur")',
         '               .dropna(subset=["valeur"]))',
         'crp_elevee = (complet["crp"] > 10).groupby(complet["service"]).mean()',
         "display(croise.round(1))",
         "print(len(long_retour), len(examens))",
         "print(crp_elevee.round(3))"],
    ))
    add(code(
        "assert croise.shape == (7, 3), \"6 services + All, F / M / All\"",
        'assert np.isclose(croise.loc["All", "All"], sejours["duree_sejour"].mean())',
        "assert len(long_retour) == len(examens) == 1178",
        'assert set(long_retour.columns) == {"id_patient", "examen", "valeur"}',
        'assert np.isclose(crp_elevee["Cardiologie"], 0.38, atol=1e-3)',
        'print("✅ Exercice 10 réussi")',
    ))
    return cells


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for correction in (False, True):
        nb._COUNTER = itertools.count()
        notebook = nb.notebook(build(correction))
        name = NAME + ("-correction" if correction else "") + ".ipynb"
        with open(OUT / name, "w", encoding="utf-8") as fh:
            json.dump(notebook, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print("wrote", OUT / name, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
