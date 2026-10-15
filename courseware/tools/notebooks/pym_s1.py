"""Python pour la médecine, séance 1 — Premiers pas avec Python. The Colab notebooks.

Writes the exercise notebook and its correction from one source:

    pym-s1-exercices.ipynb             (student: skeletons, check cells kept)
    pym-s1-exercices-correction.ipynb  (the solutions in place of the skeletons)

Clinical calculations on numbers and strings only: functions are taught in
séance 2, so each exercise gives the `def` line and the student completes the
`return` line (the notebook's first section says so). Conditions are not taught
yet either: the sex factor of Cockcroft-Gault and CKD-EPI is written with a
boolean used as an integer (`1 - 0.15 * femme`), which is the point of that
exercise.

Constants: 1 g/L of glucose = 5.551 mmol/L (M = 180.16 g/mol), 1 mg/dL of
creatinine = 88.4 µmol/L; OMS BMI classes; ESC/ESH 2018 hypertension
(≥ 140 and/or ≥ 90 mmHg); Cockcroft-Gault 1976; Mosteller 1987; CKD-EPI 2021
(Inker et al., NEJM 2021, race-free).

Writes two files and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/pym_s1.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_modecine)

COURSE = "python-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
NAME = "pym-s1-exercices.ipynb"
CORRECTION = "pym-s1-exercices-correction.ipynb"

md, code = nb.md, nb.code


def build(solution: bool) -> dict:
    """The whole notebook; `solution` swaps every skeleton for its answer."""

    def ex(todo: list[str], answer: list[str]) -> dict:
        """The one code cell that differs between the two notebooks."""
        return code(*(answer if solution else todo))

    name = CORRECTION if solution else NAME
    title = ("# Séance 1 — Calculs cliniques (correction)" if solution
             else "# Séance 1 — Calculs cliniques")
    cells: list[dict] = [
        md(
            title,
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({aim_repo.colab(COURSE, name)})",
            "",
            "**Objectifs** : calculer avec des nombres (`int`, `float`), écrire des",
            "conditions avec des booléens, convertir entre types, découper et",
            "nettoyer des chaînes de caractères, et présenter un résultat avec une",
            "f-string. Tous les exemples sont médicaux ; ce sont des **exemples",
            "pédagogiques**, pas des outils d'aide à la décision clinique.",
            "",
            "**Comment travailler**",
            "",
            "1. Fichier → *Enregistrer une copie dans Drive*, sinon votre travail est perdu.",
            "2. Exécutez les cellules **dans l'ordre** (Maj + Entrée).",
            "3. Complétez les cellules marquées `# À vous` : remplacez les `...`.",
            "4. Exécutez la cellule de vérification qui suit : elle affiche",
            "   « ✅ Exercice N réussi » ou une `AssertionError` qui dit ce qui ne va pas.",
            "5. En cas de comportement étrange : Exécution → *Redémarrer la session",
            "   et tout exécuter*.",
            "",
            "Les exercices « Lire du code » se font **sans exécuter** d'abord : vous",
            "écrivez votre prédiction, puis vous vérifiez.",
        ),
        md(
            "## Mode d'emploi : compléter une fonction",
            "",
            "Les fonctions sont l'objet de la séance 2. Pour pouvoir vérifier vos",
            "calculs sur plusieurs patients, chaque exercice vous donne une fonction",
            "déjà déclarée : la ligne `def` nomme le calcul et ses **paramètres**,",
            "et vous complétez la ligne `return`, qui donne le **résultat**. Voici",
            "un exemple complet :",
        ),
        code(
            "def kg_vers_livres(poids_kg):",
            "    return poids_kg * 2.20462",
            "",
            "",
            "print(kg_vers_livres(70))      # on « appelle » la fonction avec 70",
            "print(kg_vers_livres(3.2))     # puis avec 3.2 : un nouveau-né",
        ),
        md(
            "Dans le corps de la fonction, `poids_kg` vaut la valeur passée entre",
            "parenthèses. Les lignes de la fonction sont **indentées** de 4 espaces.",
        ),
    ]

    # ------------------------------------------------------------ exercice 1
    cells += [
        md(
            "## Exercice 1 — Indice de masse corporelle",
            "",
            "L'IMC est le poids (kg) divisé par le carré de la taille (m) :",
            "",
            "$$\\mathrm{IMC} = \\frac{\\mathrm{poids}}{\\mathrm{taille}^2}$$",
            "",
            "Complétez `calculer_imc` pour qu'elle renvoie l'IMC **arrondi à une",
            "décimale** (fonction `round`).",
        ),
        ex(
            [
                "# À vous",
                "def calculer_imc(poids_kg, taille_m):",
                "    return ...",
            ],
            [
                "def calculer_imc(poids_kg, taille_m):",
                "    return round(poids_kg / taille_m ** 2, 1)",
            ],
        ),
        code(
            "assert calculer_imc(70, 1.75) == 22.9, \"70 kg, 1,75 m : l'IMC attendu est 22.9\"",
            "assert calculer_imc(95, 1.68) == 33.7, \"95 kg, 1,68 m : l'IMC attendu est 33.7\"",
            "assert calculer_imc(3.4, 0.50) == 13.6, \"un nouveau-né de 3,4 kg et 50 cm : 13.6\"",
            "print(\"✅ Exercice 1 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 2
    cells += [
        md(
            "## Exercice 2 — Booléens : corpulence et tension",
            "",
            "L'OMS classe l'IMC de l'adulte en : maigreur (< 18,5), normal",
            "(18,5 à < 25), surpoids (25 à < 30), obésité (≥ 30).",
            "",
            "L'hypertension artérielle (ESC/ESH 2018, mesure au cabinet) est définie",
            "par une systolique ≥ 140 mmHg **et/ou** une diastolique ≥ 90 mmHg.",
            "",
            "Chaque fonction renvoie un booléen (`True` / `False`) : une comparaison",
            "suffit, sans `if`. Python accepte les encadrements `a <= x < b`.",
        ),
        ex(
            [
                "# À vous",
                "def est_en_surpoids(imc):",
                "    return ...",
                "",
                "",
                "def est_obese(imc):",
                "    return ...",
                "",
                "",
                "def est_hypertendu(pas, pad):",
                "    return ...",
            ],
            [
                "def est_en_surpoids(imc):",
                "    return 25 <= imc < 30",
                "",
                "",
                "def est_obese(imc):",
                "    return imc >= 30",
                "",
                "",
                "def est_hypertendu(pas, pad):",
                "    return pas >= 140 or pad >= 90",
            ],
        ),
        code(
            "assert est_en_surpoids(27.3) is True and est_en_surpoids(25) is True",
            "assert est_en_surpoids(24.9) is False and est_en_surpoids(30) is False, \"30 est déjà l'obésité\"",
            "assert est_obese(30) is True and est_obese(29.9) is False",
            "assert est_hypertendu(142, 85) is True, \"systolique seule élevée : HTA\"",
            "assert est_hypertendu(130, 92) is True, \"diastolique seule élevée : HTA\"",
            "assert est_hypertendu(139, 89) is False",
            "print(\"✅ Exercice 2 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 3
    cells += [
        md(
            "## Exercice 3 — Conversions d'unités",
            "",
            "Les laboratoires rendent les mêmes analyses dans des unités différentes :",
            "",
            "- glycémie : 1 g/L = 5,551 mmol/L (masse molaire du glucose : 180,16 g/mol) ;",
            "- créatininémie : 1 mg/dL = 88,4 µmol/L.",
            "",
            "Complétez les quatre conversions (sans arrondir). Attention au sens :",
            "multiplier ou diviser ?",
        ),
        ex(
            [
                "# À vous",
                "def glycemie_g_vers_mmol(glycemie_g_l):",
                "    return ...",
                "",
                "",
                "def glycemie_mmol_vers_g(glycemie_mmol_l):",
                "    return ...",
                "",
                "",
                "def creatinine_umol_vers_mgdl(creatinine_umol_l):",
                "    return ...",
                "",
                "",
                "def creatinine_mgdl_vers_umol(creatinine_mg_dl):",
                "    return ...",
            ],
            [
                "def glycemie_g_vers_mmol(glycemie_g_l):",
                "    return glycemie_g_l * 5.551",
                "",
                "",
                "def glycemie_mmol_vers_g(glycemie_mmol_l):",
                "    return glycemie_mmol_l / 5.551",
                "",
                "",
                "def creatinine_umol_vers_mgdl(creatinine_umol_l):",
                "    return creatinine_umol_l / 88.4",
                "",
                "",
                "def creatinine_mgdl_vers_umol(creatinine_mg_dl):",
                "    return creatinine_mg_dl * 88.4",
            ],
        ),
        code(
            "import math",
            "",
            "assert math.isclose(glycemie_g_vers_mmol(1.26), 6.99426), \"1,26 g/L (seuil du diabète) ≈ 7,0 mmol/L\"",
            "assert math.isclose(glycemie_mmol_vers_g(5.551), 1.0)",
            "assert math.isclose(creatinine_umol_vers_mgdl(176.8), 2.0)",
            "assert math.isclose(creatinine_mgdl_vers_umol(1.2), 106.08)",
            "print(\"✅ Exercice 3 réussi\")",
        ),
        md(
            "Remarquez `math.isclose` dans la vérification : on ne compare pas deux",
            "flottants avec `==` (voir l'exercice suivant).",
        ),
    ]

    # ------------------------------------------------------------ exercice 4
    cells += [
        md(
            "## Exercice 4 — Lire du code : divisions et flottants",
            "",
            "Une perfusion dure 135 minutes. **Que va afficher ce code ? Répondez",
            "avant d'exécuter**, en remplissant les prédictions ci-dessous (un",
            "nombre, un booléen…), puis exécutez le code et la vérification.",
            "",
            "```python",
            "duree_min = 135",
            "a = duree_min // 60",
            "b = duree_min % 60",
            "c = duree_min / 60",
            "d = int(7.9)",
            "e = 0.1 + 0.2 == 0.3",
            "f = 2 ** 3 ** 2",
            "```",
        ),
        ex(
            [
                "# À vous : vos prédictions, avant d'exécuter le code",
                "prediction_a = ...",
                "prediction_b = ...",
                "prediction_c = ...",
                "prediction_d = ...",
                "prediction_e = ...",
                "prediction_f = ...",
            ],
            [
                "prediction_a = 2          # quotient entier : 2 heures",
                "prediction_b = 15         # reste : 15 minutes",
                "prediction_c = 2.25       # la division / donne un float",
                "prediction_d = 7          # int() tronque, il n'arrondit pas",
                "prediction_e = False      # 0.1 + 0.2 vaut 0.30000000000000004",
                "prediction_f = 512        # ** se lit de droite à gauche : 2 ** 9",
            ],
        ),
        code(
            "duree_min = 135",
            "a = duree_min // 60",
            "b = duree_min % 60",
            "c = duree_min / 60",
            "d = int(7.9)",
            "e = 0.1 + 0.2 == 0.3",
            "f = 2 ** 3 ** 2",
            "print(a, b, c, d, e, f)",
            "print(0.1 + 0.2)",
        ),
        code(
            "for nom, prediction, valeur in [(\"a\", prediction_a, a), (\"b\", prediction_b, b),",
            "                                (\"c\", prediction_c, c), (\"d\", prediction_d, d),",
            "                                (\"e\", prediction_e, e), (\"f\", prediction_f, f)]:",
            "    assert prediction == valeur, f\"{nom} vaut {valeur!r}, pas {prediction!r}\"",
            "print(\"✅ Exercice 4 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 5
    cells += [
        md(
            "## Exercice 5 — Clairance de la créatinine (Cockcroft-Gault)",
            "",
            "La formule de Cockcroft et Gault (1976) estime la clairance de la",
            "créatinine en mL/min, avec la créatininémie en mg/dL :",
            "",
            "$$\\mathrm{Cl} = \\frac{(140 - \\mathrm{age}) \\times \\mathrm{poids}}{72 \\times \\mathrm{creatinine}} \\times 0.85 \\;\\mathrm{(femme)}$$",
            "",
            "Le dossier donne la créatininémie en µmol/L : réutilisez",
            "`creatinine_umol_vers_mgdl` de l'exercice 3. Le paramètre `femme` est",
            "un booléen ; sans `if`, utilisez le fait que `True` vaut 1 et `False`",
            "vaut 0 : le facteur `1 - 0.15 * femme` vaut 0,85 pour une femme et 1",
            "pour un homme. Arrondissez à une décimale.",
        ),
        ex(
            [
                "# À vous",
                "def clairance_cockcroft(age, poids_kg, creatinine_umol_l, femme):",
                "    return ...",
            ],
            [
                "def clairance_cockcroft(age, poids_kg, creatinine_umol_l, femme):",
                "    creatinine_mg_dl = creatinine_umol_vers_mgdl(creatinine_umol_l)",
                "    facteur_sexe = 1 - 0.15 * femme",
                "    return round((140 - age) * poids_kg / (72 * creatinine_mg_dl) * facteur_sexe, 1)",
            ],
        ),
        code(
            "assert clairance_cockcroft(60, 72, 88.4, femme=False) == 80.0, \"homme, 60 ans, 72 kg, 1 mg/dL : 80 mL/min\"",
            "assert clairance_cockcroft(60, 72, 88.4, femme=True) == 68.0, \"la même patiente : 80 × 0,85\"",
            "assert clairance_cockcroft(82, 58, 115, femme=True) == 30.5",
            "print(\"✅ Exercice 5 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 6
    cells += [
        md(
            "## Exercice 6 — Surface corporelle et dose de chimiothérapie",
            "",
            "Beaucoup de doses de chimiothérapie sont prescrites en mg/m² de surface",
            "corporelle (SC). La formule de Mosteller (1987) :",
            "",
            "$$\\mathrm{SC}\\,(\\mathrm{m}^2) = \\sqrt{\\frac{\\mathrm{taille\\,(cm)} \\times \\mathrm{poids\\,(kg)}}{3600}}$$",
            "",
            "1. `surface_corporelle` renvoie la SC arrondie à 2 décimales. La racine",
            "   carrée s'écrit `x ** 0.5` (ou `math.sqrt(x)`).",
            "2. `dose_mg` renvoie la dose (mg) pour une prescription en mg/m², arrondie",
            "   à l'entier (`round(x)`), calculée à partir de la SC **arrondie**.",
        ),
        ex(
            [
                "# À vous",
                "def surface_corporelle(taille_cm, poids_kg):",
                "    return ...",
                "",
                "",
                "def dose_mg(dose_mg_par_m2, taille_cm, poids_kg):",
                "    return ...",
            ],
            [
                "def surface_corporelle(taille_cm, poids_kg):",
                "    return round((taille_cm * poids_kg / 3600) ** 0.5, 2)",
                "",
                "",
                "def dose_mg(dose_mg_par_m2, taille_cm, poids_kg):",
                "    return round(dose_mg_par_m2 * surface_corporelle(taille_cm, poids_kg))",
            ],
        ),
        code(
            "assert surface_corporelle(180, 80) == 2.0",
            "assert surface_corporelle(165, 62) == 1.69",
            "assert dose_mg(75, 165, 62) == 127, \"75 mg/m² × 1,69 m² = 126,75 → 127 mg\"",
            "assert isinstance(dose_mg(75, 165, 62), int), \"round(x) sans décimale renvoie un int\"",
            "print(\"✅ Exercice 6 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 7
    cells += [
        md(
            "## Exercice 7 — Lire du code : découper un code CIM-10",
            "",
            "Le code CIM-10 `E11.9` désigne un diabète de type 2 sans complication.",
            "**Que va afficher ce code ? Répondez avant d'exécuter.** Écrivez les",
            "chaînes entre guillemets, les nombres et booléens sans.",
            "",
            "```python",
            "code_cim = \"E11.9\"",
            "a = code_cim[0]",
            "b = code_cim[:3]",
            "c = code_cim[-1]",
            "d = len(code_cim)",
            "e = \"11\" in code_cim",
            "f = code_cim.split(\".\")[1]",
            "g = code_cim[1:3] + code_cim[1:3]",
            "h = int(code_cim[1:3]) + int(code_cim[1:3])",
            "```",
        ),
        ex(
            [
                "# À vous : vos prédictions, avant d'exécuter le code",
                "prediction_a = ...",
                "prediction_b = ...",
                "prediction_c = ...",
                "prediction_d = ...",
                "prediction_e = ...",
                "prediction_f = ...",
                "prediction_g = ...",
                "prediction_h = ...",
            ],
            [
                "prediction_a = \"E\"",
                "prediction_b = \"E11\"     # indices 0, 1, 2 ; le 3 est exclu",
                "prediction_c = \"9\"       # un caractère, donc une chaîne",
                "prediction_d = 5",
                "prediction_e = True",
                "prediction_f = \"9\"       # split donne ['E11', '9']",
                "prediction_g = \"1111\"    # + sur des chaînes : concaténation",
                "prediction_h = 22        # + sur des entiers : addition",
            ],
        ),
        code(
            "code_cim = \"E11.9\"",
            "a = code_cim[0]",
            "b = code_cim[:3]",
            "c = code_cim[-1]",
            "d = len(code_cim)",
            "e = \"11\" in code_cim",
            "f = code_cim.split(\".\")[1]",
            "g = code_cim[1:3] + code_cim[1:3]",
            "h = int(code_cim[1:3]) + int(code_cim[1:3])",
            "print(repr(a), repr(b), repr(c), d, e, repr(f), repr(g), h)",
        ),
        code(
            "for nom, prediction, valeur in [(\"a\", prediction_a, a), (\"b\", prediction_b, b),",
            "                                (\"c\", prediction_c, c), (\"d\", prediction_d, d),",
            "                                (\"e\", prediction_e, e), (\"f\", prediction_f, f),",
            "                                (\"g\", prediction_g, g), (\"h\", prediction_h, h)]:",
            "    assert prediction == valeur, f\"{nom} vaut {valeur!r}, pas {prediction!r}\"",
            "print(\"✅ Exercice 7 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 8
    cells += [
        md(
            "## Exercice 8 — Extraire des valeurs d'une saisie texte",
            "",
            "Un export du dossier patient contient des valeurs saisies à la main.",
            "",
            "1. `lire_systolique` et `lire_diastolique` reçoivent une chaîne de la",
            "   forme `\"TA: 142/91 mmHg\"` et renvoient l'entier correspondant",
            "   (`split`, indexation, `int`).",
            "2. `lire_decimal` reçoit un nombre écrit à la française, éventuellement",
            "   entouré d'espaces (`\" 1,26 \"`), et renvoie le `float` (`strip`,",
            "   `replace`, `float`).",
            "3. `normaliser_code_cim` reçoit un code saisi n'importe comment",
            "   (`\" i21.4\"`) et le renvoie en majuscules, sans espaces autour.",
        ),
        ex(
            [
                "# À vous",
                "def lire_systolique(texte):",
                "    return ...",
                "",
                "",
                "def lire_diastolique(texte):",
                "    return ...",
                "",
                "",
                "def lire_decimal(texte):",
                "    return ...",
                "",
                "",
                "def normaliser_code_cim(texte):",
                "    return ...",
            ],
            [
                "def lire_systolique(texte):",
                "    return int(texte.split(\" \")[1].split(\"/\")[0])",
                "",
                "",
                "def lire_diastolique(texte):",
                "    return int(texte.split(\" \")[1].split(\"/\")[1])",
                "",
                "",
                "def lire_decimal(texte):",
                "    return float(texte.strip().replace(\",\", \".\"))",
                "",
                "",
                "def normaliser_code_cim(texte):",
                "    return texte.strip().upper()",
            ],
        ),
        code(
            "assert lire_systolique(\"TA: 142/91 mmHg\") == 142",
            "assert lire_diastolique(\"TA: 142/91 mmHg\") == 91",
            "assert lire_systolique(\"TA: 98/61 mmHg\") == 98, \"le nombre de chiffres peut varier\"",
            "assert isinstance(lire_diastolique(\"TA: 98/61 mmHg\"), int), \"il faut un int, pas un str\"",
            "assert lire_decimal(\" 1,26 \") == 1.26 and lire_decimal(\"7\") == 7.0",
            "assert normaliser_code_cim(\" i21.4\") == \"I21.4\"",
            "assert est_hypertendu(lire_systolique(\"TA: 142/91 mmHg\"), lire_diastolique(\"TA: 142/91 mmHg\"))",
            "print(\"✅ Exercice 8 réussi\")",
        ),
    ]

    # ------------------------------------------------------------ exercice 9
    cells += [
        md(
            "## Exercice 9 — Trouvez l'erreur",
            "",
            "Ce code devait calculer la clairance d'un homme de 67 ans, 70 kg, avec",
            "une créatininémie de 106 µmol/L, et l'IMC du même patient (1,72 m). Il",
            "contient **quatre** erreurs : trois font planter le programme, la",
            "quatrième donne un résultat faux sans message d'erreur. Exécutez-le,",
            "lisez les messages, puis corrigez-le dans la cellule `# À vous`.",
            "",
            "```python",
            "age = \"67\"",
            "poids_kg = 70",
            "taille_m = 1.72",
            "creatinine_umol_l = \"106\"",
            "clairance = (140 - age) * poids_kg / (72 * creatinine_umol_l / 88.4)",
            "imc = poids_kg / taille_m * taille_m",
            "message = \"Clairance : \" + clairance + \" mL/min, IMC : \" + imc",
            "```",
        ),
        ex(
            [
                "# À vous : corrigez le code (gardez les noms clairance, imc et message)",
                "age = \"67\"",
                "poids_kg = 70",
                "taille_m = 1.72",
                "creatinine_umol_l = \"106\"",
                "clairance = (140 - age) * poids_kg / (72 * creatinine_umol_l / 88.4)",
                "imc = poids_kg / taille_m * taille_m",
                "message = \"Clairance : \" + clairance + \" mL/min, IMC : \" + imc",
                "print(message)",
            ],
            [
                "age = 67                      # 1. un nombre, pas une chaîne",
                "poids_kg = 70",
                "taille_m = 1.72",
                "creatinine_umol_l = 106       # 2. idem",
                "clairance = (140 - age) * poids_kg / (72 * creatinine_umol_l / 88.4)",
                "imc = poids_kg / taille_m ** 2   # 4. / puis * : 70 / 1,72 × 1,72 = 70",
                "message = \"Clairance : \" + str(round(clairance, 1)) + \" mL/min, IMC : \" + str(round(imc, 1))",
                "# 3. on ne concatène pas str et float : str(), ou mieux une f-string :",
                "message = f\"Clairance : {clairance:.1f} mL/min, IMC : {imc:.1f}\"",
                "print(message)",
            ],
        ),
        code(
            "assert isinstance(age, int) and isinstance(creatinine_umol_l, (int, float))",
            "assert math.isclose(clairance, 59.2, abs_tol=0.05), f\"clairance attendue ≈ 59.2 mL/min, obtenu {clairance}\"",
            "assert math.isclose(imc, 23.66, abs_tol=0.01), f\"IMC attendu ≈ 23.7, obtenu {imc} : priorité des opérateurs ?\"",
            "assert message == \"Clairance : 59.2 mL/min, IMC : 23.7\", f\"message : {message!r}\"",
            "print(\"✅ Exercice 9 réussi\")",
        ),
    ]

    # ----------------------------------------------------------- exercice 10
    cells += [
        md(
            "## Exercice 10 — Un compte rendu avec des f-strings",
            "",
            "Écrivez `compte_rendu`, qui renvoie **une** chaîne de trois lignes",
            "(séparées par `\\n`), en réutilisant vos fonctions des exercices 1, 5",
            "et 6. Pour une femme de 74 ans, 61 kg, 1,58 m, créatininémie 97 µmol/L,",
            "le résultat attendu est exactement :",
            "",
            "```text",
            "Patiente : Mme Durand, 74 ans",
            "IMC : 24.4 kg/m² | SC : 1.64 m²",
            "Clairance (Cockcroft-Gault) : 43.3 mL/min",
            "```",
            "",
            "La première ligne dit « Patient » pour un homme. Comme à l'exercice 5,",
            "sans `if` : `\"Patiente\"[:7 + femme]` vaut `\"Patient\"` pour un homme",
            "(`femme` vaut `False`, donc 0) et `\"Patiente\"` pour une femme. La",
            "clairance s'affiche avec une décimale (`:.1f`).",
        ),
        ex(
            [
                "# À vous",
                "def compte_rendu(nom, age, poids_kg, taille_m, creatinine_umol_l, femme):",
                "    return ...",
                "",
                "",
                "print(compte_rendu(\"Mme Durand\", 74, 61, 1.58, 97, femme=True))",
            ],
            [
                "def compte_rendu(nom, age, poids_kg, taille_m, creatinine_umol_l, femme):",
                "    imc = calculer_imc(poids_kg, taille_m)",
                "    sc = surface_corporelle(taille_m * 100, poids_kg)",
                "    clairance = clairance_cockcroft(age, poids_kg, creatinine_umol_l, femme)",
                "    return (f\"{'Patiente'[:7 + femme]} : {nom}, {age} ans\\n\"",
                "            f\"IMC : {imc} kg/m² | SC : {sc} m²\\n\"",
                "            f\"Clairance (Cockcroft-Gault) : {clairance:.1f} mL/min\")",
                "",
                "",
                "print(compte_rendu(\"Mme Durand\", 74, 61, 1.58, 97, femme=True))",
            ],
        ),
        code(
            "attendu = (\"Patiente : Mme Durand, 74 ans\\n\"",
            "           \"IMC : 24.4 kg/m² | SC : 1.64 m²\\n\"",
            "           \"Clairance (Cockcroft-Gault) : 43.3 mL/min\")",
            "assert compte_rendu(\"Mme Durand\", 74, 61, 1.58, 97, femme=True) == attendu, \"comparez ligne à ligne avec l'énoncé\"",
            "assert compte_rendu(\"M. Leroy\", 60, 72, 1.80, 88.4, femme=False).startswith(\"Patient : M. Leroy\")",
            "assert compte_rendu(\"M. Leroy\", 60, 72, 1.80, 88.4, femme=False).endswith(\": 80.0 mL/min\")",
            "print(\"✅ Exercice 10 réussi\")",
        ),
    ]

    # ------------------------------------------------- pour aller plus loin
    cells += [
        md(
            "## Pour aller plus loin — Le DFG estimé par CKD-EPI 2021",
            "",
            "*Facultatif.* Les recommandations actuelles (KDIGO) estiment la fonction",
            "rénale par le débit de filtration glomérulaire (DFG) de l'équation",
            "CKD-EPI 2021 (Inker et al., *NEJM* 2021), en mL/min/1,73 m², avec la",
            "créatininémie $S$ en mg/dL :",
            "",
            "$$\\mathrm{DFG} = 142 \\times \\min(S/\\kappa, 1)^{\\alpha} \\times \\max(S/\\kappa, 1)^{-1.200} \\times 0.9938^{\\mathrm{age}} \\times 1.012\\;\\mathrm{(femme)}$$",
            "",
            "avec $\\kappa = 0.7$ et $\\alpha = -0.241$ pour une femme, $\\kappa = 0.9$",
            "et $\\alpha = -0.302$ pour un homme. `min` et `max` sont des fonctions de",
            "Python (`min(3, 1)` vaut `1`).",
            "",
            "Écrivez `dfg_ckd_epi(age, creatinine_umol_l, femme)`, arrondi à l'entier,",
            "toujours sans `if` : exprimez $\\kappa$, $\\alpha$ et le facteur 1,012 à",
            "partir de `femme` (par exemple `kappa = 0.9 - 0.2 * femme`). Puis",
            "comparez avec la clairance de Cockcroft-Gault pour Mme Durand.",
        ),
        ex(
            [
                "# À vous",
                "def dfg_ckd_epi(age, creatinine_umol_l, femme):",
                "    return ...",
            ],
            [
                "def dfg_ckd_epi(age, creatinine_umol_l, femme):",
                "    s = creatinine_umol_vers_mgdl(creatinine_umol_l)",
                "    kappa = 0.9 - 0.2 * femme",
                "    alpha = -0.302 + 0.061 * femme",
                "    facteur_sexe = 1 + 0.012 * femme",
                "    dfg = (142 * min(s / kappa, 1) ** alpha * max(s / kappa, 1) ** -1.200",
                "           * 0.9938 ** age * facteur_sexe)",
                "    return round(dfg)",
                "",
                "",
                "print(f\"Mme Durand : DFG {dfg_ckd_epi(74, 97, True)} mL/min/1,73 m², \"",
                "      f\"Cockcroft-Gault {clairance_cockcroft(74, 61, 97, True)} mL/min\")",
            ],
        ),
        code(
            "assert dfg_ckd_epi(60, 88.4, femme=False) == 86",
            "assert dfg_ckd_epi(74, 97, femme=True) == 53",
            "assert dfg_ckd_epi(45, 44.2, femme=True) == 118, \"créatinine sous κ : c'est min() qui joue\"",
            "print(\"✅ Pour aller plus loin réussi\")",
        ),
        md(
            "Les deux estimations diffèrent : Cockcroft-Gault dépend du poids et",
            "estime une clairance en mL/min, CKD-EPI ne dépend pas du poids et",
            "donne un DFG rapporté à 1,73 m² de surface corporelle.",
        ),
    ]
    return nb.notebook(cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for solution, name in ((False, NAME), (True, CORRECTION)):
        nb._COUNTER = itertools.count()
        with open(OUT / name, "w", encoding="utf-8") as fh:
            json.dump(build(solution), fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
