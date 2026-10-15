"""Séance 2 of *Python pour la médecine* — the exercise notebook and its correction.

Module `pym-s2-controle-et-fonctions` (conditions, boucles, fonctions, lire et
déboguer). The exercise page `exercices-s2.md` opens the student notebook in
Colab and links the correction.

Both notebooks come from ONE list of exercises: each `ex(...)` emits the
statement, then the student's skeleton (`# À vous`, signature given, body `...`)
in `pym-s2-exercices.ipynb` or the solution in its place in
`pym-s2-exercices-correction.ipynb`, then the same check cell (`assert`s that
print "✅ Exercice N réussi"). No dataset: every value is typed in the cells.

The clinical thresholds are the published ones (WHO BMI classes, ESC/ESH 2018
blood-pressure grades, Sepsis-3 qSOFA, CHA₂DS₂-VASc, paracetamol 15 mg/kg per
dose capped at 1 g, hypoglycaemia < 0.70 g/L, Mosteller body surface area);
the notebook says they are teaching examples, not decision tools.

Writes two files and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/pym_s2.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_modecine)

REPO = pathlib.Path(__file__).resolve().parents[3]
COURSE = "python-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
NAME = "pym-s2-exercices.ipynb"
CORRECTION_NAME = "pym-s2-exercices-correction.ipynb"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

md, code = nb.md, nb.code


def ex(correction: bool, statement: list[str], todo: list[str],
       solution: list[str], check: list[str],
       after: list[list[str]] | None = None) -> list[dict]:
    """One exercise: statement, skeleton or solution, optional fixed cells, check."""
    cells = [md(*statement)]
    cells.append(code(*(solution if correction else ["# À vous", *todo])))
    for extra in after or []:
        cells.append(code(*extra))
    cells.append(code(*check))
    return cells


def header(correction: bool) -> list[dict]:
    name = CORRECTION_NAME if correction else NAME
    title = ("# Séance 2 — Scores cliniques : correction" if correction
             else "# Séance 2 — Scores cliniques : conditions, boucles et fonctions")
    intro = ([
        "Ce notebook est la **correction** des exercices de la séance 2 : chaque",
        "cellule « À vous » y est remplacée par une solution. Il s'exécute de bout",
        "en bout. Ouvrez-le après avoir cherché : comparer votre code à la",
        "correction n'est utile que si vous avez d'abord écrit le vôtre.",
    ] if correction else [
        "**Objectifs** : écrire des conditions (`if / elif / else`), des boucles",
        "(`for`, `while`, `range`) et des fonctions (paramètres, valeurs par défaut,",
        "`return`) ; lire du code et prévoir ce qu'il affiche ; trouver une erreur.",
        "",
        "**Comment travailler**",
        "",
        "- Exécutez les cellules dans l'ordre (Maj + Entrée).",
        "- Complétez chaque cellule marquée `# À vous` : la signature de la fonction",
        "  est donnée, remplacez `...` par votre code.",
        "- Exécutez la cellule de **vérification** qui suit : elle affiche",
        "  « ✅ Exercice N réussi », ou une `AssertionError` qui dit ce qui ne va pas.",
        "- Pour les exercices « Lire du code », écrivez votre prédiction **avant**",
        "  d'exécuter le code.",
    ])
    return [md(
        title,
        "",
        f"[![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, name)})",
        "",
        *intro,
        "",
        "Les seuils cliniques utilisés (OMS, ESC/ESH 2018, Sepsis-3, CHA₂DS₂-VASc…)",
        "sont ceux des recommandations publiées, mais ces fonctions sont des",
        "exemples pédagogiques, **pas des outils d'aide à la décision clinique**.",
    )]


def exercises(c: bool) -> list[dict]:
    cells: list[dict] = []

    # 1 ---------------------------------------------------------------- IMC
    cells += ex(
        c,
        [
            "## Exercice 1 — IMC et classes de l'OMS",
            "",
            "L'indice de masse corporelle vaut $\\mathrm{IMC} = \\dfrac{\\mathrm{poids}}{\\mathrm{taille}^2}$",
            "(poids en kg, taille en m). L'OMS classe l'adulte en : maigreur",
            "(IMC < 18,5), normal (18,5 ≤ IMC < 25), surpoids (25 ≤ IMC < 30),",
            "obésité (IMC ≥ 30).",
            "",
            "1. Écrivez `calculer_imc(poids_kg, taille_m)` qui renvoie l'IMC.",
            "2. Écrivez `classe_imc(imc)` qui renvoie `\"maigreur\"`, `\"normal\"`,",
            "   `\"surpoids\"` ou `\"obésité\"`.",
        ],
        [
            "def calculer_imc(poids_kg, taille_m):",
            "    ...",
            "",
            "",
            "def classe_imc(imc):",
            "    ...",
        ],
        [
            "def calculer_imc(poids_kg, taille_m):",
            "    return poids_kg / taille_m ** 2",
            "",
            "",
            "def classe_imc(imc):",
            "    if imc < 18.5:",
            '        return "maigreur"',
            "    elif imc < 25:",
            '        return "normal"',
            "    elif imc < 30:",
            '        return "surpoids"',
            "    else:",
            '        return "obésité"',
        ],
        [
            "# Vérification",
            "assert calculer_imc(70, 1.75) is not None, \"calculer_imc ne renvoie rien : manque-t-il return ?\"",
            "assert round(calculer_imc(70, 1.75), 2) == 22.86",
            'assert classe_imc(17.0) == "maigreur"',
            'assert classe_imc(18.5) == "normal", "18,5 est déjà « normal »"',
            'assert classe_imc(24.99) == "normal"',
            'assert classe_imc(25) == "surpoids", "25 est déjà « surpoids »"',
            'assert classe_imc(30) == "obésité"',
            'assert classe_imc(calculer_imc(95, 1.70)) == "obésité"',
            'print("✅ Exercice 1 réussi")',
        ],
    )

    # 2 ------------------------------------------------------- tension ESC/ESH
    cells += ex(
        c,
        [
            "## Exercice 2 — Classer une pression artérielle (ESC/ESH 2018)",
            "",
            "Les recommandations ESC/ESH 2018 classent une pression artérielle mesurée",
            "au cabinet (mmHg) selon la systolique (PAS) **et** la diastolique (PAD) :",
            "",
            "| Catégorie | PAS | | PAD |",
            "|---|---|---|---|",
            "| optimale | < 120 | et | < 80 |",
            "| normale | 120–129 | et/ou | 80–84 |",
            "| normale haute | 130–139 | et/ou | 85–89 |",
            "| HTA grade 1 | 140–159 | et/ou | 90–99 |",
            "| HTA grade 2 | 160–179 | et/ou | 100–109 |",
            "| HTA grade 3 | ≥ 180 | et/ou | ≥ 110 |",
            "",
            "Quand PAS et PAD tombent dans deux catégories différentes, on retient la",
            "**plus élevée** : 150/80 est une HTA grade 1.",
            "",
            "Écrivez `rang_pas(pas)` et `rang_pad(pad)`, qui renvoient le rang de la",
            "catégorie (0 = optimale … 5 = HTA grade 3), puis `classer_tension(pas, pad)`",
            "qui renvoie le nom de la catégorie. Indice : `max(a, b)` renvoie le plus",
            "grand de deux nombres.",
        ],
        [
            "CATEGORIES = [\"optimale\", \"normale\", \"normale haute\",",
            "              \"HTA grade 1\", \"HTA grade 2\", \"HTA grade 3\"]",
            "# CATEGORIES[0] vaut \"optimale\", CATEGORIES[5] vaut \"HTA grade 3\"",
            "",
            "",
            "def rang_pas(pas):",
            "    ...",
            "",
            "",
            "def rang_pad(pad):",
            "    ...",
            "",
            "",
            "def classer_tension(pas, pad):",
            "    ...",
        ],
        [
            "CATEGORIES = [\"optimale\", \"normale\", \"normale haute\",",
            "              \"HTA grade 1\", \"HTA grade 2\", \"HTA grade 3\"]",
            "",
            "",
            "def rang_pas(pas):",
            "    if pas < 120:",
            "        return 0",
            "    elif pas < 130:",
            "        return 1",
            "    elif pas < 140:",
            "        return 2",
            "    elif pas < 160:",
            "        return 3",
            "    elif pas < 180:",
            "        return 4",
            "    else:",
            "        return 5",
            "",
            "",
            "def rang_pad(pad):",
            "    if pad < 80:",
            "        return 0",
            "    elif pad < 85:",
            "        return 1",
            "    elif pad < 90:",
            "        return 2",
            "    elif pad < 100:",
            "        return 3",
            "    elif pad < 110:",
            "        return 4",
            "    else:",
            "        return 5",
            "",
            "",
            "def classer_tension(pas, pad):",
            "    rang = max(rang_pas(pas), rang_pad(pad))",
            "    return CATEGORIES[rang]",
        ],
        [
            "# Vérification",
            "assert rang_pas(119) == 0 and rang_pas(120) == 1 and rang_pas(180) == 5",
            "assert rang_pad(79) == 0 and rang_pad(85) == 2 and rang_pad(110) == 5",
            'assert classer_tension(115, 75) == "optimale"',
            'assert classer_tension(120, 80) == "normale"',
            'assert classer_tension(118, 86) == "normale haute", "la PAD 86 suffit"',
            'assert classer_tension(150, 80) == "HTA grade 1", "la catégorie la plus élevée l\'emporte"',
            'assert classer_tension(140, 90) == "HTA grade 1"',
            'assert classer_tension(135, 105) == "HTA grade 2"',
            'assert classer_tension(185, 95) == "HTA grade 3"',
            'print("✅ Exercice 2 réussi")',
        ],
    )

    # 3 ------------------------------------------------------ lire du code (1)
    cells += ex(
        c,
        [
            "## Exercice 3 — Lire du code : booléens et conditions",
            "",
            "Lisez le code de la **deuxième** cellule ci-dessous sans l'exécuter.",
            "**Que va-t-il afficher ? Répondez avant d'exécuter**, en complétant vos",
            "prédictions dans la première cellule. Exécutez ensuite le code, puis la",
            "vérification.",
            "",
            "Rappel : la priorité est `not` > `and` > `or`, et `True` vaut 1 dans une",
            "addition.",
        ],
        [
            "prediction_a = ...        # True ou False ?",
            "prediction_b = ...        # True ou False ?",
            "prediction_c = ...        # un entier",
            "prediction_message = ...  # \"alerte\", \"surveiller\" ou \"rien\" ?",
        ],
        [
            "prediction_a = True",
            "prediction_b = True        # 90 <= 95 < 100 est vrai, not (15 < 15) aussi",
            "prediction_c = 1           # False + True + False",
            'prediction_message = "surveiller"',
        ],
        [
            "# Vérification (exécutez d'abord les deux cellules précédentes)",
            "assert ... not in (prediction_a, prediction_b, prediction_c, prediction_message), \\",
            "    \"écrivez d'abord vos quatre prédictions\"",
            "assert prediction_a == a, f\"a vaut {a}\"",
            "assert prediction_b == b, f\"b vaut {b}\"",
            "assert prediction_c == c, f\"c vaut {c}\"",
            "assert prediction_message == message, f\"message vaut {message!r}\"",
            'print("✅ Exercice 3 réussi")',
        ],
        after=[[
            "fr, pas, glasgow = 20, 95, 15",
            "a = fr >= 22 or pas <= 100",
            "b = 90 <= pas < 100 and not glasgow < 15",
            "c = (fr >= 22) + (pas <= 100) + (glasgow < 15)",
            "if c >= 2:",
            '    message = "alerte"',
            "elif c == 1:",
            '    message = "surveiller"',
            "else:",
            '    message = "rien"',
            "print(a, b, c, message)",
        ]],
    )

    # 4 ---------------------------------------------------------------- qSOFA
    cells += ex(
        c,
        [
            "## Exercice 4 — Le score qSOFA (Sepsis-3)",
            "",
            "Le *quick SOFA* (Singer et al., JAMA 2016) compte un point par critère",
            "présent chez un patient infecté :",
            "",
            "- fréquence respiratoire ≥ 22 /min ;",
            "- pression artérielle systolique ≤ 100 mmHg ;",
            "- altération de la conscience, c'est-à-dire un score de Glasgow < 15.",
            "",
            "Un score ≥ 2 identifie un patient à risque élevé de mauvaise évolution.",
            "",
            "1. Écrivez `qsofa(fr, pas, glasgow)` qui renvoie le score (0 à 3).",
            "2. Écrivez `qsofa_positif(fr, pas, glasgow)` qui renvoie `True` si le",
            "   score est ≥ 2, **en appelant** `qsofa`.",
        ],
        [
            "def qsofa(fr, pas, glasgow):",
            "    ...",
            "",
            "",
            "def qsofa_positif(fr, pas, glasgow):",
            "    ...",
        ],
        [
            "def qsofa(fr, pas, glasgow):",
            "    score = 0",
            "    if fr >= 22:",
            "        score += 1",
            "    if pas <= 100:",
            "        score += 1",
            "    if glasgow < 15:",
            "        score += 1",
            "    return score",
            "",
            "",
            "def qsofa_positif(fr, pas, glasgow):",
            "    return qsofa(fr, pas, glasgow) >= 2",
        ],
        [
            "# Vérification",
            "assert qsofa(16, 125, 15) == 0",
            "assert qsofa(22, 101, 15) == 1, \"FR = 22 compte déjà\"",
            "assert qsofa(18, 100, 15) == 1, \"PAS = 100 compte déjà\"",
            "assert qsofa(24, 95, 15) == 2, \"des if indépendants, pas une chaîne elif\"",
            "assert qsofa(30, 80, 9) == 3",
            "assert qsofa_positif(24, 95, 15) is True",
            "assert qsofa_positif(16, 125, 13) is False",
            'print("✅ Exercice 4 réussi")',
        ],
    )

    # 5 ------------------------------------------------------- CHA2DS2-VASc
    cells += ex(
        c,
        [
            "## Exercice 5 — Le score CHA₂DS₂-VASc",
            "",
            "Dans la fibrillation atriale, le score CHA₂DS₂-VASc estime le risque",
            "d'AVC (de 0 à 9) :",
            "",
            "| Critère | Points |",
            "|---|---|",
            "| **C** insuffisance cardiaque congestive | 1 |",
            "| **H** hypertension | 1 |",
            "| **A₂** âge ≥ 75 ans | 2 |",
            "| **D** diabète | 1 |",
            "| **S₂** antécédent d'AVC, d'AIT ou de thrombo-embolie | 2 |",
            "| **V** maladie vasculaire (infarctus, artériopathie, plaque aortique) | 1 |",
            "| **A** âge de 65 à 74 ans | 1 |",
            "| **Sc** sexe féminin | 1 |",
            "",
            "Écrivez la fonction `cha2ds2_vasc`. L'âge et le sexe sont obligatoires ;",
            "les antécédents sont des paramètres **nommés** qui valent `False` par",
            "défaut, si bien qu'on ne précise que ceux qui sont présents :",
            "`cha2ds2_vasc(80, femme=True, diabete=True)`.",
        ],
        [
            "def cha2ds2_vasc(age, femme, insuffisance_cardiaque=False,",
            "                 hypertension=False, diabete=False, avc_ait=False,",
            "                 maladie_vasculaire=False):",
            "    ...",
        ],
        [
            "def cha2ds2_vasc(age, femme, insuffisance_cardiaque=False,",
            "                 hypertension=False, diabete=False, avc_ait=False,",
            "                 maladie_vasculaire=False):",
            "    score = 0",
            "    if age >= 75:",
            "        score += 2",
            "    elif age >= 65:",
            "        score += 1",
            "    if femme:",
            "        score += 1",
            "    if insuffisance_cardiaque:",
            "        score += 1",
            "    if hypertension:",
            "        score += 1",
            "    if diabete:",
            "        score += 1",
            "    if avc_ait:",
            "        score += 2",
            "    if maladie_vasculaire:",
            "        score += 1",
            "    return score",
        ],
        [
            "# Vérification",
            "assert cha2ds2_vasc(50, femme=False) == 0",
            "assert cha2ds2_vasc(50, femme=True) == 1",
            "assert cha2ds2_vasc(64, False) == 0 and cha2ds2_vasc(65, False) == 1",
            "assert cha2ds2_vasc(74, False) == 1 and cha2ds2_vasc(75, False) == 2, \"l'âge ne compte qu'une fois\"",
            "assert cha2ds2_vasc(70, False, hypertension=True) == 2",
            "assert cha2ds2_vasc(80, True, diabete=True, avc_ait=True) == 6",
            "assert cha2ds2_vasc(80, True, True, True, True, True, True) == 9",
            'print("✅ Exercice 5 réussi")',
        ],
    )

    # 6 ----------------------------------------------- posologie, défauts, lambda
    cells += ex(
        c,
        [
            "## Exercice 6 — Posologie pédiatrique : valeurs par défaut",
            "",
            "Le paracétamol se prescrit chez l'enfant à 15 mg/kg par prise, sans",
            "dépasser 1 000 mg par prise.",
            "",
            "1. Écrivez `dose_paracetamol(poids_kg, mg_par_kg=15, dose_max_mg=1000)` qui",
            "   renvoie la dose d'une prise en mg, plafonnée à `dose_max_mg`. Un poids",
            "   nul ou négatif est une erreur de saisie : la fonction doit alors",
            "   **lever** une erreur, `raise ValueError(\"...\")`, plutôt que renvoyer",
            "   une dose.",
            "2. La suspension buvable pédiatrique est dosée à 2,4 % soit 24 mg/mL.",
            "   Écrivez, avec `lambda`, `volume_ml` qui convertit une dose en mg en",
            "   volume en mL.",
        ],
        [
            "def dose_paracetamol(poids_kg, mg_par_kg=15, dose_max_mg=1000):",
            "    ...",
            "",
            "",
            "volume_ml = lambda dose_mg: ...",
        ],
        [
            "def dose_paracetamol(poids_kg, mg_par_kg=15, dose_max_mg=1000):",
            "    if poids_kg <= 0:",
            "        raise ValueError(f\"poids invalide : {poids_kg} kg\")",
            "    return min(poids_kg * mg_par_kg, dose_max_mg)",
            "",
            "",
            "volume_ml = lambda dose_mg: dose_mg / 24",
        ],
        [
            "# Vérification",
            "assert dose_paracetamol(12) == 180",
            "assert dose_paracetamol(80) == 1000, \"la dose est plafonnée à 1 000 mg\"",
            "assert dose_paracetamol(20, mg_par_kg=10) == 200",
            "assert dose_paracetamol(40, dose_max_mg=500) == 500",
            "try:",
            "    dose_paracetamol(-3)",
            "except ValueError:",
            "    pass",
            "else:",
            "    raise AssertionError(\"un poids <= 0 doit lever ValueError\")",
            "assert volume_ml(180) == 7.5",
            'print("✅ Exercice 6 réussi")',
        ],
    )

    # 7 ---------------------------------------------------------- for : fièvre
    cells += ex(
        c,
        [
            "## Exercice 7 — Boucle for : compter les épisodes de fièvre",
            "",
            "Un patient a sa température mesurée toutes les 4 heures. On parle de",
            "fièvre à partir de 38,0 °C.",
            "",
            "1. `compter_mesures_fievre(temperatures, seuil=38.0)` renvoie le nombre de",
            "   mesures fébriles.",
            "2. `compter_episodes(temperatures, seuil=38.0)` renvoie le nombre",
            "   d'**épisodes** : des mesures fébriles consécutives forment un seul",
            "   épisode. Dans `[37.0, 38.2, 38.5, 37.4, 38.1]`, il y a 3 mesures",
            "   fébriles mais 2 épisodes.",
            "",
            "Indice pour la 2ᵉ question : gardez dans une variable l'état de la mesure",
            "précédente (`en_fievre = False` avant la boucle) ; un épisode commence",
            "quand on passe de « pas en fièvre » à « en fièvre ».",
        ],
        [
            "def compter_mesures_fievre(temperatures, seuil=38.0):",
            "    ...",
            "",
            "",
            "def compter_episodes(temperatures, seuil=38.0):",
            "    ...",
        ],
        [
            "def compter_mesures_fievre(temperatures, seuil=38.0):",
            "    n = 0",
            "    for t in temperatures:",
            "        if t >= seuil:",
            "            n += 1",
            "    return n",
            "",
            "",
            "def compter_episodes(temperatures, seuil=38.0):",
            "    episodes = 0",
            "    en_fievre = False",
            "    for t in temperatures:",
            "        if t >= seuil and not en_fievre:",
            "            episodes += 1",
            "        en_fievre = t >= seuil",
            "    return episodes",
        ],
        [
            "# Vérification",
            "releve = [37.0, 38.2, 38.5, 37.4, 37.9, 38.1, 36.8]",
            "assert compter_mesures_fievre(releve) == 3",
            "assert compter_episodes(releve) == 2",
            "assert compter_mesures_fievre([]) == 0 and compter_episodes([]) == 0",
            "assert compter_mesures_fievre([38.0, 38.0]) == 2, \"38,0 °C est déjà de la fièvre\"",
            "assert compter_episodes([38.0, 38.0]) == 1",
            "assert compter_episodes([38.5, 37.0, 38.5]) == 2",
            "assert compter_mesures_fievre(releve, seuil=37.5) == 4",
            'print("✅ Exercice 7 réussi")',
        ],
    )

    # 8 ------------------------------------------------------ lire du code (2)
    cells += ex(
        c,
        [
            "## Exercice 8 — Lire du code : boucles et portée",
            "",
            "Même consigne : **que va afficher ce code ? Répondez avant d'exécuter.**",
            "Tracez les boucles à la main, tour par tour, sur une feuille.",
            "",
            "Attention à `range(1, len(mesures))` et aux deux variables nommées",
            "`seuil`.",
        ],
        [
            "prediction_resultat = ...  # un entier",
            "prediction_seuil = ...     # 38.0 ou 37.5 ?",
            "prediction_k = ...         # un entier",
            "prediction_dose = ...      # un entier",
        ],
        [
            "prediction_resultat = 2   # indices 1, 2, 3 : 37.6 et 38.0 dépassent 37.5",
            "prediction_seuil = 38.0   # le seuil de la fonction est local",
            "prediction_k = 3          # 100 -> 75 -> 50 -> 25",
            "prediction_dose = 25",
        ],
        [
            "# Vérification (exécutez d'abord les deux cellules précédentes)",
            "assert ... not in (prediction_resultat, prediction_seuil, prediction_k, prediction_dose), \\",
            "    \"écrivez d'abord vos quatre prédictions\"",
            "assert prediction_resultat == resultat, f\"resultat vaut {resultat}\"",
            "assert prediction_seuil == seuil, f\"seuil vaut {seuil}\"",
            "assert prediction_k == k, f\"k vaut {k}\"",
            "assert prediction_dose == dose, f\"dose vaut {dose}\"",
            'print("✅ Exercice 8 réussi")',
        ],
        after=[[
            "seuil = 38.0",
            "",
            "def compter(mesures):",
            "    seuil = 37.5",
            "    n = 0",
            "    for i in range(1, len(mesures)):",
            "        if mesures[i] >= seuil:",
            "            n += 1",
            "    return n",
            "",
            "mesures = [38.2, 37.6, 36.9, 38.0]",
            "resultat = compter(mesures)",
            "print(resultat, seuil)",
            "",
            "k = 0",
            "dose = 100",
            "while dose > 30:",
            "    dose = dose - 25",
            "    k += 1",
            "print(k, dose)",
        ]],
    )

    # 9 --------------------------------------------------------- while : demi-vie
    cells += ex(
        c,
        [
            "## Exercice 9 — Boucle while : élimination d'un médicament",
            "",
            "Après une prise unique, la concentration plasmatique d'un médicament est",
            "divisée par deux à chaque demi-vie. On considère qu'il est éliminé quand",
            "sa concentration passe **sous** un seuil.",
            "",
            "Écrivez `temps_elimination(c0_mg_l, demi_vie_h, seuil_mg_l)` qui simule",
            "cette décroissance demi-vie par demi-vie avec une boucle `while`, et",
            "renvoie **deux valeurs** : le nombre d'heures écoulées et la concentration",
            "finale. Si la concentration de départ est déjà sous le seuil, la fonction",
            "renvoie `(0, c0_mg_l)`.",
            "",
            "Exemple : `temps_elimination(80, 6, 5)` renvoie `(30, 2.5)`",
            "(80 → 40 → 20 → 10 → 5 → 2,5 : cinq demi-vies ; 5 n'est pas sous 5).",
        ],
        [
            "def temps_elimination(c0_mg_l, demi_vie_h, seuil_mg_l):",
            "    ...",
        ],
        [
            "def temps_elimination(c0_mg_l, demi_vie_h, seuil_mg_l):",
            "    concentration = c0_mg_l",
            "    heures = 0",
            "    while concentration >= seuil_mg_l:",
            "        concentration = concentration / 2",
            "        heures += demi_vie_h",
            "    return heures, concentration",
        ],
        [
            "# Vérification",
            "assert temps_elimination(80, 6, 5) == (30, 2.5)",
            "assert temps_elimination(10, 4, 20) == (0, 10)",
            "heures, finale = temps_elimination(100, 8, 1)",
            "assert heures == 56 and finale == 0.78125, (heures, finale)",
            'print("✅ Exercice 9 réussi")',
            "",
            "# Après 5 demi-vies, il reste 1/32 ≈ 3 % de la concentration initiale :",
            "# c'est la règle usuelle « éliminé après 5 demi-vies ».",
            "print(f\"Reste après 5 demi-vies : {100 / 2 ** 5:.1f} %\")",
        ],
    )

    # 10 ------------------------------------------------------- trouvez l'erreur
    cells += ex(
        c,
        [
            "## Exercice 10 — Trouvez l'erreur",
            "",
            "Chacune des trois fonctions ci-dessous s'exécute **sans message",
            "d'erreur**, mais renvoie un résultat faux. Exécutez la vérification pour",
            "voir lesquelles échouent, trouvez le bug de chacune (un seul par",
            "fonction) et corrigez-le directement dans la cellule.",
            "",
            "- `moyenne` : moyenne d'une série de mesures ;",
            "- `est_hypoglycemie` : vrai si la glycémie est < 0,70 g/L ;",
            "- `surface_corporelle` : formule de Mosteller,",
            "  $\\sqrt{\\mathrm{taille_{cm}} \\times \\mathrm{poids_{kg}} / 3600}$ en m².",
        ],
        [
            "def moyenne(valeurs):",
            "    somme = 0",
            "    for i in range(1, len(valeurs)):",
            "        somme += valeurs[i]",
            "    return somme / len(valeurs)",
            "",
            "",
            "def est_hypoglycemie(glycemie_g_l):",
            "    resultat = False",
            "    if glycemie_g_l < 0.70:",
            "        resultat == True",
            "    return resultat",
            "",
            "",
            "def surface_corporelle(poids_kg, taille_cm):",
            "    sc = (taille_cm * poids_kg / 3600) ** 0.5",
        ],
        [
            "def moyenne(valeurs):",
            "    somme = 0",
            "    for i in range(len(valeurs)):        # range(1, ...) sautait la 1re valeur",
            "        somme += valeurs[i]",
            "    return somme / len(valeurs)",
            "",
            "",
            "def est_hypoglycemie(glycemie_g_l):",
            "    resultat = False",
            "    if glycemie_g_l < 0.70:",
            "        resultat = True                  # == comparait sans affecter",
            "    return resultat",
            "",
            "",
            "def surface_corporelle(poids_kg, taille_cm):",
            "    sc = (taille_cm * poids_kg / 3600) ** 0.5",
            "    return sc                            # sans return, la fonction renvoie None",
        ],
        [
            "# Vérification",
            "assert abs(moyenne([36.5, 37.0, 38.0]) - 37.1667) < 1e-3, moyenne([36.5, 37.0, 38.0])",
            "assert est_hypoglycemie(0.55) is True",
            "assert est_hypoglycemie(0.95) is False",
            "assert surface_corporelle(70, 175) is not None, \"surface_corporelle renvoie None\"",
            "assert round(surface_corporelle(70, 175), 2) == 1.84",
            'print("✅ Exercice 10 réussi")',
        ],
    )

    # 11 ---------------------------------------------------- pour aller plus loin
    cells += ex(
        c,
        [
            "## Exercice 11 — Pour aller plus loin : surveiller un patient heure par heure",
            "",
            "*Facultatif.* Un patient infecté est surveillé toutes les heures. Les",
            "trois séries ci-dessous donnent, pour chaque heure (indice 0, 1, 2…), sa",
            "fréquence respiratoire, sa PAS et son score de Glasgow.",
            "",
            "1. `premiere_alerte(fr, pas, glasgow)` renvoie l'indice de la **première**",
            "   heure où le qSOFA est ≥ 2 (réutilisez votre fonction de l'exercice 4),",
            "   ou `None` s'il n'y en a aucune. Utilisez `break` ou un `return` dans la",
            "   boucle.",
            "2. `heures_en_alerte(fr, pas, glasgow)` renvoie le nombre total d'heures",
            "   avec un qSOFA ≥ 2.",
            "",
            "Pour parcourir les trois séries ensemble : `for h in range(len(fr)):`",
            "puis `fr[h]`, `pas[h]`, `glasgow[h]`.",
        ],
        [
            "def premiere_alerte(fr, pas, glasgow):",
            "    ...",
            "",
            "",
            "def heures_en_alerte(fr, pas, glasgow):",
            "    ...",
        ],
        [
            "def premiere_alerte(fr, pas, glasgow):",
            "    for h in range(len(fr)):",
            "        if qsofa_positif(fr[h], pas[h], glasgow[h]):",
            "            return h",
            "    return None",
            "",
            "",
            "def heures_en_alerte(fr, pas, glasgow):",
            "    n = 0",
            "    for h in range(len(fr)):",
            "        if qsofa_positif(fr[h], pas[h], glasgow[h]):",
            "            n += 1",
            "    return n",
        ],
        [
            "# Vérification",
            "fr_h      = [18, 19, 21, 23, 24, 25, 22, 20]",
            "pas_h     = [128, 122, 115, 108, 99, 96, 104, 112]",
            "glasgow_h = [15, 15, 15, 15, 15, 14, 15, 15]",
            "assert premiere_alerte(fr_h, pas_h, glasgow_h) == 4",
            "assert heures_en_alerte(fr_h, pas_h, glasgow_h) == 2",
            "assert premiere_alerte([16, 18], [130, 125], [15, 15]) is None",
            "assert heures_en_alerte([], [], []) == 0",
            'print("✅ Exercice 11 réussi")',
        ],
    )

    cells.append(md(
        "---",
        "",
        "**Bilan.** Vous avez écrit des conditions qui reproduisent des seuils",
        "cliniques, des boucles qui parcourent des séries de mesures, et des",
        "fonctions réutilisables avec paramètres et valeurs par défaut. La séance 3",
        "introduit les structures de données (listes, dictionnaires) qui permettent",
        "de stocker un dossier patient complet.",
    ))
    return cells


def build(correction: bool) -> dict:
    return nb.notebook(header(correction) + exercises(correction))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for correction, name in ((False, NAME), (True, CORRECTION_NAME)):
        nb._COUNTER = itertools.count()
        notebook = build(correction)
        path = OUT / name
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"  wrote {path.relative_to(REPO)}  ({len(notebook['cells'])} cells)")
        print(f"        {aim_repo.colab(COURSE, name)}")


if __name__ == "__main__":
    main()
