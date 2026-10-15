"""Séance 6 of *Python pour la médecine* — the exercise notebook and its correction.

Module `pym-s6-visualisation-analyse` (lessons `matplotlib`, `seaborn`,
`analyse-exploratoire`; exercise page `exercices-s6.md`). Three parts, on three
of the course's datasets (`pym_datasets.py`):

- A. `constantes_vitales.csv`: Matplotlib, figure / axes, subplots, a threshold;
- B. `covid_hospitalisations.csv.gz`: time series with a date axis, a 7-day
  rolling mean, and a dual-axis figure to criticise;
- C. `insuffisance_cardiaque.csv`: an exploratory analysis with seaborn, then
  a Mann-Whitney test and a chi² test, and a truncated axis to criticise.

Both notebooks come from one source: `ex(...)` emits the `# À vous` skeleton in
the student notebook and the solution in the correction, followed by the same
check cell. The numbers quoted in the markdown (peaks, medians, p-values) were
read off an executed correction: a change to the data changes them.

Writes two files and nothing else (never runs `build_notebooks.main`):

    uv run python courseware/tools/notebooks/pym_s6.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)
import aim_repo  # noqa: E402  (where the notebooks live: racousin/ai_for_modecine)

REPO = pathlib.Path(__file__).resolve().parents[3]
COURSE = "python-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
NAME = "pym-s6-exercices.ipynb"
NAME_CORR = "pym-s6-exercices-correction.ipynb"
DATA = aim_repo.raw(COURSE, "donnees/")
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines()


def md(text: str) -> dict:
    return nb.md(*_lines(text))


def code(text: str) -> dict:
    return nb.code(*_lines(text))


class Builder:
    """Collects the cells of one version (student or correction)."""

    def __init__(self, correction: bool) -> None:
        self.correction = correction
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append(md(text))

    def code(self, text: str) -> None:
        self.cells.append(code(text))

    def ex(self, todo: str, solution: str, check: str | None = None) -> None:
        """The skeleton (student) or the solution (correction), then the check."""
        self.code(solution if self.correction else todo)
        if check:
            self.code(check)

    def answer(self, solution: str) -> None:
        """A written answer: an empty slot for the student, the text in the correction."""
        if self.correction:
            self.md("**Réponse.** " + textwrap.dedent(solution).strip("\n"))
        else:
            self.md("*Votre réponse :* …")


def build(correction: bool) -> dict:
    b = Builder(correction)
    this = NAME_CORR if correction else NAME
    other = NAME if correction else NAME_CORR
    if correction:
        lien = (f"Ceci est la correction. L'énoncé seul : [`{other}`]({aim_repo.colab(COURSE, other)}).")
    else:
        lien = (f"La correction est dans [`{other}`]({aim_repo.colab(COURSE, other)}) — "
                "à ouvrir après avoir cherché.")
    b.md(f"""
        # Séance 6 — Visualiser et analyser{" (correction)" if correction else ""}

        [![Open In Colab]({BADGE})]({aim_repo.colab(COURSE, this)})

        **Objectifs.** Construire des figures Matplotlib (figure, axes, subplots,
        légende, seuil), tracer des séries temporelles avec un vrai axe de dates,
        mener une analyse exploratoire avec seaborn, puis tester une différence
        entre deux groupes et l'interpréter avec prudence.

        **Données.**
        - `constantes_vitales.csv` : 12 patients de réanimation, une mesure par heure pendant 48 h (données simulées) ;
        - `covid_hospitalisations.csv.gz` : patients COVID-19 hospitalisés par département et par jour (Santé publique France, Licence Ouverte) ;
        - `insuffisance_cardiaque.csv` : 299 patients en insuffisance cardiaque (Chicco & Jurman 2020, UCI, CC BY 4.0).

        **Méthode.** Exécutez les cellules dans l'ordre. Complétez les cellules
        marquées `# À vous`, puis exécutez la cellule de vérification qui suit :
        elle affiche ✅ quand votre réponse est juste. Les questions *Lire du
        code* et *Lire un graphique* se résolvent **avant** d'exécuter quoi que ce soit.

        {lien}

        Exemples pédagogiques : aucun résultat de ce notebook n'est un outil de décision clinique.
    """)
    b.code(f"""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from scipy import stats

        URL = "{DATA}"
        sns.set_theme(style="whitegrid")

        vitaux = pd.read_csv(URL + "constantes_vitales.csv")
        covid = pd.read_csv(URL + "covid_hospitalisations.csv.gz",
                            dtype={{"dep": str}}, parse_dates=["jour"])
        ic = pd.read_csv(URL + "insuffisance_cardiaque.csv")
        print(vitaux.shape, covid.shape, ic.shape)
    """)
    b.code("vitaux.head()")

    # ------------------------------------------------------------------ A
    b.md("""
        ---
        ## Partie A — Constantes vitales en réanimation (Matplotlib)

        `vitaux` est au format **long** : une ligne par patient et par heure, avec
        la fréquence cardiaque (bpm), la pression artérielle systolique (mmHg),
        la SpO₂ (%) et la température (°C).

        ### Exercice 1 — Une courbe

        Tracez la fréquence cardiaque du patient `R01` en fonction de l'heure,
        avec l'interface objet (`fig, ax = plt.subplots()`). Nommez l'axe x
        `"Heure"`, l'axe y avec son unité (`"Fréquence cardiaque (bpm)"`), et
        donnez un titre.
    """)
    b.ex(
        todo="""
            # À vous
            p01 = ...          # les lignes du patient R01
            fig, ax = plt.subplots(figsize=(9, 3.5))
            # tracez, puis ax.set_xlabel(...), ax.set_ylabel(...), ax.set_title(...)
            ...
        """,
        solution="""
            p01 = vitaux[vitaux["id_patient"] == "R01"]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(p01["heure"], p01["frequence_cardiaque"])
            ax.set_xlabel("Heure")
            ax.set_ylabel("Fréquence cardiaque (bpm)")
            ax.set_title("Patient R01 — fréquence cardiaque sur 48 h")
        """,
        check="""
            assert isinstance(p01, pd.DataFrame) and len(p01) == 48, "p01 doit contenir les 48 lignes de R01"
            assert len(ax.get_lines()) == 1, "il faut exactement une courbe"
            assert len(ax.get_lines()[0].get_xdata()) == 48
            assert ax.get_xlabel() == "Heure"
            assert "bpm" in ax.get_ylabel(), "l'unité manque sur l'axe y"
            assert ax.get_title() != "", "ajoutez un titre"
            print("✅ Exercice 1 réussi")
        """,
    )

    b.md("""
        ### Lire du code — une grille d'axes

        Que va afficher ce code ? Répondez **avant** d'exécuter.

        ```python
        fig, axes = plt.subplots(2, 2)
        print(type(axes).__name__, axes.shape)
        print(axes[1, 0] is axes.flat[2])
        axes[0, 1].set_title("SpO2")
        print([a.get_title() for a in axes.flat])
        ```
    """)
    b.answer("""
        `ndarray (2, 2)` : `axes` est un tableau NumPy d'Axes, deux lignes et
        deux colonnes. Puis `True` : `axes.flat` parcourt la grille ligne par
        ligne (0 = haut gauche, 1 = haut droite, 2 = bas gauche, 3 = bas
        droite), donc `axes[1, 0]` est le troisième. Enfin
        `['', 'SpO2', '', '']` : seul l'axe en haut à droite a un titre.
    """)
    b.code("""
        fig, axes = plt.subplots(2, 2)
        print(type(axes).__name__, axes.shape)
        print(axes[1, 0] is axes.flat[2])
        axes[0, 1].set_title("SpO2")
        print([a.get_title() for a in axes.flat])
        plt.close(fig)
    """)

    b.md("""
        ### Exercice 2 — Les quatre constantes d'un patient

        Écrivez `tracer_constantes(df, patient)` qui crée une grille 2 × 2
        (`sharex=True`) et trace, pour ce patient, une constante par case :
        fréquence cardiaque, pression systolique, SpO₂, température. Chaque case
        a un titre (le nom de la constante) ; la figure a un titre général qui
        contient l'identifiant du patient (`fig.suptitle`). La fonction renvoie
        `fig, axes`. Appliquez-la au patient `R03`.
    """)
    b.ex(
        todo="""
            CONSTANTES = {
                "frequence_cardiaque": "Fréquence cardiaque (bpm)",
                "pression_systolique": "Pression systolique (mmHg)",
                "spo2": "SpO2 (%)",
                "temperature": "Température (°C)",
            }

            def tracer_constantes(df, patient):
                # À vous : filtrer le patient, plt.subplots(2, 2, ...), une boucle sur
                # zip(axes.flat, CONSTANTES.items()), fig.suptitle(...), fig.tight_layout()
                ...

            fig, axes = tracer_constantes(vitaux, "R03")
        """,
        solution="""
            CONSTANTES = {
                "frequence_cardiaque": "Fréquence cardiaque (bpm)",
                "pression_systolique": "Pression systolique (mmHg)",
                "spo2": "SpO2 (%)",
                "temperature": "Température (°C)",
            }

            def tracer_constantes(df, patient):
                p = df[df["id_patient"] == patient]
                fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
                for ax, (colonne, nom) in zip(axes.flat, CONSTANTES.items()):
                    ax.plot(p["heure"], p[colonne])
                    ax.set_title(nom)
                for ax in axes[1]:
                    ax.set_xlabel("Heure")
                fig.suptitle(f"Constantes vitales du patient {patient}")
                fig.tight_layout()
                return fig, axes

            fig, axes = tracer_constantes(vitaux, "R03")
        """,
        check="""
            assert fig is not None and axes is not None, "la fonction doit renvoyer fig, axes"
            assert axes.shape == (2, 2)
            assert all(len(a.get_lines()) == 1 for a in axes.flat), "une courbe par case"
            titres = [a.get_title() for a in axes.flat]
            assert all(titres) and len(set(titres)) == 4, "quatre titres différents"
            assert fig._suptitle is not None and "R03" in fig._suptitle.get_text(), "le titre général doit nommer le patient"
            spo2 = axes.flat[2].get_lines()[0].get_ydata()
            assert min(spo2) == vitaux.loc[vitaux["id_patient"] == "R03", "spo2"].min()
            print("✅ Exercice 2 réussi")
        """,
    )
    b.md("""
        Que voyez-vous chez R03 après la 20ᵉ heure ? Quelle association de signes
        cela évoque-t-il ?
    """)
    b.answer("""
        La fréquence cardiaque et la température montent, la pression
        systolique et la SpO₂ baissent : tachycardie, fièvre, hypotension et
        désaturation, le tableau d'une dégradation de type sepsis (ces données
        sont simulées pour le montrer).
    """)

    b.md("""
        ### Exercice 3 — Douze patients, trois qui se dégradent

        1. Calculez `deteriores` : la liste triée des patients dont la SpO₂ est
           descendue **sous 92 %** au moins une fois.
        2. Tracez la SpO₂ des 12 patients sur un même axe : les patients non
           détériorés en gris fin (`color="lightgrey"`, `linewidth=1`), les
           patients de `deteriores` en couleur avec un `label`, puis une ligne
           horizontale pointillée à 92 % (`ax.axhline`) et une légende.

        Le seuil de 92 % est un repère pédagogique (borne basse de la cible
        usuelle de SpO₂ chez l'adulte sans insuffisance respiratoire chronique).
    """)
    b.ex(
        todo="""
            SEUIL = 92
            deteriores = ...   # À vous : groupby + min, puis filtrer
            fig, ax = plt.subplots(figsize=(10, 4))
            # À vous : une boucle sur vitaux.groupby("id_patient"), puis ax.axhline(...)
            ...
        """,
        solution="""
            SEUIL = 92
            spo2_min = vitaux.groupby("id_patient")["spo2"].min()
            deteriores = sorted(spo2_min[spo2_min < SEUIL].index)

            fig, ax = plt.subplots(figsize=(10, 4))
            for patient, p in vitaux.groupby("id_patient"):
                if patient in deteriores:
                    ax.plot(p["heure"], p["spo2"], linewidth=2, label=patient)
                else:
                    ax.plot(p["heure"], p["spo2"], color="lightgrey", linewidth=1)
            ax.axhline(SEUIL, color="black", linestyle="--", linewidth=1,
                       label=f"Seuil {SEUIL} %")
            ax.set(xlabel="Heure", ylabel="SpO2 (%)",
                   title="SpO2 des 12 patients de réanimation")
            ax.legend(loc="lower left")
        """,
        check="""
            assert deteriores == ["R03", "R07", "R11"], f"deteriores = {deteriores}"
            lignes = ax.get_lines()
            assert len(lignes) == 13, "12 courbes + la ligne du seuil"
            assert any(list(l.get_ydata()) == [SEUIL, SEUIL] for l in lignes), "ligne du seuil absente"
            assert ax.get_legend() is not None, "ajoutez une légende"
            noms = [t.get_text() for t in ax.get_legend().get_texts()]
            assert all(p in noms for p in deteriores), "la légende doit nommer les 3 patients"
            print("✅ Exercice 3 réussi")
        """,
    )

    b.md("""
        ### Exercice 4 — Trouvez l'erreur

        Ce code doit tracer, côte à côte, la fréquence cardiaque et la
        température du patient R07. Il plante. Lisez le message d'erreur,
        expliquez-le, puis corrigez le code (en gardant `plt.subplots(1, 2)`).
    """)
    b.ex(
        todo="""
            # À vous : corrigez
            p07 = vitaux[vitaux["id_patient"] == "R07"]
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
            ax.plot(p07["heure"], p07["frequence_cardiaque"])
            ax.plot(p07["heure"], p07["temperature"])
            ax.set_xlabel("Heure")
        """,
        solution="""
            # plt.subplots(1, 2) renvoie un TABLEAU de deux Axes : un tableau NumPy
            # n'a pas de méthode plot (AttributeError). On indexe chaque case.
            p07 = vitaux[vitaux["id_patient"] == "R07"]
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
            ax[0].plot(p07["heure"], p07["frequence_cardiaque"])
            ax[0].set(xlabel="Heure", ylabel="Fréquence cardiaque (bpm)")
            ax[1].plot(p07["heure"], p07["temperature"], color="tab:red")
            ax[1].set(xlabel="Heure", ylabel="Température (°C)")
            fig.tight_layout()
        """,
        check="""
            assert len(ax) == 2 and all(len(a.get_lines()) == 1 for a in ax), "une courbe par case"
            assert all(a.get_xlabel() == "Heure" for a in ax), "nommez l'axe x des deux cases"
            print("✅ Exercice 4 réussi")
        """,
    )

    # ------------------------------------------------------------------ B
    b.md("""
        ---
        ## Partie B — Hospitalisations COVID-19 en France

        Chaque ligne de `covid` : un département (`dep`) et un jour (`jour`).
        `hosp` et `rea` comptent les patients **présents** ce jour-là à l'hôpital
        et en réanimation (un stock) ; `rad` et `dc` sont des **cumuls** depuis
        mars 2020 (retours à domicile, décès à l'hôpital).
    """)
    b.code("covid.head()")
    b.md("""
        ### Exercice 5 — La courbe nationale

        1. Construisez `national` : la somme de `hosp` et `rea` sur tous les
           départements, pour chaque jour (index = `jour`).
        2. Tracez les deux séries sur un même axe (même unité : des patients),
           avec un vrai axe de dates : `mdates.ConciseDateFormatter` sur
           `ax.xaxis`. Légende, titre, étiquette y.
        3. Calculez `pic_hosp`, la date du maximum de `hosp`.
    """)
    b.ex(
        todo="""
            national = ...      # À vous : groupby("jour") puis sum()
            pic_hosp = ...
            fig, ax = plt.subplots(figsize=(11, 4))
            # À vous : deux ax.plot avec label, puis l'axe des dates :
            # loc = mdates.AutoDateLocator(); ax.xaxis.set_major_locator(loc)
            # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ...
        """,
        solution="""
            national = covid.groupby("jour")[["hosp", "rea"]].sum()
            pic_hosp = national["hosp"].idxmax()

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(national.index, national["hosp"], label="Hospitalisés")
            ax.plot(national.index, national["rea"], label="En réanimation")
            loc = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ax.set(ylabel="Patients présents", title="COVID-19 : patients hospitalisés en France")
            ax.legend()
            print(pic_hosp.date(), national["hosp"].max())
        """,
        check="""
            assert isinstance(national.index, pd.DatetimeIndex), "jour doit être une date (parse_dates)"
            assert len(national) == covid["jour"].nunique()
            assert national["hosp"].max() == 33466 and str(pic_hosp.date()) == "2020-11-16"
            assert national["rea"].max() == 7019
            assert len(ax.get_lines()) == 2 and ax.get_legend() is not None
            assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
            print("✅ Exercice 5 réussi")
        """,
    )
    b.md("""
        Combien de vagues voyez-vous ? Le pic des réanimations a-t-il lieu à la
        même vague que le pic des hospitalisations ?
    """)
    b.answer("""
        On distingue au moins six vagues : printemps 2020 (32 131 hospitalisés
        le 14 avril), automne 2020 (33 466 le 16 novembre, le maximum), suivie
        d'un plateau jusqu'à la vague du printemps 2021 (31 262 le 12 avril),
        une petite vague à l'été 2021, l'hiver 2021-2022 (Omicron, 33 447 le
        7 février 2022), puis plusieurs vagues plus basses en 2022. Le pic des
        réanimations (7 019) a lieu dès le 8 avril 2020 : lors de la première
        vague, une plus grande part des patients hospitalisés était en
        réanimation. Avec Omicron, une part notable des patients comptés dans
        `hosp` étaient hospitalisés pour un autre motif et testés positifs.
    """)

    b.md("""
        ### Exercice 6 — Trois départements

        Comparez les hospitalisations de Paris (`75`), des Bouches-du-Rhône
        (`13`) et du Rhône (`69`). Construisez `trois` au format **large**
        (une colonne par département, index = jour) avec `pivot`, puis
        tracez-le (une courbe par colonne, légende).
    """)
    b.ex(
        todo="""
            DEPS = ["75", "13", "69"]
            trois = ...      # À vous : filtrer avec isin, puis pivot(index=..., columns=..., values=...)
            fig, ax = plt.subplots(figsize=(11, 4))
            ...
        """,
        solution="""
            DEPS = ["75", "13", "69"]
            trois = (covid[covid["dep"].isin(DEPS)]
                     .pivot(index="jour", columns="dep", values="hosp"))
            fig, ax = plt.subplots(figsize=(11, 4))
            for dep in DEPS:
                ax.plot(trois.index, trois[dep], label=dep)
            ax.set(ylabel="Patients hospitalisés", title="COVID-19 : trois départements")
            ax.legend(title="Département")
            # Équivalent seaborn, en format long :
            # sns.lineplot(data=covid[covid["dep"].isin(DEPS)], x="jour", y="hosp", hue="dep", ax=ax)
        """,
        check="""
            assert sorted(trois.columns) == ["13", "69", "75"]
            assert trois["75"].max() == 3281 and str(trois["75"].idxmax().date()) == "2020-04-14"
            assert len(ax.get_lines()) == 3 and ax.get_legend() is not None
            print("✅ Exercice 6 réussi")
        """,
    )
    b.md("""
        Le pic parisien est au printemps 2020, celui du Rhône et des
        Bouches-du-Rhône à l'automne 2020. Peut-on comparer directement les
        hauteurs des courbes pour dire quel département a été le plus touché ?
    """)
    b.answer("""
        Non : ce sont des effectifs bruts. Paris compte environ 2,1 millions
        d'habitants, le Rhône 1,9 et les Bouches-du-Rhône 2,0 millions, mais
        surtout les hôpitaux parisiens accueillent des patients d'autres
        départements (transferts). Il faudrait rapporter à la population
        (pour 100 000 habitants) et savoir où les patients résident.
    """)

    b.md("""
        ### Exercice 7 — Décès quotidiens et moyenne glissante sur 7 jours

        `dc` est un cumul. Le nombre de décès d'un jour est la différence avec la
        veille (`.diff()`). Cette série quotidienne est bruitée (moins de
        déclarations le week-end) : on la lisse par une **moyenne glissante sur
        7 jours** (`.rolling(7).mean()`).

        Construisez la série `deces_jour` (somme nationale de `dc`, puis
        `diff`) et `deces_7j`, puis tracez les deux sur le même axe : la série
        brute en gris clair, la moyenne glissante par-dessus.
    """)
    b.ex(
        todo="""
            deces_jour = ...   # À vous
            deces_7j = ...
            fig, ax = plt.subplots(figsize=(11, 4))
            ...
        """,
        solution="""
            deces_jour = covid.groupby("jour")["dc"].sum().diff()
            deces_7j = deces_jour.rolling(7).mean()

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(deces_jour.index, deces_jour, color="lightgrey", label="Quotidien")
            ax.plot(deces_7j.index, deces_7j, label="Moyenne glissante 7 jours")
            ax.set(ylabel="Décès à l'hôpital par jour", title="COVID-19 : décès quotidiens à l'hôpital")
            ax.legend()
        """,
        check="""
            assert deces_jour.isna().sum() == 1, "diff() laisse un seul NaN, le premier jour"
            assert deces_jour.max() == 605
            assert round(deces_7j.max(), 1) == 514.3
            assert str(deces_7j.idxmax().date()) == "2020-04-08"
            assert deces_7j.isna().sum() == 7, "rolling(7) : les 6 premiers jours + le NaN de diff"
            assert len(ax.get_lines()) == 2
            print("✅ Exercice 7 réussi")
        """,
    )

    b.md("""
        ### Lire un graphique — deux axes y

        Exécutez la cellule, puis répondez : que semble montrer cette figure ?
        En quoi est-elle trompeuse ?
    """)
    b.code("""
        nat = covid.groupby("jour")[["hosp", "dc"]].sum()
        d7 = nat["dc"].diff().rolling(7).mean()
        fig, ax1 = plt.subplots(figsize=(11, 4))
        ax1.plot(nat.index, nat["hosp"], color="tab:blue")
        ax1.set_ylabel("Hospitalisés", color="tab:blue")
        ax2 = ax1.twinx()                      # un second axe y, à droite
        ax2.plot(d7.index, d7, color="tab:red")
        ax2.set_ylabel("Décès par jour (7 j)", color="tab:red")
        ax2.set_ylim(0, 1200)
        ax1.set_title("Les décès ne suivent plus les hospitalisations ?");
    """)
    b.answer("""
        La figure suggère qu'à partir de 2021 les décès « décrochent » des
        hospitalisations. Mais les deux courbes ont des échelles différentes :
        celle de droite a été choisie arbitrairement (0 à 1 200) et celle de
        gauche ne part pas de zéro. En les changeant, on fait se croiser ou se
        séparer les courbes à volonté : leur position relative ne signifie rien.
        La mortalité des patients hospitalisés a peut-être réellement baissé,
        mais pour le montrer il faut tracer directement le rapport décès /
        hospitalisés. Pour comparer deux grandeurs d'unités différentes : deux
        graphiques superposés (`sharex=True`), ou chaque série rapportée à sa
        propre valeur maximale.
    """)

    # ------------------------------------------------------------------ C
    b.md("""
        ---
        ## Partie C — Analyse exploratoire : insuffisance cardiaque

        299 patients suivis après une hospitalisation pour insuffisance
        cardiaque. `deces` vaut 1 si le patient est décédé pendant le suivi.
        Unités : `fraction_ejection` en %, `creatinine` en mg/dL, `sodium` en
        mmol/L, `cpk` en UI/L, `plaquettes` en 10³/µL.

        **Question** : quelles mesures relevées à l'inclusion sont associées au décès ?
    """)
    b.code("ic.describe().T.round(2)")

    b.md("""
        ### Exercice 8 — Distributions

        1. Sur une grille 1 × 3, tracez l'histogramme (`sns.histplot`, `ax=`)
           de `age`, `fraction_ejection` et `creatinine`.
        2. Calculez `moy_creat` et `med_creat`, la moyenne et la médiane de la
           créatinine. Laquelle des trois distributions est la plus asymétrique ?
        3. Sur une nouvelle figure `fig2, ax2`, tracez la boîte à moustaches de
           `fraction_ejection` selon `deces` (`sns.boxplot`), et calculez
           `medianes_fe`, la médiane de la fraction d'éjection par groupe.
    """)
    b.ex(
        todo="""
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            # À vous : trois sns.histplot(data=ic, x=..., ax=...)
            ...
            moy_creat = ...
            med_creat = ...

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            # À vous : sns.boxplot(...)
            medianes_fe = ...
        """,
        solution="""
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            for ax, col in zip(axes, ["age", "fraction_ejection", "creatinine"]):
                sns.histplot(data=ic, x=col, bins=25, ax=ax)
            fig.tight_layout()
            moy_creat = ic["creatinine"].mean()
            med_creat = ic["creatinine"].median()
            print(f"créatinine : moyenne {moy_creat:.2f}, médiane {med_creat:.2f} mg/dL")

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=ic, x="deces", y="fraction_ejection", ax=ax2)
            ax2.set(xlabel="Décès pendant le suivi (0 = non, 1 = oui)",
                    ylabel="Fraction d'éjection (%)")
            medianes_fe = ic.groupby("deces")["fraction_ejection"].median()
            print(medianes_fe)
        """,
        check="""
            assert axes.shape == (3,) and all(len(a.patches) > 0 for a in axes), "trois histogrammes"
            assert [a.get_xlabel() for a in axes] == ["age", "fraction_ejection", "creatinine"]
            assert round(moy_creat, 2) == 1.39 and med_creat == 1.1
            assert ax2.get_ylabel() != "" and len(ax2.patches) + len(ax2.lines) > 0
            assert list(medianes_fe) == [38.0, 30.0]
            print("✅ Exercice 8 réussi")
        """,
    )
    b.answer("""
        La créatinine : moyenne 1,39 mg/dL pour une médiane de 1,1, avec une
        longue queue à droite (jusqu'à 9,4). La moyenne est tirée vers le haut
        par quelques patients à la fonction rénale très altérée ; la médiane
        décrit mieux le patient typique. La fraction d'éjection médiane est de
        38 % chez les survivants contre 30 % chez les patients décédés.
    """)

    b.md("""
        ### Lire un graphique — un axe tronqué

        Que semble montrer cette figure ? En quoi est-elle trompeuse ?
    """)
    b.code("""
        taux = ic.groupby("anemie")["deces"].mean() * 100
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.bar(["Sans anémie", "Avec anémie"], taux, color=["tab:blue", "tab:red"])
        ax.set_ylim(28, 37)
        ax.set_ylabel("Décès pendant le suivi (%)")
        ax.set_title("L'anémie, facteur de mortalité majeur");
    """)
    b.answer("""
        La barre « avec anémie » paraît cinq fois plus haute que l'autre. En
        réalité la mortalité passe de 29 % à 36 % : l'axe y commence à 28 % au
        lieu de 0, ce qui exagère l'écart. Une barre représente une quantité par
        sa longueur : son axe doit partir de zéro. Le titre affirme aussi une
        causalité et une importance que la figure ne montre pas (voir le test
        du χ² à l'exercice 10).
    """)

    b.md("""
        ### Exercice 9 — Relations entre variables

        1. Nuage de points `age` × `creatinine` coloré par `deces`
           (`sns.scatterplot`, `hue=`), sur `fig, ax`.
        2. `corr` : la matrice des corrélations de **Spearman** de toutes les
           colonnes numériques (sans `id_patient`), affichée avec `sns.heatmap`
           (`annot=True`, `fmt=".2f"`, `cmap="vlag"`, `center=0`) sur `fig3, ax3`.
        3. `var_max` : le nom de la variable dont la corrélation avec `deces`
           est la plus forte **en valeur absolue** (hors `deces` lui-même).
        4. Un `sns.pairplot` de `age`, `fraction_ejection`, `creatinine`,
           `sodium`, coloré par `deces` (stockez le résultat dans `g`).
    """)
    b.ex(
        todo="""
            fig, ax = plt.subplots(figsize=(6, 4))
            ...    # À vous : le nuage de points
            corr = ...
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ...    # À vous : la heatmap
            var_max = ...
            g = ...
        """,
        solution="""
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=ic, x="age", y="creatinine", hue="deces", alpha=0.7, ax=ax)
            ax.set(xlabel="Âge (ans)", ylabel="Créatinine (mg/dL)")

            corr = ic.drop(columns="id_patient").corr(method="spearman")
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax3)

            var_max = corr["deces"].drop("deces").abs().idxmax()
            print(var_max, round(corr.loc[var_max, "deces"], 2))

            g = sns.pairplot(ic, vars=["age", "fraction_ejection", "creatinine", "sodium"],
                             hue="deces", plot_kws={"alpha": 0.6, "s": 20})
        """,
        check="""
            assert len(ax.collections) > 0 and ax.get_legend() is not None, "nuage coloré par deces"
            assert corr.shape == (13, 13) and "id_patient" not in corr.columns
            assert round(corr.loc["creatinine", "deces"], 2) == 0.37, "corrélation de Spearman ?"
            assert len(ax3.collections) > 0, "heatmap absente"
            assert var_max == "suivi_jours"
            assert g.axes.shape == (4, 4)
            print("✅ Exercice 9 réussi")
        """,
    )
    b.md("""
        `suivi_jours` est la variable la plus corrélée au décès (−0,54). Est-ce
        un facteur de risque ? Pourquoi la créatinine est-elle plus corrélée au
        décès en Spearman (0,37) qu'en Pearson (0,29) ?
    """)
    b.answer("""
        Ce n'est pas un facteur de risque mais une conséquence : le suivi
        s'arrête au décès, donc un patient décédé tôt a forcément un suivi
        court. Cette variable ne doit pas servir à « expliquer » le décès. Pour
        la créatinine : la distribution est très asymétrique, quelques valeurs
        extrêmes pèsent lourd dans Pearson (relation linéaire) ; Spearman
        travaille sur les rangs et capte la relation monotone « plus la
        créatinine est haute, plus le décès est fréquent ».
    """)

    b.md("""
        ### Exercice 10 — Tester, puis interpréter

        1. `p_mw` : la p-valeur du test de **Mann-Whitney** comparant la
           fraction d'éjection des survivants et des patients décédés
           (`stats.mannwhitneyu`). Comparez avec le test de Welch
           (`stats.ttest_ind(..., equal_var=False)`), `p_t`.
        2. `table` : le tableau de contingence `anemie` × `deces`
           (`pd.crosstab`) ; `p_chi2` : la p-valeur du test du χ²
           (`stats.chi2_contingency`, qui renvoie un objet dont l'attribut
           `pvalue` est la p-valeur).
        3. Rédigez l'interprétation des deux résultats en trois ou quatre
           phrases, dans la cellule de texte qui suit.
    """)
    b.ex(
        todo="""
            fe_survivants = ...
            fe_deces = ...
            p_mw = ...
            p_t = ...
            table = ...
            p_chi2 = ...
        """,
        solution="""
            fe_survivants = ic.loc[ic["deces"] == 0, "fraction_ejection"]
            fe_deces = ic.loc[ic["deces"] == 1, "fraction_ejection"]
            p_mw = stats.mannwhitneyu(fe_survivants, fe_deces).pvalue
            p_t = stats.ttest_ind(fe_survivants, fe_deces, equal_var=False).pvalue
            table = pd.crosstab(ic["anemie"], ic["deces"])
            p_chi2 = stats.chi2_contingency(table).pvalue
            print(f"Mann-Whitney p = {p_mw:.1e} ; Welch p = {p_t:.1e} ; chi2 p = {p_chi2:.2f}")
            print(pd.crosstab(ic["anemie"], ic["deces"], normalize="index").round(2))
        """,
        check="""
            assert 1e-7 < p_mw < 1e-6, f"p_mw = {p_mw}"
            assert 1e-6 < p_t < 1e-4, f"p_t = {p_t}"
            assert table.shape == (2, 2) and table.values.sum() == 299
            assert round(p_chi2, 2) == 0.31, f"p_chi2 = {p_chi2}"
            print("✅ Exercice 10 réussi")
        """,
    )
    b.answer("""
        Fraction d'éjection : p ≈ 7 × 10⁻⁷ (Mann-Whitney), du même ordre avec
        Welch. Un tel écart entre les groupes serait très improbable s'il n'y
        avait aucune différence : la FEVG est plus basse chez les patients
        décédés (médiane 30 % contre 38 %), ce qui est cohérent avec la gravité
        d'une dysfonction systolique. Le test ne dit ni la taille de l'effet
        (on la lit dans les médianes) ni une causalité. Anémie : 36 % de décès
        contre 29 %, p ≈ 0,31 ; les données ne permettent pas de conclure à une
        différence. Ce n'est pas la preuve d'une absence d'effet : avec 299
        patients, un écart de cet ordre peut passer inaperçu. Enfin, ce sont des
        données d'observation : un facteur de confusion (âge, sévérité) peut
        expliquer une association.
    """)

    # ------------------------------------------------------------------ bonus
    b.md("""
        ---
        ### Pour aller plus loin — tester six variables à la fois

        Construisez `resume`, un DataFrame indexé par les variables `age`,
        `cpk`, `fraction_ejection`, `plaquettes`, `creatinine`, `sodium`, avec
        les colonnes `mediane_survivants`, `mediane_deces`, `p_valeur`
        (Mann-Whitney) et `p_bonferroni` (= `p_valeur × 6`, plafonnée à 1),
        trié par `p_valeur` croissante. Tracez ensuite une figure 2 × 3 avec un
        `sns.boxplot` par variable (`x="deces"`), la p-valeur dans le titre de
        chaque case. Pourquoi corriger les p-valeurs quand on fait plusieurs tests ?
    """)
    b.ex(
        todo="""
            VARIABLES = ["age", "cpk", "fraction_ejection", "plaquettes", "creatinine", "sodium"]
            resume = ...   # À vous
        """,
        solution="""
            VARIABLES = ["age", "cpk", "fraction_ejection", "plaquettes", "creatinine", "sodium"]
            lignes = []
            for v in VARIABLES:
                a = ic.loc[ic["deces"] == 0, v]
                d = ic.loc[ic["deces"] == 1, v]
                lignes.append({"variable": v, "mediane_survivants": a.median(),
                               "mediane_deces": d.median(),
                               "p_valeur": stats.mannwhitneyu(a, d).pvalue})
            resume = pd.DataFrame(lignes).set_index("variable").sort_values("p_valeur")
            resume["p_bonferroni"] = (resume["p_valeur"] * len(VARIABLES)).clip(upper=1)
            display(resume)

            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            for ax, v in zip(axes.flat, VARIABLES):
                sns.boxplot(data=ic, x="deces", y=v, ax=ax)
                ax.set_title(f"{v} (p = {resume.loc[v, 'p_valeur']:.1e})")
            fig.tight_layout()
        """,
        check="""
            assert list(resume.index[:2]) == ["creatinine", "fraction_ejection"]
            assert (resume["p_bonferroni"] < 0.05).sum() == 4
            assert resume.loc["cpk", "p_bonferroni"] == 1
            print("✅ Pour aller plus loin réussi")
        """,
    )
    b.answer("""
        Au seuil 0,05, chaque test a 5 % de chances de donner un « résultat
        significatif » alors qu'il n'y a pas de différence. Avec six tests, la
        probabilité d'au moins un faux positif monte à environ 26 %
        (1 − 0,95⁶). Bonferroni multiplie chaque p-valeur par le nombre de
        tests pour garder ce risque global sous 5 %. Ici, créatinine, fraction
        d'éjection, âge et sodium restent significatifs après correction ;
        plaquettes et CPK non.
    """)
    return nb.notebook(b.cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, correction in [(NAME, False), (NAME_CORR, True)]:
        nb._COUNTER = itertools.count()     # byte-identical on every regeneration
        path = OUT / name
        path.write_text(json.dumps(build(correction), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(REPO)}\n      {aim_repo.colab(COURSE, name)}")


if __name__ == "__main__":
    main()
