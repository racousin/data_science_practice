"""Séance 4 — Objets et modules: the exercise notebook and its correction.

Course *Python pour la médecine* (`content/python-medecine`, module
`pym-s4-objets-et-modules`, lesson `exercices-s4`). One source builds both
notebooks: each exercise is written once by `ex(...)`, which puts the skeleton
in the student notebook and the solution in the correction; the check cells
(asserts printing "✅ Exercice N réussi") are identical in both.

Writes two files and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/pym_s4.py

Verify the correction runs top to bottom (it reads two CSVs from GitHub,
`racousin/ai_for_medicine`, and
writes `calculs_medicaux.py` in the working directory):

    uv run --with nbclient --with ipykernel python -c "import nbformat,nbclient; \
nb=nbformat.read('ai_for_medicine/python-medecine/notebooks/pym-s4-exercices-correction.ipynb',4); \
nbclient.NotebookClient(nb, timeout=300, kernel_name='python3').execute(); print('OK')"
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
COURSE = "python-medecine"
OUT = aim_repo.out_dir(COURSE, "notebooks")
NAME = "pym-s4-exercices.ipynb"
NAME_CORR = "pym-s4-exercices-correction.ipynb"
DONNEES = aim_repo.raw(COURSE, "donnees/")
BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"


def M(text: str) -> dict:
    return nb.md(*textwrap.dedent(text).strip("\n").splitlines())


def C(text: str) -> dict:
    return nb.code(*textwrap.dedent(text).strip("\n").splitlines())


def check(n: int | str, body: str) -> dict:
    """Check cell: the asserts, then the success line."""
    return C(textwrap.dedent(body).strip("\n")
             + f'\nprint("✅ Exercice {n} réussi")')


def ex(correction: bool, statement: str, todo: str, solution: str,
       checks: str | None, n: int | str) -> list[dict]:
    cells = [M(statement)]
    if correction:
        cells.append(C(solution))
    else:
        cells.append(C("# À vous\n" + textwrap.dedent(todo).strip("\n")))
    if checks:
        cells.append(check(n, checks))
    return cells


def lire(correction: bool, titre: str, intro: str, snippet: str,
         reponse: str) -> list[dict]:
    """A 'Lire du code' item: predict the output, then run. The correction
    adds the answer after the run."""
    cells = [
        M(f"""
        ### {titre}

        {intro}

        **Que va afficher ce code ? Répondez avant d'exécuter**, en écrivant
        votre prédiction dans la cellule de texte en dessous, puis lancez la
        cellule et comparez.
        """),
        C(snippet),
        M("*Ma prédiction :* ..."),
    ]
    if correction:
        cells.append(M("**Réponse.** " + textwrap.dedent(reponse).strip()))
    return cells


# --------------------------------------------------------------------------- #
def build(correction: bool) -> dict:
    url = aim_repo.colab(COURSE, NAME_CORR if correction else NAME)
    titre = "Séance 4 — Exercices : modéliser un patient"
    if correction:
        titre += " (correction)"
    cells = [
        M(f"""
        # {titre}

        {BADGE}({url})

        **Objectifs.** Écrire des classes qui représentent des objets médicaux
        (patient, mesure, ordonnance, examen), les protéger par des propriétés,
        les brancher sur le langage avec les méthodes spéciales, les organiser par
        héritage, puis ranger des fonctions dans un module que l'on importe.

        **Comment travailler.**
        - Exécutez les cellules **dans l'ordre** (Maj + Entrée).
        - Les cellules marquées `# À vous` contiennent un squelette à compléter :
          remplacez les `...` par votre code.
        - Chaque exercice est suivi d'une cellule de **vérification** : elle affiche
          `✅ Exercice N réussi` si votre code est juste, et une `AssertionError`
          (ou une autre erreur) sinon. Ne la modifiez pas.
        - Les items « Lire du code » se font **sans écrire de code** : prédisez la
          sortie, puis exécutez.

        Les seuils et formules utilisés sont des exemples pédagogiques, pas des
        outils de décision clinique.
        """),
        C(f"""
        import csv
        import io
        import urllib.request
        from datetime import date, datetime

        DONNEES = "{DONNEES}"


        def lire_csv(nom):
            \"\"\"Télécharge un CSV du cours et renvoie la liste de ses lignes (dictionnaires).\"\"\"
            with urllib.request.urlopen(DONNEES + nom) as reponse:
                texte = reponse.read().decode("utf-8")
            return list(csv.DictReader(io.StringIO(texte)))
        """),
    ]

    # ---------------------------------------------------------------- Ex 1
    cells += ex(correction, """
        ## Exercice 1 — La classe `Patient`

        Un dossier patient regroupe une identité et des mesures. Écrivez une classe
        `Patient` dont le constructeur reçoit `nom` (str), `date_naissance` (un objet
        `date`), `poids_kg` et `taille_m`, et les range dans des attributs du même nom.

        Ajoutez une méthode `imc()` qui renvoie l'indice de masse corporelle
        IMC = poids / taille² (kg/m²), arrondi à 1 décimale avec `round`.
        """, """
        class Patient:
            def __init__(self, nom, date_naissance, poids_kg, taille_m):
                ...

            def imc(self):
                ...
        """, """
        class Patient:
            def __init__(self, nom, date_naissance, poids_kg, taille_m):
                self.nom = nom
                self.date_naissance = date_naissance
                self.poids_kg = poids_kg
                self.taille_m = taille_m

            def imc(self):
                return round(self.poids_kg / self.taille_m ** 2, 1)
        """, """
        p1 = Patient("Durand", date(1958, 3, 14), 72, 1.80)
        p2 = Patient("Martin", date(1990, 11, 2), 58, 1.64)
        assert p1.nom == "Durand" and p1.date_naissance == date(1958, 3, 14)
        assert p2.poids_kg == 58 and p2.taille_m == 1.64
        assert p1.imc() == 22.2, p1.imc()
        assert p2.imc() == 21.6, p2.imc()
        assert isinstance(p1, Patient) and p1 is not p2
        """, 1)

    # ---------------------------------------------------------------- Lire A
    cells += lire(correction, "Lire du code A — attributs de classe et d'instance",
                  "Un service compte ses lits avec un attribut de classe.", """
        class Lit:
            service = "Cardiologie"
            nb_lits = 0

            def __init__(self, numero):
                self.numero = numero
                Lit.nb_lits += 1

        a = Lit(12)
        b = Lit(14)
        b.service = "Réanimation"
        print(a.service, b.service, Lit.service)
        print(Lit.nb_lits, a.nb_lits, b.numero)
        """, """
        `Cardiologie Réanimation Cardiologie` puis `2 2 14`. `b.service = ...`
        crée un attribut d'instance propre à `b`, qui masque l'attribut de classe ;
        `a` et la classe ne changent pas. `Lit.nb_lits += 1` modifie bien l'attribut
        de classe (on passe par `Lit`), partagé par toutes les instances : `a.nb_lits`
        le lit aussi. Si l'on avait écrit `self.nb_lits += 1`, chaque lit aurait eu
        son propre compteur à 1.
        """)

    # ---------------------------------------------------------------- Ex 2
    cells += ex(correction, """
        ## Exercice 2 — Trouvez l'erreur

        Une perfusion de `volume_ml` passée en `duree_h` heures a un débit de
        volume / durée (mL/h). La classe ci-dessous s'exécute sans erreur, mais
        `Perfusion(1000, 8).debit()` échoue. Trouvez les **deux** erreurs (relisez le
        cours : qu'est-ce qui manque ?) et corrigez-les.
        """, """
        class Perfusion:
            def __init__(self, volume_ml, duree_h):
                self.volume_ml = volume_ml
                duree_h = duree_h

            def debit(volume_ml, duree_h):
                return volume_ml / duree_h
        """, """
        class Perfusion:
            def __init__(self, volume_ml, duree_h):
                self.volume_ml = volume_ml
                self.duree_h = duree_h          # erreur 1 : self. manquant

            def debit(self):                    # erreur 2 : self manquant
                return self.volume_ml / self.duree_h
        """, """
        perf = Perfusion(1000, 8)
        assert perf.duree_h == 8, "la durée doit être un attribut de l'objet"
        assert perf.debit() == 125.0
        assert Perfusion(500, 4).debit() == 125.0
        """, 2)
    if correction:
        cells.append(M("""
        Sans `self.`, `duree_h = duree_h` affecte une variable locale qui disparaît à
        la fin de `__init__`. Et `def debit(volume_ml, duree_h)` reçoit l'objet dans
        son premier paramètre : `perf.debit()` lève `TypeError: debit() missing 1
        required positional argument`.
        """))

    # ---------------------------------------------------------------- Ex 3
    cells += ex(correction, """
        ## Exercice 3 — Un âge calculé et un poids protégé

        Reprenez `Patient` et ajoutez :

        1. une méthode `age_au(date_ref)` qui renvoie l'âge en années révolues à la
           date `date_ref` (l'anniversaire doit être passé pour compter l'année) ;
        2. une **propriété** `age` (en lecture seule) qui renvoie `age_au(date.today())` ;
        3. une propriété `poids_kg` avec un **setter** qui lève `ValueError` si le
           poids est `<= 0`. Le poids est stocké dans l'attribut interne `_poids_kg`,
           et le constructeur passe lui aussi par le setter.

        Indice : `(ref.month, ref.day) < (naissance.month, naissance.day)` vaut
        `True` si l'anniversaire n'est pas encore passé, et `True` compte comme 1.
        """, """
        class Patient:
            def __init__(self, nom, date_naissance, poids_kg, taille_m):
                self.nom = nom
                self.date_naissance = date_naissance
                self.poids_kg = poids_kg      # passe par le setter
                self.taille_m = taille_m

            def imc(self):
                return round(self.poids_kg / self.taille_m ** 2, 1)

            def age_au(self, date_ref):
                ...

            @property
            def age(self):
                ...

            @property
            def poids_kg(self):
                ...

            @poids_kg.setter
            def poids_kg(self, valeur):
                ...
        """, """
        class Patient:
            def __init__(self, nom, date_naissance, poids_kg, taille_m):
                self.nom = nom
                self.date_naissance = date_naissance
                self.poids_kg = poids_kg      # passe par le setter
                self.taille_m = taille_m

            def imc(self):
                return round(self.poids_kg / self.taille_m ** 2, 1)

            def age_au(self, date_ref):
                n = self.date_naissance
                pas_encore = (date_ref.month, date_ref.day) < (n.month, n.day)
                return date_ref.year - n.year - pas_encore

            @property
            def age(self):
                return self.age_au(date.today())

            @property
            def poids_kg(self):
                return self._poids_kg

            @poids_kg.setter
            def poids_kg(self, valeur):
                if valeur <= 0:
                    raise ValueError(f"poids invalide : {valeur}")
                self._poids_kg = valeur
        """, """
        p = Patient("Durand", date(1980, 5, 20), 70, 1.75)
        assert p.age_au(date(2026, 5, 19)) == 45
        assert p.age_au(date(2026, 5, 20)) == 46
        assert p.age_au(date(2026, 12, 31)) == 46
        assert isinstance(p.age, int) and p.age == p.age_au(date.today())
        p.poids_kg = 68
        assert p._poids_kg == 68 and p.imc() == 22.2
        try:
            p.poids_kg = -3
            raise AssertionError("un poids négatif doit lever ValueError")
        except ValueError:
            pass
        assert p.poids_kg == 68, "le poids ne doit pas changer après un refus"
        try:
            Patient("Erreur", date(2000, 1, 1), 0, 1.70)
            raise AssertionError("le constructeur doit aussi refuser un poids nul")
        except ValueError:
            pass
        try:
            p.age = 30
            raise AssertionError("age doit être en lecture seule")
        except AttributeError:
            pass
        """, 3)

    # ---------------------------------------------------------------- Ex 4
    cells += ex(correction, """
        ## Exercice 4 — Une `Mesure` qui s'affiche et s'additionne

        Un résultat de laboratoire n'a de sens qu'avec son unité. Écrivez une classe
        `Mesure(valeur, unite)` avec :

        - `__str__` → `"5.4 mmol/L"` (valeur, espace, unité) ;
        - `__repr__` → `"Mesure(5.4, 'mmol/L')"` (astuce : `{self.unite!r}` met les guillemets) ;
        - `__eq__` → vrai si même valeur **et** même unité ;
        - `__add__` → une **nouvelle** `Mesure` de même unité, ou `ValueError` si les
          unités diffèrent (on n'additionne pas des g/L et des mmol/L).
        - une méthode statique `glycemie_g_vers_mmol(g_par_l)` qui convertit une
          glycémie de g/L en mmol/L : multiplier par 5,551 (masse molaire du glucose
          180,16 g/mol), arrondir à 2 décimales.
        """, """
        class Mesure:
            def __init__(self, valeur, unite):
                ...

            def __str__(self):
                ...

            def __repr__(self):
                ...

            def __eq__(self, autre):
                ...

            def __add__(self, autre):
                ...

            @staticmethod
            def glycemie_g_vers_mmol(g_par_l):
                ...
        """, """
        class Mesure:
            def __init__(self, valeur, unite):
                self.valeur = valeur
                self.unite = unite

            def __str__(self):
                return f"{self.valeur} {self.unite}"

            def __repr__(self):
                return f"Mesure({self.valeur!r}, {self.unite!r})"

            def __eq__(self, autre):
                return self.valeur == autre.valeur and self.unite == autre.unite

            def __add__(self, autre):
                if self.unite != autre.unite:
                    raise ValueError(f"unités différentes : {self.unite} et {autre.unite}")
                return Mesure(self.valeur + autre.valeur, self.unite)

            @staticmethod
            def glycemie_g_vers_mmol(g_par_l):
                return round(g_par_l * 5.551, 2)
        """, """
        g = Mesure(5.4, "mmol/L")
        assert str(g) == "5.4 mmol/L" and f"{g}" == "5.4 mmol/L"
        assert repr(g) == "Mesure(5.4, 'mmol/L')"
        assert str([g]) == "[Mesure(5.4, 'mmol/L')]"
        assert g == Mesure(5.4, "mmol/L") and g != Mesure(5.4, "g/L")
        total = Mesure(250, "mL") + Mesure(500, "mL")
        assert total == Mesure(750, "mL") and isinstance(total, Mesure)
        try:
            Mesure(1.2, "g/L") + g
            raise AssertionError("des unités différentes doivent lever ValueError")
        except ValueError:
            pass
        assert Mesure.glycemie_g_vers_mmol(1.0) == 5.55
        assert Mesure.glycemie_g_vers_mmol(1.26) == 6.99
        """, 4)

    # ---------------------------------------------------------------- Lire B
    cells += lire(correction, "Lire du code B — égalité, identité et mutabilité",
                  "Cette classe ne définit pas `__eq__`.", """
        class Temperature:
            def __init__(self, celsius):
                self.celsius = celsius

        t1 = Temperature(38.5)
        t2 = Temperature(38.5)
        t3 = t1
        t3.celsius = 39.2
        print(t1 == t2, t1 is t3, t1.celsius, t2.celsius)
        """, """
        `False True 39.2 38.5`. Sans `__eq__`, `==` compare les identifiants, comme
        `is` : `t1` et `t2` sont deux objets distincts, donc différents, même avec la
        même valeur. `t3 = t1` ne copie pas l'objet : `t3` est un second nom pour
        `t1`, et modifier `t3.celsius` modifie `t1` (les objets sont mutables).
        """)

    # ---------------------------------------------------------------- Ex 5
    cells += ex(correction, """
        ## Exercice 5 — Trier la file des urgences

        Aux urgences, l'infirmier·ère d'accueil attribue un niveau de tri de 1 (le
        plus urgent) à 5 (échelle FRENCH, SFMU 2018). On examine d'abord le niveau le
        plus urgent, et à niveau égal le patient arrivé le premier.

        Écrivez `PatientUrgence(nom, tri, arrivee)` (`arrivee` est un `datetime`) :

        - le constructeur lève `ValueError` si `tri` n'est pas un entier de 1 à 5 ;
        - `__lt__(self, autre)` : vrai si `self` doit passer **avant** `autre` ;
        - `__eq__` : vrai si même niveau et même heure d'arrivée ;
        - `__repr__` → `"PatientUrgence('Durand', tri=2)"`.

        `sorted(file)` donne alors l'ordre de passage. Indice : les tuples se
        comparent élément par élément, `(2, h1) < (2, h2)` compare les heures.
        """, """
        class PatientUrgence:
            def __init__(self, nom, tri, arrivee):
                ...

            def __lt__(self, autre):
                ...

            def __eq__(self, autre):
                ...

            def __repr__(self):
                ...
        """, """
        class PatientUrgence:
            def __init__(self, nom, tri, arrivee):
                if tri not in (1, 2, 3, 4, 5):
                    raise ValueError(f"niveau de tri invalide : {tri}")
                self.nom = nom
                self.tri = tri
                self.arrivee = arrivee

            def __lt__(self, autre):
                return (self.tri, self.arrivee) < (autre.tri, autre.arrivee)

            def __eq__(self, autre):
                return (self.tri, self.arrivee) == (autre.tri, autre.arrivee)

            def __repr__(self):
                return f"PatientUrgence({self.nom!r}, tri={self.tri})"
        """, """
        file = [
            PatientUrgence("Durand", 3, datetime(2026, 10, 15, 8, 5)),
            PatientUrgence("Martin", 1, datetime(2026, 10, 15, 8, 40)),
            PatientUrgence("Nguyen", 3, datetime(2026, 10, 15, 7, 50)),
            PatientUrgence("Petit", 5, datetime(2026, 10, 15, 7, 0)),
            PatientUrgence("Moreau", 2, datetime(2026, 10, 15, 8, 30)),
        ]
        ordre = [p.nom for p in sorted(file)]
        assert ordre == ["Martin", "Moreau", "Nguyen", "Durand", "Petit"], ordre
        assert min(file).nom == "Martin"
        assert repr(file[0]) == "PatientUrgence('Durand', tri=3)"
        assert PatientUrgence("A", 2, datetime(2026, 1, 1)) == PatientUrgence("B", 2, datetime(2026, 1, 1))
        for mauvais in (0, 6, "2"):
            try:
                PatientUrgence("X", mauvais, datetime(2026, 1, 1))
                raise AssertionError(f"tri={mauvais!r} doit lever ValueError")
            except ValueError:
                pass
        """, 5)

    # ---------------------------------------------------------------- Ex 6
    cells += ex(correction, """
        ## Exercice 6 — Une ordonnance qui contient des médicaments

        1. `Medicament(nom, dose_mg, prises_par_jour)` avec :
           - une propriété `dose_journaliere_mg` (dose × prises) ;
           - une **méthode de classe** `depuis_ligne(ligne)`, constructeur alternatif
             qui lit une ligne de texte `"paracetamol;1000;3"` (convertir les nombres
             avec `int`) ;
           - une **méthode statique** `dose_pediatrique(mg_par_kg, poids_kg)` qui
             renvoie la dose d'une prise en mg (ex. paracétamol 15 mg/kg par prise
             chez l'enfant, d'après le RCP).
        2. `Ordonnance(nom_patient)` qui contient une liste de `Medicament` :
           - `ajouter(medicament)` ;
           - `__len__` → le nombre de médicaments ;
           - `__contains__(nom)` → permet d'écrire `"paracetamol" in ordonnance` ;
           - `__str__` → une ligne par médicament : `"- paracetamol : 1000 mg x 3/j"`,
             précédées de `"Ordonnance de <nom>"`.
        """, """
        class Medicament:
            def __init__(self, nom, dose_mg, prises_par_jour):
                ...

            @property
            def dose_journaliere_mg(self):
                ...

            @classmethod
            def depuis_ligne(cls, ligne):
                ...

            @staticmethod
            def dose_pediatrique(mg_par_kg, poids_kg):
                ...


        class Ordonnance:
            def __init__(self, nom_patient):
                ...

            def ajouter(self, medicament):
                ...

            def __len__(self):
                ...

            def __contains__(self, nom):
                ...

            def __str__(self):
                ...
        """, """
        class Medicament:
            def __init__(self, nom, dose_mg, prises_par_jour):
                self.nom = nom
                self.dose_mg = dose_mg
                self.prises_par_jour = prises_par_jour

            @property
            def dose_journaliere_mg(self):
                return self.dose_mg * self.prises_par_jour

            @classmethod
            def depuis_ligne(cls, ligne):
                nom, dose, prises = ligne.strip().split(";")
                return cls(nom, int(dose), int(prises))

            @staticmethod
            def dose_pediatrique(mg_par_kg, poids_kg):
                return mg_par_kg * poids_kg


        class Ordonnance:
            def __init__(self, nom_patient):
                self.nom_patient = nom_patient
                self.medicaments = []

            def ajouter(self, medicament):
                self.medicaments.append(medicament)

            def __len__(self):
                return len(self.medicaments)

            def __contains__(self, nom):
                return any(m.nom == nom for m in self.medicaments)

            def __str__(self):
                lignes = [f"Ordonnance de {self.nom_patient}"]
                for m in self.medicaments:
                    lignes.append(f"- {m.nom} : {m.dose_mg} mg x {m.prises_par_jour}/j")
                return "\\n".join(lignes)
        """, """
        m = Medicament.depuis_ligne("paracetamol;1000;3")
        assert isinstance(m, Medicament) and m.nom == "paracetamol"
        assert m.dose_mg == 1000 and m.prises_par_jour == 3
        assert m.dose_journaliere_mg == 3000
        assert Medicament.dose_pediatrique(15, 12) == 180
        o = Ordonnance("Durand")
        assert len(o) == 0
        o.ajouter(m)
        o.ajouter(Medicament("amoxicilline", 1000, 2))
        assert len(o) == 2
        assert "paracetamol" in o and "ibuprofene" not in o
        attendu = ("Ordonnance de Durand\\n- paracetamol : 1000 mg x 3/j\\n"
                   "- amoxicilline : 1000 mg x 2/j")
        assert str(o) == attendu, str(o)
        print(o)
        """, 6)

    # ---------------------------------------------------------------- Ex 7
    cells += ex(correction, """
        ## Exercice 7 — Héritage : les examens d'un dossier

        Un examen biologique et un examen d'imagerie **sont des** examens.

        - `Examen(id_patient, libelle)` avec `resume()` → `"P0002 — potassium"`.
        - `ExamenBiologique(Examen)` ajoute `valeur`, `unite`, `borne_basse`,
          `borne_haute` ; son constructeur appelle celui de la mère avec `super()`.
          `est_normal()` → valeur dans `[borne_basse, borne_haute]`.
          `resume()` est redéfinie et réutilise celle de la mère :
          `"P0002 — potassium : 4.1 mmol/L (normal)"` (ou `(anormal)`).
        - `ExamenImagerie(Examen)` ajoute `modalite`, `conclusion` ; `resume()` →
          `"P0002 — radiographie thoracique [RX] : pas d'opacité"`.

        Les cellules suivantes chargent ensuite les vrais résultats du fichier
        `examens_biologiques.csv` (données synthétiques du cours) dans des objets.
        """, """
        class Examen:
            def __init__(self, id_patient, libelle):
                ...

            def resume(self):
                ...


        class ExamenBiologique(Examen):
            def __init__(self, id_patient, libelle, valeur, unite, borne_basse, borne_haute):
                ...

            def est_normal(self):
                ...

            def resume(self):
                ...


        class ExamenImagerie(Examen):
            ...
        """, """
        class Examen:
            def __init__(self, id_patient, libelle):
                self.id_patient = id_patient
                self.libelle = libelle

            def resume(self):
                return f"{self.id_patient} — {self.libelle}"


        class ExamenBiologique(Examen):
            def __init__(self, id_patient, libelle, valeur, unite, borne_basse, borne_haute):
                super().__init__(id_patient, libelle)
                self.valeur = valeur
                self.unite = unite
                self.borne_basse = borne_basse
                self.borne_haute = borne_haute

            def est_normal(self):
                return self.borne_basse <= self.valeur <= self.borne_haute

            def resume(self):
                etat = "normal" if self.est_normal() else "anormal"
                return f"{super().resume()} : {self.valeur} {self.unite} ({etat})"


        class ExamenImagerie(Examen):
            def __init__(self, id_patient, libelle, modalite, conclusion):
                super().__init__(id_patient, libelle)
                self.modalite = modalite
                self.conclusion = conclusion

            def resume(self):
                return f"{super().resume()} [{self.modalite}] : {self.conclusion}"
        """, """
        k = ExamenBiologique("P0002", "potassium", 4.1, "mmol/L", 3.5, 5.0)
        rx = ExamenImagerie("P0002", "radiographie thoracique", "RX", "pas d'opacité")
        assert issubclass(ExamenBiologique, Examen) and issubclass(ExamenImagerie, Examen)
        assert isinstance(k, Examen) and k.id_patient == "P0002"
        assert k.est_normal()
        assert k.resume() == "P0002 — potassium : 4.1 mmol/L (normal)", k.resume()
        assert ExamenBiologique("P9", "crp", 48.0, "mg/L", 0, 5).resume() == "P9 — crp : 48.0 mg/L (anormal)"
        assert rx.resume() == "P0002 — radiographie thoracique [RX] : pas d'opacité", rx.resume()
        assert Examen("P1", "ECG").resume() == "P1 — ECG"
        """, 7)
    cells += [
        M("""
        Chargeons maintenant le fichier. Les intervalles de référence ci-dessous sont
        **indicatifs** (adulte) : en pratique ils dépendent du laboratoire, du sexe
        et de l'âge. La boucle crée un objet par ligne, puis le polymorphisme fait
        le reste : chaque objet sait se résumer.
        """),
        C("""
        REFERENCES = {                 # examen: (borne basse, borne haute)
            "hemoglobine": (12.0, 17.0),   # g/dL
            "creatinine": (45.0, 110.0),   # µmol/L
            "potassium": (3.5, 5.0),       # mmol/L
            "crp": (0.0, 5.0),             # mg/L
        }
        lignes = lire_csv("examens_biologiques.csv")
        print(len(lignes), "lignes ; première :", lignes[0])

        examens = []
        for ligne in lignes:
            basse, haute = REFERENCES[ligne["examen"]]
            examens.append(ExamenBiologique(ligne["id_patient"], ligne["examen"],
                                            float(ligne["valeur"]), ligne["unite"],
                                            basse, haute))
        dossier_p0002 = [e for e in examens if e.id_patient == "P0002"]
        dossier_p0002.append(ExamenImagerie("P0002", "radiographie thoracique", "RX", "pas d'opacité"))
        for e in dossier_p0002:
            print(e.resume())
        """),
        M("""
        **Question.** Combien de résultats de potassium sont hors de l'intervalle
        de référence ? Stockez la réponse dans `n_potassium_anormal` en utilisant
        la méthode `est_normal()` des objets.
        """),
    ]
    if correction:
        cells.append(C("""
        n_potassium_anormal = sum(1 for e in examens
                                  if e.libelle == "potassium" and not e.est_normal())
        print(n_potassium_anormal)
        """))
    else:
        cells.append(C("""
        # À vous
        n_potassium_anormal = ...
        print(n_potassium_anormal)
        """))
    cells.append(check("7 bis", """
        attendu = sum(1 for l in lignes if l["examen"] == "potassium"
                      and not 3.5 <= float(l["valeur"]) <= 5.0)
        assert n_potassium_anormal == attendu
        """))

    # ---------------------------------------------------------------- Ex 8
    cells += [M("""
        ## Exercice 8 — Votre premier module

        Dans Colab, la commande magique `%%writefile nom.py`, en première ligne d'une
        cellule, écrit le reste de la cellule dans un fichier au lieu de l'exécuter.
        Complétez le module `calculs_medicaux.py` ci-dessous, exécutez la cellule
        (elle affiche `Writing calculs_medicaux.py`), puis importez-le.

        - `imc(poids_kg, taille_m)` → arrondi à 1 décimale ;
        - `clairance_cockcroft_gault(age, poids_kg, creatinine_mg_dl, femme)` →
          clairance de la créatinine en mL/min, arrondie à 1 décimale :
          (140 − âge) × poids / (72 × créatinine), multipliée par 0,85 chez la
          femme (Cockcroft et Gault, 1976) ;
        - `duree_sejour_jours(entree, sortie)` : deux dates au format `"JJ/MM/AAAA"`
          → nombre de jours entre les deux (utilisez `datetime.strptime(texte,
          "%d/%m/%Y")` puis `.days` sur la différence) ;
        - un bloc `if __name__ == "__main__":` qui affiche un exemple : il ne doit
          **pas** s'exécuter lors de l'import.

        Si vous modifiez le fichier après un premier import, un second `import` ne
        relit pas le fichier : la cellule d'import utilise `importlib.reload`.
        """)]
    if correction:
        cells.append(C("""
        %%writefile calculs_medicaux.py
        \"\"\"Calculs cliniques courants (exemples pédagogiques).\"\"\"
        from datetime import datetime


        def imc(poids_kg, taille_m):
            return round(poids_kg / taille_m ** 2, 1)


        def clairance_cockcroft_gault(age, poids_kg, creatinine_mg_dl, femme):
            clairance = (140 - age) * poids_kg / (72 * creatinine_mg_dl)
            if femme:
                clairance *= 0.85
            return round(clairance, 1)


        def duree_sejour_jours(entree, sortie):
            debut = datetime.strptime(entree, "%d/%m/%Y")
            fin = datetime.strptime(sortie, "%d/%m/%Y")
            return (fin - debut).days


        if __name__ == "__main__":
            print("IMC :", imc(72, 1.80))
            print("Clairance :", clairance_cockcroft_gault(60, 70, 1.0, femme=False), "mL/min")
        """))
    else:
        cells.append(C("""
        %%writefile calculs_medicaux.py
        \"\"\"Calculs cliniques courants (exemples pédagogiques).\"\"\"
        from datetime import datetime


        def imc(poids_kg, taille_m):
            ...


        def clairance_cockcroft_gault(age, poids_kg, creatinine_mg_dl, femme):
            ...


        def duree_sejour_jours(entree, sortie):
            ...


        # À vous : le bloc if __name__ == "__main__": avec un exemple
        """))
    cells += [
        C("""
        import importlib

        import calculs_medicaux as cm
        importlib.reload(cm)       # relit le fichier s'il a changé

        print(cm.__name__)         # le nom du module quand il est importé
        print(cm.imc(72, 1.80))
        """),
        M("""
        Exécutez maintenant le module comme un programme : `__name__` vaut alors
        `"__main__"` et le bloc d'exemple s'exécute. (`!` lance une commande du
        terminal depuis un notebook.)
        """),
        C("!python calculs_medicaux.py"),
        check(8, """
        from calculs_medicaux import clairance_cockcroft_gault, duree_sejour_jours
        assert cm.imc(72, 1.80) == 22.2
        assert clairance_cockcroft_gault(60, 70, 1.0, femme=False) == 77.8
        assert clairance_cockcroft_gault(60, 70, 1.0, femme=True) == 66.1
        assert clairance_cockcroft_gault(85, 50, 1.4, femme=True) == 23.2
        assert duree_sejour_jours("09/10/2026", "15/10/2026") == 6
        assert duree_sejour_jours("28/02/2024", "01/03/2024") == 2   # année bissextile
        source = open("calculs_medicaux.py", encoding="utf-8").read()
        assert 'if __name__ == "__main__":' in source, "il manque le bloc __main__"
        """),
    ]

    # ---------------------------------------------------------------- Ex 9
    cells += ex(correction, """
        ## Exercice 9 — Pour aller plus loin (facultatif) : 299 dossiers d'insuffisance cardiaque

        Le fichier `insuffisance_cardiaque.csv` contient 299 patients réels (UCI,
        Chicco et Jurman 2020, CC BY 4.0) : `id_patient`, `age`,
        `fraction_ejection` (FEVG, en %), `deces` (0/1), etc. Écrivez une classe
        `DossierIC` :

        - une méthode de classe `depuis_ligne(ligne)` qui crée l'objet à partir d'un
          dictionnaire de `csv.DictReader` (attributs `id_patient` (str),
          `age` (float), `fraction_ejection` (int), `deces` (bool)) ;
        - une propriété `categorie_fevg` selon les recommandations ESC 2021 :
          `"reduite"` si FEVG ≤ 40, `"moderement_reduite"` si 41–49,
          `"preservee"` si ≥ 50 ;
        - `__lt__` pour trier par FEVG croissante, `__repr__` → `"DossierIC('IC001', FEVG=20%)"`.

        Puis construisez la liste `dossiers` des 299 objets, et le dictionnaire
        `mortalite` : catégorie → proportion de décès (arrondie à 3 décimales).
        """, """
        class DossierIC:
            def __init__(self, id_patient, age, fraction_ejection, deces):
                ...

            @classmethod
            def depuis_ligne(cls, ligne):
                ...

            @property
            def categorie_fevg(self):
                ...

            def __lt__(self, autre):
                ...

            def __repr__(self):
                ...


        dossiers = ...
        mortalite = ...
        print(mortalite)
        """, """
        class DossierIC:
            def __init__(self, id_patient, age, fraction_ejection, deces):
                self.id_patient = id_patient
                self.age = age
                self.fraction_ejection = fraction_ejection
                self.deces = deces

            @classmethod
            def depuis_ligne(cls, ligne):
                return cls(ligne["id_patient"], float(ligne["age"]),
                           int(ligne["fraction_ejection"]), ligne["deces"] == "1")

            @property
            def categorie_fevg(self):
                if self.fraction_ejection <= 40:
                    return "reduite"
                if self.fraction_ejection < 50:
                    return "moderement_reduite"
                return "preservee"

            def __lt__(self, autre):
                return self.fraction_ejection < autre.fraction_ejection

            def __repr__(self):
                return f"DossierIC({self.id_patient!r}, FEVG={self.fraction_ejection}%)"


        dossiers = [DossierIC.depuis_ligne(l) for l in lire_csv("insuffisance_cardiaque.csv")]
        mortalite = {}
        for cat in ("reduite", "moderement_reduite", "preservee"):
            groupe = [d for d in dossiers if d.categorie_fevg == cat]
            mortalite[cat] = round(sum(d.deces for d in groupe) / len(groupe), 3)
        print(mortalite)
        print(sorted(dossiers)[:3])
        """, """
        brut = lire_csv("insuffisance_cardiaque.csv")
        assert len(dossiers) == 299 and all(isinstance(d, DossierIC) for d in dossiers)
        d0 = dossiers[0]
        assert d0.id_patient == "IC001" and d0.fraction_ejection == 20 and d0.deces is True
        assert repr(d0) == "DossierIC('IC001', FEVG=20%)"
        assert DossierIC("X", 60, 45, False).categorie_fevg == "moderement_reduite"
        assert DossierIC("X", 60, 40, False).categorie_fevg == "reduite"
        assert DossierIC("X", 60, 50, False).categorie_fevg == "preservee"
        assert sorted(dossiers)[0].fraction_ejection == min(int(l["fraction_ejection"]) for l in brut)
        for cat, (lo, hi) in {"reduite": (0, 40), "moderement_reduite": (41, 49),
                              "preservee": (50, 100)}.items():
            g = [l for l in brut if lo <= int(l["fraction_ejection"]) <= hi]
            assert mortalite[cat] == round(sum(l["deces"] == "1" for l in g) / len(g), 3), cat
        """, 9)
    if correction:
        cells.append(M("""
        La mortalité observée est plus élevée chez les patients à FEVG réduite. Ce
        n'est qu'une description de 299 patients suivis quelques mois : l'âge, la
        créatinine et d'autres variables diffèrent aussi entre les groupes.
        """))
    return nb.notebook(cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for correction, name in ((False, NAME), (True, NAME_CORR)):
        nb._COUNTER = itertools.count()     # byte-identical on every regeneration
        path = OUT / name
        path.write_text(json.dumps(build(correction), indent=1,
                                   ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(REPO)}\n      {aim_repo.colab(COURSE, name)}")


if __name__ == "__main__":
    main()
