# Héritage

Un examen biologique et un examen d'imagerie ont des points communs (un patient, une
date, un résumé) et des différences (une valeur et des bornes d'un côté, une modalité
et une conclusion de l'autre). L'**héritage** permet d'écrire le commun une seule fois.

<!-- notes: 25 minutes. Commencer par le code dupliqué : les étudiants voient seuls
le problème. super() est le point qui bloque : faire remarquer que sans lui,
id_patient n'existe pas sur l'objet (AttributeError). L'héritage multiple se présente
en une diapositive, sans exercice. -->

---

## Le problème : du code dupliqué

```python
class Examen:
    def __init__(self, id_patient, date):
        self.id_patient = id_patient
        self.date = date

class ExamenBiologique:
    def __init__(self, id_patient, date, valeur, unite):
        self.id_patient = id_patient       # recopié
        self.date = date                   # recopié
        self.valeur = valeur
        self.unite = unite
```

Toute correction dans `Examen` devrait être reportée à la main dans chaque classe
d'examen. Or un examen biologique **est un** examen : la relation « est un » est le
signe qu'il faut hériter.

---

## Classe mère, classe fille

![Examen et ses deux classes filles](assets/pym-s4-objets-et-modules/heritage_examen.png)

L'héritage crée une nouvelle classe à partir d'une classe existante en lui ajoutant de
nouveaux attributs et méthodes. La nouvelle classe est la **classe fille** ; elle
**hérite de** (ou **étend**) la **classe mère**.

```python
class ExamenBiologique(Examen):      # ExamenBiologique hérite d'Examen
    ...
```

La classe fille possède automatiquement tous les attributs et méthodes de la mère.

---

## Appeler le constructeur de la mère : `super()`

```python
class ExamenBiologique(Examen):
    def __init__(self, id_patient, date, valeur, unite,
                 borne_basse, borne_haute):
        super().__init__(id_patient, date)     # partie commune
        self.valeur = valeur
        self.unite = unite
        self.borne_basse, self.borne_haute = borne_basse, borne_haute

    def est_normal(self):
        return self.borne_basse <= self.valeur <= self.borne_haute
```

`super()` désigne la classe mère. On rencontre aussi l'écriture explicite
`Examen.__init__(self, id_patient, date)`, équivalente ici. Sans cet appel, l'objet
n'aurait ni `id_patient` ni `date`.

---

## Redéfinir une méthode

Une classe fille peut **redéfinir** une méthode de la mère pour l'adapter, et réutiliser
la version de la mère avec `super()` :

```python
class Examen:
    def resume(self):
        return f"{self.id_patient} ({self.date})"

class ExamenImagerie(Examen):
    def __init__(self, id_patient, date, modalite, conclusion):
        super().__init__(id_patient, date)
        self.modalite, self.conclusion = modalite, conclusion

    def resume(self):
        return f"{super().resume()} — {self.modalite} : {self.conclusion}"
```

---

## Polymorphisme

Des objets de classes différentes répondent au même appel, chacun à sa manière :

```python
dossier = [
    ExamenBiologique("P0002", "2026-10-12", 4.1, "mmol/L", 3.5, 5.0),
    ExamenImagerie("P0002", "2026-10-13", "IRM", "pas de lésion"),
]
for examen in dossier:
    print(examen.resume())   # chaque objet appelle sa propre méthode
```

`isinstance(examen, Examen)` vaut `True` pour les deux objets : un objet d'une classe
fille est aussi une instance de la classe mère. `issubclass(ExamenImagerie, Examen)`
teste la relation entre classes.

---

## Plusieurs niveaux, plusieurs mères

Une classe fille peut elle-même avoir des filles (`Examen` → `ExamenBiologique` →
`Ionogramme`). Python permet aussi l'**héritage multiple** : un professeur des
universités-praticien hospitalier (PU-PH) est à la fois soignant et enseignant.

![Héritage multiple : PUPH hérite de Soignant et d'Enseignant](assets/pym-s4-objets-et-modules/heritage_multiple.png)

```python
class PUPH(Soignant, Enseignant):
    ...
```

Quand deux mères définissent la même méthode, Python suit un ordre de recherche fixe
(`PUPH.__mro__`). À utiliser avec parcimonie.

---

## Ce qu'il faut retenir de la programmation objet

- En Python, **tout est objet** ; définir une classe, c'est définir un nouveau **type**.
- Une classe est une manière de **ranger l'information** avec les opérations qui s'y appliquent.
- Attributs et méthodes sont toujours **publics** : l'encapsulation repose sur des conventions et des propriétés.
- On peut **surcharger** les opérateurs (`+`, `<`, `==`, …) pour ses propres types.
- L'**héritage** partage attributs et méthodes entre classes et évite la duplication de code.

Les bibliothèques reposent sur ces notions : un réseau de neurones PyTorch est une
classe qui hérite de `torch.nn.Module` et redéfinit `forward`. Pour aller plus loin :
les classes abstraites (module `abc`) et les classes de données (`@dataclass`), qui
écrivent `__init__`, `__repr__` et `__eq__` à votre place.
