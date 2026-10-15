# Classes et objets

Jusqu'ici, un patient était un dictionnaire et ses calculs des fonctions séparées.
La programmation objet réunit les deux : une **classe** décrit ce qu'est un patient
(ses données et ce qu'on peut calculer sur lui), et chaque patient réel en est un
**objet**.

<!-- notes: 35 minutes. Partir du dictionnaire patient de la séance 3 : "où mettre la
fonction imc ?". Les étudiants butent sur self : le présenter comme "l'objet sur lequel
on travaille", pas comme un mot magique. Faire taper la classe Patient en direct. -->

---

## En Python, tout est objet

Chaque valeur manipulée est un objet, qui possède un **identifiant**, un **type** et une
**valeur**. L'identifiant (l'adresse en mémoire) ne change jamais après la création.

```python
glycemie = 5.4
print(type(glycemie), id(glycemie))   # <class 'float'> 4378...
mesures = [5.4, 6.1]
copie = mesures
print(copie is mesures)               # True : le même objet
```

`is` compare les identifiants ; `==` compare les valeurs. Un objet regroupe des
**attributs** (ses données) et des **méthodes** (les fonctions qui agissent sur lui) :
`mesures.append(7.0)` appelle la méthode `append` de la liste.

---

## Programmation procédurale ou objet

| Procédurale | Objet |
|---|---|
| des fonctions qui s'appellent pas à pas | des classes qui définissent des types |
| les données circulent à part (dictionnaires, listes) | les données sont rangées dans l'objet |
| `calculer_imc(patient["poids"], patient["taille"])` | `patient.imc()` |

Les deux styles coexistent en Python. L'objet devient utile dès que des données et les
opérations sur ces données vont toujours ensemble : un patient et son IMC, une
ordonnance et ses médicaments, un examen et son interprétation.

---

## Classe et instances

Une **classe** est un plan ; un **objet** (ou **instance**) est une réalisation de ce
plan, avec ses propres valeurs.

![Une classe Patient et trois objets créés à partir d'elle](assets/pym-s4-objets-et-modules/classe_et_instances.png)

```python
class Patient:
    def __init__(self, nom, poids_kg, taille_m):
        self.nom = nom
        self.poids_kg = poids_kg
        self.taille_m = taille_m
```

Par convention, le nom d'une classe s'écrit en `CamelCase` : `Patient`,
`ExamenBiologique`.

---

## Constructeur et instanciation

`__init__` est le **constructeur** : la méthode appelée automatiquement à la création
d'un objet. Son premier paramètre, `self`, désigne l'objet en cours de construction ;
`self.poids_kg = poids_kg` crée un **attribut d'instance**.

```python
p1 = Patient("Durand", poids_kg=72, taille_m=1.80)
p2 = Patient("Martin", poids_kg=58, taille_m=1.64)
print(p1.nom, p2.poids_kg)    # Durand 58
```

L'**instanciation** `Patient(...)` fait deux opérations : l'**allocation** (réserver la
mémoire de l'objet) puis l'**initialisation** (appeler `__init__` pour fixer son état).
On ne passe jamais `self` soi-même : Python le fournit.

---

## Méthodes d'instance

Une méthode est une fonction définie dans la classe. Elle reçoit l'objet en premier
paramètre et peut lire ou modifier son état.

```python
class Patient:
    ...
    def imc(self):
        return self.poids_kg / self.taille_m ** 2

    def peser(self, nouveau_poids):
        self.poids_kg = nouveau_poids
```

`p1.imc()` est traduit par Python en `Patient.imc(p1)` : c'est ainsi que `self` reçoit
`p1`. Après `p1.peser(70)`, `p1.imc()` vaut 70 / 1,80² ≈ 21,6 kg/m².

---

## Attributs de classe

Un attribut défini directement dans le corps de la classe est **partagé** par toutes
les instances.

```python
class Patient:
    hopital = "CHU de Bordeaux"      # attribut de classe
    def __init__(self, nom):
        self.nom = nom               # attribut d'instance

p1, p2 = Patient("Durand"), Patient("Martin")
Patient.hopital = "CHU de Lille"
print(p1.hopital, p2.hopital)        # CHU de Lille CHU de Lille
```

Attention : `p1.hopital = "Clinique X"` ne modifie pas la classe, il crée un attribut
d'instance propre à `p1` qui masque celui de la classe.

---

## Méthodes de classe et méthodes statiques

Une **méthode de classe** (`@classmethod`) reçoit la classe (`cls`) au lieu d'un objet.
Son usage typique : un constructeur alternatif.

```python
    @classmethod
    def depuis_dict(cls, d):
        return cls(d["nom"], d["poids_kg"], d["taille_m"])
```

Une **méthode statique** (`@staticmethod`) ne reçoit ni l'objet ni la classe : c'est
une fonction utilitaire rangée dans la classe parce qu'elle s'y rattache logiquement.

```python
    @staticmethod
    def livres_vers_kg(livres):
        return livres * 0.4536
```

`Patient.depuis_dict({...})` et `Patient.livres_vers_kg(160)` s'appellent sur la
classe elle-même.

---

## Les objets sont mutables

Les objets que l'on crée sont **mutables** par défaut : on peut changer leurs attributs
après création. Comme pour les listes (séance 3), deux noms peuvent désigner le même
objet.

```python
p1 = Patient("Durand", 72, 1.80)
p3 = p1              # pas de copie : un second nom pour le même objet
p3.peser(68)
print(p1.poids_kg)   # 68
```

Pour un vrai double, il faut créer un nouvel objet (`Patient(p1.nom, ...)`) ou utiliser
`copy.copy(p1)`.

---

## À retenir

- Une classe est un plan (`class Patient:`), un objet une instance de ce plan (`Patient(...)`).
- `__init__` initialise l'objet ; `self` est l'objet sur lequel on travaille.
- Attributs d'instance (`self.x`) propres à chaque objet, attributs de classe partagés.
- `@classmethod` reçoit la classe (constructeurs alternatifs), `@staticmethod` ne reçoit rien.
