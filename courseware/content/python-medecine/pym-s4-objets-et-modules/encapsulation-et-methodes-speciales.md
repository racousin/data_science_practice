# Encapsulation et méthodes spéciales

Un objet protège ses données : on les lit et on les modifie par des méthodes qui
vérifient ce qu'on leur donne. Et il peut s'intégrer au langage : s'afficher avec
`print`, se comparer avec `<`, s'additionner avec `+`.

<!-- notes: 30 minutes. Montrer d'abord le problème (un poids de -5 kg accepté sans
broncher), puis la solution @property. Le name mangling surprend : le montrer avec
dir() plutôt que de l'expliquer. Les étudiants confondent __str__ et __repr__ :
retenir "str pour le lecteur, repr pour le développeur". -->

---

## Le principe d'encapsulation

**Encapsuler**, c'est regrouper les données dans un objet et n'en permettre la
manipulation que par des méthodes dédiées, qui forment l'interface de l'objet :

- les **accesseurs** (*getters*) pour lire un attribut ;
- les **mutateurs** (*setters*) pour le modifier, en vérifiant la valeur.

Sans cela, rien n'empêche une valeur absurde :

```python
p = Patient("Durand", poids_kg=72, taille_m=1.80)
p.poids_kg = -5          # accepté
setattr(p, "poids_kg", "lourd")   # accepté aussi
print(getattr(p, "poids_kg"))     # lourd
```

---

## Un mutateur qui vérifie

```python
class Patient:
    ...
    def set_poids(self, valeur):
        if not isinstance(valeur, (int, float)):
            raise TypeError(f"{type(valeur).__name__} n'est pas un poids")
        if valeur <= 0:
            raise ValueError("le poids doit être positif")
        self.poids_kg = valeur
```

`p.set_poids(-5)` lève maintenant une `ValueError`. Mais Python n'applique pas
l'encapsulation : rien n'interdit d'écrire encore `p.poids_kg = -5` directement. Les
attributs et les méthodes sont toujours publics ; la protection repose sur des
conventions.

---

## Attributs « privés » : `_` et `__`

Un nom préfixé par `_` signifie « usage interne, ne pas toucher » — une simple
convention, l'accès reste possible :

```python
self._poids_kg = poids_kg     # p._poids_kg fonctionne toujours
```

Un nom préfixé par `__` est **renommé** par Python (*name mangling*) :

```python
class Patient:
    def __init__(self, nss):
        self.__nss = nss

p = Patient("1 85 05 75 ...")
p.__nss            # AttributeError
p._Patient__nss    # fonctionne : visible dans dir(p)
```

L'attribut n'est pas réellement privé : son nom a seulement changé.

---

## `@property` et son setter

La solution idiomatique : garder un attribut interne `_poids_kg` et exposer une
**propriété** `poids_kg` qui s'utilise comme un attribut mais passe par des méthodes.

```python
class Patient:
    def __init__(self, nom, poids_kg):
        self.nom = nom
        self.poids_kg = poids_kg       # passe déjà par le setter

    @property
    def poids_kg(self):                # accesseur
        return self._poids_kg

    @poids_kg.setter
    def poids_kg(self, valeur):        # mutateur
        if valeur <= 0:
            raise ValueError("le poids doit être positif")
        self._poids_kg = valeur
```

`p.poids_kg = -5` lève maintenant `ValueError`. Une propriété sans setter est en
lecture seule : pratique pour une valeur calculée comme un âge ou un IMC.

---

## Méthodes spéciales : `__str__` et `__repr__`

Par défaut, `print(m)` affiche `<__main__.Mesure object at 0x1101db700>`. Les méthodes
**spéciales** (ou *dunder*, encadrées de `__`) branchent un objet sur le langage.

```python
class Mesure:
    def __init__(self, valeur, unite):
        self.valeur, self.unite = valeur, unite

    def __str__(self):                 # pour le lecteur : print, f-string
        return f"{self.valeur} {self.unite}"

    def __repr__(self):                # pour le développeur : console, listes
        return f"Mesure({self.valeur!r}, {self.unite!r})"
```

`print(Mesure(5.4, "mmol/L"))` affiche `5.4 mmol/L` ;
`[Mesure(5.4, "mmol/L")]` affiche `[Mesure(5.4, 'mmol/L')]`.

---

## Surcharger les opérateurs

La **surcharge** redéfinit un opérateur existant pour l'adapter à un nouveau type.
Python traduit chaque opérateur en appel de méthode :

| Expression | Appel | Expression | Appel |
|---|---|---|---|
| `a + b` | `a.__add__(b)` | `a < b` | `a.__lt__(b)` |
| `a - b` | `a.__sub__(b)` | `a <= b` | `a.__le__(b)` |
| `a * b` | `a.__mul__(b)` | `a == b` | `a.__eq__(b)` |
| `a / b` | `a.__truediv__(b)` | `a != b` | `a.__ne__(b)` |
| `a // b` | `a.__floordiv__(b)` | `a > b` | `a.__gt__(b)` |
| `a % b`, `a ** b` | `__mod__`, `__pow__` | `a >= b` | `a.__ge__(b)` |
| `a & b` | `a.__and__(b)` | `len(a)` | `a.__len__()` |

Le OU binaire `a | b` appelle de même `a.__or__(b)`. Sans `__eq__`, `==` compare les identifiants comme `is` : deux mesures identiques ne
sont pas « égales ».

---

## Exemple : comparer et additionner des mesures

```python
class Mesure:
    ...
    def __eq__(self, autre):
        return self.valeur == autre.valeur and self.unite == autre.unite

    def __lt__(self, autre):
        return self.valeur < autre.valeur

    def __add__(self, autre):
        if self.unite != autre.unite:
            raise ValueError("unités différentes")
        return Mesure(self.valeur + autre.valeur, self.unite)
```

Avec `__lt__`, `sorted(liste_de_mesures)` et `min(...)` fonctionnent. `__add__`
renvoie un **nouvel** objet plutôt que de modifier `self`, comme le fait `+` sur les
nombres.

---

## À retenir

- Encapsuler : accéder aux données par des méthodes qui vérifient les valeurs.
- En Python, tout reste public : `_x` est une convention, `__x` un simple renommage.
- `@property` + `@x.setter` : la syntaxe d'un attribut, le contrôle d'une méthode.
- `__str__`, `__repr__`, `__eq__`, `__lt__`, `__add__`, `__len__` intègrent un objet au langage.
