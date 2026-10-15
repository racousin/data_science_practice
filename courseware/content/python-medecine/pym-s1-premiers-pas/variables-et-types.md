# Variables, nombres et booléens

Un programme manipule des **valeurs** (un poids, une créatininémie, un
résultat positif ou négatif) rangées dans des **variables**. Chaque valeur a
un **type**, qui détermine les opérations permises. Cette leçon présente la
syntaxe de base et les trois types numériques : `int`, `float` et `bool`.

<!-- notes: 35 minutes. Taper les exemples en direct dans Colab, faire prédire
les résultats avant d'exécuter (surtout // , % et 0.1 + 0.2). -->

---

## Variables et affectation

```python
poids_kg = 68.5            # une variable reçoit une valeur
age = 54
print(poids_kg, age)       # 68.5 54
```

Le signe `=` est une **affectation**, pas une égalité : la valeur de droite est
calculée, puis rangée sous le nom de gauche. `age = age + 1` est donc correct.
Ce qui suit `#` est un **commentaire**, ignoré par Python.

Les noms sont libres, sauf les mots réservés (`if`, `else`, `for`, `def`,
`return`, `class`, `in`, `True`…). Convention : minuscules, mots séparés par
`_`, sans accents, et un nom qui dit ce que contient la variable, avec son
unité : `creatinine_umol_l` plutôt que `c`.

---

## L'indentation fait partie de la syntaxe

Python n'utilise pas d'accolades : les **blocs** sont délimités par
l'indentation (4 espaces), après une ligne terminée par `:`.

```python
temperature = 38.6
if temperature >= 38.0:
    print("Fièvre")          # dans le bloc du if
print("Mesure enregistrée")  # hors du bloc
```

Une indentation incohérente provoque une `IndentationError`. Les conditions
et les boucles sont l'objet de la séance 2 ; retenez dès maintenant qu'une
espace en début de ligne n'est jamais anodine.

---

## Entiers et flottants

Python distingue les **entiers** (`int`) et les nombres à virgule, les
**flottants** (`float`), écrits avec un point.

```python
7 + 2, 7 - 2, 7 * 2     # (9, 5, 14)
7 / 2                   # 3.5  : la division donne toujours un float
7 // 2, 7 % 2           # (3, 1) : quotient et reste de la division entière
2 ** 10                 # 1024 : puissance
```

`//` et `%` servent souvent : `125 // 60` heures et `125 % 60` minutes pour
une durée de 125 min (2 h 05). `round(22.857, 1)` arrondit (`22.9`). La
notation scientifique écrit les grands et petits nombres :
`plaquettes = 2.5e5` (250 000 /µL), `dose_g = 5e-3` (5 mg).

---

## Les flottants sont approchés

Un `float` est stocké en binaire avec environ 16 chiffres significatifs :
la plupart des décimaux ne sont pas représentés exactement.

```python
0.1 + 0.2              # 0.30000000000000004
0.1 + 0.2 == 0.3       # False
round(0.1 + 0.2, 2) == 0.3   # True
```

Ce n'est pas une erreur de Python mais du format IEEE 754, commun à tous les
langages. Conséquence pratique : on ne compare jamais deux flottants avec
`==`, on arrondit ou on teste un écart (`abs(a - b) < 1e-9`).

<!-- notes: Les étudiants stallent ici : rassurer, cela n'affecte pas les
résultats cliniques (erreur au 17e chiffre), seulement les tests d'égalité. -->

---

## Booléens et comparaisons

Une comparaison produit un **booléen** (`bool`) : `True` ou `False`.

```python
pas, pad = 148, 92            # pression artérielle systolique, diastolique
pas >= 140                    # True
pas >= 140 or pad >= 90       # True  : HTA selon l'ESC/ESH 2018
not (pas < 120 and pad < 80)  # True
```

Opérateurs : `==`, `!=`, `<`, `<=`, `>`, `>=`, et les opérateurs logiques
`and`, `or`, `not`. Un booléen est un entier déguisé (`True + True` vaut `2`),
et `bool(0)` vaut `False`, `bool(3)` vaut `True`. Python accepte aussi les
encadrements : `3.5 <= potassium <= 5.0`.

---

## Typage dynamique, typage fort

Python devine le type d'une valeur à l'exécution (**typage dynamique**) : une
variable n'est qu'un nom, qui peut désigner successivement des valeurs de
types différents.

```python
resultat = 7.2
type(resultat)          # float
resultat = "positif"
type(resultat)          # str
```

Mais Python ne convertit pas à votre place (**typage fort**) :
`"3" + 3` lève `TypeError: can only concatenate str (not "int") to str`. On
convertit explicitement avec `int("3")`, `float("7.2")` ou `str(3)` ;
`int(7.9)` tronque (`7`). En revanche `"ab" * 3` est permis (`"ababab"`) :
une opération est acceptée quand elle a un sens pour les deux types.

---

## La famille des types

![Hiérarchie des types standard de Python 3](assets/pym-s1-premiers-pas/type_hierarchy.png)

Les nombres (`int`, `bool`, `float`) et les chaînes (`str`) sont vus dans
cette séance ; les listes, tuples, dictionnaires et ensembles en séance 3,
les fonctions en séance 2, les classes en séance 4.

---

## À retenir

- `=` affecte ; les noms sont en minuscules avec `_`, et portent l'unité.
- `/` donne un `float`, `//` et `%` le quotient et le reste entiers, `**` la
  puissance.
- Les flottants sont approchés : ne jamais les comparer avec `==`.
- Typage dynamique (le type suit la valeur) mais fort (pas de conversion
  implicite : `int()`, `float()`, `str()`).
