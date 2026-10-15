# Listes et tuples

Une variable contient une valeur ; un dossier patient en contient des dizaines.
Les listes et les tuples rangent plusieurs valeurs **dans un ordre**, sous un seul nom.

<!-- notes: 35 minutes. Faire taper les exemples dans Colab au fur et à mesure.
Les étudiants bloquent sur deux points : l'indice qui commence à 0, et la borne
de fin exclue du slicing. Dessiner les indices au tableau sous la liste. -->

---

## Créer une liste et lire ses éléments

Une liste s'écrit entre crochets ; ses éléments peuvent être de types différents.

```python
glycemies = [0.92, 1.05, 1.31, 0.88, 2.10]   # g/L, à jeun
dossier = ["R03", 67, True, glycemies]         # str, int, bool, list
len(glycemies)                                 # 5
```

On lit un élément par son **indice**, qui commence à 0. Un indice négatif part de la fin.

```python
glycemies[0]     # 0.92, la première mesure
glycemies[-1]    # 2.10, la dernière
1.31 in glycemies   # True
```

---

## Découper : le slicing

La syntaxe `liste[debut:fin:pas]` renvoie une **nouvelle liste**. La borne `fin` est exclue.

| indice | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `glycemies` | 0.92 | 1.05 | 1.31 | 0.88 | 2.10 |
| indice négatif | -5 | -4 | -3 | -2 | -1 |

```python
glycemies[0:2]    # [0.92, 1.05]
glycemies[::2]    # [0.92, 1.31, 2.10], une mesure sur deux
glycemies[-3:]    # [1.31, 0.88, 2.10], les trois dernières
glycemies[::-1]   # la liste à l'envers
```

On peut aussi concaténer (`+`) et répéter (`*`) : `[0] * 3` donne `[0, 0, 0]`, et `list(range(4))` donne `[0, 1, 2, 3]`.

---

## Parcourir une liste

`for` donne les éléments un à un ; `enumerate` donne aussi leur position ; `zip` parcourt deux listes en parallèle.

```python
for g in glycemies:
    print(g)

for i, g in enumerate(glycemies):
    print(f"mesure {i} : {g} g/L")

heures = ["8h", "12h", "16h", "20h", "0h"]
for h, g in zip(heures, glycemies):
    print(h, g)
```

<!-- notes: insister sur enumerate plutôt que range(len(...)) : plus lisible,
et c'est ce qu'on lit dans le code des autres. -->

---

## Modifier une liste : les méthodes

Une liste se modifie **sur place** : on remplace un élément par son indice, ou on appelle une méthode.

```python
glycemies[3] = 0.90          # corriger une saisie
glycemies.append(1.12)       # ajouter à la fin
glycemies.extend([0.95, 1.0])  # ajouter plusieurs valeurs
glycemies.insert(0, 0.85)    # insérer à la position 0
```

```python
derniere = glycemies.pop()   # retire et renvoie le dernier élément
glycemies.remove(2.10)       # retire la première occurrence de 2.10
glycemies.sort()             # trie la liste elle-même, renvoie None
triee = sorted(glycemies)    # renvoie une nouvelle liste triée
```

Autres méthodes utiles : `count`, `index`, `reverse`, `copy`, `clear`. `sort` modifie la liste, `sorted` en crée une nouvelle : c'est la même différence que nous retrouverons avec la mutabilité.

---

## Compréhension de liste

Construire une liste à partir d'une autre est si fréquent que Python a une syntaxe dédiée.

```python
# avec une boucle
en_mmol = []
for g in glycemies:
    en_mmol.append(g * 5.551)

# en une ligne
en_mmol = [g * 5.551 for g in glycemies]
```

On peut ajouter une condition pour **filtrer** : ici les glycémies à jeun au-dessus de 1,26 g/L, seuil du diabète (OMS, à confirmer sur deux dosages).

```python
hyper = [g for g in glycemies if g >= 1.26]
```

<!-- notes: 1 g/L de glucose = 5,551 mmol/L (masse molaire 180,16 g/mol).
Ce sont des exemples pédagogiques, pas un outil de décision clinique. -->

---

## Les tuples : des séquences immuables

Un tuple s'écrit entre parenthèses. Il s'indexe, se découpe et se parcourt comme une liste, mais **ne se modifie pas**.

```python
tension = (142, 91)        # (systolique, diastolique) en mmHg
tension[0]                 # 142
seul = (5,)                # la virgule fait le tuple, pas les parenthèses
tension[0] = 130           # TypeError: 'tuple' object does not support item assignment
```

| Opération | liste | tuple |
|---|---|---|
| indexer, découper, parcourir, `len`, `in` | oui | oui |
| concaténer `+`, répéter `*` | oui | oui (nouveau tuple) |
| remplacer, supprimer, `append` | oui | non |

On choisit un tuple pour une donnée de forme fixe — une mesure de tension, des coordonnées — et une liste pour une collection qui grandit.

---

## Déballer une séquence (unpacking)

On peut affecter les éléments d'une séquence à plusieurs noms en une instruction.

```python
pas, pad = (142, 91)                       # pas = 142, pad = 91
nom, (taille_m, poids_kg) = ["Durand", (1.75, 90)]
premiere, *milieu, derniere = glycemies    # *milieu reçoit une liste
a, b = b, a                                # échanger deux variables
```

C'est ce qui se passe dans `for i, g in enumerate(...)` : chaque tuple `(indice, valeur)` est déballé.

---

## À retenir

- Une liste `[...]` est ordonnée et modifiable ; un tuple `(...)` est ordonné et figé.
- Les indices commencent à 0 ; `l[debut:fin]` exclut `fin`.
- `enumerate` donne l'indice et la valeur, `zip` parcourt plusieurs listes ensemble.
- `pas, pad = tension` déballe un tuple ; `*reste` ramasse ce qui reste.
- `[expression for x in liste if condition]` transforme et filtre en une ligne.
