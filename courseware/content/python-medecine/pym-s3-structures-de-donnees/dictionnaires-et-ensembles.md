# Dictionnaires et ensembles

Une liste range ses valeurs par position. Un dossier patient se lit plutôt par
**rubrique** : l'âge, le poids, les traitements. C'est le rôle du dictionnaire.
L'ensemble, lui, garde des éléments uniques et sait comparer deux collections.

<!-- notes: 30 minutes. L'analogie qui marche : le dictionnaire Vidal, on cherche
un médicament par son nom (la clé) et on lit sa notice (la valeur). Erreurs
fréquentes : oublier les guillemets autour des clés, confondre d["cle"] et
d.get("cle"). -->

---

## Clé et valeur

Un dictionnaire associe des **clés** uniques à des **valeurs**. On accède à une valeur par sa clé, pas par une position.

```python
patient = {"id": "P001", "age": 67, "poids_kg": 72.5, "sexe": "F"}
patient = dict(id="P001", age=67, poids_kg=72.5, sexe="F")   # équivalent
patient["age"]           # 67
```

Une clé absente lève une erreur avec `[]` ; `get` renvoie `None` ou une valeur par défaut.

```python
patient["taille_cm"]             # KeyError: 'taille_cm'
patient.get("taille_cm")         # None
patient.get("taille_cm", 165)    # 165
```

---

## Ajouter, modifier, supprimer

Un dictionnaire est modifiable. Affecter une clé existante la remplace ; une clé nouvelle l'ajoute.

```python
patient["poids_kg"] = 71.0                   # remplacer
patient["taille_cm"] = 164                   # ajouter
patient.update({"fumeur": False, "age": 68}) # plusieurs à la fois
patient |= {"allergies": ["pénicilline"]}    # idem, Python ≥ 3.9
```

```python
fumeur = patient.pop("fumeur")   # retire la clé, renvoie sa valeur
"age" in patient                 # True : `in` teste les clés
```

`clear()` vide le dictionnaire. Les clés doivent être immuables (`str`, `int`, `tuple`) ; les valeurs peuvent être n'importe quoi.

---

## Parcourir un dictionnaire

Trois vues : `keys()`, `values()`, `items()`. Un `for` sur le dictionnaire lui-même parcourt les clés.

```python
for cle in patient:
    print(cle, patient[cle])

for valeur in patient.values():
    print(valeur)

for cle, valeur in patient.items():
    print(f"{cle:>10} : {valeur}")
```

Depuis Python 3.7, l'ordre de parcours est l'ordre d'insertion.

---

## Compter avec un dictionnaire

Un usage très fréquent : compter des occurrences, par exemple des codes diagnostiques CIM-10.

```python
codes = ["I10", "E11", "I10", "J18", "I10", "E11"]
compte = {}
for c in codes:
    compte[c] = compte.get(c, 0) + 1
# {'I10': 3, 'E11': 2, 'J18': 1}
```

La bibliothèque standard fait la même chose avec `collections.Counter`, qui sait aussi donner les plus fréquents.

```python
from collections import Counter
Counter(codes).most_common(1)   # [('I10', 3)]
```

---

## Dictionnaires imbriqués

Les valeurs peuvent être des listes ou d'autres dictionnaires : c'est la forme d'un dossier réel, et celle du format JSON.

```python
dossier = {
    "id": "P001",
    "antecedents": ["HTA", "diabète de type 2"],
    "traitements": {"metformine": "1000 mg x2/j", "ramipril": "5 mg/j"},
}
dossier["antecedents"][0]            # 'HTA'
dossier["traitements"]["ramipril"]   # '5 mg/j'
```

On lit de gauche à droite : chaque `[...]` descend d'un niveau.

---

## Les ensembles

Un ensemble (`set`) contient des éléments **uniques, sans ordre**. `{}` crée un dictionnaire vide ; un ensemble vide s'écrit `set()`.

```python
symptomes_grippe = {"fièvre", "toux", "myalgies", "céphalées"}
symptomes_covid = {"fièvre", "toux", "anosmie", "dyspnée"}
set(["I10", "E11", "I10"])   # {'I10', 'E11'} : doublons supprimés
```

```python
symptomes_grippe & symptomes_covid   # intersection : {'fièvre', 'toux'}
symptomes_grippe | symptomes_covid   # union
symptomes_covid - symptomes_grippe   # différence : {'anosmie', 'dyspnée'}
symptomes_grippe ^ symptomes_covid   # dans l'un ou l'autre, pas les deux
```

Le test `x in ensemble` est très rapide, même sur des milliers d'éléments.

---

## À retenir

- `dict` : clés uniques → valeurs ; `d[cle]` échoue si la clé manque, `d.get(cle, defaut)` non.
- `items()` parcourt les paires clé–valeur ; `d[c] = d.get(c, 0) + 1` compte, `Counter` aussi.
- Les structures s'imbriquent : listes dans des dictionnaires, dictionnaires dans des listes.
- `set` : éléments uniques, `&` `|` `-` `^` pour comparer deux collections.
