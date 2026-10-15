# Mutabilité et références

En Python, une variable est un **nom qui désigne un objet**, pas une boîte qui
contient une valeur. Tant que l'objet ne change pas, la nuance est invisible.
Dès qu'on modifie une liste ou un dictionnaire, elle explique des résultats
surprenants.

<!-- notes: 20 minutes. Faire prédire chaque exemple avant de l'exécuter
(main levée : « a change ou pas ? »). Le cas b = a puis b.append(...) surprend
presque tout le monde ; c'est le but. -->

---

## Mutable ou immuable

Un objet **mutable** peut être modifié après sa création ; un objet **immuable** ne le peut pas : toute « modification » crée un nouvel objet.

| Immuables | Mutables |
|---|---|
| `int`, `float`, `bool`, `str`, `tuple` | `list`, `dict`, `set` |

`id(objet)` donne l'identité d'un objet (son adresse en mémoire) ; `a is b` teste si deux noms désignent **le même** objet.

---

## Un entier ne change pas : on en crée un autre

```python
a = 120
b = a
a is b          # True : deux noms, un seul objet
a = a + 10      # 130 est un nouvel objet ; a le désigne désormais
b               # 120
```

![Réaffecter un entier](assets/pym-s3-structures-de-donnees/entier-immuable.png)

`id(a)` change après `a = a + 10`, `id(b)` non. C'est une copie implicite : l'ancienne valeur reste intacte pour `b`.

---

## Une liste change sur place

```python
a = [120, 135]
b = a           # b n'est pas une copie : c'est un second nom
b.append(128)
a               # [120, 135, 128]
a is b          # True
```

![Deux noms pour une même liste](assets/pym-s3-structures-de-donnees/liste-alias.png)

`append`, `a[0] = ...`, `sort`, `d["cle"] = ...` modifient l'objet lui-même : `id(a)` ne change pas. On parle d'**alias** quand deux noms désignent le même objet mutable.

---

## Le piège dans une fonction

Une fonction reçoit une référence vers l'objet, pas une copie. Si elle modifie une liste passée en argument, l'appelant voit la modification.

```python
def ajouter_mesure(serie, valeur):
    serie.append(valeur)

tensions = [120, 135]
ajouter_mesure(tensions, 128)
tensions        # [120, 135, 128]
```

C'est parfois voulu (méthodes comme `sort`), souvent une source de bogues. Une fonction qui modifie son argument doit le dire dans sa docstring, et en général ne rien renvoyer.

---

## Copier explicitement

Pour travailler sur une copie, on la demande.

```python
b = a.copy()        # ou list(a), ou a[:]
b.append(140)
a                   # inchangée
```

Pour une liste qui contient elle-même des listes ou des dictionnaires, `copy()` ne copie que le premier niveau. `copy.deepcopy` copie tout.

```python
import copy
dossiers_copie = copy.deepcopy(dossiers)
```

---

## À retenir

- Une variable est un nom vers un objet ; `b = a` ne copie rien.
- Immuables (`int`, `str`, `tuple`) : toute modification crée un nouvel objet.
- Mutables (`list`, `dict`, `set`) : la modification est vue par tous les noms qui les désignent.
- Copier explicitement avec `.copy()` ou `copy.deepcopy` quand on veut un objet indépendant.
- On préfère un objet immuable pour une valeur qui ne doit pas changer pendant l'exécution.
