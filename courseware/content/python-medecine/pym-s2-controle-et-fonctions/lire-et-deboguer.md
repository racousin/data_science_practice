# Lire et déboguer du code

On passe plus de temps à lire du code qu'à en écrire, et une erreur n'est pas
un échec : c'est un message précis qui dit ce qui ne va pas et où. Cette leçon
apprend à prévoir ce que fait un code, à lire un message d'erreur et à localiser
un bug.

<!-- notes: 20 min. Faire remplir le tableau de trace par la salle, une ligne
par étudiant. Puis provoquer volontairement chacune des quatre erreurs en
direct dans Colab et lire le traceback à voix haute, de bas en haut. Point
où ils calent : ils lisent la première ligne du traceback au lieu de la
dernière. -->

---

## Tracer une boucle à la main

Avant d'exécuter, on suit les variables tour par tour dans un tableau.

```python
total = 0
for h in range(1, 4):
    total = total + 2 * h
print(total)
```

| tour | `h` | `total` après le tour |
|---|---|---|
| 1 | 1 | 2 |
| 2 | 2 | 6 |
| 3 | 3 | 12 |

`range(1, 4)` s'arrête à 3 : le code affiche `12`. Prévoir la sortie, puis
exécuter pour vérifier, est la meilleure façon de comprendre un code.

---

## Lire un message d'erreur

```text
Traceback (most recent call last):
  File "<ipython-input-3>", line 5, in <module>
    print(calculer_imc(70, 0))
  File "<ipython-input-3>", line 2, in calculer_imc
    return poids_kg / taille_m ** 2
ZeroDivisionError: division by zero
```

Un traceback se lit **de bas en haut** : la dernière ligne donne le **type**
d'erreur et son message ; juste au-dessus, la ligne de code fautive et son
numéro ; plus haut, le chemin des appels qui y ont mené.

---

## Quatre erreurs fréquentes

| Erreur | Cause typique | Exemple |
|---|---|---|
| `NameError` | variable mal orthographiée ou non définie | `print(temperture)` |
| `TypeError` | opération entre types incompatibles | `"38" + 1` |
| `IndentationError` | retrait incohérent après `:` | un `if` sans bloc indenté |
| `ZeroDivisionError` | division par zéro | `poids / 0` |

Deux erreurs ne produisent **aucun** message : `range(1, n)` qui saute la
première valeur, et une fonction qui oublie `return` (elle renvoie `None`).
Seule une vérification du résultat les révèle.

---

## Déboguer avec print

Afficher les variables au point suspect est la méthode la plus simple. La
syntaxe `f"{x=}"` affiche le nom et la valeur.

```python
for t in temperatures:
    est_fievre = t >= 38.0
    print(f"{t=} {est_fievre=}")     # t=38.4 est_fievre=True
```

Dans Colab, après une erreur, la commande `%debug` dans une nouvelle cellule
ouvre le débogueur à l'endroit de l'erreur : `p variable` affiche une valeur,
`q` quitte.

---

## Échouer tôt plutôt que se tromper en silence

Un poids négatif ne doit pas produire une dose : mieux vaut arrêter le programme
avec un message clair.

```python
def dose_mg_kg(poids_kg, mg_par_kg):
    if poids_kg <= 0:
        raise ValueError(f"poids invalide : {poids_kg} kg")
    return poids_kg * mg_par_kg
```

`try / except` permet d'intercepter une erreur connue. On ne l'utilise que
lorsqu'on sait quoi faire de l'erreur ; un `except` qui l'avale cache le
problème au lieu de le résoudre.

```python
try:
    dose_mg_kg(-12, 15)
except ValueError as erreur:
    print("Saisie refusée :", erreur)
```

---

## À retenir

- Prévoir la sortie d'un code avant de l'exécuter ; tracer les boucles dans un tableau.
- Un traceback se lit de bas en haut : type d'erreur, puis ligne fautive.
- `print(f"{x=}")` et `%debug` pour inspecter les variables.
- Une erreur claire vaut mieux qu'un résultat faux : `raise`, et pas de `except` qui avale tout.
