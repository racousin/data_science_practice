# PEP 8 et pylint

Un code est lu bien plus souvent qu'il n'est écrit : par vous dans six mois, par un
collègue, par un relecteur. Des conventions communes rendent cette lecture rapide, et
un outil, `pylint`, les vérifie automatiquement.

<!-- notes: 25 minutes. Ne pas lire toute la PEP 8 : le tableau bon/mauvais suffit,
le reste se découvre avec pylint. Faire tourner pylint en direct sur un fichier mal
écrit et corriger deux ou trois messages devant la salle. Les étudiants bloquent sur
la lecture des codes de message (C0114, W0611) : montrer qu'on peut les chercher tels
quels dans un moteur de recherche. -->

---

## La PEP 8

Les *Python Enhancement Proposals* (PEP) sont les documents qui font évoluer le langage.
La **PEP 8** fixe les conventions de style du code Python ; la **PEP 257** celles des
docstrings.

Page officielle : peps.python.org/pep-0008

Ces règles sont assez strictes, mais elles ne demandent aucun effort une fois devenues
une habitude — et les outils les vérifient pour vous.

---

## Nommer

| Élément | Convention | Exemple |
|---|---|---|
| package, module | court, minuscules | `medcalc`, `scores.py` |
| fonction, variable | `minuscules_avec_tirets_bas` | `calculer_imc`, `poids_kg` |
| classe | `MotsEnCapitales` | `Patient`, `Ordonnance` |
| constante (globale) | `MAJUSCULES` | `FACTEUR_GLUCOSE = 5.551` |

Des noms explicites (`creatinine_umol` plutôt que `c`, jamais `l`, `O` ou `I`), sans
accents ni caractères spéciaux. L'anglais est la règle pour du code partagé largement ;
des noms français sont acceptables s'ils sont cohérents dans tout le projet.

---

## Mettre en forme

| Règle | Correct | Incorrect |
|---|---|---|
| indentation : 4 espaces, jamais de tabulation | `····return x` | `→return x` |
| un espace autour de `=` et des comparaisons | `x = 4`, `if x == 4:` | `x=4`, `if x==4 :` |
| pas d'espace collé aux parenthèses | `imc(poids[0], 1.7)` | `imc( poids [0] , 1.7 )` |
| pas d'espace autour de `=` d'un argument | `def f(x, unite="g/L"):` | `def f(x, unite = "g/L"):` |
| pas d'alignement artificiel | `poids = 70` | `poids         = 70` |
| comparer à `None` avec `is` | `if valeur is None:` | `if valeur == None:` |
| une ligne : 79 caractères | coupée entre parenthèses | une ligne de 140 caractères |

Les espaces servent aussi à montrer les priorités : `x = x*2 - 1` se lit mieux que
`x=x*2-1`. Guillemets simples ou doubles : au choix, mais toujours les mêmes.

---

## Lignes longues, imports, docstrings

Une ligne trop longue se coupe à l'intérieur des parenthèses, indentée de 4 espaces :

```python
clairance = clairance_cockcroft_gault(
    age, poids_kg, creatinine_umol, femme=True
)
```

Un import par ligne, en tête de fichier, en trois groupes séparés par une ligne vide :
bibliothèque standard, bibliothèques installées, code local.

```python
import math
import sys

import numpy as np

from medcalc.conversions import creatinine_umol_vers_mgdl
```

---

## Docstrings

Une docstring en tête de module et de chaque fonction dit ce qu'elle fait, ses
arguments et ce qu'elle renvoie :

```python
def imc(poids_kg, taille_m):
    """Indice de masse corporelle.

    Args:
        poids_kg: poids en kilogrammes.
        taille_m: taille en mètres.

    Returns:
        L'IMC en kg/m².
    """
    return poids_kg / taille_m ** 2
```

Les fonctions sont séparées par deux lignes vides, les blocs d'une fonction par une
seule ligne vide au plus.

---

## pylint : une note sur 10

`pylint` analyse le code, signale chaque écart aux conventions (et des erreurs
probables : variable inutilisée, import manquant) et attribue une note sur 10.

```bash
python -m pylint medcalc            # tout le package
python -m pylint mauvais_code.py    # un seul fichier
```

```
mauvais_code.py:1:0: C0114: Missing module docstring (missing-module-docstring)
mauvais_code.py:3:0: C0103: Function name "CalculIMC" doesn't conform to
    snake_case naming style (invalid-name)
mauvais_code.py:5:5: C0121: Comparison 'l == None' should be 'l is None'
mauvais_code.py:2:0: W0611: Unused import math (unused-import)
...
Your code has been rated at 0.00/10
```

Chaque ligne donne fichier, ligne:colonne, code et explication. La lettre indique la
gravité : **C** convention, **R** refactorisation, **W** avertissement, **E** erreur.
`ruff` (`ruff check .`) est une alternative récente, beaucoup plus rapide.

---

## À vous de jouer

1. À la racine de votre projet, lancez `python -m pylint medcalc`.
2. Quelle est votre note sur 10 ? Quels sont les messages les plus fréquents ?
3. Corrigez le code, relancez `pylint`, et relancez les tests : une correction de
   style ne doit jamais changer un résultat.
