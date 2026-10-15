# Modules et imports

Un **module** est un fichier Python (`.py`) qui contient des fonctions, des classes et
des variables. Les modules permettent de réutiliser du code écrit par d'autres, et de
découper son propre programme en plusieurs fichiers.

<!-- notes: 25 minutes. Faire la démonstration dans Colab : %%writefile pour créer
calculs_medicaux.py, puis import. Piège classique : après modification du fichier, un
second import ne recharge rien (importlib.reload). Insister sur __name__ == "__main__",
que les étudiants recopient sans le comprendre. -->

---

## D'où viennent les modules

Un module peut venir de trois endroits :

- la **bibliothèque standard**, livrée avec Python : `math`, `statistics`, `datetime`, `random`, `csv`… ;
- un **package installé** : `numpy`, `pandas` ;
- un **fichier local** que l'on a écrit : `calculs_medicaux.py`.

Ils servent à utiliser les fonctionnalités d'un module dans un autre, à structurer un
programme en plusieurs fichiers (programmation **modulaire**) et à profiter des
bibliothèques open source.

---

## Importer

```python
import math
math.sqrt(180 * 75 / 3600)        # surface corporelle de Mosteller, en m²

from statistics import mean, median
mean([5.4, 6.1, 7.8])             # plus besoin du préfixe

import numpy as np                # alias, par convention
from math import sqrt as racine   # alias sur une fonction
```

Seuls les noms importés existent : après `from statistics import mean`, appeler
`median(...)` lève `NameError: name 'median' is not defined`.

---

## Ne pas utiliser `import *`

`from module import *` importe tous les noms d'un coup, sans dire lesquels. Deux
modules peuvent définir le même nom, et le second écrase le premier en silence :

```python
from numpy import *
from math import *
print(sqrt([1, 4, 9]))   # TypeError : c'est math.sqrt qui a gagné
```

`numpy.sqrt` accepte une liste, `math.sqrt` un seul nombre. Avec `import numpy as np`,
`np.sqrt` dit sans ambiguïté laquelle on appelle.

---

## La bibliothèque standard

```python
from datetime import date, datetime
duree = date(2026, 10, 15) - date(2026, 10, 9)
print(duree.days)                                   # 6 jours d'hospitalisation
entree = datetime.strptime("09/10/2026 14:30", "%d/%m/%Y %H:%M")

import statistics
statistics.stdev([5.4, 6.1, 7.8, 5.9])              # écart-type des glycémies

import random
random.seed(42)                                     # résultat reproductible
random.choice(["traitement", "placebo"])            # tirage au sort d'un bras
```

`date.today()` donne la date du jour ; une différence de deux dates est un `timedelta`.

---

## Installer un package

Un package qui n'est pas dans la bibliothèque standard s'installe avec un gestionnaire
de paquets, une seule fois par environnement :

```bash
pip install pandas            # dans un terminal
conda install pandas          # avec Anaconda
```

Dans un notebook (Colab ou Jupyter), on préfixe la commande : `%pip install pandas`.
Colab fournit déjà NumPy, pandas, Matplotlib, seaborn et scikit-learn.

---

## Écrire son propre module

Un fichier `calculs_medicaux.py` placé dans le même dossier :

```python
def imc(poids_kg, taille_m):
    return poids_kg / taille_m ** 2

def glycemie_g_vers_mmol(g_par_l):
    return g_par_l * 5.551          # masse molaire du glucose : 180,16 g/mol
```

s'importe par son nom, sans l'extension :

```python
import calculs_medicaux as cm
cm.imc(72, 1.80)                 # 22.2
```

Dans Colab, la commande magique `%%writefile calculs_medicaux.py` en tête de cellule
écrit le contenu de la cellule dans ce fichier.

---

## Structurer un programme

Un programme se découpe en modules qui contiennent les fonctionnalités, et un
programme principal qui les utilise :

```
mon_projet/
  donnees.py       # chargement des données
  calculs.py       # fonctions et classes
  main.py          # programme principal
```

```python
# main.py — imports : bibliothèque standard, puis packages installés, puis modules locaux
from datetime import date

from calculs import imc

def main():
    print(date.today(), imc(72, 1.80))

if __name__ == "__main__":
    main()
```

---

## `if __name__ == "__main__":`

Chaque module a une variable `__name__`. Elle vaut `"__main__"` quand le fichier est
exécuté directement (`python main.py`), et le nom du module quand il est importé.

Le bloc `if __name__ == "__main__":` ne s'exécute donc que si l'on lance le fichier :
on peut y mettre des essais (`print(imc(72, 1.80))`) sans qu'ils se déclenchent à
chaque `import calculs`. Le programme principal se range dans une fonction `main()`.

---

## À retenir

- `import module`, `from module import nom`, `import module as alias` ; jamais `import *`.
- La bibliothèque standard couvre déjà beaucoup : `math`, `statistics`, `datetime`, `random`, `csv`.
- `pip install` (ou `%pip install` dans un notebook) ajoute un package.
- Un fichier `.py` est un module ; `if __name__ == "__main__":` protège le code de lancement.
