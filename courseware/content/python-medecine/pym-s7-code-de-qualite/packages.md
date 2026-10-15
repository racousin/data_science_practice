# Structurer un projet en package

Un script qui grossit finit par mélanger les calculs, l'affichage et les essais.
Cette leçon range le code en modules, puis en un package installable : `medcalc`,
une petite bibliothèque de calculs cliniques que l'on construit pendant toute la séance.

<!-- notes: 30 minutes. Partir du rappel de la séance 4 (modules et imports) :
la plupart des étudiants ont déjà écrit un module, peu savent pourquoi
"python main.py" échoue dès qu'on change de dossier. Les étudiants bloquent
sur la différence entre le dossier du projet (medcalc_projet/) et le dossier
du package (medcalc/) : la dessiner au tableau. -->

---

## Rappel : un programme en plusieurs fichiers

Version simple : un ou plusieurs modules qui contiennent les fonctions, et un programme
principal qui les utilise.

```
mon_projet/
  - calculs.py
  - main.py
```

```python
# calculs.py
def imc(poids_kg, taille_m):
    return poids_kg / taille_m ** 2

if __name__ == "__main__":
    print(imc(70, 1.75))
```

Le bloc `if __name__ == "__main__":` ne s'exécute que si l'on lance le fichier
(`python calculs.py`). Quand un autre fichier fait `import calculs`, `__name__` vaut
`"calculs"` et le bloc est ignoré.

---

## Le problème du dossier courant

```python
# main.py
from calculs import imc
print(imc(70, 1.75))
```

`python main.py` fonctionne si `main.py` et `calculs.py` sont dans le même dossier :
Python cherche les modules dans le dossier du script lancé. Depuis un autre dossier,
un notebook ou un autre projet, `import calculs` lève `ModuleNotFoundError`.

La solution durable est d'en faire un **package installable** : une fois installé,
il s'importe de partout, comme `numpy`.

---

## Qu'est-ce qu'un package ?

Un package Python est :

1. un **ensemble de modules** organisés en dossiers et sous-dossiers ;
2. une bibliothèque dédiée à un **domaine** ;
3. **installable** avec `pip` (ou `conda`) ;
4. **réutilisable** depuis n'importe quel autre code ou package.

NumPy (calcul numérique), pandas (tableaux de données), Matplotlib (graphiques) et SciPy
(calcul scientifique) sont des packages. Le nôtre, `medcalc`, regroupera conversions
d'unités et scores cliniques.

---

## L'arborescence d'un package

```
medcalc_projet/            <- dossier du projet
  - README.md
  - pyproject.toml
  - medcalc/               <- le package lui-même
    - __init__.py
    - conversions.py
    - scores.py
  - tests/
    - test_conversions.py
    - test_scores.py
```

- `README.md` documente le package, en Markdown (comme les cellules de texte d'un
  notebook) : ce qu'il fait, comment l'installer, un exemple d'utilisation.
- `medcalc/__init__.py` fait du dossier un package importable. Il est souvent vide ;
  il peut porter la version et exposer les fonctions principales.
- Un sous-dossier de `medcalc/` qui contient du code a lui aussi son `__init__.py`.

```python
# medcalc/__init__.py
__version__ = "0.1.0"
from medcalc.scores import imc, qsofa
```

---

## Le fichier `pyproject.toml`

Il décrit le package et la façon de l'installer, en blocs (format TOML).

```toml
[project]
name = "medcalc"                 # nom utilisé par pip
version = "0.1.0"
description = "Calculs cliniques usuels (exemples pédagogiques)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [                 # bibliothèques nécessaires
    "pandas>=2.2",
]
```

Le bloc `[build-system]` indique l'outil qui construit le package ; `setuptools`
trouve ensuite les dossiers à inclure.

```toml
[build-system]
requires = ["setuptools>=77.0.3"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["medcalc*"]
```

Référence : packaging.python.org, *Writing your pyproject.toml*.

---

## Installer le package

Depuis le dossier du projet (celui qui contient `pyproject.toml`) :

```bash
pip install .        # copie le package dans l'environnement Python
pip install -e .     # mode éditable : Python lit directement vos fichiers
```

En mode **éditable** (`-e`), une modification de `scores.py` est prise en compte sans
réinstaller : c'est le mode de travail pendant le développement. Ensuite, de n'importe
quel dossier :

```python
from medcalc import imc
imc(70, 1.75)        # 22.86
```

Dans un notebook déjà démarré, un package installé pendant la session n'est visible
qu'après redémarrage du noyau (ou en ajoutant son dossier à `sys.path`).

---

## À retenir

- `if __name__ == "__main__":` sépare le code à exécuter du code à importer.
- Un package = un dossier avec `__init__.py`, documenté par un `README.md` et décrit
  par un `pyproject.toml`.
- `pip install -e .` installe le package en mode éditable : on l'importe de partout.
