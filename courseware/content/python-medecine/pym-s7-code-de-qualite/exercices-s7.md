# Exercices — Le package medcalc

Dans ce notebook, vous construisez pas à pas un petit package Python, `medcalc`
(conversions de glycémie et de créatininémie, IMC, clairance de Cockcroft-Gault,
score qSOFA), vous le testez avec pytest, vous trouvez un bug grâce aux tests, puis vous
améliorez la qualité du code avec pylint. Comptez environ 1 h 30. Le livrable est le
notebook complété, toutes les cellules de vérification affichant ✅.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s7-exercices.ipynb)

Correction, à ouvrir après avoir cherché :
[pym-s7-exercices-correction.ipynb](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s7-exercices-correction.ipynb)

Au programme :

- Lire du code : `__name__` à l'import et au lancement d'un script.
- Le squelette du package : `README.md`, `pyproject.toml`, `pip install -e .`.
- Exercice 1 — Le module `conversions` (glycémie g/L ↔ mmol/L, créatinine µmol/L ↔ mg/dL).
- Exercice 2 — Le module `scores` (IMC, Cockcroft-Gault, qSOFA).
- Exercice 3 — Ce que le package expose, dans `__init__.py`.

- Lire du code : quels tests pytest collecte-t-il, et pourquoi `0.1 + 0.2 == 0.3` échoue.
- Exercice 4 — Tester les conversions (`pytest.approx`, `pytest.raises`).
- Exercice 5 — Tester les scores avec `@pytest.mark.parametrize` et des valeurs seuils.
- Exercice 6 — Trouvez le bug grâce aux tests : une classification de la tension artérielle.
- Lire du code : repérer les écarts à la PEP 8 avant de lancer pylint.
- Exercice 7 — Réécrire un fichier mal écrit jusqu'à 9/10 sous pylint.
- Exercice 8 — pylint sur le package entier.
- Pour aller plus loin — Exercice 9 : 100 % de couverture avec `pytest-cov`.

Les vérifications des exercices 4 et 5 injectent des bugs dans une copie du package :
des tests qui passent sans rien vérifier ne suffisent pas.

Hors de Colab : téléchargez le `.ipynb`, installez
`pip install jupyter pytest pylint pytest-cov`, puis ouvrez-le avec Jupyter. Le notebook
travaille dans un dossier `medcalc_projet` du répertoire temporaire du système ; vous
pouvez aussi reproduire le projet dans un vrai dossier et lancer `pytest` et `pylint`
depuis un terminal.

<!-- notes: 90 minutes. Les étudiants bloquent surtout sur trois points :
(1) oublier de ré-exécuter une cellule %%writefile après l'avoir modifiée ;
(2) l'exercice 5, où le test qSOFA sans valeur seuil (FR = 22) laisse passer le
bug "> 22" : c'est le message à faire passer ; (3) l'exercice 6, où il faut lire
le rapport pytest plutôt que relire le code au hasard (deux bugs : "> 180" et
"and" au lieu de "or"). Si un étudiant est perdu, ré-exécuter la cellule de
préparation repart d'un dossier vide. L'exercice 9 est facultatif. -->
