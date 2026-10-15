# Pourquoi Python en santé

Python est aujourd'hui le langage le plus utilisé pour analyser des données,
y compris en recherche clinique, en épidémiologie et en imagerie médicale.
Cette leçon situe le langage : d'où il vient, ce qu'il fait bien, ce qu'il fait
moins bien, et ce que l'on peut en attendre dans un service hospitalier ou un
laboratoire.

<!-- notes: 20 minutes. Leçon d'ouverture, sans code à taper. Demander à la salle
qui a déjà utilisé Excel, R, SAS ou SPSS : c'est la référence à laquelle ils vont
comparer. Ne pas s'attarder sur l'historique. -->

---

## Un langage né en 1991

Python a été créé par Guido van Rossum à partir de 1989 et rendu public en
1991. Son nom vient des *Monty Python*, pas du serpent. Van Rossum en a dirigé
l'évolution jusqu'en 2018 ; le langage est depuis gouverné par un comité élu
de la Python Software Foundation.

Python est libre, gratuit, multiplateforme (Windows, macOS, Linux) et
documenté sur [python.org](https://www.python.org).

```python
print("Bonjour, service de cardiologie")
```

Une ligne suffit pour un premier programme : pas de déclaration, pas de
compilation, pas de fonction `main`.

<!-- notes: Montrer que la ligne s'exécute telle quelle dans Colab. -->

---

## Un langage interprété, simple à écrire

Un programme Python est lu et exécuté **ligne par ligne** par un
*interpréteur*. On peut donc taper une instruction et voir le résultat
immédiatement, de façon **interactive** (Colab, Jupyter, shell), ou exécuter un
fichier entier, un **script**.

```python
poids_kg = 72
taille_m = 1.78
imc = poids_kg / taille_m ** 2
print(imc)        # 22.724403484408533
```

La syntaxe est proche du langage courant et les erreurs sont signalées avec la
ligne fautive : corriger un bogue est relativement simple, ce qui en fait un
bon premier langage.

---

## Mais relativement peu performant

![Langages de haut et de bas niveau](assets/pym-s1-premiers-pas/abstraction.png)

Python est un langage de **haut niveau** : éloigné de la machine, proche du
problème. Ce confort a un coût en vitesse, pour quatre raisons :
l'interprétation (au lieu d'une compilation en code machine, comme en C),
des structures de données très générales, le **typage dynamique** (le type
d'une variable est vérifié à chaque opération) et la gestion automatique de la
mémoire.

En pratique, les bibliothèques de calcul (NumPy, pandas, scikit-learn) sont
écrites en C ou en Fortran : le code Python ne fait que les piloter. Une
analyse de 300 000 séjours hospitaliers prend quelques secondes.

<!-- notes: Message à faire passer : "lent" signifie lent pour une boucle écrite
à la main sur des millions d'éléments, pas lent pour une analyse de données. On
y reviendra en séance 5 (vectorisation NumPy). -->

---

## Une histoire de versions

![Calendrier de support des versions de Python](assets/pym-s1-premiers-pas/python-versions.png)

Python 3 (2008) n'est pas compatible avec Python 2 : `print "x"` y devient
`print("x")`, et `7 / 2` vaut `3.5` au lieu de `3`. Python 2 n'est plus maintenu
depuis le 1er janvier 2020 : on rencontre encore du vieux code, on n'en écrit
plus.

Chaque version 3.x est maintenue cinq ans. Ce cours demande **Python 3.10 ou
plus récent** ; Google Colab fournit une version récente, que l'on vérifie
avec :

```python
import sys
print(sys.version)
```

---

## Python en santé

| Domaine | Exemples | Bibliothèques |
|---|---|---|
| Analyse de données cliniques | cohortes, registres, essais | pandas, NumPy, SciPy, statsmodels |
| Visualisation | courbes de survie, constantes vitales | Matplotlib, seaborn |
| Épidémiologie | incidence, modèles SIR, survie | pandas, lifelines |
| Imagerie médicale | lire un DICOM, segmenter une IRM | pydicom, nibabel, scikit-image |
| Apprentissage automatique | prédire une réadmission, classer une lésion | scikit-learn, PyTorch |
| Automatisation | comptes rendus, extraction d'un export du DPI | bibliothèque standard, pandas |

La force de Python tient moins au langage qu'à cet **écosystème** : des
centaines de milliers de bibliothèques libres, installables en une commande
(`pip install …`), et une communauté scientifique qui publie son code en
Python.

<!-- notes: Faire citer aux étudiants une tâche répétitive de leur pratique
(copier des résultats de biologie dans un tableau, recalculer une clairance…) :
c'est le fil rouge des exercices. -->

---

## Forces et limites

**Forces** : lisible, gratuit, reproductible (le script *est* la trace de
l'analyse, contrairement à une suite de clics dans un tableur), et utilisable
du petit calcul jusqu'au modèle d'apprentissage profond.

**Limites** : plus lent qu'un langage compilé pour du calcul écrit à la main,
moins adapté aux applications embarquées ou temps réel (un dispositif médical
certifié), et la liberté du typage dynamique laisse passer des erreurs qu'un
compilateur aurait refusées : il faut **tester** son code (séance 7).

Les calculs cliniques de ce cours (IMC, clairance, scores) sont des **exemples
pédagogiques**, pas des outils d'aide à la décision clinique.

---

## À retenir

- Python est un langage interprété, de haut niveau, libre et multiplateforme.
- Il est lent pour des boucles écrites à la main, rapide via ses bibliothèques
  (NumPy, pandas) écrites en C.
- On utilise Python 3 (≥ 3.10) ; Python 2 est abandonné depuis 2020.
- Son intérêt en santé vient de l'écosystème : données, statistiques, images,
  apprentissage automatique, automatisation.
