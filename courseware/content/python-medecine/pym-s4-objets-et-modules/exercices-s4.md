# Exercices — Modéliser un patient

Dans ce notebook, vous écrivez les classes d'un petit dossier médical : un patient
avec un âge calculé et un poids contrôlé, des mesures avec leur unité, une file
d'attente des urgences triée par priorité, une ordonnance, des examens organisés par
héritage, puis un module de calculs cliniques que vous importez. Comptez 75 minutes ;
le livrable est le notebook complété, dont toutes les cellules de vérification
affichent ✅.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s4-exercices.ipynb)

Correction (à ouvrir après avoir cherché) :
[pym-s4-exercices-correction.ipynb](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s4-exercices-correction.ipynb)

Au programme :

1. La classe `Patient` : constructeur, attributs et méthode `imc()`.
2. Trouvez l'erreur : une classe `Perfusion` à qui il manque `self`.
3. Une propriété `age` calculée avec `datetime` et un setter qui refuse un poids négatif.
4. Une classe `Mesure` avec `__str__`, `__repr__`, `__eq__`, `__add__` et une méthode statique de conversion.
5. Trier la file des urgences avec `__lt__` (niveaux de tri FRENCH).
6. Une `Ordonnance` qui contient des `Medicament`, avec `__len__`, `__contains__` et un constructeur alternatif `@classmethod`.
7. L'héritage `Examen` → `ExamenBiologique` / `ExamenImagerie`, appliqué au fichier `examens_biologiques.csv`.
8. Écrire le module `calculs_medicaux.py` (IMC, Cockcroft-Gault, durée de séjour) et l'importer.
9. Pour aller plus loin : 299 dossiers réels d'insuffisance cardiaque, classés par fraction d'éjection.

Deux items « Lire du code » s'intercalent : prédire la sortie avant d'exécuter.

Hors de Colab : téléchargez le fichier `.ipynb` (menu *Fichier* de Colab ou lien
GitHub) et ouvrez-le dans Jupyter. Le notebook n'utilise que la bibliothèque
standard ; il lit deux CSV sur GitHub, il faut donc une connexion internet.

<!-- notes: Les exercices 1 à 6 sont le cœur de la séance ; viser au moins le 7 en
séance, le 8 et le 9 peuvent se finir à la maison. Points de blocage observés :
oublier self (exercice 2 exprès), appeler le setter dans __init__ (exercice 3),
renvoyer une chaîne dans __str__ au lieu de l'afficher (exercice 4), oublier super()
dans l'exercice 7. Pour l'exercice 8, si un étudiant modifie son module, rappeler
que la cellule d'import fait importlib.reload. -->
