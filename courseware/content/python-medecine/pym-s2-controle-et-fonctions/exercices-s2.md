# Exercices — Scores cliniques

Vous programmez des règles cliniques réelles avec des conditions, des boucles
et des fonctions. Comptez environ 75 minutes. Le livrable est le notebook avec
toutes les cellules de vérification au vert (« ✅ Exercice N réussi »).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s2-exercices.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s2-exercices-correction.ipynb)
est à ouvrir après avoir cherché.

**Les exercices**

1. IMC et classes de l'OMS : une chaîne `if / elif / else`.
2. Classer une pression artérielle selon ESC/ESH 2018 (la catégorie la plus élevée l'emporte).
3. Lire du code : booléens et conditions — prévoir la sortie avant d'exécuter.
4. Le score qSOFA (Sepsis-3) : additionner des critères.
5. Le score CHA₂DS₂-VASc : paramètres nommés avec valeurs par défaut.
6. Posologie pédiatrique du paracétamol : dose plafonnée, `ValueError`, `lambda`.
7. Boucle `for` : compter les mesures et les épisodes de fièvre.
8. Lire du code : boucles, `range` et portée des variables.
9. Boucle `while` : temps d'élimination d'un médicament, deux valeurs renvoyées.
10. Trouvez l'erreur : trois fonctions fausses sans message d'erreur.
11. Pour aller plus loin (facultatif) : surveiller un patient heure par heure.

Hors de Colab, téléchargez le `.ipynb` (Fichier → Télécharger) et ouvrez-le
dans Jupyter ou VS Code : aucune bibliothèque à installer.

<!-- notes: Les exercices 1 à 7 sont le cœur de la séance ; 8 à 10 se font en
fin de séance ou à la maison, 11 est réservé aux plus rapides. Les
exercices 3 et 8 se corrigent à l'oral : demander les prédictions à main levée
avant d'exécuter. Points de blocage attendus : la règle « catégorie la plus
élevée » de l'exercice 2 (orienter vers deux fonctions de rang puis max), des
elif au lieu de if indépendants dans qSOFA (la vérification le signale), et
l'état « en fièvre » de l'exercice 7. -->
