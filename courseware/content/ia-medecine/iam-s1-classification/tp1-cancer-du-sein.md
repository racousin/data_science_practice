# TP 1 — Cancer du sein et régression logistique

Dans ce notebook, vous prédisez si une tumeur du sein est maligne ou
bénigne à partir de 30 mesures des noyaux cellulaires d'une cytoponction
(jeu *Breast Cancer Wisconsin*, 569 biopsies). Vous suivez la méthode de
la séance de bout en bout : explorer, visualiser, découper, entraîner une
régression logistique, l'évaluer, standardiser les variables et valider
par validation croisée. Comptez environ 1 h 30.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s1-tp.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s1-tp-correction.ipynb)
est à ouvrir après avoir cherché.

1. Charger les données ; statistiques descriptives ; équilibre des classes.
2. Visualiser : distribution d'une variable par classe, nuage de points de
   deux variables, carte des corrélations avec la cible.
3. Découper en entraînement et test (80 / 20) ; entraîner une régression
   logistique.
4. Évaluer : exactitude, matrice de confusion, précision, rappel et F1
   calculés à la main puis avec scikit-learn, rapport de classification.
5. Standardiser les variables et comparer.
6. Validation croisée à 5 blocs.

Complétez les cellules marquées `# TODO` en remplaçant les `...`. Les
cellules « Votre réponse » attendent une ou deux phrases.

Attention au codage de scikit-learn : **maligne = 0**, bénigne = 1. La
classe que l'on cherche à détecter (la classe positive) est la maligne,
d'où `pos_label=0` dans les fonctions de métriques.

Hors de Colab : ouvrez le `.ipynb` téléchargé avec `jupyter lab` ou VS Code
(paquets : pandas, numpy, matplotlib, seaborn, scikit-learn).

<!-- notes: 1 h 30. Laisser travailler, faire un point collectif après
l'exercice 3.2 (premier modèle entraîné) puis après 4.3. Points de blocage
attendus : ex. 2.3, sélectionner les 10 noms de colonnes (sort_values puis
head puis index) ; ex. 4.3, lire la matrice avec maligne = 0 comme classe
positive (VP = cm[0, 0], FN = cm[0, 1]) ; ex. 5.1, fit_transform sur
l'entraînement et transform seul sur le test. Résultats attendus :
exactitude 0,956 puis 0,974 après standardisation ; 4 cancers manqués sur 43 ;
validation croisée 0,981 ± 0,007. -->
