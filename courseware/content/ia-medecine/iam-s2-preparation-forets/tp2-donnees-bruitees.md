# TP 2 — Données bruitées et forêt aléatoire

On reprend le jeu de données du cancer du sein de la séance 1, cette fois
**bruité** comme de vraies données cliniques : 866 valeurs manquantes, des
mesures aberrantes (une aire de 7 400 pixels² pour une médiane de 560) et
trois variables catégorielles (tranche d'âge, statut ménopausique,
localisation de la tumeur). Vous le préparez étape par étape, puis vous
comparez une forêt aléatoire à la régression logistique. Comptez environ
1 h 30.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s2-tp.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s2-tp-correction.ipynb)
est à ouvrir après avoir cherché.

1. Charger les données et inspecter leur qualité (`info`, `describe`).
2. Visualiser les défauts : carte des valeurs manquantes, boîtes à
   moustaches, répartition des variables catégorielles.
3. Imputer : médiane pour les variables numériques, modalité la plus
   fréquente pour les catégorielles.
4. Écrêter les valeurs aberrantes aux bornes IQR.
5. Encoder les variables catégorielles (one-hot).
6. Séparer apprentissage et test, standardiser.
7. Entraîner la régression logistique de référence.
8. Entraîner une forêt aléatoire : matrices de confusion, importances des
   variables.
9. Régler les hyperparamètres de la forêt avec `GridSearchCV`.
10. Comparer les deux modèles en validation croisée.

Complétez les cellules marquées `# TODO` en remplaçant les `...` ; les
indices sous chaque énoncé donnent la forme du code. Hors de Colab : ouvrez
le `.ipynb` avec `jupyter lab` ou VS Code (pandas, seaborn et scikit-learn
nécessaires).

<!-- notes: Faire les exercices 1.1 à 2.3 ensemble (20 min), puis laisser
travailler. Points de blocage attendus : 3.2, oublier de réaffecter le
résultat de fit_transform à df ; 5.1, compter les colonnes créées par
drop_first ; 9.1, la recherche prend environ une minute (n_jobs=-1). Le TP
impute et écrête tout le tableau avant la séparation, comme le TP d'origine :
une remarque le signale au § 7, et c'est l'occasion de redire la règle de la
fuite avant le défi. Résultat attendu : régression logistique et forêt
autour de 0,96 ; ici la forêt ne bat pas la référence, et c'est un résultat. -->
