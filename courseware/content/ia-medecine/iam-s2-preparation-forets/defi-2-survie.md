# Défi 2 — Survie à 2 mois avec une forêt aléatoire

Même tâche qu'au défi 1 : prédire si un patient de réanimation de l'étude
SUPPORT2 est **vivant (`alive`) ou décédé (`dead`) 2 mois** après
l'admission, à partir de 28 variables mesurées à l'admission. Cette fois,
avec la méthode de la séance 2 : une préparation complète des données, une
**forêt aléatoire** et un petit réglage de ses hyperparamètres. Comptez
environ 30 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s2-defi.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s2-defi-correction.ipynb)
est à ouvrir après avoir cherché.

Le notebook suit la méthode dans le bon ordre, sans fuite d'information :

1. mettre de côté 20 % des patients pour la **validation**, avant toute
   préparation ;
2. **écrêter** les valeurs aberrantes aux bornes IQR calculées sur
   l'apprentissage ;
3. **préparer** dans un `ColumnTransformer` : imputation par la médiane avec
   indicateurs de valeur manquante, modalité la plus fréquente puis one-hot ;
4. entraîner une **forêt aléatoire** dans un `Pipeline` ;
5. régler `max_depth`, `min_samples_leaf` et `max_features` avec
   `GridSearchCV` (validation croisée à 5 plis), puis mesurer une seule fois
   sur la validation ;
6. réentraîner sur tous les patients d'apprentissage et écrire
   `submission.csv`.

## Soumettre

Le fichier `submission.csv` a deux colonnes, `patient_id,outcome`, et une
ligne par patient du test (1541 lignes). Une cellule du notebook vérifie son
format, la suivante le télécharge depuis Colab. Déposez-le sur la page du
défi : [ml-arena.com/viewchallenge/200](https://ml-arena.com/viewchallenge/200).

Repères : prédire `alive` pour tout le monde donne environ 0,59 ; la
correction obtient environ 0,78 en validation. Votre score de validation
annonce votre score sur ML-Arena, à quelques points près.

<!-- notes: 30 minutes. Le message : la validation honnête (séparée avant
toute préparation) prédit le classement. Une forêt réglée gagne environ
un point sur la régression logistique avec la même préparation (0,78 contre
0,77 en validation) : la préparation compte autant que le modèle. Pour les
rapides : essayer d'autres valeurs de la grille, ou retirer
add_indicator=True et mesurer ce que l'on perd. -->
