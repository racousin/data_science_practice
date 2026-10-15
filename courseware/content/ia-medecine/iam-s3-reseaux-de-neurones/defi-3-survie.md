# Défi 3 — Survie à 2 mois avec un réseau de neurones

Même défi qu'aux séances 1 et 2, avec le modèle du jour : prédire si un
patient de réanimation de l'étude SUPPORT2 est **vivant ou décédé 2 mois**
après son admission (6 164 patients d'entraînement, 28 variables, 1 541
patients de test dont l'issue est cachée). Comptez environ 30 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-defi.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-defi-correction.ipynb)
est à ouvrir après avoir cherché.

Le notebook :

1. Charge les données.
2. Prépare les variables dans un `ColumnTransformer` : médiane et
   standardisation pour les variables numériques, valeur la plus fréquente
   et encodage one-hot pour les variables texte.
3. Évalue sur le même jeu de validation (20 % des patients) la régression
   logistique de la séance 1 et la forêt aléatoire de la séance 2.
4. À vous : un réseau `MLPClassifier((64, 32))` sans puis avec arrêt
   précoce ; comparer l'exactitude et le rappel des décès.
5. Réentraîne le modèle choisi sur tous les patients et écrit
   `submission.csv`.

**Soumettre** : déposez `submission.csv` (colonnes `patient_id,outcome`,
une ligne par patient du test, `outcome` valant `alive` ou `dead`) sur la
page du défi :
[https://ml-arena.com/viewchallenge/201](https://ml-arena.com/viewchallenge/201).
Le score est l'**exactitude** sur les patients de test ; répondre toujours
« vivant » donne environ 0,59.

<!-- notes: 30 minutes. Résultats de la correction en validation (1 233
patients) : régression logistique 0,762, forêt 0,765, MLP sans arrêt précoce
0,705 (sur-apprentissage), MLP avec arrêt précoce 0,777 (arrêt après 20
époques). Le réseau ne gagne qu'une vingtaine de patients sur les modèles
simples : bon moment pour reprendre le message de la leçon « Comparer des
modèles ». -->
