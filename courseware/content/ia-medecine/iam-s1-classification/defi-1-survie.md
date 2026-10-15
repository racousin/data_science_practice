# Défi 1 — Survie à 2 mois avec une régression logistique

Le défi de la séance : prédire si un patient de réanimation est **vivant
ou décédé 2 mois** après son admission. Les données viennent de l'étude
SUPPORT2 : 6 164 patients d'entraînement décrits par 28 variables
(démographie, diagnostic, signes vitaux, biologie, scores de gravité), et
1 541 patients de test dont l'issue est cachée. Comptez environ 30 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s1-defi.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s1-defi-correction.ipynb)
est à ouvrir après avoir cherché.

Le notebook applique la méthode du TP 1 :

1. Charger les données ; proportion de vivants (répondre toujours
   « vivant » donne environ 0,59 : le score à battre).
2. Garder les variables numériques.
3. Un pipeline : valeurs manquantes remplacées par la médiane,
   standardisation, régression logistique ; exactitude mesurée sur 20 %
   des patients réservés à la validation.
4. Réentraîner sur tous les patients et écrire `submission.csv`.

**Soumettre** : déposez `submission.csv` (colonnes `patient_id,outcome`,
une ligne par patient du test, `outcome` valant `alive` ou `dead`) sur la
page du défi :
[https://ml-arena.com/viewchallenge/199](https://ml-arena.com/viewchallenge/199).
Le score est l'**exactitude** sur les patients de test ; le classement
compare votre modèle à ceux de la promotion. Les défis des séances 2 et 3
reprennent les mêmes patients avec les modèles suivants.

<!-- notes: 30 minutes, en fin de séance ou à la maison. La correction
obtient 0,731 d'exactitude en validation (contre 0,594 pour « toujours
vivant »). Faire remarquer que le score du classement est proche de celui
de la validation : c'est le signe d'une évaluation honnête. Les variables
texte (sexe, groupe diagnostique) sont ignorées ici, elles seront encodées
en séance 2. -->
