# TP 3 — Maladie cardiaque et réseaux de neurones

Dans ce notebook, vous prédisez la présence d'une maladie cardiaque chez
270 patients décrits par 13 variables cliniques (jeu *Statlog Heart* :
âge, type de douleur thoracique, fréquence cardiaque maximale à l'effort,
sous-décalage du segment ST, scintigraphie au thallium…). Vous appliquez
la démarche des séances 1 et 2, puis vous entraînez un réseau de neurones
et comparez honnêtement trois modèles. Comptez environ 1 h 30.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-tp.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-tp-correction.ipynb)
est à ouvrir après avoir cherché.

1. Charger les données ; types des variables (lesquelles sont des
   catégories codées en nombres ?) ; équilibre des classes.
2. Visualiser : distributions par classe, nuage de points, carte des
   corrélations.
3. Découper (80 / 20) et standardiser.
4. Modèle de référence : régression logistique.
5. Forêt aléatoire, exactitude sur l'entraînement et sur le test.
6. Réseau de neurones : un premier `MLPClassifier`, sa courbe de perte,
   puis l'arrêt précoce.
7. Comparer quatre architectures ; recherche sur grille.
8. Comparaison finale par validation croisée ; matrices de confusion des
   trois modèles côte à côte.

Complétez les cellules marquées `# TODO`. Les cellules « Votre réponse »
attendent une ou deux phrases.

Hors de Colab : ouvrez le `.ipynb` téléchargé avec `jupyter lab` ou VS Code
(paquets : pandas, numpy, matplotlib, seaborn, scikit-learn).

<!-- notes: 1 h 30. Faire un point collectif après l'exercice 5.1 (la forêt
sur-apprend : 1,00 en entraînement, 0,76 en test) puis après 6.3 (arrêt
précoce). Résultats attendus de la correction : régression logistique 0,907
en test (3 malades manqués sur 21) ; forêt 0,759 ; MLP par défaut 0,833
après 839 époques (0,995 en entraînement) ; avec arrêt précoce 0,815 en 27
époques mais rappel 0,81 au lieu de 0,71 ; recherche sur grille : tanh,
(50,), 0,001, 0,870 en test ; validation croisée : 0,837 ± 0,036 (LR),
0,819 ± 0,055 (forêt), 0,819 ± 0,043 (MLP). Message final : sur 270
patients, la régression logistique suffit. La recherche sur grille
(exercice 7.2) prend quelques secondes. -->
