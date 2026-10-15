# TP bonus — La descente de gradient pas à pas

Ce notebook facultatif ouvre la boîte de l'entraînement : vous déroulez à
la main la descente de gradient sur une fonction d'une variable, observez
l'effet du taux d'apprentissage, puis ajustez une régression linéaire avec
scikit-learn sur des données simulées dont vous connaissez les vrais
paramètres. Comptez environ 45 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-bonus.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-s3-bonus-correction.ipynb)
est à ouvrir après avoir cherché.

1. Un pas de descente de gradient sur $f(\theta) = (3\theta - 7)^2$, puis
   vingt pas.
2. Le taux d'apprentissage : 0,01, 0,05, 0,1, 0,12 ; quand la descente
   diverge-t-elle ?
3. Une autre fonction : écrire $f_2(\theta) = \theta^2 + 5\theta + 6$, sa dérivée
   et son minimum.
4. Régression linéaire avec `LinearRegression` : retrouver le poids et le
   biais des données simulées.
5. L'effet du bruit, du nombre d'exemples et du nombre de variables.

Les exercices consistent surtout à changer une valeur marquée `# TODO`,
exécuter et interpréter.

---

## Le taux d'apprentissage

Un taux trop petit avance lentement, un taux trop grand fait osciller puis
diverger, et une fonction non convexe peut piéger la descente dans un
minimum local.

![Taux d'apprentissage trop petit, trop grand, minimum local](assets/iam-s3-reseaux-de-neurones/taux-apprentissage.png)

<!-- notes: À faire à la maison ou par les étudiants rapides. Point clé de
l'exercice 2 : ici le pas multiplie l'écart au minimum par (1 − 18 η) ;
au-delà de η = 1/9 ≈ 0,11 la descente diverge (0,12 → θ ≈ −43 après 20 pas).
Le schéma vient du cours 1 (diapositive 33). LinearRegression n'utilise pas
la descente de gradient (formule exacte) : la descente sert quand il n'y en
a pas, comme pour les réseaux. -->
