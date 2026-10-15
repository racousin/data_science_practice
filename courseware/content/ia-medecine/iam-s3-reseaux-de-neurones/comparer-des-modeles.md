# Comparer des modèles

Trois modèles sont maintenant disponibles : régression logistique, forêt
aléatoire, réseau de neurones. Lequel choisir pour dépister une maladie
cardiaque ? La réponse dépend d'une comparaison **honnête** (mêmes
données, même évaluation), de la **métrique** qui compte cliniquement, et
de ce que l'on doit pouvoir **expliquer**.

<!-- notes: 20 minutes. Leçon de synthèse des trois séances. Les chiffres
sont ceux de la correction du TP 3 (maladie cardiaque, 270 patients). -->

---

## Une comparaison honnête

Pour comparer deux modèles, tout le reste doit être identique :

- le **même découpage** entraînement / test (même `random_state`) ;
- la **même préparation**, ajustée sur l'entraînement seulement ;
- la **même métrique**, calculée sur des patients que le modèle n'a pas vus.

Le jeu de test sert **une seule fois**, à la fin. Choisir un modèle ou un
hyperparamètre en regardant le test revient à l'apprendre : son score
devient trop optimiste.

<!-- notes: 3 minutes. Même règle qu'en séance 2 pour la recherche sur
grille : les choix se font en validation, le test ne sert qu'au bilan. -->

---

## Un seul découpage ne suffit pas

Sur les 54 patients de test du TP, un patient vaut près de 2 points
d'exactitude. La **validation croisée** à 5 plis (séance 2) donne cinq
scores par modèle : leur moyenne et leur écart type.

```python
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
scores = cross_val_score(modele, X, y, cv=5, scoring="accuracy")
scores.mean(), scores.std()
```

Le pipeline refait la standardisation dans chaque pli, sans fuite.

| Modèle | Exactitude (5 plis) |
|---|---|
| Régression logistique | 0,84 ± 0,04 |
| Forêt aléatoire | 0,82 ± 0,06 |
| Réseau de neurones | 0,82 ± 0,04 |

Les écarts entre modèles sont plus petits que les écarts d'un pli à
l'autre : sur 270 patients, aucun modèle n'est nettement meilleur.

<!-- notes: 5 minutes. Faire remarquer que le réseau, réglé par recherche
sur grille, ne bat pas la régression logistique : sur peu de données
tabulaires, c'est la règle plutôt que l'exception. -->

---

## La métrique qui compte cliniquement

Pour un dépistage, l'erreur grave est le **faux négatif** : un malade
déclaré sain. Le **rappel** mesure la part des malades détectés :

$$
\text{rappel} = \frac{VP}{VP + FN}
$$

![Matrices de confusion des trois modèles](assets/iam-s3-reseaux-de-neurones/matrices-confusion.png)

Sur le test du TP, la régression logistique manque 3 malades sur 21, le
réseau (avec arrêt précoce) 4, la forêt 8. À exactitude voisine, on
préfère le modèle qui manque le moins de malades ; pour en manquer encore
moins, on peut abaisser le seuil de décision de 0,5, au prix de plus de
fausses alertes.

<!-- notes: 5 minutes. Lire la case en bas à gauche de chaque matrice.
Rappel de la séance 1 : le seuil se règle avec predict_proba. -->

---

## Interprétabilité ou performance

| | Régression logistique | Forêt aléatoire | Réseau de neurones |
|---|---|---|---|
| Frontière | linéaire | non linéaire | non linéaire |
| Standardisation | importante | inutile | indispensable |
| Explication | coefficients | importance des variables | boîte noire |
| Données nécessaires | peu | moyennes | beaucoup |

Un médecin doit pouvoir justifier une décision. Un coefficient se lit :
« à autres variables égales, un angor d'effort augmente le risque ». Les
1 501 poids d'un réseau ne se lisent pas. À performance égale, le modèle
le plus simple et le plus explicable l'emporte.

<!-- notes: 4 minutes. Les réseaux de neurones deviennent indispensables
sur des images, des signaux ou du texte, et avec beaucoup de données ; pas
sur 270 patients décrits par 13 variables. -->

---

## À retenir

- Même découpage, même préparation, même métrique ; le test sert une fois.
- Validation croisée : comparer des moyennes et regarder les écarts types.
- En dépistage, regarder le rappel et les faux négatifs, pas seulement
  l'exactitude.
- Le choix de la métrique compte plus que le choix du modèle.
- À performance voisine, préférer le modèle simple et interprétable.
