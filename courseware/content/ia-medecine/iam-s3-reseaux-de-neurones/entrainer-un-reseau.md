# Entraîner un réseau : courbe de perte et arrêt précoce

Un réseau de neurones s'entraîne comme la régression logistique de la
séance 1 : on choisit une fonction de perte et on la fait baisser par
descente de gradient. Ce qui change, c'est le nombre de paramètres, et donc
le risque d'apprendre les patients d'entraînement par cœur. Cette leçon
présente la boucle d'entraînement, la rétropropagation, la courbe de perte
et l'arrêt précoce.

<!-- notes: 25 minutes. Pas de calcul de dérivées au tableau : l'intuition
suffit. Le cœur de la leçon est la figure de la courbe de perte. -->

---

## La boucle d'entraînement

![La boucle d'entraînement d'un modèle](assets/iam-s3-reseaux-de-neurones/boucle-entrainement.png)

1. **Initialiser** les poids au hasard.
2. **Prédire** pour les patients d'entraînement et calculer la **perte**.
3. **Mettre à jour** chaque poids dans le sens qui fait baisser la perte.
4. Recommencer. Un passage complet sur les données s'appelle une **époque**.

Pour une classification, la perte est l'**entropie croisée** de la
séance 1, moyennée sur les $n$ patients :

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

<!-- notes: 4 minutes. La figure vient du cours 1 (régression d'une
température) : la même boucle s'applique à un réseau, seul le modèle
change. -->

---

## Descente de gradient et rétropropagation

Chaque poids $w$ est mis à jour avec le taux d'apprentissage $\eta$ :

$$
w \leftarrow w - \eta \, \frac{\partial L}{\partial w}
$$

Il faut donc la dérivée de la perte par rapport à **chaque** poids, y
compris ceux des premières couches, loin de la sortie. La
**rétropropagation** (*backpropagation*) les calcule en remontant le
réseau de la sortie vers l'entrée, grâce à la règle de dérivation des
fonctions composées :

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial w_1}
$$

Intuition : l'erreur commise en sortie est **répartie** entre les
neurones qui y ont contribué, couche après couche. scikit-learn fait ce
calcul pour vous ; le solveur par défaut, `adam`, est une variante de la
descente de gradient qui adapte le pas de chaque poids.

<!-- notes: 5 minutes. Ne pas dériver au tableau. Retenir : un passage
avant (prédiction) puis un passage arrière (gradients), à chaque pas. -->

---

## La courbe de perte

`mlp.loss_curve_` donne la perte d'entraînement à chaque époque. Elle
baisse puis se stabilise : le réseau a **convergé**. Mais une perte
d'entraînement basse ne dit rien des nouveaux patients. Pour le savoir, on
met de côté des patients de **validation** et on suit aussi leur perte.

![Perte d'entraînement et de validation, arrêt précoce](assets/iam-s3-reseaux-de-neurones/courbe-perte-arret-precoce.png)

Sur la maladie cardiaque (100 neurones cachés, 162 patients
d'ajustement) : la perte de validation est minimale vers l'époque 16, puis
**remonte** alors que celle d'entraînement continue de baisser. Le réseau
apprend les particularités des patients d'entraînement, pas la maladie :
c'est le **sur-apprentissage**.

<!-- notes: 6 minutes. Faire lire la figure : deux courbes qui se
séparent. Lien avec la forêt de la séance 2 qui faisait 100 % sur
l'entraînement. Figure produite par courseware/tools/figures/iam_s3.py. -->

---

## L'arrêt précoce

L'**arrêt précoce** (*early stopping*) arrête l'entraînement quand le
score de validation ne s'améliore plus depuis un certain nombre
d'époques (la **patience**).

```python
mlp = MLPClassifier(max_iter=1000, random_state=42,
                    early_stopping=True,       # active l'arrêt précoce
                    validation_fraction=0.1,   # 10 % de l'entraînement en validation
                    n_iter_no_change=10)       # patience : 10 époques
```

Sur la maladie cardiaque, le réseau s'arrête après 27 époques au lieu de
839. Le score de validation est dans `mlp.validation_scores_`.

La validation est prise **dans l'entraînement** : le jeu de test reste
intact pour l'évaluation finale. Avec peu de patients, la validation est
petite (ici 22 patients) et le point d'arrêt est lui-même bruité.

<!-- notes: 4 minutes. Le réglage est automatique : c'est la forme la plus
simple de régularisation d'un réseau. -->

---

## Choisir une architecture

| `hidden_layer_sizes` | Paramètres (13 variables) | Usage |
|---|---|---|
| `(50,)` | 751 | problème simple, peu de données |
| `(100,)` | 1 501 | valeur par défaut |
| `(64, 32)` | 3 009 | motifs plus complexes |
| `(128, 64, 32)` | 12 161 | beaucoup de données nécessaires |

Commencer petit, n'ajouter des couches que si la **validation**
s'améliore. Les autres réglages (`activation`, `learning_rate_init`) se
choisissent par recherche sur grille et validation croisée, comme les
hyperparamètres de la forêt en séance 2.

<!-- notes: 3 minutes. Dans le TP, les quatre architectures donnent entre
0,83 et 0,85 sur 54 patients de test : des écarts de un ou deux patients,
pas significatifs. -->

---

## À retenir

- Entraîner = répéter prédiction, perte, mise à jour des poids, époque
  après époque.
- La rétropropagation calcule le gradient de chaque poids en remontant de
  la sortie vers l'entrée.
- La perte d'entraînement qui baisse ne suffit pas : suivre la validation.
- Arrêt précoce : `early_stopping=True` arrête quand la validation ne
  s'améliore plus.
- L'architecture la plus simple qui suffit est la bonne.
