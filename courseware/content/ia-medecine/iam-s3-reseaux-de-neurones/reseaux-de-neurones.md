# Les réseaux de neurones

La régression logistique trace une frontière **linéaire** entre deux
classes. Un réseau de neurones assemble beaucoup de petites unités de
calcul, les neurones artificiels, en couches successives : le modèle
obtenu peut tracer des frontières courbes. Cette leçon décrit le neurone
artificiel, les fonctions d'activation, le perceptron multicouche et son
utilisation dans scikit-learn.

<!-- notes: 30 minutes. Partir de la régression logistique de la séance 1 :
c'est déjà un neurone. Le message à faire passer : un réseau = des
régressions logistiques empilées, séparées par des non-linéarités. Garder
la rétropropagation pour la leçon suivante. -->

---

## Du neurone biologique au neurone artificiel

Un neurone biologique reçoit des signaux par ses dendrites, les intègre
dans le corps cellulaire et, au-delà d'un seuil, émet un potentiel d'action
le long de l'axone. Le neurone artificiel en est une caricature
mathématique.

![Neurone biologique et neurone artificiel](assets/iam-s3-reseaux-de-neurones/neurone-biologique-artificiel.png)

Il calcule une **somme pondérée** de ses entrées $x_1, \dots, x_n$, ajoute
un **biais** $b$, puis applique une **fonction d'activation** $f$ :

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b, \qquad a = f(z)
$$

Les poids $w_j$ et le biais $b$ sont les **paramètres** appris pendant
l'entraînement, comme les coefficients $\theta$ de la séance 1.

<!-- notes: 4 minutes. Faire le lien avec la régression logistique : avec
f = sigmoïde, ce neurone EST une régression logistique. L'analogie
biologique s'arrête là : rien d'autre dans le calcul n'imite le cerveau. -->

---

## Les fonctions d'activation

La fonction d'activation introduit la **non-linéarité**. Sans elle,
empiler des couches ne servirait à rien : une composition de fonctions
linéaires reste linéaire.

![Fonctions d'activation](assets/iam-s3-reseaux-de-neurones/fonctions-activation.png)

$$
\sigma(z) = \frac{1}{1 + e^{-z}} \qquad\qquad \mathrm{ReLU}(z) = \max(0, z)
$$

- La **sigmoïde** $\sigma$ ramène $z$ entre 0 et 1 : on la garde pour la
  **sortie**, qui doit être une probabilité (maladie présente ou non).
- **ReLU** (*rectified linear unit*) laisse passer les valeurs positives et
  annule les négatives. Simple et rapide à entraîner, c'est le choix par
  défaut des **couches cachées** (`activation="relu"` dans scikit-learn).
- `tanh` ressemble à la sigmoïde mais va de −1 à 1.

<!-- notes: 4 minutes. Insister sur « sans activation, tout le réseau
s'écrase en une seule régression linéaire ». -->

---

## Le perceptron multicouche

Un **perceptron multicouche** (*multi-layer perceptron*, MLP) range les
neurones en couches : chaque neurone d'une couche reçoit toutes les
sorties de la couche précédente.

![Perceptron multicouche](assets/iam-s3-reseaux-de-neurones/perceptron-multicouche.jpeg)

- la **couche d'entrée** : les variables du patient (13 pour la maladie
  cardiaque) ;
- les **couches cachées** : des représentations intermédiaires, calculées
  par le réseau ;
- la **couche de sortie** : un neurone sigmoïde qui donne la probabilité
  de la classe positive.

Avec une couche cachée, le réseau est une **composition** de deux étapes :

$$
h = \mathrm{ReLU}(W_1 x + b_1), \qquad \hat{y} = \sigma(W_2 h + b_2)
$$

<!-- notes: 5 minutes. W1 est un tableau de poids : une ligne par neurone
caché. Compter les paramètres avec la salle (diapositive suivante). -->

---

## Ce que le réseau ajoute

La régression logistique, c'est le seul neurone de sortie appliqué
directement aux variables. Les couches cachées construisent d'abord de
nouvelles variables, combinaisons non linéaires des variables d'origine,
d'où une frontière de décision qui peut se courber.

![Frontière de la régression logistique et d'un réseau](assets/iam-s3-reseaux-de-neurones/frontieres-lr-mlp.png)

Le prix à payer est le **nombre de paramètres**. Pour 13 variables :

- régression logistique : $13 + 1 = 14$ paramètres ;
- une couche cachée de 100 neurones : $13 \times 100 + 100$ pour la couche
  cachée, $100 + 1$ pour la sortie, soit **1 501** paramètres.

Avec 270 patients, 1 501 paramètres suffisent pour apprendre les données
par cœur : c'est le risque de **sur-apprentissage**, sujet de la leçon
suivante.

<!-- notes: 5 minutes. La figure est un jeu jouet en deux dimensions (deux
demi-lunes) : un vrai problème médical a rarement une frontière aussi
tordue. Annoncer que, sur la maladie cardiaque, le réseau ne battra pas la
régression logistique. -->

---

## Dans scikit-learn

`MLPClassifier` s'utilise comme les autres modèles : `fit`, `predict`,
`predict_proba`.

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp.predict(X_test_scaled)
```

`hidden_layer_sizes=(100,)` : une couche cachée de 100 neurones ;
`(64, 32)` : deux couches. `random_state` fixe les poids initiaux, tirés
au hasard.

Les variables doivent être **standardisées** (moyenne 0, écart type 1) :
un cholestérol en centaines de mg/dL à côté d'un sexe codé 0/1 déséquilibre
les sommes pondérées et ralentit l'entraînement.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ajusté sur l'entraînement
X_test_scaled = scaler.transform(X_test)        # appliqué au test
```

<!-- notes: 4 minutes. Rappeler la règle de la séance 2 : le scaler
s'ajuste sur l'entraînement seulement. -->

---

## À retenir

- Un neurone : somme pondérée + biais, puis une fonction d'activation.
- ReLU dans les couches cachées, sigmoïde en sortie pour une probabilité.
- Un perceptron multicouche compose des couches ; la non-linéarité lui
  permet des frontières courbes.
- Beaucoup plus de paramètres qu'une régression logistique : plus souple,
  mais plus de risque de sur-apprentissage.
- Toujours standardiser les variables avant un réseau.
