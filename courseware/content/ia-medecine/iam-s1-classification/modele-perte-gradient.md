# Modèle, fonction de perte et descente de gradient

Tout algorithme d'apprentissage supervisé suit le même schéma, de la
régression linéaire au réseau de neurones profond : choisir une famille de
**modèles**, mesurer leurs erreurs par une **fonction de perte**, et
**optimiser** les paramètres pour la rendre minimale. Cette leçon pose ces
trois briques et l'algorithme d'optimisation le plus utilisé : la
**descente de gradient**.

<!-- notes: 25 minutes. Garder l'exemple fil rouge (température 1980-2020)
du début à la fin : modèle initial, erreurs, gradient, modèle entraîné.
Pas de dérivation au tableau au-delà de la dérivée de θ². -->

---

## Un modèle est une fonction paramétrée

Un **modèle paramétrique** associe aux $p$ variables d'une observation une
prédiction :

$$
f_\theta : \mathbb{R}^p \to \mathbb{R}, \qquad \hat{y} = f_\theta(x)
$$

- $\theta$ regroupe les **paramètres**, les nombres que le modèle doit
  apprendre ;
- le chapeau de $\hat{y}$ signifie « estimé » : le but est que $\hat{y}$
  soit aussi proche que possible de $y$.

L'exemple le plus simple est la **régression linéaire** :

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_p x_p
$$

$\theta_0$ est l'ordonnée à l'origine ; chaque $\theta_j$ mesure l'effet de
la variable $j$ sur la prédiction. Avec $p$ variables, il y a $p + 1$
paramètres.

---

## La fonction de perte

La **perte** mesure l'écart entre les cibles réelles et les prédictions.
Pour une régression, la plus courante est l'**erreur quadratique moyenne**
(MSE, *mean squared error*) :

$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - f_\theta(x_i)\right)^2
$$

Exemple : la température moyenne mesurée tous les 5 ans de 1980 à 2020
($n = 9$), modélisée par une droite $\hat{y} = a \cdot \text{année} + b$.

![Ajuster une droite par descente de gradient](assets/iam-s1-classification/regression-temperature.png)

Le modèle initial ($a = -0{,}01$, $b = 34{,}70$) est loin des points : ses
écarts (traits gris) donnent une perte de 0,20. Le modèle entraîné
($a = 0{,}019$, $b = -23{,}47$) la ramène à 0,006.

---

## Le problème d'apprentissage

Apprendre, c'est chercher les paramètres qui **minimisent** la perte sur les
données d'entraînement :

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
$$

| Étape | Choix |
|---|---|
| 1. Données | Collecter des exemples $(x_i, y_i)$ |
| 2. Modèle | Choisir une famille $f_\theta$ |
| 3. Perte | Définir l'erreur $\mathcal{L}$ |
| 4. Optimisation | Minimiser $\mathcal{L}$ |
| 5. Résultat | Les paramètres appris $\theta^*$ |

Les modèles diffèrent par l'étape 2 et par la méthode de l'étape 4. Pour la
régression linéaire, une formule donne $\theta^*$ directement ; pour la
plupart des modèles, on procède par petits pas.

---

## Le gradient

La **dérivée** $f'(\theta)$ mesure de combien varie $f$ quand $\theta$ varie
un peu. Avec plusieurs paramètres, le **gradient** rassemble les dérivées
partielles :

$$
\nabla \mathcal{L}(\theta) = \left( \frac{\partial \mathcal{L}}{\partial \theta_0}, \frac{\partial \mathcal{L}}{\partial \theta_1}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_p} \right)
$$

Il indique la direction dans laquelle la perte **augmente** le plus vite.
Pour la faire diminuer, on avance donc dans la direction **opposée**.

---

## La descente de gradient

1. **Initialiser** les paramètres $\theta_0$ au hasard.
2. **Mettre à jour** en reculant dans le sens du gradient :

$$
\theta_{t+1} = \theta_t - \eta \, \nabla \mathcal{L}(\theta_t)
$$

3. **Répéter** tant que la perte diminue.

Sur l'exemple de la température, chaque pas déplace la droite vers les
points, et la perte décroît (graphique de droite de la diapositive
précédente) jusqu'à se stabiliser au minimum. La droite apprise prédit
$0{,}019 \times 2030 - 23{,}47 \approx 15{,}0$ °C pour 2030.

<!-- notes: Faire calculer à la main un pas sur L(θ) = θ² : dérivée 2θ ;
depuis θ = 3 avec η = 0,3 : θ ← 3 − 0,3 × 6 = 1,2. C'est le panneau du
milieu de la diapositive suivante. -->

---

## Le taux d'apprentissage

$\eta$ est le **taux d'apprentissage** (*learning rate*) : la taille du pas.
C'est un **hyperparamètre**, un réglage choisi par l'utilisateur et non
appris par le modèle.

![Trois taux d'apprentissage sur L(θ) = θ²](assets/iam-s1-classification/taux-apprentissage.png)

- trop petit : la descente est correcte mais très lente ;
- adapté : quelques pas suffisent pour atteindre le minimum ;
- trop grand : chaque pas saute par-dessus le minimum, la perte augmente,
  l'algorithme diverge.

---

## À retenir

- Modèle $f_\theta$, perte $\mathcal{L}(\theta)$, optimisation
  $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$ : le cadre de tout
  l'apprentissage supervisé.
- La MSE moyenne les carrés des écarts entre prédictions et cibles.
- Descente de gradient : $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$,
  répété jusqu'à stabilisation.
- Le taux d'apprentissage $\eta$ est un hyperparamètre : ni trop petit, ni
  trop grand.

<!-- notes: Quiz oral : comment sont initialisés les paramètres ? Comment
entraîne-t-on un modèle paramétrique ? Combien de paramètres a une régression
linéaire à 5 variables ? (6) -->
