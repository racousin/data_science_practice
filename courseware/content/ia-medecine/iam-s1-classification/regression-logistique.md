# La régression logistique

La régression linéaire prédit un nombre. Pour un diagnostic, on veut une
**classe**, et si possible une **probabilité** : « cette tumeur a 92 % de
chances d'être maligne ». La régression logistique garde la combinaison
linéaire des variables et la fait passer par une fonction qui la ramène
entre 0 et 1. C'est le modèle du TP 1 et du défi 1.

<!-- notes: 25 minutes. Le fil : la droite ne convient pas → la sigmoïde →
le modèle → pourquoi changer de perte → le code → la mise à l'échelle.
Utiliser le rayon moyen des noyaux comme unique variable, les étudiants le
retrouvent dans le TP. -->

---

## De la régression à la classification

On code la cible $y = 1$ pour une tumeur maligne, $y = 0$ pour une tumeur
bénigne, et on essaie de la prédire à partir du rayon moyen des noyaux.

![Régression linéaire et régression logistique](assets/iam-s1-classification/lineaire-vs-logistique.png)

Une droite (à gauche) prédit des valeurs inférieures à 0 et supérieures à 1,
qui ne sont pas des probabilités. Il faut une courbe en S, bornée entre 0
et 1 (à droite).

---

## La sigmoïde

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

![La fonction sigmoïde](assets/iam-s1-classification/sigmoide.png)

- elle envoie tout réel $z$ dans l'intervalle $]0, 1[$ : on peut lire sa
  sortie comme une probabilité ;
- $\sigma(0) = 0{,}5$ ; $z$ très positif donne presque 1, $z$ très négatif
  presque 0 ;
- elle est lisse et dérivable, ce qui permet la descente de gradient.

---

## Le modèle

1. **Calculer** la même combinaison linéaire qu'en régression :
   $z = \theta_0 + \theta_1 x_1 + \dots + \theta_p x_p$.
2. **Appliquer la sigmoïde** pour obtenir une probabilité :

$$
\hat{p} = P(y = 1 \mid x) = \sigma(\theta_0 + \theta_1 x_1 + \dots + \theta_p x_p)
$$

3. **Décider** avec un seuil : prédire 1 si $\hat{p} > 0{,}5$, sinon 0.

Le seuil 0,5 n'est qu'un choix par défaut. En dépistage, on peut l'abaisser
pour manquer moins de malades, au prix de plus de fausses alertes (leçon
suivante).

---

## Une autre fonction de perte : l'entropie croisée

Avec la sigmoïde, la MSE donne une perte **non convexe**, avec des
plateaux où le gradient est presque nul et où la descente peut rester
bloquée. On utilise
l'**entropie croisée** (*log loss*) :

$$
\mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

![Les deux termes de l'entropie croisée](assets/iam-s1-classification/entropie-croisee.png)

Si $y = 1$, seul $-\log(\hat{p})$ compte : il est nul si $\hat{p} = 1$ et
explose quand $\hat{p} \to 0$. Une erreur commise avec assurance coûte
beaucoup plus cher qu'une hésitation. Cette perte est convexe : la descente
de gradient trouve son minimum.

---

## Dans scikit-learn

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)              # minimise l'entropie croisée
```

`fit` apprend les $p + 1$ paramètres. Ensuite, deux sorties :

```python
y_pred = model.predict(X_test)                # classes : 0 ou 1
y_proba = model.predict_proba(X_test)[:, 1]   # probabilités de la classe 1
```

`max_iter` borne le nombre de pas de l'optimisation. Le schéma
`fit` → `predict` est le même pour tous les modèles de scikit-learn.

---

## Pourquoi mettre les variables à l'échelle

Dans le TP, l'aire des noyaux vaut des centaines, leur régularité quelques
centièmes. La surface de perte est alors très étirée : le pas de gradient,
réglé sur la direction la plus raide, avance à peine dans l'autre.

![Descente de gradient avec et sans standardisation](assets/iam-s1-classification/mise-a-l-echelle.png)

La **standardisation** ramène chaque variable à une moyenne 0 et un
écart-type 1 : $x' = (x - \mu) / \sigma$.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # μ et σ appris sur l'entraînement
X_test_s = scaler.transform(X_test)         # appliqués tels quels au test
```

<!-- notes: Insister sur fit_transform / transform : μ et σ viennent du seul
jeu d'entraînement. C'est le premier exemple de fuite de données évitée,
repris dans la leçon suivante. Quiz : P(y=1|x) = 0,73, quelle classe avec un
seuil de 0,5 ? et de 0,8 ? -->
