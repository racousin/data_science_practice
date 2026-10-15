# Hyperparamètres et validation croisée

Une forêt a des réglages : combien d'arbres, quelle profondeur, combien de
patients au minimum par feuille. Ces **hyperparamètres** ne s'apprennent pas
pendant l'entraînement : on les choisit en comparant plusieurs réglages. Pour
que cette comparaison soit honnête, il faut des patients que le modèle n'a
pas vus, et garder le jeu de test pour la toute fin.

<!-- notes: 20 minutes. Le fil : paramètres vs hyperparamètres, pourquoi un
seul découpage ne suffit pas, la validation croisée, la recherche en grille,
et la règle du test utilisé une seule fois. -->

---

## Paramètres et hyperparamètres

| | Qui les fixe ? | Exemples |
|---|---|---|
| **Paramètres** | l'algorithme, pendant `fit` | coefficients de la régression logistique, seuils des nœuds d'un arbre |
| **Hyperparamètres** | nous, avant `fit` | `n_estimators`, `max_depth`, `min_samples_leaf` |

![Trop simple, bien ajusté, trop complexe](assets/iam-s2-preparation-forets/sous-et-sur-apprentissage.png)

*De gauche à droite : sous-apprentissage, bon ajustement, surapprentissage
(R. Cousin, « Course 1 DS », Sorbonne Université).* Les hyperparamètres
règlent la complexité : un arbre de profondeur 1 est trop simple, un arbre
sans limite apprend le bruit.

---

## Apprentissage, validation, test

![Découpage en apprentissage, validation et test](assets/iam-s2-preparation-forets/decoupage-apprentissage-validation-test.png)

Choisir le réglage qui maximise le score **de test** revient à régler le
modèle sur le test : ce score devient optimiste. On sépare donc trois
rôles. L'**apprentissage** fixe les paramètres, la **validation** compare
les réglages, le **test** mesure une seule fois le modèle final.

<!-- notes: 3 minutes. Analogie : réviser sur les annales (apprentissage),
faire un examen blanc pour choisir sa méthode (validation), passer
l'examen (test). Revoir le sujet de l'examen avant de le passer, c'est la
fuite. -->

---

## La validation croisée à k plis

Un seul découpage donne un seul chiffre, qui dépend de la chance : des
patients « faciles » peuvent tomber dans la validation.

![Validation croisée à 5 plis](assets/iam-s2-preparation-forets/validation-croisee-5-plis.png)

*R. Cousin, « Course 1 DS », Sorbonne Université.* On coupe les données
d'apprentissage en $k$ plis ; chaque pli sert une fois de validation pendant
que les $k - 1$ autres servent à entraîner. Le score retenu est la moyenne
des $k$ scores, et leur écart-type mesure la stabilité :

$$
\text{score}_{CV} = \frac{1}{k}\sum_{i=1}^{k} \text{score}_i
$$

---

## En scikit-learn

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(foret, X_app, y_app, cv=5, scoring="accuracy")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

Une ligne fait les 5 découpages, les 5 entraînements et les 5 évaluations.
Si la préparation (imputation, standardisation) s'ajuste sur les données,
on la met dans un `Pipeline` avec le modèle : elle est alors réajustée sur
les 4 plis d'entraînement à chaque tour, sans fuite.

---

## La recherche en grille

`GridSearchCV` essaie toutes les combinaisons d'une grille et garde celle
dont le score moyen en validation croisée est le meilleur.

```python
grille = {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5]}
recherche = GridSearchCV(RandomForestClassifier(random_state=42),
                         grille, cv=5, scoring="accuracy")
recherche.fit(X_app, y_app)
recherche.best_params_, recherche.best_score_
```

4 × 2 = 8 combinaisons × 5 plis = 40 entraînements. Le coût croît vite :
gardez des grilles courtes.

![Exactitude en validation croisée pour une grille d'arbres](assets/iam-s2-preparation-forets/grille-recherche.png)

<!-- notes: 4 minutes. Lire la grille : beaucoup de réglages se valent à
0,005 près, c'est-à-dire moins que l'écart-type entre plis. Ne pas
surinterpréter le meilleur. -->

---

## Le test, une seule fois

`recherche.best_estimator_` est le meilleur modèle, réentraîné sur toutes
les données d'apprentissage. On le mesure **une fois** sur le test :

```python
y_pred = recherche.best_estimator_.predict(X_test)
accuracy_score(y_test, y_pred)
```

Si l'on retouche le modèle après avoir vu ce score, le test devient une
validation de plus, et l'on n'a plus d'estimation honnête. Sur ML-Arena, le
classement final joue ce rôle de test.

---

## À retenir

- Paramètres : appris par `fit`. Hyperparamètres : choisis par nous.
- Apprentissage, validation, test : trois rôles, trois jeux de patients.
- Validation croisée à $k$ plis : moyenne ± écart-type de $k$ scores.
- `GridSearchCV` : toutes les combinaisons d'une petite grille, en
  validation croisée.
- Le test se regarde une seule fois, à la fin.
