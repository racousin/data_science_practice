# Évaluer un classifieur

Un modèle sert à prédire des patients qu'il n'a **jamais vus**. L'évaluer
sur ses propres données d'entraînement ne dit rien de cette capacité : un
modèle assez souple peut les apprendre par cœur. Cette leçon présente
comment mesurer honnêtement la performance (découpage, validation croisée,
absence de fuite) et avec quelles métriques (matrice de confusion,
exactitude, précision, rappel, F1, courbe ROC).

<!-- notes: 25 minutes. Deux moitiés : « sur quelles données évaluer »
(10 min) puis « avec quelle métrique » (15 min). Les chiffres de la matrice
de confusion sont ceux que les étudiants obtiendront dans le TP 1
(exercice 4.2). -->

---

## Entraînement et test

On sépare les données **avant** tout apprentissage : le modèle apprend sur
le **jeu d'entraînement** (80 %), et on le mesure sur le **jeu de test**
(20 %), qui simule des patients nouveaux.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

`random_state` fixe le tirage pour que le résultat soit reproductible. Un
modèle bon sur l'entraînement mais mauvais sur le test a appris par cœur au
lieu de généraliser : c'est le **surapprentissage**.

---

## La validation croisée

Un seul découpage donne une seule estimation : un tirage chanceux (des cas
faciles en test) ou malchanceux la fausse. La **validation croisée à
$k$ blocs** répète l'évaluation $k$ fois.

![Validation croisée à 5 blocs](assets/iam-s1-classification/validation-croisee.png)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

La moyenne est une estimation plus fiable ; l'écart-type dit si le modèle
est stable d'un découpage à l'autre.

---

## La fuite de données

Il y a **fuite** (*data leakage*) quand une information sur le jeu de test
atteint l'entraînement. Le score mesuré est alors trop optimiste, et
s'effondre sur de vrais nouveaux patients.

- standardiser avec la moyenne et l'écart-type de **toutes** les données,
  test compris ;
- garder une variable connue seulement après l'issue (un traitement
  décidé après le diagnostic) ;
- avoir le même patient en entraînement et en test.

Le réflexe : tout ce qui s'**apprend** (moyennes, médianes, paramètres) se
calcule sur l'entraînement seul. `make_pipeline` enchaîne la préparation et
le modèle pour que la validation croisée le respecte dans chaque bloc :

```python
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
```

---

## La matrice de confusion

On choisit la classe **positive** : celle qu'on cherche à détecter, ici la
tumeur maligne. Chaque prédiction tombe dans l'une des quatre cases :

| | Prédit maligne | Prédit bénigne |
|---|---|---|
| **Maligne** | VP = 39 (vrai positif) | FN = 4 (faux négatif) |
| **Bénigne** | FP = 1 (faux positif) | VN = 70 (vrai négatif) |

(la régression logistique du TP 1 sur les 114 tumeurs de test)

L'**exactitude** (*accuracy*) est la part de prédictions correctes :

$$
\text{Exactitude} = \frac{VP + VN}{VP + VN + FP + FN} = \frac{109}{114} = 0{,}956
$$

Elle est **trompeuse si les classes sont déséquilibrées** : si 99 % des
patients sont sains, répondre toujours « sain » donne 99 % d'exactitude et
ne détecte aucun malade.

---

## Précision, rappel et F1

$$
\text{Précision} = \frac{VP}{VP + FP} = \frac{39}{40} = 0{,}975 \qquad \text{Rappel} = \frac{VP}{VP + FN} = \frac{39}{43} = 0{,}907
$$

- **précision** : parmi les tumeurs annoncées malignes, combien le sont ?
- **rappel** (ou **sensibilité**) : parmi les tumeurs malignes, combien
  a-t-on trouvées ?

En **dépistage**, le rappel prime : un cancer manqué (faux négatif) peut
être fatal, une fausse alerte mène à un examen complémentaire. Ici,
4 cancers sur 43 sont manqués.

Le **F1** résume les deux par leur moyenne harmonique, qui pénalise un
déséquilibre entre eux :

$$
F_1 = 2 \cdot \frac{\text{Précision} \cdot \text{Rappel}}{\text{Précision} + \text{Rappel}} = 0{,}940
$$

<!-- notes: Faire trancher la salle : dépistage du cancer, précision ou
rappel ? Puis un contre-exemple où la précision compte : annoncer à tort un
diagnostic grave déclenche un traitement lourd. -->

---

## La courbe ROC et l'AUC

Précision et rappel dépendent du **seuil** de décision (0,5 par défaut).
Abaisser le seuil trouve plus de cancers (rappel ↑) mais crée plus de
fausses alertes. La **courbe ROC** trace le rappel en fonction du taux de
faux positifs $FP / (FP + VN)$, pour tous les seuils.

![Courbe ROC de la régression logistique](assets/iam-s1-classification/courbe-roc.png)

L'**aire sous la courbe** (AUC) résume le modèle sans choisir de seuil :
1 pour un classement parfait, 0,5 pour le hasard.

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

---

## À retenir

- Évaluer sur des données que le modèle n'a pas vues : découpage
  entraînement / test, ou validation croisée.
- Pas de fuite : tout ce qui s'apprend se calcule sur l'entraînement seul.
- La matrice de confusion (VP, FN, FP, VN) fonde toutes les métriques.
- L'exactitude trompe sur des classes déséquilibrées ; en dépistage, on
  surveille le **rappel**.
- La courbe ROC et l'AUC jugent le modèle sur tous les seuils.

<!-- notes: Quiz oral : un modèle à 99 % d'exactitude sur des données à 99 %
de classe 0 est-il bon ? Quelle différence entre fonction de perte (guide
l'entraînement) et métrique (juge le résultat) ? Pourquoi la validation
croisée plutôt qu'un seul découpage ? -->
