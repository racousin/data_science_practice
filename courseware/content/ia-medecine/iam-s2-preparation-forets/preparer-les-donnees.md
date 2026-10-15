# Préparer des données imparfaites

En séance 1, le jeu de données du cancer du sein était propre : aucune
valeur manquante, que des nombres. Les données cliniques réelles ne le sont
jamais. Un dossier est incomplet, une mesure est saisie avec une virgule de
travers, le sexe ou le service sont du texte. La plupart des modèles
refusent ces données telles quelles : il faut les **préparer**, et le faire
sans laisser le jeu de test influencer la préparation.

<!-- notes: 30 minutes. Partir de la question « que fait un modèle d'un NaN,
d'un texte, d'une aire de 7 400 ? » avant de montrer les remèdes. Les deux
messages à garder : la donnée manquante peut être une information (figure
SUPPORT2), et toute statistique de préparation se calcule sur
l'apprentissage seul. Le TP applique les mêmes étapes sur le cancer du sein
bruité, le défi sur SUPPORT2. -->

---

## Les quatre défauts à traiter

| Défaut | Exemple clinique | Remède du jour |
|---|---|---|
| Valeur manquante | albumine non dosée | imputation par la médiane |
| Valeur aberrante | aire de 7 400 pixels² (médiane 560) | écrêtage par l'IQR |
| Variable catégorielle | localisation de la tumeur | encodage one-hot |
| Échelles différentes | âge en années, aire en pixels² | standardisation |

On commence toujours par **regarder** : `df.info()` compte les valeurs
présentes et donne les types, `df.describe()` montre les plages,
`df.isna().sum()` compte les valeurs manquantes par colonne.

<!-- notes: 3 minutes. Sur breast_cancer_noisy.csv : 866 valeurs manquantes,
une aire maximale de 7 400, trois colonnes texte. -->

---

## Valeurs manquantes

Trois stratégies simples :

- **supprimer** les lignes incomplètes : acceptable s'il y en a très peu ;
- **imputer** par la **médiane** de la colonne (variable numérique) : la
  médiane, contrairement à la moyenne, n'est pas tirée par les valeurs
  aberrantes ;
- imputer par la **modalité la plus fréquente** (variable catégorielle).

```python
from sklearn.impute import SimpleImputer
imputeur = SimpleImputer(strategy="median")
X_app_num = imputeur.fit_transform(X_app[colonnes_num])
```

`fit` calcule la médiane de chaque colonne ; `transform` remplace les `NaN`
par ces médianes.

---

## L'absence est une information

![Taux de décès selon que la valeur est manquante ou renseignée, SUPPORT2](assets/iam-s2-preparation-forets/manquant-informatif.png)

Dans SUPPORT2, le questionnaire d'autonomie manque surtout chez les patients
trop graves pour y répondre : 55 % de décès quand il manque, 18 % quand il est
renseigné. Une valeur manque rarement au hasard. L'option
`add_indicator=True` ajoute une colonne 0/1 « valeur manquante », que le
modèle peut utiliser.

<!-- notes: 4 minutes. Faire deviner pourquoi l'autonomie manque chez les
plus graves. En pratique clinique : un examen non prescrit, un patient non
interrogeable, un décès précoce. Imputer sans indicateur efface ce signal. -->

---

## Valeurs aberrantes : l'écrêtage par l'IQR

L'**écart interquartile** est $\mathrm{IQR} = Q_3 - Q_1$. Les bornes de la
boîte à moustaches sont :

$$
[\,Q_1 - 1.5\,\mathrm{IQR}\;;\;Q_3 + 1.5\,\mathrm{IQR}\,]
$$

![Aire moyenne avant et après écrêtage](assets/iam-s2-preparation-forets/ecretage-iqr.png)

**Écrêter** remplace une valeur hors bornes par la borne la plus proche
(`serie.clip(bas, haut)`) : on garde le patient, on limite l'influence de la
valeur. Une valeur extrême peut aussi être vraie : en clinique, on vérifie la
source avant de corriger.

---

## Variables catégorielles : l'encodage one-hot

Un modèle calcule avec des nombres. Coder `central = 0`, `upper_inner = 1`,
`lower_inner = 2` inventerait un ordre qui n'existe pas. L'encodage
**one-hot** crée une colonne 0/1 par modalité :

| tumor_location | central | lower_inner | upper_inner |
|---|---|---|---|
| upper_inner | 0 | 0 | 1 |
| central | 1 | 0 | 0 |
| lower_inner | 0 | 1 | 0 |

```python
pd.get_dummies(df, columns=["tumor_location"], drop_first=True)
```

`drop_first=True` retire une colonne redondante : si toutes les autres
valent 0, la modalité retirée est connue.

---

## Mise à l'échelle, et la règle de la fuite

La **standardisation** ramène chaque variable à une moyenne nulle et un
écart-type de 1 :

$$
z = \frac{x - \mu}{\sigma}
$$

Médianes, bornes IQR, $\mu$ et $\sigma$ sont des **statistiques apprises**.
Les calculer sur toutes les données laisse le test influencer le modèle :
c'est une **fuite d'information**, et le score de test devient optimiste.

![Ajuster sur l'apprentissage seul](assets/iam-s2-preparation-forets/fuite-ajustement.png)

```python
scaler = StandardScaler().fit(X_app)       # statistiques de l'apprentissage
X_app_s = scaler.transform(X_app)
X_test_s = scaler.transform(X_test)        # mêmes μ et σ pour le test
```

<!-- notes: 5 minutes. Image mentale : le test, ce sont les patients de
demain ; on ne connaît pas leur médiane au moment d'entraîner. Le TP prépare
encore tout le tableau avant la séparation, pour rester simple ; le défi
fait les choses dans l'ordre. -->

---

## À retenir

- Regarder d'abord : `info`, `describe`, `isna().sum()`.
- Manquant : médiane (numérique), modalité la plus fréquente (catégorielle),
  et un indicateur de valeur manquante.
- Aberrant : écrêter aux bornes $Q_1 - 1.5\,\mathrm{IQR}$ et
  $Q_3 + 1.5\,\mathrm{IQR}$.
- Catégoriel : one-hot. Échelles : standardisation.
- Toute statistique de préparation s'ajuste (`fit`) sur l'apprentissage seul.
