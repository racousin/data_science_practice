# Analyse exploratoire

L'analyse exploratoire (*EDA*, *exploratory data analysis*) consiste à regarder
les données avant de les modéliser : leur forme, leurs anomalies, les liens
entre variables. On la mène sur un jeu réel : 299 patients en insuffisance
cardiaque suivis quelques mois (Chicco & Jurman, 2020, UCI, CC BY 4.0).

```python
ic = pd.read_csv(URL + "insuffisance_cardiaque.csv")
ic.shape, ic["deces"].mean()   # (299, 14), 32 % de décès pendant le suivi
```

<!-- notes: 35 minutes. Exemples pédagogiques : aucun résultat de cette leçon n'est un outil de décision clinique. Les chiffres cités ont été calculés sur le fichier du cours. -->

---

## D'abord une question

Une exploration sans question produit cinquante graphiques et aucune
conclusion. On fixe la question, puis on choisit les graphiques qui y répondent.

> Quelles mesures recueillies à l'inclusion sont associées au décès pendant le suivi ?

La démarche suit toujours le même ordre :

1. **une variable** à la fois : distribution, valeurs aberrantes ;
2. **deux variables** : chaque mesure selon le groupe (décès ou non) ;
3. **toutes ensemble** : matrice de corrélation ;
4. **tester** les différences qui semblent importantes, puis les interpréter.

---

## Une variable : distribution et valeurs extrêmes

```python
ic.describe().T      # moyenne, écart-type, quartiles, min, max
ic["creatinine"].hist(bins=30)
```

La créatinine a une médiane de 1,1 mg/dL mais un maximum de 9,4 : la
distribution est **asymétrique**, la moyenne (1,39) est tirée par quelques
patients. La CPK va de 23 à 7 861 UI/L. Une valeur extrême n'est pas une
erreur : une créatinine à 9,4 mg/dL est plausible chez un insuffisant rénal
sévère. On la vérifie, on la garde si elle est plausible, et on préfère alors la
médiane et les méthodes fondées sur les rangs.

<!-- notes: Les étudiants veulent supprimer les "outliers" automatiquement (règle 1,5 IQR). Insister : on supprime une erreur de saisie (âge 999), pas un patient grave. -->

---

## Deux variables : comparer les groupes

![Fraction d'éjection selon l'issue du suivi](assets/pym-s6-visualisation-analyse/pym_s6_fe_par_deces.png)

```python
ic.groupby("deces")["fraction_ejection"].median()     # 38 % contre 30 %
pd.crosstab(ic["anemie"], ic["deces"], normalize="index")
```

La fraction d'éjection ventriculaire gauche (FEVG) est plus basse chez les
patients décédés ; les deux distributions se recouvrent pourtant largement.
Pour une variable binaire comme l'anémie, on compare des **proportions** :
36 % de décès chez les anémiques contre 29 % chez les autres.

---

## Toutes les variables : corrélations

$$
r = \frac{\mathrm{cov}(X, Y)}{\sigma_X \, \sigma_Y} \in [-1, 1]
$$

Le coefficient de **Pearson** mesure une relation **linéaire** et est sensible
aux valeurs extrêmes. Celui de **Spearman** est le Pearson calculé sur les
**rangs** : il mesure une relation **monotone** et résiste aux extrêmes.

```python
corr = ic.drop(columns="id_patient").corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
```

![Matrices de corrélation de Pearson et de Spearman](assets/pym-s6-visualisation-analyse/pym_s6_correlations.png)

Pour la créatinine, très asymétrique, la corrélation avec le décès passe de
0,29 (Pearson) à 0,37 (Spearman). **Corrélation n'est pas causalité** : une
association peut venir d'un facteur commun (l'âge, la sévérité de la maladie).

<!-- notes: Corréler une variable binaire (deces) revient à comparer des moyennes de groupes ; c'est acceptable pour explorer. -->

---

## Tester une différence

Un test part d'une hypothèse nulle $H_0$ : « pas de différence entre les
groupes ». La **p-valeur** est la probabilité d'observer un écart au moins aussi
grand si $H_0$ était vraie.

```python
from scipy import stats
fe = ic.groupby("deces")["fraction_ejection"]
stats.mannwhitneyu(fe.get_group(0), fe.get_group(1))   # p ≈ 7e-07
stats.chi2_contingency(pd.crosstab(ic["anemie"], ic["deces"]))  # p ≈ 0,31
```

- **Mann-Whitney** compare deux distributions sans les supposer normales ; le
  **t de Student** (`stats.ttest_ind`, `equal_var=False` pour Welch) compare
  deux moyennes et suppose des données à peu près normales.
- Le **χ²** teste l'indépendance de deux variables catégorielles à partir d'un
  tableau de contingence.

---

## Lire une p-valeur

- $p \approx 7 \times 10^{-7}$ pour la FEVG : un tel écart serait très
  improbable sans différence réelle entre les groupes. Cela ne dit **rien** de
  la taille de l'effet : on rapporte aussi les médianes (38 % contre 30 %).
- $p \approx 0{,}31$ pour l'anémie : les données ne permettent pas de conclure.
  Ce n'est **pas** la preuve que l'anémie est sans effet : avec 299 patients,
  une différence de 6 points peut passer inaperçue.
- Tester vingt variables au seuil 0,05, c'est s'attendre à environ un « résultat
  significatif » dû au hasard.

<!-- notes: Le seuil 0,05 est une convention, pas une frontière entre vrai et faux. Faire formuler l'interprétation à l'oral avant de montrer cette slide. -->

---

## Une mise en garde : le biais

![Durée de suivi selon l'issue](assets/pym-s6-visualisation-analyse/pym_s6_suivi_deces.png)

`suivi_jours` est la variable la plus corrélée au décès (Spearman −0,54). Ce
n'est pas un facteur de risque : un patient décédé au 10ᵉ jour a, par
construction, un suivi de 10 jours. La variable est une **conséquence** de
l'issue. De même, une association entre deux mesures peut s'expliquer par un
**facteur de confusion** (l'âge élève la créatinine et le risque de décès).
Avant de conclure, se demander comment les données ont été recueillies.

---

## À retenir

- Une question d'abord ; puis une variable, deux variables, toutes ensemble.
- Médiane, rangs et Spearman quand la distribution est asymétrique.
- Une p-valeur faible signale une différence peu compatible avec le hasard, pas sa taille ni sa cause ; une p-valeur élevée ne prouve pas l'absence d'effet.
- Corrélation n'est pas causalité : chercher les biais et les facteurs de confusion.
