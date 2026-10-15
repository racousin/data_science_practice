# seaborn : graphiques statistiques

seaborn est une interface de **haut niveau** au-dessus de Matplotlib. Elle lit
directement un DataFrame pandas, désigne les variables par leur nom de colonne,
et calcule elle-même les statistiques usuelles (moyennes, intervalles de
confiance, densités, droites de régression).

```python
import seaborn as sns
sns.set_theme()                      # active le thème par défaut de seaborn
```

<!-- notes: 30 minutes. Tous les exemples utilisent ic = pd.read_csv(".../insuffisance_cardiaque.csv"), 299 patients (Chicco & Jurman 2020). Ressources : seaborn.pydata.org/examples (galerie) et seaborn.pydata.org/tutorial.html. -->

---

## Une même signature pour toutes les fonctions

Presque toutes les fonctions acceptent les mêmes arguments :

- `data` : le DataFrame ;
- `x`, `y` : des **noms de colonnes** ;
- `hue` : une colonne qui colore les marques par groupe (aussi `size`, `style`) ;
- `ax` : l'Axes Matplotlib où dessiner.

```python
sns.scatterplot(data=ic, x="age", y="creatinine", hue="deces")
sns.histplot(data=ic, x="fraction_ejection", hue="deces")
sns.boxplot(data=ic, x="deces", y="fraction_ejection")
```

Changer de graphique, c'est changer le nom de la fonction, pas les arguments.

<!-- notes: Comparer avec Matplotlib, où il faut filtrer soi-même chaque groupe et appeler scatter deux fois. -->

---

## Trois familles, deux niveaux

![Les fonctions figure-level relplot, displot, catplot et leurs fonctions axes-level](assets/pym-s6-visualisation-analyse/seaborn-overview.png)

- **Relationnelles** : relation entre deux variables numériques (`scatterplot`, `lineplot`).
- **Distributions** : forme d'une variable (`histplot`, `kdeplot`, `ecdfplot`).
- **Catégorielles** : comparer des groupes (`boxplot`, `violinplot`, `barplot`…).

Les fonctions du bas sont **axes-level** : elles dessinent dans un Axes
(`ax=`). Les trois du haut sont **figure-level** : elles créent leur propre
figure et choisissent le tracé avec `kind=`.

<!-- notes: Le point délicat : une fonction figure-level n'accepte pas ax=. Elle renvoie un objet FacetGrid (g.fig, g.axes). -->

---

## Axes-level ou figure-level

```python
# axes-level : on lui fournit l'Axes
fig, ax = plt.subplots()
sns.histplot(data=ic, x="fraction_ejection", hue="deces", ax=ax)
# figure-level : la fonction crée sa propre figure
sns.displot(data=ic, x="fraction_ejection", hue="deces", kind="hist")
```

![histplot (axes-level)](assets/pym-s6-visualisation-analyse/hist_axe.png)

<!-- notes: Images de démonstration sur deux variables simulées (data_1, data_2) : le rendu est le même avec les données cliniques. -->

---

## Figure-level : légende et facettes

![displot (figure-level)](assets/pym-s6-visualisation-analyse/hist_figure.png)

Le figure-level place la légende hors du graphique et sait découper en
**facettes** : `col="sexe_homme"` trace un panneau par sexe. `kind="kde"`
remplace l'histogramme par une densité lissée (`kdeplot`), et
`multiple="stack"` empile les groupes au lieu de les superposer.

---

## Densité lissée : kdeplot ou displot(kind="kde")

![kdeplot (axes-level)](assets/pym-s6-visualisation-analyse/kde_axe.png)
![displot kind="kde" (figure-level)](assets/pym-s6-visualisation-analyse/kde_figure.png)

---

## Des statistiques dans le graphique

`barplot` affiche la **moyenne** de chaque groupe et une barre d'erreur
(par défaut l'intervalle de confiance à 95 %, estimé par bootstrap).

```python
sns.barplot(data=ic, x="anemie", y="creatinine", errorbar=("ci", 95))
```

![Barres et intervalles de confiance à 95 % (données d'exemple de seaborn)](assets/pym-s6-visualisation-analyse/cat-error-bar.png)

Une barre sans barre d'erreur cache la dispersion : préférer un `boxplot` ou
ajouter les points (`stripplot`) dès que les effectifs sont petits.

<!-- notes: errorbar="sd" affiche l'écart-type à la place de l'IC. Rappeler que l'IC porte sur la moyenne, pas sur les individus. -->

---

## Une droite de régression : regplot

`regplot` ajoute une droite de régression linéaire et sa bande de confiance.

```python
sns.regplot(data=ic, x="age", y="fraction_ejection")
```

![Nuage de points et droite de régression (données d'exemple de seaborn)](assets/pym-s6-visualisation-analyse/reg-fit.png)

---

## Format long : comparer des colonnes

seaborn attend un tableau **long** : une ligne par mesure, une colonne qui dit
quelle mesure. `pd.melt` passe du format large au format long.

```python
long = ic.melt(id_vars="deces", value_vars=["age", "fraction_ejection"],
               var_name="variable", value_name="valeur")
sns.boxplot(data=long, x="variable", y="valeur", hue="deces")
```

C'est le même format pour une série temporelle de plusieurs patients :
`sns.lineplot(data=vitaux, x="heure", y="spo2", hue="id_patient")`.

<!-- notes: constantes_vitales.csv est déjà au format long (une ligne par patient et par heure). -->

---

## Vue d'ensemble : pairplot, jointplot, heatmap

```python
sns.pairplot(ic[["age", "fraction_ejection", "creatinine", "deces"]],
             hue="deces")
```

![Matrice de nuages de points, coloriée par groupe (données d'exemple de seaborn)](assets/pym-s6-visualisation-analyse/scatterplot_matrix.png)

Le **pairplot** croise toutes les paires de variables (distributions sur la
diagonale).

<!-- notes: Les images de cette leçon viennent du cours d'origine (jeux d'exemple de seaborn : titanic, penguins) ; les étudiants refont les mêmes graphiques sur insuffisance_cardiaque.csv dans le notebook. Un pairplot sur 13 colonnes est illisible : choisir 3 à 5 variables. -->

---

## Jointplot et heatmap

```python
sns.jointplot(data=ic, x="age", y="fraction_ejection")
corr = ic.drop(columns="id_patient").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
```

![Jointplot : nuage et distributions marginales (données d'exemple)](assets/pym-s6-visualisation-analyse/joint.png)

Le **jointplot** montre un nuage et les deux distributions marginales. La
**heatmap** colore une matrice, ici les corrélations, avec une palette
divergente centrée sur 0.

---

## Esthétique : set_theme

```python
sns.set_theme()                                  # thème par défaut
sns.set_theme(style="darkgrid")                  # fond gris, grille
sns.set_theme(style="whitegrid", palette="pastel")
```

![style par défaut](assets/pym-s6-visualisation-analyse/set-theme-1.png)

`set_theme` règle le fond, la grille, la police et la palette de toutes les
figures suivantes, Matplotlib compris. `sns.set_theme(context="talk")` grossit
les textes pour une présentation.

---

## set_theme : darkgrid, whitegrid

![darkgrid](assets/pym-s6-visualisation-analyse/set-theme-2.png)
![whitegrid, palette pastel](assets/pym-s6-visualisation-analyse/set-theme-3.png)

---

## À retenir

- seaborn lit un DataFrame : `data=`, puis `x`, `y`, `hue` par nom de colonne.
- Trois familles (relation, distribution, catégories) ; axes-level (`ax=`) ou figure-level (`kind=`, facettes `col=`).
- Les barres d'erreur et droites de régression sont calculées pour vous : savoir ce qu'elles représentent.
- Matplotlib reste dessous : `ax.set_title`, `fig.savefig` s'appliquent aux graphiques seaborn.
