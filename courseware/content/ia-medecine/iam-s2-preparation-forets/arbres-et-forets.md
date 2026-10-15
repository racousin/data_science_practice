# Arbres de décision et forêts aléatoires

La régression logistique trace une frontière **linéaire** entre les classes.
Un **arbre de décision** procède autrement : il pose une suite de questions
simples sur les variables, comme un arbre décisionnel de recommandation
clinique. Un arbre seul apprend trop bien ses données ; une **forêt** de
centaines d'arbres légèrement différents, qui votent, corrige ce défaut.

<!-- notes: 30 minutes. Lire l'arbre de profondeur 2 à voix haute comme un
algorithme clinique avant toute formule. Puis la courbe de surapprentissage,
puis la forêt. Les importances font le lien avec le TP (exercice 8.3). -->

---

## Un arbre : une suite de questions

![Arbre de décision de profondeur 2 sur le cancer du sein](assets/iam-s2-preparation-forets/arbre-profondeur-2.png)

Chaque **nœud** teste une variable contre un seuil (`worst radius <= 16.795`) ;
on descend à gauche si c'est vrai, à droite sinon. Chaque **feuille** prédit
la classe majoritaire de ses patients d'apprentissage : ici, 328 tumeurs
bénignes sur 333 dans la feuille de gauche.

<!-- notes: 4 minutes. Lire : rayon < 16,8 et peu de points concaves → bénigne
(328 sur 333). Les couleurs : orange = maligne, bleu = bénigne, l'intensité
= la pureté. -->

---

## Choisir la question : l'impureté de Gini

Un bon seuil sépare les classes. On mesure le mélange d'un nœud par
l'**impureté de Gini**, où $p_k$ est la proportion de la classe $k$ dans le
nœud :

$$
G = 1 - \sum_k p_k^2
$$

$G = 0$ pour un nœud pur, $G = 0.5$ pour un mélange 50/50 de deux classes.
À la racine : $p = 212/569$ maligne, $G = 1 - 0.373^2 - 0.627^2 = 0.468$.
L'algorithme essaie toutes les variables et tous les seuils, et garde la
coupure qui fait le plus baisser l'impureté moyenne des deux enfants.

```python
from sklearn.tree import DecisionTreeClassifier
arbre = DecisionTreeClassifier(max_depth=2).fit(X_app, y_app)
```

---

## Un arbre profond apprend le bruit

![Exactitude d'un arbre selon sa profondeur](assets/iam-s2-preparation-forets/profondeur-surapprentissage.png)

Sans limite de profondeur, l'arbre coupe jusqu'à isoler chaque patient :
100 % d'exactitude sur l'apprentissage, mais il a appris les particularités
de ces patients, pas la règle. Sur de nouveaux patients, il plafonne vers
92 %. C'est le **surapprentissage**. Un arbre est aussi **instable** : changer
quelques patients change les premières coupures et tout l'arbre.

<!-- notes: 4 minutes. Écart entre les deux courbes = surapprentissage. La
ligne verte annonce la suite : la forêt fait mieux que le meilleur arbre. -->

---

## La forêt aléatoire

![Chaque arbre reçoit son échantillon de patients et de variables](assets/iam-s2-preparation-forets/foret-aleatoire.png)

Deux sources de hasard rendent les arbres différents :

1. **Bootstrap** : chaque arbre apprend sur un tirage de $n$ patients
   **avec remise** parmi les $n$ (certains deux fois, d'autres jamais).
2. **Variables tirées au sort** : à chaque nœud, l'arbre ne cherche la
   meilleure coupure que parmi un sous-ensemble de variables (environ
   $\sqrt{p}$ sur $p$).

La prédiction est le **vote majoritaire** des arbres.

---

## Pourquoi le vote marche

Chaque arbre surapprend, mais pas de la même façon : leurs erreurs sont en
partie indépendantes, et le vote les compense. Comme un avis collégial,
souvent plus fiable que chaque avis isolé, à condition que les avis
diffèrent. Si $\rho$ est la corrélation entre deux arbres et $\sigma^2$ la
variance d'un arbre, la variance de la moyenne de $B$ arbres vaut :

$$
\rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2
$$

Ajouter des arbres réduit le second terme ; décorréler les arbres (le tirage
des variables) réduit le premier.

```python
from sklearn.ensemble import RandomForestClassifier
foret = RandomForestClassifier(n_estimators=300, random_state=42)
foret.fit(X_app, y_app)
```

<!-- notes: 5 minutes. La formule n'est pas à retenir ; le message est :
plus d'arbres ne surapprend pas davantage, et des arbres différents valent
mieux que des arbres identiques. Une forêt n'a pas besoin de standardisation :
une coupure « x <= seuil » ne dépend pas de l'échelle. -->

---

## Les importances des variables

![Importances des variables de la forêt](assets/iam-s2-preparation-forets/importances-foret.png)

`foret.feature_importances_` additionne, pour chaque variable, la baisse
d'impureté apportée par toutes ses coupures, sur tous les arbres. Ici, la
taille et la forme concave des noyaux dominent : cohérent avec la
cytologie. Une importance dit ce que le modèle **utilise**, pas une cause ;
entre variables corrélées (rayon, périmètre, aire), elle se partage.

---

## À retenir

- Un arbre pose des questions « variable ≤ seuil » choisies pour réduire
  l'impureté de Gini $G = 1 - \sum_k p_k^2$.
- Un arbre profond surapprend et il est instable.
- Une forêt = bootstrap + variables tirées au sort + vote : moins de
  variance, sans standardisation.
- `feature_importances_` indique les variables les plus utilisées.
