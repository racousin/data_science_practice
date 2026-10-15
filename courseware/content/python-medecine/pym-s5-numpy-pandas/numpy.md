# NumPy : calculer sur des tableaux

Une liste Python range des valeurs ; elle ne sait pas calculer dessus. NumPy
apporte le **tableau** (`ndarray`) : des nombres d'un même type, rangés en
une ou plusieurs dimensions, sur lesquels une opération s'applique à tous les
éléments à la fois. C'est la brique de tout l'écosystème scientifique :
**SciPy** (tests statistiques, optimisation, signal), **pandas** (tableaux de
données), **Matplotlib** et **seaborn** (graphiques), **scikit-learn**.

```python
import numpy as np   # alias usuel ; hors de Colab : pip install numpy
```

Documentation : [numpy.org/doc](https://numpy.org/doc/stable/user/absolute_beginners.html).

<!-- notes: 30 minutes. Faire taper les exemples dans Colab au fil de la
leçon. Blocage fréquent : confondre axis=0 et axis=1 ; le dessin "axis=0 =
on descend le long des lignes, résultat par colonne" aide. -->

---

## Une liste ne calcule pas, un tableau si

Convertir des pressions artérielles de mmHg en kPa (1 mmHg = 0,133322 kPa)
demande une boucle avec une liste. Avec un tableau, l'opération s'écrit sur
le tableau entier : on parle de **calcul vectorisé**.

```python
pas_liste = [128, 151, 117, 140]
pas_liste * 2              # [128, 151, 117, 140, 128, 151, 117, 140]

pas = np.array(pas_liste)
(pas * 0.133322).round(2)  # array([17.07, 20.13, 15.6 , 18.67])
```

Le calcul vectorisé est plus court à écrire et beaucoup plus rapide : la
boucle est faite en C, à l'intérieur de NumPy.

---

## Créer un tableau, connaître sa forme

`np.array` part d'une liste (ou d'une liste de listes pour deux dimensions).
D'autres fonctions créent des tableaux remplis :

```python
np.zeros(3)                  # array([0., 0., 0.])
np.ones((2, 3))              # 2 lignes, 3 colonnes de 1
np.arange(0, 24, 6)          # array([ 0,  6, 12, 18]) — comme range
np.linspace(36, 40, 5)       # 5 valeurs régulières de 36 à 40 inclus
```

Un tableau a une **forme** (`shape`), un nombre de dimensions (`ndim`), une
taille (`size`) et un **type unique** (`dtype`) :

```python
mesures = np.array([[128, 72], [151, 88], [117, 64]])   # PAS, FC
mesures.shape, mesures.ndim, mesures.size, mesures.dtype
# ((3, 2), 2, 6, dtype('int64'))
mesures.astype(float)        # copie convertie en flottants
```

Types courants : `int64`, `int32`, `float64`, `float32`, `bool`,
`complex128`, et `object` (n'importe quel objet Python, sans calcul rapide).

---

## Indexer et découper

On accède aux éléments comme dans une liste, avec **un indice par
dimension**, séparés par une virgule :

```python
pas[0]            # 128 — premier patient
pas[-2:]          # array([117, 140]) — deux derniers
mesures[1, 0]     # 151 — ligne 1, colonne 0
mesures[:, 1]     # array([72, 88, 64]) — toute la colonne FC
mesures[1:3, 0]   # array([151, 117])
```

`:` seul veut dire « tout le long de cette dimension ». Un découpage
renvoie une **vue** sur les mêmes données : la modifier modifie le tableau
d'origine (`.copy()` pour l'éviter).

---

## Opérations et agrégations

Les opérateurs `+ - * / **` et les fonctions `np.sqrt`, `np.log`, `np.exp`
agissent élément par élément. Les **agrégations** résument un tableau :

```python
pas.mean(), pas.std(), pas.min(), pas.max(), np.median(pas)
mesures.mean(axis=0)     # array([132. , 74.67]) — une moyenne par colonne
mesures.mean(axis=1)     # une moyenne par ligne (par patient)
```

`axis=0` parcourt les lignes et donne **un résultat par colonne** ;
`axis=1` donne un résultat par ligne. Sans `axis`, tout le tableau est
résumé en un seul nombre.

$$
\mathrm{IMC} = \frac{\mathrm{poids}\ (kg)}{\mathrm{taille}\ (m)^2}
$$

```python
poids = np.array([72, 95, 58]); taille_m = np.array([1.78, 1.70, 1.62])
imc = poids / taille_m ** 2
imc.round(2)                         # array([22.72, 32.87, 22.1 ])
```

---

## Masques booléens

Une comparaison donne un tableau de booléens, un **masque**, qui sert à
compter et à sélectionner. Seuil : hypertension de grade 1 à partir de
140 mmHg de systolique (ESC/ESH 2018).

```python
hta = pas >= 140                 # array([False,  True, False,  True])
hta.sum(), hta.mean()            # (2, 0.5) : True compte pour 1
pas[hta]                         # array([151, 140])
pas[(pas >= 120) & (pas < 140)]  # & = et, | = ou, ~ = non
```

Les parenthèses sont obligatoires autour de chaque comparaison. Les mots
`and` / `or` ne fonctionnent pas sur des tableaux.

`np.where(condition, si_vrai, si_faux)` construit un tableau élément par
élément :

```python
np.where(imc >= 30, "obésité", "non")   # seuil OMS
```

---

## Broadcasting

Quand deux tableaux n'ont pas la même forme, NumPy **étire** le plus petit
le long des dimensions manquantes, sans copier les données. C'est ce qui
permet de centrer chaque colonne en une ligne.

![Broadcasting : une ligne de moyennes soustraite à chaque patient](assets/pym-s5-numpy-pandas/broadcasting.png)

Le score $z$ ramène chaque variable à une moyenne de 0 et un écart type de 1 :

$$
z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}
$$

```python
z = (mesures - mesures.mean(axis=0)) / mesures.std(axis=0)
np.allclose(z.mean(axis=0), 0)      # True
```

Règle : les formes sont comparées en partant de la droite ; deux dimensions
sont compatibles si elles sont égales ou si l'une vaut 1.

---

## Pour aller un peu plus loin

NumPy contient bien d'autres outils, à découvrir dans la documentation quand
le besoin se présente :

```python
mesures.T                            # transposée, forme (2, 3)
np.arange(12).reshape(3, 4)          # changer de forme
np.vstack([mesures, mesures])        # empiler (np.hstack, np.concatenate)
np.sort(pas)                         # copie triée
rng = np.random.default_rng(0)
rng.normal(loc=120, scale=15, size=5)  # 5 PAS simulées
G = mesures.T @ mesures              # produit matriciel (2, 2)
np.linalg.inv(G)                     # inverse (module linalg)
```

---

## À retenir

- Un `ndarray` contient des valeurs d'un seul type ; `shape` donne sa forme.
- Les opérations sont vectorisées : pas de boucle pour transformer un tableau.
- `axis=0` : un résultat par colonne ; `axis=1` : un résultat par ligne.
- Un masque booléen sélectionne et compte : `x[x >= 140]`, `(x >= 140).mean()`.
- Le broadcasting étire le plus petit tableau : `X - X.mean(axis=0)`.
