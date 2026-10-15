# pandas : lire et explorer un tableau

Les données cliniques arrivent sous forme de tableaux : un patient par
ligne, une variable par colonne, des colonnes de types différents (âge,
sexe, date d'entrée, résultat biologique). **pandas** est la bibliothèque
qui lit ces tableaux, les inspecte et les interroge. Fil rouge de la leçon :
les 299 patients du jeu *Heart failure clinical records* (Chicco et Jurman,
2020, licence CC BY 4.0), suivis après une insuffisance cardiaque.

<!-- notes: 35 minutes. Tout faire en direct dans Colab sur le vrai fichier.
Blocage fréquent : loc et iloc ; montrer l'exemple après tri, où les deux
donnent des lignes différentes. Deuxième blocage : oublier les parenthèses
autour des conditions combinées avec &. -->

---

## Series et DataFrame

Un **DataFrame** est un tableau à deux dimensions dont les lignes et les
colonnes portent des **étiquettes** : l'`index` pour les lignes, `columns`
pour les colonnes. Chaque colonne est une **Series** : un tableau NumPy
d'un seul type, accompagné de l'index.

![Anatomie d'un DataFrame](assets/pym-s5-numpy-pandas/anatomie-dataframe.png)

```python
import pandas as pd

patients = pd.DataFrame({"id_patient": ["P01", "P02", "P03"],
                         "age": [75, 55, 65],
                         "sexe": ["M", "F", "M"]})
patients["age"]          # une Series
```

Un DataFrame se construit aussi depuis une liste de lignes :
`pd.DataFrame(data=lignes, columns=["id_patient", "age", "sexe"])`.

---

## Lire un fichier CSV

En pratique, on ne tape pas les données : on les lit. `pd.read_csv` accepte
un chemin local ou une URL.

```python
URL = ("https://raw.githubusercontent.com/racousin/data_science_practice/"
       "main/website/public/modules/python-medecine/donnees/")
ic = pd.read_csv(URL + "insuffisance_cardiaque.csv")
```

![Un CSV lu par pd.read_csv (exemple du cours d'origine, données météo)](assets/pym-s5-numpy-pandas/pandas_1.png)

---

## Options de lecture

Les exports réels demandent souvent des options :

```python
pd.read_csv("export.csv",
            sep=";", decimal=",",           # CSV « à la française »
            dtype={"code_postal": str},     # garder les zéros initiaux
            parse_dates=["date_entree"],    # lire les dates
            na_values=["NA", "-", 999])     # codes de valeur manquante
```

Un fichier compressé (`.csv.gz`, `.zip`) se lit directement : `compression`
est déduit de l'extension. Pour Excel : `pd.read_excel`.

---

## Inspecter

Avant tout calcul, on regarde ce que l'on a lu :

```python
ic.shape            # (299, 14) : lignes, colonnes
ic.head()           # les 5 premières lignes (tail() : les dernières)
ic.info()           # type et nombre de valeurs non manquantes par colonne
ic.describe()       # moyenne, écart type, quartiles des colonnes numériques
```

Pour une variable qualitative, on compte les modalités :

```python
ic["sexe_homme"].value_counts()               # 1 : 194, 0 : 105
ic["deces"].value_counts(normalize=True)      # proportions
ic["deces"].mean()                            # 0.321 : taux de décès
```

La moyenne d'une colonne 0/1 est une proportion : 32,1 % des patients sont
décédés pendant le suivi.

---

## Sélectionner : [], loc, iloc

`df["col"]` donne une colonne, `df[["col1", "col2"]]` un DataFrame de
plusieurs colonnes. Pour choisir lignes **et** colonnes :

- `.loc[lignes, colonnes]` sélectionne **par étiquette** ;
- `.iloc[lignes, colonnes]` sélectionne **par position** (0, 1, 2…).

```python
ic.loc[[1, 3], ["age", "fraction_ejection"]]   # étiquettes 1 et 3
ic.loc[:, "age":"cpk"]                         # toutes les lignes, colonnes de age à cpk
ic.iloc[0, 1]                                  # 1re ligne, 2e colonne
```

![Sélection avec loc (exemple du cours d'origine)](assets/pym-s5-numpy-pandas/pandas_2.png)

Après un tri, l'index n'est plus dans l'ordre : `loc[0]` est toujours le
patient d'étiquette 0, `iloc[0]` est la ligne qui se trouve en premier.

---

## Filtrer avec des conditions

Comme avec NumPy, une condition sur une colonne donne un masque booléen,
que l'on passe entre crochets :

```python
fe_basse = ic[ic["fraction_ejection"] < 30]   # 59 patients
fe_basse["deces"].mean()                      # 0.644
ic[(ic["age"] >= 75) & (ic["anemie"] == 1)]   # & et |, avec parenthèses
ic[ic["id_patient"].isin(["IC001", "IC010"])]
```

Une fraction d'éjection inférieure à 30 % signe une dysfonction systolique
sévère : la mortalité y est deux fois plus élevée que dans l'ensemble.

Pour les textes, l'accesseur `.str` fournit `startswith`, `contains`,
`lower`… : `df[df["diagnostic_cim10"].str.startswith("I")]` garde les
maladies de l'appareil circulatoire (codes I00–I99, chapitre IX de la CIM-10).

---

## Créer des colonnes, trier

Une nouvelle colonne s'écrit comme une affectation dans un dictionnaire ;
le calcul est vectorisé :

```python
ic["creatinine_umol"] = ic["creatinine"] * 88.4   # mg/dL -> µmol/L
ic["classe_age"] = pd.cut(ic["age"], bins=[0, 60, 75, 120],
                          labels=["<60", "60-74", "75+"], right=False)
```

`pd.cut` découpe une variable continue en classes ; `right=False` rend
les intervalles fermés à gauche : [60, 75) contient 60, pas 75.

```python
ic.sort_values("fraction_ejection")                         # croissant
ic.sort_values(["deces", "age"], ascending=[False, True])   # deux clés
ic.sort_index()                                             # revenir à l'ordre de l'index
```

`sort_values` renvoie un **nouveau** DataFrame : `ic` n'est pas modifié tant
qu'on n'écrit pas `ic = ic.sort_values(...)`.

---

## Statistiques par colonne

Les méthodes statistiques s'appliquent colonne par colonne :

```python
ic[["age", "sodium", "creatinine"]].mean()      # une moyenne par colonne
ic["sodium"].median(), ic["sodium"].std()
ic["classe_age"].unique()
ic.groupby("classe_age", observed=True)["deces"].mean()
# <60 : 0.24   60-74 : 0.31   75+ : 0.61
```

La dernière ligne est un **groupby**, la clé de l'agrégation, détaillée
dans la leçon *Agréger et joindre*.

---

## À retenir

- Un DataFrame = des colonnes (Series) partageant un index.
- `read_csv` lit fichiers locaux, URL et fichiers compressés ; `sep`,
  `decimal`, `dtype`, `parse_dates`, `na_values` s'adaptent à l'export.
- Toujours inspecter : `shape`, `head`, `info`, `describe`, `value_counts`.
- `loc` sélectionne par étiquette, `iloc` par position.
- Filtrer : `df[(cond1) & (cond2)]`, une condition par paire de parenthèses.
