# Nettoyer des données

Un export d'un logiciel hospitalier n'est jamais prêt à analyser : des
lignes en double, un service orthographié de quatre façons, des dates dans
deux formats, une taille saisie en mètres au lieu de centimètres, un âge
« inconnu » codé 999. Une moyenne calculée sur ces données est fausse sans
message d'erreur. Fil rouge : `sejours_hopital_brut.csv`, 612 séjours
(données synthétiques construites pour le cours).

<!-- notes: 30 minutes. Faire deviner les défauts à la salle à partir de
value_counts et describe avant de montrer les corrections. Blocage
fréquent : oublier de réaffecter (df["x"].str.strip() ne modifie pas df).
Le piège de format="mixed" + dayfirst est réel : le montrer en direct. -->

---

## Repérer les problèmes

Trois commandes suffisent à voir la plupart des défauts :

```python
sejours = pd.read_csv(URL + "sejours_hopital_brut.csv")
sejours.info()                   # types, valeurs manquantes
sejours["sexe"].value_counts()   # M 280, F 264, Femme 28, f 22, Homme 18
sejours["taille_cm"].describe()  # min 1.56 : une taille en mètres
```

`isna().sum()` compte les valeurs manquantes par colonne ;
`duplicated().sum()` compte les lignes identiques à une ligne précédente.
On corrige ensuite **une étape à la fois**, en vérifiant chaque étape.

---

## Doublons et textes

```python
sejours.duplicated().sum()                      # 12
sejours = sejours.drop_duplicates()             # 600 lignes
sejours = sejours.reset_index(drop=True)        # index 0..599
sejours.duplicated(subset=["id_sejour"]).sum()  # doublons sur la clé seule
```

Les textes se normalisent avec l'accesseur `.str`, puis un dictionnaire
ramène chaque variante au nom canonique :

```python
service = sejours["service"].str.strip().str.lower()   # "Cardio " -> "cardio"
canonique = {"cardio": "Cardiologie", "cardiologie": "Cardiologie",
             "pneumo": "Pneumologie", ...}
sejours["service"] = service.map(canonique)
```

`map` renvoie `NaN` pour une valeur absente du dictionnaire : un
`isna().sum()` après coup signale les variantes oubliées. `replace`, lui,
laisse intactes les valeurs non listées.

---

## Dates

`pd.to_datetime` convertit du texte en dates. Donner le `format` rend la
conversion stricte ; `errors="coerce"` met `NaT` (date manquante) là où le
format ne convient pas. Deux formats se traitent donc en deux passes :

```python
iso = pd.to_datetime(sejours["date_entree"], format="%Y-%m-%d", errors="coerce")
fr = pd.to_datetime(sejours["date_entree"], format="%d/%m/%Y", errors="coerce")
sejours["date_entree"] = iso.fillna(fr)
```

Raccourci tentant : `format="mixed", dayfirst=True`. Sur les 600 séjours, il
inverse jour et mois de 155 dates ISO comme `2025-02-12`, sans erreur ni
avertissement. Un format explicite par passe est plus sûr.

```python
sejours["date_sortie"] = pd.to_datetime(sejours["date_sortie"], format="%Y-%m-%d")
duree = sejours["date_sortie"] - sejours["date_entree"]   # Timedelta
sejours["duree_sejour"] = duree.dt.days
sejours["date_entree"].dt.month                           # aussi .dt.year, .dt.day_name()
```

---

## Unités et valeurs sentinelles

Une valeur dans la mauvaise unité se repère par sa plage : une taille
adulte inférieure à 3 ne peut être qu'en mètres. On corrige les lignes
d'un masque avec `.loc` :

```python
en_metres = sejours["taille_cm"] < 3
sejours.loc[en_metres, "taille_cm"] = sejours.loc[en_metres, "taille_cm"] * 100
```

Même principe avec `np.where`, pour une glycémie saisie en mmol/L au lieu de
g/L (1 g/L de glucose = 5,551 mmol/L) :

```python
g = sejours["glycemie"]
sejours["glycemie"] = np.where(g > 4, g / 5.551, g)
```

Le seuil de 4 est un choix : 4 g/L est exceptionnel, 4 mmol/L est une
hypoglycémie. Une valeur comme 3,4 reste ambiguë ; en pratique, on demande
l'unité à la source.

Une **sentinelle** est un code qui signifie « inconnu » (999, -1, 9999) ;
elle doit devenir `NaN`, sinon elle fausse les moyennes :

```python
sejours["age"] = sejours["age"].replace(999, np.nan)   # ou na_values=[999] à la lecture
```

---

## Valeurs manquantes

`NaN` (*Not a Number*) marque une valeur absente. Les statistiques de pandas
l'ignorent (`mean()` se fait sur les valeurs présentes) ; c'est à vous de
décider quoi faire des lignes incomplètes.

```python
sejours.isna().sum()                  # combien, par colonne
sejours.dropna(subset=["poids_kg"])   # retirer les lignes sans poids
sejours["poids_kg"].fillna(sejours["poids_kg"].median())   # imputer
# série horaire : reporter la dernière mesure connue
constantes["spo2"].ffill()
```

Avant d'imputer : la valeur manque-t-elle au hasard ? Une glycémie absente
peut signifier « non prescrite » parce que le patient allait bien. Remplacer
par 0 est presque toujours faux. `astype("Int64")` garde des entiers malgré
les `NaN`.

---

## Modifier sans se tromper

Une modification doit viser le DataFrame lui-même, en une seule opération
`.loc[masque, colonne]` :

```python
sejours[sejours["age"] > 90]["age"] = 90        # FAUX : agit sur une copie
sejours.loc[sejours["age"] > 90, "age"] = 90    # juste
```

La première ligne modifie une copie intermédiaire puis la jette : `sejours`
ne change pas (pandas affiche un avertissement, `SettingWithCopyWarning`
ou `ChainedAssignmentError` selon la version). De même, pour garder un
sous-tableau que l'on modifiera ensuite : `sous = df[masque].copy()`.

---

## À retenir

- Inspecter avant de corriger : `info`, `value_counts`, `describe`,
  `isna().sum()`, `duplicated().sum()`.
- Une étape à la fois, vérifiée par un `assert` ou un comptage.
- Dates : `pd.to_datetime` avec un `format` explicite.
- Unités et sentinelles : masque + `.loc`, `np.where`, `replace(999, np.nan)`.
- Modifier avec `df.loc[masque, col] = ...`, jamais `df[masque][col] = ...`.
