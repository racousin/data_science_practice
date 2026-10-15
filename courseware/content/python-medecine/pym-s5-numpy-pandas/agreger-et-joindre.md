# Agréger et joindre

Une fois les séjours nettoyés, les questions cliniques sont des questions
de **groupes** (durée moyenne de séjour par service, mortalité par classe
d'âge) et de **jointures** (rapprocher les séjours des résultats de
biologie du même patient). Cette leçon présente les quatre opérations qui y
répondent : `groupby`, `pivot_table`, `pivot`/`melt` et `merge`.

<!-- notes: 30 minutes. Dessiner split-apply-combine au tableau avant de
montrer le code. Blocage fréquent : un merge qui multiplie les lignes ;
faire compter les lignes avant et après chaque jointure. -->

---

## groupby : séparer, appliquer, combiner

`groupby` **sépare** les lignes selon les valeurs d'une colonne, **applique**
un calcul à chaque groupe et **combine** les résultats dans un nouveau
tableau indexé par les groupes.

![Split, apply, combine](assets/pym-s5-numpy-pandas/split-apply-combine.png)

```python
sejours.groupby("service")["duree_sejour"].mean()
sejours.groupby("service").size()                    # nombre de séjours
sejours.groupby(["service", "sexe"])["deces"].mean() # deux clés
```

Les fonctions d'agrégation usuelles : `mean`, `median`, `sum`, `count`
(valeurs non manquantes), `size` (lignes), `min`, `max`, `std`.

---

## agg : plusieurs résumés à la fois

`agg` applique plusieurs fonctions, et nomme les colonnes du résultat :

```python
sejours.groupby("service").agg(
    n=("id_sejour", "count"),
    duree_moy=("duree_sejour", "mean"),
    duree_med=("duree_sejour", "median"),
    mortalite=("deces", "mean"),
)
```

Une fonction maison est acceptée, par exemple l'étendue :

```python
def etendue(x):
    return x.max() - x.min()

sejours.groupby("service")["age"].agg(["mean", etendue])
```

---

## pivot_table : un tableau croisé

`pivot_table` croise deux variables qualitatives, comme le tableau croisé
dynamique d'un tableur : une en lignes, une en colonnes, une statistique
dans les cases.

```python
sejours.pivot_table(index="service", columns="sexe",
                    values="duree_sejour", aggfunc="mean",
                    margins=True)  # ajoute la ligne et la colonne "All"
```

Pour compter des effectifs, `pd.crosstab(sejours["service"], sejours["sexe"])`
fait la même chose avec `aggfunc="count"`.

---

## Format long, format large

Un laboratoire exporte souvent une ligne **par mesure** (format long) ;
l'analyse veut une ligne **par patient**, une colonne par examen (format
large). `pivot` passe du long au large, `melt` fait l'inverse.

![Format long et format large](assets/pym-s5-numpy-pandas/long-large.png)

```python
examens = pd.read_csv(URL + "examens_biologiques.csv")      # 1178 lignes
large = examens.pivot(index="id_patient", columns="examen", values="valeur")
large.shape                                                 # (332, 4)
long = large.reset_index().melt(id_vars="id_patient", value_name="valeur")
```

`pivot` exige un seul couple (ligne, colonne) par case ; s'il y a des
doublons (deux CRP le même patient), il faut `pivot_table` avec une
`aggfunc`. Un examen non fait devient `NaN` dans le format large.

---

## merge : joindre deux tableaux

`merge` rapproche les lignes de deux tableaux qui partagent une **clé**,
comme une jointure SQL. L'argument `how` dit quoi faire des clés présentes
d'un seul côté.

![inner, left, outer](assets/pym-s5-numpy-pandas/merge-how.png)

```python
complet = sejours.merge(large.reset_index(), on="id_patient", how="left")
len(sejours), len(complet)                  # (600, 600)
```

`how="left"` garde tous les séjours, avec `NaN` si le patient n'a pas de
biologie ; `inner` ne garde que les clés communes ; `outer` garde tout.
Les clés portent un nom différent ? `left_on="patient", right_on="id_patient"`.

---

## Vérifier une jointure

Si la clé n'est pas unique du côté droit, chaque séjour est **dupliqué**
autant de fois qu'il y a de correspondances : joindre les séjours au
format long des examens donnerait 2 150 lignes au lieu de 600.

```python
sejours.merge(examens, on="id_patient").shape   # bien plus de 600 lignes
# validate : erreur si la clé n'est pas unique du côté droit
sejours.merge(large.reset_index(), on="id_patient",
              validate="many_to_one")
```

Compter les lignes avant et après chaque jointure est le réflexe à prendre.

`pd.concat` **empile** des tableaux de même structure, par exemple deux
exports mensuels : `pd.concat([janvier, fevrier], ignore_index=True)` ;
avec `axis=1`, il les place côte à côte en alignant les index.

---

## À retenir

- `groupby(cle)[col].fonction()` : séparer, appliquer, combiner.
- `agg(nom=(colonne, fonction))` pour plusieurs résumés nommés.
- `pivot_table` croise deux variables ; `pivot` / `melt` passent du long
  au large et retour.
- `merge(..., on=, how=)` joint sur une clé ; compter les lignes avant et
  après, `validate=` pour l'imposer.
