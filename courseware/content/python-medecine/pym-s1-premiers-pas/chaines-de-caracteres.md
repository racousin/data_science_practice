# Chaînes de caractères

Une grande part des données de santé est du texte : un nom, un code CIM-10,
une ligne d'ordonnance, une valeur saisie « TA: 142/91 mmHg ». Le type `str`
représente ces **chaînes de caractères**. Cette leçon montre comment les
découper, les nettoyer et en fabriquer de nouvelles avec les f-strings.

<!-- notes: 25 minutes. Le schéma d'indexation est le point dur : le faire
lire à voix haute. Les étudiants confondent s[3:6] (3 caractères) et s[3:6]
"jusqu'au 6e inclus". -->

---

## Écrire une chaîne

Une chaîne s'écrit entre guillemets simples ou doubles, ou entre triples
guillemets si elle tient sur plusieurs lignes.

```python
code_cim = "I21.4"
service = 'cardiologie'
note = """Patient de 67 ans,
douleur thoracique depuis 2 h."""
```

`len(code_cim)` donne le nombre de caractères (`5`). `+` concatène
(`"I21" + ".4"`), `*` répète (`"-" * 20`), et `in` teste la présence d'un
morceau : `"douleur" in note` vaut `True`.

---

## Indexer et découper

![Indices positifs et négatifs d'une chaîne](assets/pym-s1-premiers-pas/indexation-chaine.png)

Chaque caractère a une position, à partir de **0** ; les indices négatifs
partent de la fin. Une **tranche** `s[debut:fin]` prend de `debut` inclus à
`fin` exclu.

```python
s = "TA 142/91"
s[0], s[-1]        # ('T', '1')
s[3:6]             # '142'
s[:2], s[7:]       # ('TA', '91') : bornes omises = début ou fin
s[::2]             # 'T 4/1' : un caractère sur deux
```

---

## Les méthodes de `str`

Une **méthode** est une fonction attachée à une valeur, appelée avec un point.

```python
saisie = "  Cardiologie  "
saisie.strip()            # 'Cardiologie' : retire les espaces aux bords
saisie.strip().lower()    # 'cardiologie' : les méthodes s'enchaînent
"amoxicilline".upper()    # 'AMOXICILLINE'
"jean dupont".title()     # 'Jean Dupont'
"7,4".replace(",", ".")   # '7.4' : virgule décimale française
```

Une chaîne est **immuable** : `saisie.strip()` ne modifie pas `saisie`, elle
renvoie une nouvelle chaîne qu'il faut ranger (`saisie = saisie.strip()`).

---

## Découper et recoller : `split` et `join`

`split` coupe une chaîne en une liste de morceaux ; `join` fait l'inverse.

```python
mesure = "TA: 142/91 mmHg"
valeur = mesure.split(" ")[1]          # '142/91'
pas, pad = valeur.split("/")           # ('142', '91') : encore du texte
int(pas) - int(pad)                    # 51 : convertir avant de calculer
" | ".join(["I10", "E11.9", "N18.3"])  # 'I10 | E11.9 | N18.3'
```

Le résultat de `split` est une **liste** (séance 3) ; ici, on se contente de
prendre un élément par son indice ou de répartir deux morceaux dans deux
variables.

---

## Les f-strings

Une **f-string** (préfixe `f`) insère des valeurs dans un texte, entre
accolades. Après `:`, un **format** fixe l'affichage.

```python
nom, imc = "Mme Martin", 27.68166
print(f"{nom} : IMC = {imc}")        # Mme Martin : IMC = 27.68166
print(f"{nom} : IMC = {imc:.1f}")    # Mme Martin : IMC = 27.7
print(f"[{imc:8.2f}]")               # [   27.68] : 8 caractères de large
print(f"{0.237:.0%}")                # 24%
```

`.1f` : une décimale ; `8.2f` : deux décimales sur 8 caractères ; `%` :
pourcentage. Le format n'arrondit que l'**affichage**, la variable reste
inchangée.

---

## Aligner un petit tableau

La largeur et l'alignement (`<` à gauche, `>` à droite) permettent d'aligner
des colonnes.

```python
print(f"{'Examen':<12}{'Valeur':>8} {'Unité'}")
print(f"{'Créatinine':<12}{88.4:>8.1f} µmol/L")
print(f"{'Potassium':<12}{4.3:>8.1f} mmol/L")
```

```text
Examen        Valeur Unité
Créatinine      88.4 µmol/L
Potassium        4.3 mmol/L
```

Les noms de colonnes sont eux-mêmes des chaînes formatées : `{'Examen':<12}`
réserve 12 caractères et aligne le texte à gauche.

---

## À retenir

- Indices à partir de 0, négatifs depuis la fin ; `s[a:b]` exclut `b`.
- `strip`, `lower`, `upper`, `replace`, `split`, `join` renvoient une
  **nouvelle** chaîne (ou liste) : une chaîne est immuable.
- `split` produit du texte : convertir avec `int()` / `float()` avant de
  calculer.
- f-string : `f"{valeur:.1f}"` pour une décimale, `:>8` pour aligner.
