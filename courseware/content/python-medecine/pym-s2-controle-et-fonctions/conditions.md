# Conditions

Un programme qui exécute toujours les mêmes lignes ne sait rien décider. Les
conditions lui permettent de choisir une branche selon les données : classer un
IMC, déclencher une alerte, refuser une valeur impossible.

Les seuils cliniques de cette séance sont des exemples pédagogiques, pas des
outils d'aide à la décision.

<!-- notes: 25 min. Partir de l'organigramme avant le code : les étudiants
comprennent la logique, c'est l'indentation qui les bloque. Points où ils
calent : oublier les deux-points, mélanger = et ==, croire que plusieurs elif
peuvent s'exécuter. -->

---

## if, elif, else

```python
imc = 27.4
if imc < 18.5:
    classe = "maigreur"
elif imc < 25:
    classe = "normal"
elif imc < 30:
    classe = "surpoids"
else:
    classe = "obésité"
```

Python teste les conditions de haut en bas et exécute **la première branche
vraie, et elle seule**. `elif` et `else` sont facultatifs. Les seuils sont ceux
de l'OMS pour l'adulte.

![Organigramme d'une chaîne if / elif / else sur l'IMC](assets/pym-s2-controle-et-fonctions/organigramme-imc.png)

---

## L'indentation fait partie de la syntaxe

Le bloc exécuté sous une condition est défini par son **retrait** (4 espaces),
après les deux-points `:`.

```python
temperature = 38.6
if temperature >= 38.0:
    print("Fièvre")            # dans le bloc du if
    print("Contrôler dans 1 h") # dans le bloc du if
print("Mesure enregistrée")    # après le if : toujours exécuté
```

Il n'y a ni accolades ni `end` : un retrait incohérent provoque une
`IndentationError`, ou pire, change silencieusement la logique.

---

## Comparaisons et opérateurs booléens

Une comparaison renvoie un booléen, `True` ou `False`.

```python
pas = 95                 # pression artérielle systolique (mmHg)
pas <= 100               # True
pas == 95, pas != 95     # (True, False) — == compare, = affecte
90 <= pas < 140          # True : comparaison chaînée
```

On combine des booléens avec `and`, `or` et `not`. La priorité est
`not` > `and` > `or` ; des parenthèses rendent l'intention explicite.

```python
fievre, tachycardie = True, False
fievre and tachycardie        # False
fievre or tachycardie         # True
not fievre                    # False
False or not False and True   # True : (not False) and True, puis or
```

---

## Valeurs vraies et fausses

Toute valeur peut servir de condition : `0`, `0.0`, `""`, `None` et les
collections vides sont considérés comme faux, le reste comme vrai.

```python
bool(0), bool(1), bool(""), bool("R07")   # (False, True, False, True)
commentaire = ""
if not commentaire:
    print("Aucun commentaire saisi")
```

Un booléen est aussi un entier : `True + True` vaut `2`. C'est commode pour
compter des critères remplis.

---

## Expression conditionnelle

Quand une condition ne sert qu'à choisir une valeur, une seule ligne suffit :

```python
age = 72
categorie = "senior" if age >= 65 else "adulte"
```

C'est l'équivalent de quatre lignes `if / else`. On la réserve aux choix simples
entre deux valeurs : au-delà, la chaîne `if / elif` se lit mieux.

---

## Un score clinique : additionner des critères

Le score qSOFA (Sepsis-3, Singer et al., JAMA 2016) compte trois critères ; un
score ≥ 2 signale un risque élevé de mauvaise évolution chez un patient infecté.

```python
fr, pas, glasgow = 24, 98, 15
score = 0
if fr >= 22:
    score += 1
if pas <= 100:
    score += 1
if glasgow < 15:
    score += 1
```

Ici on veut tester **tous** les critères : trois `if` indépendants, pas une
chaîne `elif` qui s'arrêterait au premier vrai.

---

## match : choisir selon une valeur

Depuis Python 3.10, `match` compare une valeur à une série de motifs.

```python
groupe_sanguin = "O-"
match groupe_sanguin:
    case "O-":
        print("Donneur universel de globules rouges")
    case "AB+":
        print("Receveur universel de globules rouges")
    case "A+" | "A-" | "B+" | "B-" | "O+" | "AB-":
        print("Autre groupe")
    case _:
        print("Groupe inconnu")
```

`case _` attrape tout le reste. `match` sait aussi décomposer une structure
(`case (nom, age):`), utile plus tard avec les tuples.

---

## À retenir

- `if / elif / else` : seule la première branche vraie s'exécute.
- Le bloc est défini par l'indentation après `:`.
- `==` compare, `=` affecte ; `and`, `or`, `not` combinent les booléens.
- Pour cumuler des critères, des `if` indépendants, pas une chaîne `elif`.
