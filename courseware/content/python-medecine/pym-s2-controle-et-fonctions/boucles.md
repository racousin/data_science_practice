# Boucles

Une boucle répète un bloc d'instructions : pour chaque mesure d'une série, pour
chaque heure d'une journée, ou tant qu'une condition reste vraie. C'est ce qui
permet de traiter 48 heures de constantes vitales avec trois lignes de code.

<!-- notes: 30 min. Faire tracer la première boucle à la main au tableau
(tableau des tours) avant de l'exécuter. Points où ils calent : range(n)
s'arrête à n-1, oublier de mettre à jour la variable d'un while (boucle
infinie : bouton stop de Colab), initialiser le compteur dans la boucle au
lieu d'avant. Les listes sont vues en séance 3 : ici, seulement « une suite
de valeurs entre crochets ». -->

---

## for : pour chaque élément

```python
temperatures = [37.2, 38.4, 38.9, 37.8, 36.9]   # une liste (séance 3)
for t in temperatures:
    print(f"{t} °C")
```

À chaque tour, la variable `t` prend la valeur suivante de la séquence, puis le
bloc indenté s'exécute. La boucle s'arrête quand la séquence est épuisée. Une
chaîne de caractères se parcourt de la même façon, caractère par caractère.

---

## Accumuler : compteur et somme

Le motif le plus courant : une variable initialisée **avant** la boucle, mise à
jour **dans** la boucle, lue **après**.

```python
nb_fievre = 0
somme = 0.0
for t in temperatures:
    somme += t
    if t >= 38.0:
        nb_fievre += 1
moyenne = somme / len(temperatures)   # ≈ 37.84 ; nb_fievre vaut 2
```

---

## range : répéter un nombre de fois

`range(debut, fin, pas)` produit des entiers de `debut` inclus à `fin`
**exclu**.

```python
list(range(4))           # [0, 1, 2, 3]
for heure in range(0, 24, 6):
    print(f"{heure:02d}h : paracétamol")   # 00h, 06h, 12h, 18h
```

Pour avoir à la fois la position et la valeur, `enumerate` évite de gérer un
compteur à la main :

```python
for i, t in enumerate(temperatures):
    print(f"mesure {i} : {t} °C")
```

---

## while : tant que

`while` répète tant que sa condition est vraie. On l'utilise quand on ne connaît
pas le nombre de tours à l'avance, par exemple pour l'élimination d'un
médicament : la concentration est divisée par deux à chaque demi-vie.

```python
concentration, heures = 80.0, 0     # mg/L, h
while concentration >= 5.0:
    concentration = concentration / 2
    heures += 6                       # demi-vie de 6 h
print(heures, concentration)          # 30 2.5
```

![Trace de la boucle while : la concentration est divisée par deux à chaque tour](assets/pym-s2-controle-et-fonctions/demi-vie.png)

Si rien ne modifie la condition dans le bloc, la boucle ne s'arrête jamais :
dans Colab, il faut alors interrompre l'exécution (bouton ■).

---

## break et continue

`break` quitte la boucle immédiatement ; `continue` passe directement au tour
suivant.

```python
spo2 = [97, 96, None, 93, 88, 91]     # None : mesure manquante
for heure, valeur in enumerate(spo2):
    if valeur is None:
        continue               # mesure manquante : ignorée
    if valeur < 90:
        print(f"Désaturation à H{heure} : {valeur} %")
        break                  # première alerte : on s'arrête
```

Ce code affiche `Désaturation à H4 : 88 %` et ne regarde pas la mesure de H5.

---

## Boucles imbriquées

Une boucle peut en contenir une autre : la boucle intérieure fait tous ses
tours à chaque tour de la boucle extérieure.

```python
for jour in range(1, 3):
    for heure in range(8, 21, 12):
        print(f"J{jour} {heure}h")    # J1 8h, J1 20h, J2 8h, J2 20h
```

---

## À retenir

- `for x in sequence` parcourt une séquence ; `range(n)` va de 0 à `n - 1`.
- Compteurs et sommes s'initialisent avant la boucle.
- `while` pour un nombre de tours inconnu : la condition doit finir par devenir fausse.
- `break` sort de la boucle, `continue` passe au tour suivant.
