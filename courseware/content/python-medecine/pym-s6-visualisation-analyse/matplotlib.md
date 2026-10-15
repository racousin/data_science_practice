# Matplotlib : figure et axes

Matplotlib est la bibliothèque de référence pour tracer des graphiques en Python.
seaborn et les méthodes `.plot()` de pandas sont construites dessus : comprendre
ses deux objets de base, la *Figure* et les *Axes*, permet de tout personnaliser.

```python
import matplotlib.pyplot as plt   # convention d'import
```

<!-- notes: 30 minutes. Montrer en direct dans Colab, sur constantes_vitales.csv. Les étudiants confondent Axes (une zone de tracé) et Axis (un axe gradué) : insister sur l'image de la slide 2. Ressources : matplotlib.org/stable/gallery (galerie d'exemples) et le tutoriel github.com/matplotlib/AnatomyOfMatplotlib. -->

---

## Figure, Axes, Axis

![Une Figure contient des Axes, qui contiennent deux objets Axis](assets/pym-s6-visualisation-analyse/Figure_Axes.png)

- La **Figure** est la « fenêtre » : elle contient tout, y compris un titre général.
- Un **Axes** est une zone de tracé qui reçoit les données ; une figure peut en contenir plusieurs.
- Chaque Axes possède deux objets **Axis** (x et y, trois en 3D), avec graduations et étiquettes.

<!-- notes: Faire remarquer que le titre, les étiquettes x/y et la légende appartiennent à l'Axes, pas à la Figure. -->

---

## Deux façons d'écrire le même graphique

L'interface **pyplot** agit sur la figure « courante », créée automatiquement.

```python
plt.hist(ic["fraction_ejection"], bins=20)
plt.title("Fraction d'éjection")
plt.xlabel("FEVG (%)")
plt.show()
```

L'interface **objet** nomme explicitement la figure et l'axe. C'est celle à
préférer dès qu'il y a plus d'un graphique.

```python
fig, ax = plt.subplots(figsize=(6, 4))   # largeur, hauteur en pouces
ax.hist(ic["fraction_ejection"], bins=20)
ax.set_title("Fraction d'éjection")
ax.set_xlabel("FEVG (%)")
```

Les noms changent légèrement : `plt.title` devient `ax.set_title`,
`plt.xlim` devient `ax.set_xlim`. On peut aussi créer la figure seule
(`fig = plt.figure()`) puis placer un axe à la main avec
`fig.add_axes([gauche, bas, largeur, hauteur])`.

<!-- notes: ic est le DataFrame insuffisance_cardiaque.csv (séance 5). Le point bloquant : mélanger les deux styles dans une même cellule. -->

---

## Les tracés de base

Chaque méthode répond à un type de question.

```python
ax.plot(heures, fc)                  # courbe : évolution dans le temps
ax.scatter(ic["age"], ic["creatinine"])  # nuage : relation entre deux variables
ax.hist(ic["sodium"], bins=25)       # histogramme : distribution d'une variable
ax.bar(["Anémie", "Diabète"], [129, 125])  # barres : une valeur par catégorie
ax.boxplot([fe_survivants, fe_deces])      # boîtes : comparer des distributions
```

Une boîte à moustaches montre la médiane, les quartiles $Q_1$ et $Q_3$, et des
moustaches qui vont jusqu'à la dernière observation située à moins de
$1{,}5 \times (Q_3 - Q_1)$ de la boîte ; les points au-delà sont tracés un par un. `ax.barh` trace des barres horizontales, `ax.pie` un
camembert, à réserver à deux ou trois catégories. Un nuage dont la taille des
points dépend d'une troisième variable (`s=`) s'appelle un *bubble plot*.

<!-- notes: Les effectifs 129/125 sont ceux du jeu insuffisance_cardiaque (anemie = 1, diabete = 1). -->

---

## Plusieurs graphiques : subplots

`plt.subplots(nlignes, ncolonnes)` renvoie la figure et un **tableau d'Axes**,
indexé comme un tableau NumPy.

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes[0, 0].plot(p["heure"], p["frequence_cardiaque"])
axes[0, 1].plot(p["heure"], p["pression_systolique"])
axes[1, 0].plot(p["heure"], p["spo2"])
axes[1, 1].plot(p["heure"], p["temperature"])
fig.tight_layout()                   # évite les chevauchements de texte
```

Avec une seule ligne, `axes` est un tableau à une dimension :
`fig, (ax1, ax2) = plt.subplots(1, 2)`. L'ancienne écriture
`plt.subplot(1, 2, 1)` (ou `plt.subplot(121)`) sélectionne une case de la grille
puis dessine avec les fonctions `plt.*`.

<!-- notes: Erreur fréquente : axes[2] sur une grille 2×2 (IndexError) ou axes.plot (un tableau n'a pas de méthode plot). -->

---

## Titres, légende, repères

Un graphique médical doit se lire sans le code : un titre, des axes nommés
**avec leur unité**, une légende dès qu'il y a plusieurs séries.

```python
ax.plot(h, spo2_r01, label="R01")
ax.plot(h, spo2_r03, label="R03")
ax.axhline(92, color="grey", linestyle="--", label="Seuil 92 %")
ax.set(xlabel="Heure", ylabel="SpO2 (%)", title="Saturation en oxygène")
ax.legend(loc="lower left")
fig.suptitle("Réanimation — 48 premières heures")
```

`ax.axhline` et `ax.axvline` tracent une ligne de référence (seuil, début d'un
traitement). `ax.set_ylim(80, 100)` fixe les bornes ; `fig.suptitle` titre la
figure entière.

<!-- notes: Le seuil 92 % est un repère pédagogique (cible usuelle de SpO2 chez l'adulte sans BPCO : 92-96 %), pas une règle de décision. Dans la source du cours, "plt.subtitle" n'existe pas : c'est fig.suptitle / plt.suptitle. -->

---

## Enregistrer et changer de style

```python
fig.savefig("spo2.png", dpi=200, bbox_inches="tight")
fig.savefig("spo2.pdf")              # format vectoriel pour un article
```

`bbox_inches="tight"` rogne les marges blanches. Le format est déduit de
l'extension. Un **style** change l'apparence de toutes les figures suivantes :

```python
print(plt.style.available)           # la liste des styles
plt.style.use("ggplot")
```

Une matrice (corrélations, patients × heures) s'affiche en carte de chaleur
avec `ax.imshow(matrice, cmap="viridis")` suivi de `fig.colorbar(...)` ; seaborn
le fait plus simplement (leçon suivante).

<!-- notes: Dans Colab, savefig écrit dans l'espace de travail temporaire : télécharger le fichier depuis le panneau Fichiers. -->

---

## À retenir

- Une **Figure** contient des **Axes** ; chaque Axes porte un graphique, ses axes, son titre.
- Préférer `fig, ax = plt.subplots()` puis les méthodes `ax.*`.
- `plot` pour le temps, `scatter` pour une relation, `hist` et `boxplot` pour une distribution, `bar` pour des catégories.
- Toujours un titre, des axes avec unités, une légende ; `savefig` pour exporter.
