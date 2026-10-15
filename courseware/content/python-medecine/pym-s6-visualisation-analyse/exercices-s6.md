# Exercices — Insuffisance cardiaque et COVID-19

Dans ce notebook, vous tracez des constantes vitales de réanimation avec
Matplotlib, les hospitalisations COVID-19 en France avec un vrai axe de dates,
puis vous menez l'analyse exploratoire d'une cohorte de 299 patients en
insuffisance cardiaque avec seaborn, jusqu'à deux tests statistiques que vous
interprétez par écrit. Comptez environ 1 h 30. Livrable : le notebook complété,
avec toutes les cellules de vérification au vert (✅) et vos réponses rédigées.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s6-exercices.ipynb)

La correction, à ouvrir après avoir cherché :
[pym-s6-exercices-correction.ipynb](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/python-medecine/notebooks/pym-s6-exercices-correction.ipynb).

**Partie A — Constantes vitales (Matplotlib)**

1. La fréquence cardiaque d'un patient sur 48 h : une courbe, des axes nommés avec unités.
2. Les quatre constantes d'un patient sur une grille 2 × 2 (`plt.subplots`).
3. La SpO₂ des 12 patients, les trois qui se dégradent mis en évidence, un seuil à 92 %.
4. Trouvez l'erreur : un tableau d'Axes n'a pas de méthode `plot`.

**Partie B — Hospitalisations COVID-19 (séries temporelles)**

5. La courbe nationale des hospitalisations et des réanimations, et ses vagues.
6. Trois départements comparés, du format long au format large (`pivot`).
7. Les décès quotidiens et leur moyenne glissante sur 7 jours.

**Partie C — Analyse exploratoire (seaborn, scipy)**

8. Distributions : histogrammes, asymétrie, boîte à moustaches par groupe.
9. Relations : nuage de points, matrice de corrélation de Spearman, `pairplot`.
10. Tests de Mann-Whitney et du χ², et leur interprétation rédigée.

S'y ajoutent deux questions *Lire du code* et *Lire un graphique* (un axe
tronqué, un double axe y), et un exercice *Pour aller plus loin* sur les tests
multiples.

**Hors de Colab.** Téléchargez le fichier `.ipynb` (menu *Fichier →
Télécharger*), installez les bibliothèques avec
`pip install pandas matplotlib seaborn scipy jupyter`, puis ouvrez-le avec
`jupyter notebook`. Une connexion internet est nécessaire : les données sont
lues depuis GitHub.

<!-- notes: 90 minutes. Parties A et B en 45 minutes, partie C en 45 minutes. Points de blocage attendus : exercice 2 (zip sur axes.flat et un dictionnaire), exercice 5 (parse_dates déjà fait au chargement : ne pas reconvertir ; l'axe de dates ConciseDateFormatter), exercice 9 (oublier de retirer id_patient avant corr, qui plante sur une colonne texte). Faire lire à voix haute une interprétation de l'exercice 10 et corriger collectivement les formulations "l'anémie n'a pas d'effet" et "la FEVG cause le décès". Les deux figures "Lire un graphique" sont volontairement trompeuses. -->
