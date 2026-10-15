# Exercices — Un export hospitalier

Dans ce notebook, vous calculez d'abord sur des tableaux NumPy (pressions
artérielles, IMC, scores z), puis vous explorez avec pandas les 299 patients
du jeu *Heart failure clinical records* et les hospitalisations COVID-19 de
Santé publique France. Le cœur de la séance est le nettoyage, étape par
étape, d'un export de 612 séjours hospitaliers plein de défauts, que vous
joignez ensuite à des résultats de biologie. Comptez environ 1 h 15. Le
livrable est le notebook complété, dont toutes les cellules de vérification
affichent « ✅ ».

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s5-exercices.ipynb)

La [correction](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s5-exercices-correction.ipynb)
est à ouvrir après avoir cherché.

1. Pressions systoliques : statistiques et conversion en kPa, sans boucle.
2. Masques booléens : hypertension selon ESC/ESH 2018, `np.where`.
3. IMC d'un tableau de patients, puis scores z par broadcasting.
4. Lire et inspecter `insuffisance_cardiaque.csv` ; taux de décès.
5. Mortalité par sexe, par anémie, fraction d'éjection < 30 %, classes d'âge avec `pd.cut`.
6. Options de `read_csv` sur un fichier compressé : `dtype`, `parse_dates` ; pic de réanimation à Paris.
7. Nettoyer `sejours_hopital_brut.csv` en sept étapes vérifiées : doublons,
   services, sexe, dates et durée de séjour, taille, glycémie, âge 999.
8. Biologie : `pivot` du format long au format large, `merge` avec les séjours, `groupby` et `agg`.
9. Trouvez l'erreur : `and` au lieu de `&`, et une modification perdue (affectation en chaîne).
10. Pour aller plus loin (facultatif) : `pivot_table`, `melt`, proportion de CRP élevée par service.

Trois cellules « Lire du code » s'intercalent : vues NumPy et axes, `loc`
contre `iloc` après un tri, filtres enchaînés.

Pour travailler hors de Colab : téléchargez le fichier `.ipynb` (Fichier →
Télécharger), installez les bibliothèques avec `pip install numpy pandas`,
et ouvrez-le avec `jupyter lab` ou VS Code. Les données sont lues depuis
GitHub : une connexion Internet est nécessaire.

<!-- notes: Les exercices 1 à 3 vont vite (15 min) ; ne pas laisser la salle
s'y attarder. L'exercice 7 est le cœur de la séance (30 min) : faire
d'abord lire la cellule d'inspection et lister les défauts au tableau.
Blocages attendus : 7.2, oublier une variante (le message de l'assert le
dit) ; 7.4, combiner les deux passes avec fillna ; 7.5, modifier sans
.loc. L'exercice 9b est le message principal : un code qui s'exécute sans
erreur peut ne rien modifier. La partie D dépend de la partie C : un
étudiant bloqué peut exécuter les cellules de la correction pour l'exercice
7, puis reprendre. -->
