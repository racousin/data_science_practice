# Lab 2 — Un agent à règles

Vous construisez un agent qui réfléchit : gagner quand c'est possible, sinon
bloquer l'adversaire, sinon éviter le coup qui lui offre la victoire, de
préférence au centre. Il repose sur de petites fonctions sur une grille 6 × 7,
chacune vérifiée sur des positions dessinées en texte. Comptez environ 60
minutes ; le livrable est le notebook au vert et un `AgentRegles` qui bat
l'agent aléatoire au moins 90 fois sur 100.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-2.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-2-correction.ipynb)
est à ouvrir après avoir cherché.

**Les exercices**

1. `vers_grille` : de l'observation `(6, 7, 2)` à une grille d'entiers 0 / 1 / 2.
2. `coups_valides` : les colonnes dont la case du haut est vide.
3. `ligne_libre` : où tombe le pion.
   — *Lire du code 1* : `b = a` contre `a.copy()`.
4. `simuler_coup` : jouer sur une copie.
5. Les 69 fenêtres de quatre cases (quatre directions).
6. `est_gagnant` : quatre pions alignés.
   — *Trouvez l'erreur* : une recherche de coup gagnant qui abîme la grille.
7. `coup_gagnant`, la version correcte.
8. `coups_surs` : ne pas jouer sous la case qui manque à l'adversaire.
9. `AgentRegles` : les quatre règles, testées sur des positions, puis 100 parties.
   — *Lire du code 2* : gagner ou bloquer, que joue l'agent ?
10. Pour aller plus loin (facultatif) : la double menace.

Hors de Colab, téléchargez le fichier `.ipynb` et installez
`pip install "pettingzoo[classic]" matplotlib`.

<!-- notes: Le notebook reprend en tête la correction du lab 1 : un binôme en
retard peut commencer ici. Points de blocage attendus : l'ordre des lignes
(0 en haut, 5 en bas) dans ligne_libre, et l'aliasing — le « Trouvez
l'erreur » de l'exercice 8 est le même piège que le Lire du code de
l'exercice 4, le faire remarquer. Demander pourquoi « gagner » passe avant
« bloquer » (exercice 11) avant d'exécuter. -->
