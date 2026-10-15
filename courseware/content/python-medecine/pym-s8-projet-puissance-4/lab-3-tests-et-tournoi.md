# Lab 3 — Tester et organiser un tournoi

Vous transformez vos vérifications en une vraie stratégie de test : des tests
qui attrapent des erreurs volontaires, des scénarios de jeu, un module
`puissance4.py` testé avec `pytest` et noté par `pylint`. Vous mesurez
ensuite le temps par coup et la mémoire de vos agents, et vous les comparez
dans un tournoi. Comptez environ 60 minutes ; le livrable est le notebook au
vert, et les fichiers `puissance4.py` et `test_puissance4.py` à reprendre
dans votre dépôt GitHub.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-3.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-3-correction.ipynb)
est à ouvrir après avoir cherché.

**Les exercices**

1. Trois fonctions de test qui passent sur le bon code et échouent sur des mutants.
2. Des scénarios : gagner, bloquer, ne pas offrir la victoire.
   — *Lire du code 1* : quelles fonctions `pytest` exécute-t-il ?
3. `test_puissance4.py` lancé avec `pytest`, puis la note `pylint`.
4. `AgentChronometre` : mesurer chaque coup par délégation.
   — *Lire du code 2* : délégation ou héritage ?
5. `pic_memoire` avec `tracemalloc`.
6. `tournoi` : les taux de victoire de chaque agent contre chaque autre.
   — *Trouvez l'erreur* : un match où le même agent commence toujours.
7. `match_equitable`, la version corrigée.
8. `AgentReglesJournal` : quelle règle sert le plus ?
9. Pour aller plus loin (facultatif) : l'intervalle de confiance d'un taux de victoire.

Hors de Colab, téléchargez le fichier `.ipynb` et installez
`pip install "pettingzoo[classic]" matplotlib pandas pytest pylint`.

<!-- notes: L'exercice 1 est le plus important : un test qui ne peut pas
échouer ne sert à rien, d'où les mutants. Le mutant « oublie les diagonales »
attrape presque tout le monde la première fois. pytest et pylint s'exécutent
dans Colab via subprocess ; dans le dépôt GitHub, ce sont les commandes
python -m pytest et pylint. Le tournoi prend une dizaine de secondes. Finir
sur l'exercice 10 à l'oral : sur 100 parties, 55 % ne se distingue pas de
50 %. -->
