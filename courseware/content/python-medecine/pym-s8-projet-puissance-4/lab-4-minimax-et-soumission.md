# Lab 4 — Minimax et soumission sur ML-Arena

Vous écrivez un agent qui anticipe : une fonction d'évaluation, l'algorithme
minimax avec élagage alpha-bêta, puis l'approfondissement itératif, qui
cherche aussi profond que le permet la limite de **0,25 s par coup** de la
plateforme. Vous écrivez `agent.py`, le testez comme ML-Arena l'appelle, et
le soumettez au tournoi du cours. Comptez environ 90 minutes ; le livrable est
votre agent sur le classement du challenge.

**Challenge :** <https://ml-arena.com/viewchallenge/198>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s8-lab-4.ipynb)

La [correction](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s8-lab-4-correction.ipynb)
est à ouvrir après avoir cherché.

**Les exercices** (après un *Lire du code 1* : minimax sur un petit arbre en listes imbriquées)

1. `score_fenetre` : la valeur d'une fenêtre de quatre cases.
2. `evaluer` : la valeur d'une position non terminée.
3. `minimax` avec élagage alpha-bêta.
4. `meilleur_coup` : le coup, pas seulement la valeur.
   — *Trouvez l'erreur* : un minimax où l'adversaire ne joue jamais ; puis le nombre de positions évitées par l'élagage.
5. `AgentMinimax` à profondeur fixe ; puis profondeur contre temps, et où passe le temps (`cProfile`).
   — *Lire du code 2* : garder la dernière profondeur terminée.
6. `choisir_coup` : l'approfondissement itératif sous budget de temps.
7. `agent.py` testé comme sur la plateforme : ≥ 95 % contre l'aléatoire, plus de victoires que de défaites contre l'agent à règles, jamais plus de 0,2 s ; puis la soumission avec le SDK `mlarena-sdk`.
8. Pour aller plus loin (facultatif) : UCB1 et Monte Carlo Tree Search.

**Avant la séance :** créez une clé d'API sur ML-Arena (Profil → API Keys) et
ajoutez-la dans les *Secrets* de Colab sous le nom `MLARENA_API_KEY`. Ne la
collez jamais dans une cellule.

Hors de Colab, téléchargez le fichier `.ipynb`, installez
`pip install "pettingzoo[classic]" matplotlib pandas "mlarena-sdk==3.0.0"` et
définissez la variable d'environnement `MLARENA_API_KEY`.

<!-- notes: Le point délicat est la limite de temps : 0,25 s aller-retour,
sur une machine plus lente que les portables ; un seul dépassement fait perdre
le match. La mesure « profondeur contre temps » montre qu'à profondeur 5 le
pire coup dépasse déjà la limite alors que la moyenne passe : c'est le maximum
qui compte, d'où l'approfondissement itératif (TEMPS_RECHERCHE = 0,08 s). Le
test de l'exercice 8 prend environ deux minutes. Faire soumettre une seule
fois : chaque exécution de la cellule crée une soumission. Le classement ELO
évolue pendant plusieurs heures après la soumission. -->
