# Lab 1 — Comprendre le jeu et jouer au hasard

Vous découvrez le Puissance 4 tel que le représente PettingZoo : l'observation
à deux canaux, le masque des coups autorisés, la boucle de jeu. Vous écrivez
ensuite un premier agent, aléatoire, qui respecte le contrat de ML-Arena, et
la fonction `jouer_partie` qui fait s'affronter deux agents exactement comme
la plateforme les appelle. Comptez environ 60 minutes ; le livrable est le
notebook avec toutes les vérifications « ✅ » au vert.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-1.ipynb)

La [correction](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s8-lab-1-correction.ipynb)
est à ouvrir après avoir cherché.

**Les exercices**

1. Les nombres du jeu : lignes, colonnes, directions d'alignement, coups au plus.
2. Lire `env.last()` : forme de l'observation, masque, joueur dont c'est le tour.
   — *Lire du code 1* : à qui appartient chaque pion après deux coups ?
3. `afficher_plateau` : la grille en texte (X, O, .).
   — *Trouvez l'erreur* : des valeurs du masque au lieu de ses indices.
4. `colonnes_libres`, avec `enumerate`.
5. Une partie entièrement au hasard avec la boucle de PettingZoo.
6. `AgentAleatoire` : le contrat `setup` / `reset` / `choose_action`.
7. `jouer_partie` : la boucle de ML-Arena, places alternées, `reset` avant chaque partie.
   — *Lire du code 2* : un agent espion qui affiche ses appels à `reset`.
8. `jouer_match`, puis ce que donne le hasard sur 200 parties (avantage au premier joueur, durée, nulles).
9. `AgentPondere` : le centre plus probable, par héritage.
10. Pour aller plus loin (facultatif) : le film d'une partie.

Hors de Colab, téléchargez le fichier `.ipynb` (menu Fichier → Télécharger),
installez `pip install "pettingzoo[classic]" matplotlib` et ouvrez-le dans
Jupyter ou VS Code. Les règles du jeu et le contrat d'agent sont aussi sur la
[page du challenge](https://ml-arena.com/viewchallenge/198).

<!-- notes: Les exercices 1 à 8 sont le cœur ; 9 et 10 se corrigent à l'oral
(demander à main levée si celui qui commence a un avantage avant d'exécuter :
environ 55 %). Points de blocage attendus : la vue de l'observation qui change
de joueur à chaque tour (exercice 3, insister), le bug « valeurs au lieu
d'indices » du masque, et dans jouer_partie l'appel env.step(None) pour un
joueur dont la partie est finie — sans lui la boucle ne se termine pas.
L'installation de PettingZoo prend une vingtaine de secondes dans Colab. -->
