# Le projet Puissance 4

![Connect Four Project](assets/pym-s8-projet-puissance-4/connect4.png)

En binôme, vous programmez un **joueur de Puissance 4** en Python : d'abord au
hasard, puis avec des règles, enfin avec une recherche qui anticipe plusieurs
coups. Vous le testez comme un logiciel, puis vous le soumettez sur ML-Arena,
où il affronte les joueurs des autres groupes.

<!-- notes: 20 minutes. Présenter le projet, pas le code : le code vient dans
les quatre labs. Former les binômes et faire choisir un nom de groupe avant la
fin de la séance. Vérifier que chacun a un compte ML-Arena et une clé d'API
(Profil → API Keys) : sans elle, le lab 4 bloque à la soumission. Question
fréquente : « faut-il de l'IA ? » — non, un minimax bien écrit suffit à battre
la plupart des agents ; l'apprentissage par renforcement est hors du projet. -->

---

## Le jeu

![Les quatre directions d'alignement](assets/pym-s8-projet-puissance-4/directions-victoire.png)

- Une grille verticale de **6 lignes × 7 colonnes** ; deux joueurs lâchent à
  tour de rôle un pion, qui tombe dans la case libre la plus basse.
- Le premier qui aligne **quatre pions** gagne : horizontalement,
  verticalement ou dans l'une des deux diagonales.
- Grille pleine sans alignement : partie **nulle**. Une colonne pleine ne peut
  plus être jouée.

Une partie dure au plus 42 coups, et le joueur qui commence a un léger
avantage : c'est pourquoi les places alternent d'une partie à l'autre.

---

## Le jeu en Python : PettingZoo

La bibliothèque [PettingZoo](https://pettingzoo.farama.org/environments/classic/connect_four/)
fournit le jeu (`connect_four_v3`). Une partie est une boucle : `env.last()`
donne ce que voit le joueur dont c'est le tour, `env.step(colonne)` joue.

```python
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env()
env.reset()
observation, reward, terminated, truncated, info = env.last()
env.step(3)          # player_0 joue la colonne 3
```

![Les deux canaux de l'observation](assets/pym-s8-projet-puissance-4/observation-canaux.png)

L'observation est un tableau `(6, 7, 2)` : le canal 0 contient **mes** pions,
le canal 1 ceux de l'adversaire, ligne 0 en haut. L'`action_mask` (7 valeurs)
vaut 1 pour chaque colonne encore jouable.

---

## Le contrat : une classe `Agent`

Sur la plateforme, vous ne manipulez pas l'environnement : vous fournissez un
fichier `agent.py`, et ML-Arena appelle votre classe.

```python
class Agent:
    def setup(self, observation_space, action_space):
        return True                       # une fois, avant le match

    def reset(self, env_player_name, episode_index):
        return True                       # avant chaque partie : la place change

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        ...                               # renvoyer une colonne autorisée (int)
```

Trois règles : jouer une colonne autorisée par `action_mask` ; répondre en
moins de **0,25 s** par appel ; n'importer que la bibliothèque standard et
NumPy. Un plantage, un coup interdit ou un dépassement de temps font perdre le
match.

<!-- notes: Insister sur reset : l'agent ne doit pas supposer qu'il commence.
La vue de l'observation est toujours celle du joueur qui doit jouer, donc le
même code marche aux deux places. Les 0,25 s sont mesurées aller-retour, sur
une machine plus lente qu'un portable récent : viser moins de 0,1 s. -->

---

## Trois niveaux d'agents

1. **Aléatoire** : une colonne autorisée au hasard. La référence à battre.
2. **À règles** : gagner si possible, sinon bloquer l'adversaire, sinon éviter
   le coup qui lui offre la victoire, de préférence au centre.
3. **Recherche** : anticiper plusieurs coups avec **minimax** et l'élagage
   **alpha-bêta**, ou avec une recherche arborescente Monte Carlo (MCTS).

![Un arbre minimax de profondeur 2](assets/pym-s8-projet-puissance-4/arbre-minimax.png)

Minimax développe l'arbre des coups possibles : à mon tour je prends le
**maximum**, au tour adverse le **minimum**. Le coût croît comme $b^d$
(facteur de branchement $b = 7$, profondeur $d$) : la limite de temps décide de
la profondeur.

---

## La plateforme ML-Arena

Le tournoi du cours : [ml-arena.com/viewchallenge/198](https://ml-arena.com/viewchallenge/198).
La page du challenge détaille le contrat, les contraintes et la soumission.

- Un **match** compte 10 parties ; chacune rapporte +1 au gagnant, −1 au
  perdant, 0 en cas de nulle.
- Le classement est un score **ELO** : chaque agent commence à 1200 ; battre
  un adversaire mieux classé rapporte plus de points que battre un plus faible.
- La plateforme organise régulièrement de nouveaux matchs : la note évolue
  après la soumission.

On soumet depuis Colab avec le SDK : `client.submit(challenge_id=198,
files=["agent.py"])`. Le lab 4 le fait pas à pas.

---

## Organisation et livrables

Le projet se fait en **binôme** ; chaque groupe choisit un nom. Les quatre
labs construisent le projet pas à pas :

| lab | contenu |
|---|---|
| 1 | règles, PettingZoo, agent aléatoire, boucle de jeu |
| 2 | fonctions de grille, agent à règles |
| 3 | tests `pytest`, temps, mémoire, tournoi |
| 4 | minimax alpha-bêta, limite de temps, soumission |

**À rendre :**

- un dépôt **GitHub** par groupe : le code découpé en modules (un package
  Python en option), des tests `pytest`, un code propre selon `pylint`, un
  `README.md` qui explique vos choix ;
- l'agent final **soumis sur ML-Arena**.

---

## Évaluation

1. **Exactitude** : l'agent respecte-t-il les règles et joue-t-il toujours un
   coup autorisé ?
2. **Qualité du code** : est-il propre, organisé, documenté ?
3. **Tests** : les tests couvrent-ils les fonctions et l'agent ?
4. **Compréhension** : savez-vous expliquer vos choix de conception ?
5. **Performance** : l'agent tient-il la limite de temps, et quel est son
   classement ?

**À retenir**

- Un agent est une classe à trois méthodes ; `reset` est appelée avant chaque
  partie.
- Toujours jouer une colonne de l'`action_mask`, toujours en moins de 0,25 s.
- Progresser par niveaux : aléatoire, règles, recherche — et tester chaque
  niveau avant le suivant.
