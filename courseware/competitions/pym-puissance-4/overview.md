# Puissance 4

Deux joueurs, une grille verticale de 6 lignes × 7 colonnes. À tour de rôle,
chaque joueur lâche un jeton dans une colonne : le jeton tombe dans la case
libre la plus basse. Le premier qui aligne **quatre jetons** — horizontalement,
verticalement ou en diagonale — gagne. Si les 7 colonnes sont pleines sans
alignement, la partie est nulle.

Vous écrivez un **agent** : une classe Python `Agent` qui choisit la colonne à
jouer. La plateforme le fait jouer contre les agents des autres participants.

| | |
| --- | --- |
| Joueurs par match | 2 |
| Action | un entier de `0` à `6` : la colonne où lâcher le jeton |
| Observation | tableau NumPy de forme `(6, 7, 2)`, valeurs `0` ou `1` |
| Coups autorisés | `action_mask`, 7 valeurs `0` / `1` |
| Parties par match | 10 (les places de premier et second joueur alternent) |
| Classement | ELO |

Environnement utilisé : [PettingZoo Connect Four](https://pettingzoo.farama.org/environments/classic/connect_four/)
(`connect_four_v3`). Vous n'avez pas à manipuler cet environnement : la
plateforme appelle directement votre classe `Agent`.

## Ce que reçoit votre agent

### L'observation `(6, 7, 2)`

- `observation[:, :, 0]` : **vos** jetons (1 = un de vos jetons occupe la case) ;
- `observation[:, :, 1]` : les jetons de **l'adversaire**.

La vue est toujours celle du joueur qui doit jouer : le même code fonctionne
que vous commenciez ou non. La ligne `0` est **en haut** de la grille, la
ligne `5` en bas. Une case vaut `0` dans les deux canaux si elle est vide.

### Le masque `action_mask`

`action_mask[c] == 1` signifie que la colonne `c` n'est pas pleine.
**Jouez toujours une colonne autorisée** : jouer dans une colonne pleine met
fin à la partie, perdue pour vous (−1). Renvoyer autre chose qu'un entier de
`0` à `6` (ou `None` pendant la partie) est une erreur de votre agent : le
match est perdu par forfait.

## Le contrat de la classe `Agent`

Votre fichier s'appelle `agent.py` et contient une classe `Agent` sans
argument dans son constructeur, avec trois méthodes :

1. `setup(observation_space, action_space)` — appelée **une fois** avant la
   première partie. Renvoie `True`.
2. `reset(env_player_name, episode_index)` — appelée **au début de chaque
   partie** (obligatoire). `env_player_name` vaut `"player_0"` (vous commencez)
   ou `"player_1"` : votre place change d'une partie à l'autre. Renvoie `True`.
3. `choose_action(observation, reward, terminated, truncated, info, action_mask)`
   — appelée **à chacun de vos tours**. Renvoie le numéro de colonne (`int`).

### Exemple minimal : un agent aléatoire

```python
import random


class Agent:
    def setup(self, observation_space, action_space):
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        colonnes_libres = [c for c in range(7) if action_mask[c] == 1]
        return random.choice(colonnes_libres)
```

Cet agent respecte les règles mais joue au hasard. Idées pour faire mieux,
de la plus simple à la plus ambitieuse :

- jouer le coup gagnant s'il existe ;
- sinon bloquer le coup gagnant de l'adversaire ;
- préférer les colonnes centrales (elles participent à plus d'alignements) ;
- anticiper plusieurs coups à l'avance (recherche *minimax*).

## Contraintes

- **0,25 seconde** maximum par appel à `choose_action` (temps aller-retour
  mesuré par la plateforme). Au-delà, l'appel est coupé et le match est perdu.
  Un pic ponctuel de 100 à 200 ms peut arriver sur un appel : visez **moins de
  100 ms de calcul par coup**. Un agent à règles simples répond en moins
  d'une milliseconde.
- **360 secondes** maximum pour un match complet (10 parties).
- Ressources de votre agent : **3 cœurs CPU** et **3 Go de mémoire** au
  maximum ; pas de GPU.
- Vous ne pouvez importer que la bibliothèque standard de Python (3.12) et
  les paquets de l'environnement d'exécution choisi pour la soumission ;
  NumPy et pandas sont toujours disponibles.
- Une soumission : au plus 10 fichiers, 100 Mo au total. Au plus
  20 soumissions actives par participant.

## Récompenses et classement ELO

Chaque partie rapporte **+1** au gagnant, **−1** au perdant et **0** aux deux
joueurs en cas de nulle. Un match compte 10 parties ; son score est la moyenne,
entre −1 et 1.

Le classement se fait au **score ELO**. Chaque nouvel agent commence à
**1200**. Après chaque match, le gagnant prend des points au perdant ; le
nombre de points échangés dépend de l'écart entre les deux notes : battre un
adversaire mieux classé rapporte beaucoup, battre un adversaire plus faible
rapporte peu. La plateforme organise régulièrement de nouveaux matchs entre
agents de niveaux proches : votre note continue d'évoluer après votre
soumission.

Si votre agent plante, dépasse le temps ou renvoie une action invalide, le
match est arrêté et compté comme une défaite pour vous ; votre adversaire
n'est pas pénalisé.

## Soumettre depuis Google Colab

1. Sur ml-arena.com, ouvrez votre **Profil → API Keys** et copiez votre clé
   (elle commence par `mlk_`).
2. Dans Colab, installez le SDK et connectez-vous :

```python
!pip install -q "mlarena-sdk==3.0.0"

import mlarena

API_TOKEN = "mlk_user_A_REMPLACER"   # votre clé personnelle
CHALLENGE_ID = __CHALLENGE_ID__

client = mlarena.connect(api_key=API_TOKEN, base_url="https://ml-arena.com")
```

3. Écrivez votre agent dans un fichier `agent.py` (la cellule commence par
   `%%writefile agent.py`, suivie du code de votre classe `Agent`).
4. Envoyez-le :

```python
resultat = client.submit(challenge_id=CHALLENGE_ID, files=["agent.py"],
                         submission_name="mon-agent")
print(resultat)
```

5. Suivez votre soumission dans l'onglet **Submissions** du challenge, puis
   votre place dans l'onglet **Leaderboard**.
