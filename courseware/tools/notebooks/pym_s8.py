"""Séance 8 — Projet Puissance 4: the four lab notebooks of `python-medecine`.

The module `pym-s8-projet-puissance-4` has one slide lesson (the project) and
four exercise pages, one per lab. Each lab has a Colab notebook built here, in
a student version (`pym-s8-lab-N.ipynb`, "À vous" skeletons) and a correction
(`pym-s8-lab-N-correction.ipynb`), from ONE source: `ex(todo, solution)` emits
the skeleton in the first and the solution in the second.

- Lab 1: the PettingZoo env, the observation, a random game, the ML-Arena
  agent contract, `jouer_partie` (the local game loop) and `jouer_match`.
- Lab 2: the 6 × 7 grid, `coups_valides`, `ligne_libre`, `simuler_coup`, the
  69 windows of four, `est_gagnant`, `coup_gagnant`, the rule-based agent.
- Lab 3: tests (assert, mutants, scenarios, pytest, pylint), timing, memory,
  the tournament, confidence intervals.
- Lab 4: heuristic evaluation, minimax with alpha-beta, iterative deepening
  under a time budget, `agent.py`, the submission, MCTS (optional).

Each later lab starts with the corrected code of the earlier ones, from the
same strings (`L1_CODE`, `L2_CODE`, ...), so the four notebooks never drift.

The game loop `jouer_partie` calls the agent exactly as flexkit's
`run_pettingzoo_eval` does on ML-Arena (`workers/flex_v1/flexkit/aec_loop.py`
in the platform repo): `setup` once per match, `reset(env_player_name,
episode_index)` before every game, `choose_action(observation, reward, False,
False, info, action_mask=...)` on every live turn, `env.step(None)` for a
finished player. The platform draws the seats at random each game; the local
loop alternates them, which is the same in expectation and fairer over few
games. The challenge (copy of challenge 65) allows 0.25 s per call, 360 s per
match of 10 games: the final agent searches for `TEMPS_RECHERCHE` = 0.08 s.

Writes eight notebooks and nothing else — unlike `build_notebooks.py`, running
this script is safe:

    uv run python courseware/tools/notebooks/pym_s8.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "website" / "public" / "modules" / "python-medecine" / "notebooks"
COLAB = (f"https://colab.research.google.com/github/{nb.GITHUB}/blob/"
         f"{nb.BRANCH}/website/public/modules/python-medecine/notebooks/")
ASSETS = (f"https://raw.githubusercontent.com/{nb.GITHUB}/{nb.BRANCH}/"
          "courseware/content/python-medecine/assets/pym-s8-projet-puissance-4/")
CHALLENGE_ID = 198  # ML-Arena challenge "Puissance 4" (copy of challenge 65)
CHALLENGE_URL = f"https://ml-arena.com/viewchallenge/{CHALLENGE_ID}"
TEMPS_MAX = 0.25    # s per choose_action call on the platform
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

CORRECTION = False  # set by build()


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines()


def m(text: str) -> dict:
    return nb.md(*_lines(text))


def c(text: str) -> dict:
    return nb.code(*_lines(text))


def ex(todo: str, solution: str) -> dict:
    """The skeleton in the student notebook, the solution in the correction."""
    return c(solution if CORRECTION else todo)


def sol_md(text: str) -> list[dict]:
    """A markdown answer shown only in the correction."""
    return [m(text)] if CORRECTION else []


def header(n: int, title: str, objectifs: str) -> list[dict]:
    name = f"pym-s8-lab-{n}" + ("-correction" if CORRECTION else "")
    suffix = " (correction)" if CORRECTION else ""
    return [
        m(f"""
        # Lab {n} — {title}{suffix}

        [![Open In Colab]({BADGE})]({COLAB}{name}.ipynb)

        *Python pour la médecine — Séance 8, projet Puissance 4.*

        {objectifs}
        """),
        m("""
        **Comment travailler.** Exécutez les cellules dans l'ordre. Une cellule
        marquée `# À vous` contient un squelette (`...`) à compléter ; la
        cellule suivante vérifie votre réponse et affiche « ✅ Exercice N
        réussi ». Les cellules « Lire du code » demandent d'écrire votre
        prédiction **avant** d'exécuter.
        """),
    ]


SETUP = """
import subprocess
import sys

try:
    import pettingzoo
except ImportError:  # Colab : PettingZoo n'est pas préinstallé
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pettingzoo[classic]"], check=True)

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.classic import connect_four_v3

random.seed(0)
print("PettingZoo prêt")
"""


def setup_cells() -> list[dict]:
    return [m("## 0. Installation"), c(SETUP)]


def previous(code: str, labs: str) -> list[dict]:
    return [
        m(f"""
        ### Le code des labs précédents

        La cellule suivante reprend la correction des {labs} : exécutez-la
        sans la modifier (ou remplacez-la par votre propre code).
        """),
        c(code),
    ]


# =========================================================================
# Corrected code shared between notebooks
# =========================================================================

L1_AFFICHER = '''
def afficher_plateau(observation):
    """Le plateau en texte : X = mes pions, O = l'adversaire, . = vide."""
    obs = np.asarray(observation)
    lignes = []
    for ligne in range(6):
        cases = []
        for col in range(7):
            if obs[ligne, col, 0] == 1:
                cases.append("X")
            elif obs[ligne, col, 1] == 1:
                cases.append("O")
            else:
                cases.append(".")
        lignes.append(" ".join(cases))
    return "\\n".join(lignes)
'''

L1_COLONNES = '''
def colonnes_libres(action_mask):
    """Les indices des colonnes autorisées (masque à 1)."""
    return [col for col, autorise in enumerate(action_mask) if autorise]
'''

L1_PARTIE_ALEATOIRE = '''
def partie_aleatoire(seed=None):
    """Une partie entre deux joueurs au hasard : (gagnant ou None, nb de coups)."""
    env = connect_four_v3.env()
    env.reset(seed=seed)
    gagnant, nb_coups = None, 0
    for nom in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            if reward == 1:
                gagnant = nom
            action = None
        else:
            action = env.action_space(nom).sample(observation["action_mask"])
            nb_coups += 1
        env.step(action)
    env.close()
    return gagnant, nb_coups
'''

L1_ALEATOIRE = '''
class AgentAleatoire:
    """Joue une colonne autorisée au hasard (le contrat ML-Arena)."""

    def setup(self, observation_space, action_space):
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        self.episode_index = episode_index
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return random.choice(colonnes_libres(action_mask))
'''

L1_JOUER = '''
def jouer_partie(agent_a, agent_b, a_commence=True, index=0):
    """Une partie appelée comme sur ML-Arena : (résultat de agent_a, nb de coups).

    Le résultat vaut +1 (agent_a gagne), 0 (nulle) ou -1 (agent_a perd).
    """
    env = connect_four_v3.env()
    env.reset()
    if a_commence:
        places = {"player_0": agent_a, "player_1": agent_b}
    else:
        places = {"player_0": agent_b, "player_1": agent_a}
    for nom, agent in places.items():
        agent.reset(nom, index)
    gains = {nom: 0.0 for nom in env.possible_agents}
    nb_coups = 0
    for nom in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)  # un joueur dont la partie est finie sort ainsi
            continue
        action = places[nom].choose_action(
            observation["observation"], float(reward), False, False, info,
            action_mask=observation["action_mask"])
        env.step(action)
        nb_coups += 1
        for joueur, gain in env.rewards.items():
            gains[joueur] += gain
    env.close()
    nom_a = "player_0" if a_commence else "player_1"
    return gains[nom_a], nb_coups
'''

L1_MATCH = '''
def jouer_match(agent_a, agent_b, n_parties=100):
    """n parties, agent_a commence une partie sur deux. Bilan vu par agent_a."""
    env = connect_four_v3.env()
    for agent in (agent_a, agent_b):  # une fois par match, comme sur ML-Arena
        agent.setup(env.observation_space("player_0"),
                    env.action_space("player_0"))
    bilan = {"victoires": 0, "nuls": 0, "defaites": 0, "coups": []}
    for i in range(n_parties):
        resultat, nb_coups = jouer_partie(agent_a, agent_b,
                                          a_commence=(i % 2 == 0), index=i)
        if resultat > 0:
            bilan["victoires"] += 1
        elif resultat < 0:
            bilan["defaites"] += 1
        else:
            bilan["nuls"] += 1
        bilan["coups"].append(nb_coups)
    return bilan
'''

L1_PONDERE = '''
class AgentPondere(AgentAleatoire):
    """Au hasard, mais les colonnes du centre sont plus probables."""

    POIDS = [1, 2, 3, 4, 3, 2, 1]

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        colonnes = colonnes_libres(action_mask)
        poids = [self.POIDS[col] for col in colonnes]
        return random.choices(colonnes, weights=poids)[0]
'''

L1_CODE = "\n\n".join(s.strip("\n") for s in (
    L1_AFFICHER, L1_COLONNES, L1_ALEATOIRE, L1_PONDERE, L1_JOUER, L1_MATCH))

# ---------------------------------------------------------------- Lab 2
L2_CONSTANTES = '''
LIGNES, COLONNES = 6, 7
VIDE, MOI, ADV = 0, 1, 2
'''

L2_VERS_GRILLE = '''
def vers_grille(observation):
    """(6, 7, 2) -> grille 6 x 7 : 0 = vide, 1 = moi, 2 = l'adversaire."""
    obs = np.asarray(observation)
    return (obs[:, :, 0] + 2 * obs[:, :, 1]).astype(int)
'''

L2_OUTILS = '''
def grille_depuis_texte(texte):
    """Une grille depuis 6 lignes de texte : X = moi, O = l'adversaire."""
    symboles = {".": VIDE, "X": MOI, "O": ADV}
    lignes = [l.split() for l in texte.strip().splitlines()]
    return np.array([[symboles[s] for s in l] for l in lignes], dtype=int)


def observation_depuis_grille(grille):
    """L'inverse de vers_grille : l'observation (6, 7, 2) que reçoit l'agent."""
    return np.stack([grille == MOI, grille == ADV], axis=-1).astype(np.int8)


def masque_depuis_grille(grille):
    """Le masque des colonnes non pleines, comme action_mask."""
    return (grille[0] == VIDE).astype(np.int8)
'''

L2_COUPS = '''
def coups_valides(grille):
    """Les colonnes dont la case du haut est vide."""
    return [col for col in range(COLONNES) if grille[0, col] == VIDE]
'''

L2_LIGNE_LIBRE = '''
def ligne_libre(grille, col):
    """La ligne où tombe un pion lâché dans col, ou None si elle est pleine."""
    for ligne in range(LIGNES - 1, -1, -1):  # du bas (5) vers le haut (0)
        if grille[ligne, col] == VIDE:
            return ligne
    return None
'''

L2_SIMULER = '''
def simuler_coup(grille, col, joueur):
    """Une COPIE de la grille où joueur a joué dans col."""
    nouvelle = grille.copy()
    nouvelle[ligne_libre(grille, col), col] = joueur
    return nouvelle
'''

L2_FENETRES = '''
def construire_fenetres():
    """Toutes les suites de 4 cases alignées sur la grille 6 x 7."""
    fenetres = []
    for ligne in range(LIGNES):
        for col in range(COLONNES):
            for dl, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                cases = [(ligne + k * dl, col + k * dc) for k in range(4)]
                if all(0 <= l < LIGNES and 0 <= c < COLONNES for l, c in cases):
                    fenetres.append(cases)
    return fenetres


FENETRES = construire_fenetres()
'''

L2_GAGNANT = '''
def est_gagnant(grille, joueur):
    """Vrai si joueur a quatre pions alignés quelque part."""
    for fenetre in FENETRES:
        if all(grille[l, c] == joueur for l, c in fenetre):
            return True
    return False
'''

L2_COUP_GAGNANT = '''
def coup_gagnant(grille, joueur):
    """Une colonne qui fait gagner joueur tout de suite, sinon None."""
    for col in coups_valides(grille):
        if est_gagnant(simuler_coup(grille, col, joueur), joueur):
            return col
    return None
'''

L2_SURS = '''
def coups_surs(grille, coups):
    """Les coups après lesquels l'adversaire ne peut pas gagner tout de suite."""
    return [col for col in coups
            if coup_gagnant(simuler_coup(grille, col, MOI), ADV) is None]
'''

L2_REGLES = '''
class AgentRegles(AgentAleatoire):
    """Gagner, sinon bloquer, sinon un coup sûr de préférence au centre."""

    POIDS = [1, 2, 3, 4, 3, 2, 1]

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        grille = vers_grille(observation)
        coups = colonnes_libres(action_mask)
        for joueur in (MOI, ADV):  # 1. gagner, 2. bloquer
            col = coup_gagnant(grille, joueur)
            if col is not None:
                return col
        candidats = coups_surs(grille, coups) or coups  # 3. ne pas offrir
        poids = [self.POIDS[col] for col in candidats]  # 4. le centre
        return random.choices(candidats, weights=poids)[0]
'''

L2_CODE = "\n\n".join(s.strip("\n") for s in (
    L2_CONSTANTES, L2_VERS_GRILLE, L2_OUTILS, L2_COUPS, L2_LIGNE_LIBRE,
    L2_SIMULER, L2_FENETRES, L2_GAGNANT, L2_COUP_GAGNANT, L2_SURS,
    L2_REGLES))

# ---------------------------------------------------------------- Lab 3
L3_CHRONO = '''
class AgentChronometre:
    """Enveloppe un agent et mesure la durée de chaque choose_action."""

    def __init__(self, agent):
        self.agent = agent
        self.durees = []

    def setup(self, observation_space, action_space):
        return self.agent.setup(observation_space, action_space)

    def reset(self, env_player_name, episode_index):
        return self.agent.reset(env_player_name, episode_index)

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        debut = time.perf_counter()
        action = self.agent.choose_action(observation, reward, terminated,
                                          truncated, info, action_mask)
        self.durees.append(time.perf_counter() - debut)
        return action
'''

L3_TOURNOI = '''
def tournoi(agents, n_parties=50):
    """Chaque paire d'agents joue un match ; taux de victoire ligne / colonne."""
    noms = list(agents)
    taux = pd.DataFrame(np.nan, index=noms, columns=noms)
    for a, b in itertools.combinations(noms, 2):
        bilan = jouer_match(agents[a], agents[b], n_parties)
        taux.loc[a, b] = bilan["victoires"] / n_parties
        taux.loc[b, a] = bilan["defaites"] / n_parties
    return taux
'''

L3_CODE = "import itertools\n\nimport pandas as pd\n\n" + "\n\n".join(
    s.strip("\n") for s in (L3_CHRONO, L3_TOURNOI))

# ---------------------------------------------------------------- Lab 4
L4_SCORE = '''
def score_fenetre(valeurs):
    """Le score d'une fenêtre de 4 cases, de mon point de vue."""
    moi, adv = valeurs.count(MOI), valeurs.count(ADV)
    if moi and adv:      # fenêtre mixte : plus personne ne peut y aligner 4
        return 0
    if moi == 3:
        return 5
    if moi == 2:
        return 2
    if adv == 3:
        return -4
    if adv == 2:
        return -2
    return 0
'''

L4_EVALUER = '''
def evaluer(grille):
    """Somme des scores des 69 fenêtres + bonus pour la colonne centrale."""
    score = 0
    for fenetre in FENETRES:
        score += score_fenetre([grille[l, c] for l, c in fenetre])
    centre = list(grille[:, COLONNES // 2])
    return score + 3 * centre.count(MOI) - 3 * centre.count(ADV)
'''

L4_MINIMAX = '''
GAGNE = 1_000_000


class TempsEcoule(Exception):
    """Levée quand la recherche dépasse son budget de temps."""


def minimax(grille, profondeur, alpha, beta, maximise, fin=float("inf")):
    """La valeur de la position, MOI cherchant le max et ADV le min."""
    if time.perf_counter() > fin:
        raise TempsEcoule
    if est_gagnant(grille, MOI):
        return GAGNE + profondeur  # gagner tôt vaut mieux que gagner tard
    if est_gagnant(grille, ADV):
        return -GAGNE - profondeur
    coups = coups_valides(grille)
    if not coups:
        return 0                   # grille pleine : nulle
    if profondeur == 0:
        return evaluer(grille)
    coups.sort(key=lambda col: abs(col - 3))  # le centre d'abord : plus d'élagage
    if maximise:
        valeur = -float("inf")
        for col in coups:
            valeur = max(valeur, minimax(simuler_coup(grille, col, MOI),
                                         profondeur - 1, alpha, beta, False, fin))
            alpha = max(alpha, valeur)
            if alpha >= beta:
                break              # ADV ne laissera jamais arriver ici
        return valeur
    valeur = float("inf")
    for col in coups:
        valeur = min(valeur, minimax(simuler_coup(grille, col, ADV),
                                     profondeur - 1, alpha, beta, True, fin))
        beta = min(beta, valeur)
        if alpha >= beta:
            break                  # MOI ne laissera jamais arriver ici
    return valeur
'''

L4_MEILLEUR = '''
def meilleur_coup(grille, coups, profondeur, fin=float("inf")):
    """(colonne, valeur) du meilleur coup, en essayant les coups dans l'ordre."""
    meilleur, alpha = coups[0], -float("inf")
    for col in coups:
        valeur = minimax(simuler_coup(grille, col, MOI), profondeur - 1,
                         alpha, float("inf"), False, fin)
        if valeur > alpha:
            meilleur, alpha = col, valeur
    return meilleur, alpha
'''

L4_AGENT_MINIMAX = '''
class AgentMinimax(AgentAleatoire):
    """Minimax alpha-bêta à profondeur fixe."""

    def __init__(self, profondeur=4):
        self.profondeur = profondeur

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        grille = vers_grille(observation)
        coups = colonnes_libres(action_mask)
        # centre d'abord, hasard entre colonnes à même distance du centre
        coups.sort(key=lambda col: (abs(col - 3), random.random()))
        return meilleur_coup(grille, coups, self.profondeur)[0]
'''

L4_CHOISIR = '''
def choisir_coup(grille, coups, temps_max):
    """Approfondissement itératif : profondeur 1, 2, 3... tant qu'il reste du temps."""
    fin = time.perf_counter() + temps_max
    ordre = sorted(coups, key=lambda col: (abs(col - 3), random.random()))
    choix = ordre[0]
    cases_vides = int((grille == VIDE).sum())
    for profondeur in range(1, cases_vides + 1):
        try:
            choix, valeur = meilleur_coup(grille, ordre, profondeur, fin)
        except TempsEcoule:
            break          # on garde le coup de la dernière profondeur finie
        ordre = [choix] + [col for col in ordre if col != choix]
        if abs(valeur) >= GAGNE:
            break          # issue forcée trouvée : inutile de chercher plus loin
    return choix
'''

L4_CODE = "\n\n".join(s.strip("\n") for s in (
    L4_SCORE, L4_EVALUER, L4_MINIMAX, L4_MEILLEUR, L4_AGENT_MINIMAX,
    L4_CHOISIR))


def _agent_py(correction: bool) -> str:
    """The text of agent.py: self-contained, numpy + standard library only."""
    head = f'''
%%writefile agent.py
"""Agent Puissance 4 pour ML-Arena (challenge {CHALLENGE_ID}).

Minimax alpha-bêta, approfondissement itératif sous budget de temps.
Bibliothèque standard + NumPy uniquement : pas de PettingZoo sur la plateforme.
"""
import random
import time

import numpy as np

TEMPS_RECHERCHE = 0.08  # s de calcul par coup ; la plateforme coupe à {TEMPS_MAX} s
'''
    helpers = "\n\n".join(s.strip("\n") for s in (
        L2_CONSTANTES, L2_VERS_GRILLE, L2_COUPS, L2_LIGNE_LIBRE, L2_SIMULER,
        L2_FENETRES, L2_GAGNANT))
    if correction:
        search = "\n\n".join(s.strip("\n") for s in (
            L4_SCORE, L4_EVALUER, L4_MINIMAX, L4_MEILLEUR, L4_CHOISIR))
        decide = "return int(choisir_coup(grille, coups, TEMPS_RECHERCHE))"
    else:
        search = ("# À vous : collez ici score_fenetre, evaluer, GAGNE, TempsEcoule,\n"
                  "# minimax, meilleur_coup et choisir_coup (exercices 1 à 6).")
        decide = ("# À vous : remplacez le hasard par\n"
                  "        # return int(choisir_coup(grille, coups, TEMPS_RECHERCHE))\n"
                  "        return random.choice(coups)")
    agent = f'''
class Agent:
    """Le contrat ML-Arena : setup une fois, reset à chaque partie, choose_action."""

    def setup(self, observation_space, action_space):
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        self.episode_index = episode_index
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        grille = vers_grille(observation)
        masque = np.asarray(action_mask)
        coups = [col for col in range(COLONNES) if masque[col]]
        {decide}
'''
    return "\n\n\n".join(textwrap.dedent(s).strip("\n")
                         for s in (head, helpers, search, agent)) + "\n"


# =========================================================================
# Lab 1
# =========================================================================

def lab1() -> list[dict]:
    cells = header(1, "Comprendre le jeu et jouer au hasard", """
        **Objectifs.** Comprendre les règles du Puissance 4 et la façon dont
        PettingZoo représente une partie ; jouer une partie au hasard ; écrire
        un premier agent qui respecte le contrat de ML-Arena ; écrire la boucle
        `jouer_partie` qui fait s'affronter deux agents **exactement comme la
        plateforme les appelle** ; mesurer ce que donne le hasard sur 200
        parties.
        """)
    cells += setup_cells()

    cells += [
        m("""
        ---

        ## 1. Les règles

        Deux joueurs lâchent à tour de rôle un pion dans l'une des 7 colonnes
        d'une grille verticale de 6 lignes ; le pion tombe dans la case libre
        la plus basse. Le premier qui aligne quatre pions (horizontalement,
        verticalement ou en diagonale) gagne ; si la grille est pleine sans
        alignement, la partie est nulle. Une colonne pleine ne peut plus être
        jouée.

        ### Exercice 1 — Les nombres du jeu

        Complétez les constantes. Pour `NB_DIRECTIONS`, comptez les directions
        d'alignement différentes (une diagonale dans un sens ou dans l'autre
        compte pour une direction chacune ; « gauche » et « droite » sont la
        même direction horizontale).
        """),
        ex("""
        # À vous
        N_LIGNES = ...
        N_COLONNES = ...
        NB_DIRECTIONS = ...   # directions d'alignement à vérifier
        NB_COUPS_MAX = ...    # nombre maximal de coups dans une partie
        """, """
        N_LIGNES = 6
        N_COLONNES = 7
        NB_DIRECTIONS = 4     # horizontale, verticale, deux diagonales
        NB_COUPS_MAX = 42     # 6 x 7 cases, un pion par coup
        """),
        c("""
        assert (N_LIGNES, N_COLONNES) == (6, 7), "la grille fait 6 lignes et 7 colonnes"
        assert NB_DIRECTIONS == 4, "horizontale, verticale, et deux diagonales"
        assert NB_COUPS_MAX == N_LIGNES * N_COLONNES
        print("✅ Exercice 1 réussi")
        """),
        m("""
        ---

        ## 2. L'environnement PettingZoo

        [PettingZoo](https://pettingzoo.farama.org/environments/classic/connect_four/)
        fournit le jeu sous le nom `connect_four_v3`. Les deux joueurs
        s'appellent `player_0` (il commence) et `player_1`. Le schéma est
        toujours le même : `env.last()` donne ce que voit le joueur dont c'est
        le tour, `env.step(action)` joue son coup.
        """),
        c("""
        env = connect_four_v3.env()
        env.reset(seed=42)
        print("joueurs         :", env.agents)
        print("à qui de jouer  :", env.agent_selection)
        print("actions         :", env.action_space("player_0"))
        print("observations    :", env.observation_space("player_0"))
        """),
        m("""
        L'action est un entier de 0 à 6 : la colonne. L'observation est un
        dictionnaire à deux clés : `"observation"`, un tableau de forme
        `(6, 7, 2)`, et `"action_mask"`, 7 valeurs qui valent 1 pour chaque
        colonne encore jouable.

        ### Exercice 2 — Lire `env.last()`

        `env.last()` renvoie cinq valeurs. Récupérez la forme du tableau
        d'observation, le masque et le nom du joueur qui doit jouer.
        """),
        ex("""
        # À vous
        observation, reward, terminated, truncated, info = env.last()
        forme = ...    # la forme de observation["observation"]
        masque = ...   # le masque des coups autorisés
        joueur = ...   # le nom du joueur dont c'est le tour (env.agent_selection)
        print(forme, masque, joueur)
        """, """
        observation, reward, terminated, truncated, info = env.last()
        forme = observation["observation"].shape
        masque = observation["action_mask"]
        joueur = env.agent_selection
        print(forme, masque, joueur)
        """),
        c("""
        assert tuple(forme) == (6, 7, 2)
        assert list(masque) == [1] * 7, "au début, les 7 colonnes sont jouables"
        assert joueur == "player_0"
        print("✅ Exercice 2 réussi")
        """),
        m(f"""
        ### L'observation : deux canaux

        ![Les deux canaux de l'observation]({ASSETS}observation-canaux.png)

        `observation[:, :, 0]` contient **mes** pions (1 = occupé par moi),
        `observation[:, :, 1]` ceux de **l'adversaire**. La ligne 0 est en
        **haut**, la ligne 5 en bas. Surtout, la vue est toujours celle du
        joueur qui doit jouer : le même code marche que l'on commence ou non.

        ### Lire du code 1

        Que va afficher ce code ? Écrivez votre réponse **avant** d'exécuter.
        """),
        c("""
        env.reset(seed=42)
        env.step(3)        # player_0 joue la colonne 3
        env.step(3)        # player_1 joue aussi la colonne 3
        obs = env.last()[0]["observation"]
        print(env.agent_selection)
        print(obs[5, 3, 0], obs[4, 3, 0], obs[4, 3, 1])
        """),
        *sol_md("""
        **Réponse.** `player_0`, puis `1 0 1`. C'est de nouveau à `player_0`,
        donc la vue est la sienne : son pion est en bas de la colonne 3
        (`[5, 3, 0] == 1`), le pion adverse est juste au-dessus, dans le canal 1
        (`[4, 3, 1] == 1`), et rien de « à moi » en ligne 4.
        """),
        m("""
        PettingZoo sait aussi dessiner la grille : avec
        `render_mode="rgb_array"`, `env.render()` renvoie une image que
        matplotlib affiche.
        """),
        c("""
        env_image = connect_four_v3.env(render_mode="rgb_array")
        env_image.reset(seed=42)
        for col in [3, 3, 4, 2]:
            env_image.step(col)
        plt.imshow(env_image.render())
        plt.axis("off")
        plt.title("Après 4 coups : 3, 3, 4, 2")
        plt.show()
        env_image.close()
        """),
        m("""
        ### Exercice 3 — Afficher le plateau en texte

        Écrivez `afficher_plateau(observation)` qui renvoie une **chaîne** de
        6 lignes, une par ligne de la grille (de haut en bas), les 7 cases
        séparées par une espace : `X` pour mes pions, `O` pour ceux de
        l'adversaire, `.` pour une case vide. Commencez par
        `obs = np.asarray(observation)` : sur la plateforme, l'observation
        peut arriver sous forme de listes imbriquées.
        """),
        ex('''
        # À vous
        def afficher_plateau(observation):
            """Le plateau en texte : X = mes pions, O = l'adversaire, . = vide."""
            obs = np.asarray(observation)
            ...


        print(afficher_plateau(obs))
        ''', L1_AFFICHER + "\n\nprint(afficher_plateau(obs))"),
        c("""
        essai = np.zeros((6, 7, 2), dtype=int)
        essai[5, 3, 0] = 1
        essai[4, 3, 1] = 1
        texte = afficher_plateau(essai)
        assert isinstance(texte, str), "renvoyez une chaîne (return), pas un print"
        lignes = texte.splitlines()
        assert len(lignes) == 6
        assert lignes[0] == ". . . . . . ."
        assert lignes[4] == ". . . O . . ."
        assert lignes[5] == ". . . X . . ."
        assert afficher_plateau(essai.tolist()) == texte, "listes imbriquées acceptées"
        print("✅ Exercice 3 réussi")
        """),
        m("""
        ---

        ## 3. Les coups autorisés

        ### Trouvez l'erreur

        Cette fonction doit renvoyer les **numéros** des colonnes jouables.
        Exécutez-la : que renvoie-t-elle ? Pourquoi ?
        """),
        c("""
        def colonnes_libres_bug(action_mask):
            return [autorise for autorise in action_mask if autorise]

        print(colonnes_libres_bug([0, 1, 0, 1, 0, 1, 0]))
        """),
        *sol_md("""
        **Réponse.** `[1, 1, 1]` : la fonction garde les **valeurs** du masque
        (des 1), pas leurs **indices**. Un agent qui l'utilise jouerait
        toujours la colonne 1, même pleine. `enumerate` donne l'indice et la
        valeur.
        """),
        m("""
        ### Exercice 4 — `colonnes_libres`

        Écrivez la version correcte, avec `enumerate`.
        """),
        ex('''
        # À vous
        def colonnes_libres(action_mask):
            """Les indices des colonnes autorisées (masque à 1)."""
            ...
        ''', L1_COLONNES),
        c("""
        assert colonnes_libres([1, 1, 1, 1, 1, 1, 1]) == [0, 1, 2, 3, 4, 5, 6]
        assert colonnes_libres([0, 1, 0, 1, 0, 1, 0]) == [1, 3, 5]
        assert colonnes_libres(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int8)) == [3]
        print("✅ Exercice 4 réussi")
        """),
        m("""
        ### Exercice 5 — Une partie au hasard

        Complétez la boucle de jeu de PettingZoo. `env.agent_iter()` donne à
        chaque tour le nom du joueur dont c'est le tour. Si sa partie est finie
        (`terminated` ou `truncated`), il doit jouer `None` pour sortir ; et
        s'il a reçu `reward == 1`, c'est lui le gagnant. Sinon, il tire un coup
        autorisé avec `env.action_space(nom).sample(masque)`.

        La fonction renvoie `(gagnant, nb_coups)`, `gagnant` valant `None` en
        cas de nulle.
        """),
        ex('''
        # À vous
        def partie_aleatoire(seed=None):
            """Une partie entre deux joueurs au hasard : (gagnant ou None, nb de coups)."""
            env = connect_four_v3.env()
            env.reset(seed=seed)
            gagnant, nb_coups = None, 0
            for nom in env.agent_iter():
                observation, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    ...            # gagnant ? puis action = None
                    action = None
                else:
                    action = ...   # un coup autorisé au hasard
                    nb_coups += 1
                env.step(action)
            env.close()
            return gagnant, nb_coups
        ''', L1_PARTIE_ALEATOIRE),
        c("""
        for graine in range(20):
            gagnant, nb_coups = partie_aleatoire(seed=graine)
            assert gagnant in ("player_0", "player_1", None)
            assert 7 <= nb_coups <= 42, "au moins 7 coups pour aligner 4 pions"
            if gagnant is None:
                assert nb_coups == 42, "une nulle remplit la grille"
        print("exemple :", partie_aleatoire(seed=1))
        print("✅ Exercice 5 réussi")
        """),
        m(f"""
        ---

        ## 4. Le contrat d'agent de ML-Arena

        Sur [ML-Arena]({CHALLENGE_URL}), vous ne manipulez pas
        l'environnement : la plateforme appelle **votre classe `Agent`**
        (fichier `agent.py`). Elle la construit sans argument, puis :

        | appel | quand | renvoie |
        |---|---|---|
        | `setup(observation_space, action_space)` | une fois, avant le match | `True` |
        | `reset(env_player_name, episode_index)` | avant **chaque** partie : votre place (`player_0` ou `player_1`) change | `True` |
        | `choose_action(observation, reward, terminated, truncated, info, action_mask)` | à chacun de vos tours | une colonne autorisée (`int`) |

        `observation` est le tableau `(6, 7, 2)` (déjà extrait du
        dictionnaire) et `action_mask` le masque. Si `terminated` ou
        `truncated` est vrai, renvoyez `None`. La plateforme coupe un appel qui
        dure plus de **{TEMPS_MAX} s** : le match est alors perdu.

        ### Exercice 6 — `AgentAleatoire`

        Écrivez la classe : `setup` et `reset` renvoient `True` (`reset`
        retient sa place dans `self.env_player_name`), `choose_action` joue
        une colonne autorisée au hasard avec `random.choice` et
        `colonnes_libres`.
        """),
        ex('''
        # À vous
        class AgentAleatoire:
            """Joue une colonne autorisée au hasard (le contrat ML-Arena)."""

            def setup(self, observation_space, action_space):
                ...

            def reset(self, env_player_name, episode_index):
                ...

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                ...
        ''', L1_ALEATOIRE),
        c("""
        agent = AgentAleatoire()
        assert agent.setup(None, None) is True
        assert agent.reset("player_1", 0) is True
        assert agent.env_player_name == "player_1"
        vide = np.zeros((6, 7, 2), dtype=np.int8)
        for _ in range(500):
            masque = np.random.randint(0, 2, size=7)
            masque[np.random.randint(7)] = 1
            action = agent.choose_action(vide, action_mask=masque)
            assert masque[action] == 1, f"colonne {action} interdite par {masque}"
        assert agent.choose_action(vide, terminated=True, action_mask=masque) is None
        print("✅ Exercice 6 réussi")
        """),
        m("""
        ---

        ## 5. Faire jouer deux agents

        ### Exercice 7 — `jouer_partie`

        Voici la boucle qui fait jouer deux agents **comme ML-Arena les
        appelle**. Complétez les trois `...` :

        1. `places` associe `"player_0"` à l'agent qui commence et
           `"player_1"` à l'autre (selon `a_commence`) ;
        2. avant la partie, chaque agent reçoit `reset(nom_de_sa_place, index)` ;
        3. à chaque tour, l'agent de la place `nom` choisit son coup :
           `choose_action(observation["observation"], float(reward), False,
           False, info, action_mask=observation["action_mask"])`.

        La fonction renvoie le résultat **de `agent_a`** (+1, 0 ou −1) et le
        nombre de coups.
        """),
        ex('''
        # À vous
        def jouer_partie(agent_a, agent_b, a_commence=True, index=0):
            """Une partie appelée comme sur ML-Arena : (résultat de agent_a, nb de coups)."""
            env = connect_four_v3.env()
            env.reset()
            places = ...   # {"player_0": ..., "player_1": ...}
            ...            # reset de chaque agent
            gains = {nom: 0.0 for nom in env.possible_agents}
            nb_coups = 0
            for nom in env.agent_iter():
                observation, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                action = ...   # le coup de l'agent de la place nom
                env.step(action)
                nb_coups += 1
                for joueur, gain in env.rewards.items():
                    gains[joueur] += gain
            env.close()
            nom_a = "player_0" if a_commence else "player_1"
            return gains[nom_a], nb_coups
        ''', L1_JOUER),
        c("""
        class _Espion(AgentAleatoire):
            def reset(self, env_player_name, episode_index):
                self.places = getattr(self, "places", []) + [env_player_name]
                return super().reset(env_player_name, episode_index)

        espion = _Espion()
        for i in range(10):
            resultat, nb_coups = jouer_partie(espion, AgentAleatoire(),
                                              a_commence=(i % 2 == 0), index=i)
            assert resultat in (-1, 0, 1)
            assert 7 <= nb_coups <= 42
        assert espion.places == ["player_0", "player_1"] * 5, "reset avant chaque partie"
        print("✅ Exercice 7 réussi")
        """),
        m("""
        ### Lire du code 2

        `Espion` hérite d'`AgentAleatoire` et affiche chaque appel à `reset`.
        Que va afficher ce code ?
        """),
        c("""
        class Espion(AgentAleatoire):
            def reset(self, env_player_name, episode_index):
                print("reset", env_player_name, episode_index)
                return super().reset(env_player_name, episode_index)

        espion = Espion()
        for i in range(3):
            jouer_partie(espion, AgentAleatoire(), a_commence=(i % 2 == 0), index=i)
        """),
        *sol_md("""
        **Réponse.** `reset player_0 0`, `reset player_1 1`, `reset player_0 2`.
        L'espion commence une partie sur deux. Sur ML-Arena aussi la place
        change d'une partie à l'autre : un agent qui retiendrait sa place une
        fois pour toutes dans `setup` jouerait avec la mauvaise.
        """),
        m("""
        ### Exercice 8 — `jouer_match`

        Écrivez `jouer_match(agent_a, agent_b, n_parties)` : appelle `setup`
        une fois sur chaque agent (avec les espaces de `player_0`), puis joue
        `n_parties` parties, `agent_a` commençant les parties d'indice pair.
        Elle renvoie un dictionnaire `{"victoires", "nuls", "defaites",
        "coups"}` vu par `agent_a`, `"coups"` étant la liste des longueurs.
        """),
        ex('''
        # À vous
        def jouer_match(agent_a, agent_b, n_parties=100):
            """n parties, agent_a commence une partie sur deux. Bilan vu par agent_a."""
            env = connect_four_v3.env()
            for agent in (agent_a, agent_b):
                agent.setup(env.observation_space("player_0"),
                            env.action_space("player_0"))
            bilan = {"victoires": 0, "nuls": 0, "defaites": 0, "coups": []}
            ...
            return bilan
        ''', L1_MATCH),
        c("""
        bilan = jouer_match(AgentAleatoire(), AgentAleatoire(), 30)
        assert bilan["victoires"] + bilan["nuls"] + bilan["defaites"] == 30
        assert len(bilan["coups"]) == 30
        print({k: v for k, v in bilan.items() if k != "coups"})
        print("✅ Exercice 8 réussi")
        """),
        m("""
        ### Ce que donne le hasard

        Deux joueurs au hasard sur 200 parties, puis 300 parties où le même
        joueur commence toujours. Avant d'exécuter : le joueur qui commence
        a-t-il un avantage ? Combien de coups dure une partie ? Les nulles
        sont-elles fréquentes ?
        """),
        c("""
        bilan = jouer_match(AgentAleatoire(), AgentAleatoire(), 200)
        coups = np.array(bilan["coups"])
        print("victoires / nuls / défaites :", bilan["victoires"], bilan["nuls"],
              bilan["defaites"])
        print(f"durée : moyenne {coups.mean():.1f} coups, min {coups.min()}, max {coups.max()}")

        premier = [jouer_partie(AgentAleatoire(), AgentAleatoire(), a_commence=True)[0]
                   for _ in range(300)]
        print(f"le joueur qui commence gagne {np.mean(np.array(premier) > 0):.0%} des parties")

        plt.hist(coups, bins=range(7, 44), color="#2f5d9e")
        plt.xlabel("nombre de coups")
        plt.ylabel("parties")
        plt.title("Durée de 200 parties entre deux agents aléatoires")
        plt.show()
        """),
        *sol_md("""
        **Lecture.** Entre deux agents aléatoires, celui qui commence gagne
        nettement plus d'une partie sur deux (environ 55 %) : il a un pion
        d'avance. C'est pourquoi les places alternent dans un match. Une
        partie dure une vingtaine de coups ; les nulles sont rares (il faut
        remplir les 42 cases). Même deux agents identiques ne font pas
        exactement 50/50 sur 200 parties : c'est la fluctuation
        d'échantillonnage, que le lab 3 chiffre.
        """),
        m("""
        ---

        ## 6. Un hasard plus malin

        ### Exercice 9 — `AgentPondere`, par héritage

        Les colonnes du centre participent à plus d'alignements. Écrivez
        `AgentPondere`, qui **hérite** d'`AgentAleatoire` et ne redéfinit que
        `choose_action` : il tire une colonne autorisée avec
        `random.choices(colonnes, weights=poids)`, les poids étant
        `[1, 2, 3, 4, 3, 2, 1]` (colonne 0 à 6).
        """),
        ex('''
        # À vous
        class AgentPondere(AgentAleatoire):
            """Au hasard, mais les colonnes du centre sont plus probables."""

            POIDS = [1, 2, 3, 4, 3, 2, 1]

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                ...
        ''', L1_PONDERE),
        c("""
        agent = AgentPondere()
        assert isinstance(agent, AgentAleatoire), "AgentPondere hérite d'AgentAleatoire"
        assert agent.reset("player_0", 0) is True, "reset est hérité"
        vide = np.zeros((6, 7, 2), dtype=np.int8)
        tirages = [agent.choose_action(vide, action_mask=[1] * 7) for _ in range(3000)]
        assert tirages.count(3) > 2 * tirages.count(0), "le centre doit sortir plus souvent"
        assert all(agent.choose_action(vide, action_mask=[1, 0, 0, 0, 0, 0, 1]) in (0, 6)
                   for _ in range(100))
        bilan = jouer_match(AgentPondere(), AgentAleatoire(), 100)
        print("pondéré contre aléatoire :", bilan["victoires"], "victoires sur 100")
        print("✅ Exercice 9 réussi")
        """),
        m("""
        ---

        ## 7. Pour aller plus loin (facultatif)

        ### Exercice 10 — Le film d'une partie

        Écrivez `images_partie(agent_a, agent_b)` : comme `jouer_partie`, mais
        avec `render_mode="rgb_array"`, et renvoie la liste des images
        (`env.render()`) prises après chaque coup. Affichez-en six.
        """),
        ex('''
        # À vous
        def images_partie(agent_a, agent_b):
            """Les images de la grille après chaque coup d'une partie."""
            images = []
            ...
            return images
        ''', '''
        def images_partie(agent_a, agent_b):
            """Les images de la grille après chaque coup d'une partie."""
            env = connect_four_v3.env(render_mode="rgb_array")
            env.reset()
            places = {"player_0": agent_a, "player_1": agent_b}
            for nom, agent in places.items():
                agent.reset(nom, 0)
            images = []
            for nom in env.agent_iter():
                observation, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                env.step(places[nom].choose_action(
                    observation["observation"], float(reward), False, False, info,
                    action_mask=observation["action_mask"]))
                images.append(env.render())
            env.close()
            return images
        '''),
        c("""
        images = images_partie(AgentPondere(), AgentAleatoire())
        assert len(images) >= 7 and images[0].ndim == 3
        choix = np.linspace(0, len(images) - 1, 6).astype(int)
        fig, axes = plt.subplots(1, 6, figsize=(15, 2.6))
        for ax, i in zip(axes, choix):
            ax.imshow(images[i])
            ax.set_title(f"coup {i + 1}")
            ax.axis("off")
        plt.show()
        print("✅ Exercice 10 réussi")
        """),
        m("""
        ---

        **Bilan.** Vous savez lire l'observation, jouer un coup autorisé et
        faire s'affronter deux agents comme le fait ML-Arena. Le lab 2
        construit un agent qui réfléchit : gagner quand c'est possible, bloquer
        l'adversaire, ne pas lui offrir la victoire.
        """),
    ]
    return cells


# =========================================================================
# Lab 2
# =========================================================================

def lab2() -> list[dict]:
    cells = header(2, "Un agent à règles", """
        **Objectifs.** Passer de l'observation à une grille simple ; écrire
        des fonctions courtes et testables (`coups_valides`, `ligne_libre`,
        `simuler_coup`, `est_gagnant`, `coup_gagnant`) ; les assembler en un
        agent à règles — gagner, sinon bloquer, sinon ne pas offrir la
        victoire, de préférence au centre — qui bat l'agent aléatoire presque
        à tous les coups.
        """)
    cells += setup_cells()
    cells += previous(L1_CODE, "lab 1")
    cells += [
        m("""
        ---

        ## 1. Une grille plus simple à manipuler

        Deux canaux, c'est pratique pour un réseau de neurones, moins pour
        raisonner. On travaille sur une grille 6 × 7 d'entiers :
        `0` = vide, `1` = moi, `2` = l'adversaire.
        """),
        c(L2_CONSTANTES),
        m("""
        ### Exercice 1 — `vers_grille`

        Écrivez `vers_grille(observation)` en **une** opération NumPy : le
        canal 0 vaut 1 là où j'ai un pion, le canal 1 là où l'adversaire en a
        un. Renvoyez un tableau d'entiers (`.astype(int)`).
        """),
        ex('''
        # À vous
        def vers_grille(observation):
            """(6, 7, 2) -> grille 6 x 7 : 0 = vide, 1 = moi, 2 = l'adversaire."""
            obs = np.asarray(observation)
            ...
        ''', L2_VERS_GRILLE),
        c("""
        essai = np.zeros((6, 7, 2), dtype=np.int8)
        essai[5, 3, 0] = 1
        essai[4, 3, 1] = 1
        g = vers_grille(essai)
        assert g.shape == (6, 7)
        assert g[5, 3] == MOI and g[4, 3] == ADV and g.sum() == 3
        assert (vers_grille(essai.tolist()) == g).all()
        print("✅ Exercice 1 réussi")
        """),
        m("""
        Trois outils pour la suite (donnés) : construire une grille à partir
        d'un dessin en texte, et revenir à l'observation et au masque que
        reçoit un agent — pour le tester sur des positions choisies.
        """),
        c(L2_OUTILS + """

g = grille_depuis_texte('''
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . O . . .
. . O X X . .
''')
print(g)
print(masque_depuis_grille(g))
"""),
        m("""
        ---

        ## 2. Jouer un coup sur la grille

        ### Exercice 2 — `coups_valides`

        Une colonne est jouable si sa case du **haut** (ligne 0) est vide.
        """),
        ex('''
        # À vous
        def coups_valides(grille):
            """Les colonnes dont la case du haut est vide."""
            ...
        ''', L2_COUPS),
        c("""
        assert coups_valides(np.zeros((6, 7), dtype=int)) == [0, 1, 2, 3, 4, 5, 6]
        pleine_0_et_6 = np.zeros((6, 7), dtype=int)
        pleine_0_et_6[:, 0] = [MOI, ADV] * 3
        pleine_0_et_6[:, 6] = [ADV, MOI] * 3
        assert coups_valides(pleine_0_et_6) == [1, 2, 3, 4, 5]
        print("✅ Exercice 2 réussi")
        """),
        m("""
        ### Exercice 3 — `ligne_libre`

        La ligne où tombe un pion lâché dans `col` : la case vide la plus
        **basse**. Parcourez les lignes de 5 à 0 avec
        `range(LIGNES - 1, -1, -1)` ; renvoyez `None` si la colonne est pleine.
        """),
        ex('''
        # À vous
        def ligne_libre(grille, col):
            """La ligne où tombe un pion lâché dans col, ou None si elle est pleine."""
            ...
        ''', L2_LIGNE_LIBRE),
        c("""
        vide = np.zeros((6, 7), dtype=int)
        assert ligne_libre(vide, 3) == 5, "sur une grille vide, le pion tombe en bas"
        assert ligne_libre(g, 3) == 3 and ligne_libre(g, 2) == 4 and ligne_libre(g, 0) == 5
        assert ligne_libre(pleine_0_et_6, 0) is None
        print("✅ Exercice 3 réussi")
        """),
        m("""
        ### Lire du code 1

        Pour essayer un coup sans abîmer la vraie grille, il faut une copie.
        Que va afficher ce code ?
        """),
        c("""
        a = np.zeros((6, 7), dtype=int)
        b = a
        b[5, 3] = MOI
        print(a[5, 3])
        c = a.copy()
        c[5, 4] = ADV
        print(a[5, 4])
        """),
        *sol_md("""
        **Réponse.** `1` puis `0`. `b = a` ne copie rien : `a` et `b` sont deux
        noms pour **le même** tableau, modifier `b` modifie `a`. `a.copy()`
        crée un nouveau tableau, indépendant.
        """),
        m("""
        ### Exercice 4 — `simuler_coup`

        Renvoyez une **copie** de la grille où `joueur` a joué dans `col` ;
        la grille d'origine ne doit pas changer.
        """),
        ex('''
        # À vous
        def simuler_coup(grille, col, joueur):
            """Une COPIE de la grille où joueur a joué dans col."""
            ...
        ''', L2_SIMULER),
        c("""
        avant = g.copy()
        apres = simuler_coup(g, 3, MOI)
        assert apres[3, 3] == MOI, "le pion se pose sur le O de la colonne 3"
        assert (g == avant).all(), "la grille d'origine ne doit pas changer"
        assert (apres != g).sum() == 1
        print("✅ Exercice 4 réussi")
        """),
        m(f"""
        ---

        ## 3. Détecter une victoire

        ![Les quatre directions]({ASSETS}directions-victoire.png)

        Plutôt que de chercher autour de chaque pion, on énumère une fois pour
        toutes les **fenêtres** : les suites de 4 cases alignées. Un joueur a
        gagné si l'une d'elles ne contient que ses pions.

        ### Exercice 5 — Les 69 fenêtres

        Pour chaque case de départ `(ligne, col)` et chaque direction
        `(dl, dc)` parmi `(0, 1)` (horizontale), `(1, 0)` (verticale), `(1, 1)`
        (diagonale ↘) et `(-1, 1)` (diagonale ↗), la fenêtre est
        `[(ligne + k*dl, col + k*dc) for k in range(4)]` ; gardez-la si ses 4
        cases sont dans la grille. Combien en attendez-vous ?
        """),
        ex('''
        # À vous
        def construire_fenetres():
            """Toutes les suites de 4 cases alignées sur la grille 6 x 7."""
            fenetres = []
            ...
            return fenetres


        FENETRES = construire_fenetres()
        print(len(FENETRES), "fenêtres")
        ''', L2_FENETRES + '\nprint(len(FENETRES), "fenêtres")'),
        c("""
        assert len(FENETRES) == 69, "24 horizontales + 21 verticales + 24 diagonales"
        assert [(5, 0), (5, 1), (5, 2), (5, 3)] in FENETRES
        assert [(5, 0), (4, 1), (3, 2), (2, 3)] in FENETRES, "diagonale ↗"
        assert all(0 <= l < 6 and 0 <= c < 7 for f in FENETRES for l, c in f)
        print("✅ Exercice 5 réussi")
        """),
        m("""
        ### Exercice 6 — `est_gagnant`

        Vrai si **toutes** les cases d'une fenêtre au moins valent `joueur`
        (`any` et `all` font cela en une ligne, une boucle convient aussi).
        """),
        ex('''
        # À vous
        def est_gagnant(grille, joueur):
            """Vrai si joueur a quatre pions alignés quelque part."""
            ...
        ''', L2_GAGNANT),
        c("""
        horizontale = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O O . . .
        . X X X X . .
        ''')
        verticale = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . O .
        . . . . . O .
        . X X . . O .
        . X X . X O .
        ''')
        diagonale = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . X . . .
        . . X O . . .
        . X O O . . .
        X O O X . . .
        ''')
        antidiagonale = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        O . . . . . .
        X O . . . . .
        X X O . . . .
        X O X O . . .
        ''')
        assert est_gagnant(horizontale, MOI) and not est_gagnant(horizontale, ADV)
        assert est_gagnant(verticale, ADV) and not est_gagnant(verticale, MOI)
        assert est_gagnant(diagonale, MOI), "diagonale ↗"
        assert est_gagnant(antidiagonale, ADV), "diagonale ↘"
        assert not est_gagnant(g, MOI) and not est_gagnant(g, ADV)
        print("✅ Exercice 6 réussi")
        """),
        m("""
        ### Trouvez l'erreur

        Cette fonction doit renvoyer une colonne qui fait gagner `joueur`
        **tout de suite**, ou `None`. Sur la grille ci-dessous (deux X en bas),
        aucun coup ne gagne. Que renvoie-t-elle, et qu'est-il arrivé à la
        grille ?
        """),
        c("""
        def coup_gagnant_bug(grille, joueur):
            for col in coups_valides(grille):
                grille[ligne_libre(grille, col), col] = joueur
                if est_gagnant(grille, joueur):
                    return col
            return None

        deux_x = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        O O . . . . .
        X X . . . . .
        ''')
        print(coup_gagnant_bug(deux_x, MOI))
        print(deux_x)
        """),
        *sol_md("""
        **Réponse.** Elle renvoie `3` : les essais précédents (colonnes 0, 1
        et 2) sont restés sur la grille, si bien qu'au quatrième la ligne du
        bas contient quatre X. La fonction **modifie** la grille qu'on lui
        passe au lieu de travailler sur une copie : il faut `simuler_coup`.
        """),
        m("""
        ### Exercice 7 — `coup_gagnant`

        La version correcte, avec `simuler_coup`.
        """),
        ex('''
        # À vous
        def coup_gagnant(grille, joueur):
            """Une colonne qui fait gagner joueur tout de suite, sinon None."""
            ...
        ''', L2_COUP_GAGNANT),
        c("""
        deux_x = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        O O . . . . .
        X X . . . . .
        ''')
        copie = deux_x.copy()
        assert coup_gagnant(deux_x, MOI) is None
        assert (deux_x == copie).all(), "la grille ne doit pas changer"
        assert coup_gagnant(simuler_coup(deux_x, 2, MOI), MOI) == 3
        trois_o = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . O
        . . . . X . O
        . . . X X X O
        ''')
        assert coup_gagnant(trois_o, ADV) == 6
        assert coup_gagnant(trois_o, MOI) in (2, 6)
        print("✅ Exercice 7 réussi")
        """),
        m("""
        ---

        ## 4. L'agent à règles

        Les règles, dans l'ordre de priorité :

        1. si je peux gagner, je gagne ;
        2. sinon, si l'adversaire peut gagner au prochain coup, je bloque ;
        3. sinon, j'évite les coups **suicidaires** : ceux après lesquels
           l'adversaire gagne (typiquement, jouer juste sous la case qui lui
           manque) ;
        4. parmi les coups restants, je tire au hasard en favorisant le centre.

        ### Exercice 8 — `coups_surs`

        Gardez les coups de `coups` après lesquels l'adversaire n'a **pas** de
        coup gagnant.
        """),
        ex('''
        # À vous
        def coups_surs(grille, coups):
            """Les coups après lesquels l'adversaire ne peut pas gagner tout de suite."""
            ...
        ''', L2_SURS),
        c("""
        piege = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O O X . .
        . X O X X O .
        ''')
        # Les trois O de la ligne 4 n'attendent que la case (4, 0), encore en
        # l'air : jouer en colonne 0 la leur offre.
        assert 0 not in coups_surs(piege, coups_valides(piege))
        assert 4 in coups_surs(piege, coups_valides(piege))
        assert coups_surs(np.zeros((6, 7), dtype=int), list(range(7))) == list(range(7))
        print("✅ Exercice 8 réussi")
        """),
        m("""
        ### Exercice 9 — `AgentRegles`

        Assemblez les quatre règles dans `choose_action`. La classe hérite
        d'`AgentAleatoire` (donc de `setup` et `reset`). Commencez par
        `grille = vers_grille(observation)` et
        `coups = colonnes_libres(action_mask)`.
        """),
        ex('''
        # À vous
        class AgentRegles(AgentAleatoire):
            """Gagner, sinon bloquer, sinon un coup sûr de préférence au centre."""

            POIDS = [1, 2, 3, 4, 3, 2, 1]

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                if terminated or truncated:
                    return None
                grille = vers_grille(observation)
                coups = colonnes_libres(action_mask)
                ...
        ''', L2_REGLES),
        c("""
        def coup_de(agent, grille):
            return agent.choose_action(observation_depuis_grille(grille),
                                       action_mask=masque_depuis_grille(grille))

        agent = AgentRegles()
        gagner = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O . . . .
        . X X X O . .
        ''')
        assert coup_de(agent, gagner) == 0, "X X X : gagner en colonne 0"
        bloquer = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . O . . .
        . X . O . . .
        . X X O . . .
        ''')
        assert coup_de(agent, bloquer) == 3, "bloquer la verticale des O"
        assert all(coup_de(agent, piege) != 0 for _ in range(30)), "ne pas offrir (4, 0)"
        bilan = jouer_match(AgentRegles(), AgentAleatoire(), 100)
        print("règles contre aléatoire :", bilan["victoires"], "victoires sur 100")
        assert bilan["victoires"] >= 90
        print("✅ Exercice 9 réussi")
        """),
        m("""
        ### Lire du code 2

        Sur cette grille, je peux gagner en colonne 6 **et** l'adversaire peut
        gagner en colonne 0. Que joue `AgentRegles` ? Pourquoi est-ce le bon
        ordre ?
        """),
        c("""
        les_deux = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . X
        . . . . . . X
        . O O O . . X
        ''')
        print(coup_de(AgentRegles(), les_deux))
        """),
        *sol_md("""
        **Réponse.** `6`. La règle « gagner » passe avant « bloquer » : gagner
        termine la partie, l'adversaire n'aura jamais l'occasion de jouer sa
        colonne 0. Bloquer d'abord laisserait passer une victoire certaine.
        """),
        m("""
        ---

        ## 5. Pour aller plus loin (facultatif)

        ### Exercice 10 — La double menace

        Une **double menace** est un coup après lequel j'ai **deux** colonnes
        gagnantes différentes : l'adversaire ne peut en bloquer qu'une.
        Écrivez `cree_double_menace(grille, col, joueur)`. Puis ajoutez cette
        règle à un `AgentReglesPlus(AgentRegles)` (entre « bloquer » et « coup
        sûr ») et mesurez-le contre `AgentRegles`.
        """),
        ex('''
        # À vous
        def cree_double_menace(grille, col, joueur):
            """Vrai si, après joueur en col, joueur a au moins deux colonnes gagnantes."""
            ...
        ''', '''
        def cree_double_menace(grille, col, joueur):
            """Vrai si, après joueur en col, joueur a au moins deux colonnes gagnantes."""
            apres = simuler_coup(grille, col, joueur)
            gagnantes = [c for c in coups_valides(apres)
                         if est_gagnant(simuler_coup(apres, c, joueur), joueur)]
            return len(gagnantes) >= 2


        class AgentReglesPlus(AgentRegles):
            """AgentRegles + créer une double menace quand c'est possible."""

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                if terminated or truncated:
                    return None
                grille = vers_grille(observation)
                coups = colonnes_libres(action_mask)
                for joueur in (MOI, ADV):
                    col = coup_gagnant(grille, joueur)
                    if col is not None:
                        return col
                surs = coups_surs(grille, coups)
                for col in surs:
                    if cree_double_menace(grille, col, MOI):
                        return col
                candidats = surs or coups
                poids = [self.POIDS[col] for col in candidats]
                return random.choices(candidats, weights=poids)[0]


        bilan = jouer_match(AgentReglesPlus(), AgentRegles(), 60)
        print("règles+ contre règles :", bilan["victoires"], "victoires,",
              bilan["defaites"], "défaites sur 60")
        '''),
        c("""
        ouverte = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O . . . .
        . X X . . O .
        ''')
        assert cree_double_menace(ouverte, 3, MOI), ". X X X . : deux cases gagnantes"
        assert not cree_double_menace(ouverte, 0, MOI)
        print("✅ Exercice 10 réussi")
        """),
        m("""
        ---

        **Bilan.** Des fonctions courtes, chacune vérifiée sur des grilles
        choisies, puis assemblées : c'est ce qui rend l'agent fiable. Le lab 3
        transforme ces vérifications en vrais tests (pytest) et organise un
        tournoi.
        """),
    ]
    return cells


# =========================================================================
# Lab 3
# =========================================================================

def lab3() -> list[dict]:
    cells = header(3, "Tester et organiser un tournoi", """
        **Objectifs.** Concevoir une stratégie de test ; écrire des tests qui
        **détectent** de vraies erreurs ; tester l'agent sur des scénarios ;
        écrire un module et ses tests `pytest`, mesurer sa qualité avec
        `pylint` ; mesurer le temps par coup et la mémoire ; comparer les
        agents dans un tournoi et savoir combien de parties il faut pour
        conclure.
        """)
    cells += setup_cells()
    cells += previous(L1_CODE + "\n\n\n" + L2_CODE, "labs 1 et 2")
    cells += [
        m(f"""
        ---

        ## 1. Un plan de test

        | catégorie | question | comment | critère de succès |
        |---|---|---|---|
        | fonctionnel | les fonctions font-elles ce qu'elles disent ? | grilles choisies, `assert` | 100 % des tests passent |
        | fonctionnel | l'agent joue-t-il toujours un coup autorisé ? | positions aléatoires | aucun coup interdit |
        | stratégique | gagne-t-il, bloque-t-il quand il faut ? | scénarios | le coup attendu |
        | stratégique | est-il plus fort qu'un autre agent ? | match de N parties | ≥ 90 % contre l'aléatoire |
        | performance | répond-il à temps ? | chronomètre | < 0,1 s par coup (limite : {TEMPS_MAX} s) |
        | performance | tient-il en mémoire ? | `tracemalloc` | quelques Mo |

        Un test utile est un test qui **échoue** quand le code est faux. On le
        vérifie en lui donnant du code volontairement faux (un *mutant*).

        ### Exercice 1 — Des tests qui attrapent les erreurs

        Écrivez trois fonctions de test, chacune avec plusieurs `assert` :
        `test_coups_valides`, `test_ligne_libre` et `test_est_gagnant`. La
        cellule de vérification les exécute sur le bon code (ils doivent
        passer) puis sur trois mutants (ils doivent échouer) : un
        `coups_valides` qui ignore les colonnes pleines, un `ligne_libre` qui
        remplit par le haut, un `est_gagnant` qui oublie les diagonales.
        """),
        ex('''
        # À vous
        def test_coups_valides():
            ...


        def test_ligne_libre():
            ...


        def test_est_gagnant():
            ...
        ''', '''
        def test_coups_valides():
            vide = np.zeros((6, 7), dtype=int)
            assert coups_valides(vide) == list(range(7))
            pleine = vide.copy()
            pleine[:, 2] = [MOI, ADV] * 3
            assert coups_valides(pleine) == [0, 1, 3, 4, 5, 6]


        def test_ligne_libre():
            vide = np.zeros((6, 7), dtype=int)
            assert ligne_libre(vide, 0) == 5
            assert ligne_libre(simuler_coup(vide, 0, MOI), 0) == 4
            pleine = vide.copy()
            pleine[:, 2] = [MOI, ADV] * 3
            assert ligne_libre(pleine, 2) is None


        def test_est_gagnant():
            g = np.zeros((6, 7), dtype=int)
            assert not est_gagnant(g, MOI)
            g[5, 0:4] = MOI
            assert est_gagnant(g, MOI) and not est_gagnant(g, ADV)
            diag = np.zeros((6, 7), dtype=int)
            for k in range(4):
                diag[5 - k, k] = ADV          # diagonale ↗
            assert est_gagnant(diag, ADV)
            anti = np.zeros((6, 7), dtype=int)
            for k in range(4):
                anti[2 + k, k] = MOI          # diagonale ↘
            assert est_gagnant(anti, MOI)
        '''),
        c("""
        def _attrape(test, nom, mutant):
            original = globals()[nom]
            globals()[nom] = mutant
            try:
                test()
                return False
            except Exception:
                return True
            finally:
                globals()[nom] = original

        def _mutant_coups(grille):
            return list(range(COLONNES))

        def _mutant_ligne(grille, col):
            for ligne in range(LIGNES):
                if grille[ligne, col] == VIDE:
                    return ligne
            return None

        def _mutant_gagnant(grille, joueur):
            droites = [f for f in FENETRES if len({l for l, _ in f}) == 1
                       or len({c for _, c in f}) == 1]
            return any(all(grille[l, c] == joueur for l, c in f) for f in droites)

        for test in (test_coups_valides, test_ligne_libre, test_est_gagnant):
            test()  # sur le bon code : ne doit rien lever
        assert _attrape(test_coups_valides, "coups_valides", _mutant_coups), \\
            "test_coups_valides ne voit pas une colonne pleine jouée"
        assert _attrape(test_ligne_libre, "ligne_libre", _mutant_ligne), \\
            "test_ligne_libre ne voit pas un pion posé en haut"
        assert _attrape(test_est_gagnant, "est_gagnant", _mutant_gagnant), \\
            "test_est_gagnant ne teste pas les diagonales"
        print("✅ Exercice 1 réussi")
        """),
        m("""
        ### Exercice 2 — Des scénarios pour l'agent

        Un scénario est une position dessinée en texte et l'ensemble des coups
        acceptables. `verifier_scenario` (donnée) demande 20 fois son coup à
        l'agent — il tire parfois au hasard — et vérifie qu'il est toujours
        acceptable. Deux scénarios sont fournis ; ajoutez-en **au moins
        trois** : une victoire verticale, un blocage en diagonale, et un piège
        à éviter (l'ensemble attendu contient alors tous les coups sauf le
        mauvais).
        """),
        c('''
        def verifier_scenario(agent, texte, attendus, essais=20):
            grille = grille_depuis_texte(texte)
            for _ in range(essais):
                coup = agent.choose_action(observation_depuis_grille(grille),
                                           action_mask=masque_depuis_grille(grille))
                if coup not in attendus:
                    return False
            return True
        '''),
        ex('''
        # À vous : complétez la liste
        SCENARIOS = [
            ("gagner à l'horizontale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        X X X . O O O
        """, {3}),
            ("bloquer à l'horizontale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . X .
        O O O . X X .
        """, {3}),
            # ...
        ]
        ''', '''
        SCENARIOS = [
            ("gagner à l'horizontale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        X X X . O O O
        """, {3}),
            ("bloquer à l'horizontale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . X .
        O O O . X X .
        """, {3}),
            ("gagner à la verticale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . X . . . . .
        . X . O . . .
        . X . O O . .
        """, {1}),
            ("bloquer en diagonale", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . O X . . .
        . O X X . . .
        O X X O . . .
        """, {3}),
            ("ne pas offrir la case (4, 0)", """
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O O X . .
        . X O X X O .
        """, {1, 2, 3, 4, 5, 6}),
        ]
        '''),
        c("""
        assert len(SCENARIOS) >= 5, "au moins trois scénarios de plus"
        agent = AgentRegles()
        for nom, texte, attendus in SCENARIOS:
            assert verifier_scenario(agent, texte, attendus), f"échec : {nom}"
            print("ok :", nom)
        print("✅ Exercice 2 réussi")
        """),
        m("""
        ---

        ## 2. Un module, ses tests, pytest et pylint

        Dans le projet, le code vit dans des fichiers `.py`, pas dans un
        notebook. La cellule suivante écrit le module `puissance4.py` (le code
        des labs 1 et 2) avec `%%writefile`.
        """),
        c("%%writefile puissance4.py\n"
          '"""Puissance 4 : outils de grille et agents (labs 1 et 2)."""\n'
          "import random\n\nimport numpy as np\n"
          "from pettingzoo.classic import connect_four_v3\n\n\n"
          + L1_CODE + "\n\n\n" + L2_CODE),
        m("""
        ### Lire du code 1

        `pytest` exécute les fonctions dont le nom commence par `test_` dans
        les fichiers `test_*.py`. Combien de tests va-t-il exécuter dans ce
        fichier, et combien échoueront ?
        """),
        c('''
        %%writefile test_demo.py
        def test_cases():
            assert 6 * 7 == 42


        def verifie_directions():
            assert 2 + 2 == 5


        def test_centre():
            assert 7 // 2 == 3.5
        '''),
        c("""
        import subprocess
        sortie = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_demo.py"],
                                capture_output=True, text=True)
        print(sortie.stdout[-600:])
        """),
        *sol_md("""
        **Réponse.** Deux tests exécutés (`test_cases`, `test_centre`) ;
        `verifie_directions` est ignorée, malgré son assertion fausse, car son
        nom ne commence pas par `test_`. `test_centre` échoue : `7 // 2` vaut
        `3` (division entière), pas `3.5`.
        """),
        m("""
        ### Exercice 3 — `test_puissance4.py`

        Le fichier de tests importe le module. Deux tests sont fournis ;
        ajoutez-en **au moins quatre** (reprenez ceux de l'exercice 1, les
        scénarios, et un test « l'agent joue toujours un coup autorisé » sur
        des positions tirées au hasard). La cellule suivante lance `pytest`.
        """),
        ex('''
        %%writefile test_puissance4.py
        import numpy as np

        from puissance4 import (ADV, MOI, AgentAleatoire, AgentRegles, coup_gagnant,
                                coups_valides, est_gagnant, grille_depuis_texte,
                                ligne_libre, masque_depuis_grille,
                                observation_depuis_grille, simuler_coup)


        def test_grille_vide():
            vide = np.zeros((6, 7), dtype=int)
            assert coups_valides(vide) == list(range(7))
            assert ligne_libre(vide, 3) == 5


        def test_simuler_ne_modifie_pas():
            vide = np.zeros((6, 7), dtype=int)
            simuler_coup(vide, 3, MOI)
            assert vide.sum() == 0


        # À vous : au moins quatre tests de plus
        ''', '''
        %%writefile test_puissance4.py
        import numpy as np

        from puissance4 import (ADV, MOI, AgentAleatoire, AgentRegles, coup_gagnant,
                                coups_valides, est_gagnant, grille_depuis_texte,
                                ligne_libre, masque_depuis_grille,
                                observation_depuis_grille, simuler_coup)


        def test_grille_vide():
            vide = np.zeros((6, 7), dtype=int)
            assert coups_valides(vide) == list(range(7))
            assert ligne_libre(vide, 3) == 5


        def test_simuler_ne_modifie_pas():
            vide = np.zeros((6, 7), dtype=int)
            simuler_coup(vide, 3, MOI)
            assert vide.sum() == 0


        def test_colonne_pleine():
            g = np.zeros((6, 7), dtype=int)
            g[:, 2] = [MOI, ADV] * 3
            assert 2 not in coups_valides(g)
            assert ligne_libre(g, 2) is None


        def test_victoires_quatre_directions():
            for cases in ([(5, 0), (5, 1), (5, 2), (5, 3)],
                          [(5, 6), (4, 6), (3, 6), (2, 6)],
                          [(5, 0), (4, 1), (3, 2), (2, 3)],
                          [(2, 0), (3, 1), (4, 2), (5, 3)]):
                g = np.zeros((6, 7), dtype=int)
                for l, c in cases:
                    g[l, c] = ADV
                assert est_gagnant(g, ADV) and not est_gagnant(g, MOI)


        def test_coup_gagnant():
            g = grille_depuis_texte("""
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O . . . .
        . X X X O . .
        """)
            assert coup_gagnant(g, MOI) == 0
            assert coup_gagnant(g, ADV) is None


        def test_agent_regles_bloque():
            g = grille_depuis_texte("""
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . X .
        . . . . X X .
        O O O . X O .
        """)
            obs, masque = observation_depuis_grille(g), masque_depuis_grille(g)
            assert all(AgentRegles().choose_action(obs, action_mask=masque) == 3
                       for _ in range(10))


        def test_agents_jouent_des_coups_autorises():
            rng = np.random.default_rng(0)
            for _ in range(100):
                g = np.zeros((6, 7), dtype=int)
                for k in range(rng.integers(0, 30)):  # une position de jeu au hasard
                    if not coups_valides(g) or est_gagnant(g, MOI) or est_gagnant(g, ADV):
                        break
                    g = simuler_coup(g, rng.choice(coups_valides(g)), [MOI, ADV][k % 2])
                if not coups_valides(g):
                    continue
                obs, masque = observation_depuis_grille(g), masque_depuis_grille(g)
                for agent in (AgentAleatoire(), AgentRegles()):
                    assert masque[agent.choose_action(obs, action_mask=masque)] == 1


        def test_fin_de_partie_renvoie_none():
            obs = np.zeros((6, 7, 2), dtype=np.int8)
            assert AgentRegles().choose_action(obs, terminated=True,
                                               action_mask=[1] * 7) is None
        '''),
        c("""
        import re
        sortie = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_puissance4.py"],
                                capture_output=True, text=True)
        print(sortie.stdout[-1500:])
        reussis = re.search(r"(\\d+) passed", sortie.stdout)
        assert sortie.returncode == 0, "des tests échouent : lisez le rapport ci-dessus"
        assert reussis and int(reussis.group(1)) >= 6, "au moins six tests"
        print("✅ Exercice 3 réussi")
        """),
        m("""
        L'avant-dernier test fabrique 100 positions en jouant des coups au
        hasard, et vérifie que chaque agent respecte le masque sur chacune.
        La graine fixe (`default_rng(0)`) rend le test reproductible : un test
        qui échoue une fois sur dix ne sert à rien.

        `pylint` note la qualité du code sur 10 : noms, docstrings, longueur
        des lignes, imports inutilisés… Lisez ses remarques et corrigez ce qui
        a du sens dans votre propre dépôt.
        """),
        c("""
        try:
            import pylint
        except ImportError:  # Colab : pylint n'est pas préinstallé
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylint"],
                           check=True)
        sortie = subprocess.run([sys.executable, "-m", "pylint", "puissance4.py",
                                 "--disable=too-few-public-methods"],
                                capture_output=True, text=True)
        print(sortie.stdout[-1500:])
        """),
        m(f"""
        ---

        ## 3. Le temps et la mémoire

        Sur ML-Arena, chaque appel à `choose_action` doit répondre en moins de
        **{TEMPS_MAX} s**, aller-retour compris ; au-delà, le match est perdu.
        Il faut donc mesurer, et garder une marge.

        ### Exercice 4 — `AgentChronometre`

        Écrivez une classe qui **enveloppe** un agent : elle respecte le même
        contrat, transmet chaque appel à l'agent enveloppé, et enregistre la
        durée de chaque `choose_action` dans `self.durees`
        (`time.perf_counter()` avant et après). C'est la *délégation* : on
        ajoute un comportement sans toucher à l'agent.
        """),
        ex('''
        # À vous
        class AgentChronometre:
            """Enveloppe un agent et mesure la durée de chaque choose_action."""

            def __init__(self, agent):
                self.agent = agent
                self.durees = []

            def setup(self, observation_space, action_space):
                ...

            def reset(self, env_player_name, episode_index):
                ...

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                ...
        ''', L3_CHRONO),
        c("""
        class _Lent(AgentAleatoire):
            def choose_action(self, *args, **kwargs):
                time.sleep(0.01)
                return super().choose_action(*args, **kwargs)

        chrono = AgentChronometre(_Lent())
        bilan = jouer_match(chrono, AgentAleatoire(), 4)
        assert len(chrono.durees) > 4 * 3, "une durée par coup joué"
        assert min(chrono.durees) >= 0.01
        assert chrono.agent.env_player_name in ("player_0", "player_1"), "reset transmis"
        print("✅ Exercice 4 réussi")
        """),
        m("""
        ### Lire du code 2

        `AgentChronometre` **contient** un agent, il n'en **hérite** pas. Que
        vont afficher ces trois tests ?
        """),
        c("""
        chrono = AgentChronometre(AgentRegles())
        print(isinstance(chrono, AgentAleatoire))
        print(hasattr(chrono, "POIDS"))
        print(hasattr(chrono.agent, "POIDS"))
        """),
        *sol_md("""
        **Réponse.** `False`, `False`, `True`. L'enveloppe n'est pas un
        `AgentAleatoire` et n'a pas les attributs de l'agent ; elle les
        atteint par `self.agent`. La délégation permet d'envelopper **n'importe
        quel** agent — aléatoire, à règles, minimax — avec la même classe, là
        où l'héritage demanderait une sous-classe chronométrée par agent.
        """),
        c("""
        for nom, agent in [("aléatoire", AgentAleatoire()), ("pondéré", AgentPondere()),
                           ("règles", AgentRegles())]:
            chrono = AgentChronometre(agent)
            jouer_match(chrono, AgentAleatoire(), 20)
            d = np.array(chrono.durees) * 1000
            print(f"{nom:10s} moyenne {d.mean():6.2f} ms   max {d.max():6.2f} ms")
        """),
        m("""
        ### Exercice 5 — La mémoire

        `tracemalloc` (bibliothèque standard) mesure la mémoire allouée par
        Python. Écrivez `pic_memoire(agent)` : démarre la mesure
        (`tracemalloc.start()`), joue une partie contre `AgentAleatoire`,
        lit le pic (`tracemalloc.get_traced_memory()[1]`, en octets), arrête
        la mesure (`tracemalloc.stop()`) et renvoie le pic en **Mo**.
        """),
        ex('''
        # À vous
        import tracemalloc


        def pic_memoire(agent):
            """Le pic de mémoire (Mo) pendant une partie contre AgentAleatoire."""
            ...
        ''', '''
        import tracemalloc


        def pic_memoire(agent):
            """Le pic de mémoire (Mo) pendant une partie contre AgentAleatoire."""
            tracemalloc.start()
            jouer_partie(agent, AgentAleatoire())
            pic = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return pic / 1e6
        '''),
        c("""
        mo = pic_memoire(AgentRegles())
        assert isinstance(mo, float) and 0 < mo < 100
        print(f"pic pendant une partie : {mo:.2f} Mo")
        print("✅ Exercice 5 réussi")
        """),
        m("""
        ---

        ## 4. Le tournoi

        ### Exercice 6 — `tournoi`

        Chaque paire d'agents joue un match (`jouer_match`, places
        alternées). Renvoyez un `DataFrame` dont la case (ligne `a`, colonne
        `b`) est le taux de victoire de `a` contre `b` ; la diagonale reste
        `NaN`. `itertools.combinations(noms, 2)` donne les paires.
        """),
        ex('''
        # À vous
        import itertools

        import pandas as pd


        def tournoi(agents, n_parties=50):
            """Chaque paire d'agents joue un match ; taux de victoire ligne / colonne."""
            noms = list(agents)
            taux = pd.DataFrame(np.nan, index=noms, columns=noms)
            ...
            return taux
        ''', "import itertools\n\nimport pandas as pd\n\n" + L3_TOURNOI),
        c("""
        agents = {"aléatoire": AgentAleatoire(), "pondéré": AgentPondere(),
                  "règles": AgentRegles()}
        taux = tournoi(agents, 50)
        print(taux.round(2))
        assert list(taux.index) == list(agents) and taux.isna().sum().sum() == 3
        for a, b in itertools.combinations(agents, 2):
            assert taux.loc[a, b] + taux.loc[b, a] <= 1, "victoires + défaites ≤ 1"
        assert taux.loc["règles", "aléatoire"] >= 0.9
        print("✅ Exercice 6 réussi")
        """),
        m("""
        ### Trouvez l'erreur

        Ce match oppose deux agents **identiques**. Pourquoi l'agent A gagne-t-il
        nettement plus d'une partie sur deux ? Corrigez `match_biaise` dans
        `match_equitable`.
        """),
        c("""
        def match_biaise(agent_a, agent_b, n_parties):
            victoires = 0
            for i in range(n_parties):
                resultat, _ = jouer_partie(agent_a, agent_b, index=i)
                victoires += resultat > 0
            return victoires / n_parties

        print(match_biaise(AgentAleatoire(), AgentAleatoire(), 300))
        """),
        *sol_md("""
        **Réponse.** `a_commence` vaut `True` par défaut : A commence
        **toutes** les parties, et celui qui commence a un avantage (environ
        55 % contre un aléatoire). Le taux mesure la place, pas l'agent. Il
        faut alterner, comme le fait `jouer_match` (et comme ML-Arena change
        les places d'une partie à l'autre).
        """),
        m("""
        ### Exercice 7 — `match_equitable`

        Écrivez la version corrigée : `agent_a` commence une partie sur deux.
        """),
        ex('''
        # À vous
        def match_equitable(agent_a, agent_b, n_parties):
            ...
        ''', '''
        def match_equitable(agent_a, agent_b, n_parties):
            victoires = 0
            for i in range(n_parties):
                resultat, _ = jouer_partie(agent_a, agent_b, a_commence=(i % 2 == 0),
                                           index=i)
                victoires += resultat > 0
            return victoires / n_parties
        '''),
        c("""
        class _Compteur(AgentAleatoire):
            def reset(self, env_player_name, episode_index):
                self.premieres = getattr(self, "premieres", 0) + (env_player_name == "player_0")
                return super().reset(env_player_name, episode_index)

        compteur = _Compteur()
        match_equitable(compteur, AgentAleatoire(), 20)
        assert compteur.premieres == 10, "A doit commencer une partie sur deux"
        print("✅ Exercice 7 réussi")
        """),
        m("""
        ### Exercice 8 — Quelle règle sert le plus ?

        Pour comprendre un agent, on journalise ses décisions. Écrivez
        `AgentReglesJournal(AgentRegles)` : même logique, mais chaque décision
        incrémente `self.journal[raison]` (un `collections.Counter`), avec
        `raison` parmi `"gagner"`, `"bloquer"`, `"centre"`. Dans un vrai
        projet, on écrirait ces lignes dans un fichier avec le module
        `logging` (ou la bibliothèque `loguru`) plutôt que de les compter.
        """),
        ex('''
        # À vous
        from collections import Counter


        class AgentReglesJournal(AgentRegles):
            """AgentRegles qui compte la raison de chaque décision."""

            def __init__(self):
                self.journal = Counter()

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                ...
        ''', '''
        from collections import Counter


        class AgentReglesJournal(AgentRegles):
            """AgentRegles qui compte la raison de chaque décision."""

            def __init__(self):
                self.journal = Counter()

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                if terminated or truncated:
                    return None
                grille = vers_grille(observation)
                coups = colonnes_libres(action_mask)
                for joueur, raison in ((MOI, "gagner"), (ADV, "bloquer")):
                    col = coup_gagnant(grille, joueur)
                    if col is not None:
                        self.journal[raison] += 1
                        return col
                self.journal["centre"] += 1
                candidats = coups_surs(grille, coups) or coups
                poids = [self.POIDS[col] for col in candidats]
                return random.choices(candidats, weights=poids)[0]
        '''),
        c("""
        journal = AgentReglesJournal()
        bilan = jouer_match(journal, AgentRegles(), 40)
        print(dict(journal.journal), "|", bilan["victoires"], "victoires,",
              bilan["defaites"], "défaites")
        assert set(journal.journal) <= {"gagner", "bloquer", "centre"}
        assert journal.journal["gagner"] >= bilan["victoires"] > 0
        print("✅ Exercice 8 réussi")
        """),
        m("""
        ---

        ## 5. Pour aller plus loin (facultatif)

        ### Exercice 9 — Combien de parties pour conclure ?

        Un taux de victoire mesuré sur `n` parties fluctue. L'intervalle de
        confiance à 95 % (approximation normale) est :

        $$\\hat p \\pm 1{,}96\\sqrt{\\hat p(1-\\hat p)/n}$$

        Écrivez `intervalle_confiance(victoires, n)` qui renvoie
        `(bas, haut)`, bornés à [0, 1]. Combien de parties faut-il pour
        distinguer 55 % de 50 % ?
        """),
        ex('''
        # À vous
        def intervalle_confiance(victoires, n):
            """Intervalle de confiance à 95 % d'un taux de victoire."""
            ...
        ''', '''
        def intervalle_confiance(victoires, n):
            """Intervalle de confiance à 95 % d'un taux de victoire."""
            p = victoires / n
            marge = 1.96 * np.sqrt(p * (1 - p) / n)
            return max(0.0, p - marge), min(1.0, p + marge)


        for n in (10, 100, 400, 1000):
            bas, haut = intervalle_confiance(0.55 * n, n)
            print(f"55 % sur {n:4d} parties : [{bas:.1%}, {haut:.1%}]")
        '''),
        c("""
        bas, haut = intervalle_confiance(50, 100)
        assert abs(bas - 0.402) < 0.001 and abs(haut - 0.598) < 0.001
        assert intervalle_confiance(100, 100) == (1.0, 1.0)
        print("✅ Exercice 9 réussi")
        """),
        *sol_md("""
        **Lecture.** Sur 100 parties, 55 % donne [45 %, 65 %] : impossible de
        dire que l'agent fait mieux que 50 %. Il faut environ 400 parties pour
        que l'intervalle exclue 50 %. Un match de 10 parties sur ML-Arena est
        très bruité : c'est pourquoi le classement ELO agrège de nombreux
        matchs.
        """),
        m("""
        ---

        **Bilan.** Vos agents sont testés, chronométrés et comparés. Le lab 4
        construit un agent qui anticipe plusieurs coups (minimax), sous la
        contrainte de temps de la plateforme, et le soumet sur ML-Arena.
        """),
    ]
    return cells


# =========================================================================
# Lab 4
# =========================================================================

def lab4() -> list[dict]:
    cells = header(4, "Minimax et soumission sur ML-Arena", f"""
        **Objectifs.** Évaluer une position par une heuristique ; anticiper
        plusieurs coups avec l'algorithme **minimax** et l'**élagage
        alpha-bêta** ; tenir la limite de **{TEMPS_MAX} s par coup** grâce à
        l'approfondissement itératif ; écrire `agent.py`, le tester comme la
        plateforme, et le **soumettre** sur [ML-Arena]({CHALLENGE_URL}).
        """)
    cells += setup_cells()
    cells += previous(L1_CODE + "\n\n\n" + L2_CODE + "\n\n\n" + L3_CODE,
                      "labs 1 à 3")
    cells += [
        m(f"""
        ---

        ## 1. Anticiper : l'arbre de jeu

        ![Un arbre minimax de profondeur 2]({ASSETS}arbre-minimax.png)

        Depuis la position actuelle, chacun de mes coups mène à une position,
        chacune des réponses adverses à une autre, etc. : c'est l'**arbre de
        jeu**. On le développe jusqu'à une **profondeur** fixée, on note les
        feuilles avec une **fonction d'évaluation**, puis on remonte : à mon
        tour je prends le **max** (je choisis le meilleur pour moi), au tour
        adverse le **min** (l'adversaire choisit le pire pour moi).

        ### Lire du code 1

        L'arbre ci-dessus, en listes imbriquées. Que vont afficher les deux
        `print` ?
        """),
        c("""
        def minimax_arbre(noeud, maximise):
            if isinstance(noeud, int):          # une feuille : sa valeur
                return noeud
            valeurs = [minimax_arbre(enfant, not maximise) for enfant in noeud]
            return max(valeurs) if maximise else min(valeurs)

        arbre = [[3, 5], [2, 9], [0, 1]]
        print(minimax_arbre(arbre, True))
        print(minimax_arbre(arbre, False))
        """),
        *sol_md("""
        **Réponse.** `3`, puis `1`. Avec MAX à la racine : les nœuds MIN
        valent 3, 2 et 0, et MAX prend 3 — la colonne 0, même si le 9 est sous
        la colonne 1 (l'adversaire ne le laissera jamais arriver). Avec MIN à
        la racine : les nœuds MAX valent 5, 9 et 1, MIN prend 1.
        """),
        m("""
        ---

        ## 2. La fonction d'évaluation

        Au bout de l'arbre, il faut noter une position qui n'est pas finie. On
        reprend les 69 fenêtres : une fenêtre où je suis seul avec 3 pions est
        une menace, avec 2 pions un début ; une fenêtre mixte ne vaut plus
        rien.

        ### Exercice 1 — `score_fenetre`

        `valeurs` est la liste des 4 valeurs d'une fenêtre. Renvoyez : 0 si
        elle contient mes pions **et** ceux de l'adversaire ; sinon 5 pour
        3 pions à moi, 2 pour 2 pions à moi, −4 pour 3 pions adverses,
        −2 pour 2 pions adverses, 0 dans les autres cas. (Les fenêtres
        complètes sont traitées à part, comme victoires.)
        """),
        ex('''
        # À vous
        def score_fenetre(valeurs):
            """Le score d'une fenêtre de 4 cases, de mon point de vue."""
            moi, adv = valeurs.count(MOI), valeurs.count(ADV)
            ...
        ''', L4_SCORE),
        c("""
        assert score_fenetre([MOI, MOI, MOI, VIDE]) == 5
        assert score_fenetre([VIDE, MOI, VIDE, MOI]) == 2
        assert score_fenetre([ADV, ADV, VIDE, ADV]) == -4
        assert score_fenetre([ADV, VIDE, ADV, VIDE]) == -2
        assert score_fenetre([MOI, MOI, MOI, ADV]) == 0
        assert score_fenetre([MOI, VIDE, VIDE, VIDE]) == 0
        print("✅ Exercice 1 réussi")
        """),
        m("""
        ### Exercice 2 — `evaluer`

        Additionnez `score_fenetre` sur les 69 fenêtres (les valeurs d'une
        fenêtre : `[grille[l, c] for l, c in fenetre]`), et ajoutez
        3 points par pion à moi dans la colonne centrale (colonne 3), −3 par
        pion adverse.
        """),
        ex('''
        # À vous
        def evaluer(grille):
            """Somme des scores des 69 fenêtres + bonus pour la colonne centrale."""
            ...
        ''', L4_EVALUER),
        c("""
        vide = np.zeros((6, 7), dtype=int)
        assert evaluer(vide) == 0
        assert evaluer(simuler_coup(vide, 3, MOI)) > evaluer(simuler_coup(vide, 0, MOI)), \\
            "le centre vaut plus que le bord"
        trois = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        X X X . . O O
        ''')
        assert evaluer(trois) > 0
        print("évaluation :", evaluer(trois))
        print("✅ Exercice 2 réussi")
        """),
        m("""
        ---

        ## 3. Minimax avec élagage alpha-bêta

        `alpha` est la meilleure valeur que MAX est déjà sûr d'obtenir ailleurs,
        `beta` la meilleure que MIN est sûr d'obtenir. Dès que `alpha >= beta`,
        les coups restants de ce nœud ne peuvent plus changer la décision : on
        les **élague** (`break`). Le résultat est le même que minimax, en
        visitant beaucoup moins de positions — surtout si l'on essaie les
        bons coups d'abord (le centre).

        Le paramètre `fin` sert à la section 5 : si l'heure `fin` est dépassée,
        on lève `TempsEcoule` pour abandonner la recherche.

        ### Exercice 3 — `minimax`

        Complétez, dans cet ordre : 1) les fins de partie (j'ai gagné :
        `GAGNE + profondeur` ; l'adversaire a gagné : `-GAGNE - profondeur` ;
        grille pleine : 0) ; 2) `profondeur == 0` : `evaluer(grille)` ;
        3) la récursion — si `maximise`, je joue (`MOI`) et je prends le max,
        sinon l'adversaire joue (`ADV`) et prend le min, en mettant à jour
        `alpha` / `beta` et en élaguant.
        """),
        ex('''
        # À vous
        GAGNE = 1_000_000


        class TempsEcoule(Exception):
            """Levée quand la recherche dépasse son budget de temps."""


        def minimax(grille, profondeur, alpha, beta, maximise, fin=float("inf")):
            """La valeur de la position, MOI cherchant le max et ADV le min."""
            if time.perf_counter() > fin:
                raise TempsEcoule
            ...
        ''', L4_MINIMAX),
        c("""
        inf = float("inf")
        assert minimax(trois, 0, -inf, inf, True) == evaluer(trois)
        gagne = simuler_coup(trois, 3, MOI)
        assert minimax(gagne, 3, -inf, inf, False) == GAGNE + 3
        # au tour de MOI, profondeur 1 : il voit sa victoire en colonne 3
        assert minimax(trois, 1, -inf, inf, True) == GAGNE
        # au tour d'ADV, profondeur 2 : il bloque, plus de victoire immédiate
        assert minimax(trois, 2, -inf, inf, False) < GAGNE
        print("✅ Exercice 3 réussi")
        """),
        m("""
        ### Exercice 4 — `meilleur_coup`

        À la racine, on veut le **coup**, pas seulement la valeur. Pour chaque
        colonne de `coups` (dans l'ordre donné), calculez la valeur de la
        position après mon coup (`minimax` avec `maximise=False`, profondeur
        − 1, en passant le meilleur `alpha` trouvé jusqu'ici) ; gardez la
        meilleure. Renvoyez `(colonne, valeur)`.
        """),
        ex('''
        # À vous
        def meilleur_coup(grille, coups, profondeur, fin=float("inf")):
            """(colonne, valeur) du meilleur coup, en essayant les coups dans l'ordre."""
            ...
        ''', L4_MEILLEUR),
        c("""
        bloquer = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . X .
        O O O . X X .
        ''')
        assert meilleur_coup(bloquer, coups_valides(bloquer), 2)[0] == 3, "bloquer en 3"
        ouverte = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . O O . . . .
        . X X . . . .
        ''')
        col, valeur = meilleur_coup(ouverte, coups_valides(ouverte), 3)
        assert col == 3 and valeur >= GAGNE, ". X X X . : victoire forcée en 3 coups"
        print("✅ Exercice 4 réussi")
        """),
        m("""
        ### Trouvez l'erreur

        Ce minimax (sans élagage, pour faire court) ne voit aucun danger sur
        la grille `bloquer` : comparez ses valeurs à celles de votre `minimax`.
        Pourquoi les coups 0, 1, 2, 4, 5 et 6, qui laissent les O gagner en
        colonne 3, ne valent-ils pas −1 000 000 ? Trouvez la ligne fautive.
        """),
        c("""
        def minimax_bug(grille, profondeur, maximise):
            if est_gagnant(grille, MOI):
                return GAGNE
            if est_gagnant(grille, ADV):
                return -GAGNE
            if profondeur == 0 or not coups_valides(grille):
                return evaluer(grille)
            valeurs = [minimax_bug(simuler_coup(grille, col, MOI), profondeur - 1,
                                   not maximise)
                       for col in coups_valides(grille)]
            return max(valeurs) if maximise else min(valeurs)

        inf = float("inf")
        for col in coups_valides(bloquer):
            apres = simuler_coup(bloquer, col, MOI)
            print(f"colonne {col} : bug {minimax_bug(apres, 1, False):9d}   "
                  f"correct {minimax(apres, 1, -inf, inf, False):9d}")
        """),
        *sol_md("""
        **Réponse.** `simuler_coup(grille, col, MOI)` pose **toujours mon**
        pion, même au tour de l'adversaire : dans cet arbre, l'adversaire ne
        joue jamais, donc ne gagne jamais, et aucun coup ne paraît dangereux.
        Il faut `MOI if maximise else ADV`.
        """),
        m("""
        ### Combien de positions l'élagage évite-t-il ?

        On compte les appels à `simuler_coup` (une position visitée chacun)
        pour minimax sans élagage (`alpha`/`beta` jamais utilisés) et avec.
        """),
        c('''
        def minimax_naif(grille, profondeur, maximise):
            """Minimax sans élagage."""
            if est_gagnant(grille, MOI):
                return GAGNE + profondeur
            if est_gagnant(grille, ADV):
                return -GAGNE - profondeur
            coups = coups_valides(grille)
            if not coups:
                return 0
            if profondeur == 0:
                return evaluer(grille)
            joueur = MOI if maximise else ADV
            valeurs = [minimax_naif(simuler_coup(grille, col, joueur), profondeur - 1,
                                    not maximise) for col in coups]
            return max(valeurs) if maximise else min(valeurs)


        def compter_positions(fonction, *args):
            """Le nombre d'appels à simuler_coup pendant fonction(*args)."""
            global simuler_coup
            original, compteur = simuler_coup, [0]

            def compte(*a):
                compteur[0] += 1
                return original(*a)

            simuler_coup = compte
            try:
                fonction(*args)
            finally:
                simuler_coup = original
            return compteur[0]


        inf = float("inf")
        depart = simuler_coup(np.zeros((6, 7), dtype=int), 3, ADV)
        for p in (2, 3, 4):
            naif = compter_positions(minimax_naif, depart, p, True)
            elague = compter_positions(minimax, depart, p, -inf, inf, True)
            print(f"profondeur {p} : {naif:6d} positions sans élagage, {elague:5d} avec")
        '''),
        m("""
        Sans élagage, il y a environ $b^d$ positions pour un facteur de
        branchement $b$ (7 colonnes) et une profondeur $d$ : chaque niveau
        multiplie le travail par 7. Avec un bon ordre des coups, l'élagage
        ramène ce nombre vers $b^{d/2}$ : à temps égal, on va presque deux
        fois plus profond.

        ### Exercice 5 — `AgentMinimax`

        Écrivez l'agent : `__init__(self, profondeur=4)` retient la profondeur
        (sans appeler `super().__init__`, `AgentAleatoire` n'en a pas) ;
        `choose_action` trie les coups autorisés « centre d'abord » — avec un
        tirage au hasard entre colonnes à même distance du centre, pour ne pas
        jouer toujours la même partie : `coups.sort(key=lambda col:
        (abs(col - 3), random.random()))` — puis renvoie la colonne de
        `meilleur_coup`.
        """),
        ex('''
        # À vous
        class AgentMinimax(AgentAleatoire):
            """Minimax alpha-bêta à profondeur fixe."""

            def __init__(self, profondeur=4):
                ...

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                ...
        ''', L4_AGENT_MINIMAX),
        c("""
        agent = AgentMinimax(3)
        obs, masque = observation_depuis_grille(bloquer), masque_depuis_grille(bloquer)
        assert agent.choose_action(obs, action_mask=masque) == 3
        assert agent.choose_action(obs, terminated=True, action_mask=masque) is None
        bilan = jouer_match(AgentMinimax(4), AgentRegles(), 10)
        print("minimax(4) contre règles :", bilan["victoires"], "victoires,",
              bilan["defaites"], "défaites sur 10")
        assert bilan["victoires"] >= 7
        print("✅ Exercice 5 réussi")
        """),
        m(f"""
        ---

        ## 4. Profondeur contre temps

        Plus profond = plus fort, mais plus lent. Mesurons le temps par coup
        pour chaque profondeur, sur des parties contre `AgentRegles`.
        La limite de la plateforme est **{TEMPS_MAX} s** par appel,
        aller-retour compris, sur une machine qui peut être plus lente que
        celle-ci : visez **moins de 0,1 s** de calcul.
        """),
        c("""
        resultats = []
        for p in range(1, 6):
            chrono = AgentChronometre(AgentMinimax(p))
            bilan = jouer_match(chrono, AgentRegles(), 4)
            d = np.array(chrono.durees)
            resultats.append((p, d.mean(), d.max(), bilan["victoires"]))
            print(f"profondeur {p} : moyenne {d.mean() * 1000:7.1f} ms, "
                  f"max {d.max() * 1000:7.1f} ms, {bilan['victoires']}/4 victoires")

        p, moy, mx, _ = zip(*resultats)
        plt.semilogy(p, mx, "o-", label="max", color="#c0392b")
        plt.semilogy(p, moy, "o-", label="moyenne", color="#2f5d9e")
        plt.axhline(0.25, ls="--", color="gray", label="limite ML-Arena (0,25 s)")
        plt.xlabel("profondeur")
        plt.ylabel("secondes par coup (échelle log)")
        plt.legend()
        plt.show()
        """),
        *sol_md("""
        **Lecture.** Le temps est multiplié par un facteur constant à chaque
        niveau (une droite en échelle logarithmique) : c'est le $b^d$. Le
        **maximum** compte plus que la moyenne, car un seul appel trop long
        fait perdre le match ; il survient en milieu de partie, quand
        beaucoup de colonnes sont ouvertes. Une profondeur fixe est donc
        risquée : trop prudente en début et en fin de partie, trop lente au
        milieu.
        """),
        m("""
        ### Où passe le temps ?

        Le profileur `cProfile` compte le temps passé dans chaque fonction.
        """),
        c("""
        import cProfile
        import pstats

        milieu = grille_depuis_texte('''
        . . . . . . .
        . . . . . . .
        . . . X . . .
        . . O O . . .
        . X X O . . .
        O X O X . . .
        ''')
        cProfile.run("meilleur_coup(milieu, coups_valides(milieu), 4)", "profil.out")
        pstats.Stats("profil.out").sort_stats("tottime").print_stats(6)
        """),
        *sol_md("""
        **Lecture.** L'essentiel du temps est dans `est_gagnant` et
        `evaluer`, qui parcourent les 69 fenêtres à chaque position. Pistes
        d'optimisation : ne vérifier la victoire que sur les fenêtres qui
        passent par le dernier pion posé ; mettre en cache les positions déjà
        évaluées ; représenter la grille par deux entiers (*bitboards*), où
        une victoire se teste en quelques opérations binaires.
        """),
        m(f"""
        ---

        ## 5. Tenir le temps : l'approfondissement itératif

        Au lieu de fixer la profondeur, on cherche à profondeur 1, puis 2, puis
        3… tant qu'il reste du temps. Quand le budget est épuisé, `minimax`
        lève `TempsEcoule` : on abandonne la profondeur en cours et on joue le
        meilleur coup de la **dernière profondeur terminée**. Bonus : on essaie
        d'abord le meilleur coup de la profondeur précédente, ce qui améliore
        l'élagage.

        ### Lire du code 2

        Le mécanisme, sur un exemple jouet où le « temps » est remplacé par un
        compteur. Que va afficher ce code ?
        """),
        c("""
        def recherche(budget):
            meilleur = None
            for profondeur in range(1, 10):
                try:
                    if profondeur * profondeur > budget:   # coût de cette profondeur
                        raise TempsEcoule
                    meilleur = profondeur
                except TempsEcoule:
                    break
            return meilleur

        print(recherche(10), recherche(50), recherche(0))
        """),
        *sol_md("""
        **Réponse.** `3 7 None`. Les profondeurs 1, 2, 3 « coûtent » 1, 4, 9
        ≤ 10 ; la 4 coûterait 16 : l'exception interrompt la boucle et l'on
        garde 3, la dernière profondeur **terminée**. Avec un budget nul, rien
        n'est terminé : d'où, dans `choisir_coup`, une valeur de départ
        (`choix = ordre[0]`) qui est toujours un coup autorisé.
        """),
        m("""
        ### Exercice 6 — `choisir_coup`

        Complétez la boucle : pour chaque profondeur, appelez
        `meilleur_coup(grille, ordre, profondeur, fin)` dans un `try` ; sur
        `TempsEcoule`, sortez de la boucle ; sinon retenez le coup, placez-le
        en tête de `ordre`, et arrêtez si la valeur est une victoire ou une
        défaite forcée (`abs(valeur) >= GAGNE`).
        """),
        ex('''
        # À vous
        def choisir_coup(grille, coups, temps_max):
            """Approfondissement itératif : profondeur 1, 2, 3... tant qu'il reste du temps."""
            fin = time.perf_counter() + temps_max
            ordre = sorted(coups, key=lambda col: (abs(col - 3), random.random()))
            choix = ordre[0]
            cases_vides = int((grille == VIDE).sum())
            for profondeur in range(1, cases_vides + 1):
                ...
            return choix
        ''', L4_CHOISIR),
        c("""
        for grille in (np.zeros((6, 7), dtype=int), milieu, bloquer):
            debut = time.perf_counter()
            col = choisir_coup(grille, coups_valides(grille), 0.08)
            duree = time.perf_counter() - debut
            assert col in coups_valides(grille)
            assert duree < 0.15, f"{duree:.3f} s : le budget n'est pas respecté"
        assert choisir_coup(bloquer, coups_valides(bloquer), 0.08) == 3
        print("✅ Exercice 6 réussi")
        """),
        m(f"""
        ---

        ## 6. Le fichier `agent.py`

        La plateforme ne reçoit qu'un fichier : `agent.py`, qui doit contenir
        **tout** ce dont l'agent a besoin, et n'importer que la bibliothèque
        standard et NumPy (PettingZoo n'est **pas** installé côté agent). La
        cellule suivante l'écrit avec `%%writefile` ; `TEMPS_RECHERCHE` vaut
        0,08 s, une marge confortable sous les {TEMPS_MAX} s.
        """),
        c(_agent_py(CORRECTION)),
        m("""
        ### Exercice 7 — Tester `agent.py` comme la plateforme

        La cellule suivante importe le fichier, puis le fait jouer avec
        `jouer_match` (les appels de ML-Arena) : 100 parties contre
        l'aléatoire, 20 contre l'agent à règles, en mesurant **chaque** coup.
        Elle vérifie aussi qu'une observation sous forme de listes est
        acceptée. Votre `agent.py` doit gagner au moins 95 % contre
        l'aléatoire, battre l'agent à règles le plus souvent, et ne jamais
        dépasser 0,2 s.
        """),
        c("""
        import importlib

        import agent as mon_agent

        importlib.reload(mon_agent)
        assert "pettingzoo" not in open("agent.py", encoding="utf-8").read()

        a = mon_agent.Agent()
        assert a.setup({}, {}) is True and a.reset("player_1", 0) is True
        vide = np.zeros((6, 7, 2), dtype=np.int8)
        assert a.choose_action(vide.tolist(), 0.0, False, False, {}, [1] * 7) in range(7)

        chrono = AgentChronometre(mon_agent.Agent())
        contre_aleatoire = jouer_match(chrono, AgentAleatoire(), 100)
        contre_regles = jouer_match(chrono, AgentRegles(), 20)
        d = np.array(chrono.durees)
        print("contre l'aléatoire :", {k: v for k, v in contre_aleatoire.items() if k != "coups"})
        print("contre les règles  :", {k: v for k, v in contre_regles.items() if k != "coups"})
        print(f"temps par coup : moyenne {d.mean() * 1000:.0f} ms, max {d.max() * 1000:.0f} ms")
        assert contre_aleatoire["victoires"] >= 95
        assert contre_regles["victoires"] > contre_regles["defaites"]
        assert d.max() < 0.2, "trop lent : baissez TEMPS_RECHERCHE"
        print("✅ Exercice 7 réussi")
        """),
        m(f"""
        ---

        ## 7. Soumettre sur ML-Arena

        Le challenge : [{CHALLENGE_URL}]({CHALLENGE_URL}). Sa page décrit le
        contrat, les contraintes et le classement ELO.

        Il faut une clé d'API : sur ML-Arena, **Profil → API Keys** (elle
        commence par `mlk_user_`). **Ne collez jamais la clé dans une
        cellule** : un notebook se partage avec son code et ses sorties. Dans
        Colab, ajoutez-la dans le panneau **Secrets** (l'icône de clé à
        gauche) sous le nom `MLARENA_API_KEY` et autorisez ce notebook à y
        accéder. Hors de Colab, définissez la variable d'environnement
        `MLARENA_API_KEY` avant de lancer Jupyter.
        """),
        c(f"""
        import os

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "mlarena-sdk==3.0.0"], check=True)
            from google.colab import userdata
            os.environ["MLARENA_API_KEY"] = userdata.get("MLARENA_API_KEY")

        CHALLENGE_ID = {CHALLENGE_ID}
        NOM_GROUPE = "mon-binome"   # À vous : le nom de votre groupe
        """),
        m("""
        `client.submit` crée la soumission, envoie `agent.py` et la déploie.
        La plateforme joue d'abord un match de vérification ; si l'agent
        plante, dépasse le temps ou joue un coup interdit, la soumission
        échoue et `last_status_message` dit pourquoi.

        **Exécutez cette cellule une seule fois** : chaque exécution crée une
        nouvelle soumission.
        """),
        c("""
        if CHALLENGE_ID and os.environ.get("MLARENA_API_KEY"):
            import mlarena

            client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
            resultat = client.submit(challenge_id=CHALLENGE_ID, files=["agent.py"],
                                     submission_name=f"puissance4-{NOM_GROUPE}")
            print(resultat)
        else:
            print("Pas de clé MLARENA_API_KEY : soumission non envoyée.")
        """),
        m("""
        Le statut passe par `deploy_queue`, `deploy_run`, puis `active`. Une
        fois active, la soumission joue des matchs contre celles des autres
        groupes et sa note ELO évolue.
        """),
        c("""
        if CHALLENGE_ID and os.environ.get("MLARENA_API_KEY"):
            statut = client.status()
            print(statut["status"], "|", statut["last_status_message"])
            print(client.leaderboard(CHALLENGE_ID).head(10))
        """),
        m("""
        ---

        ## 8. Pour aller plus loin (facultatif) : Monte Carlo Tree Search

        MCTS n'a pas besoin de fonction d'évaluation : il joue des parties
        **au hasard** depuis les positions candidates et garde les coups qui
        gagnent le plus souvent. Chaque itération a quatre étapes :
        **sélection** (descendre l'arbre en choisissant l'enfant de meilleur
        score UCB1), **expansion** (ajouter un coup pas encore essayé),
        **simulation** (finir la partie au hasard), **rétropropagation**
        (mettre à jour victoires et visites jusqu'à la racine).

        Le score UCB1 d'un enfant équilibre **exploitation** (son taux de
        victoire) et **exploration** (les enfants peu visités) :

        $$\\mathrm{UCB1} = \\frac{g}{n} + c\\sqrt{\\frac{\\ln N}{n}}$$

        avec $g$ les victoires et $n$ les visites de l'enfant, $N$ les visites
        du parent, et $c$ la constante d'exploration (souvent $\\sqrt{2}$).

        ### Exercice 8 — `ucb1`

        Renvoyez `float("inf")` si l'enfant n'a jamais été visité (il faut
        l'essayer), sinon la formule.
        """),
        ex('''
        # À vous
        import math


        def ucb1(gains, visites, visites_parent, c=1.41):
            """Le score UCB1 d'un enfant de l'arbre MCTS."""
            ...
        ''', '''
        import math


        def ucb1(gains, visites, visites_parent, c=1.41):
            """Le score UCB1 d'un enfant de l'arbre MCTS."""
            if visites == 0:
                return float("inf")
            return gains / visites + c * math.sqrt(math.log(visites_parent) / visites)
        '''),
        c("""
        assert ucb1(0, 0, 10) == float("inf")
        assert abs(ucb1(3, 4, 10) - (0.75 + 1.41 * math.sqrt(math.log(10) / 4))) < 1e-9
        assert ucb1(3, 4, 10, c=0) == 0.75, "c = 0 : exploitation pure"
        assert ucb1(1, 2, 100) > ucb1(1, 2, 10), "plus le parent est visité, plus on explore"
        print("✅ Exercice 8 réussi")
        """),
        m("""
        Un MCTS complet, construit sur `ucb1` (donné). Chaque nœud retient la
        grille, le joueur qui vient de jouer, ses enfants, ses visites et ses
        victoires (du point de vue de ce joueur).
        """),
        c('''
        def autre(joueur):
            return ADV if joueur == MOI else MOI


        class Noeud:
            def __init__(self, grille, joueur, parent=None, coup=None):
                self.grille, self.joueur = grille, joueur  # joueur : celui qui a joué coup
                self.parent, self.coup = parent, coup
                self.enfants, self.visites, self.gains = [], 0, 0.0
                fini = est_gagnant(grille, joueur) or not coups_valides(grille)
                self.a_essayer = [] if fini else coups_valides(grille)


        def partie_au_hasard(grille, dernier):
            """Finit la partie au hasard ; renvoie le gagnant ou None (nulle)."""
            if est_gagnant(grille, dernier):
                return dernier
            g, joueur = grille.copy(), dernier
            while coups_valides(g):
                joueur = autre(joueur)
                col = random.choice(coups_valides(g))
                g[ligne_libre(g, col), col] = joueur
                if est_gagnant(g, joueur):
                    return joueur
            return None


        def mcts(grille, temps_max, c=1.41):
            racine = Noeud(grille, ADV)       # ADV vient de jouer : c'est à moi
            fin = time.perf_counter() + temps_max
            while time.perf_counter() < fin:
                noeud = racine
                while not noeud.a_essayer and noeud.enfants:        # 1. sélection
                    parent = noeud
                    noeud = max(parent.enfants,
                                key=lambda e: ucb1(e.gains, e.visites, parent.visites, c))
                if noeud.a_essayer:                                 # 2. expansion
                    col = noeud.a_essayer.pop(random.randrange(len(noeud.a_essayer)))
                    joueur = autre(noeud.joueur)
                    enfant = Noeud(simuler_coup(noeud.grille, col, joueur), joueur,
                                   noeud, col)
                    noeud.enfants.append(enfant)
                    noeud = enfant
                gagnant = partie_au_hasard(noeud.grille, noeud.joueur)  # 3. simulation
                while noeud is not None:                            # 4. rétropropagation
                    noeud.visites += 1
                    noeud.gains += 1.0 if gagnant == noeud.joueur else (
                        0.5 if gagnant is None else 0.0)
                    noeud = noeud.parent
            return max(racine.enfants, key=lambda e: e.visites).coup


        class AgentMCTS(AgentAleatoire):
            def __init__(self, temps_max=0.08):
                self.temps_max = temps_max

            def choose_action(self, observation, reward=0.0, terminated=False,
                              truncated=False, info=None, action_mask=None):
                if terminated or truncated:
                    return None
                return mcts(vers_grille(observation), self.temps_max)


        taux = tournoi({"aléatoire": AgentAleatoire(), "règles": AgentRegles(),
                        "MCTS": AgentMCTS(0.08), "minimax": AgentMinimax(3)}, 6)
        print(taux.round(2))
        '''),
        m("""
        | | Minimax | MCTS |
        |---|---|---|
        | évaluation | une heuristique à écrire | des parties au hasard |
        | temps | profondeur fixe (ou itérative) | s'arrête n'importe quand |
        | force | optimal si l'évaluation est parfaite | s'améliore avec le nombre de simulations |
        | branchement élevé | souffre ($b^d$) | supporte bien |

        En Python pur, MCTS ne fait que quelques centaines de simulations en
        0,08 s : c'est peu, et il reste souvent derrière un minimax bien
        élagué. Les programmes les plus forts combinent les deux idées :
        vérifier les coups gagnants et les blocages immédiats, puis chercher
        (AlphaGo associe MCTS et réseaux de neurones). Avec des *bitboards*,
        les deux méthodes deviennent des dizaines de fois plus rapides.
        """),
        m("""
        ---

        **Bilan.** Votre agent anticipe, tient la limite de temps et joue sur
        ML-Arena. Pour le projet : mettez ce code dans votre dépôt GitHub,
        découpé en modules, avec ses tests `pytest`, un `pylint` propre et un
        `README.md` qui explique vos choix.
        """),
    ]
    return cells


# =========================================================================

LABS = {1: lab1, 2: lab2, 3: lab3, 4: lab4}


def build(n: int, correction: bool) -> dict:
    global CORRECTION
    CORRECTION = correction
    nb._COUNTER = itertools.count()     # byte-identical on every regeneration
    return nb.notebook(LABS[n]())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in LABS:
        for correction in (False, True):
            name = f"pym-s8-lab-{n}" + ("-correction" if correction else "")
            path = OUT / f"{name}.ipynb"
            path.write_text(json.dumps(build(n, correction), indent=1,
                                       ensure_ascii=False) + "\n")
            print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
