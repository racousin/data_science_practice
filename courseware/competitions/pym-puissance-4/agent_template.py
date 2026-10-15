"""
Modèle d'agent — Puissance 4 (Python pour la médecine).

La plateforme pilote votre agent dans cet ordre :
  1. Agent()                                   constructeur sans argument
  2. setup(observation_space, action_space)    une fois, avant la première partie
     (les espaces arrivent encodés en dictionnaire — décodables avec
     flexkit.spaces.decode_space)
  3. reset(env_player_name, episode_index)     au début de CHAQUE partie
     (OBLIGATOIRE — votre place, "player_0" ou "player_1", change d'une
     partie à l'autre)
  4. choose_action(observation, reward, terminated, truncated, info, action_mask)
     à chacun de vos tours ; renvoie la colonne à jouer (0 à 6) et respecte
     action_mask (1 = colonne non pleine)

observation : tableau (6, 7, 2) ; observation[:, :, 0] = vos jetons,
observation[:, :, 1] = ceux de l'adversaire ; la ligne 0 est en haut.
"""
import random


class Agent:
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.env_player_name = ""
        self.episode_index = 0

    def setup(self, observation_space, action_space):
        # flexkit est fourni par la plateforme (inutile pour exécuter ce
        # fichier en local) ; on l'importe ici pour que `import agent`
        # fonctionne partout.
        from flexkit.spaces import decode_space
        self.observation_space = decode_space(observation_space)
        self.action_space = decode_space(action_space)
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        self.episode_index = episode_index
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        if action_mask is not None:
            legal = [i for i, ok in enumerate(action_mask) if ok]
            if legal:
                # À FAIRE : remplacer le choix au hasard par votre stratégie
                # parmi les colonnes autorisées.
                return random.choice(legal)
        return self.action_space.sample()
