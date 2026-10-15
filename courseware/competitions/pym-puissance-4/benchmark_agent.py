"""Agent de référence (benchmark) — Puissance 4, Python pour la médecine.

Règles simples, dans l'ordre :
  1. gagner tout de suite si un coup gagnant existe ;
  2. sinon bloquer un coup gagnant de l'adversaire ;
  3. sinon jouer la colonne libre la plus centrale (3, 2, 4, 1, 5, 0, 6),
     en tirant au hasard parmi les colonnes à égale distance du centre.

Observation : tableau (6, 7, 2). observation[:, :, 0] = mes jetons,
observation[:, :, 1] = jetons adverses ; la ligne 0 est en haut.
action_mask : 7 valeurs, 1 si la colonne n'est pas pleine.
"""
import random

ROWS, COLS = 6, 7


def drop_row(board, col):
    """Ligne où tombe un jeton joué en `col` (None si la colonne est pleine)."""
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == 0:
            return row
    return None


def is_win(board, player, row, col):
    """Vrai si poser `player` en (row, col) aligne quatre jetons."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 4:
            return True
    return False


class Agent:
    def __init__(self):
        self.env_player_name = ""

    def setup(self, observation_space, action_space):
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        # Plateau : 1 = moi, 2 = adversaire, 0 = vide.
        board = [[0] * COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                if observation[r][c][0]:
                    board[r][c] = 1
                elif observation[r][c][1]:
                    board[r][c] = 2
        legal = [c for c in range(COLS) if action_mask[c]]

        for player in (1, 2):  # 1 : gagner, 2 : bloquer
            for col in legal:
                row = drop_row(board, col)
                if is_win(board, player, row, col):
                    return col

        best = min(abs(col - 3) for col in legal)
        return random.choice([col for col in legal if abs(col - 3) == best])
