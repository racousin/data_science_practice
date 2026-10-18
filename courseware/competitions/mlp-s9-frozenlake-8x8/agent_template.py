"""
FrozenLake 8x8 agent template — flex_v1 contract.

The platform drives your agent:
  1. Agent()                                   zero-arg constructor
  2. setup(observation_space, action_space)    once, before the first episode
  3. choose_action(observation, reward, terminated, truncated, info, action_mask)
     once per step; return an action in {0: left, 1: down, 2: right, 3: up},
     or None once the episode ends

observation is the cell index, 0..63 (row * 8 + column).
Upload q_table.npy (shape (64, 4)) next to agent.py.
"""
from pathlib import Path

import numpy as np


class Agent:
    def __init__(self):
        self.q = np.load(Path(__file__).parent / "q_table.npy")

    def setup(self, observation_space, action_space):
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return int(np.argmax(self.q[int(observation)]))
