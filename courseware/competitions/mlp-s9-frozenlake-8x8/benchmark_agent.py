"""Benchmark for FrozenLake 8x8: always move right.

No learning. On the slippery lake it reaches the goal in about 23% of the
episodes (measured over 2,000 seeded episodes with gymnasium 1.3.0); a
trained Q-table should reach about 60%.
"""

RIGHT = 2


class Agent:
    def setup(self, observation_space, action_space):
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return RIGHT
