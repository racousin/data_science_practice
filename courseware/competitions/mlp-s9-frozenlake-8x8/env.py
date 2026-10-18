"""Gymnasium · FrozenLake-v1 8x8 — flex_v1 via flexkit.

The slippery 8x8 lake (the Gymnasium default `is_slippery=True`): the agent
moves in the intended direction with probability 1/3 only. Reward 1 for
reaching the goal, 0 otherwise, so the mean episode reward is the success rate.
`flexkit.run_gymnasium_eval` drives the episodes; the episode count comes from
the challenge's `episode_budget_brackets`.
"""
import gymnasium as gym

import flexkit


class Env:
    def __init__(self, is_evaluation: bool = False):
        self.is_evaluation = is_evaluation
        self.env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="rgb_array")

    def evaluate(self, agents, agent_infos):
        return flexkit.run_gymnasium_eval(self, agents, agent_infos)
