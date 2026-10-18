# Frozen Lake 8x8

_Source_: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>

Cross a frozen lake from the start **S** (top left) to the goal **G** (bottom right)
without falling into a hole **H**. The ice is **slippery**: the agent moves in the
intended direction with probability 1/3, and in each of the two perpendicular
directions with probability 1/3. The best policy is therefore not the shortest path.

|  |  |
| --- | --- |
| Action Space | `Discrete(4)` — 0 left, 1 down, 2 right, 3 up |
| Observation Space | `Discrete(64)` — the cell index, `row * 8 + column` |
| import | `gymnasium.make("FrozenLake-v1", map_name="8x8")` (slippery, the default) |

```text
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFFFHFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

## Reward and score

Reward is **1** when the agent reaches the goal and **0** otherwise. An episode ends
in a hole, at the goal, or after 200 steps. Your score is the **mean reward over 100
episodes**, i.e. your success rate.

| Policy | Success rate |
| --- | --- |
| random actions | 0.00 |
| always move right (the benchmark) | ≈ 0.23 |
| a tabular Q-learning agent, 10,000 training episodes with ε decay | ≈ 0.60 |
| the optimal policy (value iteration on the known model) | ≈ 0.63–0.74 |

## Your agent

Train a Q-table offline (in the starter notebook), save it with
`np.save("q_table.npy", q)`, and submit **two files**: `agent.py` and `q_table.npy`.
The platform calls `Agent()`, then `setup(observation_space, action_space)` once, then
`choose_action(observation, ...)` at each step; `observation` is the cell index.
Return `None` once the episode is over.

```python
class Agent:
    def __init__(self):
        self.q = np.load(Path(__file__).parent / "q_table.npy")

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return int(np.argmax(self.q[int(observation)]))
```

## Quick start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s9-frozenlake-q-learning.ipynb)

The starter notebook builds the Q-learning agent on the 4x4 lake, moves it to the
8x8 lake, and submits it to this challenge (`CHALLENGE_ID = __CHALLENGE_ID__`).
