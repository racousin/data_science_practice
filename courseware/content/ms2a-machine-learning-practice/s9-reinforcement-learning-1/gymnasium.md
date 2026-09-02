# Gymnasium

Gymnasium is the API every single-agent RL environment implements. It is four
methods and one five-tuple, and getting one element of that tuple wrong quietly
corrupts your value estimates.

<!-- notes: 25 minutes. Run the random-agent loop live on Frozen Lake before
explaining anything. Spend real time on terminated vs truncated — it is the
only thing on these slides that costs them a working agent. End on the
ML-Arena contract so the lab and Lab 10 have somewhere to land. -->

---

## The contract

![Frozen Lake](assets/rl/frozen_lake.gif)

```python
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Maintained by the Farama Foundation, successor to OpenAI Gym. A tutorial where
`step` returns four values predates 2022 — the signature changed, and the change
matters.

---

## reset

```python
obs, info = env.reset(seed=42)
```

Returns a **tuple**, not a bare observation. `info` is a dict — usually empty,
sometimes carrying an action mask or diagnostics.

`seed` is passed to `reset`, not to a separate `env.seed()` call. Pass it once
at the start of a run; passing the same seed on every reset gives you the same
episode forever, which looks like an agent that has learned and has not.

---

## step

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| Field | Meaning |
|---|---|
| `obs` | the next observation |
| `reward` | scalar reward for this transition |
| `terminated` | the MDP reached a terminal state |
| `truncated` | an external limit stopped the episode |
| `info` | diagnostics, never for the agent's decision |

`info` is for you, not for the policy. Reading a privileged field out of `info`
is how a benchmark result stops meaning anything.

---

## terminated is not truncated

**terminated** — the pole fell, the agent drowned, the game ended. The MDP says
there is no future. $V(s') = 0$ by definition.

**truncated** — the time limit fired at step 500. The MDP has a perfectly good
future; you stopped watching.

```python
target = r + gamma * Q[s2].max() * (not terminated)
```

Bootstrap through truncation, never through termination. Collapsing the two into
one `done` flag is the most common bug in ported Gym code: the agent learns that
surviving to the time limit is worth zero, and stops trying.

---

## Spaces

```python
print(env.observation_space)   # Discrete(16)
print(env.action_space)        # Discrete(4)
```

| Space | Shape | Example |
|---|---|---|
| `Discrete(n)` | one integer in $[0, n)$ | Frozen Lake state, four moves |
| `Box(low, high, shape)` | float array | CartPole's 4 floats, an RGB frame |
| `MultiDiscrete([a, b])` | vector of integers | independent selectors |
| `MultiBinary(n)` | 0/1 vector | a set of toggles |

The spaces are the type signature of the agent. `Discrete` observations index a
table — this session. `Box` observations need a function approximator — Session
10.

---

## Read the space, do not hardcode it

```python
assert isinstance(env.observation_space, gym.spaces.Discrete)
Q = np.zeros((env.observation_space.n, env.action_space.n))
```

An agent that hardcodes `16` works on 4x4 Frozen Lake and silently
index-errors — or worse, silently truncates — on 8x8. Read `n` from the space
and assert the type you expect at construction.

`env.action_space.sample()` gives a valid random action for any space, which is
your baseline and your smoke test.

---

## Wrappers

A wrapper is an environment that delegates to another environment and changes
one thing.

```python
env = gym.make("CartPole-v1")
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.NormalizeObservation(env)
```

They compose, and the order is the order of application: the outermost wrapper
sees the innermost environment's output last. `gym.make` already applies
`TimeLimit` for registered environments — which is where `truncated` comes from.

`env.unwrapped` reaches the underlying environment, and is how you got `P` in
the dynamic programming lesson.

---

## Vectorized environments

```python
envs = gym.make_vec("CartPole-v1", num_envs=8)
obs, info = envs.reset(seed=0)
obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
```

`obs` is now a batch of 8. The point is throughput: environment stepping is
usually the bottleneck, and eight environments in a batch turn eight tiny policy
forward passes into one.

Vectorized environments auto-reset, so an episode boundary appears as a flag
rather than a call. Session 10 uses them properly; know they exist.

---

## Seeding

```python
obs, info = env.reset(seed=SEED)
env.action_space.seed(SEED)
rng = np.random.default_rng(SEED)
```

Three separate sources of randomness: the environment's dynamics, the action
space sampler, and your own agent. Seed all three or your run is not
reproducible, and "not reproducible" in RL means you cannot tell a real
improvement from a lucky seed.

One seed is never evidence. Report a mean and a spread over at least five.

---

## A complete random agent

```python
env = gym.make("FrozenLake-v1")
obs, info = env.reset(seed=0)
env.action_space.seed(0)
total = 0.0
for _ in range(1000):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    total += r
    if term or trunc:
        obs, info = env.reset()
```

Run this first, every time, on any new environment. It tells you the reward
scale, the episode length, and the floor your agent has to beat. On 4x4 Frozen
Lake the floor is about 0.014 successes per episode.

---

## The ML-Arena shape

An ML-Arena `gymnasium` competition does not run your training loop. It imports
your `agent.py`, constructs `Agent()` with no arguments, calls `setup` once, and
then calls `choose_action` per step.

```python
from flexkit.spaces import decode_space

class Agent:
    def setup(self, observation_space, action_space):
        self.action_space = decode_space(action_space)
        return True
```

The space arguments arrive as dicts, not as objects — `decode_space` turns them
back into Gymnasium spaces.

---

## choose_action

```python
    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return int(self.q_table[observation].argmax())
```

Inference only. Train offline, save the Q-table beside `agent.py`, load it in
`__init__` or `setup`, and act greedily — there is no exploration at evaluation
time and no learning between steps.

> The competition scores the mean episode reward over a fixed number of seeded
> episodes. An agent that raises inside `setup` is marked broken and scores
> nothing, so validate what you load and fail loudly in your own tests, not on
> the leaderboard.
