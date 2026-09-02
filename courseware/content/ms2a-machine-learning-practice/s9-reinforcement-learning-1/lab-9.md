# Lab 9 — Tabular Agent on Frozen Lake

Implement Q-learning and SARSA from scratch, compare them honestly on the same
seeds, and read the learned policy back out as a grid you can explain.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.
**No RL library** — `gymnasium` and `numpy` only.

<!-- notes: They will get the terminal bootstrap wrong; the unit test in Part D
catches it, so push them to write that test early. Circulate during Part C —
"one seed is not a result" is the sentence to repeat. Keep an eye on the clock,
Part E is the part they skip and it is where the marks are. -->

---

## Setup

Work in your project repository, on a branch.

```text
src/rl/
    __init__.py
    env.py          <- make_env(seed, map_name, slippery)
    agents.py       <- QLearning, Sarsa — one class each
    train.py        <- run(agent, seeds, episodes) -> curves
    report.py       <- plots, ablation table, policy grid
tests/test_rl.py
runs/               <- gitignored: curves, Q-tables
REPORT.md
```

`runs/` stays out of git. Committing a `.npy` of Q-values is an automatic
deduction; committing the plot that summarises them is what you want.

---

## Part A — the floor (5 min)

```python
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
```

Before writing an agent, measure the random baseline over 1000 episodes with a
fixed seed, and write the number in `REPORT.md`.

Also record, in one line each: the size of $S$, the size of $A$, the reward
structure, and whether an episode can end without reaching the goal. You cannot
interpret a learning curve without those four facts.

---

## Part B — two agents (15 min)

One class each, sharing an `epsilon_greedy` helper and a `Q` array of shape
`(n_states, n_actions)`. The only difference between them is the target.

```python
class QLearning:
    def update(self, s, a, r, s2, a2, terminated):
        target = r + self.gamma * self.Q[s2].max() * (not terminated)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
```

SARSA uses `self.Q[s2, a2]` instead. Requirements:

- `* (not terminated)` present and correct in both
- `truncated` does **not** stop the bootstrap
- $\epsilon$ decays on a schedule you can state in one line
- no `try`/`except` anywhere in the update path

---

## Part C — compare, with error bars (10 min)

Train both agents on the **same** list of at least 5 seeds, for the same number
of episodes, with the same $\alpha$, $\gamma$ and $\epsilon$ schedule.

```python
curves = np.array([run(agent_cls, seed, n_episodes) for seed in SEEDS])
mean, sd = curves.mean(axis=0), curves.std(axis=0)
plt.fill_between(x, mean - sd, mean + sd, alpha=0.2)
plt.plot(x, mean)
```

Smooth each curve with a rolling mean over 100 episodes — the raw per-episode
return on Frozen Lake is 0 or 1 and plots as noise. One figure, both agents,
a shaded band per agent. Save it to `runs/` and embed it in `REPORT.md`.

If the bands overlap everywhere, say so. "SARSA is better" from a single seed is
the failure this part exists to prevent.

---

## Part D — ablation (5 min)

Vary $\alpha$ and the $\epsilon$ schedule, holding everything else fixed. Report
the mean final success rate over your seeds, plus its spread:

| $\alpha$ | $\epsilon$ schedule | Mean success | Std |
|---|---|---|---|
| 0.05 | decay 0.9995, floor 0.05 | | |
| 0.10 | decay 0.9995, floor 0.05 | | |
| 0.50 | decay 0.9995, floor 0.05 | | |
| 0.10 | fixed 0.01 | | |
| 0.10 | fixed 0.50 | | |

Five rows, one sentence under the table naming which setting broke and why.
A table with no losing row means the sweep was too narrow.

---

## Part E — read the policy (5 min)

Print the greedy policy of your best Q-table as a 4x4 grid of arrows, with the
holes and the goal marked.

```python
ARROWS = "<v>^"
grid = np.array([ARROWS[a] for a in Q.argmax(axis=1)]).reshape(4, 4)
```

Then write **one paragraph** in `REPORT.md` explaining one cell that looks
wrong. On slippery Frozen Lake the optimal policy frequently points *away* from
the goal, because the intended move has probability 0.33 and the alternative
slides you into a hole. Find such a cell and justify it.

An arrow grid with no interpretation scores zero for this part.

---

## Required tests

```python
def test_q_update_matches_hand_computation():
    """Q=0 except Q[3,1]=2.0; alpha=0.5, gamma=0.9, r=1, s'=3, not terminal.
    Target is 1 + 0.9*2.0 = 2.8, so Q[0,0] becomes 0 + 0.5*2.8 = 1.4."""

def test_greedy_policy_on_solved_table_reaches_goal():
    """With a hand-built optimal Q-table on the deterministic 4x4 map,
    the greedy rollout terminates with reward 1.0 within 20 steps."""

def test_epsilon_follows_the_declared_schedule():
    """eps(0) == 1.0, eps(2000) == max(floor, 1.0 * decay**2000), monotone."""

def test_two_runs_with_the_same_seed_are_identical():
    """Same seed, same hyperparameters -> bitwise-equal Q-tables."""
```

Docstrings only in the deliverable stubs — you write the bodies. The first test
is the one that catches the terminal-bootstrap bug, so write it before the
agent, not after.

---

## Pull request

The description states:

- the random baseline, and the final success rate of each agent
- the number of seeds and the number of episodes per seed
- which of Q-learning and SARSA won, and whether the bands actually separate
- the one policy cell you explain in Part E
- one thing you would change with another hour of compute

---

## Grading

| Criterion | Weight |
|---|---|
| Q-learning and SARSA correct, no library | 25% |
| Terminal bootstrap handled correctly | 10% |
| Comparison over ≥5 shared seeds with a band | 20% |
| Ablation table complete, with a losing row | 15% |
| Policy grid plus a real interpretation | 15% |
| Four tests passing | 15% |

---

## Automatic deductions

- an RL library used for the agent (`stable-baselines3`, `rllib`, `tianshou`)
- a single seed reported as a result
- `Q-tables`, `.npy` or `runs/` committed to git
- a bare `except` in the training loop
- hardcoded `16` or `4` instead of reading the space
- a plot with no axis labels

---

## Carry it forward

The agent you just wrote is the shape ML-Arena expects: a Q-table, a greedy
`choose_action`, no learning at evaluation time. Wrapping it in the `Agent`
class from the Gymnasium lesson is a ten-line change.

Session 10 replaces the table with a network and the same loop survives intact.
Keep `train.py` — you will run it again with a different `update`.
