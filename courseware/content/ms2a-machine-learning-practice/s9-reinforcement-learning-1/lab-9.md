# Lab 9 — Tabular Agent on Frozen Lake

Implement Q-learning and SARSA from scratch, compare them honestly on the same
seeds, and read the learned policy back out as a grid you can explain.

**Time:** 45 minutes in class (Parts A to E), plus a take-home Part F.
**Deliverable:** a merged PR in your project repository, and an accepted
submission on competition 48. **No RL library** — `gymnasium` and `numpy` only.

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
fixed seed, and write the number in `REPORT.md`. It should land near **0.015**
successes per episode — over 20,000 episodes I measure 0.0151, and 1,000
episodes carries a standard error of about 0.004, so anything from 0.005 to 0.03
is the same number. If you get 0.2, you are not on the slippery map.

Also record, in one line each: the size of $S$, the size of $A$, the reward
structure, and whether an episode can end without reaching the goal. You cannot
interpret a learning curve without those four facts.

---

## Part A' — the ceiling (5 min)

You have the model here, so you can compute the answer before you learn it. Run
value iteration on `env.unwrapped.P` with $\gamma = 0.99$ to convergence, extract
the greedy policy with the `improve()` helper from the Dynamic Programming
lesson, and roll it out for 20,000 episodes.

```python
P, gamma = env.unwrapped.P, 0.99
nS, nA = env.observation_space.n, env.action_space.n
V = np.zeros(nS)
while True:
    V_new = np.array([max(sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                          for a in range(nA)) for s in range(nS)])
    if np.abs(V_new - V).max() < 1e-10:
        break
    V = V_new
pi_star = improve(P, V, gamma, nS, nA)
```

Write that success rate in `REPORT.md`: it is your **ceiling**, and on 4x4
slippery Frozen Lake it is **0.74** (I measure 0.7426 and 0.7399 on two
evaluation runs of 20,000 episodes each). Learning cannot beat a policy computed
from the exact model — so a Q-learner reporting 0.85 is being evaluated on too
few episodes, or not greedily. In the other direction, a Q-learner that plateaus
below **0.65** has a bug, not a hyperparameter problem. That pair of numbers is
the most useful thing in this lab: without them a broken agent and a badly tuned
one look identical.

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

Train both agents on the **same** list of at least 5 seeds, for **5,000 episodes
each**, with the same $\alpha$, $\gamma$ and $\epsilon$ schedule. 5,000 is enough
to reach the ceiling with $\alpha = 0.1$ and $\epsilon$ decaying 0.9995 to a
floor of 0.05, and both agents across five seeds is well under a minute of
compute (I measure 32 s) — so the table in Part D is comparable between
students, and between your own runs.

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

For calibration: at $\alpha = 0.1$ with the decaying schedule, five seeds of
5,000 episodes and a 2,000-episode greedy evaluation per seed, I measure
Q-learning at **0.746 ± 0.010** and SARSA at **0.688 ± 0.089**. Q-learning is
sitting on the 0.74 ceiling — the gap is inside the evaluation noise, which is
about ±0.01 at 2,000 episodes. SARSA's spread is one bad seed out of five, which
is precisely what the band in Part C exists to show. Your numbers will not match
mine exactly; the shape should.

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

## Part F — Ship it to CartPole (take-home, 15 min)

Session 9 attaches **CartPole-v1 (competition `48`)**. Frozen Lake has a
`Discrete(16)` observation and CartPole has a `Box` of four floats, so your
Q-table does not fit it: you need one extra step, a **discretisation**, and it
is the block below. Bin each of the four floats and index the table with the
resulting tuple.

```python
BINS = [np.linspace(-2.4, 2.4, 7)[1:-1],      # cart position    -> 6 bins
        np.linspace(-3.0, 3.0, 7)[1:-1],      # cart velocity    -> 6 bins
        np.linspace(-0.21, 0.21, 13)[1:-1],   # pole angle       -> 12 bins
        np.linspace(-2.0, 2.0, 13)[1:-1]]     # pole ang. veloc. -> 12 bins

def encode(obs):
    return tuple(int(np.digitize(o, b)) for o, b in zip(obs, BINS))

Q = np.zeros((6, 6, 12, 12, 2))               # 5,184 states, 10,368 entries
```

`np.digitize` maps anything below the first edge to 0 and anything above the
last to the final bin, so the encoder never index-errors on an extreme
observation. The `update` from Part B is untouched — only the indexing changed.
Then wrap it in the `Agent` class from the Gymnasium lesson: `setup` loads the
saved table, `choose_action` returns `int(self.Q[encode(observation)].argmax())`.

**The numbers on competition 48.** Ranking is on **mean episode reward — higher
is better**, over seeded episodes, and the environment truncates at 500 so that
is the ceiling. A uniform-random policy scores **22.4** (sd 11.7 over 300
episodes); on the live board the `__benchmark__` row sits at **20.7** and
`qa-cartpole-random` at **23.3 ± 9.8**. The ceiling is genuinely reachable
without any learning at all: the one-line rule `0 if angle + 0.5 * angular_velocity
< 0 else 1` scores **500.00** (sd 0.00 over 300 episodes), and the board's
`qa-cartpole-linear` row is that rule at 500.0.

Between those two sits the agent you are actually writing. The binning above,
trained for 30,000 episodes at $\alpha = 0.1$ with the same $\epsilon$ schedule,
scores **431.7** across three seeds — 295.1, 500.0 and 500.0, each evaluated
greedily over 300 episodes. Read that spread: two seeds pinned the ceiling and
one did not, from identical code and identical hyperparameters. It is the
Part C lesson again, on a harder environment.

So the bar for this part is **mean episode reward > 23.3** — that alone says
your discretisation, your table and your packaging are wired up, and it is what
the checklist asks for. Getting to 430 is the same code with more episodes;
getting there *reliably*, on every seed, is Session 10's job with a network
instead of bins.

Load the table relative to `agent.py`, not to the working directory — the
platform runs your agent from somewhere you did not choose:

```python
WEIGHTS = os.path.join(os.path.dirname(__file__), "q_table.npy")
```

```python
import os, mlarena                      # uv pip install mlarena-sdk
client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
client.submit(48, files=["agent.py", "runs/q_table.npy"])
print(client.status())
print(client.leaderboard(48, top=10))
```

The table stays in gitignored `runs/` — uploading it is not committing it — and
each file is uploaded under its basename, so on the platform `q_table.npy` lands
beside `agent.py`, which is what the `__file__` path above resolves to.

A Q-table agent needs only `numpy`, so the default runtime image is fine and you
do not need the `runtime=` argument that Session 10's lab uses for torch. If a
deploy fails at import anyway, `client.runtime_options(48)` lists what that
competition offers.

> Competition `49` (MountainCar-v0) is also attached to this session. Do not
> spend the evening on it: every entry on its board, including the benchmark,
> scores exactly **-200.0**, which is the episode floor, so the leaderboard
> cannot tell a working agent from a broken one. It is a fair exercise and a
> useless scoreboard.

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

- the random baseline, the value-iteration ceiling, and the final success rate
  of each agent — three numbers on the same scale
- the number of seeds and the number of episodes per seed
- which of Q-learning and SARSA won, and whether the bands actually separate
- the one policy cell you explain in Part E
- your competition 48 score next to the 23.3 random row, and your attachment id
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

Part F is a take-home **gate**, not a weighted criterion: Session 10's lab
assumes you already have one accepted ML-Arena submission and will not be the
place to debug the packaging contract for the first time.

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
`choose_action`, no learning at evaluation time. Part F wraps it in the `Agent`
class from the Gymnasium lesson, which is a ten-line change once the four floats
are binned.

Session 10 replaces the table with a network — no bins, no ceiling at 5,184
states — and the same loop survives intact. Keep `train.py`: you will run it
again with a different `update`.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] Part A: `REPORT.md` states the random baseline over 1,000 episodes and it
      is between 0.005 and 0.03
- [ ] Part A': `REPORT.md` states the value-iteration ceiling and it rounds
      to 0.74
- [ ] Part B: the `* (not terminated)` factor is present in both
      `QLearning.update` and `Sarsa.update`, and neither `update` takes
      `truncated` as an argument
- [ ] Part C: `runs/` holds one figure with two mean curves and two shaded
      bands, built from the same ≥5 seeds and 5,000 episodes for both agents
- [ ] Part D: Q-learning's greedy evaluation, averaged over the seeds, is at
      least 0.65. The Part C training curves end far lower (~0.44) because
      epsilon is still ~0.08 at episode 5,000 — do not compare them to this
      number. Below 0.65 on the greedy evaluation is a bug — below
      that is a bug, not a tuning problem (the ceiling is 0.74)
- [ ] Part D: the ablation table has all five rows filled and at least one row
      is clearly worse than the others
- [ ] Part E: `REPORT.md` names one specific grid cell whose arrow points away
      from the goal and explains it with the 1/3 slip probability
- [ ] All four required tests pass, including
      `test_q_update_matches_hand_computation`
- [ ] `git ls-files | grep -E "runs/|[.]npy$"` returns nothing
- [ ] My submission is on the leaderboard of CartPole-v1 (#48)
- [ ] My score beats the baseline: **mean episode reward > 23.3** — the random
      row on that board. The ceiling is 500.

If the last two are not ticked you have not finished Part F, however good the
Frozen Lake agent is. Part F is take-home: it is the bridge to Session 10, and
Session 10 assumes you have already made one accepted submission.
