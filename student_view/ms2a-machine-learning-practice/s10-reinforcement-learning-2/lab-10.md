# Lab 10 — Ship an Agent to ML-Arena

Train a deep RL agent, evaluate it honestly, package it to the platform's
contract, and submit it to a live competition. The last lab of the course, and
the first time your code runs on someone else's infrastructure.

**Time:** 45 minutes. **Deliverable:** a merged PR, and an accepted submission
on an ML-Arena leaderboard.

<!-- notes: 45 minutes and it will overrun, so make them submit a random-action
agent in the first ten minutes — an accepted bad submission beats an unaccepted
good one. The packaging contract is where they will lose time: no work at
import, no network. Circulate with the leaderboard open. -->

---

## Part A — Set up and pick a competition (5 min)

Open `https://ml-arena.com`, or call `client.competitions()`, and pick a
**Gymnasium** track (single agent, ranked by mean reward) or a **PettingZoo**
track (two-player, ranked by ELO). Write the competition id and the action
space into your PR description now: a `Discrete(4)` and a `Box(-1, 1, (2,))`
are different labs, and picking the algorithm before reading the action space
is how you spend twenty minutes training something you cannot submit.

```text
src/rl/   train.py  evaluate.py  policy.py  agent.py
checkpoints/policy.pt      tests/test_agent.py      reports/eval.png
```

`agent.py` and `checkpoints/policy.pt` are what the platform receives. Nothing
else in the repository exists as far as it is concerned.

---

## Part B — Train (15 min)

PPO unless you have a reason. Stable-Baselines3 unless you have a reason.

```python
envs = gym.make_vec(ENV_ID, num_envs=8)
model = PPO("MlpPolicy", envs, n_steps=512, batch_size=256, seed=SEED)
model.learn(total_timesteps=300_000)
torch.save(model.policy.state_dict(), "checkpoints/policy.pt")
```

Requirements:

- `train.py` takes `--seed` and sets all four generators from it
- normalisation statistics saved next to the weights, if you use any
- the run logs episode return against environment steps to a file
- a written note of what the **random** agent scores, from before you trained

---

## Part C — Evaluate (10 min)

Three training seeds minimum, 20 evaluation episodes each, deterministic
policy, an eval environment that is not the training one.

```python
returns = np.stack([eval_seed(s) for s in (0, 1, 2)])
mu = returns.mean(0)
half = 1.96 * returns.std(0, ddof=1) / np.sqrt(returns.shape[0])
```

Produce `reports/eval.png`: the mean curve, the band, the individual seed
curves behind it, a horizontal line for the random baseline, and a caption
stating what the band is. A one-seed plot is not a result.

---

## Part D — Package (10 min)

The platform imports your `agent.py`, constructs `Agent()` with no arguments,
calls `setup(observation_space, action_space)` once, then `choose_action(...)`
per step.

```python
class Agent:
    def setup(self, observation_space, action_space):
        self.action_space = decode_space(action_space)
        self.net = Policy(...)
        self.net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        self.net.eval()
        return True
```

For a PettingZoo track you also implement `reset(env_player_name,
episode_index)` — roles rotate between episodes, and an agent that assumes it
is always player 0 plays the wrong side of half its games.

---

## The four rules the platform enforces

| Rule | Why | What breaks |
|---|---|---|
| Importable — `import agent` succeeds | the runner imports it | a missing dependency fails the deploy |
| No work at import time | the pod imports before the env exists | training or `gym.make` at module level times out |
| No network access at inference | the pod has no egress | a download of weights hangs, then the episode is lost |
| Deterministic given a seed | runs must be comparable | an unseeded `sample()` makes your score unreproducible |

Weights are loaded from a file you upload beside `agent.py`. Load them in
`setup`, not at import, and pass `map_location="cpu"` — there is no GPU in the
agent container.

---

## Part E — Submit (5 min)

```python
import mlarena
client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
res = client.submit(COMPETITION_ID,
                    files=["src/rl/agent.py", "checkpoints/policy.pt"])
client.status()
client.leaderboard(COMPETITION_ID, top=10)
```

`submit` creates the attachment, uploads, and deploys. `status()` says whether
it is queued, running or failed; `tail_logs` says why it failed. The key comes
from `os.environ` and never appears in the notebook. Record your rank and score
in the PR, then wait — the ranking moves as others submit, which is the point
of a leaderboard.

---

## Required tests

```python
def test_policy_emits_only_legal_actions():
    """1000 observations give actions in the space; masks respected."""

def test_checkpoint_round_trip():
    """save -> load -> same action, same observation, same seed."""
```

```python
def test_inference_has_no_training_dependency():
    """Import agent with stable_baselines3 hidden from sys.modules."""

def test_episode_return_on_fixed_seed():
    """One episode at seed 0 beats the recorded random baseline."""
```

The third catches the real failure: your training framework is not installed in
the agent container. Test it by deleting the import, not by believing you did
not use it.

---

## Pull request

The description states:

- the competition id, the environment, and the action space
- the algorithm and why it fits that action space
- the random baseline, and your evaluated mean with its band
- the leaderboard position at submission time, and the total entries
- one thing that failed on the platform but worked locally, and the cause

---

## Grading

| Criterion | Weight |
|---|---|
| Submission accepted and running on the leaderboard | 25% |
| Training script reproducible: seeded, logged, re-runnable | 15% |
| Multi-seed evaluation with a correctly labelled band | 20% |
| `agent.py` respects the four packaging rules | 15% |
| Four tests passing | 15% |
| PR description complete | 10% |

Position on the leaderboard is worth nothing. Being on it is worth 25%.

---

## Automatic deductions

- weights loaded, or an environment constructed, at import time
- a network call inside `choose_action`
- an API key in the source or in the git history
- a bare `except` around the inference path — a crashed agent must crash
- an evaluation curve from a single seed presented as a result
- checkpoints over 100 MB committed to git rather than uploaded

---

## Carry it forward

You have now shipped a model to an evaluation harness you do not control, under
a contract you had to read. That is the same exercise as the project, and the
project is half the grade.

Ten labs: a dataset, a leak-free pipeline, a tuned baseline, a trained network,
two vision models, a fine-tuned classifier, a judged RAG system, a tabular
agent, and this. Pick the track where you have the most left to say.
