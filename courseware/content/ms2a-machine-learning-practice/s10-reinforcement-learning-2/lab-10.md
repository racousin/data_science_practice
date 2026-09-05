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

## Part A — Set up and read the target (5 min)

This session attaches two competitions. **LunarLander-v3 (competition `43`)** is
the Gymnasium track, ranked by mean episode reward, and it is the one this lab
is written for. Connect-Four (competition `65`) is a PettingZoo track ranked by
ELO; it is optional and there is a note at the end of the lab about what it
would take.

**The numbers on competition 43.** Ranking is on **mean episode reward — higher
is better**. A uniform-random policy scores **-185.6** (sd 111.2 over 200
episodes); the competition's own `__benchmark__` row scores **-266.6**, an
untrained agent that crashes, and sits at rank 661 of 708. LunarLander is
considered *solved* by the Gymnasium convention at **200**, which is also
roughly the middle of this board: the median of the 708 entries is **217.7** and
the best is **293.05 ± 2.35** over 300 episodes. Take **200** as the bar and 250
as a good afternoon. The leaderboard prints a 95% confidence interval — two
agents whose intervals overlap are tied, not ranked.

Write the competition id and the action space into your PR description now:
`LunarLander-v3` is `Discrete(4)` over a `Box` of 8 floats, and picking the
algorithm before reading the action space is how you spend twenty minutes
training something you cannot submit.

```text
src/rl/   train.py  evaluate.py  policy.py  agent.py
checkpoints/policy.pt      tests/test_agent.py      reports/eval.png
```

`agent.py` and `checkpoints/policy.pt` are what the platform receives. Nothing
else in the repository exists as far as it is concerned.

---

## Part B — Train (15 min)

PPO unless you have a reason. Stable-Baselines3 unless you have a reason.
Build the batch of environments with SB3's own `make_vec_env`: handing
`gym.make_vec(...)` to `PPO` raises `ValueError: The environment is of type
... not a Gymnasium environment` before a single step runs.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

envs = make_vec_env(ENV_ID, n_envs=8, seed=SEED)
model = PPO("MlpPolicy", envs, n_steps=512, batch_size=256, seed=SEED)
model.learn(total_timesteps=300_000)
```

Now **export** the policy — do not save `model.policy.state_dict()`. That
state dict carries SB3's own module names (`mlp_extractor.policy_net.0.weight`,
`action_net.weight`) plus a critic your inference module does not have, so
loading it into your own `Policy` in Part D raises `Missing key(s) in
state_dict`. Copy the three layers on the policy path into your module's naming
and save that:

```python
import torch, torch.nn as nn

class Policy(nn.Module):                      # policy.py — copy it verbatim into agent.py too
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions))
    def forward(self, x):
        return self.net(x)                    # logits; argmax is the greedy action

obs_dim   = int(envs.observation_space.shape[0])
n_actions = int(envs.action_space.n)          # int(): action_space.n is a numpy
                                              # scalar, and torch.load refuses to
                                              # unpickle one under weights_only
src = model.policy.state_dict()
policy = Policy(obs_dim, n_actions)
policy.load_state_dict({
    "net.0.weight": src["mlp_extractor.policy_net.0.weight"],
    "net.0.bias":   src["mlp_extractor.policy_net.0.bias"],
    "net.2.weight": src["mlp_extractor.policy_net.2.weight"],
    "net.2.bias":   src["mlp_extractor.policy_net.2.bias"],
    "net.4.weight": src["action_net.weight"],
    "net.4.bias":   src["action_net.bias"],
})
torch.save({"obs_dim": obs_dim, "n_actions": n_actions,
            "state_dict": policy.state_dict()}, "checkpoints/policy.pt")
```

The default `MlpPolicy` is exactly 2 × 64 with `tanh`, which is why `Policy`
above matches it layer for layer. Change `policy_kwargs` and you must change
`Policy` to match — the key names are positional.

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
        # flexkit ships inside the platform runtime image, not on PyPI — import
        # it here so `import agent` and your own tests work on your laptop too.
        from flexkit.spaces import decode_space
        self.action_space = decode_space(action_space)
        ckpt = torch.load(WEIGHTS, map_location="cpu")
        self.net = Policy(ckpt["obs_dim"], ckpt["n_actions"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        return True
```

`Policy` is the class from Part B. Copy it into `agent.py` rather than importing
`policy.py` — `agent.py` and `checkpoints/policy.pt` are the only two files the
platform receives.

---

## The four rules the platform enforces

| Rule | Why | What breaks |
|---|---|---|
| Importable — `import agent` succeeds | the runner imports it | a missing dependency fails the deploy; a new attachment defaults to the dependency-free image, so a torch agent needs `runtime=` at submit time |
| No work at import time | the pod imports before the env exists | training or `gym.make` at module level times out |
| No network access at inference | the pod has no egress | a download of weights hangs, then the episode is lost |
| Deterministic given a seed | runs must be comparable | an unseeded `sample()` makes your score unreproducible |

Weights are loaded from a file you upload beside `agent.py`. Load them in
`setup`, not at import, and pass `map_location="cpu"` — there is no GPU in the
agent container.

---

## Part E — Submit (5 min)

```bash
uv pip install mlarena-sdk
```

The distribution is `mlarena-sdk` and it imports as `mlarena`. `uv pip install
mlarena` gets you an unrelated package with no `connect`.

A fresh attachment defaults to the **dependency-free** runtime image. If your
`agent.py` imports torch, tensorflow or jax you must pin the matching runtime
with `runtime=`, or the deploy dies at import with `ModuleNotFoundError: No
module named 'torch'` — `client.runtime_options(COMPETITION_ID)` lists what that
competition offers.

```python
import os, mlarena
client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
res = client.submit(COMPETITION_ID,
                    files=["src/rl/agent.py", "checkpoints/policy.pt"],
                    runtime={"language": "python", "framework": "torch"})
print(client.status())
print(client.leaderboard(COMPETITION_ID, top=10))
```

`submit` creates the attachment, pins the runner, uploads, and deploys, and
returns a dict with `attache_agent_id`. `status()` says whether it is queued,
running or failed; when it says `deploy_failed`, print the reason:

```python
for line in client.tail_logs(COMPETITION_ID, res["attache_agent_id"]):
    print(line)
```

The key comes from `os.environ` and never appears in the notebook. Record your
rank and score in the PR, then wait — the ranking moves as others submit, which
is the point of a leaderboard.

Getting on the board is what is graded. Do the random-action agent first, in the
first ten minutes: it will land near **-186** on competition 43, which proves
the whole path works, and every later submission is then only a training
problem.

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
    """Mean return over 20 seeded episodes beats the random baseline you
    recorded in Part B (about -186 on LunarLander-v3)."""
```

The third catches the real failure: your training framework is not installed in
the agent container. Test it by deleting the import, not by believing you did
not use it.

---

## Pull request

The description states:

- the competition id, the environment, and the action space
- the algorithm and why it fits that action space
- the random baseline you measured, and your evaluated mean with its band
- your leaderboard score next to the two published numbers — random -185.6 and
  solved 200 — and which of your own numbers you believe
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

## If you finish early — the PettingZoo track

Competition `65` (Connect-Four) is ranked by **ELO against the live
population**, starting at 1200; the reference agent sits at **1248** and the
board has four entries, so there is no absolute score to beat — the target is to
finish above 1248 over enough games to mean anything. The mean-reward column on
that board is the mean game outcome and does *not* determine the rank.

It is a strictly harder lab than the Gymnasium track and this session does not
teach the training half of it. What is missing, if you want it:

- a vectorized wrapper, because SB3 cannot consume a PettingZoo environment
  directly — `supersuit.pettingzoo_env_to_vec_env_v1(...)` over the parallel API;
- self-play: PPO trained against a frozen copy of itself, plus a pool of past
  checkpoints so it does not forget (see the Multi-Agent lesson's "league");
- logit masking during **training**, not only at inference — an unmasked policy
  never learns which moves are legal;
- `reset(env_player_name, episode_index)` in `agent.py`, because roles rotate
  between episodes and an agent that assumes it is always player 0 plays the
  wrong side of half its games.

Do it after the Gymnasium submission is accepted, never instead of it.

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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] Part B: `train.py --seed 0` runs end to end and writes
      `checkpoints/policy.pt`; `torch.load(...)` on it returns a dict with the
      keys `obs_dim`, `n_actions`, `state_dict`
- [ ] Part B: my `REPORT.md` records what the **random** agent scored, measured
      before training
- [ ] Part C: `reports/eval.png` shows three seed curves, a band, a horizontal
      random-baseline line, and a caption saying which band it is
- [ ] Part D: `import agent` succeeds in a fresh interpreter with
      `sys.modules["stable_baselines3"] = None`, which blocks the import —
      deleting the entry only forces a re-import, so the check passes for the
      very agent it exists to catch — and constructs nothing at
      import time
- [ ] All four required tests pass, including
      `test_inference_has_no_training_dependency`
- [ ] `client.status()` reports `active`, not `deploy_failed` — if it says
      `ModuleNotFoundError: No module named 'torch'` you submitted without
      `runtime={"language": "python", "framework": "torch"}`
- [ ] My submission is on the leaderboard of LunarLander-v3 (#43)
- [ ] My score beats the baseline: **mean episode reward > -185.6** — the
      uniform-random floor. Solved is 200; the median of the 708 entries is
      217.7; the best is 293.05 ± 2.35.

If the last two are not ticked you have not finished the lab, however good the
code is.
