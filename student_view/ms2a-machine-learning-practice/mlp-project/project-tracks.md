# Tracks

Three tracks, one competition each, pick one. They differ in what you upload,
what scores it, and which half of the course they lean on. The structure is
fixed; the subjects are announced separately.

<!-- notes: 25 minutes. Ten of them is the students reading the comparison table
and arguing. Do not let anyone leave undecided — make them write the track on
paper before the end. The failure-mode slide of each track is the part worth
lecturing; the rest they can read. -->

---

## The three, in one line each

| Track | Kernel | You upload |
|---|---|---|
| Prediction | `file_v1` | a predictions file |
| Agent | `flex_v1` | code that acts in an environment |
| Generative / LLM | `file_v1` + a judge | generated text |

All three are the same platform, the same team object, the same leaderboard,
the same freeze. What changes is the shape of the work between now and then.

---

## Prediction track — what you submit

A single file: a CSV of predictions on a test set whose labels you never see.

```text
id,label
1042,pneumonia
1043,healthy
```

The competition's `env.evaluate(submission_path)` parses it and returns the
metric its page states. The format is opaque to the platform — a misspelled
column name is a zero, not a helpful error.

Data are tabular or images. Everything runs on your machine; the platform only
scores.

---

## Prediction track — what it draws on, and how it goes wrong

Sessions **2** (a leak-free preprocessing pipeline), **3** (gradient boosting,
cross-validation, hyperparameter search) and **5** (transfer learning on a
pretrained backbone).

The characteristic failure is **overfitting the public leaderboard**. Sixty
submissions, each keeping whatever nudged the public score up, is a hill-climb
on a few hundred labelled examples. It produces a model tuned to the noise of
that subset and a final ranking that collapses.

---

## Prediction track — the split discipline

> Your local cross-validation is the number you optimise. The leaderboard is a
> sanity check, not a loss to minimise.

- fix a validation protocol in week one and do not change it mid-term
- track the gap between local CV and public score — a growing gap is the alarm
- budget your submissions; each one you tune against costs a degree of freedom
- assume a private half decides the final ranking, from the first submission on

---

## Agent track — what you submit

Code, not an answer. The competition's `env.py` owns the evaluation loop and
calls your agent over the platform's proxy. The single-agent (`gymnasium`)
contract:

```python
class Agent:
    def __init__(self): ...
    def setup(self, observation_space, action_space): ...
    def choose_action(self, observation, reward=0.0,
                      terminated=False, truncated=False, info=None): ...
```

Zero-argument constructor, one `setup` call, then one `choose_action` per step.
Your trained weights ship alongside `agent.py` as a file the agent loads.

---

## Agent track — what it draws on

Sessions **9** (MDPs, value functions, tabular control) and **10** (DQN, policy
gradients, PPO, multi-agent). Lab 10 already puts an agent on ML-Arena, so the
mechanics are not new by the time the project opens.

The multi-agent (`pettingzoo`) contract adds one required method:

```python
class Agent:
    def reset(self, env_player_name, episode_index):
        self.player = env_player_name
```

Roles rotate between episodes. An agent that assumes it is always the first
player is wrong half the time, and nothing in the loop tells it so.

---

## Agent track — the packaging contract

The characteristic failure here is not RL. It is an agent that trains
beautifully and then dies on the platform:

| Symptom | Cause |
|---|---|
| Timeout before the first step | training or a 2 GB checkpoint load at import time |
| `ModuleNotFoundError` | a training-only dependency called at inference |
| Score varies run to run | unseeded RNG, wall-clock branching, threads |
| Collapses after episode 1 | state carried across episodes, no `reset` |

Import must be cheap: load the checkpoint in `setup`, keep inference imports to
what the runtime image contains, and seed everything.

---

## Agent track — ELO changes the target

Multi-agent competitions are ranked by **ELO** against the current population,
not by an absolute score. Two consequences:

- your rating moves when *other people* submit, without you doing anything
- "improving" means beating the field as it stands, so an agent that beat last
  month's field may be mid-table today

Submit early and keep submitting. A single upload the night before the freeze
has no rating history and gets ranked on a handful of matches.

---

## Generative / LLM track — what you submit

Generated text in a file: answers to a held-out set of questions, summaries,
classifications with a justification, a pitch. The competition's `env.evaluate`
runs a **judge model** with a fixed rubric over your output and returns a score.

Sessions **7** (tokenization, embeddings, attention, the transformer) and **8**
(decoding, instruction tuning and PEFT, RAG, evaluating generation) are the
ones you need. Session 8's "Evaluating Generation" is effectively the
specification of this track.

---

## Generative / LLM track — the two failure modes

**Unreproducible generation.** Sampling at temperature 0.8 with no seed means
you cannot regenerate your own submission tomorrow, which triggers the
reproducibility rule and a zero. Pin the model revision, pin the seed or use
greedy decoding, and commit the prompt as a file.

**Prompt-tuning against a judge you cannot see.** Thirty submissions to
discover the judge likes bullet points is the same hill-climb as overfitting a
public leaderboard, with a noisier signal.

---

## Generative / LLM track — the honest response

Two pieces of engineering, both cheap:

```text
prompts/v3.txt          # the exact prompt, versioned, referenced by commit
eval/judge.py           # your own judge, your own rubric, your own held-out set
```

A local judge harness — an open-weights model scoring a set you labelled or
reserved — lets you measure a prompt change before spending a submission on it.
Run it three times and report the spread: a judge is a noisy instrument, and a
+0.4 improvement inside a $\pm$ 1.2 band is not an improvement.

---

## Comparison

| | Prediction | Agent | Generative / LLM |
|---|---|---|---|
| Kernel | `file_v1` | `flex_v1` | `file_v1` + judge |
| You submit | a predictions file | `agent.py` + weights | generated text |
| Scored on | a fixed metric vs private labels | episode return, or ELO | a judge model's rubric |
| Sessions | 2, 3, 5 | 9, 10 | 7, 8 |
| Main risk | overfitting the public board | breaking the packaging contract | unreproducible generation |
| Suits you if | you like validation and feature work | you like systems and debugging | you like evaluation design |

---

## How to choose

Choose on the failure mode, not on the topic. Every track is winnable, and
every one has a wall you hit around week six:

- **Prediction** — a validation protocol you stop trusting
- **Agent** — a training run that works locally and not on the platform
- **Generative** — a metric that moves for reasons you cannot explain

Pick the wall you would rather spend two weeks against. With no preference,
take Prediction: it has the shortest path to a first scored submission, and a
first scored submission is milestone 3.

---

## The subjects are not the structure

The three tracks above are fixed. The **specific competitions** for this term —
the datasets, the environments, the judge, the baselines to beat — are
announced separately and attached to this module on ML-Arena.

Nothing in this lesson changes when they are. Choose the track now; the subject
will not surprise you.
