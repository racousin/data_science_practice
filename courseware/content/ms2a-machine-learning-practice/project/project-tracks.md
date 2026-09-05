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
calls your agent over the platform's proxy. This term's Agent track is
**SuperTuxKart Grand Prix (#169)**, four karts in one race, and its contract is
a zero-argument constructor and a single method:

```python
class Agent:
    def __init__(self): ...

    def act(self, obs) -> dict:
        return {"steer": 0.0, "acceleration": 1.0, "brake": 0,
                "drift": 0, "nitro": 0, "fire": 0, "rescue": 0}
```

`env.py` calls `act` once per kart per step with a **2-second budget per call**.
`steer` is −1..1, `acceleration` is 0..1, the other five are 0 or 1, and a
missing key defaults to coasting — so a partial dict is legal and silently
mediocre. Your trained weights ship alongside `agent.py` as a file the agent
loads.

---

## Agent track — the observation

`obs` is a plain dict, every coordinate egocentric to your kart:

| Key | Shape | Meaning |
|---|---|---|
| `velocity`, `front`, `center_path` | `[3]` each | motion, heading, vector to the track centre |
| `center_path_distance`, `distance_down_track` | float | lateral offset, progress along the lap |
| `max_steer_angle`, `energy` | float | steering limit, nitro reserve |
| `powerup`, `attachment` | int | what you are holding, what is stuck to you |
| `paths` | list of `{start, end, width}` | the nearest track segments ahead |
| `karts` | list of `[3]` | the nearest opponents |
| `items` | list of `{pos, type}` | the nearest items |

Score is cumulative race reward — distance progress, a position-among-karts
bonus, and a finish bonus — but see the ELO slide: that number is not your rank.

---

## Agent track — what it draws on

Sessions **9** (MDPs, value functions, tabular control) and **10** (DQN, policy
gradients, PPO, multi-agent). Lab 10 already puts an agent on ML-Arena, so the
mechanics are not new by the time the project opens.

Those labs, and the two Reference playgrounds (#168 and #170), use the *other*
`flex_v1` contract — the gymnasium one, which is `setup` plus `choose_action`
rather than `act`:

```python
class Agent:
    def setup(self, observation_space, action_space):
        from flexkit.spaces import decode_space
        self.action_space = decode_space(action_space)   # not a Space yet
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None): ...
```

`setup` receives the spaces **dict-encoded**, not as Gymnasium objects:
`MultiBinary(6)` arrives as `{'type': 'multi_binary', 'shape': [6]}`. Decode
before you call `.sample()` or read `.n` / `.shape`, or the first step raises
`AttributeError: 'dict' object has no attribute 'sample'`.

**Read the competition's own agent template before you write a line.** An agent
that exposes `choose_action` to a competition whose `env.py` calls `act`
exposes no method the environment ever calls, and the failure looks like a
timeout.

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

The board also shows a mean-reward column, and on #169 it contradicts the rank:
the rank-1 agent shows **946.41** while a rank-3 agent shows **11,210.39**. The
reward column is information. ELO is the score.

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

## This term's three competitions

The three tracks above are fixed. These are the competitions attached to this
module, and they are not going to change under you:

| Track | Competition | Metric | You upload |
|---|---|---|---|
| Prediction | 2-Month Survival Prediction (**#172**) | accuracy | `submission.csv` |
| Agent | SuperTuxKart Grand Prix (**#169**) | ELO rating | `agent.py` + weights |
| Generative | The Round (**#171**) | USD raised | `pitch.txt`, exactly one file |

Two things to know about #172 before you pick it. It is the **same competition
you already submitted to in Lab 3**, so your Lab 3 pipeline carries over and
the project starts from a working submission. And its board already holds 82
entries from an earlier cohort, run between 23 June and 6 July 2026 — those
rows are a reference, not the classmates you are ranked against.

---

## The numbers you are graded against

"Absolute score vs the published baseline" is 15% of the project grade. This is
the published baseline. Every number was read from the live leaderboards on
2 September 2026; reproduce any of them with `client.leaderboard(<id>)`.

**Prediction — #172, accuracy, higher is better.** About 59% of the patients are
`alive`, so always guessing `alive` scores **0.594** — and that is exactly where
the platform reference `__benchmark__` sits. That is the bar. The median of the
82-row board is **0.787** and the top entry is **0.811**, so the useful range is
narrow: 80 of the 82 entries beat the baseline, and the work is in the last two
points.

**Agent — #169, ELO, higher is better.** There is no absolute bar. Every agent
starts at **1200** and your rating moves when other people submit. On 2
September the board holds four agents: `__benchmark__` and `Luigi` at **1200**,
`baseline-kart-test` and `flexfix-kart-vmcheck` at **1184**. Beating the
baseline means finishing above `__benchmark__`'s 1200.

**Generative — #171, USD raised, higher is better.** The reference pitch
`__benchmark__` raises **$27.1M** over 14 runs. That is the bar. The top of the
board is **$46.15M** (two pitches tied), and two of the nine entries raise
**$0** — a pitch can score nothing. The ~**$65M** on the page is the total
capital across the panel, a ceiling, not a target.

---

## Check yourself

1. You write an Agent-track agent with `setup` and `choose_action`, upload it to
   #169, and every race times out. What is wrong, and where would you have found
   out in one minute?

   **Answer.** #169's `env.py` calls `act(obs)`, not `choose_action`, so your
   agent exposes no method the environment ever calls. The competition's own
   agent template states which of the two contracts it uses; read it before you
   write the class.

2. Run this. You should get exactly the output shown.

   ```python
   CONTRACT = {"steer": (-1.0, 1.0), "acceleration": (0.0, 1.0),
               "brake": (0, 1), "drift": (0, 1), "nitro": (0, 1),
               "fire": (0, 1), "rescue": (0, 1)}
   action = {"steer": -1.4, "acceleration": 1.0}

   print(sorted(set(CONTRACT) - set(action)))
   # -> ['brake', 'drift', 'fire', 'nitro', 'rescue']
   print([k for k, v in action.items() if not CONTRACT[k][0] <= v <= CONTRACT[k][1]])
   # -> ['steer']
   ```

   **Answer.** The five missing keys are legal — they default to coasting — so
   nothing tells you they are missing. The out-of-range `steer` is the kind of
   bug this check exists to catch. Assert the contract in your own code, because
   the platform will not.

3. Your Generative-track pitch raises $24M. Have you beaten the published
   baseline, and what is the $65M figure on the competition page?

   **Answer.** No. The reference pitch `__benchmark__` raises $27.1M, so $24M
   is below the bar. The $65M is the total capital across the 16 investors — a
   ceiling nobody has reached (the board top is $46.15M), not a target.
