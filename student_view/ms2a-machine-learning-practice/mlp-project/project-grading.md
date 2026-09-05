# Deliverables and Grading

What you hand in, what each part is worth, and the small list of findings that
cost you marks automatically regardless of how well the rest went.

<!-- notes: 20 minutes. Read the automatic-deductions table out loud, slowly —
it is the slide that changes behaviour. The AI-accountability slide at the end
is not a warning, it is a description of how the defense actually works: ask
someone to explain a line and see what happens. -->

---

## The deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Private GitHub repository, instructor a collaborator | GitHub |
| 2 | `README.md` — install and reproduce in three commands | repo root |
| 3 | `DATASET.md`, or an environment description | repo root |
| 4 | Tests on committed fixtures | `tests/` |
| 5 | Report, six pages maximum | repo |
| 6 | The ML-Arena submission itself, named | the platform |
| 7 | Defense | in person |

Seven items. Six of them live in the repository, which is the point.

---

## The README, in three commands

A stranger with your repository URL and nothing else reaches your submitted
artefact by running exactly this:

```bash
git clone git@github.com:you/project.git && cd project
uv sync
uv run python -m project.reproduce --config configs/best.yaml
```

The third writes the file you submitted — on the Agent track, the `agent.py`
and weights you uploaded. If the run is stochastic the README states the seed
and the tolerance, and the score lands inside it.

Three commands, not "run the cells in order, except cell 14".

---

## Describing your inputs

| Track | The file, and what it has to answer |
|---|---|
| Prediction | `DATASET.md` as in Lab 1 — one row, columns, source, licence, split |
| Agent | observation and action spaces, reward, termination, training seeds |
| Generative | model and revision, decoding parameters, prompt file, local judge |

One question in three forms: what did the model actually see?

---

## Tests

At least three, running on fixtures committed under `tests/fixtures/` — a dozen
rows, never the real dataset. One of them must be a reproduction test:

```python
def test_pipeline_reproduces_reference_score():
    """Running the pipeline on the fixture scores 0.82 +/- 0.01."""
```

The rule from the labs stands: a test that cannot fail is not a test. Before
you hand in, break one line of your pipeline on purpose and check that the
suite goes red. If it does not, you have three assertions that assert nothing.

---

## The report

Six pages maximum, Markdown or PDF, in the repository. Five sections:

1. **The problem** — task, metric, baseline, in half a page
2. **What you tried and what failed** — a table, including the losers
3. **The validation protocol** — the split, why it is right, its gap to the board
4. **The final number** — honest, with its variance, the submission name on the
   board, and the commit that produced it
5. **Another month** — what you would do next, and why that

---

## Section 2 is the one that is read first

A report with no failures in it is a report that has been laundered.

| Experiment | Local CV | Board | Kept |
|---|---|---|---|
| Baseline, class prior | 0.31 | 0.30 | reference |
| Gradient boosting, default | 0.74 | 0.72 | yes |
| + target encoding | 0.79 | 0.71 | **no** — leak |
| + tuned depth and shrinkage | 0.81 | 0.80 | yes |
| Stacked with a neural net | 0.81 | 0.79 | no — no gain |

Five rows like this say more about your work than five pages of prose. The
third row — a local gain that did not transfer — is worth more marks than the
second.

---

## Grading

| Axis | Criterion | Weight |
|---|---|---|
| Leaderboard | Rank in the cohort at the freeze | 25% |
| Leaderboard | Absolute score vs the published baseline | 15% |
| Leaderboard | 3+ scored submissions, spread over the term | 10% |
| Repository | Reproduces from a clean clone, three commands | 15% |
| Repository | Package structure, git history, pinned deps | 10% |
| Repository | Tests that can fail | 10% |
| Repository | The report | 10% |
| Repository | Defense | 5% |

Fifty on each axis. The third leaderboard line is free marks for submitting
early, and every year some teams do not collect it.

---

## Automatic deductions

Applied on inspection, before any judgement about quality:

| Finding | Cost |
|---|---|
| Data or model weights committed to git | −10% |
| A credential in the source or in the history | −10%, and rotate it today |
| A bare `except` | −5% each, capped at −10% |
| `.get(key, default)` for required configuration | −5% |
| A result that cannot be reproduced from the repo | project grade **0** |
| An unattributed copy, code or prose | project grade **0** |

The last two are not gradients. They are the two ways a project stops being
your work or stops being a result.

---

## The defense

Twenty minutes per team, at the end of the term.

- **8 minutes** — the problem, what you did, the final number. The report's
  figures are enough; no separate deck.
- **12 minutes** — questions, addressed to a named person, not to the team.
- **One live task** — change something in the repository (the split, a
  hyperparameter, a line of the prompt) and predict the effect before running.

Bring the repository. Nothing else is needed.

---

## AI-assisted work: the rule

You were taught to drive a coding agent in the second session of the 12-hour
module, and you are expected to use one. The rule is not *do not use it*.

> You are accountable for every line in the repository, and you must be able to
> explain any of them under questioning.

"The assistant wrote that" is not an answer at the defense. It is the same
answer as "I do not know what this does", and it is graded the same way.

---

## What that means in the repository

Generated code nobody read has a recognisable shape, and it counts as
**negative value** — the grader still has to read it:

- helpers and imports that nothing calls
- a six-line docstring on a two-line function
- `try/except` around code that cannot raise
- a base class with exactly one implementation

Delete it. A short repository you can defend outscores a large one you cannot.

---

## Attribution

External code — a GitHub repository, a paper's reference implementation, a
tutorial, an answer you found — is cited in the `README.md` with a link and one
sentence saying what you changed.

Citing costs you nothing. Not citing is the deduction that ends the project,
and it is trivially detectable in a term where everyone solves the same three
problems.

---

## The thing being graded

> Not "did you get a good score". Whether the score is yours, whether you can
> get it again, and whether you can explain how you got it.

Those three questions are the leaderboard axis, the reproducibility rule, and
the defense. Everything in these three lessons follows from them.
