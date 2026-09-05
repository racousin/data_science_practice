# Lab 2 — Agent-Driven Feature

Same pairs, same repository as Lab 1. You will add a feature using an agent and
open a pull request that would survive a real review.

**Time:** 45 minutes in class for Parts A–E; Part F is five more, in the room
if there is time and before the next session otherwise. **Deliverable:** a
merged PR, a short retrospective file, and a scored submission on
competition `180`.

---

## Part A — Set the context (10 min)

If you still have the throwaway `agent-sandbox` branch from *Setup*, throw it
away first: `git restore . && git switch main && git branch -D agent-sandbox`.

1. `git switch main && git pull && git switch -c feature/readability`
2. Confirm `git status` is clean.
3. Create `CLAUDE.md` (or `CONVENTIONS.md` for Aider) covering:
   - the install / test / lint commands
   - your `src/` layout and test-mirroring convention
   - the fail-fast rule: no defaults for required arguments, no bare `except`
   - one explicit "do not": no new dependencies without asking

Write it yourself, or run `/init` and then **edit it** — an unedited `/init`
draft does not count.

---

## Part B — Plan before code (10 min)

The feature: a `readability` module implementing the Flesch reading-ease score.

$$
206.835 - 1.015 \times \frac{\text{words}}{\text{sentences}} - 84.6 \times \frac{\text{syllables}}{\text{words}}
$$

The formula is the easy part. *Word*, *sentence* and *syllable* are not defined
by it, and syllable counting has no canonical answer — so the two sections
below pin one. Those rules are the specification, they are what competition
`180` grades against, and you hand them to your agent **verbatim**.

---

## Part B — the pinned specification

**Sentences.** Count the maximal *runs* of characters drawn from `.!?`.
`"Wait... no!"` is **two** sentences, not four — the `...` is one run. A text
with no such punctuation counts as **one** sentence, never zero.

**Words.** Split on whitespace, then strip leading and trailing characters that
are not letters or digits. Tokens that become empty are dropped.
`"end."` → `end`; `"--"` → dropped; `"under_scores"` → `under_scores`, because
the stripping is only at the ends.

**Syllables**, per word. Lowercase it and count the maximal runs of `aeiouy`
(`y` counts). Then: if the word ends in `e` **and** that count is greater than
1, subtract 1. The result is never less than 1.

---

## Part B — worked syllable counts

| word | vowel runs | ends in `e`? | syllables |
|---|---|---|---|
| `time` | `i`, `e` → 2 | yes, and count > 1 → −1 | **1** |
| `the` | `e` → 1 | yes, but count is 1 → no change | **1** |
| `place` | `a`, `e` → 2 | yes → −1 | **1** |
| `queueing` | `ueuei` is *one* run → 1 | no | **1** |
| `rhythm` | `y` → 1 | no | **1** |
| `dryly` | `y`, `y` → 2 | no | **2** |
| `reevaluation` | `ee`, `a`, `ua`, `io` → 4 | no | **4** |
| `42` | none → 0 | no | **1** (the floor) |

`queueing` is the one worth staring at: `u e u e i` are five *contiguous*
vowels, so the rule sees a single run. That is not how English works, and it is
still the answer — the spec is the spec. An implementation that is right about
the arithmetic and has its own opinion about the rules scores half. Part F has
the measured numbers.

---

## Part B — ask for the plan

Ask for a **plan only**, and paste the specification into the prompt:

```text
> Read src/textstats/ and tests/. Here is the specification for a Flesch
> reading-ease score, which is fixed and not up for negotiation:
> <paste the three pinned rules, verbatim>
> Propose how to add it: module, signature, syllable-counting approach, edge
> cases, and the tests you would write. Do not write any code.
```

Pasting the rules is the whole trick. Without them the agent invents a
syllable heuristic, you have no way to say it is wrong, and the leaderboard
disagrees with you twenty times.

**Save the plan** into `RETRO.md` under a heading `## Plan`. Then push back on
it at least once — a real objection, in writing, before any code exists.

---

## Part C — Test first (10 min)

```text
> Write the tests from the plan. Do not write the implementation.
```

Read every test. At minimum you must have:

- **a known value.** `flesch_reading_ease("The cat sat on the mat.")` is
  `pytest.approx(116.145)` — 6 words, 1 sentence, 6 syllables. Do that
  arithmetic yourself before you accept the number.
- **one case per pinned rule.** `"Wait... no!"` is two sentences; `dryly` is
  two syllables; `queueing` is one; `42` is one. If your module exposes the
  counters, assert on them directly. If it does not, one whole-text assertion
  covers the `y` rule five times over:
  `flesch_reading_ease("Rhythm myths fly by dryly.")` is
  `pytest.approx(100.24)`.
- **empty input → raises `ValueError`**, does not return `0.0`.
- **text with no sentence-ending punctuation** — one sentence, never zero.

If the generated tests do not fail for the right reason, they are not tests.
Run them and confirm they fail:

```bash
uv run --all-extras pytest -v
```

`--all-extras` is what installs `pytest` when your Session 1 `pyproject.toml`
declares it under `[project.optional-dependencies]`. If yours declares a
`[dependency-groups] dev` instead, the flag is a harmless no-op.

---

## Part D — Implement and verify (10 min)

```text
> Now implement it so the tests pass. Do not modify the tests.
```

Let the loop run. Then, yourself:

```bash
git diff
uv run --all-extras pytest
```

**Reject and re-prompt** if you see any of: a bare `except`, a default value for
a required argument, a new dependency, or an edited test.

---

## Part E — Retrospective (5 min) + PR

Finish `RETRO.md`:

```markdown
## Plan
<the plan you were given>

## My objection
<what you pushed back on, and why>

## What I rejected
<at least one thing the agent produced that you refused, and why>

## What I could not explain
<any line you had to go and understand — or "none", honestly>
```

Push and open the PR now. Your partner's review and the merge are
**homework** — they need a second person to stop what they are doing, and five
minutes of class time does not buy that.

---

## Part F — Put it on the board (5 min)

**PAIE S2 — Flesch reading-ease** (competition `180`) runs your module against
twenty hidden texts and the reference implementation of the specification in
Part B. Same rules, same tie-breaks, no taste involved — which is the point:
this is the one part of the lab that is settled by a number rather than by a
reader.

Copy `readability.py` out of your package into a flat directory. It must not
import anything from `textstats`, because only the files you upload are there.
Put a six-line `agent.py` next to it:

```python
from readability import flesch_reading_ease


class Agent:
    def __init__(self):
        pass

    def flesch_reading_ease(self, text):
        return flesch_reading_ease(text)
```

Then submit both files:

```bash
uv pip install mlarena-sdk
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=180, files=["agent.py", "readability.py"])
print(client.status())                             # queue_info / run_info / message
```

The package is `mlarena-sdk`; it imports as `mlarena`. Once the run has
finished, `client.leaderboard(180)` and the competition page both show your
score. If `client.competition(180)` raises `CompetitionNotFoundError`, the
competition is not open yet — tell your teacher, it is one command on their
side.

---

## The number you are aiming at

The ranked column is **pass rate**: the fraction of the twenty texts you answer
within `1e-6` of the reference. It runs 0% to 100% and **higher is better**.
Measured on this exact evaluation set:

| implementation | pass rate | mean abs error |
|---|---|---|
| the starter you are given (`raise NotImplementedError`) | 0.0% | 0.0000 (nothing to measure) |
| every rule guessed: vowel *letters* not runs, no `y`, no silent `e`, no stripping, one sentence per `.!?` character | 20.0% | 43.6401 |
| the spec followed except that `y` is not a vowel | 50.0% | 5.4622 |
| the pinned specification, implemented exactly | **100.0%** | 0.0000 |

**A completed Lab 2 scores 100.0%, 20 of 20.** Unusually, the bar is a ceiling
rather than a target: the spec is pinned, so anything below it means your rules
and the specification disagree somewhere. Mean absolute error tells you how
badly — around `5` is one rule, above `40` is several. The competition overview
has the full ladder, one rung per rule.

Note the third row. Dropping a single rule — `y` — costs you half the board
while leaving an implementation that looks entirely reasonable in review. That
is the argument for pinning a specification before you prompt, made in numbers.

---

## Grading

| Criterion | Weight |
|---|---|
| `CLAUDE.md` is specific to this project, not generic | 15% |
| Tests written before implementation, and meaningful | 25% |
| `RETRO.md` shows real pushback, not a transcript | 20% |
| Diff is clean: no silent failure, no unapproved deps | 20% |
| Competition `180` pass rate 100.0% (20/20) | 10% |
| PR reviewed by your partner before merge | 10% |

---

## The rule for this lab

> If you cannot explain a line, it does not merge.

You may be asked to walk through any line of the diff. "The agent wrote it" is
not an answer.

---

## If you finish early

Ask the agent to review its own work with a fresh session:

```text
> /clear
> Review the diff between main and this branch for silent failure handling,
> missing edge cases, and tests that assert nothing. Do not fix anything.
```

A clean context finds things the authoring context is blind to. Add whatever it
finds — and whether you agreed — to `RETRO.md`.

---

## Did you validate this session?

- [ ] `uv sync --all-extras && uv run pytest` is green in a fresh clone of my
      repository, on the `feature/readability` branch
- [ ] `CLAUDE.md` names my install / test / lint commands, my `src/` layout and
      at least one "do not", and is not unedited `/init` output (Part A)
- [ ] `RETRO.md` contains the plan the agent proposed **and** the objection I
      wrote before any code existed (Part B)
- [ ] My tests contain the known value `pytest.approx(116.145)`, an empty-input
      case expecting `ValueError`, and a case with no terminal punctuation
      (Part C)
- [ ] Those tests failed before the implementation existed, and pass now
      (Parts C and D)
- [ ] `git diff main...HEAD` shows no bare `except`, no default value on a
      required argument, no edited test and no new dependency (Part D)
- [ ] `RETRO.md` names at least one thing the agent produced that I refused,
      and why (Part E)
- [ ] My submission is on the leaderboard of **PAIE S2 — Flesch reading-ease**
      (`#180`)
- [ ] My score reaches the bar: **pass rate = 100.0%** (20/20). 50% means the
      arithmetic is right and one rule is wrong

If the last two are not ticked you have not finished the lab, however good the
code is.
