# The Brief

The project is half the grade of this course. It is one ML-Arena competition
track of your choosing, plus the repository that produced the submission. Both
are graded, and neither one rescues the other.

<!-- notes: 20 minutes, run it in Session 3, once they have seen a leaderboard.
Put the real dates on the board — every date in this file is a placeholder. Save five minutes for the reproducibility rule; it is the one they
skim past and the one that costs them. -->

---

## What is being asked for

Two artefacts, one term:

- an **ML-Arena submission** on one of three tracks, scored on a public
  leaderboard against your cohort and against a published baseline
- the **repository** that produced it — installable, tested, and reproducible
  by someone who has never seen it

The competition makes the work measurable. The repository makes it defensible.
A number nobody can reproduce is not a result, it is a claim.

---

## Where the grade comes from

| Component | Share |
|---|---|
| Continuous assessment — the ten labs | 50% of the course |
| Project | 50% of the course |
| — leaderboard performance | half the project |
| — repository quality | half the project |

The second axis is the 12-hour *Python AI Engineering* module coming back to
collect: git history, a package that installs, tests that can fail, pinned
dependencies. You were taught all of it; this is where it is examined.

---

## Teams of one or two

Two is the default. One is allowed and is not marked more gently.

A team is a **first-class object on the platform**, not a line in an email. You
create it on the competition page and invite your partner; from then on
submissions attribute to the team rather than to whoever happened to click.

One team per competition per person, and it is set at the declaration deadline.
Changing partners later means abandoning your submission history, which is part
of what you are graded on.

---

## Timeline

Every date below is a **placeholder** until the term calendar is fixed. The
anchors — which session each milestone follows — are not.

| # | Milestone | Anchor | Date |
|---|---|---|---|
| 1 | Team and track declared | after Session 3 | `TBD` |
| 2 | Repository skeleton pushed | after Session 4 | `TBD` |
| 3 | First scored submission on the board | after Session 5 | `TBD` |
| 4 | Mid-point check | after Session 7 | `TBD` |
| 5 | Freeze — the leaderboard closes | after Session 10 | `TBD` |
| 6 | Defense | exam week | `TBD` |

Milestone 4 is a fifteen-minute conversation, not a deliverable: you show the
board, the repository, and the experiment table as it stands.

---

## Milestone 1 — declare

Three facts: team members, chosen track, repository URL.

The repository is **private**, with the instructor added as a collaborator on
the day it is created — not on the day of the freeze. A grader who cannot clone
it cannot grade it.

---

## Milestone 2 — the skeleton

Not a placeholder commit:

```text
pyproject.toml        # installable, pinned
src/<yourpkg>/        # the package, importable
tests/                # at least one test that can fail
README.md             # install, and how to run
.gitignore            # data/, weights, .env
```

A repository whose first real commit lands in the last week is visible in the
history and is marked as such.

---

## Milestone 3 — get on the board early

The first submission exists to prove the pipeline, not to score. Submit a
baseline — a constant predictor, a random agent, a one-sentence prompt — and
confirm it is scored.

```python
import mlarena, os

client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
client.submit(competition_id=COMP_ID, files=["submission.csv"])
print(client.status())
```

`os.environ[...]`, not `os.getenv(..., "")`. A missing key should crash here,
loudly, rather than send an unauthenticated request you spend an hour
debugging.

---

## Milestone 5 — the freeze is the leaderboard

At the freeze timestamp, the board is read and that is your performance grade.
There is no email extension, no "it was training", no submission accepted
afterwards.

```python
client.leaderboard(COMP_ID, top=20)
```

Consequence: your best submission must be on the board *before* the deadline,
not merely producible before it. Teams lose marks every year to a run that
started at 22:00 and finished at 01:00.

---

## The reproducibility rule

> A submission you cannot reproduce from your repository scores **zero** on
> both axes.

Not "loses points". Zero. If the grader clones your repository, follows your
README, and cannot regenerate the artefact you submitted, there is nothing left
to grade — the leaderboard row is unattributable and the repository is a
different piece of work.

Reproducible means: same seed, same pinned versions, same data snapshot, same
command, and a result within the noise you documented.

---

## A good project versus a leaderboard-chasing one

| | Leaderboard-chasing | Good |
|---|---|---|
| Submissions | many, undocumented | few, each traceable to a commit |
| Choices | tried until something scored | motivated, then measured |
| Validation | the public leaderboard | a local split you trust |
| Failures | deleted | written down in the report |
| Repository | a notebook and a CSV | a package, tests, a README |
| Final number | best of 60 tries | the one you can defend |

Both can top the board. Only one of them passes.

---

## What "motivated" means in practice

You will make perhaps six decisions that matter: the validation protocol, the
model family, the features or the representation, the regularisation, the
compute budget, and what to do about the errors you looked at.

For each of them the report should be able to say what you tried, what it
scored on **your** validation, and why you kept what you kept. That is the
difference between an engineer and a random search with a human in the loop.

---

## The engineering stance carries over

The labs' rules are the project's rules, and they are checked the same way:

- crash at the boundary — no silent defaults for required configuration
- no bare `except`
- data and weights stay out of git
- credentials come from the environment
- every result comes from a command, not from a cell you ran once

None of this is style. Every item on that list is a way a result stops being
trustworthy.

---

## Start with the track

The next lesson describes the three tracks and what each one asks of you. Read
it, pick one within a week, and start submitting.

The most common failure of this project is not a bad model. It is a team that
spent five weeks deciding.
