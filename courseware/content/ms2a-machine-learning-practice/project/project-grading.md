# Deliverables and Grading

What you hand in at the freeze, how the leaderboard becomes a mark, and what
the oral asks. The project is 50 % of the course grade. Within the project: 25 %
leaderboard, 75 % oral.

<!-- notes: 20 minutes. Show the anchors table and work one example on the
board. Then read the oral rubric slowly: O3 (evaluation methodology, 30 %) is
the largest line, and it is earned during the term, not the night before. Every
member must be able to explain every component: say it twice. -->

---

## The project in the course grade

| Part | Share of the project |
|---|---|
| Leaderboard, against fixed anchors | 25 % |
| Oral, both members | 75 % |

The project is **50 %** of the course grade.

The leaderboard is not ranked against your cohort: it is marked against three
reference agents, so every team can reach 20/20.

---

## Leaderboard: the anchors

| Anchor | System | Score | Mark |
|---|---|---|---|
| A0 | stage 1: the model, a structured answer, a parser | 15.8 | 6 / 20 |
| A1 | stage 2 with routing: the calculator loop without files, stage 1 with files | 16.7 | 12 / 20 |
| A2 | the reference system: reading, data view, code, repair | 40.8 | 18 / 20 |
| | A2 + 5 points or more | ≥ 45.8 | 20 / 20 |

Scores on the private set. Linear between two anchors; proportional below A0.

Example: a score halfway between A1 and A2 earns 15 / 20.

---

## Leaderboard: the score that counts

The live leaderboard **is** the mark: your team's chosen submission, scored
on the private set, at the freeze. The logs and the task display of a
deployment come from its test run only (dev tasks); the private set is never
shown. Your best local score on `dev.json` is not the mark.

---

## The oral

10 minutes of presentation, 10 minutes of questions, both members.

| | Criterion | Weight |
|---|---|---|
| O1 | Problem understanding | 15 % |
| O2 | System design | 20 % |
| O3 | Evaluation methodology | 30 % |
| O4 | Resource engineering | 10 % |
| O5 | Individual mastery (Q&A on any component) | 25 % |

O3 and O5 are more than half the oral.

---

## What each criterion looks for

- **O1**: the families, the score, the constraints, where the points are
- **O2**: why each component of your system exists, with the measurement
  that justified it
- **O3**: your own validation set, an ablation table, an error analysis by
  family, the gap between local and leaderboard scores explained
- **O4**: model choice, seconds per call, memory, the 40 s: measured,
  not assumed
- **O5**: each member explains any component, including the one the
  partner wrote

---

## Standard questions

Expect several of these:

- Which change gave the most points, and how do you know?
- Show a change you removed. Why did it not pay?
- Where does your system lose the most points today?
- Why does your local score differ from the leaderboard?
- How long does a call of 8 tasks take, and where does the time go?
- Why this model and not another of the five?
- Walk us through what happens to one task, from objective to answer.

---

## Deliverables at the freeze

1. **The submission** on the leaderboard that your team chooses: its score
   at the freeze is the leaderboard mark
2. **The repository**, at a tagged commit that reproduces the submitted
   `agent.py` and your local dev score: a README, pinned dependencies, one
   evaluation command
3. **Slides**, PDF, 12 at most, for the oral

All three by **2026-11-03 23:59**. A deployment counts if it is **queued**
by then. The GPU runs one job at a time (about 10 min each), so the queue is
long near the freeze: queue your final submission a day early.

---

## Calendar

| Date | What |
|---|---|
| now | leaderboard open |
| 2026-10-15 | teams declared |
| **2026-11-03 23:59** | **freeze**: last deployment queued, slides (PDF), repository tag |
| 2026-11-04 | orals |

---

## Check yourself

1. Your team scores 8 points more on `dev.json` than on the leaderboard.
   Which number is your mark?

   **Answer.** The leaderboard: it is the private set, which has other draws
   and table types `dev.json` does not have. The gap measures how much your
   system is keyed on the dev tasks.

2. Your ablation table shows a change that gained 3 points locally and lost 2
   on the board. Keep the row or drop it?

   **Answer.** Keep it, and explain the gap: it is exactly what O3 asks for.
   A table with only gains does not show a method.

3. Your partner wrote the fitting tool. The jury asks you what it does when
   the training rows have missing values. What should you be able to do?

   **Answer.** Answer it. O5 (25 %) is individual mastery of every component,
   whoever wrote it.
