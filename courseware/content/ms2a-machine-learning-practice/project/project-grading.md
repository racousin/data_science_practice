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

| Anchor | Agent | Score | Mark |
|---|---|---|---|
| A0 | the model answers directly | 0.6 | 6 / 20 |
| A1 | the kit as shipped | 13.9 | 12 / 20 |
| A2 | the instructor harness | ≈ 61 | 18 / 20 |
| | A2 + 5 points or more | ≥ ≈ 66 | 20 / 20 |

Linear between two anchors; proportional below A0.

Example: a score halfway between A1 and A2 earns 15 / 20.

---

## Leaderboard: the final set

The live leaderboard is **indicative**. The mark uses a final run:

- after the freeze, the private set is **regenerated**: a new draw, the same
  families, paraphrased prompts
- your team's chosen submission runs **twice** on it
- your score is the mean of the two runs

A harness tuned to the wording of the live private set loses points here. A
harness that reads the prompt and the files does not.

---

## The oral

10 minutes of presentation, 10 minutes of questions, both members.

| | Criterion | Weight |
|---|---|---|
| O1 | Problem understanding | 15 % |
| O2 | Harness design | 20 % |
| O3 | Evaluation methodology | 30 % |
| O4 | Resource engineering | 10 % |
| O5 | Individual mastery (Q&A on any component) | 25 % |

O3 and O5 are more than half the oral.

---

## What each criterion looks for

- **O1**: the levels, the score, the constraints, where the points are
- **O2**: why each component of your harness exists, with the measurement
  that justified it
- **O3**: your own validation set, an ablation table, an error analysis by
  family, the gap between local and leaderboard scores explained
- **O4**: model choice, seconds per task, memory, the time budget: measured,
  not assumed
- **O5**: each member explains any component, including the one the
  partner wrote

---

## Standard questions

Expect several of these:

- Which change gave the most points, and how do you know?
- Show a change you removed. Why did it not pay?
- Where does your harness lose the most points today?
- Why does your local score differ from the leaderboard?
- How long does a level-3 task take, and where does the time go?
- Why this model and not a larger one?
- Walk us through what happens to one task, from prompt to answer.

---

## Deliverables at the freeze

1. **The submission** on the leaderboard that your team chooses for the
   final run
2. **The repository**, at a tagged commit that reproduces the submitted
   `agent.py` and your local dev score: a README, pinned dependencies, one
   evaluation command
3. **Slides**, PDF, 12 at most, for the oral

All three by **2026-11-20 23:59**. A deployment counts if it is **queued**
by then. The GPU runs one job at a time (about 9 min each), so the queue
is long near the freeze: aim to queue your final submission by
**2026-11-19**.

---

## Calendar

| Date | What |
|---|---|
| now | leaderboard open |
| 2026-10-15 | teams declared |
| 2026-11-16 | checkpoint: 1-page design + ablation draft (feedback only) |
| 2026-11-19 | aim to queue your final deployment |
| **2026-11-20 23:59** | **freeze**: last deployment queued, slides, repository tag |
| 2026-11-21 → 11-23 | final runs on the regenerated private set |
| 2026-11-24 → 11-27 | orals |

---

## Check yourself

1. Your team's live score is between A1 and A2. After the freeze it drops by
   6 points. What most likely happened?

   **Answer.** The final set is regenerated with paraphrased prompts and a new
   draw. A harness that keyed on the live wordings, or that ran close to the
   time limit, loses points there. A general one keeps its score.

2. Your ablation table shows a change that gained 3 points locally and lost 2
   on the board. Keep the row or drop it?

   **Answer.** Keep it, and explain the gap: it is exactly what O3 asks for.
   A table with only gains does not show a method.

3. Your partner wrote the level-3 tool. The jury asks you what it does when
   the fit takes longer than the task's time budget. What should you be able to do?

   **Answer.** Answer it. O5 (25 %) is individual mastery of every component,
   whoever wrote it.
