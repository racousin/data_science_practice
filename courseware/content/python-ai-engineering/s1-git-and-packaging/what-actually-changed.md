# What Actually Changed

This comes at the end of Session 1 and not at the end of the module, on purpose:
you will use these tools in Sessions 2, 3 and 4 and across all 30 hours of
*MS2A - Machine Learning Practice*. A tool taught in the final hour is a demo.
Taught now, it is leverage.

It also comes *after* git, tests and review rather than before, and that order is
not decoration. An agent that edits your repository is only safe on top of a
branch you can throw away, a test suite it has to satisfy, and a diff you read.
You now have all three.

<!-- notes: 20 minutes. Resist the urge to evangelise. The room contains both
students who use these tools daily and students who have never opened one. -->

---

## What that actually buys you

Not "writes code for you". More precisely:

- **Repository-wide edits** — rename a concept across 40 files, consistently.
- **Unfamiliar-codebase navigation** — "where does the leaderboard ordering come from?"
- **Mechanical translation** — a notebook into a tested module; JS into TS.
- **Test-driven grind** — write the failing test, let the loop close it.

These are the tasks that are tedious rather than hard. That is the honest
description of the current sweet spot.

---

## What it does not buy you

- **Judgement about what to build.** The agent optimises the metric you name.
  Naming the wrong metric is a Session 3 problem and the agent will not save you.
- **Correctness you did not check.** Plausible and correct are different
  properties. The agent produces the first reliably and the second often.
- **Understanding.** If you cannot explain the diff, you cannot defend it in
  your project viva, and you should not merge it.

---

## The accountability rule

> You are the author of every line you merge, whoever typed it.

This is not a moral position, it is how the grading works. Your project is
assessed on repository quality and you will be asked to explain your code. "The
agent wrote it" is not an answer to "why is the learning rate scheduled here?".

<!-- notes: Say this plainly and once. Repeating it turns into moralising and
they stop listening. -->

---

## Where the failures come from

Almost every bad outcome traces to one of three causes:

1. **Underspecified task** — you asked for "improve the model", got 400 lines of
   changes, and cannot tell what happened.
2. **No verification** — no tests, so neither of you knows if it works.
3. **Too much at once** — one enormous change instead of five reviewable ones.

The rest of this session is three techniques against exactly those three causes.

---

## What we will cover

1. **Setup** — Claude Code, and an open alternative for those without access.
2. **The core loop** — explore, plan, act, verify. Where you intervene.
3. **Context engineering** — what the agent knows, and how you control it.
4. **Guardrails** — permissions, scope, and tests as the contract.

Then **Lab 3**: extend your `textstats` package using an agent, with a pull
request that a human would actually approve.
