# What Actually Changed

This session is placed second, not last, on purpose: you will use these tools in
Sessions 3 and 4 and across all 30 hours of *MS2A - Machine Learning Practice*. A
tool taught in the final hour is a demo. Taught now, it is leverage.

<!-- notes: 20 minutes. Resist the urge to evangelise. The room contains both
students who use these tools daily and students who have never opened one. -->

---

## Three generations of assistance

| Generation | What it sees | What it does | Your job |
|---|---|---|---|
| **Autocomplete** | the current line | suggests the next token | accept / reject |
| **Chat** | what you paste | answers, you copy back | transcribe and integrate |
| **Agent** | your repository | reads, edits, runs commands, iterates | specify and verify |

The jump that matters is the third. An agent has a **feedback loop**: it runs
your tests, reads the failure, and tries again. Chat cannot do that — it never
sees whether its answer worked.

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

Then a lab: extend the Session 1 package using an agent, with a pull request
that a human would actually approve.

---

## Check yourself

1. Chat models and agents both write code. Name the one capability the table
   gives the agent and not chat, and say why it changes your job.

   **Answer.** A feedback loop: the agent runs your tests, reads the failure and
   tries again — chat never sees whether its answer worked. So your job moves
   from transcribing to specifying and verifying.

2. A classmate says "the agent will pick a good metric for my model". Which of
   the three things this lesson says an agent does *not* buy you is that, and
   what happens?

   **Answer.** Judgement about what to build. The agent optimises the metric you
   name; naming the wrong one is a Session 3 problem and the agent will not save
   you from it.

3. Almost every bad outcome traces to one of three causes. Name them.

   **Answer.** Underspecified task, no verification, and too much at once — and
   the rest of the session is three techniques against exactly those.
