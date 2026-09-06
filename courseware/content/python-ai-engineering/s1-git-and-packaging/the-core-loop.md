# The Core Loop

Every agent, whatever the brand, runs the same four-step cycle. Knowing where
you sit in it is the difference between driving and spectating.

<!-- notes: 35 minutes. Do this live on the projector with a real task in a real
repo. The demo is the lesson; the slides are the notes. -->

---

## The cycle

![The agent loop](assets/s1-git-and-packaging/the-core-loop/agent-loop.png)

**Your leverage is highest at PLAN and VERIFY.** Those are the two steps
students skip.

---

## Step 1 — Explore

Before any edit, the agent should know what exists. You can ask for this
explicitly, and on an unfamiliar codebase you should:

```text
> Read src/textstats/ and tests/. Summarise the public API and
> tell me what is NOT covered by tests. Do not change anything yet.
```

"Do not change anything yet" is doing real work in that prompt. Without it you
get an answer *and* a diff you did not ask for.

---

## Step 2 — Plan

Make the approach explicit before the code exists. Reviewing a paragraph costs
thirty seconds; reviewing 300 lines costs twenty minutes.

```text
> I want to add a `readability_score` function using Flesch–Kincaid.
> Propose an approach: signature, where it goes, what the edge cases are,
> and what tests you would write. Do not write code yet.
```

Claude Code has a dedicated **plan mode** (`Shift+Tab` cycles into it) where the
agent researches and proposes but cannot edit. Use it for anything you cannot
describe in one sentence.

---

## Why planning works

A wrong plan is cheap and obvious. Wrong code is expensive and plausible.

When you read a plan you catch: the wrong file, a misunderstood requirement, an
approach that ignores an existing helper. All three are invisible in a diff
until you have already spent the attention.

---

## Step 3 — Act

Now let it work. Two habits:

**Keep the unit small.** One function, one bug, one refactor. If the response is
touching six files you did not expect, interrupt.

**Watch the commands.** The agent will run things. Read what it is about to run.
This is why permissions exist — next lesson.

---

## Step 4 — Verify

This is the step that separates a tool from a toy.

```text
> Run the tests and fix anything that fails.
```

Now the loop closes on its own: it edits, runs `pytest`, reads the traceback,
edits again. You are supervising a search, not typing.

**No tests means no loop.** With nothing to check against, the agent stops when
the code *looks* finished. This is the strongest practical argument for the test
suite you wrote in Lab 1.

---

## Test-first, deliberately

The strongest pattern available to you:

```text
> Write a failing test for: longest_word() should raise ValueError
> on a string containing only whitespace. Do not fix the implementation.
```

Then, once you have read the test and agree it encodes what you meant:

```text
> Now make it pass.
```

You have defined correctness before any implementation existed. The agent cannot
satisfy the test by misunderstanding you, because you read the test.

---

## Anatomy of a good prompt

Compare:

> add caching

against:

> `load_dataset()` in `src/textstats/data.py` re-reads the CSV on every call.
> Add an in-memory cache keyed by the resolved absolute path. Invalidate when
> the file's mtime changes. Add tests for: cache hit, cache miss on a modified
> file, and two different paths. Keep the public signature unchanged.

The second names the file, the problem, the mechanism, the edge cases, and the
constraint. It is thirty seconds of typing that replaces two rounds of
correction.

---

## The four ingredients

1. **Where** — the file or function, by name.
2. **What** — the observable behaviour you want.
3. **How you will know** — the tests, the metric, the command that must pass.
4. **What must not change** — signatures, dependencies, file layout.

The fourth is the one people forget, and it is why an agent "helpfully" upgrades
your dependencies.

---

## Interrupting well

When you interrupt, say what was wrong. The context is preserved, so a
correction is cheaper than a restart.

```text
[Esc]
> Stop. You are editing tests/ but I asked you to change src/.
> Revert the test changes and only modify src/textstats/core.py.
```

Restarting from scratch throws away the exploration you already paid for.

---

## When to abandon the session

Start over — `/clear` — when:

- the agent has looped twice on the same failure
- the conversation has drifted far from the current task
- you changed your mind about the approach

A conversation carrying three abandoned approaches makes the fourth one worse.
Clear it, and put what you learned into the prompt.

---

## Recap

| Step | Your move |
|---|---|
| Explore | ask for a summary, forbid edits |
| Plan | demand the approach in prose, read it |
| Act | keep the unit small, watch the commands |
| Verify | tests must exist and must run |
