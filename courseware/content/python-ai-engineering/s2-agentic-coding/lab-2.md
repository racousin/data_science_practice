# Lab 2 — Agent-Driven Feature

Same pairs, same repository as Lab 1. You will add a feature using an agent and
open a pull request that would survive a real review.

**Time:** 45 minutes. **Deliverable:** a merged PR + a short retrospective file.

---

## Part A — Set the context (10 min)

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

Ask for a **plan only**:

```text
> Read src/textstats/ and tests/. Propose how to add a Flesch reading-ease
> score: module, signature, syllable-counting approach, edge cases, and the
> tests you would write. Do not write any code.
```

**Save the plan** into `RETRO.md` under a heading `## Plan`. Then push back on
it at least once — a real objection, in writing, before any code exists.

---

## Part C — Test first (10 min)

```text
> Write the tests from the plan. Do not write the implementation.
```

Read every test. At minimum you must have:

- a known-value case (compute one by hand and check it)
- empty input → raises, does not return `0.0`
- text with no sentence-ending punctuation

If the generated tests do not fail for the right reason, they are not tests.
Run them and confirm they fail:

```bash
uv run pytest -v
```

---

## Part D — Implement and verify (10 min)

```text
> Now implement it so the tests pass. Do not modify the tests.
```

Let the loop run. Then, yourself:

```bash
git diff
uv run pytest
```

**Reject and re-prompt** if you see any of: a bare `except`, a default value for
a required argument, a new dependency, or an edited test.

---

## Part E — Retrospective + PR (5 min)

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

Push, open the PR, get your partner's review, merge.

---

## Grading

| Criterion | Weight |
|---|---|
| `CLAUDE.md` is specific to this project, not generic | 20% |
| Tests written before implementation, and meaningful | 25% |
| `RETRO.md` shows real pushback, not a transcript | 25% |
| Diff is clean: no silent failure, no unapproved deps | 20% |
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
