# Context Engineering

The agent knows exactly what is in its context window. Everything you find
frustrating about its behaviour is, most of the time, something it could not see
or something misleading that it could.

<!-- notes: 30 minutes. This is the lesson that separates students who find
agents useful from students who find them annoying. -->

---

## What is in there

At any moment the context holds:

- the system prompt (the tool's own instructions)
- your project instructions (`CLAUDE.md` / `CONVENTIONS.md`)
- the conversation so far
- the contents of every file it has read this session
- the output of every command it has run

It does **not** hold: your repository as a whole, your intentions, last week's
session, or anything you did not say.

---

## Two failure modes

| | Symptom | Cause |
|---|---|---|
| **Starvation** | invents an API you already have; wrong conventions | it never read the relevant file |
| **Pollution** | drifts, repeats an abandoned approach, gets slower | the window is full of dead ends |

Starvation is fixed by pointing. Pollution is fixed by `/clear`.

---

## Point at files explicitly

Do not make it search when you already know:

```text
> Look at src/textstats/metrics.py and tests/test_metrics.py.
> Follow the same structure for the new readability module.
```

Naming an existing file as the pattern to follow is the single highest-return
habit in this lesson. It replaces a paragraph of style description.

---

## CLAUDE.md

A file at your repository root, read automatically at the start of every
session. It is the place for things that are true about your project and would
otherwise be repeated every time.

```bash
claude
> /init
```

`/init` writes a first draft by inspecting the repo. Then you edit it — the
draft is a starting point, not a deliverable.

---

## What belongs in it

```markdown
# CLAUDE.md

## Commands
- Install: `uv sync --all-extras`
- Test: `uv run --all-extras pytest`
- Lint: `uv run --all-extras ruff check src/ tests/`

## Conventions
- Python 3.11+, full type hints on public functions.
- `src/` layout. New modules go in `src/textstats/`.
- Tests mirror the source tree: `src/x/y.py` -> `tests/test_y.py`.

## Fail fast
No default values for required config. No silent `except`.
A missing key should raise `KeyError`, not fall back to a default.

## Do not
- Do not add dependencies without asking.
- Do not edit `uv.lock` by hand.
```

---

## What does not belong in it

- **Anything the code already says.** It can read the code.
- **A description of the file tree.** It will drift within a week.
- **Essays.** Every token here is in the context of every session forever.

The test for a line in `CLAUDE.md`: *would I have to say this again next time?*
If not, delete it.

---

## Why "do not" rules matter

Negative constraints are what stop the expensive surprises: an upgraded
dependency, a rewritten lockfile, a "helpful" refactor of a module you did not
mention. They cost one line each.

<!-- notes: The dependency one bites students in Session 4 — the agent upgrades
torch, the training script stops matching the lab notes. -->

---

## Scoping to a directory

A `CLAUDE.md` deeper in the tree applies to work in that subtree. Useful for a
monorepo, or for a `notebooks/` directory with different conventions from `src/`.

The nearest file wins for its subtree; the root file still applies.

---

## Aider's equivalent

`CONVENTIONS.md`, loaded explicitly:

```bash
aider --read CONVENTIONS.md
```

Same content, same purpose. Aider also wants you to name the files in scope:

```bash
aider src/textstats/core.py tests/test_core.py
```

Being explicit about scope is a virtue in both tools; Aider just enforces it.

---

## Session memory

Claude Code keeps notes across sessions. Add one with `#`:

```text
# the leaderboard is ordered by ELO for agent competitions, mean_reward otherwise
```

Use it for facts you discovered that are not written down anywhere. Do not use
it for facts the repository already records — that is what `CLAUDE.md` and the
code are for.

---

## Slash commands

A reusable prompt, stored as a Markdown file in `.claude/commands/`:

```markdown
<!-- .claude/commands/review.md -->
Review the current diff for:
1. Silent failure handling (bare except, defaulted required config)
2. Missing tests on new public functions
3. Type hints on public signatures

Report findings as a list. Do not fix anything.
```

Then `/review` in any session. Team conventions become one keystroke, and they
live in the repository where they can be reviewed like code.

---

## Managing a long session

- `/clear` between unrelated tasks — cheaper than you think
- Summarise before switching: *"Summarise what we changed and why"*, then clear
- Prefer several focused sessions over one that has seen everything

A fresh session with a good prompt beats a tired session with a long history,
nearly every time.

---

## Checklist

- [ ] `CLAUDE.md` exists and holds commands + conventions + "do not"s
- [ ] It contains nothing the code already states
- [ ] You name files explicitly instead of making it search
- [ ] You `/clear` between unrelated tasks
- [ ] At least one `/review` style command is in `.claude/commands/`

---

## Check yourself

1. The agent invents a helper you already wrote, and separately it has started
   repeating an approach you abandoned twenty minutes ago. Name each failure
   mode and its fix.

   **Answer.** Inventing what exists is **starvation** — it never read the
   relevant file; fix it by pointing at the file by name. Repeating a dead end
   is **pollution** — the window is full of abandoned work; fix it with
   `/clear`.

2. Run this in your project. You should get exactly the output shown, and you
   have just satisfied the last line of the checklist above.

   ```bash
   mkdir -p .claude/commands
   printf 'Review the current diff for silent failure handling.\n' > .claude/commands/review.md
   cat .claude/commands/review.md
   # -> Review the current diff for silent failure handling.
   ```

   **Answer.** The file is now invokable as `/review` in any session in this
   repository — a reusable prompt that lives in the repo and can be reviewed
   like code.

3. You are about to add a line to `CLAUDE.md`. What is the one test it has to
   pass, and name two kinds of content that fail it.

   **Answer.** *Would I have to say this again next time?* If not, delete it.
   Anything the code already says fails it, and so does a description of the
   file tree (it drifts within a week) — as do essays, because every token here
   is in the context of every session forever.

4. You discover that the leaderboard is ordered by ELO for agent competitions
   and by mean reward otherwise. Does that go in `CLAUDE.md`, or behind `#`?

   **Answer.** Behind `#` — session memory is for facts you discovered that are
   not written down anywhere. `CLAUDE.md` is for what the project already
   requires of you: commands, conventions, and "do not"s.
