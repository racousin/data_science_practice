# Coding Agents

One lesson on the tools that write code for you: what they are, what they cost,
how good they actually are, and the loop you have to run to get anything
trustworthy out of them.

It comes *after* git, tests and review rather than before, and that order is not
decoration. An agent that edits your repository is only safe on top of a branch
you can throw away, a test suite it has to satisfy, and a diff you read. You now
have all three.

<!-- notes: 60 minutes, and it replaces the six-lesson block this session used
to carry. Resist evangelising and resist a feature comparison — the room already
contains daily users and people who have never opened one. The two things that
transfer are the taxonomy and the loop; the price and score tables are there so
students stop asking and start choosing. -->

---

## Three generations

![Three generations](assets/s1-git-and-packaging/assistant-landscape/generations.png)

The jump that matters is the third. An **agent** has a feedback loop: it runs
your tests, reads the failure, and tries again. Chat cannot — it never finds out
whether its answer worked.

Everything else about a tool is interface.

---

## Where the tools sit

![The map](assets/s1-git-and-packaging/assistant-landscape/assistant-map.png)

Two axes predict how a tool behaves:

- **Terminal / CI, or editor UI.** A terminal tool can be scripted, run in a
  workflow, and used over SSH. An editor tool sees your cursor and your open
  files.
- **Proprietary product, or open source.** Determines whether you can run it
  against a model of your choosing, and whether it works with no network.

---

## The families

| Family | Examples | Shape |
|---|---|---|
| **Terminal agents** | Claude Code, Codex CLI, Gemini CLI | run in your repo, read/edit/execute, loop on tests |
| **Editor agents** | Cursor, Windsurf, Copilot agent mode | the same loop inside the IDE, with cursor context |
| **Notebook agents** | Gemini in Colab | the loop inside a notebook, over your cells and data |
| **Inline completion** | Copilot, Codeium, Tabnine | next-lines suggestions as you type |
| **Open-source agents** | Aider, Continue.dev, OpenHands | same loop, you pick the model |
| **Local models** | Ollama + any of the above | no network, no account, weaker |
| **Review bots** | CodeRabbit, Copilot review, `gh` workflows | run on the pull request, comment on the diff |

<!-- notes: Say plainly that this table will be wrong within a year and the
column that will still be right is "Shape". -->

---

## What they cost

**Checked 2026-09-06.** These move constantly; the row that will still be true
next term is the *shape* of the price, not the number. Follow the links before
you pay for anything.

| Tool | Free tier | Individual paid |
|---|---|---|
| [Gemini in Colab](https://colab.research.google.com) | **yes** — built in, nothing to install | included with Colab Pro |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | **yes** — 60 req/min, 1 000 req/day, Apache-2.0 | paid API key for more |
| [GitHub Copilot](https://github.com/features/copilot/plans) | yes — 2 000 completions/month | Pro **$10**, Pro+ **$39**, Max **$100** |
| [Claude Code](https://claude.com/pricing) | no | Pro **$20**, Max 5× **$100**, Max 20× **$200** |
| [Cursor](https://cursor.com/pricing) | Hobby — limited agent requests | Pro **$20**, Pro+ ≈3×, Ultra ≈20× |
| [Aider](https://aider.chat) | the tool is free | you pay the model's API bill |
| Aider + [Ollama](https://ollama.com) | **free, entirely** | — |

Per month, per person. Two things students consistently miss:

- **Students get most of this free.** Check
  [education.github.com](https://education.github.com) and the model vendors'
  student offers before paying — Copilot in particular is free with a verified
  university email.
- **A subscription and an API key are different products.** Under a
  subscription, a long agent session is included. Under an API key it is
  metered: Anthropic's list price is **$5 / $25** per million input / output
  tokens for Opus 5, **$2 / $10** for Sonnet 5, **$1 / $5** for Haiku 4.5. An
  afternoon of careless agent use on a metered key is a real number on a real
  card.

---

## Gemini is already in Colab

Worth stating on its own, because this course runs its notebooks in Colab and
this is the one option that costs nothing and installs nothing.

Colab ships a Gemini assistant in the notebook itself: a chat panel (the spark
icon in the footer), code generation and transformation in a cell, error
explanation with a diff view, and a **Data Science Agent** that plans a
multi-step analysis, runs it, reads its own results and corrects itself.

It is **enabled by default for eligible accounts** — the requirements are a
Google account aged 18+ and a supported locale, not a paid plan. If you do not
see the spark icon, that is what to check.

The consequence for you: even with no budget, no account and no install, you can
run the loop this lesson teaches. Colab is where you have a free GPU anyway.

---

## How good are they, actually

Coding benchmarks are a coarse ordering, not a measurement of your work.
**SWE-bench Verified**, the most-quoted one, asks a model to close 500 real
GitHub issues so the repository's own tests pass. A snapshot of the published
leaderboard (llm-stats.com, 2026-09-06):

| Model | SWE-bench Verified |
|---|---|
| Claude Fable 5 | 0.950 |
| Claude Opus 4.8 | 0.886 |
| Claude Sonnet 5 | 0.852 |
| Gemini 3.1 Pro | 0.806 |
| DeepSeek-V4-Pro-Max | 0.806 |

---

## Why you should not trust that table very far

Three reasons, and all three are the lesson:

1. **Contamination.** Those 500 tasks were public before several of these models
   were trained. Audits have shown frontier models reproducing the *verbatim*
   gold patch on some of them — partly remembering, not solving. This is why the
   benchmark's own successors (SWE-bench Pro) exist and why some vendors stopped
   reporting Verified in early 2026.
2. **It is out of date on arrival.** The newest model is usually not on the
   board yet. This table will be wrong before your project is due.
3. **It is not your repository.** The tasks are Python library bugs with a test
   suite already written. Your ML code has a different shape and, until Lab 1,
   no tests at all.

> The score that decides your tool is the one you measure on your own task.
> Give two tools the same real ticket from your project and compare the diffs.

---

## The harness is not the model

Every one of these tools is a **harness** around a **model**, and most let you
swap the model. That is why "which tool" and "which model" are two questions.

The pattern that holds across vendors: a **large** model for design, unfamiliar
code and hard debugging; a **small, fast** one for mechanical edits, renames and
boilerplate. Claude Code's `/model` switches between Opus 5, Sonnet 5 and
Haiku 4.5 for exactly that reason.

Use the default until you have a measured reason to change it. "Bigger model" is
not a fix for an underspecified task.

---

## Five questions that pick the tool

Not benchmark scores. These:

1. **What can it see?** One file, your open tabs, or the whole repository plus
   the output of commands it ran?
2. **Can it run things?** If not, it cannot verify, and you are back to chat.
3. **Where does your code go?** Some tools send the repository to a server. Read
   your employer's policy before you find out.
4. **What does it cost, and who pays?** Subscription, per-token API, or free.
5. **Can you leave?** A workflow built entirely on one product's interface does
   not transfer.

Question 3 is the one students underweight and employers do not.

---

## Setup — pick one and have it running

Everyone should leave this lesson with a working agent. Three routes; take the
first one available to you.

**Claude Code** — a terminal agent that runs where your code is:

```bash
npm install -g @anthropic-ai/claude-code          # needs Node.js 18+
curl -fsSL https://claude.ai/install.sh | bash    # macOS / Linux, no Node
```

```bash
cd my_project
claude
```

The first run walks you through authentication. Start it **inside** the
repository — the working directory defines what it can see. The same tool is
also a VS Code / JetBrains extension, a desktop app, and
[claude.ai/code](https://claude.ai/code); the mental model is identical.

---

## Setup — the free routes

**Gemini CLI**, open source, free with a Google account:

```bash
npm install -g @google/gemini-cli
cd my_project
gemini
```

**Aider**, open source and model-agnostic — it commits every change it makes, so
`git diff` and `git revert` are your review interface:

```bash
uv tool install aider-chat
cd my_project
aider
```

**Aider + Ollama**, fully local: no account, no network, nothing leaves the
machine.

```bash
# install ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:7b
aider --model ollama/qwen2.5-coder:7b
```

Be straight about the gap: a 7B local model is materially weaker at multi-file
work and slow without a GPU. The loop is identical; the quality of each step is
not. If it is your only option, keep the tasks small and read everything.

---

## First contact

In your `textstats` repository from Lab 1 — `git status` clean, and:

```bash
git switch -c agent-sandbox
```

so nothing here lands on `main`. Then, in order:

```text
> what does this project do?

> add a docstring to every public function in src/

> run the tests
```

Watch what happens between your message and the answer: it reads files, greps,
runs commands. That is the loop.

---

## Useful from minute one

| Command | Effect |
|---|---|
| `/init` | Generate a `CLAUDE.md` describing the project |
| `/clear` | Wipe the conversation, keep the session |
| `/model` | Switch model (Opus 5 / Sonnet 5 / Haiku 4.5) |
| `#` prefix | Save the line to memory |
| `!` prefix | Run a shell command directly in the session |
| `Esc` | Interrupt — use it early and often |
| `/help` | Everything else |

`Esc` is the one to internalise today. Stopping a wrong direction after ten
seconds costs nothing; letting it finish costs a review.

---

## The core loop

![The agent loop](assets/s1-git-and-packaging/assistant-landscape/agent-loop.png)

Every agent, whatever the brand, runs this cycle. **Your leverage is highest at
PLAN and VERIFY** — and those are the two steps students skip.

---

## Explore, and forbid edits while you do

```text
> Read src/textstats/ and tests/. Summarise the public API and
> tell me what is NOT covered by tests. Do not change anything yet.
```

"Do not change anything yet" is doing real work in that prompt. Without it you
get an answer *and* a diff you did not ask for.

---

## Plan — read a paragraph, not 300 lines

```text
> I want a Connect-Four player: win if I can, block if the opponent can,
> otherwise play the centre-most legal column. Propose an approach: module
> layout, the win-detection helper, the edge cases, and what tests you would
> write. Do not write code yet.
```

A wrong plan is cheap and obvious. Wrong code is expensive and plausible.
Reading a plan catches the wrong file, a misunderstood requirement, an approach
that ignores an existing helper — all three invisible in a diff until you have
already spent the attention.

Claude Code has a dedicated **plan mode** (`Shift+Tab` cycles into it) where the
agent researches and proposes but cannot edit. Use it for anything you cannot
describe in one sentence.

---

## Act — keep the unit small, watch the commands

One function, one bug, one refactor. If the response is touching six files you
did not expect, interrupt.

The agent will run things. Read what it is about to run.

---

## Verify — this is what separates a tool from a toy

```text
> Run the tests and fix anything that fails.
```

Now the loop closes on its own: it edits, runs `pytest`, reads the traceback,
edits again. You are supervising a search, not typing.

**No tests means no loop.** With nothing to check against, the agent stops when
the code *looks* finished. That is the strongest practical argument for the test
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

Four ingredients: **where** (the file, by name), **what** (the observable
behaviour), **how you will know** (the tests, the command that must pass), and
**what must not change** (signatures, dependencies, layout). The fourth is the
one people forget, and it is why an agent "helpfully" upgrades your torch.

---

## Context is the whole game

The agent knows exactly what is in its context window: the system prompt, your
project instructions, the conversation, every file it has read this session, and
the output of every command it has run. It does **not** know your repository as a
whole, your intentions, or last week's session.

| | Symptom | Fix |
|---|---|---|
| **Starvation** | invents an API you already have; wrong conventions | point at the file |
| **Pollution** | drifts, repeats an abandoned approach, gets slower | `/clear` |

```text
> Look at src/textstats/metrics.py and tests/test_metrics.py.
> Follow the same structure for the new connect4 module.
```

Naming an existing file as the pattern to follow is the single highest-return
habit in this lesson. It replaces a paragraph of style description.

---

## CLAUDE.md

A file at your repository root, read automatically at the start of every
session. `/init` writes a first draft by inspecting the repo; then you **edit
it** — the draft is a starting point, not a deliverable.

```markdown
# CLAUDE.md

## Commands
- Install: `uv sync --all-extras`
- Test: `uv run --all-extras pytest`
- Lint: `uv run --all-extras ruff check src/ tests/`

## Conventions
- Python 3.11+, full type hints on public functions.
- `src/` layout. Tests mirror the source tree: `src/x/y.py` -> `tests/test_y.py`.

## Fail fast
No default values for required config. No silent `except`.

## Do not
- Do not add dependencies without asking.
- Do not edit `uv.lock` by hand.
```

The test for a line in it: *would I have to say this again next time?* If not,
delete it. Nothing that the code already says, no file-tree description, no
essays — every token here is in the context of every session forever. The "do
not" rules are what stop the expensive surprises, and they cost one line each.

Aider's equivalent is `CONVENTIONS.md`, loaded explicitly with
`aider --read CONVENTIONS.md`.

---

## Guardrails — permissions

Claude Code asks before actions with consequences: writing files, running
commands, hitting the network.

| Mode | Behaviour |
|---|---|
| default | prompts before edits and commands |
| accept edits | file edits auto-approved, commands still prompt |
| plan | read-only — cannot edit or run anything |
| bypass | no prompts at all |

Approving `uv run pytest` forty times a session teaches you to approve without
reading, and *that reflex* is the actual danger — not any single command.
Allowlist the safe, frequent ones in `.claude/settings.json` (or `/permissions`
from inside the session) so the prompts you do get are rare enough to be worth
reading:

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run --all-extras pytest:*)",
      "Bash(git status)",
      "Bash(git diff:*)"
    ]
  }
}
```

Use plan mode for exploration. Use bypass mode for nothing you cannot throw
away.

---

## Guardrails — git is the real safety net

Permissions limit what happens. Git is what lets you undo it.

```bash
git status                                # must be CLEAN before you start
git switch -c feature/connect4            # never work on main
claude
> ...work...
git diff                                  # 100% agent output, unambiguous
git restore .                             # discard everything, instantly
```

A clean tree before you delegate is what makes `git diff` mean "what the agent
did". Working on a branch turns "the agent broke my project" into
`git switch main`.

<!-- notes: Every student who loses work this term will have been on main with
uncommitted changes. Say it now. -->

---

## Guardrails — secrets

The agent reads the files you point it at, and sometimes files it finds. Keep
credentials in `.env`, gitignored, never in tracked source, and assume anything
in the repository may end up in a prompt.

If a key does get committed, the fix is **rotate**, not delete. The history is
public the moment it is pushed.

---

## Reviewing agent-written code

Read the diff as if a stranger wrote it, because one did. The same four
questions as any pull request: does it do what I asked; what does it do on bad
input; are the tests real; can I explain every line.

Agent-written Python has a recognisable failure signature — grep for it:

```python
try:                              # 1. silent failure — violates fail-fast
    value = config["threshold"]
except KeyError:
    value = 0.5                   # a typo in the config is now a wrong answer

def train(data, lr=0.001, epochs=10, batch=32, seed=42, device="cpu"):
    ...                           # 2. six ways to run something you never meant

except Exception:                 # 3. over-broad catching
    pass

def test_train():                 # 4. a test that asserts nothing
    assert train(df) is not None
```

All four make the code *look* robust while removing your ability to find out it
is wrong.

---

## Scale of change

| Diff size | What actually happens |
|---|---|
| < 100 lines | read properly, real comments |
| 100–300 | skimmed, one or two comments |
| > 300 | approved on trust |

An agent can produce 500 lines in a minute. Your review capacity did not change.

---

## Where the failures come from

Almost every bad outcome traces to one of three causes, and this lesson is three
techniques against exactly those three:

1. **Underspecified task** — you asked for "improve the model", got 400 lines,
   and cannot tell what happened. → *Plan first.*
2. **No verification** — no tests, so neither of you knows if it works. →
   *Verify in the loop.*
3. **Too much at once** — one enormous change instead of five reviewable ones. →
   *Keep the unit small.*

---

## What none of them do

- **Decide what to build.** The agent optimises the objective you name. Naming
  the wrong one is a Session 3 problem and no tool will save you from it.
- **Guarantee correctness.** *Plausible* and *correct* are different properties.
  These tools produce the first reliably and the second often.
- **Understand your problem better than you do.** They have your repository.
  They do not have the conversation you had with your supervisor.

---

## The rule for this course

> You are the author of every line you merge, whoever typed it.

Not a moral position — it is how the grading works. Your project is assessed on
repository quality and you will be asked to explain your code. "The agent wrote
it" is not an answer to "why is the learning rate scheduled here?".

<!-- notes: Say this plainly and once. Repeating it turns into moralising and
they stop listening. -->

---

## Checklist before you merge agent work

- [ ] On a branch, not `main`
- [ ] Working tree was clean before the session
- [ ] `git diff` read in full
- [ ] Tests pass, and the new tests assert behaviour
- [ ] No bare `except`, no defaults on required config
- [ ] No new dependency you did not approve
- [ ] You can explain every line

Then **Lab 2**: an agent writes a Connect-Four player, and a leaderboard of
other people's agents decides whether you were right to merge it.
