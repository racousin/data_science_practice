# Guardrails & Review

An agent runs commands on your machine and pushes to your repository. The
controls below are what keep that from being reckless.

<!-- notes: 30 minutes. The review section is the graded part — spend more time
there than on permissions. -->

---

## Permissions

Every agent has a permission model. Claude Code asks before actions with
consequences: writing files, running commands, hitting the network.

| Mode | Behaviour |
|---|---|
| default | prompts before edits and commands |
| accept edits | file edits auto-approved, commands still prompt |
| plan | read-only — cannot edit or run anything |
| bypass | no prompts at all |

Use plan mode for exploration. Use bypass mode for nothing you cannot throw
away.

---

## Allowlisting

Approving `uv run pytest` forty times a session teaches you to approve without
reading. That is the actual danger — not any single command, but the reflex.

Allowlist the safe, frequent ones in `.claude/settings.json` —
`/permissions` edits the same list from inside the session:

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run --all-extras pytest:*)",
      "Bash(uv run --all-extras ruff check:*)",
      "Bash(git status)",
      "Bash(git diff:*)"
    ]
  }
}
```

Now the prompts you *do* get are rare enough to be worth reading.

---

## Git is the real safety net

Permissions limit what happens. Git is what lets you undo it.

```bash
git switch -c feature/agent-readability   # never work on main
git diff                                  # before every commit
git restore .                             # discard everything, instantly
```

Working on a branch turns "the agent broke my project" into `git switch main`.

<!-- notes: Every student who loses work this term will have been on main with
uncommitted changes. Say it now. -->

---

## Commit before you delegate

A clean working tree before you start means `git diff` shows exactly what the
agent did, with nothing of yours mixed in.

```bash
git status          # must be clean
claude
> ...work...
git diff            # 100% agent output, unambiguous
```

---

## Never let it commit unreviewed

Some tools commit automatically (Aider does, by default). That is fine — the
commit is a checkpoint, not an endorsement. What must not happen is a push
without a read.

```bash
git log --oneline -5     # what did it do
git diff main...HEAD     # the whole change, one screen at a time
```

---

## Secrets

The agent reads the files you point it at, and sometimes files it finds. Two
consequences:

- Keep credentials in `.env`, gitignored, never in tracked source.
- Assume anything in the repository may end up in a prompt.

If a key does get committed, the fix is **rotate**, not delete. The history is
public the moment it is pushed.

---

## Reviewing agent-written code

Read the diff as if a stranger wrote it, because one did. Four questions, in
this order:

1. **Does it do what I asked?** Not "is it good code" — is it the task.
2. **What does it do on bad input?** Silent fallback, or a clean crash?
3. **Are the tests real?** Do they assert behaviour, or just that nothing threw?
4. **Can I explain every line?** If not, it does not merge.

---

## The specific smells

Agent-written Python has a recognisable failure signature. Grep for it:

```python
# 1. Silent failure — violates fail-fast
try:
    value = config["threshold"]
except KeyError:
    value = 0.5          # now a typo in the config is a silent wrong answer

# 2. Defensive defaults on required inputs
def train(data, lr=0.001, epochs=10, batch=32, seed=42, device="cpu"):
    ...                  # six ways to run something you never intended

# 3. Over-broad catching
except Exception:
    pass                 # the bug is now invisible

# 4. Tests that assert nothing meaningful
def test_train():
    model = train(df)
    assert model is not None
```

All four make the code *look* robust while removing your ability to find out it
is wrong.

---

## The fail-fast correction

```python
# Instead of the silent default:
threshold = config["threshold"]        # KeyError, immediately, with the key name

# Instead of the bare except:
try:
    data = load(path)
except FileNotFoundError:
    raise ValueError(f"dataset missing: {path}") from None
```

Catch only what you can actually handle. Let everything else crash where it
happened, not three functions later with a nonsense value.

---

## Scale of change

| Diff size | What actually happens |
|---|---|
| < 100 lines | read properly, real comments |
| 100–300 | skimmed, one or two comments |
| > 300 | approved on trust |

An agent can produce 500 lines in a minute. Your review capacity did not change.
Ask for small changes not because the agent cannot do more, but because you
cannot check more.

---

## The tests are the contract

The strongest guardrail is not a permission dialog. It is:

```bash
uv run --all-extras pytest
```

Tests you wrote, or read and agreed with, encode what correct means. They are
checked mechanically, every iteration, without your attention. Everything else
in this lesson is a supplement to that.

---

## Checklist before merging agent work

- [ ] On a branch, not `main`
- [ ] Working tree was clean before the session
- [ ] `git diff` read in full
- [ ] Tests pass, and the new tests assert behaviour
- [ ] No bare `except`, no defaults on required config
- [ ] No new dependency you did not approve
- [ ] You can explain every line
