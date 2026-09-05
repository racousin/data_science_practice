# GitHub Desktop

Reference only — not covered in class. A graphical client for the Git commands
in Session 1.

---

## What it is for

Two things it does genuinely better than the terminal:

- **Reviewing a diff before committing.** Side-by-side, with per-line staging.
- **Resolving conflicts.** A three-pane view beats reading `<<<<<<<` markers.

Everything else is the same operations with buttons.

---

## Install

Download from [desktop.github.com](https://desktop.github.com). macOS and
Windows only; on Linux use `gitg`, `GitKraken`, or your IDE's Git panel.

Sign in with GitHub — this also configures your credentials, so `git push` works
from the terminal afterwards.

---

## Mapping to commands

| GitHub Desktop | Terminal |
|---|---|
| Current Repository → Add | `git clone` / `git init` |
| Changes tab, tick a file | `git add <file>` |
| Commit to `main` | `git commit -m "..."` |
| Push origin | `git push` |
| Fetch origin | `git fetch` / `git pull` |
| Current Branch → New Branch | `git switch -c <name>` |
| Branch → Merge into current | `git merge <branch>` |
| History tab | `git log` |
| Right-click a commit → Revert | `git revert <sha>` |

---

## The recommendation

Use it for **review**, use the terminal for **everything else**.

The reason is not purism. The terminal is where your agent works, where CI runs,
where a server has no GUI, and where every error message you will search for was
written. A workflow you can only perform by clicking does not transfer.

---

## Partial staging

The one workflow worth learning here. In the Changes tab, click individual lines
or hunks to stage part of a file — the graphical equivalent of `git add -p`.

Use it when you fixed a bug and reformatted something on the way: two commits,
not one.
