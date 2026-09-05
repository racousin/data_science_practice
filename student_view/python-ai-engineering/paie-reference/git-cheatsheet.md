# Git Cheatsheet

Reference for Session 1. Not lectured — keep it open while you work.

---

## Setup

| Command | Description |
|---|---|
| `git init` | Initialise a new repository |
| `git clone <url>` | Clone a remote repository |
| `git config --global user.name "Name"` | Set your username globally |
| `git config --global user.email "email"` | Set your email globally |
| `git config --list --show-origin` | Show all settings and where they come from |

---

## Everyday cycle

| Command | Description |
|---|---|
| `git status` | Working directory status |
| `git add <file>` | Stage a file |
| `git add .` | Stage everything below the current directory |
| `git add -p` | Stage interactively, hunk by hunk |
| `git commit -m "message"` | Commit staged changes |
| `git commit -am "message"` | Stage tracked changes and commit |
| `git commit --amend` | Rewrite the last commit |
| `git diff` | Unstaged changes |
| `git diff --staged` | Staged changes |
| `git diff HEAD` | Everything since the last commit |

---

## Branching

| Command | Description |
|---|---|
| `git branch` | List local branches |
| `git switch <branch>` | Switch branch |
| `git switch -c <name>` | Create and switch |
| `git checkout -b <name>` | Same, older syntax |
| `git merge <branch>` | Merge a branch into the current one |
| `git merge --abort` | Abandon an in-progress merge |
| `git rebase main` | Replay your commits on top of `main` |
| `git branch -d <branch>` | Delete a merged branch |
| `git branch -D <branch>` | Force-delete |

---

## Remotes

| Command | Description |
|---|---|
| `git remote -v` | List remotes |
| `git remote add origin <url>` | Add a remote |
| `git fetch` | Download refs, change nothing locally |
| `git pull` | Fetch and merge |
| `git push origin <branch>` | Push a branch |
| `git push -u origin <branch>` | Push and set upstream |
| `git push --force-with-lease` | Force-push, refusing to clobber others' work |

---

## History

| Command | Description |
|---|---|
| `git log` | Commit history |
| `git log --oneline` | One line per commit |
| `git log --graph --decorate` | With branch topology |
| `git log --stat` | With per-file change counts |
| `git log -p <file>` | Full diffs for one file |
| `git show <commit>` | One commit in detail |
| `git blame <file>` | Who last changed each line |
| `git reflog` | Every position HEAD has held — your undo of last resort |

---

## Undoing

| Command | Description |
|---|---|
| `git restore <file>` | Discard working-directory changes |
| `git restore --staged <file>` | Unstage, keep the edit |
| `git reset --soft HEAD~1` | Undo last commit, keep changes staged |
| `git reset --mixed HEAD~1` | Undo last commit, keep changes unstaged |
| `git reset --hard HEAD~1` | Undo last commit, **discard changes** |
| `git revert <commit>` | New commit that undoes an old one — safe when shared |

**Rule:** `reset` rewrites history — local, unpushed commits only.
`revert` adds history — safe anywhere.

---

## Stashing

| Command | Description |
|---|---|
| `git stash` | Shelve current changes |
| `git stash -u` | Include untracked files |
| `git stash pop` | Restore and remove the latest stash |
| `git stash list` | List stashes |
| `git stash apply` | Restore without removing |
| `git stash drop` | Delete the latest stash |

---

## Tags

| Command | Description |
|---|---|
| `git tag` | List tags |
| `git tag <name>` | Lightweight tag |
| `git tag -a <name> -m "message"` | Annotated tag |
| `git push origin <tag>` | Push one tag |
| `git push origin --tags` | Push all tags |

Tag the commit you submitted. It is how you answer "which version produced this
result?" three months later.

---

## Inspecting a mess

```bash
git status                        # where am I
git log --oneline --graph -20     # what happened
git diff HEAD                     # what have I changed
git reflog                        # what have I done, including undone things
```

Four commands, in that order. They resolve most "my repository is broken"
situations without anyone losing work.
