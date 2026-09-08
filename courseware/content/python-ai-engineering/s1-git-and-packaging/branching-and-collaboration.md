# Branching, Pull Requests & Review

Branches are how two people work on one codebase without waiting for each other.
Remotes are how the work gets back together. The pull request is where a second
person reads it before it lands — which is the part you are actually graded on.

<!-- notes: 60 minutes, and it is the longest lesson in the session. Two live
demos carry it: create a real conflict and resolve it (15 min), then do a real
pull request on the projector — branch, push, open, have a student review it,
address the comment, merge (15 min). Fifteen minutes of doing beats thirty of
describing. Cut the GitHub Desktop slide first if you are behind. -->

---

## What a branch is

A branch is a *movable pointer to a commit*. That is the whole implementation.

Creating a branch writes 41 bytes to disk. It is not a copy of your files, it
does not duplicate history, and it costs nothing. This is why Git workflows use
branches liberally where older tools used them sparingly.


---

## Two branches, one history

![A branch is a pointer](assets/s1-git-and-packaging/branching-and-collaboration/branch-pointers.png)

`main` points at `C`. `feature/scaling` points at `E`. Both share `A` and `B`.

---

## Creating and switching

```bash
git switch -c feature/scaling    # create and switch
git switch main                  # switch back
git branch                       # list local branches
git branch -d feature/scaling    # delete (refuses if unmerged)
```

`git switch` and `git restore` were introduced in Git 2.23 to split the two jobs
`git checkout` used to do. You will still see `git checkout -b` everywhere; it
is the same thing.

---

## Naming

Pick a convention and hold it. A common one:

```text
feature/<short-description>
fix/<short-description>
experiment/<short-description>
```

The branch name is read by your teammates in the pull-request list. `feature/`
plus three words beats `marie-branch-2` every time.

---

## Merging

Bring `feature/scaling` back into `main`:

```bash
git switch main
git merge feature/scaling
```

Two things can happen.

---

### Fast-forward

If `main` has not moved since you branched, Git just slides the pointer forward.
No merge commit, no new history.

![Fast-forward merge](assets/s1-git-and-packaging/branching-and-collaboration/Git_Fast-forward_Merge.png)

---

### Three-way merge

If `main` *has* moved, Git builds a new commit with two parents.

![Three-way merge](assets/s1-git-and-packaging/branching-and-collaboration/Git_Three-way_Merge.png)

`M` is the merge commit. Its content is Git's reconciliation of the two lines of
work, computed against their common ancestor `B`.

---

## Conflicts

A conflict happens when both branches changed the **same lines** of the **same
file**. Git will not guess; it stops and asks.

```text
Auto-merging src/model.py
CONFLICT (content): Merge conflict in src/model.py
Automatic merge failed; fix conflicts and then commit the result.
```

---

## Resolving a conflict

Git writes both versions into the file, marked:

```python
<<<<<<< HEAD
learning_rate = 0.01
=======
learning_rate = 0.001
>>>>>>> feature/scaling
```

- Above `=======` is what `main` (where you are) says.
- Below is what the incoming branch says.

Edit the file into the state you actually want, delete all three markers, then:

```bash
git add src/model.py
git commit
```

<!-- notes: Stress that resolving is an editorial decision, not a mechanical
one. Sometimes the correct answer is neither side. -->

---

## Useful during a conflict

```bash
git status                  # which files are still conflicted
git diff                    # the conflicted hunks
git merge --abort           # give up, return to pre-merge state
git checkout --ours <file>  # take this branch's version wholesale
git checkout --theirs <file># take the incoming version wholesale
```

`--ours` / `--theirs` are for generated files (a lockfile, a notebook output) —
not for source code you should be reading.

---

## Remotes

A remote is a named URL pointing at another copy of the repository.

```bash
git remote -v                                   # list
git remote add origin git@github.com:me/proj.git
git push -u origin main                         # first push, sets upstream
git push                                        # subsequent pushes
```

`origin` is a convention, not a keyword. It is the name given to the remote you
cloned from.

---

## The remote workflow

![Remote workflow](assets/s1-git-and-packaging/branching-and-collaboration/remote.png)

---

## Getting other people's work

```bash
git fetch      # download refs, change nothing in your files
git pull       # fetch + merge into the current branch
```

`fetch` then `log` is the cautious version: you can see what arrived before it
touches your working directory.

```bash
git fetch
git log --oneline HEAD..origin/main   # what they have that you do not
```

---

## Cloning

```bash
git clone git@github.com:racousin/data_science_practice_2025.git
cd data_science_practice_2025
```

`clone` does `init` + `remote add origin` + `fetch` + `switch` to the default
branch, in one step. You get the entire history, not just the latest state.

---

## Keeping a branch current

Your feature branch will fall behind `main` while you work. Two ways to catch
up:

```bash
git merge main       # safe; adds a merge commit
git rebase main      # linear history; rewrites your commits
```


---

## Merge or rebase

![Merge versus rebase](assets/s1-git-and-packaging/branching-and-collaboration/merge-vs-rebase.png)

**Rule:** rebase only commits you have not pushed. Rebasing shared history
rewrites SHAs under your collaborators' feet, and their next `pull` becomes a
mess.

---

## What a pull request is

Everything above is git on your machine. Now the second person.

A pull request is a request to merge one branch into another, plus a place to
discuss it before it happens.

It is not a git feature. Git has `git merge`, and it does not ask anyone. The
pull request is GitHub's addition: a merge that waits for a review and a green
test run.

![The pull-request loop](assets/s1-git-and-packaging/branching-and-collaboration/pr-lifecycle.png)

---

## The loop, as commands

This is the loop you will use for every assignment and every project commit:

```bash
git switch -c feature/readability      # 1. branch off main
# ... work, commit in small steps ...
git push -u origin feature/readability # 2. publish the branch
```

Then on GitHub: *Compare & pull request*. Or, without leaving the terminal:

```bash
gh pr create --fill
gh pr view --web
```

Review happens. You push again to the **same branch**, and the pull request
updates itself — there is nothing to re-open.

```bash
git add -p && git commit -m "Handle the no-punctuation case"
git push
```



---

## The three merge buttons

| Button | Result | Use when |
|---|---|---|
| **Create a merge commit** | keeps every commit, adds a merge node | a real feature branch |
| **Squash and merge** | the whole PR becomes one commit on `main` | messy history, small change |
| **Rebase and merge** | replays commits onto `main`, no merge node | you want a linear history |

---



## What review is for

Not catching typos. A linter does that faster and without an opinion.

Review catches the three things a machine cannot:

- **The wrong problem solved correctly.**
- **A design that will be expensive in three weeks.**
- **Code that works and that nobody but the author can change.**

In your project grade, repository quality sits beside leaderboard performance.
This is the axis it measures.

---

## If you would rather click: GitHub Desktop

Not required, and not examinable — but two things are genuinely easier in a
graphical client than in the terminal:

- **Reviewing a diff before committing.** Side-by-side, with per-line staging —
  the equivalent of `git add -p`, only you can see what you are choosing.
- **Resolving conflicts.** A three-pane view beats reading `<<<<<<<` markers.

Download it from [desktop.github.com](https://desktop.github.com) — macOS and
Windows only; on Linux use `gitg`, GitKraken, or your IDE's Git panel. Signing
in also configures your credentials, so `git push` works from the terminal
afterwards.

---

## GitHub Desktop, mapped to the commands

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

**The recommendation: use it for review, use the terminal for everything else.**
The reason is not purism. The terminal is where your agent works, where CI runs,
where a server has no GUI, and where every error message you will search for was
written. A workflow you can only perform by clicking does not transfer.

---

## Recap

```bash
git switch -c feature/x     # branch
git merge <branch>          # integrate
git merge --abort           # back out of a bad merge
git fetch / git pull        # get remote work
git push -u origin <branch> # publish a branch
git rebase main             # linearise (unpushed work only)
gh pr create --fill         # open the pull request
```

Next: making the project inside that repository reproducible.
