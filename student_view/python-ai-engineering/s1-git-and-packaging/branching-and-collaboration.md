# Branching & Collaboration

Branches are how two people work on one codebase without waiting for each other.
Remotes are how the work gets back together.

<!-- notes: 50 minutes. The conflict resolution demo is the part they remember —
budget 15 minutes for it and actually create a conflict live. -->

---

## What a branch is

A branch is a *movable pointer to a commit*. That is the whole implementation.

Creating a branch writes 41 bytes to disk. It is not a copy of your files, it
does not duplicate history, and it costs nothing. This is why Git workflows use
branches liberally where older tools used them sparingly.

```text
        A---B---C   main
             \
              D---E   feature/scaling
```

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

```text
before:   A---B   main
               \
                D---E   feature

after:    A---B---D---E   main, feature
```

![Fast-forward merge](/api/academic_courses/assets/lessons/30/Git_Fast-forward_Merge.png)

---

### Three-way merge

If `main` *has* moved, Git builds a new commit with two parents.

```text
        A---B---C-------M   main
             \         /
              D---E---/     feature
```

![Three-way merge](/api/academic_courses/assets/lessons/30/Git_Three-way_Merge.png)

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

![Remote workflow](/api/academic_courses/assets/lessons/30/Git_Remote_Workflow.png)

---

## Getting other people's work

```bash
git fetch      # download refs, change nothing in your files
git pull       # fetch + merge into the current branch
```

![Fetch, merge, pull](/api/academic_courses/assets/lessons/30/Git_Fetch_Merge_Pull.png)

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

## The pull-request workflow

This is the loop you will use for every assignment and every project commit.

1. `git switch -c feature/thing` — branch off `main`
2. Work. Commit in small, readable steps.
3. `git push -u origin feature/thing`
4. Open a **pull request** on GitHub
5. A teammate reviews and comments
6. Push fixes to the same branch — the PR updates itself
7. Merge, then delete the branch

---

## Why the PR matters here

The pull request is where review happens, and review is half of what this module
is teaching. In your project grade, repository quality is a graded axis
alongside leaderboard performance.

A reviewable PR:

- changes one thing
- has a title a stranger can understand
- is small enough to read in ten minutes
- has a green test run

<!-- notes: Tell them the honest number: a 900-line PR gets rubber-stamped, a
90-line PR gets read. Reviewer attention is the scarce resource. -->

---

## Keeping a branch current

Your feature branch will fall behind `main` while you work. Two ways to catch
up:

```bash
git merge main       # safe; adds a merge commit
git rebase main      # linear history; rewrites your commits
```

**Rule:** rebase only commits you have not pushed. Rebasing shared history
rewrites SHAs under your collaborators' feet, and their next `pull` becomes a
mess.

---

## Recap

```bash
git switch -c feature/x     # branch
git merge <branch>          # integrate
git merge --abort           # back out of a bad merge
git fetch / git pull        # get remote work
git push -u origin <branch> # publish a branch
git rebase main             # linearise (unpushed work only)
```

Next: making the project inside that repository reproducible.
