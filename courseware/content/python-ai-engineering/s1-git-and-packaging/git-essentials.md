# Git Essentials

The eight commands that cover 95% of daily use, and the four that get you out of
trouble.

<!-- notes: 40 minutes. Students type along. Install and authentication were
done in *Accounts & Toolchain* — do not repeat them here. Do not lecture past
`git log`; the undo section is where they actually need you. -->

---

## Before you start

`git` is installed and configured — that was *Accounts & Toolchain*. Confirm:

```bash
git --version
git config --list --show-origin | grep user
```

`--show-origin` is the flag that saves you later, when a repository-local
config overrides your global one and you cannot work out why your commits carry
the wrong name.

---

## Creating a repository

```bash
mkdir my_project
cd my_project
git init
```

`git init` creates the `.git/` directory. That directory *is* the repository —
delete it and you have an ordinary folder again; copy it and you have copied
the entire history.

---

## The status → add → commit cycle

```bash
echo "Initial content" > example.txt
git status
```

```text
On branch main
No commits yet
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	example.txt
```

Untracked means Git can see the file but is not watching it.

---

## Staging

```bash
git add example.txt      # one file
git add src/             # a directory
git add .                # everything below the current directory
git add -p               # interactively, hunk by hunk
```

`git add -p` is worth learning early. It walks you through each change and asks
whether it belongs in this commit. It is how you keep commits honest when you
have done three things at once.

---

## Committing

```bash
git commit -m "Add example.txt with initial content"
```

A commit message has one job: explain *why*, to someone reading the log in six
months. The diff already says *what*.

| Poor | Better |
|---|---|
| `update` | `Fix off-by-one in window indexing` |
| `fixed stuff` | `Handle empty dataframe in load_csv` |
| `wip` | `Add failing test for duplicate IDs` |

---

## Reading history

```bash
git log --oneline --graph --decorate -20
```

```text
* 3f9a1c2 (HEAD -> main) Add example.txt with initial content
```

Other views you will want:

```bash
git log --stat            # which files changed, how much
git log -p src/model.py   # full diff, restricted to one file
git log --author="Marie"  # one person's commits
git blame src/model.py    # who last touched each line
```

---

## Seeing what changed

The three diffs, and the difference between them:


```bash
git diff             # working directory vs staging  (not yet added)
git diff --staged    # staging vs last commit        (added, not yet committed)
git diff HEAD        # working directory vs last commit (everything)
```


---

## The three diffs

![The three diffs](assets/s1-git-and-packaging/git-essentials/three-diffs.png)

<!-- notes: This trips up nearly everyone. Point at each arrow on the diagram as
you name the command. -->

---

## Undoing — the four cases

The command you need depends on how far the change has travelled.

![Which undo](assets/s1-git-and-packaging/git-essentials/undo-map.png)

---

## Undoing — the four cases, as commands

| Situation | Command |
|---|---|
| Edited a file, want it back | `git restore <file>` |
| Staged something by mistake | `git restore --staged <file>` |
| Last commit message is wrong | `git commit --amend` |
| Want to undo a pushed commit | `git revert <sha>` |

`git revert` creates a *new* commit that undoes an old one. It is the only safe
option once the commit is shared — it does not rewrite anything.

---

## The dangerous one

```bash
git reset --hard <sha>
```

Discards commits *and* your uncommitted work, permanently. Use it only on
commits you have never pushed, and only when you mean it.

If you have just destroyed something and it was ever committed, this usually
saves you:

```bash
git reflog
```

`reflog` records every position `HEAD` has occupied, including ones no branch
points to any more. Find the SHA, `git checkout` it.

---

## .gitignore

Never commit: virtual environments, data, model weights, credentials, editor
noise. Write the rules once, at the top of the project:

```text
__pycache__/
*.py[cod]
.venv/
.env
data/
*.ckpt
.DS_Store
.ipynb_checkpoints/
```

<!-- notes: Emphasise credentials. A pushed API key is compromised even if you
delete it in the next commit — the history is public. Rotate, do not hide. -->

---

## If a file is already tracked

Adding it to `.gitignore` does nothing — ignore rules only apply to untracked
files. You must remove it from the index:

```bash
git rm --cached secrets.env
git commit -m "Stop tracking secrets.env"
```

The file stays on your disk; Git stops watching it. Note that it remains in the
history, which is why a leaked credential must be rotated, not just removed.

---

## The eight commands

Everything so far, on one card:

```bash
git init                 # start a repository
git status               # what is going on right now
git add <path>           # stage changes
git commit -m "..."      # snapshot the staged changes
git log --oneline        # read history
git diff                 # inspect changes
git restore <file>       # undo working-directory changes
git revert <sha>         # undo a shared commit, safely
```

The full cheatsheet, including everything in the next lesson, is in
[Reference → Git Cheatsheet](/courses/python-ai-engineering/s1-git-and-packaging/course/git-cheatsheet).
