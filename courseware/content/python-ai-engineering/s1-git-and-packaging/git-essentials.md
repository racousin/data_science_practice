# Git Essentials

Install, configure, and the eight commands that cover 95% of daily use.

<!-- notes: 45 minutes. Students type along. Do not lecture past `git log` —
the undo section is where they actually need you. -->

---

## Install

| Platform | Command |
|---|---|
| macOS | `brew install git` (or Xcode command line tools) |
| Debian / Ubuntu | `sudo apt install git` |
| Windows | [git-scm.com/download/win](https://git-scm.com/download/win) — includes Git Bash |

Verify:

```bash
git --version
```

Anything from 2.30 onwards is fine for this course.

---

## Configure once, per machine

Git stamps your name and email into every commit. Set them before your first
commit, or you will be rewriting history to fix attribution.

```bash
git config --global user.name "Marie Durand"
git config --global user.email "marie.durand@example.edu"
git config --global init.defaultBranch main
```

Check what is set:

```bash
git config --list --show-origin
```

<!-- notes: --show-origin is the one that saves them later, when a repo-local
config overrides the global one and they cannot work out why. -->

---

## Authenticating with GitHub

GitHub stopped accepting passwords over HTTPS in 2021. Two options:

**SSH key** (recommended — set once, works everywhere):

```bash
ssh-keygen -t ed25519 -C "marie.durand@example.edu"
cat ~/.ssh/id_ed25519.pub
```

Paste the public key into *GitHub → Settings → SSH and GPG keys*. Test it:

```bash
ssh -T git@github.com
```

**Personal access token** — a generated string used as your HTTPS password.
Fine, but expires and has to be stored somewhere.

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

<!-- notes: This trips up nearly everyone. Draw the three areas on the board
again and put each diff command as an arrow between two of them. -->

---

## Undoing — the four cases

The command you need depends on how far the change has travelled.

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
