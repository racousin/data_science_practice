# Accounts & Toolchain

Five things to install or sign up for, once. Do them now, together, in the room:
this is what stops Session 3 from being about Python installation.

<!-- notes: 20-30 minutes, everyone on keyboards, walk the room. Do NOT lecture
over this. The single most valuable thing you do today is make sure all four
verification commands print something on all thirty laptops. Put them on the
board and leave them there. -->

---

## The five

![The toolchain](assets/s1-git-and-packaging/accounts-and-tools/toolchain.png)

---

## What each one is for

| | What | Why it is on the list |
|---|---|---|
| **GitHub account** | free | every deliverable this year is a repository URL |
| **VS Code** | free | one window: editor, terminal, notebook, agent |
| **git** | free | the client that talks to GitHub |
| **uv** | free | Python versions, environments, packages, one tool |
| **Google account** | free | Colab, for the sessions that need a GPU |

Nothing here costs money, and nothing needs administrator rights on a lab
machine except `git` on Windows.

---

## 1 — GitHub

Sign up at [github.com](https://github.com). Three decisions, made once:

- **Username.** It appears on everything you hand in for the next two years, and
  on your CV after that. `marie-durand` is a username; `xX_dark_coder_Xx` is a
  decision you will regret in an interview.
- **Email.** Use the one you actually read.
- **Two-factor authentication.** GitHub requires it for accounts that push code.
  Set it up with an authenticator app rather than SMS.

<!-- notes: education.github.com gives students Copilot Pro and more with a
university email. Mention it; do not spend class time on the verification flow,
it can take days to come back. -->

---

## 2 — VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com). Install
exactly three extensions and stop:

| Extension | What it gives you |
|---|---|
| **Python** (Microsoft) | interpreter selection, debugging, test discovery |
| **Jupyter** (Microsoft) | notebooks inside the editor, same window |
| **Ruff** (Astral) | linting and formatting as you type |


---

## The one VS Code habit that matters

**Open the folder, not the file.**

`File → Open Folder…`, and pick the project root. VS Code then knows where your
`pyproject.toml`, your `.venv` and your `.git` are. Opening a lone `.py` file
gives you a text editor with syntax colouring and nothing else: no imports
resolved, no tests discovered, no git panel.

The integrated terminal (`` Ctrl+` ``) then opens *in that folder*, which is the
working directory every command in this course assumes.

---

## 3 — git

| Platform | How |
|---|---|
| macOS | `brew install git`, or accept the Xcode command-line-tools prompt |
| Debian / Ubuntu | `sudo apt install git` |
| Windows | [git-scm.com/download/win](https://git-scm.com/download/win) |

```bash
git --version
```

**Windows students:** the installer includes **Git Bash**. Use it — or WSL — as
your terminal for this course. Every shell command in these lessons is written
for a POSIX shell; PowerShell has different names for most of them, and mixing
the two is a reliable way to lose an afternoon.

Set your identity before your first commit:

```bash
git config --global user.name "Marie Durand"
git config --global user.email "marie@users.noreply.github.com"
git config --global init.defaultBranch main
```

---

## 4 — uv

One binary that replaces `pyenv`, `virtualenv`, `pip` and `pip-tools`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux / WSL
```

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

```bash
uv --version
uv python install 3.12
```

If `uv --version` says `command not found` right after a successful install, the
installer added a directory to your `PATH` that this shell has not read. Close
the terminal and open a new one. *The Shell* explains the mechanism.

---

## 5 — Google / Colab

Open [colab.research.google.com](https://colab.research.google.com), create a
notebook, and run one cell:

```python
import sys, torch
print(sys.version, torch.__version__)
```

`torch` is already installed there. That is the entire reason Sessions 3 and 4
use Colab, and *Notebooks & Colab* covers the consequences.

---

## Three machines, one repository

![Where code runs](assets/s1-git-and-packaging/accounts-and-tools/where-code-runs.png)

Keep this picture. By the end of the session your code runs in all three, and
the only thing they share is what you pushed to GitHub. Anything you did not
commit exists on exactly one of them.

---

## Verification — do not skip this

Paste these four lines. All four must print a version. A missing one is a
problem *now*, not in week six.

```bash
git --version
uv --version
uv run --python 3.12 python -c "print('python ok')"
code --version
```

<!-- notes: `code --version` fails on macOS until the student runs "Shell
Command: Install 'code' command in PATH" from the palette. It is not important —
do not let the room stall on it. -->

Then confirm you can see your own profile at `github.com/<your-username>`.

---

## Connecting git to GitHub

You will push code today, and GitHub stopped accepting passwords in 2021. Pick
one of these. SSH is the one that keeps working.

**SSH key** — set once:

```bash
ssh-keygen -t ed25519 -C "marie@users.noreply.github.com"
cat ~/.ssh/id_ed25519.pub
```

Paste the output into *GitHub → Settings → SSH and GPG keys → New SSH key*, then:

```bash
ssh -T git@github.com
```

You want `Hi <username>! You've successfully authenticated`.

**GitHub CLI** — easier, and useful again in the pull-request lesson:

```bash
gh auth login
```

It walks you through a browser login and configures git's credentials for you.

---

## When it does not work

The failures are always the same five:

| Symptom | Cause |
|---|---|
| `git: command not found` | not installed, or the terminal predates the install |
| `uv: command not found` | same — open a new terminal |
| `Permission denied (publickey)` | the SSH key is not on your GitHub account |
| `Support for password authentication was removed` | HTTPS with a password; use SSH or `gh auth login` |
| imports work in the terminal, not in VS Code | VS Code is pointed at a different interpreter |

The last one is fixed by *Python: Select Interpreter* in the command palette
(`Ctrl+Shift+P`), choosing the one inside your project's `.venv`.
