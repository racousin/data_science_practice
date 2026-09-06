# Session Plan — Teacher's Run-Sheet

> **Not teaching material.** Unpublished and out of the deck. This is the plan
> for running Session 1: what to lecture, what to demo, what to cut, and what to
> do when the room falls behind.

Session 1 is the engineering floor for both modules — the 12h *AI Engineering*
and the 30h *Machine Learning Practice* that follows it. Nothing after it works
if this does not land.

---

## 1. The contract

**One deliverable, one URL.** Every student leaves with a GitHub repository
containing a tested, installable Python package whose tests run themselves on
GitHub's machines, plus a Colab notebook that installs it from that repository.

Assessment for this session lives entirely on the students' GitHub accounts. The
two ML-Arena competitions attached to the module (179, 180) are **optional** parts
of Labs 1 and 3 — an outside opinion, not a requirement.

Six things, and a student can check all six themselves:

| | Evidence |
|---|---|
| a GitHub repository | the URL |
| an installable package | `import textstats` from outside the project |
| a test suite | `uv run pytest` — 8+ green |
| CI | a green run in the Actions tab, badge in the README |
| a reviewed pull request | merged, with a comment that is not "LGTM" |
| Colab | a notebook that `pip install`s from the tag and runs |

---

## 2. What is written, and what fits

The module holds **725 minutes** of written material against a **180-minute**
slot. That is deliberate: the written lessons are the reference students keep
for the year, and the run-sheet below is the subset you lecture.

| | Lessons | Minutes |
|---|---|---|
| Core — the 3h path | 15 | ~410 written, ~145 lectured |
| Extension — same session, not lectured | 6 | ~185 |
| Reference — never lectured | 3 | 45 |
| Labs | 3 | 150 |

The deck is 288 slides. You will show perhaps 120 of them. Nothing is lost by
skipping a slide — the web lesson is where the student reads it afterwards.

---

## 3. Before the session

Send this to the cohort **three days ahead**. It is worth twenty minutes of
class time.

> Before Session 1, please do the following. It takes 20 minutes and it is not
> optional — the session starts with everyone typing.
>
> 1. Create a GitHub account: <https://github.com>. Use a username you would put
>    on a CV. Enable two-factor authentication.
> 2. Install VS Code: <https://code.visualstudio.com>, plus the **Python**,
>    **Jupyter** and **Ruff** extensions.
> 3. Install git. Windows: <https://git-scm.com/download/win> — it includes
>    **Git Bash**, which you will use as your terminal all term.
> 4. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
>    (Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).
> 5. Open <https://colab.research.google.com> and run one cell.
>
> Then paste these four lines into a terminal. All four must print a version:
>
> ```
> git --version
> uv --version
> uv run --python 3.12 python -c "print('python ok')"
> code --version
> ```
>
> If any fails, come 15 minutes early.

Expect roughly a third of the room not to have done it. Block 0 exists for them.

---

## 4. The 3-hour run-sheet

`L` = lecture, `D` = live demo on the projector, `H` = hands on keyboards.

| Minutes | Block | Lesson | Mode |
|---|---|---|---|
| 0–8 | What we are building today | `session-map` | L |
| 8–25 | Accounts, install, verify | `accounts-and-tools` | **H** |
| 25–40 | The shell: paths, PATH, streams, errors | `the-shell` | L + H |
| 40–45 | Why version control | `why-version-control` | L |
| 45–65 | Git: status / add / commit / log / diff / undo | `git-essentials` | **D** |
| 65–80 | Branching, merging, a real conflict | `branching-and-collaboration` | **D** |
| 80–90 | Pull requests and review | `pull-requests-and-review` | **D** |
| **90–100** | **break** | | |
| 100–110 | Environments: venv, uv, lock | `python-environments` | L |
| 110–125 | Packaging and tests | `packaging-and-tests` | L + D |
| 125–137 | CI: one workflow file, watch it go green then red | `github-actions` | **D** |
| 137–147 | Notebooks and Colab: the kernel, the ephemeral VM | `notebooks-and-colab` | **D** |
| 147–180 | **Lab 1**, in the room, you circulating | `lab-1` | **H** |

Lab 1 needs 60 minutes and gets 33. That is intentional: students get to Part E
(CI) in the room, where you can unblock them, and finish Parts F–G at home.

**Labs 2 and 3, and the six agentic-coding lessons, do not fit in three hours.**
See §7 for the three ways of dealing with that; decide before the session, not
during it.

---

## 5. The five live demos

The demos are the session. Slides are the notes students read afterwards.

### D1 — Git, from nothing (45–65)

Empty directory, projector, no notes. Narrate every command.

```bash
mkdir demo && cd demo && git init
echo "hello" > a.txt
git status                      # untracked
git add a.txt && git status     # staged
git commit -m "Add a.txt"
echo "world" >> a.txt
git diff                        # unstaged
git add a.txt && git diff       # nothing! -- this is the moment
git diff --staged               # there it is
git restore --staged a.txt
git log --oneline --graph
```

The pause after `git add a.txt && git diff` printing nothing is the single most
valuable ten seconds of the session. Let it sit before explaining.

---

### D2 — A conflict, resolved (65–80)

```bash
git switch -c feature/x
echo "HELLO" > a.txt && git commit -am "Shout"
git switch main
echo "bonjour" > a.txt && git commit -am "Translate"
git merge feature/x             # CONFLICT
cat a.txt                       # show the markers
```

Then resolve it **badly** first — take one side wholesale — and ask the room
what was lost. Then resolve it properly.

### D3 — A pull request (80–90)

On a prepared throwaway repository with a partner in the room:

push a branch → open the PR → have a student leave a `suggestion` comment →
apply it with one click → merge → delete the branch.

Ninety seconds of GitHub UI beats ten minutes of description.

---

### D4 — CI going red (125–137)

Commit `.github/workflows/tests.yml`, push, switch to the Actions tab, wait for
green. Then break one assertion, push, and **wait for the red**. The red run is
the demo; the green one is just setup.

If the room is short on time, have the green run already in history and only do
the red.

### D5 — The notebook that lies (137–147)

```python
x = 1        # cell 1, run it
print(x)     # cell 2, run it -> 1
```

Now edit cell 1 to `x = 2` and **do not re-run it**. Re-run cell 2. It still
prints 1. Then *Restart & Run All* and it prints 2.

Ten seconds, and every student understands the kernel model.

---

## 6. Where the room stalls, and the fix

| Stall | Fix |
|---|---|
| `command not found` right after installing | open a new terminal — it is `PATH`, always |
| `Permission denied (publickey)` | the SSH key is not on the account; or use `gh auth login` |
| Windows student in PowerShell | move them to Git Bash before anything else |
| VS Code cannot import the package | *Python: Select Interpreter* → the project `.venv` |
| `uv sync` then `Failed to spawn: pytest` | pytest is in `[project.optional-dependencies]`; it belongs in `[dependency-groups]` |
| CI red, local green | uncommitted file — almost always `uv.lock` |
| Repository is 200 MB | `.venv/` was committed; `.gitignore` first, then `git rm -r --cached .venv` |

The last three are worth putting on the board pre-emptively at the start of
Lab 1.

---

## 7. The agentic-coding block — three options

Six lessons, ~160 minutes, plus Lab 3. It cannot go inside the 180.

**Option A — homework, recommended.** Lecture nothing today. Set
`assistant-landscape` → `guardrails-and-review` as reading plus Lab 3 for the
following week, and open Session 2 with a 15-minute recap and a live agent
demo. Costs 15 minutes of Session 2; keeps Session 1 coherent.

**Option B — a 25-minute demo, no theory.** Cut the shell block to 10 minutes
and Lab 1 to 20 in the room. Show one agent loop live — plan, test-first,
implement, review the diff, reject something — and point at the lessons. The
room sees the shape; nobody practises it.

**Option C — a fifth session.** The honest answer if agentic coding matters as
much as the rest. This module is currently three sessions of content in four
slots wearing four names; adding a slot for it would also fix the numbering that
`COURSE_STATE.md` §1b flags.

Whichever you pick, say it out loud at the start of the session. Students who
expect Claude Code and get git are disappointed by a scheduling decision, not by
the content.

---

## 8. The three labs

| | Lab | Shape | When | Deliverable |
|---|---|---|---|---|
| 1 | Ship a Package to GitHub | solo | in the room, finish at home | repository URL |
| 2 | Pull Request & Review | **pairs** | homework | a merged PR + a review given |
| 3 | Agent-Driven Feature | pairs | homework | a merged PR + `RETRO.md` |

Lab 2 must be paired and cannot be faked alone: it requires a review *given* on
someone else's repository and a review *received* on yours. Pair students at the
end of the session, in the room, and write the pairs down — leaving it to them
produces four students with no partner.

Each lab ends with a **"Did you validate this lab?"** checklist whose every row
is objectively verifiable by the student. Marking is reading those rows against
the repository.

The fast grading pass, per student, in under two minutes:

```bash
git clone <url> /tmp/g && cd /tmp/g
uv sync && uv run pytest -q          # 25% of Lab 1
git log --oneline --graph | head -20 # history quality
git ls-files | grep -E '\.venv|__pycache__'   # must be empty
gh run list --limit 3                # CI actually ran
```

---

## 9. What changed in this rebuild (2026-09-06)

For the record, since the module was previously *Git & Python Packaging*:

- **New:** `session-map`, `accounts-and-tools`, `the-shell`,
  `pull-requests-and-review`, `notebooks-and-colab`, `assistant-landscape`.
- **Promoted from reference to taught:** `github-actions` (now a full CI/CD
  lesson) and `ide-syntax-linting` (now linting, formatting, types and
  pre-commit). Both kept their slugs; both were rewritten.
- **Labs renumbered:** Lab 1 rewritten around the GitHub + CI + Colab
  deliverable; Lab 2 is now the pull-request-and-review lab; the old Lab 2
  (agent-driven feature) became Lab 3.
- **Competitions demoted to optional.** Session 1 is assessed from GitHub.
- **30 figures authored**, generated by
  `tools/figures/s1_git_and_packaging.py`. Session 1 had no taught PPTX to lift
  stills from, so every diagram is drawn by that script and traceable to it.

---

## 10. Open decisions

1. **The agentic block** — options A / B / C in §7. Needs a call before the
   session, and it changes the deck you present.
2. **Windows policy.** The lessons assume Git Bash or WSL and say so once, in
   `accounts-and-tools`. If the cohort is mostly Windows, decide which and put
   it in the pre-flight email.
3. **Public or private lab repositories.** The lessons recommend **public**:
   free CI minutes, unrestricted branch protection, and `pip install
   git+https://…` works in Colab with no token. If the school requires private,
   Lab 1 Part G needs a token step and CI minutes become finite.
4. **Whether to keep 179/180 attached at all.** They are optional today. If the
   evaluation for this session ends up being purely GitHub-based, detaching them
   removes two cards from the module page that no required lesson uses.
5. **Pair assignment for Labs 2 and 3.** Fixed pairs for both, or reshuffle
   between them. Reshuffling doubles the number of repositories each student has
   to read, which is the point of the exercise.
