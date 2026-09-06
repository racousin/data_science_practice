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

Assessment for this session lives on the students' GitHub accounts, with one
exception: **Lab 2 requires a submission to competition 65** (PettingZoo ·
Connect-Four). That is deliberate — it is the only thing in the session a
student cannot mark themselves, because an ELO board scores their agent against
other people's rather than against an answer key. *Being on the board* is graded;
rank is not.

The two competitions this module used to carry — 179 (`PAIE S1 — textstats`) and
180 (`PAIE S2 — Flesch reading-ease`) — were retired on 2026-09-06 with the
lessons that used them.

Seven things, and a student can check all seven themselves:

| | Evidence |
|---|---|
| a GitHub repository | the URL |
| an installable package | `import textstats` from outside the project |
| a test suite | `uv run pytest` — 8+ green |
| CI | a green run in the Actions tab, badge in the README |
| a reviewed pull request | merged, with a comment that is not "LGTM" |
| Colab | a notebook that `pip install`s from the tag and runs |
| an agent on a leaderboard | a rated entry on competition 65 |

---

## 2. What is written, and what fits

The module holds **560 published minutes** against a **180-minute** slot. That
is deliberate: the written lessons are the reference students keep for the year,
and the run-sheet below is the subset you lecture.

| | Lessons | Minutes |
|---|---|---|
| Core — the 3h path | 10 | 375 written, ~145 lectured |
| Extension — same session, not lectured | 1 | 30 |
| Reference — never lectured | 1 | 5 |
| Labs | 3 | 150 |

It was 725 minutes across 25 lessons until the 2026-09-06 consolidation folded
nine lessons into three (§9). Nothing was cut except duplication: the same
material, in three fewer places to lose it.

You will show perhaps 120 slides of the deck. Nothing is lost by skipping one —
the web lesson is where the student reads it afterwards.

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
| 40–62 | Why version control, then status / add / commit / log / diff / undo | `git-essentials` | L + **D** |
| 62–90 | Branching, a real conflict, then the pull request | `branching-and-collaboration` | **D** |
| **90–100** | **break** | | |
| 100–110 | Environments: venv, uv, lock | `python-environments` | L |
| 110–125 | Packaging and tests | `packaging-and-tests` | L + D |
| 125–137 | CI: one workflow file, watch it go green then red | `github-actions` | **D** |
| 137–147 | Notebooks and Colab: the kernel, the ephemeral VM | `notebooks-and-colab` | **D** |
| 147–180 | **Lab 1**, in the room, you circulating | `lab-1` | **H** |

Lab 1 needs 75 minutes and gets 33. That is intentional: students get to Part E
(the pull request and its review) in the room, where you can unblock them, and
finish Parts F–G at home.

**Form the groups before the lab starts, not at 147.** Groups of 3 map one
`textstats` function to each student; a group of 2 gives someone two. Write the
groups down — leaving it to the room produces four students with nobody.

**Lab 2 and the agentic-coding lesson do not fit in three hours.** See §7 for
the three ways of dealing with that; decide before the session, not during it.

---

## 5. The five live demos

The demos are the session. Slides are the notes students read afterwards.

### D1 — Git, from nothing (40–62)

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

### D2 — A conflict, resolved (62–78)

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

### D3 — A pull request (78–90)

On a prepared throwaway repository with a partner in the room:

push a branch → open the PR → have a student leave a `suggestion` comment →
apply it with one click → merge → delete the branch.

Ninety seconds of GitHub UI beats ten minutes of description. D2 and D3 are now
one lesson, so run them back to back without switching slide decks: the conflict
you just resolved is the change you open the pull request for.

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

**One lesson, 60 minutes** (`assistant-landscape`, *Coding Agents*), plus Lab 2.
It was six lessons and ~160 minutes until 2026-09-06; even at 60 it does not fit
inside the 180 alongside everything above.

**Option A — homework, recommended.** Lecture nothing today. Set *Coding Agents*
as reading plus Lab 2 for the following week, and open Session 2 with a
15-minute recap and a live agent demo. Costs 15 minutes of Session 2; keeps
Session 1 coherent.

**Option B — a 25-minute demo, no theory.** Cut the shell block to 10 minutes
and Lab 1 to 20 in the room. Show one agent loop live — plan, test-first,
implement, review the diff, reject something — and point at the lesson. The room
sees the shape; nobody practises it. The consolidation makes this the easiest it
has been: one lesson, one slide deck, one URL to send them to.

**Option C — a fifth session.** The honest answer if agentic coding matters as
much as the rest. This module is currently three sessions of content in four
slots wearing four names; adding a slot for it would also fix the numbering that
`COURSE_STATE.md` §1b flags.

Whichever you pick, say it out loud at the start of the session. Students who
expect Claude Code and get git are disappointed by a scheduling decision, not by
the content.

**One thing to check before the session, whichever option you pick:** Lab 2
submits to competition **65**, which this repository does not own. Confirm it is
still public and started (`client.competition(65)` returns 200 to a student
token) — a lab whose leaderboard 404s is worse than no leaderboard.

---

## 8. The two labs

| | Lab | Shape | When | Deliverable |
|---|---|---|---|---|
| 1 | Ship a Package to GitHub | **groups of 2-3** | in the room, finish at home | one repository URL per group |
| 2 (slug `lab-3`) | Ship an Agent to Connect Four | groups | homework | a merged PR + `RETRO.md` + a rated agent on comp 65 |

Lab 1 absorbed the old *Pull Request & Review* lab on 2026-09-06 (§9c). It is
the same package, built by a group instead of one student: three functions,
one owner each, one branch and one pull request each, and nobody merges their
own. Review is therefore inside Lab 1 rather than a lab of its own, and it
still cannot be faked alone — a group of one has nobody to merge its work.

Keep the same groups for Lab 2; its review and merge are homework and a
reshuffle strands people mid-week.

Each lab ends with a **"Did you validate this lab?"** checklist whose every row
is objectively verifiable by the student. Marking is reading those rows against
the repository.

The fast grading pass, per student, in under two minutes:

```bash
git clone <url> /tmp/g && cd /tmp/g
uv sync && uv run pytest -q          # it runs from a fresh clone
git log --oneline --graph | head -20 # history quality
git log --format='%an' | sort -u     # every member of the group is in it
git ls-files | grep -E '\.venv|__pycache__'   # must be empty
gh run list --limit 3                # CI actually ran
```

---

## 9. What changed

### 9c. The two labs merged (2026-09-06, third pass)

`lab-2` (*Pull Request & Review*, server lesson 136) was folded into `lab-1`;
14 lessons → 13, 520 published minutes → 490.

Lab 1 was a solo lab and Lab 2 the one that needed a partner. That split had
stopped working: the website had already cut the entire review half of
`branching-and-collaboration`, so Lab 2 asked for a skill the session no longer
taught, and it was homework nobody could start until they had found a partner.

Lab 1 is now **one group lab**, and its body is the old Lab 1 — the same
`textstats` package, the same specification, the same CI, tag and Colab steps.
What was added is the split: **groups of 2 or 3**, one or more of the three
functions owned by each student, the skeleton merged before anyone branches,
and a pull request per function that **somebody else in the group reviews and
merges**. That is the part of Lab 2 worth keeping. Branch protection, the
deliberate merge conflict and the review rubric went with the rest of it.

**Lab 3 is now Lab 2.** The slug is immutable server-side, so `lab-3` names the
second lab; only the title and the body's H1 changed.

`delete_lesson(136)` must run server-side **before** the next publish, or
`reorder_lessons` rejects the manifest.

### 9b. The consolidation (2026-09-06, second pass)

Nine lessons became three. 25 lessons → 16; 725 published minutes → 560.

- `why-version-control` → **`git-essentials`.** The motivation and the
  three-trees model now open the lesson whose commands they explain.
- `pull-requests-and-review` + `Reference — GitHub Desktop` →
  **`branching-and-collaboration`**, retitled *Branching, Pull Requests &
  Review*. Branching and the pull request were one workflow taught as two
  lessons; the GUI page was a reference nobody was sent to and is now the last
  two slides of the lesson it belongs to.
- `what-actually-changed` + `setup` + `the-core-loop` + `context-engineering` +
  `guardrails-and-review` → **`assistant-landscape`**, retitled *Coding Agents*.
  It now carries a **dated price table** and a **dated benchmark table**, which
  the six-lesson version deliberately refused to. That refusal was wrong:
  students were choosing a tool anyway, and doing it from marketing copy. The
  tables are stamped with the date they were checked and the lesson says in
  three ways why not to trust the benchmark far.
- **Lab 3 replaced.** Was *Agent-Driven Feature* (build a Flesch score, optional
  submission to comp 180). Is now *Ship an Agent to Connect Four*: an agent-built
  Connect-Four player, **required** submission to comp **65**. Its baseline
  ladder is measured over 400 self-play games and the interesting row is that
  *win-now-without-blocking* (+0.485) scores **worse than playing the centre
  column every time** (+0.780) — "plausible, not correct", as a number.
- **Competitions 179 and 180 retired**, packages deleted from
  `competitions/`. `detach_competition(14, 179)` / `(14, 180)` and eight
  `delete_lesson` calls have to run server-side **before** the publish.

### 9a. The rebuild (2026-09-06, first pass)

Since the module was previously *Git & Python Packaging*:

- **New:** `session-map`, `accounts-and-tools`, `the-shell`,
  `pull-requests-and-review`, `notebooks-and-colab`, `assistant-landscape`.
- **Promoted from reference to taught:** `github-actions` (now a full CI/CD
  lesson) and `ide-syntax-linting` (now linting, formatting, types and
  pre-commit). Both kept their slugs; both were rewritten.
- **Labs renumbered:** Lab 1 rewritten around the GitHub + CI + Colab
  deliverable; Lab 2 is now the pull-request-and-review lab; the old Lab 2
  (agent-driven feature) became Lab 3.
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
4. ~~Whether to keep 179/180 attached at all.~~ **Settled 2026-09-06:**
   detached and retired, and the module now carries one competition (65) that a
   required lab actually uses. The open question underneath it is new: comp 65
   is **not ours**. Its overview page is the PettingZoo blurb and states no
   baseline, and a challenge we do not own can be stopped or edited by its
   creator. Either adopt it (write the overview, state the measured ladder from
   Lab 2 Part E) or build a Session 1 package that replaces it.
5. **Group size.** Lab 1 is written for 3 — one `textstats` function each — and
   works at 2. At 4 somebody owns nothing, so a cohort that does not divide by
   3 should be padded with 2s, not with 4s.
