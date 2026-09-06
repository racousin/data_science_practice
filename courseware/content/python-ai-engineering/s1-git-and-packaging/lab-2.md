# Lab 2 — Pull Request & Review

Lab 1 was you and a repository. This one needs a second person, because review
is the half of version control that cannot be practised alone.

**Pair up.** You keep your own `textstats` repository from Lab 1; you will
contribute to your partner's and they will contribute to yours.

**Deliverable:** in *your* repository — one merged pull request that you
authored, one review you gave on your partner's, and one resolved merge
conflict.

<!-- notes: Pairing is the whole lab; if the room is odd, make one trio and have
the third person review both. Part D (the conflict) is the part that actually
teaches — protect the time for it. -->

---

## Part A — Grant each other access
In **your** repository: *Settings → Collaborators → Add people*, add your
partner. They do the same for you. Accept the invitation from your email or from
`github.com/notifications`.

Then clone each other's:

```bash
git clone git@github.com:<partner>/textstats.git partner-textstats
```

You now have two repositories on disk. Keep them straight — `pwd` before every
command for the rest of this lab.

---

## Part B — Protect your main branch
In your own repository: *Settings → Branches → Add branch ruleset*, targeting
`main`:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass → select `test`
- ✅ Require approvals: **1**

Push directly to `main` now and watch it be refused:

```bash
git switch main
echo "x" >> README.md && git commit -am "direct push" && git push
```

```text
! [remote rejected] main -> main (protected branch hook declined)
```

Undo it locally — `git reset --hard origin/main` — and note what just happened:
your process is now enforced by the server, not by your intentions.

---

## Part C — Contribute to your partner's package
In **their** repository, add one function on a branch:

```bash
cd partner-textstats
git switch -c feature/average-word-length
```

```python
def average_word_length(text: str) -> float:
    """Mean token length, over the tokens of `text`.

    Raises ValueError if `text` contains no tokens.
    """
```

Use *their* conventions, not yours: same tokenisation rule as their
`word_count`, same error behaviour as their `longest_word`, tests in the same
style, exported from `__init__.py` if that is what they do.


---

## Part C — Push it and open the pull request

Write the test **first**, watch it fail, then implement.

```bash
uv sync && uv run pytest -q
git push -u origin feature/average-word-length
```

Open the pull request with a real description:

```markdown
## Why
`textstats` reports counts but nothing about token size. A mean length is
the smallest useful statistic it is missing.

## How to check it
uv sync && uv run pytest -q   →  11 passed

## Not in this PR
Median and standard deviation — separate change if wanted.
```

**Check:** CI runs on their repository, against your branch, and is green.

---

## Part D — Review their pull request
Now review the pull request **they** opened on **your** repository. On the
*Files changed* tab, leave at least **three** comments, and at least one of each:

| Kind | Example |
|---|---|
| A question | "Why strip here rather than in `word_count`? It duplicates line 8" |
| A concrete defect, with a location | "`average_word_length(' ')` divides by zero — no test covers it" |
| A `suggestion` block | a two-line fix the author applies with one click |

Then choose a verdict: **Request changes** if something must change,
**Approve** if you would ship it. Not "Comment" — this lab requires a decision.

**"LGTM" does not count.** A comment that does not name a location and a
consequence is not a review.

The author addresses the comments by pushing to the same branch:

```bash
git add -p && git commit -m "Raise on whitespace-only input"
git push          # the PR updates itself
```

Re-review, approve, **Squash and merge**, delete the branch.

---

## Part E — Cause a conflict on purpose
Both of you, in **your own** repository, at the same time:

1. `git switch main && git pull`
2. Each of you creates a branch and edits **the same line** of `README.md` —
   the project description line.
3. Both push and open pull requests.
4. Merge the first one.


---

## Part E — Resolve it

The second pull request now says *"This branch has conflicts that must be
resolved"*. Resolve it locally:

```bash
git switch main && git pull
git switch feature/<your-branch>
git merge main
```

```text
CONFLICT (content): Merge conflict in README.md
```


---

## Part E — Reading the markers

Open the file. Git has written both versions into it:

```text
<<<<<<< HEAD
Text statistics utilities for the MS2A AI Engineering course.
=======
A small library of text metrics: counts, lengths, frequencies.
>>>>>>> main
```

Edit it into the sentence you actually want — often neither side verbatim —
delete all three markers, then:

```bash
git add README.md
git commit                 # git pre-fills the merge message
git push
```

**Check:** `git log --graph --oneline -10` shows the merge commit, and the pull
request is now mergeable.

---

## Part F — Read your own history
```bash
git log --graph --oneline --decorate -15
git log --format='%an' | sort | uniq -c
```

The second command must show **both** names. A repository that only ever had one
author has not exercised anything this lab is about.

---

## Grading

| Criterion | Weight |
|---|---|
| A merged PR you authored on your partner's repository, CI green | 25% |
| Three substantive review comments, one of them a `suggestion` | 25% |
| Branch protection active on your `main`, and demonstrably enforcing | 15% |
| A resolved conflict, with both contributions surviving | 20% |
| Both authors present in your repository's history | 15% |

---

## Common failures

- **Reviewing on the *Conversation* tab.** Line comments live on *Files changed*.
- **`git pull` in the middle of a conflict.** Finish or `git merge --abort`; do
  not start a second merge inside the first.
- **Resolving by taking one side wholesale.** `--ours` / `--theirs` are for
  lockfiles, not for prose or code you should be reading.
- **Leaving the `=======` markers in.** They are valid text and Git will happily
  commit them. Your tests will not be so relaxed.
- **A green PR that nobody read.** Approval is a claim about your attention.

---

## If you finish early

- Open an **issue** on your partner's repository describing a real limitation,
  then a PR whose description says `Closes #<n>`, and watch the issue close on
  merge.
- Add a `.github/pull_request_template.md` with the *Why / How to check / Not in
  this PR* headings, so every future PR starts with them.
- Try `gh pr create`, `gh pr view --web`, `gh pr review --approve` from the
  terminal.

---

## Did you validate this lab?

- [ ] A pull request I authored is **merged** on my partner's repository, and its
      check run was green before the merge
- [ ] My repository's `main` refuses a direct push — I tried it and saw
      `protected branch hook declined`
- [ ] I left ≥ 3 line comments on my partner's PR, including one `suggestion`
      block, and gave an explicit Approve or Request-changes verdict
- [ ] At least one of my comments named a file and a line, and stated a
      consequence
- [ ] `git log --graph --oneline` in my repository shows a merge commit from a
      conflict I resolved by hand
- [ ] The final `README.md` contains both contributions, sensibly merged — not
      one side deleted
- [ ] `git log --format='%an' | sort -u` prints two names
- [ ] `grep -r '<<<<<<<' .` finds nothing

The last one takes two seconds and catches the single most common way this lab
is handed in broken.
