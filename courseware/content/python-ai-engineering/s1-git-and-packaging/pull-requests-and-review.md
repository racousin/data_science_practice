# Pull Requests & Code Review

The branch lesson was git on your machine. This is the part that involves a
second person — which is the part you are actually graded on.

<!-- notes: 30 minutes. Do a live PR on the projector: branch, push, open, have
a student review it, address the comment, merge. Fifteen minutes of doing beats
thirty of describing. -->

---

## What a pull request is

A request to merge one branch into another, plus a place to discuss it before it
happens.

It is not a git feature. Git has `git merge`, and it does not ask anyone. The
pull request is GitHub's addition: a merge that waits for a review and a green
test run.

![The pull-request loop](assets/s1-git-and-packaging/pull-requests-and-review/pr-lifecycle.png)

---

## The loop, as commands

```bash
git switch -c feature/readability     # 1. branch off main
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

## Writing one a stranger can review

![Anatomy of a good pull request](assets/s1-git-and-packaging/pull-requests-and-review/pr-anatomy.png)


---

## Four sections, none of them long


| Section | Answers |
|---|---|
| **Why** | what problem this solves — the diff already says *what* |
| **How to check it** | the exact command, and what it should print |
| **Not in this PR** | the thing you deliberately left out |
| **Screenshot / output** | for anything visual or numeric |

The third one is what stops a reviewer asking for scope you already decided
against.

---

## Size is the whole game

![Reviewer attention](assets/s1-git-and-packaging/pull-requests-and-review/review-size.png)

A 900-line pull request is not reviewed. It is approved.

This is not a claim about diligence — it is a claim about attention. Ask for
small changes not because the agent or the author cannot produce more, but
because **you** cannot check more.


---

## One pull request, one idea

> If the title needs the word "and", it is two.

---

## Reviewing: what to actually look for

Four questions, in this order. Stop at the first one that fails.

1. **Does it do what was asked?** Not "is this good code" — is this the task.
2. **What does it do on bad input?** A clean crash, or a plausible wrong answer?
3. **Are the tests real?** Do they assert behaviour, or only that nothing threw?
4. **Can I explain it?** If the reviewer cannot, it does not merge.

Everything else — naming, formatting, style — is either automated (see *Code
Quality*) or not worth a comment.

---

## Writing a comment that helps

| Instead of | Write |
|---|---|
| "This is wrong" | "`longest_word('')` hits `max()` on an empty sequence — line 12" |
| "Add tests" | "No test covers the tie-break. `longest_word('aaaa bbbb')` should be `aaaa`" |
| "LGTM" | "Checked the three spec rules against the tests; the digit case is missing" |
| "Why?" | "Why strip punctuation here rather than in `tokenize`? It duplicates line 8" |

A useful comment names a **location** and a **consequence**. "LGTM" on a
200-line diff is a statement about the reviewer, not the code.

GitHub gives you three verdicts. Use all three:

- **Comment** — questions, no verdict.
- **Approve** — you read it and you would ship it.
- **Request changes** — something must change before merge.

---

## Suggested changes

For a one-line fix, do not describe it — write it:

````markdown
```suggestion
    if not tokens:
        raise ValueError("empty text")
```
````

The author applies it with one click, and it lands as a commit with both of you
in the history.

---

## The three merge buttons

| Button | Result | Use when |
|---|---|---|
| **Create a merge commit** | keeps every commit, adds a merge node | a real feature branch |
| **Squash and merge** | the whole PR becomes one commit on `main` | messy history, small change |
| **Rebase and merge** | replays commits onto `main`, no merge node | you want a linear history |

For your labs, **squash** is usually right: a clean `main` where each commit is
one reviewed change. Then delete the branch — GitHub offers a button, take it.

---

## Draft pull requests

Open the PR *before* the work is finished:

```bash
gh pr create --draft --fill
```

CI runs on it, so you find out that your tests fail on Linux while you are still
writing, not after you have asked someone to read it. Mark it ready when it is.

---

## Issues, and linking to them

An issue is a piece of work that has been described but not started. A pull
request that says

```text
Closes #12
```

in its description closes issue 12 automatically when it merges. That single
line is what turns a list of issues into a record of what was done and by which
change.

---

## Protecting main

*Settings → Branches → Add rule*, on `main`:

- **Require a pull request before merging** — no direct pushes
- **Require status checks to pass** — the CI you write in the next lesson
- **Require a review** — one approval

Now the process is a rule rather than a good intention. This is what makes "the
tests are the contract" true instead of aspirational.

<!-- notes: On a free plan, branch protection on a *private* repo is limited.
Public repos get it all. Tell students to make their lab repositories public —
there is nothing secret in them and the tooling is better. -->

---

## Forks — and when you need one

You cannot push a branch to a repository you do not have write access to. So for
someone else's project you **fork** it — your own copy under your account — and
open the pull request from there.

```bash
gh repo fork owner/project --clone
```

For your labs you have write access, so you branch. Forks are for contributing
to projects that are not yours, which is how every open-source contribution you
will ever make begins.

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

## Checklist before you request a review

- [ ] The branch is up to date with `main`
- [ ] CI is green
- [ ] The diff is one idea, and the title says which
- [ ] The description says why, and how to check it
- [ ] You have read your own diff, top to bottom, on GitHub
- [ ] Nothing is in it that you cannot explain

The fifth is the one people skip. Reading your own diff in the GitHub view —
not in your editor — catches the debug print, the commented-out block, and the
file you did not mean to add.
