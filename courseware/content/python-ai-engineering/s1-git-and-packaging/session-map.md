# What Session 1 Builds

Three hours, one artifact: a Python package that lives on GitHub, proves itself
with tests, and can be used from a notebook you do not own.

<!-- notes: 10 minutes, no keyboards yet. The point of this lesson is that every
student can name the deliverable before the first command. Ask at the end: "what
are you handing in?" — if the room cannot answer, do not move on. -->

---

## The line this session draws

![From your laptop to Colab](assets/s1-git-and-packaging/session-map/session-map.png)

Read it left to right. Each box is a place your code has to survive.

---

## Four excuses it removes


| Place | Removes the excuse |
|---|---|
| Git | "I broke it and I cannot get back" |
| GitHub | "It is on my other laptop" |
| CI | "It works on my machine" |
| Colab | "I do not have a GPU" |

Sessions 2, 3 and 4 assume all four. So does the 30-hour module after it, and so
does your project.

---

## What you hand in

![The deliverable](assets/s1-git-and-packaging/session-map/the-deliverable.png)

Note what is *not* on that list: a model, a score, a dataset. Session 1 is
engineering only. The machine learning starts in Session 2 and stands on top of
this.

---

## Why this comes first

You already write Python. What you have probably not done is write Python that
a **second person** — or a second machine, or a grader — has to run.

Everything in this session is a consequence of that one change:

- A **package** exists because `import my_stuff` has to work from a directory
  that is not yours.
- **Tests** exist because nobody will read your code to find out whether it
  works.
- **Git** exists because two people editing one file is otherwise a negotiation.
- **CI** exists because "I ran the tests" is a claim, and a green tick is
  evidence.

---

## The shape of the three hours

| Block | What | Minutes |
|---|---|---|
| 0 | Accounts and toolchain — everyone, working | 25 |
| 1 | The shell | 15 |
| 2 | Git, GitHub and review | 45 |
| — | break | 10 |
| 3 | Environments, packaging, tests | 25 |
| 4 | Linting and CI | 12 |
| 5 | Notebooks and Colab | 10 |
| — | **Lab 1** — build the thing | 38 |

The written lessons hold considerably more than that. What is lectured is the
spine; the rest is there because you will need it in week six and there will be
nobody to ask.

---

## Two rules for the session

**Type everything.** Reading a `git` command and running it are different
skills, and only one of them is examinable.

**Break things on purpose.** The lessons ask you to create a merge conflict, to
delete a commit, to run a notebook out of order. That is deliberate: the first
time you meet a conflict should not be at 23:00 the night before a deadline.

---

## What "done" looks like

At the end of the session, this sequence runs on a machine that is not yours
and produces green output:

```bash
git clone https://github.com/<you>/<your-repo>.git
cd <your-repo>
uv sync
uv run pytest
```

And this cell runs in a fresh Colab notebook:

```python
!pip install -q git+https://github.com/<you>/<your-repo>.git
import textstats; print(textstats.word_count("hello world"))
```

If both work, you are done. If either does not, the lab is not finished —
whatever the code looks like.
