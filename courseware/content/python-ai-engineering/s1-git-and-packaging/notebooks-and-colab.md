# Notebooks & Colab

Sessions 2, 3 and 4 are notebook-shapedk. Two ideas make both usable: the kernel, and the fact that Colab's machine is not yours.

<!-- notes: 25 minutes. The out-of-order demo is the lesson — do it live and let
them watch the number be wrong. Everything else follows from the kernel model. -->

---

## What a notebook is

![The kernel model](assets/s1-git-and-packaging/notebooks-and-colab/kernel-model.png)


---

## Two things, joined loosely


- **A document** — cells of code and text, in the order you left them.
- **A kernel** — one long-running Python process holding all your variables.

The document is what you see and what git stores. The kernel is what actually
has state. Every confusing notebook bug is the gap between the two.

> A notebook is not a program. It is a REPL (Read-Eval-Print Loop) with a scrollback that you are
> allowed to edit.

---

## The consequence

![Out of order](assets/s1-git-and-packaging/notebooks-and-colab/out-of-order.png)

Cells run in the order **you** ran them, which is recorded in the `[n]` counter
beside each one — not the order they appear on screen.


---

## Restart & Run All

A notebook can show you a correct-looking sequence and a result that no fresh
run reproduces. Nothing is broken; you simply ran cell 3 twice.

**Restart & Run All** is the only honest check. Do it before you show a notebook
to anyone, and before you conclude a number is real.

<!-- notes: The live demo: define x = 1, print x, edit the cell to x = 2 without
re-running, then re-run the print. Two cells, ten seconds, and the room gets it. -->

---

## Shell and magics from inside a cell

This is where the morning's shell lesson pays off:

```python
!pip install seaborn        # ! runs a shell command
!ls -la data/
```

```python
%time model.fit(X, y)       # time this line
%%time                      # time the whole cell
%matplotlib inline
```

`!` starts a **new shell** each time, so `!cd data` does nothing that lasts. Use
`%cd` if you really need to move.

---

## Notebooks in git

A `.ipynb` is JSON, and it stores the outputs alongside the code. Two
consequences:

- **The diff is unreadable.** Re-running a notebook changes execution counts and
  base64 image blobs, so `git diff` reports a hundred changed lines for one
  edited character.
- **You commit whatever was printed** — including a dataframe of personal data,
  or an API key you echoed once.

---

## The division of labour

![Notebook to package](assets/s1-git-and-packaging/notebooks-and-colab/notebook-to-package.png)

The notebook is where you find out what the code should be. The package is where
that code lives once you know.


---

## What goes where

| Belongs in a notebook | Belongs in `src/` |
|---|---|
| plots, `df.head()`, one-off checks | any function called twice |
| trying three models to see | the one you kept |
| the narrative of an analysis | anything a test asserts on |

The rule that keeps it honest: **a notebook may not be imported.** The moment
you want to reuse a cell, it moves.

---

## Colab

![What Colab gives you](assets/s1-git-and-packaging/notebooks-and-colab/colab-anatomy.png)

A hosted notebook: a virtual machine at Google, with Python, CUDA, `torch`,
`pandas` and `sklearn` already installed, and optionally a GPU. Free, no setup,
and reachable from a phone.

Everything else about it follows from the word *hosted*.

---

## Getting a GPU

*Runtime → Change runtime type → T4 GPU*, then confirm:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If that prints `False`, you changed the setting but did not restart the runtime,
or the free GPU quota is exhausted for now. Sessions 3 and 4 need it; Session 2
does not.

---

## The filesystem is temporary

The VM is deleted when you disconnect — after roughly 90 minutes idle, or a
12-hour maximum. Everything on its disk goes with it.

Four ways things get in and out:

```python
# 1. install code from GitHub — this is what your package is for
!pip install -q git+https://github.com/<you>/<repo>.git

# 2. fetch data
!wget -q https://example.com/data.csv

# 3. mount your Drive — survives the runtime
from google.colab import drive; drive.mount('/content/drive')

# 4. take a file back to your laptop
from google.colab import files; files.download('model.pt')
```

Train a model for forty minutes, close the tab, and the weights are gone. That
is not a bug report, it is the deal.

---

## Using *your* package in Colab

This is the Session 1 deliverable, and the reason packaging was worth the effort:

```python
!pip install -q git+https://github.com/marie-durand/textstats.git@v0.1.0
```

```python
from textstats import word_count
print(word_count("the quick brown fox"))     # 4
```


---

## Three things to notice


- **No files were copied.** Colab installed from GitHub, so the code is exactly
  what you pushed — not a stale paste.
- **`@v0.1.0` pins a tag.** Without it you get whatever `main` says today, and
  a notebook that worked in October fails in December.
- **A private repository needs a token.** Public is simpler, and there is nothing
  secret in a lab repository.

If the import fails, the usual cause is that your `pyproject.toml` is not at the
repository root, or the package is not under `src/`.

---

## Colab and GitHub

Colab opens a notebook straight from a repository:

```text
colab.research.google.com/github/<owner>/<repo>/blob/<branch>/<path>.ipynb
```

That URL is how the challenge notebooks in this course reach you, and it is worth
knowing you can produce one for your own repository. *File → Save a copy in
GitHub* writes it back, commit message and all.

---

## When to stop using a notebook

Three signals, any one of which means the work has outgrown it:

1. You are scrolling to find a function you wrote earlier.
2. You copied a cell instead of calling it.
3. Something matters enough that being wrong about it would be expensive.

At that point, extract to `src/`, write the test, and import it back. The
notebook stays — it becomes the place you *use* the code rather than the place
you keep it.
