## What a shell is

A program that reads a line, finds the executable it names, runs it with the
rest of the line as arguments, and prints what comes back. That is all.

It is not Python, it is not the editor, and it is not "the black window". It is
a very small language whose vocabulary is the programs installed on your
machine.

```bash
uv run pytest -q tests/
```

One command (`uv`), three arguments. The shell does not know what any of them
mean.

---

## Where you are standing

![Paths](assets/s1-git-and-packaging/the-shell/paths.png)

Half of all "the file is not there" reports are a working-directory problem.

---

## Three commands that settle it


```bash
pwd          # print working directory — where am I?
ls           # what is here?
cd ..        # go up one level
```

`cd` with no argument goes home. `cd -` goes back to where you just were.

---

## The ten commands

There are thousands. These ten cover this course:

```bash
pwd                      # where am I
ls -la                   # list, including hidden files, with detail
cd path/                 # move
cat file.txt             # print a small file
less file.txt            # page through a big one  (q to quit)
mkdir -p a/b/c           # create directories
cp src.py backup.py      # copy
mv old.py new.py         # move, and also rename
rm file.txt              # delete — no undo, no bin
head -20 / tail -20      # first / last 20 lines
```

`ls -la` is the one to make a habit. Hidden files start with a dot, and `.git`,
`.venv`, `.gitignore` and `.github` are all hidden — a plain `ls` shows you a
directory that looks empty when it is not.

---

## Two habits, worth more than the ten commands

**Tab completion.** Type three characters and press Tab. The shell finishes the
path, and if it does not, the thing you were about to type does not exist. It is
a spell-checker for paths that runs before you press Enter.

**History.** Up-arrow walks back through what you ran. `Ctrl+R` searches it:

```text
Ctrl+R  pytest        →  uv run --all-extras pytest -q tests/test_core.py
```


---

## Why `command not found` happens

![PATH resolution](assets/s1-git-and-packaging/the-shell/path-resolution.png)

`PATH` is a list of directories, separated by colons. The shell searches it left
to right and runs the first match.

```bash
echo $PATH
which python
which -a python      # every match, in search order
```


---

## Three consequences you will meet this term


1. **You installed something and the shell cannot see it.** The installer edited
   your shell's startup file; this shell started before that. Open a new one.
2. **The wrong Python runs.** `which -a python` shows you why: a different one
   is earlier on the `PATH`. This is what a virtual environment manipulates.
3. **It works in the terminal but not in VS Code.** Two different shells, two
   different `PATH`s. Usually the interpreter selection, not the shell.

---

## Environment variables

Variables the shell hands to every program it starts.

```bash
echo $HOME
export MLARENA_API_KEY=mlk_user_xxx      # set for this shell and its children
env | grep MLARENA                       # check
```

Two properties that catch people out:

- **They die with the shell.** Close the terminal and the variable is gone. To
  make one persist, put the `export` line in `~/.zshrc` or `~/.bashrc`.
- **They are inherited, not shared.** A program you start gets a *copy*. It
  cannot change yours.

<!-- notes: This is where the API-key lab step comes from, and where students
lose twenty minutes wondering why their key "disappeared". -->

---

## Never commit a secret

An API key in a shell variable is fine. An API key in a tracked file is a
published API key.

```bash
echo "MLARENA_API_KEY=mlk_user_xxx" > .env
echo ".env" >> .gitignore
```

If you push one by accident: **rotate it**, do not delete it. The history is
public from the moment it lands on GitHub, and deleting the line in the next
commit changes nothing.

---

## Output, errors, and the number at the end

![Streams and pipes](assets/s1-git-and-packaging/the-shell/streams-and-pipes.png)


---

## Three channels and a number

Every program gets three channels and returns one integer:

| | |
|---|---|
| **stdout** | the answer |
| **stderr** | the complaint — *this is where the traceback goes* |
| **stdin** | input, if any |
| **exit code** | `0` means success; anything else means failure |

```bash
uv run pytest ; echo "exit code: $?"
```

`$?` holds the exit code of the last command. This is the mechanism behind CI:
GitHub does not read your test output, it reads that number.

---

## Redirection and pipes

```bash
uv run pytest > out.txt          # stdout to a file (stderr still on screen)
uv run pytest > out.txt 2>&1     # both to the file
uv run pytest 2> errors.txt      # only the errors
uv run pytest | tail -20         # last 20 lines only
uv run pytest | grep FAILED      # only the lines that matter
```

The distinction between `>` and `2>` is the reason a student writes
`pytest > log.txt`, sees an empty file, and concludes the tests printed nothing.
They printed to stderr.

---

## Chaining

```bash
uv sync && uv run pytest      # run the second only if the first succeeded
uv run pytest || echo "FAILED"
uv sync ; uv run pytest       # run both, regardless
```

`&&` is the one to use. It respects the exit code, so a broken install does not
produce a confusing test failure ten seconds later.

---

## Reading an error

```text
Traceback (most recent call last):
  File "/Users/marie/projects/textstats/src/textstats/core.py", line 12, in longest_word
    return max(tokens, key=len)
ValueError: max() arg is an empty sequence
```

Read it **bottom up**:

1. The last line is the error type and message.
2. The line above it is the code that raised it.
3. The lines above that are how you got there.

The path in the traceback is also information: if it does not point where you
expect, you are running a different copy of your code than you think.

---

## Stopping and cleaning up

| Keys | Effect |
|---|---|
| `Ctrl+C` | interrupt the running program |
| `Ctrl+D` | end of input — exits a Python REPL or the shell |
| `Ctrl+L` | clear the screen |
| `q` | quit `less`, `git log`, `man` |

`Ctrl+C` in the middle of a training run is how you stop it. It is not how you
stop `git rebase` — that one wants `git rebase --abort`.

---

## Wildcards

The shell expands these *before* the program sees them:

```bash
ls *.py                  # every .py in this directory
ls tests/test_*.py       # matching a prefix
rm -rf build/            # a directory and everything under it
```

`rm -rf` deletes without confirmation and without a bin. Type the path, look at
it, then press Enter — in that order.

---

## Deliberately not covered

`sed`, `awk`, `grep` beyond a literal search, shell scripting, job control,
`vim`. You will meet them; none is needed here, and each one is a lesson of its
own.

Two exceptions worth knowing exist:

```bash
grep -rn "def load_dataset" src/     # find a definition
man ls                               # the manual for a command  (q to quit)
```

---

## A five-minute drill

From a fresh terminal, without using the mouse:

```bash
cd ~
mkdir -p scratch/demo && cd scratch/demo
echo "hello" > a.txt
cat a.txt
ls -la
cd ../..
rm -rf scratch
```
