# textstats — Session 1

The three functions from **Lab 1**, graded on twenty texts you have not seen.

This is the leaderboard half of "ship a package": your tests prove the code does
what *you* expect, this proves it does what the *specification* says. Those are
different claims, and the gap between them is where most bugs live.

## What you submit

An `agent.py` exposing the Lab 1 functions as methods:

```python
class Agent:
    def word_count(self, text): ...
    def char_frequencies(self, text): ...
    def longest_word(self, text): ...
```

You can upload `core.py` from your package alongside `agent.py` and import it —
the agent directory is on `sys.path`, so `from core import word_count` works.
That is the better move: it means the leaderboard is grading the code you
actually shipped, not a copy that has since drifted.

## The specification

Twenty hidden texts, three checks each, sixty checks in total. Your score is the
fraction that match the reference **exactly**.

**Tokens** are whitespace-separated and keep their punctuation.
`"the end."` is two tokens, and the second one is `end.` — four characters.

**`word_count(text)`** — the number of tokens.

**`char_frequencies(text)`** — a `dict` mapping each character to its count,
case-folded, with whitespace dropped. Punctuation and digits are characters and
do count. `"Aa b"` gives `{"a": 2, "b": 1}`. A `Counter` is fine; it arrives as
a plain object either way.

**`longest_word(text)`** — the longest token. **Ties go to the first one** in
the text: `longest_word("aaaa bbbb")` is `"aaaa"`.

## What is not graded here

`longest_word("")` must raise `ValueError` — that is Lab 1 Part C, and your
pytest suite is where it is checked. It is deliberately absent from this
competition: the platform records a raised exception as an agent crash, so a
leaderboard that *required* one would mark every correct submission broken.
Every text you are sent is non-empty.

## Scoring, and what a low score means

Score is the pass rate over all sixty checks; the leaderboard breaks it down per
function so you can see which of the three is wrong.

One thing to know about how the platform runs your code: **the first exception
ends the run.** Calls after it short-circuit rather than reaching your agent. So
if you crash on the ninth text you keep credit for the first eight and lose the
rest, and the result message names the case and the method that crashed. A score
that looks strangely round — `0.13`, `0.27` — usually means a crash early on,
not a subtly wrong answer. Read the message.
