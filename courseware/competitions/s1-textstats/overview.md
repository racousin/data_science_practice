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

## Baselines

The leaderboard column is **Pass rate** — the fraction of the sixty checks that
match the reference. It runs from 0% to 100% and **higher is better**.

The starter you are handed scores **0.0%**. Every method in it raises
`NotImplementedError`, the first raise latches the channel, and the run ends on
the very first call: `0/60 checks passed before the agent crashed`. That is the
floor, and it is what an untouched submission looks like.

The reference implementation — the three Lab 1 functions written exactly as the
specification above says — scores **100.0%, 60 of 60**. That is the bar. Unlike
a modelling competition it is also the ceiling: the checks are exact comparisons
against a pinned spec, so a correct implementation takes all of them and there
is nothing above. The useful ladder is therefore the one *below* the bar — what
a particular mistake actually costs:

| what the agent does | Pass rate | checks |
|---|---|---|
| the starter, untouched — every method raises | 0.0% | 0/60 |
| `word_count` + `longest_word` right, `char_frequencies` wrong, **and it raises on the ninth text** | 26.7% | 16/60 |
| one function right, the other two wrong but never raising | 33.3% | 20/60 |
| `word_count` + `longest_word` right, `char_frequencies` wrong, no crash | 66.7% | 40/60 |
| all three written, but `char_frequencies` never case-folds | 91.7% | 55/60 |
| all three written, but `longest_word` gives ties to the *last* token | 91.7% | 55/60 |
| the reference | 100.0% | 60/60 |

Rows two and four are the same three functions. The only difference is that one
of them raises partway through, and it costs **40 percentage points** — every
call after the first exception short-circuits, so a crash is worth far more than
a wrong answer. Fix crashes before you fix logic.

The two 91.7% rows are the ones to be suspicious of: a single misread rule costs
five checks out of sixty, which is high enough to look like success. Both are
one rule wide — five of the twenty texts contain upper-case characters, and five
contain a tie at the maximal token length — and the per-function columns on the
leaderboard tell you which of the three to open.

**You have completed Lab 1 when you score 100% (60/60).** That is exactly
reachable with the three functions the lab asks for: the reference does it, and
the build refuses to publish this competition unless it still does.
