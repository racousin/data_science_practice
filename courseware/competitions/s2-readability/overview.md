# Flesch reading-ease — Session 2

The module you build with a coding agent in **Lab 2**, checked against a
specification that leaves nothing to taste.

$$
206.835 - 1.015 \times \frac{\text{words}}{\text{sentences}} - 84.6 \times \frac{\text{syllables}}{\text{words}}
$$

The formula is the easy part. Syllable counting has no canonical answer, which
is exactly why this is the Session 2 competition: the work is turning an
ambiguous ask into a specification precise enough that a generated
implementation is either right or wrong. The spec below is that specification.
Hand it to your agent **verbatim**.

## What you submit

An `agent.py` exposing one method:

```python
class Agent:
    def flesch_reading_ease(self, text): ...
```

Upload `readability.py` from your package alongside it and import it — the agent
directory is on `sys.path` — so the leaderboard grades the code you shipped.

## The specification

**Sentences.** Count the maximal *runs* of characters drawn from `.!?`.
`"Wait... no!"` is **two** sentences, not four — the `...` is one run. A text
with no such punctuation counts as **one** sentence, never zero.

**Words.** Split on whitespace, then strip leading and trailing characters that
are not letters or digits. Tokens that become empty are dropped.
`"end."` → `end`; `"--"` → dropped; `"under_scores"` → `under_scores`, because
the stripping is only at the ends.

**Syllables**, per word. Lowercase it and count the maximal runs of `aeiouy`
(`y` counts). Then: if the word ends in `e` **and** that count is greater than
1, subtract 1. The result is never less than 1.

| word | vowel runs | ends in `e`? | syllables |
|---|---|---|---|
| `time` | `i`, `e` → 2 | yes, and count > 1 → −1 | **1** |
| `the` | `e` → 1 | yes, but count is 1 → no change | **1** |
| `place` | `a`, `e` → 2 | yes → −1 | **1** |
| `queueing` | `ueuei` is *one* run → 1 | no | **1** |
| `rhythm` | none | no | **1** (the floor) |
| `dryly` | `y`, `y` → 2 | no | **2** |
| `reevaluation` | `ee`, `a`, `uaio` … → 4 | no | **4** |

`queueing` is the one worth staring at: `u e u e i` are five *contiguous*
vowels, so the rule sees a single run. That is not how English works, and it is
still the answer — the spec is the spec. Heuristics that "look more correct"
score zero here, which is the point of the exercise.

## Scoring

Twenty hidden texts. A text counts as passed when your answer is within
`1e-6` of the reference. Score is the fraction passed.

The leaderboard also shows **mean absolute error**, which is the useful number
while you are debugging: a pass rate of 0 with an error of `0.4` is a rounding
or tie-breaking detail, while an error of `40` means the syllable rule is wrong.

## What is not graded here

Empty input must raise `ValueError` — Lab 2 Part C, checked by your own tests.
The platform records a raised exception as a crash, so every text sent here has
at least one word. And as in Session 1, **the first exception ends the run**:
you keep credit for what passed before it and the result message names the case.
