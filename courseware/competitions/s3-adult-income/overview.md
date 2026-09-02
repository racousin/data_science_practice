# Adult Census Income — Session 3

Predict whether a person's income exceeds $50K a year, from the census record
taken at interview time. Tabular, mixed numeric and categorical, and **~24%
positive** — which is the whole reason this dataset is here.

This is **Lab 3** with the last step attached. The lab has you write down a
metric before you look at a result and build a pipeline that does not leak; this
gives you one number on a split whose labels you have never seen, so the honesty
of the protocol is testable rather than merely asserted.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 39,073 | `id` + the 14 features |
| `y_train.csv` | 39,073 | `id,prediction` — 1 means >50K |
| `X_test.csv` | 9,769 | `id` + the same 14 features |

A stratified 80/20 split of openml `adult` (version 2) at `random_state=42`.
Ids are freshly assigned and the source ordering is discarded.

## What you submit

`submission.csv` — one row per test id, in any order:

```csv
id,prediction
te_00000,0
te_00001,1
```

`prediction` is **1** for >50K and **0** otherwise. Every id in `X_test.csv`
must appear exactly once. Anything else — a missing id, an unknown id, a
duplicate, a probability instead of a class — is rejected with a message naming
the line, and scores 0. That is deliberate: a scorer that silently imputed your
missing rows would be lying to you.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(competition_id=<id>, files=["submission.csv"])
print(client.leaderboard(<id>).head())
```

## Scoring

**F1 on the positive class.** This is the metric Lab 3 Part A asks you to
justify, and the justification is in the class balance: always predicting `0`
gets you **76% accuracy and an F1 of 0.00**. Accuracy, precision and recall are
all on the leaderboard so the threshold tradeoff from Part E is visible, but
ranking is on F1.

The logistic-regression pipeline from Part C — impute, scale, one-hot, no
leakage — scores **F1 = 0.656** at the default 0.5 threshold. That is the bar,
and here is the ladder above it on this exact split:

| model | threshold | F1 |
|---|---|---|
| always predict `0` | — | 0.000 |
| logistic regression | 0.50 | 0.656 |
| logistic regression | 0.38 | 0.692 |
| gradient boosting | 0.50 | 0.717 |
| gradient boosting | 0.38 | 0.731 |

Both moves help and they compose, but the model change is worth roughly twice
the threshold change. Report which you tried first — Part E asks you to state
what moving the threshold did to precision and recall, and "it went up 0.036"
is a real answer.

Submissions are scored once and deterministically — the same file always gets
the same number.
