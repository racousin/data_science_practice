# Bank Term Deposit — Session 2

A Portuguese bank ran a direct marketing campaign: 45,211 phone calls, and for
each one, whether the client subscribed a term deposit afterwards. Predict that
outcome for calls whose result you have not been shown.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bank-marketing.ipynb)

This is the **classification** challenge of Session 2, and the notebook above
contains **no code on purpose**. It lists each step in English and leaves the
cell empty; you write it. Everything you need is in the
[bike-demand notebook](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bike-demand.ipynb), which is the same five
steps on a continuous target.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 36,168 | `id` + the 16 features |
| `y_train.csv` | 36,168 | `id,prediction` — 1 means subscribed |
| `X_test.csv` | 9,043 | `id` + the same 16 features |

A stratified 80/20 split of a public direct-marketing dataset. The rows are
shuffled and the ids are freshly assigned, so a row's id says nothing about
where it came from. The source serves the columns anonymised as `V1..V16`; they
have been restored to their UCI names, and the restoration is checked against
the data (age lies in 18–95, `month` carries twelve month abbreviations, and so
on) rather than taken on trust.

| column | meaning |
|---|---|
| `age`, `job`, `marital`, `education` | client demographics |
| `default`, `balance`, `housing`, `loan` | credit position; `balance` is in euros and can be negative |
| `contact`, `day`, `month`, `duration` | this call: channel, date, and length in seconds |
| `campaign`, `pdays`, `previous`, `poutcome` | campaign history; `pdays = -1` means never previously contacted |

Two of these repay a close look before you model anything. `pdays` uses `-1` as
a flag rather than a quantity, so it is not on the same scale as its other
values. And `duration` is the length of the call being predicted from — it is
only known once the call has ended, so a model that leans on it could not run
before placing the call. It is left in the data; deciding what to do about that
is part of the exercise.

## What you submit

`submission.csv` — one row per test id, in any order:

```csv
id,prediction
te_a2bc08ec9a5a,0
te_cbc8d2c09b81,1
```

`prediction` is **1** for subscribed and **0** otherwise. Every id in
`X_test.csv` must appear exactly once. A missing id, an unknown id, a duplicate,
or a probability instead of a class is rejected with a message naming the line,
and scores 0. Probabilities are rejected rather than silently thresholded at
0.5, because the threshold is your decision and a scorer that picked one for you
would be reporting its choice as your result.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
print(client.leaderboard(<id>).head())
```

## Scoring

**F1 on the positive class.** The target is **11.7% positive**, and that single
number is why:

> Always predicting `0` scores **88.3% accuracy** and an **F1 of 0.00**.

An accuracy leaderboard would rank that constant above most honest models.
Accuracy, precision and recall are all displayed so the tradeoff your threshold
made is visible, but ranking is on F1.

The reference baseline — `pd.get_dummies`, `StandardScaler`, then
`LogisticRegression(max_iter=1000)` at the default 0.5 threshold — scores
**F1 = 0.4528** (90.14% accuracy, precision 0.645, recall 0.349). Here is the
ladder on this exact split:

| model | threshold | F1 |
|---|---|---|
| always predict `0` | — | 0.000 |
| logistic regression | 0.50 | 0.453 |
| logistic regression | 0.30 | 0.568 |
| logistic regression | 0.20 | 0.581 |
| gradient boosting | 0.50 | 0.538 |
| gradient boosting | 0.30 | 0.632 |

Note which move is cheaper. Going from 0.50 to 0.30 costs you one line and buys
0.115 of F1; changing the model family buys 0.085. The default threshold is not
a property of the problem — it is `predict()`'s convention, and on an imbalanced
target it is rarely the right one. You cannot make that choice at all if you
kept `predict()`'s labels instead of `predict_proba()`'s probabilities.

Scaling is in the baseline for a reason: `balance` spans −8,019 to 102,127 while
`campaign` spans 1 to 63, and the solver does not converge in a sensible number
of iterations if you hand it both untouched.

Submissions are scored once and deterministically — the same file always gets
the same number.

The underlying data is public, so the held-out labels can be looked up rather
than predicted. That is not modelling, it is not what is being assessed, and a
score far above the gradient-boosting row is visible on the leaderboard for what
it is. Fit something.
