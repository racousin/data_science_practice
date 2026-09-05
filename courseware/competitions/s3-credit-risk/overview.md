# Credit Risk — Session 3

1,000 loan applications, 20 attributes each, and whether the applicant turned
out to be a **bad credit risk**. Predict that for applications you have not been
shown.

This is the **classification** challenge of Session 3.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-credit-risk.ipynb)

The notebook above contains **no code on purpose**. It names each step in
English and leaves the cell empty; you write it. You have just worked through
the [diabetes notebook](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-diabetes-progression.ipynb),
which is the same protocol on a continuous target — split, cross-validate,
*then* submit.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 700 | `id` + the 20 attributes |
| `y_train.csv` | 700 | `id,prediction` — 1 means bad credit risk |
| `X_test.csv` | 300 | `id` + the same 20 attributes |

A stratified 70/30 split of openml `credit-g` (version 1) at `random_state=42`.
Thirteen attributes are categorical (`checking_status`, `credit_history`,
`purpose`, `savings_status`, `employment`, `housing`, `job`, …) and seven are
numeric (`duration`, `credit_amount`, `age`, `installment_commitment`, …).

**30% of applicants are bad risks**, and that number decides your metric.

### One thing to notice about this dataset

It carries `personal_status` (which encodes sex alongside marital status) and
`foreign_worker`. Using either in a real credit decision would be unlawful in
the EU, and this challenge is not a model you would deploy — it is a controlled
setting for practising validation. It is still worth asking what your model
leans on, and `.coef_` will tell you. That question is the subject of the
fairness material later in the year.

## What you submit

`submission.csv` — one row per test id, in any order:

```csv
id,prediction
te_00000,0
te_00001,1
```

`prediction` is **1** for a bad credit risk and **0** otherwise. A missing id,
an unknown id, a duplicate, or a probability instead of a class is rejected with
a message naming the line, and scores 0. Probabilities are rejected rather than
silently thresholded at 0.5, because the threshold is your decision.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
```

## Scoring

**F1 on the positive (bad-risk) class.** Always predicting `0` — every applicant
is fine — scores **70% accuracy and an F1 of 0.00**. Accuracy cannot tell that
constant apart from a real model, which is why it does not rank. Accuracy,
precision and recall are all displayed.

## The bar, and the trap

| model | train acc | test acc | **test F1** |
|---|---|---|---|
| always predict `0` | — | 0.700 | 0.000 |
| **LogisticRegression @ 0.50** (the benchmark) | 0.793 | 0.770 | **0.571** |
| LogisticRegression @ 0.35 | — | 0.757 | 0.633 |
| RandomForest, 500 trees, unrestricted | **1.000** | 0.757 | 0.482 |
| RandomForest, 500 trees, `min_samples_leaf=10` | 0.801 | 0.737 | 0.347 |

Look at the forest's first column. **1.000** — it classifies all 700 training
applicants correctly, every one. It then scores 0.482 on the held-out 300,
below a logistic regression that never got a single training row perfect.

A training accuracy of 1.000 is not a good model. It is a model that has
memorised the answer key, and the only way to find that out before the
leaderboard does is to hold data back or cross-validate.

The second row up is the other lesson: moving the threshold from 0.50 to 0.35
buys 0.06 of F1 for one line of code, more than any model change on this table.
Pick that threshold on your own validation split, never on the leaderboard.

Submissions are scored once and deterministically — the same file always gets
the same number.
