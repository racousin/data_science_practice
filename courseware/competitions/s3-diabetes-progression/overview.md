# Diabetes Progression — Session 3

Predict a quantitative measure of disease progression one year after baseline,
from ten clinical measurements taken at baseline. 265 training rows, ten
features, one continuous target.

This is the **regression** challenge of Session 3, and it exists to make one
thing impossible to argue with:

> **Your training score is not your score.**

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-diabetes-progression.ipynb)

The worked notebook fits two models — plain `LinearRegression` and a 500-tree
`RandomForestRegressor` — and submits **both**, so the leaderboard settles the
argument rather than the text. Open it, paste your API key, run all.

## The data

| file | rows | contents |
|---|---|---|
| `X.csv` | 265 | `id` + the 10 features |
| `y.csv` | 265 | `id,prediction` — the progression score |
| `X_submission.csv` | 177 | `id` + the same 10 features |

`X.csv` and `y.csv` are the labelled data — fit on them, and carve your own
validation split out of them. `X_submission.csv` is what the leaderboard
scores; its labels are held back, so it cannot serve as a validation set.

A random 60/40 split of scikit-learn's `load_diabetes` (raw, unscaled) at
`random_state=42`. No missing values. The target runs from 25 to 346, mean 152.

| column | meaning |
|---|---|
| `age` | years |
| `sex` | coded 1 / 2 |
| `bmi` | body mass index |
| `blood_pressure` | average blood pressure |
| `total_cholesterol`, `ldl`, `hdl` | serum panel |
| `cholesterol_ratio` | total cholesterol ÷ HDL |
| `log_triglycerides` | log of serum triglycerides |
| `glucose` | blood sugar |

Only 265 training rows, deliberately. Small `n` is the regime where a flexible
model can memorise, and memorising is what you are here to catch.

## What you submit

`submission.csv` — one row per id in `X_submission.csv`, in any order:

```csv
id,prediction
te_00000,171.4
te_00001,92.0
```

`prediction` is a real number. Every id in `X_submission.csv` must appear exactly
once; a missing id, an unknown id, a duplicate, a non-numeric value or a `NaN`
is rejected with a message naming the line, and scores 0.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
```

## Scoring

**R²**, on the 177 held-out patients. Ranking is on R² rather than RMSE because
the leaderboard sorts descending and needs a higher-is-better number; R² = 0 is
exactly the model that predicts the training mean. RMSE and MAE are shown too,
in the target's units.

## The point of this challenge

Six models, on this exact split. Read the columns in order.

| model | train R² | 5-fold CV R² | **test R²** |
|---|---|---|---|
| predict the training mean | — | — | 0.000 |
| **LinearRegression** | 0.507 | **0.427** | **0.516** |
| RidgeCV | 0.507 | 0.427 | 0.513 |
| RandomForest, 500 trees, `min_samples_leaf=10` | 0.637 | 0.400 | 0.502 |
| RandomForest, 500 trees, `max_depth=3` | 0.595 | 0.385 | 0.494 |
| RandomForest, 500 trees, unrestricted | **0.920** | 0.358 | 0.492 |
| HistGradientBoosting | 0.913 | 0.311 | 0.445 |

Three things are true here, and all three are checkable:

1. **The two best training scores are the two worst test scores.** The
   unrestricted forest (0.920) and the boosted trees (0.913) finish last and
   second-to-last on data they have not seen. The straight line, dead last on
   training at 0.507, wins.
2. **Cross-validation gets the order exactly right.** Rank those six models by
   their 5-fold CV score on the *training set alone* and you get precisely the
   test-set ranking — all six, in order, without ever looking at a test label.
3. **The forest is not broken.** Restrict it (`min_samples_leaf=10`) and its
   training score falls from 0.920 to 0.637 while its test score *rises* to
   0.502. Giving up training accuracy bought generalisation.

The gap between column 1 and column 3 is called overfitting. Column 2 is the
tool that lets you see it before the leaderboard does — which matters, because
you get one honest look at the test set and the leaderboard is it.

A warning that follows directly: **the leaderboard is a test set too.** Submit
twenty variants, keep the one that scored best, and you have fitted 177 rows by
hand. Choose your model on CV; use the leaderboard to confirm, not to search.

Submissions are scored once and deterministically — the same file always gets
the same number.
