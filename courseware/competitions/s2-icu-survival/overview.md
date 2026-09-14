# Critical Care Survival — Session 2

Will this patient be alive in two months? For 2,732 seriously ill adults
of the SUPPORT cohort, predict whether the patient **died within 60 days of
study entry**, from what the chart held on the third day: the diagnosis, the
vital signs, the day-3 labs, the functional status.

**The model is fixed.** You do not choose, tune or submit a model. The scorer
always fits scikit-learn's default `LogisticRegression()` on the numbers you
give it, and ranks the test patients by ROC AUC. What you submit is a
**feature matrix**. Every point of AUC comes from your preprocessing, which is
this session's whole subject.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/courseware/competitions/s2-icu-survival/starter.ipynb)

**The starter notebook** downloads the data, builds the most naive feature
matrix there is, checks it with the scorer's own model, writes a valid
`submission.csv.gz` and submits it. It is plumbing, not a solution: everything
between reading the data and writing the file is yours.

Then read **`EXPERTISE.pdf`**, which comes with the data. SUPPORT was a
prognostic study, and its investigators wrote down what they measured, what a
normal value is, and what they did when a lab was missing: why an empty
albumin is not "unknown", why a creatinine of 700 can be an ordinary value,
why a diagnosis is not a scale. Each preprocessing decision worth making here
has a clinical reason. `DICTIONARY.pdf` describes every column.

## The data

The SUPPORT cohort (Study to Understand Prognoses and Preferences for Outcomes
and Risks of Treatments): **9,105 seriously ill adults** admitted to five US
academic medical centres between 1989 and 1994, with one of nine qualifying
diagnoses. The physiology is the worst value recorded on the third day after
study entry, as the investigators collected it for their prognostic model. The
public file is on hbiostat.org and the UCI repository.

| file | rows | contents |
|---|---|---|
| `train.csv.gz` | 6,373 | `id`, the 31 columns, and the target `dead` |
| `test.csv.gz` | 2,732 | `id` and the same 31 columns |
| `sample_submission.csv.gz` | one row per train and test id | the benchmark's submission: the format to follow |
| `EXPERTISE.pdf` | | what clinicians know about the columns |
| `DICTIONARY.pdf` | | each column: meaning, unit, fill rate, examples |

`dead` is **1** when the patient died within 60 days of study entry, and 0
otherwise: 46.2% of the training rows (46.2% of the test
rows).

**What the columns hold.** Every column is what the chart held on day 3 of the
study, plus the bill. Five sites, each charting in its own units.
Several spellings of sex and race. Ages of 999. Sodium values above 1,000. A
glucose column that pandas reads as text because one site writes "not done"
in it. Labs that are empty because nobody ordered them, and a
functional-status questionnaire that is empty when the patient could not
answer.

**Train and test.** Each row is one patient, and the two files have the same
31 columns. `charges`, the hospital bill, is filled for 98% of the training
patients and empty for every test patient.

## What you submit

**`submission.csv.gz`**: a gzip-compressed CSV, and the file must have exactly
this name:

```csv
id,feature_1,feature_2,...
tr_3f9c2a71b0de,0.13,-1.20,...
te_8d01c4e97a55,1.02,0.00,...
```

- an `id` column, plus **1 to 300** feature columns, all **numeric** and
  **finite** (no text, no NaN, no inf);
- **every id of `test.csv.gz`**;
- ids of `train.csv.gz`: **all of them, or any subset of at least 4,000**.
  Which rows the model learns from is a preprocessing decision too: you may
  leave out rows you do not trust;
- no labels: the scorer has its own.

`df.to_csv("submission.csv.gz", index=False)` writes it; pandas compresses from
the extension. A file that breaks a rule (a missing test id, an unknown or
duplicated id, a duplicated or unnamed column, a text column, a NaN, a file
that is not gzip) is rejected with a message naming the first offender, scores
0 and never reaches the leaderboard. It still uses one of your submissions of
the day.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=192, files=["submission.csv.gz"])
print(client.leaderboard(192).head())
```

## Scoring

The scorer does this, and nothing else:

```python
model = LogisticRegression()          # scikit-learn 1.8.0, all defaults
model.fit(X[your train ids], y[your train ids])
score = roc_auc_score(y_test, model.predict_proba(X[test ids])[:, 1])
```

No scaling, no imputation, no tuning of its own. **ROC AUC on the test rows**
ranks the leaderboard; it measures how well your features let the model put
the patients who died above the others, whatever the threshold. The
deterministic scorer gives the same file the same number every time. Install
`scikit-learn==1.8.0` and your local check reproduces it.

Beside the score, the leaderboard shows:

| column | what it tells you |
|---|---|
| Train AUC | the same AUC on your own train rows |
| Features | how many columns you sent |
| Train rows | how many train ids you kept |
| Converged | 1 if the solver reached its optimum within its 100 iterations |

Two warnings come with a submission's result. **"lbfgs did not converge"**: the
solver stopped before the optimum, which is what happens when columns live on
very different scales. **"train AUC is … above test AUC"**: when the train AUC
is more than 0.05 above the test AUC, a feature carries the target on the
train rows and not on the test rows (a column known only after the outcome, or
a target statistic computed on the rows it describes).

## The ladder

Every row is the same `LogisticRegression()` on this exact split. Only the
features change.

| features | test AUC |
|---|---|
| a constant column ("always alive") | 0.500 |
| the numeric columns as pandas reads them, empty cells set to 0 | 0.757 |
| **the benchmark:** the same columns, median-imputed and standardized | **0.8499** |
| + repaired values and harmonised units | 0.876 |
| + labs imputed as the investigators did, absences kept as information | 0.900 |
| + categories encoded as categories, orders as orders | 0.916 |
| + the bedside formulas | 0.937 |
| *gradient boosting on the raw columns, for reference* | *0.907* |

The benchmark knows nothing about patients. Each rung above it is one item of
`EXPERTISE.pdf` put into numbers, and none of them requires a different model.
The last row is a flexible model given the raw columns, which a straight line
on good features gets close to; a score well above it is not preprocessing.
The pass bar for this module is a test AUC of **0.905**.

## Rules

- **Features come from the columns you were given.** Transform, combine,
  encode, drop: all of it is the exercise.
- **Target statistics only out-of-fold within train.** A target encoding fitted
  on the rows it encodes is caught by the train/test gap; do not try to hide it.
- **No external data and no labels from outside.** Joining the public SUPPORT
  files, or any other file, is external data and is forbidden. Looking things up is not
  preprocessing, it is not what is assessed, and a score far above the
  boosting reference is conspicuous.
- **No model smuggled in as a feature.** A column holding another model's
  prediction cannot be blocked by the scorer, and it is exactly what this
  challenge is not about.
- **Your notebook is part of the grade.** The leaderboard shows what your
  features are worth; the notebook shows that they are yours and why you built
  them.
