# DPE Energy Label — Session 2

Is this dwelling an energy sieve? For 30,146 French houses and flats, predict
whether the official energy label (the *DPE*) is **E, F or G**, from what the
diagnostician recorded during the visit: the building's age, its insulation,
its heating, its hot water, its ventilation.

**The model is fixed.** You do not choose, tune or submit a model. The scorer
always fits scikit-learn's default `LogisticRegression()` on the numbers you
give it, and ranks the test dwellings by ROC AUC. What you submit is a
**feature matrix**. Every point of AUC comes from your preprocessing, which is
this session's whole subject.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/courseware/competitions/s2-dpe-energy-label/starter.ipynb)

**The starter notebook** downloads the data, builds the most naive feature
matrix there is, checks it with the scorer's own model, writes a valid
`submission.csv.gz` and submits it. It is plumbing, not a solution: everything
between reading the data and writing the file is yours.

Then read **`EXPERTISE.md`**, which comes with the data. The DPE is a regulated
calculation, and the documents that define it say how each input is used: why
a missing construction year is recoverable, why an insulation quality is an
order, why a département number is not a quantity. Each preprocessing decision
worth making here has a written reason. `DICTIONNAIRE.md` describes every column.

## The data

ADEME's open data, *DPE Logements existants (depuis juillet 2021)*, Licence
Ouverte 2.0: diagnoses issued between **1 July 2024 and 31 December 2025**,
houses and flats only, **12,500 dwellings in each of 8 départements**, one per
climate zone (Nord, Bas-Rhin, Rhône, Ille-et-Vilaine, Loire-Atlantique,
Gironde, Vaucluse, Bouches-du-Rhône). A DPE that was later replaced by another
one is not included.

| file | rows | contents |
|---|---|---|
| `train.csv.gz` | 69,854 | `id`, the 99 columns, and the target `classe_efg` |
| `test.csv.gz` | 30,146 | `id` and the same 99 columns |
| `sample_submission.csv.gz` | 100,000 | the benchmark's submission: the format to follow |
| `EXPERTISE.md` | | what the regulation says about the columns |
| `DICTIONNAIRE.md` | | each column: meaning, unit, fill rate, examples |

`classe_efg` is **1** when the label is E, F or G, and 0 for A to D:
17.5% of the training rows (17.3% of the test rows).

**The values are exactly as ADEME publishes them.** French labels, codes stored
as numbers, empty cells that mean "not applicable", a construction year of 1300, a
ceiling 249 m high. Nothing has been cleaned: that is the exercise.

**What is not in the data, and why.** The label is computed by the 3CL method
as a threshold on the dwelling's calculated energy consumption and greenhouse
gas emissions. Every column the software *calculates* (consumptions, emissions,
costs, heat losses, the global insulation grade) would be the answer in
disguise, so none of them is here. What remains is what the diagnostician
*observed*. Identifiers, dates and addresses are removed too; a street address
typed into a description reads `[adresse retirée]`.

**The split is by building.** Many flats have a DPE "généré à partir des
données DPE immeuble": one diagnosis of the building, copied to each flat, and
some buildings were diagnosed more than once. Every dwelling sharing a building
DPE, an address or a building identifier with another is on the same side of
the split, so a copy in `train.csv.gz` does not give away a row of
`test.csv.gz`.

## What you submit

**`submission.csv.gz`** — a gzip-compressed CSV, and the file must have exactly
this name:

```csv
id,feature_1,feature_2,...
tr_3f9c2a71b0de,0.13,-1.20,...
te_8d01c4e97a55,1.02,0.00,...
```

- an `id` column, plus **1 to 300** feature columns, all **numeric** and
  **finite** (no text, no NaN, no inf);
- **every id of `test.csv.gz`**;
- ids of `train.csv.gz`: **all of them, or any subset of at least 20,000**.
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
client.submit(challenge_id=191, files=["submission.csv.gz"])
print(client.leaderboard(191).head())
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
E/F/G dwellings above the others, whatever the threshold. The deterministic
scorer gives the same file the same number every time. Install
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
train rows (a target statistic computed on the rows it describes, typically).

## The ladder

Every row is the same `LogisticRegression()` on this exact split. Only the
features change.

| features | test AUC |
|---|---|
| the numeric columns as pandas reads them, empty cells set to 0 | 0.641 |
| **the benchmark:** the same columns, median-imputed and standardized | **0.7606** |
| + domain cleaning: construction years, implausible values, structural gaps | 0.821 |
| + the codes treated as categories | 0.888 |
| + insulation qualities as an order | 0.924 |
| + features built from the regulation | 0.927 |
| *gradient boosting on the raw columns, for reference* | *0.934* |

The benchmark knows nothing about dwellings. Each rung above it is one item of
`EXPERTISE.md` put into numbers, and none of them requires a different model.
The last row is a flexible model given the raw columns, which a straight line
on good features gets close to; a score well above it is not preprocessing.

## Rules

- **Features come from the columns you were given.** Transform, combine,
  encode, drop: all of it is the exercise.
- **Target statistics only out-of-fold within train.** A target encoding fitted
  on the rows it encodes is caught by the train/test gap; do not try to hide it.
- **No external data and no labels from outside.** The same dwellings, labels
  included, are public on data.ademe.fr. Identifiers, addresses and dates have
  been removed and the surfaces slightly perturbed so that rows cannot simply be
  looked up, but a determined search could still find some. Looking labels up
  is not preprocessing, it is not what is assessed, and a score above the
  boosting reference is conspicuous.
- **No model smuggled in as a feature.** A column holding another model's
  prediction cannot be blocked by the scorer, and it is exactly what this
  challenge is not about.
- **Your notebook is part of the grade.** The leaderboard shows what your
  features are worth; the notebook shows that they are yours and why you built
  them.
