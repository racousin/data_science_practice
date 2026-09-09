# Lab 1 — Modelling & Evaluation

Same two datasets as Session 2. In that session you looked at them and fitted
one linear model on each — an EDA and a first baseline. Everything that decides
*which* model you keep was still missing.

That is this lab. The data does not change, the columns do not change; what
changes is the model, and the machinery that ranks models honestly before the
leaderboard does. Both scores move a long way for that reason alone.

**Deliverable:** a better submission on each of the two challenges, and the
model-selection half of your toolkit.

<!-- notes: Section 1 in the room, walked through on the projector — it is the
worked notebook and it runs top to bottom in about four minutes plus the grid
search. Sections 2 and 3 are homework. Section 2 is deliberately the *same*
challenge they submitted to in Session 2, so the only thing that can explain
the jump is what they learned here; say that out loud, it is the argument of
the whole session. -->

---

## Section 1 — Bike Sharing Demand

The challenge you already submitted to. Continuous target, ranked on **−MAE**.

**Challenge:** <https://ml-arena.com/viewchallenge/183>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-bike-demand.ipynb)

**This notebook is worked.** It runs top to bottom. Paste your `mlk_user_...`
key into cell 2 and run all; the grid search in section 6 takes a few minutes
and everything else is seconds.

It does four things in an order that is itself the lesson: **split first**
(chronologically — this data is a forecast), **transform second** with every
statistic learned on the training rows only, **compare candidates** on training
*and* validation, then **search** with `GridSearchCV` and spend the test set
once.

---

## Challenge 1 — Run it, then read the two columns

| candidate | train MAE | validation MAE |
|---|---|---|
| `LinearRegression` | 79.09 | 103.78 |
| `Ridge(alpha=100)` | 79.16 | 102.35 |
| `RandomForest`, depth 10 | 21.58 | **65.88** |
| `RandomForest`, unrestricted | **7.88** | 66.51 |

The last two rows are the lab. The unrestricted forest is off by 7.9 bikes an
hour on the rows it was fitted to and by 66.5 on rows it has not seen; the
shallower forest is three times *worse* on training and better where it counts.
The only column that could have told you that is the right-hand one — and you
only have it because the split came first.

Two more things to notice while it runs:

- **Ridge buys 1.4 MAE and raises the training error.** That is not a
  disappointment, it is a diagnosis: with 19 columns against 8,899 rows the
  linear model is *underfitting*, and a penalty cannot fix a model that is too
  simple.
- **`hour` is never re-encoded.** Session 2 needed it as 24 categories and it
  was worth about 38 MAE there. The tree splits on `hour < 7.5` by itself. What
  cost you a feature-engineering step is free once the model family can bend.

---

## Challenge 1 — What that is worth

| model | −MAE |
|---|---|
| Session 2's linear model | −138.88 |
| the same data, `RandomForest` chosen on a chronological split | **−48.06** |

**Done when:** your name is on the leaderboard of challenge 183 at or above
**−60**. The notebook itself reaches −48.06; the bar leaves room for a
different search.

---

## Section 2 — Bank Term Deposit

Also the challenge you already submitted to. 11.7% positive, ranked on **F1**.

**Challenge:** <https://ml-arena.com/viewchallenge/184>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-bank-marketing.ipynb)

**This notebook has no code in it.** Each step is written out in English and the
cell below it is empty — the steps are the ones you just watched, on a
classification target. Continue the work you started in Session 2; the protocol
is the same and the bar is higher.

Three things differ from the bike notebook, and each one is a decision you have
to make rather than a line you copy:

- **The split is stratified, not chronological.** These rows have no time order
  to protect, and an 11.7% target does have a class ratio to protect. Using the
  bike notebook's `iloc` split here is wrong for a reason you should be able to
  state.
- **The threshold is yours.** `predict()` cuts at 0.5 because that is the
  library's convention, not because your problem says so. On this target it is
  worth more than any model in your grid — sweep it on validation, never on the
  leaderboard.
- **`duration` is the length of the call you are predicting the outcome of.**
  Same trap as Session 2, same requirement: decide, and write down why.

| model | threshold | F1 |
|---|---|---|
| logistic regression (your Session 2 submission) | 0.50 | 0.453 |
| gradient boosting | 0.50 | 0.553 |
| gradient boosting | 0.24, tuned on validation | **0.639** |

**Done when:** you are on the leaderboard of challenge 184 above **F1 = 0.55**.
That is up from Session 2's 0.453 — the validation bar is the only thing that
moved, and it moved because you now know how to clear it.

---

## Section 3 — Finish your toolkit

Session 2 left `dskit` deliberately short. You built the part that prepares a
dataset; you have now written the part that *chooses a model* twice, in two
notebooks, on two different metrics. Third time is a function.

```text
dskit/
├── .github/workflows/tests.yml
├── pyproject.toml
├── README.md
├── src/dskit/
│   ├── __init__.py
│   ├── visualization.py   ← seaborn plots            (Session 2)
│   ├── stats.py           ← describe a table         (Session 2)
│   ├── preprocessing.py   ← missing values, encoding (Session 2)
│   ├── selection.py       ← splits and search        (new)
│   └── evaluation.py      ← scoring and thresholds   (new)
└── tests/test_dskit.py
```

---

## Section 3 — The two new modules

They hold exactly the code you have now written twice:

**`selection.py`**

- `ordered_split(X, y, test_size=0.2)` — the last fraction, in place, no
  shuffle. What the bike notebook needed.
- `stratified_split(X, y, test_size=0.2, random_state=0)` — a thin wrapper that
  makes `stratify=y` impossible to forget. What the bank notebook needed.
- `search(model, grid, X, y, cv, scoring)` — `GridSearchCV`, returning the
  fitted search *and* `cv_results_` as a tidy frame sorted by rank, because the
  spread across folds is the half everybody drops.

**`evaluation.py`**

- `compare(candidates, X_train, y_train, X_val, y_val, metric)` — one row per
  candidate, a train column and a validation column. This is the table from
  section 5 of both notebooks, and it is the single most reused thing in the
  session.
- `best_threshold(y_true, proba, metric=f1_score)` — sweep, argmax, return the
  threshold and its score.

Two rules carried over from Session 1: it installs from GitHub into Colab, and
`uv run pytest` is green on a fresh clone. A function without a test is a
function you will not trust in three weeks.

---

## Deliverables

1. A score on **challenge 183** at or above **−60**.
2. A score on **challenge 184** above **F1 = 0.55**.
3. `dskit` on GitHub with `selection.py` and `evaluation.py` in it, tested, CI
   green.

---

## Did you finish?

- [ ] My split was made **before** any imputation, encoding or `fit`
- [ ] The bike split is chronological and the bank split is stratified, and I
      can say in one sentence why they are not the same
- [ ] Every candidate I report has a training score **and** a validation score
- [ ] I compared the gap between my top candidates against `std_test_score`,
      and said whether I found a winner or a region
- [ ] My test slice was scored **once**, after the choice was made
- [ ] I did not switch models because one of them won on the test set
- [ ] My bank threshold was chosen on validation, not on the leaderboard
- [ ] `uv run pytest` is green on a fresh clone of `dskit`
- [ ] Challenge 183: **−MAE ≥ −60**
- [ ] Challenge 184: **F1 > 0.55**

If a validation score surprises you on the upside, assume a leak until you have
found the reason it is real.
