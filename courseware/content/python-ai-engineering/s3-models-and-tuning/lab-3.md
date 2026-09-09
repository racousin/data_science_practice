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

## Challenge 1 — Bike Sharing Demand

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


## Challenge 2 — Bank Term Deposit

Also the challenge you already submitted to. 11.7% positive, ranked on **F1**.

**Challenge:** <https://ml-arena.com/viewchallenge/184>

Continue the work you started and evaluate and improve your performances.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s3-bank-marketing.ipynb)





---

## Complete your toolkit


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

## Complete your toolkit — The two new modules examples (DO THE ONES YOU USE!)

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


---

## Deliverables

1. A score on **challenge 183** at or above **−60**.
2. A score on **challenge 184** above **F1 = 0.55**.
3. `dskit` on GitHub with `selection.py` and `evaluation.py` in it.
