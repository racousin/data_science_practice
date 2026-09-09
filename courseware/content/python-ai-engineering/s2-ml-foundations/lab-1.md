# Lab 1 — Two Challenges and a Toolkit

Two datasets, one continuous target and one class label, both taken end to end:
look at the data, build the columns your plots ask for, fit a linear model,
submit. Then turn the code you wrote twice into a package you will install on
every dataset you meet for the rest of the year.

**Deliverable:** a submission on each of the two challenges, and a GitHub
repository holding your toolkit.

<!-- notes: Section 1 in the room, walked through on the projector — it is the
worked notebook and it runs. Sections 2 and 3 are homework; 3 is the one worth
the marks and the one students will skip if you do not say so out loud. The
whole lab leans on Session 1: same src/ layout, same pytest, same CI workflow,
same install-from-GitHub in Colab. -->

---


## Section 1 — Bike Sharing Demand

Predict hourly bike rentals from the calendar and the weather. Continuous
target, ranked on **−MAE**.

**Challenge:** <https://ml-arena.com/viewchallenge/183>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bike-demand.ipynb)

**This notebook is worked.** It runs top to bottom: it downloads the data,
plots it, fills the holes, encodes the words, fits a `LinearRegression` and
submits. Paste your `mlk_user_...` key into cell 2 and run all.

---

## Section 1 — Run it, then read it

Running it takes four minutes. Reading it is the lab.

| stage | −MAE |
|---|---|
| predict the training mean | −174.98 |
| `LinearRegression` on the raw columns | −138.88 |
| the same model, `hour` as 24 categories | −100.28 |

The last two rows are the **same model**. What moved the number was one column,
and the notebook's section 8 tells you which one and why — it is the shape you
already saw in plot 3b. That is the whole argument of Session 2: on this
dataset the columns are worth more than the model.

**Done when:** your name is on the leaderboard of challenge 183 with a score at
or above **−138.88**, the notebook's benchmark.

---

## Section 2 — Bank Term Deposit

45,211 marketing calls; predict which clients subscribed. Binary target, **11.7%
positive**, ranked on **F1**.

**Challenge:** <https://ml-arena.com/viewchallenge/184>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bank-marketing.ipynb)

**This notebook has no code in it.** Each step is written out in English and the
cell below it is empty. The steps are the ones you just watched; write them.

---

## Section 2 — The bar, and the two traps

| model | threshold | F1 |
|---|---|---|
| always predict `0` | — | 0.000 |
| logistic regression | 0.50 | 0.453 |

Always predicting `0` is **88.3% accurate** and finds not one subscriber. That
gap is why the challenge ranks on F1 and not on accuracy.

The two things that are genuinely new here, both flagged in the notebook where
they bite:

- **The threshold is yours.** `predict()` cuts at 0.5 because that is its
  convention, not because your problem says so. Moving it costs one line and
  buys 0.128 of F1 — more than any model swap in Session 3 will.
- **`duration` is the length of the call you are predicting the outcome of.**
  You only know it once the call is over. Decide whether to use it, and write
  down why.

**Done when:** you are on the leaderboard of challenge 184 above **F1 = 0.453**.

---

## Section 3 — Build your toolkit

You have now written data import, summary statistics, and plots — twice each,
in two notebooks. Go back through both, find every block you wrote a second
time, and put it in a package.

**The third time is a function.** Same shape as Session 1's `textstats`, with
real code in it: build it on GitHub, then `pip install` it from the notebook —
on these two datasets, and on every one that follows.

```text
dskit/
├── .github/workflows/tests.yml
├── pyproject.toml
├── README.md
├── src/dskit/
│   ├── __init__.py
│   ├── visualization.py   ← seaborn plots
│   ├── stats.py           ← describe a table
│   └── preprocessing.py   ← missing values, encoding
└── tests/test_dskit.py
```

---

## Deliverables

1. A score on **challenge 183** at or above **−138.88**.
2. A score on **challenge 184** above **F1 = 0.453**.
3. A public GitHub repository that will be usefull for your next dataset
