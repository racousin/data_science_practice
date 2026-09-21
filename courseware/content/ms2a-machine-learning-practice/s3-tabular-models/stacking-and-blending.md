# Stacking and Blending

Every model in this session was built, validated and tuned on its own. By the
end of Lab 3 you will have several of them, and the best one is not the only
thing you know. This lesson is about using all of them at once: how the
combination is fitted without leaking, and how much — often how little — it is
worth.

<!-- notes: 50 minutes, the last modelling lesson before Lab 3. Show the
two-panel ladder early and let them read the left panel honestly: on California
housing the stack buys nothing, because one booster dominates. Then the right
panel, where the models are complementary and it buys three percent. The leak
demo is worth doing live: fit a 1-NN, print its training RMSE, watch the
meta-model hand it the weight. They will try a stack in the lab, and the aim is
that they measure it rather than assume it. -->

---

## Four ways to put models together

![Voting: different algorithms on the same data. Bagging: one algorithm on resampled data](assets/tabular/votbag.png)

*Trees and Ensembles* built the first two rows of this table. The two new ones
differ in exactly one way — the combination rule is **fitted from data**
instead of fixed in advance.

| | Diversity comes from | The combination rule is |
|---|---|---|
| Voting / averaging | different algorithms | fixed: the mode, or the mean |
| Bagging | resampled rows | fixed: the mean |
| **Blending** | different algorithms | fitted, on one held-out split |
| **Stacking** | different algorithms | fitted, on out-of-fold predictions |

Everything difficult about the last two follows from that one difference: the
rows the rule is fitted on must be rows no base model has been trained on.

---

## The two levels

![Stacking: base learners at level 0, a meta-learner at level 1](assets/tabular/stacking.jpg)

The vocabulary is fixed and worth getting right:

- **Level 0** — the *base models*. Ordinary models on the ordinary features.
- **Level 1** — the *meta-model*. One model whose features are the base models'
  predictions, one column per base model.
- **Meta-features** — that matrix of predictions, written $Z$ below. It has one
  row per training row and one column per base model.

A stack is therefore just a second model fitted on a very short, very
informative feature table. The only new question is where the numbers in $Z$
come from.

---

## When does averaging two models help?

Take two models with errors $e_1$, $e_2$, root-mean-square errors $\sigma_1 \le
\sigma_2$, and correlation $\rho$ between the errors. The equal-weight average
has

$$
\mathrm{MSE}_{\mathrm{avg}} = \frac{1}{4}\left(\sigma_1^2 + \sigma_2^2 + 2\rho\,\sigma_1\sigma_2\right)
$$

That is below $\sigma_1^2$ — the better model alone — exactly when

$$
\frac{\sigma_2}{\sigma_1} < \sqrt{\rho^2 + 3} - \rho
$$

It is an identity, not a rule of thumb, and it is worth reading at three
points. Two equally good models always gain, unless their errors are perfectly
correlated. With independent errors, a model $\sqrt{3} \approx 1.73$ times worse
still helps. At $\rho = 0.9$ — two boosters on the same table — the tolerance
collapses to 5%: anything worse than that drags the mean down.

---

## The frontier, on real pairs

![Every pair of the six base models, against the frontier](assets/tabular/stack-diversity.png)

Each point is one pair of base models on one train/test split — 180 pairs over
the two datasets everything measured in this lesson runs on: **California
housing**, where a gradient booster dominates, and a **generated task** whose
signal is half smooth and linear, half sharp interactions, where nothing does.

The shaded region is where the formula says the plain average wins, and every
point falls on the side the formula puts it.

This is why a plain average of everything you have usually loses. Your models
are not equally good, and they are not independent: two boosters on the same
table sit near $\rho = 0.9$, so the weaker one has to be within a few percent
of the stronger to earn its place in a mean.

A fitted rule does not have this problem. It can give the weaker model a small
weight instead of half.

---

## Diversity is a property of the errors

Nothing about an algorithm's name makes it diverse. Measure it:

```python
E = Z - y_tr[:, None]        # one error column per base model
print(np.round(np.corrcoef(E.T), 2))
```

On a typical feature table that matrix comes back full of 0.8 to 0.97. The
0.97 pairs — a random forest and extra trees, two boosters with different
seeds — are the ones to drop: they cost a fifth of your fitting budget and add
a column the meta-model cannot use.

> Look for a low correlation with the leader, not for a high score. The second
> model in a stack is chosen on what it gets wrong.

---

## Blending: one split

![Blending: base models fit on one part, the meta-model on the other](assets/tabular/blending.jpg)

Split the training data once. Fit the base models on the larger part, predict
the smaller part, fit the meta-model on those predictions.

---

## Blending in code

```python
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

X_a, X_b, y_a, y_b = train_test_split(X_tr, y_tr, test_size=0.2)
Z_b = np.column_stack(
    [clone(m).fit(X_a, y_a).predict(X_b) for m in base]
)
meta = RidgeCV().fit(Z_b, y_b)
```

One fit per base model, and the arithmetic is obvious. Two costs: the
meta-model is fitted on a fifth of the rows, so its weights are noisy, and the
base models never see that fifth, so they are each slightly worse than they
could be.

---

## Stacking: k folds, every row covered

![Blending spends one split; stacking covers every row](assets/tabular/stack-blend-vs-stack.png)

Run the blend split $k$ times, rotating which fold is held out. Every training
row then gets a prediction from a model that was not fitted on it, and the
meta-model trains on all of them.

---

## Out-of-fold predictions in code

They are **out-of-fold** predictions: the same numbers `cross_val_score`
produces on each held-out fold in *Model Selection and Validation*, kept
instead of being collapsed into a score.

```python
from sklearn.model_selection import KFold, cross_val_predict

cv = KFold(5, shuffle=True, random_state=0)
Z = np.column_stack(
    [cross_val_predict(m, X_tr, y_tr, cv=cv) for m in base]
)
```

`Z` has one row per training row and one column per base model. It costs $k$
fits per base model instead of one.

---

## The test-time half

The base models in `Z` were each fitted on four fifths of the data, and they no
longer exist. For the test set you refit each base model on **all** the
training rows and predict once:

```python
from sklearn.base import clone

T = np.column_stack(
    [clone(m).fit(X_tr, y_tr).predict(X_te) for m in base]
)
pred = meta.predict(T)
```

That asymmetry is deliberate, and it is the one place where a stack is
approximate: the meta-model's weights were fitted against models trained on
$4/5$ of the data, then applied to models trained on all of it. With $k = 5$
and up the difference is small. With $k = 2$ it is not.

---

## scikit-learn does all of it

```python
from sklearn.ensemble import StackingRegressor

stack = StackingRegressor(
    [("gbm", HistGradientBoostingRegressor()),
     ("rf", RandomForestRegressor()),
     ("mlp", make_pipeline(StandardScaler(), MLPRegressor()))],
    final_estimator=RidgeCV(), cv=5, n_jobs=-1)
```

`StackingClassifier` is the same class for classification. Its default
`stack_method="auto"` puts `predict_proba` columns in `Z` wherever the base
model has them, which is what you want: a hard label throws away most of what
the base model knows.

`passthrough=True` appends the original features to `Z`, which lets the
meta-model learn *where* each base model is reliable. It also multiplies the
meta-model's job by the width of your table; try it last, and only with a
regularised meta-model.

Use these classes rather than assembling it by hand. They do the refit above
for you, and they do the out-of-fold construction correctly.

---

## The failure mode: predicting rows you trained on

> If a base model ever predicts a row it was fitted on, its prediction on that
> row is too good, and the meta-model learns to trust it. Validation looks
> excellent; the leaderboard does not.

![In-sample meta-features hand the weight to the model that memorises](assets/tabular/stack-leak.png)

The left panel adds a 1-nearest-neighbour regressor to the six honest models.
Its in-sample RMSE is exactly zero: it returns each training row's own target.
Fit the meta-model on in-sample predictions and it hands that model a weight of
**1.00**, against 0.01 when the same predictions are taken out-of-fold. The
right panel is what that costs on the test set: 0.848 against 0.502, a model
**69% worse** than the one two lines of correct protocol would have given you.

This is the leakage of *Model Selection and Validation*, one level up. Nothing
in the code raises; the only symptom is a validation score you cannot
reproduce.

---

## Keep the meta-model boring

`Z` has one column per base model, those columns are strongly correlated, and
each one is already a fitted prediction of the target. There is almost nothing
left to learn, and a flexible meta-model will find it in the noise.

```python
from scipy.optimize import nnls

w, _ = nnls(np.column_stack([Z, np.ones(len(Z))]), y_tr)
```

Non-negative least squares is the default worth reaching for: the weights come
out non-negative and roughly summing to one, so they read as a mixture and a
model with nothing to add gets exactly zero rather than a small negative
weight that cancels another column. `RidgeCV` and a regularised
`LogisticRegression` are the other two reasonable answers.

A gradient-boosted meta-model on a handful of correlated columns overfits the
out-of-fold predictions and gives back the gain you just bought.

---

## What it actually buys

![Four steps on two datasets](assets/tabular/stack-ladder.png)

Six base models, six train/test splits, the same four steps on each dataset.
Test RMSE, and the change against the best single model:

| | California housing | Complementary task |
|---|---|---|
| best single model | 0.504 | 1.510 |
| equal average of all six | 0.544 (−7.9%) | 2.208 (−46%) |
| blend, one 80/20 split | 0.536 (−6.4%) | 1.530 (−1.3%) |
| stack, 5-fold out-of-fold | **0.502 (+0.4%)** | **1.458 (+3.5%)** |

---

## Reading the ladder

Three things to take from it. The equal average **loses** on both, for the
reason the frontier gives. Blending loses too, and by more than you would
expect: its meta-model saw 1,200 rows, so the weights wobble — its spread across
the six splits is four times the stack's — and its base models were fitted on
four fifths of the data and never refitted. And the stack's gain is not a
property of stacking: it is nothing where a booster dominates, and three and a
half percent where no single family fits the whole signal.

---

## What the weights say

![Solo score and stack weight, on both datasets](assets/tabular/stack-weights.png)

Read the two rows of each column against each other. The booster leads both
stacks, and in neither is the order of the weights the order of the scores.

On the left the random forest is tied for second at 0.555 and gets **nothing**:
extra trees, at the same score, already says everything it says. The MLP and the
ridge are both worse and both get a little, because they are wrong elsewhere. On
the right the MLP is the runner-up and keeps 0.14 — it is the only model of the
six that fits the linear half of that signal.

A meta-model that puts 0.89 of the weight on one base model, as the left column
does, is telling you something useful: the others have nothing to add, and the
way forward is a better base model, not a better combination.

---

## Blending or stacking

| | Blending | Stacking |
|---|---|---|
| Base fits | 1 per model | k per model |
| Meta-model trained on | one held-out split | every training row |
| Weight noise | high — few rows | low |
| The way it goes wrong | the blend set is small, so the weights wobble | a base model that predicts its own rows |
| Use it when | fits are expensive, or rows are plentiful | otherwise |

Blending exists because it is cheap and because a single held-out split is easy
to reason about in a team. Stacking is the better default at the sizes you
work with in this course.

---

## Under a time-ordered split

Everything above assumed rows are exchangeable. If they are not — the rain
challenge, anything with a timestamp — the folds that build `Z` must be
time-ordered, with the same `gap` as the rest of your protocol.

`cross_val_predict` will not do it for you:

```python
cross_val_predict(m, X, y, cv=TimeSeriesSplit(5))
# ValueError: cross_val_predict only works for partitions
```

`TimeSeriesSplit` is not a partition — the first block is never in a test fold,
so some rows have no out-of-fold prediction at all. Build `Z` yourself, and fit
the meta-model on the rows that have one:

```python
from sklearn.model_selection import TimeSeriesSplit

Z = np.full((len(y), len(base)), np.nan)
for fit, out in TimeSeriesSplit(5, gap=48).split(X):
    for j, m in enumerate(base):
        fitted = clone(m).fit(X[fit], y[fit])
        Z[out, j] = fitted.predict(X[out])
ok = ~np.isnan(Z).any(axis=1)
```

The meta-model then trains on the later rows only, which is both the cost and
the point: the weights are fitted on the regime the model will be used in.

---

## Combining distributions, not points

![A linear pool against a quantile average](assets/tabular/stack-pooling.png)

Lab 3 scores a distribution with CRPS, so there are two different things
"averaging two forecasts" can mean:

- **Linear pool** — average the *probabilities* at each value. In members:
  throw both models' members into one bag.
- **Quantile average** — average the *quantiles* at each level, which is also
  called Vincentization. In members: sort each model's, then average them
  position by position.

$$
F_{\mathrm{pool}}(x) = \frac{F_1(x) + F_2(x)}{2}
\qquad
F^{-1}_{\mathrm{quant}}(\tau) = \frac{F_1^{-1}(\tau) + F_2^{-1}(\tau)}{2}
$$

They are not the same forecast. The pool is always at least as wide as its
widest member and goes bimodal when the two disagree; the quantile average
keeps the shape and moves the location.

---

## Which one, and why

Two Gaussian forecasters 2.4 standard deviations apart, each with the right
spread, scored by CRPS over 1,500 draws:

| The disagreement is | Linear pool | Quantile average |
|---|---|---|
| bias — the truth is between them | 0.612 | **0.557** |
| real — the truth is one or the other | **0.879** | 0.929 |

So the question is not which combination is better in general, it is what the
disagreement means. Two models fitted on the same data that disagree are
usually disagreeing out of estimation error, and the quantile average is the
right answer; genuinely different scenarios call for the pool.

A linear pool of already-calibrated members comes out too wide, and the rank
histogram of *Probabilistic Prediction* says so: a **dome**, not a flat bar.
Read it before you decide which of the two you want — dome means narrow the
members, which here means averaging the quantiles instead of pooling them.

---

## A recipe, and when to stop

1. Get one model as good as you can make it. This is most of the score.
2. Add models from **different families**, not different seeds. Check the error
   correlation matrix before you spend a fitting budget on the fourth booster.
3. Build `Z` with the same CV object as the rest of your protocol — the same
   `gap`, the same groups, the same stratification — by hand where that object
   is not a partition.
4. Fit NNLS or `RidgeCV` on `Z`. Read the weights.
5. Score the whole thing on data none of it has touched, and compare against
   the best single model. If the gap is inside your CV's spread, ship the
   single model.

Two levels of stacking is where public solutions stop being reproducible and
start being folklore. The Netflix Prize was won by a blend of more than a
hundred models that Netflix never put into production: it said the engineering
effort was not justified by the accuracy it bought. That trade-off has not
changed, and on a table of your size it arrives much earlier.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   rng = np.random.default_rng(0)
   a = rng.normal(size=200_000)
   b = 0.8 * a + np.sqrt(1 - 0.8**2) * rng.normal(size=200_000)
   e1, e2 = a, 1.5 * b      # sigma2/sigma1 = 1.5, rho = 0.8
   rms = np.sqrt(np.mean(np.square([e1, e2, (e1 + e2) / 2]), axis=1))
   print(rms.round(3), round(float(np.sqrt(0.8**2 + 3) - 0.8), 3))
   # -> [1.001 1.5   1.189] 1.108
   ```

   The frontier at $\rho = 0.8$ is 1.108 and the second model is 1.5 times
   worse, so averaging must lose: 1.189 against 1.001.

2. You build `Z` with `cross_val_predict`, fit the meta-model, and report the
   meta-model's cross-validated score on `Z` as the stack's score. What is
   wrong with that number?

   **Answer.** The folds that built `Z` and the folds scoring the meta-model
   are different splits of the same rows, so a row's meta-features came from a
   base model that saw rows now in the meta-model's test fold. Score the stack
   end to end, on data outside both: a test set, or an outer loop of nested CV.

3. Your six base models are two boosters, two random forests with different
   seeds, a kNN and a ridge. The stack gives the leading booster a weight near
   1 and everything else near 0. What does that tell you, and what do you do?

   **Answer.** There is no complementary model in the set. The second booster
   copies the leader, the two forests copy each other, and the kNN and the ridge
   are too much worse to clear the frontier. Stacking cannot create diversity it
   was not given: replace the duplicates with a different family, or keep the
   single model and spend the time elsewhere.
