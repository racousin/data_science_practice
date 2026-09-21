# Stacking and Blending

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
