# Scaling and Normalization

Scaling changes no information. It changes the geometry the optimiser walks
over, and for half the models in this course that is the difference between
convergence and a plateau.

<!-- notes: 25 minutes. Lead with the "who cares" table — students routinely
scale before a random forest and believe it helped. Demo: k-NN on income in
euros versus income in thousands, same data, different neighbours. -->

---

## Who needs it

| Model | Scaling | Why |
|---|---|---|
| Decision tree, random forest, gradient boosting | **no** | splits are order-based; a monotone map changes nothing |
| Linear / logistic regression, unregularised | no effect on fit | coefficients absorb the scale |
| Ridge, Lasso, elastic net | **yes** | the penalty is applied to raw coefficients |
| k-NN, k-means, SVM with RBF | **yes** | they compute distances |
| Neural networks | **yes** | gradient magnitudes and initialisation assume it |
| PCA | **yes** | it maximises variance, which has units |

Scaling before a tree costs only time. Skipping it before a distance-based model
costs the model: a column in euros dominates a column in years by four orders of
magnitude, and "nearest" comes to mean "similar income".

---

## StandardScaler

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)
```

Zero mean, unit variance, unbounded range. The default, and what regularised
linear models, SVMs and neural networks expect.

It does not make a distribution normal. It shifts and rescales one; a skewed
column comes out skewed.

---

## MinMaxScaler

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler

X_tr = MinMaxScaler().fit_transform(X_tr)
```

Maps to $[0, 1]$. Use it when a bounded range is required — image pixels, some
activation functions, anything assuming a unit interval.

Both bounds come from the training set, so a larger test value maps above 1.
That is correct, not a bug: clipping it would hide the fact that production is
outside the range you trained on.

---

## RobustScaler

$$
x' = \frac{x - Q_2}{Q_3 - Q_1}
$$

```python
from sklearn.preprocessing import RobustScaler

X_tr = RobustScaler().fit_transform(X_tr)
```

Median and interquartile range instead of mean and standard deviation. A single
extreme value moves a mean and a standard deviation; it moves neither quartile.

Reach for it when the column has genuine outliers you decided to keep — which,
after the previous lesson, is a decision you made on purpose.

---

## Choosing

| Situation | Scaler |
|---|---|
| Default, any gradient or distance method | `StandardScaler` |
| A bounded range is required | `MinMaxScaler` |
| Heavy tails, outliers kept | `RobustScaler` |
| Sparse matrix, zeros must stay zero | `MaxAbsScaler` |
| Row vectors compared by direction | `Normalizer` (per row, not per column) |

`Normalizer` is the odd one out: it rescales each **row** to unit norm, which is
what cosine similarity wants and almost never what a tabular feature wants.
Confusing it with the column scalers is a common and silent error.

---

## Skew is a different problem

Scaling moves a distribution; it does not reshape it. For a column with a long
right tail — revenue, population, counts — the reshaping tool is a transform:

```python
df["revenue_log"] = np.log1p(df["revenue"])
```

`log1p` computes $\log(1 + x)$, so a zero stays a zero. Plain `log` on a column
containing zeros produces `-inf`, which fails silently in some estimators and
loudly in others.

---

## Box-Cox and Yeo-Johnson

$$
x^{(\lambda)} = \frac{x^{\lambda} - 1}{\lambda}
$$

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson", standardize=True)
X_tr = pt.fit_transform(X_tr)
```

Both fit the exponent $\lambda$ by maximum likelihood, to bring the column as
close to normal as it can get. Box-Cox requires strictly positive values;
Yeo-Johnson handles zeros and negatives, which is why it is the default.

$\lambda$ is a fitted parameter. Fitted on train.

---

## Scaling the target

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(regressor=Ridge(),
                                   func=np.log1p, inverse_func=np.expm1)
```

For a skewed target, regressing on the log and inverting the prediction usually
helps. Doing it by hand is where people get hurt: they train on `log(y)`, report
an RMSE of 0.31, and forget that the metric is now in log units and not
comparable to anything.

`TransformedTargetRegressor` applies the inverse before `predict` returns, so
the score is in the original units and no bookkeeping is left to a human.

---

## What not to scale, and the rule again

- **one-hot columns** — indicators, not magnitudes; scaling turns 0/1 into two
  meaningless real numbers and makes the coefficients unreadable
- **already-bounded features** — a proportion in $[0, 1]$ is fine
- **the inputs of a tree model** — no benefit, and it hides the raw thresholds
  from anyone reading the model

In a `ColumnTransformer` this is free: the scaler sits on the numeric branch and
never sees the encoder's output. `fit` on train, `transform` on test — a scaler
is two numbers per column, so the leak is small, but it is the same leak and the
fix costs nothing.

> Scale for the model, not for the data. If the model computes no distance and
> no gradient, the scaler is decoration.

---

## Check yourself

1. Of a random forest, a Ridge regression and a k-NN classifier, which one is
   unaffected by scaling — and what property decides it?

   **Answer.** The random forest. What decides it is whether the model consults
   *order* or *magnitude*: a tree splits on order and scaling is a monotone map,
   so every split it could have made it can still make. Ridge is affected because
   the penalty applies to the raw coefficients, k-NN because it computes
   distances. Scaling before a tree costs only time; skipping it before the other
   two costs the model.

2. Run this. You should get exactly the output shown.

   ```python
   from sklearn.preprocessing import MinMaxScaler

   mm = MinMaxScaler().fit([[0.0], [10.0]])            # train range 0-10
   print(mm.transform([[5.0], [10.0], [15.0]]).ravel().tolist())
   # -> [0.5, 1.0, 1.5]
   ```

   **Answer.** 1.5 is outside $[0, 1]$ and that is correct, not a bug: both
   bounds came from the training set, so a larger test value maps above 1.
   Clipping it would hide the fact that production is outside the range you
   trained on.

3. Your revenue column contains zeros. What does `np.log(0)` return, and what
   does `np.log1p(0)` return?

   **Answer.** `np.log(0)` returns `-inf`, which fails silently in some
   estimators and loudly in others. `np.log1p(0)` returns `0.0`, because it
   computes $\log(1 + x)$ — which is why it is the transform to reach for on a
   count or a revenue column.
