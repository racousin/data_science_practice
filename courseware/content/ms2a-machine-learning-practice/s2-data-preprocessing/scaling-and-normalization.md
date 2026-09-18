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
| Linear / logistic regression, unregularised | low | coefficients absorb the scale |
| Ridge, Lasso, elastic net | **yes** | the penalty is applied to raw coefficients |
| k-NN, k-means, SVM with RBF | **yes** | they compute distances |
| Neural networks | **yes** | gradient magnitudes and initialisation assume it |
| PCA | **yes** | it maximises variance, which has units |

Scaling before a tree costs only time. Skipping it before a distance-based model
costs the model: a column in euros dominates a column in years by four orders of
magnitude, and "nearest" comes to mean "similar income".

---



## Scaling and k-NN

![k-NN regression error by scaler](assets/preprocessing/knn-mse-by-scaler.png)

k-NN regressor, 5-fold cross-validation, each scaler fitted on the training
folds only; L2 is the row-wise `Normalizer`.

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



## Skew is a different problem

Scaling moves a distribution; it does not reshape it. For a column with a long
right tail — revenue, population, counts — the reshaping tool is a transform:

```python
df["revenue_log"] = np.log1p(df["revenue"])
```

`log1p` computes $\log(1 + x)$, so a zero stays a zero. Plain `log` on a column
containing zeros produces `-inf`, which fails silently in some estimators and
loudly in others.


![0_--L5CABcACqMNTH0.png](assets/preprocessing/0_--L5CABcACqMNTH0.png)


---

## Column scalers 

![Population against MedInc under each scaler](assets/preprocessing/scalers-medinc-vs-population.png)

Min-Max (top right), Standard and MaxAbs (middle) keep the shape of the raw
cloud (top left) and change only the axes; the row-wise L2 `Normalizer`
(bottom) changes the shape.

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
