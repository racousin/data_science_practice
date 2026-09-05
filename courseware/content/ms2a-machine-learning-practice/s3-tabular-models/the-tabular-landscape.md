# The Tabular Landscape

Deep learning took vision and text. It did not take tables. On a few thousand
rows of mixed numeric and categorical columns, a tree ensemble still wins — and
the models in this lesson are the ones it has to beat.

<!-- notes: 30 minutes. Ask the room who expects a neural network to win on their
project data; most say yes. Correct that early. Keep kNN and SVM short — they are
reference points, not candidates. Spend the time on the tree, because everything
in the next two lessons is built out of it. -->

---

## Deep learning did not take tables

Repeated benchmarks on medium-sized tabular data — Shwartz-Ziv and Armon (2022),
Grinsztajn et al. (2022) — reach the same conclusion: tree ensembles beat tuned
neural networks on most datasets, with a fraction of the tuning budget.

Three structural reasons:

- Columns have no translation invariance and no ordering. The inductive bias
  that makes convolution work has no analogue in a table.
- Features are heterogeneous in scale, type and meaning. A tree splits each on
  its own terms; a dense layer mixes them all in the first matmul.
- Real tables are small. Five thousand rows is normal, and that starves a
  network while being plenty for boosting.

---

## Linear regression

$$
\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j
$$

Fitted by minimising the residual sum of squares — in closed form when $p$ is
small, by gradient descent when it is not.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
```

One coefficient per feature, each readable as "one unit of $x_j$ moves the
prediction by $\beta_j$, all else equal". No other model gives that for free.

---

## Logistic regression

The same linear score, squashed to a probability and fitted by maximum
likelihood:

$$
P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}
$$

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_train)
```

`C` is the *inverse* regularisation strength, which trips up everyone once.
Small `C` means a strongly regularised model. Search it on a log scale from
$10^{-3}$ to $10^{3}$.

---

## The rule: fit the baseline first

> Before any boosting, fit a regularised linear or logistic model on the same
> split with the same metric, and write the number down.

It costs two minutes and buys three things: proof the pipeline runs end to end,
a floor any later model must clear, and an early warning. A linear model scoring
0.99 AUC on a hard problem is not a triumph — it is leakage, found for free.

---

## Regularization

Unpenalised least squares with correlated features gives huge coefficients that
cancel each other out. Penalising their size fixes it.

$$
\hat{\beta} = \arg\min_\beta \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda R(\beta)
$$

| Penalty | $R(\beta)$ | Effect |
|---|---|---|
| Ridge (L2) | sum of squared coefficients | shrinks together, keeps all features |
| Lasso (L1) | sum of absolute coefficients | drives coefficients to exactly zero |
| Elastic net | a mix of the two | sparse, but stable under collinearity |

---

## Which penalty

```python
from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5).fit(X_train, y_train)
```

Ridge when features are correlated and you want all of them, lasso when you want
the model to choose a subset, elastic net when both — which, on real data with
groups of correlated columns, is usually. Scale first: the penalty is on the
coefficient, and an unscaled coefficient carries whatever unit its column uses.

Lasso on correlated features picks one of the group arbitrarily and zeroes the
rest. Do not read that choice as a statement about importance.

---

## k-nearest neighbours

![k-nearest neighbours](assets/tabular/knn.png)

$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(X_tr, y_tr)
```

Small $k$ overfits, large $k$ underfits, and the method degrades above roughly
twenty dimensions, where every point is about equally far from every other. Keep
it as a sanity check, not a submission.

---

## Support vector machines

![Maximum-margin separator](assets/tabular/svm.png)

An SVM finds the separating hyperplane with the widest margin, allowing
violations at a cost `C`:

$$
\min_{w, b} \frac{1}{2} |w|^2 + C \sum_{i=1}^n \xi_i
$$

The kernel trick replaces every inner product with $K(x_i, x_j)$, giving a
non-linear boundary without ever building the high-dimensional features.

```python
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_train, y_train)
```

Training is quadratic to cubic in rows, so above roughly 50,000 it stops being
practical. That, not accuracy, is why it lost the tabular crown.

---

## The decision tree

![A decision tree](assets/tabular/tree.png)

A tree recursively splits the feature space on one column at a time, greedily
choosing the split that most reduces impurity in the children. Prediction walks
from the root to a leaf and returns the leaf's mean or majority class.

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20).fit(X_tr, y_tr)
```

No scaling, no encoding of ordinals, invariance to monotone transformations, and
a boundary you can print and read. That combination is why trees survive.

---

## Impurity

![Choosing a split by weighted Gini impurity](assets/tabular/gini.png)

Two measures of how mixed a node is, over class proportions $p_i$:

$$
G = 1 - \sum_{i=1}^C p_i^2, \qquad H = - \sum_{i=1}^C p_i \log_2 p_i
$$

The split chosen is the one with the largest drop from the parent's impurity to
the weighted average of its children's — the information gain.

Gini and entropy almost always pick the same split. Gini is cheaper. Use it and
stop tuning `criterion` — the depth parameters matter a hundred times more.

---

## Why one tree is not enough

Grown without limit, a tree splits until every leaf is pure — one training row
per leaf, training error zero, test error terrible.

The failure mode is **variance**: refit on a 90% resample and the top split can
change, taking the whole structure with it. A model whose shape depends on which
rows you happened to draw is not describing the population.

Pruning with `max_depth` and `min_samples_leaf` trades that variance for bias.
Averaging many trees, the next lesson, trades it for almost nothing.

---

## Where this leaves you

| Situation | Reach for |
|---|---|
| First model, always | regularised linear / logistic |
| Coefficients must be defended to a regulator | linear, and stop there |
| Fewer than ~1,000 rows | regularised linear, or a small SVM |
| Anything else, tabular | gradient boosting (next two lessons) |
| Fewer than 20 features, low dimension, quick check | kNN |

The rest of this session assumes gradient boosting and spends its time on what
decides whether yours is any good: the validation protocol and the search.

---

## Check yourself

1. Name two of the three structural reasons tree ensembles still beat tuned
   neural networks on medium-sized tables.

   **Answer.** Any two of: columns have no translation invariance and no
   ordering, so convolution's inductive bias has no analogue; features are
   heterogeneous in scale, type and meaning — a tree splits each on its own
   terms while a dense layer mixes them all in the first matmul; real tables
   are small, and five thousand rows starves a network while being plenty for
   boosting.

2. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.linear_model import LogisticRegression
   X = np.arange(20).reshape(-1, 1) / 10.0
   y = (X.ravel() > 1.0).astype(int)
   small = LogisticRegression(C=0.01).fit(X, y).coef_[0, 0]
   large = LogisticRegression(C=100).fit(X, y).coef_[0, 0]
   print(f"{small:.2f} {large:.2f}")        # -> 0.05 11.94
   ```

   `C` is the *inverse* regularisation strength: the small `C` is the strongly
   regularised model, and its coefficient is the one that got shrunk.

3. Your first model, a regularised logistic regression, scores 0.99 AUC on a
   problem everyone told you was hard. What is the reading?

   **Answer.** Leakage, found for free. That early warning is one of the three
   things fitting the baseline first buys you — the other two being proof the
   pipeline runs end to end and a floor any later model must clear.
