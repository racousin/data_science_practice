# Feature Selection

Feature selection is where you take back the columns that engineering added
and could not justify. Width is a cost you pay every run, and every selector
is a fitted step that belongs inside the Pipeline.

<!-- notes: 20 minutes. Most of this lesson is one sklearn call per method, so
it can move faster than the engineering half. Slow down twice: on selection
outside the pipeline as the second-largest leak of the course, and on PCA as a
projection that destroys the explanation. The ball-in-a-cube figure is the one
number to say out loud — a quarter of a percent in ten dimensions. End on the
loop; step 5 is the line to write on the wall. -->

---

## Why selection: the curse of dimensionality

Adding a column adds a dimension. In high dimensions volume grows faster than
any sample can fill it: points become equidistant, distance-based methods
degenerate, and the rows needed for the same density grow exponentially. The
practical symptoms arrive long before the theoretical ones:

- training slows in proportion to the width
- variance rises — the model fits noise in columns that carry none
- collinear columns split an effect between them and destabilise coefficients
- nobody can explain what the model uses

---

## Volume outruns the sample

![Ratio of the volume of a ball inscribed in a cube to the volume of the cube, on a logarithmic vertical axis, for dimensions 1 to 10: it starts at 1 and falls steadily to about 0.0025 at ten dimensions](assets/preprocessing/curse-of-dimensionality-ball-in-cube.png)

The ball inscribed in a cube holds 79% of the volume in 2D, 52% in 3D and
0.25% in 10D. Almost all of a high-dimensional box is corners — far from the
centre, and far from any row you sampled.

---

## Three families

| Family | How it decides | Cost | Blind spot |
|---|---|---|---|
| **Filter** | a statistic per feature against the target | one pass | interactions |
| **Wrapper** | train the model on candidate subsets | one fit per step | overfits on small data |
| **Embedded** | selection happens during fitting | nearly free | tied to that model |

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV

filt = SelectKBest(mutual_info_classif, k=20)
wrap = RFECV(estimator=LogisticRegression(), cv=5, scoring="roc_auc")
emb = LassoCV(cv=5)                     # L1 drives coefficients to exactly zero
```

Use `f_classif` / `f_regression` for linear association, `mutual_info_*` for any
dependence, `chi2` for non-negative counts.

All three are fitted steps and belong **inside** the pipeline. Selecting
features on the full dataset and cross-validating the survivors is the
second-largest leak in this course, and it buys several points of imaginary AUC.

---

## Selection methods disagree

![Feature scores from four selection methods](assets/preprocessing/feature-selection-methods-heatmap.png)

Breast-cancer data: the 15 of 30 features with the highest average score, each
method's scores min-max scaled.

---

## Correlation and VIF

```python
corr = X.corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print(pairs[pairs > 0.95])
```

Two columns correlated at 0.98 carry one piece of information and two
coefficients. Drop one — the cheaper to collect, or the easier to explain.

Pairwise correlation misses a column that is a linear combination of three
others. The variance inflation factor catches it, by regressing each feature on
all the rest:

$$
VIF_j = \frac{1}{1 - R_j^2}
$$

Above 10 is the usual alarm. It matters for linear models, where it makes
coefficients unstable and their signs arbitrary; trees are indifferent.

---

## Permutation importance

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=0)
```

Shuffle one column in the validation set, re-score, measure how far the score
fell. Model-agnostic, computed on held-out data, and it answers the question you
actually asked. Prefer it to `feature_importances_`, which is computed on the
training set and biased towards high-cardinality continuous columns.

---

## Correlation hides importance

![Box plot of permutation importances on the test set for the 30 breast-cancer features of a random forest: every feature's decrease in accuracy sits within about plus or minus 0.01 of zero, and most boxes are flat at zero](assets/preprocessing/permutation-importance-collinear-features.png)

Breast-cancer data, random forest at 97% test accuracy: shuffling any one of
the 30 features barely moves the score. Both importances mislead under
correlation — shuffling one of a correlated pair leaves the information
available through the other, and neither looks important.

---

## PCA, and when it is the wrong tool

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95).fit(X_tr_scaled)   # keep 95% of the variance
```

PCA rotates the data onto orthogonal axes ordered by variance and keeps the
first few. It needs scaled inputs — variance has units — and is fitted on train
like anything else. Right tool: decorrelating a wide block of measurements,
compressing before a distance-based method, visualising at two components.

Wrong tool when you need to know which columns matter, because every output
mixes every input; when the structure is non-linear; and when the signal has low
variance, which PCA discards first — being unsupervised, it has never seen the
target.

> Prefer selection to projection whenever anyone will ask why the model made a
> decision.

---

## What PCA keeps

![Scatter of a two-dimensional Gaussian cloud stretched along a diagonal, with two orthogonal arrows drawn from its centre: a long arrow along the direction of greatest spread and a short arrow across it](assets/preprocessing/pca-principal-axes-gaussian-scatter.png)

A Gaussian cloud and its two principal axes, each drawn at the length of its
standard deviation (3 and 1). PCA keeps the long axis; the short one is the
low-variance direction it discards first — whether or not the target lives
there.

---

## Explained variance

![PCA cumulative explained variance](assets/preprocessing/pca-cumulative-explained-variance.png)

Scaled breast-cancer features: ten components reach the 95% threshold.

---

## The loop

1. engineer features from domain knowledge, not from a library
2. put every one of them in the Pipeline
3. cross-validate: does the score move outside the noise?
4. drop what does not earn its place — width is a cost you pay every run
5. re-check the leak: could this value be computed at prediction time?

Step 5 is the one to write on the wall. A feature that will not exist at
inference — tomorrow's price, a field back-filled by an operator, an aggregate
over the full dataset — makes a model that validates beautifully and predicts
nothing.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.linear_model import LinearRegression

   rng = np.random.default_rng(0)
   a, b, c = rng.normal(size=(3, 200))
   d = a + b + c + rng.normal(scale=0.1, size=200)
   X = np.column_stack([a, b, c])
   r2 = LinearRegression().fit(X, d).score(X, d)
   print(round(1 / (1 - r2), 1))                            # -> 310.7
   print(abs(np.corrcoef([a, b, c, d]))[3, :3].round(2))  # -> [0.54 0.58 0.63]
   ```

   **Answer.** `d` is `a + b + c` plus noise. No pair is correlated above 0.63,
   so the 0.95 screen keeps all four columns; a VIF of 310 says `d` is almost
   entirely explained by the other three — the case pairwise correlation misses.

2. Name two situations in which PCA is the wrong tool, and say why.

   **Answer.** When you need to know which columns matter — every component mixes
   every input, so the explanation is gone. When the signal has low variance —
   PCA discards low-variance directions first, and being unsupervised it has
   never seen the target. (A third: when the structure is non-linear.)

3. `SelectKBest` is fitted on the full dataset, then the survivors are
   cross-validated. Which leak is this, and why is the score optimistic?

   **Answer.** The second-largest leak in this course: the selector is a fitted
   step, and fitted on every row it chose the survivors with the labels of the
   folds it is later scored on — several points of imaginary AUC. Inside the
   pipeline it is re-fitted on each training fold.
