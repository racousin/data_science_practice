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


![image-68.png](assets/preprocessing/image-68.png)


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


![1_8bV8on4rNCsHv9vR0yivQA.png](assets/preprocessing/1_8bV8on4rNCsHv9vR0yivQA.png)

---

## PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95).fit(X_tr_scaled)   # keep 95% of the variance
```

PCA rotates the data onto orthogonal axes ordered by variance and keeps the
first few. It needs scaled inputs — variance has units — and is fitted on train
like anything else. Right tool: decorrelating a wide block of measurements,
compressing before a distance-based method, visualising at two components.

![Scatter of a two-dimensional Gaussian cloud stretched along a diagonal, with two orthogonal arrows drawn from its centre: a long arrow along the direction of greatest spread and a short arrow across it](assets/preprocessing/pca-principal-axes-gaussian-scatter.png)

---

## Explained variance

![PCA cumulative explained variance](assets/preprocessing/pca-cumulative-explained-variance.png)
