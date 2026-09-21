

## Voting and bagging

![Voting combines different models; bagging, one model on resampled data](assets/tabular/voting-vs-bagging.png)

Voting combines **different algorithms** on the **same data**; bagging combines
the **same algorithm** on **different bootstrap samples**.

---

## Voting: different model families

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

vote = VotingClassifier([
    ("lr", make_pipeline(StandardScaler(), LogisticRegression())),
    ("rf", RandomForestClassifier()),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier())),
    ("svm", make_pipeline(StandardScaler(),
                          CalibratedClassifierCV(SVC(), ensemble=False))),
], voting="soft")
vote.fit(X_train, y_train)
y_pred = vote.predict(X_test)
```

Diversity here comes from genuinely different algorithms. The scale-sensitive
members carry their own `StandardScaler`; the forest needs none.

---

## Soft or hard voting

- `voting="soft"` averages the predicted probabilities. It keeps the confidence
  information that a majority vote discards. Every member must expose
  `predict_proba` and be roughly calibrated: wrap a plain `SVC` in
  `CalibratedClassifierCV(SVC(), ensemble=False)`. `SVC(probability=True)`
  does the same job and is deprecated from scikit-learn 1.9 (*Probabilistic
  Prediction*).
- `voting="hard"` takes the majority of the predicted labels. It is the only
  option when a member has no `predict_proba`, such as a plain `SVC`.

Use soft voting unless a member forces hard.

---

## Voting for regression

```python
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

model = VotingRegressor(estimators=[
    ("svr", make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0))),
    ("tree", DecisionTreeRegressor(max_depth=4)),
    ("knn", make_pipeline(StandardScaler(),
                          KNeighborsRegressor(n_neighbors=5))),
])
model.fit(X_train, y_train)
```

`VotingRegressor` averages the members' predictions; `weights=` makes the
average weighted. There is no hard or soft choice for regression.

---

## Bagging in three steps

Diversity from **resampling the data** instead:

1. Draw $B$ bootstrap samples: random sampling **with replacement**
2. Train one model on each sample
3. Aggregate: majority vote (classification) or average (regression)

![Three bootstrap samples drawn from one original sample](assets/tabular/bootstrap-samples.png)

---

## Bootstrap aggregating

Draw $n$ rows with replacement from $n$. A given row is never drawn with
probability

$$
\left(1 - \frac{1}{n}\right)^n \to e^{-1} \approx 0.368
$$

so each tree sees a different 63.2% of the distinct rows.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=200, n_jobs=-1)
```

---

## Bagging fixes variance, not bias

Averaging $B$ independent estimators divides their variance by $B$ and leaves
the bias alone.

> Bagging **reduces variance**. It stabilises unstable models, which is exactly
> the weakness of a single deep tree.

That is why the base learner should be deliberately overfit. Bagging does
essentially nothing for bias: averaging many underfit models gives you one
underfit model.

---

## Random forests

Bagging, plus random feature selection at each split:

- At each node, consider only $m$ random features
- Classification: $m = \sqrt{p}$; regression: $m = p/3$
- This **decorrelates** the trees: more diversity, a better ensemble

![Each tree gets its own data sample and feature sample](assets/tabular/random-forest.png)

The figure draws one feature sample per tree; scikit-learn draws a fresh one
at **every split**, which decorrelates the trees further.

---

## Why decorrelation matters

![A random forest averages the predictions of all its trees](assets/tabular/rf.png)

Bootstrap samples alone leave the trees correlated: a dominant feature is the top
split in nearly all of them. The variance of the average is

$$
Var\Big(\frac{1}{B}\sum_{b=1}^B f_b(x)\Big) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2
$$

---

## The floor is the correlation

The second term vanishes with more trees; the first does not, being floored by
the correlation $\rho$. Random forests attack $\rho$ directly, considering only a
random subset of features at **each split**.

Without the feature subsampling, every bootstrap tree would pick the same
dominant feature at the root and the trees would be near-copies. The second
source of randomness is what makes a forest more than bagged trees.

---

## Random forest in practice

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500, max_features="sqrt", min_samples_leaf=2,
    n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

- `n_estimators`: more is never worse, only slower; 300–500 is a plateau.
- `max_features`: the correlation knob. `"sqrt"` for classification, `p/3` for
  regression. Lower means more decorrelation and more bias per tree.
- `min_samples_leaf`: the only depth control worth touching.

A forest with defaults is the strongest model you can produce without thinking:
an excellent second baseline and a poor final answer.

---

## Regression forests and importances

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=500, max_features=1/3, min_samples_leaf=2,
    n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
```

Set `max_features` yourself for regression: the default, `1.0`, uses every
feature at every split, which makes the forest plain bagged trees.

`feature_importances_` recovers part of the interpretability you lost when you
stopped using a single readable tree. It is the impurity decrease per feature,
measured on the training data: the next lesson's warning on importance applies.
