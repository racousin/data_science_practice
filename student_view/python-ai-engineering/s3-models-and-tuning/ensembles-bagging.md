# Ensembles: Bagging

Every model so far has one weakness. Combining many of them cancels part of it —
and the version that averages *unstable* models is the best general-purpose
tabular model you will meet before boosting.

<!-- notes: ~15 of the 30 minutes budgeted for ensembles. The bias/variance
framing is what makes bagging and boosting distinct rather than two names for
"use several models". -->

---

## Why ensembles

![Wisdom of the crowd](/api/academic_courses/assets/lessons/160/voting-vs-bagging.png)

- Single models have weaknesses — either bias or variance
- Idea: combine several models for better performance
- "Wisdom of the crowd" — aggregate diverse opinions

The word doing the work is **diverse**. Averaging ten copies of the same model
gains nothing; averaging ten models that fail on different rows gains a lot.

---

## Voting: different model families

```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

model = VotingClassifier(estimators=[
    ('svm',  SVC(kernel='rbf', C=1.0)),
    ('tree', DecisionTreeClassifier(max_depth=4)),
    ('knn',  KNeighborsClassifier(n_neighbors=5)),
], voting='hard')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

```python
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

model = VotingRegressor(estimators=[
    ('svr',  SVR(kernel='rbf', C=1.0)),
    ('tree', DecisionTreeRegressor(max_depth=4)),
    ('knn',  KNeighborsRegressor(n_neighbors=5)),
])
```

Diversity here comes from using genuinely different algorithms.

---

## Bagging (bootstrap aggregating)

Diversity from **resampling the data** instead:

1. Create $B$ bootstrap samples — random sampling **with replacement**
2. Train one model on each sample
3. Aggregate — majority vote (classification) or average (regression)

![Bootstrap samples](/api/academic_courses/assets/lessons/160/bootstrap-samples.png)

> Bagging **reduces variance**. It stabilises unstable models — which is exactly
> the weakness in the decision-tree cons table.

It does essentially nothing for bias. Averaging many underfit models gives you
one underfit model.

---

## Random Forest

Bagging, plus random feature selection at each split:

- At each node, consider only $m$ random features
- Classification: $m = \sqrt{p}$   |   Regression: $m = p/3$
- This **decorrelates** the trees → more diversity → a better ensemble

![Random forest](/api/academic_courses/assets/lessons/160/random-forest.png)

Without the feature subsampling, every bootstrap tree would pick the same
dominant feature at the root and the trees would be near-copies. The second
source of randomness is what makes a forest more than bagged trees.

---

## In scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
importances = rf.feature_importances_
```

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                           max_features='sqrt', random_state=42)
```

`feature_importances_` recovers part of the interpretability that was lost when
you stopped using a single readable tree.
