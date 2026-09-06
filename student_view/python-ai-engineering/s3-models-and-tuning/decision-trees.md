# Decision Trees

A model you can read out loud. Weak on its own, and the building block of the
two best tabular models in the course.

<!-- notes: ~10 minutes. The worked Gini example is the core; do it on the board
rather than reading the slide. Land "weak alone, powerful in ensembles" — it sets
up the next two lessons. -->

---

## The idea

- Recursive **binary partitioning** of the feature space
- Each internal node = a question about one feature (*is $x_1 > 3.5$?*)
- Each leaf = a prediction (a class label or a value)

![Anatomy of a tree](/api/academic_courses/assets/lessons/158/tree-anatomy.png)

![A partition and the tree that produces it](/api/academic_courses/assets/lessons/158/tree-partition-and-structure.png)

Goal: find the feature $j$ and threshold $t$ that best separate the labels.

![Decision regions](/api/academic_courses/assets/lessons/158/tree-decision-regions.png)

---

## Splitting for classification: Gini impurity

Minimise the Gini impurity within each child node:

$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$

- $G = 0$ → pure node (all the same class)
- $G = 0.5$ → maximum impurity (binary case)

$$
\text{Prediction:} \quad \hat{y} = \arg\max_k p_k \quad \text{(majority class in the leaf)}
$$

![Gini impurity by candidate feature](/api/academic_courses/assets/lessons/158/gini-by-feature.png)

---

## Splitting for regression: MSE

Minimise the MSE within each child node:

$$
\text{Prediction:} \quad \hat{y} = \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i \quad \text{(mean of the leaf)}
$$

![A regression tree](/api/academic_courses/assets/lessons/158/tree-regression-example.png)

---

## The split algorithm

For each node, test all possible splits $(j, t)$:

$$
\text{For each feature } j \in 1, \ldots, p
$$

$$
\text{For each unique value } t \text{ of feature } j
$$

$$
D_{left} = \{x_i : x_i^{(j)} \leq t\}
\qquad
D_{right} = \{x_i : x_i^{(j)} > t\}
$$

$$
G_{split}(j, t) = \frac{n_{left}}{n} G_{left} + \frac{n_{right}}{n} G_{right}
$$

$$
(j^*, t^*) = \arg\min_{j, t} \; G_{split}(j, t)
$$

Exhaustive, greedy, and locally optimal — it never revisits an earlier split.

---

## Worked example

| Student | Hours studied | Sleep | Result |
|---|---|---|---|
| 1 | 2 | 8 | Fail |
| 2 | 3 | 6 | Fail |
| 3 | 5 | 7 | Pass |
| 4 | 6 | 5 | Pass |
| 5 | 7 | 8 | Pass |
| 6 | 8 | 4 | Pass |

At the root: 4 Pass, 2 Fail, so

$$
p_{pass} = \frac{4}{6}, \quad p_{fail} = \frac{2}{6}
$$

### Candidate split $x_1 \leq 5$

Left (students 1, 2, 3): 1 Pass, 2 Fail; right (4, 5, 6): 3 Pass, 0 Fail.

$$
G_{left} = 1 - \left(\frac{1}{3}\right)^2 - \left(\frac{2}{3}\right)^2 = 0.44
\qquad
G_{right} = 0
$$

$$
G_{split} = \frac{3}{6} \times 0.44 + \frac{3}{6} \times 0 = 0.22
$$

### Candidate split $x_1 \leq 3$

Left (students 1, 2): 0 Pass, 2 Fail — **pure**; right (3, 4, 5, 6): 4 Pass, 0
Fail — **pure**.

$$
G_{left} = 1 - 0^2 - 1^2 = 0
\qquad
G_{right} = 1 - 1^2 - 0^2 = 0
$$

$$
G_{split} = \frac{2}{6} \times 0 + \frac{4}{6} \times 0 = 0
$$

**Conclusion:** $x_1 \leq 3$ gives lower impurity, so the algorithm picks it. In
practice it tests every threshold for every feature and keeps the best.

---

## Controlling complexity

Trees grow until their leaves are pure — which is overfitting by construction,
the same interpolation problem as the degree-$(n-1)$ polynomial.

| Parameter | Effect |
|---|---|
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required in a leaf |
| `max_features` | Number of features considered per split |

---

## In scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
plot_tree(tree, filled=True)
```

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=4)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
```

No scaling anywhere. A threshold test $x_j \leq t$ is invariant to the units of
$x_j$, which is why trees are the one family in this session that does not need
`StandardScaler`.

---

## Pros and cons

| Pros | Cons |
|---|---|
| Interpretable (white-box) | High variance — unstable |
| No feature scaling needed | Axis-aligned splits only |
| Handles mixed feature types | Overfits easily without pruning |
| Fast training and prediction | **Weak alone — powerful in ensembles** |

That last row is the next two lessons.
