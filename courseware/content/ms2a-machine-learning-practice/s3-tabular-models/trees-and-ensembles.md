# Trees and Ensembles

A decision tree is a model you can read out loud: weak on its own, and the
building block of the two best tabular models in the course. One tree has low
bias and ruinous variance. Every ensemble method is a different answer to the
same question: how do you keep the flexibility and throw away the variance?

<!-- notes: About 65 minutes, the theoretical core of the session. Trees first,
about 20 minutes: do the worked Gini example on the board rather than reading
the slide, and land "weak alone, powerful in ensembles", because it sets up
everything after it. Ensembles second, about 45 minutes: draw the bias/variance
decomposition on the board before showing the slide. The correlation term in
the random-forest variance is the single idea that explains why feature
subsampling exists; do not skip it. Work AdaBoost through step by step, because
it is the one where the weight update is legible; gradient boosting is what
they will actually use. -->

---

## The decision tree

- Recursive **binary partitioning** of the feature space
- Each internal node asks a question about one feature: *is $x_1 > 3.5$?*
- Each leaf holds a prediction: a class label or a value

![A partition of the plane and the tree that produces it](assets/tabular/tree-decision-regions.png)

Prediction walks from the root to a leaf and returns the leaf's majority class
or mean.

---

## The same geometry in a k-d tree

![A k-d tree: a partition built on the points themselves](assets/tabular/tree-partition-and-structure.png)

A k-d tree cuts the plane the same way, one axis per node, but to **balance the
cells**, not to separate labels. The textbook version, drawn here, alternates
axes and splits at the median point; scikit-learn's `KDTree`, the index behind
`KNeighborsClassifier(algorithm="kd_tree")`, splits the widest dimension
instead. The circle around Q is a nearest-neighbour query. A decision tree
splits where the **labels** separate best. Same shape, different criterion.

---

## Impurity: how mixed is a node

Two measures over the class proportions $p_k$, the share of the node's samples
that belong to class $k$:

$$
G = 1 - \sum_{k=1}^{K} p_k^2, \qquad H = - \sum_{k=1}^{K} p_k \log_2 p_k
$$

| Node | Gini $G$ | Entropy $H$ |
|---|---|---|
| pure: one class | 0 | 0 |
| 50/50, two classes | 0.5, the maximum | 1 bit, the maximum |

A leaf predicts its majority class:

$$
\hat{y} = \arg\max_k p_k
$$

---


## The split algorithm

For each node, test every candidate split $(j, t)$: each feature
$j \in 1, \ldots, p$, and each unique value $t$ of that feature.

$$
D_{left} = \{x_i : x_i^{(j)} \leq t\} \qquad D_{right} = \{x_i : x_i^{(j)} > t\}
$$

$$
G_{split}(j, t) = \frac{n_{left}}{n} G_{left} + \frac{n_{right}}{n} G_{right}
$$

$$
(j^*, t^*) = \arg\min_{j, t} \; G_{split}(j, t)
$$

The parent's impurity is fixed, so the lowest $G_{split}$ is the largest gain.
Then recurse into both children. Exhaustive, greedy, and locally optimal: it
never revisits an earlier split.

---

## Controlling complexity

Trees grow until their leaves are pure, which is overfitting by construction:
the same interpolation problem as the degree-$(n-1)$ polynomial of *The Tabular
Landscape*, which *Model Selection and Validation* dissects.

| Parameter | Effect |
|---|---|
| `max_depth` | maximum tree depth |
| `min_samples_split` | minimum samples required to split a node |
| `min_samples_leaf` | minimum samples required in a leaf |
| `max_features` | number of features considered per split |

Set `max_depth` and `min_samples_leaf` first.

---

## Depth on a regression

![A regression tree at depth 2 and at depth 5](assets/tabular/tree.png)

A regression tree predicts one constant per leaf, so its curve is a staircase.
At depth 2 it has at most four steps and misses the shape; at depth 5 it spends
leaves on single noisy points. Depth is the bias–variance dial.

---


## In scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
plot_tree(tree, filled=True)
```

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
```

`sklearn.tree.export_text(tree)` prints the same rules as indented text.

---

## No scaling needed

A threshold test $x_j \leq t$ is invariant to the units of $x_j$, and to any
monotone transformation of it. Trees are the one family in this session that
needs no `StandardScaler`, and every ensemble built from trees inherits that.

No scaling, no encoding of ordinals, invariance to monotone transformations, and
a boundary you can print and read. That combination is why trees survive.
