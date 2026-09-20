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

![Anatomy of a tree: root, decision nodes, leaves, sub-trees](assets/tabular/tree-anatomy.png)

Prediction walks from the root to a leaf and returns the leaf's majority class
or mean.

---

## A partition and the tree that produces it

![A partition of the plane and the tree that produces it](assets/tabular/tree-decision-regions.png)

Every leaf is an axis-aligned rectangle, so the boundary is a staircase. At each
node the goal is the same: find the feature $j$ and threshold $t$ that best
separate the labels.

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

## Information gain

The split chosen is the one with the largest drop from the parent's impurity
$I$ to the weighted average of its children's:

$$
\Delta I = I(D) - \frac{n_{left}}{n} I(D_{left}) - \frac{n_{right}}{n} I(D_{right})
$$

With entropy as $I$, that drop is the **information gain**; with Gini it is the
Gini decrease, which is what `criterion="gini"` maximises.

Gini and entropy almost always pick the same split. Gini is cheaper, having no
logarithm. Use it and stop tuning `criterion`: the depth parameters matter a
hundred times more.

---

## Choosing the split: a worked example

![Weighted Gini impurity for four candidate splits](assets/tabular/gini-by-feature.png)

14 rows, 9 yes and 5 no: the parent's Gini is 0.459. Age leaves the purest
children, a weighted 0.343, so it wins with a gain of 0.116. Entropy ranks the
four features in the same order.

---

## The same example, binary

The worked example splits Age three ways, as ID3 and C4.5 do. CART, the
algorithm behind scikit-learn's trees, makes binary splits only, so it tests
*middle age* against the rest:

$$
\frac{4}{14} \times 0 + \frac{10}{14} \times 0.5 = 0.357
$$

Still the best split on the table: Student, the runner-up, scores 0.367. A
many-valued feature reaches its levels over several binary splits instead of
one.

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

## The regression tree

![A regression tree on car prices](assets/tabular/tree-regression-example.png)

Same algorithm, different impurity: each split minimises the squared error of
the children around their means (`criterion="squared_error"`), and each leaf
predicts the **mean** of its training rows, 23,000 or 50,000 here.

A leaf returns the mean of its training rows, so a tree never predicts outside
the range of the training target: it cannot extrapolate.

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

---

## Why one tree is not enough

Grown without limit, a tree splits until every leaf is pure — down to single
training rows where the labels force it — training error zero, test error
terrible.

The failure mode is **variance**: refit on a 90% resample and the top split can
change, taking the whole structure with it. A model whose shape depends on which
rows you happened to draw is not describing the population.

Pruning with `max_depth` and `min_samples_leaf` trades that variance for bias.
Averaging many trees, the rest of this lesson, trades it for almost nothing.

---

## Where the error comes from

For squared loss, the expected test error of a model $\hat{f}$ at a point
decomposes into three terms:

$$
\mathbb{E}[(y - \hat{f}(x))^2] = (\mathbb{E}[\hat{f}(x)] - f(x))^2 + Var(\hat{f}(x)) + \sigma^2
$$

Bias is how wrong the average model is; variance is how much it moves when the
training sample changes; noise is irreducible. A deep tree is the low-bias,
high-variance extreme and a linear model is the opposite. Which of the two you
attack determines which ensemble you build.

---

## Why ensembles

- Single models have a weakness: bias or variance
- Idea: combine several models for better performance
- "Wisdom of the crowd": aggregate diverse opinions

The word doing the work is **diverse**. Averaging ten copies of the same model
gains nothing; averaging ten models that fail on different rows gains a lot.

Diversity comes from different algorithms (voting), different rows (bagging),
different features (random forests) or different targets (boosting).

---

## Two strategies

![Bagging and boosting, built from the same decision tree](assets/tabular/bagging-vs-boosting.png)

Same building block, opposite failure mode addressed. **Bagging** averages deep
trees trained in parallel to cut variance; **boosting** chains shallow trees
trained in sequence to cut bias. The full comparison closes the boosting part.

---

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

---

## Out-of-bag error

Each tree ignored about 37% of the rows; predicting each row with only the trees
that did not see it gives a validation estimate for free.

```python
rf = RandomForestClassifier(n_estimators=500, oob_score=True).fit(X, y)
print(rf.oob_score_)
```

OOB is roughly a leave-one-out estimate for the cost of one fit. It is no
substitute for the protocol of *Model Selection and Validation*: it says nothing
about a preprocessing step fitted outside it, and the boosters you will use
offer no such shortcut (scikit-learn's `GradientBoosting*` does, only with
`subsample < 1`); a booster needs a held-out set for early stopping anyway.

---

## Boosting: fix what the last model got wrong

- Train models **sequentially**, each correcting the previous one's errors
- Focus on the hard examples: misclassified points, or large residuals
- Reduce **bias**: make weak learners strong

![Weak learners combined into a strong one](assets/tabular/boosting-weak-learners.png)

The mirror image of bagging: same building block, opposite failure mode.

---

## Weak on purpose

Bagging builds its members in ignorance of each other. Boosting builds them in
sequence, each focused on what the current ensemble handles badly.

The base learner is deliberately **weak** — a stump, or a depth-3 tree. It
underfits alone; the sequence corrects the bias, and shallowness keeps each step
small. A boosted model can therefore overfit by adding trees, which makes the
number of trees a hyperparameter you must validate. A forest is not.

---

## AdaBoost

![AdaBoost reweighting: each round focuses on the last round's mistakes](assets/tabular/adaboost.jpg)

Each round trains a weak learner on weighted data, then upweights the examples
it got wrong. The final model is a weighted vote of all the rounds.

---

## AdaBoost, step by step

Labels and learner outputs are coded $\pm 1$: $y_i, h_t(x_i) \in \{-1, +1\}$.

**Step 1 — initialise** uniform sample weights:

$$
w_i = \frac{1}{n} \qquad \forall i \in 1, \ldots, n
$$

**Step 2 — for each round** $t = 1, \ldots, T$:

**2a.** Train a weak learner $h_t$ on the weighted data.

---

## AdaBoost, the error and the vote weight

**2b.** Compute the weighted error:

$$
\epsilon_t = \sum_{i=1}^{n} w_i \, \mathbb{1}\big(h_t(x_i) \neq y_i\big)
$$

**2c.** Compute the learner's weight; a good learner gets a high $\alpha_t$:

$$
\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}
$$

---

## AdaBoost, the weight update

**2d.** Update the sample weights, so that misclassified points get heavier, then
normalise:

$$
w_i \leftarrow w_i \, e^{-\alpha_t y_i h_t(x_i)}
\qquad\text{then}\qquad
w_i \leftarrow \frac{w_i}{\sum_j w_j}
$$

$y_i h_t(x_i)$ is $+1$ on a correct row and $-1$ on a wrong one, so:

$$
w_i \leftarrow w_i \, e^{-\alpha_t} \quad \mathrm{if\ correct}, \qquad w_i \leftarrow w_i \, e^{\alpha_t} \quad \mathrm{if\ wrong}
$$

After the update, $h_t$ scores exactly 50% on the new weights, so the next
learner has to find something new.

---

## AdaBoost, the final vote

**Step 3 — final prediction**, a weighted vote:

$$
H(x) = \text{sign}\Big( \sum_{t=1}^{T} \alpha_t h_t(x) \Big)
$$

A learner with weighted error $\epsilon_t$ votes with weight $\alpha_t$. The
exponential loss behind the update makes AdaBoost brittle under label noise: a
mislabelled row is upweighted forever.

---

## The same update, with an indicator

Many texts, and scikit-learn, write the update with an indicator instead. With
$\pm 1$ labels,

$$
1 - y_i h_t(x_i) = 2 \cdot \mathbb{1}\big(h_t(x_i) \neq y_i\big)
$$

so the update above factors as

$$
w_i \, e^{-\alpha_t y_i h_t(x_i)} = e^{-\alpha_t} \, w_i \, \exp\big(2\alpha_t \, \mathbb{1}(h_t(x_i) \neq y_i)\big)
$$

The first factor is the same for every row and cancels in the normalisation.

---

## Twice the α

So the indicator form, which leaves correct rows alone and scales only the
wrong ones up, is the same algorithm **only if its α is twice this one**, the
½ dropped:

$$
w_i \leftarrow w_i \exp\big(\alpha^\prime_t \, \mathbb{1}(h_t(x_i) \neq y_i)\big), \qquad \alpha^\prime_t = \ln \frac{1 - \epsilon_t}{\epsilon_t} = 2\alpha_t
$$

Doubling every $\alpha_t$ leaves the sign of the vote unchanged, so both forms
predict the same. Mixing them, the ½ with the indicator, is a common slip: it
scales wrong rows by

$$
\sqrt{\frac{1 - \epsilon_t}{\epsilon_t}} \quad \mathrm{instead\ of} \quad \frac{1 - \epsilon_t}{\epsilon_t}
$$

---

## Reading $\alpha_t$

| $\epsilon_t$ | $\alpha_t$ | Effect |
|---|---|---|
| low, 0.1 | high, 1.10 | strong vote |
| high, 0.4 | low, 0.20 | weak vote |
| 0.5, random | 0 | **ignored entirely** |
| above 0.5, e.g. 0.7 | negative, −0.42 | used inverted |

A learner that is no better than a coin gets exactly zero weight, because
$\ln(1) = 0$: the formula throws away useless models for free. One worse than
chance gets negative weight and is used inverted. scikit-learn instead discards
any learner at or below chance and stops boosting.

---

## AdaBoost in scikit-learn

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    learning_rate=0.5)
```

The stump is the default base learner, written out here so you can see it.
`learning_rate` multiplies every $\alpha_t$: below 1 it shrinks each step and
needs more rounds. The `algorithm` argument is gone as of scikit-learn 1.8;
only the discrete SAMME update above remains.

---

## Gradient boosting

Generalise: instead of reweighting, fit each new tree to the **negative gradient
of the loss** at the current predictions. For squared loss that gradient is the
residual, so each tree predicts what the ensemble still gets wrong.

$$
r_i^{(m)} = - \left[ \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}
$$

$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$

The shrinkage $\nu$ scales each correction down. The libraries call it
`learning_rate` — the same name the MLP lesson's step size $\eta$ goes by, a
different quantity. Any differentiable loss works, which is why one algorithm
covers regression, classification and ranking.

---

## Gradient boosting's three dials

| Parameter | Name | Role |
|---|---|---|
| `n_estimators` ($T$) | boosting rounds | how many corrections are added |
| `learning_rate` ($\nu$) | shrinkage | how much of each correction is kept |
| `max_depth` | tree depth | how much each tree can correct: shallow, 4–8 |

`max_depth` stays shallow because boosting wants *weak* learners: a deep tree
already has low bias and leaves the ensemble nothing to correct. How $\nu$ and
$T$ trade against each other, and why early stopping picks $T$, is the next
lesson.

---

## Gradient boosting in code

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
gb.fit(X_train, y_train)
```

```python
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
```

---

## Which implementation

`GradientBoostingClassifier` and `GradientBoostingRegressor` are the textbook
implementation: exact splits, one core, slow beyond about 10,000 rows.
`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are
scikit-learn's fast, binned version of the same algorithm.

In practice you fit XGBoost, LightGBM or CatBoost. The next lesson compares the
three and covers the parameters that matter.

---

## Bagging versus boosting

| | Bagging | Boosting |
|---|---|---|
| Attacks | **variance** | **bias** |
| Base learner | deep, low-bias, overfitting trees | shallow, weak, underfitting trees |
| Training | parallel, independent | sequential, each on the last's error |
| Diversity from | bootstrap rows (+ random features) | reweighting or residuals |
| More trees | plateaus: rarely overfits | too many rounds overfits |
| Parallelisable | trivially | not across trees |
| Tuning sensitivity | low | high |
| Examples | random forest | AdaBoost, XGBoost, LightGBM, CatBoost |

The "more trees" row is the practical one: adding trees to a random forest is
close to free, adding rounds to a booster is not. Bagging is what you run in
twenty minutes; boosting is what you run to win.

---

## Blending and stacking

![Stacking](assets/tabular/stacking.jpg)

**Blending** splits the training data once: base models fit on one part, predict
the rest, and a meta-model trains on those predictions. Simple, fast, wasteful.

**Stacking** does the same with k-fold, so every row gets an **out-of-fold**
prediction from a model that never saw it. The meta-model trains on those.

```python
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
stack = StackingClassifier(
    [("gb", LGBMClassifier()), ("rf", RandomForestClassifier())],
    final_estimator=LogisticRegression(), cv=5)
```

Keep the meta-model boring — regularised logistic or ridge. A gradient-boosted
meta-model on five correlated columns overfits the out-of-fold predictions and
throws away the gain you just bought.

---

## The failure mode: leaking into the meta-model

> If a base model ever predicts a row it was trained on, its prediction on that
> row is too good, and the meta-model learns to trust it. Validation looks
> excellent; the leaderboard does not.

This is the same mistake as fitting a scaler before splitting, one level up. Use
`StackingClassifier` rather than assembling it by hand; if you build it yourself,
assert that no base model saw the rows it predicted.

Expect one to three percent, after everything else is done. A tuned single
gradient-boosting model gets you most of the way.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   def gini(*counts):
       n = sum(counts)
       return 1 - sum((c / n) ** 2 for c in counts)
   parent = gini(9, 5)
   age = (5 * gini(2, 3) + 4 * gini(4, 0) + 5 * gini(3, 2)) / 14
   print(f"{parent:.3f} {age:.3f} {parent - age:.3f}")
   # -> 0.459 0.343 0.116
   ```

   The parent's impurity, the Age split's, and the drop that makes Age win.

2. Bagging and boosting attack different terms of the error decomposition.
   Which is which, and what does each imply about the base learner?

   **Answer.** Bagging attacks variance, so its base learner is deliberately
   overfit: averaging $B$ estimators divides variance by $B$ and leaves bias
   alone. Boosting attacks bias, so its base learner is deliberately weak: a
   stump, or a depth-3 tree.

3. Bootstrapping already gives every tree different rows. Why does a random
   forest also subsample *features* at each split?

   **Answer.** The variance of the average is
   $\rho\sigma^2 + (1-\rho)\sigma^2/B$. More trees only shrink the
   second term, so the correlation $\rho$ between trees is the floor. Feature
   subsampling attacks $\rho$ directly.
