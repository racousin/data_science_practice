# Trees and Ensembles

One tree has low bias and ruinous variance. Every ensemble method is a different
answer to the same question: how do you keep the flexibility and throw away the
variance?

<!-- notes: 40 minutes, the theoretical core of the session. Draw the
bias/variance decomposition on the board before showing the slide. The
correlation term in the random-forest variance is the single idea that explains
why feature subsampling exists — do not skip it. Boosting gets the second half.
-->

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

## Two strategies

| | Bagging | Boosting |
|---|---|---|
| Attacks | variance | bias |
| Base learner | deep, overfitting trees | shallow, underfitting trees |
| Trained | in parallel, independently | sequentially, each on the last's error |
| Effect of more trees | plateaus, does not overfit | eventually overfits |
| Parallelisable | trivially | not across trees |
| Tuning sensitivity | low | high |

Bagging is what you run in twenty minutes; boosting is what you run to win.

---

## Voting and bagging

![Voting versus bagging](assets/tabular/votbag.png)

Voting combines **different algorithms** on the **same data**; bagging combines
the **same algorithm** on **different bootstrap samples**.

```python
from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(
    [("lr", LogisticRegression()), ("rf", RandomForestClassifier()),
     ("knn", KNeighborsClassifier())], voting="soft")
```

Use `voting="soft"`: averaging probabilities keeps confidence information that a
majority vote discards. Every member must expose `predict_proba` and be roughly
calibrated.

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
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, n_jobs=-1)
```

Averaging $B$ independent estimators divides their variance by $B$ and leaves
the bias alone. That is why the base learner should be deliberately overfit:
bagging can fix variance, not bias.

---

## Random forests

![Random forest](assets/tabular/rf.png)

Bootstrap samples alone leave the trees correlated: a dominant feature is the top
split in nearly all of them. The variance of the average is

$$
Var\left(\frac{1}{B}\sum_{b=1}^B f_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2
$$

The second term vanishes with more trees; the first does not, being floored by
the correlation $\rho$. Random forests attack $\rho$ directly, considering only a
random subset of features at **each split**.

---

## Random forest in practice

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=-1)
```

- `n_estimators` — more is never worse, only slower; 300–500 is a plateau.
- `max_features` — the correlation knob. `"sqrt"` for classification, `p/3` for
  regression. Lower means more decorrelation and more bias per tree.
- `min_samples_leaf` — the only depth control worth touching.

A forest with defaults is the strongest model you can produce without thinking:
an excellent second baseline and a poor final answer.

---

## Out-of-bag error

Each tree ignored about 37% of the rows; predicting each row with only the trees
that did not see it gives a validation estimate for free.

```python
rf = RandomForestClassifier(n_estimators=500, oob_score=True).fit(X, y)
print(rf.oob_score_)
```

OOB is roughly a leave-one-out estimate for the cost of one fit. It is no
substitute for the next lesson's protocol: it says nothing about a preprocessing
step fitted outside it, and boosting has no equivalent.

---

## Boosting: fix what the last model got wrong

Bagging builds its members in ignorance of each other. Boosting builds them in
sequence, each focused on what the current ensemble handles badly.

The base learner is deliberately **weak** — a stump, or a depth-3 tree. It
underfits alone; the sequence corrects the bias, and shallowness keeps each step
small. A boosted model can therefore overfit by adding trees, which makes the
number of trees a hyperparameter you must validate. A forest is not.

---

## AdaBoost

![AdaBoost reweighting](assets/tabular/adaboost.jpg)

Each round trains a weak learner on weighted data, then upweights the examples
it got wrong. A learner with weighted error $\epsilon_t$ receives weight

$$
\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}
$$

and the sample weights update as $w_i \leftarrow w_i e^{-\alpha_t y_i h_t(x_i)}$.

```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5)
```

A learner at chance ($\epsilon_t = 0.5$) gets weight zero; one worse than chance
gets negative weight and is used inverted. The exponential loss makes AdaBoost
brittle under label noise — a mislabelled row is upweighted forever.

---

## Gradient boosting

Generalise: instead of reweighting, fit each new tree to the **negative gradient
of the loss** at the current predictions. For squared loss that gradient is the
residual, so each tree predicts what the ensemble still gets wrong.

$$
r_i^{(m)} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}
$$

$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$

The shrinkage $\nu$ scales each correction down: small $\nu$ needs more trees and
generalises better. Any differentiable loss works, which is why one algorithm
covers regression, classification and ranking.

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

1. Bagging and boosting attack different terms of the error decomposition.
   Which is which, and what does each imply about the base learner?

   **Answer.** Bagging attacks variance, so its base learner is deliberately
   overfit — averaging $B$ estimators divides variance by $B$ and leaves bias
   alone. Boosting attacks bias, so its base learner is deliberately weak: a
   stump, or a depth-3 tree.

2. Run this. You should get exactly the output shown.

   ```python
   n = 10000
   print(round(1 - (1 - 1/n) ** n, 3))       # -> 0.632
   ```

   Each bootstrap tree sees about 63.2% of the distinct rows, which is also
   why the other 37% can be reused as an out-of-bag validation estimate.

3. Bootstrapping already gives every tree different rows. Why does a random
   forest also subsample *features* at each split?

   **Answer.** The variance of the average is
   $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. More trees only shrink the
   second term, so the correlation $\rho$ between trees is the floor. Feature
   subsampling attacks $\rho$ directly.
