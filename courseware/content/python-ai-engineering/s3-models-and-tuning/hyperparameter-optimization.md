# Hyperparameter Optimization

Every model in this session came with a dial: $K$, $C$, `max_depth`, $\lambda$,
`learning_rate`. This is how you set them without lying to yourself about the
result.

<!-- notes: ~10 minutes per the deck plan. Keep the parameter/hyperparameter
table on screen while doing the search code — the whole lesson is that
distinction plus one sklearn call. -->

---

## Parameters versus hyperparameters

| Parameters ($\theta$) | Hyperparameters |
|---|---|
| Learned during training | Set **before** training |
| Weights, coefficients, splits | $K$, $C$, $\lambda$, `max_depth`, `lr`, … |
| Minimise **training** loss | Optimise **validation** performance |

The second row is why they cannot be learned the same way: there is no gradient
of the validation score with respect to `max_depth`. You search instead.

---

## Grid search and random search

![Grid search versus random search](assets/s3-models-and-tuning/hyperparameter-optimization/grid-vs-random-search.png)

- **Grid search** — every combination of a discrete list per hyperparameter.
  Exhaustive, and exponential in the number of hyperparameters.
- **Random search** — sample combinations from ranges. Same budget, more distinct
  values tried **per hyperparameter**.

Look carefully at the figure: with 9 trials, grid search tests 3 distinct values
of each hyperparameter; random search tests 9. When only one of the two actually
matters — which is the usual case — random search has explored it three times as
finely for the same cost.

---

## Model selection

![Selecting among candidates](assets/s3-models-and-tuning/hyperparameter-optimization/model-selection-flow.png)

Train each candidate on the training set, score each on held-out data, keep the
best. The loop is the same whether the candidates differ by hyperparameter or by
model family.

---

## In scikit-learn

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators':  [50, 100, 200],
    'max_depth':     [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
}
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

`cv=5` means each of the $3 \times 4 \times 3 = 36$ combinations is fitted five
times — 180 fits. Grid search gets expensive fast, which is the practical
argument for `RandomizedSearchCV` with an explicit `n_iter`.

`best_score_` is a **cross-validated** score on the training data. It is not your
test score, and reporting it as one is the "using the test set to choose" leak
from the validation lesson.
