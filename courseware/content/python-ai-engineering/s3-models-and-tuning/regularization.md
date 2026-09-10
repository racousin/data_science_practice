# Regularization

One extra term in the loss, one hyperparameter, and simple way to avoid overfiting.

<!-- notes: ~15 minutes per the deck plan. The KKT slide is optional depth — skip
it if the room is struggling, but the constrained view is what makes the diamond
picture make sense. -->

---

## The idea

![With and without regularization](assets/s3-models-and-tuning/regularization/regularization-effect.png)

> Penalising large parameters forces the model to find simpler solutions that
> generalise better.

$$
\theta^* = \arg\min_\theta \; \frac{1}{n}\|Y - X\theta\|^2 + \lambda \cdot R(\theta)
$$

$\lambda$ controls the strength of the regularisation. It is a hyperparameter:

| | |
|---|---|
| $\lambda = 0$ | no penalty → the standard loss, back to square one |
| $\lambda \rightarrow \infty$ | all coefficients driven to 0 → the model predicts a constant |

Everything useful is in between, and you find it on the validation set.

---

## Two penalties

$$
\|\theta\|_1 = |\theta_1| + |\theta_2| + \ldots + |\theta_p|
\qquad
\|\theta\|_2^2 = \theta_1^2 + \theta_2^2 + \ldots + \theta_p^2
$$

![Unit balls for different norms](assets/s3-models-and-tuning/regularization/lp-norm-balls.png)


---

## The constrained view (KKT)

$$
\min_\theta \|y - X\theta\|^2 + \lambda\|\theta\|
\qquad \Longleftrightarrow \qquad
\min_\theta \|y - X\theta\|^2 \quad \text{s.t.} \quad \|\theta\| \leq t
$$

$$
\forall \lambda \geq 0, \; \exists\, t \geq 0 \quad
\text{such that both problems have the same solution (KKT)}
$$

So regularisation is the same thing as confining $\theta$ to a ball of radius
$t$. The solution lands where the loss contours first touch that ball:

![Ridge, Lasso and Elastic Net constraint regions](assets/s3-models-and-tuning/regularization/ridge-lasso-elasticnet-regions.png)

The $L_1$ ball has **corners on the axes**. Contours touch corners, and a corner
means some $\theta_j$ is exactly zero. The $L_2$ ball is round and has no
corners, so it shrinks coefficients without ever zeroing them.

---

## Ridge, Lasso, Elastic Net

$$
\text{Ridge:} \quad \min_\theta \|y - X\theta\|^2 + \lambda\|\theta\|_2^2
$$

$$
\text{Lasso:} \quad \min_\theta \|y - X\theta\|^2 + \lambda\|\theta\|_1
$$

$$
\text{Elastic Net:} \quad \min_\theta \|y - X\theta\|^2 + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2
$$

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge   = Ridge(alpha=1.0)
lasso   = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

> **Note:** `alpha` in sklearn is $\lambda$ in the formulas.

---

## The same thing for classification

$$
\text{Ridge (L2):} \quad \min_\theta -\sum_{i=1}^{n}\big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\big] + \lambda\|\theta\|_2^2
$$

$$
\text{Lasso (L1):} \quad \min_\theta -\sum_{i=1}^{n}\big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\big] + \lambda\|\theta\|_1
$$

$$
\text{Elastic Net:} \quad \min_\theta -\sum_{i=1}^{n}\big[\ldots\big] + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2
$$

$$
\text{where} \quad \hat{p}_i = \sigma(\theta^T x_i) = \frac{1}{1 + e^{-\theta^T x_i}}
$$

```python
from sklearn.linear_model import LogisticRegression
ridge   = LogisticRegression(penalty='l2', C=1.0)
lasso   = LogisticRegression(penalty='l1', C=10, solver='saga')
elastic = LogisticRegression(penalty='elasticnet', C=10, l1_ratio=0.5, solver='saga')
```

> **Note:** in sklearn, $C = 1/\lambda$. It runs the other way — **larger `C`
> means less regularisation.** This inversion catches people every year.
