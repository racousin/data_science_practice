# Probabilistic Prediction

A regression model that answers "0.3 mm" hides what it does not know. This
lesson predicts a whole distribution instead, with tools you already use
(scikit-learn, LightGBM), and scores it with CRPS, the metric of this session's
challenge, *European Rain Forecast*.

<!-- notes: ~60 minutes. Budget: 5 on why, 20 on quantile regression, 15 on the
hurdle model, 15 on CRPS, 5 on questions. Do the CRPS worked example by hand,
term by term. Say it twice: the crps_ensemble on the slide is exactly what
challenge 177 computes. -->

---

## A number is not enough

A regression model says "0.3 mm of rain at 9:00 tomorrow". That could mean:

- near-certain drizzle of 0.3 mm
- a 30% chance of about 1 mm, dry otherwise
- a 3% chance of a 10 mm downpour, dry otherwise

Same mean, three different days. The picnic, the farmer and the storm-drain
operator each care about a different part of the distribution, and the point
prediction threw that part away.

> Predict a distribution, not a number.

---

## The plan: predict quantiles

The simplest way to describe a distribution with a tabular model: predict a few
of its **quantiles**.

| Level | Meaning |
|---|---|
| 0.05 | 5% chance the outcome is below this value |
| 0.5 | the median |
| 0.95 | 95% chance the outcome is below this value |

One gradient-boosting model per level. No new library, no distribution
assumption, and the quantiles convert directly into the ensemble members the
challenge asks for.

---

## The pinball loss

A regressor trained on squared error predicts the mean. To predict the
$\tau$-quantile, train on the pinball loss:

$$
\rho_\tau(y, q) = \max\left(\tau (y - q), (\tau - 1)(y - q)\right)
$$

![Pinball loss for three quantile levels](assets/tabular/prob-pinball.png)

Under-predicting costs $\tau$ per unit, over-predicting $1 - \tau$. At
$\tau = 0.9$, being too low is 9 times more expensive, so the model aims high:
exactly at the 90% quantile. At $\tau = 0.5$ it is the median.

---

## Quantile regression in LightGBM

```python
import numpy as np
import lightgbm as lgb

levels = [0.05, 0.25, 0.5, 0.75, 0.95]
models = {t: lgb.LGBMRegressor(objective="quantile", alpha=t, verbose=-1)
             .fit(X_tr, y_tr) for t in levels}

Q = np.column_stack([models[t].predict(X_val) for t in levels])
Q = np.sort(Q, axis=1)     # independent models can cross: sort each row
```

Same thing in scikit-learn:
`HistGradientBoostingRegressor(loss="quantile", quantile=t)`.

Score one level with
`sklearn.metrics.mean_pinball_loss(y_val, Q[:, j], alpha=levels[j])`.

---

## A quantile fan

![Five quantile models on a 1-D problem](assets/tabular/prob-quantile-fan.png)

Five models on a problem whose noise grows with $x$. The band widens where the
data are noisy, which no mean model and no fixed ± interval can do.

---

## Check it: coverage and width

The 5%–95% band should contain 90% of new outcomes.

```python
lo, hi = Q[:, 0], Q[:, -1]
coverage = np.mean((y_val >= lo) & (y_val <= hi))   # target: 0.90
width = np.mean(hi - lo)                            # smaller is better
```

Always report both. Infinite width gives perfect coverage; zero width gives a
point. A good model reaches the target coverage with the narrowest band.

---

## Rain is zero-inflated

![Paris hourly rain and wet-hour share by month](assets/tabular/prob-rain-europe.png)

Over the 45 cities, only 14.5% of hours are wet (≥ 0.1 mm). When it rains, the
median is 0.2 mm, the 90th percentile 1.4 mm. A spike at zero plus a long tail:
one quantile model fitted on everything struggles with that shape.

---

## A hurdle model: two questions

1. **Does it rain?** A classifier on all rows, target `rain >= 0.1`.
2. **How much, if it rains?** Quantile models on the wet rows only.

```python
wet = y_tr >= 0.1
clf = lgb.LGBMClassifier(verbose=-1).fit(X_tr, wet)

levels = np.linspace(0.05, 0.95, 10)
amount = {t: lgb.LGBMRegressor(objective="quantile", alpha=t, verbose=-1)
             .fit(X_tr[wet], np.log1p(y_tr[wet])) for t in levels}

p_wet = clf.predict_proba(X_val)[:, 1]
q_wet = np.column_stack([amount[t].predict(X_val) for t in levels])
q_wet = np.sort(np.expm1(q_wet), axis=1)
```

`log1p` tames the tail. It is safe for quantiles because the transform is
increasing: undo it with `expm1` after predicting.

---

## From the two parts to members

With $M$ members at levels $\tau_k = (k - 0.5)/M$: the lowest $(1 - p)$ share
of members is exactly 0, the rest are read from the wet-amount quantiles.

```python
def hurdle_members(p_wet, levels, q_wet, m=50):
    """p_wet (n,), q_wet (n, L) sorted amount quantiles given wet."""
    tau = (np.arange(1, m + 1) - 0.5) / m
    p = np.clip(p_wet, 1e-6, 1)[:, None]
    u = (tau[None, :] - (1 - p)) / p          # level inside the wet part
    out = np.zeros(u.shape)
    for i in range(len(u)):
        w = u[i] > 0
        out[i, w] = np.interp(u[i, w], levels, q_wet[i])
    return out

members = hurdle_members(p_wet, levels, q_wet)   # (n_val, 50)
```

A 30% chance of rain gives 35 members at 0 and 15 wet ones.

---

## CRPS: scoring a distribution

The continuous ranked probability score measures the area between the forecast
CDF and the observation, seen as a step:

$$
\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) - \mathbf{1}\{x \geq y\}\right)^2 dx
$$

![Forecast CDF, observation step and the CRPS area](assets/tabular/prob-crps-area.png)

In millimetres, lower is better. It is proper: the best expected score comes
from reporting the distribution you actually believe.

---

## CRPS of an ensemble

For $M$ members $x_i$:

$$
\mathrm{CRPS} = \underbrace{\frac{1}{M}\sum_{i} |x_i - y|}_{\text{accuracy}} - \underbrace{\frac{1}{2M^2}\sum_{i,j} |x_i - x_j|}_{\text{spread credit}}
$$

With one member, the credit is zero and CRPS is the absolute error: **CRPS is
MAE for distributions.**

```python
def crps_ensemble(members, y):
    """members (n, M), y (n,). CRPS per row, in the units of y."""
    x = np.sort(members, axis=1)
    M = x.shape[1]
    accuracy = np.abs(x - y[:, None]).mean(axis=1)
    k = np.arange(1, M + 1)
    spread = x @ (2 * k - M - 1) / M**2     # sorted form, O(M)
    return accuracy - spread

print(crps_ensemble(members, y_val).mean())
```

This is exactly what challenge 177 computes.

---

## Worked example

Members $\{0, 0, 1, 3\}$, observed $y = 1$ mm.

- Accuracy: $(1 + 1 + 0 + 2)/4 = 1.0$
- Spread: pairwise distances sum to $2 \times (1 + 3 + 1 + 3 + 2) = 20$, so
  $20 / (2 \times 16) = 0.625$
- CRPS $= 1.0 - 0.625 = 0.375$

Compare with point forecasts: "1 mm" scores 0 (lucky), "0 mm" scores 1.0. The
ensemble hedges between dry and wet and pays little either way.

---

## Quantiles are members

For sorted members at levels $\tau_k = (k - 0.5)/M$:

$$
\mathrm{CRPS} = \frac{2}{M}\sum_{k=1}^{M} \rho_{\tau_k}\left(y, x_{(k)}\right)
$$

CRPS is the average pinball loss over all levels. So quantile models trained on
the pinball loss already optimise CRPS, level by level — which is why the
recipe above is the right one for the challenge.
