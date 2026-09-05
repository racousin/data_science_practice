# Time Series Models

When rows are ordered, every technique in the previous lesson breaks. Shuffling
the folds trains on the future, and the score you get is a measurement of a
model you cannot deploy.

<!-- notes: 25 minutes. Start with the shuffled-split demo: fit on shuffled folds,
report a beautiful score, then re-fit with TimeSeriesSplit and watch it collapse.
That contrast is the lesson. Keep ARIMA short — it is a baseline here, not a
subject. -->

---

## The assumption that breaks

Cross-validation assumes rows are exchangeable: any row could have been in any
fold. Ordered data is not exchangeable.

- Consecutive rows are correlated, so a "held-out" row next to a training row is
  nearly a copy of it.
- The distribution moves. A model validated on 2019 says nothing about 2024.
- The deployment task is **extrapolation** — predict $t+1$ from $\le t$ — and a
  random fold asks for interpolation, which is easier.

A random split does not overestimate the score slightly; it routinely turns a
useless model into an excellent-looking one.

---

## Forward chaining

![Expanding-window time series folds](assets/tabular/tsfold.png)

Train on the past, validate on the future, then move forward and repeat. Each
fold's training set ends before its validation set begins.

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
```

Expanding window keeps all history; sliding keeps a fixed length and forgets.
Use sliding when the process is clearly non-stationary, expanding otherwise.

Sort by time first: `TimeSeriesSplit` splits on position, not on the timestamp
column, and will silently do the wrong thing on unsorted rows.

---

## The gap, and what the horizon means

`gap` inserts unused rows between train and validation. If the target is the
return over the next seven days, the last seven training rows have targets that
overlap the validation window; without a gap they leak. The gap is the target's
horizon.

> Validate at the horizon you will deploy at. A model tuned on one-step-ahead
> error and deployed thirty steps ahead has not been evaluated.

Multi-step forecasting is either **recursive** — feed predictions back as lags,
compounding error — or **direct**: one model per horizon, no compounding, $h$
times the training cost.

---

## Lag features

```python
for lag in (1, 2, 3, 7, 14, 28):
    df[f"y_lag_{lag}"] = df.groupby("series_id")["y"].shift(lag)
```

Lags turn a forecasting problem into a supervised tabular one. Pick them from the
domain: 1 and 2 for momentum, 7 for a weekly cycle, 28 or 365 for a seasonal one.

Every lag must be at least as long as the forecast horizon, or the feature will
not exist at prediction time. `groupby` before `shift` when the table holds
several series, or the tail of one series lands in the head of the next.

---

## Rolling-window features

```python
g = df.groupby("series_id")["y"]
df["y_roll_mean_7"] = g.shift(1).rolling(7).mean()
df["y_roll_std_28"] = g.shift(1).rolling(28).std()
```

Rolling means, deviations and extremes carry level and volatility raw lags miss.

The `shift(1)` before `rolling` is the whole correctness argument: without it the
window includes the current row and the feature contains the target. It is the
most common leak in time-series feature engineering, and it produces a model that
scores superbly and predicts nothing.

Add calendar features too, and encode strong cycles as sine and cosine pairs.

---

## Trend and seasonality

A series is usefully read as trend plus seasonality plus remainder.

```python
from statsmodels.tsa.seasonal import STL
res = STL(y, period=12).fit()
```

Trees cannot extrapolate a trend: every prediction is an average of training leaf
values, so a model trained on a rising series forecasts a flat line below its
next true value. Difference the target and model the change, or fit the trend
separately and let the tree learn the remainder.

Seasonality is the opposite — give a tree a month index and a seasonal lag and it
handles it well.

---

## Classical baselines

| Model | Predicts | Beat it by |
|---|---|---|
| Naive | $\hat{y}_{t+h} = y_t$ | any signal at all |
| Seasonal naive | $\hat{y}_{t+h} = y_{t+h-m}$ | modelling non-seasonal structure |
| Exponential smoothing (ETS) | weighted recent history, trend, season | multivariate information |
| ARIMA / SARIMA | autoregression on a differenced series | non-linearity, exogenous features |

$$
y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t
$$

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
fit = ExponentialSmoothing(y_tr, trend="add", seasonal="add",
                           seasonal_periods=12).fit()
```

Seasonal naive is the baseline that embarrasses people: on a strongly seasonal
business series it is often within a few percent of an elaborate model.

---

## Classical or boosting

| Situation | Use |
|---|---|
| One series, a few hundred points | ETS or ARIMA |
| One series, strong known seasonality | seasonal naive, then ETS |
| Many related series, covariates available | boosting on lag features |
| Non-linear effects, calendar and promotions | boosting on lag features |

Boosting on lag and rolling features is the default for the messy case: it takes
exogenous columns, handles hundreds of series in one model, and assumes no
stationarity. It pays by being unable to extrapolate and by needing features the
classical models derive for free. Fit both — the baseline costs a minute and
tells you whether the boosting model earned its complexity.

---

## Backtesting discipline

- Sort by time, then split. The final test period is the most recent block, held
  out before any feature is built.
- Build lags and rolling windows **inside** the pipeline, or verify by hand that
  every feature at time $t$ uses only data available at $t$.
- Report the metric per fold. A model whose error triples in the last fold has
  not learned the process; it has learned a regime that ended.
- Refit on all data before deploying, with the number of rounds early stopping
  chose on the last fold.

Every leak in this session is the same failure: a score computed with information
the deployed model will not have. Here that information is the future, and it
gets in through a missing `shift`.

---

## Check yourself

1. What breaks if you drop the `shift(1)` from
   `g.shift(1).rolling(7).mean()`?

   **Answer.** The window then includes the current row, so the feature
   contains the target. The model scores superbly and predicts nothing — the
   most common leak in time-series feature engineering.

2. Run this. You should get exactly the output shown.

   ```python
   import pandas as pd
   s = pd.Series([1, 2, 3, 4, 5])
   print(s.rolling(2).mean().tolist())            # -> [nan, 1.5, 2.5, 3.5, 4.5]
   print(s.shift(1).rolling(2).mean().tolist())   # -> [nan, nan, 1.5, 2.5, 3.5]
   ```

   Row 2 of the first line already knows `y_2`. Row 2 of the second does not.

3. Your target is the return over the next seven days. What do you pass as
   `gap` to `TimeSeriesSplit`, and why that number?

   **Answer.** `gap=7`. The gap is the target's horizon: without it the last
   seven training rows have targets computed over the validation window, and
   they leak.
