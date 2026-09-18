# Feature Engineering

Feature engineering is where domain knowledge enters a model: a ratio, a
timestamp taken apart, a per-customer summary. Every one of these is either a
fitted step or a leak in waiting, so all of them run inside the Pipeline.

<!-- notes: 20 minutes. Spend real time on cyclical encoding and on the
group-aggregation leak; the arithmetic, datetime and binning slides are one
example each and can move fast. Let the three figures do the talking: the
week of bike demand, the hour on a circle, the staircase that binning gives a
linear model. Close on the inference-time question — could this value be
computed when the model is asked to predict? — because the next lesson turns
it into a loop. -->

---

## The highest-leverage feature is arithmetic

```python
df["price_per_m2"] = df["price"] / df["surface_m2"]
df["days_since_signup"] = (df["order_ts"] - df["signup_ts"]).dt.days
df["debt_to_income"] = df["debt"] / df["income"].clip(lower=1)
```

A ratio, a difference, a rate. Three lines that a gradient-boosted tree would
need many splits to approximate and a linear model could never represent at all.
`clip(lower=1)` is not cosmetic: a zero denominator produces `inf`, which is not
`NaN` and survives every missing-value check you wrote this morning.

---

## Interactions and polynomials

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True,
                          include_bias=False)
```

`PolynomialFeatures` generates every product up to `degree`;
`interaction_only=True` keeps the cross-terms and drops the squares. It is also
a combinatorial trap: 50 columns at degree 2 become 1,325. Apply it to a handful
of columns you chose, never to the whole matrix, and never in front of a tree
model — trees build interactions by construction.

---

## Datetime decomposition

```python
ts = pd.to_datetime(df["datetime"])
df["hour"] = ts.dt.hour
df["dayofweek"] = ts.dt.dayofweek
df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
```

A timestamp as an integer is nearly useless; its components are not.

Add the domain calendar too — holidays, paydays, school terms, promotion weeks.
That is the part no library gives you.

---

## What the components carry

![Average hourly bike-sharing demand over one week, Sunday to Saturday: every weekday shows two sharp commute peaks, morning and evening, while Saturday and Sunday show a single broad afternoon hump](assets/preprocessing/bike-demand-by-hour-of-week.png)

Bike-sharing demand, hour by hour across one week: two commute peaks on every
weekday, one afternoon hump at the weekend. Demand depends on the hour and on
whether the offices are open, and neither is recoverable from a Unix epoch by
any split a tree can make.

---

## Cyclical encoding

Hour 23 and hour 0 are one hour apart, and the integers say 23. December and
January are adjacent, and the integers say 11. Project the value onto a circle:

$$
\sin\left(\frac{2 \pi h}{24}\right), \quad \cos\left(\frac{2 \pi h}{24}\right)
$$

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Do this for hour, day of week and month whenever the model is linear or a
network. Trees can carve the discontinuity out with extra splits, so the gain
there is smaller.

---

## The hour on a circle

![The 24 hours of the day plotted as points with sin(hour) on the horizontal axis and cos(hour) on the vertical axis, coloured from dark at hour 0 to yellow at hour 23: they lie evenly on a circle, and hour 23 sits next to hour 0 at the top](assets/preprocessing/hour-on-the-unit-circle.png)

Each hour becomes a point on the unit circle, and 23 is a neighbour of 0
again. Both columns are needed: the sine alone maps 03:00 and 09:00 to the same
value (0.7071 each), and the cosine alone maps 06:00 and 18:00 to the same value
(0.0 each).

---

## Binning

```python
df["age_band"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 120])
df["income_q"] = pd.qcut(df["income"], q=5, labels=False)
```

`cut` uses the edges you supply — legal ages, tax brackets, tariff bands. `qcut`
uses quantiles and produces balanced bins. Binning trades information for
robustness to outliers and gives a linear model a non-linear response.

---

## Ten bins, two models

![Regression on a noisy sine wave before and after binning the input into ten one-hot bins: before, the linear fit is a straight line and the tree follows every wiggle; after, both models produce the same staircase](assets/preprocessing/binning-linear-vs-tree.png)

Left, on the raw input: a straight line from the linear model, every wiggle from
the tree. Right, after ten one-hot bins: the same staircase from both. Before a
tree, binning is almost always a loss: the tree was going to find the threshold
anyway, and it would have found a better one.

---

## Group aggregations, and the leak inside them

```python
agg = train.groupby("customer_id")["amount"].agg(["mean", "std", "count"])
train = train.join(agg, on="customer_id")
test = test.join(agg, on="customer_id")  # train statistics, applied to test
```

Per-group summaries — a customer's mean basket, a station's median temperature,
a store's order count — are among the strongest tabular features there are.

They leak in two ways. Computing `agg` over train **and** test puts test rows
into a training feature. Aggregating the *target* per group puts the label into
the feature, exactly as in target encoding: it needs the same out-of-fold
treatment, or a strictly past-only window when the rows are ordered in time.
