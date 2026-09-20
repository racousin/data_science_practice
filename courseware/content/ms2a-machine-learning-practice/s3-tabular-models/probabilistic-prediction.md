# Probabilistic Prediction

A model that answers "rain" or "0.3 mm" has made a decision it had no business
making. This lesson is about predicting what you actually know — a probability
for a class, a distribution for a number — and about scoring it so that honesty
is the winning strategy. It ends with CRPS, the score behind this session's
challenge, *European Rain Forecast*.

<!-- notes: 80 minutes. Short of time, skip parametric distributions and the
two conformal slides; the lab needs neither. Rough budget: 5 on decisions, 15
on proper scores, 15 on calibration, 15 on predictive distributions and the
hurdle model, 15 on CRPS, 10 on checking and the challenge, 5 on the lab
recipe and the questions.
Draw the umbrella line p > C/L on the board first: everything after it is about
producing a p you can put into that inequality. Do the CRPS worked example by
hand, term by term. Show the four rank histograms without their titles and ask
the room to diagnose each before revealing. Say it twice: the crps_ensemble on
the slide is exactly what challenge 177 computes. -->

---

## A forecast is an input to a decision

Take an umbrella? Carrying it costs $C$; getting soaked without it costs $L$.
With a probability $p$ of rain:

| | Rain ($p$) | Dry ($1 - p$) | Expected cost |
|---|---|---|---|
| Take it | $C$ | $C$ | $C$ |
| Leave it | $L$ | 0 | $pL$ |

Take it when $pL > C$, that is when $p > C/L$. The threshold belongs to the
person deciding, not to the model: a commuter in a suit and one in a raincoat
act differently on the same forecast. Admitting a patient, reviewing a
transaction and gritting a road have the same structure.

A classifier that outputs a label has fixed one threshold, usually 0.5, before
anyone knew the costs.

> Predict the probability. Decide with the costs. Keep the two steps apart.

<!-- notes: The patient version: admit when P(deterioration) exceeds the cost
of a bed over the cost of a missed case. A probability also lets you move the
threshold next quarter without retraining. -->

---

## A number is not enough either

A regression model says "0.3 mm of rain at 9:00 tomorrow". That could mean:

- a 30% chance of about 1 mm, dry otherwise
- near-certain drizzle of 0.3 mm
- a 3% chance of a 10 mm downpour, dry otherwise

Same mean, three different days. The picnic, the farmer and the storm-drain
operator each care about a different part of the distribution, and the point
prediction threw that part away.

A **predictive distribution** — a probability for every possible outcome —
keeps it. The rest of this lesson is how to produce one and how to score it.

---

## Proper scoring rules

A scoring rule $S(p, y)$ scores a forecast $p$ against the outcome $y$; here,
lower is better. If you believe the outcome follows $q$, your expected score for
reporting $p$ is the average of $S(p, Y)$ over outcomes $Y$ drawn from $q$. The
rule is **proper** if reporting your belief is optimal:

$$
\mathbb{E}_{Y \sim q}\left[S(q, Y)\right] \leq \mathbb{E}_{Y \sim q}\left[S(p, Y)\right] \quad \forall p
$$

and **strictly proper** if $p = q$ is the only optimum.

Under a strictly proper rule the best strategy is to say what you believe:
nothing is gained by shading the number towards what the scorer seems to want.
Every loss this lesson trains on — log loss, the Brier score, the NLL, CRPS,
and the pinball loss for its quantile — rewards the honest report. The scores
that do not — accuracy, AUC, and a first version of the challenge's skill
score — each get a slide.

---

## Log loss and Brier score

For a binary outcome $y \in \{0, 1\}$ and a forecast probability $p$:

$$
\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log p_i + (1 - y_i) \log(1 - p_i)\right]
$$

$$
\mathrm{BS} = \frac{1}{n}\sum_{i=1}^n (p_i - y_i)^2
$$

```python
from sklearn.metrics import log_loss, brier_score_loss
p = clf.predict_proba(X_val)[:, 1]
print(log_loss(y_val, p), brier_score_loss(y_val, p))
```

Both are strictly proper. Log loss is the negative log-likelihood, the loss
logistic regression and boosted classifiers minimise; it punishes a confident
miss without bound — $p = 0.001$ on an event that happens costs 6.9. The Brier
score is the squared error of the probability, bounded in $[0, 1]$.

---

## Improper rules invite hedging or exaggeration

![Optimal report against belief for three scores](assets/tabular/prob-improper-scores.png)

Score a forecast by $|p - y|^k$. For $k = 2$, the Brier score, the best report
is your belief. The absolute error ($k = 1$) pays you to exaggerate to 0 or 1;
$k = 3$ pays you to hedge towards 0.5. A metric that looks reasonable can pay
you to lie: check that it is proper before you optimise it.

---

## Accuracy and AUC are not proper

**Accuracy** only sees which side of 0.5 you land on. Reports of 0.51 and 0.99
earn the same, so nothing rewards the honest number: it is not strictly proper.

**AUC** only sees the ranking. Any increasing transformation of the scores
leaves it unchanged:

```python
from sklearn.metrics import roc_auc_score, log_loss
p = clf.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, p), roc_auc_score(y_val, p**4))  # equal
print(log_loss(y_val, p), log_loss(y_val, p**4))           # not
```

A model can have an AUC of 0.92 and probabilities nearly three times too high:
`class_weight="balanced"` does exactly that to a 10% class. Use AUC to ask
"does it rank?" and a proper score to ask "can I act on the numbers?".

---

## Calibration

A forecaster is **calibrated** if, among all the occasions it said $p$, the
event happened a fraction $p$ of the time:

$$
P(Y = 1 \mid \hat{p} = p) = p
$$

Of the days it said 30%, it rained on 30%. Calibration is a property of many
forecasts together; a single forecast is neither calibrated nor miscalibrated.

Calibrated is not the same as useful. Announcing the base rate every hour —
14.5% for a wet hour in the challenge data — is perfectly calibrated and tells
nobody anything.

---

## Reliability diagrams

![Reliability diagram, calibrated and overconfident](assets/tabular/prob-reliability.png)

Bin the forecasts and plot the observed frequency in each bin against the mean
forecast. A calibrated model sits on the diagonal. An overconfident one is
flatter than the diagonal: its 90% events happen 75% of the time, its 1% events
10%. The histogram below shows where the forecasts are.

---

## Drawing one in scikit-learn

```python
from sklearn.calibration import CalibrationDisplay, calibration_curve
obs, pred = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
CalibrationDisplay.from_predictions(y_val, p, n_bins=10,
                                    strategy="quantile")
```

`strategy="quantile"` puts the same number of forecasts in every bin. With the
default uniform bins, the extreme bins of an imbalanced problem hold a handful
of rows and the curve there is noise.

The **expected calibration error** summarises the diagram as the bin-weighted
gap to the diagonal. It moves with the binning: report the diagram, not only
the number.

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n} \left|\bar{y}_b - \bar{p}_b\right|
$$

---

## Which models come out miscalibrated

| Model | Typical distortion | Why |
|---|---|---|
| Logistic regression | close to calibrated | it minimises log loss directly |
| Boosting, many rounds | overconfident | late rounds push training rows to 0 and 1 |
| Random forest | under-confident at the ends | averaged trees rarely all agree |
| SVM | no probabilities at all | `decision_function` is a distance |
| Naive Bayes | overconfident | counts correlated evidence twice |
| Deep networks | overconfident | driven to low training loss |
| `class_weight`, resampling | overpredicts the minority | learned a different base rate |

For an SVM use `CalibratedClassifierCV(SVC(), ensemble=False)`;
`SVC(probability=True)` does the same and is deprecated from 1.9. For the
others, look at a reliability diagram on held-out rows whenever the numbers,
not only the ranking, will be used.

---

## Recalibration: Platt and isotonic

Fit a monotone map from the model's score $s$ to a probability, on rows the
model did not train on. Platt scaling fits a logistic curve with two
parameters:

$$
p = \frac{1}{1 + e^{-(a s + b)}}
$$

| | Platt, `"sigmoid"` | Isotonic, `"isotonic"` |
|---|---|---|
| Map | logistic in the score | any non-decreasing step function |
| Parameters | 2 | up to one per calibration row |
| Rows needed | a few hundred | a thousand or more |
| Fixes | S-shaped distortion | any monotone distortion |
| Fails by | being too rigid | overfitting, creating ties |

Both are monotone: recalibration fixes the numbers, not the ranking. For
multiclass networks, scikit-learn 1.8 adds `method="temperature"`.

---

## CalibratedClassifierCV

```python
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(model, method="isotonic", cv=5)
cal.fit(X_tr, y_tr)
p = cal.predict_proba(X_val)[:, 1]
```

With `cv=5` it fits five copies of the model, calibrates each on the fold it
did not see, and averages the five pairs. The calibrator never sees a
prediction on a training row: those are overconfident by construction, and a
map fitted on them learns to make things worse.

Already have a fitted model and a separate calibration set? Freeze it:

```python
from sklearn.frozen import FrozenEstimator
cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
cal.fit(X_cal, y_cal)          # rows the model never trained on
```

`cv="prefit"` is gone in 1.8. On time-ordered data the calibration rows come
after the training rows, like any validation set.

---

## Class weights and prior shift

`class_weight="balanced"` or undersampling trains the model in a world where
the classes are balanced, and its probabilities describe that world. If the
model saw a positive rate $\pi$ and the deployment rate is $\pi'$, correct the
odds:

$$
\frac{p'}{1 - p'} = \frac{p}{1 - p} \cdot \frac{\pi' / (1 - \pi')}{\pi / (1 - \pi)}
$$

```python
def shift_prior(p, pi_seen, pi_true):
    odds = p / (1 - p)
    odds *= (pi_true / (1 - pi_true)) / (pi_seen / (1 - pi_seen))
    return odds / (1 + odds)
```

With `"balanced"`, `pi_seen` is 0.5. The same formula handles a base rate that
moves between training and deployment, provided you know the new rate.

---

## The Brier score decomposes

Group the forecasts into bins of equal value: bin $k$ holds $n_k$ forecasts
$p_k$, of which a fraction $\bar{o}_k$ came true; $\bar{o}$ is the base rate.
Murphy (1973):

$$
\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}
$$

$$
\mathrm{REL} = \frac{1}{n}\sum_k n_k (p_k - \bar{o}_k)^2, \quad \mathrm{RES} = \frac{1}{n}\sum_k n_k (\bar{o}_k - \bar{o})^2, \quad \mathrm{UNC} = \bar{o}(1 - \bar{o})
$$

- **Reliability**: the gap between what you said and what happened. Lower is
  better; calibration drives it to 0.
- **Resolution**: how far the outcomes in your bins move away from the base
  rate. Higher is better; this is the information.
- **Uncertainty**: the variance of the outcome itself. Nothing you do changes
  it.

---

## Sharpness subject to calibration

Gneiting, Balabdaoui and Raftery (2007) state the goal of probabilistic
forecasting in one line: **maximise the sharpness of the predictive
distribution, subject to calibration.**

- Calibration is the constraint, the reliability term: the forecasts must mean
  what they say.
- Sharpness is the objective, what buys resolution: among calibrated
  forecasts, prefer the one that concentrates its probability — near 0 and 1,
  or in a narrow interval.

Climatology is calibrated with no sharpness. A deterministic forecast is
perfectly sharp and never calibrated. A proper score rewards both at once,
which is why you optimise a proper score rather than either property alone.

---

## Skill scores

A raw score depends on how hard the problem is: a Brier score of 0.10 is
excellent for a 50% event and worse than useless for a 10% one. A skill score
compares with a reference forecast on the same cases:

$$
\mathrm{SS} = 1 - \frac{S}{S_{\mathrm{ref}}}
$$

1 is perfect, 0 is as good as the reference, below 0 is worse. The classic
reference is **climatology**, the base rate issued every time. Its Brier score
is the uncertainty term, so the Brier skill score is

$$
\mathrm{BSS} = 1 - \frac{\mathrm{BS}}{\mathrm{UNC}} = \frac{\mathrm{RES} - \mathrm{REL}}{\mathrm{UNC}}
$$

Challenge 177 also compares with a reference, recent climatology, but divides
by a fixed number instead: the end of this lesson shows why.

---

## From ŷ to a predictive distribution

A regressor usually returns one number, an estimate of the conditional mean. A
probabilistic regressor returns the conditional distribution $F(y \mid x)$ in
one of three forms:

| Form | What the model outputs | Example |
|---|---|---|
| Parametric | the parameters of a known family | mean and variance of a Gaussian |
| Quantiles | values at a set of levels | the 5%, 50% and 95% amounts |
| Ensemble | a set of samples, or members | 50 plausible rain amounts |

They convert into each other: sample a parametric model to get members, sort
members to read quantiles, treat quantiles as members. The challenge asks for
the third form.

---

## Parametric distributions

Predict the parameters of a family, for instance $\mu(x)$ and $\sigma(x)$ of a
Gaussian, and score with the negative log-likelihood:

$$
\mathrm{NLL} = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}
$$

A two-stage recipe works with any regressor: fit the mean, then fit a second
model to the squared out-of-fold residuals, whose expectation is $\sigma^2$.
Non-negative targets have their own families, fitted with a log link:

| Target | Family | scikit-learn / LightGBM |
|---|---|---|
| Counts | Poisson | `loss="poisson"` / `objective="poisson"` |
| Positive amounts | Gamma | `loss="gamma"` / `objective="gamma"` |
| Zeros and amounts | Tweedie, 1 < power < 2 | `TweedieRegressor(power=1.5)` / `objective="tweedie"` |

Tweedie with power between 1 and 2 is a compound Poisson–Gamma: a point mass
at zero plus a positive part, the shape of hourly rain. The catch: these
objectives fit the mean only; a distribution also needs the dispersion and a
sampler. On tables, most practitioners go to quantiles instead.

---

## The pinball loss

For a quantile level $\tau \in (0, 1)$, a forecast $q$ and an outcome $y$:

$$
\rho_\tau(y, q) = \max\left(\tau (y - q), (\tau - 1)(y - q)\right)
$$

![Pinball loss for three quantile levels](assets/tabular/prob-pinball.png)

Too low costs $\tau$ per unit, too high $1 - \tau$, so the minimiser is the
$\tau$-quantile. At $\tau = 0.5$ it is half the absolute error: the median.

---

## Quantile regression with trees

```python
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

levels = [0.05, 0.25, 0.5, 0.75, 0.95]
hgb = {t: HistGradientBoostingRegressor(loss="quantile", quantile=t)
       .fit(X_tr, y_tr) for t in levels}
lgbm = {t: lgb.LGBMRegressor(objective="quantile", alpha=t, verbose=-1)
        .fit(X_tr, y_tr) for t in levels}
Q = np.column_stack([lgbm[t].predict(X_val) for t in levels])
```

One model per level, each trained on its own loss. The older
`GradientBoostingRegressor(loss="quantile", alpha=t)` works too, and so does the
linear `QuantileRegressor(quantile=t, alpha=0)`, whose `alpha` is an L1 penalty,
not the level. Three libraries, three names for $\tau$.

Score each level with `mean_pinball_loss(y_val, Q[:, j], alpha=levels[j])`.

---

## A quantile fan

![Five quantile models on a 1-D problem](assets/tabular/prob-quantile-fan.png)

Five `HistGradientBoostingRegressor` models on a problem whose noise grows with
$x$. The band widens where the data are noisy, which no mean model and no
constant ± interval can do. The 5–95% band should hold 90% of new points; it
holds 88.8%. Extreme quantiles fitted in-sample tend to under-cover a little —
the gap conformal prediction closes.

---

## Quantile crossing

Independent models know nothing of each other. Where data are thin the 0.75
model can predict below the 0.5 model, an impossible distribution. Sort each
row:

```python
Q = np.sort(Q, axis=1)          # q_0.05 <= q_0.25 <= ... <= q_0.95
```

Rearrangement never takes the curves further from the true, monotone ones
(Chernozhukov et al., 2010). For CRPS the order of members does not matter —
the score sorts them — but intervals and interpolation between levels need
monotone quantiles.

---

## Prediction intervals and coverage

The central 90% interval runs from the 5% quantile to the 95% quantile. Two
numbers describe it on held-out data:

```python
lo, hi = Q[:, 0], Q[:, -1]                          # 5% and 95%
coverage = np.mean((y_val >= lo) & (y_val <= hi))   # target: 0.90
width = np.mean(hi - lo)                            # sharpness
```

Coverage without width is meaningless: predict $(-\infty, \infty)$. Width
without coverage is meaningless: predict a point.

Check coverage per group as well. 90% overall can be 99% in Athens and 70% in
Glasgow.

---

## Split conformal prediction

A wrapper that turns any point model into intervals with a guarantee:

1. Fit the model on the training rows.
2. On $n$ calibration rows, compute the scores $s_i = |y_i - \hat{y}(x_i)|$.
3. Take $\hat{q}$, the $\lceil (n + 1)(1 - \alpha) \rceil$-th smallest score.
4. Predict $[\hat{y}(x) - \hat{q}, \hat{y}(x) + \hat{q}]$.

```python
s = np.abs(y_cal - model.predict(X_cal))
k = int(np.ceil((len(s) + 1) * (1 - alpha)))
qhat = np.sort(s)[k - 1] if k <= len(s) else np.inf
lo, hi = pred - qhat, pred + qhat
```

If the calibration rows and the new row are exchangeable, the new outcome falls
inside with probability at least $1 - \alpha$ — for any model, at any sample
size. With fewer than $(1 - \alpha)/\alpha$ calibration rows the quantile is
$+\infty$ and the interval is the whole line: valid, and useless.

---

## Conformalised quantile regression

Split conformal gives every row the same width. CQR (Romano et al., 2019)
starts from quantile models, so the width adapts, and only corrects their
coverage:

```python
s = np.maximum(lo_cal - y_cal, y_cal - hi_cal)    # negative inside
k = int(np.ceil((len(s) + 1) * (1 - alpha)))
qhat = np.sort(s)[k - 1] if k <= len(s) else np.inf
lo, hi = lo_new - qhat, hi_new + qhat
```

If the quantile models over-cover, $\hat{q}$ is negative and the band shrinks;
if they under-cover, as in the fan above, it widens.

What conformal does not promise:

- coverage per group — 90% over all rows, not 90% for Glasgow in January
- validity under drift — a 2023 calibration set says little about 2026
- quality — a bad model gets wide, valid intervals
- a whole distribution — it calibrates one interval, CRPS scores them all

---

## Ensembles as distributions

An ensemble of $M$ members is an empirical distribution. Where the members come
from decides what spread they carry:

| Source | Spread it carries |
|---|---|
| Bagging members, random seeds | the model's uncertainty only |
| Analogues: what followed the $K$ most similar past cases | the outcome's own noise |
| Weather ensembles, e.g. ECMWF's 50 perturbed runs | initial state and physics |

A seed ensemble of point models is far too narrow: it varies where the model is
unsure, not where the rain is. Analogues are cheap and honest, zeros included:

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=50).fit(Z_tr)    # scaled features
_, idx = nn.kneighbors(Z_val)
members = y_tr[idx]                                # (n_val, 50)
```

<!-- notes: Physical weather ensembles are under-dispersed too, and national
services post-process them statistically before issuing probabilities. The
analogue quality depends entirely on the distance: scale the features and
weight the ones that matter. -->

---

## Rain is zero-inflated

![Paris hourly rain and wet-hour share by month](assets/tabular/prob-rain-europe.png)

Over the 45 cities, 14.5% of hours are wet (≥ 0.1 mm, the resolution of the
data). When it rains, the median is 0.2 mm, the 90th percentile 1.4 mm and the
99th 4.5 mm. The wet share runs from 7.9% in Athens to 28.3% in Glasgow.

---

## A hurdle model

No single family fits a spike at zero and a long right tail. Split the question
in two — does it rain, and how much if it does:

$$
F(x \mid z) = 1 - p(z) + p(z)\, G(x \mid z), \qquad x \geq 0
$$

- $p(z)$ = P(wet | z): a classifier on all rows, target `rain >= 0.1`.
  Calibrate it: it sets how many members are exactly zero.
- $G$, the amount given wet: quantile regressors fitted on the wet rows only,
  at a grid of levels.

The classifier sees every row and the regressors see only the 14.5% that are
wet, so each part learns from the data that concern it.

---

## Fitting the two parts

```python
wet = y_tr >= 0.1
clf = lgb.LGBMClassifier(verbose=-1).fit(X_tr, wet)
levels = np.linspace(0.05, 0.95, 10)
amount = {t: lgb.LGBMRegressor(objective="quantile", alpha=t,
                               verbose=-1)
          .fit(X_tr[wet], np.log1p(y_tr[wet])) for t in levels}

p_wet = clf.predict_proba(X_val)[:, 1]
q_wet = np.column_stack([amount[t].predict(X_val) for t in levels])
q_wet = np.sort(np.expm1(q_wet), axis=1)
```

Fitting on `log1p` tames the tail and is safe for quantiles: a quantile of
$\log(1 + y)$ is $\log(1 + \cdot)$ of the same quantile of $y$, because the
transform is increasing. It is not safe for a mean. The sort repairs any
crossing before the levels are interpolated.

---

## Members from a hurdle model

The mixture's quantile at level $\tau$ is 0 when $\tau \leq 1 - p$, and
otherwise the amount quantile $G^{-1}(u)$ at $u = (\tau - 1 + p)/p$. Evaluate
it at the midpoint levels $\tau_k = (k - 0.5)/M$:

```python
def hurdle_members(p_wet, levels, q_wet, m=50):
    """p_wet (n,), q_wet (n, L) sorted amount quantiles given wet."""
    tau = (np.arange(1, m + 1) - 0.5) / m
    p = np.clip(p_wet, 1e-6, 1)[:, None]
    u = (tau[None, :] - (1 - p)) / p                   # (n, m)
    out = np.zeros(u.shape)
    for i in range(len(u)):
        wet = u[i] > 0
        out[i, wet] = np.interp(u[i, wet], levels, q_wet[i])
    return out
```

`np.interp` holds the end values beyond the outer levels, so the tail stops at
the top fitted quantile: add a 0.99 level if the heavy hours matter.

---

## CRPS: the definition

The continuous ranked probability score compares the forecast CDF $F$ with the
observation $y$, seen as a step:

$$
\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) - \mathbf{1}\{x \geq y\}\right)^2 dx
$$

![Forecast CDF, observation step and the CRPS area](assets/tabular/prob-crps-area.png)

---

## CRPS: the kernel form

The integral is the Brier score of the event "$Y \leq x$", summed over every
threshold $x$: strictly proper, and in the units of $y$, millimetres here. With
$X$ and $X'$ independent draws from $F$ it can also be written

$$
\mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \frac{1}{2}\mathbb{E}|X - X'|
$$

- The first term is accuracy: how far the forecast's draws land from what
  happened.
- The second is a spread credit: a wide distribution gets part of its distance
  back.

Together they reward the right amount of spread. Put all the mass on one point
and the credit vanishes: CRPS becomes the absolute error. It is MAE generalised
to distributions.

---

## CRPS of an ensemble

Replace $F$ by the empirical distribution of $M$ members $x_i$:

$$
\mathrm{CRPS} = \frac{1}{M}\sum_{i=1}^{M} |x_i - y| - \frac{1}{2M^2}\sum_{i=1}^{M}\sum_{j=1}^{M} |x_i - x_j|
$$

The double sum costs $O(M^2)$. Sort the members, $x_{(k)}$ being the $k$-th
smallest, and it takes one pass:

$$
\sum_{i=1}^{M}\sum_{j=1}^{M} |x_i - x_j| = 2\sum_{k=1}^{M} (2k - M - 1)\, x_{(k)}
$$

With $M = 1$ the second term is zero and CRPS is $|x_1 - y|$: a deterministic
forecast is scored by its absolute error.

---

## Quantiles are members

For an ensemble, CRPS is exactly twice the mean pinball loss of the sorted
members at the midpoint levels $\tau_k = (k - 0.5)/M$:

$$
\mathrm{CRPS} = \frac{2}{M}\sum_{k=1}^{M} \rho_{\tau_k}\left(y, x_{(k)}\right)
$$

For a continuous $F$ it is twice the pinball loss integrated over all levels.
Two consequences:

- Quantile models trained on the pinball loss optimise CRPS level by level.
- The best $M$ members to represent a distribution are its quantiles at
  levels $(k - 0.5)/M$. Random samples cost about $\mathbb{E}|X - X'|/(2M)$
  more on average.

---

## Fair CRPS, units and skill

- **Fair CRPS** divides the spread term by $2M(M - 1)$ instead of $2M^2$, which
  removes the penalty for sampling a small ensemble. Challenge 177 uses the
  standard estimator above, like `properscoring.crps_ensemble` and the default
  of `scoringrules.crps_ensemble`.
- **Units.** CRPS is in mm. A 5 mm miss in a downpour weighs as much as fifty
  0.1 mm misses, so the wet hours dominate any total.
- **Skill.** The textbook skill score compares with a reference on the same
  cases, pooled:

$$
\mathrm{CRPSS} = 1 - \frac{\sum \mathrm{CRPS}}{\sum \mathrm{CRPS}_{\mathrm{ref}}}
$$

---

## A worked example

One hour, 1.0 mm observed. Four forecasts:

| Forecast | Members (mm) | Mean abs. error | Spread credit | CRPS |
|---|---|---|---|---|
| Dry, certain ($M = 1$) | 0 | 1.000 | 0.000 | 1.000 |
| Sharp, wrong place | 2.6, 2.8, 3.0, 3.2, 3.4 | 2.000 | 0.160 | 1.840 |
| Vague | 0, 0, 0.2, 1.5, 4.0 | 1.260 | 0.760 | 0.500 |
| Sharp, right place | 0.6, 0.8, 1.0, 1.2, 1.5 | 0.260 | 0.176 | 0.084 |

Sharp and wrong is punished almost like a point forecast: its credit is small.
Vague earns a large credit but starts far away. The winner is close and tight.

One hour cannot show calibration. That needs many hours.

<!-- notes: Do the vague row on the board with the sorted formula: coefficients
(2k - M - 1) for M = 5 are -4, -2, 0, 2, 4; so the credit is
(2 x 1.5 + 4 x 4.0) / 25 = 0.76. -->

---

## Over many hours

![CRPS skill of six forecasters on simulated rain](assets/tabular/prob-crps-forecasters.png)

20 000 simulated hours with known rain distributions. Even the truth scores
only +0.17: rain is noisy. Zero beats the mean, and a seed ensemble barely
improves on the mean.

---

## CRPS in numpy

```python
import numpy as np

def crps_ensemble(members, y):
    """CRPS of ensembles members (..., M) against y (...)."""
    x = np.sort(np.asarray(members, dtype=float), axis=-1)
    y = np.asarray(y, dtype=float)[..., None]
    m = x.shape[-1]
    k = np.arange(1, m + 1)
    skill = np.abs(x - y).mean(axis=-1)
    spread = ((2 * k - m - 1) * x).sum(axis=-1) / m**2
    return skill - spread
```

It is the challenge's estimator, and it takes the challenge's shapes directly:
members `(45, 2, M)` against observations `(45, 2)`. The challenge clips
members below 0 to 0 before scoring; clip yours when you validate.

---

## Checked against properscoring

```python
import properscoring as ps
rng = np.random.default_rng(0)
ens = rng.gamma(0.5, 1.0, size=(1000, 30))
obs = rng.gamma(0.5, 1.0, size=1000)
gap = crps_ensemble(ens, obs) - ps.crps_ensemble(obs, ens)
print(np.abs(gap).max() < 1e-12)         # True
print(crps_ensemble([[2.0]], [0.5]))    # [1.5]: M = 1 is |x - y|
```

Mind the argument order: `properscoring` and `scoringrules` take the
observations first. `scoringrules.crps_ensemble(..., estimator="fair")` gives
the fair variant, which is not what the challenge computes.

---

## PIT and rank histograms

If $F$ is the true distribution of $y$, the probability integral transform
$u = F(y)$ is uniform on $[0, 1]$. For an ensemble, the rank of $y$ among the
$M$ members is uniform over the $M + 1$ positions. Histogram it over many
cases: a calibrated forecaster gives a flat histogram.

Rain adds a twist: when $y = 0$ and several members are 0, the rank is a tie.
Break ties at random, or every dry hour piles into the lowest bin:

```python
def obs_rank(members, y, rng):
    below = (members < y[:, None]).sum(axis=1)
    ties = (members == y[:, None]).sum(axis=1)
    return below + rng.integers(0, ties + 1)      # 0 .. M
```

For a CDF with a jump at $y$, the same fix is the randomised PIT: $u$ drawn
uniformly between $F(y^-)$ and $F(y)$.

---

## Reading a rank histogram

![Four rank histograms: flat, U-shaped, dome, sloped](assets/tabular/prob-rank-histograms.png)

U: widen the members. Dome: narrow them. Slope: shift the centre.

---

## Coverage of central intervals

The rank histogram's tails in one number: the share of observations inside
the ensemble's central intervals.

```python
for level in (0.5, 0.8, 0.9):
    lo, hi = np.quantile(members, [(1 - level) / 2, (1 + level) / 2],
                         axis=1)
    print(level, np.mean((y >= lo) & (y <= hi)))
```

Mind small ensembles. $M$ calibrated members hold $y$ between their lowest
and highest member with probability $(M - 1)/(M + 1)$: 90% for 19 members. The
interpolated 5–95% range of the same 19 members covers only about 82%.

Flat and covered are necessary, not sufficient: climatology passes both. Read
them next to the CRPS, never instead of it.

---

## A skill score can stop being proper

CRPS is proper; a skill score built from it need not be. A first version of
challenge 177 averaged a ratio over runs and horizons, with $A_r$ and $B_r$
the agent's and the reference's mean CRPS in run $r$ at one horizon:

$$
\mathrm{score} = \frac{1}{R}\sum_{r=1}^{R} \left(1 - \frac{A_r}{B_r}\right)
$$

A mean of ratios is not a ratio of means. $B_r$ depends on the outcome and is
small exactly when Europe is dry, so dry runs weigh the most and a dry bias
pays. Replayed on 1,700 runs, "always 0 mm" beat a per-city, per-month
climatology, +0.079 to +0.050, and halving every amount gained about 7
standard errors while making the pooled CRPS worse.

The fix is a denominator the outcome cannot move: a fixed constant per month.
Above the floor at −1, the expected score is then a fixed affine function of
the expected CRPS, and proper again.

---

## The rain challenge

Challenge 177, *European Rain Forecast*. Every 6 hours your agent is called
once:

- **In**: the last 48 hours of 8 weather features for 45 European cities, and
  the horizons `[6, 48]`.
- **Out**: `{"rain": (45, 2, M)}` — for each city and horizon, $M$ members in
  mm, $1 \leq M \leq 100$, the same $M$ everywhere, all finite.
- **Reference**: recent climatology — the city's own 48 observed hours, used
  as a 48-member ensemble for both horizons.
- **Score**: the CRPS skill below, per run and horizon, floored at −1; the
  run score is the mean of the two horizons, the leaderboard the mean over
  runs. The reference scores exactly 0; positive is better.
- **Budget**: 60 s for the call, 3 CPU, 3 GiB; model files load from next to
  `agent.py`.

$$
\mathrm{skill}_h = \frac{\overline{\mathrm{CRPS}}_{\mathrm{ref}} - \overline{\mathrm{CRPS}}_{\mathrm{agent}}}{C_h(\mathrm{month\ of\ the\ valid\ hour})}
$$

$C_h$ is a fixed, published table: the reference's mean CRPS per calendar
month of the valid hour, over every hour of 2020–2024 taken as an issue time.

After a dry spell the reference is 48 zeros: on a dry hour its CRPS is 0 and
cannot be beaten. Your skill is earned when the weather changes.

<!-- notes: The request also carries 48 `observed` flags (False = the feed
dropped that hour and the previous value was carried forward), and the
visibility feature is always missing. A forecast whose target hour is missing
from the feed is dropped for everyone. The run details report skill_6h,
skill_48h, and the raw crps_6h and crps_48h in mm; "always 0 mm" scores about
0. -->

---

## A recipe for the lab

1. **Table.** One row per city and issue time. Features from the 48-hour
   window: recent rain totals, humidity, clouds, hour, month, city. Targets:
   rain at +6 h and at +48 h.
2. **Baseline.** Per-city, per-month climatology quantiles as members. Score
   them, and the 48-hour reference, with `crps_ensemble`.
3. **Hurdle model.** A calibrated LightGBM classifier for wet; LightGBM
   quantile models on the wet rows, in `log1p`; members from
   `hurdle_members`.
4. **Validate like the challenge.** A time-ordered split with a 48-hour gap
   (the gap is the target's horizon — *Time Series Models*, next); CRPS skill
   per run and horizon, averaged over runs.
5. **Diagnose.** A rank histogram with random ties, a reliability diagram of
   `p_wet`, CRPS by city and horizon.

The lab builds this end to end. Keep the order: a scored baseline first.

---

## Validate like the challenge scores

```python
def run_skill(ens, ref, y, c):
    """One run, as challenge 177 scores it. ens (45, 2, M), ref (45, 48)
    the window's rain, y (45, 2), c (2,) = C_h for the valid month."""
    a = crps_ensemble(np.clip(ens, 0, None), y).mean(axis=0)   # (2,)
    r = crps_ensemble(ref[:, None, :], y).mean(axis=0)         # (2,)
    return np.maximum((r - a) / c, -1.0).mean()

score = np.mean([run_skill(*run) for run in val_runs])  # leaderboard
```

The table `C` (12 months × 2 horizons) is the reference's mean CRPS over
every hour of 2020–2024 taken as an issue time, by month of the valid hour. It
is published as `REFERENCE_CRPS[h][month]` on the challenge page and in the lab
notebooks, where this function is called `skill`. Copy it: recomputed over
6-hourly runs alone it drifts by up to 0.004 mm. What this avoids:

- a per-run ratio: improper, it pays a dry bias
- a random split: the +48 h targets of training rows overlap the validation
  window
- reporting +6 h alone: +48 h is harder and counts the same

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   x, y = np.array([0.0, 0.0, 0.5, 3.0]), 1.0
   accuracy = np.abs(x - y).mean()
   credit = np.abs(x[:, None] - x[None, :]).mean() / 2
   print(accuracy, credit, accuracy - credit)  # -> 1.125 0.59375 0.53125
   ```

   The last number is the CRPS of that four-member ensemble, in mm.

2. You submit the conditional mean of rain, $M = 1$. A colleague submits
   0 mm everywhere and gets a lower CRPS. How?

   **Answer.** With one member CRPS is the absolute error, minimised by the
   median. The median is 0 whenever P(wet) < 0.5, which is most hours; the
   mean pays on every dry one.

3. Your rank histogram has tall bars at both ends. Diagnosis, and a fix?

   **Answer.** Under-dispersed: the observation falls outside the members too
   often. Use members that carry the outcome's noise — quantile models or
   analogues, not seeds.
