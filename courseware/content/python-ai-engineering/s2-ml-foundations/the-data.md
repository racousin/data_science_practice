# The Data

Before a model, a table. This lesson fixes what a dataset *is* — its two
dimensions, its notation, and the column that makes the problem supervised.

<!-- notes: 35 minutes. Run the penguins snippets live; they take ten seconds
and the room reads the output faster than a slide. The two numbers (n, p) and
the "y is missing at prediction time" framing are what the rest of the session
assumes. -->

---

## Two numbers describe any dataset

- **$n$ — the number of observations.** How many rows, how many independent
  units you measured.
- **$p$ — the dimension.** How many variables you recorded per observation.

$$
X \in \mathbb{R}^{n \times p}
$$

Everything downstream is sensitive to the ratio. $n \gg p$ is the classical
statistical regime. Once $p$ approaches $n$, least squares stops having a unique
solution and regularisation stops being optional.

The same phenomenon can be recorded at very different $(n, p)$:

![9 observations, 2 variables](assets/s2-ml-foundations/the-data/temperature-2var.png)

![9 observations, 3 variables](assets/s2-ml-foundations/the-data/temperature-3d.png)

Adding a variable moves you right along $p$; adding a year moves you down along
$n$. They are not interchangeable, and which one is scarce decides which models
are available to you.

---

## A concrete one

The Palmer Archipelago penguins: 344 individuals, 7 measurements each.

```python
import seaborn as sns
df = sns.load_dataset("penguins")
df.shape                       # -> (344, 7)
```

So $n = 344$ and $p = 7$.

```python
df.head(3)
```

| | species | island | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g | sex |
|---|---|---|---|---|---|---|---|
| 0 | Adelie | Torgersen | 39.1 | 18.7 | 181.0 | 3750.0 | Male |
| 1 | Adelie | Torgersen | 39.5 | 17.4 | 186.0 | 3800.0 | Female |
| 2 | Adelie | Torgersen | 40.3 | 18.0 | 195.0 | 3250.0 | Female |

Note already: two columns are text, not numbers, and the counts below are going
to come back as 342 rather than 344. Both are the next-to-last lesson's problem.

---

## Look at the marginals

```python
df[["flipper_length_mm", "body_mass_g"]].describe()
```

| | flipper_length_mm | body_mass_g |
|---|---|---|
| count | 342.0 | 342.0 |
| mean | 200.9 | 4201.8 |
| std | 14.1 | 802.0 |
| min | 172.0 | 2700.0 |
| 25% | 190.0 | 3550.0 |
| 50% | 197.0 | 4050.0 |
| 75% | 213.0 | 4750.0 |
| max | 231.0 | 6300.0 |

`count` is 342, not 344 — `describe` drops missing values silently, and that is
your first evidence they exist. On both columns the mean sits above the median
(200.9 against 197.0; 4201.8 against 4050), so both are right-skewed. And the
standard deviation on `body_mass_g` is 802 g: predicting the mean for every
penguin is a model, and it is wrong by about 800 g. That is the number anything
you fit has to beat.

Condition on a categorical and the picture changes:

```python
df.groupby("species")["body_mass_g"].agg(["count", "mean", "std"])
```

| species | count | mean | std |
|---|---|---|---|
| Adelie | 151 | 3701 | 459 |
| Chinstrap | 68 | 3733 | 384 |
| Gentoo | 123 | 5076 | 504 |

The within-species standard deviations (≈ 460) are well under the pooled 802.
Species carries a large part of the variance in mass — which is exactly the
statement "this feature is informative", made before any model was fitted.

---

## Look at the joint distribution

Marginals hide structure. Plot every pair at once:

```python
sns.pairplot(df, hue="species")
```

![All six pairs of the four numeric variables, coloured by species](assets/s2-ml-foundations/the-data/penguins-pairplot.png)

Read off, in one figure: `flipper_length_mm` and `body_mass_g` are strongly and
near-linearly related; Gentoo separates cleanly from the other two on almost any
pair; Adelie and Chinstrap overlap everywhere except `bill_length_mm`. That last
observation predicts which classes a model will confuse, before you train one.

Do this first, every time. It costs one line.

---

## Features and target

![Features and label in a table](assets/s2-ml-foundations/the-data/features-and-label.png)

Split the columns in two. The **features** (explanatory variables) are what you
are allowed to look at. The **target** (label, variable to explain) is the
column you must produce.

$$
X \in \mathbb{R}^{n \times p} \qquad Y \in \mathbb{R}^{n}
$$

$$
D = (X, Y) = \big(x_i, y_i\big)_{1 \le i \le n}
$$

One row $(x_i, y_i)$ is **an observation**, or **a sample**: $x_i \in
\mathbb{R}^p$ and $y_i$ is its label. On the penguins, choosing
`body_mass_g` as the target leaves $p = 6$ features and $n = 342$ usable rows.

---

## What makes it supervised

This is the definition the whole session rests on, so state it exactly.

You are given a **training set** in which the target is *observed*:

$$
D_{\text{train}} = \big(x_i, y_i\big)_{i=1}^{n} \qquad y_i \ \text{known}
$$

You must produce predictions on data where it is *not*:

$$
x_{n+1}, x_{n+2}, \ldots \qquad y_{n+1}, y_{n+2}, \ldots \ \text{unknown}
$$

The labels exist — a penguin has a mass whether or not you weighed it — you
simply do not have them at prediction time, and often never will. Supervision
means the labelled pairs were available *during training*, not that they are
available when the model is used.

Everything hard about machine learning follows from this asymmetry. Fitting the
training set is an interpolation exercise with a known answer. The task is to
be right on rows you have not seen, and the training set is the only evidence
you have about them.

---

## Three kinds of learning problem

![Supervised, unsupervised, reinforcement learning](assets/s2-ml-foundations/the-data/learning-types.png)

| | |
|---|---|
| **Supervised** | Learning from labelled pairs — features and a target. $Y$ is given. |
| **Unsupervised** | Learning structure from unlabelled data; no $Y$ at all. Clustering, density estimation, dimensionality reduction. |
| **Reinforcement** | Learning from interaction and a reward signal, where the data is generated by your own actions. |

This session and the next are entirely supervised. Reinforcement learning is the
back half of *MS2A - Machine Learning Practice*.

---

## Two tasks, decided by the type of $y$

$$
\text{regression:}\quad y \in \mathbb{R}
\qquad
\text{classification:}\quad y \in \{1, 2, \ldots, K\}
$$

The task is fixed by the target's type, not by the model you reach for.

![The same features under a continuous and a discrete target](assets/s2-ml-foundations/the-data/penguins-targets.png)

Same $X$, same 342 rows, same axes. Left, `body_mass_g` — continuous, and the
colour varies smoothly across the cloud. Right, `species` — three levels, and
the question becomes where to draw the boundaries.

| Target column | Example values | $y \in$ | Task |
|---|---|---|---|
| `body_mass_g` | 3750, 3800, 3250 | $\mathbb{R}$ | Regression |
| `flipper_length_mm` | 181, 186, 195 | $\mathbb{R}$ | Regression |
| `species` | Adelie, Gentoo, Chinstrap | $\{1,2,3\}$ | Classification, $K = 3$ |
| `sex` | Male, Female | $\{0,1\}$ | Binary classification |
| `body_mass_g > 4050` | True, False | $\{0,1\}$ | Binary classification |

The last row is the same measurement as the first. Thresholding a continuous
target turns regression into classification: you lose resolution — 4051 g and
6300 g become the same label — and you buy an easier target and a decision you
can act on directly. That choice is yours, and it changes your loss, your
metric, and your baseline.
