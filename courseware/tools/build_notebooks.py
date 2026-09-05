#!/usr/bin/env python3
"""Build the Session 2 challenge notebooks.

Two notebooks, deliberately different in kind:

* `aie-s2-bike-demand.ipynb`  — **worked**. Runs end to end and produces a
  submission that scores the challenge's declared baseline. This is the one a
  student reads to learn the shape of the task.
* `aie-s2-bank-marketing.ipynb` — **guided**. The same steps in English, with
  empty cells. No code: the student writes it, having just read the other one.

Both are written to `website/public/modules/python-ai-engineering/challenges/`,
which is what makes the Colab link work — Colab reads the file from GitHub at
`colab.research.google.com/github/<owner>/<repo>/blob/<branch>/<path>`.

    python courseware/tools/build_notebooks.py

Regenerate after editing this file; the .ipynb files are committed so the Colab
links resolve, but this script is the source.
"""
from __future__ import annotations

import itertools
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "website" / "public" / "modules" / "python-ai-engineering" / "challenges"

GITHUB = "racousin/data_science_practice"
BRANCH = "main"

BIKE_ID = 183
BANK_ID = 184


def colab_url(name: str) -> str:
    rel = f"website/public/modules/python-ai-engineering/challenges/{name}"
    return f"https://colab.research.google.com/github/{GITHUB}/blob/{BRANCH}/{rel}"


_COUNTER = itertools.count()


def _cell_id() -> str:
    """nbformat >= 4.5 requires a per-cell id, and warns loudly without one.
    Derived from a counter so regenerating is byte-identical (the sync test
    depends on that)."""
    return f"c{next(_COUNTER):03d}"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": _src(lines)}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": _src(lines)}


def _src(lines) -> list[str]:
    text = "\n".join(lines)
    out = text.splitlines(keepends=True)
    return out or [""]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# 1. Bike sharing demand — the worked notebook
# --------------------------------------------------------------------------- #
def build_bike() -> dict:
    name = "aie-s2-bike-demand.ipynb"
    return notebook([
        md(
            f"# AIE S2 — Bike Sharing Demand",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(name)})",
            "",
            "**Regression.** Predict how many bikes are rented in a given hour,",
            "from the calendar and the weather.",
            "",
            f"Challenge: <https://ml-arena.com/viewchallenge/{BIKE_ID}>",
            "",
            "This notebook is the worked example. It runs top to bottom and ends",
            "with a scored submission. Five steps, and every later challenge in",
            "the course is the same five:",
            "",
            "1. get the data",
            "2. look at it",
            "3. turn it into numbers",
            "4. fit a model and measure it *before* submitting",
            "5. predict the test set and submit",
        ),
        md("---", "", "## 0. Setup",
           "",
           "`pandas`, `seaborn` and `scikit-learn` are already in Colab. The",
           "ML-Arena client is what downloads the data and uploads your answer.",
           "",
           "The distribution is called **`mlarena-sdk`** and it imports as",
           "`mlarena`. Do not `pip install mlarena` — that is an unrelated",
           "package by another author, and none of the calls below exist in it."),
        code("!pip install -q mlarena-sdk"),
        md("---", "", "## 1. Get the data",
           "",
           "Paste your personal API key from your ML-Arena **Profile** page. It",
           "starts with `mlk_user_`. `download_dataset` writes the three public",
           "files into the working directory."),
        code(
            "import mlarena",
            "",
            'API_KEY = "mlk_user_..."   # <- paste yours here',
            f"CHALLENGE_ID = {BIKE_ID}",
            "",
            "client = mlarena.connect(api_key=API_KEY)",
            'client.download_dataset(CHALLENGE_ID, ".")',
        ),
        md("---", "", "## 2. Read it",
           "",
           "Three files. `X_train` and `y_train` are what you learn from;",
           "`X_test` is what you must predict. `y_test` does not exist on your",
           "side — that is the whole point of the exercise."),
        code(
            "import pandas as pd",
            "",
            'X_train = pd.read_csv("X_train.csv")',
            'y_train = pd.read_csv("y_train.csv")["prediction"]',
            'X_test = pd.read_csv("X_test.csv")',
            "",
            'print("X_train", X_train.shape, " y_train", y_train.shape, " X_test", X_test.shape)',
            "X_train.head()",
        ),
        md("`X_train` is (13903, 13): **n = 13,903 observations** and 13 columns,",
           "of which one is the `id`, so **p = 12 features**. The target lives in",
           "a separate file, aligned by `id`.",
           "",
           "Check the column types before anything else — they decide what you",
           "are allowed to do with each one."),
        code(
            "X_train.dtypes",
        ),
        md("Four columns are text (`season`, `holiday`, `workingday`, `weather`)",
           "and the rest are numbers. A linear model cannot consume the text ones",
           "as they are; step 3 is about that.",
           "",
           "Now the target and the numeric features:"),
        code(
            'print(y_train.describe().round(2))',
            "X_train.describe().round(2)",
        ),
        md("---", "", "## 3. Look at it",
           "",
           "This is the step that is easiest to skip and cheapest to do. Five",
           "plots, and each one changes a decision you are about to make.",
           "",
           "First, put the target back next to the features so seaborn can plot",
           "them together."),
        code(
            'df = X_train.assign(count=y_train)',
            'df.head(3)',
        ),
        md("### 3a. The target's distribution",
           "",
           "Always look at what you are predicting first."),
        code(
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            'sns.set_theme(style="whitegrid")',
            "",
            "fig, ax = plt.subplots(figsize=(9, 3.5))",
            'sns.histplot(df["count"], bins=60, ax=ax, color="#2f6f9f")',
            'ax.set_title(f"Hourly rental count — mean {df[\'count\'].mean():.0f}, median {df[\'count\'].median():.0f}")',
            'ax.set_xlabel("count")',
            "plt.show()",
        ),
        md("Strongly right-skewed, and bounded below by 0. Two consequences worth",
           "holding on to: the mean sits well above the median, and a model with",
           "an unbounded output — such as a linear one — is free to predict",
           "negative rentals. It will."),
        md("### 3b. The hour of the day",
           "",
           "The single most informative plot in this dataset. Split it by",
           "`workingday`, because there is no reason a Tuesday and a Sunday should",
           "look alike."),
        code(
            "fig, ax = plt.subplots(figsize=(9, 4))",
            'sns.lineplot(data=df, x="hour", y="count", hue="workingday",',
            '             palette=["#c1553b", "#2f6f9f"], ax=ax)',
            'ax.set_title("Average rentals by hour — working days vs the rest")',
            'ax.set_xticks(range(0, 24, 2))',
            "plt.show()",
        ),
        md("Two completely different shapes. Working days have a sharp **commute",
           "double peak** at 08:00 and 17:00–18:00; non-working days have a single",
           "broad afternoon hump around 13:00–15:00.",
           "",
           "Note what this implies for the model. The relationship between `hour`",
           "and `count` is not a line — it is not even monotone. A model that",
           "multiplies `hour` by one coefficient cannot represent either curve.",
           "Remember this when you read your score."),
        md("### 3c. Temperature",
           "",
           "The obvious weather candidate."),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))",
            'sns.regplot(data=df, x="temp", y="count", ax=axes[0],',
            '            scatter_kws=dict(alpha=0.08, s=6, color="#2f6f9f"),',
            '            line_kws=dict(color="#c1553b"))',
            'axes[0].set_title("Rentals vs temperature (°C)")',
            'sns.boxplot(data=df, x="weather", y="count", ax=axes[1],',
            '            order=["clear", "misty", "rain", "heavy_rain"], color="#2f6f9f")',
            'axes[1].set_title("Rentals by weather")',
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("Warmer is busier, roughly linearly — this one a linear model *can*",
           "use. Weather degrades demand in the order you would guess.",
           "",
           "`heavy_rain` is worth a second look: the box is almost invisible",
           "because there are only a handful of such hours in the data. A category",
           "with three observations will not support a reliable coefficient."),
        code(
            'df["weather"].value_counts()',
        ),
        md("### 3d. What correlates with what",
           "",
           "A correlation heatmap on the numeric columns, to catch redundancy",
           "before it reaches the model."),
        code(
            'num = df.select_dtypes("number")',
            "fig, ax = plt.subplots(figsize=(7.5, 5.5))",
            'sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r",',
            "            center=0, ax=ax, annot_kws={\"size\": 8})",
            'ax.set_title("Correlation, numeric columns")',
            "plt.show()",
        ),
        md("`temp` and `feel_temp` correlate at **0.99** — they are the same",
           "measurement twice. Keeping both is not fatal here, but it makes the",
           "two coefficients individually meaningless: the fit can trade one",
           "against the other freely.",
           "",
           "Now look at the `hour` row: it correlates **0.39** with `count`,",
           "about the same as `temp`. Read that against plot 3b, which showed",
           "`hour` driving the target far harder than any other column. The",
           "correlation understates it badly, because correlation measures only",
           "the *linear* part of a relationship and the hour effect rises, falls,",
           "rises and falls again. A single number here cannot see a shape like",
           "that — which is why you plotted it. Section 8 puts a figure on how",
           "much this one column is really worth."),
        md("### 3e. Season and year"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))",
            'sns.barplot(data=df, x="season", y="count", ax=axes[0],',
            '            order=["spring", "summer", "fall", "winter"], color="#2f6f9f")',
            'axes[0].set_title("Rentals by season")',
            'sns.barplot(data=df, x="year", y="count", ax=axes[1], color="#2f6f9f")',
            'axes[1].set_title("Rentals by year (0 = first, 1 = second)")',
            "plt.tight_layout()",
            "plt.show()",
        ),
        md("The system grew substantially between the two years. `year` is a",
           "genuine feature here, not noise."),
        md("---", "", "## 4. Turn it into numbers",
           "",
           "`f(x)` is a function on real vectors: the text columns have to become",
           "numbers. `pd.get_dummies` one-hot encodes every non-numeric column and",
           "leaves the numeric ones alone.",
           "",
           "The `reindex` on the second line is not optional. `X_test` may not",
           "contain every category `X_train` does, so encoding the two frames",
           "independently can give them different columns. `reindex` forces the",
           "test matrix to have exactly the training columns, in the same order."),
        code(
            'X_train_enc = pd.get_dummies(X_train.drop(columns=["id"]))',
            'X_test_enc = pd.get_dummies(X_test.drop(columns=["id"])).reindex(',
            "    columns=X_train_enc.columns, fill_value=0)",
            "",
            'print("encoded:", X_train_enc.shape, "->", list(X_train_enc.columns[:6]), "...")',
            'assert list(X_train_enc.columns) == list(X_test_enc.columns)',
        ),
        md("---", "", "## 5. Fit, and measure before you submit",
           "",
           "You get one score per submission, and you should never learn anything",
           "from the leaderboard you could have learned at home. Hold out part of",
           "the training data, fit on the rest, and score yourself first."),
        code(
            "from sklearn.model_selection import train_test_split",
            "from sklearn.linear_model import LinearRegression",
            "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error",
            "import numpy as np",
            "",
            "A_tr, A_va, b_tr, b_va = train_test_split(",
            "    X_train_enc, y_train, test_size=0.2, random_state=0)",
            "",
            "model = LinearRegression().fit(A_tr, b_tr)",
            "pred_va = model.predict(A_va)",
            "",
            'print(f"R2   {r2_score(b_va, pred_va):.4f}")',
            'print(f"RMSE {np.sqrt(mean_squared_error(b_va, pred_va)):.2f}")',
            'print(f"MAE  {mean_absolute_error(b_va, pred_va):.2f}")',
        ),
        md("Read those three numbers against the only baseline that matters —",
           "predicting the training mean for every hour, which is what **R² = 0**",
           "means:"),
        code(
            "baseline = np.full(len(b_va), b_tr.mean())",
            'print(f"predict-the-mean  R2 {r2_score(b_va, baseline):.4f}  '
            'RMSE {np.sqrt(mean_squared_error(b_va, baseline)):.2f}")',
            "",
            f'print(f"negative predictions: {{(pred_va < 0).sum()}} of {{len(pred_va)}}")',
        ),
        md("So the model does carry real signal — but it also predicts a negative",
           "number of bicycles a few hundred times, exactly as plot 3a warned.",
           "That is the linear family's unbounded output space showing up in the",
           "output."),
        md("---", "", "## 6. Predict the test set and write the submission",
           "",
           "Refit on **all** the training data now — the holdout has done its job",
           "and more data is better. Then write the two required columns."),
        code(
            "model = LinearRegression().fit(X_train_enc, y_train)",
            "predictions = model.predict(X_test_enc)",
            "",
            'submission = pd.DataFrame({"id": X_test["id"], "prediction": predictions})',
            'submission.to_csv("submission.csv", index=False)',
            "",
            'print(submission.shape)',
            "submission.head()",
        ),
        md("Check the file before uploading it. One row per test id, no missing",
           "values, no duplicates:"),
        code(
            'assert len(submission) == len(X_test)',
            'assert submission["id"].is_unique',
            'assert submission["prediction"].notna().all()',
            'print("submission.csv looks well-formed")',
        ),
        md("---", "", "## 7. Submit"),
        code(
            'result = client.submit(challenge_id=CHALLENGE_ID, files=["submission.csv"])',
            "print(result)",
        ),
        code(
            "client.leaderboard(CHALLENGE_ID).head(10)",
        ),
        md("---", "", "## 8. Where to go from here",
           "",
           "This model scores about **R² = 0.40**. The challenge page lists the",
           "ladder above it; the first rung is worth far more than you would",
           "expect, and it is not a change of model:",
           "",
           "- **`hour` is being treated as a number.** The encoding says 23:00 is",
           "  twenty-three times 1:00, and that midnight is a fall of 23 from",
           "  23:00. Plot 3b showed the real shape. Try",
           "  `X_train[\"hour\"] = X_train[\"hour\"].astype(str)` before",
           "  `get_dummies` and re-run — on this split it takes R² from 0.40 to",
           "  about 0.70, with the same `LinearRegression`.",
           "- **Predictions can be negative.** `np.clip(predictions, 0, None)` is",
           "  free and cannot hurt.",
           "- **`temp` and `feel_temp` are the same column twice.** Try dropping",
           "  one.",
           "- `month` and `weekday` have the same problem as `hour`.",
           "",
           "Everything past that — trees, ensembles, tuning — is Session 3."),
    ]), name


# --------------------------------------------------------------------------- #
# 2. Bank marketing — the guided notebook (no code)
# --------------------------------------------------------------------------- #
def build_bank() -> dict:
    name = "aie-s2-bank-marketing.ipynb"
    return notebook([
        md(
            "# AIE S2 — Bank Term Deposit",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(name)})",
            "",
            "**Classification.** A Portuguese bank ran a direct marketing",
            "campaign: 45,211 calls, and for each one, whether the client",
            "subscribed a term deposit afterwards. Predict that outcome for calls",
            "you have not been shown.",
            "",
            f"Challenge: <https://ml-arena.com/viewchallenge/{BANK_ID}>",
            "",
            "---",
            "",
            "**This notebook contains no code, and that is deliberate.** You have",
            "just read the bike-demand notebook, which is the same five steps on a",
            "regression target. Here the steps are written out in English and the",
            "cells are empty. Write them yourself.",
            "",
            "Two things genuinely differ from the bike challenge, and both are",
            "flagged below where they bite: the target is a **class**, not a",
            "quantity, and it is **imbalanced** — only 11.7% of clients said yes.",
        ),
        md("---", "", "## 0. Setup",
           "",
           "Install the ML-Arena client. `pandas`, `seaborn` and `scikit-learn`",
           "are already available in Colab.",
           "",
           "The distribution is **`mlarena-sdk`** and it imports as `mlarena`.",
           "`pip install mlarena` installs an unrelated package — check you have",
           "the right one."),
        code(""),
        md("---", "", "## 1. Get the data",
           "",
           "Connect with your personal `mlk_user_...` key from your ML-Arena",
           f"Profile page, and download the dataset for challenge **{BANK_ID}**",
           "into the working directory. Same two calls as last time."),
        code(""),
        md("---", "", "## 2. Read it",
           "",
           "Load `X_train.csv`, the `prediction` column of `y_train.csv`, and",
           "`X_test.csv`.",
           "",
           "Then answer three questions before going further:",
           "",
           "- What are **n** and **p**?",
           "- Which columns are numeric and which are text?",
           "- **What fraction of the training target is 1?** Compute it now. This",
           "  number decides your metric and your reading of every result that",
           "  follows."),
        code(""),
        md("---", "", "## 3. Look at it",
           "",
           "Aim for five or six plots. For each one, ask what decision it changes",
           "— a plot that changes nothing was not worth making.",
           "",
           "**3a. The target.** Count the two classes. Confirm the imbalance you",
           "computed above, and work out what accuracy you would get by always",
           "predicting the majority class. Write that number down; it is the",
           "number your model has to be judged against.",
           "",
           "**3b. A numeric feature against the target.** Compare the",
           "distribution of `duration` (call length in seconds) for clients who",
           "subscribed and clients who did not. A boxplot or a pair of histograms",
           "will do. The separation is large — then read the note in step 4 about",
           "why that should make you uneasy rather than pleased.",
           "",
           "**3c. A categorical feature against the target.** Plot the",
           "**subscription rate** — not the raw count — per level of `poutcome`",
           "(the outcome of the previous campaign), `month`, and `job`. Raw counts",
           "will mostly tell you which levels are common; the rate is what tells",
           "you which are predictive. `groupby(col)[target].mean()` gives it to",
           "you directly.",
           "",
           "**3d. Age.** Plot the subscription rate against age in bins. It is not",
           "monotone. Note what that means for a model that multiplies age by a",
           "single coefficient.",
           "",
           "**3e. Correlation.** Heatmap the numeric columns. Check whether any",
           "pair is nearly redundant.",
           "",
           "**3f. `pdays`.** Look at its distribution. The value `-1` is a flag",
           "meaning \"never previously contacted\", not a quantity — so the column",
           "mixes a category and a measurement in one place. Decide what to do",
           "about it, and be able to say why."),
        code(""),
        code(""),
        code(""),
        md("---", "", "## 4. Turn it into numbers",
           "",
           "Same problem as before: nine of the sixteen columns are text. One-hot",
           "encode them, and align the test columns to the training ones — an",
           "encoding done independently on the two frames will not match.",
           "",
           "**Then scale the numeric columns.** This is the step the bike notebook",
           "did not need. `balance` runs from −8,019 to 102,127 while `campaign`",
           "runs from 1 to 63; the solver behind `LogisticRegression` will not",
           "converge in a sensible number of iterations if you hand it both",
           "untouched. `StandardScaler` fitted on the training matrix is enough.",
           "",
           "**A decision to make here.** `duration` is the length of the call you",
           "are predicting the outcome of. You only know it once the call has",
           "ended — so a model that relies on it could never run *before* placing",
           "the call. It is in the data and you may use it. Decide whether you",
           "want to, and say why in a comment. Both answers are defensible; an",
           "unexamined answer is not."),
        code(""),
        md("---", "", "## 5. Fit, and measure before you submit",
           "",
           "Hold out 20% of the training data, fit `LogisticRegression` on the",
           "rest, and score yourself on the holdout.",
           "",
           "Report **four** numbers: accuracy, precision, recall and F1. Then:",
           "",
           "- Compare your accuracy with the always-predict-0 accuracy from step",
           "  3a. If they are close, your accuracy is telling you nothing.",
           "- Look at precision against recall. A model that almost never predicts",
           "  1 will have decent precision and terrible recall, and F1 is the",
           "  number that refuses to let you ignore that.",
           "- A confusion matrix makes all four legible at once. Plot it.",
           "",
           "The challenge ranks on **F1 on the positive class**, for exactly the",
           "reason step 3a made visible."),
        code(""),
        code(""),
        md("---", "", "## 6. The threshold is yours",
           "",
           "`predict()` labels a client as 1 when the predicted probability",
           "exceeds 0.5. That 0.5 is a convention, not a property of the problem,",
           "and on a target that is 11.7% positive it is rarely the best choice.",
           "",
           "Use `predict_proba()` to get the probabilities, then:",
           "",
           "- Sweep the threshold from 0.05 to 0.60 and plot F1 against it on your",
           "  holdout.",
           "- Read off the best one and keep it.",
           "",
           "On this dataset the move is worth more than changing model family.",
           "Do this on the **holdout**, never on the leaderboard."),
        code(""),
        md("---", "", "## 7. Predict the test set and write the submission",
           "",
           "Refit on all the training data, predict `X_test`, and apply the",
           "threshold you chose.",
           "",
           "The submission is `submission.csv` with columns `id` and `prediction`,",
           "one row per test id. **`prediction` must be the integer 0 or 1** — a",
           "probability is rejected outright rather than thresholded for you.",
           "",
           "Assert the three things before uploading: one row per test id, ids",
           "unique, and every value in {0, 1}."),
        code(""),
        md("---", "", "## 8. Submit",
           "",
           "Submit the file and look at the leaderboard. Compare your score with",
           "the baseline quoted on the challenge page (F1 = 0.4528). If you are",
           "below it, the most likely reasons in order: you left the threshold at",
           "0.5, you did not scale, or your test columns are misaligned with your",
           "training columns."),
        code(""),
        md("---", "", "## 9. Write down what you did",
           "",
           "Two or three sentences, in the notebook:",
           "",
           "- Which metric you are reporting and why that one.",
           "- What you did about `duration`, and why.",
           "- What threshold you chose, and what it cost you in precision to buy",
           "  that recall.",
           "",
           "Being able to answer those is the point of the exercise. The score is",
           "just the receipt."),
    ]), name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    global _COUNTER
    for builder in (build_bike, build_bank):
        _COUNTER = itertools.count()
        nb, name = builder()
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  wrote {path.relative_to(REPO)}  ({n_md} md + {n_code} code cells)")
        print(f"        {colab_url(name)}")


if __name__ == "__main__":
    main()
