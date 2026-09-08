#!/usr/bin/env python3
"""Build the bike-sharing split for the Session 2 regression challenge.

Public  -> data/X_train.csv, data/y_train.csv, data/X_test.csv
Private -> data/y_test.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (the notebook's baseline: impute +
             get_dummies + LinearRegression, no tuning — so the declared
             benchmark is exactly what a student following the worked notebook
             produces)

**The split is chronological, not random.** The openml table is one row per
hour in calendar order (verified: `year`*100+`month` is monotone and `hour`
cycles 0-23 over 729 day boundaries), so a random split puts 20:00 in the test
set while 19:00 and 21:00 stay in training — adjacent hours whose weather is the
same reading and whose count is within a few bikes. A model can then interpolate
what it is supposed to forecast, and every score is optimistic. The first 80% of
the hours train, the last 20% are held out: the second year from mid-August on.
That is the same shape as the problem the operator actually has.

The cost is that the test set contains no summer and no `heavy_rain` hour. That
is not a defect of the split — it is what forecasting forward means, and
`reindex(columns=..., fill_value=0)` in the baseline is what keeps the encoding
aligned through it.

Deterministic given `MLARENA_ID_SALT`: the chronological cut, then opaque ids
(`../_dataset_ids.py`). Re-running under the same salt reproduces the same files
byte for byte, so the leaderboard stays comparable across rebuilds; under a
different salt every id changes.

**Train ships in calendar order; test ships shuffled.** The shuffle exists to
stop the reproducible split being replayed onto the published ids, and what it
protects is `y_test` — the training targets are handed out in `y_train.csv`, so
permuting them defended nothing while destroying the one thing a student needs
in order to carve a validation set the honest way (the *last* rows, not random
ones). `X_test` is still shuffled, so a row's published id says nothing about
its position in the reproducible cut.

The id key is `train-v2` / `test-v2`. Bumping it was deliberate: the rows behind
every id changed when the split went chronological, so a submission written
against the old `X_test.csv` is nonsense against the new one. Under the old key
it would still have validated — same 3,476 ids, every one present — and scored
silently. It now fails on the first check with "missing 3476 of 3476 test ids",
which is the error a student can act on.

**Missing values are punched in on purpose** (`MISSING`, `punch_holes`). The
source table has none, and a first challenge whose CSV is already a clean
rectangle teaches that data arrives that way. Session 2's Data Preparation
lesson spends a third of its time on missingness; this is what makes a student
run those lines rather than read them. The mask is seeded from a *public*
constant, unlike the ids: which cells are blank is visible in the file anyway,
so there is nothing to keep secret, and reproducibility is the only property
needed. Two *mechanisms* — one block, two scatters — because the lesson teaches
reading those apart off `sns.heatmap(df.isna())`:

  temp + feel_temp   3% of rows, the SAME rows — a sensor down for a while
  windspeed          5% of rows, scattered — dropped readings
  weather            2% of rows, scattered — the one categorical hole

The benchmark is fitted on the CSVs **as written and read back**, not on the
in-memory frame. pandas re-infers dtypes on read (the True/False columns come
back as bool), and the baseline must be computed through exactly the path a
student's `pd.read_csv` takes or the declared score would not be reachable.

    python prepare_data.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

TEST_SIZE = 0.2
# Public on purpose — see the module docstring. Not the id salt.
MISSING_SEED = 20260908
# Scattered holes: (column, fraction of rows). Independent draws per column.
MISSING_SCATTER = [("windspeed", 0.05), ("weather", 0.02)]
# The block: temp and feel_temp go together, in whole contiguous runs of hours,
# because that is what a thermometer being down looks like. Punched before the
# test split is shuffled, so the co-occurrence survives into both files and the
# run structure is legible in X_train, which ships in calendar order.
MISSING_OUTAGE = (["temp", "feel_temp"], 0.03, 60)   # columns, fraction, run length
# Mixed into the mask seed so the two splits do not lose the same row positions.
# A dict rather than a hash of the name: `hash()` is salted per process.
SPLIT_OFFSET = {"train": 0, "test": 1}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DATASET = "s2-bike-demand"

sys.path.insert(0, os.path.dirname(HERE))
from _dataset_ids import shuffle_and_label  # noqa: E402


def punch_holes(X, split):
    """Blank `MISSING_*` into a copy of X, reproducibly and per split.

    Called on the chronologically-ordered split, before the test half is
    shuffled: an outage is a run of consecutive *hours*, so it has to be drawn
    against calendar order to mean anything. The split name is mixed into the
    seed so train and test do not lose the same row positions.
    """
    X = X.copy()
    rng = np.random.default_rng([MISSING_SEED, SPLIT_OFFSET[split]])
    n = len(X)

    for col, frac in MISSING_SCATTER:
        rows = rng.choice(n, size=int(round(n * frac)), replace=False)
        X.loc[X.index[rows], col] = np.nan

    columns, frac, run = MISSING_OUTAGE
    starts = rng.choice(n - run, size=int(round(n * frac / run)), replace=False)
    rows = np.unique(np.concatenate([np.arange(s, s + run) for s in starts]))
    for col in columns:
        X.loc[X.index[rows], col] = np.nan
    return X


def baseline_matrix(train, test):
    """The worked notebook's step 4, exactly: impute with statistics learned on
    the training frame, one-hot encode, align the test columns onto the train's.

    Kept in one function because `prepare_data.py` and the notebook must produce
    the same matrix — `benchmark_expected_score` is asserted to be reachable by
    a student who followed the notebook, and imputing the test set with its own
    median would quietly break that (and teach leakage besides).
    """
    # `fillna(medians)` names the numeric columns and so touches only those; the
    # bare `fillna("unknown")` then catches whatever NaN is left, which by that
    # point is the text columns only. Two calls rather than a dtype scan for the
    # categoricals, because `select_dtypes("object")` does not mean the same
    # thing across pandas 2 and 3 and this has to run identically in Colab.
    medians = train.median(numeric_only=True)
    enc_train = pd.get_dummies(train.fillna(medians).fillna("unknown"))
    enc_test = pd.get_dummies(test.fillna(medians).fillna("unknown")).reindex(
        columns=enc_train.columns, fill_value=0)
    return enc_train, enc_test


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml Bike_Sharing_Demand (version 2) …")
    bunch = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
    X = bunch.data.reset_index(drop=True)
    y = bunch.target.astype(int).reset_index(drop=True)
    print(f"  {len(X)} rows, {X.shape[1]} features, "
          f"count min={y.min()} max={y.max()} mean={y.mean():.1f}")

    # Past predicts future. The rows are already in calendar order, so the cut
    # is positional; see the module docstring for why a random split is wrong
    # here. Asserted rather than assumed — a future openml revision that ships
    # the same rows sorted differently would silently reintroduce the leak.
    order = X["year"].astype(int) * 100 + X["month"].astype(int)
    if not (order.diff().dropna() >= 0).all():
        raise SystemExit(
            "openml returned the bike rows out of calendar order; the "
            "chronological split below assumes row position IS time.")
    cut = int(len(X) * (1 - TEST_SIZE))
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_test, y_test = X.iloc[cut:], y.iloc[cut:]
    print(f"  cut at row {cut}: train ends year {X_train['year'].iloc[-1]} "
          f"month {X_train['month'].iloc[-1]}, test is year "
          f"{X_test['year'].iloc[0]} months "
          f"{sorted(X_test['month'].unique().tolist())}")

    X_train, X_test = punch_holes(X_train, "train"), punch_holes(X_test, "test")

    # Opaque ids. The cut is trivially reproducible — it is two slices — so a
    # sequential id over it would hand back y_test to anyone who read this file.
    # Train keeps calendar order (`shuffle=False`): its targets are published
    # anyway, and the order is what lets a student hold out the last hours
    # rather than random ones. See ../_dataset_ids.py.
    X_train, y_train = shuffle_and_label(
        X_train, y_train, dataset=DATASET, split="train-v2", prefix="tr",
        shuffle=False)
    X_test, y_test = shuffle_and_label(
        X_test, y_test, dataset=DATASET, split="test-v2", prefix="te")
    for name, frame in (("train", X_train), ("test", X_test)):
        holes = frame.isna().sum()
        print(f"  {name} missing: "
              + ", ".join(f"{c}={v}" for c, v in holes[holes > 0].items()))

    X_train.to_csv(os.path.join(DATA, "X_train.csv"), index=False)
    pd.DataFrame({"id": X_train["id"], "prediction": y_train}).to_csv(
        os.path.join(DATA, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA, "X_test.csv"), index=False)
    pd.DataFrame({"id": X_test["id"], "prediction": y_test}).to_csv(
        os.path.join(DATA, "y_test.csv"), index=False)
    print(f"  train {len(X_train)} rows / test {len(X_test)} rows")

    print("fitting the baseline (impute + get_dummies + LinearRegression) …")
    tr = pd.read_csv(os.path.join(DATA, "X_train.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_test.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y_train.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_test.csv"))["prediction"]

    Xtr, Xte = baseline_matrix(tr.drop(columns=["id"]), te.drop(columns=["id"]))
    preds = LinearRegression().fit(Xtr, ytr).predict(Xte)

    pd.DataFrame({"id": te["id"], "prediction": np.round(preds, 6)}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    # Scored the way env.py scores it, from the rounded file that is uploaded.
    written = pd.read_csv(os.path.join(DATA, "benchmark_submission.csv"))["prediction"]
    resid = yte - written
    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(resid.abs().mean())
    mean_mae = float((yte - ytr.mean()).abs().mean())
    print(f"  benchmark  -MAE={-mae:.6f}  RMSE={rmse:.6f}")
    print(f"  reference  predict-the-mean -MAE={-mean_mae:.4f}")

    # The overview's third rung, and the payoff of the lesson's feature
    # engineering section: the same model with `hour` read as 24 categories.
    tr_h, te_h = tr.copy(), te.copy()
    tr_h["hour"] = tr_h["hour"].astype(str)
    te_h["hour"] = te_h["hour"].astype(str)
    Htr, Hte = baseline_matrix(tr_h.drop(columns=["id"]), te_h.drop(columns=["id"]))
    hour_preds = LinearRegression().fit(Htr, ytr).predict(Hte)
    print(f"  reference  hour one-hot     -MAE="
          f"{-float(np.abs(yte - hour_preds).mean()):.4f}")

    print(f"\n  set benchmark_expected_score = {round(-mae, 6)}")
    print(f"\nwrote {DATA}")
    print("  public : X_train.csv y_train.csv X_test.csv")
    print("  private: y_test.csv")
    print("  bench  : benchmark_submission.csv")


if __name__ == "__main__":
    main()
