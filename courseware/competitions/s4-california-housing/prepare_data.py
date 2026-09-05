#!/usr/bin/env python3
"""Build the California-housing split for the Session 4 regression challenge.

Public  -> data/X_train.csv, data/y_train.csv, data/X_test.csv
Private -> data/y_test.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (plain LinearRegression — the bar the
             MLP has to clear, NOT what the worked notebook produces)

The benchmark is deliberately *not* the notebook's own model, which is the
break from Sessions 2 and 3. A torch training run is reproducible on one
machine and not across machines — different BLAS, different thread counts — so
pinning a float the notebook must hit to 1e-6 would be a lie the test suite
would eventually catch. The benchmark is therefore the linear model, i.e. the
best Session 2 could do on this data, and the whole point of Session 4 is to
beat it. `test_challenges.py` asserts the notebook clears
`notebook_expected_min_score` rather than equalling the benchmark.

Reference numbers on this split, measured (see the overview's table):

    predict the training mean               test R2  0.000
    LinearRegression                        test R2  0.576   <- the benchmark
    MLP 8-64-64-1, standardised, Adam 60ep  test R2  0.785

That is a bigger jump than anything in Sessions 2 or 3, and it is the reason
this dataset was chosen: the relationship between the features and the price is
genuinely non-linear (geography alone is two coordinates that only matter
jointly), so capacity buys something real here rather than memorising.

Note the target: median house value in units of $100,000, **capped at 5.0**.
992 of the 20,640 districts sit exactly on that cap, which is visible as a
spike in the histogram and is worth pointing at rather than hiding.

Deterministic: one random 80/20 split at SEED. The benchmark is fitted on the
CSVs as written and read back, so the declared score is reachable through
exactly the path a student's `pd.read_csv` takes.

    python prepare_data.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# sklearn's names are terse and CamelCase; these are the same columns, spelled
# the way the rest of the course spells columns.
RENAME = {
    "MedInc": "median_income",          # tens of thousands of dollars
    "HouseAge": "house_age",            # median age of the houses, years
    "AveRooms": "avg_rooms",            # rooms per household
    "AveBedrms": "avg_bedrooms",        # bedrooms per household
    "Population": "population",         # people in the block group
    "AveOccup": "avg_occupancy",        # household members per household
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def _assert_mapping(X, y):
    """The rename is only safe if the columns still mean what we named them."""
    checks = [
        ("median_income in tens of k$", 0.4 < X["median_income"].min() and X["median_income"].max() <= 15.1),
        ("house_age in years", 1 <= X["house_age"].min() and X["house_age"].max() <= 52),
        ("avg_bedrooms < avg_rooms", (X["avg_bedrooms"] < X["avg_rooms"]).all()),
        ("latitude is California", 32 < X["latitude"].min() and X["latitude"].max() < 42.5),
        ("longitude is California", -125 < X["longitude"].min() and X["longitude"].max() < -114),
        ("target capped at 5", abs(y.max() - 5.0) < 1e-9),
        ("no missing values", not X.isna().any().any()),
    ]
    bad = [n for n, ok in checks if not ok]
    if bad:
        raise SystemExit(f"column rename failed its sanity check for: {bad}")
    print(f"  column mapping verified ({len(checks)} checks)")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("loading sklearn california_housing …")
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X = X.rename(columns=RENAME).reset_index(drop=True)
    y = y.reset_index(drop=True)
    _assert_mapping(X, y)
    print(f"  {len(X)} districts, {X.shape[1]} features, "
          f"target {y.min():.2f}-{y.max():.2f} mean {y.mean():.3f} "
          f"({int((y >= 5.0).sum())} at the cap)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train.insert(0, "id", [f"tr_{i:05d}" for i in range(len(X_train))])
    X_test.insert(0, "id", [f"te_{i:05d}" for i in range(len(X_test))])

    X_train.to_csv(os.path.join(DATA, "X_train.csv"), index=False)
    pd.DataFrame({"id": X_train["id"], "prediction": y_train}).to_csv(
        os.path.join(DATA, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA, "X_test.csv"), index=False)
    pd.DataFrame({"id": X_test["id"], "prediction": y_test}).to_csv(
        os.path.join(DATA, "y_test.csv"), index=False)
    print(f"  train {len(X_train)} rows / test {len(X_test)} rows")

    print("fitting the baseline (LinearRegression) on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X_train.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_test.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y_train.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_test.csv"))["prediction"]

    Xtr = tr.drop(columns=["id"])
    Xte = te.drop(columns=["id"])
    preds = LinearRegression().fit(Xtr, ytr).predict(Xte)

    pd.DataFrame({"id": te["id"], "prediction": np.round(preds, 6)}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    written = pd.read_csv(os.path.join(DATA, "benchmark_submission.csv"))["prediction"]
    resid = yte - written
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(resid.abs().mean())
    mean_rmse = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
    print(f"  benchmark  R2={r2:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
    print(f"  reference  predict-the-mean RMSE={mean_rmse:.4f}  (R2 = 0)")
    print(f"\n  set benchmark_expected_score = {round(r2, 6)}")
    print(f"\nwrote {DATA}")


if __name__ == "__main__":
    main()
