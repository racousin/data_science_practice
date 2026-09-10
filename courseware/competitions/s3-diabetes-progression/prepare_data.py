#!/usr/bin/env python3
"""Build the diabetes-progression split for the Session 3 regression challenge.

Public  -> data/X.csv, data/y.csv, data/X_submission.csv
Private -> data/y_submission.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (plain LinearRegression — the honest
             baseline the worked notebook ends up defending)

This dataset is chosen for one property, and the whole session hangs off it: a
heavy random forest fits the training set almost perfectly and *still loses to a
straight line* on data it has not seen. n is small (265 training rows), the
signal is largely linear, and the noise is real, which is exactly the regime
where training score stops being evidence.

Reference numbers on this split, for the overview and the notebook:

    LinearRegression      train R2 0.5072   5-fold CV 0.4273   TEST 0.5157
    RandomForest(n=500)   train R2 0.9201   5-fold CV 0.3584   TEST 0.4922

The forest's 0.92 is the lie; its CV score saw through it without ever touching
the test set.

`scaled=False` is deliberate — sklearn's default hands back mean-centred,
norm-scaled columns that no one can plot or interpret. Raw gives age in years,
a real BMI and an actual cholesterol panel.

Deterministic: one random 60/40 split at SEED. The benchmark is fitted on the
CSVs as written and read back, so the declared score is reachable through
exactly the path a student's `pd.read_csv` takes.

    python prepare_data.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.4
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# sklearn ships this dataset with the clinical panel abbreviated to s1..s6.
# The mapping is documented upstream and re-checked below from the values.
RENAME = {
    "bp": "blood_pressure",
    "s1": "total_cholesterol",   # tc
    "s2": "ldl",                 # low-density lipoprotein
    "s3": "hdl",                 # high-density lipoprotein
    "s4": "cholesterol_ratio",   # tch = total cholesterol / HDL
    "s5": "log_triglycerides",   # ltg
    "s6": "glucose",             # glu
}


def _assert_mapping(X):
    """The rename is only safe if the columns still mean what we named them."""
    ratio = X["total_cholesterol"] / X["hdl"]
    checks = [
        ("cholesterol_ratio == tc/hdl", np.corrcoef(ratio, X["cholesterol_ratio"])[0, 1] > 0.95),
        ("age in years", 18 <= X["age"].min() and X["age"].max() <= 100),
        ("bmi plausible", 15 <= X["bmi"].min() and X["bmi"].max() <= 50),
        ("blood_pressure plausible", 50 <= X["blood_pressure"].min() and X["blood_pressure"].max() <= 200),
        ("log_triglycerides is a log", X["log_triglycerides"].max() < 10),
        ("hdl < total_cholesterol", (X["hdl"] < X["total_cholesterol"]).all()),
    ]
    bad = [n for n, ok in checks if not ok]
    if bad:
        raise SystemExit(f"column rename failed its sanity check for: {bad}")
    print(f"  column mapping verified ({len(checks)} checks)")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("loading sklearn diabetes (scaled=False) …")
    X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
    X = X.rename(columns=RENAME).reset_index(drop=True)
    y = y.reset_index(drop=True)
    _assert_mapping(X)
    print(f"  {len(X)} rows, {X.shape[1]} features, "
          f"target {y.min():.0f}-{y.max():.0f} mean {y.mean():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train.insert(0, "id", [f"tr_{i:05d}" for i in range(len(X_train))])
    X_test.insert(0, "id", [f"te_{i:05d}" for i in range(len(X_test))])

    X_train.to_csv(os.path.join(DATA, "X.csv"), index=False)
    pd.DataFrame({"id": X_train["id"], "prediction": y_train}).to_csv(
        os.path.join(DATA, "y.csv"), index=False)
    X_test.to_csv(os.path.join(DATA, "X_submission.csv"), index=False)
    pd.DataFrame({"id": X_test["id"], "prediction": y_test}).to_csv(
        os.path.join(DATA, "y_submission.csv"), index=False)
    print(f"  train {len(X_train)} rows / test {len(X_test)} rows")

    print("fitting the baseline (LinearRegression) on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_submission.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_submission.csv"))["prediction"]

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
