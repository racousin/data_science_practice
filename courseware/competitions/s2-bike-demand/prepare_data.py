#!/usr/bin/env python3
"""Build the bike-sharing split for the Session 2 regression challenge.

Public  -> data/X_train.csv, data/y_train.csv, data/X_test.csv
Private -> data/y_test.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (the notebook's baseline: get_dummies
             + LinearRegression, no tuning — so the declared benchmark is exactly
             what a student following the worked notebook produces)

Deterministic given `MLARENA_ID_SALT`: one 80/20 split at SEED, then a
salt-derived shuffle and opaque ids (`../_dataset_ids.py`). Re-running under the
same salt reproduces the same files byte for byte, so the leaderboard stays
comparable across rebuilds; under a different salt every id changes. SEED is
public — this file is in a public repo — and the salt is what stops the split
from being replayed straight onto the ids.

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
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DATASET = "s2-bike-demand"

sys.path.insert(0, os.path.dirname(HERE))
from _dataset_ids import shuffle_and_label  # noqa: E402


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml Bike_Sharing_Demand (version 2) …")
    bunch = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
    X = bunch.data.reset_index(drop=True)
    y = bunch.target.astype(int).reset_index(drop=True)
    print(f"  {len(X)} rows, {X.shape[1]} features, "
          f"count min={y.min()} max={y.max()} mean={y.mean():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    # Salted shuffle + opaque ids. SEED is public — this file is committed to a
    # public repository — so a sequential id over the split's own order would
    # hand back y_test to anyone who re-ran the two lines above. See
    # ../_dataset_ids.py.
    X_train, y_train = shuffle_and_label(
        X_train, y_train, dataset=DATASET, split="train", prefix="tr")
    X_test, y_test = shuffle_and_label(
        X_test, y_test, dataset=DATASET, split="test", prefix="te")

    X_train.to_csv(os.path.join(DATA, "X_train.csv"), index=False)
    pd.DataFrame({"id": X_train["id"], "prediction": y_train}).to_csv(
        os.path.join(DATA, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA, "X_test.csv"), index=False)
    pd.DataFrame({"id": X_test["id"], "prediction": y_test}).to_csv(
        os.path.join(DATA, "y_test.csv"), index=False)
    print(f"  train {len(X_train)} rows / test {len(X_test)} rows")

    print("fitting the baseline (get_dummies + LinearRegression) on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X_train.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_test.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y_train.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_test.csv"))["prediction"]

    Xtr = pd.get_dummies(tr.drop(columns=["id"]))
    Xte = pd.get_dummies(te.drop(columns=["id"])).reindex(
        columns=Xtr.columns, fill_value=0)
    preds = LinearRegression().fit(Xtr, ytr).predict(Xte)

    pd.DataFrame({"id": te["id"], "prediction": np.round(preds, 6)}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    # Scored the way env.py scores it, from the rounded file that is uploaded.
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
    print("  public : X_train.csv y_train.csv X_test.csv")
    print("  private: y_test.csv")
    print("  bench  : benchmark_submission.csv")


if __name__ == "__main__":
    main()
