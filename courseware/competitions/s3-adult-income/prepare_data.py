#!/usr/bin/env python3
"""Build the Adult (Census Income) split for the Session 3 competition.

Public  -> data/X_train.csv, data/y_train.csv, data/X_test.csv
Private -> data/y_test.csv          (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv   (the logistic-regression pipeline
             from Lab 3 Part C, so the creator-side benchmark scores something
             a student can actually beat)

Deterministic: one stratified 80/20 split at SEED. Re-running reproduces the
exact same files, so the leaderboard stays comparable across rebuilds.

    python prepare_data.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml adult (version 2) …")
    bunch = fetch_openml("adult", version=2, as_frame=True)
    X = bunch.data.reset_index(drop=True)
    y = (bunch.target == ">50K").astype(int).reset_index(drop=True)
    print(f"  {len(X)} rows, {X.shape[1]} features, positive rate {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    # Fresh contiguous ids. The original openml row order is discarded so a
    # submission cannot be reconstructed by index alignment against the source.
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
    print(f"  test positive rate {y_test.mean():.4f}")

    print("fitting the Lab 3 Part C baseline for the benchmark submission …")
    features_train = X_train.drop(columns=["id"])
    features_test = X_test.drop(columns=["id"])
    num = features_train.select_dtypes("number").columns
    cat = features_train.select_dtypes(exclude="number").columns
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(features_train, y_train)
    preds = pipe.predict(features_test)
    pd.DataFrame({"id": X_test["id"], "prediction": preds}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    acc = float((preds == y_test).mean())
    print(f"  benchmark F1={f1:.6f} accuracy={acc:.6f}")
    print(f"\nwrote {DATA}")
    print("  public : X_train.csv y_train.csv X_test.csv")
    print("  private: y_test.csv")
    print("  bench  : benchmark_submission.csv")


if __name__ == "__main__":
    main()
