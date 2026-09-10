#!/usr/bin/env python3
"""Build the credit-risk split for the Session 3 classification challenge.

Public  -> data/X.csv, data/y.csv, data/X_submission.csv
Private -> data/y_submission.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (get_dummies + scaler + LogisticRegression)

The guided half of Session 3, and it carries the same lesson as the worked one
in a form the student has to find alone: on these 700 training rows a random
forest reaches **exactly 1.000 training accuracy** — it memorises every
applicant — and still loses to logistic regression on the held-out third.

Reference numbers on this split:

    LogisticRegression    train acc 0.7929   TEST acc 0.7700   TEST F1 0.5714
    RandomForest(n=500)   train acc 1.0000   TEST acc 0.7567   TEST F1 0.4823

A training accuracy of 1.000 is not a good model. It is a model that has learned
the answer key.

The target is coded so that **1 = bad credit risk** — the minority class (30%)
and the one that costs money to miss, which is why the challenge ranks on F1.

Deterministic: one stratified 70/30 split at SEED. The benchmark is fitted on
the CSVs as written and read back.

    python prepare_data.py
"""
import os

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.3
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml credit-g (version 1) …")
    bunch = fetch_openml("credit-g", version=1, as_frame=True)
    X = bunch.data.reset_index(drop=True)
    target = bunch.target.astype(str).reset_index(drop=True)
    if set(target.unique()) != {"good", "bad"}:
        raise SystemExit(f"unexpected target levels: {target.unique()}")
    # 1 = bad credit risk: the minority class and the expensive error.
    y = (target == "bad").astype(int)
    print(f"  {len(X)} rows, {X.shape[1]} features, bad-risk rate {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
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
    print(f"  test bad-risk rate {y_test.mean():.4f}")

    print("fitting the reference baseline on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_submission.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_submission.csv"))["prediction"]

    Xtr = pd.get_dummies(tr.drop(columns=["id"]))
    Xte = pd.get_dummies(te.drop(columns=["id"])).reindex(
        columns=Xtr.columns, fill_value=0)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    preds = model.predict(scaler.transform(Xte))

    pd.DataFrame({"id": te["id"], "prediction": preds}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    written = pd.read_csv(os.path.join(DATA, "benchmark_submission.csv"))["prediction"]
    tp = int(((written == 1) & (yte == 1)).sum())
    fp = int(((written == 1) & (yte == 0)).sum())
    fn = int(((written == 0) & (yte == 1)).sum())
    tn = int(((written == 0) & (yte == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(yte)
    print(f"  benchmark  F1={f1:.6f} accuracy={acc:.6f} "
          f"precision={precision:.4f} recall={recall:.4f}")
    print(f"  reference  always-0 -> accuracy={1 - yte.mean():.4f}, F1=0.0000")
    print(f"\n  set benchmark_expected_score = {round(f1, 6)}")
    print(f"\nwrote {DATA}")


if __name__ == "__main__":
    main()
