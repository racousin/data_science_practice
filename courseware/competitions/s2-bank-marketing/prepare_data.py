#!/usr/bin/env python3
"""Build the bank-marketing split for the Session 2 classification challenge.

Public  -> data/X_train.csv, data/y_train.csv, data/X_test.csv
Private -> data/y_test.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (get_dummies + StandardScaler +
             LogisticRegression — the reference baseline the overview quotes)

openml serves this dataset with its columns anonymised to V1..V16. They are the
UCI `bank-full.csv` columns in order, and the mapping is verifiable from the
data itself rather than taken on trust: V1 ranges 18-95 (age), V6 spans
-8019..102127 (balance), V10 is 1-31 (day), V11 carries the twelve month
abbreviations, V12 is 0-4918 seconds (duration), V16 is the four poutcome
levels. Restoring the names is what makes the dataset explorable at all.

Deterministic: one stratified 80/20 split at SEED. The benchmark is fitted on
the CSVs as written and read back, so the declared score is reachable through
exactly the path a student's `pd.read_csv` takes.

    python prepare_data.py
"""
import os

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# UCI bank-full.csv column order, restored over openml's V1..V16.
NAMES = ["age", "job", "marital", "education", "default", "balance", "housing",
         "loan", "contact", "day", "month", "duration", "campaign", "pdays",
         "previous", "poutcome"]


def _assert_mapping(X):
    """The rename is only safe if the data still looks like what we named it."""
    checks = [
        ("age", X["age"].min() >= 18 and X["age"].max() <= 95),
        ("balance", X["balance"].min() < -1000 and X["balance"].max() > 50000),
        ("day", set(X["day"].unique()) <= set(range(1, 32))),
        ("month", "may" in set(X["month"].astype(str)) and X["month"].nunique() == 12),
        ("duration", X["duration"].min() >= 0 and X["duration"].max() > 4000),
        ("poutcome", {"success", "failure", "unknown"} <= set(X["poutcome"].astype(str))),
        ("marital", {"married", "single", "divorced"} == set(X["marital"].astype(str))),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        raise SystemExit(f"column rename failed its sanity check for: {bad}")
    print(f"  column mapping verified ({len(checks)} checks)")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml bank-marketing (version 1) …")
    bunch = fetch_openml("bank-marketing", version=1, as_frame=True)
    X = bunch.data.reset_index(drop=True)
    if list(X.columns) != [f"V{i}" for i in range(1, 17)]:
        raise SystemExit(f"unexpected openml columns: {list(X.columns)}")
    X.columns = NAMES
    _assert_mapping(X)
    # openml codes the target as '1'/'2'; '2' is the minority class and the UCI
    # positive rate is 11.7%, which identifies it as "yes, subscribed".
    y = (bunch.target.astype(str) == "2").astype(int).reset_index(drop=True)
    print(f"  {len(X)} rows, {X.shape[1]} features, positive rate {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
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
    print(f"  test positive rate {y_test.mean():.4f}")

    print("fitting the reference baseline on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X_train.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_test.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y_train.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_test.csv"))["prediction"]

    Xtr = pd.get_dummies(tr.drop(columns=["id"]))
    Xte = pd.get_dummies(te.drop(columns=["id"])).reindex(
        columns=Xtr.columns, fill_value=0)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
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
