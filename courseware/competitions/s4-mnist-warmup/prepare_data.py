#!/usr/bin/env python3
"""Build the MNIST warm-up split for the Session 4 competition.

Public  -> data/X_test.csv          5,000 unlabelled images to predict
           data/sample_train.csv    2,000 labelled images, as a format reference
Private -> data/y_test.csv          the held-back labels
Benchmark -> data/benchmark_submission.csv   multinomial logistic regression on
             raw pixels, i.e. a bar an MLP should clear comfortably.

Pixel convention, and it matters: columns `p0 … p783` are the 28x28 image
flattened row-major as uint8 0-255 — exactly what `torchvision.datasets.MNIST`
hands you *before* `ToTensor()` divides by 255. Lab 4 trains on the full
torchvision train split; `sample_train.csv` is here so you can assert your
loader agrees with ours before you trust a submission.

The evaluation rows are drawn from the whole 70k MNIST corpus with a private
seed and shuffled, so they line up with no published ordering.

    python prepare_data.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

SEED = 20260907          # private: fixes which rows are held out
N_TEST = 5000
N_SAMPLE_TRAIN = 2000
N_BENCHMARK_FIT = 20000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PIXELS = [f"p{i}" for i in range(784)]


def main():
    os.makedirs(DATA, exist_ok=True)

    print("fetching openml mnist_784 (this pulls ~15 MB) …")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)
    print(f"  {X.shape[0]} images, {X.shape[1]} pixels, labels {sorted(set(y.tolist()))}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X))
    test_idx = order[:N_TEST]
    rest_idx = order[N_TEST:]
    sample_idx = rest_idx[:N_SAMPLE_TRAIN]
    fit_idx = rest_idx[:N_BENCHMARK_FIT]

    test_ids = [f"te_{i:05d}" for i in range(N_TEST)]
    X_test = pd.DataFrame(X[test_idx], columns=PIXELS)
    X_test.insert(0, "id", test_ids)
    X_test.to_csv(os.path.join(DATA, "X_test.csv"), index=False)
    pd.DataFrame({"id": test_ids, "label": y[test_idx]}).to_csv(
        os.path.join(DATA, "y_test.csv"), index=False)

    sample = pd.DataFrame(X[sample_idx], columns=PIXELS)
    sample.insert(0, "label", y[sample_idx])
    sample.insert(0, "id", [f"sm_{i:05d}" for i in range(N_SAMPLE_TRAIN)])
    sample.to_csv(os.path.join(DATA, "sample_train.csv"), index=False)

    label_counts = pd.Series(y[test_idx]).value_counts().sort_index().to_dict()
    print(f"  test {N_TEST} rows, class counts {label_counts}")
    print(f"  sample_train {N_SAMPLE_TRAIN} rows")

    print(f"fitting the benchmark (logistic regression on {N_BENCHMARK_FIT} images) …")
    clf = LogisticRegression(max_iter=200)
    clf.fit(X[fit_idx].astype(np.float32) / 255.0, y[fit_idx])
    preds = clf.predict(X[test_idx].astype(np.float32) / 255.0)
    pd.DataFrame({"id": test_ids, "label": preds}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)
    acc = float((preds == y[test_idx]).mean())
    print(f"  benchmark accuracy={acc:.6f}")

    for name in ("X_test.csv", "y_test.csv", "sample_train.csv",
                 "benchmark_submission.csv"):
        size = os.path.getsize(os.path.join(DATA, name)) / 1e6
        print(f"  {name:28s} {size:7.2f} MB")


if __name__ == "__main__":
    main()
