#!/usr/bin/env python3
"""Build the forest-cover split for the Session 4 classification challenge.

Public  -> data/X.csv, data/y.csv, data/X_submission.csv
Private -> data/y_submission.csv                  (uploaded to the ENV folder, never published)
Benchmark -> data/benchmark_submission.csv  (StandardScaler + LogisticRegression —
             the bar the MLP has to clear, NOT what any notebook produces)

As with the Session 4 regression package, the benchmark is the *linear* model
rather than the notebook's own: a torch run is reproducible on one machine and
not across machines, so a float pinned to 1e-6 would be a lie. The benchmark is
the best Session 2 could do here, and beating it is the exercise.

Reference numbers on this split, measured (see the overview's table):

    predict the most common class                  accuracy 0.143
    LogisticRegression on standardised features    accuracy 0.696   <- benchmark
    MLP 54-128-64-7, standardised, Adam 60 epochs  accuracy 0.814

**The subsample is balanced on purpose**: 2,500 rows of each of the seven cover
types. The full 581,012-row dataset is 49% lodgepole pine and 0.47% cottonwood,
and on that distribution accuracy stops meaning anything — a model that never
predicts cottonwood loses almost nothing. Balanced, chance is exactly 1/7 =
0.143, accuracy equals balanced accuracy, and macro-F1 tracks it, so a single
number on the leaderboard is honest. Cottonwood has 2,747 rows in the source,
which is what caps the per-class count at 2,500.

Class codes, as they appear in `y.csv` and as they must appear in a
submission:

    1 Spruce/Fir        3 Ponderosa Pine     5 Aspen        7 Krummholz
    2 Lodgepole Pine    4 Cottonwood/Willow  6 Douglas-fir

They are 1-based, and `torch.nn.CrossEntropyLoss` wants 0-based indices. That
is a deliberate speed bump, not an oversight — the guided notebook makes the
student handle it, because getting it wrong is either a loud `IndexError` or a
silent wasted output unit, and both are worth meeting once under supervision.

Deterministic: the per-class sample and the stratified 80/20 split both run at
SEED. The benchmark is fitted on the CSVs as written and read back, so the
declared score is reachable through exactly the path a student's `pd.read_csv`
takes.

    python prepare_data.py
"""
import os

import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
PER_CLASS = 2500
TEST_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

COVER_TYPES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
}

# sklearn's names, spelled the way the rest of the course spells columns. The
# binary blocks are 0-indexed upstream and 1-indexed in the UCI documentation
# everyone reads; this restores the documented numbering.
RENAME = {
    "Horizontal_Distance_To_Hydrology": "h_dist_hydrology",
    "Vertical_Distance_To_Hydrology": "v_dist_hydrology",
    "Horizontal_Distance_To_Roadways": "h_dist_roadways",
    "Horizontal_Distance_To_Fire_Points": "h_dist_fire_points",
    "Hillshade_9am": "hillshade_9am",
    "Hillshade_Noon": "hillshade_noon",
    "Hillshade_3pm": "hillshade_3pm",
    "Elevation": "elevation", "Aspect": "aspect", "Slope": "slope",
}
RENAME.update({f"Wilderness_Area_{i}": f"wilderness_area_{i + 1}" for i in range(4)})
RENAME.update({f"Soil_Type_{i}": f"soil_type_{i + 1}" for i in range(40)})

QUANTITATIVE = ["elevation", "aspect", "slope", "h_dist_hydrology",
                "v_dist_hydrology", "h_dist_roadways", "hillshade_9am",
                "hillshade_noon", "hillshade_3pm", "h_dist_fire_points"]


def _assert_mapping(X):
    """The rename is only safe if the columns still mean what we named them."""
    binary = [c for c in X.columns if c.startswith(("wilderness_area_", "soil_type_"))]
    checks = [
        ("elevation in metres", 1800 < X["elevation"].min() and X["elevation"].max() < 4000),
        ("aspect is a compass bearing", 0 <= X["aspect"].min() and X["aspect"].max() <= 360),
        ("slope in degrees", 0 <= X["slope"].min() and X["slope"].max() <= 90),
        ("hillshade is a 0-255 index", all(
            0 <= X[c].min() and X[c].max() <= 255
            for c in ("hillshade_9am", "hillshade_noon", "hillshade_3pm"))),
        ("44 binary indicator columns", len(binary) == 44),
        ("indicators are 0/1", set(pd.unique(X[binary].values.ravel())) <= {0, 1}),
        ("exactly one wilderness area per row",
         (X[[f"wilderness_area_{i}" for i in range(1, 5)]].sum(axis=1) == 1).all()),
        ("exactly one soil type per row",
         (X[[f"soil_type_{i}" for i in range(1, 41)]].sum(axis=1) == 1).all()),
        ("no missing values", not X.isna().any().any()),
    ]
    bad = [n for n, ok in checks if not ok]
    if bad:
        raise SystemExit(f"column rename failed its sanity check for: {bad}")
    print(f"  column mapping verified ({len(checks)} checks)")


def main():
    os.makedirs(DATA, exist_ok=True)

    print("loading sklearn covtype (581k rows) …")
    X, y = fetch_covtype(return_X_y=True, as_frame=True)
    X = X.rename(columns=RENAME)
    _assert_mapping(X)
    counts = y.value_counts().sort_index()
    print("  source class counts: "
          + ", ".join(f"{c}={counts[c]}" for c in sorted(counts.index)))
    if counts.min() < PER_CLASS:
        raise SystemExit(f"class {counts.idxmin()} has only {counts.min()} rows "
                         f"< PER_CLASS={PER_CLASS}")

    print(f"balancing to {PER_CLASS} rows per cover type …")
    frame = X.assign(_target=y)
    balanced = pd.concat(
        [frame[frame["_target"] == c].sample(PER_CLASS, random_state=SEED)
         for c in sorted(counts.index)]
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    y_bal = balanced.pop("_target")
    print(f"  {len(balanced)} rows, {balanced.shape[1]} features, "
          f"{y_bal.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        balanced, y_bal, test_size=TEST_SIZE, random_state=SEED, stratify=y_bal
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

    print("fitting the baseline (StandardScaler + LogisticRegression) on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_submission.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_submission.csv"))["prediction"]

    scaler = StandardScaler().fit(tr.drop(columns=["id"]))
    model = LogisticRegression(max_iter=2000).fit(
        scaler.transform(tr.drop(columns=["id"])), ytr)
    preds = model.predict(scaler.transform(te.drop(columns=["id"])))

    pd.DataFrame({"id": te["id"], "prediction": preds}).to_csv(
        os.path.join(DATA, "benchmark_submission.csv"), index=False)

    written = pd.read_csv(os.path.join(DATA, "benchmark_submission.csv"))["prediction"]
    accuracy = float((written.values == yte.values).mean())
    f1s = []
    for c in sorted(COVER_TYPES):
        tp = int(((written == c) & (yte == c)).sum())
        precision = tp / int((written == c).sum()) if int((written == c).sum()) else 0.0
        recall = tp / int((yte == c).sum()) if int((yte == c).sum()) else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    macro_f1 = sum(f1s) / len(f1s)
    majority = float((yte == yte.mode()[0]).mean())
    print(f"  benchmark  accuracy={accuracy:.6f}  macro_f1={macro_f1:.6f}")
    print("  per-class F1: "
          + ", ".join(f"{c}:{f:.3f}" for c, f in zip(sorted(COVER_TYPES), f1s)))
    print(f"  reference  most-common-class accuracy={majority:.4f}  (chance = 1/7 = 0.1429)")
    print(f"\n  set benchmark_expected_score = {round(accuracy, 6)}")
    print(f"\nwrote {DATA}")


if __name__ == "__main__":
    main()
