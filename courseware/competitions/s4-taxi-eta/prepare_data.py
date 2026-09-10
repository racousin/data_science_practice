#!/usr/bin/env python3
"""Build the NYC green-taxi split for the Session 4 challenge (Lab 1).

Public    -> data/X.csv, data/y.csv, data/X_submission.csv
Private   -> data/y_submission.csv          (uploaded to the ENV folder, never published)
             data/y_test.csv                the same file under the name the platform
                                            needs to start the challenge and to build
                                            its upload gate (columns + ids)
Benchmark -> data/benchmark_submission.csv  (linear quantile regression at
             tau = 0.9 — the bar the lab's network has to clear, NOT what the
             lab's notebook produces)

The task: promise an arrival time the ride beats 9 times out of 10. The target
is the trip's duration in minutes and the loss is the pinball loss at
tau = 0.9, so a minute late costs nine times a minute of padding and the best
promise is the 90th percentile of the duration, not its mean.

**Source.** NYC TLC trip records, green taxis, January 2024: the official
parquet and the zone lookup, both pinned by sha256. They are downloaded once
into data/source/ (gitignored with the rest of data/), and the build stops on a
checksum mismatch rather than building from a different file.

**Cleaning.** The row counts are asserted, because the lab's notebook checks
depend on them (131 batches of 256 per epoch):

    56,551 raw trips
 -> 52,625 picked up in January 2024, 1-120 min long, 0.1-50 miles
 -> 52,305 whose pickup and drop-off boroughs are both Manhattan, Queens,
           Brooklyn or the Bronx. Staten Island, EWR and unknown zones are 0.6%
           of trips; standardising a one-hot column that is almost always 0
           turns its rare 1 into a value of 30-200, which an MLP extrapolates.

**The split is chronological.** Rows sorted by pickup time (a stable sort, so
trips picked up in the same second keep their file order on every machine);
the first 80% — 41,844 trips, 1 January 00:03 to 25 January 20:56 — to fit on,
the remaining 10,461, to 31 January 23:57, to predict. X.csv ships in pickup
order (`shuffle_and_label(..., shuffle=False)`, the s2-bike-demand precedent)
so a student can hold out the LAST rows for validation; X_submission.csv is
shuffled. No timestamp column is shipped.

**12 features:** trip_distance (miles), pickup_hour (hour + minute / 60),
day_of_week (0 = Monday), is_weekend, and the pickup and drop-off borough, one
hot, four columns each. trip_distance is the metered distance, known only once
the ride is over; an app would use the planned route's length, and this data
stands in for it with the metered one. The target is rounded to 4 decimals.

**The benchmark is not a network.** A torch run is reproducible on one machine
and not across machines, so the declared benchmark is scikit-learn's
QuantileRegressor(quantile=0.9, alpha=0.0): the best straight line under the
same loss, solved exactly as a linear program ("highs-ipm"; "highs" agrees to
1e-13 and takes ten times longer). sklearn's default alpha is 1.0, an L1
penalty — hence alpha=0.0. It is fitted on the raw CSV columns as written and
read back, and scored from the rounded file exactly as env.py scores it.

Reference rows on this split, -Pinball and promises kept (the first three are
printed below; the MLP rows are the lab's 12-64-64-1 recipe over five seeds):

    always 24.7 min, the training 90th percentile   -2.2331   89.4%
    LinearRegression (MSE)                           -2.0866   55.7%
    the lab's MLP trained on MSE                     -1.74     54%
    QuantileRegressor, tau = 0.9   <- the benchmark  -1.0429   88.4%
    the lab's MLP trained on pinball (solution)      -0.94     89%

Deterministic given `MLARENA_ID_SALT` (see ../_dataset_ids.py): re-running
under the same salt reproduces the same files byte for byte. Source the salt
file rather than `$(cat ...)` it: it holds one `export MLARENA_ID_SALT=...`
line, and the whole line is not the salt (blake2b refuses it as a key).

    . competitions/.id-salt
    uv run --with pandas --with pyarrow --with scikit-learn \\
        python competitions/s4-taxi-eta/prepare_data.py
"""
import hashlib
import os
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

TAU = 0.9
FIT_SHARE = 0.8
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SOURCE = os.path.join(DATA, "source")
DATASET = "nyc-green-taxi-2024-01"

TRIPS = ("https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet",
         "1512a953bed564ac68dcb622827424dfb4f68ad93c2a4572e50ef6ab0b09c1ad")
ZONES = ("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
         "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed")
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
# raw -> after the date/duration/distance filter -> within the four boroughs
EXPECTED_ROWS = (56551, 52625, 52305)

sys.path.insert(0, os.path.dirname(HERE))
from _dataset_ids import shuffle_and_label  # noqa: E402


def fetch(url, sha256):
    """Download once into data/source/, and refuse any file other than the one
    every number in config.py and overview.md was measured on."""
    os.makedirs(SOURCE, exist_ok=True)
    path = os.path.join(SOURCE, os.path.basename(url))
    if not os.path.exists(path):
        print(f"  downloading {url}")
        part = path + ".part"
        urllib.request.urlretrieve(url, part)
        os.replace(part, path)
    with open(path, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != sha256:
        raise SystemExit(
            f"{path}: sha256 {got}, expected {sha256}.\n"
            f"This is not the file the challenge was measured on. Delete it to "
            f"download again; if TLC has republished the month, every number in "
            f"config.py and overview.md has to be measured again.")
    return path


def build_frame():
    trips = pd.read_parquet(fetch(*TRIPS))
    zones = pd.read_csv(fetch(*ZONES))
    borough = dict(zip(zones["LocationID"], zones["Borough"]))

    pickup = trips["lpep_pickup_datetime"]
    trips["duration_min"] = (
        trips["lpep_dropoff_datetime"] - pickup).dt.total_seconds() / 60
    keep = ((pickup >= "2024-01-01") & (pickup < "2024-02-01")
            & trips["duration_min"].between(1, 120)
            & trips["trip_distance"].between(0.1, 50))
    df = (trips[keep]
          .sort_values("lpep_pickup_datetime", kind="stable")
          .reset_index(drop=True))
    df["pickup_borough"] = df["PULocationID"].map(borough)
    df["dropoff_borough"] = df["DOLocationID"].map(borough)
    inside = (df["pickup_borough"].isin(BOROUGHS)
              & df["dropoff_borough"].isin(BOROUGHS))
    df = df[inside].reset_index(drop=True)

    counts = (len(trips), int(keep.sum()), len(df))
    print(f"  {counts[0]} raw trips -> {counts[1]} after the date, duration "
          f"and distance filter -> {counts[2]} within the four boroughs")
    if counts != EXPECTED_ROWS:
        raise SystemExit(
            f"row counts {counts} != {EXPECTED_ROWS}. The file passed its "
            f"checksum, so the cleaning behaved differently (a pandas change?). "
            f"The notebook's 131-batch check and the overview's numbers assume "
            f"{EXPECTED_ROWS[-1]} trips.")
    return df


def features(df):
    """The 12 columns a student gets. Integer-valued columns stay integers."""
    t = df["lpep_pickup_datetime"]
    X = pd.DataFrame({
        "trip_distance": df["trip_distance"],           # miles, metered
        "pickup_hour": t.dt.hour + t.dt.minute / 60,    # 13.5 = 13:30
        "day_of_week": t.dt.dayofweek,                  # 0 = Monday
        "is_weekend": (t.dt.dayofweek >= 5).astype(int),
    })
    for side, column in (("pickup", "pickup_borough"),
                         ("dropoff", "dropoff_borough")):
        for b in BOROUGHS:
            X[f"{side}_{b.lower()}"] = (df[column] == b).astype(int)
    return X


def pinball_score(truth, pred):
    """(-pinball, % of promises kept), as env.py computes them."""
    d = np.asarray(truth, dtype=float) - np.asarray(pred, dtype=float)
    loss = float(np.mean(np.maximum(TAU * d, (TAU - 1) * d)))
    return -loss, 100 * float(np.mean(d <= 0))


def main():
    os.makedirs(DATA, exist_ok=True)

    print("loading NYC TLC green-taxi trips, January 2024 …")
    df = build_frame()
    X, y = features(df), df["duration_min"].round(4)

    n_fit = int(FIT_SHARE * len(df))
    t = df["lpep_pickup_datetime"]
    print(f"  fit rows 0-{n_fit - 1}: {t.iloc[0]} .. {t.iloc[n_fit - 1]}; "
          f"submission rows: {t.iloc[n_fit]} .. {t.iloc[-1]}")
    X_fit, y_fit = X.iloc[:n_fit], y.iloc[:n_fit]
    X_sub, y_sub = X.iloc[n_fit:], y.iloc[n_fit:]
    print(f"  90th percentile of the duration: fit {y_fit.quantile(TAU):.2f} "
          f"min, submission {y_sub.quantile(TAU):.2f} min")

    # Opaque ids (../_dataset_ids.py). The fit rows keep pickup order: their
    # targets ship in y.csv anyway, and the order is what lets a student hold
    # out the last days rather than random trips. The submission rows are
    # shuffled, so an id says nothing about its position in the cut.
    X_fit, y_fit = shuffle_and_label(
        X_fit, y_fit, dataset=DATASET, split="train", prefix="tr", shuffle=False)
    X_sub, y_sub = shuffle_and_label(
        X_sub, y_sub, dataset=DATASET, split="test", prefix="te")

    X_fit.to_csv(os.path.join(DATA, "X.csv"), index=False)
    pd.DataFrame({"id": X_fit["id"], "prediction": y_fit}).to_csv(
        os.path.join(DATA, "y.csv"), index=False)
    X_sub.to_csv(os.path.join(DATA, "X_submission.csv"), index=False)
    pd.DataFrame({"id": X_sub["id"], "prediction": y_sub}).to_csv(
        os.path.join(DATA, "y_submission.csv"), index=False)
    shutil.copyfile(os.path.join(DATA, "y_submission.csv"),
                    os.path.join(DATA, "y_test.csv"))
    print(f"  fit {len(X_fit)} rows / submission {len(X_sub)} rows, "
          f"{X_fit.shape[1] - 1} features")

    print("fitting the benchmark (QuantileRegressor, tau = 0.9) on the CSVs …")
    tr = pd.read_csv(os.path.join(DATA, "X.csv"))
    te = pd.read_csv(os.path.join(DATA, "X_submission.csv"))
    ytr = pd.read_csv(os.path.join(DATA, "y.csv"))["prediction"]
    yte = pd.read_csv(os.path.join(DATA, "y_submission.csv"))["prediction"]
    Xtr, Xte = tr.drop(columns=["id"]), te.drop(columns=["id"])

    qr = QuantileRegressor(quantile=TAU, alpha=0.0, solver="highs-ipm")
    preds = qr.fit(Xtr, ytr).predict(Xte)
    bench = os.path.join(DATA, "benchmark_submission.csv")
    pd.DataFrame({"id": te["id"], "prediction": np.round(preds, 6)}).to_csv(
        bench, index=False)

    # Scored the way env.py scores it, from the rounded file that is uploaded.
    neg, kept = pinball_score(yte, pd.read_csv(bench)["prediction"])
    print(f"  benchmark  -Pinball={neg:.6f}  promise kept {kept:.1f}%")

    const = float(ytr.quantile(TAU))
    c_neg, c_kept = pinball_score(yte, np.full(len(yte), const))
    print(f"  reference  always {const:.2f} min (the fit 90th percentile) "
          f"-Pinball={c_neg:.6f}  kept {c_kept:.1f}%")
    lin = LinearRegression().fit(Xtr, ytr)
    l_neg, l_kept = pinball_score(yte, lin.predict(Xte))
    print(f"  reference  LinearRegression (MSE)          "
          f"-Pinball={l_neg:.6f}  kept {l_kept:.1f}%")
    shift = float(np.quantile(ytr - lin.predict(Xtr), TAU))
    s_neg, s_kept = pinball_score(yte, lin.predict(Xte) + shift)
    print(f"  reference  LinearRegression + {shift:.2f} min "
          f"(its fit residuals' 90th pct) -Pinball={s_neg:.6f}  kept {s_kept:.1f}%")

    print(f"\n  set benchmark_expected_score = {round(neg, 6)}")
    print(f"\nwrote {DATA}")
    print("  public : X.csv y.csv X_submission.csv")
    print("  private: y_submission.csv y_test.csv (identical)")
    print("  bench  : benchmark_submission.csv")


if __name__ == "__main__":
    main()
