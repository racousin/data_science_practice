#!/usr/bin/env python3
"""Reference solution for MLP S1 — Multi-Source Store Sales.

Builds the three submissions the package and the live checks use, from the
files `prepare_data.py` wrote into data/:

    data/benchmark_submission.csv   the benchmark: the exercise's recipe
    data/minimal_submission.csv     "Your first push" on the challenge page
    data/malformed_submission.csv   the benchmark with every second row removed

It assembles the data the way a student does, source by source:

1. **Files.** The four training stores, each with its own quirk: CityMart is a
   plain CSV; Greenfield_Grocers is pipe-separated, has three empty lines above
   its header, upper-case column names and two empty trailing columns;
   SuperSaver_Outlet is a workbook whose `Quantity` sheet holds the target and
   whose `Info` sheet has a blank first header cell, so every name sits one
   column to the right of its data; HighStreet_Bazaar is JSON records.
2. **API.** `unit_cost` per item.
3. **Scraping.** `customer_score` and `total_reviews` per item.
4. **Database.** `weekly_footfall` per store, from `retail.stores`.

Steps 2 and 3 read `data/full_generated.csv` instead of the network, so the
script runs offline; `prepare_data.py` asserts those columns equal the live API
and the live page for all 2000 items. Step 4 hard-codes the five rows of the
database contract below (no network either).

The model is `get_simple_baseline` from module4_exercise1.ipynb: fill NaN with
-1, drop store_name and last_modified, StandardScaler, LinearRegression, 5-fold
CV for the printed score, then one fit on all four stores.

**The benchmark is the exercise's own reference solution: three sources.**
Measured on the 409 Neighborhood_Market items (MAE; the pass bar is 20):

    constant: CityMart's mean (minimal_submission)   26.37   fail
    constant: mean of the four stores                23.18   fail
    files + API + scraping          (benchmark)       3.51   pass   cv 40.03
    files + API + scraping + footfall as a feature    2.15   pass   cv 15.97
    files + API + scraping, footfall as a multiplier  2.11   pass

The benchmark stops at the three sources the original exercise had, so its CV
score can be asserted equal to the one that solution printed (40.0277), which
ties this script to it. The database's footfall is the step beyond it.

Neighborhood_Market's footfall is 13400, not 1.0 x 12000: the generator made
its target with one regression fitted on the four stores pooled (average effect
1.10) plus a +3.76 shift, so its effective multiplier is about 1.117. With
12000 in the table both footfall models scored 21.35 / 21.39 and failed the
bar; the database states the store's footfall as its sales actually behave.

    uv run --with pandas --with scikit-learn --with openpyxl \\
        python competitions/mlp-s1-store-sales/reference_solution.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# The database contract: retail.stores.weekly_footfall.
WEEKLY_FOOTFALL = {
    "CityMart": 14400,
    "Greenfield_Grocers": 8400,
    "SuperSaver_Outlet": 13200,
    "HighStreet_Bazaar": 16800,
    "Neighborhood_Market": 13400,
}

FILE_COLS = ["mass", "dimension_length", "dimension_width",
             "dimension_height", "days_since_last_purchase",
             "package_volume", "stock_age"]
API_COLS = ["unit_cost"]
SCRAPED_COLS = ["customer_score", "total_reviews"]
DB_COLS = ["weekly_footfall"]
DROP = ["store_name", "last_modified"]
TARGET = "quantity_sold"

# module4_exercise_model_test-Copy1.ipynb printed this CV MAE for the
# files + API + scraping model; prepare_data.py reproduces it too.
REFERENCE_CV_MAE = 40.02769508950211
N_TRAIN, N_TEST = 1591, 409


def path(name):
    return os.path.join(DATA, name)


# --------------------------------------------------------------------------- #
# 1. Files
# --------------------------------------------------------------------------- #
def read_files():
    city = pd.read_csv(path("CityMart_data.csv"), index_col="item_code")

    green = pd.read_csv(path("Greenfield_Grocers_data.csv"), sep="|",
                        header=3, index_col="ITEM_CODE")
    green = green.drop(columns=["1", "Unnamed: 12"])
    green.columns = [c.lower() for c in green.columns]
    green.index.name = "item_code"

    sheets = pd.read_excel(path("SuperSaver_Outlet_data.xlsx"),
                           sheet_name=None)
    info = sheets["Info"]
    info.columns = ["item_code", "store_name", *FILE_COLS, "empty"]
    info = info.drop(columns=["empty"])
    saver = (sheets["Quantity"].merge(info, on="item_code", validate="1:1")
             .set_index("item_code"))

    high = pd.read_json(path("HighStreet_Bazaar_data.json"),
                        orient="records").set_index("item_code")

    train = pd.concat([city, green, saver, high], axis=0)
    test = pd.read_csv(path("Neighborhood_Market_data.csv"),
                       index_col="item_code")
    if len(train) != N_TRAIN or len(test) != N_TEST:
        raise SystemExit(f"expected {N_TRAIN} train / {N_TEST} test rows, "
                         f"got {len(train)} / {len(test)}")
    if not train.index.is_unique or not test.index.is_unique:
        raise SystemExit("item_code is not unique after reading the files")
    if train[TARGET].isna().any():
        raise SystemExit("quantity_sold has missing values in the files")
    return train, test


# --------------------------------------------------------------------------- #
# 2-4. API, scraping, database
# --------------------------------------------------------------------------- #
def add_sources(frame):
    full = pd.read_csv(path("full_generated.csv"), index_col="item_code")
    api = full[API_COLS]
    scraped = full[SCRAPED_COLS].astype(float)  # a scraper reads text
    out = (frame.join(api, how="left", validate="1:1")
           .join(scraped, how="left", validate="1:1"))
    out["weekly_footfall"] = out["store_name"].map(WEEKLY_FOOTFALL)
    added = API_COLS + SCRAPED_COLS + DB_COLS
    if out[added].isna().any().any():
        raise SystemExit("an item or a store has no value in a joined source")
    return out


# --------------------------------------------------------------------------- #
# The model — module4_exercise1.ipynb's get_simple_baseline, with only the
# branches this challenge uses (standard scaler, linear model, MAE).
# --------------------------------------------------------------------------- #
def get_simple_baseline(data, fillna_value=-1, drop_cols=None, k_fold=5,
                        target_col=None, X_data_test=None):
    data = data.copy().fillna(fillna_value)
    X_data_test = X_data_test.copy().fillna(fillna_value)
    if drop_cols:
        data = data.drop(columns=drop_cols)
        X_data_test = X_data_test.drop(columns=drop_cols)
    y = data[target_col]
    X = data.drop(columns=[target_col])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_data_test = scaler.transform(X_data_test)
    model = LinearRegression()
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        model.fit(X[train_index], y.iloc[train_index])
        scores.append(mean_absolute_error(y.iloc[test_index],
                                          model.predict(X[test_index])))
    model.fit(X, y)
    return np.mean(scores), model.predict(X_data_test)


def fit(train, test, features, target=None):
    target = train[TARGET] if target is None else target
    data = train[DROP + features].assign(**{TARGET: target})
    return get_simple_baseline(data, drop_cols=DROP, target_col=TARGET,
                               X_data_test=test[DROP + features])


def write(frame, name):
    frame.to_csv(path(name), index=False)
    print(f"  wrote data/{name}  ({len(frame)} rows)")


def main():
    train, test = read_files()
    train, test = add_sources(train), add_sources(test)

    three = FILE_COLS + API_COLS + SCRAPED_COLS
    cv3, pred3 = fit(train, test, three)
    if abs(cv3 - REFERENCE_CV_MAE) > 1e-6:
        raise SystemExit(f"CV MAE {cv3!r} != the exercise's reference "
                         f"solution {REFERENCE_CV_MAE!r}")
    cv4, pred4 = fit(train, test, three + DB_COLS)
    per_visitor = train[TARGET] / train["weekly_footfall"]
    _, pred_rate = fit(train, test, three, target=per_visitor)
    pred_mult = pred_rate * test["weekly_footfall"].to_numpy()

    ids = test.index.to_numpy()
    benchmark = pd.DataFrame({"item_code": ids, TARGET: pred3})
    city_mean = pd.read_csv(path("CityMart_data.csv"))[TARGET].mean()
    minimal = pd.DataFrame({"item_code": ids, TARGET: city_mean})
    print("writing …")
    write(benchmark, "benchmark_submission.csv")
    write(minimal, "minimal_submission.csv")
    write(benchmark.iloc[::2], "malformed_submission.csv")

    # Creator-side only: the hidden target is not something a student has.
    truth = pd.read_csv(path("neighborhood_market_target.csv"),
                        index_col="item_code")[TARGET].reindex(ids)
    if truth.isna().any():
        raise SystemExit("the target does not cover every test item")
    rows = [
        ("constant: CityMart mean (minimal)", np.full(N_TEST, city_mean), None),
        ("constant: mean of the four stores",
         np.full(N_TEST, train[TARGET].mean()), None),
        ("files + API + scraping (benchmark)", pred3, cv3),
        ("+ footfall as a feature", pred4, cv4),
        ("footfall as a multiplier", pred_mult, None),
    ]
    print("\nMAE on the 409 Neighborhood_Market items (bar: 20)")
    for label, pred, cv in rows:
        mae = float(np.mean(np.abs(pred - truth.to_numpy())))
        cv_text = "" if cv is None else f"  cv {cv:.4f}"
        verdict = "pass" if mae <= 20 else "fail"
        print(f"  {label:38s} {mae:9.6f}  {verdict}{cv_text}")


if __name__ == "__main__":
    main()
