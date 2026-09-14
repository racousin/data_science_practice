# Session 1 of MS2A - Machine Learning Practice ("Data Collection"), course 15.
# The Data Science Practice module 4 exercise, unchanged: predict quantity_sold
# for the 409 Neighborhood_Market items from four other stores (1,591 items)
# whose data is spread over four source types — files (CSV, pipe-separated CSV,
# XLSX, JSON), an authenticated API (unit_cost), a JavaScript-rendered page
# (customer_score, total_reviews) and the course SQL database (retail.stores,
# retail.data_dictionary). prepare_data.py recovers the never-published target
# by replaying the public generator notebook, and asserts it reproduces every
# file students download.
#
# Measured ladder (prepare_data.py and reference_solution.py, 2026-09-14): the
# exercise's get_simple_baseline recipe — fillna -1, drop store_name and
# last_modified, StandardScaler, LinearRegression — trained on the four stores.
# MAE on all 409 items / on the 214 even-numbered items the course database's
# board scores / 5-fold CV on the training stores:
#
#     constant: CityMart's mean (minimal)      26.37            fail
#     constant: mean of the four stores        23.18 / 22.95    fail
#     files                                    19.89 / 20.15    pass   cv 45.04
#     + unit_cost (API)                        17.05 / 17.23    pass   cv 44.10
#     + customer_score, total_reviews          3.51 /  3.55     pass   cv 40.03
#     + weekly_footfall as a feature           2.15             pass   cv 15.97
#     footfall as a multiplier (y / footfall)  2.11             pass
#
# The benchmark is the files + API + scraping model — the original exercise's
# reference solution, whose CV score reference_solution.py asserts. The
# database's footfall is the step beyond it. Neighborhood_Market's footfall in
# retail.stores is 13,400, not 1.0 x 12,000: the generator made its target with
# one regression fitted on the four stores pooled plus a +3.76 shift, an
# effective multiplier of ~1.117. At 12,000 both footfall models scored
# 21.35 / 21.39 and failed the bar.
#
# Ranked on -MAE, the s2-bike-demand precedent: the leaderboard sorts score
# DESC with no lower-is-better flag (modelmanager/modelmanager/competitions.py
# :210-213), so the error is negated. RMSE and the row count ride along.
#
# The target is not secret: the seeded generator notebook is public, and
# re-running it gives neighborhood_market_target.csv exactly. The ids cannot be
# salted either, because the student reads them from the original files. Treat
# it as coursework, not a secure benchmark.
CONFIG = {
    "name": "MLP S1 — Multi-Source Store Sales",
    "kernel_version": "file_v1",
    "module_slug": "s1-data-collection",
    "label": "Multi-Source Store Sales",
    "metric": "neg_mae",
    "metric2": "rmse",
    "public_files": ["CityMart_data.csv", "Greenfield_Grocers_data.csv",
                     "SuperSaver_Outlet_data.xlsx",
                     "HighStreet_Bazaar_data.json",
                     "Neighborhood_Market_data.csv"],
    # env.py reads neighborhood_market_target.csv. y_test.csv is the same file
    # under the name the platform requires: start refuses a file_v1 CSV
    # challenge without an env file called y_test.csv (backend lifecycle.py),
    # and its header and ids become the upload check (env_files.py), which
    # rejects a wrong header or item set before a deploy is spent. The
    # s4-taxi-eta precedent.
    "private_files": ["neighborhood_market_target.csv", "y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # files + API + scraping through get_simple_baseline (reference_solution.py).
    # Its 5-fold CV MAE, 40.0277, equals what the exercise's own reference
    # solution printed.
    "benchmark_expected_score": -3.513043,
    "benchmark_score_tol": 1e-6,
    # A course that states its own bar: the original exercise passed a
    # submission with MAE <= 20 — ERROR_THRESHOLD=20 in
    # tests/data-science-practice/module4/exercise1.sh. `attach` writes this,
    # not the benchmark score, onto the module link (compared with >=).
    "pass_threshold": -20.0,
    "dataset_label": "Multi-Source Store Sales",
    "dataset_description": (
        "Item data from five stores of one retail chain. Four stores come with "
        "their sales (quantity_sold), each in its own format: "
        "CityMart_data.csv, Greenfield_Grocers_data.csv (pipe-separated), "
        "SuperSaver_Outlet_data.xlsx (two sheets) and "
        "HighStreet_Bazaar_data.json. Neighborhood_Market_data.csv holds the "
        "409 items to predict, without their sales. unit_cost, "
        "customer_score, total_reviews and the stores' weekly_footfall are "
        "not in these files: they come from an API, a web page and the course "
        "database."
    ),
    # Deterministic scorer: the same CSV always yields the same -MAE.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "neg_mae", "label": "-MAE", "source": "env",
         "agg": "mean", "format": "number", "precision": 2,
         "higher_is_better": True},
        {"key": "rmse", "label": "RMSE", "source": "env",
         "agg": "mean", "format": "number", "precision": 2,
         "higher_is_better": False},
        # Always 409 on a scored run; shown so the row count is visible.
        {"key": "n_rows", "label": "Items scored", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0},
    ],
    "is_public_initial": False,
}
