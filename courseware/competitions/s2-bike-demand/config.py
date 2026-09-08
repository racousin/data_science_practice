# Session 2 of MS2A - AI Engineering ("Machine Learning Foundations"), the
# regression half. Hourly bike rentals from calendar + weather columns, scored
# on a held-out 20% split.
#
# Ranked on -MAE: the leaderboard sorts score DESC with no lower-is-better flag
# (modelmanager/modelmanager/competitions.py:210-213), so the primary metric has
# to increase with quality. Negating MAE gives that while keeping the target's
# units — -103.74 is "off by 103.74 bikes an hour", and 0 is perfect. RMSE and
# R2 ride along for display.
CONFIG = {
    "name": "AIE S2 — Bike Sharing Demand",
    "kernel_version": "file_v1",
    "module_slug": "s2-ml-foundations",
    "label": "Bike Sharing Demand",
    "metric": "neg_mae",
    "metric2": "rmse",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # get_dummies + LinearRegression on the CSVs as written — the worked
    # notebook's model, so the bar is exactly what a student who followed it
    # gets. Reference points on this split: predicting the training mean is
    # -140.08; this baseline is -103.74 (RMSE 137.92, R2 0.3993).
    "benchmark_expected_score": -103.739243,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Bike Sharing Demand",
    "dataset_description": (
        "Random 80/20 split of a public bike-sharing dataset, 17,379 hourly "
        "observations from a bike-share system, shuffled and re-identified. "
        "X_train.csv + y_train.csv to fit on, X_test.csv to predict. The test "
        "counts are held back."
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
        {"key": "r2", "label": "R²", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "n_negative", "label": "Negative preds", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
