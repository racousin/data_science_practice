# Session 2 of MS2A - AI Engineering ("Machine Learning Foundations"), the
# regression half. Hourly bike rentals from calendar + weather columns.
#
# Split is CHRONOLOGICAL — the last 20% of the hours are held out, so the task is
# to forecast forward rather than to interpolate. It was a random 80/20 until
# 2026-09-08, which put 20:00 in the test set while 19:00 and 21:00 stayed in
# training: same weather reading, count within a few bikes, so the model could
# look up most of what it was being asked to predict. The leak was worth about
# 35 MAE — the same baseline scored -103.74 under it and scores -138.88 without.
#
# The features also carry missing values now (see prepare_data.py), which is why
# the benchmark pipeline starts with an imputation step.
#
# Ranked on -MAE: the leaderboard sorts score DESC with no lower-is-better flag
# (modelmanager/modelmanager/competitions.py:210-213), so the primary metric has
# to increase with quality. Negating MAE gives that while keeping the target's
# units — -138.88 is "off by 138.88 bikes an hour", and 0 is perfect. RMSE rides
# along for display. R2 was dropped on 2026-09-08: a second view of the same
# residuals is an invitation to quote whichever is kinder, and under a
# chronological split it scores the forecast against a test-window mean the
# forecaster could not have known.
# Attached to TWO modules since 2026-09-09: s2-ml-foundations (EDA + a first
# linear model) and s3-models-and-tuning (model choice and honest evaluation on
# the same rows). `module_slug` below drives `make competitions-attach` only, so
# it still names session 2; session 3's link is written by `make publish` from
# the `competitions:` block in course.yaml, and carries the higher bar.
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
    # Median/"unknown" imputation + get_dummies + LinearRegression on the CSVs as
    # written — the worked notebook's model, so the bar is exactly what a student
    # who followed it gets. Reference points on this split: predicting the
    # training mean is -174.98 (the system grew, so the past mean under-shoots
    # the future); this baseline is -138.88 (RMSE 183.20); the same baseline with
    # `hour` one-hot encoded is -100.28.
    "benchmark_expected_score": -138.879599,
    "benchmark_score_tol": 1e-6,
    # The worked notebook no longer STOPS at the benchmark: it submits the
    # baseline, then engineers features and submits again, which is the point of
    # the session. So its final `submission.csv` is the engineered model and the
    # test asserts a floor rather than equality with the benchmark. Measured
    # -99.42 (hour as 24 categories + missing flags + temp x workingday, clipped
    # at 0); the floor keeps headroom for a different BLAS.
    "notebook_expected_min_score": -105.0,
    "dataset_label": "Bike Sharing Demand",
    "dataset_description": (
        "17,379 hourly observations from a bike-share system over two years, "
        "split by time: the first 80% of the hours train, the last 20% are held "
        "back. X_train.csv + y_train.csv to fit on, X_test.csv to predict. "
        "X_train.csv ships in calendar order so you can hold out the last hours "
        "for validation; X_test.csv is shuffled. Some feature cells are missing "
        "— temp and feel_temp together in runs, windspeed and weather scattered "
        "— and nothing fits until you deal with them."
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
        {"key": "n_negative", "label": "Negative preds", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
