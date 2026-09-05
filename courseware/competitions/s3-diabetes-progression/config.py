# Session 3 of MS2A - AI Engineering ("Models, Ensembles & Tuning"), the
# regression half and the session's centrepiece. Diabetes progression from ten
# baseline clinical measurements, on a deliberately small 60/40 split.
#
# The dataset earns its place by making overfitting unmissable: a 500-tree
# forest fits the 265 training rows to R2 = 0.92 and scores 0.49 held out,
# while plain linear regression fits to 0.51 and scores 0.52. The benchmark is
# the linear model, so the leaderboard bar is the honest one.
#
# Ranked on R2, not RMSE: the leaderboard sorts score DESC with no
# lower-is-better flag (modelmanager/modelmanager/competitions.py:214), so the
# primary metric has to increase with quality. R2 = 0 is exactly the
# predict-the-mean model, which makes the sign of the score meaningful on its
# own. RMSE and MAE ride along in the target's units.
CONFIG = {
    "name": "AIE S3 — Diabetes Progression",
    "kernel_version": "file_v1",
    "module_slug": "s3-models-and-tuning",
    "label": "Diabetes Progression",
    "metric": "r2",
    "metric2": "rmse",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # Plain LinearRegression on the CSVs as written. Reference points on this
    # split: predict-the-mean is R2 = 0.0 (RMSE 76.54); this baseline is
    # R2 = 0.5157 (RMSE 53.23); the heavy random forest is R2 = 0.4922.
    "benchmark_expected_score": 0.515744,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Diabetes Progression",
    "dataset_description": (
        "Random 60/40 split of sklearn's `load_diabetes` (raw, unscaled): 442 "
        "patients, ten baseline clinical measurements, and disease progression "
        "one year later. X_train.csv + y_train.csv to fit on, X_test.csv to "
        "predict. The test targets are held back."
    ),
    # Deterministic scorer: the same CSV always yields the same R2.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "r2", "label": "R²", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "rmse", "label": "RMSE", "source": "env",
         "agg": "mean", "format": "number", "precision": 2,
         "higher_is_better": False},
        {"key": "mae", "label": "MAE", "source": "env",
         "agg": "mean", "format": "number", "precision": 2,
         "higher_is_better": False},
        {"key": "n_negative", "label": "Negative preds", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
