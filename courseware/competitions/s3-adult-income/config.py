# Session 3 of MS2A - AI Engineering ("Data Science in a Nutshell"). The Lab 3
# pipeline, scored on a held-out split of UCI Adult.
CONFIG = {
    "name": "PAIE S3 — Adult Census Income",
    "kernel_version": "file_v1",
    # "Data Science in a Nutshell" (#16) was split; Lab 3, which submits here,
# now lives in s3-models-and-tuning. course.yaml declares it there.
    "module_slug": "s3-models-and-tuning",
    "label": "Adult Census Income",
    "metric": "f1",
    "metric2": "accuracy",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    "benchmark_expected_score": 0.656175,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Adult Census Income",
    "dataset_description": (
        "Stratified 80/20 split of UCI Adult (openml `adult`, version 2). "
        "X_train.csv + y_train.csv to fit on, X_test.csv to predict. The test "
        "labels are held back."
    ),
    # Deterministic scorer: the same CSV always yields the same F1.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "f1", "label": "F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
        {"key": "accuracy", "label": "Accuracy", "source": "env",
         "agg": "mean", "format": "percent", "precision": 2},
        {"key": "precision", "label": "Precision", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
        {"key": "recall", "label": "Recall", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
    ],
    "is_public_initial": False,
}
