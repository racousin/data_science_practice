# Session 3 of MS2A - AI Engineering ("Models, Ensembles & Tuning"), the
# classification half, guided rather than worked. German credit applications;
# predict a bad credit risk, on a stratified 70/30 split.
#
# Same lesson as s3-diabetes-progression in a form the student has to find:
# a 500-tree forest reaches exactly 1.000 training accuracy on the 700 training
# rows and still loses to logistic regression held out (F1 0.482 vs 0.571).
#
# Ranked on F1 because the target is 30% positive and always predicting 0
# scores 70% accuracy, which no model should be allowed to tie by doing
# nothing.
CONFIG = {
    "name": "AIE S3 — Credit Risk",
    "kernel_version": "file_v1",
    "module_slug": "s3-models-and-tuning",
    "label": "Credit Risk",
    "metric": "f1",
    "metric2": "accuracy",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # get_dummies + StandardScaler + LogisticRegression(max_iter=2000) on the
    # CSVs as written. Reference points on this split: always-0 is F1 = 0.0 at
    # 70.00% accuracy; this baseline is F1 = 0.5714 at 77.00%; the heavy random
    # forest is F1 = 0.4823 at 75.67% -- from 1.000 training accuracy.
    "benchmark_expected_score": 0.571429,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Credit Risk",
    "dataset_description": (
        "Stratified 70/30 split of openml `credit-g` (version 1), 1,000 German "
        "credit applications with 20 attributes. The target is coded 1 for a "
        "bad credit risk (30% of rows). X_train.csv + y_train.csv to fit on, "
        "X_test.csv to predict. The test labels are held back."
    ),
    # Deterministic scorer: the same CSV always yields the same F1.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "f1", "label": "F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "accuracy", "label": "Accuracy", "source": "env",
         "agg": "mean", "format": "percent", "precision": 2,
         "higher_is_better": True},
        {"key": "precision", "label": "Precision", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "recall", "label": "Recall", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
    ],
    "is_public_initial": False,
}
