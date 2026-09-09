# Session 2 of MS2A - AI Engineering ("Machine Learning Foundations"), the
# classification half. Direct-marketing calls from a Portuguese bank; predict
# whether the client subscribed a term deposit, scored on a held-out 20% split.
#
# Ranked on F1 because the target is 11.7% positive: always predicting 0 scores
# 88.3% accuracy, so accuracy cannot separate a model from a constant.
# Attached to TWO modules since 2026-09-09: s2-ml-foundations (EDA + a first
# linear model) and s3-models-and-tuning (model choice and honest evaluation on
# the same rows). `module_slug` below drives `make competitions-attach` only, so
# it still names session 2; session 3's link is written by `make publish` from
# the `competitions:` block in course.yaml, and carries the higher bar.
CONFIG = {
    "name": "AIE S2 — Bank Term Deposit",
    "kernel_version": "file_v1",
    "module_slug": "s2-ml-foundations",
    "label": "Bank Term Deposit",
    "metric": "f1",
    "metric2": "accuracy",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # get_dummies + StandardScaler + LogisticRegression(max_iter=1000) on the
    # CSVs as written. Reference points on this split: always-0 is F1 = 0.0 at
    # 88.30% accuracy; this baseline is F1 = 0.4528 at 90.14%.
    "benchmark_expected_score": 0.452761,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Bank Term Deposit",
    "dataset_description": (
        "Stratified 80/20 split of a public direct-marketing dataset, 45,211 "
        "calls from a Portuguese bank, shuffled and re-identified. The "
        "anonymised V1..V16 columns are restored to their UCI names. "
        "X_train.csv + y_train.csv to fit on, X_test.csv to predict. The test "
        "labels are held back."
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
