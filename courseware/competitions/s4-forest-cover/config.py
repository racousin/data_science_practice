# Session 4 of MS2A - AI Engineering ("PyTorch in a Nutshell"), the
# classification half and the one the student writes unaided. Forest cover type
# from 54 cartographic features, seven classes, balanced.
#
# Same argument as the regression half, on a class label: LogisticRegression
# reaches 0.696 accuracy on this split and a 54-128-64-7 MLP reaches 0.814.
# Seven classes make `CrossEntropyLoss` over real logits the natural shape, so
# the student meets the standard torch classification head rather than a
# one-output binary special case.
#
# THE BENCHMARK IS THE LINEAR MODEL, not an MLP — see s4-california-housing's
# config for why, and prepare_data.py for the measured ladder.
#
# Ranked on accuracy, which is honest here only because the classes are
# balanced 1/7 by construction: chance is 0.1429, accuracy equals balanced
# accuracy, and macro-F1 tracks it. On the raw 581k-row distribution (49%
# lodgepole pine) none of that would hold.
CONFIG = {
    "name": "AIE S4 — Forest Cover Type",
    "kernel_version": "file_v1",
    "module_slug": "s4-pytorch-nutshell",
    "label": "Forest Cover Type",
    "metric": "accuracy",
    "metric2": "macro_f1",
    "public_files": ["X_train.csv", "y_train.csv", "X_test.csv"],
    "private_files": ["y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # StandardScaler + LogisticRegression(max_iter=2000) on the CSVs as written.
    # Reference points on this split: most-common-class is 0.1429; this baseline
    # is 0.695714 (macro-F1 0.691952); the taught MLP is 0.814.
    "benchmark_expected_score": 0.695714,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "Forest Cover Type",
    "dataset_description": (
        "Balanced subsample of the UCI Covertype dataset (sklearn's "
        "`fetch_covtype`): 2,500 rows of each of the seven cover types, split "
        "80/20 stratified. 54 cartographic features — ten quantitative, four "
        "wilderness-area indicators, forty soil-type indicators. The target is "
        "the cover type, coded 1-7. X_train.csv + y_train.csv to fit on, "
        "X_test.csv to predict. The test targets are held back."
    ),
    # Deterministic scorer: the same CSV always yields the same accuracy.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "accuracy", "label": "Accuracy", "source": "env",
         "agg": "mean", "format": "percent", "precision": 2},
        {"key": "macro_f1", "label": "Macro F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
        # The column that notices a model has quietly stopped predicting a whole
        # cover type — which is exactly what an under-trained 7-way head does.
        {"key": "worst_class_f1", "label": "Worst class F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
    ],
    "is_public_initial": False,
}
