# Session 4 of MS2A - AI Engineering ("PyTorch in a Nutshell"). The warm-up
# competition Lab 4 Part D submits to — the dry run for the project.
CONFIG = {
    "name": "AIE S4 — MNIST Warm-up",
    "kernel_version": "file_v1",
    "module_slug": "s4-pytorch-nutshell",
    "label": "MNIST Warm-up",
    "metric": "accuracy",
    "metric2": "macro_f1",
    "public_files": ["X_submission.csv", "sample_train.csv"],
    "private_files": ["y_submission.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    "benchmark_expected_score": 0.9134,
    "benchmark_score_tol": 1e-6,
    "dataset_label": "MNIST Warm-up",
    "dataset_description": (
        "5,000 unlabelled 28x28 digits to predict (X_submission.csv) plus 2,000 "
        "labelled ones as a format reference (sample_train.csv). Columns "
        "p0..p783 are the image flattened row-major, uint8 0-255. Train on the "
        "full torchvision MNIST split, as Lab 4 Part C describes."
    ),
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "accuracy", "label": "Accuracy", "source": "env",
         "agg": "mean", "format": "percent", "precision": 2},
        {"key": "macro_f1", "label": "Macro F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
        {"key": "worst_class_f1", "label": "Worst digit F1", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
    ],
    "is_public_initial": False,
}
