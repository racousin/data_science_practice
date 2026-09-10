# Session 4 of MS2A - AI Engineering ("PyTorch in a Nutshell"), the regression
# half and the session's worked example. Median house value per California
# census district, from eight numeric features.
#
# The dataset earns its place by being a problem where the linear model that
# closed Session 2 is visibly not enough and a small MLP visibly is:
# LinearRegression reaches R2 = 0.576 on this split and an 8-64-64-1 MLP
# reaches 0.776. That is a bigger jump than anything in Sessions 2 or 3 and it
# is the argument for spending a session on PyTorch at all.
#
# THE BENCHMARK IS THE LINEAR MODEL, not the notebook's MLP. Sessions 2 and 3
# pinned the benchmark to what the worked notebook produced; that is not
# available here, because a torch training run is bit-reproducible on one
# machine and not across machines. So the benchmark is the bar rather than the
# answer, and `notebook_expected_min_score` is what the notebook test asserts
# instead. See prepare_data.py.
#
# Ranked on R2 for the same reason as every other regression challenge here:
# the leaderboard sorts score DESC with no lower-is-better flag
# (modelmanager/modelmanager/competitions.py:214), so the primary metric has to
# increase with quality.
CONFIG = {
    "name": "AIE S4 — California Housing",
    "kernel_version": "file_v1",
    "module_slug": "s4-pytorch-nutshell",
    "label": "California Housing",
    "metric": "r2",
    "metric2": "rmse",
    "public_files": ["X.csv", "y.csv", "X_submission.csv"],
    "private_files": ["y_submission.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # Plain LinearRegression on the CSVs as written — what Session 2 could do.
    # Reference points on this split: predict-the-mean is R2 = 0.0
    # (RMSE 1.1449); this baseline is 0.575788 (RMSE 0.7456); the same
    # predictions clipped to the target's documented range are 0.603856; the
    # taught MLP is 0.776 (0.775 before clipping).
    "benchmark_expected_score": 0.575788,
    "benchmark_score_tol": 1e-6,
    # What the worked notebook has to clear for the test suite to pass. The MLP
    # measures 0.776 with a fixed seed and 0.776-0.779 across seeds 0-2, so this
    # is a floor with real headroom rather than a pinned float — it fails on a
    # broken notebook and survives a different BLAS.
    "notebook_expected_min_score": 0.72,
    "dataset_label": "California Housing",
    "dataset_description": (
        "Random 80/20 split of sklearn's `fetch_california_housing`: 20,640 "
        "California census districts from the 1990 census, eight numeric "
        "features, and the district's median house value in units of $100,000 "
        "(capped at 5.0). X.csv + y.csv to fit on, X_submission.csv to "
        "predict. Its targets are held back."
    ),
    # Deterministic scorer: the same CSV always yields the same R2.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "r2", "label": "R²", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        {"key": "rmse", "label": "RMSE", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": False},
        {"key": "mae", "label": "MAE", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": False},
        # Not decoration. The target cannot leave [0.15, 5.0] and a linear model
        # leaves it 15 times; an MLP with a plain linear output layer will too.
        # Clipping to the range is worth +0.028 R2 on the benchmark for free,
        # and this column is how a competitor notices there is something to
        # clip.
        {"key": "n_out_of_range", "label": "Impossible preds", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
