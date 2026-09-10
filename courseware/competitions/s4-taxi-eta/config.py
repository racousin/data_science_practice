# Session 4 of MS2A - AI Engineering ("PyTorch in a Nutshell"), Lab 1. NYC
# green-taxi trips, January 2024: promise an arrival time the ride beats 9
# times out of 10. The student writes the loss (pinball, tau = 0.9) and the
# training loop of a 12-64-64-1 MLP, then submits its promises.
#
# The challenge exists for one contrast: the same network trained on MSE aims
# at the average trip, keeps about half its promises and scores about -1.74;
# trained on the loss the student writes, it scores about -0.94. Writing the
# loop is what lets you choose the loss.
#
# Ranked on -Pinball: the leaderboard sorts score DESC with no lower-is-better
# flag (modelmanager/modelmanager/competitions.py:214), so the loss is negated
# — the s2-bike-demand -MAE precedent. Precision 4, not 3: the bar is
# -1.042916, and at three decimals a score just under it (-1.0431) and one just
# over it (-1.0427) would both read -1.043. No metric2: the number worth showing
# beside the loss is the share of promises kept, and it is not an error.
#
# THE BENCHMARK IS THE LINEAR MODEL, not the lab's network: a torch run is
# reproducible on one machine and not across machines. It is
# QuantileRegressor(quantile=0.9, alpha=0.0), the best straight line under the
# same loss, solved exactly and deterministic. `build_competitions.py attach`
# writes it onto the module link as the pass threshold (compared with >=), so
# the bar reads "beat the best straight line under this loss". Clearing it does
# NOT show that the student wrote the loop: an MSE-trained net rescaled by one
# factor fitted on the training rows scores -0.93 to -0.96, and scikit-learn's
# gradient boosting with a quantile loss about -0.93. The notebook's check
# cells are that evidence; the leaderboard is not.
CONFIG = {
    "name": "AIE S4 — Taxi Arrival Promise",
    "kernel_version": "file_v1",
    "module_slug": "s4-pytorch-nutshell",
    "label": "Taxi Arrival Promise",
    "metric": "neg_pinball",
    "public_files": ["X.csv", "y.csv", "X_submission.csv"],
    # y_test.csv is the same file under the name the platform requires: start
    # refuses a file_v1 CSV challenge without an env file called y_test.csv
    # (backend lifecycle.py), and its columns and ids become the upload gate
    # (env_files.py), which rejects a wrong header or id set before a deploy
    # slot is spent. env.py reads y_submission.csv.
    "private_files": ["y_submission.csv", "y_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv",
    # QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs-ipm") on the raw
    # CSV columns as written. Reference points on this split (prepare_data.py
    # prints the first three): always 24.7 min, the fit rows' 90th percentile,
    # -2.233092 with 89.4% of promises kept; LinearRegression on MSE -2.086596,
    # 55.7%; this benchmark -1.042916, 88.4%; the lab's MLP on MSE about -1.74,
    # 54%; the same MLP on pinball about -0.94, 89%.
    "benchmark_expected_score": -1.042916,
    "benchmark_score_tol": 1e-6,
    # What the solution notebook must clear for the test suite to pass: a floor
    # rather than the measured value, because torch is not bit-reproducible
    # across machines. The planned solution notebook measured -0.930 to -0.946
    # over seeds 0-4; the test also asserts it beats the benchmark.
    "notebook_expected_min_score": -0.99,
    "dataset_label": "Taxi Arrival Promise",
    "dataset_description": (
        "52,305 NYC green-taxi trips from January 2024 (public TLC trip "
        "records), split by time. X.csv + y.csv hold the first 41,844 trips, "
        "1 to 25 January, in pickup order, so the last rows can be held out "
        "for validation; X_submission.csv holds the remaining 10,461, "
        "shuffled. 12 numeric features; the target is the trip's duration in "
        "minutes. Predict a duration the trip beats 9 times out of 10: scored "
        "with the pinball loss at tau = 0.9."
    ),
    # Deterministic scorer: the same CSV always yields the same -Pinball.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "neg_pinball", "label": "-Pinball (min)", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        # Neither direction is better: about 90 is calibrated, 100 is padding.
        {"key": "promise_kept_pct", "label": "Promise kept (%)", "source": "env",
         "agg": "mean", "format": "number", "precision": 1},
        {"key": "avg_promise_min", "label": "Avg promise (min)", "source": "env",
         "agg": "mean", "format": "number", "precision": 2,
         "higher_is_better": False},
        {"key": "n_negative", "label": "Negative preds", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
