# MS2A - Machine Learning Practice, Session 2 ("Data Preprocessing"). The
# SUPPORT cohort — 9,105 seriously ill adults admitted to 5 US hospitals,
# 1989-1994 — served as a hospital export would look: did the patient die
# within 60 days of study entry?
#
# The model is FIXED. The scorer fits scikit-learn 1.8.0's default
# LogisticRegression() on the train rows the student submits and ranks the test
# rows by ROC AUC, so every point of AUC comes from preprocessing: units that
# differ by site, placeholders, text where numbers should be, labs that are
# missing because nobody ordered them, a post-outcome column. The student
# submits features, not predictions: `submission.csv.gz` = id + 1..300 numeric
# finite columns, every test id, and any subset of >= 4,000 train ids.
#
# Why `submission.csv.gz` and not the platform default `submission.csv`: a `.csv`
# upload is checked for an exact column-set match against `agent_template`, which
# a feature matrix cannot satisfy; `.csv.gz` skips that check by design. For the
# same reason the private labels are NOT named y_test.csv (whose upload would
# auto-extract the test ids into agent_template). Both need the platform's
# settable `submission_filename` (see the DPE challenge spec, "What ML-Arena
# needs"); the mechanism is identical to s2-dpe-energy-label (challenge 191).
#
# Ranked on ROC AUC, descending (the leaderboard's only direction), which is
# correct here. `metric` is the key; the displayed name is the schema label.
# Train AUC rides along as metric2: a train AUC far above the test AUC is the
# visible sign of a feature that knows the outcome on the train rows (here the
# discharge charges, or a target encoding fitted on the rows it encodes).
#
# Memory is not a concern here: ~9,100 rows x at most 300 float64 columns is
# ~22 MB. The engine is DPE's anyway (same kernel, same scorer), and the 4Gi
# env limit is simply unused.
CONFIG = {
    "name": "MLP S2 — Critical Care Survival",
    "kernel_version": "file_v1",
    "module_slug": "s2-data-preprocessing",
    "label": "Critical Care Survival",
    "metric": "auc",
    "metric2": "auc_train",
    "submission_filename": "submission.csv.gz",
    "max_upload_size_bytes": 50 * 1024 * 1024,
    # jobpod-file-v1-highmem-1: the engine challenge 191 runs on (file_v1,
    # high-memory pool). Shared so the two Session 2 challenges behave alike.
    "engine_id": 27,
    # Course bar on module 20 (s2-data-preprocessing). None until the ladder in
    # teacher_expert_pipeline.py has been measured on the shipped split; then
    # set to the AUC that the documented preprocessing steps reach. `attach`
    # falls back to benchmark_expected_score while this is None. A bar above
    # the benchmark is accepted by the builder only up to
    # expert_expected_score.
    "pass_threshold": 0.905,
    # The test AUC the package's own reference solution (the expert pipeline:
    # units, placeholders, normal-value imputation, indicators, encodings,
    # medical formulas) reaches on the shipped split. Pinned from the ladder,
    # like pass_threshold; it is the ceiling the builder checks the bar
    # against, never something a student sees.
    "expert_expected_score": 0.936766,
    "public_files": [
        "train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
        "EXPERTISE.pdf", "DICTIONARY.pdf",
    ],
    # Uploaded to the ENV folder next to env.py (build_competitions.py uploads
    # env.py itself separately, from the package root), never to the dataset.
    "private_files": ["labels_train.csv", "labels_test.csv"],
    "benchmark_file": "data/benchmark_submission.csv.gz",
    # No domain knowledge: the numeric columns as pandas reads them,
    # median-imputed and standardized (SimpleImputer + StandardScaler fitted on
    # the train rows), every train and test row. Scored by env.py on the file
    # prepare_data.py writes. The ladder on this split is in overview.md and in
    # teacher_expert_pipeline.py's output.
    # None until prepare_data.py has written the split and scored the
    # benchmark file with env.py; pin the value it prints here (the build
    # refuses to run while it is None).
    "benchmark_expected_score": 0.849869,
    # AUC is a rank statistic over ~2,700 test rows and lbfgs is deterministic,
    # so the score reproduces; 1e-5 rather than 1e-6 leaves room for a
    # different BLAS in the scoring image (the test s.e. of the AUC is ~0.01).
    "benchmark_score_tol": 1e-5,
    "dataset_label": "Critical Care Survival",
    # dataset.description is varchar(500) on the platform.
    "dataset_description": (
        "The SUPPORT cohort: 9,105 seriously ill adults admitted to 5 US "
        "hospitals, 1989-1994: site, demographics, diagnosis, day-3 vital signs "
        "and labs, functional status, charges (31 columns). "
        "train.csv.gz carries the target dead (died within 60 days of study "
        "entry); test.csv.gz does not. EXPERTISE.pdf explains the clinical "
        "knowledge, DICTIONARY.pdf each column, sample_submission.csv.gz the "
        "format."
    ),
    # Deterministic scorer: the same file always yields the same AUC.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "auc", "label": "ROC AUC", "source": "env",
         "agg": "mean", "format": "number", "precision": 4,
         "higher_is_better": True},
        # Neither direction is better: close to the test AUC is healthy, far
        # above it is a target leak on the train rows.
        {"key": "auc_train", "label": "Train AUC", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
        {"key": "n_features", "label": "Features", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0},
        {"key": "n_train_rows", "label": "Train rows", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0},
        {"key": "converged", "label": "Converged", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0,
         "higher_is_better": True},
    ],
    "is_public_initial": False,
}
