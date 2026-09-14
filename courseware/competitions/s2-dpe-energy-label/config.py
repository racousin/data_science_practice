# MS2A - Machine Learning Practice, Session 2 ("Data Preprocessing"). ADEME's
# public DPE records for 100,000 dwellings in 8 départements (one per climate
# zone), DPEs issued 2024-07-01 .. 2025-12-31: is the energy label E, F or G?
#
# The model is FIXED. The scorer fits scikit-learn 1.8.0's default
# LogisticRegression() on the train rows the student submits and ranks the test
# rows by ROC AUC, so every point of AUC comes from preprocessing. The student
# submits features, not predictions: `submission.csv.gz` = id + 1..300 numeric
# finite columns, every test id, and any subset of >= 20,000 train ids.
#
# Why `submission.csv.gz` and not the platform default `submission.csv`: a `.csv`
# upload is checked for an exact column-set match against `agent_template`, which
# a feature matrix cannot satisfy; `.csv.gz` skips that check by design. For the
# same reason the private labels are NOT named y_test.csv (whose upload would
# auto-extract 30,000 ids into agent_template). Both need the platform's
# settable `submission_filename` (see the challenge spec, "What ML-Arena needs").
#
# Ranked on ROC AUC, descending (the leaderboard's only direction), which is
# correct here. `metric` is the key; the displayed name is the schema label.
# Train AUC rides along as metric2: a train AUC far above the test AUC is the
# visible sign of a target encoding fitted on the rows it encodes.
#
# Memory: the scorer holds up to 100,000 rows x 300 float64 columns. Measured
# peak RSS is in the package's local test; run it on an engine with
# env_memory_limit >= 3Gi (the 1024Mi default is too small).
CONFIG = {
    "name": "MLP S2 — DPE Energy Label",
    "kernel_version": "file_v1",
    # Detached from module 20 (s2-data-preprocessing) on 2026-09-15: Session 2
    # keeps Critical Care Survival (192) as its only challenge. None means
    # `make competitions-attach` leaves this one off every module; the
    # challenge itself stays live and public.
    "module_slug": None,
    "label": "DPE Energy Label",
    "metric": "auc",
    "metric2": "auc_train",
    "submission_filename": "submission.csv.gz",
    "max_upload_size_bytes": 200 * 1024 * 1024,
    # jobpod-file-v1-highmem-1: file_v1 on the high-memory pool, env 4Gi. The
    # default file_v1 engine (12) caps env memory at 1024Mi, and the scorer
    # peaks at ~1.0 GB on a dense 100k x 250 submission.
    "engine_id": 27,
    # Course bar on module 20: above the benchmark (0.7606) on purpose — that
    # gap is the point, 0.85 takes the documented domain steps (cleaning,
    # one-hot codes: 0.888). The builder accepts a bar above the benchmark
    # only up to expert_expected_score, the test AUC of the package's own
    # reference solution (teacher_expert_pipeline.py on the shipped split).
    "pass_threshold": 0.85,
    "expert_expected_score": 0.92705,
    "public_files": [
        "train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
        "EXPERTISE.md", "DICTIONNAIRE.md",
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
    # Measured on the shipped split (prepare_data.py, 2026-09-14): 31 numeric
    # columns (39 read as numeric, 8 never filled), lbfgs converges in 20
    # iterations; train AUC 0.755410.
    "benchmark_expected_score": 0.760606,
    # AUC is a rank statistic over 30,000 rows and lbfgs is deterministic, so
    # the score reproduces; 1e-5 rather than 1e-6 leaves room for a different
    # BLAS in the scoring image (the test s.e. of the AUC is ~0.003).
    "benchmark_score_tol": 1e-5,
    "dataset_label": "DPE Energy Label",
    # dataset.description is varchar(500) on the platform.
    "dataset_description": (
        "ADEME energy diagnoses (DPE, Licence Ouverte 2.0) of 100,000 houses "
        "and flats in 8 départements, July 2024 to December 2025, as the "
        "diagnostician recorded them. train.csv.gz carries the target "
        "classe_efg (label E, F or G); test.csv.gz does not. EXPERTISE.md "
        "explains the regulation, DICTIONNAIRE.md each column, "
        "sample_submission.csv.gz the format."
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
