# Session 8 of MS2A - Machine Learning Practice ("NLP 2"), course 15.
# A fixed 0.42M-parameter GPT (arith_gpt.py: 2 layers, 4 heads, width 128,
# 160-token context, tied embeddings, a 19-character vocabulary) has to
# multiply two 4-digit numbers. Participants submit only weights, trained from
# scratch — the architecture, the greedy decoder, the 128-token budget and the
# parser are the grader's. What they control is the training data: above all,
# what the model writes before the "#" that precedes its answer.
#
# Why file_v1: the submission is a safetensors file the env loads as data, so
# nothing the participant wrote runs. flex_v1 would run their agent.py, which
# could compute the answer without the model.
#
# The lesson is chain of thought as extra computation per token. Measured
# (scratchpad runs, Apple M4 GPU, one seed each, 2026-09-30, 1000 products):
#
#     answer only "#7006652", 10 min             0% exact, 45% per digit
#     the scratchpad of overview hints 1-3       90.4% exact at 10 min
#     the best format found (hint 4, further)    99% at 2.3 min, 100% at 10 min
#
# The notebook trains the first, its hints lead to the second, the third is
# the top of the board (teacher_reference_solution.py, local only).
CONFIG = {
    "name": "MLP S8 — Arithmetic GPT",
    "kernel_version": "file_v1",
    "module_slug": "s8-nlp-2",
    "label": "Arithmetic GPT — teach a fixed tiny GPT to multiply 4-digit numbers: design the chain of thought it trains on",
    "metric": "accuracy",
    "submission_filename": "weights.safetensors",
    # fp32 is 1.7 MB; arith_gpt.load_weights refuses more than 5 MB.
    "max_upload_size_bytes": 5 * 1024 * 1024,
    # Decoding stops itself at env.DECODE_BUDGET_SEC (300 s); this is the
    # platform maximum, so the pod deadline never cuts a scored run.
    "simulation_timeout_sec": 600,
    "public_files": ["arith_gpt.py", "dev_problems.csv"],
    # env.py imports arith_gpt from next to itself — the same bytes students get.
    "private_files": ["arith_gpt.py", "test_problems.csv"],
    # The starter notebook's answer-only model, trained by prepare_data.py
    # (3000 steps on CPU, seed 0): exact 0.0, per digit 0.4566.
    "benchmark_file": "data/benchmark.safetensors",
    "benchmark_expected_score": 0.0,
    "benchmark_score_tol": 0.002,
    # Half the products right needs a chain of thought: answer only stays at 0.
    "pass_threshold": 0.5,
    # teacher_reference_solution.py --fmt cot, 10 min on the M4 GPU: 0.941
    # exact, 0.991 per digit. The bar leaves room for a slower GPU.
    "expert_expected_score": 0.941,
    "dataset_label": "Arithmetic GPT",
    "dataset_description": (
        "arith_gpt.py is the fixed model, decoder, answer parser and scorer the "
        "leaderboard runs. dev_problems.csv holds 200 4-digit x 4-digit "
        "products with their answers, for scoring locally with "
        "arith_gpt.evaluate_problems. There is no training set: generate your own."
    ),
    # Deterministic scorer: greedy decoding, a fixed test set.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "accuracy", "label": "Exact", "source": "env",
         "agg": "mean", "format": "percent", "precision": 1,
         "higher_is_better": True},
        {"key": "digit_accuracy", "label": "Per digit", "source": "env",
         "agg": "mean", "format": "percent", "precision": 1,
         "higher_is_better": True},
        {"key": "decode_seconds", "label": "Decode time", "source": "env",
         "agg": "mean", "format": "seconds", "precision": 1,
         "higher_is_better": False},
    ],
    "is_public_initial": False,
}
