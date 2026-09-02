# Competition package metadata, read by tools/build_competitions.py.
#
# Session 1 of MS2A - AI Engineering ("Git & Python Packaging"). Grades the
# three functions Lab 1 asks for, on hidden texts.
CONFIG = {
    "name": "PAIE S1 — textstats",
    "kernel_version": "flex_v1",
    "module_slug": "s1-git-and-packaging",
    "label": "textstats correctness",
    "metric": "pass_rate",
    "benchmark_file": "agent.py",
    "benchmark_expected_score": 1.0,
    "agent_template_file": "agent_template.py",
    # Deterministic scorer: the hidden set is fixed, so one scored run per
    # submission is exact. Without this the competition inherits the server
    # default of 2 constraint + 10 score runs, i.e. 12 identical pods. 1 is the
    # floor the settings schema allows (`ge=1`), not 0.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "pass_rate", "label": "Pass rate", "source": "env",
         "agg": "mean", "format": "percent", "precision": 1},
        {"key": "word_count_ok", "label": "word_count", "source": "env",
         "agg": "mean", "format": "percent", "precision": 0},
        {"key": "char_frequencies_ok", "label": "char_frequencies", "source": "env",
         "agg": "mean", "format": "percent", "precision": 0},
        {"key": "longest_word_ok", "label": "longest_word", "source": "env",
         "agg": "mean", "format": "percent", "precision": 0},
    ],
    "is_public_initial": False,
}
