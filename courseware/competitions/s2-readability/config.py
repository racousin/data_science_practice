# Session 2 of MS2A - AI Engineering ("Agentic Coding"). Grades the Flesch
# reading-ease module Lab 2 has you build with an agent, against a pinned spec.
CONFIG = {
    "name": "PAIE S2 — Flesch reading-ease",
    "kernel_version": "flex_v1",
    "module_slug": "s2-agentic-coding",
    "label": "Flesch reading-ease",
    "metric": "pass_rate",
    "benchmark_file": "agent.py",
    "benchmark_expected_score": 1.0,
    "agent_template_file": "agent_template.py",
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "pass_rate", "label": "Pass rate", "source": "env",
         "agg": "mean", "format": "percent", "precision": 1},
        {"key": "mean_abs_error", "label": "Mean abs error", "source": "env",
         "agg": "mean", "format": "number", "precision": 4},
    ],
    "is_public_initial": False,
}
