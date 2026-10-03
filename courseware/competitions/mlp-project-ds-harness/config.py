# The project of MS2A - Machine Learning Practice, course 15 ("DS-Harness").
# Teams build a harness around one small LLM (the models of the GPU VM's
# offline HF cache) to solve data-science tasks written in prose: arithmetic
# and statistics (L1), word problems and multi-step table queries (L2),
# forecasts and tabular predictions (L3). 25 % of the project grade is this
# leaderboard, 75 % an oral. Design record: thinking/06_synthesis.md.
#
# Why flex_v1: the harness IS the submission, so it has to run — agent.py +
# dsh.py in an agent container with the GPU, called task batch by task batch
# (scoring.run_agent). Pinned to the GPU engine GSM8K (165) runs on.
#
# The repository is public: the generators (draft/tasks/), the secret seed of
# the private split (data/private_seed.txt), the instructor anchor (anchors/)
# and the experiments stay local. See .gitignore in this folder.
#
# Anchors on the private set, platform rules: A0 naive 0.6, A1 kit baseline
# (the leaderboard row "A1 — kit baseline"), A2 instructor 61.0 (README.md).
CONFIG = {
    "name": "MLP Project — DS-Harness",
    "kernel_version": "flex_v1",
    "module_slug": "mlp-project",
    "label": "DS-Harness — build a harness around a small LLM to solve data-science tasks",
    "metric": "score",
    # Pinned on prod to machine 34 (gpu-vm-1, the GPU VM, agent GPU 0). The
    # builder still speaks the pre-machine engine_id API: set the machine in
    # the creator settings, not here.
    "engine_id": None,
    # The whole job: model load + the private set. The env stops sending tasks
    # at eval_context["deadline_monotonic"] (set max(5 s, 5 %) earlier): a
    # batch whose full budget no longer fits is not sent.
    "simulation_timeout_sec": 420,
    # Agent() gets min(60, max(5, 10 x this)) s to load its model
    # (workers/flex_v1/executor/main.py); every solve() call carries its own
    # timeout= from scoring.batch_budget.
    "agent_max_time_per_step_second": 6.0,
    # A no-model placeholder: the platform runs a flex_v1 benchmark on the
    # kernel's first agent runtime, which has no torch (backend run_benchmark).
    # It checks env, delivery, scoring and metrics; the kit baseline is measured
    # as an ordinary submission on the torch runtime. Score on the 30-task test
    # run (benchmark jobs are is_test): 0.0, 30/30 answered, 5 format errors.
    "benchmark_file": "benchmark_agent.py",
    # Uploads are validated against it (class Agent, solve); without one the
    # platform default template requires predict(self, data).
    "agent_template_file": "agent_template.py",
    "benchmark_expected_score": 0.0,
    "benchmark_score_tol": 1e-9,
    # The code (scorer, platform env, local runners, starter kit) is the public
    # repository github.com/racousin/ds-harness (build_repo.py); the platform
    # ships the data only.
    "public_files": ["dev.json"],
    # env.py imports scoring from next to itself — the same bytes as the repository's.
    "private_files": ["private.json", "dev.json", "scoring.py"],
    "dataset_label": "DS-Harness",
    "dataset_description": (
        "dev.json: 178 public tasks with their answers. The evaluation code the "
        "leaderboard runs, the local runners and the starter kit are in the "
        "repository github.com/racousin/ds-harness."
    ),
    # One test run on a dev subset, one scored run on the private set.
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "score", "label": "Score (/100)", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "level_1", "label": "L1", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "level_2", "label": "L2", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "level_3", "label": "L3", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "heldout", "label": "Unseen families", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "tasks_answered", "label": "Answered", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0, "higher_is_better": True},
        {"key": "format_errors", "label": "Format errors", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0, "higher_is_better": False},
    ],
    "is_public_initial": False,
}
