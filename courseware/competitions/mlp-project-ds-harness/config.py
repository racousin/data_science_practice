# The project of MS2A - Machine Learning Practice, course 15 ("DS-Harness").
# Teams build an AI system around a small language model: it reads an objective
# and its files, decides what kind of task it faces, chooses what reaches the
# model and which tools it calls, parses and checks one number per task, all
# inside 40 s per call of 8 tasks. Spec: platform repo
# docs/plan_challenge_194_ds_harness.md.
#
# Why flex_v1: the system IS the submission, so it has to run -- agent.py + dsh.py
# in an agent container with the GPU, called batch by batch (scoring.run_agent).
# Pinned on prod to machine 34 (gpu-vm-1, agent GPU 0).
#
# The repository is public: the generators (gen/), the seed of the private split
# (data/private_seed.txt), the anchors (anchors2/) and the measurements stay
# local. See .gitignore in this folder.
CONFIG = {
    "name": "MLP Project — DS-Harness",
    "kernel_version": "flex_v1",
    "module_slug": "mlp-project",
    "label": "DS-Harness — build an AI system around a small language model",
    "metric": "score",
    # The builder still speaks the pre-machine engine_id API: the machine is set
    # in the creator settings (machine 34), not here.
    "engine_id": None,
    # The whole job: container start + model load (60 s) + 15 calls x 40 s at most
    # (11 min). The margin keeps the platform deadline out of the way: the only time
    # rules a participant meets are the 60 s load and the 40 s per call.
    "simulation_timeout_sec": 900,
    # Agent() gets min(60, max(5, 10 x this)) s to load its model
    # (workers/flex_v1/executor/main.py): 6.0 gives the 60 s. Each solve() call
    # carries its own timeout= (scoring.CALL_TIMEOUT_S).
    "agent_max_time_per_step_second": 6.0,
    # A no-model placeholder: the platform runs a flex_v1 benchmark on the kernel's
    # first agent runtime, which has no torch. It checks the env, the file delivery
    # (it reads every file), the scoring and the metrics: 0.0 on the 16-task test run.
    "benchmark_file": "benchmark_agent.py",
    "agent_template_file": "agent_template.py",
    "benchmark_expected_score": 0.0,
    "benchmark_score_tol": 1e-9,
    # The code (scorer, local runner, kit) is the public repository
    # github.com/racousin/ds-harness; the platform ships the data only.
    "public_files": ["dev.json"],
    # env.py imports scoring from next to itself -- the same bytes as the
    # repository's. hf_models.json narrows the agent's model cache to the five models.
    "private_files": ["private.json", "dev.json", "scoring.py", "hf_models.json"],
    "dataset_label": "DS-Harness",
    "dataset_description": (
        "dev.json: 180 public tasks with their files, answers and types. The scorer the "
        "leaderboard runs, the local runner and the starter kit are in the repository "
        "github.com/racousin/ds-harness."
    ),
    # One test run (16 dev tasks, 2 calls), one scored run (120 private tasks, 15 calls).
    "deployment_nb_constraint_run": 1,
    "deployment_nb_initial_score_run": 1,
    "metrics_schema": [
        {"key": "score", "label": "Score (/100)", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "calc", "label": "No files", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "table", "label": "Tables", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "fit", "label": "Fits", "source": "env",
         "agg": "mean", "format": "number", "precision": 1, "higher_is_better": True},
        {"key": "tasks_answered", "label": "Answered", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0, "higher_is_better": True},
        {"key": "format_errors", "label": "Format errors", "source": "env",
         "agg": "mean", "format": "integer", "precision": 0, "higher_is_better": False},
    ],
    "is_public_initial": False,
}
