"""DS-Harness agent (challenge 194). Upload this file as agent.py, with the modules it
imports (e.g. dsh.py from github.com/racousin/ds-harness), and choose the PyTorch runtime.

A solve() call that raises or misses its timeout ends the run and the deployment fails:
no score. Catch errors per task, answer with a placeholder instead of raising, and stop at
each task's time_budget_s. The job ends 399 s after it starts, model loading included:
plan for 330 s of answering (about 2 s per level-1/2 task, 8 s per level-3 task).
Packages: torch, transformers, accelerate, pandas, numpy, sympy, matplotlib (no
scikit-learn, scipy or statsmodels). Formats and delivery: schema.md.
"""


class Agent:
    def __init__(self):
        # Load your model here: at most 60 s on the platform.
        pass

    def solve(self, tasks):
        # tasks: list of {"id", "prompt", "files", "answer_type", "time_budget_s"}.
        # Return one {"id", "answer", "trace"} per task.
        return [{"id": t["id"], "answer": 0.0, "trace": ""} for t in tasks]
