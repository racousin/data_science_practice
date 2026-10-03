"""Start-gate benchmark: a well-typed placeholder for every task, no model.

The platform runs a flex_v1 benchmark on the first agent runtime of the kernel
(backend run_benchmark), which has no torch, so the kit baseline cannot be the
benchmark. This agent checks the env, the delivery, the scoring and the metrics
on the platform; the kit baseline is measured as an ordinary submission on the
torch runtime.
"""
PLACEHOLDER = {"number": 0.0, "category": "", "list": [], "vector": [], "predictions": []}


class Agent:
    def __init__(self):
        pass

    def solve(self, tasks):
        return [{"id": t["id"], "answer": PLACEHOLDER[t["answer_type"]], "trace": ""} for t in tasks]
