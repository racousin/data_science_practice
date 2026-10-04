"""The start-gate benchmark: answers 0 to every task, no model.

The platform runs a flex_v1 benchmark on the kernel's first agent runtime, which has no
torch, so it cannot load a model. It checks the env, the file delivery, the scoring and
the metrics. It also reads every file it is given, so a delivery fault shows up here."""


class Agent:
    def solve(self, tasks):
        for t in tasks:
            for p in t["files"]:
                with open(p) as f:
                    f.read()
        return [{"id": t["id"], "answer": 0} for t in tasks]
