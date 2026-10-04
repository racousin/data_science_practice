"""DS-Harness agent (challenge 194). Upload this file as agent.py, with the modules it
imports (e.g. dsh.py from github.com/racousin/ds-harness), and choose the PyTorch runtime.

Agent() has 60 s to load its model (one of the five models the platform mounts:
Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct,
DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-1.7B). solve() is then called 15 times with 8
tasks; each call must return within 40 s. A call that raises or takes longer ends the
run and the deployment fails: catch errors per task and answer a number anyway.
Formats: schema.md.
"""


class Agent:
    def __init__(self):
        # Load your model here.
        pass

    def solve(self, tasks):
        # tasks: list of {"id", "objective", "files": [paths]}.
        # Return one {"id", "answer": <a number>} per task.
        return [{"id": t["id"], "answer": 0} for t in tasks]
