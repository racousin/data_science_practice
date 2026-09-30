"""MLP S8 — Arithmetic GPT (file_v1).

The model is fixed; the participant submits only its weights. `arith_gpt.py`
(the architecture, the greedy decoder, the answer parser and the scorer) is
handed out byte for byte and imported here, so a student's local score on
dev_problems.csv is computed by the same code as the leaderboard's.

Submission — `weights.safetensors`, written by `arith_gpt.save_weights`: the
fixed GPT's tensors (without the tied `head.weight`), float32/16 or bfloat16,
at most 5 MB. Anything else is rejected with a message naming the problem,
and never scored. The file is data: safetensors has no code path, and nothing
the participant wrote is executed.

Score — exact-match accuracy on the 1000 private 4-digit x 4-digit products of
test_problems.csv, a fraction. Per-digit accuracy is reported beside it.

Time — the executor has no wallclock of its own: only the pod deadline stops
a run, and a killed run records no score. Decoding therefore stops itself at
DECODE_BUDGET_SEC; problems not reached score wrong, and the message says so.
"""
import csv
import os
import time

import torch

from arith_gpt import MAX_NEW_TOKENS, WeightsError, evaluate_problems, load_weights

TEST_FILE = "test_problems.csv"
# Well inside simulation_timeout_sec (config.py): loading the image, torch and
# the weights, then delivering the outcome, all come out of the same budget.
DECODE_BUDGET_SEC = 300


class Env:
    def __init__(self, is_evaluation=True):
        self.is_evaluation = is_evaluation
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_FILE)
        with open(path, newline="") as fh:
            self.problems = [(int(r["a"]), int(r["b"]), int(r["answer"]))
                             for r in csv.DictReader(fh)]

    def evaluate(self, submission_path):
        # The env container's CPU quota is small; one thread avoids oversubscription.
        torch.set_num_threads(1)
        try:
            model = load_weights(submission_path)
        except WeightsError as exc:
            raise self.ParticipantSubmissionError(f"weights.safetensors rejected: {exc}") from None

        started = time.monotonic()
        result = evaluate_problems(model, self.problems, deadline=started + DECODE_BUDGET_SEC)
        seconds = time.monotonic() - started

        n = len(self.problems)
        unanswered = n - result["answered"]
        # Fractions: the console's "percent" format multiplies by 100.
        detail = {
            "accuracy": round(result["accuracy"], 4),
            "digit_accuracy": round(result["digit_accuracy"], 4),
            "decode_seconds": round(seconds, 1),
        }
        message = (f"exact {result['accuracy']:.1%}, per digit {result['digit_accuracy']:.1%} "
                   f"on {n} products. {unanswered} had no valid answer (no end token within "
                   f"{MAX_NEW_TOKENS} tokens, no '#', or a malformed number)")
        if seconds > DECODE_BUDGET_SEC:
            message += f"; decoding hit the {DECODE_BUDGET_SEC} s budget, the rest scored wrong"
        return {"agent_results": [{
            "agent_index": 0,
            "score": detail["accuracy"],
            "steps": 1,
            "metrics_detail": detail,
            "info_message": message + ".",
        }]}
