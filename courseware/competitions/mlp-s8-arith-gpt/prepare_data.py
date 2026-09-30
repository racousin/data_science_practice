#!/usr/bin/env python3
"""Build data/ for mlp-s8-arith-gpt.

    source ../.id-salt                 # MLARENA_ID_SALT: the test set's secret
    python prepare_data.py             # problem sets; trains the benchmark if missing
    python prepare_data.py --retrain   # retrain benchmark.safetensors too

Writes:
    data/dev_problems.csv       public   200 products with their answers
    data/test_problems.csv      private  1000 products — the leaderboard
    data/benchmark.safetensors  benchmark — the answer-only model the starter
                                notebook trains ("1234*5678=#7006652"). It
                                scores ~0% exact: the point of the challenge.
                                Retraining it can move the score; re-pin
                                benchmark_expected_score with `python
                                localtest.py mlp-s8-arith-gpt --submission
                                data/benchmark.safetensors`.

The dev set's seed is public. The test set's is derived from MLARENA_ID_SALT
(../_dataset_ids.py): this file is in a public repository, and a public seed
would hand out the 1000 private prompts to train on — memorising them is
easier than learning to multiply.

`train` is the loop the starter notebook shows, and the one
teacher_reference_solution.py (local only) calls with a chain-of-thought format.
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arith_gpt import GPT, PAD, encode, evaluate_problems, load_weights, save_weights  # noqa: E402

DATA = HERE / "data"
N_DEV, N_TEST = 200, 1000
DEV_SEED = 8_2026


def test_seed() -> int:
    """The private test set's seed: keyed by the salt, never in the repository."""
    sys.path.insert(0, str(HERE.parent))
    from _dataset_ids import _digest

    return int.from_bytes(_digest("mlp-s8-arith-gpt", "test", size=8), "big")


def operand(rng: random.Random) -> int:
    return rng.randint(1000, 9999)


def problem_set(seed: int, n: int, exclude=frozenset()) -> list[tuple]:
    rng, rows, seen = random.Random(seed), [], set(exclude)
    while len(rows) < n:
        a, b = operand(rng), operand(rng)
        if (a, b) not in seen:
            seen.add((a, b))
            rows.append((a, b, a * b))
    return rows


def write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "answer"])
        w.writerows(rows)


def answer_only(a: int, b: int) -> str:
    """The completion the model learns to write after the prompt "a*b="."""
    return f"#{a * b}\n"


def make_batch(rng: random.Random, n: int, completion):
    """Prompt + completion token ids; the loss (label != -100) sees the completion only."""
    seqs, plen = [], []
    for _ in range(n):
        a, b = operand(rng), operand(rng)
        prompt = f"{a}*{b}="
        seqs.append(encode(prompt + completion(a, b)))
        plen.append(len(prompt))
    T = max(map(len, seqs)) - 1
    x = torch.full((n, T), PAD)
    y = torch.full((n, T), -100)
    for i, s in enumerate(seqs):
        s = torch.tensor(s)
        x[i, : len(s) - 1] = s[:-1]
        y[i, plen[i] - 1 : len(s) - 1] = s[plen[i]:]
    return x, y


def train(completion, minutes: float, device: str, lr: float = 1e-3, batch: int = 256,
          seed: int = 0, max_steps: int | None = None, log_every: int = 500) -> GPT:
    """AdamW, 100-step warmup, cosine to 0.1*lr over the wall-clock budget."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    budget, t0, step = minutes * 60, time.monotonic(), 0
    while (elapsed := time.monotonic() - t0) < budget and (max_steps is None or step < max_steps):
        frac = elapsed / budget if max_steps is None else step / max_steps
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        x, y = make_batch(rng, batch, completion)
        loss = torch.nn.functional.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % log_every == 0:
            print(f"  step {step}  {elapsed / 60:.1f} min  loss {loss.item():.4f}", flush=True)
    return model.cpu().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    dev = problem_set(DEV_SEED, N_DEV)
    test = problem_set(test_seed(), N_TEST, exclude={(a, b) for a, b, _ in dev})
    write_csv(DATA / "dev_problems.csv", dev)
    write_csv(DATA / "test_problems.csv", test)
    print(f"dev {len(dev)} / test {len(test)} problems")

    bench = DATA / "benchmark.safetensors"
    if args.retrain or not bench.exists():
        # CPU and a step count, not a clock: the same file on any machine.
        print("training the answer-only benchmark (3000 steps, CPU)")
        save_weights(train(answer_only, minutes=60, device="cpu", max_steps=3000), str(bench))
    torch.set_num_threads(1)
    result = evaluate_problems(load_weights(str(bench)), test)
    print(f"benchmark on the test set: exact {result['accuracy']:.4f}  "
          f"per digit {result['digit_accuracy']:.4f}  answered {result['answered']}")


if __name__ == "__main__":
    main()
