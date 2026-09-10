#!/usr/bin/env python3
"""Run a competition package's env.py locally, the way the platform would.

Mirrors the two worker contracts closely enough to catch authoring bugs before
a pod ever starts:

* flex_v1 — builds an `AgentProxy` per agent around a local `Agent()` instance
  and reproduces the executor's **error latch**: once a call raises, the channel
  is broken and every later call short-circuits to a failure
  (`workers/flex_v1/executor/agent_channel.py:154`). Env code that assumes it
  can keep calling after a crash fails here rather than on the leaderboard.
* file_v1 — calls `env.evaluate(submission_path)` on a local file.

It also enforces the `metrics_detail` equal-mapping contract against the
package's declared `metrics_schema`, which the executor enforces at run time
(`workers/shared/executor/metric_contract.py`).

Usage:
    python localtest.py s3-adult-income
    python localtest.py s3-adult-income --submission data/benchmark_submission.csv
"""
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config(pkg_dir):
    ns = {}
    exec((pkg_dir / "config.py").read_text(), ns)
    return ns["CONFIG"]


class LocalProxy:
    """AgentProxy + AgentChannel, with the real latch semantics."""

    def __init__(self, agent, index):
        self._agent = agent
        self._index = index
        self.last_error = None
        self.is_broken = False
        self.n_calls = 0

    def call(self, method, *args, timeout=None, default=..., catch_errors=False, **kwargs):
        self.last_error = None
        if self.is_broken:
            return self._fail("channel already broken by an earlier failure",
                              default, catch_errors)
        self.n_calls += 1
        fn = getattr(self._agent, method, None)
        if fn is None:
            self.is_broken = True
            return self._fail(f"agent has no method {method!r}", default, catch_errors)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self.is_broken = True
            return self._fail(f"{type(exc).__name__}: {exc}", default, catch_errors)

    def _fail(self, message, default, catch_errors):
        self.last_error = message
        if default is not ...:
            return default
        if catch_errors:
            return None
        raise RuntimeError(message)


def check_metrics(results, schema):
    """Equal-mapping: metrics_detail keys == declared source:'env' keys."""
    expected = {d["key"] for d in (schema or []) if d.get("source") == "env"}
    problems = []
    for row in results:
        got = set((row.get("metrics_detail") or {}).keys())
        if got != expected:
            problems.append(
                f"agent_index={row.get('agent_index')}: metrics_detail keys {sorted(got)} "
                f"!= declared env keys {sorted(expected)}"
            )
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--agent", default="agent.py",
                    help="flex_v1 only: agent file to run (default: the benchmark agent)")
    ap.add_argument("--agents", type=int, default=1,
                    help="flex_v1 only: how many copies of the agent to evaluate")
    ap.add_argument("--submission", default=None,
                    help="file_v1 only: submission file (default: the package's benchmark)")
    args = ap.parse_args()

    pkg_dir = (HERE / args.package).resolve()
    if not pkg_dir.is_dir():
        sys.exit(f"No such package: {pkg_dir}")
    cfg = load_config(pkg_dir)
    kind = cfg["kernel_version"]
    print(f"== {cfg['name']}  [{kind}]  {pkg_dir.name}")

    # Stage env.py + its private files into a scratch dir, exactly as the
    # platform lays out the env folder. This is what catches "env.py reads its
    # private label file from next to itself but the file was never uploaded".
    stage = Path(tempfile.mkdtemp(prefix=f"envstage-{pkg_dir.name}-"))
    shutil.copy2(pkg_dir / "env.py", stage / "env.py")
    for rel in cfg.get("private_files", []):
        src = pkg_dir / "data" / rel
        if not src.exists():
            src = pkg_dir / rel
        if not src.exists():
            sys.exit(f"Missing private env file {rel!r} — run prepare_data.py first.")
        shutil.copy2(src, stage / Path(rel).name)
    print(f"   env staged in: {stage}")

    sys.path.insert(0, str(pkg_dir))
    env_module = load_module("env", stage / "env.py")
    env = env_module.Env(is_evaluation=True)

    if kind == "file_v1":
        submission = Path(args.submission) if args.submission \
            else pkg_dir / cfg["benchmark_file"]
        if not submission.is_absolute():
            submission = (pkg_dir / submission).resolve()
        if not submission.exists():
            sys.exit(f"Missing submission file: {submission}")
        print(f"   submission: {submission}")
        outcome = env.evaluate(str(submission))
    else:
        agent_module = load_module("agent", pkg_dir / args.agent)
        proxies = [LocalProxy(agent_module.Agent(), i) for i in range(args.agents)]
        print(f"   agent: {args.agent}  x{args.agents}")
        outcome = env.evaluate(proxies, [{} for _ in proxies])
        for p in proxies:
            print(f"   agent[{p._index}] calls={p.n_calls} broken={p.is_broken}")

    results = outcome["agent_results"]
    print(json.dumps(outcome, indent=2, default=str)[:4000])

    problems = check_metrics(results, cfg.get("metrics_schema"))
    for row in results:
        if row.get("is_agent_code_error"):
            problems.append(f"agent_index={row.get('agent_index')}: is_agent_code_error set")

    # Only hold the *reference* solution to the declared benchmark score — the
    # deliberately-broken agents exist to be run with --agent and to fail.
    is_reference = (kind == "file_v1"
                    and args.submission is None) or (kind != "file_v1"
                                                     and args.agent == cfg["benchmark_file"])
    expected_score = cfg.get("benchmark_expected_score") if is_reference else None
    if not is_reference:
        print("   (not the package's reference solution — benchmark score not enforced)")
    if expected_score is not None:
        for row in results:
            got = row.get("score")
            if got is None or abs(got - expected_score) > cfg.get("benchmark_score_tol", 1e-9):
                problems.append(
                    f"agent_index={row.get('agent_index')}: score {got} != "
                    f"expected benchmark score {expected_score}"
                )

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nOK — env contract, metrics schema and benchmark score all check out.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
