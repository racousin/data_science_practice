"""Create (idempotently) the "Gymnasium · FrozenLake-v1 8x8" challenge on prod
ML-Arena as a copy of challenge 48 (Gymnasium · CartPole-v1, flex_v1,
`gymnasium` env image), for MS2A Machine Learning Practice, Session 9.

    PYTHONPATH=<mlarena-sdk 4.6> python create.py

Steps (each safe to re-run; the challenge id lives in state.json):
  1. create the copy (copy_from_challenge_id=48, hidden) unless state.json has it
  2. env.py, settings (100 episodes per run), agent template, overview
  3. upload benchmark_agent.py as the benchmark agent.py, run it, check it
  4. start the challenge, make it public
  5. read it back with the student key
Keys are read from courseware/.env and never printed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mlarena.client import MLArenaClient

HERE = Path(__file__).resolve().parent
COURSEWARE = HERE.parents[1]
BASE_URL = "https://ml-arena.com"
SOURCE_ID = 48
NAME = "Gymnasium · FrozenLake-v1 8x8"
DESCRIPTION = (
    "Cross the slippery 8x8 frozen lake to the goal without falling into a hole. "
    "Score: success rate over 100 episodes. Built for tabular Q-learning."
)
STATE = HERE / "state.json"
N_EPISODES = 100
METRICS = [{
    "key": "reward", "label": "Success rate", "source": "score", "agg": "mean",
    "order": "desc", "format": "number", "unit": None, "precision": 3,
    "is_ranking": True, "visible": True,
}]
BENCH_MIN, BENCH_MAX = 0.10, 0.40  # always-right succeeds in ~23% of episodes


def load_env() -> dict:
    env = {}
    for line in (COURSEWARE / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def read_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def write_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def wait_for_benchmark(c: MLArenaClient, cid: int, timeout_s: int = 1800) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        run = c.benchmark_status(cid)["run"] or {}
        state = run.get("job_status")
        if state != last:
            print(f"  benchmark: {state}")
            last = state
        if state in ("completed", "failed", "reaped", "cancelled"):
            return run
        time.sleep(10)
    raise TimeoutError(f"benchmark did not finish in {timeout_s}s (last={last})")


def check_benchmark(run: dict) -> None:
    if run.get("job_status") != "completed":
        raise RuntimeError(f"benchmark {run.get('job_status')}: "
                           f"{run.get('job_error_type')}: {run.get('job_error_message')}")
    if run.get("env_error_type"):
        raise RuntimeError(f"benchmark env error {run['env_error_type']}: "
                           f"{run.get('env_error_message')}")
    (seat,) = run["submission_results"]
    if seat["agent_error_type"]:
        raise RuntimeError(f"benchmark agent error {seat['agent_error_type']}: "
                           f"{seat['agent_error_message']}")
    print(f"  score={seat['score']} n_episodes={seat['n_episodes']} "
          f"action_time_max={seat['action_time_max_sec']:.4f}s")
    if seat["n_episodes"] != N_EPISODES:
        raise RuntimeError(f"expected {N_EPISODES} episodes, got {seat['n_episodes']}")
    if not BENCH_MIN <= seat["score"] <= BENCH_MAX:
        raise RuntimeError(f"benchmark score {seat['score']} outside "
                           f"[{BENCH_MIN}, {BENCH_MAX}]")


def main() -> None:
    env = load_env()
    c = MLArenaClient(env["MLARENA_CREATOR_API_KEY"], BASE_URL)

    # 1. create once
    state = read_state()
    if "challenge_id" not in state:
        created = c.create_challenge(NAME, "flex_v1", description=DESCRIPTION,
                                     copy_from_challenge_id=SOURCE_ID, is_public=False)
        print(f"[1] created: {created}")
        state["challenge_id"] = created["challenge_id"]
        write_state(state)
    cid = state["challenge_id"]
    print(f"[1] challenge {cid}: {BASE_URL}/viewchallenge/{cid}")
    started = c.creator_challenge(cid)["status"] == "started"

    # 2. env, settings, template, overview
    if not started:
        env_py = (HERE / "env.py").read_text()
        if c.list_env_files(cid)["files"]["env.py"]["content"] != env_py:
            reply = c.update_env_file_content(cid, "env.py", env_py)
            print(f"[2] env.py written: {reply.get('validation')}")
        c.update_settings(cid, episode_budget_brackets=[[1_000_000_000, N_EPISODES]],
                          metrics=METRICS)
        print(f"[2] settings: {N_EPISODES} episodes per run, metric 'Success rate'")
        c.update_agent_template(cid, (HERE / "agent_template.py").read_text())
        print("[2] agent template set")
    overview = (HERE / "overview.md").read_text().replace("__CHALLENGE_ID__", str(cid))
    if c.challenge_markdown(cid)["content"] != overview:
        c.set_challenge_markdown(cid, overview)
        print("[2] overview set")

    # 3. benchmark
    if not started:
        bench_src = (HERE / "benchmark_agent.py").read_text()
        files = c.list_benchmark_files(cid)["files"]
        run = c.benchmark_status(cid)["run"]
        current = files.get("agent.py", {}).get("content")
        if current != bench_src or set(files) != {"agent.py"} or not run \
                or run["job_status"] != "completed":
            for fname in set(files) - {"agent.py"}:
                c.delete_benchmark_file(cid, fname)
            c.update_benchmark_file_content(cid, "agent.py", bench_src)
            print("[3] benchmark agent.py uploaded; running the benchmark")
            c.run_benchmark(cid)
            run = wait_for_benchmark(c, cid)
        check_benchmark(run)
        print("[3] benchmark completed")

        # 4. start
        c.start_challenge(cid)
        print("[4] challenge started")
    c.update_challenge(cid, is_public=True)
    final = c.creator_challenge(cid)
    if final["status"] != "started" or final["is_public"] is not True:
        raise SystemExit(f"[4] unexpected final state: status={final['status']} "
                         f"is_public={final['is_public']}")
    print("[4] started and public")

    # 5. student view (read only)
    student = MLArenaClient(env["MLARENA_STUDENT_API_KEY"], BASE_URL)
    seen = student.challenge(cid)
    if seen["name"] != NAME or seen["status"] != "started":
        raise SystemExit(f"[5] student sees name={seen['name']!r} status={seen['status']!r}")
    print(f"[5] visible to the student key: {seen['name']} ({seen['status']})")

    state.update({"url": f"{BASE_URL}/viewchallenge/{cid}", "status": final["status"],
                  "is_public": final["is_public"], "copied_from": SOURCE_ID})
    write_state(state)


if __name__ == "__main__":
    sys.exit(main())
