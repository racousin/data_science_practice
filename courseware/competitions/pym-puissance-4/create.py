"""Create (idempotently) the "Puissance 4 — Python pour la médecine" challenge on
prod ML-Arena as a copy of challenge 65 (PettingZoo · Connect-Four).

    PYTHONPATH=<mlarena-sdk 3.0.0> python create.py

Steps (each safe to re-run; the challenge id lives in state.json):
  1. snapshot challenge 65 -> source_snapshot.json (read only)
  2. create the copy (copy_from_challenge_id=65, hidden) unless state.json has it
  3. compare the copy with 65 field by field; fail on any unexpected difference
  4. French agent template, French overview, cover image check
  5. upload benchmark_agent.py as the benchmark agent.py, run it, check it
  6. start the challenge, make it public
  7. read it back with the student key
Keys are read from courseware/.env and never printed.
"""
from __future__ import annotations

import filecmp
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from mlarena.client import MLArenaClient

HERE = Path(__file__).resolve().parent
COURSEWARE = HERE.parents[1]
BASE_URL = "https://ml-arena.com"
SOURCE_ID = 65
NAME = "Puissance 4 — Python pour la médecine"
DESCRIPTION = (
    "Puissance 4 sur une grille 6×7 : écrivez en Python un agent qui choisit la "
    "colonne où lâcher son jeton. Votre agent joue les deux places, en "
    "alternance, contre les agents des autres participants ; classement ELO."
)
STATE = HERE / "state.json"

# Rows that legitimately differ between two challenges.
IDENTITY_KEYS = {"id", "challenge_id", "created_at_ts", "environment_id", "user_id",
                 "name", "description", "benchmark_simulation_result_id"}
# Changed on purpose: the template is translated to French.
INTENDED = {"configuration.agent_template"}


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


def snapshot(c: MLArenaClient, cid: int) -> dict:
    challenge = c.challenge(cid)
    for runtime in challenge.get("runtime_options", []):
        runtime.pop("requirement", None)  # long pip freeze, not config
    return {
        "creator_challenge": c.creator_challenge(cid),
        "challenge": challenge,
        "tags": c.challenge_tags(cid),
        "env_files": c.list_env_files(cid)["files"],
        "benchmark_files": c.list_benchmark_files(cid)["files"],
        "markdown": c.challenge_markdown(cid)["content"],
    }


def diff(a, b, path="") -> list[str]:
    """Paths where a and b differ, skipping identity keys."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in IDENTITY_KEYS:
                continue
            out += diff(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        return out
    return [] if a == b else [path]


def compare(src: dict, new: dict) -> list[str]:
    s, n = src["creator_challenge"], new["creator_challenge"]
    out = []
    for section in ("configuration", "evaluation", "environment", "machine"):
        out += diff(s[section], n[section], section)
    cfg_s = dict(s["config"]["values"])
    cfg_n = dict(n["config"]["values"])
    out += diff(cfg_s, cfg_n, "config.values")
    out += diff({"kernel": s["configuration"]["kernel_version"]},
                {"kernel": n["configuration"]["kernel_version"]})
    if [t["name"] for t in src["tags"]] != [t["name"] for t in new["tags"]]:
        out.append("tags")
    if set(src["env_files"]) != set(new["env_files"]):
        out.append("env_files.names")
    for fname, meta in src["env_files"].items():
        if fname in new["env_files"] and meta["content"] != new["env_files"][fname]["content"]:
            out.append(f"env_files.{fname}")
    return out


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
    results = run["submission_results"]
    if len(results) != 2:
        raise RuntimeError(f"expected 2 seats in the benchmark run, got {len(results)}")
    for seat in results:
        if seat["agent_error_type"]:
            raise RuntimeError(f"benchmark agent error {seat['agent_error_type']}: "
                               f"{seat['agent_error_message']}")
        print(f"  seat {seat['agent_attached_player_id']}: score={seat['score']} "
              f"n_episodes={seat['n_episodes']} outcome={seat['game_outcome']} "
              f"status={seat['submission_status']} "
              f"action_time_max={seat['action_time_max_sec']:.4f}s")


def main() -> None:
    env = load_env()
    c = MLArenaClient(env["MLARENA_CREATOR_API_KEY"], BASE_URL)

    # 1. source snapshot
    src = snapshot(c, SOURCE_ID)
    (HERE / "source_snapshot.json").write_text(
        json.dumps(src, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"[1] snapshot of challenge {SOURCE_ID} written")

    # 2. create once
    state = read_state()
    if "challenge_id" not in state:
        created = c.create_challenge(NAME, "flex_v1", description=DESCRIPTION,
                                     copy_from_challenge_id=SOURCE_ID, is_public=False)
        print(f"[2] created: {created}")
        state["challenge_id"] = created["challenge_id"]
        write_state(state)
    cid = state["challenge_id"]
    print(f"[2] challenge {cid}: {BASE_URL}/viewchallenge/{cid}")

    new = snapshot(c, cid)
    started = new["creator_challenge"]["status"] == "started"

    # 3. field-by-field comparison with the source
    diffs = [d for d in compare(src, new) if d not in INTENDED]
    if diffs:
        raise SystemExit(f"[3] copy differs from challenge {SOURCE_ID} on: {diffs}")
    print(f"[3] configuration, evaluation, environment, machine, config values, "
          f"tags and env files identical to challenge {SOURCE_ID}")

    # 4. French template, overview, cover image
    template = (HERE / "agent_template.py").read_text()
    if new["creator_challenge"]["configuration"]["agent_template"] != template:
        if started:
            raise SystemExit("[4] agent template differs but the challenge is started")
        c.update_agent_template(cid, template)
        print("[4] French agent template set")
    overview = (HERE / "overview.md").read_text().replace("__CHALLENGE_ID__", str(cid))
    if new["markdown"] != overview:
        c.set_challenge_markdown(cid, overview)
        print("[4] French overview set")
    with tempfile.TemporaryDirectory() as tmp:
        src_img = c.challenge_image(SOURCE_ID, os.path.join(tmp, "src"))
        new_img = c.challenge_image(cid, os.path.join(tmp, "new"))
        if not filecmp.cmp(src_img, new_img, shallow=False):
            raise SystemExit("[4] cover image differs from the source's")
    print("[4] cover image identical to the source's")

    # 5. benchmark
    bench_src = (HERE / "benchmark_agent.py").read_text()
    if not started:
        files = c.list_benchmark_files(cid)["files"]
        run = c.benchmark_status(cid)["run"]
        current = files.get("agent.py", {}).get("content")
        if current != bench_src or set(files) != {"agent.py"} or not run \
                or run["job_status"] != "completed":
            for fname in set(files) - {"agent.py"}:
                c.delete_benchmark_file(cid, fname)
            c.update_benchmark_file_content(cid, "agent.py", bench_src)
            print("[5] benchmark agent.py uploaded; running the benchmark")
            c.run_benchmark(cid)
            run = wait_for_benchmark(c, cid)
        check_benchmark(run)
        print("[5] benchmark completed without env or agent error")

        # 6. start
        c.start_challenge(cid)
        print("[6] challenge started")
    if not new["creator_challenge"]["is_public"] or not started:
        c.update_challenge(cid, is_public=True)
        print("[6] challenge public")

    final = c.creator_challenge(cid)
    if final["status"] != "started" or final["is_public"] is not True:
        raise SystemExit(f"[6] unexpected final state: status={final['status']} "
                         f"is_public={final['is_public']}")

    # 7. student view (read only)
    student = MLArenaClient(env["MLARENA_STUDENT_API_KEY"], BASE_URL)
    seen = student.challenge(cid)
    if seen["name"] != NAME or seen["status"] != "started":
        raise SystemExit(f"[7] student sees name={seen['name']!r} status={seen['status']!r}")
    print(f"[7] visible to the student key: {seen['name']} ({seen['status']})")

    state.update({"url": f"{BASE_URL}/viewchallenge/{cid}", "status": final["status"],
                  "is_public": final["is_public"], "copied_from": SOURCE_ID})
    write_state(state)


if __name__ == "__main__":
    sys.exit(main())
