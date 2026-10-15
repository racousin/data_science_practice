"""Create (idempotently) the three "Survie à 2 mois" challenges of the course
*IA pour la médecine* on prod ML-Arena, one per session, each a copy of
challenge 172 ("2-Month Survival Prediction": file_v1, SUPPORT2, accuracy).

    PYTHONPATH=<mlarena-sdk 3.0.0> python create.py [s1 s2 s3]

Steps (each safe to re-run; the challenge ids live in state.json):
  1. snapshot challenge 172 -> source_snapshot.json (read only; the private
     y_test.csv and other large blobs are stored as sha256, never as content)
  2. per session: create the copy (copy_from_challenge_id=172, hidden) unless
     state.json has it
  3. compare the copy with 172 field by field (benchmark files: step 6); fail on
     any difference
  4. French overview (overview.md + session_<s>.md), cover image check
  5. datasets: the copy serves 172's dataset files, byte for byte (uploaded
     from 172 if the copy did not carry them)
  6. benchmark: 172's submission.csv (always `alive`; the copy carries only a
     header stub, so it is uploaded); run it, check the score
     equals the share of `alive` in 172's y_test
  7. course-data gate: the files the overview links (ai_for_medicine) must
     match the scorer: hosted X_test ids == y_test ids, hosted y_train_survie
     == the served y_train. Fails before anything is started otherwise.
  8. start, make public, read back with the student key
Keys are read from courseware/.env and never printed.
"""
from __future__ import annotations

import csv
import filecmp
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from mlarena.client import MLArenaClient

HERE = Path(__file__).resolve().parent
COURSEWARE = HERE.parents[1]
BASE_URL = "https://ml-arena.com"
SOURCE_ID = 172
STATE = HERE / "state.json"
HOSTED = ("https://raw.githubusercontent.com/racousin/ai_for_medicine/main/"
          "ia-medecine/donnees/support2/")

SESSIONS = {
    "s1": {
        "name": "Survie à 2 mois — Séance 1 : régression logistique",
        "title": "Séance 1 : régression logistique",
        "description": (
            "Prédire la survie à 2 mois de patients de réanimation (étude SUPPORT2) "
            "par régression logistique. Classification binaire, exactitude."),
    },
    "s2": {
        "name": "Survie à 2 mois — Séance 2 : préparation des données et forêt aléatoire",
        "title": "Séance 2 : préparation des données et forêt aléatoire",
        "description": (
            "Prédire la survie à 2 mois de patients de réanimation (étude SUPPORT2) : "
            "imputation, encodage, forêt aléatoire. Classification binaire, exactitude."),
    },
    "s3": {
        "name": "Survie à 2 mois — Séance 3 : réseau de neurones",
        "title": "Séance 3 : réseau de neurones",
        "description": (
            "Prédire la survie à 2 mois de patients de réanimation (étude SUPPORT2) "
            "avec un réseau de neurones. Classification binaire, exactitude."),
    },
}

# Rows that legitimately differ between two challenges.
IDENTITY_KEYS = {"id", "challenge_id", "created_at_ts", "environment_id", "user_id",
                 "name", "description", "benchmark_simulation_result_id"}


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


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def snapshot(c: MLArenaClient, cid: int) -> dict:
    challenge = c.challenge(cid)
    for runtime in challenge.get("runtime_options", []):
        runtime.pop("requirement", None)
    challenge.pop("statistics", None)  # live counters, not config
    return {
        "creator_challenge": c.creator_challenge(cid),
        "challenge": challenge,
        "tags": c.challenge_tags(cid),
        "env_files": c.list_env_files(cid)["files"],
        "benchmark_files": c.list_benchmark_files(cid)["files"],
        "markdown": c.challenge_markdown(cid)["content"],
    }


def sanitized(snap: dict) -> dict:
    """The snapshot as committed: private ground truth and big blobs as sha256."""
    out = json.loads(json.dumps(snap, default=str))
    cfg = out["creator_challenge"]["configuration"]
    cfg["agent_template"] = {"sha256": sha(cfg["agent_template"])}
    for group in ("env_files", "benchmark_files"):
        for fname, meta in out[group].items():
            if fname != "env.py":
                meta["content"] = {"sha256": sha(meta["content"]),
                                   "lines": meta["content"].count("\n")}
    return out


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
    out += diff(s["config"]["values"], n["config"]["values"], "config.values")
    if [t["name"] for t in src["tags"]] != [t["name"] for t in new["tags"]]:
        out.append("tags")
    for group in ("env_files",):
        if set(src[group]) != set(new[group]):
            out.append(f"{group}.names")
        for fname, meta in src[group].items():
            if fname in new[group] and meta["content"] != new[group][fname]["content"]:
                out.append(f"{group}.{fname}")
    for key in ("kernel_version", "submission_filename", "submission_layout",
                "number_of_agents", "agent_memory_limit", "agent_cpu_limit",
                "max_upload_size_bytes", "max_upload_files", "machine"):
        if src["challenge"][key] != new["challenge"][key]:
            out.append(f"challenge.{key}")
    return out


def dataset_files(c: MLArenaClient, cid: int) -> dict[str, bytes]:
    """{label: bytes} of every dataset file a challenge serves (creator view)."""
    out = {}
    for ds in c.creator_datasets(cid)["datasets"]:
        for f in ds["files"]:
            if f["label"] in out:
                raise SystemExit(f"challenge {cid} serves {f['label']} twice")
            resp = requests.get(f["download_url"], timeout=300)
            resp.raise_for_status()
            out[f["label"]] = resp.content
    return out


def ensure_datasets(c: MLArenaClient, cid: int, src_ds: list, src_files: dict) -> None:
    current = c.creator_datasets(cid)["datasets"]
    if not current:
        if len(src_ds) != 1:
            raise SystemExit(f"[5] challenge {SOURCE_ID} has {len(src_ds)} datasets, expected 1")
        ds = c.create_dataset(cid, src_ds[0]["label"], src_ds[0]["description"])
        with tempfile.TemporaryDirectory() as tmp:
            for f in src_ds[0]["files"]:
                path = os.path.join(tmp, f["label"])
                Path(path).write_bytes(src_files[f["label"]])
                c.upload_dataset_file(cid, ds["id"], path)
        print(f"[5] dataset '{ds['label']}' created and filled from challenge {SOURCE_ID}")
    elif [(d["label"], d["description"]) for d in current] != \
            [(d["label"], d["description"]) for d in src_ds]:
        raise SystemExit(f"[5] datasets differ from challenge {SOURCE_ID}: "
                         f"{[d['label'] for d in current]}")
    served = dataset_files(c, cid)
    if {k: sha(v) for k, v in served.items()} != {k: sha(v) for k, v in src_files.items()}:
        raise SystemExit(f"[5] dataset files differ from challenge {SOURCE_ID}'s")
    print(f"[5] serves {sorted(served)}, byte-identical to challenge {SOURCE_ID}")


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


def check_benchmark(run: dict, expected: float) -> float:
    if run.get("job_status") != "completed":
        raise RuntimeError(f"benchmark {run.get('job_status')}: "
                           f"{run.get('job_error_type')}: {run.get('job_error_message')}")
    if run.get("env_error_type"):
        raise RuntimeError(f"benchmark env error {run['env_error_type']}: "
                           f"{run.get('env_error_message')}")
    results = run["submission_results"]
    if len(results) != 1:
        raise RuntimeError(f"expected 1 result in the benchmark run, got {len(results)}")
    seat = results[0]
    if seat["agent_error_type"]:
        raise RuntimeError(f"benchmark submission error {seat['agent_error_type']}: "
                           f"{seat['agent_error_message']}")
    if abs(seat["score"] - expected) > 1e-6:
        raise RuntimeError(f"benchmark score {seat['score']} != expected {expected}")
    return seat["score"]


def course_data_gate(y_test: list[dict], served_y_train: bytes) -> list[str]:
    """Mismatches between the files the overview links and what the scorer uses."""
    hosted = {}
    for fname in ("X_test.csv", "y_train_survie.csv"):
        resp = requests.get(HOSTED + fname, timeout=120)
        resp.raise_for_status()
        hosted[fname] = read_csv(resp.text)
    problems = []
    x_ids = [r["patient_id"] for r in hosted["X_test.csv"]]
    y_ids = {r["patient_id"] for r in y_test}
    if len(x_ids) != len(set(x_ids)) or set(x_ids) != y_ids:
        problems.append(
            f"hosted X_test.csv: {len(set(x_ids) & y_ids)}/{len(y_ids)} of its "
            f"{len(x_ids)} patient_ids are in the scorer's y_test.csv")
    if hosted["y_train_survie.csv"] != read_csv(served_y_train.decode()):
        problems.append("hosted y_train_survie.csv differs from the served y_train.csv")
    return problems


def main(keys: list[str]) -> None:
    env = load_env()
    c = MLArenaClient(env["MLARENA_CREATOR_API_KEY"], BASE_URL)

    # 1. source snapshot
    src = snapshot(c, SOURCE_ID)
    src_ds = c.creator_datasets(SOURCE_ID)["datasets"]
    src_files = dataset_files(c, SOURCE_ID)
    committed = sanitized(src)
    committed["datasets"] = [
        {"label": d["label"], "description": d["description"],
         "files": [{"label": f["label"], "file_size_bytes": f["file_size_bytes"],
                    "sha256": sha(src_files[f["label"]])} for f in d["files"]]}
        for d in src_ds]
    (HERE / "source_snapshot.json").write_text(
        json.dumps(committed, indent=2, ensure_ascii=False) + "\n")
    src_bench = {k: v["content"] for k, v in src["benchmark_files"].items()}
    y_test = read_csv(src["env_files"]["y_test.csv"]["content"])
    alive_share = round(sum(r["outcome"] == "alive" for r in y_test) / len(y_test), 6)
    print(f"[1] snapshot of challenge {SOURCE_ID} written "
          f"({len(y_test)} test patients, {alive_share:.4f} alive)")

    state = read_state()
    ready = []
    for key in keys:
        spec = SESSIONS[key]
        print(f"=== {key}: {spec['name']}")

        # 2. create once
        entry = state.setdefault(key, {})
        if "challenge_id" not in entry:
            created = c.create_challenge(spec["name"], "file_v1",
                                         description=spec["description"],
                                         copy_from_challenge_id=SOURCE_ID,
                                         is_public=False)
            print(f"[2] created: {created}")
            entry["challenge_id"] = created["challenge_id"]
            write_state(state)
        cid = entry["challenge_id"]
        print(f"[2] challenge {cid}: {BASE_URL}/viewchallenge/{cid}")

        new = snapshot(c, cid)
        if new["creator_challenge"]["name"] != spec["name"]:
            raise SystemExit(f"[2] challenge {cid} is named {new['creator_challenge']['name']!r}")
        started = new["creator_challenge"]["status"] == "started"

        # 3. field-by-field comparison with the source
        diffs = compare(src, new)
        if diffs:
            raise SystemExit(f"[3] copy differs from challenge {SOURCE_ID} on: {diffs}")
        print(f"[3] configuration, evaluation, environment, machine, config values, "
              f"tags, env files (env.py, y_test.csv) and submission layout identical "
              f"to challenge {SOURCE_ID}")

        # 4. overview, cover image
        overview = ((HERE / "overview.md").read_text()
                    .replace("__SESSION_TITLE__", spec["title"])
                    .replace("__SESSION_BLOCK__", (HERE / f"session_{key}.md").read_text().strip()))
        if new["markdown"] != overview:
            c.set_challenge_markdown(cid, overview)
            print("[4] French overview set")
        if c.challenge_markdown(cid)["content"] != overview:
            raise SystemExit("[4] overview did not round-trip")
        with tempfile.TemporaryDirectory() as tmp:
            src_img = c.challenge_image(SOURCE_ID, os.path.join(tmp, "src"))
            new_img = c.challenge_image(cid, os.path.join(tmp, "new"))
            if not filecmp.cmp(src_img, new_img, shallow=False):
                raise SystemExit("[4] cover image differs from the source's")
        print("[4] overview in place; cover image identical to the source's")

        # 5. datasets
        if started:
            served = dataset_files(c, cid)
            if {k: sha(v) for k, v in served.items()} != {k: sha(v) for k, v in src_files.items()}:
                raise SystemExit("[5] started challenge serves other files than the source")
            print("[5] datasets byte-identical to the source's")
        else:
            ensure_datasets(c, cid, src_ds, src_files)

        # 6. benchmark: 172's submission.csv (the copy carries only a header stub)
        bench = {k: v["content"] for k, v in c.list_benchmark_files(cid)["files"].items()}
        run = c.benchmark_status(cid)["run"]
        if bench != src_bench:
            if started:
                raise SystemExit("[6] started challenge has another benchmark than the source")
            for fname in set(bench) - set(src_bench):
                c.delete_benchmark_file(cid, fname)
            for fname, content in src_bench.items():
                c.update_benchmark_file_content(cid, fname, content)
            bench = {k: v["content"] for k, v in c.list_benchmark_files(cid)["files"].items()}
            if bench != src_bench:
                raise SystemExit("[6] benchmark files did not round-trip")
            print(f"[6] benchmark {sorted(src_bench)} copied from challenge {SOURCE_ID}")
            run = None
        if not started and (not run or run["job_status"] != "completed"):
            c.run_benchmark(cid)
            print("[6] benchmark started")
            run = wait_for_benchmark(c, cid)
        score = check_benchmark(run, alive_share)
        entry["benchmark_score"] = score
        write_state(state)
        print(f"[6] benchmark completed, score {score} (= share of alive in y_test)")
        ready.append((key, cid, new["creator_challenge"]))

    # 7. course data must match the scorer before anything goes live
    problems = course_data_gate(y_test, src_files["y_train.csv"])
    if problems:
        raise SystemExit("[7] the course-hosted data does not match challenge "
                         f"{SOURCE_ID}'s scorer; nothing started:\n  - "
                         + "\n  - ".join(problems))
    print("[7] hosted X_test ids == y_test ids; hosted y_train_survie == served y_train")

    # 8. start, public, student read-back
    student = MLArenaClient(env["MLARENA_STUDENT_API_KEY"], BASE_URL)
    for key, cid, cc in ready:
        if cc["status"] != "started":
            c.start_challenge(cid)
            print(f"[8] {key}: challenge {cid} started")
        if not cc["is_public"]:
            c.update_challenge(cid, is_public=True)
            print(f"[8] {key}: challenge {cid} public")
        final = c.creator_challenge(cid)
        if final["status"] != "started" or final["is_public"] is not True:
            raise SystemExit(f"[8] {key}: unexpected final state: status={final['status']} "
                             f"is_public={final['is_public']}")
        seen = student.challenge(cid)
        if seen["name"] != SESSIONS[key]["name"] or seen["status"] != "started":
            raise SystemExit(f"[8] student sees name={seen['name']!r} status={seen['status']!r}")
        seen_files = sorted(f["label"] for d in student.datasets(cid)["datasets"]
                            for f in d["files"])
        if seen_files != sorted(src_files):
            raise SystemExit(f"[8] student sees datasets {seen_files}")
        print(f"[8] {key}: visible to the student key: {seen['name']} ({seen['status']}), "
              f"datasets {seen_files}")
        state[key].update({"url": f"{BASE_URL}/viewchallenge/{cid}", "status": final["status"],
                           "is_public": final["is_public"]})
    state["copied_from"] = SOURCE_ID
    write_state(state)


if __name__ == "__main__":
    args = sys.argv[1:] or list(SESSIONS)
    unknown = set(args) - set(SESSIONS)
    if unknown:
        raise SystemExit(f"unknown session(s) {sorted(unknown)}; choose from {list(SESSIONS)}")
    sys.exit(main(args))
