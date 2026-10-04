"""Replace the data of challenge 194 with this package: stop, delete the old submissions,
upload the env files, settings, template, dataset and benchmark, re-run the benchmark (0.0
expected), start, then the overview.

    set -a; . ../../.env; set +a          # MLARENA_CREATOR_API_KEY
    python3 deploy.py --dry-run
    python3 deploy.py

Deleting the submissions is deliberate: scores computed on other data mean nothing.
"""
import os
import re
import json
import sys
import time

sys.path.insert(0, os.environ.get("MLARENA_SDK_PATH", "/Users/raphaelcousin/reinforcement_learning_challenge/mlarena-sdk"))
from mlarena.client import MLArenaClient  # noqa: E402

PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)
from config import CONFIG  # noqa: E402

CID = 194
DRY = "--dry-run" in sys.argv
c = MLArenaClient(token=os.environ["MLARENA_CREATOR_API_KEY"], base_url="https://ml-arena.com")
ENV_FILES = {"env.py": f"{PKG}/env.py", "scoring.py": f"{PKG}/data/scoring.py",
             "dev.json": f"{PKG}/data/dev.json", "private.json": f"{PKG}/data/private.json",
             "hf_models.json": f"{PKG}/hf_models.json"}


if open(f"{PKG}/public/scoring.py").read() != open(f"{PKG}/data/scoring.py").read():
    sys.exit("data/scoring.py differs from public/scoring.py: cp public/scoring.py data/scoring.py")
if re.search(r"\{[A-Z0-9_]+\}", open(f"{PKG}/overview.md").read()):
    sys.exit("overview.md still has {PLACEHOLDERS}: run anchors2/fill_numbers.py")


def step(msg, fn=None):
    print(("[dry] " if DRY else "") + msg, flush=True)
    if not DRY and fn is not None:
        r = fn()
        print("   ->", json.dumps(r, default=str)[:300], flush=True)
        return r


cc = c.creator_challenge(CID)
print("status", cc["status"], "machine", cc["machine_name"])
if cc["status"] == "started":
    step("stop", lambda: c.stop_challenge(CID))

subs = c.creator_submissions(CID)["submissions"]
for s in subs:
    if s["submission_name"] == "__benchmark__":
        continue
    step(f"delete submission {s['submission_id']} {s['username']} {s['submission_name']} {s.get('score')}",
         lambda s=s: c.soft_delete_submission(CID, s["submission_id"]))

have = set(c.list_env_files(CID)["files"])
for name in sorted(have - set(ENV_FILES)):
    if name != "env.py":
        step(f"delete env file {name}", lambda name=name: c.delete_env_file(CID, name))
for name, path in ENV_FILES.items():
    step(f"upload env file {name} <- {path}", lambda path=path: {k: v for k, v in c.upload_env_file(CID, path).items()
                                                                if k != "content"})

metrics = []
for m in CONFIG["metrics_schema"]:
    d = {"key": "reward" if m["key"] == "score" else m["key"], "label": m["label"],
         "source": "score" if m["key"] == "score" else "env", "agg": m["agg"],
         "order": "desc" if m["higher_is_better"] else "asc", "format": m["format"], "unit": None,
         "precision": m["precision"], "is_ranking": m["key"] == "score", "visible": True}
    metrics.append(d)
step(f"settings timeout={CONFIG['simulation_timeout_sec']} step={CONFIG['agent_max_time_per_step_second']} "
     f"metrics={[m['key'] for m in metrics]}",
     lambda: c.update_settings(CID, simulation_timeout_sec=CONFIG["simulation_timeout_sec"],
                               agent_max_time_per_step_second=CONFIG["agent_max_time_per_step_second"],
                               deployment_nb_constraint_run=1, deployment_nb_initial_score_run=1,
                               metrics=metrics)["configuration"]["simulation_timeout_sec"])
step("agent template", lambda: bool(c.update_agent_template(CID, open(f"{PKG}/agent_template.py").read())))

ds = next(d for d in c.creator_datasets(CID)["datasets"] if d["label"] == CONFIG["dataset_label"])
step(f"dataset {ds['id']} description", lambda: bool(c.update_dataset(CID, ds["id"], description=CONFIG["dataset_description"])))
for f in ds["files"]:
    step(f"dataset file delete {f['label']}", lambda f=f: c.delete_dataset_file(CID, ds["id"], f["id"]))
step("dataset file upload dev.json", lambda: {k: v for k, v in c.upload_dataset_file(CID, ds["id"], f"{PKG}/data/dev.json").items()
                                              if k != "download_url"})

step("benchmark file", lambda: bool(c.update_benchmark_file_content(CID, "agent.py", open(f"{PKG}/benchmark_agent.py").read())))
if DRY:
    sys.exit(0)
step("run benchmark", lambda: c.run_benchmark(CID))
t0 = time.time()
last = None
while True:
    try:
        st = c.benchmark_status(CID)["run"]
    except Exception as e:  # a network blip while the job runs: poll again
        print("   poll failed:", e, flush=True)
        time.sleep(10)
        continue
    state = st.get("job_status")
    if state != last:
        res = st.get("submission_results") or [{}]
        print(f"   benchmark: {state} score={res[0].get('score')} "
              f"agent_err={res[0].get('agent_error_type')} env_err={st.get('env_error_type')} "
              f"{(st.get('env_error_message') or '')[:300]}", flush=True)
        last = state
    if state in ("completed", "failed"):
        break
    if time.time() - t0 > 1800:
        sys.exit("benchmark timeout")
    time.sleep(10)
res = (st.get("submission_results") or [{}])[0]
if state != "completed" or st.get("env_error_type") or res.get("score") != 0.0:
    sys.exit("benchmark did not pass: not starting")
step("start", lambda: c.start_challenge(CID))
step("overview", lambda: bool(c.set_challenge_markdown(CID, open(f"{PKG}/overview.md").read())))
