#!/usr/bin/env python3
"""Sync the course's competition packages to ML-Arena.

Sibling of `publish_mlarena.py`: that one owns modules and lessons, this one
owns the competitions those modules link to. Both are pure client-side
compositions of the public SDK surface — no new endpoints — per the
frontend<->SDK parity rule in `mlarena-sdk/PROCESS.md`.

    competitions/<pkg>/config.py     the manifest (name, kind, module, metrics)
                      env.py         the scorer
                      overview.md    the competition page
                      agent.py       flex_v1: the reference solution, run as
                                     the creator-side benchmark
                      data/          file_v1: public + private files, built by
                                     the package's prepare_data.py

Modes
-----
    build     create/update, upload, benchmark, verify, start     (creator key)
    overview  re-publish overview.md alone, started or not         (creator key)
    refresh   replace the data of a STARTED competition, in place  (creator + user key)
    status    show what is live                                   (creator key)
    publish   flip the competitions public                        (creator key)
    attach    link each competition to its course module          (TEACHER key)
    teardown  stop + hide                                         (creator key)

`build` is idempotent: server ids are recorded in
`competitions/.mlarena-state.json`, keyed by base URL. Commit that file — it is
a lockfile, not an artifact. A competition that is already started is left
alone, because the platform locks settings after start.

`overview` is the small escape hatch: the competition page is the one thing the
platform does *not* lock at start, so a wording fix does not need a stop, a
re-benchmark, or the loss of every score on the board. It uploads nothing else.

`refresh` is the big one: it stops the competition, replaces the
dataset files and the private ground truth, re-benchmarks, and starts it again.
Use it when `prepare_data.py` has produced a *different* split or different ids
— every score already on the board was computed against data that no longer
exists, so refresh deletes those agents rather than leaving stale numbers
ranked. It refuses to run if anyone but you is on the board.

The benchmark is the real test: it runs the actual worker pipeline (JobPod,
env.py, and for flex_v1 an agent container) against the package's reference
solution, and `build` fails unless the resulting score equals the package's
declared `benchmark_expected_score`. A green run means the competition scores
what it says it scores.

Env vars:
    MLARENA_API_KEY          creator-scope token (mlk_creator_...)
    MLARENA_TEACHER_API_KEY  teacher-scope token, `attach` only
    MLARENA_USER_API_KEY     user-scope token, `refresh` only — deleting a
                             stale agent is a /direct_attache_agents route
    MLARENA_BASE_URL         defaults to https://ml-arena.com
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSEWARE = HERE.parent
PACKAGES_DIR = COURSEWARE / "competitions"
STATE_FILE = PACKAGES_DIR / ".mlarena-state.json"
DEFAULT_SDK = COURSEWARE.parent.parent.parent / "mlarena-sdk"

# Build order = course order. Session 1 has no package of its own: its lab
# submits to the existing PettingZoo Connect-Four challenge (65), which this
# repository does not own. `s1-textstats` (179) and `s2-readability` (180) were
# retired on 2026-09-06 with the lessons that used them.
PACKAGES = ["s2-bike-demand", "s2-bank-marketing",
            "s3-adult-income", "s3-diabetes-progression", "s3-credit-risk",
            "s4-mnist-warmup", "s4-california-housing", "s4-forest-cover"]


# --------------------------------------------------------------------------- #
# config / state
# --------------------------------------------------------------------------- #
def load_config(pkg: str) -> dict:
    pkg_dir = PACKAGES_DIR / pkg
    ns: dict = {}
    exec((pkg_dir / "config.py").read_text(), ns)
    cfg = dict(ns["CONFIG"])
    cfg["_dir"] = pkg_dir
    cfg["_pkg"] = pkg
    return cfg


def read_state(base_url: str) -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text()).get(base_url, {})
    return {}


def write_state(base_url: str, state: dict) -> None:
    whole = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    whole[base_url] = state
    STATE_FILE.write_text(json.dumps(whole, indent=2, sort_keys=True) + "\n")


def connect(scope_env: str, base_url: str):
    sdk_path = os.environ.get("MLARENA_SDK_PATH", str(DEFAULT_SDK))
    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    import mlarena
    token = os.environ.get(scope_env)
    if not token:
        sys.exit(f"{scope_env} is not set.")
    return mlarena.connect(api_key=token, base_url=base_url)


# --------------------------------------------------------------------------- #
# benchmark = the end-to-end test
# --------------------------------------------------------------------------- #
def _benchmark_score(status: dict):
    results = status.get("agent_results") or []
    return results[0].get("score") if results else None


def wait_for_benchmark(client, cid: int, timeout_s: int = 1800) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        st = client.benchmark_status(cid)
        state = st.get("status")
        if state != last:
            print(f"      benchmark: {state} score={_benchmark_score(st)}")
            last = state
        if state == "completed":
            if st.get("success") is False:
                raise RuntimeError(f"benchmark ran but reported failure: {st}")
            return st
        if state == "failed":
            raise RuntimeError(f"benchmark failed: {st}")
        time.sleep(8)
    raise TimeoutError(f"benchmark did not finish in {timeout_s}s (last={last})")


def verify_benchmark(cfg: dict, status: dict) -> None:
    """Fail fast unless the reference solution scored exactly what it should."""
    expected = cfg.get("benchmark_expected_score")
    if expected is None:
        return
    got = _benchmark_score(status)
    tol = cfg.get("benchmark_score_tol", 1e-9)
    if got is None or abs(float(got) - float(expected)) > tol:
        raise RuntimeError(
            f"{cfg['name']}: benchmark scored {got}, expected {expected} "
            f"(tol {tol}). The env does not grade what the package claims."
        )
    results = status.get("agent_results") or [{}]
    if results[0].get("is_agent_code_error"):
        raise RuntimeError(
            f"{cfg['name']}: benchmark flagged is_agent_code_error: "
            f"{results[0].get('agent_code_error_message')}"
        )
    print(f"      benchmark verified: score={got} == expected {expected}")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def find_existing(client, name: str):
    for c in client.creator_competitions():
        if c.get("name") == name:
            return c
    return None


def build_one(client, cfg: dict, state: dict, base_url: str) -> int:
    name, pkg_dir = cfg["name"], cfg["_dir"]
    print(f"\n=== {name}  [{cfg['kernel_version']}]  ({cfg['_pkg']})")

    existing = find_existing(client, name)
    if existing and existing.get("is_started"):
        print(f"    already live: id={existing['id']} — settings are locked, skipping.")
        return existing["id"]
    if existing:
        cid = existing["id"]
        print(f"    reusing existing unstarted competition id={cid}")
    else:
        comp = client.create_competition(
            name=name,
            kernel_version=cfg["kernel_version"],
            description=cfg.get("label", name),
            is_public=cfg.get("is_public_initial", False),
        )
        cid = comp["competition_id"]
        print(f"    created id={cid} (is_public={cfg.get('is_public_initial', False)})")

    client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())
    print("    overview.md published")

    settings = {
        "evaluation_metric": cfg["metric"],
        "evaluation_deployment_nb_constraint_run": cfg["deployment_nb_constraint_run"],
        "evaluation_deployment_nb_initial_score_run": cfg["deployment_nb_initial_score_run"],
        "evaluation_metrics_schema": cfg["metrics_schema"],
    }
    if cfg.get("metric2"):
        settings["evaluation_metric2"] = cfg["metric2"]
    client.update_settings(cid, **settings)
    print(f"    settings: metric={cfg['metric']} "
          f"runs={cfg['deployment_nb_constraint_run']}+"
          f"{cfg['deployment_nb_initial_score_run']} "
          f"metrics_schema={[d['key'] for d in cfg['metrics_schema']]}")

    validation = client.update_env_file_content(cid, "env.py",
                                                (pkg_dir / "env.py").read_text())
    print(f"    env.py uploaded; structural check="
          f"{validation.get('validation', validation.get('status', 'ok'))}")

    for rel in cfg.get("private_files", []):
        src = pkg_dir / "data" / rel
        if not src.exists():
            src = pkg_dir / rel
        if not src.exists():
            raise SystemExit(f"missing private file {rel} — run {cfg['_pkg']}/prepare_data.py")
        client.upload_env_file(cid, str(src))
        print(f"    private env file: {rel}")

    if cfg.get("agent_template_file"):
        client.update_agent_template(
            cid, (pkg_dir / cfg["agent_template_file"]).read_text())
        print(f"    agent template: {cfg['agent_template_file']}")

    public = cfg.get("public_files") or []
    if public:
        label = cfg.get("dataset_label", name)
        existing_ds = next(
            (d for d in client.creator_datasets(cid).get("datasets", [])
             if d.get("label") == label), None)
        if existing_ds:
            ds_id = existing_ds["id"]
            have = {f["label"] for f in existing_ds.get("files", [])}
            print(f"    reusing dataset id={ds_id} (has {sorted(have)})")
        else:
            ds = client.create_dataset(cid, label=label,
                                       description=cfg.get("dataset_description"))
            ds_id, have = ds["id"], set()
            print(f"    dataset created id={ds_id}")
        for fname in public:
            if fname in have:
                continue
            path = pkg_dir / "data" / fname
            if not path.exists():
                raise SystemExit(f"missing public file {fname} — run {cfg['_pkg']}/prepare_data.py")
            client.upload_dataset_file(cid, ds_id, str(path))
            print(f"    dataset file: {fname} ({path.stat().st_size/1e6:.2f} MB)")

    bench = pkg_dir / cfg["benchmark_file"]
    if cfg["kernel_version"] == "file_v1":
        client.update_benchmark_file_content(cid, "submission.csv", bench.read_text())
    else:
        client.update_benchmark_file_content(cid, "agent.py", bench.read_text())
    print(f"    benchmark file: {cfg['benchmark_file']}")

    client.run_benchmark(cid)
    status = wait_for_benchmark(client, cid)
    verify_benchmark(cfg, status)

    client.start_competition(cid)
    print(f"    STARTED id={cid}")

    state.setdefault("competitions", {})[cfg["_pkg"]] = {
        "id": cid,
        "name": name,
        "module_slug": cfg["module_slug"],
        "label": cfg["label"],
    }
    write_state(base_url, state)
    return cid


def do_build(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    state = read_state(args.base_url)
    for pkg in args.packages:
        cfg = load_config(pkg)
        build_one(client, cfg, state, args.base_url)
    do_status(args)


def refresh_one(client, user_client, cfg: dict, base_url: str) -> int:
    """Replace the data of an already-started competition, in place.

    The platform locks settings, datasets and the agent template once a
    competition starts, so this has to stop it first. Stopping leaves agents and
    results untouched (`stop_competition` only clears `is_started`), which is
    exactly the problem when the ids have changed underneath them: a score
    computed against the old X_test is meaningless against the new one but still
    ranks. So the stale agents are deleted, not left on the board.
    """
    name, pkg_dir = cfg["name"], cfg["_dir"]
    print(f"\n=== refresh {name}  ({cfg['_pkg']})")

    existing = find_existing(client, name)
    if not existing:
        raise SystemExit(f"{name}: not on the server — use `build`, not `refresh`.")
    cid = existing["id"]

    me = client.profile().get("username")
    board = client.leaderboard(cid)
    rows = board.to_dict("records") if hasattr(board, "to_dict") else list(board)
    others = sorted({r["Username"] for r in rows if r.get("Username") != me})
    if others:
        raise SystemExit(
            f"{name}: {len(others)} other competitor(s) on the board ({others}). "
            f"Refreshing invalidates their scores — stop the competition and "
            f"decide deliberately rather than through this script."
        )
    stale = [r for r in rows if r.get("AgentName") != "__benchmark__"]

    client.stop_competition(cid)
    print(f"    stopped id={cid}")

    for r in stale:
        user_client.delete_agent(cid, r["agentAttachId"])
        print(f"    deleted stale agent {r['agentAttachId']} "
              f"({r.get('AgentName')}, {cfg['metric']}={r.get('MeanReward')})")

    client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())
    client.update_env_file_content(cid, "env.py", (pkg_dir / "env.py").read_text())
    print("    overview.md + env.py re-uploaded")

    # Settings are locked while started, so they can only be re-applied here.
    # A refresh that changes the metric and not the schema would fail the
    # worker's equal-mapping check on the first run, not on the upload.
    settings = {
        "evaluation_metric": cfg["metric"],
        "evaluation_deployment_nb_constraint_run": cfg["deployment_nb_constraint_run"],
        "evaluation_deployment_nb_initial_score_run": cfg["deployment_nb_initial_score_run"],
        "evaluation_metrics_schema": cfg["metrics_schema"],
    }
    if cfg.get("metric2"):
        settings["evaluation_metric2"] = cfg["metric2"]
    client.update_settings(cid, **settings)
    print(f"    settings: metric={cfg['metric']} "
          f"metrics_schema={[d['key'] for d in cfg['metrics_schema']]}")

    for rel in cfg.get("private_files", []):
        src = pkg_dir / "data" / rel
        if not src.exists():
            raise SystemExit(f"missing private file {rel} — run {cfg['_pkg']}/prepare_data.py")
        client.upload_env_file(cid, str(src))
        print(f"    private env file replaced: {rel}")

    public = cfg.get("public_files") or []
    if public:
        label = cfg.get("dataset_label", name)
        ds = next((d for d in client.creator_datasets(cid).get("datasets", [])
                   if d.get("label") == label), None)
        if ds is None:
            raise SystemExit(f"{name}: no dataset labelled {label!r} to refresh")
        ds_id = ds["id"]
        if cfg.get("dataset_description"):
            client.update_dataset(cid, ds_id,
                                  description=cfg["dataset_description"])
        # upload_dataset_file always adds a row, so replacing means delete first.
        for f in ds.get("files", []):
            if f["label"] in public:
                client.delete_dataset_file(cid, ds_id, f["id"])
                print(f"    dataset file deleted: {f['label']}")
        for fname in public:
            path = pkg_dir / "data" / fname
            if not path.exists():
                raise SystemExit(f"missing public file {fname} — run {cfg['_pkg']}/prepare_data.py")
            client.upload_dataset_file(cid, ds_id, str(path))
            print(f"    dataset file uploaded: {fname} "
                  f"({path.stat().st_size / 1e6:.2f} MB)")

    bench = pkg_dir / cfg["benchmark_file"]
    fname = "submission.csv" if cfg["kernel_version"] == "file_v1" else "agent.py"
    client.update_benchmark_file_content(cid, fname, bench.read_text())
    print(f"    benchmark file: {cfg['benchmark_file']}")

    client.run_benchmark(cid)
    verify_benchmark(cfg, wait_for_benchmark(client, cid))

    client.start_competition(cid)
    print(f"    RESTARTED id={cid}")
    return cid


def do_refresh(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    user_client = connect("MLARENA_USER_API_KEY", args.base_url)
    for pkg in args.packages:
        refresh_one(client, user_client, load_config(pkg), args.base_url)
    do_status(args)


def do_status(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    state = read_state(args.base_url)
    # The public `competition(id)` route 404s while a competition is hidden, so
    # status reads the creator listing instead.
    mine = {c["id"]: c for c in client.creator_competitions()}
    print("\n===== competitions =====")
    for pkg in args.packages:
        entry = (state.get("competitions") or {}).get(pkg)
        if not entry:
            print(f"  {pkg:18s} not built")
            continue
        detail = mine.get(entry["id"], {})
        print(f"  {pkg:18s} id={entry['id']:<5} "
              f"started={detail.get('is_started')} public={detail.get('is_public')} "
              f"-> module {entry['module_slug']}")
        print(f"  {'':18s} {args.base_url}/viewcompetition/{entry['id']}")


def do_overview(args):
    """Re-publish overview.md and nothing else.

    `build` skips a started competition outright, because settings, datasets and
    the agent template are locked once it starts. The markdown is not — so a
    correction to the competition page is one call, and does not have to go
    through `refresh` and throw away the leaderboard to fix a sentence.
    """
    client = connect("MLARENA_API_KEY", args.base_url)
    state = read_state(args.base_url)
    for pkg in args.packages:
        entry = (state.get("competitions") or {}).get(pkg)
        if not entry:
            print(f"  {pkg}: not built, skipping")
            continue
        body = (PACKAGES_DIR / pkg / "overview.md").read_text()
        client.set_challenge_markdown(entry["id"], body)
        print(f"  {pkg}: overview.md -> competition {entry['id']} "
              f"({len(body)} chars)")


def do_publish(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    state = read_state(args.base_url)
    for pkg, entry in (state.get("competitions") or {}).items():
        client.update_competition(entry["id"], is_public=True)
        print(f"published {pkg} id={entry['id']}")


def do_teardown(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    state = read_state(args.base_url)
    for pkg, entry in (state.get("competitions") or {}).items():
        try:
            client.update_competition(entry["id"], is_public=False)
            client.stop_competition(entry["id"])
            print(f"stopped + hid {pkg} id={entry['id']}")
        except Exception as exc:
            print(f"  could not stop {pkg}: {exc}")


def do_attach(args):
    """Link each competition to its course module. Needs a TEACHER-scope key.

    Module ids come from the course's publish lockfile
    (`content/<course>/.mlarena-state.json`), written by publish_mlarena.py.

    The link carries the course's **pass threshold** — the value a student's
    best leaderboard score has to reach for the module to count the challenge
    validated — and it is `benchmark_expected_score`: the bar is "match the
    worked baseline", uniformly, so it is derived here rather than typed into
    the course editor. It was typed in, once, and then s2-bike-demand changed
    metric from R2 to -MAE on 2026-09-08 and its bar stayed at the old R2
    (0.399304). Under -MAE nothing reaches 0.4, so the challenge read
    "NEEDS >= 0.40" and could not be validated by anyone. Hence the
    reconciliation below: an existing attachment is not skipped, it is checked.
    """
    teacher = connect("MLARENA_TEACHER_API_KEY", args.base_url)
    state = read_state(args.base_url)
    course_state_file = COURSEWARE / "content" / args.course / ".mlarena-state.json"
    if not course_state_file.exists():
        sys.exit(f"No course lockfile at {course_state_file} — publish the course first.")
    modules = json.loads(course_state_file.read_text())[args.base_url]["modules"]

    for pkg in args.packages:
        entry = (state.get("competitions") or {}).get(pkg)
        if not entry:
            print(f"  {pkg}: not built, skipping")
            continue
        module_id = modules.get(entry["module_slug"])
        if module_id is None:
            sys.exit(f"{pkg}: module {entry['module_slug']!r} is not in the lockfile")
        threshold = load_config(pkg)["benchmark_expected_score"]
        detail = teacher.get_module(module_id)
        already = {c["competition_id"]: c for c in (detail.get("competitions") or [])}
        link = already.get(entry["id"])
        if link is None:
            teacher.attach_competition(module_id, entry["id"], label=entry["label"],
                                       pass_threshold=threshold)
            print(f"  {pkg}: competition {entry['id']} -> module {module_id} "
                  f"({entry['module_slug']}), pass >= {threshold}")
            continue
        live = link.get("pass_threshold")
        if live is not None and abs(live - threshold) <= 1e-6:
            print(f"  {pkg}: already attached to module {module_id}, "
                  f"pass >= {live}")
            continue
        teacher.update_challenge_link(module_id, entry["id"],
                                      pass_threshold=threshold)
        print(f"  {pkg}: pass threshold {live} -> {threshold} "
              f"(module {module_id})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="build",
                    choices=["build", "overview", "refresh", "status",
                             "publish", "attach", "teardown"])
    ap.add_argument("--base-url", default=os.environ.get("MLARENA_BASE_URL",
                                                         "https://ml-arena.com"))
    ap.add_argument("--course", default="python-ai-engineering")
    ap.add_argument("--only", action="append", dest="only",
                    help="restrict to one package (repeatable)")
    args = ap.parse_args()
    args.packages = args.only or PACKAGES
    unknown = [p for p in args.packages if p not in PACKAGES]
    if unknown:
        sys.exit(f"unknown package(s): {unknown}; known: {PACKAGES}")
    {"build": do_build, "overview": do_overview, "refresh": do_refresh,
     "status": do_status, "publish": do_publish, "attach": do_attach,
     "teardown": do_teardown}[args.mode](args)


if __name__ == "__main__":
    main()
