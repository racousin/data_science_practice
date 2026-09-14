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
              --keep-agents: same split, renamed files — keep the board (creator key)
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

`refresh --keep-agents` is the same run with both of those safeguards off,
for the one case that does not need them: the split, the ids and the bytes are
identical and only the *file names* changed, so every score on the board is
still the score. Nothing here verifies that — it is your claim. On a real data
change it would leave the board silently wrong, which is what plain `refresh`
exists to prevent.

The benchmark is the real test: it runs the actual worker pipeline (JobPod,
env.py, and for flex_v1 an agent container) against the package's reference
solution, and `build` fails unless the resulting score equals the package's
declared `benchmark_expected_score`. A green run means the competition scores
what it says it scores.

Packages belong to a course (`PACKAGES_BY_COURSE`), and every mode works on
the packages of `--course` only — so `publish` on one course never flips
another course's challenge public, and `attach` only looks up module slugs in
the course lockfile they belong to.

Optional config.py keys, applied before the benchmark (settings lock at start):
    submission_filename      file_v1: the file participants upload, and the
                             name the benchmark is stored under. Absent means
                             the platform default, "submission.csv". A name
                             not ending in ".csv" (e.g. "submission.csv.gz")
                             skips the platform's CSV column check, and its
                             benchmark is uploaded as bytes, not as text.
    max_upload_size_bytes    the participant upload cap (platform default 100 MB).
    engine_id                pin the challenge to this engine (e.g. one with the
                             memory the scorer needs). Absent or None keeps the
                             kind's default engine. Pinning goes through the
                             admin configuration route, so the creator key must
                             belong to an admin account.
Optional config.py keys read by `attach` only:
    pass_threshold           the score a student must reach for the module to
                             count the challenge validated, for a course that
                             states its own bar. Absent or None means the bar
                             is benchmark_expected_score. A bar above the
                             benchmark is accepted only when the package also
                             declares expert_expected_score and the bar is at
                             most that value; otherwise it is refused as a
                             typo, since nothing proves it reachable.
    expert_expected_score    the score the package's own reference solution
                             (its teacher pipeline) reaches on the shipped
                             split — the ceiling a pass_threshold above the
                             benchmark is checked against. Absent or None
                             means no such claim, and the bar may not exceed
                             the benchmark.
Public and private files are looked up in `data/` and then in the package
directory, so a hand-written file (EXPERTISE.md) need not be copied into data/.

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
import re
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
# retired on 2026-09-06 with the lessons that used them. Session 4's three —
# `s4-mnist-warmup` (182), `s4-california-housing` (187) and `s4-forest-cover`
# (188) — were retired on 2026-09-10 and replaced by `s4-taxi-eta`, Lab 1's
# challenge. Retiring a package means removing it HERE too: `attach` re-links
# every listed package that has a lockfile entry, detached or not.
#
# `ms2a-machine-learning-practice` (course 15) builds its Session 1
# multi-source challenge and its Session 2 preprocessing challenge from this
# directory too. Its module slugs live in that course's lockfile, not
# python-ai-engineering's, which is why the list is keyed by course: run it
# with `--course ms2a-machine-learning-practice`
# (`make competitions COURSE=ms2a-machine-learning-practice`).
PACKAGES_BY_COURSE = {
    "python-ai-engineering": ["s2-bike-demand", "s2-bank-marketing",
                              "s3-adult-income", "s3-diabetes-progression",
                              "s3-credit-risk", "s4-taxi-eta"],
    "ms2a-machine-learning-practice": ["mlp-s1-store-sales",
                                       "s2-dpe-energy-label",
                                       "s2-icu-survival"],
}
PACKAGES = [p for pkgs in PACKAGES_BY_COURSE.values() for p in pkgs]

# The platform's own default (`CompetitionConfiguration.submission_filename`,
# modelmanager/modelmanager/competitions.py). A package without the key is
# built exactly as before this key existed.
DEFAULT_SUBMISSION_FILENAME = "submission.csv"
MAX_SUBMISSION_FILENAME_LEN = 128   # backend settings schema bound
SUBMISSION_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# --------------------------------------------------------------------------- #
# config / state
# --------------------------------------------------------------------------- #
def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == value)                                 # not NaN


def validate_config(cfg: dict) -> None:
    """Fail on a malformed optional key here, before anything is uploaded."""
    pkg = cfg["_pkg"]
    if "submission_filename" in cfg:
        name = cfg["submission_filename"]
        if cfg["kernel_version"] != "file_v1":
            raise SystemExit(f"{pkg}: submission_filename is a file_v1 setting, "
                             f"not {cfg['kernel_version']}")
        # The backend's rule (validate_submission_filename in
        # creator_competition/_schemas.py): a name secure_filename() stores
        # unchanged. Checked here so a bad name stops the build before the
        # challenge is created, not at update_settings one step later.
        if (not isinstance(name, str)
                or not SUBMISSION_FILENAME_RE.fullmatch(name)
                or ".." in name or name.endswith((".", "_"))
                or len(name) > MAX_SUBMISSION_FILENAME_LEN):
            raise SystemExit(f"{pkg}: submission_filename must be a bare file "
                             f"name of at most {MAX_SUBMISSION_FILENAME_LEN} "
                             f"characters (letters, digits, '.', '_', '-'; "
                             f"starting with a letter or digit, no '..', not "
                             f"ending in '.' or '_'), got {name!r}")
    if "max_upload_size_bytes" in cfg:
        size = cfg["max_upload_size_bytes"]
        if not _is_int(size) or size < 1:
            raise SystemExit(f"{pkg}: max_upload_size_bytes must be a positive "
                             f"int, got {size!r}")
    # The platform's dataset columns are varchar(200) / varchar(500); a longer
    # value fails at create_dataset, after the challenge already exists.
    for key, limit in (("dataset_label", 200), ("dataset_description", 500)):
        if len(cfg.get(key) or "") > limit:
            raise SystemExit(f"{pkg}: {key} is {len(cfg[key])} characters, "
                             f"the platform stores at most {limit}")
    engine_id = cfg.get("engine_id")
    if engine_id is not None and (not _is_int(engine_id) or engine_id < 1):
        raise SystemExit(f"{pkg}: engine_id must be a positive int or None, "
                         f"got {engine_id!r}")
    # Both keys may be present as None ("to be pinned"), which reads as absent.
    expert = cfg.get("expert_expected_score")
    if expert is not None and not _is_number(expert):
        raise SystemExit(f"{pkg}: expert_expected_score must be a number or "
                         f"None, got {expert!r}")
    bar = cfg.get("pass_threshold")
    if bar is not None:
        if not _is_number(bar):
            raise SystemExit(f"{pkg}: pass_threshold must be a number, "
                             f"got {bar!r}")
        # A bar nothing in the package reaches is a typo or a stale number,
        # not a course decision. The benchmark is the floor every package
        # proves; a bar above it is a course asking for the domain steps
        # (s2-dpe-energy-label: 0.85 over a 0.7606 benchmark), and is only
        # accepted up to the score the package's own reference solution
        # reaches, declared as expert_expected_score.
        bench = cfg.get("benchmark_expected_score")
        if expert is not None:
            if bar > expert:
                raise SystemExit(f"{pkg}: pass_threshold {bar} is above "
                                 f"expert_expected_score {expert} — the "
                                 f"package's own reference solution would "
                                 f"not validate")
        elif bench is not None and bar > bench:
            raise SystemExit(f"{pkg}: pass_threshold {bar} is above the "
                             f"benchmark's own score {bench}; declare "
                             f"expert_expected_score to allow a bar above "
                             f"the benchmark")


def load_config(pkg: str) -> dict:
    pkg_dir = PACKAGES_DIR / pkg
    ns: dict = {}
    exec((pkg_dir / "config.py").read_text(), ns)
    cfg = dict(ns["CONFIG"])
    cfg["_dir"] = pkg_dir
    cfg["_pkg"] = pkg
    validate_config(cfg)
    return cfg


def pass_threshold(cfg: dict) -> float:
    """The bar `attach` writes onto the module link: the package's own
    `pass_threshold` when it declares one (a None reads as absent), else its
    benchmark score."""
    if cfg.get("pass_threshold") is not None:
        return cfg["pass_threshold"]
    if cfg["benchmark_expected_score"] is None:
        raise SystemExit(f"{cfg['_pkg']}: benchmark_expected_score is not pinned, "
                         f"so there is no bar to attach — run prepare_data.py "
                         f"and set it in config.py")
    return cfg["benchmark_expected_score"]


def submission_filename(cfg: dict) -> str:
    """The name participants upload, and the benchmark's name on the server."""
    return cfg.get("submission_filename", DEFAULT_SUBMISSION_FILENAME)


def package_file(cfg: dict, rel: str, role: str) -> Path:
    """Locate a public or private file: `data/` (generated) or the package
    directory (hand-written, e.g. EXPERTISE.md).

    Present in both with different bytes is refused rather than resolved by a
    precedence rule: one of the two is stale, and the build cannot tell which.
    """
    candidates = [p for p in (cfg["_dir"] / "data" / rel, cfg["_dir"] / rel)
                  if p.is_file()]
    if not candidates:
        raise SystemExit(f"missing {role} file {rel} — run {cfg['_pkg']}/prepare_data.py")
    if len(candidates) == 2 and candidates[0].read_bytes() != candidates[1].read_bytes():
        raise SystemExit(f"{cfg['_pkg']}: {role} file {rel} exists in data/ and in "
                         f"the package directory with different contents")
    return candidates[0]


def preflight(cfg: dict) -> None:
    """Every local file the run will upload, checked before the first request.

    `refresh` stops a live challenge before it uploads anything, so a file
    found missing halfway through would leave the challenge stopped.
    """
    pkg_dir = cfg["_dir"]
    # `verify_benchmark` skips the check when no score is declared, and the
    # build then starts the challenge: an unpinned package would go live with
    # a benchmark nobody verified.
    if cfg.get("benchmark_expected_score") is None:
        raise SystemExit(f"{cfg['_pkg']}: benchmark_expected_score is not pinned "
                         f"— run prepare_data.py and set it in config.py")
    for name in ("overview.md", "env.py"):
        if not (pkg_dir / name).is_file():
            raise SystemExit(f"{cfg['_pkg']}: {name} missing")
    for rel in cfg.get("private_files", []):
        package_file(cfg, rel, "private")
    for rel in cfg.get("public_files") or []:
        package_file(cfg, rel, "public")
    bench = pkg_dir / cfg["benchmark_file"]
    if not bench.is_file():
        raise SystemExit(f"missing benchmark file {cfg['benchmark_file']} — "
                         f"run {cfg['_pkg']}/prepare_data.py")
    name = submission_filename(cfg) if cfg["kernel_version"] == "file_v1" else None
    if name and name.lower().endswith(".gz"):
        with bench.open("rb") as fh:
            if fh.read(2) != b"\x1f\x8b":
                raise SystemExit(f"{cfg['_pkg']}: {cfg['benchmark_file']} is uploaded "
                                 f"as {name} but is not gzip-compressed")


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
# settings, engine, benchmark file — shared by build and refresh
# --------------------------------------------------------------------------- #
def pin_engine(client, cid: int, cfg: dict) -> None:
    """Point the challenge at `engine_id`, when the package names one.

    Without the key the challenge keeps the engine the platform chose for its
    kind at creation, which is what every package did before this key existed.
    """
    engine_id = cfg.get("engine_id")
    if engine_id is None:
        print("    engine: the kind's default (no engine_id in config.py)")
        return
    resp = client.update_challenge_configuration(cid, engine_id=engine_id)
    if resp["engine_id"] != engine_id:
        raise RuntimeError(f"{cfg['name']}: asked for engine {engine_id}, the "
                           f"server reports engine {resp['engine_id']}")
    print(f"    engine pinned: id={engine_id}")


def apply_settings(client, cid: int, cfg: dict) -> None:
    settings = {
        "evaluation_metric": cfg["metric"],
        "evaluation_deployment_nb_constraint_run": cfg["deployment_nb_constraint_run"],
        "evaluation_deployment_nb_initial_score_run": cfg["deployment_nb_initial_score_run"],
        "evaluation_metrics_schema": cfg["metrics_schema"],
    }
    if cfg.get("metric2"):
        settings["evaluation_metric2"] = cfg["metric2"]
    # Sent only when declared, so a package without them changes nothing.
    for key in ("submission_filename", "max_upload_size_bytes"):
        if key in cfg:
            settings[key] = cfg[key]
    resp = client.update_settings(cid, **settings)

    # The response is the stored configuration. Checked because the benchmark
    # is uploaded under the name computed here, and `run_benchmark` looks for
    # the name the server holds: a disagreement would otherwise surface as a
    # "benchmark file is required" error one step later, far from its cause.
    if cfg["kernel_version"] == "file_v1" and resp["submission_filename"] != submission_filename(cfg):
        raise RuntimeError(f"{cfg['name']}: the server's submission file is "
                           f"{resp['submission_filename']!r}, the package expects "
                           f"{submission_filename(cfg)!r}")
    if "max_upload_size_bytes" in cfg and resp["max_upload_size_bytes"] != cfg["max_upload_size_bytes"]:
        raise RuntimeError(f"{cfg['name']}: the server's upload cap is "
                           f"{resp['max_upload_size_bytes']}, the package declares "
                           f"{cfg['max_upload_size_bytes']}")

    extra = ""
    if cfg["kernel_version"] == "file_v1":
        extra += f" submission={submission_filename(cfg)}"
    if "max_upload_size_bytes" in cfg:
        extra += f" max_upload={cfg['max_upload_size_bytes'] / 1024 / 1024:.0f} MiB"
    print(f"    settings: metric={cfg['metric']} "
          f"runs={cfg['deployment_nb_constraint_run']}+"
          f"{cfg['deployment_nb_initial_score_run']} "
          f"metrics_schema={[d['key'] for d in cfg['metrics_schema']]}{extra}")


def upload_benchmark(client, cid: int, cfg: dict) -> None:
    """Store the reference solution where `run_benchmark` looks for it.

    A `.csv` submission keeps the text route every package has always used. Any
    other name goes up as bytes: the JSON content route writes text, so a
    gzip file would not survive it. The multipart route stores the file under
    the part's name, which `filename=` sets to the configured name.
    """
    bench = cfg["_dir"] / cfg["benchmark_file"]
    if cfg["kernel_version"] != "file_v1":
        client.update_benchmark_file_content(cid, "agent.py", bench.read_text())
        print(f"    benchmark file: {cfg['benchmark_file']} -> agent.py")
        return

    name = submission_filename(cfg)
    if name.lower().endswith(".csv"):
        client.update_benchmark_file_content(cid, name, bench.read_text())
    else:
        resp = client.upload_benchmark_file(cid, str(bench), filename=name)
        if resp["filename"] != name:
            raise RuntimeError(f"{cfg['name']}: benchmark stored as "
                               f"{resp['filename']!r}, not {name!r}")
    print(f"    benchmark file: {cfg['benchmark_file']} -> {name} "
          f"({bench.stat().st_size / 1e6:.2f} MB)")


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
    preflight(cfg)
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

    pin_engine(client, cid, cfg)

    client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())
    print("    overview.md published")

    apply_settings(client, cid, cfg)

    validation = client.update_env_file_content(cid, "env.py",
                                                (pkg_dir / "env.py").read_text())
    print(f"    env.py uploaded; structural check="
          f"{validation.get('validation', validation.get('status', 'ok'))}")

    for rel in cfg.get("private_files", []):
        client.upload_env_file(cid, str(package_file(cfg, rel, "private")))
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
            path = package_file(cfg, fname, "public")
            # The server labels a file with the uploaded part's name.
            if path.name in have:
                continue
            client.upload_dataset_file(cid, ds_id, str(path))
            print(f"    dataset file: {fname} ({path.stat().st_size/1e6:.2f} MB)")

    upload_benchmark(client, cid, cfg)

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


def refresh_one(client, user_client, cfg: dict, base_url: str,
                keep_agents: bool = False) -> int:
    """Replace the data of an already-started competition, in place.

    The platform locks settings, datasets and the agent template once a
    competition starts, so this has to stop it first. Stopping leaves agents and
    results untouched (`stop_competition` only clears `is_started`), which is
    exactly the problem when the ids have changed underneath them: a score
    computed against the old submission set is meaningless against the new one
    but still ranks. So the stale agents are deleted, not left on the board.

    `keep_agents=True` is the narrow exception: the bytes and the ids are
    unchanged and only the *file names* differ, so every score on the board was
    computed against data that still exists and stays valid. It is the caller's
    claim, not something this script can verify — pass it only when
    `prepare_data.py` produced the same split, and never to get past the
    other-competitors guard on a real data change.
    """
    name, pkg_dir = cfg["name"], cfg["_dir"]
    print(f"\n=== refresh {name}  ({cfg['_pkg']})"
          f"{'  [keep-agents]' if keep_agents else ''}")
    preflight(cfg)

    existing = find_existing(client, name)
    if not existing:
        raise SystemExit(f"{name}: not on the server — use `build`, not `refresh`.")
    cid = existing["id"]

    me = client.profile().get("username")
    board = client.leaderboard(cid)
    rows = board.to_dict("records") if hasattr(board, "to_dict") else list(board)
    others = sorted({r["Username"] for r in rows if r.get("Username") != me})
    if others and not keep_agents:
        raise SystemExit(
            f"{name}: {len(others)} other competitor(s) on the board ({others}). "
            f"Refreshing invalidates their scores — stop the competition and "
            f"decide deliberately rather than through this script."
        )
    stale = [] if keep_agents else [r for r in rows if r.get("AgentName") != "__benchmark__"]

    client.stop_competition(cid)
    print(f"    stopped id={cid}")
    if keep_agents:
        kept = [r for r in rows if r.get("AgentName") != "__benchmark__"]
        print(f"    keeping {len(kept)} agent(s) on the board "
              f"({sorted({r.get('Username') for r in kept})})")

    for r in stale:
        user_client.delete_agent(cid, r["agentAttachId"])
        print(f"    deleted stale agent {r['agentAttachId']} "
              f"({r.get('AgentName')}, {cfg['metric']}={r.get('MeanReward')})")

    pin_engine(client, cid, cfg)

    client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())
    client.update_env_file_content(cid, "env.py", (pkg_dir / "env.py").read_text())
    print("    overview.md + env.py re-uploaded")

    # Settings are locked while started, so they can only be re-applied here.
    # A refresh that changes the metric and not the schema would fail the
    # worker's equal-mapping check on the first run, not on the upload.
    apply_settings(client, cid, cfg)

    for rel in cfg.get("private_files", []):
        client.upload_env_file(cid, str(package_file(cfg, rel, "private")))
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
        # upload_dataset_file always adds a row, so replacing means delete
        # first — and delete *every* file, not just the ones public_files still
        # names. A rename (X_train.csv -> X.csv) or a dropped file would
        # otherwise leave the old row served alongside the new one.
        for f in ds.get("files", []):
            client.delete_dataset_file(cid, ds_id, f["id"])
            print(f"    dataset file deleted: {f['label']}")
        for fname in public:
            path = package_file(cfg, fname, "public")
            client.upload_dataset_file(cid, ds_id, str(path))
            print(f"    dataset file uploaded: {fname} "
                  f"({path.stat().st_size / 1e6:.2f} MB)")

    upload_benchmark(client, cid, cfg)

    client.run_benchmark(cid)
    verify_benchmark(cfg, wait_for_benchmark(client, cid))

    client.start_competition(cid)
    print(f"    RESTARTED id={cid}")
    return cid


def do_refresh(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    # Deleting a stale agent is the only thing here that needs a user key, so a
    # keep-agents refresh does not ask for one.
    user_client = (None if args.keep_agents
                   else connect("MLARENA_USER_API_KEY", args.base_url))
    for pkg in args.packages:
        refresh_one(client, user_client, load_config(pkg), args.base_url,
                    keep_agents=args.keep_agents)
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


def _built_entries(args) -> list:
    """Lockfile entries of the selected packages. The lockfile holds every
    course's challenges, so iterating all of it would publish (or tear down)
    another course's work."""
    built = read_state(args.base_url).get("competitions") or {}
    return [(pkg, built[pkg]) for pkg in args.packages if pkg in built]


def do_publish(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    for pkg, entry in _built_entries(args):
        client.update_competition(entry["id"], is_public=True)
        print(f"published {pkg} id={entry['id']}")


def do_teardown(args):
    client = connect("MLARENA_API_KEY", args.base_url)
    for pkg, entry in _built_entries(args):
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

    A course that states its own bar declares `pass_threshold` in config.py,
    and that value is written instead (mlp-s1-store-sales: -20.0, the MAE <= 20
    of the original exercise). It is still read from the package, so it cannot
    drift from the metric the way a hand-typed value did. A bar above the
    benchmark (s2-dpe-energy-label: 0.85, "do the domain steps") is allowed
    when the package declares `expert_expected_score` at or above it — see
    `validate_config`; before that rule the DPE bar had to be typed on the
    link by hand, and this reconciliation would have reverted it.
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
        threshold = pass_threshold(load_config(pkg))
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


def select_packages(course: str, only) -> list:
    if course not in PACKAGES_BY_COURSE:
        sys.exit(f"no competition packages for course {course!r}; "
                 f"known: {sorted(PACKAGES_BY_COURSE)}")
    in_course = PACKAGES_BY_COURSE[course]
    if not only:
        return list(in_course)
    unknown = [p for p in only if p not in PACKAGES]
    if unknown:
        sys.exit(f"unknown package(s): {unknown}; known: {PACKAGES}")
    elsewhere = [p for p in only if p not in in_course]
    if elsewhere:
        owners = {p: c for c, pkgs in PACKAGES_BY_COURSE.items() for p in pkgs}
        sys.exit(f"package(s) not in course {course!r}: "
                 f"{ {p: owners[p] for p in elsewhere} } — pass --course")
    return list(only)


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
    ap.add_argument("--keep-agents", action="store_true",
                    help="refresh only: the split is unchanged and only the "
                         "file names differ, so leave the leaderboard alone")
    args = ap.parse_args()
    args.packages = select_packages(args.course, args.only)
    {"build": do_build, "overview": do_overview, "refresh": do_refresh,
     "status": do_status, "publish": do_publish, "attach": do_attach,
     "teardown": do_teardown}[args.mode](args)


if __name__ == "__main__":
    main()
