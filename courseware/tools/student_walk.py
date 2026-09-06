#!/usr/bin/env python3
"""Walk a published ML-Arena course exactly as an enrolled student sees it.

Two jobs, kept separate on purpose:

  dump    read the course through the *student* token and write it to disk —
          landing page, every module overview, every lesson body, every
          attached competition card. What the dump contains is what a student
          can actually reach; what 404s is recorded as a 404, not skipped.

  check   the same walk, but assert the things a student needs to be true:
          lesson bodies non-empty, images resolvable, attached competitions
          openable, a stated baseline on every competition.

The point of dumping before reviewing is that a review of the *repo markdown*
reviews the intent; a review of this dump reviews what was delivered. The two
drift, and the drift is where the student-facing bugs live.

    export MLARENA_STUDENT_API_KEY=mlk_user_...
    python tools/student_walk.py dump  --course python-ai-engineering --out ../student_view
    python tools/student_walk.py check --course ms2a-machine-learning-practice
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_SDK = Path(__file__).resolve().parents[3].parent / "mlarena-sdk"
if _SDK.exists():
    sys.path.insert(0, str(_SDK))

import mlarena  # noqa: E402

BASE_URL = os.environ.get("MLARENA_BASE_URL", "https://ml-arena.com")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
NOTES_RE = re.compile(r"<!--\s*notes:.*?-->", re.S)


def _client(env_var: str, scope: str):
    token = os.environ.get(env_var)
    if not token:
        raise SystemExit(f"{env_var} is not set (need a {scope}-scope key)")
    return mlarena.connect(token, base_url=BASE_URL)


def walk(client, slug: str) -> dict:
    """Fetch the whole course through the student surface. Errors are data."""
    course = client.course(slug)
    out = {"slug": slug, "course": {k: v for k, v in course.items() if k != "modules"},
           "modules": []}
    for mod in course.get("modules", []):
        entry = {"module": {k: v for k, v in mod.items() if k != "lessons"},
                 "lessons": [], "competitions": []}
        try:
            entry["overview"] = client.module_overview(slug, mod["slug"])
        except Exception as exc:                    # noqa: BLE001 — recorded, not handled
            entry["overview_error"] = f"{type(exc).__name__}: {exc}"
        for les in mod.get("lessons", []):
            try:
                body = client.lesson(slug, mod["slug"], les["slug"])
            except Exception as exc:                # noqa: BLE001
                entry["lessons"].append({**les, "error": f"{type(exc).__name__}: {exc}"})
                continue
            entry["lessons"].append(body)
        for comp in mod.get("competitions", []):
            cid = comp["competition_id"]
            try:
                entry["competitions"].append({**comp, "detail": client.competition(cid)})
            except Exception as exc:                # noqa: BLE001
                entry["competitions"].append({**comp, "error": f"{type(exc).__name__}: {exc}"})
        out["modules"].append(entry)
    return out


def cmd_dump(args) -> int:
    client = _client("MLARENA_STUDENT_API_KEY", "user")
    data = walk(client, args.course)
    root = Path(args.out) / args.course
    root.mkdir(parents=True, exist_ok=True)
    (root / "_course.json").write_text(json.dumps(data["course"], indent=2, default=str))
    index = []
    stale: list[str] = []
    for m in data["modules"]:
        mslug = m["module"]["slug"]
        mdir = root / mslug
        mdir.mkdir(exist_ok=True)
        (mdir / "_module.json").write_text(json.dumps(
            {k: v for k, v in m.items() if k != "lessons"}, indent=2, default=str))
        written = {"_module.json"}
        for les in m["lessons"]:
            name = les.get("slug", "unknown")
            if "error" in les:
                (mdir / f"{name}.ERROR.txt").write_text(les["error"])
                written.add(f"{name}.ERROR.txt")
                index.append(f"{mslug}/{name}\tERROR\t{les['error']}")
                continue
            body = les.get("body_md") or ""
            (mdir / f"{name}.md").write_text(body)
            written.add(f"{name}.md")
            index.append(f"{mslug}/{name}\t{len(body)}\t{les.get('estimated_minutes')}min")

        # A dump that keeps the lessons a restructure deleted is a dump of a
        # course that does not exist — and it is read as evidence of what was
        # delivered. Writing over the survivors is not enough; the ones that are
        # gone have to go.
        for path in sorted(mdir.iterdir()):
            if path.is_file() and path.name not in written:
                path.unlink()
                stale.append(f"{mslug}/{path.name}")

    (root / "_index.tsv").write_text("\n".join(index) + "\n")
    print(f"dumped {len(data['modules'])} modules -> {root}")
    if stale:
        print(f"removed {len(stale)} file(s) for lessons no longer in the course:")
        for name in stale:
            print(f"  {name}")
    return 0


def cmd_check(args) -> int:
    client = _client("MLARENA_STUDENT_API_KEY", "user")
    data = walk(client, args.course)
    problems: list[str] = []
    c = data["course"]
    if not c.get("is_enrolled"):
        problems.append(f"COURSE\tnot-enrolled\tstudent cannot record progress on '{args.course}'")
    for m in data["modules"]:
        mslug = m["module"]["slug"]
        if "overview_error" in m:
            problems.append(f"{mslug}\toverview-404\t{m['overview_error']}")
        for les in m["lessons"]:
            lslug = les.get("slug", "?")
            if "error" in les:
                problems.append(f"{mslug}/{lslug}\tlesson-error\t{les['error']}")
                continue
            body = les.get("body_md") or ""
            if not body.strip():
                problems.append(f"{mslug}/{lslug}\tempty-body\t0 chars")
            for w in les.get("directive_warnings") or []:
                problems.append(f"{mslug}/{lslug}\tdirective-warning\t{w}")
            for src in IMG_RE.findall(body):
                if not src.startswith(("http://", "https://", "/")):
                    problems.append(f"{mslug}/{lslug}\tunuploaded-image\t{src}")
            if NOTES_RE.search(body):
                problems.append(f"{mslug}/{lslug}\tspeaker-notes-in-body\tteacher notes shipped to students")
        for comp in m["competitions"]:
            cid = comp["competition_id"]
            if "error" in comp:
                problems.append(f"{mslug}\tcompetition-unreachable\t#{cid} {comp['error']}")
                continue
            det = comp["detail"]
            if not det.get("is_started"):
                problems.append(f"{mslug}\tcompetition-not-started\t#{cid} {det.get('name')}")
    for p in problems:
        print(p)
    print(f"\n{len(problems)} problem(s) on '{args.course}'", file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="write the student view to disk")
    d.add_argument("--course", required=True)
    d.add_argument("--out", default="student_view")
    d.set_defaults(func=cmd_dump)
    k = sub.add_parser("check", help="assert the student view is followable")
    k.add_argument("--course", required=True)
    k.set_defaults(func=cmd_check)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
