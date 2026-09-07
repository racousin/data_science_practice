#!/usr/bin/env python3
"""Is the live course what this repo says it is?

Read-only: it never writes to ML-Arena and never touches your files.

This is the *inspection* tool, not the guard. It compares the repo against the
live course, which sees every difference in either direction and cannot say
which side moved — a lesson you edited here and a lesson someone edited on the
website look identical to it. Gating a publish on that would block every
publish that has anything to publish, so the guard is a different comparison
and lives inside the publisher (`lesson_sync.py`, three-way, against what was
last published).

What this is for: reading a difference once you know there is one — after a
publish, or when `make publish` has refused and you want the diff.

Two tiers, because they cost different amounts:

    make check-sync QUICK=1     1 request.  Structure only: which lessons exist,
                                their order, titles, kind, published flag,
                                estimated minutes. Catches a lesson unpublished,
                                renamed, reordered, added or deleted on the site.

    make check-sync             1 + N requests (~57 for a 12h course, a few
                                seconds). Everything above, plus the body of
                                every lesson compared line by line.

    make check-sync DIFF=1      ... and print the diff, so you can see what to
                                copy back into the markdown.

Exit status is 1 if anything differs, in either direction.

What "the same" means
---------------------
The markdown on disk keeps repo-relative image paths so the PPTX build works;
the publisher rewrites them to served URLs on the way up. Both sides are
therefore reduced to the image's basename before comparing, and trailing
whitespace is ignored. Everything else is compared literally, speaker-notes
comments included.
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_SDK = Path(__file__).resolve().parents[3].parent / "mlarena-sdk"
if _SDK.exists():
    sys.path.insert(0, str(_SDK))

# One definition of "the same body", shared with the publisher's guard and with
# `pull_mlarena.py`, so no two of the three can drift apart about it.
from lesson_sync import IMAGE_RE, normalise  # noqa: E402,F401

import mlarena  # noqa: E402
import yaml  # noqa: E402

# Fields the manifest owns. `estimated_minutes` is included because it is the
# one a teacher most plausibly nudges in the editor without thinking of it as
# an edit.
META = ("title", "kind", "is_published", "estimated_minutes")


def declared(spec: dict) -> dict:
    return {
        "title": spec["title"],
        "kind": spec.get("kind", "lesson"),
        "is_published": bool(spec.get("is_published", False)),
        "estimated_minutes": spec.get("estimated_minutes"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("content", help="course content directory (holds course.yaml)")
    ap.add_argument("--base-url",
                    default=os.environ.get("MLARENA_BASE_URL", "https://ml-arena.com"))
    ap.add_argument("--module", action="append", help="only this module slug")
    ap.add_argument("--quick", action="store_true",
                    help="structure only — one request, no bodies")
    ap.add_argument("--diff", action="store_true", help="print the unified diff")
    args = ap.parse_args()

    token = os.environ.get("MLARENA_API_KEY")
    if not token:
        raise SystemExit("MLARENA_API_KEY is not set (needs a mlk_teacher_... key)")

    content = Path(args.content)
    manifest = yaml.safe_load((content / "course.yaml").read_text())
    client = mlarena.connect(token, base_url=args.base_url)

    # One request. Carries every module, every lesson id, and the metadata the
    # quick tier compares — unpublished lessons included, for a key that can
    # manage the course.
    course = client.course(manifest["course"]["slug"])
    live = {m["slug"]: {l["slug"]: l for l in m.get("lessons", [])}
            for m in course.get("modules", [])}
    live_order = {m["slug"]: [l["slug"] for l in m.get("lessons", [])]
                  for m in course.get("modules", [])}

    problems: list[str] = []
    diffs: list[tuple] = []
    checked = 0

    for module in manifest["modules"]:
        mslug = module["slug"]
        if args.module and mslug not in args.module:
            continue
        if mslug not in live:
            problems.append(f"MISSING  module {mslug} — in course.yaml, not on the server")
            continue

        want = [s["slug"] for s in module.get("lessons", [])]
        if want != live_order[mslug]:
            problems.append(
                f"ORDER    {mslug} — the server's lesson order is not the manifest's\n"
                f"         repo: {', '.join(want)}\n"
                f"         live: {', '.join(live_order[mslug])}")

        for spec in module.get("lessons", []):
            lslug = spec["slug"]
            row = live[mslug].get(lslug)
            if row is None:
                problems.append(f"MISSING  {mslug}/{lslug} — in course.yaml, not on the server")
                continue

            for field in META:
                a, b = declared(spec)[field], row.get(field)
                if a != b:
                    problems.append(
                        f"META     {mslug}/{lslug} (#{row['id']}) — {field}: "
                        f"repo {a!r} vs live {b!r}")

            if args.quick:
                continue

            local = normalise((content / spec["file"]).read_text())
            remote = client.get_lesson(row["id"])
            remote = normalise((remote.get("lesson", remote).get("body_md") or ""))
            checked += 1
            if local != remote:
                diffs.append((f"{mslug}/{lslug}", row["id"], local, remote))

        for lslug, row in live[mslug].items():
            if lslug not in want:
                problems.append(
                    f"ORPHAN   {mslug}/{lslug} (#{row['id']}) — on the server, "
                    f"not in course.yaml")

    for name, lesson_id, local, remote in diffs:
        delta = list(difflib.ndiff(local, remote))
        added = sum(1 for line in delta if line.startswith("+ "))
        removed = sum(1 for line in delta if line.startswith("- "))
        print(f"BODY     {name} (#{lesson_id}) — live has +{added}/-{removed} lines vs repo")
        if args.diff:
            for line in difflib.unified_diff(local, remote, fromfile=f"repo/{name}",
                                             tofile=f"live/{name}", lineterm="", n=2):
                print("         " + line)
    for line in problems:
        print(line)

    total = len(diffs) + len(problems)
    scope = "structure" if args.quick else f"structure + {checked} bodies"
    print(f"\n{scope} compared — {total} difference(s)")
    if not total:
        print("in sync: the live course is what this repo says it is")
    if diffs:
        print("\nA BODY difference is the repo and the site disagreeing; it does not\n"
              "say which one moved. `make publish` refuses if it was the site (it\n"
              "compares against what it last published), and `make pull` is how the\n"
              "site's copy comes back here. `make check-sync DIFF=1` shows the text.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
