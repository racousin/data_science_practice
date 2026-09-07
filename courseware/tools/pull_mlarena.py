#!/usr/bin/env python3
"""Bring a website edit back into the Markdown — the inverse of `publish_mlarena.py`.

`make check-sync` tells you someone edited a lesson in the ML-Arena course
editor and that the next `make publish` will overwrite it. This is what stops
that being a choice between losing the edit and hand-copying a diff: it rewrites
the source Markdown from the live body, so the repo is the source of truth again
and the publish is a no-op.

Scope: **lesson bodies only.**

That is deliberate, not an omission. A body is a file this tool owns end to end,
so overwriting it is safe and reversible through git. Everything else — a title,
a lesson deleted on the website, a reordering — lives in `course.yaml`, whose
comments carry the reasoning behind every structural decision in the course. A
YAML round-trip deletes all of them. So structural drift is *reported*, as the
exact edits to make by hand, and the tool touches nothing.

Image paths are rewritten back on the way down. The Markdown keeps repo-relative
paths so the PPTX build works; the publisher rewrote them to
`/api/academic_courses/assets/lessons/<id>/<file>` on the way up. Each served URL
is matched to a local file by basename — first against the paths the local body
already uses, then against `assets/<module>/<lesson>/`. A URL that matches
neither is left alone and reported, because inventing a path would produce a
lesson that publishes a broken image.

A pull also records what the publisher's guard calls the **baseline**: once the
repo holds the live body, that body is ours, and the next `make publish` can
prove the server was not touched since. Bodies only, and that limit is load
bearing — the titles and orders this tool only *reports* are not absorbed, so
baselining them would tell the next publish it was free to overwrite the very
edit it just printed as a to-do. See `lesson_sync.py`.

Usage::

    export MLARENA_API_KEY=mlk_teacher_...
    python tools/pull_mlarena.py content/python-ai-engineering --dry-run
    python tools/pull_mlarena.py content/python-ai-engineering --module s1-git-and-packaging
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same file, so the two tools cannot disagree about what "the same" means: the
# SDK path insertion, the image normalisation and the metadata field list all
# come from the checker.
from check_sync import META, declared, normalise  # noqa: E402

import lesson_sync  # noqa: E402
from publish_mlarena import load_state, save_state  # noqa: E402

import mlarena  # noqa: E402
import yaml  # noqa: E402

# An image the publisher rewrote. Matching the media route rather than "any
# absolute URL" is what keeps genuinely external images — a CI badge, say —
# out of this: they were written by hand and there is no local file behind them.
SERVED_IMAGE_RE = re.compile(
    r"(!\[[^\]]*\]\()((?:https?://[^)\s/]+)?/api/academic_courses/assets/[^)\s]+)(\))")


def resolve_asset(url: str, content: Path, local_body: str, module_slug: str,
                  lesson_slug: str) -> str | None:
    """The repo-relative path a served image URL came from, or None."""
    name = os.path.basename(url.split("?", 1)[0])

    # The body being replaced is the best evidence: it is the same lesson, and
    # its paths are the ones the deck build already resolves.
    for _alt, rel in re.findall(r"!\[([^\]]*)\]\((?!https?://|/)([^)\s]+)\)", local_body):
        if os.path.basename(rel) == name:
            return rel

    # A lesson whose body was replaced wholesale on the website has no local
    # paths left to match. Fall back to where the figure scripts put them.
    candidate = f"assets/{module_slug}/{lesson_slug}/{name}"
    if (content / candidate).is_file():
        return candidate

    matches = sorted(p for p in (content / "assets").rglob(name) if p.is_file())
    if len(matches) == 1:
        return str(matches[0].relative_to(content))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("content", help="course content directory (holds course.yaml)")
    ap.add_argument("--base-url",
                    default=os.environ.get("MLARENA_BASE_URL", "https://ml-arena.com"))
    ap.add_argument("--module", action="append", help="only this module slug")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be written, write nothing")
    ap.add_argument("--diff", action="store_true", help="print the unified diff")
    args = ap.parse_args()

    token = os.environ.get("MLARENA_API_KEY")
    if not token:
        raise SystemExit("MLARENA_API_KEY is not set (needs a mlk_teacher_... key)")

    content = Path(args.content)
    manifest = yaml.safe_load((content / "course.yaml").read_text())
    client = mlarena.connect(token, base_url=args.base_url)

    course = client.course(manifest["course"]["slug"])
    live = {m["slug"]: {l["slug"]: l for l in m.get("lessons", [])}
            for m in course.get("modules", [])}
    live_order = {m["slug"]: [l["slug"] for l in m.get("lessons", [])]
                  for m in course.get("modules", [])}

    written: list[tuple[str, int, int]] = []
    unresolved: list[str] = []
    todo: list[str] = []
    # Every body this run compared, rewritten or already identical: either way
    # the repo and the server now hold the same text, which is what the
    # baseline records.
    baselined: dict[str, str] = {}

    for module in manifest["modules"]:
        mslug = module["slug"]
        if args.module and mslug not in args.module:
            continue
        if mslug not in live:
            todo.append(f"module {mslug} is in course.yaml and not on the server")
            continue

        want = [s["slug"] for s in module.get("lessons", [])]
        if want != live_order[mslug]:
            todo.append(
                f"reorder {mslug} — put the lessons in course.yaml in the server's order:\n"
                f"    {', '.join(live_order[mslug])}")

        for spec in module.get("lessons", []):
            lslug = spec["slug"]
            row = live[mslug].get(lslug)
            if row is None:
                todo.append(
                    f"delete {mslug}/{lslug} — deleted on the website; drop its entry "
                    f"from course.yaml and remove {spec['file']}")
                continue

            for field in META:
                a, b = declared(spec)[field], row.get(field)
                if a != b:
                    todo.append(
                        f"{mslug}/{lslug} (#{row['id']}) — set {field}: {a!r} -> {b!r}")

            path = content / spec["file"]
            local_raw = path.read_text()
            remote = client.get_lesson(row["id"])
            body = (remote.get("lesson", remote).get("body_md") or "")
            body = body.replace("\r\n", "\n")

            def _back(match: re.Match) -> str:
                rel = resolve_asset(match.group(2), content, local_raw, mslug, lslug)
                if rel is None:
                    unresolved.append(f"{mslug}/{lslug}: {match.group(2)}")
                    return match.group(0)
                return match.group(1) + rel + match.group(3)

            body = SERVED_IMAGE_RE.sub(_back, body)
            body = body.rstrip("\n") + "\n"

            baselined[f"{mslug}/{lslug}"] = lesson_sync.digest(body)

            if normalise(local_raw) == normalise(body):
                continue

            delta = list(difflib.ndiff(normalise(local_raw), normalise(body)))
            added = sum(1 for line in delta if line.startswith("+ "))
            removed = sum(1 for line in delta if line.startswith("- "))
            written.append((f"{mslug}/{lslug}", added, removed))
            if args.diff:
                for line in difflib.unified_diff(
                        normalise(local_raw), normalise(body),
                        fromfile=f"repo/{lslug}", tofile=f"live/{lslug}",
                        lineterm="", n=2):
                    print("    " + line)
            if not args.dry_run:
                path.write_text(body)

        for lslug, row in live[mslug].items():
            if lslug not in want:
                todo.append(
                    f"add {mslug}/{lslug} (#{row['id']}) — on the server, not in "
                    f"course.yaml; `pull` does not create files")

    verb = "would rewrite" if args.dry_run else "rewrote"
    for name, added, removed in written:
        print(f"{verb} {name} — +{added}/-{removed} lines from live")
    if not written:
        print("no lesson body differs from the live course")

    if unresolved:
        print(f"\n{len(unresolved)} uploaded image(s) left as served URLs — no file "
              f"under assets/ has that basename:")
        for line in unresolved:
            print(f"    {line}")
        print("    the deck build needs a repo-relative path (`make slides` runs with "
              "--strict-assets); add the file or fix the reference by hand.")

    if todo:
        print(f"\n{len(todo)} change(s) this tool does not make — edit course.yaml "
              f"by hand (its comments are the record of why the course is shaped "
              f"the way it is):")
        for line in todo:
            print(f"  * {line}")

    if baselined and not args.dry_run:
        state = load_state(str(content), args.base_url)
        lessons = state["published"]["lessons"]
        for key, body_digest in baselined.items():
            entry = dict(lessons.get(key) or {})
            entry["body"] = body_digest
            lessons[key] = entry
        save_state(str(content), args.base_url, state)
        print(f"\nbaselined {len(baselined)} lesson body(ies) — the next `make publish` "
              f"can tell a website edit from a local one.")

    if written and not args.dry_run:
        print("Rebuild what reads these files: `make check-slides` then `make slides`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
