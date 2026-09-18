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

Structure means everything `course.yaml` declares and the server holds: the
course fields, each module's title / icon / summary and its challenge
attachments (id and label), the module order, and each lesson's presence,
order, title, kind, published and gated flags and estimated minutes.

Exit status is 1 if anything differs, in either direction.

`structure()` is shared with `pull_mlarena.py`: each finding carries the edit
that makes the repo match the website (`Finding.fix`), or None for the few a
pull reports but does not make — a module added, removed or reordered.

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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_SDK = Path(__file__).resolve().parents[3].parent / "mlarena-sdk"
if _SDK.exists():
    sys.path.insert(0, str(_SDK))

# One definition of "the same body", shared with the publisher's guard and with
# `pull_mlarena.py`, so no two of the three can drift apart about it.
from lesson_sync import COURSE_FIELDS, IMAGE_RE, _brief, _loose, normalise  # noqa: E402,F401

import mlarena  # noqa: E402
import yaml  # noqa: E402

# Fields the manifest owns. `estimated_minutes` is included because it is the
# one a teacher most plausibly nudges in the editor without thinking of it as
# an edit; `gated` because the publisher writes it on every run.
META = ("title", "kind", "is_published", "gated", "estimated_minutes")
MODULE_META = ("title", "icon", "summary")


def declared(spec: dict) -> dict:
    return {
        "title": spec["title"],
        "kind": spec.get("kind", "lesson"),
        "is_published": bool(spec.get("is_published", False)),
        "gated": bool(spec.get("gated", False)),
        "estimated_minutes": spec.get("estimated_minutes"),
    }


def declared_links(module: dict) -> list[tuple[int, str | None]]:
    return [(c["competition_id"], c.get("label"))
            for c in module.get("competitions") or []]


def live_links(module: dict) -> list[tuple[int, str | None]]:
    """The course payload serves a module's links in position order."""
    return [(c["challenge_id"], c.get("label")) for c in module["challenges"]]


def _links(links: list) -> str:
    return ", ".join(f"{cid} {_brief(label)!r}" for cid, label in links) or "(none)"


@dataclass
class Finding:
    """One way the repo and the live course differ.

    `fix` is the edit that makes the repo match the website — a tuple whose
    first element names it (see `pull_mlarena.apply`) — or None when a pull
    only reports it.
    """
    tag: str
    text: str
    fix: tuple | None = None

    def __str__(self) -> str:
        return f"{self.tag:<8} {self.text}"


def structure(manifest: dict, course: dict, modules: list[str] | None) -> list[Finding]:
    """Everything but the lesson bodies: one walk shared by check-sync, which
    prints it, and pull, which applies it. `modules` scopes the walk; the
    course fields and the module order are only compared on a full run."""
    found: list[Finding] = []
    live_modules = {m["slug"]: m for m in course["modules"]}
    want_modules = [m["slug"] for m in manifest["modules"]]

    if modules is None:
        spec = manifest["course"]
        for field in COURSE_FIELDS:
            a, b = spec.get(field), course.get(field)
            if not _loose(None if a is None else str(a), b):
                found.append(Finding(
                    "COURSE", f"{field}: repo {_brief(a)!r} vs live {_brief(b)!r}",
                    ("course", field, b)))
        if want_modules != list(live_modules):
            found.append(Finding(
                "ORDER", "the server's module order is not the manifest's\n"
                f"         repo: {', '.join(want_modules)}\n"
                f"         live: {', '.join(live_modules)}"))
        for mslug in live_modules:
            if mslug not in want_modules:
                found.append(Finding(
                    "ORPHAN", f"module {mslug} (#{live_modules[mslug]['module_id']}) — "
                              f"on the server, not in course.yaml"))

    for module in manifest["modules"]:
        mslug = module["slug"]
        if modules is not None and mslug not in modules:
            continue
        live_module = live_modules.get(mslug)
        if live_module is None:
            found.append(Finding("MISSING", f"module {mslug} — in course.yaml, not on the server"))
            continue

        for field in MODULE_META:
            a, b = module.get(field), live_module.get(field)
            if not _loose(a, b):
                found.append(Finding(
                    "META", f"module {mslug} — {field}: repo {_brief(a)!r} vs live {_brief(b)!r}",
                    ("module", mslug, field, b)))

        a, b = declared_links(module), live_links(live_module)
        if a != b:
            found.append(Finding(
                "LINK", f"{mslug} — challenges\n"
                        f"         repo: {_links(a)}\n         live: {_links(b)}",
                ("links", mslug, b)))

        live = {l["slug"]: l for l in live_module["lessons"]}
        want = [s["slug"] for s in module.get("lessons") or []]
        for spec in module.get("lessons") or []:
            lslug = spec["slug"]
            row = live.get(lslug)
            if row is None:
                found.append(Finding(
                    "MISSING", f"{mslug}/{lslug} — in course.yaml, not on the server",
                    ("lesson-delete", mslug, lslug, spec["file"])))
                continue
            for field in META:
                a, b = declared(spec)[field], row.get(field)
                if a != b:
                    found.append(Finding(
                        "META", f"{mslug}/{lslug} (#{row['id']}) — {field}: "
                                f"repo {a!r} vs live {b!r}",
                        ("lesson", mslug, lslug, field, b)))
        for lslug, row in live.items():
            if lslug not in want:
                found.append(Finding(
                    "ORPHAN", f"{mslug}/{lslug} (#{row['id']}) — on the server, "
                              f"not in course.yaml",
                    ("lesson-add", mslug, row)))
        if want != list(live):
            found.append(Finding(
                "ORDER", f"{mslug} — the server's lesson order is not the manifest's\n"
                         f"         repo: {', '.join(want)}\n"
                         f"         live: {', '.join(live)}",
                ("lesson-order", mslug, list(live))))
    return found


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
    problems = structure(manifest, course, args.module)

    diffs: list[tuple] = []
    checked = 0
    if not args.quick:
        live = {m["slug"]: {l["slug"]: l for l in m["lessons"]} for m in course["modules"]}
        for module in manifest["modules"]:
            mslug = module["slug"]
            if args.module and mslug not in args.module:
                continue
            for spec in module.get("lessons") or []:
                row = live.get(mslug, {}).get(spec["slug"])
                if row is None:
                    continue
                local = normalise((content / spec["file"]).read_text())
                remote = client.get_lesson(row["id"])
                remote = normalise((remote.get("lesson", remote).get("body_md") or ""))
                checked += 1
                if local != remote:
                    diffs.append((f"{mslug}/{spec['slug']}", row["id"], local, remote))

    for name, lesson_id, local, remote in diffs:
        delta = list(difflib.ndiff(local, remote))
        added = sum(1 for line in delta if line.startswith("+ "))
        removed = sum(1 for line in delta if line.startswith("- "))
        print(f"BODY     {name} (#{lesson_id}) — live has +{added}/-{removed} lines vs repo")
        if args.diff:
            for line in difflib.unified_diff(local, remote, fromfile=f"repo/{name}",
                                             tofile=f"live/{name}", lineterm="", n=2):
                print("         " + line)
    for finding in problems:
        print(finding)

    total = len(diffs) + len(problems)
    scope = "structure" if args.quick else f"structure + {checked} bodies"
    print(f"\n{scope} compared — {total} difference(s)")
    if not total:
        print("in sync: the live course is what this repo says it is")
    else:
        print("\nA difference does not say which side moved. `make publish` refuses if\n"
              "it was the website (it compares against what it last published).\n"
              "`make pull` makes the repo match the website — everything above except\n"
              "a module added, removed or reordered, which it lists as an edit to make\n"
              "by hand. `make check-sync DIFF=1` shows the text of a BODY difference.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
