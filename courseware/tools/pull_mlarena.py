#!/usr/bin/env python3
"""Make the repo match the website — the inverse of `publish_mlarena.py`.

The website is where the course gets edited while it is being taught: a lesson
trimmed in the course editor, a lab unpublished until its session, a challenge
detached. `make check-sync` tells you that happened; this brings it back, so
the repo is the website again and the next `make publish` is a no-op rather
than an overwrite.

What it writes:

* **lesson bodies** — each Markdown file, from the live body;
* **course.yaml** — every lesson's title, kind, published and gated flags and
  estimated minutes, the lesson order, lessons added or deleted on the website,
  each module's title / icon / summary and its challenge attachments (id and
  label), and the course fields;
* **images** uploaded in the course editor, which have no file here yet —
  downloaded next to the lesson's other images.

course.yaml is edited a line at a time (`course_yaml.py`), never round-tripped
through a YAML library: its comments are the record of why the course is
shaped the way it is, and they survive. The result is re-parsed and compared
with the live course again before anything is written, so an edit that did
not land is a crash rather than a manifest that is quietly wrong.

What it only reports: a **module** added, deleted or reordered on the website.
Those are restructures (module slugs are immutable server-side; a new module
brings a directory, lessons and attachments with it), rare, and worth a person
reading.

Image paths are rewritten back on the way down. The Markdown keeps
repo-relative paths so the PPTX build works; the publisher rewrote them to
`/api/academic_courses/assets/lessons/<id>/<file>` on the way up. Each served
URL is matched to a local file by basename — first against the paths the local
body already uses, then against `assets/<module>/<lesson>/`, then anywhere under
`assets/` if the name is unique. A URL that matches nothing was uploaded on the
website: it is downloaded into the directory this lesson's (or failing that,
its module's) images already live in.

A pull also records what the publisher's guard calls the **baseline**: once
the repo holds what the server holds, that state is ours, and the next `make
publish` can prove the server was not touched since. Recorded field by field,
and only where the repo and the server now agree (`lesson_sync.adopt`) — a
module reorder this tool merely reported keeps its old baseline, so the
publish still refuses over it.

Usage::

    python tools/pull_mlarena.py content/python-ai-engineering --dry-run
    python tools/pull_mlarena.py content/python-ai-engineering --module s1-git-and-packaging
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same file, so the two tools cannot disagree about what "the same" means: the
# SDK path insertion, the image normalisation and the structural walk all come
# from the checker.
from check_sync import normalise, structure  # noqa: E402

import lesson_sync  # noqa: E402
from course_yaml import CourseYaml  # noqa: E402
from publish_mlarena import LOCAL_IMAGE_RE, build_plan, load_state, save_state  # noqa: E402

import mlarena  # noqa: E402
import yaml  # noqa: E402

# An image the publisher rewrote, or one uploaded in the course editor. Matching
# the media route rather than "any absolute URL" is what keeps genuinely
# external images — a CI badge, say — out of this: they were written by hand
# and there is no local file behind them.
SERVED_IMAGE_RE = re.compile(
    r"(!\[[^\]]*\]\()((?:https?://[^)\s/]+)?/api/academic_courses/assets/[^)\s]+)(\))")
MEDIA_RE = re.compile(r"/api/academic_courses/assets/lessons/(\d+)/([^?#)\s]+)")

# The order edits are applied in. Deletions and additions first, so the reorder
# sees the final set of lessons; everything else addresses items by slug and
# does not care.
ORDER = ("lesson-delete", "lesson-add", "lesson", "lesson-order", "links",
         "module", "course")


def resolve_asset(url: str, content: Path, local_body: str, module_slug: str,
                  lesson_slug: str) -> str | None:
    """The repo-relative path a served image URL came from, or None."""
    name = unquote(os.path.basename(url.split("?", 1)[0]))

    # The body being replaced is the best evidence: it is the same lesson, and
    # its paths are the ones the deck build already resolves.
    for _alt, rel in LOCAL_IMAGE_RE.findall(local_body):
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


def asset_dir(bodies: list[str], module_slug: str, lesson_slug: str) -> str:
    """Where an image uploaded on the website goes: the directory the lesson's
    own images already live in, else its module's, else the per-lesson
    directory `python-ai-engineering` uses. `bodies` is [lesson, *siblings]."""
    for group in (bodies[:1], bodies[1:]):
        dirs = Counter(os.path.dirname(rel) for body in group
                       for _alt, rel in LOCAL_IMAGE_RE.findall(body)
                       if rel.startswith("assets/"))
        if dirs:
            return dirs.most_common(1)[0][0]
    return f"assets/{module_slug}/{lesson_slug}"


def download(client, url: str, dest: Path) -> None:
    """Fetch an uploaded image to `dest`. An existing file there must be the
    same bytes — two different images with one name is a crash, not a pick."""
    lesson_id, name = MEDIA_RE.search(url).groups()
    with tempfile.TemporaryDirectory() as tmp:
        got = Path(client.download_lesson_media(int(lesson_id), unquote(name), dest_dir=tmp))
        if dest.exists():
            if dest.read_bytes() != got.read_bytes():
                raise SystemExit(f"{dest} exists and is not the image at {url}; "
                                 f"rename one of them and re-run")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), dest)


def lesson_dir(module: dict) -> str:
    """Where a lesson added on the website gets its file: the directory its
    module's other lessons are in, else the module slug."""
    dirs = Counter(os.path.dirname(s["file"]) for s in module.get("lessons") or [])
    return dirs.most_common(1)[0][0] if dirs else module["slug"]


def apply(doc: CourseYaml, fix: tuple, manifest: dict) -> str | None:
    """Make one structural edit. Returns the Markdown file to delete, if any."""
    op, *rest = fix
    if op == "course":
        doc.set_course(*rest)
    elif op == "module":
        doc.set_module(*rest)
    elif op == "lesson":
        mslug, lslug, field, value = rest
        # The manifest leaves defaults unwritten (kind: lesson, gated: false);
        # `is_published` is always written.
        if (field, value) in (("kind", "lesson"), ("gated", False)):
            value = None
        doc.set_lesson(mslug, lslug, field, value)
    elif op == "lesson-order":
        doc.reorder_lessons(*rest)
    elif op == "links":
        doc.set_links(*rest)
    elif op == "lesson-delete":
        mslug, lslug, path = rest
        doc.remove_lesson(mslug, lslug)
        return path
    elif op == "lesson-add":
        mslug, row = rest
        module = next(m for m in manifest["modules"] if m["slug"] == mslug)
        doc.add_lesson(mslug, [
            ("title", row["title"], True),
            ("slug", row["slug"], False),
            ("kind", None if row["kind"] == "lesson" else row["kind"], False),
            ("file", f"{lesson_dir(module)}/{row['slug']}.md", False),
            ("estimated_minutes", row.get("estimated_minutes"), False),
            ("is_published", bool(row["is_published"]), False),
            ("gated", True if row["gated"] else None, False),
        ])
    else:
        raise ValueError(f"unknown fix {fix!r}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("content", help="course content directory (holds course.yaml)")
    ap.add_argument("--base-url",
                    default=os.environ.get("MLARENA_BASE_URL", "https://ml-arena.com"))
    ap.add_argument("--module", action="append", help="only this module slug")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be written, write nothing")
    ap.add_argument("--diff", action="store_true", help="print the unified diffs")
    args = ap.parse_args()

    token = os.environ.get("MLARENA_API_KEY")
    if not token:
        raise SystemExit("MLARENA_API_KEY is not set (needs a mlk_teacher_... key)")

    content = Path(args.content)
    yaml_path = content / "course.yaml"
    source = yaml_path.read_text()
    manifest = yaml.safe_load(source)
    client = mlarena.connect(token, base_url=args.base_url)
    course = client.course(manifest["course"]["slug"])
    scope = args.module
    verb = "would " if args.dry_run else ""

    # ---- course.yaml ------------------------------------------------------ #
    findings = structure(manifest, course, scope)
    todo = [f for f in findings if f.fix is None]
    fixes = sorted((f for f in findings if f.fix), key=lambda f: ORDER.index(f.fix[0]))
    doc = CourseYaml(source)
    stale_files = [path for f in fixes if (path := apply(doc, f.fix, manifest))]
    edited = doc.text()
    pulled = yaml.safe_load(edited)
    missed = [f for f in structure(pulled, course, scope) if f.fix]
    if missed:
        raise SystemExit("course.yaml edits did not land — nothing written:\n"
                         + "\n".join(f"  {f}" for f in missed))

    for f in fixes:
        print(f"{verb}edit course.yaml — {f}")
    if args.diff and edited != source:
        sys.stdout.writelines("    " + line for line in difflib.unified_diff(
            source.splitlines(True), edited.splitlines(True),
            "repo/course.yaml", "pulled/course.yaml", n=1))
    if edited != source and not args.dry_run:
        yaml_path.write_text(edited)

    # ---- lesson bodies ---------------------------------------------------- #
    live = {m["slug"]: {l["slug"]: l for l in m["lessons"]} for m in course["modules"]}
    written: list[tuple[str, int, int]] = []
    fetched: list[tuple[str, str]] = []
    unresolved: list[str] = []
    bodies: dict[str, str] = {}   # live bodies as served, for the baseline
    resolved: dict[str, str] = {}  # served URL -> repo path, once per run

    for module in pulled["modules"]:
        mslug = module["slug"]
        if (scope and mslug not in scope) or mslug not in live:
            continue
        specs = module.get("lessons") or []
        for spec in specs:
            lslug = spec["slug"]
            row = live[mslug].get(lslug)
            if row is None:
                continue
            path = content / spec["file"]
            local_raw = path.read_text() if path.is_file() else ""
            remote = client.get_lesson(row["id"])
            served = (remote.get("lesson", remote).get("body_md") or "").replace("\r\n", "\n")
            bodies[f"{mslug}/{lslug}"] = served

            def back(match: re.Match) -> str:
                url = match.group(2)
                rel = resolved.get(url) or resolve_asset(url, content, local_raw, mslug, lslug)
                if rel is None and MEDIA_RE.search(url):
                    siblings = [(content / s["file"]).read_text() for s in specs
                                if s is not spec and (content / s["file"]).is_file()]
                    name = unquote(MEDIA_RE.search(url).group(2)).rsplit("/", 1)[-1]
                    rel = f"{asset_dir([local_raw, *siblings], mslug, lslug)}/{name}"
                    fetched.append((f"{mslug}/{lslug}", rel))
                    if not args.dry_run:
                        download(client, url, content / rel)
                if rel is None:
                    unresolved.append(f"{mslug}/{lslug}: {url}")
                    return match.group(0)
                resolved[url] = rel
                return match.group(1) + rel + match.group(3)

            body = SERVED_IMAGE_RE.sub(back, served).rstrip("\n") + "\n"
            if normalise(local_raw) == normalise(body):
                continue
            delta = list(difflib.ndiff(normalise(local_raw), normalise(body)))
            written.append((f"{mslug}/{lslug}",
                            sum(1 for line in delta if line.startswith("+ ")),
                            sum(1 for line in delta if line.startswith("- "))))
            if args.diff:
                for line in difflib.unified_diff(
                        normalise(local_raw), normalise(body),
                        fromfile=f"repo/{lslug}", tofile=f"live/{lslug}",
                        lineterm="", n=2):
                    print("    " + line)
            if not args.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)

    for name, added, removed in written:
        print(f"{verb}rewrite {name} — +{added}/-{removed} lines from live")
    for name, rel in fetched:
        print(f"{verb}download {rel} — uploaded on the website to {name}")
    for rel in stale_files:
        print(f"{verb}delete {rel} — its lesson was deleted on the website")
        if not args.dry_run and (content / rel).is_file():
            (content / rel).unlink()
    if not (fixes or written or fetched):
        print("the repo already matches the live course")

    if unresolved:
        print(f"\n{len(unresolved)} image(s) left as served URLs — not a lesson "
              f"upload, so there is nothing to download:")
        for line in unresolved:
            print(f"    {line}")
        print("    the deck build needs a repo-relative path (`make slides` runs with "
              "--strict-assets); add the file or fix the reference by hand.")

    if todo:
        print(f"\n{len(todo)} change(s) this tool does not make — restructure "
              f"course.yaml by hand:")
        for finding in todo:
            print(f"  {finding}")

    # ---- baseline --------------------------------------------------------- #
    if args.dry_run:
        return 0
    full_run = scope is None
    plan = build_plan(str(content), pulled["course"],
                      [m for m in pulled["modules"] if full_run or m["slug"] in scope],
                      publish_all=False, full_run=full_run)
    state = load_state(str(content), args.base_url)
    state["published"] = lesson_sync.adopt(
        state["published"], lesson_sync.live_snapshot(course, bodies), plan,
        modules=scope, course_meta=full_run, module_order=full_run)
    # The id map, so the publisher updates these lessons in place and knows a
    # lesson deleted on the website from one never published.
    for module in course["modules"]:
        if full_run or module["slug"] in scope:
            state["modules"][module["slug"]] = module["module_id"]
            for key in [k for k in state["lessons"] if k.split("/", 1)[0] == module["slug"]]:
                del state["lessons"][key]
            for lesson in module["lessons"]:
                state["lessons"][f"{module['slug']}/{lesson['slug']}"] = lesson["id"]
    save_state(str(content), args.base_url, state)
    print(f"\nbaselined {len(bodies)} lesson(s) — the next `make publish` can tell a "
          f"website edit from a local one.")
    if written or fetched:
        print("Rebuild what reads these files: `make check-slides` then `make slides`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
