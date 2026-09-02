#!/usr/bin/env python3
"""Sync the Markdown courseware into an ML-Arena course.

``mlarena.author_course_from_dir`` is create-only: re-running it duplicates
every module and lesson. Teaching material is edited constantly, so this script
does an **idempotent sync** instead — create on first run, update in place after
— while staying a pure client-side composition of the public SDK methods (no
new endpoint, per the frontend↔SDK parity rule).

Identity is by slug, recorded in ``.mlarena-state.json`` next to ``course.yaml``:

    { "https://ml-arena.com": { "course_id": 31,
                                "modules": {"s1-git-and-packaging": 12},
                                "lessons": {"s1-git-and-packaging/why-git": 88} } }

Commit that file — it is what makes a second run an update rather than a
duplicate. If it is lost, the script re-resolves modules by slug from the server
and only the lesson→id map has to be rebuilt (also by slug).

Token scopes (see mlarena-sdk/PROCESS.md):
  * creating the course itself works with **any** scope and flips the account to
    teacher;
  * authoring modules/lessons requires a **``mlk_teacher_…``** token.

Usage::

    export MLARENA_API_KEY=mlk_teacher_...
    python tools/publish_mlarena.py content/python-ai-engineering --dry-run
    python tools/publish_mlarena.py content/python-ai-engineering
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import yaml

STATE_FILENAME = ".mlarena-state.json"

# ![alt](path) where path is a local, relative file — remote URLs and already
# uploaded media URLs are left alone.
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://|/)([^)\s]+)\)")

COURSE_META_FIELDS = {"name", "code", "description", "visibility",
                      "instructor_name", "start_date", "end_date"}
MODULE_META_FIELDS = {"title", "summary", "icon", "visibility"}


# --------------------------------------------------------------------------- #
# SDK loading
# --------------------------------------------------------------------------- #


def load_sdk(sdk_path: str | None):
    """Import `mlarena`, optionally from a local checkout.

    Fails loud with an actionable message rather than degrading to raw HTTP —
    the SDK is the contract with the backend, and hand-rolled requests would
    drift from it.
    """
    if sdk_path:
        resolved = os.path.abspath(os.path.expanduser(sdk_path))
        if not os.path.isdir(os.path.join(resolved, "mlarena")):
            raise SystemExit(f"--sdk-path {resolved!r} does not contain a 'mlarena' package")
        sys.path.insert(0, resolved)
    try:
        import mlarena
    except ImportError as exc:
        raise SystemExit(
            "the 'mlarena' SDK is not importable.\n"
            "  pip install -e ../../mlarena-sdk\n"
            "  or pass --sdk-path /path/to/mlarena-sdk"
        ) from exc
    return mlarena


# --------------------------------------------------------------------------- #
# Manifest + state
# --------------------------------------------------------------------------- #


def load_manifest(base: str) -> dict:
    for name in ("course.yaml", "course.yml", "course.json"):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh) if name.endswith(".json") else yaml.safe_load(fh)
    raise SystemExit(f"no course.yaml in {base}")


def load_state(base: str, base_url: str) -> dict:
    path = os.path.join(base, STATE_FILENAME)
    if not os.path.isfile(path):
        return {"course_id": None, "modules": {}, "lessons": {}}
    with open(path, encoding="utf-8") as fh:
        all_state = json.load(fh)
    entry = all_state.get(base_url) or {}
    return {
        "course_id": entry.get("course_id"),
        "modules": entry.get("modules") or {},
        "lessons": entry.get("lessons") or {},
    }


def save_state(base: str, base_url: str, state: dict) -> None:
    path = os.path.join(base, STATE_FILENAME)
    all_state = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            all_state = json.load(fh)
    all_state[base_url] = state
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_state, fh, indent=2, sort_keys=True)
        fh.write("\n")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def read_body(base: str, lesson: dict) -> str:
    if lesson.get("file"):
        path = os.path.join(base, lesson["file"])
        if not os.path.isfile(path):
            raise SystemExit(f"lesson file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    if "body_md" in lesson:
        return lesson["body_md"]
    raise SystemExit(f"lesson {lesson.get('title')!r} has neither 'file' nor 'body_md'")


def meta(spec: dict, allowed: set) -> dict:
    return {k: v for k, v in spec.items() if k in allowed and v is not None}


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #


class Syncer:
    def __init__(self, client, base: str, state: dict, dry_run: bool, publish: bool,
                 skip_media: bool = False):
        self.c = client
        self.base = base
        self.state = state
        self.dry_run = dry_run
        self.publish = publish
        self.skip_media = skip_media
        self.skipped_media: list[str] = []
        self.actions: list[str] = []

    def log(self, verb: str, what: str) -> None:
        prefix = "would " if self.dry_run else ""
        line = f"  {prefix}{verb:<7} {what}"
        self.actions.append(line)
        print(line)

    # -- course ------------------------------------------------------------ #
    def sync_course(self, spec: dict) -> int:
        course_id = spec.get("course_id") or self.state.get("course_id")
        fields = meta(spec, COURSE_META_FIELDS)

        if course_id:
            self.log("update", f"course #{course_id} ({fields.get('name')})")
            if not self.dry_run:
                self.c.update_course(course_id, **fields)
            return course_id

        self.log("create", f"course {fields.get('name')!r}")
        if self.dry_run:
            return -1
        created = self.c.create_course(
            name=spec["name"],
            code=spec.get("code"),
            start_date=spec.get("start_date"),
            end_date=spec.get("end_date"),
            instructor_name=spec.get("instructor_name"),
            slug=spec.get("slug"),
            description=spec.get("description"),
            visibility=spec.get("visibility"),
        )
        print(f"    enrollment_link: {created.get('enrollment_link')}")
        print(f"    join_code:       {created.get('join_code')}")
        return created["id"]

    # -- modules ----------------------------------------------------------- #
    def resolve_module(self, spec: dict) -> tuple[int, dict | None]:
        """Return (module_id, server_detail). Creates the module if unknown."""
        slug = spec.get("slug") or slugify(spec["title"])
        module_id = spec.get("module_id") or self.state["modules"].get(slug)

        if module_id is None and not self.dry_run:
            # State file lost or first run: look the slug up on the server
            # before creating, so a re-run never duplicates.
            for m in self.c.list_modules():
                if m.get("slug") == slug:
                    module_id = m["id"]
                    break

        if module_id is None:
            self.log("create", f"module {slug}")
            if self.dry_run:
                return -1, None
            created = self.c.create_module(
                title=spec["title"], slug=slug, summary=spec.get("summary"),
                icon=spec.get("icon"), visibility=spec.get("visibility", "private"),
            )
            module_id = created["id"]
            self.state["modules"][slug] = module_id
            return module_id, created

        self.log("update", f"module {slug} (#{module_id})")
        self.state["modules"][slug] = module_id
        if self.dry_run:
            return module_id, None
        fields = meta(spec, MODULE_META_FIELDS)
        if fields:
            self.c.update_module(module_id, **fields)
        return module_id, self.c.get_module(module_id)

    # -- lessons ----------------------------------------------------------- #
    def sync_lessons(self, module_id: int, module_slug: str, detail: dict | None,
                     specs: list) -> None:
        existing = {}
        if detail:
            for lsn in detail.get("lessons") or []:
                existing[lsn["slug"]] = lsn["id"]

        ordered: list[int] = []
        for spec in specs:
            slug = spec.get("slug") or slugify(spec["title"])
            key = f"{module_slug}/{slug}"
            body = read_body(self.base, spec)
            lesson_id = existing.get(slug) or self.state["lessons"].get(key)

            if lesson_id is None:
                self.log("create", f"lesson {key} ({len(body)} chars)")
                if self.dry_run:
                    self.upload_media(-1, body)
                    continue
                lesson = self.c.create_lesson(
                    module_id, title=spec["title"], kind=spec.get("kind", "lesson"),
                    slug=slug, body_md=body, gated=bool(spec.get("gated", False)),
                )
                lesson_id = lesson["id"]
                self.state["lessons"][key] = lesson_id
            else:
                self.log("update", f"lesson {key} ({len(body)} chars)")
                if self.dry_run:
                    self.upload_media(-1, body)
                    continue

            # Media upload needs the lesson id, so the body is written only once
            # the local image paths have been rewritten to served URLs.
            body = self.upload_media(lesson_id, body)

            updates: dict = {
                "title": spec["title"],
                "body_md": body,
                "gated": bool(spec.get("gated", False)),
            }
            if spec.get("estimated_minutes") is not None:
                updates["estimated_minutes"] = spec["estimated_minutes"]
            updates["is_published"] = bool(spec.get("is_published", False) or self.publish)
            self.c.update_lesson(lesson_id, **updates)

            self.state["lessons"][key] = lesson_id
            ordered.append(lesson_id)

        if len(ordered) > 1 and not self.dry_run:
            self.c.reorder_lessons(module_id, ordered)

    # -- media ------------------------------------------------------------- #
    def upload_media(self, lesson_id: int, body: str) -> str:
        """Upload every locally-referenced image and rewrite the body to the
        served URLs.

        Markdown keeps repo-relative paths (so the PPTX build and a local
        preview both work); ML-Arena needs its own URLs. Rewriting happens here
        rather than in the source files so the two outputs stay in sync.
        """
        replacements: dict[str, str] = {}
        for _alt, rel in LOCAL_IMAGE_RE.findall(body):
            if rel in replacements:
                continue
            path = os.path.join(self.base, rel)
            if not os.path.isfile(path):
                raise SystemExit(
                    f"lesson image not found: {path}\n"
                    f"  referenced as ![...]({rel})"
                )
            if self.dry_run:
                replacements[rel] = rel
                self.log("upload", f"media {rel}")
                continue
            if self.skip_media:
                # Leave the relative path in the body untouched: it renders as a
                # broken image for now, but a re-run once the server-side upload
                # works will find it and rewrite it. Rewriting to a placeholder
                # would make the reference unrecoverable.
                self.skipped_media.append(rel)
                self.log("skip", f"media {rel}")
                continue
            result = self.c.upload_lesson_media(lesson_id, path)
            replacements[rel] = result["url"]

        for rel, url in replacements.items():
            body = body.replace(f"]({rel})", f"]({url})")
        return body

    # -- competitions ------------------------------------------------------ #
    def sync_competitions(self, module_id: int, detail: dict | None, specs: list) -> None:
        attached = set()
        if detail:
            attached = {c["competition_id"] for c in (detail.get("competitions") or [])}
        for comp in specs:
            cid = comp["competition_id"]
            if cid in attached:
                continue
            self.log("attach", f"competition #{cid} -> module #{module_id}")
            if not self.dry_run:
                self.c.attach_competition(module_id, cid, label=comp.get("label"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("course_dir", help="directory containing course.yaml")
    ap.add_argument("--base-url", default=os.environ.get("MLARENA_BASE_URL",
                                                         "https://ml-arena.com"))
    ap.add_argument("--api-key", default=os.environ.get("MLARENA_API_KEY"),
                    help="defaults to $MLARENA_API_KEY")
    ap.add_argument("--sdk-path", default=os.environ.get("MLARENA_SDK_PATH"),
                    help="local mlarena-sdk checkout to import from")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without touching the server")
    ap.add_argument("--publish", action="store_true",
                    help="force is_published=True on every lesson")
    ap.add_argument("--module", action="append",
                    help="sync only this module slug (repeatable)")
    ap.add_argument("--skip-media", action="store_true",
                    help="publish lesson text without uploading images. The "
                         "relative paths stay in the body, so a later run "
                         "uploads and rewrites them.")
    args = ap.parse_args()

    if not args.api_key and not args.dry_run:
        raise SystemExit("no API key: set $MLARENA_API_KEY or pass --api-key")

    scope = args.api_key.split("_")[1] if args.api_key and args.api_key.count("_") >= 3 else None
    if scope and scope != "teacher":
        print(
            f"warning: token scope is '{scope}'. Creating the course works with any\n"
            f"         scope, but authoring modules/lessons requires a 'mlk_teacher_…'\n"
            f"         token (mint one on your ML-Arena Profile page). Expect a 403\n"
            f"         on the first /api/teacher/* call.",
            file=sys.stderr,
        )

    base = os.path.abspath(args.course_dir)
    manifest = load_manifest(base)
    course_spec = dict(manifest.get("course") or {})
    modules = manifest.get("modules") or []
    if args.module:
        wanted = set(args.module)
        modules = [m for m in modules
                   if (m.get("slug") or slugify(m["title"])) in wanted]
        if not modules:
            raise SystemExit(f"no module matched {sorted(wanted)}")

    state = load_state(base, args.base_url)

    client = None
    if not args.dry_run:
        mlarena = load_sdk(args.sdk_path)
        client = mlarena.connect(api_key=args.api_key, base_url=args.base_url)

    print(f"ML-Arena sync -> {args.base_url}")
    syncer = Syncer(client, base, state, args.dry_run, args.publish,
                    skip_media=args.skip_media)

    def checkpoint() -> None:
        """Persist the id map so a failure part-way never orphans what was
        already created — re-running must update, not duplicate."""
        if not args.dry_run:
            save_state(base, args.base_url, state)

    try:
        course_id = syncer.sync_course(course_spec)
        if course_id > 0:
            state["course_id"] = course_id
        checkpoint()

        cover = course_spec.get("cover")
        if cover and not args.dry_run:
            syncer.log("upload", f"cover {cover}")
            client.set_course_cover(course_id, os.path.join(base, cover))

        # Which modules the course already carries. link_module rejects a
        # duplicate link with an error, so a second run would die on the first
        # module without this — breaking the idempotency this script exists to
        # provide. Checking beforehand is preferable to catching the error,
        # which would also swallow genuine link failures.
        already_linked: set[int] = set()
        if not args.dry_run:
            already_linked = {m["module_id"]
                              for m in client.list_course_modules(course_id)}

        ordered_modules: list[int] = []
        for position, spec in enumerate(modules):
            module_slug = spec.get("slug") or slugify(spec["title"])
            module_id, detail = syncer.resolve_module(spec)
            checkpoint()
            syncer.sync_lessons(module_id, module_slug, detail,
                                spec.get("lessons") or [])
            syncer.sync_competitions(module_id, detail,
                                     spec.get("competitions") or [])
            checkpoint()

            if not args.dry_run:
                if module_id not in already_linked:
                    syncer.log("link", f"module {module_slug} -> course #{course_id}")
                    client.link_module(course_id, module_id, position=position)
                ordered_modules.append(module_id)

        if len(ordered_modules) > 1:
            client.reorder_modules(course_id, ordered_modules)
    except Exception as exc:
        checkpoint()
        print(f"\nsync stopped: {exc}", file=sys.stderr)
        if not args.dry_run:
            print(f"state saved to {os.path.join(base, STATE_FILENAME)} — "
                  f"re-running resumes rather than duplicating.", file=sys.stderr)
        raise

    if not args.dry_run:
        checkpoint()
        print(f"\nstate written to {os.path.join(base, STATE_FILENAME)} — commit it.")
        print(f"course: {args.base_url}/courses/{course_spec.get('slug')}")
        if syncer.skipped_media:
            unique = sorted(set(syncer.skipped_media))
            print(f"\n{len(unique)} image(s) NOT uploaded (--skip-media):")
            for rel in unique:
                print(f"  {rel}")
            print("Their markdown references are unchanged, so re-running "
                  "without --skip-media will upload and rewrite them.")
    else:
        print(f"\n{len(syncer.actions)} action(s) planned. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
