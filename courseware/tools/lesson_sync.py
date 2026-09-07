#!/usr/bin/env python3
"""What "the live course still holds what we published" means, in one place.

`publish_mlarena.py` overwrites the server — bodies, titles, flags, lesson
order. That is safe exactly when the server still holds what this repo last put
there, and unsafe the moment someone edits the course in the ML-Arena editor:
the edit is destroyed silently and is not recoverable from this repo. It has
happened (COURSE_STATE §1d — a lesson renamed on the website, put back by a
full-course run that was not checked first).

`check_sync.py` compares **repo against live**, which cannot tell the two
directions apart: a lesson you edited here and a lesson someone edited there
look identical to it. So it cannot gate a publish on its own — every publish
worth running has repo-vs-live differences, that is what it is for.

The publisher therefore records what it published — the **baseline**, under
`"published"` in `.mlarena-state.json` — and the guard is a three-way compare:

    baseline   what the server held right after our last publish
    live       what it holds now
    plan       what this run is about to send

    live != baseline  ->  the website changed under us. Publishing destroys
                          that change. Refuse.
    live == baseline  ->  the server holds our own copy; overwriting it can
                          lose nothing, whatever the plan says.

With no baseline for an item (first guarded run, or the state file was lost)
there is nothing to compare against, so the check falls back to `live` vs
`plan` — check_sync's comparison, with check_sync's blind spot. Equal means
nothing can be lost and the item is cleared; different is genuinely ambiguous
and is reported as such rather than guessed at.

Only what the publisher writes is guarded. `kind` is set when a lesson is
created and never updated, so a `kind` changed on the website survives a
publish and is not this module's business.

Baseline metadata is recorded **as the server reported it**, never as we sent
it, so a field the backend renames or normalises on the way in (module
`visibility` arrives back as `is_published`) cannot surface as drift on the way
out. Bodies are the exception — the digest is of what we sent, because the
server is not re-read after the write; `check_sync` is what verifies that
assumption end to end, and does.
"""
from __future__ import annotations

import hashlib
import os
import re

IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")

# Fields of the live payload the publisher overwrites, and can therefore
# destroy. Module `visibility` and lesson `kind` are deliberately absent: the
# first is not readable back in the form it is written, the second is never
# updated. `estimated_minutes` is guarded only when the manifest declares one,
# because that is the only case in which the publisher sends it.
COURSE_FIELDS = ("name", "code", "description", "visibility",
                 "instructor_name", "start_date", "end_date")
MODULE_FIELDS = ("title", "summary", "icon")
LESSON_FIELDS = ("title", "is_published", "gated", "estimated_minutes")


def normalise(body: str) -> list[str]:
    """Reduce a body to what the two sides should agree on.

    The markdown on disk keeps repo-relative image paths so the PPTX build
    works, and the publisher rewrites them to served URLs on the way up — so
    both sides are reduced to the image's basename. Trailing whitespace is
    ignored; everything else, `<!-- notes: -->` comments included, is literal.
    """
    body = IMAGE_RE.sub(lambda m: m.group(1) + os.path.basename(m.group(2)) + m.group(3),
                        body)
    return [line.rstrip() for line in body.replace("\r\n", "\n").strip().split("\n")]


def digest(body: str) -> str:
    """A body's identity, normalised — small enough to commit, exact enough to
    prove the server was not touched."""
    return hashlib.sha256("\n".join(normalise(body)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Snapshots
#
# baseline, live and plan are all the same shape, so comparing them is one walk
# rather than three special cases:
#
#   {"course":  {field: value},
#    "modules": {slug: {field: value, "lessons": [lesson-slug, ...]}},
#    "lessons": {"module/lesson": {field: value, "body": digest}},
#    "module_order": [module-slug, ...]}
# --------------------------------------------------------------------------- #


def empty_snapshot() -> dict:
    return {"course": {}, "modules": {}, "lessons": {}, "module_order": []}


def live_snapshot(course: dict, bodies: dict[str, str]) -> dict:
    """The server as it is now, from one `client.course(slug)` payload plus
    whatever bodies were fetched (`{"module/lesson": body_md}`)."""
    snap = empty_snapshot()
    snap["course"] = {f: course.get(f) for f in COURSE_FIELDS}
    snap["module_order"] = [m["slug"] for m in course.get("modules") or []]
    for module in course.get("modules") or []:
        mslug = module["slug"]
        lessons = module.get("lessons") or []
        snap["modules"][mslug] = {f: module.get(f) for f in MODULE_FIELDS}
        snap["modules"][mslug]["lessons"] = [l["slug"] for l in lessons]
        for lesson in lessons:
            key = f"{mslug}/{lesson['slug']}"
            entry = {f: lesson.get(f) for f in LESSON_FIELDS}
            if key in bodies:
                entry["body"] = digest(bodies[key])
            snap["lessons"][key] = entry
    return snap


def absorb(baseline: dict, fresh: dict, modules: list[str] | None,
           course_meta: bool, module_order: bool) -> dict:
    """Fold a snapshot of what we just wrote into the recorded baseline.

    Scoped, and that is the whole point: a `--module s1` run leaves every other
    module's baseline exactly as it was. Recording the live state of a module
    this run never looked at would silently accept a website edit to it and
    disarm the guard for the next run — the failure this module exists to stop.
    """
    out = {
        "course": dict(baseline.get("course") or {}),
        "modules": dict(baseline.get("modules") or {}),
        "lessons": dict(baseline.get("lessons") or {}),
        "module_order": list(baseline.get("module_order") or []),
    }
    if course_meta:
        out["course"] = dict(fresh.get("course") or {})
    if module_order:
        out["module_order"] = list(fresh.get("module_order") or [])
    for mslug, entry in (fresh.get("modules") or {}).items():
        if modules is not None and mslug not in modules:
            continue
        out["modules"][mslug] = dict(entry)
    for key, entry in (fresh.get("lessons") or {}).items():
        if modules is not None and key.split("/", 1)[0] not in modules:
            continue
        # A run that did not send a body must not drop the digest of the one it
        # sent last time — that record is the only evidence the body on the
        # server is ours.
        if "body" not in entry:
            previous = (baseline.get("lessons") or {}).get(key) or {}
            if "body" in previous:
                entry = dict(entry, body=previous["body"])
        out["lessons"][key] = dict(entry)
    return out


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #


def _brief(value) -> str:
    """A field value on one readable line. Course descriptions and module
    summaries are paragraphs; printed raw they bury the finding."""
    if value is None:
        return "(unset)"
    text = " ".join(str(value).split())
    return text if len(text) <= 88 else text[:87] + "…"


def _loose(a, b) -> bool:
    """Equal once whitespace stops counting.

    Only used against the plan, never against the baseline. The plan's text
    comes from a YAML block scalar, which keeps the trailing newline the
    backend strips on the way in — a difference that is not an edit and that
    the baseline comparison never sees, both of its sides having come from the
    server.
    """
    if isinstance(a, str) and isinstance(b, str):
        return ("\n".join(l.rstrip() for l in a.strip().splitlines())
                == "\n".join(l.rstrip() for l in b.strip().splitlines()))
    return a == b


class Drift:
    """One thing the publish would overwrite that this repo did not write."""

    def __init__(self, where: str, what: str, live, expected, verified: bool):
        self.where = where
        self.what = what
        self.live = live
        self.expected = expected
        self.verified = verified  # False: no baseline, so live was compared to plan

    def __str__(self) -> str:
        held = "published" if self.verified else "repo"
        tag = "" if self.verified else "   (no baseline — either side could be the newer)"
        if self.what == "body":
            return (f"  {self.where}\n"
                    f"      body — the live text is not the one this repo {held}{tag}")
        if self.what == "deleted":
            return (f"  {self.where}\n"
                    f"      deleted on the website (was #{self.expected}) — "
                    f"publishing puts it back")
        if self.what == "extra lesson":
            return (f"  {self.where}\n"
                    f"      lesson {self.live!r} exists on the website and not in "
                    f"course.yaml")
        return (f"  {self.where}\n"
                f"      {self.what}{tag}\n"
                f"        {held:<9} {_brief(self.expected)}\n"
                f"        live      {_brief(self.live)}")


def _compare(where: str, field: str, live: dict, base: dict | None, plan: dict,
             out: list[Drift]) -> None:
    """live vs baseline where there is one, live vs plan where there is not."""
    if field not in plan:
        return  # the publisher does not write it, so it cannot destroy it
    if base is not None and field in base:
        if live.get(field) != base[field]:
            out.append(Drift(where, field, live.get(field), base[field], True))
    elif not _loose(live.get(field), plan[field]):
        out.append(Drift(where, field, live.get(field), plan[field], False))


def drift(baseline: dict, live: dict, plan: dict,
          known_ids: dict | None = None) -> list[Drift]:
    """Everything `plan` would overwrite that the server did not get from us.

    Walks the plan, so a module or lesson this run does not touch is not
    reported, and one that does not exist on the server yet has nothing to
    lose and is skipped.

    `known_ids` is the state file's lesson->id map: a key in it that the server
    no longer carries was deleted on the website, which a publish undoes by
    recreating it (COURSE_STATE §1g had to undo that by hand). Without the map
    that case is indistinguishable from a lesson that has never been published.
    """
    found: list[Drift] = []
    base_modules = baseline.get("modules") or {}
    base_lessons = baseline.get("lessons") or {}

    for field in COURSE_FIELDS:
        if field in (plan.get("course") or {}):
            _compare("course", field, live.get("course") or {},
                     baseline.get("course"), plan["course"], found)

    if plan.get("module_order") and live.get("module_order"):
        base_order = baseline.get("module_order")
        want = plan["module_order"]
        have = live["module_order"]
        # A partial run does not reorder the course, so it does not plan one.
        if base_order:
            if have != base_order:
                found.append(Drift("course", "module order", ", ".join(have),
                                   ", ".join(base_order), True))
        elif have != want:
            found.append(Drift("course", "module order", ", ".join(have),
                               ", ".join(want), False))

    for mslug, spec in (plan.get("modules") or {}).items():
        live_module = (live.get("modules") or {}).get(mslug)
        if live_module is None:
            continue  # not on the server yet: the publish creates it
        base_module = base_modules.get(mslug)
        for field in MODULE_FIELDS:
            _compare(f"module {mslug}", field, live_module, base_module, spec, found)

        want, have = spec.get("lessons") or [], live_module.get("lessons") or []
        base_order = (base_module or {}).get("lessons")
        if base_order is not None:
            if have != base_order:
                found.append(Drift(f"module {mslug}", "lesson order",
                                   ", ".join(have), ", ".join(base_order), True))
        elif have != want:
            found.append(Drift(f"module {mslug}", "lesson order",
                               ", ".join(have), ", ".join(want), False))
        # A lesson added on the website is not deleted by a publish, but the
        # reorder that follows it is computed without it, so it moves.
        for extra in have:
            if extra not in want:
                found.append(Drift(f"module {mslug}", "extra lesson", extra, None, True))

    for key, spec in (plan.get("lessons") or {}).items():
        live_lesson = (live.get("lessons") or {}).get(key)
        if live_lesson is None:
            if known_ids and key in known_ids:
                found.append(Drift(key, "deleted", None, known_ids[key], True))
            continue  # otherwise it has never been published: the run creates it
        base_lesson = base_lessons.get(key)
        for field in (*LESSON_FIELDS, "body"):
            _compare(key, field, live_lesson, base_lesson, spec, found)
    return found


def report(found: list[Drift]) -> str:
    """The refusal, and what to do about it."""
    unverified = [d for d in found if not d.verified]
    lines = [
        "",
        f"publish refused — {len(found)} change(s) on the website that this repo "
        f"did not put there.",
        "Publishing overwrites each of them, silently and unrecoverably:",
        "",
    ]
    lines += [str(d) for d in found]
    lines += [
        "",
        "  make check-sync DIFF=1        read them without writing anything",
        "  make pull MODULE=<slug>       bring the lesson bodies back into the repo",
        "  make publish FORCE=1          overwrite them deliberately",
        "",
        "Structural drift — a title, an order, a lesson added on the website — is",
        "not something `pull` writes: it lives in course.yaml, whose comments are",
        "the record of why the course is shaped the way it is. `make pull-dry`",
        "prints those as the edits to make by hand.",
    ]
    if unverified:
        lines += [
            "",
            f"{len(unverified)} of these have no baseline: the state file has no record",
            "of what was last published for them, so the live copy was compared with",
            "the repo instead and either side could be the newer one. Look before you",
            "force. A publish (or a `make pull`) records the baseline, and the next",
            "run tells you which side moved.",
        ]
    return "\n".join(lines)
