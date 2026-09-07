#!/usr/bin/env python3
"""Tests for the publish guard.

    uv run --with pytest pytest courseware/tools/test_lesson_sync.py -v

No network and no SDK: every case is three dicts and the walk over them. What
is covered, and why each one exists:

* a publish of local edits is **not** blocked — the failure mode that would
  make the guard get switched off within a week;
* an edit made on the website **is** blocked, body and metadata alike, because
  the metadata case is the one that actually cost a lesson title (COURSE_STATE
  §1d);
* with no baseline the check degrades to repo-vs-live and says so, rather than
  guessing which side is newer;
* a trailing newline is not an edit — YAML block scalars carry one and the
  backend strips it, and a guard that cried wolf on all five course-level
  fields on every run would also get switched off;
* `absorb` is scoped, so a one-module publish cannot silently baseline a
  website edit to a module it never looked at.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lesson_sync import absorb, digest, drift, empty_snapshot, live_snapshot  # noqa: E402

BODY = "# Git essentials\n\nA commit is a snapshot.\n"
EDITED = "# Git essentials\n\nA commit is a snapshot of the whole tree.\n"


def snapshot(title="Git Essentials", body=BODY, lessons=("git-essentials",)):
    """A snapshot with one module and one lesson, in the shape all three sides
    share."""
    snap = empty_snapshot()
    snap["modules"]["s1"] = {"title": "Session 1", "lessons": list(lessons)}
    snap["lessons"]["s1/git-essentials"] = {
        "title": title, "is_published": True, "gated": False, "body": digest(body),
    }
    return snap


# --------------------------------------------------------------------------- #
# The server still holds what we published: publishing is safe
# --------------------------------------------------------------------------- #


def test_local_edits_do_not_block_a_publish():
    """The whole point. repo != live is what a publish is for; only
    live != baseline means someone else moved."""
    baseline = snapshot()
    live = snapshot()
    plan = snapshot(title="Git, Essentially", body=EDITED)
    assert drift(baseline, live, plan) == []


def test_a_new_lesson_has_nothing_to_lose():
    plan = snapshot()
    plan["lessons"]["s1/new-lesson"] = {"title": "New", "body": digest("x")}
    assert drift(snapshot(), snapshot(), plan) == []


def test_a_module_not_yet_on_the_server_is_skipped():
    plan = snapshot()
    plan["modules"]["s5"] = {"title": "Session 5", "lessons": []}
    assert drift(snapshot(), snapshot(), plan) == []


# --------------------------------------------------------------------------- #
# The website moved: publishing destroys it
# --------------------------------------------------------------------------- #


def test_a_body_edited_on_the_website_is_caught():
    found = drift(snapshot(), snapshot(body=EDITED), snapshot())
    assert [(d.where, d.what, d.verified) for d in found] == [
        ("s1/git-essentials", "body", True)]


def test_a_title_edited_on_the_website_is_caught():
    """COURSE_STATE §1d: the edit that was actually lost was a title, not a
    body, so a body-only guard would not have saved it."""
    found = drift(snapshot(), snapshot(title="Accounts & Setup"), snapshot())
    assert [(d.what, d.expected, d.live) for d in found] == [
        ("title", "Git Essentials", "Accounts & Setup")]


def test_a_lesson_unpublished_on_the_website_is_caught():
    live = snapshot()
    live["lessons"]["s1/git-essentials"]["is_published"] = False
    assert [d.what for d in drift(snapshot(), live, snapshot())] == ["is_published"]


def test_a_reorder_on_the_website_is_caught():
    order = ("git-essentials", "packaging")
    baseline, live, plan = (snapshot(lessons=order), snapshot(lessons=order[::-1]),
                            snapshot(lessons=order))
    assert [d.what for d in drift(baseline, live, plan)] == ["lesson order"]


def test_a_lesson_added_on_the_website_is_caught():
    """A publish does not delete it, but the reorder that follows moves it —
    which is why the order drifts as well, and both are reported."""
    live = snapshot(lessons=("git-essentials", "someone-elses-lesson"))
    assert [d.what for d in drift(snapshot(), live, snapshot())] == [
        "lesson order", "extra lesson"]


def test_a_lesson_deleted_on_the_website_is_caught():
    """It is not overwritten, it is *recreated* — COURSE_STATE §1g had to undo
    exactly that by hand. Only the id map tells this apart from a lesson that
    has never been published."""
    live = empty_snapshot()
    live["modules"]["s1"] = {"title": "Session 1", "lessons": []}
    found = drift(snapshot(), live, snapshot(),
                  known_ids={"s1/git-essentials": 29})
    assert [(d.what, d.expected) for d in found] == [
        ("lesson order", "git-essentials"), ("deleted", 29)]
    assert "#29" in str(found[1])


def test_a_lesson_never_published_is_not_a_deletion():
    """Same live course, no recorded id: the lesson is new, not deleted, and
    the run creates it."""
    live = empty_snapshot()
    live["modules"]["s1"] = {"title": "Session 1", "lessons": []}
    found = drift(snapshot(), live, snapshot(), known_ids={})
    assert [d.what for d in found] == ["lesson order"]


def test_only_fields_the_publisher_sends_are_guarded():
    """`kind` is set at creation and never updated, so a publish cannot destroy
    a change to it and the guard must not refuse over one."""
    baseline, live, plan = snapshot(), snapshot(), snapshot()
    live["lessons"]["s1/git-essentials"]["kind"] = "lab"
    assert drift(baseline, live, plan) == []


# --------------------------------------------------------------------------- #
# No baseline: the check degrades to repo-vs-live, and says so
# --------------------------------------------------------------------------- #


def test_no_baseline_and_agreement_clears():
    assert drift(empty_snapshot(), snapshot(), snapshot()) == []


def test_no_baseline_and_disagreement_is_reported_as_unverified():
    found = drift(empty_snapshot(), snapshot(body=EDITED), snapshot())
    assert len(found) == 1 and found[0].verified is False
    assert "no baseline" in str(found[0])


def test_a_trailing_newline_is_not_an_edit():
    """A YAML block scalar keeps the newline the backend strips."""
    live, plan = empty_snapshot(), empty_snapshot()
    live["course"] = {"description": "Four sessions."}
    plan["course"] = {"description": "Four sessions.\n"}
    assert drift(empty_snapshot(), live, plan) == []


def test_an_image_rewritten_to_a_served_url_is_not_an_edit():
    """The publisher uploads and rewrites paths on the way up; both sides are
    compared by basename, or every lesson with a figure would drift."""
    assert (digest("![](assets/s1/git/three-trees.png)")
            == digest("![](/api/academic_courses/assets/lessons/29/three-trees.png)"))


# --------------------------------------------------------------------------- #
# Recording the baseline
# --------------------------------------------------------------------------- #


def test_absorb_leaves_untouched_modules_alone():
    """The one that keeps a partial publish honest: baselining the live state
    of a module this run never checked would accept a website edit to it and
    disarm the guard for the next run."""
    baseline = snapshot()
    baseline["modules"]["s2"] = {"title": "Session 2", "lessons": ["losses"]}
    baseline["lessons"]["s2/losses"] = {"title": "Losses", "body": digest(BODY)}

    fresh = snapshot(title="Git, Essentially")
    fresh["modules"]["s2"] = {"title": "Renamed On The Website", "lessons": ["losses"]}
    fresh["lessons"]["s2/losses"] = {"title": "Renamed", "body": digest(EDITED)}

    out = absorb(baseline, fresh, modules=["s1"], course_meta=True, module_order=False)
    assert out["lessons"]["s1/git-essentials"]["title"] == "Git, Essentially"
    assert out["modules"]["s2"]["title"] == "Session 2"
    assert out["lessons"]["s2/losses"]["body"] == digest(BODY)

    # ... and the next run still sees the untouched module's drift: its title,
    # and the body under it.
    assert [(d.where, d.what) for d in drift(out, fresh, fresh)] == [
        ("module s2", "title"), ("s2/losses", "title"), ("s2/losses", "body")]


def test_absorb_keeps_the_digest_of_a_body_this_run_did_not_send():
    """A metadata-only re-read carries no bodies; dropping the recorded digest
    would leave the next run with no evidence the live body is ours."""
    fresh = live_snapshot(
        {"modules": [{"slug": "s1", "title": "Session 1",
                      "lessons": [{"slug": "git-essentials", "title": "Git Essentials"}]}]},
        bodies={})
    out = absorb(snapshot(), fresh, modules=None, course_meta=False, module_order=False)
    assert out["lessons"]["s1/git-essentials"]["body"] == digest(BODY)


def test_live_snapshot_reads_a_real_payload_shape():
    """The fields are the ones `client.course(slug)` actually returns."""
    snap = live_snapshot({
        "name": "MS2A - AI Engineering", "visibility": "public",
        "modules": [{"slug": "s1", "title": "Session 1", "summary": "Shell, git.",
                     "icon": "git-branch", "is_published": True,
                     "lessons": [{"slug": "git-essentials", "id": 29, "kind": "lesson",
                                  "title": "Git Essentials", "is_published": True,
                                  "gated": False, "estimated_minutes": 30}]}],
    }, bodies={"s1/git-essentials": BODY})
    assert snap["course"]["name"] == "MS2A - AI Engineering"
    assert snap["module_order"] == ["s1"]
    assert snap["modules"]["s1"]["icon"] == "git-branch"
    assert "is_published" not in snap["modules"]["s1"]  # written as `visibility`
    assert snap["lessons"]["s1/git-essentials"]["body"] == digest(BODY)
