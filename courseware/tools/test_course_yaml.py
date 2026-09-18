#!/usr/bin/env python3
"""Tests for `make pull`'s course.yaml editing and baselining.

    uv run --with pytest --with pyyaml --with requests pytest courseware/tools/test_course_yaml.py -v

No network: the live course is a dict in the shape `client.course(slug)` serves.
What is covered, and why each one exists:

* an edit touches only its own line — both real manifests survive a no-op
  byte for byte, and an inline comment survives a rewrite of its line;
* a reorder moves each comment run with the item it sits above, which is the
  reason this is not a ruamel.yaml round-trip;
* the whole pull, structure -> edit -> re-parse, leaves nothing it could fix:
  the end-to-end property `make pull-all` relies on;
* `adopt` baselines only what the repo and the server agree on, so a
  difference the pull merely reported still stops the next publish.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from course_yaml import CourseYaml, render, split_value  # noqa: E402
from lesson_sync import adopt, digest, drift, empty_snapshot  # noqa: E402

CONTENT = Path(__file__).resolve().parent.parent / "content"

MANIFEST = '''\
# header comment
course:
  name: "Course"
  slug: course
  visibility: public
  start_date: "2026-09-14"
  description: |
    Line one.

    Line two.

modules:
  # ---------------------------------------------------------------- session 1
  - title: "Session 1"
    slug: s1
    icon: database
    summary: >
      A summary that
      folds.
    # why 177 is attached
    competitions:
      - competition_id: 177
        label: "Weather"
      - competition_id: 190
        label: "Store Sales"
    lessons:
      - title: "Alpha"     # retitled on the website
        slug: alpha
        file: s1/alpha.md
        estimated_minutes: 20
        is_published: true            # core
      # comment that belongs to beta
      - title: "Beta"
        slug: beta
        kind: exercise
        file: s1/beta.md
        estimated_minutes: 45
        is_published: true
      - title: "Gamma"
        slug: gamma
        file: s1/gamma.md
        is_published: true

  # ---------------------------------------------------------------- session 2
  - title: "Session 2"
    slug: s2
    lessons:
      - title: "Delta"
        slug: delta
        file: s2/delta.md
        is_published: false
'''


def lesson(slug, title, lid, **kw):
    row = {"id": lid, "slug": slug, "title": title, "kind": "lesson",
           "is_published": True, "gated": False, "estimated_minutes": None}
    row.update(kw)
    return row


def live_course():
    """MANIFEST as the server holds it after a round of website edits."""
    return {
        "name": "Course", "code": None, "slug": "course", "visibility": "public",
        "instructor_name": None, "start_date": "2026-09-15",
        "end_date": None, "description": "Line one.\n\nLine two.",
        "modules": [
            {"slug": "s1", "module_id": 19, "title": "Session 1 — Renamed",
             "icon": "database", "summary": "A summary that folds.",
             "challenges": [{"challenge_id": 190, "label": "Sales"},
                            {"challenge_id": 205, "label": "New one"}],
             "lessons": [
                 lesson("beta", "Exercise — Beta", 2, kind="exercise",
                        estimated_minutes=45, is_published=False),
                 lesson("alpha", "Alpha", 1, estimated_minutes=25),
                 lesson("epsilon", "Epsilon", 9, gated=True),
             ]},
            {"slug": "s2", "module_id": 20, "title": "Session 2", "icon": None,
             "summary": None, "challenges": [],
             "lessons": [lesson("delta", "Delta", 4, is_published=False)]},
        ],
    }


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #


def test_render_quotes_whatever_would_parse_as_something_else():
    assert render("lab-1", quote=False) == "lab-1"
    assert render("true", quote=False) == '"true"'
    assert render("2026-09-14", quote=False) == '"2026-09-14"'
    assert render("45", quote=False) == '"45"'
    assert render('Lab — "one"', quote=True) == '"Lab — \\"one\\""'
    assert render(False, quote=True) == "false"


def test_split_value_keeps_the_inline_comment():
    assert split_value(' "A # B"   # note') == (" ", '"A # B"', "   # note")
    assert split_value(" true            # core") == (" ", "true", "            # core")
    assert split_value(" 'it''s'") == (" ", "'it''s'", "")


# --------------------------------------------------------------------------- #
# The editor touches only what it means to
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", sorted(CONTENT.glob("*/course.yaml")), ids=lambda p: p.parent.name)
def test_the_real_manifests_survive_a_noop(path):
    text = path.read_text()
    assert CourseYaml(text).text() == text


def test_a_title_rewrite_keeps_its_comment_and_quotes():
    doc = CourseYaml(MANIFEST)
    doc.set_lesson("s1", "alpha", "title", "Alpha: the first")
    doc.set_lesson("s1", "alpha", "is_published", False)
    assert '      - title: "Alpha: the first"     # retitled on the website' in doc.text()
    assert "        is_published: false            # core" in doc.text()
    assert len(doc.text().split("\n")) == len(MANIFEST.split("\n"))


def test_a_missing_key_is_inserted_among_its_siblings_and_none_removes_it():
    doc = CourseYaml(MANIFEST)
    doc.set_lesson("s1", "gamma", "estimated_minutes", 30)
    doc.set_lesson("s1", "beta", "estimated_minutes", None)
    parsed = yaml.safe_load(doc.text())
    gamma = parsed["modules"][0]["lessons"][2]
    assert gamma["estimated_minutes"] == 30
    assert "estimated_minutes" not in parsed["modules"][0]["lessons"][1]


def test_a_folded_summary_is_rewritten_as_a_folded_block():
    doc = CourseYaml(MANIFEST)
    long = "Where data comes from and what it costs to get it " * 3
    doc.set_module("s1", "summary", long.strip())
    assert "    summary: >\n" in doc.text()
    assert yaml.safe_load(doc.text())["modules"][0]["summary"].strip() == long.strip()
    # The comment after the block is not swallowed by it.
    assert "    # why 177 is attached\n    competitions:" in doc.text()


def test_a_reorder_moves_each_comment_with_its_item():
    doc = CourseYaml(MANIFEST)
    doc.reorder_lessons("s1", ["gamma", "beta", "alpha"])
    text = doc.text()
    assert [l["slug"] for l in yaml.safe_load(text)["modules"][0]["lessons"]] == \
        ["gamma", "beta", "alpha"]
    assert '      # comment that belongs to beta\n      - title: "Beta"' in text
    # The module after it is untouched.
    assert text.split("  # -------------")[2] == MANIFEST.split("  # -------------")[2]


def test_links_are_made_exactly_the_live_list():
    doc = CourseYaml(MANIFEST)
    doc.set_links("s1", [(190, "Sales"), (205, "New one")])
    doc.set_links("s2", [(7, "Created with its key")])
    parsed = yaml.safe_load(doc.text())
    assert parsed["modules"][0]["competitions"] == [
        {"competition_id": 190, "label": "Sales"},
        {"competition_id": 205, "label": "New one"}]
    assert parsed["modules"][1]["competitions"] == [
        {"competition_id": 7, "label": "Created with its key"}]
    assert "    # why 177 is attached" in doc.text()


def test_add_and_remove_a_lesson():
    doc = CourseYaml(MANIFEST)
    doc.remove_lesson("s1", "beta")
    doc.add_lesson("s1", [("title", "Epsilon", True), ("slug", "epsilon", False),
                          ("kind", None, False), ("file", "s1/epsilon.md", False),
                          ("is_published", True, False)])
    lessons = yaml.safe_load(doc.text())["modules"][0]["lessons"]
    assert [l["slug"] for l in lessons] == ["alpha", "gamma", "epsilon"]
    assert lessons[-1] == {"title": "Epsilon", "slug": "epsilon",
                           "file": "s1/epsilon.md", "is_published": True}


# --------------------------------------------------------------------------- #
# The pull, end to end, without the network
# --------------------------------------------------------------------------- #


def test_a_pull_leaves_nothing_it_could_fix():
    from check_sync import structure
    from pull_mlarena import ORDER, apply

    manifest = yaml.safe_load(MANIFEST)
    course = live_course()
    findings = structure(manifest, course, None)
    kinds = {f.fix[0] for f in findings if f.fix}
    assert kinds == {"course", "module", "links", "lesson", "lesson-order",
                     "lesson-add", "lesson-delete"}

    doc = CourseYaml(MANIFEST)
    stale = [p for f in sorted((f for f in findings if f.fix),
                               key=lambda f: ORDER.index(f.fix[0]))
             if (p := apply(doc, f.fix, manifest))]
    pulled = yaml.safe_load(doc.text())
    assert [f for f in structure(pulled, course, None) if f.fix] == []
    assert stale == ["s1/gamma.md"]
    epsilon = pulled["modules"][0]["lessons"][2]
    assert epsilon == {"title": "Epsilon", "slug": "epsilon", "file": "s1/epsilon.md",
                       "is_published": True, "gated": True}
    # Defaults stay unwritten, and the comments are all still there.
    assert "kind" not in pulled["modules"][0]["lessons"][1]
    for line in MANIFEST.splitlines():
        if line.strip().startswith("#") and "gamma" not in line:
            assert line in doc.text()


# --------------------------------------------------------------------------- #
# adopt: the baseline a pull leaves behind
# --------------------------------------------------------------------------- #


def snap(order=("a", "b"), title="A", body="x"):
    s = empty_snapshot()
    s["modules"]["s1"] = {"title": "S1", "lessons": list(order)}
    s["lessons"]["s1/a"] = {"title": title, "is_published": True, "gated": False,
                            "body": digest(body)}
    s["lessons"]["s1/b"] = {"title": "B", "is_published": True, "gated": False,
                            "body": digest("y")}
    return s


def test_adopt_baselines_what_the_pull_brought_back():
    old, live = snap(title="A", body="x"), snap(title="A2", body="x2")
    plan = snap(title="A2", body="x2")
    base = adopt(old, live, plan, None, course_meta=False, module_order=False)
    assert drift(base, live, plan) == []


def test_adopt_keeps_the_old_baseline_where_the_repo_still_differs():
    """A reorder the pull only reported must still stop the next publish."""
    old = snap(order=("a", "b"))
    live = snap(order=("b", "a"), title="A2")
    plan = snap(order=("a", "b"), title="A2")
    base = adopt(old, live, plan, None, course_meta=False, module_order=False)
    assert base["modules"]["s1"]["lessons"] == ["a", "b"]
    assert base["lessons"]["s1/a"]["title"] == "A2"
    assert [d.what for d in drift(base, live, plan)] == ["lesson order"]


def test_adopt_is_scoped_and_forgets_a_lesson_deleted_on_both_sides():
    old = snap()
    old["lessons"]["s2/z"] = {"title": "Z"}
    live, plan = snap(order=("a",)), snap(order=("a",))
    for s in (live, plan):
        del s["lessons"]["s1/b"]
    base = adopt(old, live, plan, ["s1"], course_meta=False, module_order=False)
    assert "s1/b" not in base["lessons"]
    assert base["lessons"]["s2/z"] == {"title": "Z"}
