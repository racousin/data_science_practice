#!/usr/bin/env python3
"""Report slides whose content does not fit, even at the smallest font.

`build_slides.py` steps down a font ladder until the content fits the body box.
When even the last rung does not fit, it renders anyway — and the overflow falls
off the bottom of the slide, invisibly. This finds those slides before a class
does.

    uv run --with python-pptx --with pyyaml --with matplotlib \
        python tools/check_slide_overflow.py content/python-ai-engineering

    ... --module s1-git-and-packaging     # one module
    ... --verbose                         # per-block heights, to pick a split

Exit status is 1 if anything overflows, so it works as a build gate.

It measures with the same `Measurer` the renderer lays slides out with, so what
it reports is what the deck does. A figure has already given up all the height
it can by the time a slide is listed here — it shrinks to fit the text down to
IMAGE_MIN_H — so the remaining fix is editorial: split the slide with a `---`
and give the second half a heading. Of the 4.95 in a titled slide has, a figure
at full size takes 3.4, which leaves room for one short paragraph or a
four-row table and nothing else.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_slides as bs


def describe(block) -> str:
    if block.kind == "para":
        return block.text[:72]
    if block.kind == "list":
        return f"{len(block.items)} items: {str(block.items[0][1])[:48]}"
    if block.kind == "code":
        return f"{len(block.lines)} lines: {(block.lines[0][:48] if block.lines else '')}"
    if block.kind == "table":
        return f"{len(block.rows)} rows: {' | '.join(block.rows[0])[:48]}"
    if block.kind == "image":
        return os.path.basename(getattr(block, "src", ""))
    return str(getattr(block, "text", ""))[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("content", help="course content directory (holds course.yaml)")
    ap.add_argument("--module", action="append", help="only this module slug")
    ap.add_argument("--verbose", action="store_true", help="show per-block heights")
    ap.add_argument("--cache", default="build/slides/.mathcache",
                    help="where display-math PNGs are cached (shared with the build)")
    args = ap.parse_args()

    course = bs.load_manifest(args.content)
    base = os.path.abspath(args.content)
    measure = bs.Measurer(
        [base, os.path.join(base, "assets"), os.path.dirname(base)],
        bs.MathRenderer(args.cache),
    )
    titled = (bs.BODY_BOTTOM - bs.BODY_TOP) / bs.Emu(914400)
    untitled = (bs.BODY_BOTTOM - bs.Inches(1.0)) / bs.Emu(914400)
    smallest = bs.SIZE_LADDER[-1]

    problems = 0
    for module in course["modules"]:
        slug = module.get("slug") or bs._slug(module["title"])
        if args.module and slug not in args.module:
            continue
        for lesson in module.get("lessons", []):
            if not lesson.get("in_deck", True):
                continue
            body = bs.read_lesson_body(args.content, lesson)
            for n, chunk in enumerate(bs.split_slides(body), 1):
                title, blocks = bs.parse_blocks(bs.NOTES_RE.sub("", chunk))
                if not title and not blocks:
                    continue
                avail = titled if title else untitled
                height, image_heights = measure.plan(blocks, smallest, avail)
                if height <= avail:
                    continue
                problems += 1
                print(
                    f"{slug}/{lesson['slug']} slide {n}: "
                    f"{title or '(untitled)'} — needs {height:.2f} in of {avail:.2f}"
                )
                if args.verbose:
                    for i, b in enumerate(blocks):
                        h = (image_heights[i] + 0.16 if b.kind == "image"
                             else measure.block_height(b, smallest))
                        print(f"    {b.kind:8} {h:5.2f}  {describe(b)}")

    print(f"{problems} overflowing slide(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
