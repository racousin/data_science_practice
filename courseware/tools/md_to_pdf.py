#!/usr/bin/env python3
"""Render a Markdown document to a PDF, for the documents that ship with a dataset.

    uv run --with weasyprint --with markdown python tools/md_to_pdf.py IN.md OUT.pdf [--title T]

Markdown -> HTML with python-markdown (tables, fenced code, definition lists,
attribute lists), then HTML -> PDF with WeasyPrint. The stylesheet is inline so
the result does not depend on the machine. Pure Python: no pandoc, no LaTeX.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; } }
html { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 8pt 0; border-bottom: 2px solid #1c5cab; padding-bottom: 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; color: #1c5cab; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 0; padding-left: 18pt; }
li { margin-bottom: 2.5pt; }
code { font-family: Menlo, "DejaVu Sans Mono", Consolas, monospace; font-size: 9.2pt; background: #f2f2f0; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f5f3; border: 1px solid #ddd; padding: 6pt 8pt; font-size: 8.8pt; line-height: 1.35;
      white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0; font-size: 9.2pt; page-break-inside: auto; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #e9eef6; font-weight: 600; }
tr { page-break-inside: avoid; }
blockquote { margin: 6pt 0 8pt 12pt; padding-left: 8pt; border-left: 3px solid #c8c8c8; color: #333; }
hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; }
em { color: #333; }
a { color: #1c5cab; text-decoration: none; }
"""


def md_to_html(text: str, title: str | None) -> str:
    import markdown
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "def_list", "attr_list", "md_in_html"],
        output_format="html5")
    head = f"<title>{title}</title>" if title else ""
    return f"<!doctype html><html><head><meta charset='utf-8'>{head}<style>{CSS}</style></head><body>{body}</body></html>"


def convert(src: Path, dst: Path, title: str | None = None) -> Path:
    from weasyprint import HTML
    html = md_to_html(src.read_text(encoding="utf-8"), title)
    dst.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(src.parent)).write_pdf(str(dst))
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--title")
    args = ap.parse_args(argv)
    out = convert(args.src, args.dst, args.title)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    sys.exit(main())
