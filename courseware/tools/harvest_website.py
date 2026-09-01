#!/usr/bin/env python3
"""Harvest the React course pages into rough Markdown.

One-off migration aid: the website content lives as JSX under
``website/src/pages/**`` and the courseware content lives as Markdown under
``courseware/content/**``. This script does the mechanical 80% of the
conversion so the remaining 20% (trimming, rewriting, re-ordering) is hand
work on Markdown instead of on JSX.

The output is *raw material*, not a deliverable — it lands in ``_harvest/``
(gitignored) and is then edited into the real lesson files.

Usage::

    python tools/harvest_website.py website/src/pages/data-science-practice/module1 \
        --out _harvest/dsp-m1

Conversions:
  ``<div data-slide>``      -> ``---`` slide separator
  ``<Title order={N}>``     -> ``#`` * N
  ``<Text>`` / ``<p>``      -> paragraph
  ``<List>``/``<List.Item>``-> ``-`` bullets
  ``<CodeBlock code=… language=…>`` -> fenced code block
  ``<InlineMath>``          -> ``$…$``
  ``<BlockMath>``           -> ``$$…$$``
  ``<Image src=…>``         -> ``![alt](src)``
  ``<Table>``               -> GitHub table (best effort)
  ``<Anchor href=…>``       -> ``[text](href)``
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys

# --------------------------------------------------------------------------- #
# JSX attribute helpers
# --------------------------------------------------------------------------- #

# code={`...`} | code="..." | code='...' | code={"..."}
_ATTR_PATTERNS = [
    r'{}=\{{`(?P<v>[^`]*)`\}}',
    r'{}=\{{"(?P<v>(?:[^"\\]|\\.)*)"\}}',
    r"{}=\{{'(?P<v>(?:[^'\\]|\\.)*)'\}}",
    r'{}="(?P<v>[^"]*)"',
    r"{}='(?P<v>[^']*)'",
]


def attr(tag: str, name: str) -> str | None:
    """Extract a JSX attribute value from an opening tag, whatever its quoting."""
    for pat in _ATTR_PATTERNS:
        m = re.search(pat.format(re.escape(name)), tag)
        if m:
            return _unescape_js(m.group("v"))
    return None


def _unescape_js(s: str) -> str:
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\\`", "`")
        .replace("\\\\", "\\")
    )


def _text(s: str) -> str:
    """Collapse a JSX text run into a single clean line of Markdown."""
    s = re.sub(r"""\{['"](.*?)['"]\}""", r"\1", s)  # {'literal'}
    s = re.sub(r"\{/\*.*?\*/\}", "", s, flags=re.S)  # {/* comment */}
    s = html.unescape(s)
    s = s.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

TAG_RE = re.compile(r"<(/?)([A-Za-z][\w.]*)((?:[^<>\"'{}]|\"[^\"]*\"|'[^']*'|\{(?:[^{}`]|`[^`]*`)*\})*)(/?)>", re.S)


def tokenize(src: str):
    """Yield ('text', str) and ('tag', name, attrs, closing, selfclose)."""
    pos = 0
    for m in TAG_RE.finditer(src):
        if m.start() > pos:
            yield ("text", src[pos : m.start()])
        yield ("tag", m.group(2), m.group(3) or "", m.group(1) == "/", m.group(4) == "/")
        pos = m.end()
    if pos < len(src):
        yield ("text", src[pos:])


# --------------------------------------------------------------------------- #
# Converter
# --------------------------------------------------------------------------- #

INLINE_STRONG = {"strong", "b"}
INLINE_EM = {"em", "i"}
BLOCK_TEXT = {"Text", "p", "Paper", "Alert", "Card", "Blockquote"}
SKIP_TAGS = {
    "Container", "Grid", "Grid.Col", "Flex", "Box", "Stack", "Group", "Space",
    "div", "span", "Center", "SimpleGrid", "ScrollArea", "Divider", "ThemeIcon",
    "Tabs", "Tabs.List", "Tabs.Tab", "Tabs.Panel", "Badge", "Accordion",
}


class Harvester:
    def __init__(self) -> None:
        self.out: list[str] = []
        self.list_depth = 0
        self.buf: list[str] = []
        self.in_table = False
        self.table_rows: list[list[str]] = []
        self.table_cell: list[str] | None = None
        self.table_head = False
        self.first_slide = True
        self.captures: list[tuple[str, str, list[str]]] = []

    # -- emit helpers ------------------------------------------------------ #
    def flush(self) -> None:
        line = _text(" ".join(self.buf))
        self.buf.clear()
        if not line:
            return
        if self.list_depth:
            self.out.append(f"{'  ' * (self.list_depth - 1)}- {line}")
        else:
            self.out.append(line)
            self.out.append("")

    def emit(self, line: str = "") -> None:
        self.flush()
        self.out.append(line)

    def capture(self, kind: str, meta: str) -> None:
        """Start buffering inline text for a construct that needs its own body
        (a heading, a link) rather than letting it land in the page flow."""
        self.captures.append((kind, meta, self.buf))
        self.buf = []

    def release(self) -> tuple[str, str]:
        """Close the innermost capture, returning ``(meta, collected_text)``."""
        if not self.captures:
            return "", _text(" ".join(self.buf))
        _kind, meta, outer = self.captures.pop()
        body = _text(" ".join(self.buf))
        self.buf = outer
        return meta, body

    # -- main loop --------------------------------------------------------- #
    def run(self, src: str) -> str:
        # Drop imports / export / component boilerplate before the first return.
        src = re.sub(r"^\s*import .*?$", "", src, flags=re.M)
        src = re.sub(r"^\s*export default .*?$", "", src, flags=re.M)

        for tok in tokenize(src):
            if tok[0] == "text":
                self.on_text(tok[1])
            else:
                _, name, attrs, closing, selfclose = tok
                self.on_tag(name, attrs, closing, selfclose)
        self.flush()
        return self.cleanup("\n".join(self.out))

    def on_text(self, raw: str) -> None:
        # Strip JS expressions that are not simple string literals.
        if self.in_table and self.table_cell is not None:
            self.table_cell.append(raw)
            return
        self.buf.append(raw)

    def on_tag(self, name: str, attrs: str, closing: bool, selfclose: bool) -> None:
        tag = f"<{name}{attrs}>"

        # ---- slides ------------------------------------------------------ #
        if name == "div" and "data-slide" in attrs and not closing:
            self.flush()
            if self.first_slide:
                self.first_slide = False
            else:
                self.emit("---")
                self.emit("")
            return

        # ---- tables ------------------------------------------------------ #
        if name.startswith("Table") or name in {"table", "thead", "tbody", "tr", "th", "td"}:
            self.on_table(name, closing, selfclose)
            return

        # ---- headings ---------------------------------------------------- #
        if name == "Title" and not closing:
            self.flush()
            raw_order = attr(tag, "order")
            if raw_order is None:
                m = re.search(r"order=\{(\d)\}", tag)
                raw_order = m.group(1) if m else "2"
            n = max(1, min(6, int(raw_order)))
            self.capture("heading", "#" * n)
            return
        if name == "Title" and closing:
            level, body = self.release()
            if body:
                self.out.append(f"{level} {body}")
                self.out.append("")
            return

        # ---- code -------------------------------------------------------- #
        if name == "CodeBlock" and not closing:
            self.flush()
            code = attr(tag, "code")
            lang = attr(tag, "language") or "bash"
            if code is not None:
                self.out.append(f"```{lang}")
                self.out.extend(code.strip("\n").split("\n"))
                self.out.append("```")
                self.out.append("")
            return

        # ---- math -------------------------------------------------------- #
        if name in {"InlineMath", "BlockMath"} and not closing:
            expr = attr(tag, "math") or ""
            if expr:
                if name == "InlineMath":
                    self.buf.append(f" ${expr}$ ")
                else:
                    self.emit(f"$$\n{expr}\n$$")
                    self.emit("")
            return

        # ---- images ------------------------------------------------------ #
        if name == "Image" and not closing:
            src = attr(tag, "src")
            alt = attr(tag, "alt") or ""
            if src:
                self.emit(f"![{alt}]({src})")
                self.emit("")
            return

        # ---- links ------------------------------------------------------- #
        if name == "Anchor" and not closing:
            self.capture("link", attr(tag, "href") or "")
            return
        if name == "Anchor" and closing:
            href, label = self.release()
            if label:
                self.buf.append(f" [{label}]({href}) ")
            return

        # ---- inline emphasis --------------------------------------------- #
        if name in INLINE_STRONG:
            self.buf.append("**")
            return
        if name in INLINE_EM:
            self.buf.append("*")
            return
        if name == "Code":
            self.buf.append("`")
            return

        # ---- lists ------------------------------------------------------- #
        if name == "List" and not closing:
            self.flush()
            self.list_depth += 1
            return
        if name == "List" and closing:
            self.flush()
            self.list_depth = max(0, self.list_depth - 1)
            if self.list_depth == 0:
                self.out.append("")
            return
        if name == "List.Item":
            self.flush()
            return

        # ---- block text --------------------------------------------------- #
        if name in BLOCK_TEXT:
            self.flush()
            return

        if name in SKIP_TAGS or name.startswith("Icon"):
            return

        # Unknown component: flush so its children start on a fresh line.
        self.flush()

    # -- tables ------------------------------------------------------------ #
    def on_table(self, name: str, closing: bool, selfclose: bool) -> None:
        base = name.split(".")[-1].lower()
        if base in {"table"}:
            if closing:
                self.emit_table()
            else:
                self.flush()
                self.in_table = True
                self.table_rows = []
            return
        if base in {"thead", "th"} and not closing:
            self.table_head = True
        if base == "tr" and not closing:
            self.table_rows.append([])
        if base in {"th", "td"}:
            if closing:
                if self.table_cell is not None and self.table_rows:
                    self.table_rows[-1].append(_text(" ".join(self.table_cell)))
                self.table_cell = None
            else:
                self.table_cell = []

    def emit_table(self) -> None:
        self.in_table = False
        rows = [r for r in self.table_rows if r]
        self.table_rows = []
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        self.out.append("| " + " | ".join(rows[0]) + " |")
        self.out.append("|" + "|".join(["---"] * width) + "|")
        for r in rows[1:]:
            self.out.append("| " + " | ".join(r) + " |")
        self.out.append("")

    # -- post ------------------------------------------------------------- #
    @staticmethod
    def cleanup(md: str) -> str:
        # Tighten emphasis/code spans, but only within a line — these must not
        # swallow the newline that separates a paragraph from a fenced block.
        md = re.sub(r"\*\*[ \t]+", "**", md)
        md = re.sub(r"[ \t]+\*\*", "**", md)
        md = re.sub(r"`[ \t]+", "`", md)
        md = re.sub(r"[ \t]+`", "`", md)
        # A fence must start its own line and be surrounded by blank lines.
        md = re.sub(r"(?<!\n)(```)", r"\n\n\1", md)
        md = re.sub(r"(```)(?![\w+-]*\n)", r"\1\n", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = re.sub(r"^\s*[-*]\s*$", "", md, flags=re.M)
        return md.strip() + "\n"


def convert(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return Harvester().run(fh.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="file or directory of .js course pages")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    files = []
    if os.path.isfile(args.src):
        files = [args.src]
    else:
        for root, _, names in os.walk(args.src):
            files += [os.path.join(root, n) for n in names if n.endswith(".js")]

    if not files:
        raise SystemExit(f"no .js files under {args.src}")

    os.makedirs(args.out, exist_ok=True)
    for f in sorted(files):
        rel = os.path.relpath(f, args.src if os.path.isdir(args.src) else os.path.dirname(args.src))
        dest = os.path.join(args.out, rel.replace(os.sep, "__")[:-3] + ".md")
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(convert(f))
        print(f"{f} -> {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
