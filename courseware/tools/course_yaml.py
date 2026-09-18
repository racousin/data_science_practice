#!/usr/bin/env python3
"""Edit course.yaml in place, a line at a time, so its comments survive.

`course.yaml`'s comments are the record of why each course is shaped the way it
is — which lesson was merged into which, which challenge was detached and when.
A PyYAML round-trip deletes every one of them. ruamel.yaml keeps them, but it
hangs a comment written above a list item on the item *before* it, so a reorder
would move each comment onto the wrong lesson. So `make pull` edits the text:
it rewrites the one line that holds a value, moves whole item blocks (the
comment run directly above an item travels with it), and inserts or drops whole
items. Every line it did not mean to touch is byte for byte what it was.

It understands the one shape this repo writes, and nothing more general:

    course:                 a mapping of scalars and block scalars
    modules:                a list of mappings, first key on the dash line
      - title: ...
        summary: >          a block scalar (> or |)
        competitions:       a list of mappings, first key on the dash line
        lessons:            the same

Anything else is a crash with the line it choked on, not a guess. The caller
re-parses the result with PyYAML and compares it with what it meant to write
(`pull_mlarena.py` does); an edit that did not land is a crash as well, never a
silently-wrong manifest.
"""
from __future__ import annotations

import json
import re
import textwrap

import yaml

# A scalar that can go unquoted: this file writes slugs, paths and kinds bare.
_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*\Z")
_WIDTH = 80


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _blank(line: str) -> bool:
    return not line.strip()


def _comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def render(value, quote: bool) -> str:
    """A scalar as YAML. Strings are double-quoted when `quote` (the style of
    the value being replaced, or of the key for a new line) or when a bare
    spelling would parse as something else — `true`, a number, a date."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        raise TypeError(f"course_yaml writes str, int and bool, not {value!r}")
    if not quote and _PLAIN.match(value) and yaml.safe_load(value) == value:
        return value
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(value, ensure_ascii=False)


def split_value(rest: str) -> tuple[str, str, str]:
    """`' "A # B"   # note'` -> (`' '`, `'"A # B"'`, `'   # note'`).

    The gap after the colon, the value token, and what follows it — kept, so
    an inline comment (`is_published: true   # core`) survives a rewrite.
    """
    body = rest.lstrip(" ")
    gap = rest[: len(rest) - len(body)]
    if body.startswith('"'):
        i = 1
        while i < len(body) and body[i] != '"':
            i += 2 if body[i] == "\\" else 1
        end = i + 1
    elif body.startswith("'"):
        i = 1
        while i < len(body):
            if body[i] == "'" and body[i + 1: i + 2] != "'":
                break
            i += 2 if body[i] == "'" else 1
        end = i + 1
    else:
        hit = re.search(r"\s#", body)
        end = len((body[:hit.start()] if hit else body).rstrip())
    return gap, body[:end], body[end:]


class CourseYaml:
    """The manifest as lines, addressed by slug rather than by position.

    Every method re-locates what it edits from scratch, so edits can be applied
    in any order without an index going stale.
    """

    def __init__(self, text: str):
        self.lines = text.split("\n")

    def text(self) -> str:
        return "\n".join(self.lines)

    # ------------------------------------------------------------------ #
    # Locating
    # ------------------------------------------------------------------ #

    def _end(self, start: int, indent: int) -> int:
        """One past the last line of the block that opens at `start`: every
        following line deeper than `indent`, comments included, trailing blank
        lines excluded."""
        last = start
        for j in range(start + 1, len(self.lines)):
            line = self.lines[j]
            if _blank(line):
                continue
            if _indent(line) <= indent:
                break
            last = j
        return last + 1

    def _key(self, start: int, end: int, key: str, key_indent: int,
             dash: bool) -> int | None:
        """The line of `key` in the mapping spanning [start, end) — a list item
        when `dash` (its first key shares the `- ` line), else a plain one."""
        own = re.compile(rf" {{{key_indent}}}{re.escape(key)}:(\s|$)")
        first = re.compile(rf" {{{key_indent - 2}}}- {re.escape(key)}:(\s|$)")
        for j in range(start, end):
            line = self.lines[j]
            if _blank(line) or _comment(line):
                continue
            if dash and j == start:
                if first.match(line):
                    return j
            elif _indent(line) == key_indent and own.match(line):
                return j
        return None

    def _top(self, key: str) -> tuple[int, int]:
        """(line, end) of a top-level key."""
        for j, line in enumerate(self.lines):
            if re.match(rf"{re.escape(key)}:(\s|$)", line):
                return j, self._end(j, 0)
        raise SystemExit(f"course.yaml has no top-level `{key}:`")

    def _items(self, key_line: int, end: int) -> tuple[int, list[tuple[int, int]]]:
        """(dash indent, [(start, end), ...]) of the list under `key_line`."""
        dashes = [j for j in range(key_line + 1, end)
                  if not _comment(self.lines[j])
                  and self.lines[j].lstrip(" ").startswith("- ")]
        if not dashes:
            return _indent(self.lines[key_line]) + 2, []
        d = min(_indent(self.lines[j]) for j in dashes)
        starts = [j for j in dashes if _indent(self.lines[j]) == d]
        return d, [(s, self._end(s, d)) for s in starts]

    def _scalar(self, line: int):
        _, token, _ = split_value(self.lines[line].split(":", 1)[1])
        return yaml.safe_load(token) if token else None

    def _find(self, key_line: int, end: int, ident: str, value) -> tuple[int, int, int]:
        """(dash indent, start, end) of the item under `key_line` whose `ident`
        key holds `value`."""
        d, items = self._items(key_line, end)
        for s, e in items:
            j = self._key(s, e, ident, d + 2, dash=True)
            if j is not None and self._scalar(j) == value:
                return d, s, e
        raise SystemExit(f"course.yaml: no item with {ident}: {value!r} under line "
                         f"{key_line + 1}")

    def _module(self, mslug: str) -> tuple[int, int, int]:
        top, end = self._top("modules")
        return self._find(top, end, "slug", mslug)

    def _list(self, mslug: str, key: str, create: bool) -> tuple[int, int, int] | None:
        """(key line, end, key indent) of a module's `lessons:` or
        `competitions:` list, or None if the module has no such key."""
        d, s, e = self._module(mslug)
        j = self._key(s, e, key, d + 2, dash=True)
        if j is None:
            if not create:
                return None
            self.lines.insert(e, f"{' ' * (d + 2)}{key}:")
            j = e
        return j, self._end(j, d + 2), d + 2

    # ------------------------------------------------------------------ #
    # Scalars and block scalars
    # ------------------------------------------------------------------ #

    def _set(self, start: int, end: int, key_indent: int, dash: bool, key: str,
             value, quote: bool) -> None:
        j = self._key(start, end, key, key_indent, dash)
        if value is None:
            if j is not None:
                del self.lines[j: self._end(j, key_indent)]
            return
        if isinstance(value, str) and "\n" in value.strip("\n"):
            self._set_block(end, key_indent, key, value, j)
            return
        if j is None:
            # After the last one-line key, so a new scalar lands among its
            # siblings rather than after a `lessons:` list.
            at = start
            for k in range(start, end):
                line = self.lines[k]
                if _blank(line) or _comment(line) or _indent(line) > key_indent:
                    continue
                token = split_value(line.split(":", 1)[1])[1]
                if token and token[0] not in ">|":
                    at = k
            self.lines.insert(at + 1, f"{' ' * key_indent}{key}: {render(value, quote)}")
            return
        head, rest = self.lines[j].split(":", 1)
        gap, token, trailer = split_value(rest)
        if token[:1] in (">", "|"):
            self._set_block(end, key_indent, key, value, j)
            return
        was_quoted = token[:1] in ('"', "'")
        self.lines[j] = f"{head}:{gap or ' '}{render(value, was_quoted)}{trailer}"

    def _set_block(self, end: int, key_indent: int, key: str, value: str,
                   j: int | None) -> None:
        """Write a block scalar, keeping its indicator (`>` folds, `|` keeps
        line breaks). A value a folded block cannot hold exactly — a line that
        starts with a space, say — is caught by the caller's re-parse."""
        pad = " " * (key_indent + 2)
        if j is None:
            j, stop = end, end
            head, style, trailer = " " * key_indent + key, ">", ""
        else:
            head, rest = self.lines[j].split(":", 1)
            _, token, trailer = split_value(rest)
            style = token if token[:1] in (">", "|") else ">"
            stop = self._end(j, key_indent)
        paras = value.rstrip("\n").split("\n")
        body: list[str] = []
        if style[0] == ">":
            for n, para in enumerate(paras):
                if n:
                    body.append("")
                body += [pad + chunk for chunk in textwrap.wrap(
                    para, width=_WIDTH - len(pad), break_long_words=False,
                    break_on_hyphens=False)]
        else:
            body = [pad + para if para else "" for para in paras]
        self.lines[j:stop] = [f"{head}: {style}{trailer}"] + body

    def set_course(self, field: str, value, quote: bool = True) -> None:
        top, end = self._top("course")
        self._set(top + 1, end, 2, False, field, value, quote)

    def set_module(self, mslug: str, field: str, value, quote: bool = True) -> None:
        d, s, e = self._module(mslug)
        self._set(s, e, d + 2, True, field, value, quote)

    def set_lesson(self, mslug: str, lslug: str, field: str, value) -> None:
        j, end, _ = self._list(mslug, "lessons", create=False)
        d, s, e = self._find(j, end, "slug", lslug)
        self._set(s, e, d + 2, True, field, value, quote=field == "title")

    # ------------------------------------------------------------------ #
    # Lists
    # ------------------------------------------------------------------ #

    def _lead(self, start: int, floor: int, d: int) -> int:
        """First line of the comment run directly above the item at `start`
        (no blank line in between), which travels with the item."""
        j = start
        while j - 1 > floor and _comment(self.lines[j - 1]) \
                and _indent(self.lines[j - 1]) >= d:
            j -= 1
        return j

    def _reorder(self, key_line: int, end: int, ident: str, order: list) -> None:
        d, items = self._items(key_line, end)
        ids = [self._scalar(self._key(s, e, ident, d + 2, dash=True)) for s, e in items]
        if sorted(map(str, ids)) != sorted(map(str, order)):
            raise SystemExit(f"course.yaml: cannot reorder {ids} into {order} — "
                             f"not the same set")
        leads = [self._lead(s, key_line, d) for s, _ in items]
        chunks = {ident_: self.lines[lead: e]
                  for ident_, lead, (_, e) in zip(ids, leads, items)}
        # Blank lines between items belong to the position, not to the item.
        gaps = [self.lines[items[k][1]: leads[k + 1]] for k in range(len(items) - 1)]
        gaps.append(self.lines[items[-1][1]: end])
        out = self.lines[key_line + 1: leads[0]]
        for k, ident_ in enumerate(order):
            out += chunks[ident_] + gaps[k]
        self.lines[key_line + 1: end] = out

    def _remove(self, key_line: int, end: int, ident: str, value) -> None:
        """Drop an item. The comment run above it stays: it may explain more
        than this one item, and a stale comment is visible in the diff while a
        deleted one is not."""
        _, s, e = self._find(key_line, end, ident, value)
        del self.lines[s:e]

    def _append(self, key_line: int, end: int, key_indent: int,
                fields: list[tuple[str, object, bool]]) -> None:
        d, items = self._items(key_line, end)
        if not items:
            d = key_indent + 2
        (k0, v0, q0), *rest = fields
        block = [f"{' ' * d}- {k0}: {render(v0, q0)}"]
        block += [f"{' ' * (d + 2)}{k}: {render(v, q)}" for k, v, q in rest if v is not None]
        self.lines[end:end] = block

    def reorder_lessons(self, mslug: str, order: list[str]) -> None:
        j, end, _ = self._list(mslug, "lessons", create=False)
        self._reorder(j, end, "slug", order)

    def remove_lesson(self, mslug: str, lslug: str) -> None:
        j, end, _ = self._list(mslug, "lessons", create=False)
        self._remove(j, end, "slug", lslug)

    def add_lesson(self, mslug: str, fields: list[tuple[str, object, bool]]) -> None:
        """Append a lesson item; `fields` is [(key, value, quote), ...], first
        key on the dash line, None values skipped."""
        j, end, key_indent = self._list(mslug, "lessons", create=True)
        self._append(j, end, key_indent, fields)

    def set_links(self, mslug: str, links: list[tuple[int, str | None]]) -> None:
        """Make a module's `competitions:` list exactly `links`
        ([(challenge id, label)], in order): drop the detached, append the
        attached, rewrite the labels, then reorder. Comments stay put."""
        found = self._list(mslug, "competitions", create=bool(links))
        if found is None:
            return
        want = dict(links)
        j, end, key_indent = found
        d, items = self._items(j, end)
        have = [self._scalar(self._key(s, e, "competition_id", d + 2, dash=True))
                for s, e in items]
        for cid in have:
            if cid not in want:
                j, end, _ = self._list(mslug, "competitions", create=False)
                self._remove(j, end, "competition_id", cid)
        for cid, label in links:
            j, end, key_indent = self._list(mslug, "competitions", create=False)
            if cid not in have:
                self._append(j, end, key_indent,
                             [("competition_id", cid, False), ("label", label, True)])
                continue
            d, s, e = self._find(j, end, "competition_id", cid)
            self._set(s, e, d + 2, True, "label", label, quote=True)
        if links:
            j, end, _ = self._list(mslug, "competitions", create=False)
            self._reorder(j, end, "competition_id", [cid for cid, _ in links])
