# Courseware

Markdown source of truth for the taught modules, plus the two build paths that
consume it.

```text
                     content/<course>/            ← you edit this
                    course.yaml + *.md
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      tools/build_slides.py       tools/publish_mlarena.py
              │                           │
        build/slides/*.pptx        ML-Arena course
```

One source, two outputs. Nothing is authored twice.

**Status:** `python-ai-engineering` (12h) is complete. The 30h module
(*ML en pratique*) is not started — see `../CURRICULUM_PLAN.md`.

---

## Layout

```text
courseware/
├── Makefile
├── content/
│   └── python-ai-engineering/
│       ├── course.yaml               # the manifest — structure, order, metadata
│       ├── assets/                   # images referenced by the lessons
│       ├── s1-git-and-packaging/     # one directory per module (= per session)
│       │   └── *.md                  # one file per lesson
│       ├── s2-agentic-coding/
│       ├── s3-data-science-nutshell/
│       ├── s4-pytorch-nutshell/
│       ├── reference/                # demoted, self-study material
│       └── .mlarena-state.json       # id map — committed, see "Publishing"
├── tools/
│   ├── build_slides.py               # Markdown -> PPTX
│   ├── mathrender.py                 # LaTeX -> Unicode / PNG
│   ├── publish_mlarena.py            # Markdown -> ML-Arena (idempotent)
│   └── harvest_website.py            # one-off: React JSX -> Markdown
└── build/                            # generated, gitignored
```

Directory names are the module **slugs** and file names are the lesson
**slugs**, which keeps the tree round-trippable with the SDK's
`export_course_to_dir`. Display order lives in `course.yaml`, not in the
filenames.

---

## Writing a lesson

A lesson file is ordinary Markdown with two conventions:

**`---` on its own line starts a new slide.** It is a horizontal rule, so
ML-Arena renders it as a separator and the deck builder splits on it. The first
heading in each slide becomes the slide title.

**`<!-- notes: ... -->` becomes the speaker-notes pane.** It is an HTML comment,
so it is invisible on the web.

````markdown
# Lesson Title

An opening paragraph.

<!-- notes: 30 minutes. Do the demo before the theory. -->

---

## A slide

- a bullet
- another one

```python
print("code blocks render as a dark panel")
```

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
````

Supported blocks: paragraphs, headings, bullet and numbered lists (nested),
fenced code, tables, block quotes, images, and display math.

---

## Math

| Form | Rendered as |
|---|---|
| `$...$` inline | transliterated to Unicode — `$\theta$` → θ, `$10^6$` → 10⁶ |
| `$$...$$` display | a real PNG via matplotlib mathtext |

PowerPoint cannot put an image inside a line of text, hence the split. Keep
inline math to short spans (a symbol, an exponent); anything with a fraction or
a summation belongs in a display block.

Display math uses matplotlib's **mathtext**, a LaTeX subset — no system LaTeX
needed. An unsupported macro fails the build with the offending expression
rather than silently emitting a broken slide.

---

## Images

Reference them relative to the course directory:

```markdown
![Three-way merge](assets/git/Git_Three-way_Merge.png)
```

The deck builder resolves the path locally. The publisher uploads the file via
`upload_lesson_media` and rewrites the reference to the served URL, so the same
Markdown works in both outputs. A missing image fails the build
(`--strict-assets`) rather than producing a slide with a hole in it.

SVG is not supported by python-pptx — convert to PNG first:

```bash
uv run --with cairosvg python -c \
  "import cairosvg; cairosvg.svg2png(url='in.svg', write_to='out.png', output_width=1400)"
```

---

## Building slides

```bash
make slides                              # all sessions
make slides-one MODULE=s2-agentic-coding # one
make pdf                                 # slides + PDF (needs LibreOffice)
```

Output: `build/slides/<module-slug>.pptx`, one deck per 3-hour session, 16:9,
with a cover, a divider per lesson, and speaker notes.

Body text is auto-fitted: the builder estimates the content height and steps
down a font ladder until the slide fits. A slide that comes out small is a
slide with too much on it — split it with a `---`.

---

## Publishing to ML-Arena

```bash
export MLARENA_API_KEY=mlk_teacher_...
make publish-dry     # print the plan, touch nothing
make publish
```

### Token scope

Authoring goes through `/api/teacher/*`, which requires a **`mlk_teacher_…`**
token. A `mlk_user_…` token is rejected:

```json
{"error": "Key scope 'user' cannot access 'teacher' route"}
```

Creating the *course* works with any scope and flips your account to teacher —
so if you have no teacher key yet: create the course once with the user key,
then mint a teacher key from your ML-Arena Profile page and re-run.

### Idempotency

The SDK's `author_course_from_dir` is create-only — re-running it duplicates
every module and lesson. `publish_mlarena.py` syncs instead: it records the
server ids in `.mlarena-state.json`, keyed by base URL, and updates in place on
later runs.

**Commit that file.** Without it the script re-resolves modules by slug from the
server, but the lesson map has to be rebuilt. It is a lockfile, not an artifact.

The script is a pure client-side composition of the public SDK methods — no new
endpoint — per the frontend↔SDK parity rule in `mlarena-sdk/PROCESS.md`.

---

## Harvesting from the React site

`tools/harvest_website.py` converts the JSX course pages under
`website/src/pages/**` into rough Markdown. It handles `data-slide` → `---`,
`<Title>` → headings, `<CodeBlock>` → fenced code, `<List>` → bullets,
`<InlineMath>`/`<BlockMath>` → `$`/`$$`, and `<Table>` → GitHub tables.

```bash
make harvest        # everything -> _harvest/  (gitignored)
```

The output is **raw material**, not a deliverable. It gets trimmed, rewritten,
and reorganised into `content/` by hand — which is where the actual editorial
work of the 2026 restructure lives.

---

## Adding the second module

When *ML en pratique* (30h) starts:

```bash
mkdir -p content/ml-en-pratique
# write course.yaml with 10 modules
make slides COURSE=ml-en-pratique
make publish COURSE=ml-en-pratique
```

The tooling is course-agnostic; only `content/` grows.

---

## Known gaps

- **Course dates in `course.yaml` are placeholders** (2026-09-07 → 2026-09-11).
  Set the real week before the first publish.
- **The warm-up competition is not attached.** `course.yaml` has the
  `competitions:` block for session 4 commented out, pending a decision on
  `CURRICULUM_PLAN.md` §9.2 and a competition id.
- **No lesson `mlarena:` directives are used yet.** If any are added,
  `preview_lesson` should be wired into `publish_mlarena.py` to validate them
  before publishing.
