# Courseware

Markdown source of truth for the taught modules, the competitions they link to,
and the build paths that consume both.

```text
     content/<course>/            competitions/<pkg>/   ← you edit these
    course.yaml + *.md            config.py + env.py
            │                             │
   ┌────────┴────────┐                    │
   ▼                 ▼                    ▼
build_slides.py  publish_mlarena.py  build_competitions.py
   │                 │                    │
build/slides     ML-Arena course ◄──attach── ML-Arena competitions
   *.pptx        (modules+lessons)         (one per session)
```

One source per artefact, nothing authored twice. `course.yaml` names the
competition ids its modules link to, so the attachment is declared in the same
manifest as everything else.

**Status:** both modules are written — `python-ai-engineering` (12h) and
`ms2a-machine-learning-practice` (30h). See `../CURRICULUM_PLAN.md` for the restructure that
produced them.

---

## Layout

```text
courseware/
├── Makefile
├── content/
│   ├── python-ai-engineering/        # 12h — 4 sessions
│   │   ├── course.yaml               # the manifest — structure, order, metadata
│   │   ├── assets/                   # images referenced by the lessons
│   │   ├── s1-git-and-packaging/     # one directory per module (= per session)
│   │   │   └── *.md                  # one file per lesson
│   │   ├── s2-shell-notebooks-colab/
│   │   ├── s3-data-science-nutshell/
│   │   ├── s4-pytorch-nutshell/
│   │   ├── reference/                # demoted, self-study material
│   │   └── .mlarena-state.json       # id map — committed, see "Publishing"
│   └── ms2a-machine-learning-practice/               # 30h — 10 sessions
│       ├── course.yaml
│       ├── assets/{collect,tabular,nn,cv,nlp,rl}/
│       ├── s1-data-collection/ … s10-reinforcement-learning-2/
│       ├── project/                  # the ML-Arena project brief (50% of the grade)
│       ├── reference/
│       └── .mlarena-state.json
├── competitions/                     # one competition per taught session
│   ├── s1-textstats/                 # flex_v1 — Lab 1's three functions
│   ├── s2-readability/               # flex_v1 — Lab 2's Flesch score
│   ├── s3-adult-income/              # file_v1 — Lab 3's pipeline
│   ├── s4-mnist-warmup/              # file_v1 — Lab 4's submission dry run
│   ├── localtest.py                  # run an env.py the way the worker would
│   └── .mlarena-state.json           # id lockfile — committed
├── tools/
│   ├── build_slides.py               # Markdown -> PPTX
│   ├── mathrender.py                 # LaTeX -> Unicode / PNG
│   ├── publish_mlarena.py            # Markdown -> ML-Arena (idempotent)
│   ├── build_competitions.py         # competition packages -> ML-Arena
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
make slides-one MODULE=s2-shell-notebooks-colab # one
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

### Publishing text without images

`--skip-media` (or `make publish NO_MEDIA=1`) publishes lesson bodies without
uploading their images, leaving the repo-relative paths in place. Those lessons
render with broken image links until you re-run **without** the flag, which
uploads each file and rewrites the bodies. Nothing is lost — the reference is
still in the markdown, which is why the flag does not substitute a placeholder.

Use it only when the server's media route is unavailable.

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

## Competitions

Each taught session has one competition, built from a package under
`competitions/` and linked to that session's module. Sessions 1 and 2 grade the
lab's *code* (`flex_v1` — competitors upload `agent.py`); Sessions 3 and 4 grade
a *submission file* (`file_v1`). The reference module has none.

```bash
export MLARENA_API_KEY=mlk_creator_...
make competitions                 # build, benchmark, verify, start
make competitions-status
make competitions-publish         # flip them public when the course is ready
MLARENA_TEACHER_API_KEY=mlk_teacher_... make competitions-attach
```

`make competitions` refuses to start a competition whose reference solution does
not score the benchmark its package declares, so a green run is evidence the
scoring works — not just that the upload did. See `competitions/README.md` for
the package layout and the platform behaviours the envs are written around.

Note the two different keys. Competition authoring is `creator` scope; attaching
a competition to a module is `/api/teacher/*`, and API-key auth requires the
scope to match **exactly** — a creator key is rejected with
`Key scope 'creator' cannot access 'teacher' route`.

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

## Two courses, one toolchain

Every target takes `COURSE=`, defaulting to `python-ai-engineering`:

```bash
make slides  COURSE=ms2a-machine-learning-practice
make publish COURSE=ms2a-machine-learning-practice
```

The tooling is course-agnostic; only `content/` grows. Each course carries its
own `.mlarena-state.json`, so the two publish independently.

---

## Known gaps

- **`ms2a-machine-learning-practice` is published text-only.** All 12 modules and
  73 lessons are live on course #15, but its 132 images are **not** uploaded.
  The cause is server-side, not in this directory:

  `backend/app/services/course_content.py:25` reads
  `os.environ.get("PATH_COURSES", "/app/storage/courses")`. `PATH_COURSES` is
  not set in `k8s-manifests_prod/backend-deployment.yaml`, and nothing is
  mounted at that path — its three siblings (`PATH_COMPETITIONS`, `PATH_USERS`,
  `PATH_REPOSITORY`) each have both an env var and a PVC. So
  `os.makedirs()` in `backend/app/views/teacher/lessons.py:198` hits the
  read-only image filesystem and every upload returns 500.

  Fix: provision a `courses-volume` PV/PVC, mount it at `/app/storage/courses`,
  set `PATH_COURSES`, and make that line `os.environ["PATH_COURSES"]` so a
  missing mount fails at boot rather than at upload time. Then:

  ```bash
  make publish COURSE=ms2a-machine-learning-practice   # no NO_MEDIA
  ```

  which uploads the 132 files and rewrites every affected body.

- **Course dates in `course.yaml` are placeholders** — 2026-09-07 → 2026-09-11
  for the 12h module, 2026-09-14 → 2026-11-27 for the 30h one. Set the real
  term dates before the first publish.
- **The project competitions are not attached.** `ms2a-machine-learning-practice`'s `project`
  module describes three tracks (CURRICULUM_PLAN.md §7); the competition ids do
  not exist yet, so there is no `competitions:` block. Add one per track once
  they are created.
- **`python-ai-engineering`'s four competitions are hidden.** Ids 179–182 are
  live, benchmarked and attached to modules 14–17 (`make competitions-status`),
  and `course.yaml` declares them. But they are still `is_public=False`, so
  enrolled students get a 404 on the competition pages. One command closes it,
  once the real term dates are set:

  ```bash
  make competitions-publish
  ```
- **No lesson `mlarena:` directives are used yet.** If any are added,
  `preview_lesson` should be wired into `publish_mlarena.py` to validate them
  before publishing.
