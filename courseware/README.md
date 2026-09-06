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
│   ├── python-ai-engineering/        # 12h — 3 modules (see "Session numbering")
│   │   ├── course.yaml               # the manifest — structure, order, metadata
│   │   ├── assets/                   # images referenced by the lessons
│   │   ├── s1-git-and-packaging/     # one directory per module (= per session)
│   │   │   └── *.md                  # one file per lesson
│   │   ├── s3-data-science-nutshell/
│   │   ├── s4-pytorch-nutshell/      # self-study lessons live in the session
│   │   └── .mlarena-state.json       #   they belong to, marked `in_deck: false`
│   └── ms2a-machine-learning-practice/               # 30h — 10 sessions
│       ├── course.yaml
│       ├── assets/{collect,tabular,nn,cv,nlp,rl}/
│       ├── s1-data-collection/ … s10-reinforcement-learning-2/
│       ├── project/                  # the ML-Arena project brief (50% of the grade)
│       ├── reference/
│       └── .mlarena-state.json
├── competitions/                     # one or more per taught session, except
│   │                                 # Session 1, whose Lab 3 borrows comp 65
│   ├── s2-bike-demand/               # file_v1 — regression, worked
│   ├── s2-bank-marketing/            # file_v1 — classification, guided
│   ├── s3-adult-income/              # file_v1 — Lab 3's pipeline
│   ├── s3-diabetes-progression/      # file_v1 — the overfitting demo, worked
│   ├── s3-credit-risk/               # file_v1 — the same, guided
│   ├── s4-california-housing/        # file_v1 — the PyTorch MLP, worked
│   ├── s4-forest-cover/              # file_v1 — the same, guided
│   ├── s4-mnist-warmup/              # file_v1 — Lab 4's submission dry run
│   ├── localtest.py                  # run an env.py the way the worker would
│   └── .mlarena-state.json           # id lockfile — committed
├── tools/
│   ├── build_slides.py               # Markdown -> PPTX
│   ├── check_slide_overflow.py       # slides that render off the bottom
│   ├── check_sync.py                 # has the live course drifted from here?
│   ├── figures/                      # committed generators for authored figures
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

`python-ai-engineering` stores them under
**`assets/<module-slug>/<lesson-slug>/`**, mirroring the lesson tree — so an
image's owner is readable from its path, and cutting a lesson makes the images
to delete obvious. Reference them relative to the course directory:

```markdown
![Three-way merge](assets/s1-git-and-packaging/branching-and-collaboration/Git_Three-way_Merge.png)
```

Most files are stills lifted from the taught PPTX decks. Figures **authored** for
the course are generated by a committed script instead, so every one can be
traced to the code that drew it:

```bash
uv run --with seaborn --with scikit-learn --with pandas \
    python tools/figures/s2_ml_foundations.py
uv run --with matplotlib --with numpy \
    python tools/figures/s1_git_and_packaging.py
```

Session 1 has no taught deck to lift stills from, so **all 30** of its diagrams
come out of that second script. Which lesson directory a figure lands in is the
`lesson = "…"` line at the top of each function — when lessons merge, that line
moves with the PNG.

Add a figure by adding a function there, not by dropping a PNG into the tree.

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

## Session numbering

`python-ai-engineering` holds **four** modules — `s1-git-and-packaging`,
`s2-ml-foundations`, `s3-models-and-tuning`, `s4-pytorch-nutshell` — titled
*Session 1* … *Session 4*. Agentic Coding and the Shell/Notebooks/Colab material
were both folded into Session 1, whose title now names all three scopes; the slot
that freed is what the two ML modules were built into, from the taught decks.

One thing still carries the old numbering: the retired server module
`s3-data-science-nutshell` (#16), whose five lessons were split across
`s2-ml-foundations` and `s3-models-and-tuning` and which must be deleted
server-side. Module *slugs* are immutable server-side, so it cannot simply be
renamed. The two competitions that also carried it (`PAIE S1 — textstats`,
`PAIE S2 — Flesch reading-ease`) were retired on 2026-09-06. See
`COURSE_STATE.md` §1b, §1d, §5.3.

---

## Building slides

```bash
make slides                              # all sessions
make slides-one MODULE=s1-git-and-packaging # one
make check-slides                        # slides whose content overflows
make pdf                                 # slides + PDF (needs LibreOffice)
```

Output: `build/slides/<module-slug>.pptx`, one deck per 3-hour session, 16:9,
with a cover, a divider per lesson, and speaker notes.

Body text is auto-fitted: the builder measures the content and steps down a font
ladder until the slide fits. A slide that comes out small is a slide with too
much on it — split it with a `---`.

Measuring is `Measurer` in `build_slides.py`, and the renderer and
`check_slide_overflow` share one instance, so what the check reports is what the
deck does. Everything is measured from the real thing: an image from its file, a
formula from the PNG mathtext renders, and a table row from the text that wraps
inside it.

Figures are the one block that flexes. A figure alone on a slide grows to fill
the body box (**4.95 in**); sharing with text it shrinks — keeping **3.0 in** for
as long as a rung of the ladder allows, then giving way down to **2.2 in**. It is
never enlarged past 110 dpi, so a small screenshot stays small and sharp.

Only when a figure is at that 2.2 in floor and the slide *still* does not fit is
there nothing left to give: the builder renders anyway, the overflow falls off
the bottom silently, and `make check-slides` reports it. It exits non-zero, so it
gates a build. `VERBOSE=1` prints the per-block heights, which is how you pick
the split point, and the fix from there is editorial — split with a `---` and
give the second half a heading.

All four sessions are clean.

A lesson marked `in_deck: false` in `course.yaml` is skipped by the deck builder
and published as usual — that is how the `Reference — …` self-study lessons sit
inside a session on the web without being lectured from its deck.

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

### Publishing is one-way — check before you publish

`publish_mlarena.py` calls `update_lesson(body_md=…)`, which **replaces** the
server's body with the file's. An edit made in the ML-Arena course editor is
therefore destroyed by the next `make publish`, silently, and it is not
recoverable from this repo. Nothing syncs back.

`make check-sync` is the guard. It is read-only — it never writes to ML-Arena
and never touches your files — and it exits non-zero, so it gates a publish.

```bash
export MLARENA_API_KEY=mlk_teacher_...
make check-sync QUICK=1     # 1 request. structure only
make check-sync             # 1 + N requests (~57, a few seconds). + every body
make check-sync DIFF=1      # ... and print the diff, so you can copy it back
make check-sync MODULE=s2-ml-foundations
```

| Tier | Cost | Catches |
|---|---|---|
| `QUICK=1` | one request | a lesson added, deleted, renamed, reordered, unpublished, or re-timed on the site |
| default | one request per lesson | all of the above, **plus any edit to a lesson body** |

Both compare against `course.yaml` and the markdown, not against a recorded
snapshot, so there is no state to keep current and nothing to seed.

What it treats as equal: the markdown on disk keeps repo-relative image paths so
the deck build works, and the publisher rewrites them to served URLs on the way
up — so both sides are reduced to the image's basename before comparing, and
trailing whitespace is ignored. Everything else is literal, `<!-- notes: -->`
comments included.

The four verdicts:

| | Means |
|---|---|
| `BODY` | the lesson text differs — someone edited it on the website |
| `META` | title, kind, published flag or estimated minutes differ |
| `ORDER` | the server's lesson order is not the manifest's |
| `MISSING` / `ORPHAN` | declared but absent / present but undeclared — the residue of a move (see §1c of `COURSE_STATE.md`) |

**When it reports `BODY`, decide which side wins before publishing.** Copy the
change into the markdown, or accept that the publish will overwrite it. There is
no merge.

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

Each taught session has at least one competition linked to its module. Sessions
2, 3 and 4 build theirs from a package under `competitions/` — two apiece, one
per model family or target type they teach — plus the notebooks that go with
them (`tools/build_notebooks.py`, output under
`website/public/modules/python-ai-engineering/challenges/`, which is the path
the Colab links resolve against on GitHub). They grade a *submission file*
(`file_v1`).

**Session 1 has no package.** Since 2026-09-06 its Lab 3 submits to the existing
PettingZoo · Connect-Four challenge (**65**, `flex_v1`, ELO-ranked), which this
repository does not own, does not build and cannot benchmark. It is the only
attachment in the course that `make competitions` knows nothing about — it is
declared in `course.yaml` and nowhere else. The `Reference — …` self-study
lessons have none.

Within each pair the first challenge ships a **worked** notebook that runs top
to bottom and the second ships a **guided** one — the same protocol in English
with empty cells. `test_challenges.py` enforces both properties: the worked
notebooks are executed and their submissions scored, and the guided ones are
asserted to contain no code.

`build_notebooks.py` also emits three notebooks that are not challenges and need
no ML-Arena account — the pandas/seaborn pre-flight for Session 2 and the two
Session 4 warm-ups. They are reached from their lessons rather than from a
competition page.

Challenge ids come from `competitions/.mlarena-state.json` rather than being
typed into the generator, so a notebook written before its competition exists
picks up the real id on the next regeneration; the sync test then fails if
nobody re-ran the script.

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
- **Session 1's challenge is not ours.** Lab 3 submits to competition **65**
  (PettingZoo · Connect-Four), which this repository does not build and cannot
  benchmark. Its overview page is the PettingZoo blurb and states no baseline.
  Either adopt it — write the overview, state the measured ladder from Lab 3
  Part E — or build a Session 1 package to replace it. The two packages that
  used to serve this module (179, 180) were retired on 2026-09-06;
  `detach_competition(14, 179)` / `(14, 180)` still has to be run server-side,
  because `publish_mlarena.py` only ever attaches.
- **No lesson `mlarena:` directives are used yet.** If any are added,
  `preview_lesson` should be wired into `publish_mlarena.py` to validate them
  before publishing.
