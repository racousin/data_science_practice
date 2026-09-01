# Curriculum Plan — 2026 Restructure

Plan for reorganizing the two teaching modules (UE) and migrating the content to
ML-Arena. Status: **draft, pre-migration**. Decisions still open are listed at the
end.

## 1. Constraints

| Module (UE) | Volume | Rhythm | Role |
|---|---|---|---|
| **Python AI Engineering** | 12h | 4 × 3h, same week | Mise à niveau, front-loaded |
| **Machine Learning en pratique** | 30h | 2 × 3h per week, 10 sessions | Core course + project |

Both keep a dedicated course page. The final project accounts for 50% of the
grade of *ML en pratique* (continuous assessment is the other 50%).

## 2. Key structural fact

The new split **cuts across the two existing website courses**. Content moves
between them — this is the bulk of the mechanical work.

| New module | Sourced from |
|---|---|
| Python AI Engineering (12h) | `data-science-practice` m1, m2, m3 + `python-deep-learning` m1, m2 |
| ML en pratique (30h) | `data-science-practice` m4–m9 + `python-deep-learning` m3, m4 |

`python-deep-learning` is dismembered: m1+m2 go up front into the 12h module,
m3+m4 land in the middle of the 30h module. It ceases to exist as a standalone
course.

## 3. Python AI Engineering — 12h (4 × 3h)

| # | Session | Source | Volume today | Action |
|---|---|---|---|---|
| 1 | Git + Python packaging | DSP m1 + m2 | 137 slides, 27 files | **Cut ~60%** — currently 2 sessions of material |
| 2 | Claude Code (+ open alternative) | — | none | **Build from scratch** |
| 3 | Data science in a nutshell | DSP m3 | 55 slides | Fits; light trim |
| 4 | PyTorch in a nutshell | PDL m1 + m2 | 155 slides | **Cut ~65%** |

### Decision: Claude Code goes at position 2, not 4

Placing it second means students *use* it in sessions 3–4 and across all 30h of
ML en pratique. Placed last it becomes a demo with no downstream leverage.
Session 1 ends with git + environment working, which is the exact prerequisite
for setting up an agentic coding tool. The counter-argument (students need
judgment before the tool) is weak here: this is a mise à niveau for engineering
students who already write Python.

### Session 1 — what to keep from DSP m1 + m2

- Keep: install/config, first commits, remotes, branching & merging, collaborating (PR flow), cheatsheet.
- Keep: venv/uv, installing packages, building a package, tests.
- Demote to reference pages (not class time): GitHub Desktop, GitHub Actions, IDE, syntax & linting.

### Session 4 — what to keep from PDL m1 + m2

- Keep: tensors, autograd (torch perspective), optimizers, a minimal MLP end-to-end.
- Demote to reference: autograd mathematical perspective, advanced gradient mechanics, historical context.

## 4. Machine Learning en pratique — 30h (10 × 3h)

### Capacity problem

Counting the existing content at an honest pace: Data collection 1, Preprocessing 1,
Tabular 1, Advanced NN 1.5–2, CV 2, Generative 1, NLP 2–3, RL 2 → **~12 sessions of
material for 10 slots**. The overflow is not one session, it is two to three.

### Target lineup

| # | Session | Source |
|---|---|---|
| 1 | Data Collection | DSP m4 |
| 2 | Data Preprocessing | DSP m5 |
| 3 | Tabular Models | DSP m6 |
| 4 | Advanced Neural Networks with PyTorch | PDL m3 + m4 |
| 5 | Computer Vision 1 — classification & transfer learning | DSP m7 |
| 6 | Computer Vision 2 — detection, segmentation, **image generation** | DSP m7 |
| 7 | NLP 1 — tokenization → transformers | DSP m8 |
| 8 | NLP 2 — LLMs, fine-tuning, RAG, agents, **text generation** | DSP m8 |
| 9 | Reinforcement Learning 1 — MDP → model-free | DSP m9 |
| 10 | Reinforcement Learning 2 — deep RL, policy gradient, multi-agent | DSP m9 |

### Decision: no standalone "Génération" session

Split it instead — VAE/GAN/diffusion into CV2, autoregressive generation into
NLP2. Rationale:

- Existing generative content is a single file (`module7/course/GenerativeModel.js`), the thinnest topic in the repo.
- DSP m8 already covers LLM generation at length.
- This preserves 2 RL sessions, which feed ML-Arena and the project.

### Required cuts

- **Multi-GPU scaling** (PDL m4) → reference page. Students have no multi-GPU hardware.
- **NLP** → trim ~40%. DSP m8 is 209 slides, the largest module in the repo, for 2 sessions.
- **3D CNN** and **image enhancement** (DSP m7) → reference pages.

### Format gap

**DSP m5 (Preprocessing) has zero `data-slide` markers** — the only content module
never converted to slide format. Needs conversion before migration.

## 5. Content inventory (measured)

| Module | Files | Lines | Slides |
|---|---|---|---|
| DSP m1 Git | 15 | 3676 | 116 |
| DSP m2 Python Env | 12 | 2220 | 21 |
| DSP m3 DS Methodology | 11 | 2036 | 55 |
| DSP m4 Data Collection | 15 | 22212 | 29 |
| DSP m5 Preprocessing | 10 | 2872 | **0** |
| DSP m6 Tabular | 10 | 2736 | 49 |
| DSP m7 Image | 26 | 6383 | 74 |
| DSP m8 NLP | 20 | 8608 | 209 |
| DSP m9 RL | 23 | 4168 | 59 |
| PDL m1 Foundations | 8 | 3156 | 109 |
| PDL m2 Autodiff | 9 | 2483 | 46 |
| PDL m3 Training | 20 | 2164 | 73 |
| PDL m4 Performance | 7 | 1607 | 54 |

## 6. Work to produce

1. **Claude Code session (3h)** — entirely new. Only genuine content gap.
2. **Preprocessing slide conversion** — DSP m5, 10 files.
3. **Trims** — Session 1 (−60%), Session 4 (−65%), NLP (−40%).
4. **Generative merges** — image generation into CV2, text generation into NLP2.
5. **Reference-page demotions** — GitHub Desktop/Actions, IDE, linting, autograd maths, multi-GPU, 3D CNN, enhancement.

## 7. Project

### Prior art

2025 offered two options (Permuted MNIST continual learning, Bipedal Walker RL),
teams of 1–2, private GitHub repo with `racousin` as collaborator, ML-Arena
leaderboard already used for the freeze deadline.

### What is now possible on ML-Arena

- `file_v1` — any submission format, including LLM-judged text.
- `flex_v1` — code/agent competitions, with `gymnasium` / `pettingzoo` presets.

### Proposed shape

One competition family with **tracks mirroring the course arc**, students pick one:

- **Prediction track** — `file_v1`, tabular or vision submission.
- **Agent track** — `flex_v1`, gymnasium or pettingzoo.
- **Generative / LLM track** — `file_v1`, LLM-judged output.

Grading keeps two axes: leaderboard performance + repository quality (git
hygiene, packaging, tests) — the second axis closes the loop with the 12h module.

### Warm-up competition (proposal)

End the 12h module with a throwaway ML-Arena mini-competition (single `file_v1`
submission). It forces git + package + torch in anger and gets every student onto
the platform weeks before the real project opens. Low cost — the platform is ours.

## 8. Migration to ML-Arena

Target repo: `reinforcement_learning_challenge/` (two levels up from this folder).

Mapping onto the ML-Arena course-content layer:

| Website today | ML-Arena |
|---|---|
| Course (`coursesData` entry) | `AcademicCourse` |
| Module (`moduleN`) | `CourseModule` (owned, reusable, live-linked) |
| Course page (`course/*.js`) | `Lesson` |
| Exercise page (`exercise/*.js`) | `Lesson` (exercise-typed) |
| Project page | Competition attached via `ModuleCompetitionLink` |

Authoring goes through `/course-editor` (`frontend/src/pages/CourseAuthoring/`),
backed by `backend/app/views/teacher/`. Students consume via
`frontend/src/pages/CourseLearner/`, the SDK, or the course-scoped MCP server.

**Sequence:** settle the structure below → do the trims and the Claude Code build
on the current website → then migrate. Migrating first would mean porting content
that is about to be deleted.

Files that encode the current structure and will need rewriting:

- `website/src/components/SideNavigation.js` — `coursesData`, `courseContentData`, `exerciseContentData`, `pytorchCourseContentData`, `pytorchExerciseContentData`
- `website/src/courses/DataSciencePractice.js` + `DataSciencePracticeOverview.js`
- `website/src/courses/PythonDeepLearning.js` + `PythonDeepLearningOverview.js`
- `website/src/pages/data-science-practice/project-pages/`

## 9. Open decisions

1. **Project theme for 2026** — undecided. Tracks proposal above needs a call.
2. **Warm-up competition at the end of the 12h** — yes/no.
3. **Course naming** — final French/English titles for the two modules.
4. **Does the 30h reserve time for project work / defense**, or are all 10 sessions content? If a defense session is wanted, one more content session must go.
5. **Claude Code session content** — Claude Code only, or Claude Code + an open-weights alternative for students without access.
