# Audit findings — pae-s2

17 findings

## 1. [blocker / content / fix in courseware] s2-agentic-coding/lab-2

**Problem.** Lab 2 asks for a Flesch reading-ease module and mandates a self-check it makes impossible: it gives only the formula, never defining a word, a sentence, or a syllable. The pinned specification that decides correctness exists solely in competition 180's overview, which no s2 lesson mentions. A student can satisfy every stated Part C requirement, get a green suite, and be 50% wrong.

**Evidence.**
```
lab-2.md Part C: "a known-value case (compute one by hand and check it)" — but the only spec in the lab is the formula block at Part B. I wrote src/textstats/readability.py plus the three mandated tests (known value, empty raises, no terminal punctuation) and got `5 passed in 0.03s`. Scoring that exact module against competition 180's env.py CASES: `pass_rate = 10/20 = 0.5`; failures include `'Wait... what happened?! Nobody knows.' delta=16.9200` (sentence-run rule) and `'Rhythm myths fly by dryly.' delta=16.9200` (y counts as a vowel). `grep -rniE "competition|arena|submit" student_view/python-ai-engineering/s2-agentic-coding/*.md` returns zero hits in any lesson body.
```

**Proposed fix.** Move the specification into the lab. In lab-2.md Part B, after the formula, paste the three pinned rules verbatim from competitions/s2-readability/overview.md ("**Sentences.** Count the maximal *runs* of characters drawn from `.!?` …", "**Words.** Split on whitespace, then strip leading and trailing characters that are not letters or digits …", "**Syllables**, per word. Lowercase it and count the maximal runs of `aeiouy` …") together with the seven-row worked table (`time`, `the`, `place`, `queueing`, `rhythm`, `dryly`, `reevaluation`). Replace Part C's bullet "a known-value case (compute one by hand and check it)" with "a known-value case: `flesch_reading_ease(\"The cat sat on the mat.\") == pytest.approx(116.145)` (6 words / 1 sentence / 6 syllables), plus one case per pinned rule — `\"Wait... no!\"` is two sentences, `dryly` is two syllables, `queueing` is one."

---

## 2. [blocker / validation / fix in courseware] s2-agentic-coding/lab-2 (root cause in s1-git-and-packaging/packaging-and-tests:71)

**Problem.** `uv run pytest`, the verify command Lab 2 Parts C and D are built on, does not run on a repository built to Lab 1's own spec. Lab 1 Part B requires pytest "as a dev extra", and `uv run` / `uv sync` install the project's default dependency groups, not `[project.optional-dependencies]` extras. This breaks the VERIFY step of the loop that is the session's entire thesis, and it breaks it for the agent too — "> Run the tests and fix anything that fails" produces a spawn error rather than a test failure, so the loop never closes.

**Evidence.**
```
Built the Lab 1 package exactly as Part B specifies (pyproject with `[project.optional-dependencies]\ndev = ["pytest>=8", "ruff>=0.6"]`), then ran lab-2.md:62 verbatim:
$ uv run pytest -v
Creating virtual environment at: .venv
   Built textstats @ file:///private/tmp/s2lab
Installed 1 package in 3ms
error: Failed to spawn: `pytest`
  Caused by: No such file or directory (os error 2)
`uv sync && uv run pytest` fails identically. `uv run --extra dev pytest -q` → `2 passed`. uv 0.10.12.
```

**Proposed fix.** Root fix, one edit that repairs s1 and s2 together: in s1-git-and-packaging/packaging-and-tests.md:71-72 replace
```
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
```
with
```
[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]
```
(uv syncs the `dev` group by default), and change lab-1.md:51's check from ``uv pip install -e ".[dev]"`` to ``uv sync``. If the extras form must stay, then instead change lab-2.md:62 to `uv run --extra dev pytest -v`, lab-2.md:77 to `uv run --extra dev pytest`, guardrails-and-review.md:188 to `uv run --extra dev pytest`, and context-engineering.md:75 to ``- Test: `uv run --extra dev pytest` ``.

---

## 3. [blocker / competition / fix in courseware] python-ai-engineering (course 14), competitions 179 180 181 182

**Problem.** All four course-14 competitions are is_public=False, so every student-facing surface for them fails. The two best baselines in the entire audit (181's F1=0.656 bar, 182's 91.3% baseline / 97% target) are the ones no student can read.

**Evidence.**
```
With the student token: GET /api/competitions/179 -> 404 (same for 180/181/182); GET /api/competition_asset/179/markdown/overview -> 500; GET /api/leaderboard/competition/179 -> 500; GET /api/competitions/179/datasets -> 500. With the creator token, creator_competitions() returns {'id': 179, ..., 'is_public': False, 'is_started': True, 'role': 'owner'} for all four. Anonymous GET is also 404.
```

**Proposed fix.** Run `python tools/build_competitions.py publish` (courseware/competitions/README.md: "Competitions are created hidden (is_public=False) and started. Flip them public when the course is ready"), or PUT is_public=true on 179-182 with the creator token. Then re-verify with the student token that /api/competitions/179 returns 200. Until this is done nothing else in course 14 can be assessed by a student.

---

## 4. [blocker / competition / fix in courseware] python-ai-engineering/ (all 4 sessions; competitions 179,180,181,182)

**Problem.** Every competition in the 12h course is invisible to students. All four were created with is_public=False and never flipped, so the only gradeable artefact of the whole course 404s for an enrolled student.

**Evidence.**
```
Student token mlk_user_...: `c.competition(179)` -> CompetitionNotFoundError: Not Found; same for 180, 181, 182. Creator token reaches all four (HTTP 200 on /api/competition_asset/179/markdown/overview). courseware/README.md 'Known gaps' already records this: "Ids 179-182 are live, benchmarked and attached to modules 14-17 ... But they are still is_public=False, so enrolled students get a 404".
```

**Proposed fix.** Run `MLARENA_API_KEY=mlk_creator_... make competitions-publish` in courseware/ (i.e. `python tools/build_competitions.py publish`), after setting the real term dates in content/python-ai-engineering/course.yaml. Then re-verify with the student token that competition(179..182) returns 200.

---

## 5. [blocker / competition / fix in platform] competitions 179, 180, 181, 182 (course 14, sessions 1-4)

**Problem.** All four PAIE competitions 404 to a student token, so the single validation surface that already produces an objective number is unreachable for the entire 12-hour course. Any design that folds competition score into progress is dead here until they are flipped public.

**Evidence.**
```
Live with the student key: `c.competition(179/180/181/182)` -> `CompetitionNotFoundError: Not Found` for all four. The module payloads carry the same error inline, e.g. student_view/python-ai-engineering/s1-git-and-packaging/_module.json: {"competition_id": 179, ..., "error": "CompetitionNotFoundError: Not Found"}. Cross-check: all 17 competitions attached to course 15 return is_public=True, is_started=True.
```

**Proposed fix.** Run `make competitions-publish` in tmp/data_science_practice/courseware (it exists and flips is_public — Makefile target `competitions-publish`), then re-run `python tools/student_walk.py check --course python-ai-engineering` and confirm zero `competition-unreachable` lines. Until that runs, do not author any `mlarena:target` block against 179-182: an unreachable target renders a red light the student cannot clear.

---

## 6. [major / competition / fix in both] competition 180 / s2-agentic-coding module page

**Problem.** The module page advertises competition 180 by name to every student, but opening it 404s and its leaderboard 500s. Because the pinned Flesch spec lives only in that overview, the unreachable page is not an optional extra for this module — it is the only authoritative statement of what Lab 2's tests are supposed to encode.

**Evidence.**
```
With the student token: `c.module_overview("python-ai-engineering","s2-agentic-coding")["competitions"]` → `[{"competition_id": 180, "label": "Flesch reading-ease", "name": "PAIE S2 — Flesch reading-ease"}]`, while `c.competition(180)` → `CompetitionNotFoundError: Not Found` and `c.leaderboard(180)` → `500 Server Error ... /api/leaderboard/competition/180`. The same 404 is recorded in the module dump: `_module.json` → `"error": "CompetitionNotFoundError: Not Found"`.
```

**Proposed fix.** Run `python tools/build_competitions.py publish` to flip 180 to `is_public=True` before the 2026-09-07 start date, and add to lab-2.md a Part F that links it: a ```mlarena:competition id=180``` directive block (the resolver already supports `competition`, `leaderboard` and `submit` types — backend/app/services/lesson_directives.py:203) plus the sentence "Submit `agent.py` and `readability.py` with `c.submit(180, files=[\"agent.py\", \"readability.py\"])`; a completed lab scores 1.0."

---

## 7. [major / baseline / fix in courseware] competition 180 (courseware/competitions/s2-readability/overview.md)

**Problem.** The overview defines the metric precisely but never states, in a number, what score means "I have done the lab". The only starter a student is handed scores 0.0 by construction, so there is no reproducible reference point either. The reference value 1.0 exists only in config.py, which students never see.

**Evidence.**
```
overview.md "## Scoring" says only: "Twenty hidden texts. A text counts as passed when your answer is within `1e-6` of the reference. Score is the fraction passed." No target appears anywhere in the file. `python competitions/localtest.py s2-readability --agent agent_template.py` → `"score": 0.0`, `"info_message": "0/20 texts within 1e-06 before the agent crashed. case #0 failed: NotImplementedError: "`. The reference solution scores `"score": 1.0`. `config.py` has `"benchmark_expected_score": 1.0` but is a creator-side file.
```

**Proposed fix.** Append to overview.md's "## Scoring" section: "Score runs from 0.0 to 1.0, higher is better. The reference implementation of the spec above scores **1.0** — every one of the twenty texts matches. Anything below 1.0 means your implementation and the spec disagree somewhere, and `mean absolute error` tells you whether it is a rounding detail (< 1) or a wrong rule (> 5). The starter you are given scores 0.0: it raises `NotImplementedError`. **A completed Lab 2 scores 1.0.**"

---

## 8. [major / content / fix in courseware] s2-agentic-coding/setup

**Problem.** The install command requires Node.js, which the course never mentions, never installs, and never lists as a prerequisite — and the lesson's own speaker note names this as the thing students get stuck on. The macOS/Linux native installer, which needs no Node at all, is not offered.

**Evidence.**
```
setup.md:17 is `npm install -g @anthropic-ai/claude-code`. `grep -rniE "\bnode(\.js)?\b|npm|nvm"` across all 29 published lesson bodies returns exactly three hits: this line, `each node.` in s4/autograd.md, and setup.md:7 — the speaker note `<!-- notes: 20 minutes, hands on keyboards. Walk the room. The students who get stuck get stuck on Node or on auth, not on the concepts. -->`. The course description lists prerequisites as "It assumes you already write Python" and nothing else.
```

**Proposed fix.** Insert before setup.md:17: "`npm` comes with Node.js 18+. If `node --version` prints nothing, either install Node from nodejs.org first, or skip npm entirely and use the native installer: `curl -fsSL https://claude.ai/install.sh | bash` (macOS/Linux). Neither route needs an existing Node install if you use the second one."

---

## 9. [major / structure / fix in courseware] s2-agentic-coding/setup (position 1) vs s2-agentic-coding/guardrails-and-review (position 4)

**Problem.** The first thing the session has a student do with an agent is issue an editing prompt against their graded Lab 1 repository, with no branch and no clean-tree check. The rules that prevent that arrive 95 minutes and three lessons later, in a lesson whose own note says this is how students lose work.

**Evidence.**
```
setup.md:45-52 "## First contact / Try these, in order, in your Session 1 repository: `> what does this project do?` `> add a docstring to every public function in src/`". guardrails-and-review.md, position 4: "```bash\ngit switch -c feature/agent-readability   # never work on main\n```" and "## Commit before you delegate / A clean working tree before you start means `git diff` shows exactly what the agent did", plus the note "<!-- notes: Every student who loses work this term will have been on main with uncommitted changes. Say it now. -->".
```

**Proposed fix.** Insert two lines above setup.md's `> what does this project do?` block: "First: `git status` must be clean, and `git switch -c agent-sandbox` so nothing here lands on `main`. *Guardrails & Review* explains why; do it now anyway." And change "Try these, in order, in your Session 1 repository:" to "Try these, in order, on that branch:".

---

## 10. [major / validation / fix in courseware] courseware/tools/build_competitions.py:121-140 (verify_benchmark) and :173 (set_competition_markdown)

**Problem.** The build's one guarantee is opt-in and does not cover the page. `verify_benchmark` returns silently when `benchmark_expected_score` is absent, and nothing checks that the number appears in overview.md — which is why 179 and 180 have a machine-verified 1.0 on the server that never reaches the student.

**Evidence.**
```
build_competitions.py:123-126: `expected = cfg.get("benchmark_expected_score")` / `if expected is None: return`. Line 173 publishes the page unconditionally: `client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())`, and it runs *before* the benchmark at :242. Live proof the number exists but is unpublished: benchmark_status(179) -> score=[1.0], benchmark_status(180) -> score=[1.0], yet neither overview.md contains the string "1.0".
```

**Proposed fix.** Two edits. (1) Replace `if expected is None: return` with `raise RuntimeError(f"{cfg['name']}: config.py declares no benchmark_expected_score")` so the guarantee is mandatory, not opt-in. (2) Add `verify_overview_contract(cfg, overview_text)` called immediately before line 173: parse the Baseline block, and refuse to publish unless the metric name equals `cfg["metric"]`, the direction word is present, and the block's Baseline/Reference/Passing numbers equal `cfg["baseline_score"]` / `cfg["benchmark_expected_score"]` / `cfg["target_score"]` within `cfg["benchmark_score_tol"]`. Add `baseline_score`, `target_score` and `metric_direction` as required keys to all four config.py files. The failure message should read like the existing one: "the env does not grade what the package claims" becomes "the page does not state what the package grades".

---

## 11. [major / platform / fix in platform] backend/app/services/lesson_directives.py:113-131, 133-176, 180-196

**Problem.** All three directive resolvers load competitions with `Competition.query.get(...)` and run the leaderboard query with no visibility check, so an ```mlarena:competition id=179``` or ```mlarena:leaderboard id=179``` block in a public course lesson would hand an anonymous reader the hidden competition's name, description and full standings — the exact data /api/competitions/179 refuses. No lesson uses a directive today, so this is latent, but it becomes live the moment the courseware adopts them (see the finding below).

**Evidence.**
```
lesson_directives.py:118-119 `competition = Competition.query.get(competition_id)` with no user_can_see_competition call; :157-163 build_leaderboard_query is called with only competition_id/is_elo_ranked/aggregate; the returned payload at :127-130 includes name, description, is_public.
```

**Proposed fix.** In _resolve_competition and _resolve_submit, replace `Competition.query.get(competition_id)` with a lookup that then checks `user_can_see_competition(competition)` (import from app.views.creator_competition._helpers) and raises DirectiveError('competition <n> is not visible to this reader') on failure — which the non-strict consumption path already turns into a dropped-with-warning block. Gate _resolve_leaderboard the same way before calling build_leaderboard_query.

---

## 12. [major / content / fix in courseware] courseware/content/python-ai-engineering/s1-git-and-packaging/lab-1.md, s2-agentic-coding/lab-2.md, s3-data-science-nutshell/lab-3.md

**Problem.** Labs 1-3 never mention the competition attached to their own module, never mention ML-Arena, and never say anything is submitted. The module card advertises 'textstats correctness' / 'Flesch reading-ease' / 'Adult Census Income' while the lab it belongs to says the deliverable is a GitHub URL. Zero of the ~102 published lessons across both courses uses the ```mlarena:submit``` / ```mlarena:competition``` / ```mlarena:leaderboard``` directive blocks the platform built for exactly this.

**Evidence.**
```
`grep -rn '```mlarena' student_view/` -> 0 matches; SDK `lesson(...)['directives']` is `[]` and `directive_warnings` is `[]` for lab-1. `grep -rli 'ml-arena|mlarena' student_view/python-ai-engineering/` matches only lab-4.md, github-actions.md and three s3 lessons — not lab-1, lab-2 or lab-3. lab-1.md's Grading table lists four criteria, none of them a leaderboard score.
```

**Proposed fix.** Add a closing 'Part F — Submit' section to lab-1.md, lab-2.md and lab-3.md containing the competition's baseline number in prose plus a ```mlarena:submit id=179``` (resp. 180, 181) block, and add the competition row to each lab's Grading table. Do the same for the MS2A lab-N.md files, which attach 17 competitions across ten sessions. Republish with `make publish`.

---

## 13. [major / platform / fix in both] COVERAGE HOLE — backend/app/services/lesson_directives.py, frontend/src/components/Course/directiveCards.tsx, courseware/tools/build_slides.py

**Problem.** Finding 180 recommends lesson directives as the primary design for the validation layer. Nobody checked whether the feature has ever run. It has not — not in these two courses, and not anywhere on the platform — and three known defects sit directly in its path, so the recommendation is being made about untested machinery.

**Evidence.**
```
`SELECT count(*) FROM lesson WHERE body_md LIKE '%mlarena:%'` = 0 across the whole lesson table, not just courses 14/15. The three defects: build_slides.py:76 FENCE_RE is `^```([\w+-]*)\s*$`, which cannot match an info string containing `:` or a space, so an opening ```` ```mlarena:checkpoint ```` fence never sets in_fence while the closing bare ``` does — every subsequent `---` is swallowed and the deck collapses (finding 189, confirmed by reading split_slides at :92-104). directiveCards.tsx:154-169 matches a fence to a payload by competition_id then falls back to the first directive of that type (finding 190). lesson_directives.py:113-131 resolves `Competition.query.get(...)` with no visibility check (finding 168).
```

**Proposed fix.** Before committing to option (b), prototype one directive on one lesson through `POST /api/teacher/lessons/<id>/preview` (teacher/lessons.py:241, strict mode) and rebuild that module's deck. Fix build_slides.py's FENCE_RE to `^```(\S.*)?$` and directiveCards' matcher first — both are small, and both are prerequisites, not follow-ups. Note the resolver already returns `is_public` and `is_started`, so a competition directive can carry the reachability signal finding 159 says the module card lacks.

---

## 14. [minor / content / fix in courseware] s2-agentic-coding/setup

**Problem.** The lesson names `/config` as the way to switch models and never mentions `/model`, which is the actual command; the Guardrails lesson likewise sends students to hand-edit `.claude/settings.json` without mentioning `/permissions`, which does the same job from inside the session. In a 20-minute hands-on-keyboard lesson these cost real minutes.

**Evidence.**
```
setup.md:79-86 "## Model choice … `/config` switches between them." The "Useful from minute one" table lists `/init`, `/clear`, `/config`, `#`, `!`, `Esc`, `/help` and no `/model`. Both commands exist in the shipped CLI: `strings /Users/raphaelcousin/.local/share/claude/versions/2.1.258 | grep -oE '"/(model|permissions|config)"'` → `"/config"`, `"/model"`, `"/permissions"` (Claude Code 2.1.258). The model names in the lesson (Opus 5, Sonnet 5, Haiku 4.5) are all correct — only the command is wrong.
```

**Proposed fix.** In setup.md, add a row to the commands table: `| \`/model\` | Switch model (Opus 5 / Sonnet 5 / Haiku 4.5) |`, and change "`/config` switches between them." to "`/model` switches between them; `/config` holds everything else." In guardrails-and-review.md, above the settings.json block, add: "`/permissions` edits this file from inside the session — it is the same list."

---

## 15. [minor / structure / fix in courseware] s2-agentic-coding/lab-2 (Part E) and course.yaml estimated_minutes

**Problem.** Part E budgets 5 minutes for four written retrospective sections plus push, open PR, obtain a partner's review, and merge — a step that requires a second human to context-switch. The session's per-lesson minutes also sum to exactly 180 for a 3-hour slot, so there is no slack anywhere to absorb the overrun, and the split is 135 lecture / 45 lab against a course description that promises "Every session is half lecture, half lab."

**Evidence.**
```
lab-2.md: "## Part E — Retrospective + PR (5 min)" followed by a four-heading RETRO.md template and "Push, open the PR, get your partner's review, merge." course.yaml s2 lessons: 20 + 20 + 35 + 30 + 30 + 45 = 180 minutes. _course.json description: "Every session is half lecture, half lab."
```

**Proposed fix.** Rebalance inside the existing 180: cut Setup 20→15 (it is an install plus four checkboxes) and The Core Loop 35→30, and give Lab 2 45→55 with Part E at 15 min. If the minutes cannot move, change Part E's heading to "## Part E — Retrospective (5 min) + PR (homework)" and move "get your partner's review, merge" out of the timed block.

---

## 16. [minor / structure / fix in courseware] content/python-ai-engineering/course.yaml (module s1-git-and-packaging)

**Problem.** Session 1 is budgeted at 230 minutes inside a 3-hour (180-minute) session, and the split is 170 minutes of lecture to 60 of lab — not the "half lecture, half lab" the course description promises. Either the lab or ~50 minutes of lecture will be cut live, and the lab is the part that is graded, so students will most likely be sent home to do Parts D and E unsupervised, which is where the pair-review and conflict-resolution learning actually is.

**Evidence.**
```
course.yaml:45,50,55,60,65,70 estimated_minutes: 10 + 45 + 50 + 30 + 35 + 60 = 230.
course.yaml:23-24 course description: "A 12-hour mise à niveau in four 3-hour sessions… Every session is half lecture, half lab."
(Same arithmetic from the served index: `awk -F'\t' '$1 ~ /^s1-/ {…}' _index.tsv` -> TOTAL 230 min.)
```

**Proposed fix.** Bring the lecture block to ~110 minutes and the lab to ~70, so 180 holds and the halves are honest: cut `git-essentials` 45 -> 35 by moving Install + Authenticating-with-GitHub into a short pre-session "Before Session 1" reference lesson (they are setup, not teaching); cut `branching-and-collaboration` 50 -> 40 by moving the Naming and Keeping-a-branch-current sections to the existing paie-reference/git-cheatsheet; cut `packaging-and-tests` 35 -> 30 by dropping Coverage to reference. Update the six `estimated_minutes` in course.yaml to 10/35/40/30/30/70 (= 215… trim `why-version-control` to 5 for 180 if you want it exact).

---

## 17. [minor / structure / fix in courseware] courseware/content/python-ai-engineering/course.yaml lines 131-157

**Problem.** The session does not fit its slot and does not match the split the course advertises. Estimated minutes sum to 185 in a 3-hour (180 min) session with no break, and the lecture/lab ratio is 76/24 rather than the promised half and half. In practice the lab is what gets cut — and the lab is what the competition depends on.

**Evidence.**
```
course.yaml s3 block: estimated_minutes 30 + 35 + 40 + 35 + 45 = 185. _course.json description: "A 12-hour mise à niveau in four 3-hour sessions … Every session is half lecture, half lab." 140/185 = 76% lecture. lab-3's own parts sum to exactly 45 (5+5+15+10+10) with zero slack, and Part D's 10 minutes must absorb the sparse-input TypeError above; `fetch_openml("adult", version=2)` alone took 50 s here on a warm connection, before thirty students hit openml.org at once.
```

**Proposed fix.** Move 20 minutes from lecture to lab in course.yaml: the-ml-pipeline 30→25 (move "Where the time actually goes" to paie-reference), evaluation-metrics 40→30 (drop "## Ranking metrics" — MAP/NDCG are never used again in this course and never appear in Lab 3), validation-and-overfitting 35→30, lab-3 45→60 to cover Parts A–F. New total 25+35+30+30+60 = 180 exactly, with 60 minutes of lab.

---

