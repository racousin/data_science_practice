# Audit findings — pae-s1

19 findings

## 1. [blocker / validation / fix in courseware] s2-agentic-coding/lab-2 (root cause in s1-git-and-packaging/packaging-and-tests:71)

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

## 2. [blocker / validation / fix in courseware] s1-git-and-packaging/packaging-and-tests (+ lab-1 grading table)

**Problem.** The reproduce-and-verify sequence the module states three times — `uv sync` then `uv run pytest` — cannot work with the pyproject.toml the same module teaches. pytest is declared as a `[project.optional-dependencies]` extra, and `uv sync` does not install extras. Worse, the two install commands the session teaches actively undo each other: `uv sync` uninstalls the pytest that `uv pip install -e ".[dev]"` put in. This is the criterion worth 30% of the lab grade, so every pair that follows the lesson literally fails the heaviest grading axis on a fresh clone.

**Evidence.**
```
packaging-and-tests.md:71-72 declares `[project.optional-dependencies]` / `dev = ["pytest>=8", "ruff>=0.6"]`; :84-85 teach `uv pip install -e ".[dev]"`; :257-258 and python-environments.md:158-159 both give `uv sync` / `uv run pytest`; lab-1.md:101 grades `| uv sync && uv run pytest green on a fresh clone | 30% |`.
Built that exact project in /tmp/paie-lab1, committed, then:
  $ git clone /tmp/paie-lab1 /tmp/paie-lab1-clone2 && cd /tmp/paie-lab1-clone2 && uv sync && uv run pytest
  Installed 1 package in 3ms
   + textstats==0.1.0 (from file:///private/tmp/paie-lab1-clone2)
  error: Failed to spawn: `pytest`
    Caused by: No such file or directory (os error 2)   [exit 2]
And in the working copy where pytest WAS installed:
  $ uv sync
  Uninstalled 5 packages in 91ms
   - pytest==9.1.1  (…iniconfig, packaging, pluggy, pygments)
  $ uv run --no-sync pytest -q  ->  error: Failed to spawn: `pytest`
With the fix applied it passes: `uv sync && uv run pytest -q` -> `9 passed in 0.04s` (uv 0.10.12).
```

**Proposed fix.** In packaging-and-tests.md, replace lines 71-72
  [project.optional-dependencies]
  dev = ["pytest>=8", "ruff>=0.6"]
with
  [dependency-groups]
  dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]
(uv installs dependency-groups on a bare `uv sync`; verified green). Then replace line 85 `uv pip install -e ".[dev]"    # with the dev extras` with `uv pip install -e . --group dev   # with the dev tools` — note `uv pip install -e ".[dev]"` against a dependency-group only prints `warning: … does not have an extra named 'dev'` and installs nothing, which is precisely the silent failure the course's own fail-fast section warns about. Mirror the same change in lab-1.md:51. If you prefer to keep extras, then every `uv sync` in the module (python-environments.md:137,158,204; packaging-and-tests.md:257; lab-1.md:101) must become `uv sync --extra dev`.

---

## 3. [blocker / competition / fix in both] s1-git-and-packaging/lab-1 + competition 179

**Problem.** Session 1's attached competition is invisible from the session. No lesson in s1 contains the words competition, leaderboard, ML-Arena or submit; lab-1 has no submission part; and its stated deliverable ("a GitHub repository URL") names no place to hand it in. Meanwhile competition 179 — whose own overview calls itself "the leaderboard half of 'ship a package'" and is the only external check on whether the three functions are right — 404s to a student token. The net effect on this module is that the exercise has no external oracle and no hand-in path at all.

**Evidence.**
```
$ grep -rniE 'ml-arena|mlarena|leaderboard|competition|submit' student_view/python-ai-engineering/s1-git-and-packaging/*.md
  branching-and-collaboration.md:228:alongside leaderboard performance.   <- the only hit, and it is about the project grade
By contrast s4-pytorch-nutshell/lab-4.md:84 has a whole "## Part D — Submit to ML-Arena (10 min)" with `client.submit(competition_id=<id>, path="submission.csv")`, and lab-4.md:128 grades `| ML-Arena submission accepted | 10% |`.
With the student key: `c.competition(179)` -> `CompetitionNotFoundError: Not Found`; `c.datasets(179)` -> `datasets failed`; and _module.json records the same: `"error": "CompetitionNotFoundError: Not Found"`.
Also: no lesson in the entire course uses the platform's embed directives — `grep -rn 'mlarena:' student_view/python-ai-engineering/` returns nothing, and `c.lesson(...)` returns `"directives": []` for every s1 lesson.
```

**Proposed fix.** Two halves. (courseware) Add a "## Part F — Put it on the leaderboard (10 min)" to lab-1.md before the Grading table, and give the competition a graded row (e.g. `| textstats leaderboard entry | 10% |`, rebalancing the other four). The platform resolves these fenced directives server-side (backend/app/services/lesson_directives.py:203-205), so the body can be literally:
  ```mlarena:competition id=179
  ```
  Upload `agent.py` wrapping the `core.py` you just wrote — the agent directory is on `sys.path`, so `from core import word_count, char_frequencies, longest_word` works.
  ```mlarena:submit id=179
  ```
and repeat the three-line SDK snippet from lab-4.md:94-98 with the user-scoped key, since Session 1 is the first time a student is asked to submit anything. (platform) Flip 179 to is_public=True — `python tools/build_competitions.py publish` per courseware/competitions/README.md — otherwise Part F 404s.

---

## 4. [blocker / competition / fix in courseware] python-ai-engineering (course 14), competitions 179 180 181 182

**Problem.** All four course-14 competitions are is_public=False, so every student-facing surface for them fails. The two best baselines in the entire audit (181's F1=0.656 bar, 182's 91.3% baseline / 97% target) are the ones no student can read.

**Evidence.**
```
With the student token: GET /api/competitions/179 -> 404 (same for 180/181/182); GET /api/competition_asset/179/markdown/overview -> 500; GET /api/leaderboard/competition/179 -> 500; GET /api/competitions/179/datasets -> 500. With the creator token, creator_competitions() returns {'id': 179, ..., 'is_public': False, 'is_started': True, 'role': 'owner'} for all four. Anonymous GET is also 404.
```

**Proposed fix.** Run `python tools/build_competitions.py publish` (courseware/competitions/README.md: "Competitions are created hidden (is_public=False) and started. Flip them public when the course is ready"), or PUT is_public=true on 179-182 with the creator token. Then re-verify with the student token that /api/competitions/179 returns 200. Until this is done nothing else in course 14 can be assessed by a student.

---

## 5. [blocker / platform / fix in platform] backend/app/views/competition_asset.py:19-46, backend/app/views/competitions.py:918-960, backend/app/views/leaderboard.py:75-77

**Problem.** Three student-facing routes wrap `get_visible_competition_or_404()` in a bare `except Exception` and return 500. A legitimate visibility 404 is reported to the student as a server error, which is what a course-14 student hits on every one of 179-182 and which violates the repo's own Fail Fast rule ("No silent try/except").

**Evidence.**
```
get_visible_competition_or_404 calls `abort(404)` (_helpers.py:212), which raises werkzeug NotFound — a subclass of Exception. competition_asset.py:22 `get_visible_competition_or_404(competition_id)` sits inside `try:` whose handler at :41-46 returns `jsonify({"error": "Failed to read the competition overview markdown"}), 500`. Measured: student token on /api/competition_asset/179/markdown/overview -> 500, /api/competitions/179/datasets -> 500, /api/leaderboard/competition/179 -> 500, while /api/competitions/179 (which lets the abort propagate) correctly returns 404.
```

**Proposed fix.** In all three handlers, hoist `get_visible_competition_or_404(competition_id)` above the `try:` block, or add `except HTTPException: raise` as the first handler so werkzeug aborts propagate untouched. Also drop the overview route's silent creation of DEFAULT_MARKDOWN_CONTENT on GET (competition_asset.py:28-34): a read request should not write a placeholder file, and that write is what makes 169/170 look like they have an overview.

---

## 6. [blocker / competition / fix in courseware] python-ai-engineering/ (all 4 sessions; competitions 179,180,181,182)

**Problem.** Every competition in the 12h course is invisible to students. All four were created with is_public=False and never flipped, so the only gradeable artefact of the whole course 404s for an enrolled student.

**Evidence.**
```
Student token mlk_user_...: `c.competition(179)` -> CompetitionNotFoundError: Not Found; same for 180, 181, 182. Creator token reaches all four (HTTP 200 on /api/competition_asset/179/markdown/overview). courseware/README.md 'Known gaps' already records this: "Ids 179-182 are live, benchmarked and attached to modules 14-17 ... But they are still is_public=False, so enrolled students get a 404".
```

**Proposed fix.** Run `MLARENA_API_KEY=mlk_creator_... make competitions-publish` in courseware/ (i.e. `python tools/build_competitions.py publish`), after setting the real term dates in content/python-ai-engineering/course.yaml. Then re-verify with the student token that competition(179..182) returns 200.

---

## 7. [blocker / competition / fix in platform] competitions 179, 180, 181, 182 (course 14, sessions 1-4)

**Problem.** All four PAIE competitions 404 to a student token, so the single validation surface that already produces an objective number is unreachable for the entire 12-hour course. Any design that folds competition score into progress is dead here until they are flipped public.

**Evidence.**
```
Live with the student key: `c.competition(179/180/181/182)` -> `CompetitionNotFoundError: Not Found` for all four. The module payloads carry the same error inline, e.g. student_view/python-ai-engineering/s1-git-and-packaging/_module.json: {"competition_id": 179, ..., "error": "CompetitionNotFoundError: Not Found"}. Cross-check: all 17 competitions attached to course 15 return is_public=True, is_started=True.
```

**Proposed fix.** Run `make competitions-publish` in tmp/data_science_practice/courseware (it exists and flips is_public — Makefile target `competitions-publish`), then re-run `python tools/student_walk.py check --course python-ai-engineering` and confirm zero `competition-unreachable` lines. Until that runs, do not author any `mlarena:target` block against 179-182: an unreachable target renders a red light the student cannot clear.

---

## 8. [major / content / fix in courseware] s1-git-and-packaging/lab-1 (Part B) vs competitions/s1-textstats/overview.md

**Problem.** The lab specifies the three functions loosely enough that two correct-looking implementations disagree, and the precise spec lives only in the competition overview a student cannot open. `longest_word` gives no tie-break rule; `char_frequencies` says "ignoring whitespace and case" but never says punctuation and digits count. Since Part C asks students to write their own tests, a wrong reading of the spec produces a green suite that certifies the wrong behaviour — the exact failure the module is meant to prevent.

**Evidence.**
```
lab-1.md:44-48 is the entire spec a student gets:
  def char_frequencies(text: str) -> dict[str, int]:
      """Count of each character, ignoring whitespace and case."""
  def longest_word(text: str) -> str:
      """The longest token. Raises ValueError on empty input."""
overview.md pins what the lab omits: "**Tokens** are whitespace-separated and keep their punctuation. `"the end."` is two tokens, and the second one is `end.` — four characters." … "Punctuation and digits are characters and do count." … "**Ties go to the first one** in the text: `longest_word("aaaa bbbb")` is `"aaaa"`."
The divergence is real, not theoretical:
  $ python3 -c "t='aaaa bbbb'.split(); print(repr(max(t,key=len)), repr(sorted(t,key=len)[-1]))"
  'aaaa' 'bbbb'
Both are defensible readings of "the longest token"; one scores 0 on 20/20 tie cases.
```

**Proposed fix.** Paste the three rules into lab-1.md Part B, as a block directly under the three signatures: "**Specification.** Tokens are whitespace-separated and keep their punctuation — `\"the end.\"` is two tokens and the second is `end.` (four characters). `char_frequencies` folds case and drops whitespace; punctuation and digits are characters and do count, so `\"Aa b\"` gives `{\"a\": 2, \"b\": 1}`. `longest_word` breaks ties by taking the **first** token of maximal length: `longest_word(\"aaaa bbbb\") == \"aaaa\"`." Then add to Part C's required-test list a fourth bullet: "- one test for each of the three specification rules above (punctuation kept in a token, digits counted, tie taken from the left)".

---

## 9. [major / validation / fix in courseware] s1-git-and-packaging/lab-1 (Part B check) and packaging-and-tests.md:94

**Problem.** Part B's self-check fails when the student has done everything correctly. `python -c "import textstats"` run from the home directory uses whatever interpreter is on PATH — not the project `.venv` the package was just installed into — so it raises ModuleNotFoundError; on a stock macOS `python` is not even on PATH. The lesson's own advice contradicts the check ('`uv run` … guarantees the command runs in the project environment, with no activation state to get wrong', python-environments.md).

**Evidence.**
```
lab-1.md:51-52: "**Check:** `uv pip install -e \".[dev]\"` succeeds, and `python -c \"import textstats\"` works from your home directory."
  $ cd /tmp/paie-lab1 && uv venv && uv pip install -e ".[dev]"   -> Installed 6 packages … + textstats==0.1.0
  $ cd ~ && python3 -c "import textstats; print(textstats.__file__)"
  ModuleNotFoundError: No module named 'textstats'
  $ cd ~ && bash -c 'source /tmp/paie-lab1/.venv/bin/activate && python -c "import textstats; print(textstats.__file__)"'
  /private/tmp/paie-lab1/src/textstats/__init__.py
Same defect at packaging-and-tests.md:94, `python -c "import my_project; print(my_project.__file__)"`, presented as "Check it worked".
```

**Proposed fix.** lab-1.md:51-52 -> "**Check:** `uv pip install -e . --group dev` succeeds, and from *outside* the project directory `uv run --project ~/paie-lab1-... python -c \"import textstats; print(textstats.__file__)\"` prints a path under your `src/`. (Plain `python` from your home directory will not see it — it is a different interpreter; that is what the environment is for.)" packaging-and-tests.md:93-95 -> replace the bare `python -c ...` with `uv run python -c "import my_project; print(my_project.__file__)"`.

---

## 10. [major / platform / fix in platform] backend/app/views/leaderboard.py:76 and :205-210

**Problem.** `GET /api/leaderboard/competition/<id>` returns HTTP 500 for a competition the caller cannot see, and for one that does not exist. `get_visible_competition_or_404()` calls `abort(404)`, but it is invoked inside the route's `try:` block and the bare `except Exception` at the bottom swallows the werkzeug NotFound and rewrites it as 500. Concretely for this module: `leaderboard(179)` gives a student a server error rather than the 404 that would tell them the competition is not published yet, and it makes hidden-vs-broken indistinguishable while debugging the course rollout.

**Evidence.**
```
$ curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer mlk_user_…' https://ml-arena.com/api/leaderboard/competition/179  ->  500  {"error":"Failed to fetch leaderboard"}
$ … /api/leaderboard/competition/99999  ->  500  {"error":"Failed to fetch leaderboard"}   (nonexistent id, same 500)
$ … /api/leaderboard/competition/176   ->  200  [{"AgentName":"rf-baseline",…}]   (public control)
backend/app/views/leaderboard.py:75-77:
    def get_leaderboard(competition_id):
        try:
            get_visible_competition_or_404(competition_id)
backend/app/views/creator_competition/_helpers.py:210-213 -> `abort(404)`.
backend/app/views/leaderboard.py:205-210 -> `except Exception as e: … return jsonify({"error": "Failed to fetch leaderboard"}), 500`.
```

**Proposed fix.** Move the visibility check out of the try in backend/app/views/leaderboard.py: put `get_visible_competition_or_404(competition_id)` on the line immediately before `try:` (line 76 -> line 75). Alternatively, if it must stay inside, add `except HTTPException: raise` (from werkzeug.exceptions) as the first handler before `except Exception`. The same swallow-the-abort shape exists at backend/app/views/teacher/leaderboard.py:206 and should get the same treatment.

---

## 11. [major / content / fix in courseware] s1-git-and-packaging/git-essentials.md:262

**Problem.** The lesson's closing pointer to the Git cheatsheet is a relative link that resolves to a nonexistent lesson. Lesson URLs are `/courses/{slug}/{module}/{kind}/{lesson}`, so from the git-essentials page `../paie-reference/git-cheatsheet` resolves to `/courses/python-ai-engineering/s1-git-and-packaging/paie-reference/git-cheatsheet` — which still matches the route, with moduleSlug=`s1-git-and-packaging` and lessonSlug=`git-cheatsheet`, and 404s. The cheatsheet is the one artefact the lab tells students to keep open, and it is one relative segment too shallow.

**Evidence.**
```
git-essentials.md:261-262: "The full cheatsheet, including everything in the next lesson, is in [Reference → Git Cheatsheet](../paie-reference/git-cheatsheet)."
Route: frontend/src/App.js:151 `/courses/:slug/:moduleSlug/:kind/:lessonSlug`; frontend/src/components/Course/courseNavigation.ts:16-30 `kindToSegment` -> 'course' for kind 'lesson'. CourseMarkdown.tsx overrides h1-h3/div/pre/code only — no `a` component — so the anchor is a plain browser-resolved relative href.
The resulting lookup, via the SDK with the student key:
  c.lesson('python-ai-engineering','s1-git-and-packaging','git-cheatsheet') -> Lesson not found
  c.lesson('python-ai-engineering','paie-reference','git-cheatsheet')       -> 200, id 54, module_id 18
```

**Proposed fix.** In courseware/content/python-ai-engineering/s1-git-and-packaging/git-essentials.md:262 replace `(../paie-reference/git-cheatsheet)` with the absolute path `(/courses/python-ai-engineering/paie-reference/course/git-cheatsheet)`. Prefer absolute course paths for every cross-module link — relative ones cannot be written correctly because the `{kind}` segment sits between the module and the lesson slug. (Platform follow-up, optional: give CourseMarkdown an `a` component that rewrites `../<module>/<lesson>` to the canonical shape, so authors can keep writing the intuitive form.)

---

## 12. [major / platform / fix in both] s4-pytorch-nutshell/autograd:5 (and s1-git-and-packaging/git-essentials:262)

**Problem.** The two markdown links that connect the sessions to the Reference module both point at a URL shape the router does not have, so both are dead. The real lesson route carries a `kind` segment (`course` or `exercise`) between the module slug and the lesson slug, which a `../`-relative link cannot account for. These are the only two inbound links to paie-reference in the whole course, so the module is unreachable by navigation.

**Evidence.**
```
autograd.md:5: "...not the mathematics, which is in [Reference → Autograd, the Mathematics](../paie-reference/autograd-mathematics)."

frontend/src/App.js:151  <Route path="/courses/:slug/:moduleSlug/:kind/:lessonSlug" element={<LessonReader/>} />
frontend/src/components/Course/courseNavigation.ts:30
  return `/courses/${courseSlug}/${moduleSlug}/${kindToSegment(lesson.kind)}/${lesson.slug}`;
courseNavigation.ts:16-18  kindToSegment(kind) => kind === 'exercise' ? 'exercise' : 'course'

So this lesson lives at /courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd, and `../paie-reference/autograd-mathematics` resolves against the base dir .../s4-pytorch-nutshell/course/ to
  /courses/python-ai-engineering/s4-pytorch-nutshell/paie-reference/autograd-mathematics
— six segments, matching no route in App.js:148-151.

$ grep -rn -- '](\.\./' student_view/python-ai-engineering/
s4-pytorch-nutshell/autograd.md:5:...(../paie-reference/autograd-mathematics)
s1-git-and-packaging/git-essentials.md:262:...(../paie-reference/git-cheatsheet)
```

**Proposed fix.** Courseware side, now: make both links absolute. autograd.md:5 -> `[Reference → Autograd, the Mathematics](/courses/python-ai-engineering/paie-reference/course/autograd-mathematics)`; git-essentials.md:262 -> `[Reference → Git Cheatsheet](/courses/python-ai-engineering/paie-reference/course/git-cheatsheet)`.
Platform side, durable: relative cross-lesson links are a trap that every author will fall into, because the `kind` segment is invisible from the markdown. Either add a `mlarena:lesson module=<slug> lesson=<slug>` directive alongside the existing competition/leaderboard/submit types in backend/app/services/lesson_directives.py:250, or have the teacher-side validator (app/views/teacher/lessons.py:242, already run in strict mode) reject a relative link that does not resolve to a real lesson.

---

## 13. [major / validation / fix in courseware] courseware/tools/build_competitions.py:121-140 (verify_benchmark) and :173 (set_competition_markdown)

**Problem.** The build's one guarantee is opt-in and does not cover the page. `verify_benchmark` returns silently when `benchmark_expected_score` is absent, and nothing checks that the number appears in overview.md — which is why 179 and 180 have a machine-verified 1.0 on the server that never reaches the student.

**Evidence.**
```
build_competitions.py:123-126: `expected = cfg.get("benchmark_expected_score")` / `if expected is None: return`. Line 173 publishes the page unconditionally: `client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())`, and it runs *before* the benchmark at :242. Live proof the number exists but is unpublished: benchmark_status(179) -> score=[1.0], benchmark_status(180) -> score=[1.0], yet neither overview.md contains the string "1.0".
```

**Proposed fix.** Two edits. (1) Replace `if expected is None: return` with `raise RuntimeError(f"{cfg['name']}: config.py declares no benchmark_expected_score")` so the guarantee is mandatory, not opt-in. (2) Add `verify_overview_contract(cfg, overview_text)` called immediately before line 173: parse the Baseline block, and refuse to publish unless the metric name equals `cfg["metric"]`, the direction word is present, and the block's Baseline/Reference/Passing numbers equal `cfg["baseline_score"]` / `cfg["benchmark_expected_score"]` / `cfg["target_score"]` within `cfg["benchmark_score_tol"]`. Add `baseline_score`, `target_score` and `metric_direction` as required keys to all four config.py files. The failure message should read like the existing one: "the env does not grade what the package claims" becomes "the page does not state what the package grades".

---

## 14. [major / platform / fix in platform] mlarena-sdk/mlarena/client.py:161, :170, :192, :1338

**Problem.** competition(), competitions() and leaderboard() send no Authorization header, so an enrolled student using the SDK or MCP is treated as anonymous and cannot open any non-public course competition — the exact class that 179-182 belong to.

**Evidence.**
```
client.py:192 `resp = self._request("GET", self._url(f"/competitions/{competition_id}"), timeout=30)` — no `headers=self._headers()`, unlike every scoped call (e.g. :225 datasets passes it). Same at :161/:170 (competitions) and :1338 (leaderboard). Observed effect: `mlarena.connect(CREATOR).competition(179)` raises CompetitionNotFoundError, while the same creator token via raw requests with a bearer header returns 200 — the owner is being told their own competition does not exist. The backend's visibility_filter (_helpers.py:229-233) grants access on `is_public OR owned OR assistant`, and the enrolled-student side-channel in competitions.py:213-217 is likewise keyed on `current_user.is_authenticated`.
```

**Proposed fix.** Add `headers=self._headers()` to the four calls. These routes accept an anonymous caller, so the change is backward-compatible and only widens what an authenticated caller can see. Without it, publishing 179-182 still leaves them invisible to any SDK/MCP student even after enrollment is fixed.

---

## 15. [major / platform / fix in platform] backend/app/services/lesson_directives.py:113-131, 133-176, 180-196

**Problem.** All three directive resolvers load competitions with `Competition.query.get(...)` and run the leaderboard query with no visibility check, so an ```mlarena:competition id=179``` or ```mlarena:leaderboard id=179``` block in a public course lesson would hand an anonymous reader the hidden competition's name, description and full standings — the exact data /api/competitions/179 refuses. No lesson uses a directive today, so this is latent, but it becomes live the moment the courseware adopts them (see the finding below).

**Evidence.**
```
lesson_directives.py:118-119 `competition = Competition.query.get(competition_id)` with no user_can_see_competition call; :157-163 build_leaderboard_query is called with only competition_id/is_elo_ranked/aggregate; the returned payload at :127-130 includes name, description, is_public.
```

**Proposed fix.** In _resolve_competition and _resolve_submit, replace `Competition.query.get(competition_id)` with a lookup that then checks `user_can_see_competition(competition)` (import from app.views.creator_competition._helpers) and raises DirectiveError('competition <n> is not visible to this reader') on failure — which the non-strict consumption path already turns into a dropped-with-warning block. Gate _resolve_leaderboard the same way before calling build_leaderboard_query.

---

## 16. [major / content / fix in courseware] courseware/content/python-ai-engineering/s1-git-and-packaging/lab-1.md, s2-agentic-coding/lab-2.md, s3-data-science-nutshell/lab-3.md

**Problem.** Labs 1-3 never mention the competition attached to their own module, never mention ML-Arena, and never say anything is submitted. The module card advertises 'textstats correctness' / 'Flesch reading-ease' / 'Adult Census Income' while the lab it belongs to says the deliverable is a GitHub URL. Zero of the ~102 published lessons across both courses uses the ```mlarena:submit``` / ```mlarena:competition``` / ```mlarena:leaderboard``` directive blocks the platform built for exactly this.

**Evidence.**
```
`grep -rn '```mlarena' student_view/` -> 0 matches; SDK `lesson(...)['directives']` is `[]` and `directive_warnings` is `[]` for lab-1. `grep -rli 'ml-arena|mlarena' student_view/python-ai-engineering/` matches only lab-4.md, github-actions.md and three s3 lessons — not lab-1, lab-2 or lab-3. lab-1.md's Grading table lists four criteria, none of them a leaderboard score.
```

**Proposed fix.** Add a closing 'Part F — Submit' section to lab-1.md, lab-2.md and lab-3.md containing the competition's baseline number in prose plus a ```mlarena:submit id=179``` (resp. 180, 181) block, and add the competition row to each lab's Grading table. Do the same for the MS2A lab-N.md files, which attach 17 competitions across ten sessions. Republish with `make publish`.

---

## 17. [minor / content / fix in courseware] s1-git-and-packaging/packaging-and-tests.md:232

**Problem.** The coverage command taught in the lesson errors out, because pytest-cov is not among the dev dependencies the same lesson declares twelve lines earlier.

**Evidence.**
```
packaging-and-tests.md:71-72 declares `dev = ["pytest>=8", "ruff>=0.6"]`; :232 gives `uv run pytest --cov=my_project`.
  $ cd /tmp/paie-lab1 && uv run pytest --cov=textstats
  ERROR: usage: pytest [options] [file_or_dir] …
  pytest: error: unrecognized arguments: --cov=textstats
```

**Proposed fix.** Add `"pytest-cov>=5"` to the dev list in packaging-and-tests.md:72 (it is already in the fix for the dependency-groups change above), so the block reads `dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]`.

---

## 18. [minor / content / fix in courseware] s1-git-and-packaging/lab-1.md:16

**Problem.** Part A's `.gitignore` list omits `*.egg-info/`, which the editable install creates inside `src/`. Following Part A literally therefore commits build artefacts — and it contradicts python-environments.md, which lists `*.egg-info/` under "What never goes in Git" twenty minutes earlier. It also never says to commit the lockfile, although lab-1.md:101 grades `uv sync` on a fresh clone, which needs one.

**Evidence.**
```
lab-1.md:16: "Commit a `.gitignore` covering `.venv/`, `__pycache__/`, `.env`, `data/`." — four entries.
python-environments.md "What never goes in Git": `.venv/`, `__pycache__/`, `*.egg-info/`, `.env`.
After building the lab with exactly Part A's four rules:
  $ git ls-files
  src/textstats.egg-info/PKG-INFO
  src/textstats.egg-info/SOURCES.txt
  src/textstats.egg-info/dependency_links.txt
  src/textstats.egg-info/requires.txt
  src/textstats.egg-info/top_level.txt
  …
```

**Proposed fix.** lab-1.md:16 -> "4. Commit a `.gitignore` covering `.venv/`, `__pycache__/`, `*.egg-info/`, `.env`, `data/` — and do commit `uv.lock`, it is what makes the fresh-clone check below reproducible."

---

## 19. [minor / structure / fix in courseware] content/python-ai-engineering/course.yaml (module s1-git-and-packaging)

**Problem.** Session 1 is budgeted at 230 minutes inside a 3-hour (180-minute) session, and the split is 170 minutes of lecture to 60 of lab — not the "half lecture, half lab" the course description promises. Either the lab or ~50 minutes of lecture will be cut live, and the lab is the part that is graded, so students will most likely be sent home to do Parts D and E unsupervised, which is where the pair-review and conflict-resolution learning actually is.

**Evidence.**
```
course.yaml:45,50,55,60,65,70 estimated_minutes: 10 + 45 + 50 + 30 + 35 + 60 = 230.
course.yaml:23-24 course description: "A 12-hour mise à niveau in four 3-hour sessions… Every session is half lecture, half lab."
(Same arithmetic from the served index: `awk -F'\t' '$1 ~ /^s1-/ {…}' _index.tsv` -> TOTAL 230 min.)
```

**Proposed fix.** Bring the lecture block to ~110 minutes and the lab to ~70, so 180 holds and the halves are honest: cut `git-essentials` 45 -> 35 by moving Install + Authenticating-with-GitHub into a short pre-session "Before Session 1" reference lesson (they are setup, not teaching); cut `branching-and-collaboration` 50 -> 40 by moving the Naming and Keeping-a-branch-current sections to the existing paie-reference/git-cheatsheet; cut `packaging-and-tests` 35 -> 30 by dropping Coverage to reference. Update the six `estimated_minutes` in course.yaml to 10/35/40/30/30/70 (= 215… trim `why-version-control` to 5 for 180 if you want it exact).

---

