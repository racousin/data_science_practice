## [blocker/competition] s1-git-and-packaging/lab-1 + competition 179

Session 1's attached competition is invisible from the session. No lesson in s1 contains the words competition, leaderboard, ML-Arena or submit; lab-1 has no submission part; and its stated deliverable ("a GitHub repository URL") names no place to hand it in. Meanwhile competition 179 — whose own overview calls itself "the leaderboard half of 'ship a package'" and is the only external check on whether the three functions are right — 404s to a student token. The net effect on this module is that the exercise has no external oracle and no hand-in path at all.

EVIDENCE:
```
$ grep -rniE 'ml-arena|mlarena|leaderboard|competition|submit' student_view/python-ai-engineering/s1-git-and-packaging/*.md
  branching-and-collaboration.md:228:alongside leaderboard performance.   <- the only hit, and it is about the project grade
By contrast s4-pytorch-nutshell/lab-4.md:84 has a whole "## Part D — Submit to ML-Arena (10 min)" with `client.submit(competition_id=<id>, path="submission.csv")`, and lab-4.md:128 grades `| ML-Arena submission accepted | 10% |`.
With the student key: `c.competition(179)` -> `CompetitionNotFoundError: Not Found`; `c.datasets(179)` -> `datasets failed`; and _module.json records the same: `"error": "CompetitionNotFoundError: Not Found"`.
Also: no lesson in the entire course uses the platform's embed directives — `grep -rn 'mlarena:' student_view/python-ai-engineering/` returns nothing, and `c.lesson(...)` returns `"directives": []` for every s1 lesson.
```

FIX: Two halves. (courseware) Add a "## Part F — Put it on the leaderboard (10 min)" to lab-1.md before the Grading table, and give the competition a graded row (e.g. `| textstats leaderboard entry | 10% |`, rebalancing the other four). The platform resolves these fenced directives server-side (backend/app/services/lesson_directives.py:203-205), so the body can be literally:
  ```mlarena:competition id=179
  ```
  Upload `agent.py` wrapping the `core.py` you just wrote — the agent directory is on `sys.path`, so `from core import word_count, char_frequencies, longest_word` works.
  ```mlarena:submit id=179
  ```
and repeat the three-line SDK snippet from lab-4.md:94-98 with the user-scoped key, since Session 1 is the first time a student is asked to submit anything. (platform) Flip 179 to is_public=True — `python tools/build_competitions.py publish` per courseware/competitions/README.md — otherwise Part F 404s.

---

## [blocker/competition] competition 182 (PAIE S4 — MNIST Warm-up), attached to module s4-pytorch-nutshell

Competition 182 was created with is_public=False and never flipped, so a student token cannot open it, list its datasets, or read its leaderboard. Concretely for this module: Lab 4 Part D (10% of the lab grade, and the module's stated second deliverable — "a leaderboard entry") is impossible, X_test.csv is unobtainable, and every number that would tell the student whether their model worked lives only in overview.md, which is behind the same 404.

EVIDENCE:
```
With the student token:
  c.competition(182)  -> CompetitionNotFoundError: Not Found
  c.datasets(182)     -> MLArenaError: datasets failed: Failed to retrieve datasets
  c.leaderboard(182)  -> HTTPError: 500 Server Error
yet the module overview happily advertises it:
  c.module_overview("python-ai-engineering","s4-pytorch-nutshell")["competitions"]
  [{"competition_id": 182, "label": "MNIST Warm-up", "name": "PAIE S4 — MNIST Warm-up"}]
and student_view/.../s4-pytorch-nutshell/_module.json records the same dead link:
  {"competition_id": 182, ..., "error": "CompetitionNotFoundError: Not Found"}
The course description itself promises the deliverable: "You finish the module with a packaged, tested, version-controlled Python project that trains a neural network — and a submission on ML-Arena."
```

FIX: Run `python tools/build_competitions.py publish` (documented in courseware/competitions/README.md, "Publishing") to flip 182 to is_public=True before the 2026-09-07 start date. Then re-run `c.competition(182)` and `c.datasets(182)` with the student key as an acceptance check — publishing the course without this check is what let the dead link ship.

---

## [blocker/platform] backend/app/views/competition_asset.py:19-46, backend/app/views/competitions.py:918-960, backend/app/views/leaderboard.py:75-77

Three student-facing routes wrap `get_visible_competition_or_404()` in a bare `except Exception` and return 500. A legitimate visibility 404 is reported to the student as a server error, which is what a course-14 student hits on every one of 179-182 and which violates the repo's own Fail Fast rule ("No silent try/except").

EVIDENCE:
```
get_visible_competition_or_404 calls `abort(404)` (_helpers.py:212), which raises werkzeug NotFound — a subclass of Exception. competition_asset.py:22 `get_visible_competition_or_404(competition_id)` sits inside `try:` whose handler at :41-46 returns `jsonify({"error": "Failed to read the competition overview markdown"}), 500`. Measured: student token on /api/competition_asset/179/markdown/overview -> 500, /api/competitions/179/datasets -> 500, /api/leaderboard/competition/179 -> 500, while /api/competitions/179 (which lets the abort propagate) correctly returns 404.
```

FIX: In all three handlers, hoist `get_visible_competition_or_404(competition_id)` above the `try:` block, or add `except HTTPException: raise` as the first handler so werkzeug aborts propagate untouched. Also drop the overview route's silent creation of DEFAULT_MARKDOWN_CONTENT on GET (competition_asset.py:28-34): a read request should not write a placeholder file, and that write is what makes 169/170 look like they have an overview.

---

## [blocker/platform] backend/app/views/academic_courses/legacy.py:29-90

GET /api/academic_courses/ has no auth decorator and serializes courses with the full AcademicCourse.to_dict(), which includes join_code and enrollment_link. Anyone on the internet can harvest the enrolment secret of every course on the platform, including private ones, and self-enrol.

EVIDENCE:
```
Anonymous request (no Authorization header): `curl https://ml-arena.com/api/academic_courses/?show_all=true` -> 200, body begins `[{"code":null,...,"enrollment_link":"d1b1d31cc91042229f15dcdce119953d","id":12,...`. With the student token the same route returned join codes for all 15 courses: `13 private 'Knowledge Graph & Embeddings' join=A9QEVXAM`, `9 private 'TU Wien - Machine Learning 2026S - GSM8k' join=JJERDJ35`, `2 private 'Sorbonne - L2 Mathematiques 2026' join=84QMYEFP`. Route decorator at legacy.py:29 is only `@bp.route("/", methods=["GET"])`; serialization is `data = course.to_dict()` at legacy.py:85.
```

FIX: Add `@login_required` to get_courses and replace `data = course.to_dict()` (legacy.py:85) with a learner-safe serializer that emits id/name/code/slug/description/visibility/instructor_name/start_date/end_date/is_enrolled only. Add join_code and enrollment_link back per row only when `can_manage_course(course)` is true (import from ._helpers). Also filter the `show_all` branch (legacy.py:61) by `can_view_course` so private courses a caller has no relationship to are not listed at all.

---

## [blocker/platform] backend/app/views/academic_courses/consumption.py:194-210 and legacy.py:149-155; frontend/src/pages/CourseLearner/CourseLanding.tsx:201

A course with visibility=public is fully browsable but cannot be joined: the only enrolment route requires a secret enrollment_link or join_code, and the learner-facing course_landing payload never carries either. This is not a missing default (create_course at legacy.py:133-134 always mints both, and courses 14/15 do have GR1WFC63 / N1DX2QA4) and not a missing authoring field (ShareTab.tsx:51 shows it to the teacher) — it is a missing self-enrol route plus a missing landing field. Consequence: is_enrolled is permanently false, so LessonProgress rows are never created and every progress surface is dead.

EVIDENCE:
```
`c.course("python-ai-engineering")` key set is ['can_manage','code','competition_ids','cover_url','description','end_date','id','instructor_name','is_enrolled','modules','name','progress','slug','start_date','visibility'] — no join_code, no enrollment_link. `GET /api/academic_courses/14/progress/me` -> 403 {"error":"Not enrolled in this course"}. `find_course_by_link_or_code` (_helpers.py:23-30) matches only enrollment_link or join_code, never slug. CourseLanding.tsx:201 renders `<JoinCodeForm label="Join code" />` with an empty field on a course whose code is nowhere on the page.
```

FIX: Backend: add `POST /api/academic_courses/<string:slug>/enroll` in legacy.py that resolves by slug and enrols when `course.visibility in (Visibility.PUBLIC.value, Visibility.UNLISTED.value)`, reusing the body of enroll_in_course (identity checks, end_date check, 409 on duplicate); keep the code path for private courses. Frontend: in CourseLanding.tsx:196-211 render a primary `Join this course` button calling that route when `!course.is_enrolled && course.visibility !== 'private'`, and keep JoinCodeForm only for private. SDK: `enroll_in_course(..., slug=None)` in mlarena-sdk/mlarena/client.py:1497. MCP: accept `slug` in `join_course` (mlarena-mcp/mlarena_mcp/server.py:110-128).

---

## [blocker/platform] frontend/src/pages/Competition/View.js:48-95 and frontend/src/components/CompetitionHeader.js:47

When GET /api/competitions/<id> answers 404, View.js never checks response.ok and stores the error body as competitionInfo, while CompetitionHeader catches the axios rejection and returns null. The student who clicks a module competition card lands on a page with no title, no header, empty tabs and no message — a silent blank, not an error.

EVIDENCE:
```
`GET /api/competitions/179` with the student token -> 404 {"error":"Not Found"}. View.js:51-64: `const competitionResponse = await fetch(...); const competitionData = await competitionResponse.json(); ... setCompetitionData(competitionData)` — no `.ok` check anywhere in the effect. CompetitionHeader.js:47: `if (!competition) return null;`.
```

FIX: In View.js:51-64 check `if (!competitionResponse.ok)` and set an error state; render a dedicated panel for 404 that says the competition is not open to you yet and links back to the course, instead of the tab shell. Same for the /environment fetch at View.js:66-73. In CompetitionHeader.js:18-23 keep the caught error in state and render the same panel rather than `return null`.

---

## [blocker/platform] backend/app/views/academic_courses/consumption.py:82-87; frontend/src/pages/CourseLearner/ModuleOverview.tsx:65-90; frontend/src/pages/CourseLearner/CourseLanding.tsx:100-118

_serialize_module_competition returns only competition_id/label/name, with no visibility or availability signal, so both learner surfaces unconditionally render an `Open competition` link into a page the student cannot open. This is the (b) half of the attached-but-invisible problem.

EVIDENCE:
```
consumption.py:82-87 returns exactly {"competition_id", "label", "name"}. ModuleOverview.tsx:80-87 always renders `<Button component={RouterLink} to={`/viewcompetition/${competition.competition_id}`}>Open competition</Button>`. CourseLanding.tsx:109-113 always `navigate(`/viewcompetition/${c.competition_id}`)`. Live: module s1-git-and-packaging advertises competition 179 (label "textstats correctness") and 179 is 404 for the student token.
```

FIX: Extend _serialize_module_competition (consumption.py:82-87) to `{"competition_id", "label", "name", "is_public": bool(link.competition.is_public), "is_started": bool(link.competition.is_started), "viewable": user_can_see_competition(link.competition)}` (import from app.views.creator_competition._helpers). In ModuleOverview.tsx:80-87 replace the Button with a disabled `Not open yet` badge when `!competition.viewable`; in CourseLanding.tsx:100-118 drop the onClick and render the badge greyed with the same tooltip. Mirror the new fields in frontend/src/services/coursesApi.ts ModuleCompetitionSummary (~line 701) and in the MCP list_attached_competitions output (mlarena-mcp/mlarena_mcp/server.py:242-264).

---

## [blocker/platform] backend/app/views/competition_asset.py:19-46

get_markdown_overview wraps get_visible_competition_or_404 in a bare `except Exception`, which catches Werkzeug's NotFound and converts a deliberate 404 into a 500. The visibility guard is defeated and the student gets a server error for a competition that is simply not published. It also violates the Fail Fast rule in CLAUDE.md (no silent try/except).

EVIDENCE:
```
`GET /api/competition_asset/179/markdown/overview` with the student token -> 500 {"error":"Failed to read the competition overview markdown"}, while `GET /api/competitions/179` -> 404. Source: line 22 `get_visible_competition_or_404(competition_id)` is inside the `try:` opened at line 21, and line 41 is `except Exception as e:` returning 500.
```

FIX: Move the `get_visible_competition_or_404(competition_id)` call above the `try:` block (competition_asset.py:21-22), or narrow the handler to `except (OSError, IOError)` and add `except HTTPException: raise` before it. Apply the same treatment to update_markdown_overview (line 48) and delete_markdown_overview (line 81), which share the pattern.

---

## [blocker/competition] competition 169 (SuperTuxKart Grand Prix) and 170 (Bitcoin Intraday), attached to modules mlp-project and mlp-reference of course 15

Two competitions still carry the platform's placeholder overview. 169 is one of the three MS2A project tracks — half the course grade — and its brief for a student is the literal string 'Write your markdown here.' There is no task description, no submission contract, no metric, no baseline.

EVIDENCE:
```
`GET /api/competition_asset/169/markdown/overview` -> 592 bytes, beginning `**Competition Overview**\n\nWrite your markdown here.` followed by an auto-appended `<!-- mlarena:quickstart -->` Colab block. Same for 170 (634 bytes). The constant is at backend/app/views/competition_asset.py:14 `DEFAULT_MARKDOWN_CONTENT = "**Competition Overview**\n\nWrite your markdown here."`. Contrast: comp 176's overview is 7513 bytes with a measured 'Repères' table.
```

FIX: Courseware: write real overview.md for 169 and 170 (task, action/observation contract, metric, and a measured baseline table like comp 178's) and push with client.set_competition_markdown. Platform: in backend/app/views/creator_competition/settings start_competition, refuse to start (or return a blocking `warnings` entry) when the stored overview.md still equals DEFAULT_MARKDOWN_CONTENT, and badge such competitions 'Overview not written' in the creator list — the same class of guard the courseware already applies to benchmark_expected_score.

---

## [blocker/platform] competition 171 (engine 23), competition 169 (engine 22)

The Generative and Agent project tracks both run on `local_vm` engines whose VM is unreachable, and a submission does not queue — it hard-fails after blocking the HTTP request for 60 s. Two of the three tracks for a deliverable worth 50% of the course were un-submittable during my session, and the failure message gives the student no cause.

EVIDENCE:
```
`c.submit(competition_id=171, files=['pitch.txt'])` → `requests.exceptions.ReadTimeout ... (read timeout=60)`; agent 8507 then reports `status: deploy_failed, last_status_message: 'Deployment failed: Failed to queue deployment job'`. Retry via `c.deploy_agent(171, 8507)` → same 60 s timeout. `c.competition(171)['engine']` = `{'id': 23, 'k8s_workload_value': 'local_vm', 'vm_health_checked_at': '2026-09-02T20:41:10', 'vm_health_ok': False}`; comp 169 engine 22 same, `vm_health_ok: False`. The real error is swallowed at backend/app/views/direct_attache_agents/deploy.py:46-55 (`except Exception: ... return None`) and re-raised as the generic string at deploy.py:229. frontend/src/components/VMHealthBanner.tsx promises the opposite: "submissions will queue and run automatically once it is back online."
```

FIX: Two edits. (1) backend/app/views/direct_attache_agents/deploy.py: stop returning `None` from `apply_agent_deployment`'s except block — re-raise, and surface the underlying exception text in `last_status_message` so the student reads "GPU machine unreachable" instead of "Failed to queue deployment job" (this is the repo's own fail-fast rule). (2) Either make the deploy actually enqueue when `engine.vm_health_ok is False` (matching the banner's promise) or reject with HTTP 503 and the banner's text, and do not block the request for 60 s.

---

## [blocker/competition] competition 172 (module s3-tabular-models)

Every dataset file for competition 172 is gone from storage, so the download path the overview advertises is dead. This is the only downloadable data anywhere in module s3, and Lab 3 has no other dataset of its own.

EVIDENCE:
```
`c.download_dataset(172, ...)` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found`. Checking each of the three files listed by `c.datasets(172)` individually: `X_train.csv id=20 size=2493985 -> HTTP 404`, `X_test.csv id=22 size=624783 -> HTTP 404`, `y_train.csv id=31 size=108474 -> HTTP 404`, each returning `<Code>NoSuchKey</Code><Message>The specified key does not exist.</Message>` from `storage.googleapis.com/.../prod/datasets/6/`. The overview says: "You can also download the data from the **Datasets** tab and work locally." The same three files ARE live at `https://raw.githubusercontent.com/racousin/SCAI-4EUWorkshopAIinMedicineWorkshop/main/Hands-On-Session-1/data/` (HTTP 200, 2493985 / 108474 / 624783 bytes — byte-identical sizes to the dead rows).
```

FIX: Re-upload the three files to dataset 6 from the workshop repo (`upload_dataset_file(172, 6, ...)` with a creator key) — the sizes match exactly, so it is a straight restore of objects that were deleted from the bucket. Until that lands, edit the overview's Quickstart section to replace "You can also download the data from the **Datasets** tab and work locally." with "You can also read the three CSVs directly from the workshop repo: `https://raw.githubusercontent.com/racousin/SCAI-4EUWorkshopAIinMedicineWorkshop/main/Hands-On-Session-1/data/{X_train,y_train,X_test}.csv`."

---

## [blocker/competition] competition 174 (Clinical Note Triage, attached to s7-nlp-1)

The only dataset attached to the Session 7 capstone competition is unreachable. Both files return 404 from GCS, so no student can start the competition, and the linked Colab starter dies on its data cell. This is the single external artefact that could prove a student did Lab 7.

EVIDENCE:
```
`uv run python get.py` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://storage.googleapis.com/rlarena-417509-render-experiments-eu/prod/datasets/8/train.csv?...`. Direct GET on both signed URLs from `client.datasets(174)`:
  train.csv  id=28 size=5740082 -> HTTP 404  body: `<Code>NoSuchKey</Code>...No such object: rlarena-417509-render-experiments-eu/prod/datasets/8/train.csv`
  test.csv   id=29 size=1422816 -> HTTP 404  (same, .../prod/datasets/8/test.csv)
The DB rows still exist with their byte sizes. The overview's "Quick start" Colab (raw URL returns 200) has as cell 4: `client.download_dataset(COMPETITION_ID, "data/")` — it dies there.
```

FIX: Re-upload the two objects to `prod/datasets/8/train.csv` and `prod/datasets/8/test.csv` in `rlarena-417509-render-experiments-eu` (or repoint `dataset_file` rows 28 and 29 at wherever they now live, e.g. the R2 bucket). Verify with `client.download_dataset(174, "data/")` returning two paths whose sizes match 5740082 / 1422816. Until that is done, do not point students at competition 174 from any lesson.

---

## [blocker/content] s10-reinforcement-learning-2/lab-10

Part E's submit snippet cannot produce an accepted submission for a torch agent — which is every agent the lab tells you to build. A fresh attachment defaults to the framework=`none` runtime image (no torch), so `import agent` fails and the deploy dies. The lab's own table warns "a missing dependency fails the deploy" but hands the student the exact call that causes it, and never mentions that `submit()` takes a `runtime=` argument.

EVIDENCE:
```
lab-10.md Part E, verbatim: `res = client.submit(COMPETITION_ID, files=["src/rl/agent.py", "checkpoints/policy.pt"])`. I ran exactly that with a torch agent on comp 48: attachment 8510 -> status `deploy_failed`, message `File "/app/storage/competitions/48/agent/8510/agent.py", line 3, in <module> import torch / ModuleNotFoundError: No module named 'torch'`. `c.agent_runtime(8510)` -> `{'framework': 'none', 'id': 181}`. Identical files submitted as `c.submit(48, files=FILES, runtime={"language":"python","framework":"torch"})` -> attachment 8511, `c.agent_runtime(8511)` -> `{'framework':'torch','framework_version':'2.12.0','id':167}`, status `active`, "Deployment completed successfully". Both attachments deleted afterwards.
```

FIX: In lab-10.md Part E replace the snippet with `res = client.submit(COMPETITION_ID, files=["src/rl/agent.py", "checkpoints/policy.pt"], runtime={"language": "python", "framework": "torch"})` and add one sentence above it: "A new attachment defaults to the dependency-free runtime image. If your agent.py imports torch, tensorflow or jax you must pin the matching runtime with `runtime=` or the deploy fails at import with ModuleNotFoundError — `client.runtime_options(COMPETITION_ID)` lists what is available." Add the same sentence to the "four rules" table row for "Importable". Platform side: `backend/app/views/direct_attache_agents/create_delete.py:125` picks the default with an unordered `.first()`, so which image a student silently gets is whatever Postgres returns first — order it explicitly, or make the competition carry a declared default runtime.

---

## [blocker/structure] courses 14 and 15 (python-ai-engineering, ms2a-machine-learning-practice)

Neither course has a join code or an enrollment link, so no student can enrol; without enrolment every progress surface is dead and nothing a student does is recorded.

EVIDENCE:
```
`c.course('python-ai-engineering')` -> join_code=None, enrollment_link=None, is_enrolled=False, visibility='public'; same for ms2a-machine-learning-practice. `c.my_progress(14)` and `c.my_progress(15)` -> AuthenticationError: Not enrolled in this course. `c.mark_lesson_complete(115)` -> AuthenticationError: Not enrolled in any course containing this lesson.
```

FIX: Generate a join_code for both courses through the teacher surface (mlk_teacher_ key, /api/teacher/*) and add the resulting code/link to content/<course>/course.yaml so `make publish` carries it. Until then, publish_mlarena.py should fail the build when a published course has neither join_code nor enrollment_link, rather than emitting a course nobody can enter.

---

## [blocker/platform] backend/app/views/competition_asset.py:22-46

A blanket `except Exception` swallows the 404 abort raised by get_visible_competition_or_404, so requesting the overview of a competition you cannot see returns HTTP 500 with the message 'Failed to read the competition overview markdown'. The student sees a server error instead of 'not found', and the operator sees a false storage-failure log line.

EVIDENCE:
```
`curl -H 'Authorization: Bearer mlk_user_...' https://ml-arena.com/api/competition_asset/179/markdown/overview` -> 500 {"error": "Failed to read the competition overview markdown"}; the same URL with the creator key -> 200. Public comp 173 -> 200 for both. Source: line 23 calls get_visible_competition_or_404(competition_id) inside the try; line 39 `except Exception as e` catches the werkzeug NotFound and line 46 returns 500. This is the 'No silent try/except' rule in CLAUDE.md.
```

FIX: Move `get_visible_competition_or_404(competition_id)` above the `try:` block, or add `except HTTPException: raise` before the generic handler, so a hidden or missing competition returns its real 404 and only genuine storage errors return 500.

---

## [blocker/validation] competition 173 (attached to s5-computer-vision-1) / dataset 7

Every dataset file for Session 5's competition is a dead object. The DB rows exist and the backend happily signs URLs for them, but the GCS objects are gone, so a student cannot obtain a single training image. Session 5's only measurable deliverable is unreachable.

EVIDENCE:
```
`c.download_dataset(173, dest_dir=...)` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found`. Fetching each signed URL directly: train_images.npz (declared 29,722,203 B) HTTP 404, y_train.csv (246,129 B) HTTP 404, test_images.npz (7,439,321 B) HTTP 404 — all three `<Code>NoSuchKey</Code> ... No such object: rlarena-417509-render-experiments-eu/prod/datasets/7/...`. The dataset's own description still reads "Public training/test data + starter notebook."
```

FIX: Re-upload the three objects to gs://rlarena-417509-render-experiments-eu/prod/datasets/7/{train_images.npz,y_train.csv,test_images.npz}; or upload them to the R2 datasets bucket and set `dataset_file.storage_backend='r2'` for dataset 7 (the column is at modelmanager/modelmanager/competitions.py:548 and backend/app/views/competitions.py:937 already dispatches on it via `dataset_download_url`). Separately, add a `dataset_download_url` health check so a signed URL is never handed to a student for an object that does not exist — the current code path turns a missing object into a 404 the student reads as their own mistake. And either ship the promised starter notebook or change the dataset description from "Public training/test data + starter notebook." to "Public training/test data." — the file list is only train_images.npz, y_train.csv, test_images.npz.

---

## [blocker/baseline] competition 173 (attached to s5-computer-vision-1)

The competition states no target score, and the only reference the platform publishes for it scores below random guessing — so a student has no way to know whether they have done the lab, and the one automated signal actively misleads them.

EVIDENCE:
```
`c.competition(173)['description']` is one line: "Blood cells — 8 types, 28×28 RGB images". `c.leaderboard(173)`: 26 rows, `Metric` is 'reward' for every row, MeanReward ranges 0.040759–0.959296, and the `__benchmark__` row is rank 26 (last) at MeanReward=0.040759. With 8 classes, chance accuracy is 0.125, so the published benchmark is a third of chance.
```

FIX: Rewrite the description to state the contract in numbers, e.g.: "Eight blood-cell types, 28×28 RGB. Metric: accuracy on a held-out split, higher is better, range 0–1. Guessing scores 0.125. A logistic regression on raw pixels scores 0.749 — that is the baseline you must beat. A small CNN trained from scratch reaches ~0.92; the current best is 0.959." Then either re-run the creator benchmark so `__benchmark__` reflects the 0.749 pixel baseline instead of 0.0408, or remove the broken benchmark agent so students are not shown a sub-chance reference. Also fix the metric label: the leaderboard reports `Metric='reward'` for an accuracy number.

---

## [blocker/validation] all 102 lessons in python-ai-engineering + ms2a-machine-learning-practice

Not one lesson ends with a self-check. A student has no way, anywhere in 102 lessons, to find out whether they understood the lesson they just read. This is the whole of the owner's requirement, and it is at zero.

EVIDENCE:
```
`for pat in "check yourself" "self-check" "expected output" "you should see" "if you got"; do grep -ril "$pat" --include='*.md' . | wc -l; done` -> 0 0 0 0 0. Closing sections that do exist: 10x `## Carry it forward` (all MLP labs, prose: "This dataset is the input to Lab 2"), 8x `## Checklist` (prose bullets), 3x `## Recap` (a summary table), 3x `## What to take away`. The only comparable-against-truth artifact in the corpus is 6 lessons with an inline `print(...) # expected` comment, best of them python-ai-engineering/s4-pytorch-nutshell/autograd.md:192 `## Worked example — one gradient step by hand`: "print(w.grad)  # tensor([-12.])" followed by "Check it: $L = (wx - t)^2$, so $\partial L/\partial w = 2(wx - t)x = 2(2 - 5)(2) = -12$." That is the exact shape to copy: runnable snippet, stated expected value, the derivation that turns a mismatch into a diagnosis.
```

FIX: Add a `mlarena:checkpoint` directive and put one at the end of every lesson. Syntax (the existing fence grammar already captures a body — lesson_directives.py:34 says it is "captured so a future directive type can read inline config without a grammar change"):

```mlarena:checkpoint id=autograd-1 kind=run
prompt: Run this. What does w.grad print?
snippet: |
  w = torch.tensor([1.0], requires_grad=True)
  loss = (w * 2 - 5) ** 2
  loss.backward(); print(w.grad)
expect: "tensor([-12.])"
why: |
  dL/dw = 2(wx - t)x = 2(2-5)(2) = -12. A different sign means you read the
  chain rule backwards; None means requires_grad was not set on a leaf.
```

kind ∈ {run (snippet+expect), numeric (expect+compare+tol), mcq (choices+answer)}. `expect` stays plaintext — checkpoints are diagnostic, not graded, and body_md is served verbatim anyway (92/102 lessons already leak speaker notes), so hiding answers would be theatre. Author ~35 concept lessons and all 14 labs. Content cost is the real cost here (a day of writing); the engineering is in the next findings.

---

## [blocker/validation] backend/app/services/lesson_directives.py:202 + modelmanager/modelmanager/lesson_progress.py:31 + backend/app/views/academic_courses/consumption.py:369

RECOMMENDED DESIGN (primary, option b — the lesson directive). The three options are not equal: (a) pure markdown gives the student an answer key but gives the system no state, so nothing can ever say 'what remains'; (c) folding competition score into progress needs a schema change and covers only 16 of 102 lessons. (b) subsumes both: the fence body is readable markdown so it degrades to (a) for any consumer that ignores directives, and a second directive type delivers (c) without a migration.

EVIDENCE:
```
The machinery is already built and unused. `_HANDLERS` at lesson_directives.py:202 is 3 entries {competition, leaderboard, submit} with the comment "Adding a type is a one-line change here"; strict mode already refuses to publish a broken directive (teacher/lessons.py:257) and non-strict already drops-with-warning for readers (consumption.py:272). `lesson()` already returns `directives` + `directive_warnings` to the SDK (confirmed live: lesson keys include both) and MCP `get_lesson` (mlarena-mcp/mlarena_mcp/server.py:199-208) passes them straight through. Zero lessons in either course use any directive today: `grep -rn '```mlarena:' --include='*.md' student_view/` -> 0 hits.
```

FIX: Ship two handlers plus one column.

SERVER. (1) `_resolve_checkpoint(args, body)` in _HANDLERS: parse the fence body as YAML; strict mode raises DirectiveError on a missing `prompt`, an unknown `kind`, a missing `expect`, or a duplicate `id` within the lesson — so `make publish` refuses a malformed checkpoint. Payload = {id, kind, prompt, snippet, expect, compare, tol, why, choices} plus `state` = this caller's stored result. (2) `_resolve_target(args)`: args `id=<competition_id> metric=<name> min=<float>` (or `max=`); payload = {competition_id, name, metric, min, your_best, met, n_runs, last_run}. `your_best` reuses build_leaderboard_query(competition_id=..., is_elo_ranked=..., aggregate='user', student_ids=[current_user.id]) — the exact call teacher/course_content.py:344 already makes, so no new query. Strict mode fails when the competition does not exist or is invisible to the author, which would have caught competitions 179-182 before publish.

STATE. One nullable JSON column `lesson_progress.checkpoints` = {"<checkpoint id>": true|false}. Add `checkpoints: Optional[dict[str, bool]]` to LessonProgressContext (_schemas.py:72) — note it is `extra='forbid'`, so a client sending the field today gets a 400; the field must be declared. `POST /lessons/<id>/complete` merges it. In `_content_progress` (teacher/course_content.py:255) each per_module row gains `checkpoints: {passed, total}`, and `my_progress` gains `targets: [...]`. A module is green only when every published lesson is complete AND every target in it is met — that is option (c)'s semantics with the number authored in markdown instead of in a new table.

STUDENT SEES. Web: a card at the end of the lesson — prompt, a Reveal that shows `expect` + `why`, and Got it / Not yet writing into progress; the target card is a meter ("your best 0.907 / target 0.913 — not yet") with the submit CTA. SDK: directives already arrive; `my_progress()` gains the counts. MCP: an AI tutor can now say "3 unchecked checkpoints in Session 4, and your MNIST submission is under the line".

COST. backend ~140 LOC + 1 alembic rev (one nullable JSON column). frontend ~160 LOC: 2 cards in directiveCards.tsx + 2 cases at CourseMarkdown.tsx:64-72. SDK 2 lines (a `checkpoints=` kwarg on mark_lesson_complete, client.py:1627). MCP 1 tool arg. courseware: 2 regexes in build_slides.py + 1 assertion in student_walk.py.

WORKED EXAMPLE — python-ai-engineering/s4-pytorch-nutshell/lab-4.md. BEFORE (lines 96-102 and the rubric at 120-129): "client = mlarena.connect(api_key=\"mlk_user_...\") / client.submit(competition_id=<id>, path=\"submission.csv\") / print(client.leaderboard(<id>).head())" then "Getting on the board matters; your position does not." and "| ML-Arena submission accepted | 10% |". AFTER: fix the call to `client.submit(competition_id=182, files=[\"submission.csv\"])`, then add the measured line that already exists in competitions/s4-mnist-warmup/overview.md — "Multinomial logistic regression on raw pixels scores 91.3%. A correctly wired MLP clears 97%." — followed by:

```mlarena:target id=182 metric=accuracy min=0.913
label: Beat the logistic-regression line
hint: |
  Below 91.3%? Re-run test_overfits_one_batch from Part B — it fails for
  exactly the three reasons that put you there: normalisation, a softmax
  before CrossEntropyLoss, or a missing model.eval().
```

and a `## Check yourself` section with two checkpoints:

```mlarena:checkpoint id=lab4-fresh-clone kind=run
prompt: From a fresh clone of your branch, does the project install and test?
snippet: |
  git clone <your-repo> /tmp/lab4 && cd /tmp/lab4
  uv sync && uv run pytest -q
expect: "4 passed"
why: |
  Green on your own machine proves nothing — it may pass on files you never
  committed. Part A's 20% is this command on a clone, not on your laptop.
```

```mlarena:checkpoint id=lab4-overfit kind=numeric
prompt: What loss did test_overfits_one_batch reach after 200 steps on 32 examples?
expect: 0.1
compare: lt
why: |
  Above 0.1 and the model cannot memorise 32 examples: shapes, loss,
  optimizer wiring or zero_grad is wrong. Fix it before Part C.
```

and finally rewrite the Grading table with a third column so half of it is machine-settled: "| `uv sync && uv run pytest` green on a fresh clone | 20% | checkpoint lab4-fresh-clone |", "| MNIST accuracy ≥ 0.913 on competition 182 | 10% | target 182 |", the remaining rows "| ... | review |". The student now knows, before submitting anything, exactly which parts of their grade they have already secured.

---

## [blocker/platform] modelmanager/modelmanager/lesson_progress.py:31-34 + backend/app/views/academic_courses/consumption.py:369-391

The platform can only record 'I opened this' and 'I clicked the button'. There is no representation anywhere of 'I demonstrably got this right', so no amount of courseware writing can make progress mean correctness without a platform change.

EVIDENCE:
```
LessonProgress carries exactly four state fields: `status` (not_started/in_progress/completed), `completed_at`, `last_viewed_at`, and the (user, lesson, course) key. `mark_lesson_complete` (consumption.py:369) sets `progress.status = COMPLETED` from a request body whose only field is `course_id` — no evidence of any kind is required or accepted. The LessonReader UI is honest about it: LessonReader.tsx:147 renders a plain `Mark complete` button. `_content_progress` (teacher/course_content.py:303-308) then computes pct as completed/total over that self-declaration, and that same number is what both the student landing page and the teacher dashboard show.
```

FIX: Add the `checkpoints` JSON column and the `checkpoints` request field described in the design finding, and report `checkpoints_passed / checkpoints_total` alongside `completed/total` in both `my_progress` and `_content_progress`. Keep `Mark complete` for lessons with no checkpoints — the point is not to gate, it is to distinguish the two states in the payload so the UI and the SDK can show 'read' separately from 'verified'.

---

## [blocker/competition] competitions 179, 180, 181, 182 (course 14, sessions 1-4)

All four PAIE competitions 404 to a student token, so the single validation surface that already produces an objective number is unreachable for the entire 12-hour course. Any design that folds competition score into progress is dead here until they are flipped public.

EVIDENCE:
```
Live with the student key: `c.competition(179/180/181/182)` -> `CompetitionNotFoundError: Not Found` for all four. The module payloads carry the same error inline, e.g. student_view/python-ai-engineering/s1-git-and-packaging/_module.json: {"competition_id": 179, ..., "error": "CompetitionNotFoundError: Not Found"}. Cross-check: all 17 competitions attached to course 15 return is_public=True, is_started=True.
```

FIX: Run `make competitions-publish` in tmp/data_science_practice/courseware (it exists and flips is_public — Makefile target `competitions-publish`), then re-run `python tools/student_walk.py check --course python-ai-engineering` and confirm zero `competition-unreachable` lines. Until that runs, do not author any `mlarena:target` block against 179-182: an unreachable target renders a red light the student cannot clear.

---

## [blocker/platform] course 14 + course 15 (no join_code, no enrollment_link) / backend/app/views/academic_courses/consumption.py:289-320

Every progress write requires enrollment, and neither course is joinable, so the entire progress layer — ticks, percentage, next-lesson pointer, and any future checkpoint state — is unreachable for a real student today. The validation layer would ship into a surface no one can reach.

EVIDENCE:
```
Live: `c.my_progress(14)` and `c.my_progress(15)` both raise `AuthenticationError: Not enrolled in this course`. `_resolve_progress_course` (consumption.py:311-315) returns 403 "Not enrolled in any course containing this lesson" when the caller has no enrollment intersecting the lesson's modules, and both `/view` and `/complete` route through it. `_course.json` for course 14 shows `"is_enrolled": false` with no join code in the payload.
```

FIX: Generate a join code for both courses (the generator already exists: `gen_join_code` in backend/app/services/course_content.py) and put the enrol URL in the course description and in the first lesson of each course. Add the assertion to student_walk.py's check: it already flags `COURSE not-enrolled` (student_walk.py cmd_check) but that line is advisory — make a missing join_code a non-zero exit.

---

## [blocker/platform] ADJUDICATION of findings 141, 157, 183 and the audit's stated ground truth

The claim that neither course is joinable, and that the whole progress layer is therefore unreachable, is false. Enrolment is complete and working end to end today. Acting on the wrong belief would spend the five-day window building a self-enrol route that already ships.

EVIDENCE:
```
DB: course 14 join_code=GR1WFC63 enrollment_link=0ed552047ca44d4ab58add1d15a98573; course 15 join_code=N1DX2QA4 enrollment_link=78eab0f30fdd491aa7685bc0c842001a — neither is null. Live: `curl https://ml-arena.com/api/academic_courses/enroll/GR1WFC63` returns 200 with course_id 14 and competition_ids [179,180,181,182]. frontend/src/App.js:146 defines `/enroll/:enrollmentLink`. frontend/src/pages/CourseLearner/CourseLanding.tsx:201 renders `{!course.is_enrolled && <JoinCodeForm label="Join code" />}`, and components/Course/JoinCodeForm.tsx:27 navigates to `/enroll/${trimmed}`. CourseCatalog.tsx:128 and CourseUnavailable.tsx:37 render the same form. Finding 157 is right that the codes exist and wrong that a self-enrol route and landing field are missing.
```

FIX: No code change. Hand the two join codes to the cohort (the teacher Share tab already displays them) and delete findings 141/183 from the backlog. The only real gap is that a student who has not been given a code has no way to obtain one — which is the intended design for a class, not a defect.

---

## [blocker/competition] COLLAPSE of findings 54, 68, 80 — dataset_file rows for datasets 6, 7, 8 (competitions 172, 173, 174)

Three agents independently concluded the data for three competitions is gone and the competitions are dead. The bytes are intact in R2; only the storage_backend column was never flipped during the R2 migration, so the backend signs GCS URLs for objects that were deleted after the copy. One root cause, one UPDATE, three blockers retired.

EVIDENCE:
```
`gcloud storage ls gs://rlarena-417509-render-experiments-eu/prod/datasets/{6,7,8}/...` — "One or more URLs matched no objects" for all six keys. Same keys in R2 via the backend pod's own credentials: prod/datasets/6/ -> X_test.csv 624783, X_train.csv 2493985, y_train.csv 108474; prod/datasets/7/ -> test_images.npz 7439321, train_images.npz 29722203, y_train.csv 246129; prod/datasets/8/ -> test.csv 1422816, train.csv 5740082. Every size is byte-identical to the DB's file_size_bytes. DB shows these 8 rows at storage_backend='gcs' while datasets 9/10/11 are 'r2'. backend/app/dataset_storage.py:24-34 branches purely on that column.
```

FIX: `UPDATE dataset_file SET storage_backend='r2' WHERE dataset_id IN (6,7,8)` — 8 rows — then GET /api/competitions/172/datasets with a user token and fetch one signed URL to confirm 200. Do not regenerate or re-upload anything.

---

## [major/competition] competition 180 / s2-agentic-coding module page

The module page advertises competition 180 by name to every student, but opening it 404s and its leaderboard 500s. Because the pinned Flesch spec lives only in that overview, the unreachable page is not an optional extra for this module — it is the only authoritative statement of what Lab 2's tests are supposed to encode.

EVIDENCE:
```
With the student token: `c.module_overview("python-ai-engineering","s2-agentic-coding")["competitions"]` → `[{"competition_id": 180, "label": "Flesch reading-ease", "name": "PAIE S2 — Flesch reading-ease"}]`, while `c.competition(180)` → `CompetitionNotFoundError: Not Found` and `c.leaderboard(180)` → `500 Server Error ... /api/leaderboard/competition/180`. The same 404 is recorded in the module dump: `_module.json` → `"error": "CompetitionNotFoundError: Not Found"`.
```

FIX: Run `python tools/build_competitions.py publish` to flip 180 to `is_public=True` before the 2026-09-07 start date, and add to lab-2.md a Part F that links it: a ```mlarena:competition id=180``` directive block (the resolver already supports `competition`, `leaderboard` and `submit` types — backend/app/services/lesson_directives.py:203) plus the sentence "Submit `agent.py` and `readability.py` with `c.submit(180, files=[\"agent.py\", \"readability.py\"])`; a completed lab scores 1.0."

---

## [major/platform] backend/app/views/leaderboard.py:75-210

`GET /api/leaderboard/competition/<id>` returns HTTP 500 for any competition the caller cannot see or that does not exist, instead of 404. `get_visible_competition_or_404` calls `abort(404)`, which raises a werkzeug `NotFound` — an `Exception` subclass — and the route's blanket `except Exception` swallows it and rewrites it as a 500. This turns every unpublished course competition into an opaque server error for students and for the SDK.

EVIDENCE:
```
$ curl -s -o /dev/null -w '%{http_code}' https://ml-arena.com/api/leaderboard/competition/999999 → 500, body `{"error":"Failed to fetch leaderboard"}`; same for 179/180/181/182; competition 176 (public) returns 200. Source: leaderboard.py:76 `get_visible_competition_or_404(competition_id)` sits inside the `try:` opened at line 75; _helpers.py:208-213 `def get_visible_competition_or_404(...): ... abort(404)`; leaderboard.py:204-210 `except Exception as e: ... return jsonify({"error": "Failed to fetch leaderboard"}), 500`.
```

FIX: In backend/app/views/leaderboard.py, add `from werkzeug.exceptions import HTTPException` and insert, immediately before the existing `except Exception as e:` at line 204:
```python
    except HTTPException:
        raise
```
so aborts propagate with their own status. (Equivalently, hoist the `get_visible_competition_or_404(competition_id)` call above the `try:`.)

---

## [major/platform] backend/app/views/leaderboard.py:76 and :205-210

`GET /api/leaderboard/competition/<id>` returns HTTP 500 for a competition the caller cannot see, and for one that does not exist. `get_visible_competition_or_404()` calls `abort(404)`, but it is invoked inside the route's `try:` block and the bare `except Exception` at the bottom swallows the werkzeug NotFound and rewrites it as 500. Concretely for this module: `leaderboard(179)` gives a student a server error rather than the 404 that would tell them the competition is not published yet, and it makes hidden-vs-broken indistinguishable while debugging the course rollout.

EVIDENCE:
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

FIX: Move the visibility check out of the try in backend/app/views/leaderboard.py: put `get_visible_competition_or_404(competition_id)` on the line immediately before `try:` (line 76 -> line 75). Alternatively, if it must stay inside, add `except HTTPException: raise` (from werkzeug.exceptions) as the first handler before `except Exception`. The same swallow-the-abort shape exists at backend/app/views/teacher/leaderboard.py:206 and should get the same treatment.

---

## [major/platform] backend/app/views/leaderboard.py:205 and backend/app/views/competitions.py:956

The leaderboard and datasets routes rewrite an intentional 404 into an opaque 500. `get_visible_competition_or_404()` raises a Werkzeug `NotFound`, which the broad `except Exception` swallows and replaces with a generic 500 body. A student given a competition id they cannot see — or who mistypes one — gets a server error instead of "not found" and cannot tell a permissions problem from an outage. This is the silent-except the repo's own fail-fast rule forbids, and the correct pattern already exists 500 lines away in the same file.

EVIDENCE:
```
With the student token against prod: `GET /api/leaderboard/competition/181` → `500 {"error":"Failed to fetch leaderboard"}`; `GET /api/leaderboard/competition/999999` (an id that cannot exist) → the identical `500`; `GET /api/competitions/181` → correctly `404`; `GET /api/competitions/181/datasets` → `500 {"error":"Failed to retrieve datasets"}`. Source: leaderboard.py:77 calls `get_visible_competition_or_404(competition_id)` inside a `try:` whose only handler is `except Exception as e:` at 204-210; competitions.py:923 likewise, handler at 955-963. competitions.py:457-460 already does it right: `except HTTPException:` / `# get_visible_competition_or_404() raises 404 (missing/hidden); let it … propagate instead of being rewritten to a generic 500 by the broad handler below.` / `raise`.
```

FIX: Insert the same three lines above the broad handler in both places. In backend/app/views/leaderboard.py immediately before `except Exception as e:` (line 204) add `    except HTTPException:` / `        # get_visible_competition_or_404() raises 404 (missing/hidden); let it propagate.` / `        raise`, and the identical block in backend/app/views/competitions.py before `except Exception as e:` (line 955); competitions.py already imports `HTTPException`, leaderboard.py needs `from werkzeug.exceptions import HTTPException`. A student probing a hidden or nonexistent competition then gets 404, matching `/api/competitions/<id>`.

---

## [major/platform] backend/app/views/leaderboard.py:75 and :204

The leaderboard route wraps `get_visible_competition_or_404()` inside a broad `try:` whose `except Exception` swallows werkzeug's NotFound and re-emits it as HTTP 500. Any competition the caller cannot see — or that does not exist — returns `{"error":"Failed to fetch leaderboard"}` with status 500 instead of a 404. Lab 4 Part D tells the student to call exactly this (`client.leaderboard(<id>).head()`), so a hidden competition or a mistyped id gives a student an opaque server error rather than "not found". It is also a direct violation of the repo's own "No silent try/except" rule in CLAUDE.md.

EVIDENCE:
```
$ for cid in 176 43 182 999999; do curl -s -o /dev/null -w "$cid %{http_code}\n" -H "Authorization: Bearer mlk_user_..." https://ml-arena.com/api/leaderboard/competition/$cid; done
176 200
43 200
182 500
999999 500
$ curl -s .../competition/999999 -> {"error":"Failed to fetch leaderboard"}

leaderboard.py:74-75   try:
                           get_visible_competition_or_404(competition_id)
leaderboard.py:204-209 except Exception as e:
                           service_logger.error(...)
                           return jsonify({"error": "Failed to fetch leaderboard"}), 500
(_helpers.py:208-213: get_visible_competition_or_404 calls abort(404), which raises werkzeug.exceptions.NotFound — an Exception subclass.)
```

FIX: In backend/app/views/leaderboard.py, add `from werkzeug.exceptions import HTTPException` and insert an explicit re-raise immediately before line 204's handler:

    except HTTPException:
        raise
    except Exception as e:
        ...

This lets 404/403 propagate as themselves and keeps the 500 for genuine failures. Same pattern applies to any other route in app/views/ that calls abort() inside a broad try.

---

## [major/platform] s4-pytorch-nutshell/autograd:5 (and s1-git-and-packaging/git-essentials:262)

The two markdown links that connect the sessions to the Reference module both point at a URL shape the router does not have, so both are dead. The real lesson route carries a `kind` segment (`course` or `exercise`) between the module slug and the lesson slug, which a `../`-relative link cannot account for. These are the only two inbound links to paie-reference in the whole course, so the module is unreachable by navigation.

EVIDENCE:
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

FIX: Courseware side, now: make both links absolute. autograd.md:5 -> `[Reference → Autograd, the Mathematics](/courses/python-ai-engineering/paie-reference/course/autograd-mathematics)`; git-essentials.md:262 -> `[Reference → Git Cheatsheet](/courses/python-ai-engineering/paie-reference/course/git-cheatsheet)`.
Platform side, durable: relative cross-lesson links are a trap that every author will fall into, because the `kind` segment is invisible from the markdown. Either add a `mlarena:lesson module=<slug> lesson=<slug>` directive alongside the existing competition/leaderboard/submit types in backend/app/services/lesson_directives.py:250, or have the teacher-side validator (app/views/teacher/lessons.py:242, already run in strict mode) reject a relative link that does not resolve to a real lesson.

---

## [major/platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

EVIDENCE:
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

FIX: Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## [major/platform] mlarena-sdk/mlarena/client.py — no overview reader; mlarena-mcp/mlarena_mcp/server.py:242 list_attached_competitions

The competition overview — the only place any baseline is published — is unreachable from the SDK and from MCP. A student consuming the course through an AI editor can list competitions and submit to them but can never read what the target is. This violates the frontend/SDK parity rule in CLAUDE.md.

EVIDENCE:
```
`grep -n markdown mlarena/client.py` returns only `set_competition_markdown` (:592, a creator-scope PUT). There is no getter for GET /api/competition_asset/{id}/markdown/overview, which exists and is student-readable (I fetched all 21 overviews with raw requests + the student bearer token). In MCP, `list_attached_competitions` (server.py:242-264) returns only `competition_id`, `name`, `label`, `module_slug`, `module_title`; the `whats_next` prompt (server.py:392) tells the agent to "call list_attached_competitions() and suggest a competition to work on" with no way to read the task.
```

FIX: Add `def competition_overview(self, competition_id: int) -> str` to client.py mirroring GET /api/competition_asset/{id}/markdown/overview with `headers=self._headers()`, and expose it as an MCP tool `get_competition_overview(competition_id)` next to `leaderboard` in server.py. Both are pure additions over an existing route, so no new endpoint is needed and the parity rule is satisfied.

---

## [major/platform] mlarena-sdk/mlarena/client.py:161, :170, :192, :1338

competition(), competitions() and leaderboard() send no Authorization header, so an enrolled student using the SDK or MCP is treated as anonymous and cannot open any non-public course competition — the exact class that 179-182 belong to.

EVIDENCE:
```
client.py:192 `resp = self._request("GET", self._url(f"/competitions/{competition_id}"), timeout=30)` — no `headers=self._headers()`, unlike every scoped call (e.g. :225 datasets passes it). Same at :161/:170 (competitions) and :1338 (leaderboard). Observed effect: `mlarena.connect(CREATOR).competition(179)` raises CompetitionNotFoundError, while the same creator token via raw requests with a bearer header returns 200 — the owner is being told their own competition does not exist. The backend's visibility_filter (_helpers.py:229-233) grants access on `is_public OR owned OR assistant`, and the enrolled-student side-channel in competitions.py:213-217 is likewise keyed on `current_user.is_authenticated`.
```

FIX: Add `headers=self._headers()` to the four calls. These routes accept an anonymous caller, so the change is backward-compatible and only widens what an authenticated caller can see. Without it, publishing 179-182 still leaves them invisible to any SDK/MCP student even after enrollment is fixed.

---

## [major/platform] modelmanager/modelmanager/module_competition_link.py:18; backend/app/views/academic_courses (module_overview payload)

A course cannot record its own passing bar. ModuleCompetitionLink carries only a display label, so a per-cohort target has nowhere to live and must be smuggled into a 200-character string or into competition markdown a teacher may not own.

EVIDENCE:
```
module_competition_link.py:18 `label = db.Column(db.String(200), nullable=True)  # optional display label` — and to_dict() (:31-37) returns id/module_id/competition_id/position/label only. The student-facing module payload confirms it: module_overview("ms2a-machine-learning-practice","s3-tabular-models") returns `"competitions": [{"competition_id": 172, "label": "2-Month Survival Prediction — tabular binary classification: gradient boosting judged against a baseline you can defend", "name": "..."}]` — the word "baseline" is there, the number is not. courseware course.yaml has the same shape: `competitions: - competition_id: 172 / label: "..."`.
```

FIX: Add nullable `baseline_score`, `target_score` (Numeric) and `metric_direction` (String(6), 'higher'/'lower') to ModuleCompetitionLink, expose them in to_dict(), accept them in POST /api/teacher/modules/{id}/competitions and in the SDK's attach_competition(), add the matching keys to course.yaml's competitions block, and surface them in the student module_overview payload. That gives a teacher a per-cohort bar without touching a competition they do not own, and gives SDK/MCP students the target without needing the markdown route at all.

---

## [major/platform] modelmanager/modelmanager/lesson.py:74-75 -> backend/app/views/academic_courses/consumption.py:272-281

Lesson.to_dict returns body_md verbatim and consumption.lesson_body ships it unchanged, so the teacher's HTML-comment speaker notes reach every SDK and MCP consumer raw. The web renderer only appears to hide them: CourseMarkdown uses rehype-raw, so the comments are emitted into the DOM and are visible in view-source.

EVIDENCE:
```
109 `<!-- notes: ... -->` blocks across 92 of the ~102 published lesson bodies in the student_view dump. Via SDK: `c.lesson("ms2a-machine-learning-practice","s7-nlp-1","lab-7")["body_md"]` contains `<!-- notes: They will want to start with the transformer. Do not let them — the baseline is 10 lines and it is the number the whole lab is judged against. Circulate during Part C: ... -->`. rehype-raw is wired at frontend/src/components/Course/CourseMarkdown.tsx:153.
```

FIX: Strip at READ, once, on the consumption route — not at publish (the teacher must keep the notes) and not per-consumer (four implementations, and it would break the parity rule). Add `strip_author_comments(body_md: str) -> str` to backend/app/services/lesson_directives.py (regex `<!--(?!\[if).*?-->` with DOTALL, applied after directive extraction) and call it in consumption.py:273 so `data = lesson.to_dict()` is followed by `data["body_md"] = strip_author_comments(lesson.body_md)`. Leave the raw body on the teacher routes (backend/app/views/teacher/lessons.py GET /lessons/<id> and preview_lesson). Apply the same helper to backend/app/views/competition_asset.py:36 for non-manager callers, which currently ships a `<!-- mlarena:quickstart -->` sentinel to students in 12 of the 17 MS2A overviews.

---

## [major/platform] frontend/src/pages/CourseLearner/ModuleOverview.tsx:92-141, CourseLanding.tsx:128-154, LessonReader.tsx:99; type at frontend/src/services/coursesApi.ts:769-775

Competition performance is computed and shipped in my_progress but rendered by no learner page. All three CourseLearner pages call useMyProgress and use only completedSet and next_lesson; ProgressCompetitionCell exists only in the type file. So the answer to 'is competition performance folded into progress?' is: on the backend yes, in the student UI no — the frontend is the consumer that is behind.

EVIDENCE:
```
`grep -rn ProgressCompetitionCell frontend/src` returns only coursesApi.ts:169, :187, :774. `GET /api/academic_courses/11/progress/me` returns a populated `competitions` array ([{competition_id:172, name:'2-Month Survival Prediction', ranked_by:'accuracy', value:null, n_runs:0, best_agent_name:null, last_run:null}, ...]) computed by _competition_results at backend/app/views/teacher/course_content.py:314-380. CourseLanding.tsx:143 uses only `progress?.next_lesson`; ModuleOverview.tsx:95 uses only `completedSet`.
```

FIX: In ModuleOverview.tsx, extend the CompetitionItem card (line 65-90) to take the matching ProgressCompetitionCell from useMyProgress and render `n_runs === 0 ? 'Not submitted' : `${ranked_by} ${value}` (n_runs runs, last <last_run>)`. In CourseLanding.tsx add a per-module competition status chip next to the lesson/minutes badges (line 91-119). Mirror the same rollup as an SDK convenience on my_progress and as an MCP `my_progress` field so all four consumers show it.

---

## [major/platform] backend/app/views/academic_courses/consumption.py:109-136 and :431-454; frontend/src/pages/CourseLearner/CourseLanding.tsx:184-194

'Progress' means lessons-marked-complete and nothing else. A student who submitted to all 17 competitions and clicked no checkbox reads 0%. The bar is labelled 'Your progress' with no qualifier, which actively misinforms.

EVIDENCE:
```
_own_progress_summary (consumption.py:116-135) counts only LessonProgress rows with status COMPLETED over published lessons. CourseLanding.tsx:187 renders `<Text size="sm" c="dimmed">Your progress</Text>` over `{course.progress.completed}/{course.progress.total}`. Live on course 11: content.pct 0.0 while the course's three competitions are listed separately and ignored by the bar.
```

FIX: Relabel the bar 'Lessons read' in CourseLanding.tsx:187. Add a sibling `competitions` line: `<n> of <m> competitions submitted`, derived from `progress.competitions.filter(c => c.n_runs > 0).length`. Longer term, add a `completion` block to my_progress (consumption.py:448-453) that reports, per module, `{lessons_completed, lessons_total, competitions_submitted, competitions_total, required_met}` so a single field answers 'am I done with this session'.

---

## [major/platform] modelmanager/modelmanager/module_competition_link.py:12-24

There is no notion of a competition being REQUIRED for a module, and no target score on the link, so 'done with session 3' is undefinable by construction. The link carries only module_id, competition_id, position and a display label.

EVIDENCE:
```
module_competition_link.py:14-18 is the complete column list: module_id, competition_id, position, label. Neither _serialize_module_competition (consumption.py:82-87) nor _competition_results (teacher/course_content.py:314-380) has anything to compare a student's `value` against.
```

FIX: Add two nullable columns to ModuleCompetitionLink with an Alembic revision: `is_required = db.Column(db.Boolean, nullable=False, server_default=text('false'))` and `target_score = db.Column(db.Float, nullable=True)`. Accept both in AttachCompetitionRequest (backend/app/views/teacher/_schemas.py) and in attach_competition (backend/app/views/teacher/modules.py:301-306); emit both from _serialize_module_competition; carry `target` alongside `value` in _competition_results entries (teacher/course_content.py:350-359); expose fields in AddCompetitionModal.tsx (a 'Required for this session' switch and a 'Target score' number input) and in the SDK's attach_competition (mlarena-sdk/mlarena/client.py:1734).

---

## [major/platform] backend/app/views/teacher/courses.py:95-98; backend/app/views/teacher/modules.py:286-289; frontend/src/pages/CourseAuthoring/AddCompetitionModal.tsx:49

Nothing warns a teacher that the competition they are attaching is invisible to students. The picker route deliberately includes the teacher's own non-public competitions but returns only {id, name}; the attach route validates existence only; the modal renders a bare Select. This is the (a) half of the attached-but-invisible problem, and it is exactly how courses 14's four 404s got shipped.

EVIDENCE:
```
teacher/courses.py:90-98: `query = query.filter(visibility_filter())` (which is `is_public OR owner OR assistant`) then `return jsonify([{"id": c.id, "name": c.name} for c in competitions])`. teacher/modules.py:286-289 is the entire validation: `if Competition.query.get(payload.competition_id) is None: return 400`. AddCompetitionModal.tsx:49: `const options = competitions.map((c) => ({ value: String(c.id), label: c.name }));`.
```

FIX: teacher/courses.py:96 -> `{"id": c.id, "name": c.name, "is_public": bool(c.is_public), "is_started": bool(c.is_started)}`. teacher/modules.py: after the existence check, if `not competition.is_public or not competition.is_started`, still create the link but return the 201 body with `"warnings": ["Competition <n> is not public — enrolled students will get a 404 until you publish it."]`. AddCompetitionModal.tsx:49 -> render the option with a `renderOption` badge 'Hidden' / 'Not started', and show a yellow Alert under the Select when the picked competition is not public. Surface the same warnings array in SDK attach_competition (client.py:1734) and MCP.

---

## [major/platform] backend/app/views/creator_competition/_helpers.py:188-205 (user_can_see_competition) and :216-233 (visibility_filter)

Course enrolment grants no competition visibility. A competition is either world-public or owner-only; there is no 'visible to students of the courses that attach it'. That is why the courseware README documents 'everyone else gets a 404, including enrolled students' as expected behaviour — the platform gives a teacher no way to run a course-private competition, so the only option is to publish it to the whole internet.

EVIDENCE:
```
_helpers.py:194-205 is the complete rule: is_public, else anonymous->False, admin->True, owner->True, CompetitionCreatorAssistant->True. No reference to ModuleCompetitionLink / CourseModuleLink / UserCourseEnrollment anywhere in the file. Live: competitions 179-182 are 404 for the student token; tmp/data_science_practice/courseware/competitions/README.md states the same.
```

FIX: Add a fourth clause to user_can_see_competition (_helpers.py:205): the competition is reachable through ModuleCompetitionLink -> CourseModuleLink to a course where `is_enrolled(course.id) or can_manage_course(course)` (import the helpers from app.views.academic_courses._helpers). Add the matching subquery clause to visibility_filter (_helpers.py:229-233) so list endpoints agree. This is the single change that makes the 'attach to a module' gesture actually mean something for a private cohort.

---

## [major/platform] backend/app/services/lesson_directives.py:113-131, 133-176, 180-196

All three directive resolvers load competitions with `Competition.query.get(...)` and run the leaderboard query with no visibility check, so an ```mlarena:competition id=179``` or ```mlarena:leaderboard id=179``` block in a public course lesson would hand an anonymous reader the hidden competition's name, description and full standings — the exact data /api/competitions/179 refuses. No lesson uses a directive today, so this is latent, but it becomes live the moment the courseware adopts them (see the finding below).

EVIDENCE:
```
lesson_directives.py:118-119 `competition = Competition.query.get(competition_id)` with no user_can_see_competition call; :157-163 build_leaderboard_query is called with only competition_id/is_elo_ranked/aggregate; the returned payload at :127-130 includes name, description, is_public.
```

FIX: In _resolve_competition and _resolve_submit, replace `Competition.query.get(competition_id)` with a lookup that then checks `user_can_see_competition(competition)` (import from app.views.creator_competition._helpers) and raises DirectiveError('competition <n> is not visible to this reader') on failure — which the non-strict consumption path already turns into a dropped-with-warning block. Gate _resolve_leaderboard the same way before calling build_leaderboard_query.

---

## [major/platform] backend/app/views/competition_asset.py:19; mlarena-sdk/mlarena/client.py:592 (write-only); mlarena-mcp/mlarena_mcp/server.py:242-264

Parity break: the competition overview markdown — the document that carries the task description, the data contract and the baseline — is readable over REST but has no SDK method and no MCP tool. The SDK has set_competition_markdown (write) with no getter. An MCP-first student can list a course's attached competitions and get id/name/label and nothing else, so the AI editor the platform advertises cannot see the assignment.

EVIDENCE:
```
`grep -n 'markdown/overview\|def get_competition_markdown' mlarena-sdk/mlarena/client.py mlarena-mcp/mlarena_mcp/server.py` -> no matches. mlarena-sdk/mlarena/client.py:592 defines set_competition_markdown only. mlarena-mcp/mlarena_mcp/server.py:250-263 builds each entry from competition_id/name/label/module_slug/module_title. Manual REST call to /api/competition_asset/172/markdown/overview with the student token returns the full 4218-byte brief.
```

FIX: Add `def competition_overview(self, competition_id: int) -> str` to mlarena-sdk/mlarena/client.py (GET /api/competition_asset/<id>/markdown/overview, return resp.json()['content']) next to competition() at line 179, document it in mlarena-sdk/README.md. Add an MCP tool `get_competition_overview(competition_id)` and an MCP resource `competition://{id}/overview` in mlarena-mcp/mlarena_mcp/server.py alongside lesson_resource (line 348-366), and register it in read_tools.

---

## [major/baseline] modelmanager/modelmanager/competitions.py CompetitionConfiguration (columns at lines 270-318); backend/app/views/creator_competition/benchmark.py:267-290

The platform has no field for 'the score to beat'. run_benchmark executes the reference solution but stores nothing, so the number a student needs is only ever prose inside overview.md — unreachable from the module card, the progress payload, or the leaderboard. `grep -rn benchmark modelmanager/modelmanager/*.py` returns zero hits.

EVIDENCE:
```
`grep -rn 'benchmark' modelmanager/modelmanager/competitions.py modelmanager/modelmanager/evaluation*.py` -> no output. The courseware carries benchmark_expected_score in its own config.py (e.g. tmp/data_science_practice/courseware/competitions/s3-adult-income/config.py: `"benchmark_expected_score": 0.656175`) precisely because the platform has nowhere to put it. my_progress competition cells (teacher/course_content.py:350-359) carry `value` with no comparator.
```

FIX: Add `benchmark_score = db.Column(db.Float, nullable=True)` and `benchmark_score_label = db.Column(db.String(120), nullable=True)` to CompetitionConfiguration (modelmanager/modelmanager/competitions.py, near submission_filename at line 311) with an Alembic revision; write benchmark_score from the benchmark run outcome in backend/app/views/creator_competition/benchmark.py:267-290; expose it in get_competition (backend/app/views/competitions.py:371-397) and in _competition_results entries as `target`; render it on the leaderboard as a reference line and next to the student's value in the ModuleOverview competition card. Add `benchmark_score=` to SDK update_settings (client.py:410).

---

## [major/baseline] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

EVIDENCE:
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

FIX: For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## [major/validation] competition 170 (Bitcoin Intraday, attached to mlp-reference)

Comp 170's leaderboard is unusable as a self-check in two independent ways: it ranks a one-run agent above agents with 2000+ runs on raw mean reward, and it renders every score as 0.00 because the display precision is 2 while returns are order 1e-4.

EVIDENCE:
```
I deployed the template random agent (agent 8508) on 2026-09-02. After a single run it scored 0.029587 and took **rank 1**, above `carry-test` (0.000469, NumberOfRuns 2111) and `__benchmark__` (-0.000024, NumberOfRuns 2173); RewardCi95 is 0.0 for every row. `c.agent_games(8508)` → `competition_context: {'frontend_precision': 2, ...}`; frontend/src/utils/leaderboardFormat.ts:16-25 `formatScore` calls `toLocaleString` with `minimumFractionDigits: precision, maximumFractionDigits: precision`, so 0.000469 renders "0.00" and -0.000024 renders "-0.00".
```

FIX: Two changes. (1) Set this competition's `evaluation_frontend_precision` to 6 (creator Settings → Evaluation) so the board shows 0.000469 rather than 0.00. (2) In frontend/src/utils/leaderboardFormat.ts, make `formatScore` fall back to significant-digit formatting when `Math.abs(value) > 0 && Math.abs(value) < 0.5 * 10**-precision`, so a mis-set precision can never render a non-zero score as 0.00. Separately, for `IsContinuous` competitions rank on MeanReward30d or suppress agents below a minimum run count, so one lucky hour cannot top a 2000-run board.

---

## [major/competition] s7-nlp-1/lab-7 and s8-nlp-2/lab-8 (competitions 174, 178, 165)

Neither lab mentions its attached competition. The deliverable of both labs is "a merged PR in your project repository" that the platform never sees, so a student has no externally-verified way to know they got it right — which is exactly the failure the course owner is asking about. The platform already has the mechanism to fix this and it is used nowhere.

EVIDENCE:
```
`grep -ci "competition|174|178|leaderboard|ml-arena|submit" s7-nlp-1/lab-7.md` = 0. In `lab-8.md` the only hit is line 188, "Point it at your submission before the leaderboard does" — no id, no link. Meanwhile `_module.json` for s7 lists competitions 174 and 178 and s8 lists 165. `backend/app/services/lesson_directives.py:203-205` registers the directive types `competition`, `leaderboard` and `submit` (fenced as ```mlarena:competition id=174```), and `grep -rn '```mlarena:'` over all 73 published lesson bodies returns 0 hits.
```

FIX: Add a "Part F — put it on the board (5 min)" to Lab 7: a ```mlarena:competition id=174``` directive, then "Run the same two models on the Clinical Note Triage data and submit `submission.csv`. The `tfidf-logreg-starter` on that board scores **0.4797 F1-macro** and the best current entry scores **0.6084**. You have done the lab when your fine-tuned model is above 0.4797 and you can reproduce that number locally within 0.02." Add the same shape to Lab 8 pointing at 165 with its own threshold. Do this only after competition 174's dataset is restored.

---

## [major/baseline] competition 174 (Clinical Note Triage)

The overview names a baseline but gives no number, so "you have done the lab" is undefined; and the leaderboard column is labelled `reward` rather than the F1-macro the overview promises, so a student cannot tell what 0.48 is or which direction is better. The number the student needs already exists on the board — it is just not written down anywhere a student reads.

EVIDENCE:
```
Overview, verbatim and complete on this point: "A **TF-IDF + linear model** (logistic regression / linear SVM) is the starter baseline and is famously hard to beat on clinical text — a good lesson in itself." No figure appears anywhere in the overview. `client.leaderboard(174)`: `Metric == "reward"` and `MetricsSchema is None` for all 27 rows; values span 0.020908 (`__benchmark__`, the constant-class floor) to 0.608398 (`pubmedbert-logreg-v2`), with `tfidf-logreg-starter` at 0.479719. Compare competition 178, which carries a full `MetricsSchema` (`{'key':'reward','label':'Accuracy','higher_is_better':True,'precision':4,...}`) and a five-row measured baseline table in its overview.
```

FIX: (a) Add to 174's overview, under "## Task", the table it is missing: `__benchmark__` (constant class) 0.0209 | TF-IDF + logistic regression starter 0.4797 | best entry to date 0.6084, followed by "F1-macro, higher is better, range 0 to 1. Below 0.4797 the transformer has not paid for itself." (b) Set `metrics_schema` on competition 174 so the leaderboard column reads "F1-macro" with `higher_is_better: True` instead of the generic `reward`.

---

## [major/baseline] competition 165 (GSM8k)

The GSM8k overview states no target score, no metric range and no direction; it documents a return contract that contradicts the template the platform actually serves; and it points at a reference solution file that is not distributed with the competition and names a checkpoint absent from its own cache list. A student has nothing to aim at and nothing to start from.

EVIDENCE:
```
Overview: "Score = number of replies matching the gold answer to within `1e-6`, summed across all delivered subsets" — the 0..50 ceiling has to be inferred from "K=5 disjoint subsets of 10 questions"; the leaderboard's `Metric` is the bare string `score` and the best of 866 agents is 18.00. Contract drift: overview says `agent.answer(questions: list[str]) -> list[float]`, while `competition(165)["agent_template"]` says "expects a tuple: (solutions: list[float], thinking_traces: list[str])". Missing file: "See `agent_template.py` ... and `Agent.py` for a SmolLM3-3B baseline" — `client.datasets(165)` returns `{"datasets": []}`, and `SmolLM3-3B` is not in the overview's own pre-populated-cache list (which has `HuggingFaceTB/SmolLM2-1.7B-Instruct`).
```

FIX: In 165's overview: (1) replace the scoring sentence's tail with "Score is the count of correct replies out of the 50 questions sent (5 subsets x 10), higher is better. The provided random-guess starter scores about 0; a cached 1.5B instruct model with 3-shot prompting scores around 10; the best entry to date is 18." (2) change the interface line to the tuple form the template serves: `agent.answer(questions: list[str]) -> tuple[list[float], list[str]]`, noting the bare `list[float]` is still accepted. (3) Either attach `Agent.py` as a competition dataset file or delete the reference and correct `SmolLM3-3B` to `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

---

## [major/competition] competition 165 (GSM8k) — agent_template

The starter agent the platform serves is truncated mid-class: it calls `self._solve_one(q)` but that method was cut off, so the template raises on its very first call. The docstring tells the student to "override" a method that does not exist, which sends them looking for a base class rather than at the missing lines.

EVIDENCE:
```
`competition(165)["agent_template"]` ends `'...return solutions, traces\n\n    \n'` (943 chars). Executing it as the platform would: `hasattr(Agent(), "_solve_one") -> False`; `Agent().answer(["Natalia sold clips to 48 friends..."]) -> AttributeError: 'Agent' object has no attribute '_solve_one'`. Its own docstring: "Override `_solve_one` with your own logic."
```

FIX: Re-upload the template with the missing method appended, so it runs unchanged and scores the floor:
```
    def _solve_one(self, question: str) -> tuple[float, str]:
        """Replace this. Returns (numeric answer, reasoning trace)."""
        return float(self.rng.randint(0, 100)), "random guess"
```
Verify by exec'ing the served template and calling `answer([...])` without editing it.

---

## [major/baseline] competitions 48, 49, 43, 65

None of the four competitions states, anywhere a student can reach, what score counts as done, which direction the metric runs, or what its range is. The published description is a one- or two-sentence stub lifted from the Farama docs; two of them are broken sentences where the hyperlink text was stripped. The only baseline visible is a `__benchmark__` leaderboard row that no page explains, and on comp 43 it sits at rank 661 of 708.

EVIDENCE:
```
`c.competition(48)['description']` == "This environment is part of the Classic Control environments which contains general information about the environment.\n\nSource: https://gymnasium.farama.org/environments/classic_control/cart_pole/" (same stub for 49). `c.competition(43)['description']` == "A classic rocket trajectory optimization problem". `c.competition(65)['description']` == "This environment is part of the classic environments . Please read that page first for general information." — note the orphaned space before the period. The competition payload has no `overview`/`rules`/`evaluation`/`metric` key at all (`sorted(d.keys())` on all four). Measured from the leaderboards: comp 48 `__benchmark__` 20.7 (a random agent — `qa-cartpole-random` scores 23.3) against a 500 ceiling; comp 49 `__benchmark__` -200.0, which is the floor and also the score of every other entry; comp 43 `__benchmark__` -266.6 at rank 661/708 while 460 of 708 entries already score >= 200.
```

FIX: Write an overview.md for each of the four and publish it, each ending with a Target line, in the shape the courseware/competitions/ packages already use: comp 48 "Metric: mean episode reward over seeded episodes, higher is better, range 0-500. A random policy scores about 21. Target: 475+ (CartPole-v1 is considered solved at 475 averaged over 100 episodes)." comp 49 "Metric: mean episode reward, higher is better, range -200 to about -90. Every episode costs -1 per step and truncates at 200, so -200 means you never reached the flag — it is both the random score and the floor. Target: -110." comp 43 "Metric: mean episode reward, higher is better. A random policy scores about -180; the median submission scores 218. Target: 200+ (LunarLander is considered solved at 200 averaged over 100 episodes)." comp 65 "Ranked by ELO against the live population, not by mean reward; 1200 is the starting rating and the benchmark sits at 1248. Target: beat 1248 over at least 50 games."

---

## [major/baseline] competitions 8, 47, 48, 49, 43, 65, 165, 173, 174, 169, 170, 171

Twelve of the seventeen attached competitions state no numeric floor. Each says only that a Colab notebook holds a 'random-action baseline'. A student cannot tell a working submission from a broken one — the first CartPole agent scoring 21 has no way to know 21 IS the random floor.

EVIDENCE:
```
Grepping every overview for a number: comp 48 (CartPole) and 49 (MountainCar) and 47 (CarRacing) and 170 contain only "a minimal **random-action** baseline you can deploy in a couple of minutes"; comp 173 says "The starter baseline flattens the pixels into a random forest" with no score; comp 174 says "A TF-IDF + linear model ... is the starter baseline and is famously hard to beat" with no score; comp 43 quotes "solved around 200" but never the floor. By contrast comp 177 carries a six-row measured table with error bars, 176 states 0.500 / 0.813, 172 states "always guessing alive scores ~0.59. That is the bar to beat."
```

FIX: Add a '## Baselines' table to each overview in the exact form comp 177 already uses. Numbers I measured and that can be pasted in today: CartPole-v1 random = 20.98 (sd 10.89, 300 eps) and heuristic `0 if angle+0.5*angvel < 0 else 1` = 500.00; LunarLander-v3 random = -187.54 (sd 115.01, 200 eps); FrozenLake-v1 4x4 slippery random = 0.0120. For 173 and 174, run the shipped starter notebook once and paste its score. Enforce it by adding a `baselines` block to config.py that build_competitions.py refuses to publish without.

---

## [major/baseline] competition 47 (attached to s6-computer-vision-2)

The competition page is three lines of copied upstream boilerplate. It states no target, no metric direction, no range, and the leaderboard labels a Gymnasium reward as 'accuracy' — so a student cannot tell whether -33.876 is good, bad, or which way the number should move.

EVIDENCE:
```
Full `c.competition(47)['description']`: "This environment is part of the Box2D environments which contains general information about the environment.\n\nSource: https://gymnasium.farama.org/environments/box2d/car_racing/". `c.datasets(47)` -> `{"datasets": []}`. `c.leaderboard(47)` -> one row: AgentName `__benchmark__`, MeanReward -33.87589, Metric `accuracy`, NumberOfRuns 1, RewardCi95 None.
```

FIX: Replace the description with the contract in numbers: "CarRacing-v3. Your agent sees a 96×96×3 RGB frame and returns a 3-vector (steer, gas, brake). Metric: mean episode return over N episodes, higher is better. A random policy scores about -34 (that is the __benchmark__ row). Staying on the track for a full lap scores roughly 900. Anything above 0 means you are driving rather than spinning; 300+ is a pass." Also fix the metric label — `evaluation.metric` for comp 47 is 'accuracy' on a reward-valued column, which is the platform-wide mislabelling showing up where it does concrete harm.

---

## [major/platform] frontend/src/pages/CourseLearner/ (CourseLanding.tsx, ModuleOverview.tsx, LessonReader.tsx) + frontend/src/hooks/courseLearner/useMyProgress.ts

The student's own competition results are fetched by the learner UI and then never rendered. The one piece of objective evidence the platform already computes about a student dies in the client.

EVIDENCE:
```
`GET /{course_id}/progress/me` returns `competitions` (consumption.py:466), built by `_competition_results` with `best_agent_name`, `value`, `n_runs`, `last_run` per competition. `MyProgress` in services/coursesApi.ts:774 declares `competitions: ProgressCompetitionCell[]`. useMyProgress.ts returns the whole object. But `grep -rn 'progress.competitions' frontend/src/pages/CourseLearner/` -> no hits; the only `.competitions` references in CourseLearner are `module.competitions` (the static attachment list). CourseLanding.tsx:184-192 renders only the lesson-count bar.
```

FIX: Render the `competitions` array on CourseLanding and ModuleOverview as a row per competition: name, your best value, and — once `mlarena:target` exists — the target beside it with met/not-met. The data is already on the wire; this is a rendering change, no new endpoint.

---

## [major/baseline] backend/app/views/teacher/course_content.py:350-359 (payload consumed by academic_courses/consumption.py:466)

Even where the platform reports a student's competition score, it reports a bare number with nothing to compare it to. `value: 0.907` answers 'what did I get' and never 'did I pass', which is precisely the question the owner is asking about.

EVIDENCE:
```
The per-competition cell is {competition_id, name, best_agent_name, ranked_by, value, n_runs, last_run} — no target, no threshold, no pass flag. There is nowhere on the platform for such a number to live either: `Evaluation` (modelmanager/modelmanager/evaluations.py) has metric, metric2, is_elo_score, frontend_precision, metrics_schema and no expected/target score; `ModuleCompetitionLink` (module_competition_link.py:12-18) has only module_id, competition_id, position, label. `grep -rn benchmark_expected_score modelmanager/` -> nothing; the measured number exists only in the courseware packages' config.py.
```

FIX: Do not add a target column — that is why option (c) loses as a primary. Author the number in the lesson via `mlarena:target id=182 metric=accuracy min=0.913`, resolve it server-side against the existing per-user leaderboard query, and return {min, your_best, met} in the directive payload and in `my_progress.targets`. The threshold then lives beside the lesson that teaches it, versioned in git with the rest of the courseware, and changes without a migration.

---

## [major/platform] backend/app/views/leaderboard.py:204-210

A blanket `except Exception` swallows the 404 that `get_visible_competition_or_404` raises and returns 500 instead. The student gets a server error for the exact call PAIE Lab 4 tells them to make. This is also a Fail-Fast violation per CLAUDE.md.

EVIDENCE:
```
Live with the student key: `c.leaderboard(48, top=3)` -> OK, DataFrame of 3. `c.leaderboard(182, top=3)` -> `HTTPError: 500 Server Error ... /api/leaderboard/competition/182?limit=3`. `c.leaderboard(999999, top=3)` -> the same 500, so it is not about competition 182 specifically. `get_leaderboard` (leaderboard.py:75) calls `get_visible_competition_or_404` (creator_competition/_helpers.py:208-213), which calls Werkzeug's `abort(404)`; `NotFound` is an `Exception` subclass, so the handler at leaderboard.py:204 catches it and leaderboard.py:210 returns `{"error": "Failed to fetch leaderboard"}, 500`.
```

FIX: Re-raise HTTP exceptions before the generic handler: `except HTTPException: raise` immediately above `except Exception as e:` at leaderboard.py:204 (import `from werkzeug.exceptions import HTTPException`). A hidden or missing competition must answer 404 so the SDK raises CompetitionNotFoundError and the student sees 'this competition is not open to you', not 'the server broke'.

---

## [major/platform] frontend/src/components/Course/directiveCards.tsx:154-169

`matchDirective` resolves a fence to a payload by `payload.competition_id` only, then falls back to the first directive of that type. A lesson with two checkpoints would render the first one's payload twice — the design needs a generic id match before it can put more than one checkpoint on a page.

EVIDENCE:
```
directiveCards.tsx:162-167: `if (args.id != null) { const byId = sameType.find((d) => String(d.payload.competition_id) === String(args.id)); if (byId) return byId; } return sameType[0];`. Every existing directive type keys on a competition, so the bug is latent today (0 directives are in use in either course).
```

FIX: Match on the resolved directive's own `args.id` first — `sameType.find((d) => String(d.args.id) === String(args.id))` — and keep the competition_id comparison as a second attempt for the three existing types. Drop the `sameType[0]` fallback when `args.id` was supplied: returning the wrong card silently is worse than the DirectivePlaceholder.

---

## [major/platform] COLLAPSE and EXTENSION of findings 5, 16, 24, 31, 122, 144, 160, 188 — backend/app/views/

Eight findings report one bug, and all eight under-count it. The audit names three routes; there are fifteen unguarded sites, and the correct pattern already exists in the same file as one of them.

EVIDENCE:
```
`grep -rn 'get_visible_competition_or_404(' backend/app/views/ | grep -v 'def \|import'` = 22 call sites. `grep -rn 'except HTTPException' backend/app/views/` = 3 (ranking.py:117, competitions.py:457, direct_attache_agents/monitor.py:171). A scan for call sites inside a `try:` whose nearest `except` is a bare `except Exception` with no HTTPException guard returns 15: leaderboard.py:77, competition_asset.py:23 and :122, teams.py:20/:44/:202, agent_attached_result.py:140, competitions.py:458/:595/:802/:842/:923, direct_attache_agents/file.py:356, status.py:208/:306. Verified live: /api/leaderboard/competition/179 and /99999 both return 500 {"error":"Failed to fetch leaderboard"}; /api/competition_asset/179/markdown/overview and /99999/... both return 500. The guarded route behaves correctly — /api/competitions/179 returns a real 404.
```

FIX: Add `except HTTPException: raise` immediately above each of the 15 bare `except Exception` blocks, copying competitions.py:457-460 verbatim including its comment — or better, extract one `@propagates_aborts` decorator and apply it, since 22 call sites will keep growing.

---

## [major/platform] COVERAGE HOLE — backend/app/services/lesson_directives.py, frontend/src/components/Course/directiveCards.tsx, courseware/tools/build_slides.py

Finding 180 recommends lesson directives as the primary design for the validation layer. Nobody checked whether the feature has ever run. It has not — not in these two courses, and not anywhere on the platform — and three known defects sit directly in its path, so the recommendation is being made about untested machinery.

EVIDENCE:
```
`SELECT count(*) FROM lesson WHERE body_md LIKE '%mlarena:%'` = 0 across the whole lesson table, not just courses 14/15. The three defects: build_slides.py:76 FENCE_RE is `^```([\w+-]*)\s*$`, which cannot match an info string containing `:` or a space, so an opening ```` ```mlarena:checkpoint ```` fence never sets in_fence while the closing bare ``` does — every subsequent `---` is swallowed and the deck collapses (finding 189, confirmed by reading split_slides at :92-104). directiveCards.tsx:154-169 matches a fence to a payload by competition_id then falls back to the first directive of that type (finding 190). lesson_directives.py:113-131 resolves `Competition.query.get(...)` with no visibility check (finding 168).
```

FIX: Before committing to option (b), prototype one directive on one lesson through `POST /api/teacher/lessons/<id>/preview` (teacher/lessons.py:241, strict mode) and rebuild that module's deck. Fix build_slides.py's FENCE_RE to `^```(\S.*)?$` and directiveCards' matcher first — both are small, and both are prerequisites, not follow-ups. Note the resolver already returns `is_public` and `is_started`, so a competition directive can carry the reachability signal finding 159 says the module card lacks.

---

## [major/platform] SEQUENCING COUPLING on finding 156 — backend/app/views/academic_courses/legacy.py:29-90

The unauthenticated join-code leak is real, and it is also currently the only way anyone discovers a code without being handed one. Fixing it in isolation, before the codes have been distributed to the cohort, silently removes the enrolment path this audit just established is working.

EVIDENCE:
```
Anonymous `curl 'https://ml-arena.com/api/academic_courses/?show_all=true'` returns 200 and an array of 15 courses; every element carries join_code and enrollment_link, including private ones (course 13 'Knowledge Graph & Embeddings', visibility=private, join_code A9QEVXAM). Response keys include both fields. The route at legacy.py:29 has no auth decorator and serialises via the full AcademicCourse.to_dict().
```

FIX: Strip join_code and enrollment_link from the listing serialiser (keep them in the teacher-scope payload only) — and in the same change, distribute GR1WFC63 / N1DX2QA4 to the MS2A cohort, or the enrolment path goes dark the moment the leak closes.

---

## [major/platform] backend/app/views/competition_asset.py:19-46

The route that serves every competition overview — the sole carrier of every baseline in this audit — is unauthenticated and writes to disk on a GET. A read of an id whose overview file is absent creates that file containing the placeholder text, which means the placeholder overviews the audit attributes to lazy authoring may have been written by the platform itself on first read.

EVIDENCE:
```
competition_asset.py:20-21 `@bp.route("/<int:competition_id>/markdown/overview", methods=["GET"])` followed directly by `def get_markdown_overview` — no decorator, no rate limit, unlike the PUT/POST at :49 which carries @admin_required. Lines 27-35: `if not os.path.exists(overview_path): ... manage_storage.write_file(overview_path, DEFAULT_MARKDOWN_CONTENT)` where DEFAULT_MARKDOWN_CONTENT (line 14) is exactly `**Competition Overview**\n\nWrite your markdown here.` Live: competitions 169 and 170 return content beginning with that literal string (findings 106, 120, 161).
```

FIX: Make the GET read-only: return 404 when overview.md is absent rather than creating it, and move file creation to the authoring PUT. Add @auth_required('user') or leave it public deliberately, but decide — right now it is the one competition surface with no access control at all.

---

## [major/platform] EXTENSION of findings 128 and 169 — mlarena-mcp/mlarena_mcp/server.py:110-320

The audit reports that MCP cannot read a competition overview. It also cannot obtain the competition's data. An MCP-first student can join a course, read every lesson and submit an agent, but can neither read the assignment nor download the dataset it is about.

EVIDENCE:
```
Full tool list from server.py: join_course, list_my_courses, set_active_course, get_course, get_module, get_lesson, next_lesson, my_progress, mark_lesson_complete, list_attached_competitions, submit_agent, agent_status, leaderboard. There is no overview tool and no datasets/download tool. list_attached_competitions (:242-264) returns only competition_id, name, label, module_slug, module_title. The SDK has both `datasets()` and `download_dataset()` and neither is surfaced.
```

FIX: Add two MCP tools backed by existing SDK/REST calls: one reading GET /api/competition_asset/<id>/markdown/overview, one wrapping client.datasets(<id>) to return labels plus signed URLs. Both are pure compositions of public routes, so neither breaks the parity rule.

---

## [minor/platform] mlarena-sdk/mlarena/client.py:240 download_dataset()

download_dataset() has no way to select individual files, so touching Session 1's own competition dataset at all means pulling all seven weather CSVs — 4.45 GB — even to look at the schema of one year. The lesson that teaches "the single biggest performance lever in data collection is not reading what you do not need" is attached to a competition whose data can only be fetched whole.

EVIDENCE:
```
Signature: `def download_dataset(self, competition_id: int, dest_dir: str = ".") -> list[str]` — "Download every published dataset file for a competition into `dest_dir`", loops `for ds in payload.get("datasets", [])` / `for f in ds.get("files", [])` with no filter. datasets(177) file sizes: 722112434 + 720558948 + 721757245 + 721423092 + 723244638 + 720659712 + 122421412 bytes = 4.45 GB.
```

FIX: Add an optional filter: `def download_dataset(self, competition_id, dest_dir=".", files: list[str] | None = None)` and skip any file whose `label` is not in `files` when the argument is given. Keep the current behaviour when it is omitted so the starter notebooks are unaffected.

---

## [minor/platform] frontend/src (leaderboard rendering) and backend leaderboard payload

Every competition's creator-side reference score is published on the public board as a row literally named `__benchmark__`, unexplained and ranked as if it were a competitor. It is the one measured baseline the platform always has, and it is presented in the least usable way possible.

EVIDENCE:
```
`grep -rn "__benchmark__" frontend/src` returns nothing — the frontend does not special-case it; it is written by backend/app/views/creator_competition/benchmark.py:325,342,534 and lifecycle.py:584 as an ordinary agent name. Consequences on live boards: 43's `__benchmark__` ranks 661 of 708 at -266.62; 8's ranks 1038 of 1084; 165's ranks 833 of 866 at 0.0; 65's ranks 1 with Elo 1248; 172's ranks 81 of 82. A student sees a row named `__benchmark__` under the creator's username and has no way to know it is the reference.
```

FIX: Flag it in the payload (`"IsReference": true`) and render it as a pinned, visually distinct row labelled "Reference (creator benchmark)" that is excluded from the competitor ranking, the way a par line is drawn rather than played. Then the contract's Reference number is verifiable by every student on every competition without the creator writing anything.

---

## [minor/platform] backend/app/views/academic_courses (module_overview) — courses 14 and 15

The module payload advertises competitions the caller cannot open and gives no reachability signal, so the student's first click is a 404 with no explanation.

EVIDENCE:
```
As the student token: module_overview("python-ai-engineering","s1-git-and-packaging") returns `"competitions": [{"competition_id": 179, "label": "textstats correctness", "name": "PAIE S1 — textstats"}]` and `"is_enrolled": false`, while GET /api/competitions/179 with the same token returns 404. Both courses also have `join_code: null` and `enrollment_link: null`, so the student cannot self-enroll to gain the course side-channel that competitions.py:213-217 provides.
```

FIX: Include `is_visible_to_me` (or `is_public` plus the enrollment-derived grant) on each competition entry in the module payload, and have the CourseLearner module page render an unreachable competition as "not open yet" rather than a live link. Independently, issue a join_code for courses 14 and 15 so enrollment — and therefore the non-public course-competition side-channel and all progress tracking — actually works.

---

## [minor/content] competitions 65 (Connect-Four) and 169 (SuperTuxKart) — ELO-ranked

Both are ranked by ELO and neither page says so; the board shows a reward column that contradicts the rank, which makes any absolute baseline meaningless for these two.

EVIDENCE:
```
Leaderboard payload: 65 `IsEloRanked: true`, rows ordered 1248/1200/1184/1184 Elo while MeanReward reads 0.40/0.0/-0.4/-0.8; 169 `IsEloRanked: true`, rank 1 `__benchmark__` MeanReward 946.41, rank 2 `Luigi` 7940.42, rank 3 `flexfix-kart-vmcheck` 11210.39 — reward increasing as rank worsens. Neither overview contains the string "ELO". project-tracks.md does explain it ("your rating moves when *other people* submit") but that lesson is in a different module from s10 and mlp-project only.
```

FIX: Add to both overviews, in place of an absolute baseline: "Ranking: ELO against the current population, starting at 1200. Your rating moves when other people submit. There is no absolute bar — the target is to finish above the reference agent, which currently sits at <E>. The reward column is shown for information and does not determine rank." And suppress or de-emphasise the MeanReward column on ELO-ranked boards so it cannot be read as the score.

---

## [minor/content] competition 170 (Bitcoin Intraday Trading) — leaderboard display

Even once an overview is written, the board cannot show a baseline: every score rounds to 0.00 at the configured precision.

EVIDENCE:
```
Leaderboard rows for 170: `carry-test` = 4.686e-4, `__benchmark__` = -2.439e-5, `groupg7` = -4.026e-5, with `FrontendPrecision: 2`. All three render as 0.00. Contrast 177, which sets FrontendPrecision 4 for a metric on a 0-1 scale.
```

FIX: Set the competition's frontend precision to at least 6 (`client.update_settings(170, ...)`), or change env.py to report the metric in basis points so the numbers are legible, and state the unit in the Baseline block. A baseline that renders as 0.00 next to a score of 0.00 is not a baseline.

---

## [minor/platform] frontend/src/pages/CourseLearner/CourseLanding.tsx:100-118 (badge text at line 115)

The module-index badge prefers the competition's `label` over its `name`, and the MS2A labels are full sentences: 18 of 22 exceed 40 characters and the longest is 135. Inside a Mantine Badge these collapse to an unreadable ellipsised stub or blow out the row. ModuleOverview.tsx:76-77 already does it correctly (name as the title, label as dimmed subtext).

EVIDENCE:
```
Measured across all _module.json files: max label length 135 (comp 177, 'Global Weather Forecast — a source that never stops: observations arrive daily and the forecast is scored once the weather has happened'), 18/22 over 40 chars. CourseLanding.tsx:115: `{c.label || c.name || `#${c.competition_id}`}` inside `<Badge size="sm" ...>`.
```

FIX: CourseLanding.tsx:115 -> `{c.name || c.label || `#${c.competition_id}`}`, and change the Tooltip label at line 101 from the static 'Open competition' to `{c.label ?? 'Open competition'}` so the long sentence is still reachable on hover.

---

## [minor/platform] frontend/src/pages/CourseLearner/CourseLanding.tsx:218 and :220

The course landing's module index is titled 'Competitions' and its empty state reads 'This course has no competitions yet.' On a 12-module, 73-lesson course the heading over the session list says Competitions, which is the single most confusing label on the student path.

EVIDENCE:
```
CourseLanding.tsx:217-225: `<Title order={3}>Competitions</Title>` ... `{orderedModules.length === 0 ? <Text c="dimmed">This course has no competitions yet.</Text> : orderedModules.map(...)}` — the mapped items are modules (ModuleRow), not competitions.
```

FIX: Change line 218 to `<Title order={3}>Sessions</Title>` (or 'Course content') and line 220 to 'This course has no sessions yet.' Keep 'Competitions' only for the SimpleCompetitionRow branch (line 34-62), which really is a bare competition entry.

---

## [minor/platform] backend/app/views/competitions.py:274-320 (_resolve_user_course_for_competition)

The 'Back to <course>' breadcrumb on a competition page resolves only for enrolled students and managers, so a student browsing a public course who opens one of its competitions loses the thread back to the course entirely. Combined with the missing self-enrol path this means nobody currently gets the breadcrumb on courses 14/15.

EVIDENCE:
```
competitions.py:313-315: `managed = next((c for c in courses if can_manage_course(c)), None); course = managed or next((c for c in courses if is_enrolled(c.id)), None); if course is None: return None`. The consumer is frontend/src/components/CompetitionHeader.js:56-66.
```

FIX: Add a third fallback in competitions.py:313-315: when neither managed nor enrolled matches, fall back to the first course with `can_view_course(course)` true (public/unlisted), returning it with `can_manage: False`. The header link is read-only navigation and leaks nothing a public catalog does not.

---

## [minor/platform] backend/app/views/academic_courses/consumption.py:406-428 (_next_lesson) and :431-440 (my_progress)

'Where do I start?' is only answerable to an enrolled student: next_lesson lives behind my_progress, which 403s otherwise, so a public-course browser gets no entry point and CourseLanding renders no Start button. _next_lesson also ignores lesson.gated and parent_lesson_id nesting, so it can point at a lesson the caller cannot open.

EVIDENCE:
```
consumption.py:439-440 `if not (is_enrolled(course_id) or can_manage_course(course)): return 403`. CourseLanding.tsx:146 `const continueButton = course.is_enrolled && next ? (...) : null;` — nothing renders for a non-enrolled visitor. _next_lesson's filter at consumption.py:420 is `if lesson.is_published and lesson.id not in completed` with no gated check.
```

FIX: Add `first_lesson` to the course_landing payload (consumption.py:194-210) — the lowest (module.position, lesson.position) published, non-gated lesson — and render it in CourseLanding.tsx:146 as a 'Start reading' button when not enrolled. Add `and not lesson.gated` to the _next_lesson filter at consumption.py:420 unless the caller is enrolled.

---

## [minor/platform] mlarena-sdk/mlarena/client.py (project Milestone 1)

The project requires a first-class platform team by Milestone 1, and the module drives every other action through the SDK, but the SDK has no team methods at all. The routes exist and accept the student bearer token, so this is a pure parity gap against the repo's own frontend/SDK parity rule.

EVIDENCE:
```
`grep -n "team" mlarena-sdk/mlarena/client.py` → no matches. backend/app/views/teams.py exposes 12 routes (create, invite, respond, leave, delete, search, ...). Bearer auth works on them: `GET /api/teams/competition/171/team` with the student token → 404 `{"message":"No team found"}`, `GET /api/teams/invitations/received` → 200 `[]`. project-brief.md: "A team is a **first-class object on the platform**, not a line in an email. You create it on the competition page and invite your partner".
```

FIX: Add `team(competition_id)`, `create_team(competition_id, name)`, `invite_to_team(team_id, username)`, `received_invitations()`, `respond_to_invitation(invitation_id, accept)` to mlarena-sdk/mlarena/client.py as thin wrappers over the existing `/api/teams/*` routes, and add the corresponding lines to project-brief's Milestone 1 so declaration is one script.

---

## [minor/validation] competitions 172 and 8 (leaderboard payload)

The central discipline of s3 — never rank two models whose gap is inside the noise — cannot be applied to the leaderboards the two modules attach, because the leaderboard exposes no dispersion at all.

EVIDENCE:
```
`model-selection-and-validation.md` teaches "Read the spread, not just the mean" and "If two models differ by less than one fold standard deviation, you have not shown that one is better." But `c.leaderboard(8)` returns `RewardCi95 = None` and `NEpisodes = 0` for all 1084 rows (including the `__benchmark__` row, which has `NumberOfRuns = 25`), and `c.leaderboard(172)` the same for all 82 rows. Comp 8's top three are 0.9990 / 0.9980 / 0.9980, each with `NumberOfRuns = 1` — a 0.001 gap on single runs of a task whose permutations are re-randomised every episode.
```

FIX: Platform: populate `RewardCi95` and `NEpisodes` in the leaderboard payload for competitions that run multiple episodes/runs (they are already columns in the response; they are just null). Courseware, meanwhile: add one sentence to `model-selection-and-validation.md` under "Local CV and the leaderboard": "ML-Arena's leaderboard reports a single mean with no interval, so a 0.001 gap between two rows is not evidence of anything — treat adjacent ranks as tied and keep your own fold standard deviation as the ruler."

---

## [minor/platform] competition 165 (GSM8k), engine 19

The Session 8 competition's GPU VM has been reporting unhealthy all evening, so a student submitting today gets a queued agent and no score, with nothing on the competition page explaining why.

EVIDENCE:
```
`client.competition(165)["engine"]` at 20:18 UTC: `{'id': 19, 'k8s_workload_value': 'local_vm', 'vm_health_checked_at': '2026-09-02T20:18:09.238110', 'vm_health_ok': False}`; re-polled at 20:35: `'vm_health_checked_at': '2026-09-02T20:35:55.183258', 'vm_health_ok': False`. The SDK's own docstring for `competition()`: "`vm_health_ok=False` means the GPU VM is currently unreachable and submissions will queue rather than run." The 866-row leaderboard's most recent runs are from 2026-06-23.
```

FIX: Bring engine 19's local_vm back up (check the vmapi bridge on the GPU host and re-run the health poll), or surface the state to students: the competition detail already carries `vm_health_ok`, so render a banner on the competition page when it is False rather than silently accepting deployments that queue. Until then, do not attach 165 to a session whose lab week is running.

---

## [minor/platform] competitions 48, 49, 65 (leaderboard Metric column)

The leaderboard labels the score column `accuracy` on three of the four competitions while the number displayed is mean episode reward or an ELO-ranked mean outcome. A student who has just been taught "the only honest metric is episode return" sees a CartPole score of 500 labelled accuracy and an ELO-ranked Connect-Four score of 0.4 labelled accuracy.

EVIDENCE:
```
`c.leaderboard(48)` -> `Metric` == 'accuracy' for every row, `MeanReward` 500.0 / 23.3 / 20.7. `c.leaderboard(49)` -> `Metric` == 'accuracy', `MeanReward` -200.0. `c.leaderboard(65)` -> `Metric` == 'accuracy', `IsEloRanked` True, `EloScore` 1248. Comp 43 is correct: `Metric` == 'reward'.
```

FIX: Set the metric name on competitions 48, 49 and 65 to `reward` (matching comp 43) so the leaderboard column reads what it is. Comp 65 additionally displays both `EloScore` and a `MeanReward` labelled accuracy while ranking by ELO — surface only the ELO column when `IsEloRanked` is true.

---

## [minor/content] 92 of 102 lessons

Teacher speaker notes ship raw to SDK and MCP consumers. Relevant to this design because it settles a question about it: body_md is served verbatim, so a checkpoint's expected answer cannot be hidden from an SDK student — which is fine (checkpoints are diagnostic), but it must be a deliberate choice, not a discovery made later.

EVIDENCE:
```
92 of 102 lesson files contain an HTML comment; 109 comment blocks total. Example, ms2a-machine-learning-practice/s3-tabular-models/model-selection-and-validation.md:6-9: "<!-- notes: 35 minutes, the lesson that decides whether their project is worth anything. Do the leakage catalogue as a quiz: show each snippet, ask the room what is wrong, then explain. Budget 12 minutes for leakage alone — it is the part they will actually get wrong. -->". student_walk.py already flags this as `speaker-notes-in-body`.
```

FIX: Strip `<!-- notes: ... -->` in publish_mlarena.py before upload and keep them in the markdown source for build_slides.py, which is the only consumer that needs them. Separately, record in courseware/README.md that checkpoint `expect` values are intentionally public. Note the one genuinely useful thing in those notes — 'Do the leakage catalogue as a quiz: show each snippet, ask the room what is wrong' — is exactly a checkpoint that was written for the teacher and never given to the student.

---

## [minor/platform] backend/app/views/academic_courses/_schemas.py:72-83

LessonProgressContext is `extra='forbid'`, so any client that starts sending checkpoint results before the server declares the field gets a 400 rather than a tolerated no-op. This is correct Fail-Fast behaviour but it fixes the rollout order: server first, then SDK/MCP/frontend.

EVIDENCE:
```
`model_config = ConfigDict(extra="forbid")` at _schemas.py:81, with `course_id: Optional[int] = None` as the only field. `mark_lesson_complete` (consumption.py:376) validates the body through it before touching progress.
```

FIX: Add `checkpoints: Optional[dict[str, bool]] = None` to LessonProgressContext in the same change that adds the `lesson_progress.checkpoints` column, and deploy the backend before shipping the SDK kwarg (client.py:1627) and the MCP tool arg. Neither the frontend nor the SDK should send the field until `GET /lessons/.../` payloads show a checkpoint directive resolving.

---

## [minor/platform] REFINEMENT of finding 129 — mlarena-sdk/mlarena/client.py

Finding 129 names four header-less methods; the real set is six, and one apparent member is a false positive that would waste a fix.

EVIDENCE:
```
client.py:62-70 `_request` sets only allow_redirects and timeout — auth is passed per-call as `headers=self._headers()`. Scanning every method for a `self._request(...)` with no headers kwarg: competitions (:136), competition (:179), leaderboard (:1324), global_ranking, user_global_rank, list_tags (plus its private helper _resolve_tag_names). download_dataset and tail_logs also match the scan but are correct by design — download_dataset's header-less call is the pre-signed object URL, documented in its own docstring, and it reaches it via `self.datasets()`, which does send headers.
```

FIX: Add `headers=self._headers()` to the six real cases. Leave download_dataset and tail_logs alone.

---

## [minor/platform] MECHANISM behind finding 157 — frontend/src/pages/CourseLearner/LessonReader.tsx:30-50, backend/app/views/academic_courses/consumption.py:259-266

There is a second enrolment prompt, inside the lesson reader, that carries the join code automatically — and it can never fire on these two courses, because it is triggered only by a gated lesson and neither course gates anything.

EVIDENCE:
```
consumption.py:261-266 returns 403 with `{"error": "This lesson requires enrollment", "gated": True, "join_code": course.join_code}` — the payload hands the code to the client. LessonReader.tsx:30-50 renders 'Enroll to read this lesson' with the code pre-filled from that response. DB: `count(*) FILTER (WHERE l.gated)` = 0 for both courses (73 published / 0 drafts / 0 gated on course 15; 29 / 0 / 0 on course 14).
```

FIX: No change needed for enrolment (the landing-page form already covers it). Worth knowing as a lever: gating one lesson per module would surface a code-prefilled enrol prompt at the point of need, but that trades away public browsability.

---

