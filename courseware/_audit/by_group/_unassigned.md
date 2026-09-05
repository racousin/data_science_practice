## [major/platform/platform] backend/app/views/leaderboard.py:75-210
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

## [minor/competition/courseware] courseware/competitions/s2-readability/overview.md:51
In the worked syllable table — whose entire purpose is teaching that a run must be *contiguous* — the `reevaluation` row lists a run that is not contiguous. `t` separates `ua` from `io`. The row directly under it lectures the student on contiguity, so the error lands exactly where a careful reader is checking their understanding.

EVIDENCE:
```
overview.md:51 `| \`reevaluation\` | \`ee\`, \`a\`, \`uaio\` … → 4 | no | **4** |`. Actual runs under the pinned rule: `re.findall(r"[aeiouy]+", "reevaluation")` → `['ee', 'a', 'ua', 'io']`. The reference implementation agrees the total is 4 (`_count_syllables("reevaluation") == 4`), so only the labelling is wrong.
```
FIX: Replace the middle cell of overview.md:51 with ``\`ee\`, \`a\`, \`ua\`, \`io\` → 4`` (drop the ellipsis — there are exactly four runs, and listing all four is the point).

---

## [major/baseline/courseware] competitions/s1-textstats/overview.md
The overview never states a number that counts as "you have done the lab". It explains the metric fully but names no target, so a student on 0.83 cannot tell whether that is a pass, a near-miss, or a broken tie-break. The only number that exists — `benchmark_expected_score: 1.0` — lives in config.py, which is creator-side and never shipped to students. Direction and range are inferable from "the fraction that match the reference exactly" but are never stated as such.

EVIDENCE:
```
overview.md, "Scoring, and what a low score means": "Score is the pass rate over all sixty checks; the leaderboard breaks it down per function so you can see which of the three is wrong." — that is the whole of it; grep for a target number returns only the crash illustrations `0.13`, `0.27`. config.py:  "benchmark_expected_score": 1.0,  and the reference agent.py is a straight `len(text.split())` / dict-count / first-longest implementation, i.e. the target is exactly reachable by the Lab 1 functions.
```
FIX: Add, immediately after the "Score is the pass rate over all sixty checks" sentence in overview.md: "**Target: 1.0.** The metric runs 0.0 to 1.0 and higher is better. A correct implementation of the three Lab 1 functions scores exactly 1.0 — the reference solution this competition was built against does, and the build refuses to publish if it does not. Anything below 1.0 means one of the three functions disagrees with the specification above; the per-function breakdown on the leaderboard tells you which."

---

## [major/platform/platform] backend/app/views/leaderboard.py:205 and backend/app/views/competitions.py:956
The leaderboard and datasets routes rewrite an intentional 404 into an opaque 500. `get_visible_competition_or_404()` raises a Werkzeug `NotFound`, which the broad `except Exception` swallows and replaces with a generic 500 body. A student given a competition id they cannot see — or who mistypes one — gets a server error instead of "not found" and cannot tell a permissions problem from an outage. This is the silent-except the repo's own fail-fast rule forbids, and the correct pattern already exists 500 lines away in the same file.

EVIDENCE:
```
With the student token against prod: `GET /api/leaderboard/competition/181` → `500 {"error":"Failed to fetch leaderboard"}`; `GET /api/leaderboard/competition/999999` (an id that cannot exist) → the identical `500`; `GET /api/competitions/181` → correctly `404`; `GET /api/competitions/181/datasets` → `500 {"error":"Failed to retrieve datasets"}`. Source: leaderboard.py:77 calls `get_visible_competition_or_404(competition_id)` inside a `try:` whose only handler is `except Exception as e:` at 204-210; competitions.py:923 likewise, handler at 955-963. competitions.py:457-460 already does it right: `except HTTPException:` / `# get_visible_competition_or_404() raises 404 (missing/hidden); let it … propagate instead of being rewritten to a generic 500 by the broad handler below.` / `raise`.
```
FIX: Insert the same three lines above the broad handler in both places. In backend/app/views/leaderboard.py immediately before `except Exception as e:` (line 204) add `    except HTTPException:` / `        # get_visible_competition_or_404() raises 404 (missing/hidden); let it propagate.` / `        raise`, and the identical block in backend/app/views/competitions.py before `except Exception as e:` (line 955); competitions.py already imports `HTTPException`, leaderboard.py needs `from werkzeug.exceptions import HTTPException`. A student probing a hidden or nonexistent competition then gets 404, matching `/api/competitions/<id>`.

---

## [minor/platform/platform] mlarena-sdk/mlarena/client.py:240 download_dataset()
download_dataset() has no way to select individual files, so touching Session 1's own competition dataset at all means pulling all seven weather CSVs — 4.45 GB — even to look at the schema of one year. The lesson that teaches "the single biggest performance lever in data collection is not reading what you do not need" is attached to a competition whose data can only be fetched whole.

EVIDENCE:
```
Signature: `def download_dataset(self, competition_id: int, dest_dir: str = ".") -> list[str]` — "Download every published dataset file for a competition into `dest_dir`", loops `for ds in payload.get("datasets", [])` / `for f in ds.get("files", [])` with no filter. datasets(177) file sizes: 722112434 + 720558948 + 721757245 + 721423092 + 723244638 + 720659712 + 122421412 bytes = 4.45 GB.
```
FIX: Add an optional filter: `def download_dataset(self, competition_id, dest_dir=".", files: list[str] | None = None)` and skip any file whose `label` is not in `files` when the argument is given. Keep the current behaviour when it is omitted so the starter notebooks are unaffected.

---

## [major/platform/platform] backend/app/views/leaderboard.py:75 and :204
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

## [major/platform/platform] mlarena-sdk/mlarena/client.py — no overview reader; mlarena-mcp/mlarena_mcp/server.py:242 list_attached_competitions
The competition overview — the only place any baseline is published — is unreachable from the SDK and from MCP. A student consuming the course through an AI editor can list competitions and submit to them but can never read what the target is. This violates the frontend/SDK parity rule in CLAUDE.md.

EVIDENCE:
```
`grep -n markdown mlarena/client.py` returns only `set_competition_markdown` (:592, a creator-scope PUT). There is no getter for GET /api/competition_asset/{id}/markdown/overview, which exists and is student-readable (I fetched all 21 overviews with raw requests + the student bearer token). In MCP, `list_attached_competitions` (server.py:242-264) returns only `competition_id`, `name`, `label`, `module_slug`, `module_title`; the `whats_next` prompt (server.py:392) tells the agent to "call list_attached_competitions() and suggest a competition to work on" with no way to read the task.
```
FIX: Add `def competition_overview(self, competition_id: int) -> str` to client.py mirroring GET /api/competition_asset/{id}/markdown/overview with `headers=self._headers()`, and expose it as an MCP tool `get_competition_overview(competition_id)` next to `leaderboard` in server.py. Both are pure additions over an existing route, so no new endpoint is needed and the parity rule is satisfied.

---

## [major/structure/courseware] ms2a-machine-learning-practice — all 12 modules; python-ai-engineering — s1/s2/s3
The lesson bodies never name the attached competition, so the overview page is the sole carrier of the baseline; and the labs assign it no weight, so a student has no reason to believe a target exists at all.

EVIDENCE:
```
Grepping every published lesson body for each attached competition's name: "Global Weather", "Allergies", "Survival", "Permuted MNIST", "Blood Cell", "Clinical Note", "Prédire la source", "GSM8", "SuperTuxKart", "The Round", "AquaControl", "Bitcoin" all appear only in the module metadata (_module.json), never in a lesson. Only lab-10 sends a student to a competition ("Open `https://ml-arena.com`, or call `client.competitions()`, and pick a…"). In course 14, lab-1's grading table is 30/25/25/20 with no leaderboard row, lab-2's is 20/25/25/20/10 likewise, lab-3's is 20/15/30/20/15 likewise; only lab-4 has "| ML-Arena submission accepted | 10% |".
```
FIX: In each lab that has an attached competition, add one line naming it and quoting its bar, sourced from the same yaml the verify-external check reads: e.g. lab-7 already has "## Part B — The baseline you must beat" worth 20% — make it "the TF-IDF baseline on competition 174 scores 0.480 F1-macro; beat it and report both numbers on the same test set". Where the intent really is "getting on the board matters; your position does not" (lab-4, lab-10), say exactly that next to the number so a student knows the bar is participation, not rank.

---

## [major/platform/platform] modelmanager/modelmanager/module_competition_link.py:18; backend/app/views/academic_courses (module_overview payload)
A course cannot record its own passing bar. ModuleCompetitionLink carries only a display label, so a per-cohort target has nowhere to live and must be smuggled into a 200-character string or into competition markdown a teacher may not own.

EVIDENCE:
```
module_competition_link.py:18 `label = db.Column(db.String(200), nullable=True)  # optional display label` — and to_dict() (:31-37) returns id/module_id/competition_id/position/label only. The student-facing module payload confirms it: module_overview("ms2a-machine-learning-practice","s3-tabular-models") returns `"competitions": [{"competition_id": 172, "label": "2-Month Survival Prediction — tabular binary classification: gradient boosting judged against a baseline you can defend", "name": "..."}]` — the word "baseline" is there, the number is not. courseware course.yaml has the same shape: `competitions: - competition_id: 172 / label: "..."`.
```
FIX: Add nullable `baseline_score`, `target_score` (Numeric) and `metric_direction` (String(6), 'higher'/'lower') to ModuleCompetitionLink, expose them in to_dict(), accept them in POST /api/teacher/modules/{id}/competitions and in the SDK's attach_competition(), add the matching keys to course.yaml's competitions block, and surface them in the student module_overview payload. That gives a teacher a per-cohort bar without touching a competition they do not own, and gives SDK/MCP students the target without needing the markdown route at all.

---

## [minor/platform/platform] frontend/src (leaderboard rendering) and backend leaderboard payload
Every competition's creator-side reference score is published on the public board as a row literally named `__benchmark__`, unexplained and ranked as if it were a competitor. It is the one measured baseline the platform always has, and it is presented in the least usable way possible.

EVIDENCE:
```
`grep -rn "__benchmark__" frontend/src` returns nothing — the frontend does not special-case it; it is written by backend/app/views/creator_competition/benchmark.py:325,342,534 and lifecycle.py:584 as an ordinary agent name. Consequences on live boards: 43's `__benchmark__` ranks 661 of 708 at -266.62; 8's ranks 1038 of 1084; 165's ranks 833 of 866 at 0.0; 65's ranks 1 with Elo 1248; 172's ranks 81 of 82. A student sees a row named `__benchmark__` under the creator's username and has no way to know it is the reference.
```
FIX: Flag it in the payload (`"IsReference": true`) and render it as a pinned, visually distinct row labelled "Reference (creator benchmark)" that is excluded from the competitor ranking, the way a par line is drawn rather than played. Then the contract's Reference number is verifiable by every student on every competition without the creator writing anything.

---

## [minor/platform/platform] backend/app/views/academic_courses (module_overview) — courses 14 and 15
The module payload advertises competitions the caller cannot open and gives no reachability signal, so the student's first click is a 404 with no explanation.

EVIDENCE:
```
As the student token: module_overview("python-ai-engineering","s1-git-and-packaging") returns `"competitions": [{"competition_id": 179, "label": "textstats correctness", "name": "PAIE S1 — textstats"}]` and `"is_enrolled": false`, while GET /api/competitions/179 with the same token returns 404. Both courses also have `join_code: null` and `enrollment_link: null`, so the student cannot self-enroll to gain the course side-channel that competitions.py:213-217 provides.
```
FIX: Include `is_visible_to_me` (or `is_public` plus the enrollment-derived grant) on each competition entry in the module payload, and have the CourseLearner module page render an unreachable competition as "not open yet" rather than a live link. Independently, issue a join_code for courses 14 and 15 so enrollment — and therefore the non-public course-competition side-channel and all progress tracking — actually works.

---

## [blocker/platform/platform] backend/app/views/academic_courses/legacy.py:29-90
GET /api/academic_courses/ has no auth decorator and serializes courses with the full AcademicCourse.to_dict(), which includes join_code and enrollment_link. Anyone on the internet can harvest the enrolment secret of every course on the platform, including private ones, and self-enrol.

EVIDENCE:
```
Anonymous request (no Authorization header): `curl https://ml-arena.com/api/academic_courses/?show_all=true` -> 200, body begins `[{"code":null,...,"enrollment_link":"d1b1d31cc91042229f15dcdce119953d","id":12,...`. With the student token the same route returned join codes for all 15 courses: `13 private 'Knowledge Graph & Embeddings' join=A9QEVXAM`, `9 private 'TU Wien - Machine Learning 2026S - GSM8k' join=JJERDJ35`, `2 private 'Sorbonne - L2 Mathematiques 2026' join=84QMYEFP`. Route decorator at legacy.py:29 is only `@bp.route("/", methods=["GET"])`; serialization is `data = course.to_dict()` at legacy.py:85.
```
FIX: Add `@login_required` to get_courses and replace `data = course.to_dict()` (legacy.py:85) with a learner-safe serializer that emits id/name/code/slug/description/visibility/instructor_name/start_date/end_date/is_enrolled only. Add join_code and enrollment_link back per row only when `can_manage_course(course)` is true (import from ._helpers). Also filter the `show_all` branch (legacy.py:61) by `can_view_course` so private courses a caller has no relationship to are not listed at all.

---

## [blocker/platform/platform] backend/app/views/academic_courses/consumption.py:194-210 and legacy.py:149-155; frontend/src/pages/CourseLearner/CourseLanding.tsx:201
A course with visibility=public is fully browsable but cannot be joined: the only enrolment route requires a secret enrollment_link or join_code, and the learner-facing course_landing payload never carries either. This is not a missing default (create_course at legacy.py:133-134 always mints both, and courses 14/15 do have GR1WFC63 / N1DX2QA4) and not a missing authoring field (ShareTab.tsx:51 shows it to the teacher) — it is a missing self-enrol route plus a missing landing field. Consequence: is_enrolled is permanently false, so LessonProgress rows are never created and every progress surface is dead.

EVIDENCE:
```
`c.course("python-ai-engineering")` key set is ['can_manage','code','competition_ids','cover_url','description','end_date','id','instructor_name','is_enrolled','modules','name','progress','slug','start_date','visibility'] — no join_code, no enrollment_link. `GET /api/academic_courses/14/progress/me` -> 403 {"error":"Not enrolled in this course"}. `find_course_by_link_or_code` (_helpers.py:23-30) matches only enrollment_link or join_code, never slug. CourseLanding.tsx:201 renders `<JoinCodeForm label="Join code" />` with an empty field on a course whose code is nowhere on the page.
```
FIX: Backend: add `POST /api/academic_courses/<string:slug>/enroll` in legacy.py that resolves by slug and enrols when `course.visibility in (Visibility.PUBLIC.value, Visibility.UNLISTED.value)`, reusing the body of enroll_in_course (identity checks, end_date check, 409 on duplicate); keep the code path for private courses. Frontend: in CourseLanding.tsx:196-211 render a primary `Join this course` button calling that route when `!course.is_enrolled && course.visibility !== 'private'`, and keep JoinCodeForm only for private. SDK: `enroll_in_course(..., slug=None)` in mlarena-sdk/mlarena/client.py:1497. MCP: accept `slug` in `join_course` (mlarena-mcp/mlarena_mcp/server.py:110-128).

---

## [blocker/platform/platform] backend/app/views/competition_asset.py:19-46
get_markdown_overview wraps get_visible_competition_or_404 in a bare `except Exception`, which catches Werkzeug's NotFound and converts a deliberate 404 into a 500. The visibility guard is defeated and the student gets a server error for a competition that is simply not published. It also violates the Fail Fast rule in CLAUDE.md (no silent try/except).

EVIDENCE:
```
`GET /api/competition_asset/179/markdown/overview` with the student token -> 500 {"error":"Failed to read the competition overview markdown"}, while `GET /api/competitions/179` -> 404. Source: line 22 `get_visible_competition_or_404(competition_id)` is inside the `try:` opened at line 21, and line 41 is `except Exception as e:` returning 500.
```
FIX: Move the `get_visible_competition_or_404(competition_id)` call above the `try:` block (competition_asset.py:21-22), or narrow the handler to `except (OSError, IOError)` and add `except HTTPException: raise` before it. Apply the same treatment to update_markdown_overview (line 48) and delete_markdown_overview (line 81), which share the pattern.

---

## [major/platform/platform] modelmanager/modelmanager/lesson.py:74-75 -> backend/app/views/academic_courses/consumption.py:272-281
Lesson.to_dict returns body_md verbatim and consumption.lesson_body ships it unchanged, so the teacher's HTML-comment speaker notes reach every SDK and MCP consumer raw. The web renderer only appears to hide them: CourseMarkdown uses rehype-raw, so the comments are emitted into the DOM and are visible in view-source.

EVIDENCE:
```
109 `<!-- notes: ... -->` blocks across 92 of the ~102 published lesson bodies in the student_view dump. Via SDK: `c.lesson("ms2a-machine-learning-practice","s7-nlp-1","lab-7")["body_md"]` contains `<!-- notes: They will want to start with the transformer. Do not let them — the baseline is 10 lines and it is the number the whole lab is judged against. Circulate during Part C: ... -->`. rehype-raw is wired at frontend/src/components/Course/CourseMarkdown.tsx:153.
```
FIX: Strip at READ, once, on the consumption route — not at publish (the teacher must keep the notes) and not per-consumer (four implementations, and it would break the parity rule). Add `strip_author_comments(body_md: str) -> str` to backend/app/services/lesson_directives.py (regex `<!--(?!\[if).*?-->` with DOTALL, applied after directive extraction) and call it in consumption.py:273 so `data = lesson.to_dict()` is followed by `data["body_md"] = strip_author_comments(lesson.body_md)`. Leave the raw body on the teacher routes (backend/app/views/teacher/lessons.py GET /lessons/<id> and preview_lesson). Apply the same helper to backend/app/views/competition_asset.py:36 for non-manager callers, which currently ships a `<!-- mlarena:quickstart -->` sentinel to students in 12 of the 17 MS2A overviews.

---

## [major/platform/platform] frontend/src/pages/CourseLearner/ModuleOverview.tsx:92-141, CourseLanding.tsx:128-154, LessonReader.tsx:99; type at frontend/src/services/coursesApi.ts:769-775
Competition performance is computed and shipped in my_progress but rendered by no learner page. All three CourseLearner pages call useMyProgress and use only completedSet and next_lesson; ProgressCompetitionCell exists only in the type file. So the answer to 'is competition performance folded into progress?' is: on the backend yes, in the student UI no — the frontend is the consumer that is behind.

EVIDENCE:
```
`grep -rn ProgressCompetitionCell frontend/src` returns only coursesApi.ts:169, :187, :774. `GET /api/academic_courses/11/progress/me` returns a populated `competitions` array ([{competition_id:172, name:'2-Month Survival Prediction', ranked_by:'accuracy', value:null, n_runs:0, best_agent_name:null, last_run:null}, ...]) computed by _competition_results at backend/app/views/teacher/course_content.py:314-380. CourseLanding.tsx:143 uses only `progress?.next_lesson`; ModuleOverview.tsx:95 uses only `completedSet`.
```
FIX: In ModuleOverview.tsx, extend the CompetitionItem card (line 65-90) to take the matching ProgressCompetitionCell from useMyProgress and render `n_runs === 0 ? 'Not submitted' : `${ranked_by} ${value}` (n_runs runs, last <last_run>)`. In CourseLanding.tsx add a per-module competition status chip next to the lesson/minutes badges (line 91-119). Mirror the same rollup as an SDK convenience on my_progress and as an MCP `my_progress` field so all four consumers show it.

---

## [major/platform/platform] backend/app/views/academic_courses/consumption.py:109-136 and :431-454; frontend/src/pages/CourseLearner/CourseLanding.tsx:184-194
'Progress' means lessons-marked-complete and nothing else. A student who submitted to all 17 competitions and clicked no checkbox reads 0%. The bar is labelled 'Your progress' with no qualifier, which actively misinforms.

EVIDENCE:
```
_own_progress_summary (consumption.py:116-135) counts only LessonProgress rows with status COMPLETED over published lessons. CourseLanding.tsx:187 renders `<Text size="sm" c="dimmed">Your progress</Text>` over `{course.progress.completed}/{course.progress.total}`. Live on course 11: content.pct 0.0 while the course's three competitions are listed separately and ignored by the bar.
```
FIX: Relabel the bar 'Lessons read' in CourseLanding.tsx:187. Add a sibling `competitions` line: `<n> of <m> competitions submitted`, derived from `progress.competitions.filter(c => c.n_runs > 0).length`. Longer term, add a `completion` block to my_progress (consumption.py:448-453) that reports, per module, `{lessons_completed, lessons_total, competitions_submitted, competitions_total, required_met}` so a single field answers 'am I done with this session'.

---

## [major/platform/platform] modelmanager/modelmanager/module_competition_link.py:12-24
There is no notion of a competition being REQUIRED for a module, and no target score on the link, so 'done with session 3' is undefinable by construction. The link carries only module_id, competition_id, position and a display label.

EVIDENCE:
```
module_competition_link.py:14-18 is the complete column list: module_id, competition_id, position, label. Neither _serialize_module_competition (consumption.py:82-87) nor _competition_results (teacher/course_content.py:314-380) has anything to compare a student's `value` against.
```
FIX: Add two nullable columns to ModuleCompetitionLink with an Alembic revision: `is_required = db.Column(db.Boolean, nullable=False, server_default=text('false'))` and `target_score = db.Column(db.Float, nullable=True)`. Accept both in AttachCompetitionRequest (backend/app/views/teacher/_schemas.py) and in attach_competition (backend/app/views/teacher/modules.py:301-306); emit both from _serialize_module_competition; carry `target` alongside `value` in _competition_results entries (teacher/course_content.py:350-359); expose fields in AddCompetitionModal.tsx (a 'Required for this session' switch and a 'Target score' number input) and in the SDK's attach_competition (mlarena-sdk/mlarena/client.py:1734).

---

## [major/platform/platform] backend/app/views/creator_competition/_helpers.py:188-205 (user_can_see_competition) and :216-233 (visibility_filter)
Course enrolment grants no competition visibility. A competition is either world-public or owner-only; there is no 'visible to students of the courses that attach it'. That is why the courseware README documents 'everyone else gets a 404, including enrolled students' as expected behaviour — the platform gives a teacher no way to run a course-private competition, so the only option is to publish it to the whole internet.

EVIDENCE:
```
_helpers.py:194-205 is the complete rule: is_public, else anonymous->False, admin->True, owner->True, CompetitionCreatorAssistant->True. No reference to ModuleCompetitionLink / CourseModuleLink / UserCourseEnrollment anywhere in the file. Live: competitions 179-182 are 404 for the student token; tmp/data_science_practice/courseware/competitions/README.md states the same.
```
FIX: Add a fourth clause to user_can_see_competition (_helpers.py:205): the competition is reachable through ModuleCompetitionLink -> CourseModuleLink to a course where `is_enrolled(course.id) or can_manage_course(course)` (import the helpers from app.views.academic_courses._helpers). Add the matching subquery clause to visibility_filter (_helpers.py:229-233) so list endpoints agree. This is the single change that makes the 'attach to a module' gesture actually mean something for a private cohort.

---

## [major/platform/platform] backend/app/views/competition_asset.py:19; mlarena-sdk/mlarena/client.py:592 (write-only); mlarena-mcp/mlarena_mcp/server.py:242-264
Parity break: the competition overview markdown — the document that carries the task description, the data contract and the baseline — is readable over REST but has no SDK method and no MCP tool. The SDK has set_competition_markdown (write) with no getter. An MCP-first student can list a course's attached competitions and get id/name/label and nothing else, so the AI editor the platform advertises cannot see the assignment.

EVIDENCE:
```
`grep -n 'markdown/overview\|def get_competition_markdown' mlarena-sdk/mlarena/client.py mlarena-mcp/mlarena_mcp/server.py` -> no matches. mlarena-sdk/mlarena/client.py:592 defines set_competition_markdown only. mlarena-mcp/mlarena_mcp/server.py:250-263 builds each entry from competition_id/name/label/module_slug/module_title. Manual REST call to /api/competition_asset/172/markdown/overview with the student token returns the full 4218-byte brief.
```
FIX: Add `def competition_overview(self, competition_id: int) -> str` to mlarena-sdk/mlarena/client.py (GET /api/competition_asset/<id>/markdown/overview, return resp.json()['content']) next to competition() at line 179, document it in mlarena-sdk/README.md. Add an MCP tool `get_competition_overview(competition_id)` and an MCP resource `competition://{id}/overview` in mlarena-mcp/mlarena_mcp/server.py alongside lesson_resource (line 348-366), and register it in read_tools.

---

## [major/baseline/platform] modelmanager/modelmanager/competitions.py CompetitionConfiguration (columns at lines 270-318); backend/app/views/creator_competition/benchmark.py:267-290
The platform has no field for 'the score to beat'. run_benchmark executes the reference solution but stores nothing, so the number a student needs is only ever prose inside overview.md — unreachable from the module card, the progress payload, or the leaderboard. `grep -rn benchmark modelmanager/modelmanager/*.py` returns zero hits.

EVIDENCE:
```
`grep -rn 'benchmark' modelmanager/modelmanager/competitions.py modelmanager/modelmanager/evaluation*.py` -> no output. The courseware carries benchmark_expected_score in its own config.py (e.g. tmp/data_science_practice/courseware/competitions/s3-adult-income/config.py: `"benchmark_expected_score": 0.656175`) precisely because the platform has nowhere to put it. my_progress competition cells (teacher/course_content.py:350-359) carry `value` with no comparator.
```
FIX: Add `benchmark_score = db.Column(db.Float, nullable=True)` and `benchmark_score_label = db.Column(db.String(120), nullable=True)` to CompetitionConfiguration (modelmanager/modelmanager/competitions.py, near submission_filename at line 311) with an Alembic revision; write benchmark_score from the benchmark run outcome in backend/app/views/creator_competition/benchmark.py:267-290; expose it in get_competition (backend/app/views/competitions.py:371-397) and in _competition_results entries as `target`; render it on the leaderboard as a reference line and next to the student's value in the ModuleOverview competition card. Add `benchmark_score=` to SDK update_settings (client.py:410).

---

## [minor/platform/platform] frontend/src/pages/CourseLearner/CourseLanding.tsx:100-118 (badge text at line 115)
The module-index badge prefers the competition's `label` over its `name`, and the MS2A labels are full sentences: 18 of 22 exceed 40 characters and the longest is 135. Inside a Mantine Badge these collapse to an unreadable ellipsised stub or blow out the row. ModuleOverview.tsx:76-77 already does it correctly (name as the title, label as dimmed subtext).

EVIDENCE:
```
Measured across all _module.json files: max label length 135 (comp 177, 'Global Weather Forecast — a source that never stops: observations arrive daily and the forecast is scored once the weather has happened'), 18/22 over 40 chars. CourseLanding.tsx:115: `{c.label || c.name || `#${c.competition_id}`}` inside `<Badge size="sm" ...>`.
```
FIX: CourseLanding.tsx:115 -> `{c.name || c.label || `#${c.competition_id}`}`, and change the Tooltip label at line 101 from the static 'Open competition' to `{c.label ?? 'Open competition'}` so the long sentence is still reachable on hover.

---

## [minor/platform/platform] frontend/src/pages/CourseLearner/CourseLanding.tsx:218 and :220
The course landing's module index is titled 'Competitions' and its empty state reads 'This course has no competitions yet.' On a 12-module, 73-lesson course the heading over the session list says Competitions, which is the single most confusing label on the student path.

EVIDENCE:
```
CourseLanding.tsx:217-225: `<Title order={3}>Competitions</Title>` ... `{orderedModules.length === 0 ? <Text c="dimmed">This course has no competitions yet.</Text> : orderedModules.map(...)}` — the mapped items are modules (ModuleRow), not competitions.
```
FIX: Change line 218 to `<Title order={3}>Sessions</Title>` (or 'Course content') and line 220 to 'This course has no sessions yet.' Keep 'Competitions' only for the SimpleCompetitionRow branch (line 34-62), which really is a bare competition entry.

---

## [minor/platform/platform] backend/app/views/competitions.py:274-320 (_resolve_user_course_for_competition)
The 'Back to <course>' breadcrumb on a competition page resolves only for enrolled students and managers, so a student browsing a public course who opens one of its competitions loses the thread back to the course entirely. Combined with the missing self-enrol path this means nobody currently gets the breadcrumb on courses 14/15.

EVIDENCE:
```
competitions.py:313-315: `managed = next((c for c in courses if can_manage_course(c)), None); course = managed or next((c for c in courses if is_enrolled(c.id)), None); if course is None: return None`. The consumer is frontend/src/components/CompetitionHeader.js:56-66.
```
FIX: Add a third fallback in competitions.py:313-315: when neither managed nor enrolled matches, fall back to the first course with `can_view_course(course)` true (public/unlisted), returning it with `can_manage: False`. The header link is read-only navigation and leaks nothing a public catalog does not.

---

## [minor/platform/platform] backend/app/views/academic_courses/consumption.py:406-428 (_next_lesson) and :431-440 (my_progress)
'Where do I start?' is only answerable to an enrolled student: next_lesson lives behind my_progress, which 403s otherwise, so a public-course browser gets no entry point and CourseLanding renders no Start button. _next_lesson also ignores lesson.gated and parent_lesson_id nesting, so it can point at a lesson the caller cannot open.

EVIDENCE:
```
consumption.py:439-440 `if not (is_enrolled(course_id) or can_manage_course(course)): return 403`. CourseLanding.tsx:146 `const continueButton = course.is_enrolled && next ? (...) : null;` — nothing renders for a non-enrolled visitor. _next_lesson's filter at consumption.py:420 is `if lesson.is_published and lesson.id not in completed` with no gated check.
```
FIX: Add `first_lesson` to the course_landing payload (consumption.py:194-210) — the lowest (module.position, lesson.position) published, non-gated lesson — and render it in CourseLanding.tsx:146 as a 'Start reading' button when not enrolled. Add `and not lesson.gated` to the _next_lesson filter at consumption.py:420 unless the caller is enrolled.

---

## [minor/content/courseware] courseware/content/ms2a-machine-learning-practice/course.yaml (estimated_minutes across all session modules)
The course description promises 'ten 3-hour sessions', but summed lesson estimated_minutes run 3.3-4.0 h per session, and none of the budgets include the competition submission the session is graded on. The '~N min' badge CourseLanding renders (CourseLanding.tsx:27-29, 95-99) therefore tells a student the reading alone overruns the class.

EVIDENCE:
```
Per-module sums from _index.tsv: s3-tabular-models 240 min (4.0 h), s2-data-preprocessing 230, s4-advanced-neural-networks 230, s8-nlp-2 225, s7-nlp-1 220, s9 220, s10 215. python-ai-engineering s1-git-and-packaging 230 min against a stated 3-hour session.
```
FIX: Rebalance estimated_minutes in course.yaml so each session module sums to <=180 min, and add an explicit `estimated_minutes` allowance for the attached competition (or state 'plus ~45 min submission, out of class' in each session module's summary).

---

## [minor/platform/platform] mlarena-sdk/mlarena/client.py (project Milestone 1)
The project requires a first-class platform team by Milestone 1, and the module drives every other action through the SDK, but the SDK has no team methods at all. The routes exist and accept the student bearer token, so this is a pure parity gap against the repo's own frontend/SDK parity rule.

EVIDENCE:
```
`grep -n "team" mlarena-sdk/mlarena/client.py` → no matches. backend/app/views/teams.py exposes 12 routes (create, invite, respond, leave, delete, search, ...). Bearer auth works on them: `GET /api/teams/competition/171/team` with the student token → 404 `{"message":"No team found"}`, `GET /api/teams/invitations/received` → 200 `[]`. project-brief.md: "A team is a **first-class object on the platform**, not a line in an email. You create it on the competition page and invite your partner".
```
FIX: Add `team(competition_id)`, `create_team(competition_id, name)`, `invite_to_team(team_id, username)`, `received_invitations()`, `respond_to_invitation(invitation_id, accept)` to mlarena-sdk/mlarena/client.py as thin wrappers over the existing `/api/teams/*` routes, and add the corresponding lines to project-brief's Milestone 1 so declaration is one script.

---

## [blocker/structure/both] courses 14 and 15 (python-ai-engineering, ms2a-machine-learning-practice)
Neither course has a join code or an enrollment link, so no student can enrol; without enrolment every progress surface is dead and nothing a student does is recorded.

EVIDENCE:
```
`c.course('python-ai-engineering')` -> join_code=None, enrollment_link=None, is_enrolled=False, visibility='public'; same for ms2a-machine-learning-practice. `c.my_progress(14)` and `c.my_progress(15)` -> AuthenticationError: Not enrolled in this course. `c.mark_lesson_complete(115)` -> AuthenticationError: Not enrolled in any course containing this lesson.
```
FIX: Generate a join_code for both courses through the teacher surface (mlk_teacher_ key, /api/teacher/*) and add the resulting code/link to content/<course>/course.yaml so `make publish` carries it. Until then, publish_mlarena.py should fail the build when a published course has neither join_code nor enrollment_link, rather than emitting a course nobody can enter.

---

## [blocker/structure/courseware] ms2a-machine-learning-practice/s1..s7,s9/lab-*.md
Nine of the ten MLP labs never tell the student the attached competition exists. The lab ends at a PR in the student's own repo, so the competition is decorative and no student will ever discover the ramp, however good it is.

EVIDENCE:
```
`for f in */*/lab-*.md; do grep -ci 'ml-arena|leaderboard|competition id|submit.*competition' $f; done` -> 0 for s1,s2,s3,s4,s5,s6,s7 labs; 1 for s8 and s9; only s10/lab-10.md scores 13. Labs 1-9 all close with "**Deliverable:** a merged PR in your project repository."
```
FIX: Add a closing 'Part F — Put it on the board' section (5 minutes) to labs 1 through 9, naming the session's ramp competition id, the measured trivial floor, and the reference score to beat, e.g. for lab-7: 'Submit your TF-IDF baseline to the Spooky Author competition. Majority class scores macro-F1 0.1917; the pipeline in Part B scores 0.8227.'

---

## [blocker/platform/platform] backend/app/views/competition_asset.py:22-46
A blanket `except Exception` swallows the 404 abort raised by get_visible_competition_or_404, so requesting the overview of a competition you cannot see returns HTTP 500 with the message 'Failed to read the competition overview markdown'. The student sees a server error instead of 'not found', and the operator sees a false storage-failure log line.

EVIDENCE:
```
`curl -H 'Authorization: Bearer mlk_user_...' https://ml-arena.com/api/competition_asset/179/markdown/overview` -> 500 {"error": "Failed to read the competition overview markdown"}; the same URL with the creator key -> 200. Public comp 173 -> 200 for both. Source: line 23 calls get_visible_competition_or_404(competition_id) inside the try; line 39 `except Exception as e` catches the werkzeug NotFound and line 46 returns 500. This is the 'No silent try/except' rule in CLAUDE.md.
```
FIX: Move `get_visible_competition_or_404(competition_id)` above the `try:` block, or add `except HTTPException: raise` before the generic handler, so a hidden or missing competition returns its real 404 and only genuine storage errors return 500.

---

## [minor/validation/courseware] courseware/README.md 'Known gaps'
The README still says the MS2A course's 132 lesson images are not uploaded and blames an unset PATH_COURSES. That is fixed, so a reader is sent to chase a resolved server-side bug instead of the four real gaps in the same section.

EVIDENCE:
```
courseware/README.md: "**`ms2a-machine-learning-practice` is published text-only.** All 12 modules and 73 lessons are live on course #15, but its 132 images are **not** uploaded. The cause is server-side...". The courses NFS export landed in commit cf29fefb ('course media: mount the courses NFS export into the backend') and the images now resolve 200.
```
FIX: Delete that bullet from courseware/README.md 'Known gaps' and replace it with the still-open items: the missing join_code on both courses, the 179-182 is_public flip, and the labs that do not reference their competitions.

---

## [blocker/validation/both] all 102 lessons in python-ai-engineering + ms2a-machine-learning-practice
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

## [blocker/validation/both] backend/app/services/lesson_directives.py:202 + modelmanager/modelmanager/lesson_progress.py:31 + backend/app/views/academic_courses/consumption.py:369
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

## [blocker/platform/platform] modelmanager/modelmanager/lesson_progress.py:31-34 + backend/app/views/academic_courses/consumption.py:369-391
The platform can only record 'I opened this' and 'I clicked the button'. There is no representation anywhere of 'I demonstrably got this right', so no amount of courseware writing can make progress mean correctness without a platform change.

EVIDENCE:
```
LessonProgress carries exactly four state fields: `status` (not_started/in_progress/completed), `completed_at`, `last_viewed_at`, and the (user, lesson, course) key. `mark_lesson_complete` (consumption.py:369) sets `progress.status = COMPLETED` from a request body whose only field is `course_id` — no evidence of any kind is required or accepted. The LessonReader UI is honest about it: LessonReader.tsx:147 renders a plain `Mark complete` button. `_content_progress` (teacher/course_content.py:303-308) then computes pct as completed/total over that self-declaration, and that same number is what both the student landing page and the teacher dashboard show.
```
FIX: Add the `checkpoints` JSON column and the `checkpoints` request field described in the design finding, and report `checkpoints_passed / checkpoints_total` alongside `completed/total` in both `my_progress` and `_content_progress`. Keep `Mark complete` for lessons with no checkpoints — the point is not to gate, it is to distinguish the two states in the payload so the UI and the SDK can show 'read' separately from 'verified'.

---

## [blocker/platform/platform] course 14 + course 15 (no join_code, no enrollment_link) / backend/app/views/academic_courses/consumption.py:289-320
Every progress write requires enrollment, and neither course is joinable, so the entire progress layer — ticks, percentage, next-lesson pointer, and any future checkpoint state — is unreachable for a real student today. The validation layer would ship into a surface no one can reach.

EVIDENCE:
```
Live: `c.my_progress(14)` and `c.my_progress(15)` both raise `AuthenticationError: Not enrolled in this course`. `_resolve_progress_course` (consumption.py:311-315) returns 403 "Not enrolled in any course containing this lesson" when the caller has no enrollment intersecting the lesson's modules, and both `/view` and `/complete` route through it. `_course.json` for course 14 shows `"is_enrolled": false` with no join code in the payload.
```
FIX: Generate a join code for both courses (the generator already exists: `gen_join_code` in backend/app/services/course_content.py) and put the enrol URL in the course description and in the first lesson of each course. Add the assertion to student_walk.py's check: it already flags `COURSE not-enrolled` (student_walk.py cmd_check) but that line is advisory — make a missing join_code a non-zero exit.

---

## [major/validation/courseware] all 14 labs (kind: exercise)
7 of 14 labs state no acceptance criterion a student can settle alone; the other 7 have exactly one, and it is always 'tests pass' or 'submission accepted'. Zero of 14 state a score threshold. Every lab closes with a weighted rubric addressed to the grader, not to the student.

EVIDENCE:
```
All 14 have a `## Grading` weight table; 11 also have `## Automatic deductions`. Criteria the student can settle: `uv sync && uv run pytest` green on a fresh clone (PAIE lab-1 30%, PAIE lab-4 20%); 'N tests passing' where the tests are actually specified by a `## Required tests` docstring block (MLP lab-3, 7, 8, 9, 10, and PAIE lab-4 Part B) = 6 labs; 'ML-Arena submission accepted' (PAIE lab-4 10%, MLP lab-10 'Submission accepted and running on the leaderboard' 25%). Everything else is judgment: PAIE lab-2 is 100% judgment — "`CLAUDE.md` is specific to this project, not generic | 20%", "`RETRO.md` shows real pushback, not a transcript | 25%". Numeric thresholds anywhere in the 14 labs: two, and both are inside a required-test docstring rather than an acceptance criterion — PAIE lab-4 "200 steps on 32 examples drives the loss below 0.1" and MLP lab-4 "loss below 0.01". Score thresholds against an attached competition: 0/14. Labs whose rubric is circular because the student both writes and grades the tests (no `## Required tests` spec): MLP lab-1, 2, 4, 5, 6.
```
FIX: Split every `## Grading` table into a third column 'Settled by' with values `checkpoint <id>` / `target <competition id>` / `review`, and move at least one row per lab into the machine-settled half. Where the lab has a competition attached, add an `mlarena:target` block with a real number: the numbers already exist, measured, in the courseware packages — competitions/s3-adult-income/config.py:13 `benchmark_expected_score: 0.656175` and competitions/s4-mnist-warmup/config.py:13 `0.9134`. For the five circular labs, add a `## Required tests` docstring block in the shape MLP lab-9 already uses.

---

## [major/platform/platform] frontend/src/pages/CourseLearner/ (CourseLanding.tsx, ModuleOverview.tsx, LessonReader.tsx) + frontend/src/hooks/courseLearner/useMyProgress.ts
The student's own competition results are fetched by the learner UI and then never rendered. The one piece of objective evidence the platform already computes about a student dies in the client.

EVIDENCE:
```
`GET /{course_id}/progress/me` returns `competitions` (consumption.py:466), built by `_competition_results` with `best_agent_name`, `value`, `n_runs`, `last_run` per competition. `MyProgress` in services/coursesApi.ts:774 declares `competitions: ProgressCompetitionCell[]`. useMyProgress.ts returns the whole object. But `grep -rn 'progress.competitions' frontend/src/pages/CourseLearner/` -> no hits; the only `.competitions` references in CourseLearner are `module.competitions` (the static attachment list). CourseLanding.tsx:184-192 renders only the lesson-count bar.
```
FIX: Render the `competitions` array on CourseLanding and ModuleOverview as a row per competition: name, your best value, and — once `mlarena:target` exists — the target beside it with met/not-met. The data is already on the wire; this is a rendering change, no new endpoint.

---

## [major/baseline/both] backend/app/views/teacher/course_content.py:350-359 (payload consumed by academic_courses/consumption.py:466)
Even where the platform reports a student's competition score, it reports a bare number with nothing to compare it to. `value: 0.907` answers 'what did I get' and never 'did I pass', which is precisely the question the owner is asking about.

EVIDENCE:
```
The per-competition cell is {competition_id, name, best_agent_name, ranked_by, value, n_runs, last_run} — no target, no threshold, no pass flag. There is nowhere on the platform for such a number to live either: `Evaluation` (modelmanager/modelmanager/evaluations.py) has metric, metric2, is_elo_score, frontend_precision, metrics_schema and no expected/target score; `ModuleCompetitionLink` (module_competition_link.py:12-18) has only module_id, competition_id, position, label. `grep -rn benchmark_expected_score modelmanager/` -> nothing; the measured number exists only in the courseware packages' config.py.
```
FIX: Do not add a target column — that is why option (c) loses as a primary. Author the number in the lesson via `mlarena:target id=182 metric=accuracy min=0.913`, resolve it server-side against the existing per-user leaderboard query, and return {min, your_best, met} in the directive payload and in `my_progress.targets`. The threshold then lives beside the lesson that teaches it, versioned in git with the rest of the courseware, and changes without a migration.

---

## [major/platform/platform] backend/app/views/leaderboard.py:204-210
A blanket `except Exception` swallows the 404 that `get_visible_competition_or_404` raises and returns 500 instead. The student gets a server error for the exact call PAIE Lab 4 tells them to make. This is also a Fail-Fast violation per CLAUDE.md.

EVIDENCE:
```
Live with the student key: `c.leaderboard(48, top=3)` -> OK, DataFrame of 3. `c.leaderboard(182, top=3)` -> `HTTPError: 500 Server Error ... /api/leaderboard/competition/182?limit=3`. `c.leaderboard(999999, top=3)` -> the same 500, so it is not about competition 182 specifically. `get_leaderboard` (leaderboard.py:75) calls `get_visible_competition_or_404` (creator_competition/_helpers.py:208-213), which calls Werkzeug's `abort(404)`; `NotFound` is an `Exception` subclass, so the handler at leaderboard.py:204 catches it and leaderboard.py:210 returns `{"error": "Failed to fetch leaderboard"}, 500`.
```
FIX: Re-raise HTTP exceptions before the generic handler: `except HTTPException: raise` immediately above `except Exception as e:` at leaderboard.py:204 (import `from werkzeug.exceptions import HTTPException`). A hidden or missing competition must answer 404 so the SDK raises CompetitionNotFoundError and the student sees 'this competition is not open to you', not 'the server broke'.

---

## [major/structure/courseware] tmp/data_science_practice/courseware/tools/build_slides.py:76 and :95
Adding any `mlarena:` directive fence to a lesson silently destroys that lesson's slide deck — the fence regex does not match a directive info string, so the closing ``` toggles the fence state the wrong way and every subsequent `---` slide break is swallowed. This blocks the recommended design until fixed, and it is two lines.

EVIDENCE:
```
`FENCE_RE = re.compile(r"^```([\w+-]*)\s*$")` at build_slides.py:76 cannot match ```` ```mlarena:checkpoint id=1 ```` (colon, equals and a space are all outside the class, and there is trailing content). `split_slides` at :95 toggles `in_fence` only on FENCE_RE or a bare ```` ``` ````. Reproduced: a 4-slide body containing one directive fence returns 2 slides — `split_slides(body)` -> slides: 2, with slide 1 = '\n## A\n\n```mlarena:checkpoint id=1\n```\n\n-' and everything after it merged in.
```
FIX: Widen the opening-fence pattern to `^```([\w+-]*(?::[\w+-]+)?)(\s.*)?$` in both places, and add a `directive` branch in `parse_blocks` (build_slides.py:130-140) that renders a checkpoint as a titled callout box rather than a dark code panel — a checkpoint is the most useful thing to have on a slide, not something to hide.

---

## [major/structure/courseware] tmp/data_science_practice/courseware/tools/student_walk.py:11-13 (docstring) vs cmd_check
The check harness advertises the exact assertion this whole audit is about — 'a stated baseline on every competition' — and does not implement it. The one automated guard against shipping an unvalidatable course is a comment.

EVIDENCE:
```
Docstring, student_walk.py lines 11-13: "check   the same walk, but assert the things a student needs to be true: lesson bodies non-empty, images resolvable, attached competitions openable, a stated baseline on every competition." `cmd_check` asserts: not-enrolled, overview-404, lesson-error, empty-body, directive-warning, unuploaded-image, speaker-notes-in-body, competition-unreachable, competition-not-started. There is no baseline assertion, and no lab/checkpoint assertion.
```
FIX: Add three checks to cmd_check: (1) every competition whose module has a lab must be referenced by an `mlarena:target` in that module, else `no-target`; (2) every lesson of kind `exercise` must contain at least one `mlarena:checkpoint` or `mlarena:target`, else `lab-without-acceptance-criterion`; (3) every reachable competition's student-visible description must contain a digit-bearing baseline sentence, else `competition-without-baseline`. Wire `make check` to run it for both courses so a regression fails the build.

---

## [minor/content/courseware] 6 lessons in python-ai-engineering (setup.md, python-environments.md, context-engineering.md, guardrails-and-review.md, validation-and-overfitting.md, training-loop-end-to-end.md)
The only interactive-looking affordance in the corpus is 38 GFM checkboxes that render disabled and store nothing — they look like the validation feature and are not one, which is worse than having nothing.

EVIDENCE:
```
38 `- [ ]` lines across 6 files (setup.md 4, python-environments.md 5, context-engineering.md 5, guardrails-and-review.md 7, validation-and-overfitting.md 8, training-loop-end-to-end.md 9). CourseMarkdown.tsx:152 loads `remarkGfm` and the components map (CourseMarkdown.tsx:79-103) overrides only `div`, `pre` and `code` — no `input` override — so react-markdown renders GFM task items as disabled checkboxes. Nothing is persisted. They are also self-attesting rather than checkable: setup.md "- [ ] It can answer 'what does this project do?' using your actual files."
```
FIX: Convert each checklist to a single `mlarena:checkpoint kind=mcq` or `kind=run` where the item is actually verifiable (python-environments.md:50 `which python  # should print .../my_project/.venv/bin/python` is already one), and keep the rest as a plain prose `## Checklist` with bullets rather than boxes, so a box in the corpus always means 'this is tracked'.

---

## [minor/content/both] 92 of 102 lessons
Teacher speaker notes ship raw to SDK and MCP consumers. Relevant to this design because it settles a question about it: body_md is served verbatim, so a checkpoint's expected answer cannot be hidden from an SDK student — which is fine (checkpoints are diagnostic), but it must be a deliberate choice, not a discovery made later.

EVIDENCE:
```
92 of 102 lesson files contain an HTML comment; 109 comment blocks total. Example, ms2a-machine-learning-practice/s3-tabular-models/model-selection-and-validation.md:6-9: "<!-- notes: 35 minutes, the lesson that decides whether their project is worth anything. Do the leakage catalogue as a quiz: show each snippet, ask the room what is wrong, then explain. Budget 12 minutes for leakage alone — it is the part they will actually get wrong. -->". student_walk.py already flags this as `speaker-notes-in-body`.
```
FIX: Strip `<!-- notes: ... -->` in publish_mlarena.py before upload and keep them in the markdown source for build_slides.py, which is the only consumer that needs them. Separately, record in courseware/README.md that checkpoint `expect` values are intentionally public. Note the one genuinely useful thing in those notes — 'Do the leakage catalogue as a quiz: show each snippet, ask the room what is wrong' — is exactly a checkpoint that was written for the teacher and never given to the student.

---

## [minor/platform/platform] backend/app/views/academic_courses/_schemas.py:72-83
LessonProgressContext is `extra='forbid'`, so any client that starts sending checkpoint results before the server declares the field gets a 400 rather than a tolerated no-op. This is correct Fail-Fast behaviour but it fixes the rollout order: server first, then SDK/MCP/frontend.

EVIDENCE:
```
`model_config = ConfigDict(extra="forbid")` at _schemas.py:81, with `course_id: Optional[int] = None` as the only field. `mark_lesson_complete` (consumption.py:376) validates the body through it before touching progress.
```
FIX: Add `checkpoints: Optional[dict[str, bool]] = None` to LessonProgressContext in the same change that adds the `lesson_progress.checkpoints` column, and deploy the backend before shipping the SDK kwarg (client.py:1627) and the MCP tool arg. Neither the frontend nor the SDK should send the field until `GET /lessons/.../` payloads show a checkpoint directive resolving.

---

## [blocker/structure/courseware] academic_course rows 14 and 15 / courseware/content/*/course.yaml
Nobody in 194 findings mentioned the calendar. Course 14 starts in five days and the enrolment route refuses enrolment once the course has ended, so every remediation in this audit is bounded by a window nobody costed.

EVIDENCE:
```
DB: course 14 start_date=2026-09-07 end_date=2026-09-11; course 15 start_date=2026-09-14 end_date=2026-11-27. Today is 2026-09-02. backend/app/views/academic_courses/legacy.py:157-159 — `if course.end_date and course.end_date < today: return ... 410` on both GET and POST /enroll/<link>. courseware/README.md still calls these dates "placeholders". Enrollment count query: courses 14 and 15 do not appear in `SELECT course_id, count(*) FROM user_course_enrollment GROUP BY course_id` — zero students enrolled in either.
```
FIX: Decide now whether 2026-09-07 is the real course-14 start. If yes, the only work that can land before it is the four items in the narrative's first paragraph; everything else is post-launch. If the dates are still placeholders, set the real ones in course.yaml and republish, because the 410 makes end_date a hard enrolment cutoff, not a display field.

---

## [blocker/platform/platform] ADJUDICATION of findings 141, 157, 183 and the audit's stated ground truth
The claim that neither course is joinable, and that the whole progress layer is therefore unreachable, is false. Enrolment is complete and working end to end today. Acting on the wrong belief would spend the five-day window building a self-enrol route that already ships.

EVIDENCE:
```
DB: course 14 join_code=GR1WFC63 enrollment_link=0ed552047ca44d4ab58add1d15a98573; course 15 join_code=N1DX2QA4 enrollment_link=78eab0f30fdd491aa7685bc0c842001a — neither is null. Live: `curl https://ml-arena.com/api/academic_courses/enroll/GR1WFC63` returns 200 with course_id 14 and competition_ids [179,180,181,182]. frontend/src/App.js:146 defines `/enroll/:enrollmentLink`. frontend/src/pages/CourseLearner/CourseLanding.tsx:201 renders `{!course.is_enrolled && <JoinCodeForm label="Join code" />}`, and components/Course/JoinCodeForm.tsx:27 navigates to `/enroll/${trimmed}`. CourseCatalog.tsx:128 and CourseUnavailable.tsx:37 render the same form. Finding 157 is right that the codes exist and wrong that a self-enrol route and landing field are missing.
```
FIX: No code change. Hand the two join codes to the cohort (the teacher Share tab already displays them) and delete findings 141/183 from the backlog. The only real gap is that a student who has not been given a code has no way to obtain one — which is the intended design for a class, not a defect.

---

## [major/platform/platform] COLLAPSE and EXTENSION of findings 5, 16, 24, 31, 122, 144, 160, 188 — backend/app/views/
Eight findings report one bug, and all eight under-count it. The audit names three routes; there are fifteen unguarded sites, and the correct pattern already exists in the same file as one of them.

EVIDENCE:
```
`grep -rn 'get_visible_competition_or_404(' backend/app/views/ | grep -v 'def \|import'` = 22 call sites. `grep -rn 'except HTTPException' backend/app/views/` = 3 (ranking.py:117, competitions.py:457, direct_attache_agents/monitor.py:171). A scan for call sites inside a `try:` whose nearest `except` is a bare `except Exception` with no HTTPException guard returns 15: leaderboard.py:77, competition_asset.py:23 and :122, teams.py:20/:44/:202, agent_attached_result.py:140, competitions.py:458/:595/:802/:842/:923, direct_attache_agents/file.py:356, status.py:208/:306. Verified live: /api/leaderboard/competition/179 and /99999 both return 500 {"error":"Failed to fetch leaderboard"}; /api/competition_asset/179/markdown/overview and /99999/... both return 500. The guarded route behaves correctly — /api/competitions/179 returns a real 404.
```
FIX: Add `except HTTPException: raise` immediately above each of the 15 bare `except Exception` blocks, copying competitions.py:457-460 verbatim including its comment — or better, extract one `@propagates_aborts` decorator and apply it, since 22 call sites will keep growing.

---

## [major/validation/courseware] SCOPE CLOSURE on finding 28 — all 102 published lesson bodies vs courseware/content
Finding 28 found the published lab-4 is an older draft than its source and nobody established whether that drift was systemic. It is not. Exactly one lesson of 102 differs, which means one republish closes it permanently — but the republish must be sequenced after the package-name fix or it ships a known-broken install line.

EVIDENCE:
```
Normalised diff (image refs neutralised, whitespace collapsed) of every file under student_view/<course>/<module>/*.md against courseware/content/<course>/{<module>|reference|project}/<lesson>.md: 102 published / 102 authored, per course 29/29 and 73/73, one mismatch — python-ai-engineering/s4-pytorch-nutshell/lab-4 (4098 published bytes vs 4312 authored). Zero NO-SOURCE, zero other drift.
```
FIX: First edit courseware/content/python-ai-engineering/s4-pytorch-nutshell/lab-4.md:93 `uv pip install mlarena` -> `uv pip install mlarena-sdk` (finding 29 — wrong in the source too, so the republish alone does not fix it). Then `make publish COURSE=python-ai-engineering`. No other lesson needs republishing.

---

## [major/platform/platform] SEQUENCING COUPLING on finding 156 — backend/app/views/academic_courses/legacy.py:29-90
The unauthenticated join-code leak is real, and it is also currently the only way anyone discovers a code without being handed one. Fixing it in isolation, before the codes have been distributed to the cohort, silently removes the enrolment path this audit just established is working.

EVIDENCE:
```
Anonymous `curl 'https://ml-arena.com/api/academic_courses/?show_all=true'` returns 200 and an array of 15 courses; every element carries join_code and enrollment_link, including private ones (course 13 'Knowledge Graph & Embeddings', visibility=private, join_code A9QEVXAM). Response keys include both fields. The route at legacy.py:29 has no auth decorator and serialises via the full AcademicCourse.to_dict().
```
FIX: Strip join_code and enrollment_link from the listing serialiser (keep them in the teacher-scope payload only) — and in the same change, distribute GR1WFC63 / N1DX2QA4 to the MS2A cohort, or the enrolment path goes dark the moment the leak closes.

---

## [major/platform/platform] backend/app/views/competition_asset.py:19-46
The route that serves every competition overview — the sole carrier of every baseline in this audit — is unauthenticated and writes to disk on a GET. A read of an id whose overview file is absent creates that file containing the placeholder text, which means the placeholder overviews the audit attributes to lazy authoring may have been written by the platform itself on first read.

EVIDENCE:
```
competition_asset.py:20-21 `@bp.route("/<int:competition_id>/markdown/overview", methods=["GET"])` followed directly by `def get_markdown_overview` — no decorator, no rate limit, unlike the PUT/POST at :49 which carries @admin_required. Lines 27-35: `if not os.path.exists(overview_path): ... manage_storage.write_file(overview_path, DEFAULT_MARKDOWN_CONTENT)` where DEFAULT_MARKDOWN_CONTENT (line 14) is exactly `**Competition Overview**\n\nWrite your markdown here.` Live: competitions 169 and 170 return content beginning with that literal string (findings 106, 120, 161).
```
FIX: Make the GET read-only: return 404 when overview.md is absent rather than creating it, and move file creation to the authoring PUT. Add @auth_required('user') or leave it public deliberately, but decide — right now it is the one competition surface with no access control at all.

---

## [major/baseline/courseware] courseware/competitions/{s1-textstats,s2-readability}/overview.md vs s3-adult-income/overview.md
Findings 2, 13, 119, 124 and 130 all say the baseline contract is missing or inconsistent without naming that the house style already exists, fully worked, in the same directory. Two of the four packaged competitions use it and two do not, so the fix is a copy job on two files rather than a design exercise across twenty-one.

EVIDENCE:
```
s3-adult-income/overview.md:50-63 states 'always predict 0 gets you 76% accuracy and an F1 of 0.00', then 'logistic regression ... scores F1 = 0.656 ... That is the bar', then a four-row ladder (0.000 / 0.656 / 0.692 / 0.717). s4-mnist-warmup/overview.md:61-62 states '91.3%' and 'clears 97%'. s1-textstats/overview.md and s2-readability/overview.md contain no target at all — their only number is `benchmark_expected_score: 1.0` in config.py, which is creator-side. Confirmed the ladders are true: live `__benchmark__` rows score 0.656175 on 181 and 0.9134 on 182, matching the config values exactly. 179 and 180 benchmark at 1.0.
```
FIX: Add a Baselines section to s1-textstats/overview.md and s2-readability/overview.md in the s3-adult-income shape. For 179 the ladder is measurable today: score agent_broken.py and a partial implementation against env.py via competitions/localtest.py and publish the resulting pass_rates alongside the reference 1.0. Same for 180.

---

## [major/platform/courseware] ADJUDICATION and CORRECTION of findings 6 and 36 — frontend/src/App.js:149-151, components/Course/courseNavigation.ts:16-31
Two agents give incompatible explanations for the same two dead links. Finding 36 says the router has no URL shape that a relative link can produce; finding 6 says it matches with moduleSlug preserved. Both are wrong about the mechanism, and neither states the fix, which is that a relative link cannot express this target at all.

EVIDENCE:
```
App.js:151 `<Route path="/courses/:slug/:moduleSlug/:kind/:lessonSlug" .../>` — the shape exists and has four segments, so finding 36's premise fails. From /courses/python-ai-engineering/s1-git-and-packaging/course/git-essentials, `../paie-reference/git-cheatsheet` resolves to /courses/python-ai-engineering/s1-git-and-packaging/paie-reference/git-cheatsheet, which binds slug=python-ai-engineering, moduleSlug=s1-git-and-packaging, kind=paie-reference, lessonSlug=git-cheatsheet — the kind segment absorbs the module name, which is not what finding 6 states. courseNavigation.ts:16-18 `kindToSegment` maps DB kind 'lesson' -> 'course' and 'exercise' -> 'exercise'; DB confirms paie-reference/git-cheatsheet and paie-reference/autograd-mathematics are both kind='lesson'. Only two such links exist in the whole published corpus (git-essentials.md:262, autograd.md:5).
```
FIX: Replace both with absolute hrefs: `/courses/python-ai-engineering/paie-reference/course/git-cheatsheet` and `/courses/python-ai-engineering/paie-reference/course/autograd-mathematics`. A relative link can never be correct here because the target's kind segment is not derivable from the source lesson.

---

## [minor/platform/platform] REFINEMENT of finding 129 — mlarena-sdk/mlarena/client.py
Finding 129 names four header-less methods; the real set is six, and one apparent member is a false positive that would waste a fix.

EVIDENCE:
```
client.py:62-70 `_request` sets only allow_redirects and timeout — auth is passed per-call as `headers=self._headers()`. Scanning every method for a `self._request(...)` with no headers kwarg: competitions (:136), competition (:179), leaderboard (:1324), global_ranking, user_global_rank, list_tags (plus its private helper _resolve_tag_names). download_dataset and tail_logs also match the scan but are correct by design — download_dataset's header-less call is the pre-signed object URL, documented in its own docstring, and it reaches it via `self.datasets()`, which does send headers.
```
FIX: Add `headers=self._headers()` to the six real cases. Leave download_dataset and tail_logs alone.

---

## [minor/content/courseware] EXTENSION of finding 153 — courseware/README.md, 'Known gaps'
The README is stale on two counts, not one. Finding 153 catches the images note; the project-competitions note is equally false and points a reader at work that is already done.

EVIDENCE:
```
README says 'The project competitions are not attached ... the competition ids do not exist yet, so there is no `competitions:` block.' course.yaml declares competitions [172, 169, 171] for mlp-project and [168, 170] for mlp-reference, and the DB confirms 22 module_competition_link rows across both courses including all five. The images note is stale too: the backend deployment now sets PATH_COURSES=/app/storage/courses (confirmed by reading the live deploy env), which is the exact variable the note says is unset.
```
FIX: Delete both bullets. The four genuinely-open gaps in that section are the placeholder dates, the hidden 179-182, the unused directives, and — newly — the s1/s2 overviews with no stated baseline.

---

## [minor/platform/platform] MECHANISM behind finding 157 — frontend/src/pages/CourseLearner/LessonReader.tsx:30-50, backend/app/views/academic_courses/consumption.py:259-266
There is a second enrolment prompt, inside the lesson reader, that carries the join code automatically — and it can never fire on these two courses, because it is triggered only by a gated lesson and neither course gates anything.

EVIDENCE:
```
consumption.py:261-266 returns 403 with `{"error": "This lesson requires enrollment", "gated": True, "join_code": course.join_code}` — the payload hands the code to the client. LessonReader.tsx:30-50 renders 'Enroll to read this lesson' with the code pre-filled from that response. DB: `count(*) FILTER (WHERE l.gated)` = 0 for both courses (73 published / 0 drafts / 0 gated on course 15; 29 / 0 / 0 on course 14).
```
FIX: No change needed for enrolment (the landing-page form already covers it). Worth knowing as a lever: gating one lesson per module would surface a code-prefilled enrol prompt at the point of need, but that trades away public browsability.

---

## [minor/structure/courseware] COVERAGE — courseware/tools/build_slides.py, all 17 module decks; all 14 lab rubrics
Two surfaces nobody exercised come back clean, and saying so matters as much as the defects: it stops the owner treating the deck pipeline and the grading arithmetic as unknown risk during a five-day window.

EVIDENCE:
```
`make slides COURSE=python-ai-engineering` and `make slides COURSE=ms2a-machine-learning-practice` both succeed with --strict-assets, producing 17 decks (course 14: 83/73/78/88/65 slides; course 15: 89/91/88/89/90/86/89/89/90/90/47/55). No missing-asset failure, so every image reference in the authoring tree resolves locally. Separately, extracting percentage weights from all 14 published lab bodies: every rubric sums to exactly 100 (e.g. PAIE lab-1 30+25+25+20, MLP lab-4 15+10+10+15+10+15+10+15).
```
FIX: Nothing to fix. Note only that the deck build is currently green and finding 189 means the first `mlarena:` directive added to any lesson silently breaks that module's deck — so re-run `make slides` as an acceptance check on any directive work.

---

