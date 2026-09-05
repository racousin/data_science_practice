# Audit findings — pae-s4

20 findings

## 1. [blocker / content / fix in courseware] s4-pytorch-nutshell/lab-4

**Problem.** The published Lab 4 Part D is an older draft than the authoring source. The authored file already fixes three defects — it names competition 182, uses the correct `files=` kwarg, and tells the student to download X_test.csv and warns about the uint8 0-255 pixel convention. None of that reached the server, so the lesson students actually read still has all three defects.

**Evidence.**
```
diff courseware/content/python-ai-engineering/s4-pytorch-nutshell/lab-4.md student_view/python-ai-engineering/s4-pytorch-nutshell/lab-4.md
86,90c86,87
< **PAIE S4 — MNIST Warm-up** (competition `182`) takes a single
< `submission.csv` of predictions on 5,000 held-out digits. Download `X_test.csv`
< from the competition's data tab; the columns `p0 … p783` are the image
< flattened row-major as `uint8` 0-255, i.e. what `datasets.MNIST` gives you
< before `ToTensor()`.
---
> The warm-up competition takes a single `submission.csv` of predictions on the
> held-out test set.
100,101c97,98
< client.submit(competition_id=182, files=["submission.csv"])
< print(client.leaderboard(182).head())
---
> client.submit(competition_id=<id>, path="submission.csv")
> print(client.leaderboard(<id>).head())

The published line is not merely stylistically wrong, it raises:
  $ python -c 'c.submit(competition_id=182, path="submission.csv")'
  TypeError: MLArenaClient.submit() got an unexpected keyword argument 'path'
(submit's real signature, mlarena-sdk/mlarena/client.py:1226, is `submit(competition_id, agent=None, files=None, ...)`)
```

**Proposed fix.** Republish the course from courseware/content/python-ai-engineering/ (`make publish`, or `author_course_from_dir`). No editing needed for these three — the authored file is already correct. Verify afterwards with `c.lesson("python-ai-engineering","s4-pytorch-nutshell","lab-4")` and grep the body for `competition_id=182`.

---

## 2. [blocker / content / fix in courseware] s4-pytorch-nutshell/lab-4 (authored line 93, published line 90)

**Problem.** `uv pip install mlarena` installs the wrong package. PyPI's `mlarena` is an unrelated third-party ML toolkit; the ML-Arena SDK is published as `mlarena-sdk` (it imports as `mlarena`). This is wrong in BOTH the published lesson and the authoring source, so the republish in the previous finding will not fix it.

**Evidence.**
```
$ curl -s https://pypi.org/pypi/mlarena/json | jq -r '.info.summary, .info.author'
An algorithm-agnostic machine learning toolkit for model training, diagnostics and optimization
Mena Wang

$ uv run --with mlarena --no-project python -c "import mlarena; mlarena.connect(api_key='mlk_user_x')"
AttributeError: module 'mlarena' has no attribute 'connect'
top-level names: ['PreProcessor', 'preprocessor', 'utils', 'version']

$ curl -s https://pypi.org/pypi/mlarena-sdk/json | jq -r '.info.name, .info.version'
mlarena-sdk
0.3.0

$ grep -rn "install mlarena" tmp/data_science_practice/
student_view/.../lab-4.md:90:uv pip install mlarena
courseware/content/.../lab-4.md:93:uv pip install mlarena
```

**Proposed fix.** In courseware/content/python-ai-engineering/s4-pytorch-nutshell/lab-4.md line 93, replace `uv pip install mlarena` with `uv pip install mlarena-sdk`, then republish. (Add a one-line note under it: "the distribution is `mlarena-sdk`; it imports as `mlarena`." — the mismatch is exactly the kind of thing a student will re-type from memory.)

---

## 3. [blocker / competition / fix in platform] competition 182 (PAIE S4 — MNIST Warm-up), attached to module s4-pytorch-nutshell

**Problem.** Competition 182 was created with is_public=False and never flipped, so a student token cannot open it, list its datasets, or read its leaderboard. Concretely for this module: Lab 4 Part D (10% of the lab grade, and the module's stated second deliverable — "a leaderboard entry") is impossible, X_test.csv is unobtainable, and every number that would tell the student whether their model worked lives only in overview.md, which is behind the same 404.

**Evidence.**
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

**Proposed fix.** Run `python tools/build_competitions.py publish` (documented in courseware/competitions/README.md, "Publishing") to flip 182 to is_public=True before the 2026-09-07 start date. Then re-run `c.competition(182)` and `c.datasets(182)` with the student key as an acceptance check — publishing the course without this check is what let the dead link ship.

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

## 6. [blocker / content / fix in courseware] python-ai-engineering/s4-pytorch-nutshell/lab-4 (published lesson body)

**Problem.** The published Lab 4 Part D — the only lab in course 14 that tells a student to submit — contains a submit call that raises, and an install line that pulls the wrong PyPI package. The authoring source has already been fixed; the fix was never republished.

**Evidence.**
```
Published body (student_view/python-ai-engineering/s4-pytorch-nutshell/lab-4.md:97-98): `client.submit(competition_id=<id>, path="submission.csv")` / `print(client.leaderboard(<id>).head())`. The SDK signature is `submit(self, competition_id, agent=None, files=None, ...)` (mlarena-sdk/mlarena/client.py:1226) and :1244 raises `SubmissionError("Provide exactly one of agent= or files=")` when both are None — `path=` is also an unexpected kwarg. The source file courseware/content/python-ai-engineering/s4-pytorch-nutshell/lab-4.md:100-101 already reads `client.submit(competition_id=182, files=["submission.csv"])`. Line 93 of both says `uv pip install mlarena`; `mlarena` on PyPI is an unrelated package — the SDK is `mlarena-sdk`.
```

**Proposed fix.** Republish the course content (`make publish` / `client.author_course_from_dir`) so the fixed lab-4 body reaches the server, and change `uv pip install mlarena` to `uv pip install mlarena-sdk` in courseware/content/python-ai-engineering/s4-pytorch-nutshell/lab-4.md:93 first. Diff of all 42 drifted lesson bodies shows the rest of the drift is benign image-path rewriting (`assets/git/commit-main.png` -> `/api/academic_courses/assets/lessons/28/commit-main.png`); lab-4 is the only substantive one.

---

## 7. [blocker / competition / fix in courseware] python-ai-engineering/ (all 4 sessions; competitions 179,180,181,182)

**Problem.** Every competition in the 12h course is invisible to students. All four were created with is_public=False and never flipped, so the only gradeable artefact of the whole course 404s for an enrolled student.

**Evidence.**
```
Student token mlk_user_...: `c.competition(179)` -> CompetitionNotFoundError: Not Found; same for 180, 181, 182. Creator token reaches all four (HTTP 200 on /api/competition_asset/179/markdown/overview). courseware/README.md 'Known gaps' already records this: "Ids 179-182 are live, benchmarked and attached to modules 14-17 ... But they are still is_public=False, so enrolled students get a 404".
```

**Proposed fix.** Run `MLARENA_API_KEY=mlk_creator_... make competitions-publish` in courseware/ (i.e. `python tools/build_competitions.py publish`), after setting the real term dates in content/python-ai-engineering/course.yaml. Then re-verify with the student token that competition(179..182) returns 200.

---

## 8. [blocker / competition / fix in platform] competitions 179, 180, 181, 182 (course 14, sessions 1-4)

**Problem.** All four PAIE competitions 404 to a student token, so the single validation surface that already produces an objective number is unreachable for the entire 12-hour course. Any design that folds competition score into progress is dead here until they are flipped public.

**Evidence.**
```
Live with the student key: `c.competition(179/180/181/182)` -> `CompetitionNotFoundError: Not Found` for all four. The module payloads carry the same error inline, e.g. student_view/python-ai-engineering/s1-git-and-packaging/_module.json: {"competition_id": 179, ..., "error": "CompetitionNotFoundError: Not Found"}. Cross-check: all 17 competitions attached to course 15 return is_public=True, is_started=True.
```

**Proposed fix.** Run `make competitions-publish` in tmp/data_science_practice/courseware (it exists and flips is_public — Makefile target `competitions-publish`), then re-run `python tools/student_walk.py check --course python-ai-engineering` and confirm zero `competition-unreachable` lines. Until that runs, do not author any `mlarena:target` block against 179-182: an unreachable target renders a red light the student cannot clear.

---

## 9. [major / content / fix in courseware] s1-git-and-packaging/git-essentials.md:262

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

## 10. [major / validation / fix in courseware] s4-pytorch-nutshell/lab-4, Part C

**Problem.** Part C asks the student to train an MLP on MNIST and write final metrics to results.json, but states no target number anywhere — not a validation accuracy, not a loss, not a baseline. The grading rubric grades process (eval mode, no_grad, early stopping) but never outcome. The only self-check number in the whole lab is `loss below 0.1` in the Part B overfit test, which passes on random labels and says nothing about MNIST. The numbers that would answer "did I get it right" (91.3% / 97%) exist only in competition 182's overview, which the student cannot open.

**Evidence.**
```
$ grep -n "9[0-9]%\|accuracy\|0\.9" student_view/python-ai-engineering/s4-pytorch-nutshell/lab-4.md
(no matches)
Part C in full reads: "Train an MLP on MNIST. Requirements: seeded / train-validation split, stratified / both losses printed every epoch / early stopping with best-checkpoint restore / final metrics written to results.json"
I followed it exactly and got val_acc 0.975 after 5 epochs — with no way, from the lesson alone, to know whether 0.975 was good, bad, or evidence of a bug.
```

**Proposed fix.** Append to Part C, after the results.json bullet: "**What good looks like.** A correctly wired 784-128-10 MLP with Adam at `lr=1e-3` reaches **≥97% validation accuracy within 5 epochs** (about 4 seconds an epoch on a laptop CPU). Below 95%, stop and re-read the failure table in *Training Loop End to End* — the cause is almost always normalisation, a softmax before `CrossEntropyLoss`, or a missing `model.eval()`. A linear model on the same pixels gets 91.3%; if you are under that, your network is not learning at all."

---

## 11. [major / content / fix in courseware] s4-pytorch-nutshell/lab-4:25 (Part A) vs :68 (Part C)

**Problem.** Part A instructs "Add `torch` to `pyproject.toml` dependencies and re-lock", but Part C's code imports `torchvision`, which is a separate distribution. A student who follows Part A literally produces a repo that fails the lab's own top grading criterion — "`uv sync && uv run pytest` green on a fresh clone", 20% of the grade — with ModuleNotFoundError as soon as any test touches the data module.

**Evidence.**
```
lab-4.md:25  "Add `torch` to `pyproject.toml` dependencies and re-lock."
lab-4.md:68  "from torchvision import datasets, transforms"
lab-4.md:120 Grading: "| `uv sync && uv run pytest` green on a fresh clone | 20% |"
torchvision is named nowhere else in the entire course except one passing mention in why-tensors.md:131 ("Gymnasium, PettingZoo, torchvision, Hugging Face all speak it"), where it is not presented as a dependency.
My own package needed it: `from torchvision import datasets, transforms` in src/mlp/data.py, and `uv run --with torch` alone was not enough to run the tests.
```

**Proposed fix.** Change lab-4.md line 25 to: "Add `torch` **and `torchvision`** to `pyproject.toml` dependencies and re-lock — Part C loads MNIST through `torchvision.datasets`, and a fresh clone with only `torch` fails `uv run pytest` with `ModuleNotFoundError: No module named 'torchvision'`."

---

## 12. [major / content / fix in courseware] s4-pytorch-nutshell/modules-and-optimizers:20

**Problem.** The lesson's only import statement is `import torch.nn as nn`, which binds the name `nn` but NOT `torch`. Two later blocks then call bare `torch.*` and both raise NameError when copied as written — including the `MLP` subclass, which is the exact class Lab 4 Part A requires the student to put in `src/mlp/model.py`. The failure is inside `forward()`, so it surfaces at first call rather than at definition, which is a confusing place for a beginner to land.

**Evidence.**
```
$ cat blk1.py   # verbatim from the lesson
import torch.nn as nn
layer = nn.Linear(in_features=10, out_features=5)
x = torch.randn(32, 10)     # batch of 32
$ uv run --with torch python blk1.py
NameError: name 'torch' is not defined

$ cat blk2.py   # the MLP block, verbatim
import torch.nn as nn
class MLP(nn.Module):
    ...
    def forward(self, x):
        x = torch.relu(self.fc1(x))
$ uv run --with torch python blk2.py
  File "blk2.py", line 8, in forward
    x = torch.relu(self.fc1(x))
NameError: name 'torch' is not defined

Every other Session 4 lesson gets this right (why-tensors:62, tensor-mechanics:16, autograd:36, training-loop-end-to-end:57 all say `import torch`); modules-and-optimizers is the only one that does not.
```

**Proposed fix.** In courseware/content/python-ai-engineering/s4-pytorch-nutshell/modules-and-optimizers.md, change line 20 from
```
import torch.nn as nn
```
to
```
import torch
import torch.nn as nn
```
(`import torch.nn as nn` binds only `nn`; the lesson uses `torch.randn` and `torch.relu`.) Alternatively change `torch.relu(...)` in the MLP block to `nn.functional.relu(...)` — but adding the import is better, since the student copies this class straight into src/mlp/model.py.

---

## 13. [major / platform / fix in both] s4-pytorch-nutshell/autograd:5 (and s1-git-and-packaging/git-essentials:262)

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

## 14. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 15. [major / platform / fix in platform] mlarena-sdk/mlarena/client.py:161, :170, :192, :1338

**Problem.** competition(), competitions() and leaderboard() send no Authorization header, so an enrolled student using the SDK or MCP is treated as anonymous and cannot open any non-public course competition — the exact class that 179-182 belong to.

**Evidence.**
```
client.py:192 `resp = self._request("GET", self._url(f"/competitions/{competition_id}"), timeout=30)` — no `headers=self._headers()`, unlike every scoped call (e.g. :225 datasets passes it). Same at :161/:170 (competitions) and :1338 (leaderboard). Observed effect: `mlarena.connect(CREATOR).competition(179)` raises CompetitionNotFoundError, while the same creator token via raw requests with a bearer header returns 200 — the owner is being told their own competition does not exist. The backend's visibility_filter (_helpers.py:229-233) grants access on `is_public OR owned OR assistant`, and the enrolled-student side-channel in competitions.py:213-217 is likewise keyed on `current_user.is_authenticated`.
```

**Proposed fix.** Add `headers=self._headers()` to the four calls. These routes accept an anonymous caller, so the change is backward-compatible and only widens what an authenticated caller can see. Without it, publishing 179-182 still leaves them invisible to any SDK/MCP student even after enrollment is fixed.

---

## 16. [major / competition / fix in courseware] 11 of 14 labs; python-ai-engineering/s4-pytorch-nutshell/lab-4.md:97

**Problem.** Labs do not point at the competition attached to their own module, and the one that tries ships an unfilled placeholder. The objective number the platform already computes is invisible from the page that should be sending the student to it.

**Evidence.**
```
Cross-referencing each lab against its module's `_module.json`: only MLP lab-9, MLP lab-10 and PAIE lab-4 mention ML-Arena at all; the other 11 never do, including s3-tabular-models/lab-3 (competition 172 attached), s5-computer-vision-1/lab-5 (173), s7-nlp-1/lab-7 (174 and 178), s8-nlp-2/lab-8 (165). No lab quotes the competition id it is attached to. PAIE lab-4 line 97 ships `client.submit(competition_id=<id>, path="submission.csv")` and line 98 `print(client.leaderboard(<id>).head())` — literal `<id>`. It also uses `path=` while the SDK signature (mlarena-sdk/mlarena/client.py:1226) and the competition's own overview both use `files=[...]`.
```

**Proposed fix.** In PAIE lab-4 replace lines 96-98 with `client.submit(competition_id=182, files=["submission.csv"])` / `print(client.leaderboard(182).head())`. In each of the other 10 labs add the module's competition as an `mlarena:target` (or at minimum an `mlarena:competition id=<n>` card) so the link is a rendered widget rather than prose a student has to go hunting for.

---

## 17. [minor / competition / fix in courseware] competition 182 / courseware/competitions/s4-mnist-warmup/prepare_data.py:52

**Problem.** 86% of the evaluation images are byte-identical to images in the torchvision MNIST *train* split that Lab 4 Part C tells the student to train on. prepare_data.py draws the 5,000 eval rows from the whole 70k OpenML corpus (60k train + 10k test), so most of them land inside the student's training data. The competition is honest that it is a plumbing check, but the leakage means the "clears 97%" bar is met by memorisation as much as by correct wiring, and the leaderboard rewards training longer rather than training right.

**Evidence.**
```
prepare_data.py:52: `order = rng.permutation(len(X))` over `fetch_openml("mnist_784")` (70,000 rows), `test_idx = order[:5000]`.

Measured: loaded torchvision MNIST(train=True) (60,000 images), hashed each 28x28 uint8 array, and compared against the 5,000 rows of X_test.csv:
  OVERLAP: 4300/5000 eval images are byte-identical to an image in the torchvision TRAIN split Part C tells you to train on (86.0%)
Consistent with the effect: my model scored 0.9862 on the competition test set but only 0.975 on its own held-out validation split.
```

**Proposed fix.** Either (a) change prepare_data.py to draw the eval rows from the OpenML rows corresponding to the MNIST *test* split only — `order = rng.permutation(np.arange(60000, 70000))` — and regenerate data/ + benchmark_submission.csv + benchmark_expected_score; or (b) if the leak is acceptable for a plumbing check, say so explicitly in overview.md's "What this measures, honestly" section: "About 86% of these 5,000 images are also in the torchvision train split you trained on, so treat your leaderboard number as an upper bound — your own validation accuracy is the honest one." Option (b) is cheap and preserves the deliberate plumbing-check framing.

---

## 18. [minor / structure / fix in courseware] paie-reference (module 18)

**Problem.** The module summary says the material is "Self-study, linked from the sessions, never lectured", but four of its six lessons are linked from nowhere and mentioned by name nowhere — GitHub Desktop, GitHub Actions, IDE/Syntax/Linting, and A Short History of Deep Learning. The two that do have inbound links (git-cheatsheet, autograd-mathematics) are linked with the broken URLs in the previous finding. Net result: zero of six reference lessons is reachable by following a link, and 85 minutes of authored material is discoverable only by browsing the module list.

**Evidence.**
```
$ grep -rn "paie-reference\|Reference →" student_view/python-ai-engineering/ --  (excluding _index/_module.json)
s4-pytorch-nutshell/autograd.md:5   -> autograd-mathematics   (broken URL)
s1-git-and-packaging/git-essentials.md:262 -> git-cheatsheet      (broken URL)

$ for t in "GitHub Desktop" "GitHub Actions" "pre-commit" "AlexNet"; do grep -rn "$t" s1-git-and-packaging s2-agentic-coding s3-data-science-nutshell s4-pytorch-nutshell; done
(no matches for any)

Ruff is the sharpest case: s1-git-and-packaging/packaging-and-tests.md:72 puts `ruff>=0.6` in pyproject.toml and s2-agentic-coding/guardrails-and-review.md:40 allowlists `Bash(uv run ruff check:*)`, but neither points at ide-syntax-linting.md, which is the lesson that explains what ruff is and how to configure it.

_module.json summary: "Material demoted out of class time. Self-study, linked from the sessions, never lectured."
```

**Proposed fix.** Add one absolute link at each natural point of need, using the /courses/<slug>/<module>/course/<lesson> shape from the previous finding: packaging-and-tests.md near line 72 -> ide-syntax-linting; guardrails-and-review.md near line 40 -> ide-syntax-linting; s1-git-and-packaging/packaging-and-tests.md (or lab-1's CI mention) -> github-actions; s1-git-and-packaging/git-essentials.md near the client discussion -> github-desktop; s4-pytorch-nutshell/why-tensors.md:131 (the ecosystem bullet) -> deep-learning-history. If a lesson genuinely has no caller, either delete it or change the module summary from "linked from the sessions" to "browse it when you need it" so the claim is true.

---

## 19. [minor / validation / fix in courseware] paie-reference/autograd-mathematics

**Problem.** Twenty-five minutes of derivations — forward vs reverse mode, the delta recursion, parameter gradients, vanishing/exploding — with no way for the reader to check that any of it landed. There is no exercise, no worked numeric example, no question. This is the module's densest page and its least verifiable one, and it is unusual because Session 4's taught lessons all give the reader a checkable number (autograd.md's `w.grad == -12`, modules-and-optimizers' `55` parameters, training-loop's overfit-to-~0). The page even ships the verification tool and then never asks the reader to use it.

**Evidence.**
```
autograd-mathematics.md, "Checking a hand-written gradient":
  torch.autograd.gradcheck(my_function, (x.double().requires_grad_(),))
`my_function` is never defined and the reader is never asked to run it. No other runnable block, expected output, or question appears in the 5,451-byte body.
By contrast s4-pytorch-nutshell/autograd.md:191-205 gives `print(w.grad)  # tensor([-12.])` and then derives -12 by hand so the reader can confirm.
```

**Proposed fix.** Add a closing section, ~10 lines, that closes the loop with the page's own machinery:

"## Check it yourself\n\nOne hidden layer, by hand and by autograd. Derive $\\partial L/\\partial W^{(1)}$ from the recursion above for a 2-1-1 network with $\\sigma = \\tanh$ and squared error, then confirm it:\n\n```python\nimport torch\nW1 = torch.tensor([[0.5, -0.5]], dtype=torch.float64, requires_grad=True)\nW2 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)\nx  = torch.tensor([[1.0], [2.0]], dtype=torch.float64)\ny  = torch.tensor([[1.0]], dtype=torch.float64)\na1 = torch.tanh(W1 @ x)\nloss = ((W2 @ a1 - y) ** 2).sum()\nloss.backward()\nprint(W1.grad)   # your hand-derived delta^(1) (a^(0))^T should match to ~1e-12\n```\n\nIf they disagree, the discrepancy is in your transpose or in $\\sigma'$ — those are the two places it always is. `float64` is deliberate: `float32` does not have the precision for the comparison to mean anything."

---

## 20. [minor / structure / fix in courseware] s4-pytorch-nutshell/lab-4:7

**Problem.** The lab is budgeted at 45 minutes and the per-part budgets sum to exactly 45 (A 10 + B 10 + C 15 + D 10), leaving zero minutes for Part E — which is worth 10% of the grade and asks for five substantive written points, including "one thing you tried that did not help". A student working to the stated clock will write the PR description in no time at all, which is exactly the deliverable the lab says is not filler.

**Evidence.**
```
lab-4.md:7   "**Time:** 45 minutes. **Deliverable:** a merged PR + a leaderboard entry."
lab-4.md:11  "## Part A — Package it (10 min)"
lab-4.md:32  "## Part B — Tests that catch real bugs (10 min)"
lab-4.md:57  "## Part C — Train (15 min)"
lab-4.md:84  "## Part D — Submit to ML-Arena (10 min)"
lab-4.md:106 "## Part E — Pull request"   <- no budget
lab-4.md:118 "That last one is not filler. A PR with only successes describes a process that did not happen."
lab-4.md:129 Grading: "| PR description covers all five points | 10% |"
(Compute is not the constraint — measured on a laptop CPU, Part C's whole training run is ~15 seconds: MNIST load 3.0s, one-batch overfit 0.8s, 3.6s per epoch. The 15 minutes is all typing.)
```

**Proposed fix.** Change line 7 to "**Time:** 55 minutes." and line 106 to "## Part E — Pull request (10 min)". Alternatively cut Part C to 10 min (the compute is 15 seconds; the budget is for writing the loop, and most of it is already on the slide in *Training Loop End to End*) and give the 5 minutes to Part E.

---

