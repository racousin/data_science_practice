# Course state — audited 2026-09-02, corrections applied and published 2026-09-03,
# Session 1 rebuilt and republished 2026-09-06

What the two ML-Arena courses actually are today, what a student hits when they try to
follow them, and what to build next. Produced by walking both courses end to end with a
student token — 102 lesson bodies, 22 module→competition attachments, all four
consumer surfaces — and verifying every claim against the live platform.

Companion documents:

| | |
|---|---|
| Platform-side gaps (ML-Arena itself) | `../../../docs/plan_course_student_path_gaps.md` |
| Every finding, with evidence | `_audit/by_group/*.md`, `_audit/raw_findings.json` |
| Findings needing a platform change | `_audit/platform_findings.md` |
| Re-run the student walk | `tools/student_walk.py` (see bottom) |

---

## 1. What is live

| | Course 14 | Course 15 |
|---|---|---|
| Name | MS2A - AI Engineering | MS2A - Machine Learning Practice |
| Slug | `python-ai-engineering` | `ms2a-machine-learning-practice` |
| Volume | 12h — 4 × 3h, one week | 30h — 10 × 3h, ten weeks |
| Modules / lessons | 4 / 56 | 12 / 74 |
| Competitions attached | 10 (179-188) | 17 |
| Dates | **2026-09-07 → 2026-09-11** | **2026-09-14 → 2026-11-27** |
| Join code | `GR1WFC63` | `N1DX2QA4` |
| Enrolled students | 1 (test account) | 1 (test account) |
| Visibility | public | public |

Both courses are **published, browsable and joinable**. Every lesson body serves, and
every image reference returns 200 — the "132 images never uploaded" note that used
to be in `README.md` was stale and has been removed.

The course-14 row is as of 2026-09-06: the two ML modules were built on 09-05 and
Session 1 was rebuilt on 09-06 (§1c). The 3 / 30 that used to be here predated both.

The dates were **confirmed real on 2026-09-03**, though `course.yaml` still carries the
PLACEHOLDER comment above them — delete it so the next reader does not re-open the
question. They are not cosmetic: `academic_courses/legacy.py:157-159` returns **410 on
the enrol route once `end_date` has passed**, so an expired course cannot be joined at
all.

### 1b. Course 14 restructured and republished, 2026-09-05

Session 2 was **Agentic Coding**. Its six lessons now sit at the end of
`s1-git-and-packaging`, and competition **#180** went with them — Lab 2 is the lesson
that submits to it. The freed slot briefly became **Session 2 — Shell, Notebooks &
Colab** (`s2-shell-notebooks-colab`, module #31), holding one *unpublished*
`session-plan` lesson — the authoring brief for that build, not teaching material.
That module has since been folded in as well; see the end of this section.

Module slugs are immutable (`update_module` accepts title/summary/icon/visibility only),
so a new identity meant a new module. Server module **#15** and its lessons **34-39**
were deleted with `delete_module(15, force=True)` — the six lessons exist again as
**131-136** under module #14. `publish_mlarena.py` never deletes, so that step was
manual; a restructure of this shape always needs it, or the live course shows the same
lessons twice. It also has to happen *before* `reorder_modules`, which rejects an id
list that is not exactly the course's linked set — that is what failed the publish run.

Verified live at that point: 5 modules / 30 lessons in the order above, 12 lessons in
session 1, no relative image paths left in any of the 30 bodies, all 12 image refs 200.

**The `Reference` module was then removed too, same day.** Its six lessons were
self-study material parked outside the sessions, reachable only through three inbound
links, and they now live in the session they belong to: `github-desktop`,
`github-actions`, `ide-syntax-linting`, `git-cheatsheet` at the end of
`s1-git-and-packaging` (lessons **138-141**), `autograd-mathematics` and
`deep-learning-history` at the end of `s4-pytorch-nutshell` (**142-143**). Module **#18**
and its lessons **51-56** are gone — unlinked from the course first, then
`delete_module(18)`; unlinking before the publish is what lets `reorder_modules` see the
exact four-module set, and it keeps the deletion reversible until the new lessons are
verified live.

Three things had to move with them. The titles carry a `Reference — ` prefix, because the
module was what told a student the material is never lectured and it no longer exists.
The three cross-links (`git-essentials`, `autograd`, `why-tensors`) were repointed from
`paie-reference` to the new module slugs. And `build_slides.py` learned `in_deck: false`,
without which the four reference lessons would have been rendered into the session 1 deck
they were demoted out of — the s1 and s4 decks are byte-for-byte the same 173 and 94
slides as before the move.

Verified live: 4 modules / 30 lessons, all six lessons serving at their new paths, the
`paie-reference/*` paths 404, and all three cross-links resolving.

**Session 2 was then folded in too, and deleted.** Session 1 is retitled **"Session 1 —
Git & Python Packaging - Shell, Notebooks & Colab"**, which is now the whole engineering
floor in one module: git, packaging, agentic coding, the four reference lessons, and the
shell/notebook/Colab scope it has yet to carry content for. `session-plan` moved with the
title — last lesson of module #14, `is_published: false`, `in_deck: false` — so the brief
for that material sits in the module that owes it. Module **#31** and lesson **137** are
gone (`delete_module(31, force=True)`, run before the publish so `reorder_modules` saw
the exact three-module set).

Verified live: **3 modules / 30 lessons**, 17 in session 1, no relative image paths in
any body, all 12 image refs 200.

### 1c. Session 1 rebuilt and republished, 2026-09-06

Session 1 was *Git & Python Packaging*; it is now **"Session 1 — The Engineering
Floor"** and carries the whole engineering floor: shell, git, GitHub review,
environments, packaging, tests, linting, CI, notebooks and Colab, then the
agentic-coding block on top. **25 lessons, 27k words, live on module #14.**

Six lessons created — `session-map` (173), `accounts-and-tools` (174),
`the-shell` (175), `pull-requests-and-review` (176), `notebooks-and-colab` (177),
`assistant-landscape` (178) — plus `lab-3` (179).

Two `Reference — …` lessons were **promoted to taught** rather than duplicated:
`github-actions` (139) is now a full CI/CD lesson and `ide-syntax-linting` (140)
a code-quality one. Both kept their slugs, so both updated in place and nothing
had to be deleted server-side — the first restructure of this module that needed
no manual deletion at all.

Labs renumbered around a GitHub deliverable: Lab 1 (33) ships the package, gets
CI green and installs it in Colab; **Lab 2 (136) is now the paired
pull-request-and-review lab**, reusing the slug the agent lab had; the agent lab
is Lab 3 (179). Competitions 179/180 stay attached but are **optional** parts of
Labs 1 and 3 — this session is assessed from the students' GitHub accounts.

The module is deliberately over-provisioned: **725 written minutes against a
180-minute slot**, marked `# core` / `# extension` per lesson in `course.yaml`.
The 3h run-sheet, five scripted live demos, the stall table and the open
decisions are in the unpublished `session-plan` lesson (144), retitled
*Teacher's Run-Sheet*.

**30 figures were authored** by `tools/figures/s1_git_and_packaging.py` — Session
1 has no taught PPTX to lift stills from, so all of its diagrams are drawn by
that script.

Verified live: 25 lessons in order, **all 34 image references 200**, no relative
image paths in any body, `session-plan` confirmed `is_published=False`, and
`student_walk.py check` reports nothing but the known speaker-notes item.
`../student_view/` was re-dumped — it had been the 2026-09-02 baseline, so it
also picks up the 09-03 and 09-05 republishes.

**`Reference — pandas & seaborn` then moved to Session 2, same day.** It teaches
the libraries Session 2 is the first to use, and `s2-ml-foundations/the-data`
links straight to it, so it now sits last in module **#32** as lesson **180**;
module #14 is 24 lessons.

The move cost one manual deletion, and the publish that skipped it **failed**:

```
reorder_lessons failed: ordered_ids must be exactly this module's lesson ids
```

`publish_mlarena.py` never deletes, so lesson 170 stayed in module #14 while the
manifest stopped listing it, and `reorder_lessons` rejects a list that is not the
module's exact set — the same constraint §1b records for `reorder_modules`, one
level down. **`delete_lesson` has to run before the publish, not after.** The
failed run changed nothing live (it stops at the first module and module #32 was
never reached), so recovery was: delete 170, drop its key from
`.mlarena-state.json`, re-publish.

Four references moved with it, and none of them is in the lesson body: the
`the-data` callout (which said "in Session 1"), the `build_preflight` docstring
and the Colab notebook text in `tools/build_notebooks.py`, and one line of
`README.md`. Regenerating the notebooks changed exactly one line of
`aie-s0-pandas-seaborn.ipynb`, which is also the evidence that the other eight
were already in sync.

---

**A silent deck defect was found and fixed.** `build_slides.py` steps down a font
ladder until content fits the 4.95 in body box; when the last rung still does not
fit it renders anyway and the overflow falls off the bottom of the slide, with no
warning. An image block costs a flat **3.4 in**, so adding figures to Session 1
took it from 12 overflowing slides to 37. All 37 are now split at a real boundary
with a heading, and `make check-slides` (`tools/check_slide_overflow.py`, exits
non-zero) makes the next one visible. **`s2-ml-foundations` has 34 and
`s3-models-and-tuning` has 38 — left as they are, deliberately.**

---

**There is now no Session 2** — the course is Sessions 1, 3, 4. The course description
still promises "four 3-hour sessions", and the labs, deck eyebrows and competition
names (`PAIE S2 — Flesch reading-ease`) all carry the old numbers. Module slugs cannot
be renumbered at all. Settle the numbering with the Session 1 split (§5.3):
that module now authors **~5h of lecture plus two labs against a 3h slot**, a 173-slide
deck, and splitting it is what would supply a real Session 2.

Session 1's bodies are unchanged throughout all of this. They still read "this session is
placed second" and "your Session 1 repository"; rewriting waits on the split.

---

## 2. What a student hits

**All of the below has been fixed and is live.** This section is kept as the record of
what was wrong, because the same defects will recur in the next course authored from this
toolchain. What was executed against production is listed in §2b.

- **All four course-14 competitions 404 to a student.** They were created
  `is_public=False` (by design — `build_competitions.py` creates hidden) and the
  documented `make competitions-publish` step was never run. **Done** — all four are
  `is_public=True` and verified reachable with a student token. Their benchmarks were
  green throughout (1.0 / 1.0 / 0.656 / 0.9134, against the live `__benchmark__` rows).
- **10 of the 14 labs never mentioned their competition.** Every session has one
  attached and rendered as a card, but the lab — the thing the student actually does —
  ended at "a merged PR in your project repository". Two disconnected assessment
  systems, and the student had no reason to discover the leaderboard. *Fixed in this
  pass.*
- **Not one of the 102 lessons had a self-check.** No expected output, no question with
  an answer, nothing to compare against. *Fixed in this pass.*
- **Two labs' verify command could not work.** `uv sync && uv run pytest` — worth 30%
  of Lab 1 and the whole VERIFY step of Lab 2 — fails on a repo built to the course's
  own spec, because pytest is declared as a `[project.optional-dependencies]` extra and
  `uv sync` does not install extras. Worse, `uv sync` *uninstalls* the pytest that the
  lesson's own `uv pip install -e ".[dev]"` had just installed. *Fixed in this pass.*
- **Sessions overrun their slots by ~25%.** Not one taught session in either course
  fits, and lecture is 74-81% of every session against a promised 50%. See §5 — this is
  a curriculum decision, not a bug to patch.

---

## 2b. What was executed against production, 2026-09-03

In this order, each verified before moving on:

1. **`make competitions-publish`** — competitions 179-182 flipped `is_public=True`.
   Verified: `competition(179..182)` all return 200 to a student token (was 404).
2. **`make publish`** on both courses — 51 + 239 actions, shipping the 130 corrections
   and creating the new `mlp-reference/reference-playgrounds` lesson.
3. **`UPDATE dataset_file SET storage_backend='r2' WHERE dataset_id IN (6,7,8)`** —
   8 rows. Verified before and after by fetching a real signed URL as a student:
   competitions 172 / 173 / 174 went **404 → 200**. GCS holds no dataset objects at all
   any more (dataset 9 works only because its rows already said `r2`), so this was a
   stale column, never data loss.
4. **Overview markdown for 179 and 180** pushed via `set_competition_markdown`, adding
   the measured baseline ladders. Done with the targeted SDK call rather than
   `make competitions`, which would have re-run all four benchmarks for a text change.

Final state, `tools/student_walk.py check`, both courses: **zero problems** other than
the speaker-notes item, which needs the platform-side fix in
`docs/plan_course_student_path_gaps.md` §4.3. The test student was enrolled in both
courses to confirm the loop closes end to end — `enroll` → `my_progress` → `next_lesson`
all return correctly.

Not run, deliberately: `make competitions` (a full rebuild + benchmark re-run for all
four packages). Nothing needs it today; run it when a package's `env.py` or reference
solution actually changes.

---

## 2c. End-to-end verification, 2026-09-03

The corrections were not trusted — they were tested.

**A real student submission to all four course-14 competitions**, walking exactly the path
the labs now describe (`mlk_user_` token, `client.submit(...)`, the leaderboard read):

| Comp | Lab claims | Actually scored | |
|---|---|---|---|
| 179 textstats | pass_rate 1.000 | **1.000** | matches |
| 180 Flesch | pass_rate 1.000 | **1.000** | matches |
| 181 Adult | F1 0.656 is the bar | **0.656175** | matches *exactly* |
| 182 MNIST | MLP scores 0.982, bar 0.913 | **0.9878** | beats both |

Two defects surfaced that only a real submission could find, both now fixed and republished:

1. **Lab 1 Part F's `agent.py` omitted `__init__`.** Upload validation compares the class
   against the competition's starter template and rejects the submission before anything
   runs: `Missing method '__init__(self)' - required by template`. The code the lab handed
   the student could not be submitted. Part F now carries the method and documents the rule.
2. **Lab 4 never stated the submission schema.** Competition 182 requires `id,label`; Lab 3
   — the lab immediately before it — uses `id,prediction`, so a student carries the wrong
   header forward and is rejected with `Column mismatch. Expected: ['id', 'label']`. Lab 4
   now states the schema and names the trap.

Comp 181 scoring *exactly* 0.656175 also settled a live question: the Lab 3 checklist
demanded `F1 > 0.656`, which the lab's own prescribed pipeline cannot satisfy. Changed to
`≥`.

**Every self-check snippet executed.** 137 assertions across all 103 sections; 132 run, 5
documented as unrunnable. **Zero wrong code outputs** — the authoring pass's claim to have
run them holds. Nine defects were found elsewhere in the self-checks and fixed; the two
worst were checklist gates a correct student fails:

* Lab 9 asked for ≥0.65 on the *Part C* training curve, which a correct agent ends at ~0.44
  (epsilon is still ~0.08 at episode 5,000). 0.65 is the *Part D* greedy number.
* Lab 10's "remove `stable_baselines3` from `sys.modules`" check passes for the exact defect
  it exists to catch — popping the entry forces a re-import. Now `sys.modules[...] = None`.
* Lab 4's ≥0.97 gate failed 1 of 10 correct reference runs (measured 0.9690-0.9762) → ≥0.96.
* A segmentation question's premise (0.02 mIoU) was unreachable — two-class background-only
  gives 0.49. A mis-cited table column, a false `float64` precision claim, a `None`-vs-`np.nan`
  fixture, and an uncheckable row citing a `TBD` milestone date.

**The three `local_vm` engines are down.** Engines 19, 22 and 23 all point at
`vm_host = 91.168.161.7` (the home DGX) and all report `vm_health_ok = false`. That is one
host, not three faults, and it takes competitions 165 (GSM8k), 169 (SuperTuxKart) and 171
(The Round) with it — including two of the three graded project tracks. Their pages load;
their submission paths do not.

---

## 3. Baselines — the state of all 21 competitions

The owner's requirement was "for each course, baseline should be clear and well
defined". Measured against that:

| Verdict | At audit | Now | Competitions |
|---|---|---|---|
| **clear and measured** | 6 | **8** | 177, 176, 172, 178, 181, 182, **+179, +180** |
| stated but unmeasured | 3 | 3 | 43, 48, 168 |
| vague — a starter is named, no number | 8 | 8 | 8, 47, 49, 65, 165, 173, 174, 171 |
| absent | 2 | 2 | 169, 170 (both serve the platform's placeholder text) |
| unreachable to a student | 4 | **0** | 179-182, all flipped public 2026-09-03 |

179 and 180 moved into the top row on 2026-09-03: both now carry a measured ladder
produced by scoring real agents through `competitions/localtest.py`. For 179 the most
instructive rung is that the shipped `agent_broken.py` and an implementation with the
same three functions differing only by a crash score **0.267 vs 0.667** — the flex_v1
error latch costs 40 points, stated on the page as a number. For 180 each rung breaks
exactly one pinned Flesch rule (0.45 / 0.50 / 0.70 / 0.95), so a student's score names
which rule they got wrong.

The remaining 13 are the standing gap: **12 of the 17 MS2A competitions still state no
numeric floor.**

**The house style already exists** and should be copied everywhere:
`competitions/s3-adult-income/overview.md:50-63` states the trivial floor
(`F1 = 0.000`), the reference score labelled *"that is the bar"* (`0.656`), two rungs
above it (`0.692`, `0.717`) and the direction. `s4-mnist-warmup`, 177, 176 and 178 do
the same. Everything else says "beat the baseline" and names no number.

Twelve of the seventeen MS2A competitions point at a Colab notebook holding a
"random-action baseline" and state no score at all. A student whose first CartPole
agent scores 21 cannot learn from the page that 21 *is* the random floor — measured:
random = **20.98 ± 10.89**.

---

## 4. Build list — 7 simple competitions, every number measured

The gap this closes: several sessions attach a competition far too hard to be the
*first* thing a student does after the lecture. Session 5 hands them Blood Cell
Classification; Session 6, which teaches detection and diffusion, hands them
**CarRacing-v3** — a pixel-input RL control task whose every prerequisite lands in
Sessions 9-10. A student who fails cannot tell whether they failed at the concept or at
the engineering.

All data below is already in this repo under `website/public/modules/`. Every number
was produced by running the trivial and the reference solution on the actual data —
none is estimated.

| # | Session | Competition | Data | Metric | Trivial | Reference |
|---|---|---|---|---|---|---|
| 1 | MLP S1 | Store Sales — assemble four sources | `module4/exercise/` — 4 files, 1212 rows | MAE ↓ | mean = **51.65** | Ridge = **24.75** |
| 2 | MLP S4 | Mushroom Edibility | `sandbox/mushroom_cleaned.csv`, 53 732 rows | accuracy ↑ | majority = **0.5467** | MLP(64,64) = **0.9847** |
| 3 | MLP S5 | Fashion-MNIST warm-up | openml fetch, 12k/5k | accuracy ↑ | majority = **0.0948** | logreg = **0.8396** |
| 4 | MLP S6 | OOD detection by reconstruction | MNIST + Fashion-MNIST, 10% anomalies | ROC AUC ↑ | constant = **0.5000** | PCA-32 = **0.9545** |
| 5 | MLP S7 | Spooky Author ID | `llau/tp10/data.csv`, 19 579 sentences | macro-F1 ↑ | majority = **0.1917** | TF-IDF+logreg = **0.8227** |
| 6 | MLP S8 | Math word problems | `scripts/generate_math_dataset.py`, new seed | exact match ↑ | always 0 = **0.0050** | Qwen2.5-0.5B 3-shot = **0.4550** |
| 7 | MLP S9 | FrozenLake-v1 4×4 slippery | gymnasium preset | mean reward ↑ | random = **0.0120** | Lab 9's Q-learning = **0.7290** |

Notes that make these worth building rather than just listing:

- **#1 has a measured teaching point.** A student who fails to parse the pipe-delimited
  `Greenfield_Grocers` source (UPPERCASE headers, three junk `||||` lines, trailing
  separator) scores **MAE 61.53 — worse than guessing the mean.** That is the whole
  lesson of Session 1, as a number on a leaderboard.
- **#2 is the best ramp in the set**: +35 accuracy points from a linear model (0.6362)
  to a small dense net, trained in seconds on CPU. The student cannot be in doubt about
  whether the session's lesson worked.
- **#3 should ship before Blood Cell (173)**, not instead of it. Stretch rungs measured:
  RandomForest 0.8656, MLP 0.8692, 2-block CNN **0.8970**.
- **#4 was designed twice.** The obvious version (digit 9 as the anomaly) was measured
  and *does not work* — PCA-16/32/64 gave AUC 0.5033 / 0.4897 / 0.5032. Hence the
  cross-dataset version. A synthetic shape-counting variant was also built and rejected
  because a naive white-pixel-area regression beat both intended solutions.
- **#6 uses the same `answer(questions)` contract as GSM8k (165)**, so it is a true dry
  run of the submission path before the hard one. Qwen2.5-0.5B is already on 165's
  model allowlist. Per-category: word problems 0.741, arithmetic 0.528, percentage
  0.125.
- **#7 is literally Lab 9's environment**, and `s9/gymnasium.md` already states "on 4×4
  Frozen Lake the floor is about 0.014" — measurement confirms 0.0120. Note that comp 5,
  which `module9_exercise2.ipynb` still links to, now 404s.

**Two attachment changes, no build required:**

- Link **CartPole-v1 (48)** to `s10-reinforcement-learning-2` and demote it from S9
  (comp 172 is already linked to two modules, so this is supported). Measured: the
  one-line heuristic `0 if (angle + 0.5·angular_velocity) < 0 else 1` scores a **perfect
  500.00 with no training at all** — exactly the "get an accepted submission in the
  first ten minutes" that Lab 10's own speaker notes ask for. LunarLander (43) then
  becomes the real target; its unstated floor is random = **−187.54**.
- **Session 6 → CarRacing (47)** is a genuine curriculum mismatch. Either move 47 to
  S10 and attach build #4, or accept it and say on the page that it is a preview of
  Session 9.

---

## 5. Open decisions — these are the owner's, not the tooling's

1. ~~Are the dates real?~~ **Confirmed real 2026-09-03** — course 14 opens 2026-09-07.
   `course.yaml` still carries the PLACEHOLDER comment above them; worth deleting so the
   next reader does not re-open the question.
2. ~~Flip 179-182 public?~~ **Done 2026-09-03**, dates confirmed real.
3. **The 25% overrun.** Course 14 authors 795 taught minutes against 720; course 15
   authors 2190 against 1800. Lecture is 74-81% everywhere against a promised
   "half lecture, half lab". Either cut ~25% of lecture minutes, or change both course
   descriptions to stop promising a ratio no session comes within 20 points of. The
   second is honest and free; the first is better teaching.
4. **Session 6's competition.** See above.
5. **The two French competition pages** (176, 178) sit inside an all-English course.
6. **The MS2A project tracks are not ready.** Track 1 (169, SuperTuxKart) serves the
   platform's placeholder text and runs on an unreachable `local_vm` engine; track 3
   (171) likewise. The rubric spends 15% of the project grade on "absolute score vs the
   published baseline" for tracks that publish none. This is half the course grade.

---

## 6. Conventions introduced in this pass

Both are plain GitHub-flavoured Markdown — deliberately **not** raw HTML and **not**
`mlarena:` directives. `<details>` renders on the web but `tools/build_slides.py` would
print the tags onto a slide; and the directive machinery has never run in production
(`SELECT count(*) FROM lesson WHERE body_md LIKE '%mlarena:%'` = 0 platform-wide) and
has three defects in its path, one of which silently breaks a module's deck build. See
`docs/plan_course_student_path_gaps.md` §3 for why directives are the right end state
anyway.

**`## Check yourself`** — last slide of every teaching lesson. Two to four items
answerable from that lesson alone, with the answer given so the student can self-mark,
including at least one runnable snippet with its true expected output.

**`## Did you validate this session?`** — last slide of every lab. A GFM task list
(real checkboxes in the web reader) whose every row is objectively verifiable by the
student alone, ending with the two rows that matter:

```markdown
- [ ] My submission is on the leaderboard of <competition> (#<id>)
- [ ] My score beats the baseline: **<metric> <comparator> <number>**
```

"Understood gradient descent" is not a checkbox. "My run log shows the loss decreasing
for 5 consecutive epochs" is.

**Baseline sentences** state the measured ladder, never "beat the baseline". House
style: `competitions/s3-adult-income/overview.md:50-63`.

---

## 7. Re-running the walk

`tools/student_walk.py` reads a published course exactly as an enrolled student does and
writes it to disk, so a review of the *dump* reviews what was delivered rather than what
was intended — the two drift, and the drift is where the student-facing bugs live.

```bash
export MLARENA_STUDENT_API_KEY=mlk_user_...
python tools/student_walk.py dump  --course python-ai-engineering --out ../student_view
python tools/student_walk.py check --course ms2a-machine-learning-practice
```

`dump` writes `student_view/<course>/<module>/<lesson>.md` plus `_index.tsv` and
`_module.json`; a lesson that 404s is recorded as a `.ERROR.txt`, not skipped. `check`
asserts the walk is followable — non-empty bodies, resolvable images, openable
competitions, directive warnings — and exits non-zero with one `where / what / detail`
line per problem.

The dump in `../student_view/` is the 2026-09-02 baseline: diff a fresh dump against it
to see exactly what a republish changed.
