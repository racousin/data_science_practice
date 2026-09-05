# Audit findings — pae-s3

11 findings

## 1. [blocker / competition / fix in courseware] python-ai-engineering/s3-data-science-nutshell/lab-3

**Problem.** Lab 3 has no submission step. It never mentions ML-Arena, competition 181, or submission.csv — so even once 181 is public, a student reading the module never arrives at it. The competition package says it is meant to BE the lab's last step, and the lab's data source (fetch_openml, full dataset, cross-validated) is a different split from the competition's (X_train.csv / X_test.csv with te_ ids), so a student who does the lab as written cannot submit without redoing the work.

**Evidence.**
```
`grep -n -i -E "ml-arena|arena|submit|competition|leaderboard|submission" student_view/python-ai-engineering/s3-data-science-nutshell/lab-3.md` → zero matches. lab-3.md line 5: "**Time:** 45 minutes. **Deliverable:** a notebook or script in your repository, plus a written `RESULTS.md`." courseware/competitions/s3-adult-income/overview.md: "This is **Lab 3** with the last step attached." The identical grep over s4-pytorch-nutshell/lab-4.md returns "84:## Part D — Submit to ML-Arena (10 min)" — the pattern exists elsewhere in the course; s3 is the one missing it.
```

**Proposed fix.** Add a "## Part F — Submit (5 min)" section to courseware/content/python-ai-engineering/s3-data-science-nutshell/lab-3.md, inserted before "## Grading": code block `import mlarena` / `client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page` / `client.download_dataset(181, dest_dir="data")   # X_train.csv y_train.csv X_test.csv`, followed by the prose "Refit your Part C (or Part D) pipeline on `data/X_train.csv` — drop the `id` column — against `data/y_train.csv`, predict `data/X_test.csv`, and write `submission.csv` with columns `id,prediction` where 1 means >50K. Then `client.submit(competition_id=181, files=[\"submission.csv\"])`. The leaderboard F1 is the number your RESULTS.md has to explain." Also replace the Deliverable line with "**Deliverable:** a notebook or script in your repository, a written `RESULTS.md`, and a leaderboard entry on competition 181." and add "| Submitted to ML-Arena | 10% |" to the Grading table (taking 5% each from the metric-justification and threshold rows).

---

## 2. [blocker / baseline / fix in courseware] python-ai-engineering/s3-data-science-nutshell/lab-3

**Problem.** The lab gives the student no number to check any result against. The only baseline it names — DummyClassifier(most_frequent) scored with f1 — is exactly 0.000, so Part D's "Beats baseline?" column is trivially "yes" for every model and cannot distinguish a correct pipeline from a leaking or broken one. This contradicts lesson 1 of the same module, which insists success must be a number.

**Evidence.**
```
lab-3.md Part B: `print(cross_val_score(dummy, X, y, cv=cv, scoring="f1").mean())` followed by "Record it. Every later number is measured against this one." I ran it: `DUMMY f1 = 0.0`. the-ml-pipeline.md: "What does success look like, **as a number**?" … "If you cannot answer the last one, you cannot tell whether your model is good." Part D's comparison table ships with every cell empty and no expected values appear anywhere in the module. Measured with the lab's own code on the lab's own dataset (sklearn 1.8.0, StratifiedKFold(5, shuffle=True, random_state=42)): logistic regression 0.657 ± 0.006; HistGradientBoosting (dense-fixed) 0.713 ± 0.006; best threshold 0.35 → 0.689.
```

**Proposed fix.** In lab-3.md, after Part E and before "## Grading", add a section "## Expected numbers" containing: "If your pipeline is leak-free and stratified, 5-fold F1 on this dataset should land close to: | model | F1 (mean ± std) | / | DummyClassifier(most_frequent) | 0.000 (accuracy 0.76) | / | Logistic regression, threshold 0.50 | 0.657 ± 0.006 | / | Logistic regression, best threshold ≈0.35 | 0.689 | / | HistGradientBoosting, threshold 0.50 | 0.713 ± 0.006 |. A logistic-regression F1 much **above** 0.68 at threshold 0.50 means you leaked — go back to the checklist in *Validation & Overfitting*. Much below 0.60 means your preprocessing is dropping a column." And change Part B's "Record it. Every later number is measured against this one." to "Record it. It is the argument for your metric choice, not a bar — an F1 of 0.000 is beaten by anything. The bar is the logistic-regression pipeline you build in Part C."

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

## 6. [major / content / fix in courseware] python-ai-engineering/s3-data-science-nutshell/lab-3 (Part D)

**Problem.** Part D does not run as written. It hands the student a bare `HistGradientBoostingClassifier` import and 10 minutes, clearly intending it to be dropped into the Part C pipeline — but the Part C ColumnTransformer's OneHotEncoder emits sparse output, which HistGradientBoostingClassifier rejects. Under cross_val_score the real error is hidden behind "All the 5 fits failed. It is very likely that your model is misconfigured." The lesson also recommends XGBoost/LightGBM while the lab silently uses a third estimator that was never introduced.

**Evidence.**
```
Ran the lab verbatim on sklearn 1.8.0 (satisfying the `scikit-learn>=1.5` pinned in s1/python-environments.md line 127): `gb = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier())]); gb.fit(X, y)` → `TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array.` Via `cross_val_score` the surfaced message is `ValueError: All the 5 fits failed.` Rebuilding the transformer with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fits in 26 s and gives F1 = 0.713 ± 0.006. models-and-objectives.md: "| Tabular, moderate size | gradient boosting (XGBoost / LightGBM) |".
```

**Proposed fix.** In lab-3.md Part D, replace the bare `from sklearn.ensemble import HistGradientBoostingClassifier` block with that import plus the sentence: "`HistGradientBoostingClassifier` is scikit-learn's own gradient booster — no XGBoost install needed, and XGBoost and LightGBM behave the same way here. Unlike `LogisticRegression` it cannot read sparse input, so rebuild the ColumnTransformer for this pipeline with `OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)`. If you forget, `cross_val_score` reports `All the 5 fits failed`; the real error underneath is `TypeError: Sparse data was passed for X, but dense data is required`." and show the one-argument-different `pre_dense = ColumnTransformer([...])` block in full so it is copy-pasteable.

---

## 7. [major / validation / fix in courseware] python-ai-engineering/s3-data-science-nutshell/validation-and-overfitting

**Problem.** The lesson's canonical train/val/test split snippet is unstratified, and the lab in the same module lists an unstratified split on this imbalanced target as an automatic fail worth 30% of the grade. A student who copies the lesson's own code is punished for it.

**Evidence.**
```
validation-and-overfitting.md, section "## The split": `X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)` — no `stratify`. lab-3.md "## Automatic fail": "Any of these zeroes the leakage component: … an unstratified split on this imbalanced target", and that component is "Zero leakage: all preprocessing inside the Pipeline | 30%". Two sections later the same lesson says of StratifiedKFold "For classification, this should be your default rather than a special case" — making its own preceding snippet the exception it just told you not to make.
```

**Proposed fix.** In validation-and-overfitting.md "## The split", add `stratify=y` to the first call and `stratify=y_temp` to the second, and append after the table: "`stratify=` is not optional on a classification target. Without it, a 15% test split of a 24%-positive dataset can land anywhere from 20% to 28% positive, and your test score moves with the draw rather than with the model. Lab 3 fails an unstratified split for exactly this reason."

---

## 8. [major / content / fix in courseware] python-ai-engineering/s3-data-science-nutshell (the-ml-pipeline, models-and-objectives, evaluation-metrics, validation-and-overfitting)

**Problem.** 140 minutes of lecture contain no self-check of any kind: no worked exercise with a stated answer, no question, no runnable snippet whose expected output is printed. Every code block is an isolated fragment referencing unbound variables (`y_true`, `y_pred`, `X`), so nothing can be executed and compared. A student cannot tell whether they understood evaluation metrics until the lab — and the lab has no expected numbers either, so they never find out.

**Evidence.**
```
`grep -rn -i -E "^#+ .*(exercise|quiz|check|try it|your turn|practice|answer)" s3-data-science-nutshell/*.md` returns only "## Leak 2 — a feature that encodes the answer" and "## Checklist" — no self-check headings anywhere. `the-ml-pipeline.md` has 0 code fences across 11 sections and 30 estimated minutes. Every fence in evaluation-metrics.md is of the form `precision_score(y_true, y_pred)` with `y_true`/`y_pred` never bound to anything.
```

**Proposed fix.** Add one runnable check-yourself block per lecture lesson with the answer printed. The highest-value one, in evaluation-metrics.md immediately after "## The confusion matrix" (I ran it; the numbers are exact): a "## Check yourself" section with `y_true = np.array([1]*5 + [0]*45)   # 5 positives in 50` / `y_pred = np.array([1,1,1,0,0] + [1] + [0]*44)` / `tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()`, then "Work out precision, recall, F1 and accuracy on paper from `tp=3, fp=1, fn=2, tn=44` before you call the metric functions. You should get **precision 0.750, recall 0.600, F1 0.667, accuracy 0.940** — and the gap between 0.940 and 0.667 is the whole lesson." Same shape elsewhere: in models-and-objectives give the `[10,12,100]` vs `[10,12,20]` arrays and ask for MAE and MSE before revealing 26.7 / 2133; in validation-and-overfitting give four unlabelled train/val pairs and ask the student to name each against the "Diagnosing from two numbers" block.

---

## 9. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 10. [major / content / fix in courseware] courseware/content/python-ai-engineering/s1-git-and-packaging/lab-1.md, s2-agentic-coding/lab-2.md, s3-data-science-nutshell/lab-3.md

**Problem.** Labs 1-3 never mention the competition attached to their own module, never mention ML-Arena, and never say anything is submitted. The module card advertises 'textstats correctness' / 'Flesch reading-ease' / 'Adult Census Income' while the lab it belongs to says the deliverable is a GitHub URL. Zero of the ~102 published lessons across both courses uses the ```mlarena:submit``` / ```mlarena:competition``` / ```mlarena:leaderboard``` directive blocks the platform built for exactly this.

**Evidence.**
```
`grep -rn '```mlarena' student_view/` -> 0 matches; SDK `lesson(...)['directives']` is `[]` and `directive_warnings` is `[]` for lab-1. `grep -rli 'ml-arena|mlarena' student_view/python-ai-engineering/` matches only lab-4.md, github-actions.md and three s3 lessons — not lab-1, lab-2 or lab-3. lab-1.md's Grading table lists four criteria, none of them a leaderboard score.
```

**Proposed fix.** Add a closing 'Part F — Submit' section to lab-1.md, lab-2.md and lab-3.md containing the competition's baseline number in prose plus a ```mlarena:submit id=179``` (resp. 180, 181) block, and add the competition row to each lab's Grading table. Do the same for the MS2A lab-N.md files, which attach 17 competitions across ten sessions. Republish with `make publish`.

---

## 11. [minor / content / fix in courseware] python-ai-engineering/s3-data-science-nutshell/the-ml-pipeline and models-and-objectives

**Problem.** Two references point at things a PAIE student does not have: "your project" (PAIE is a 12-hour mise à niveau with four labs and no project) and "these slides" (the lesson is published as a web page; an SDK or MCP reader has no slides at all). The opening line of models-and-objectives also frames the entire lesson as prerequisite material for a different course.

**Evidence.**
```
the-ml-pipeline.md, final line: "Budget accordingly, especially in your project." — "the project" has no antecedent in course 14; the course description promises "a packaged, tested, version-controlled Python project that trains a neural network", not a graded project with a time budget. models-and-objectives.md line 3: "Everything in *MS2A - Machine Learning Practice* — tabular, vision, NLP, reinforcement learning — is a variation on what is on these slides."
```

**Proposed fix.** In the-ml-pipeline.md replace "Budget accordingly, especially in your project." with "Budget accordingly — including in Lab 3, where Parts A and B cost you ten of your minutes before you have trained anything." In models-and-objectives.md replace "is a variation on what is on these slides" with "is a variation on what is on this page", and change the lead-in to "The shared vocabulary. You will meet all of it again in *MS2A - Machine Learning Practice*; the point of this lesson is that none of it is new when you do."

---

