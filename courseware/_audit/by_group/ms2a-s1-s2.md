# Audit findings — ms2a-s1-s2

19 findings

## 1. [blocker / validation / fix in courseware] s1-data-collection + s2-data-preprocessing (all 14 lessons)

**Problem.** There is no way for a student to check that they got anything right. No lesson states an expected output, no lab ships a solution or a reference number, both labs' only deliverable is "a merged PR in your project repository" graded by a human, and the one automated checker the platform actually provides — the attached competition — is never mentioned in any lesson body. I could not verify a single thing I built in either session.

**Evidence.**
```
grep -rin "solution|answer key|check your|quiz|self-check|expected output|you should see|should print" over s1-data-collection/*.md s2-data-preprocessing/*.md returns exactly one hit, and it is the word "blocks" in a sentence about entity resolution. grep -rin "competition|leaderboard|ml-arena|weather|allerg|submit|baseline" over the same 14 lesson bodies returns zero hits (matches occur only in _module.json metadata). Both labs end with "**Deliverable:** a merged PR in your project repository." And as a non-enrolled student even the tick is unavailable: mark_lesson_complete(63) -> AuthenticationError: Not enrolled in any course containing this lesson.
```

**Proposed fix.** Append a "Part F — Check yourself" to each lab that names the module's competition and a number. For lab-2.md: "### Part F — Check yourself (5 min)\nThe module's competition, *Allergies : profils IgE et symptômes cutanés* (id 176), is the same problem shape as this lab: 241 numeric columns with 'not measured' gaps, 3 categoricals, a binary target. Feed its `train.csv` through your `build_pipeline()`, fit a `LogisticRegression`, and submit with `mlarena.connect(...).submit(competition_id=176, files=['submission.csv'])`. A constant submission scores ROC AUC 0.500; the shipped benchmark 0.644; a plain random forest 0.813. If your leak-free pipeline lands under 0.70 something in it is wrong — go back to Part B. Two scores less than 0.03 apart are not distinguishable." For lab-1.md the equivalent Part F must be written against a collection-shaped task (see the separate finding on competition 177).

---

## 2. [blocker / structure / fix in courseware] ms2a-machine-learning-practice/course.yaml, s1-data-collection and s2-data-preprocessing

**Problem.** Neither session fits its stated slot. The course promises "ten 3-hour sessions" and "Every session is half lecture, half lab" — 90 minutes of lecture, 90 of lab. Session 1's six lessons sum to 155 minutes and Session 2's to 185, before the lab. And inside each lab the numbered parts already consume the whole 45-minute budget, leaving the documentation part and the PR write-up with zero time.

**Evidence.**
```
course.yaml estimated_minutes: s1 = 20+30+25+30+25+25 = 155 (+45 lab = 200); s2 = 20+35+30+35+25+40 = 185 (+45 lab = 230). Session budget is 180. Course description: "ten 3-hour sessions… Every session is half lecture, half lab." lab-1.md: Part A 5 + Part B 15 + Part C 15 + Part D 10 = 45 of a stated 45, with "Part E — DATASET.md" (five bullets) and "## Pull request" (four bullets) unbudgeted. lab-2.md: 5+15+10+10+5 = 45, plus an unbudgeted four-bullet PR description.
```

**Proposed fix.** Two edits. (1) In course.yaml, cut Session 2 to fit: fold `scaling-and-normalization` (25 min) into `feature-engineering-and-selection` as a section, and move the KNN/Iterative-imputer half of `missing-values` to the reference module — that brings s2 to 130 lecture minutes. Do the same for s1 by merging `databases` and `apis` into one "Queried sources" lesson (40 min), bringing s1 to 130. (2) In both labs, retime the parts to sum to 35 and give the write-up its own budget: lab-1 A 5 / B 12 / C 10 / D 8 / E 5 / PR 5 = 45; lab-2 A 5 / B 12 / C 8 / D 10 / E 5 / PR 5 = 45.

---

## 3. [major / content / fix in courseware] s1-data-collection/lab-1 (Part C)

**Problem.** The four assertions the lab tells the student to put in checks.py contradict the merge printed immediately above them. `validate="m:1"` declares that the left frame has many rows per key, so `df[KEY].is_unique` is guaranteed to fail. A student who pastes the block gets an AssertionError on line 2 and no way to tell whether their join is wrong or the lab is.

**Evidence.**
```
lab-1.md Part C prints `df = a.merge(b, on=KEY, how="left", validate="m:1")` and then `assert len(df) == len(a)` / `assert df[KEY].is_unique` / … . Running exactly that (1187 patients merged on Blood_Month_sample onto 12 monthly rows): "PASS  assert len(df) == len(a)" / "FAIL  assert df[KEY].is_unique" / "PASS  assert df['value'].between(LO,HI).all()" / "PASS  assert df['ts'].notna().all()".
```

**Proposed fix.** Replace `assert df[KEY].is_unique` with `assert df[ID].is_unique          # ID is the grain of `a` (what one row is), NOT the join key — an m:1 join has many rows per KEY by construction` and add ID to the KEY line above so the two names are visibly different.

---

## 4. [major / content / fix in courseware] s2-data-preprocessing/lab-2 (Part C)

**Problem.** Part C is broken twice over in ten minutes of allotted time. Its single code line raises KeyError because Part A's split put the target in `y_tr`, not `X_tr`. And once repaired, the outcome the lab promises — "The naive column will look better and score worse" — does not happen when you follow the instruction, because `cross_val_score` on a pre-computed column cannot see the leak that produced it. The naive encoder wins the comparison the lab calls "the deliverable".

**Evidence.**
```
Part A: `X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)`. Part C: `naive = X_tr["city"].map(X_tr.groupby("city")[TARGET].mean())`. Run on comp 176's train.csv: "Part C FAILS AS WRITTEN -> KeyError : 'Column not found: skin_symptoms'". Repaired and run as instructed: "naive  cross_val_score(as lab-2 Part C instructs)=0.7455   held-out test AUC=0.6724" / "oof    cross_val_score(as lab-2 Part C instructs)=0.6502   held-out test AUC=0.6750" — the naive column scores 0.09 AUC HIGHER under the instructed measurement.
```

**Proposed fix.** Replace the Part C code block and its closing sentence with: "```python\nnaive_map = y_tr.groupby(X_tr[HIGH_CARD]).mean()          # the target lives in y_tr\nnaive_tr  = X_tr[HIGH_CARD].map(naive_map).fillna(y_tr.mean())\nnaive_te  = X_te[HIGH_CARD].map(naive_map).fillna(y_tr.mean())\n```\nScore each encoding **on `X_te`**, not with `cross_val_score` on the pre-computed column — a column that was already fitted on all of `X_tr` leaks into every fold, so cross-validation will rank the naive encoder *higher*. Seeing that inversion is the point: it is the same mistake the-preprocessing-contract warns about, and it is why the encoder has to live inside the Pipeline."

---

## 5. [major / content / fix in courseware] s2-data-preprocessing/missing-values, the-preprocessing-contract, lab-2

**Problem.** `cat_pipe` is used in three ColumnTransformer code blocks across the session and is never defined in any lesson. No lesson in Session 2 shows one complete, runnable ColumnTransformer — yet "ColumnTransformer with both branches, remainder='drop'" is 20% of the Lab 2 grade. The student has to invent the categorical branch and has nothing to compare it against.

**Evidence.**
```
grep -rn "cat_pipe" s2-data-preprocessing/: the-preprocessing-contract.md:111, missing-values.md:192, lab-2.md:54 — three uses. grep for its assignment returns nothing; only `num_pipe = Pipeline([...])` at missing-values.md:187 is ever constructed. I had to write the categorical branch myself before Part B of the lab would run at all.
```

**Proposed fix.** In missing-values.md, in the "In the pipeline, not in the dataframe" block, insert the missing four lines between `num_pipe` and `pre` so the block is complete and runnable:\n```python\ncat_pipe = Pipeline([\n    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),\n    ("encode", OneHotEncoder(handle_unknown="ignore")),\n])\n```\nand add one sentence after the block: "On the 1,187-row clinical table of Session 2's competition this produces a (949, 483) matrix from 248 columns — check the shape, it is the fastest way to catch a column that fell into neither list."

---

## 6. [major / content / fix in courseware] s2-data-preprocessing/the-preprocessing-contract (and feature-engineering-and-selection)

**Problem.** Session 2's very first lesson scores a model with `cross_val_score(..., cv=5, scoring="roc_auc")`, and Session 2's last lesson uses `RFECV(..., cv=5, scoring="roc_auc")`. Cross-validation, folds and ROC AUC are all introduced in Session 3. A student following the course in order is asked to accept that a number produced by machinery they have not met is the thing that proves their pipeline does not leak — which is the whole argument of the session.

**Evidence.**
```
the-preprocessing-contract.md:124-126 (`from sklearn.model_selection import cross_val_score` / `cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")`) is the first appearance of either term in the course. They are explained only at s3-tabular-models/model-selection-and-validation.md:39-41. Session 2 also uses `Ridge`, `LogisticRegression`, `LassoCV` and `IsolationForest` before s3-tabular-models/the-tabular-landscape introduces any model.
```

**Proposed fix.** Add a four-line box immediately before the cross_val_score block in the-preprocessing-contract.md: "> **Borrowed from Session 3.** `cross_val_score(pipe, X, y, cv=5)` splits the training rows into five parts, fits on four and scores on the fifth, five times, and returns the five scores. `scoring="roc_auc"` is the ranking quality of a binary classifier: 0.5 is a coin flip, 1.0 is perfect, higher is better. You do not need more than that today — Session 3 does it properly. Read the number here as *a score that should go down when you stop leaking*."

---

## 7. [major / content / fix in courseware] s1-data-collection/web-scraping (and s1-data-collection/lab-1 Part D)

**Problem.** The lesson's only cleaner function raises ValueError on the exact string the lesson names two lines below it, and Lab 1 Part D then requires a test asserting that that same string parses correctly. A student who copies the lesson's function and writes the lab's required test gets a red test with no hint of what is missing.

**Evidence.**
```
web-scraping.md: `def to_float(text): return float(text.strip().replace("€", "").replace(",", "."))` followed by 'HTML is presentation. `"1 234,50 €"` is a string containing a non-breaking space…'. lab-1.md Part D: `def test_parse_handles_european_decimals(): """'1 234,50 €' parses to 1234.50."""`. Running the lesson's function: `'1 234,50 €' -> RAISES ValueError could not convert string to float: '1 234.50 '` and `'1\xa0234,50\xa0€' -> RAISES ValueError`.
```

**Proposed fix.** Replace the function with one that actually handles the case the paragraph describes: "```python\ndef to_float(text):\n    cleaned = (text.replace(\"\\u00a0\", \"\")    # non-breaking space, the one you cannot see\n                   .replace(\" \", \"\")\n                   .replace(\"\\u20ac\", \"\")\n                   .replace(\",\", \".\"))\n    return float(cleaned)          # to_float(\"1\\u00a0234,50\\u00a0\\u20ac\") == 1234.50\n```" and keep the following paragraph, changing its last sentence to "The naive version — strip, drop the symbol, swap the comma — raises `ValueError` on that string, which is why the test comes first."

---

## 8. [major / competition / fix in courseware] competition 177 attached to module s1-data-collection

**Problem.** The competition attached to the data-collection session is a live weather-forecasting competition that requires an `Agent` class with a `predict(request)` method returning a (120, 3) forecast panel. Session 1 teaches no modelling of any kind and never mentions the agent contract; the earliest lesson that could support it is Session 3. It is also unusable as a session self-check: the first run produces no score by design and the earliest score matures six hours later, against a lab that is 45 minutes long.

**Evidence.**
```
grep for "class Agent|def predict|\\.fit(" across s1-data-collection/*.md returns no code, only the words "model" in prose. Competition 177 overview: "So your **first run produces no score**: it has issued forecasts but nothing has matured yet… That is expected." module_overview("s1-data-collection") lists it as the module's only competition; no lesson body references it.
```

**Proposed fix.** Either detach 177 from s1-data-collection and attach it to s3-tabular-models (time-series-models), or keep it and use it as a *collection* exercise, which is what the module label already claims: add to lab-1.md "### Part F — A source that never stops (optional)\nThe module's competition (id 177) exposes the same weather feed you would otherwise scrape: `client.download_dataset(177)` gives hourly observations for ~990 cities. Use its 2026 file as your file source in Part A, and make `observed` — the per-hour flag marking hours the upstream feed dropped — the completeness dimension of your `checks.py`. You are collecting from it, not forecasting with it; the forecast is Session 3's problem."

---

## 9. [major / competition / fix in courseware] competition 176 — starter.ipynb (shipped as a dataset file) and the Colab notebook it mirrors

**Problem.** The official starter for the Session 2 competition commits the exact error that Session 2 lists as an automatic deduction. It concatenates train and test, then one-hot encodes the union, so the encoding vocabulary is fitted on test rows. The module's own label for this competition advertises "an encoding contract fitted on train only".

**Evidence.**
```
starter.ipynb cell: `X_all = pd.concat([train.drop(columns=[TARGET]), test], keys=["train", "test"])` then `cat = pd.get_dummies(X_all[CATEG].astype(str), dummy_na=True)`. Measured: "dummy columns fitted over train+test: 118" / "dummy columns that exist ONLY because test rows were in the frame: 3 ['French_Residence_Department_deptCCCC', 'French_Residence_Department_deptIII', 'French_Residence_Department_deptNNN']". lab-2.md "Automatic deductions" lists "a statistic computed over the full dataframe before the split".
```

**Proposed fix.** Rewrite the "2. Préparer les variables" cell to fit on train and apply to test — `enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(train[CATEG].astype(str))`, then transform each frame separately — and add one line of markdown above it: "On ajuste l'encodage sur `train` uniquement puis on l'applique à `test` : `pd.get_dummies` sur la concaténation des deux ferait entrer les catégories du test dans le vocabulaire (3 colonnes ici)." Keep the same measured baseline; the number is unaffected (5-fold CV 0.807 ± 0.017 either way).

---

## 10. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 11. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 12. [major / platform / fix in platform] backend/app/services/lesson_directives.py:113-131, 133-176, 180-196

**Problem.** All three directive resolvers load competitions with `Competition.query.get(...)` and run the leaderboard query with no visibility check, so an ```mlarena:competition id=179``` or ```mlarena:leaderboard id=179``` block in a public course lesson would hand an anonymous reader the hidden competition's name, description and full standings — the exact data /api/competitions/179 refuses. No lesson uses a directive today, so this is latent, but it becomes live the moment the courseware adopts them (see the finding below).

**Evidence.**
```
lesson_directives.py:118-119 `competition = Competition.query.get(competition_id)` with no user_can_see_competition call; :157-163 build_leaderboard_query is called with only competition_id/is_elo_ranked/aggregate; the returned payload at :127-130 includes name, description, is_public.
```

**Proposed fix.** In _resolve_competition and _resolve_submit, replace `Competition.query.get(competition_id)` with a lookup that then checks `user_can_see_competition(competition)` (import from app.views.creator_competition._helpers) and raises DirectiveError('competition <n> is not visible to this reader') on failure — which the non-strict consumption path already turns into a dropped-with-warning block. Gate _resolve_leaderboard the same way before calling build_leaderboard_query.

---

## 13. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 14. [major / content / fix in courseware] s1-data-collection/lab-1 -> s3-tabular-models/lab-3 -> s4-advanced-neural-networks/lab-4

**Problem.** Labs 3 and 4 are written for a supervised classification table of a few thousand rows, but Lab 1 — which chooses the dataset the whole chain runs on — never requires a target column, a task type or a minimum size. A student whose Lab-1 dataset has no label, or 300 rows, hits Lab 3 with nothing to stratify and Lab 4 with nothing to overfit.

**Evidence.**
```
`grep -n "target\|label\|predict" s1-data-collection/lab-1.md` returns **nothing**; Part A asks only for "two of the four" source types and "what is one row of the final table?". Lab 3 Part A hard-codes `METRIC = "roc_auc"` and `return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, and Part B fits `LogisticRegressionCV`. Lab 4 Part A requires "features `float32`, class labels `int64`" and Part E's `test_forward_shape` requires "(batch_size, n_classes)". Nothing between Lab 1 and Lab 4 states this constraint.
```

**Proposed fix.** Add to `lab-1.md` Part A, right after "what is one row of the final table?": "and **what you will predict from it** — one column, present for every row. Labs 3 and 4 fit supervised models on this table, so it needs a target and at least ~2,000 rows. A classification target (two or more classes) is the path the rest of the course is written for; a regression target works but you will substitute `KFold` for `StratifiedKFold` and `neg_root_mean_squared_error` for `roc_auc` throughout. If your two sources cannot give you that, say so now, not in week three."

---

## 15. [major / competition / fix in courseware] ms2a-machine-learning-practice/s1-data-collection (competition 177)

**Problem.** Global Weather Forecast is scored asynchronously — the forecast is recorded and only scored on a later run once the observations arrive. A student in a 45-minute lab submits and gets nothing back, so it cannot be the first exercise even though it is the best-documented competition on the platform.

**Evidence.**
```
Comp 177 overview: "Every six hours your agent receives 48 hours of observations for the whole panel" and "**You are forecasting hours that have not happened yet.** Your forecast is recorded, and scored on a later run once the observations actually arrive." Lab 1 is 45 minutes and never mentions it (grep for ml-arena in s1-data-collection/lab-1.md = 0).
```

**Proposed fix.** Keep 177 as the session's ambitious/ongoing track but attach a same-session ramp first: 'Store Sales — Assemble Four Sources' (file_v1). Data already in-repo at website/public/modules/data-science-practice/module4/exercise/: module4_exercise_train.zip (CityMart_data.csv 415 rows; Greenfield_Grocers_data.csv 401 rows, pipe-delimited, 3 junk lines, UPPERCASE headers), HighStreet_Bazaar_data.json (396 rows, epoch-ms dates), SuperSaver_Outlet_data.xlsx (379 rows, target only). Union 1212 rows, random 80/20, metric MAE lower-better. Measured: train-mean = 51.648; Ridge on all three parsed sources = 24.750 (R2 0.7466); and — the teaching point — dropping Greenfield = 61.531, worse than the trivial baseline.

---

## 16. [minor / baseline / fix in courseware] competition 177 — overview.md and agent_template

**Problem.** Both the overview and the starter agent's docstring tell the student the baselines are reproducible "with test_local.py", but that file is not shipped anywhere a student can reach. The competition's dataset contains only the seven weather CSVs; there is no student route to a competition's env files.

**Evidence.**
```
Overview: "Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026… Reproduce them with `test_local.py`." datasets(177) lists exactly: 2020, 2021, 2022, 2023, 2024, 2025, 2026 (ongoing) — no test_local.py. The file exists only creator-side at /Users/raphaelcousin/reinforcement_learning_challenge/tests-dummy/catalog/global_weather/test_local.py (15,442 bytes).
```

**Proposed fix.** Publish test_local.py as a file on the competition's dataset (`create_dataset` / `upload_dataset_file`), alongside a small replay window so it runs without the 4.4 GB pull. If that is not intended, change both sentences to "Measured by replaying 339 real 6-hourly runs from June to August 2026; the replay harness is creator-side, so treat these as published numbers rather than something you can re-derive locally."

---

## 17. [minor / competition / fix in courseware] competition 177 — overview.md, "Baselines" section

**Problem.** The paragraph under the baseline table refers to a "Samples" column that the table does not have. A student reading for the resolution of the leaderboard is sent to a column that is not there.

**Evidence.**
```
Table header at overview.md line 92: `| Agent | Score | ± | Temp | Wind | Rain BSS |`. Line 104: "The **Samples** column shows how much evidence each score rests on."
```

**Proposed fix.** Replace that sentence with "On the leaderboard itself, the **Samples** column shows how much evidence each agent's score rests on — a score over 300 samples and one over 30,000 are not the same claim."

---

## 18. [minor / competition / fix in courseware] competition 176 — overview.md and starter.ipynb

**Problem.** The competition page for the Session 2 module is entirely in French, including the metric definition, the submission format and the baseline table, while the course description and all fourteen lesson bodies of s1 and s2 are in English. A student who reads the session in English and clicks through to the only self-check the module offers changes language mid-task.

**Evidence.**
```
Course description (_course.json): "A 30-hour applied machine learning course in ten 3-hour sessions…". Competition 176 overview: "## La métrique\n\n**ROC AUC** — la probabilité qu'un patient tiré au hasard parmi ceux qui ont des symptômes cutanés reçoive un score plus élevé…" and "## Repères\n| Approche | ROC AUC |". The module label attaching it is in English: "Allergen Chip Challenge — real clinical open data: missing values, mixed types, an encoding contract fitted on train only".
```

**Proposed fix.** Add an English block immediately after the title, before "## Démarrer en 2 minutes": "> **In English.** Predict the probability that a patient shows skin symptoms from a 241-component IgE profile plus 7 demographic columns. Submit `submission.csv` with columns `patient_id,skin_symptoms` for the 639 test ids. Metric: **ROC AUC, higher is better** — 0.500 = chance, 1.0 = perfect. Reference scores on this exact split: constant 0.500, shipped benchmark 0.644, logistic regression 0.785, random forest 0.813. Sampling noise on 639 patients is ±0.016. The rest of this page is in French."

---

## 19. [minor / content / fix in courseware] competitions 176 and 178, attached to modules s2-data-preprocessing and s7-nlp-1 of course 15

**Problem.** Two competition briefs are written entirely in French inside a course whose description, all 73 lesson bodies and all module summaries are in English. A student following session 2 or session 7 hits a language switch at exactly the moment the task specification and the scoring contract are handed over.

**Evidence.**
```
comp 176 overview (7513 bytes): 'Ouvrir le notebook de départ dans Google Colab', '## Repères', 'Soumission constante | 0,500'. comp 178 overview (6404 bytes): '# Prédire la source', '## La tâche', '## Le score', 'hasard uniforme (1/30) | 0.0327'. Course 15 description and every lesson in student_view/ms2a-machine-learning-practice/ are English.
```

**Proposed fix.** Either translate 176 and 178 overview.md to English (they are the two best-written briefs on the platform — worth keeping) and push with client.set_competition_markdown, or state the language switch explicitly in the module summary for s2-data-preprocessing and s7-nlp-1 in courseware/content/ms2a-machine-learning-practice/course.yaml.

---

