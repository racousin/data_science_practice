# Audit findings — ms2a-s3-s4

22 findings

## 1. [blocker / competition / fix in both] competition 172 (module s3-tabular-models)

**Problem.** Every dataset file for competition 172 is gone from storage, so the download path the overview advertises is dead. This is the only downloadable data anywhere in module s3, and Lab 3 has no other dataset of its own.

**Evidence.**
```
`c.download_dataset(172, ...)` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found`. Checking each of the three files listed by `c.datasets(172)` individually: `X_train.csv id=20 size=2493985 -> HTTP 404`, `X_test.csv id=22 size=624783 -> HTTP 404`, `y_train.csv id=31 size=108474 -> HTTP 404`, each returning `<Code>NoSuchKey</Code><Message>The specified key does not exist.</Message>` from `storage.googleapis.com/.../prod/datasets/6/`. The overview says: "You can also download the data from the **Datasets** tab and work locally." The same three files ARE live at `https://raw.githubusercontent.com/racousin/SCAI-4EUWorkshopAIinMedicineWorkshop/main/Hands-On-Session-1/data/` (HTTP 200, 2493985 / 108474 / 624783 bytes — byte-identical sizes to the dead rows).
```

**Proposed fix.** Re-upload the three files to dataset 6 from the workshop repo (`upload_dataset_file(172, 6, ...)` with a creator key) — the sizes match exactly, so it is a straight restore of objects that were deleted from the bucket. Until that lands, edit the overview's Quickstart section to replace "You can also download the data from the **Datasets** tab and work locally." with "You can also read the three CSVs directly from the workshop repo: `https://raw.githubusercontent.com/racousin/SCAI-4EUWorkshopAIinMedicineWorkshop/main/Hands-On-Session-1/data/{X_train,y_train,X_test}.csv`."

---

## 2. [blocker / content / fix in courseware] s3-tabular-models/lab-3 (Part C) and s3-tabular-models/gradient-boosting-in-practice

**Problem.** Lab 3 Part C requires early stopping with an in-fold eval set inside the mandated `Pipeline`, and 15% of the grade rides on it, but the only early-stopping pattern the module ever teaches is on a bare estimator. Transposing it the obvious way crashes, and the module never shows the step that makes it work.

**Evidence.**
```
Lesson `gradient-boosting-in-practice.md:79-83` teaches `model = lgb.LGBMRegressor(...); model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=[lgb.early_stopping(100), ...])` on a bare estimator. Lab 3 Part C mandates `boosted = Pipeline([("prep", preprocessor), ("model", LGBMClassifier(...))])` and then "add early stopping with the eval set taken **from inside the training fold**". I ran the composition on comp-172 data: `boosted.fit(X_a, y_a, model__eval_set=[(X_b, y_b)], model__callbacks=[lgb.early_stopping(100)])` -> `ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: sex: str, race: str, income: str, dzgroup: str, dzclass: str, ca: str`. The eval set goes to LightGBM raw, unprocessed by the `prep` step. Grading table: "Boosting with early stopping on an in-fold eval set | 15%".
```

**Proposed fix.** Add a section to `gradient-boosting-in-practice.md` after the "Early stopping" block, titled "Early stopping inside a Pipeline", with the working pattern: "`eval_set` is handed straight to the estimator, so it never passes through `prep`. Fit the preprocessor on the training part, then transform both sides yourself:" followed by `prep = clone(preprocessor).fit(X_a)` / `model.fit(prep.transform(X_a), y_a, eval_set=[(prep.transform(X_b), y_b)], callbacks=[lgb.early_stopping(100)])` — and add the sentence "Inside `cross_val_score` there is no hook for this: split each training fold by hand, or use `lgb.cv` with the preprocessing applied per fold." I verified this exact form works (`best_iteration_ = 100`).

---

## 3. [blocker / content / fix in courseware] s3-tabular-models/lab-3 (Part D)

**Problem.** Lab 3's Optuna study cannot run in its stated budget. The 40-trial study it mandates costs roughly an hour of compute for a 12-minute part, and the lab's own snippet sets `timeout=600`, which silently truncates the "fixed budget declared in advance" that the rubric grades at 20%.

**Evidence.**
```
I timed the lab's own configuration on comp-172 data (4931 train rows, 5-fold `StratifiedKFold`, `n_jobs=-1`, `LGBMClassifier(n_estimators=2000)` as in the lab's Part C snippet): **5 TPE trials = 478.8s**, i.e. ~64 minutes extrapolated for the mandated 40. Lab text: "## Part D — The Optuna study (12 min)" and `study.optimize(objective, n_trials=40, timeout=600)`, with the requirement "`n_trials` fixed in advance and stated in the report — not 'until it stopped improving'" and "Optuna study: fixed budget, log-scale spaces, seeded, persisted | 20%". With `timeout=600` the study stops after ~6 trials, so the reported budget is never the budget that ran. Root cause is the finding above: with no in-CV early stopping, every trial fits 5 x 2000 trees.
```

**Proposed fix.** Change the Part D snippet to `study.optimize(objective, n_trials=25, timeout=None)` and add a sentence under Requirements: "Cap `n_estimators` at 400 in the objective and drop it only for the Part E refit — a 2000-tree fit five times per trial is about 95 s of compute per trial, and forty of those do not fit in this lab." Then change the Requirements bullet "`n_trials` fixed in advance and stated in the report" to add: "if your objective takes more than 20 s per trial, halve `n_trials` rather than adding a `timeout` — a timeout makes the budget you report different from the budget that ran."

---

## 4. [blocker / competition / fix in platform] COLLAPSE of findings 54, 68, 80 — dataset_file rows for datasets 6, 7, 8 (competitions 172, 173, 174)

**Problem.** Three agents independently concluded the data for three competitions is gone and the competitions are dead. The bytes are intact in R2; only the storage_backend column was never flipped during the R2 migration, so the backend signs GCS URLs for objects that were deleted after the copy. One root cause, one UPDATE, three blockers retired.

**Evidence.**
```
`gcloud storage ls gs://rlarena-417509-render-experiments-eu/prod/datasets/{6,7,8}/...` — "One or more URLs matched no objects" for all six keys. Same keys in R2 via the backend pod's own credentials: prod/datasets/6/ -> X_test.csv 624783, X_train.csv 2493985, y_train.csv 108474; prod/datasets/7/ -> test_images.npz 7439321, train_images.npz 29722203, y_train.csv 246129; prod/datasets/8/ -> test.csv 1422816, train.csv 5740082. Every size is byte-identical to the DB's file_size_bytes. DB shows these 8 rows at storage_backend='gcs' while datasets 9/10/11 are 'r2'. backend/app/dataset_storage.py:24-34 branches purely on that column.
```

**Proposed fix.** `UPDATE dataset_file SET storage_backend='r2' WHERE dataset_id IN (6,7,8)` — 8 rows — then GET /api/competitions/172/datasets with a user token and fetch one signed URL to confirm 200. Do not regenerate or re-upload anything.

---

## 5. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 6. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 7. [major / platform / fix in platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

**Problem.** 14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

**Evidence.**
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

**Proposed fix.** Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## 8. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 9. [major / competition / fix in courseware] s3-tabular-models/lab-3, s4-advanced-neural-networks/lab-4, competitions 172 and 8

**Problem.** Both modules attach a competition whose label promises exactly the skill the session teaches, and not one of the fourteen lesson bodies mentions it. Both labs send the student to a private project repo graded by a human instead. The one mechanism in the course that could answer "how would I know I got it right" with a number is attached and never used.

**Evidence.**
```
`grep -rniE "competition|ml-arena|172|survival" s3-tabular-models/*.md` returns only generic mentions of "the leaderboard" in `model-selection-and-validation.md`; comp 172 is never named. `grep -rniE "competition|mnist|permut" s4-advanced-neural-networks/*.md` returns exactly one hit, `essential-layers.md:130` — the word "permute" about tensor contiguity. Module labels: "2-Month Survival Prediction — tabular binary classification: gradient boosting judged against a baseline you can defend" and "1 Minute Permuted MNIST — a deep network under a 60-second budget: initialisation, schedules, throughput, profiling". Lab 3 Setup: "Work in the project repository, on a branch, on Lab 2's output." Lab 4 Setup: "Use the processed dataset from Lab 2." Meanwhile the platform supports lesson-embedded competition cards — `backend/app/services/lesson_directives.py:200-205` registers `competition`, `leaderboard` and `submit` handlers — and `grep -rn "mlarena:" courseware/content/` returns nothing for the entire course; `c.lesson(...,"lab-3")` returns `'directives': []`.
```

**Proposed fix.** Add a final part to each lab that uses the attached competition as the external check. In `lab-3.md`, before "## Required tests", insert "## Part F — Submit it (5 min)" containing a ```mlarena:submit id=172``` directive block and the text "Write `submission.csv` (`patient_id,outcome`) from your Part E model and submit once. Your held-out test accuracy and the leaderboard number should agree to within a fold standard deviation; if they do not, say so in the report — that disagreement is the finding." followed by a ```mlarena:leaderboard id=172 top=5``` block. Do the same in `lab-4.md` with `id=8`, or — if comp 8 is genuinely not a Session-4 exercise (see the next finding) — detach it and attach nothing rather than advertise a link that does not exist.

---

## 10. [major / baseline / fix in courseware] competition 8 (module s4-advanced-neural-networks)

**Problem.** Competition 8's overview states no score that counts as having done the lab, gives the student nothing to download, and its official starter path is the Session-3 toolkit rather than anything from Session 4. A student has a random benchmark at 0.0989 and a rank-1 at 0.9990 and no idea what to aim for.

**Evidence.**
```
Overview `### Evaluation Metric` reads in full: "The performance is measured by classification accuracy on the test set (between 0 and 1)." No target number appears anywhere on the page. `c.datasets(8)` returns `{"datasets": []}`. The platform `__benchmark__` row scores 0.0989 (rank 1038/1084); rank 1 is 0.9990. The linked starter (`ml-arena/competition-baseline/permuted_mnist/agent_baseline.ipynb`, HTTP 200) recommends in its last cell `from sklearn.linear_model import LogisticRegression` and, beyond that, "a small MLP (`sklearn.neural_network.MLPClassifier` or PyTorch), PCA to speed up, or subsampling" — none of initialisation, schedules, throughput or profiling, the four things the module label promises. I measured the notebook's own suggestion on permuted MNIST: **0.9267 accuracy**.
```

**Proposed fix.** Add an `## Expected scores` section to competition 8's `overview.md`: "The template agent predicts random labels and scores **0.099** — that is the floor, not a submission. The logistic-regression upgrade in the starter notebook's last cell scores **0.927**. A small MLP trained inside the minute reaches **0.96-0.98**. Below 0.90, something in your agent is wrong; above 0.98 you are competing for the top of the board." And, since accuracy direction is never stated, change "(between 0 and 1)" to "(between 0 and 1; higher is better, and the score you see is the mean over the 10 episodes)."

---

## 11. [major / competition / fix in courseware] competition 8 (module s4-advanced-neural-networks)

**Problem.** Competition 8's overview misstates the resources a submission is actually graded under. A student who sizes a model to the documented 4 GB / 2 cores is sizing to the wrong machine.

**Evidence.**
```
Overview `## Competition Rules / 1. Resource Limits` says "Memory: 4 GB RAM" and "CPU: 2 cores (no GPU)". The live competition record from `c.competition(8)` says `agent_memory_limit: '3Gi'` and `agent_cpu_limit: '3000m'` — 3 GiB, not 4 GB, and 3 cores, not 2. The time limit is the one number that does match (`agent_max_time_per_step_second: 60.0`).
```

**Proposed fix.** In competition 8's `overview.md`, replace "   - Memory: 4 GB RAM\n   - CPU: 2 cores (no GPU)" with "   - Memory: 3 GiB RAM\n   - CPU: 3 cores (no GPU)" so the page matches `agent_memory_limit`/`agent_cpu_limit` on the competition record.

---

## 12. [major / competition / fix in courseware] competition 8 quickstart — ml-arena/competition-baseline/permuted_mnist/agent_baseline.ipynb, cell 11

**Problem.** The competition's own starter notebook recommends a configuration that lands within a few seconds of the 60-second hard budget on the graded hardware, and a timeout scores zero. The first thing a student is told to try is the thing most likely to give them 0.0.

**Evidence.**
```
Notebook cell 11 ("Where to go from here") gives `LogisticRegression(max_iter=200, n_jobs=-1).fit(X, ...)` as the recommended replacement for the random baseline, described as "A strong-yet-fast baseline that fits in 60s on CPU". Measured on permuted MNIST (60k x 784) with BLAS pinned to 3 threads to match `agent_cpu_limit: '3000m'`: fit+predict **47s**, plus the `np.asarray(list)` round-trip the agent contract mandates at **6.3s** = **53.3s of a 60s budget** before any transport, on an M-series laptop. Overview: "Agents that exceed the time limit fail the task" and "Timeouts count as 0% accuracy".
```

**Proposed fix.** Change the notebook's suggested snippet to `LogisticRegression(max_iter=30, n_jobs=-1)` (or subsample to 20k rows) and add one line under it: "Measured: ~0.93 accuracy in about 20 s of the 60 s budget on the grading pod's 3 cores. `max_iter=200` scores the same and takes about 50 s — close enough to the limit that a slow pod scores you 0.0."

---

## 13. [major / content / fix in courseware] s4-advanced-neural-networks/lab-4 (Part A and Setup)

**Problem.** Lab 4 depends on two artefacts from Lab 2 that Lab 2 does not produce. The very first step of the session's lab has nothing to open.

**Evidence.**
```
Lab 4 Setup: "Use the processed dataset from Lab 2." Part A: "Wire a `torch.utils.data.Dataset` over **the Parquet from Lab 2**" and "the train/val/test split from Lab 2, unchanged — do not re-split". Lab 2's Setup file tree lists `src/preprocess/{__init__,pipeline,features}.py`, `tests/test_pipeline.py`, `models/pipeline_<date>.joblib`, `PREPROCESSING.md` — no output Parquet; its Part E is `joblib.dump(pipe, f"models/pipeline_{date.today()}.joblib")`, i.e. a fitted `Pipeline`, and `grep -n "parquet" lab-2.md` returns only line 17, "Lab 1's Parquet file is the **input**". Lab 2 Part A produces a two-way split (`X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)`), never a validation split, and never persists the indices.
```

**Proposed fix.** Either (a) add a Part F to `lab-2.md`: "Write the transformed matrix and the split indices next to the artefact: `pd.DataFrame(pipe.transform(X_tr)).to_parquet('data/processed/train.parquet')`, same for val and test, plus `np.save('data/processed/split_idx.npy', ...)`. Lab 4 reads these." — and change Lab 2 Part A to a three-way split; or (b) change `lab-4.md` Part A to "Load Lab 2's fitted pipeline with `joblib.load('models/pipeline_<date>.joblib')`, apply it to Lab 1's Parquet, and carve a validation split off `X_tr` with the Lab 2 seed — record the split sizes in `RESULTS.md`." Option (a) is better because Lab 4 also forbids re-splitting.

---

## 14. [major / content / fix in courseware] s4-advanced-neural-networks/lab-4 (Part D) and s4-advanced-neural-networks/performance-and-profiling

**Problem.** The mandated throughput harness crashes on the machine the lab explicitly tells students to fall back to, and the mandated AMP block silently no-ops there — so a student can report a "measured speedup" from code that did nothing, for 15% of the grade.

**Evidence.**
```
Lab 4 Part D: "Measure samples/sec with and without: ten warm-up steps, fifty timed steps, `torch.cuda.synchronize()` on both sides." then "No CUDA device? Run **the same measurement** on CPU or MPS and report the result". On a CPU/MPS machine (torch 2.7.1, arm64, `cuda.is_available: False`, `mps: True`) I ran it: `torch.cuda.synchronize()` -> `AssertionError: Torch not compiled with CUDA enabled`. The AMP block from `performance-and-profiling.md` (`torch.amp.GradScaler("cuda")` + `torch.amp.autocast("cuda", dtype=torch.float16)`) ran to completion emitting only `UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.` and `UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling`. Grading table: "Mixed precision with a measured, fully specified speedup | 15%".
```

**Proposed fix.** In `lab-4.md` Part D, replace the bullet with: "**Mixed precision.** Pick your device once — `dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')` — and pass it to `torch.amp.autocast(dev, ...)`; `GradScaler` is only needed for CUDA fp16. Time it with a device-aware barrier (`torch.cuda.synchronize()` on CUDA, `torch.mps.synchronize()` on MPS, nothing on CPU), ten warm-up and fifty timed steps." And add to `performance-and-profiling.md` under the AMP snippet: "`autocast("cuda")` on a machine without CUDA does not raise — it warns once and runs in fp32. If your speedup is exactly 1.00x, check that autocast actually engaged before reporting it."

---

## 15. [major / content / fix in courseware] s1-data-collection/lab-1 -> s3-tabular-models/lab-3 -> s4-advanced-neural-networks/lab-4

**Problem.** Labs 3 and 4 are written for a supervised classification table of a few thousand rows, but Lab 1 — which chooses the dataset the whole chain runs on — never requires a target column, a task type or a minimum size. A student whose Lab-1 dataset has no label, or 300 rows, hits Lab 3 with nothing to stratify and Lab 4 with nothing to overfit.

**Evidence.**
```
`grep -n "target\|label\|predict" s1-data-collection/lab-1.md` returns **nothing**; Part A asks only for "two of the four" source types and "what is one row of the final table?". Lab 3 Part A hard-codes `METRIC = "roc_auc"` and `return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, and Part B fits `LogisticRegressionCV`. Lab 4 Part A requires "features `float32`, class labels `int64`" and Part E's `test_forward_shape` requires "(batch_size, n_classes)". Nothing between Lab 1 and Lab 4 states this constraint.
```

**Proposed fix.** Add to `lab-1.md` Part A, right after "what is one row of the final table?": "and **what you will predict from it** — one column, present for every row. Labs 3 and 4 fit supervised models on this table, so it needs a target and at least ~2,000 rows. A classification target (two or more classes) is the path the rest of the course is written for; a regression target works but you will substitute `KFold` for `StratifiedKFold` and `neg_root_mean_squared_error` for `roc_auc` throughout. If your two sources cannot give you that, say so now, not in week three."

---

## 16. [major / competition / fix in courseware] ms2a-machine-learning-practice/s4-advanced-neural-networks (competition 8)

**Problem.** 1 Minute Permuted MNIST is a wallclock-constrained engineering challenge on a legacy agent contract (train(X_train, y_train) rather than the flex_v1 Agent), scored 0 on timeout. As the first exercise after a session on layers, initialisation and schedules, it confounds 'did I understand the lesson' with 'did I fit inside 60 seconds'.

**Evidence.**
```
s4/_module.json agent_template is the legacy shape: "class Agent: def __init__(self, output_dim: int = 10, seed: int = None) ... def train(self, X_train, y_train)", with agent_max_time_per_step_second 60.0. Overview: "accuracy = 0.0  # Timeout penalty" and "Timeouts count as 0% accuracy". No baseline number is stated anywhere in the overview.
```

**Proposed fix.** Attach 'Mushroom Edibility' (file_v1) as the S4 ramp before 8. Data: website/public/modules/sandbox/mushroom_cleaned.csv, 54,035 rows -> 53,732 after dropping 303 exact duplicates, 8 numeric features, binary class; stratified 80/20; metric accuracy, higher better. Measured: majority class = 0.5467; LogisticRegression + StandardScaler = 0.6362; MLP(64,64) 60 epochs = 0.9847. The +35-point linear-to-dense gap is the session's whole argument, and it runs in seconds on CPU.

---

## 17. [minor / structure / fix in courseware] mlp-project (competition 172, Prediction track)

**Problem.** The Prediction track's leaderboard is already populated by a different cohort, so "Rank in the cohort at the freeze" — 25% of the project grade — will be computed against 82 entries that are not the cohort.

**Evidence.**
```
`c.leaderboard(172)` → 82 rows, 31 distinct usernames, all LastRun between 2026-06-23 17:52 and 2026-07-06 17:11, top score 0.810513 (`nikolay78989`). Course start_date in course.yaml is 2026-09-14. project-grading.md: "| Leaderboard | Rank in the cohort at the freeze | 25% |".
```

**Proposed fix.** Either clone competition 172 as a fresh per-term competition and re-attach the clone to mlp-project (leaving the June board as history), or change the project-grading row to "Rank among MS2A submissions at the freeze" and state in project-brief that the visible board contains an earlier cohort whose 0.81 top score is a reference, not the cohort you are ranked against.

---

## 18. [minor / content / fix in courseware] s3-tabular-models/* and s4-advanced-neural-networks/*

**Problem.** Both sessions introduce libraries that are not in the prerequisite environment and never tell the student to install them. One of them gates 10% of Lab 4's grade.

**Evidence.**
```
`grep -rniE "pip install|uv add|uv pip|requirements" s3-tabular-models/*.md s4-advanced-neural-networks/*.md` returns nothing (the only hit is the unrelated word "Requirements:" in lab-3.md:103). New in these two sessions: `lightgbm`, `optuna`, `shap`, `statsmodels`, `torch`, and TensorBoard. I ran `from torch.utils.tensorboard import SummaryWriter` in a torch-only environment: `ModuleNotFoundError: No module named 'tensorboard'` — `tensorboard` is a separate package, and Lab 4 grades "TensorBoard logs committed: scalars, grad norm, one histogram | 10%".
```

**Proposed fix.** Add a two-line Setup block at the top of `lab-3.md` ("`uv add lightgbm optuna shap statsmodels` — none of these are in the 12h module's environment; commit the lockfile change in the first commit of this branch.") and at the top of `lab-4.md` ("`uv add torch tensorboard` — `tensorboard` is a separate package from `torch`; `from torch.utils.tensorboard import SummaryWriter` raises `ModuleNotFoundError` without it.").

---

## 19. [minor / content / fix in courseware] s4-advanced-neural-networks/data-pipelines-and-training-loop and s4-advanced-neural-networks/lab-4

**Problem.** Lab 4's file layout requires a function that the course uses but never defines, so the student has to invent the one piece of seeding logic the lesson calls out as easy to get wrong.

**Evidence.**
```
`grep -rn "seed_worker" .` over the whole course returns exactly two lines: `data-pipelines-and-training-loop.md:106` (`dl = DataLoader(ds, shuffle=True, generator=g, worker_init_fn=seed_worker)`) and `lab-4.md:21` (`seed.py <- seed_everything + seed_worker`). `grep -rn "def seed_worker" .` returns nothing. The lesson's own prose says the worker RNGs "are otherwise duplicated — every worker applying the same 'random' crop", i.e. this is the failure the function exists to prevent.
```

**Proposed fix.** In `data-pipelines-and-training-loop.md`, extend the Seeding code block with the definition it references: `def seed_worker(worker_id):` / `    s = torch.initial_seed() % 2**32` / `    np.random.seed(s); random.seed(s)` — placed immediately above the `g = torch.Generator()` lines.

---

## 20. [minor / content / fix in courseware] s3-tabular-models/trees-and-ensembles, s3-tabular-models/model-selection-and-validation, s3-tabular-models/lab-3

**Problem.** Three code blocks in s3 reference names that are never imported or defined, so they raise `NameError` for a student who copies them.

**Evidence.**
```
`trees-and-ensembles.md:208` uses `LGBMClassifier()` inside the `StackingClassifier` snippet and `model-selection-and-validation.md:161` uses `LGBMClassifier()` in the Pipeline snippet — `grep -n "import lightgbm\|from lightgbm" ` shows the only lightgbm import in the module is `import lightgbm as lgb` in `gradient-boosting-in-practice.md:79`, a different lesson, and it binds `lgb`, not `LGBMClassifier`. `lab-3.md:124-125` uses `clone(boosted)` and `scorer(final, X_te, y_te)`, neither of which is imported or defined anywhere in the lab.
```

**Proposed fix.** Add `from lightgbm import LGBMClassifier` as the first line of both snippets (trees-and-ensembles.md:206 and model-selection-and-validation.md:160). In lab-3.md Part E, prepend to the snippet `from sklearn.base import clone` and `from sklearn.metrics import get_scorer` and change `test_score = scorer(final, X_te, y_te)` to `test_score = get_scorer(METRIC)(final, X_te, y_te)     # called exactly once`.

---

## 21. [minor / validation / fix in both] competitions 172 and 8 (leaderboard payload)

**Problem.** The central discipline of s3 — never rank two models whose gap is inside the noise — cannot be applied to the leaderboards the two modules attach, because the leaderboard exposes no dispersion at all.

**Evidence.**
```
`model-selection-and-validation.md` teaches "Read the spread, not just the mean" and "If two models differ by less than one fold standard deviation, you have not shown that one is better." But `c.leaderboard(8)` returns `RewardCi95 = None` and `NEpisodes = 0` for all 1084 rows (including the `__benchmark__` row, which has `NumberOfRuns = 25`), and `c.leaderboard(172)` the same for all 82 rows. Comp 8's top three are 0.9990 / 0.9980 / 0.9980, each with `NumberOfRuns = 1` — a 0.001 gap on single runs of a task whose permutations are re-randomised every episode.
```

**Proposed fix.** Platform: populate `RewardCi95` and `NEpisodes` in the leaderboard payload for competitions that run multiple episodes/runs (they are already columns in the response; they are just null). Courseware, meanwhile: add one sentence to `model-selection-and-validation.md` under "Local CV and the leaderboard": "ML-Arena's leaderboard reports a single mean with no interval, so a 0.001 gap between two rows is not evidence of anything — treat adjacent ranks as tied and keep your own fold standard deviation as the ruler."

---

## 22. [minor / content / fix in courseware] ms2a-machine-learning-practice/mlp-project (competitions 172, 169, 171)

**Problem.** The project module reuses competition 172 as its prediction track — the same competition already attached to s3-tabular-models. A student who did Lab 3 has already submitted to it, so the project's prediction track is pre-solved for them and unranked against classmates who arrive fresh.

**Evidence.**
```
mlp-project/_module.json lists 172 ('2-Month Survival Prediction'), 169 (SuperTuxKart) and 171 (The Round); s3-tabular-models/_module.json lists 172 as its sole competition. courseware/README.md also records "**The project competitions are not attached.** ... the competition ids do not exist yet" — which is now stale, three are attached.
```

**Proposed fix.** Give the project prediction track its own competition (a different held-out split, or a different dataset), and leave 172 to s3-tabular-models. If reuse is intended, say so explicitly in mlp-project/project-tracks.md and state that Lab 3 submissions carry over.

---

