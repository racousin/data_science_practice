# Audit findings — ms2a-s5-s6

23 findings

## 1. [blocker / platform / fix in platform] frontend/src/pages/Competition/View.js:48-95 and frontend/src/components/CompetitionHeader.js:47

**Problem.** When GET /api/competitions/<id> answers 404, View.js never checks response.ok and stores the error body as competitionInfo, while CompetitionHeader catches the axios rejection and returns null. The student who clicks a module competition card lands on a page with no title, no header, empty tabs and no message — a silent blank, not an error.

**Evidence.**
```
`GET /api/competitions/179` with the student token -> 404 {"error":"Not Found"}. View.js:51-64: `const competitionResponse = await fetch(...); const competitionData = await competitionResponse.json(); ... setCompetitionData(competitionData)` — no `.ok` check anywhere in the effect. CompetitionHeader.js:47: `if (!competition) return null;`.
```

**Proposed fix.** In View.js:51-64 check `if (!competitionResponse.ok)` and set an error state; render a dedicated panel for 404 that says the competition is not open to you yet and links back to the course, instead of the tab shell. Same for the /environment fetch at View.js:66-73. In CompetitionHeader.js:18-23 keep the caught error in state and render the same panel rather than `return null`.

---

## 2. [blocker / competition / fix in courseware] ms2a-machine-learning-practice/s6-computer-vision-2 (competition 47)

**Problem.** Session 6 teaches object detection, segmentation, autoencoders, GANs and diffusion, and its only attached competition is CarRacing-v3 — a pixel-input reinforcement-learning control task. There is no label to predict, no metric from the session, and nothing in Lab 6 that produces a submittable artefact for it.

**Evidence.**
```
course.yaml:287-289 attaches competition_id 47 with label "CarRacing-v3 — vision with no label to predict: a convolutional encoder reading raw frames inside a control loop". Lab 6 is titled 'Detect or Generate' and its two branches produce a segmentation/detection model or a VAE — neither is a Gymnasium agent.py. Comp 47 overview states no baseline number at all.
```

**Proposed fix.** Detach 47 from s6-computer-vision-2 (leave it in a reference module if it is wanted at all) and attach the S6 ramp instead: 'Out-of-Distribution Detection by Reconstruction' (file_v1) — train on 20,000 openml MNIST digits, test on 4,500 MNIST + 500 Fashion-MNIST at 10% anomaly rate, submit id,score, metric ROC AUC. Measured: constant score = 0.5000; PCA-32 reconstruction error = 0.9545 (PCA-16 0.9056, PCA-64 0.9854).

---

## 3. [blocker / validation / fix in platform] competition 173 (attached to s5-computer-vision-1) / dataset 7

**Problem.** Every dataset file for Session 5's competition is a dead object. The DB rows exist and the backend happily signs URLs for them, but the GCS objects are gone, so a student cannot obtain a single training image. Session 5's only measurable deliverable is unreachable.

**Evidence.**
```
`c.download_dataset(173, dest_dir=...)` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found`. Fetching each signed URL directly: train_images.npz (declared 29,722,203 B) HTTP 404, y_train.csv (246,129 B) HTTP 404, test_images.npz (7,439,321 B) HTTP 404 — all three `<Code>NoSuchKey</Code> ... No such object: rlarena-417509-render-experiments-eu/prod/datasets/7/...`. The dataset's own description still reads "Public training/test data + starter notebook."
```

**Proposed fix.** Re-upload the three objects to gs://rlarena-417509-render-experiments-eu/prod/datasets/7/{train_images.npz,y_train.csv,test_images.npz}; or upload them to the R2 datasets bucket and set `dataset_file.storage_backend='r2'` for dataset 7 (the column is at modelmanager/modelmanager/competitions.py:548 and backend/app/views/competitions.py:937 already dispatches on it via `dataset_download_url`). Separately, add a `dataset_download_url` health check so a signed URL is never handed to a student for an object that does not exist — the current code path turns a missing object into a 404 the student reads as their own mistake. And either ship the promised starter notebook or change the dataset description from "Public training/test data + starter notebook." to "Public training/test data." — the file list is only train_images.npz, y_train.csv, test_images.npz.

---

## 4. [blocker / baseline / fix in both] competition 173 (attached to s5-computer-vision-1)

**Problem.** The competition states no target score, and the only reference the platform publishes for it scores below random guessing — so a student has no way to know whether they have done the lab, and the one automated signal actively misleads them.

**Evidence.**
```
`c.competition(173)['description']` is one line: "Blood cells — 8 types, 28×28 RGB images". `c.leaderboard(173)`: 26 rows, `Metric` is 'reward' for every row, MeanReward ranges 0.040759–0.959296, and the `__benchmark__` row is rank 26 (last) at MeanReward=0.040759. With 8 classes, chance accuracy is 0.125, so the published benchmark is a third of chance.
```

**Proposed fix.** Rewrite the description to state the contract in numbers, e.g.: "Eight blood-cell types, 28×28 RGB. Metric: accuracy on a held-out split, higher is better, range 0–1. Guessing scores 0.125. A logistic regression on raw pixels scores 0.749 — that is the baseline you must beat. A small CNN trained from scratch reaches ~0.92; the current best is 0.959." Then either re-run the creator benchmark so `__benchmark__` reflects the 0.749 pixel baseline instead of 0.0408, or remove the broken benchmark agent so students are not shown a sub-chance reference. Also fix the metric label: the leaderboard reports `Metric='reward'` for an accuracy number.

---

## 5. [blocker / competition / fix in courseware] s6-computer-vision-2 -> competition 47 (Gymnasium · CarRacing-v3)

**Problem.** Session 6's attached competition is a reinforcement-learning control task, and Session 6 teaches no reinforcement learning. Every prerequisite for it — MDPs, policies, the Gymnasium API, the agent contract — lands in Sessions 9 and 10, three and four sessions later. A student who reaches this competition after Session 6 can do nothing but submit the random-policy template.

**Evidence.**
```
`grep -rniE '\breward\b|\bpolicy\b|\bepisode\b|\baction\b|reinforcement|gymnasium|\bagent\b|control loop' s5-computer-vision-1/*.md s6-computer-vision-2/*.md` returns zero matches across all twelve lessons. The Gymnasium API is taught in s9-reinforcement-learning-1/gymnasium.md. Comp 47's agent_template is the flexkit Gymnasium stub whose `choose_action` body is `return self.action_space.sample()`. It is the only Gymnasium competition in the course not attached to s9 or s10 (48/49 -> s9, 43/65 -> s10). Statistics: `unique_participants: 1`, and that one is the creator.
```

**Proposed fix.** Detach competition 47 from s6-computer-vision-2 in courseware/content/ms2a-machine-learning-practice/course.yaml and attach a vision competition the module actually prepares a student for — competition 8 (1 Minute Permuted MNIST) already exists and is a pure image-classification file_v1 task, or reuse 173 once its data is restored. If CarRacing must stay attached somewhere, move it to s10-reinforcement-learning-2, which teaches the deep-RL machinery (from-tables-to-networks, policy gradients, actor-critic) it needs. If the intent is genuinely 'a convolutional encoder inside a control loop' as the module label claims, that is a Session 10 lesson, not a Session 6 one.

---

## 6. [blocker / competition / fix in platform] COLLAPSE of findings 54, 68, 80 — dataset_file rows for datasets 6, 7, 8 (competitions 172, 173, 174)

**Problem.** Three agents independently concluded the data for three competitions is gone and the competitions are dead. The bytes are intact in R2; only the storage_backend column was never flipped during the R2 migration, so the backend signs GCS URLs for objects that were deleted after the copy. One root cause, one UPDATE, three blockers retired.

**Evidence.**
```
`gcloud storage ls gs://rlarena-417509-render-experiments-eu/prod/datasets/{6,7,8}/...` — "One or more URLs matched no objects" for all six keys. Same keys in R2 via the backend pod's own credentials: prod/datasets/6/ -> X_test.csv 624783, X_train.csv 2493985, y_train.csv 108474; prod/datasets/7/ -> test_images.npz 7439321, train_images.npz 29722203, y_train.csv 246129; prod/datasets/8/ -> test.csv 1422816, train.csv 5740082. Every size is byte-identical to the DB's file_size_bytes. DB shows these 8 rows at storage_backend='gcs' while datasets 9/10/11 are 'r2'. backend/app/dataset_storage.py:24-34 branches purely on that column.
```

**Proposed fix.** `UPDATE dataset_file SET storage_backend='r2' WHERE dataset_id IN (6,7,8)` — 8 rows — then GET /api/competitions/172/datasets with a user token and fetch one signed URL to confirm 200. Do not regenerate or re-upload anything.

---

## 7. [major / validation / fix in courseware] courseware/tools/build_competitions.py:121-140 (verify_benchmark) and :173 (set_competition_markdown)

**Problem.** The build's one guarantee is opt-in and does not cover the page. `verify_benchmark` returns silently when `benchmark_expected_score` is absent, and nothing checks that the number appears in overview.md — which is why 179 and 180 have a machine-verified 1.0 on the server that never reaches the student.

**Evidence.**
```
build_competitions.py:123-126: `expected = cfg.get("benchmark_expected_score")` / `if expected is None: return`. Line 173 publishes the page unconditionally: `client.set_competition_markdown(cid, (pkg_dir / "overview.md").read_text())`, and it runs *before* the benchmark at :242. Live proof the number exists but is unpublished: benchmark_status(179) -> score=[1.0], benchmark_status(180) -> score=[1.0], yet neither overview.md contains the string "1.0".
```

**Proposed fix.** Two edits. (1) Replace `if expected is None: return` with `raise RuntimeError(f"{cfg['name']}: config.py declares no benchmark_expected_score")` so the guarantee is mandatory, not opt-in. (2) Add `verify_overview_contract(cfg, overview_text)` called immediately before line 173: parse the Baseline block, and refuse to publish unless the metric name equals `cfg["metric"]`, the direction word is present, and the block's Baseline/Reference/Passing numbers equal `cfg["baseline_score"]` / `cfg["benchmark_expected_score"]` / `cfg["target_score"]` within `cfg["benchmark_score_tol"]`. Add `baseline_score`, `target_score` and `metric_direction` as required keys to all four config.py files. The failure message should read like the existing one: "the env does not grade what the package claims" becomes "the page does not state what the package grades".

---

## 8. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 9. [major / platform / fix in platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

**Problem.** 14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

**Evidence.**
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

**Proposed fix.** Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## 10. [major / baseline / fix in courseware] competitions 8 173 174 48 165 171 (measured reference already on the board, absent from the page)

**Problem.** For six competitions the baseline number the overview omits is already sitting on the public leaderboard as an unlabelled `__benchmark__` or starter row. The information exists; it was simply never written down.

**Evidence.**
```
Live boards, student token: 173 `rf-pixel-baseline` = 0.748911 (rank 23/26) vs an overview that says only "The starter baseline flattens the pixels into a random forest"; 174 `tfidf-logreg-starter` = 0.479719 (rank 9/27) vs "a TF-IDF + linear model is the starter baseline"; 8 `random0` = 0.0958 and `__benchmark__` = 0.0989 vs no number at all; 48 `qa-cartpole-random` = 23.3 ± 9.81 and `qa-cartpole-linear` = 500.0 vs no number; 171 `__benchmark__` = 27,100,000 vs only a $65M ceiling; 165 `__benchmark__` = 0.0 vs "see Agent.py for a SmolLM3-3B baseline".
```

**Proposed fix.** Write these six numbers into the Baseline block of each overview, quoting the leaderboard row name so a student can verify it: e.g. for 173, "Baseline: `rf-pixel-baseline` — flattened pixels into a random forest — scores 0.749 F1-macro. Higher is better."; for 48, "Baseline: a random policy scores 23.3 ± 9.8 mean episode reward; a linear policy scores 500.0, the environment's ceiling." Then register the same numbers in the external yaml from the verify-external finding so CI keeps them true.

---

## 11. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 12. [major / baseline / fix in both] competitions 8, 47, 48, 49, 43, 65, 165, 173, 174, 169, 170, 171

**Problem.** Twelve of the seventeen attached competitions state no numeric floor. Each says only that a Colab notebook holds a 'random-action baseline'. A student cannot tell a working submission from a broken one — the first CartPole agent scoring 21 has no way to know 21 IS the random floor.

**Evidence.**
```
Grepping every overview for a number: comp 48 (CartPole) and 49 (MountainCar) and 47 (CarRacing) and 170 contain only "a minimal **random-action** baseline you can deploy in a couple of minutes"; comp 173 says "The starter baseline flattens the pixels into a random forest" with no score; comp 174 says "A TF-IDF + linear model ... is the starter baseline and is famously hard to beat" with no score; comp 43 quotes "solved around 200" but never the floor. By contrast comp 177 carries a six-row measured table with error bars, 176 states 0.500 / 0.813, 172 states "always guessing alive scores ~0.59. That is the bar to beat."
```

**Proposed fix.** Add a '## Baselines' table to each overview in the exact form comp 177 already uses. Numbers I measured and that can be pasted in today: CartPole-v1 random = 20.98 (sd 10.89, 300 eps) and heuristic `0 if angle+0.5*angvel < 0 else 1` = 500.00; LunarLander-v3 random = -187.54 (sd 115.01, 200 eps); FrozenLake-v1 4x4 slippery random = 0.0120. For 173 and 174, run the shipped starter notebook once and paste its score. Enforce it by adding a `baselines` block to config.py that build_competitions.py refuses to publish without.

---

## 13. [major / competition / fix in courseware] ms2a-machine-learning-practice/s5-computer-vision-1 (competition 173)

**Problem.** Blood Cell Classification — 8 classes of 28x28 RGB, ~13.7k train images in .npz, ranked on F1-macro over imbalanced cell types — is the first thing a student does after their first-ever CV lecture, with no stated baseline. A student who scores badly cannot tell whether they misunderstood convolution or mishandled the npz/imbalance.

**Evidence.**
```
course.yaml:243-245 attaches 173 as the sole S5 competition. Its overview: "**Metric:** **F1-macro** — the unweighted mean of per-class F1" and "a model that aces the common classes and ignores the rare ones scores poorly on F1-macro", with no number anywhere. Lab 5 itself insists on a from-scratch baseline first — "They will all want to skip the baseline. Do not let them" — but the competition gives them no such rung.
```

**Proposed fix.** Build 'Fashion-MNIST Warm-up' (file_v1) as the S5 ramp and attach it before 173: prepare_data.py fetches openml Fashion-MNIST exactly as s4-mnist-warmup/prepare_data.py fetches mnist_784; 12,000 train / 5,000 test, metric accuracy with macro-F1 second. Measured: majority class = acc 0.0948 / macro-F1 0.0173; logistic regression on raw pixels = acc 0.8396 / macro-F1 0.8371 (9 s); 2-block CNN, 421,834 params, 12 CPU epochs = acc 0.8970 / macro-F1 0.8944. Also add the measured starter score to 173's overview.

---

## 14. [major / content / fix in courseware] 92 of 102 lesson bodies across both courses (e.g. s5-computer-vision-1/training-cnns, s7-nlp-1/lab-7)

**Problem.** Teacher speaker notes are embedded as HTML comments in the published lesson bodies. The web renderer hides them, but the SDK and MCP hand students the raw markdown, so students read the instructor's private classroom management notes about them.

**Evidence.**
```
`grep -rl '<!-- notes:' --include='*.md' student_view/ | wc -l` -> 92 of 102 lessons. s5-computer-vision-1/training-cnns.md:8 served to a student token: "<!-- notes: 30 minutes. Show a batch of augmented images on screen before explaining any of it — half the room will spot an augmentation that destroys their own label. -->". s7-nlp-1/lab-7.md:9: "<!-- notes: They will want to start with the transformer. Do not let them ... -->".
```

**Proposed fix.** Strip `<!-- notes: ... -->` blocks in courseware/tools/publish_mlarena.py before the body is sent to the server (the deck builder already consumes them for the PPTX notes pane, so nothing is lost), then republish both courses. Do not rely on the web renderer hiding them — SDK and MCP consumers are first-class per the parity rule.

---

## 15. [major / baseline / fix in both] competition 47 (attached to s6-computer-vision-2)

**Problem.** The competition page is three lines of copied upstream boilerplate. It states no target, no metric direction, no range, and the leaderboard labels a Gymnasium reward as 'accuracy' — so a student cannot tell whether -33.876 is good, bad, or which way the number should move.

**Evidence.**
```
Full `c.competition(47)['description']`: "This environment is part of the Box2D environments which contains general information about the environment.\n\nSource: https://gymnasium.farama.org/environments/box2d/car_racing/". `c.datasets(47)` -> `{"datasets": []}`. `c.leaderboard(47)` -> one row: AgentName `__benchmark__`, MeanReward -33.87589, Metric `accuracy`, NumberOfRuns 1, RewardCi95 None.
```

**Proposed fix.** Replace the description with the contract in numbers: "CarRacing-v3. Your agent sees a 96×96×3 RGB frame and returns a 3-vector (steer, gas, brake). Metric: mean episode return over N episodes, higher is better. A random policy scores about -34 (that is the __benchmark__ row). Staying on the track for a full lap scores roughly 900. Anything above 0 means you are driving rather than spinning; 300+ is a pass." Also fix the metric label — `evaluation.metric` for comp 47 is 'accuracy' on a reward-valued column, which is the platform-wide mislabelling showing up where it does concrete harm.

---

## 16. [major / structure / fix in courseware] s5-computer-vision-1/*.md and s6-computer-vision-2/*.md (all 12 lessons)

**Problem.** No lesson in either module mentions its attached competition, and the labs send the student somewhere else entirely. The competitions are attached at module level and are therefore invisible from inside the reading flow — the only place a student would ever look for 'what do I submit'. Meanwhile the platform already has a directive mechanism built for exactly this, and not one lesson in either course uses it.

**Evidence.**
```
`grep -rni 'blood|carracing|competition|leaderboard|ml-arena' s5-computer-vision-1/*.md s6-computer-vision-2/*.md` -> zero hits in lesson bodies (only in _module.json metadata). lab-5.md Part A instead says "your own project's images if your track is a vision track, otherwise a subset of a `torchvision.datasets` set — `Flowers102`, `OxfordIIITPet`, `EuroSAT`, `FGVCAircraft`". `c.lesson(...)` returns `directives=[] warnings=[]` for all 12 lessons, and `grep -rn '```mlarena:' student_view/ courseware/content/` returns nothing across both entire courses — while backend/app/services/lesson_directives.py:202-206 registers three working handlers: `competition`, `leaderboard`, `submit`.
```

**Proposed fix.** Add a directive block to each lab. In courseware/content/ms2a-machine-learning-practice/s5-computer-vision-1/lab-5.md, after the '## Setup' block, insert a new '## Part F — Submit (5 min)' section containing a ```mlarena:competition id=173``` fence, a ```mlarena:leaderboard id=173 top=5``` fence, and the sentence "Retrain your best model on the full training split, write `image_id,label` for every row of test_images.npz, and submit. You have done the lab when you beat 0.749 — the pixel-baseline row on that board." Do the same in lab-6.md once s6 has a competition it can actually prepare a student for. This is a two-line change per lab and it is the only thing that turns either module's competition from an orphan into the session's proof of work.

---

## 17. [major / content / fix in courseware] s5-computer-vision-1/lab-5

**Problem.** The lab is not survivable in its stated time on the hardware it says it needs. Parts B, C and D specify six full training runs and budget 30 minutes for them; measured on a laptop CPU that is 169 minutes of pure compute before a single line of the code, the four tests, the confusion matrix or the PR is written. The lab's own speaker note underestimates by 5x, so even the instructor's fallback plan does not hold.

**Evidence.**
```
Lab spec: Part B "the same number of epochs as Part C" (=13), Part C "3 epochs at lr=1e-3" then "unfreeze, 8–10 epochs", Part D "Fine-tune the same backbone three times" (3 x 13 epochs). Measured (torch 2.7.1, CPU, 4 threads, batch 32 @224, quiescent machine): resnet18 full fine-tune 4.58 s/batch, frozen head 1.59 s/batch, 3-block baseline 1.91 s/batch. At 6 classes x 300 images (mid-range of the lab's own "4 to 10 classes, 100 to 500 images per class") = 1260 train images = 40 batches/epoch: Part B 16.6 min + Part C1 3.2 + Part C2 30.5 + Part D 3x39.7 = 169.3 min vs a 30-min budget. At the smallest legal size (4 classes x 100) it is still 38 min. The speaker note says "ResNet-18 at 128 pixels fine-tunes on CPU in about ten minutes if not [a GPU]"; measured at 128px it is 1.91 s/batch = 16.5 min for one 13-epoch run, and the lab needs four such runs = 66 min.
```

**Proposed fix.** Two edits to courseware/content/ms2a-machine-learning-practice/s5-computer-vision-1/lab-5.md. (1) Under '## Setup', add: "**This lab needs a GPU.** On a free Colab T4 the six runs below take about twelve minutes. On a laptop CPU they take about three hours — measured at 4.6 s/batch for a resnet18 fine-tune, batch 32 at 224px. Without a GPU, run every model at 128px with the backbone frozen and say so in the PR." (2) Cut Part D from three full fine-tunes to three *frozen-backbone, 3-epoch* runs (measured 3.2 min each) — the ablation still answers 'which augmentation is worse than basic', which is the only thing Part D grades. Also correct the speaker note's "about ten minutes" to "about seventeen minutes per run at 128 pixels, and four runs are needed".

---

## 18. [major / content / fix in courseware] s6-computer-vision-2/diffusion-models

**Problem.** The lesson's central training-objective code block crashes when run as written, and it is the block a student following Lab 6 branch B's diffusion option copies. Worse, on some batch shapes it does not crash — it silently applies the wrong noise level to every image, which is undebuggable for a beginner.

**Evidence.**
```
Running the block verbatim (`t = torch.randint(0, T, (x0.size(0),), device=x0.device)` then `xt = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * noise`) with x0 of shape (8,3,32,32): `RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 3`. With x0 of shape (8,3,32,8) — where B happens to equal W — the same line runs with no error and produces a tensor of the right shape and completely wrong per-image noise levels. The lesson presents this indexing twice and the prose between them says "Sample a random `t` per image in the batch, corrupt in one operation, done.", which is precisely the case that fails.
```

**Proposed fix.** In courseware/content/ms2a-machine-learning-practice/s6-computer-vision-2/diffusion-models.md, in both the '## Jumping to any step' and '## The training objective' code blocks, replace\n    xt = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * noise\nwith\n    ab = alpha_bar[t].view(-1, 1, 1, 1)      # (B,) -> (B,1,1,1), broadcasts over (B,C,H,W)\n    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise\nand add one sentence after the second block: "The `.view(-1, 1, 1, 1)` is not cosmetic: without it a per-image `t` broadcasts against the width axis, which raises on most shapes and silently corrupts the batch when B happens to equal W."

---

## 19. [major / content / fix in courseware] s5-computer-vision-1/lab-5

**Problem.** The lab's Part C instruction and its Part E required test contradict each other: the object Part C tells you to use as the eval pipeline cannot be introspected the way the test demands, so the test as specified throws before it can assert anything. It is worth 15% of the grade and the lab calls it "the one that catches the bug this session is built around".

**Evidence.**
```
Part C: "Use `weights.transforms()` for the evaluation pipeline. Do not hard-code the ImageNet normalization constants." Part E: `test_eval_transform_is_deterministic` — "No transform in the eval pipeline has a class name starting with 'Random', and applying it twice to one image gives identical tensors." Checked: `ResNet18_Weights.IMAGENET1K_V1.transforms()` returns type `ImageClassification`; `hasattr(tf, 'transforms')` is False; iterating it raises `TypeError: 'ImageClassification' object is not iterable`; `[type(t).__name__ for t in tf.transforms]` raises `AttributeError: 'ImageClassification' object has no attribute 'transforms'`. The same check on a hand-built `T.Compose` works fine (['Resize','CenterCrop','ToTensor']).
```

**Proposed fix.** In courseware/content/ms2a-machine-learning-practice/s5-computer-vision-1/lab-5.md, replace the Part E docstring\n    """No transform in the eval pipeline has a class name starting with\n    'Random', and applying it twice to one image gives identical tensors."""\nwith\n    """Applying the eval transform twice to one image gives identical tensors.\n    `weights.transforms()` returns an ImageClassification object, not a Compose,\n    so test the behaviour, not the class names."""\nThe behavioural half is the real check and it passes against both a Compose and an ImageClassification.

---

## 20. [major / validation / fix in courseware] s5-computer-vision-1 (5 lessons) + s6-computer-vision-2 (5 lessons)

**Problem.** Ten of the twelve lessons give the student nothing to check their own understanding against. They end in a recommendation table — advice a student can agree with without having understood anything. The two lessons that do provide a check show exactly the pattern that works, so the fix is to copy it, not to invent it.

**Evidence.**
```
Lessons that end in a 'what to actually use / what to pick' table with no verifiable question: pooling-and-architectures ('What to pick in 2026'), training-cnns ('A recipe with numbers'), transfer-learning ('The default recipe'), object-detection ('What to actually use'), segmentation ('What to actually use'), autoencoders-and-vaes ('Where VAEs actually live now'), gans ('Where GANs stand now'), diffusion-models ('Choosing'). The two exceptions: convolution.md gives a worked example whose answer (9, 24, 6, 3) and a five-row output-size table (5, 3, 4, 32, 112) a student can independently reproduce — I checked all of them by hand and they are correct; object-detection.md says "Test it on identical boxes (1.0) and disjoint boxes (0.0); that two-line test catches the bug in seconds" — I ran it, and without the `max(0, ...)` clamps a disjoint pair does return exactly 1.0 as the lesson claims.
```

**Proposed fix.** Append a three-line '## Check yourself' block with the answer to each of the eight lessons above, in the style convolution.md already uses. Concretely: pooling-and-architectures — "A 224x224 input through three 2x2 max-pools at stride 2, then AdaptiveAvgPool2d(1) on 256 channels: what is the shape after each step? (224->112->56->28, then (B,256,1,1).)"; training-cnns — "You train with RandomResizedCrop in both pipelines and validation accuracy is 4 points below training from epoch 1 with no overfitting curve. Which of the two bugs in this lesson is it? (The eval transform is non-deterministic — the train/eval split section.)"; transfer-learning — "Frozen backbone, model.train(), validation far below training from epoch 1. Which buffer is drifting? (BatchNorm running_mean/running_var — call freeze_bn.)"; segmentation — "Your model reports 98% pixel accuracy and 0.02 mIoU. What did it predict? (Background everywhere — use CE + Dice.)"; autoencoders-and-vaes — "You switch the recon term to reduction='mean' and samples become the dataset mean. By what factor did the KL term's relative weight change on 3x64x64 inputs? (12288x.)"; gans — "d_loss falls to 0 and g_loss explodes. Who is winning and what do you lower? (D; lower lr_D, add label smoothing.)"; diffusion-models — "A GAN samples in one pass, DDPM in 1000, DPM-Solver++ in 20-30. Which of the three needs retraining to switch to? (None — the sampler is a one-line change.)"; object-detection already has one.

---

## 21. [minor / content / fix in courseware] s6-computer-vision-2/gans

**Problem.** The conditional-GAN discriminator line does not run for any realistic embedding dimension — `expand_as` cannot broadcast a (B, E) embedding onto a (B, 1, H, W) spatial map.

**Evidence.**
```
Running the block verbatim with B=16, E=32, H=W=64: the generator line `z_y = torch.cat([z, embed(y)], dim=1)` is fine (shape (16,132)), but `d_in = torch.cat([x, embed(y).expand_as(x[:, :1])], dim=1)` raises `RuntimeError: The expanded size of the tensor (64) must match the existing size (32) at non-singleton dimension 3. Target sizes: [16, 1, 64, 64]. Tensor sizes: [16, 32]`.
```

**Proposed fix.** In courseware/content/ms2a-machine-learning-practice/s6-computer-vision-2/gans.md, in the '## Conditional GANs' block, replace\n    d_in = torch.cat([x, embed(y).expand_as(x[:, :1])], dim=1)\nwith\n    ymap = embed(y)[..., None, None].expand(-1, -1, x.size(2), x.size(3))   # (B, E, H, W)\n    d_in = torch.cat([x, ymap], dim=1)                                      # (B, 3+E, H, W)

---

## 22. [minor / content / fix in courseware] s6-computer-vision-2/segmentation

**Problem.** The palette-PNG warning states the failure backwards. As written it tells a student that plainly opening a palette mask gives RGB triplets — the opposite of what PIL does — so a student who hits the real bug (a stray `.convert("RGB")` in their loader) will look in the wrong place.

**Evidence.**
```
Lesson text: "Palette PNGs: opening one without `.convert(\"P\")` awareness gives RGB triplets instead of class indices." Tested by writing a mode-'P' PNG with classes 0-7 and reopening it: `Image.open(...).mode` is `'P'`; `np.array(...)` gives `ndim=2, dtype=uint8, unique=[0,1,2,3,4,5,6,7]` and the lesson's own following assert (`mask.ndim == 2 and mask.dtype == np.uint8`) passes. It is `np.array(Image.open(...).convert("RGB"))` that gives `ndim=3`.
```

**Proposed fix.** In courseware/content/ms2a-machine-learning-practice/s6-computer-vision-2/segmentation.md, replace "Palette PNGs: opening one without `.convert(\"P\")` awareness gives RGB triplets instead of class indices." with "Palette PNGs: `Image.open` already gives you mode `P`, and `np.array` on it gives class indices — so the assert below passes. The bug is the `.convert(\"RGB\")` you copied from your image loader into your mask loader: that turns class 3 into a colour triplet and the assert fires. Masks are never converted."

---

## 23. [minor / content / fix in courseware] s6-computer-vision-2/lab-6

**Problem.** Lab 6 Part A offers an option its own module proves is impossible in the time given, and lists a dataset that is not obtainable through the interface the labs otherwise use.

**Evidence.**
```
lab-6.md: "## Part A — Choose a branch and get the data (5 min) ... **Branch A** ... Penn-Fudan pedestrians, Oxford-IIIT Pet masks, a Roboflow public set, or 100–300 images you annotate yourself." object-detection.md, in the same module: "A box takes 5–15 seconds of human time" — so 100 images at one box each is 8–25 minutes and 300 images is up to 75 minutes, against a 5-minute budget. Checked torchvision 0.22.1: `hasattr(datasets, 'PennFudan')` and `'PennFudanPed'` are both False (OxfordIIITPet, EuroSAT, Flowers102, FGVCAircraft are all True), so Penn-Fudan needs a manual download from an external page. Roboflow needs an account and an API key.
```

**Proposed fix.** In courseware/content/ms2a-machine-learning-practice/s6-computer-vision-2/lab-6.md, replace the Branch A dataset sentence with: "A small annotated set you can obtain in one line: `OxfordIIITPet(root=..., target_types='segmentation', download=True)` — take 2–5 breeds and 100–300 images. Do not annotate your own images here; at the 5–15 seconds per box from the detection lesson that is an hour, not five minutes. If you want your own data, annotate it before the session." Also add the missing dependency line to '## Setup': "Branch A needs `segmentation_models_pytorch` (which pulls `timm`) or `ultralytics`; install before the session — `smp.Unet('resnet18', encoder_weights='imagenet')` downloads weights on first call and will not work offline."

---

