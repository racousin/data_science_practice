# Audit findings — ms2a-project

32 findings

## 1. [blocker / content / fix in courseware] competitions 169 (SuperTuxKart Grand Prix) and 170 (Bitcoin Intraday Trading)

**Problem.** Both overviews are the platform's default placeholder text. There is no task description, metric, direction, or baseline — and 169 is one of the three graded project tracks of course 15.

**Evidence.**
```
GET /api/competition_asset/169/markdown/overview returns exactly "**Competition Overview**\n\nWrite your markdown here." plus the auto-appended `<!-- mlarena:quickstart -->` block (592 bytes total). Identical for 170 (634 bytes). The placeholder is written by the GET route itself: backend/app/views/competition_asset.py:14 `DEFAULT_MARKDOWN_CONTENT = "**Competition Overview**\n\nWrite your markdown here."` and :31-34 writes it when overview.md is missing.
```

**Proposed fix.** Write a real overview.md for both, following the 177/178 template, and publish it with `client.set_competition_markdown(169, ...)`. Minimum content: the task, the observation/action shape, the metric with its direction, and the Baseline block defined in the contract finding below. For 169 the numbers to quote already exist on the board (`__benchmark__` = 946.41, `baseline-kart-test` = 1777.24) and the page must additionally state that ranking is by ELO. For 170 the reward is fractional return at ~1e-5 scale, so the page must state the scale and the board's FrontendPrecision must be raised from 2.

---

## 2. [blocker / baseline / fix in courseware] ms2a-machine-learning-practice/mlp-project/project-grading.md and project-tracks.md

**Problem.** The project rubric spends 15% of the project grade on a published baseline that does not exist for two of the three tracks, and project-tracks.md explicitly defers the baseline to the competition page — which for 169 is a placeholder and for 171 gives only a ceiling.

**Evidence.**
```
project-grading.md:112 `| Leaderboard | Absolute score vs the published baseline | 15% |`. project-tracks.md (last section): "The **specific competitions** for this term — the datasets, the environments, the judge, **the baselines to beat** — are announced separately and attached to this module on ML-Arena." project-brief.md: "scored on a public leaderboard against your cohort and against a published baseline". The three competitions attached to mlp-project are 172, 169, 171 (student_view/ms2a-machine-learning-practice/mlp-project/_module.json). 169's overview is the placeholder; 171's only number is "There's ~$65M of capital on the table … Nobody raises all of it".
```

**Proposed fix.** Publish a baseline number for 169 and 171 before the term opens. For 171 the number already exists: `__benchmark__` = 27,100,000 on the live board — write "Baseline: the reference pitch raises $27.1M. Higher is better." For 169, deploy the shipped Colab heuristic agent under a documented name and quote its score (a candidate exists already: `baseline-kart-test` = 1777.24). If a track cannot carry a baseline, drop the 15% row from project-grading.md and redistribute it to the cohort-rank row rather than grading against a missing artifact.

---

## 3. [blocker / competition / fix in both] competition 169 (SuperTuxKart Grand Prix) and 170 (Bitcoin Intraday), attached to modules mlp-project and mlp-reference of course 15

**Problem.** Two competitions still carry the platform's placeholder overview. 169 is one of the three MS2A project tracks — half the course grade — and its brief for a student is the literal string 'Write your markdown here.' There is no task description, no submission contract, no metric, no baseline.

**Evidence.**
```
`GET /api/competition_asset/169/markdown/overview` -> 592 bytes, beginning `**Competition Overview**\n\nWrite your markdown here.` followed by an auto-appended `<!-- mlarena:quickstart -->` Colab block. Same for 170 (634 bytes). The constant is at backend/app/views/competition_asset.py:14 `DEFAULT_MARKDOWN_CONTENT = "**Competition Overview**\n\nWrite your markdown here."`. Contrast: comp 176's overview is 7513 bytes with a measured 'Repères' table.
```

**Proposed fix.** Courseware: write real overview.md for 169 and 170 (task, action/observation contract, metric, and a measured baseline table like comp 178's) and push with client.set_competition_markdown. Platform: in backend/app/views/creator_competition/settings start_competition, refuse to start (or return a blocking `warnings` entry) when the stored overview.md still equals DEFAULT_MARKDOWN_CONTENT, and badge such competitions 'Overview not written' in the creator list — the same class of guard the courseware already applies to benchmark_expected_score.

---

## 4. [blocker / competition / fix in courseware] mlp-project/project-tracks (competition 169)

**Problem.** The Agent-track competition attached to this module (169, SuperTuxKart) uses a completely different agent contract from the one the lesson teaches. project-tracks teaches the gymnasium contract (`setup` + `choose_action`) and a pettingzoo variant (`reset(env_player_name, episode_index)`); comp 169 calls `act(obs) -> dict`. An agent written exactly as the lesson specifies exposes no method the environment ever calls.

**Evidence.**
```
project-tracks.md: "The single-agent (`gymnasium`) contract:\n```python\nclass Agent:\n    def __init__(self): ...\n    def setup(self, observation_space, action_space): ...\n    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None): ...\n```". Comp 169's own agent_template (SDK `c.competition(169)['agent_template']`): "Implements the flex_v1 zero-arg `Agent()` contract. env.py calls `act(obs) -> action` once per kart per step." Env source: /Users/raphaelcousin/reinforcement_learning_challenge/tests-dummy/catalog/kart_flex/env.py:293 `reply = agents[i].call("act", proj, timeout=ACT_TIMEOUT, ...)`.
```

**Proposed fix.** In project-tracks.md, replace the "Agent track — what you submit" code block and the paragraph beneath it. Delete the sentence "The single-agent (`gymnasium`) contract:" plus its code block and the "The multi-agent (`pettingzoo`) contract adds one required method" block, and put in their place the contract the attached competition actually uses: `class Agent:` / `def __init__(self): ...` / `def act(self, obs) -> dict: ...` returning `{"steer": -1..1, "acceleration": 0..1, "brake": 0/1, "drift": 0/1, "nitro": 0/1, "fire": 0/1, "rescue": 0/1}`, with the observation keys (`velocity`, `front`, `center_path`, `paths`, `karts`, `items`, `powerup`) copied verbatim from https://github.com/ml-arena/competition-baseline/blob/main/kart/agent_baseline.ipynb. Keep the gymnasium/pettingzoo contracts only if a gymnasium or pettingzoo competition is also attached to the Agent track.

---

## 5. [blocker / competition / fix in courseware] competition 169 and competition 170 (overview.md)

**Problem.** Both competitions ship with the platform's untouched default overview. The Agent project track (169) and one of the two Reference playgrounds (170) therefore have no task description, no observation/action spec, no reward definition, no metric direction and no baseline on the page students are told to read.

**Evidence.**
```
`curl -H 'Authorization: Bearer mlk_user_...' https://ml-arena.com/api/competition_asset/169/markdown/overview` returns exactly: "**Competition Overview**\n\nWrite your markdown here.\n\n<!-- mlarena:quickstart -->\n## Quick start ...". Identical for 170. That string is `DEFAULT_MARKDOWN_CONTENT` in backend/app/views/competition_asset.py:14. Contrast comp 168 and 172, which have full authored overviews. project-tracks.md tells the student the specs are "announced separately and attached to this module on ML-Arena".
```

**Proposed fix.** Write and PUT an overview for each. For 169, paste the observation/action documentation that already exists in the kart Colab notebook (section "1. Define your agent") plus the reward definition (track progress + placement + finish bonus), the 2 s per-call budget, and the fact that ranking is ELO not raw reward. For 170, state the action space (buy/sell/hold), that reward is mark-to-market fractional return per minute bar, that position carries across runs, and the direction/scale (a good agent is ~1e-3, not ~1).

---

## 6. [blocker / platform / fix in platform] competition 171 (engine 23), competition 169 (engine 22)

**Problem.** The Generative and Agent project tracks both run on `local_vm` engines whose VM is unreachable, and a submission does not queue — it hard-fails after blocking the HTTP request for 60 s. Two of the three tracks for a deliverable worth 50% of the course were un-submittable during my session, and the failure message gives the student no cause.

**Evidence.**
```
`c.submit(competition_id=171, files=['pitch.txt'])` → `requests.exceptions.ReadTimeout ... (read timeout=60)`; agent 8507 then reports `status: deploy_failed, last_status_message: 'Deployment failed: Failed to queue deployment job'`. Retry via `c.deploy_agent(171, 8507)` → same 60 s timeout. `c.competition(171)['engine']` = `{'id': 23, 'k8s_workload_value': 'local_vm', 'vm_health_checked_at': '2026-09-02T20:41:10', 'vm_health_ok': False}`; comp 169 engine 22 same, `vm_health_ok: False`. The real error is swallowed at backend/app/views/direct_attache_agents/deploy.py:46-55 (`except Exception: ... return None`) and re-raised as the generic string at deploy.py:229. frontend/src/components/VMHealthBanner.tsx promises the opposite: "submissions will queue and run automatically once it is back online."
```

**Proposed fix.** Two edits. (1) backend/app/views/direct_attache_agents/deploy.py: stop returning `None` from `apply_agent_deployment`'s except block — re-raise, and surface the underlying exception text in `last_status_message` so the student reads "GPU machine unreachable" instead of "Failed to queue deployment job" (this is the repo's own fail-fast rule). (2) Either make the deploy actually enqueue when `engine.vm_health_ok is False` (matching the banner's promise) or reject with HTTP 503 and the banner's text, and do not block the request for 60 s.

---

## 7. [blocker / baseline / fix in courseware] mlp-project/project-grading (competitions 169, 171)

**Problem.** project-grading allocates 15% of the project grade to "Absolute score vs the published baseline", but no baseline is published for two of the three tracks. Comp 169 has no overview at all; comp 171's overview states the pot size but no number a student is expected to beat.

**Evidence.**
```
project-grading.md grading table row: "| Leaderboard | Absolute score vs the published baseline | 15% |". project-brief.md: "scored on a public leaderboard against your cohort and against a published baseline". Comp 169 overview = default placeholder (see previous finding). Comp 171 overview says only "There's ~**$65M** of capital on the table across the panel. Nobody raises all of it" — no target. Meanwhile 171's `__benchmark__` row sits at $27,100,000 and the board top is $46,150,000, numbers that appear nowhere on the page. Only comp 172 states one: "About **59%** of patients are `alive`, so always guessing `alive` scores ~0.59. That is the bar to beat."
```

**Proposed fix.** Add one sentence to each track's overview in the same shape as 172's. For 171: "The platform benchmark pitch raises $27.1M and the current board top is $46.2M; a submission that raises less than $27.1M has not yet beaten the baseline." For 169: state the benchmark agent's mean race reward (946) and that ELO 1200 is the starting rating, so "beating the baseline" means a rating above the `__benchmark__` agent's.

---

## 8. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 9. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 10. [major / platform / fix in platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

**Problem.** 14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

**Evidence.**
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

**Proposed fix.** Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## 11. [major / platform / fix in platform] mlarena-sdk/mlarena/client.py:161, :170, :192, :1338

**Problem.** competition(), competitions() and leaderboard() send no Authorization header, so an enrolled student using the SDK or MCP is treated as anonymous and cannot open any non-public course competition — the exact class that 179-182 belong to.

**Evidence.**
```
client.py:192 `resp = self._request("GET", self._url(f"/competitions/{competition_id}"), timeout=30)` — no `headers=self._headers()`, unlike every scoped call (e.g. :225 datasets passes it). Same at :161/:170 (competitions) and :1338 (leaderboard). Observed effect: `mlarena.connect(CREATOR).competition(179)` raises CompetitionNotFoundError, while the same creator token via raw requests with a bearer header returns 200 — the owner is being told their own competition does not exist. The backend's visibility_filter (_helpers.py:229-233) grants access on `is_public OR owned OR assistant`, and the enrolled-student side-channel in competitions.py:213-217 is likewise keyed on `current_user.is_authenticated`.
```

**Proposed fix.** Add `headers=self._headers()` to the four calls. These routes accept an anonymous caller, so the change is backward-compatible and only widens what an authenticated caller can see. Without it, publishing 179-182 still leaves them invisible to any SDK/MCP student even after enrollment is fixed.

---

## 12. [major / baseline / fix in courseware] competitions 8 173 174 48 165 171 (measured reference already on the board, absent from the page)

**Problem.** For six competitions the baseline number the overview omits is already sitting on the public leaderboard as an unlabelled `__benchmark__` or starter row. The information exists; it was simply never written down.

**Evidence.**
```
Live boards, student token: 173 `rf-pixel-baseline` = 0.748911 (rank 23/26) vs an overview that says only "The starter baseline flattens the pixels into a random forest"; 174 `tfidf-logreg-starter` = 0.479719 (rank 9/27) vs "a TF-IDF + linear model is the starter baseline"; 8 `random0` = 0.0958 and `__benchmark__` = 0.0989 vs no number at all; 48 `qa-cartpole-random` = 23.3 ± 9.81 and `qa-cartpole-linear` = 500.0 vs no number; 171 `__benchmark__` = 27,100,000 vs only a $65M ceiling; 165 `__benchmark__` = 0.0 vs "see Agent.py for a SmolLM3-3B baseline".
```

**Proposed fix.** Write these six numbers into the Baseline block of each overview, quoting the leaderboard row name so a student can verify it: e.g. for 173, "Baseline: `rf-pixel-baseline` — flattened pixels into a random forest — scores 0.749 F1-macro. Higher is better."; for 48, "Baseline: a random policy scores 23.3 ± 9.8 mean episode reward; a linear policy scores 500.0, the environment's ceiling." Then register the same numbers in the external yaml from the verify-external finding so CI keeps them true.

---

## 13. [major / baseline / fix in courseware] competition 168 (AquaControl)

**Problem.** The only number on the page is unreachable and points the wrong way, so it misleads rather than orients.

**Evidence.**
```
Overview: "A perfect operator scores ~24 reward units per day." Live board, every row: `Arwen` -49.977, `baseline-valve-heuristic` -98.325, `__benchmark__` -99.946, `Jane Austen` -99.946. The page's own reward spec explains the gap — `break_penalty = 100 on terminal pipe-break only` against a `demand_reward` capped near 24/day — but never connects the two.
```

**Proposed fix.** Replace the sentence with a measured block: "Ceiling: a perfect operator that never breaks a pipe scores ~24. Baseline: the shipped valve heuristic scores -98.3 — it survives the day but under-supplies. Reference: -99.9. Best so far: -50.0. Higher is better. A score near -100 means you broke a pipe: the terminal `break_penalty` of 100 dominates everything else, so the first thing to fix is never opening V1 with both V2 and V4 closed." That turns the same numbers into a diagnostic.

---

## 14. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 15. [major / baseline / fix in courseware] competition 168 (AquaControl, attached to mlp-reference)

**Problem.** The only number on the AquaControl page is a theoretical ceiling, not a baseline. Nothing tells a student what the starter they are handed actually scores, so there is no way to know whether a run is progress or a total failure.

**Evidence.**
```
Overview: "A perfect operator scores ~24 reward units per day." I deployed the competition's own template random agent (the one its Quick-start Colab hands you) as agent 8505 on 2026-09-02: it scored **-99.939 ± 0.023 over 10 episodes**, terminating after 69 of 1440 steps (`agent_games(8505)` → `env_nb_steps: 69`, `reward: -99.93933`). The whole board sits between -49.98 and -99.95 — no agent has ever been within 74 points of the stated "~24".
```

**Proposed fix.** Append to the Reward section of comp 168's overview: "For calibration: the random-action starter scores about **-99.9** (it opens V1 with V2 and V4 shut, deadheads the pump and bursts a pipe within ~70 of the 1440 steps, taking the -100 break penalty). The valve heuristic on the board scores **-98.3**. An agent that survives a full day without a break scores **above 0**; that, not ~24, is the first target."

---

## 16. [major / validation / fix in both] competition 170 (Bitcoin Intraday, attached to mlp-reference)

**Problem.** Comp 170's leaderboard is unusable as a self-check in two independent ways: it ranks a one-run agent above agents with 2000+ runs on raw mean reward, and it renders every score as 0.00 because the display precision is 2 while returns are order 1e-4.

**Evidence.**
```
I deployed the template random agent (agent 8508) on 2026-09-02. After a single run it scored 0.029587 and took **rank 1**, above `carry-test` (0.000469, NumberOfRuns 2111) and `__benchmark__` (-0.000024, NumberOfRuns 2173); RewardCi95 is 0.0 for every row. `c.agent_games(8508)` → `competition_context: {'frontend_precision': 2, ...}`; frontend/src/utils/leaderboardFormat.ts:16-25 `formatScore` calls `toLocaleString` with `minimumFractionDigits: precision, maximumFractionDigits: precision`, so 0.000469 renders "0.00" and -0.000024 renders "-0.00".
```

**Proposed fix.** Two changes. (1) Set this competition's `evaluation_frontend_precision` to 6 (creator Settings → Evaluation) so the board shows 0.000469 rather than 0.00. (2) In frontend/src/utils/leaderboardFormat.ts, make `formatScore` fall back to significant-digit formatting when `Math.abs(value) > 0 && Math.abs(value) < 0.5 * 10**-precision`, so a mis-set precision can never render a non-zero score as 0.00. Separately, for `IsContinuous` competitions rank on MeanReward30d or suppress agents below a minimum run count, so one lucky hour cannot top a 2000-run board.

---

## 17. [major / structure / fix in courseware] mlp-reference (competitions 168, 170)

**Problem.** The Reference module attaches two RL environments as "self-study playgrounds", but none of its five lessons — nor any lesson in the course — mentions either of them. A student opening Reference sees multi-GPU scaling, CNN backprop, 3D CNNs, image enhancement and AutoML, plus two RL competitions with no lesson connecting them to anything.

**Evidence.**
```
`grep -rniE "aquacontrol|bitcoin|water network|supertuxkart" student_view/ms2a-machine-learning-practice/` returns no hits outside `_module.json` / `_course.json` metadata. The module summary in course.yaml reads "Material demoted out of class time. Self-study, linked from the sessions, never lectured." The competition labels ("AquaControl — self-study playground: MultiBinary valve control") are the only guidance a student gets.
```

**Proposed fix.** Either move competitions 168 and 170 to the modules whose material they exercise (168 → s9-reinforcement-learning-1, since it is a Gymnasium MultiBinary control task; 170 → s10-reinforcement-learning-2, since it is the stateful/continuous case), or add a sixth Reference lesson "Reference playgrounds" that, in half a page each, says what AquaControl and Bitcoin Intraday are, which session's material they extend, what the starter scores, and what a good score looks like.

---

## 18. [major / content / fix in courseware] mlp-reference/automl-and-custom-objectives

**Problem.** The custom-evaluation-metric code block raises TypeError on any xgboost >= 2.0, including the exact version the platform's own runtime ships. It is the lesson's only worked example of the "objective vs eval metric" point it is making.

**Evidence.**
```
Lesson code: `model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=rmspe)`. Running it: `xgboost 3.2.0` → `TypeError: XGBModel.fit() got an unexpected keyword argument 'eval_metric'`; `inspect.signature(xgb.XGBRegressor.fit)` params are `['self','X','y','sample_weight','base_margin','eval_set','verbose','xgb_model','sample_weight_eval_set','base_margin_eval_set','feature_weights']`. The competition runtime `scikit_learn_xgboost_lightgbm 1.8.0_3.2.0_4.6.0` (runtime id 182, offered on comps 168/169/170) is xgboost 3.2.0. The asymmetric-objective block above it does still work — I verified it trains and over-predicts as the lesson claims.
```

**Proposed fix.** Replace the two lines with: `model = xgb.XGBRegressor(eval_metric=rmspe, early_stopping_rounds=50)` then `model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])`, and add one sentence: "`eval_metric` and `early_stopping_rounds` moved from `fit()` to the constructor in xgboost 2.0; the old form now raises TypeError."

---

## 19. [major / content / fix in courseware] mlp-reference/image-enhancement

**Problem.** The lesson's headline rule — the same enhancement in the train and eval transform, randomness only in train — is demonstrated with a Compose that cannot run, because `cv2.CLAHE` is not a callable. The student's takeaway snippet raises TypeError.

**Evidence.**
```
Lesson code: `train_tf = T.Compose([clahe, T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), norm])`. Running it: `type: <class 'cv2.CLAHE'> callable? False`; `clahe(img)` → `TypeError: 'cv2.CLAHE' object is not callable`. The `clahe` name is also used 40 lines before it is defined (`clahe = cv2.createCLAHE(...)` appears in the later CLAHE section, which I verified does run).
```

**Proposed fix.** Before the Compose block, define the transform the Compose can actually use, and reference it: `def clahe_tf(img):  # PIL -> PIL` / `a = np.array(img); lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB); lab[:,:,0] = _clahe.apply(lab[:,:,0]); return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))`, then write `T.Compose([T.Lambda(clahe_tf), T.RandomResizedCrop(224), ...])` in both pipelines. Move the `cv2.createCLAHE` line up so the name is defined before first use.

---

## 20. [major / content / fix in courseware] mlp-project/project-tracks

**Problem.** The Agent-track contract omits that `setup()` receives dict-encoded space specs, not Gymnasium space objects. A student who writes the lesson's contract for any flex_v1 gymnasium competition (168, 170) gets an AttributeError on the first step, with nothing in the lesson to explain it. Lab 10 gets this right; the project lesson, which is where a student goes when writing the project, drops it.

**Evidence.**
```
Repro against the real encoder: `encode_space(gym.spaces.MultiBinary(6))` → `{'type': 'multi_binary', 'shape': [6]}` (a dict). Lesson-shaped agent storing that and calling `.sample()` → `AttributeError: 'dict' object has no attribute 'sample'`. workers/flex_v1/flexkit/gym_loop.py:72-88 passes `encode_space(...)` output to `setup`. Lab 10 (s10-reinforcement-learning-2/lab-10.md:82) does write `self.action_space = decode_space(action_space)`.
```

**Proposed fix.** In the Agent-track code block of project-tracks.md, add the decode line and one sentence: "`setup` receives the spaces **dict-encoded**, not as Gymnasium objects — decode them with `from flexkit.spaces import decode_space` before calling `.sample()` or reading `.n`/`.shape`."

---

## 21. [major / baseline / fix in both] competitions 8, 47, 48, 49, 43, 65, 165, 173, 174, 169, 170, 171

**Problem.** Twelve of the seventeen attached competitions state no numeric floor. Each says only that a Colab notebook holds a 'random-action baseline'. A student cannot tell a working submission from a broken one — the first CartPole agent scoring 21 has no way to know 21 IS the random floor.

**Evidence.**
```
Grepping every overview for a number: comp 48 (CartPole) and 49 (MountainCar) and 47 (CarRacing) and 170 contain only "a minimal **random-action** baseline you can deploy in a couple of minutes"; comp 173 says "The starter baseline flattens the pixels into a random forest" with no score; comp 174 says "A TF-IDF + linear model ... is the starter baseline and is famously hard to beat" with no score; comp 43 quotes "solved around 200" but never the floor. By contrast comp 177 carries a six-row measured table with error bars, 176 states 0.500 / 0.813, 172 states "always guessing alive scores ~0.59. That is the bar to beat."
```

**Proposed fix.** Add a '## Baselines' table to each overview in the exact form comp 177 already uses. Numbers I measured and that can be pasted in today: CartPole-v1 random = 20.98 (sd 10.89, 300 eps) and heuristic `0 if angle+0.5*angvel < 0 else 1` = 500.00; LunarLander-v3 random = -187.54 (sd 115.01, 200 eps); FrozenLake-v1 4x4 slippery random = 0.0120. For 173 and 174, run the shipped starter notebook once and paste its score. Enforce it by adding a `baselines` block to config.py that build_competitions.py refuses to publish without.

---

## 22. [major / content / fix in courseware] s5-computer-vision-1/lab-5

**Problem.** The lab is not survivable in its stated time on the hardware it says it needs. Parts B, C and D specify six full training runs and budget 30 minutes for them; measured on a laptop CPU that is 169 minutes of pure compute before a single line of the code, the four tests, the confusion matrix or the PR is written. The lab's own speaker note underestimates by 5x, so even the instructor's fallback plan does not hold.

**Evidence.**
```
Lab spec: Part B "the same number of epochs as Part C" (=13), Part C "3 epochs at lr=1e-3" then "unfreeze, 8–10 epochs", Part D "Fine-tune the same backbone three times" (3 x 13 epochs). Measured (torch 2.7.1, CPU, 4 threads, batch 32 @224, quiescent machine): resnet18 full fine-tune 4.58 s/batch, frozen head 1.59 s/batch, 3-block baseline 1.91 s/batch. At 6 classes x 300 images (mid-range of the lab's own "4 to 10 classes, 100 to 500 images per class") = 1260 train images = 40 batches/epoch: Part B 16.6 min + Part C1 3.2 + Part C2 30.5 + Part D 3x39.7 = 169.3 min vs a 30-min budget. At the smallest legal size (4 classes x 100) it is still 38 min. The speaker note says "ResNet-18 at 128 pixels fine-tunes on CPU in about ten minutes if not [a GPU]"; measured at 128px it is 1.91 s/batch = 16.5 min for one 13-epoch run, and the lab needs four such runs = 66 min.
```

**Proposed fix.** Two edits to courseware/content/ms2a-machine-learning-practice/s5-computer-vision-1/lab-5.md. (1) Under '## Setup', add: "**This lab needs a GPU.** On a free Colab T4 the six runs below take about twelve minutes. On a laptop CPU they take about three hours — measured at 4.6 s/batch for a resnet18 fine-tune, batch 32 at 224px. Without a GPU, run every model at 128px with the backbone frozen and say so in the PR." (2) Cut Part D from three full fine-tunes to three *frozen-backbone, 3-epoch* runs (measured 3.2 min each) — the ablation still answers 'which augmentation is worse than basic', which is the only thing Part D grades. Also correct the speaker note's "about ten minutes" to "about seventeen minutes per run at 128 pixels, and four runs are needed".

---

## 23. [major / platform / fix in platform] frontend/src/components/Course/directiveCards.tsx:154-169

**Problem.** `matchDirective` resolves a fence to a payload by `payload.competition_id` only, then falls back to the first directive of that type. A lesson with two checkpoints would render the first one's payload twice — the design needs a generic id match before it can put more than one checkpoint on a page.

**Evidence.**
```
directiveCards.tsx:162-167: `if (args.id != null) { const byId = sameType.find((d) => String(d.payload.competition_id) === String(args.id)); if (byId) return byId; } return sameType[0];`. Every existing directive type keys on a competition, so the bug is latent today (0 directives are in use in either course).
```

**Proposed fix.** Match on the resolved directive's own `args.id` first — `sameType.find((d) => String(d.args.id) === String(args.id))` — and keep the competition_id comparison as a second attempt for the three existing types. Drop the `sameType[0]` fallback when `args.id` was supplied: returning the wrong card silently is worse than the DirectivePlaceholder.

---

## 24. [major / platform / fix in platform] EXTENSION of findings 128 and 169 — mlarena-mcp/mlarena_mcp/server.py:110-320

**Problem.** The audit reports that MCP cannot read a competition overview. It also cannot obtain the competition's data. An MCP-first student can join a course, read every lesson and submit an agent, but can neither read the assignment nor download the dataset it is about.

**Evidence.**
```
Full tool list from server.py: join_course, list_my_courses, set_active_course, get_course, get_module, get_lesson, next_lesson, my_progress, mark_lesson_complete, list_attached_competitions, submit_agent, agent_status, leaderboard. There is no overview tool and no datasets/download tool. list_attached_competitions (:242-264) returns only competition_id, name, label, module_slug, module_title. The SDK has both `datasets()` and `download_dataset()` and neither is surfaced.
```

**Proposed fix.** Add two MCP tools backed by existing SDK/REST calls: one reading GET /api/competition_asset/<id>/markdown/overview, one wrapping client.datasets(<id>) to return labels plus signed URLs. Both are pure compositions of public routes, so neither breaks the parity rule.

---

## 25. [minor / structure / fix in courseware] content/python-ai-engineering/course.yaml (module s1-git-and-packaging)

**Problem.** Session 1 is budgeted at 230 minutes inside a 3-hour (180-minute) session, and the split is 170 minutes of lecture to 60 of lab — not the "half lecture, half lab" the course description promises. Either the lab or ~50 minutes of lecture will be cut live, and the lab is the part that is graded, so students will most likely be sent home to do Parts D and E unsupervised, which is where the pair-review and conflict-resolution learning actually is.

**Evidence.**
```
course.yaml:45,50,55,60,65,70 estimated_minutes: 10 + 45 + 50 + 30 + 35 + 60 = 230.
course.yaml:23-24 course description: "A 12-hour mise à niveau in four 3-hour sessions… Every session is half lecture, half lab."
(Same arithmetic from the served index: `awk -F'\t' '$1 ~ /^s1-/ {…}' _index.tsv` -> TOTAL 230 min.)
```

**Proposed fix.** Bring the lecture block to ~110 minutes and the lab to ~70, so 180 holds and the halves are honest: cut `git-essentials` 45 -> 35 by moving Install + Authenticating-with-GitHub into a short pre-session "Before Session 1" reference lesson (they are setup, not teaching); cut `branching-and-collaboration` 50 -> 40 by moving the Naming and Keeping-a-branch-current sections to the existing paie-reference/git-cheatsheet; cut `packaging-and-tests` 35 -> 30 by dropping Coverage to reference. Update the six `estimated_minutes` in course.yaml to 10/35/40/30/30/70 (= 215… trim `why-version-control` to 5 for 180 if you want it exact).

---

## 26. [minor / content / fix in both] competitions 65 (Connect-Four) and 169 (SuperTuxKart) — ELO-ranked

**Problem.** Both are ranked by ELO and neither page says so; the board shows a reward column that contradicts the rank, which makes any absolute baseline meaningless for these two.

**Evidence.**
```
Leaderboard payload: 65 `IsEloRanked: true`, rows ordered 1248/1200/1184/1184 Elo while MeanReward reads 0.40/0.0/-0.4/-0.8; 169 `IsEloRanked: true`, rank 1 `__benchmark__` MeanReward 946.41, rank 2 `Luigi` 7940.42, rank 3 `flexfix-kart-vmcheck` 11210.39 — reward increasing as rank worsens. Neither overview contains the string "ELO". project-tracks.md does explain it ("your rating moves when *other people* submit") but that lesson is in a different module from s10 and mlp-project only.
```

**Proposed fix.** Add to both overviews, in place of an absolute baseline: "Ranking: ELO against the current population, starting at 1200. Your rating moves when other people submit. There is no absolute bar — the target is to finish above the reference agent, which currently sits at <E>. The reward column is shown for information and does not determine rank." And suppress or de-emphasise the MeanReward column on ELO-ranked boards so it cannot be read as the score.

---

## 27. [minor / content / fix in both] competition 170 (Bitcoin Intraday Trading) — leaderboard display

**Problem.** Even once an overview is written, the board cannot show a baseline: every score rounds to 0.00 at the configured precision.

**Evidence.**
```
Leaderboard rows for 170: `carry-test` = 4.686e-4, `__benchmark__` = -2.439e-5, `groupg7` = -4.026e-5, with `FrontendPrecision: 2`. All three render as 0.00. Contrast 177, which sets FrontendPrecision 4 for a metric on a 0-1 scale.
```

**Proposed fix.** Set the competition's frontend precision to at least 6 (`client.update_settings(170, ...)`), or change env.py to report the metric in basis points so the numbers are legible, and state the unit in the Baseline block. A baseline that renders as 0.00 next to a score of 0.00 is not a baseline.

---

## 28. [minor / content / fix in courseware] mlp-project/project-brief (Milestone 3)

**Problem.** The only runnable snippet in the whole Project module is correct for one of the three tracks. A student on the Generative or Agent track who copies it gets a validation failure or uploads the wrong artefact — on the very milestone whose stated purpose is "to prove the pipeline".

**Evidence.**
```
project-brief.md: "```python\nclient.submit(competition_id=COMP_ID, files=[\"submission.csv\"])\n```". Running exactly that against the Generative track: `SubmissionError: Agent 8506 did not pass upload validation: pitch.txt file not found`. Comp 171 has `max_upload_files: 1` and requires `pitch.txt`; comp 169 requires `agent.py` (+ optional weights).
```

**Proposed fix.** Replace the single `files=["submission.csv"]` line with the three track-specific forms: Prediction `files=["submission.csv"]`; Generative `files=["pitch.txt"]` (the competition fixes the filename — a wrong name is rejected at upload, not scored); Agent `files=["agent.py", "policy.pt"]`.

---

## 29. [minor / structure / fix in courseware] mlp-project (competition 172, Prediction track)

**Problem.** The Prediction track's leaderboard is already populated by a different cohort, so "Rank in the cohort at the freeze" — 25% of the project grade — will be computed against 82 entries that are not the cohort.

**Evidence.**
```
`c.leaderboard(172)` → 82 rows, 31 distinct usernames, all LastRun between 2026-06-23 17:52 and 2026-07-06 17:11, top score 0.810513 (`nikolay78989`). Course start_date in course.yaml is 2026-09-14. project-grading.md: "| Leaderboard | Rank in the cohort at the freeze | 25% |".
```

**Proposed fix.** Either clone competition 172 as a fresh per-term competition and re-attach the clone to mlp-project (leaving the June board as history), or change the project-grading row to "Rank among MS2A submissions at the freeze" and state in project-brief that the visible board contains an earlier cohort whose 0.81 top score is a reference, not the cohort you are ranked against.

---

## 30. [minor / content / fix in courseware] mlp-reference (module summary)

**Problem.** The module's own summary says its lessons are "linked from the sessions", but two of the five are referenced from nowhere in the course, and no cross-reference anywhere in the course is an actual clickable link — they are all prose mentions, so a student has to already know the Reference module exists and navigate to it manually.

**Evidence.**
```
course.yaml mlp-reference summary: "Self-study, linked from the sessions, never lectured." Inbound mentions found by grep across all ten session modules: s3-tabular-models/gradient-boosting-in-practice.md:184 ("it is in the Reference module") → AutoML; s4-advanced-neural-networks/performance-and-profiling.md:205 → multi-GPU; s5-computer-vision-1/images-as-tensors.md:61 ("Reference module, *3D CNNs*") → 3D CNNs. `grep -rniE "enhanc|clahe|histogram equal" s2-data-preprocessing s5-computer-vision-1` → no matches, and "backpropagation" appears only in s7 lessons — so Image Enhancement and CNN Backpropagation have no inbound reference. `grep -rnoE "\]\([a-z0-9-]+/[a-z0-9-]+\)"` over the whole student dump → no matches, i.e. zero lesson-to-lesson hyperlinks.
```

**Proposed fix.** Add the two missing pointers: one sentence at the end of s5-computer-vision-1/convolution.md — "The backward pass of this operation is derived in the Reference module, *CNN Backpropagation*; read it only if you are writing a custom layer." — and one at the end of s2-data-preprocessing/the-preprocessing-contract.md — "Image-side preprocessing (histogram equalisation, CLAHE, denoising) is in the Reference module, *Image Enhancement*." Make all six of these mentions real markdown links to `/course/ms2a-machine-learning-practice/mlp-reference/<slug>` so they are clickable.

---

## 31. [minor / structure / fix in courseware] course.yaml (s9-reinforcement-learning-1, s10-reinforcement-learning-2)

**Problem.** Both sessions budget more lecture time than the whole session has. The course description promises "ten 3-hour sessions" that are "half lecture, half lab"; s9 is 175 minutes of lecture plus a 45-minute lab and s10 is 170 + 45, so the lab starts 5 to 10 minutes before the session ends. The author's own speaker notes already concede this.

**Evidence.**
```
Summing `estimated_minutes` from course.yaml: s9 lecture=175 lab=45 total=220; s10 lecture=170 lab=45 total=215, against a 180-minute slot. Course description: "ten 3-hour sessions ... Every session is half lecture, half lab." lab-9.md speaker note: "Keep an eye on the clock, Part E is the part they skip and it is where the marks are." lab-10.md speaker note: "45 minutes and it will overrun". (The overrun is course-wide: every session totals 200-240 minutes.)
```

**Proposed fix.** Either restate the promise or cut the lecture. Cheapest honest fix for these two modules: mark `markov-decision-processes` (40) and `dynamic-programming` (35) in s9, and `multi-agent-rl` (35) in s10, as pre-session reading in course.yaml — that lands s9 at 100 lecture + 45 lab and s10 at 135 + 45 — and change the course description sentence to "Every session is roughly half lecture, half lab, with two lessons per session read before class."

---

## 32. [minor / content / fix in courseware] ms2a-machine-learning-practice/mlp-project (competitions 172, 169, 171)

**Problem.** The project module reuses competition 172 as its prediction track — the same competition already attached to s3-tabular-models. A student who did Lab 3 has already submitted to it, so the project's prediction track is pre-solved for them and unranked against classmates who arrive fresh.

**Evidence.**
```
mlp-project/_module.json lists 172 ('2-Month Survival Prediction'), 169 (SuperTuxKart) and 171 (The Round); s3-tabular-models/_module.json lists 172 as its sole competition. courseware/README.md also records "**The project competitions are not attached.** ... the competition ids do not exist yet" — which is now stale, three are attached.
```

**Proposed fix.** Give the project prediction track its own competition (a different held-out split, or a different dataset), and leave 172 to s3-tabular-models. If reuse is intended, say so explicitly in mlp-project/project-tracks.md and state that Lab 3 submissions carry over.

---

