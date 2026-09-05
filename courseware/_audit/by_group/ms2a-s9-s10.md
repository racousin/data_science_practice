# Audit findings — ms2a-s9-s10

26 findings

## 1. [blocker / platform / fix in platform] frontend/src/pages/Competition/View.js:48-95 and frontend/src/components/CompetitionHeader.js:47

**Problem.** When GET /api/competitions/<id> answers 404, View.js never checks response.ok and stores the error body as competitionInfo, while CompetitionHeader catches the axios rejection and returns null. The student who clicks a module competition card lands on a page with no title, no header, empty tabs and no message — a silent blank, not an error.

**Evidence.**
```
`GET /api/competitions/179` with the student token -> 404 {"error":"Not Found"}. View.js:51-64: `const competitionResponse = await fetch(...); const competitionData = await competitionResponse.json(); ... setCompetitionData(competitionData)` — no `.ok` check anywhere in the effect. CompetitionHeader.js:47: `if (!competition) return null;`.
```

**Proposed fix.** In View.js:51-64 check `if (!competitionResponse.ok)` and set an error state; render a dedicated panel for 404 that says the competition is not open to you yet and links back to the course, instead of the tab shell. Same for the /environment fetch at View.js:66-73. In CompetitionHeader.js:18-23 keep the caught error in state and render the same panel rather than `return null`.

---

## 2. [blocker / platform / fix in platform] backend/app/views/academic_courses/consumption.py:82-87; frontend/src/pages/CourseLearner/ModuleOverview.tsx:65-90; frontend/src/pages/CourseLearner/CourseLanding.tsx:100-118

**Problem.** _serialize_module_competition returns only competition_id/label/name, with no visibility or availability signal, so both learner surfaces unconditionally render an `Open competition` link into a page the student cannot open. This is the (b) half of the attached-but-invisible problem.

**Evidence.**
```
consumption.py:82-87 returns exactly {"competition_id", "label", "name"}. ModuleOverview.tsx:80-87 always renders `<Button component={RouterLink} to={`/viewcompetition/${competition.competition_id}`}>Open competition</Button>`. CourseLanding.tsx:109-113 always `navigate(`/viewcompetition/${c.competition_id}`)`. Live: module s1-git-and-packaging advertises competition 179 (label "textstats correctness") and 179 is 404 for the student token.
```

**Proposed fix.** Extend _serialize_module_competition (consumption.py:82-87) to `{"competition_id", "label", "name", "is_public": bool(link.competition.is_public), "is_started": bool(link.competition.is_started), "viewable": user_can_see_competition(link.competition)}` (import from app.views.creator_competition._helpers). In ModuleOverview.tsx:80-87 replace the Button with a disabled `Not open yet` badge when `!competition.viewable`; in CourseLanding.tsx:100-118 drop the onClick and render the badge greyed with the same tooltip. Mirror the new fields in frontend/src/services/coursesApi.ts ModuleCompetitionSummary (~line 701) and in the MCP list_attached_competitions output (mlarena-mcp/mlarena_mcp/server.py:242-264).

---

## 3. [blocker / content / fix in both] s10-reinforcement-learning-2/lab-10

**Problem.** Part E's submit snippet cannot produce an accepted submission for a torch agent — which is every agent the lab tells you to build. A fresh attachment defaults to the framework=`none` runtime image (no torch), so `import agent` fails and the deploy dies. The lab's own table warns "a missing dependency fails the deploy" but hands the student the exact call that causes it, and never mentions that `submit()` takes a `runtime=` argument.

**Evidence.**
```
lab-10.md Part E, verbatim: `res = client.submit(COMPETITION_ID, files=["src/rl/agent.py", "checkpoints/policy.pt"])`. I ran exactly that with a torch agent on comp 48: attachment 8510 -> status `deploy_failed`, message `File "/app/storage/competitions/48/agent/8510/agent.py", line 3, in <module> import torch / ModuleNotFoundError: No module named 'torch'`. `c.agent_runtime(8510)` -> `{'framework': 'none', 'id': 181}`. Identical files submitted as `c.submit(48, files=FILES, runtime={"language":"python","framework":"torch"})` -> attachment 8511, `c.agent_runtime(8511)` -> `{'framework':'torch','framework_version':'2.12.0','id':167}`, status `active`, "Deployment completed successfully". Both attachments deleted afterwards.
```

**Proposed fix.** In lab-10.md Part E replace the snippet with `res = client.submit(COMPETITION_ID, files=["src/rl/agent.py", "checkpoints/policy.pt"], runtime={"language": "python", "framework": "torch"})` and add one sentence above it: "A new attachment defaults to the dependency-free runtime image. If your agent.py imports torch, tensorflow or jax you must pin the matching runtime with `runtime=` or the deploy fails at import with ModuleNotFoundError — `client.runtime_options(COMPETITION_ID)` lists what is available." Add the same sentence to the "four rules" table row for "Importable". Platform side: `backend/app/views/direct_attache_agents/create_delete.py:125` picks the default with an unordered `.first()`, so which image a student silently gets is whatever Postgres returns first — order it explicitly, or make the competition carry a declared default runtime.

---

## 4. [blocker / content / fix in courseware] s10-reinforcement-learning-2/lab-10

**Problem.** Part B and Part D do not compose. Part B saves an SB3 `ActorCriticPolicy` state_dict; Part D loads it into a student-written `Policy`. The key names do not match and `load_state_dict` raises. Reconstructing SB3's internal module tree (`mlp_extractor.policy_net.*`, `action_net.*`, `value_net.*`) is the hardest step in the lab and it is a single unexplained line — no lesson in Session 10 ever shows those key names.

**Evidence.**
```
lab-10.md Part B: `torch.save(model.policy.state_dict(), "checkpoints/policy.pt")`. Part D: `self.net = Policy(...)` then `self.net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))`. I built `Policy` to the only spec the course gives (making-deep-rl-work.md: "MLP hidden layers | 2 x 64 for states", "Activation | tanh for PPO") and got: `RuntimeError: Error(s) in loading state_dict for Policy: Missing key(s) in state_dict: "net.0.weight", ... Unexpected key(s) in state_dict: "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight", "action_net.weight", "action_net.bias", "value_net.weight", "value_net.bias".`
```

**Proposed fix.** In lab-10.md Part B replace the save line with an export that matches what Part D loads, and show it: `pol = model.policy; torch.save({"obs_dim": ..., "n_actions": ..., "w": [pol.mlp_extractor.policy_net[0].weight, pol.mlp_extractor.policy_net[0].bias, pol.mlp_extractor.policy_net[2].weight, pol.mlp_extractor.policy_net[2].bias, pol.action_net.weight, pol.action_net.bias]}, "checkpoints/policy.pt")` — i.e. state explicitly that the deliverable is a *re-export* of the three policy-path layers into your own module's naming, not `model.policy.state_dict()`. Then add one sentence: "`model.policy.state_dict()` carries SB3's own module names (`mlp_extractor.policy_net.0.weight`, `action_net.weight`, `value_net.*`) plus the critic, which your inference module does not have — loading it directly raises `Missing key(s)`."

---

## 5. [major / baseline / fix in courseware] courseware/competitions/s4-mnist-warmup/overview.md:61-65

**Problem.** The overview's self-diagnosis rule is factually wrong, and I falsified it by measurement. It says being below the 91.3% logistic-regression line means your bug is normalisation / a stray softmax / a missing model.eval(). I built the first of those bugs deliberately — an MLP trained on `(x/255 - 0.1307)/0.3081` but fed raw uint8 0-255 at prediction time — and it scored 0.9384, comfortably ABOVE the 91.3% line. The stated check therefore does not catch the bug it names; the only number that discriminates is the 97% figure in the sentence before it.

**Evidence.**
```
overview.md:61-65: "Multinomial logistic regression on raw pixels scores **91.3%**. An MLP that is wired up correctly clears 97%. If you are below the logistic-regression line, the problem is not your architecture — it is the normalisation, a `softmax` before `CrossEntropyLoss`, or a missing `model.eval()` ..."

Scored with the real env.py against the real y_test.csv:
  submission_correct.csv      score=0.9862  macro-F1=0.9863
  submission_raw_pixels.csv   score=0.9384  macro-F1=0.9400   <- the normalisation bug the sentence names
  benchmark_submission.csv    score=0.9134  (== config.py benchmark_expected_score, reproduces exactly)
```

**Proposed fix.** Replace overview.md lines 63-65 with: "An MLP that is wired up correctly clears **97%**. Below 97% — even well above the logistic line — the problem is not your architecture: it is the normalisation (feeding raw 0-255 to a model trained on `(x/255 - 0.1307)/0.3081` scores about 94%), a `softmax` before `CrossEntropyLoss`, or a missing `model.eval()`. Below 91.3% you are behind a linear model and something is badly wrong. The one-batch overfit test from Part B would have told you which."

---

## 6. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 7. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 8. [major / platform / fix in platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

**Problem.** 14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

**Evidence.**
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

**Proposed fix.** Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## 9. [major / baseline / fix in courseware] competitions 8 173 174 48 165 171 (measured reference already on the board, absent from the page)

**Problem.** For six competitions the baseline number the overview omits is already sitting on the public leaderboard as an unlabelled `__benchmark__` or starter row. The information exists; it was simply never written down.

**Evidence.**
```
Live boards, student token: 173 `rf-pixel-baseline` = 0.748911 (rank 23/26) vs an overview that says only "The starter baseline flattens the pixels into a random forest"; 174 `tfidf-logreg-starter` = 0.479719 (rank 9/27) vs "a TF-IDF + linear model is the starter baseline"; 8 `random0` = 0.0958 and `__benchmark__` = 0.0989 vs no number at all; 48 `qa-cartpole-random` = 23.3 ± 9.81 and `qa-cartpole-linear` = 500.0 vs no number; 171 `__benchmark__` = 27,100,000 vs only a $65M ceiling; 165 `__benchmark__` = 0.0 vs "see Agent.py for a SmolLM3-3B baseline".
```

**Proposed fix.** Write these six numbers into the Baseline block of each overview, quoting the leaderboard row name so a student can verify it: e.g. for 173, "Baseline: `rf-pixel-baseline` — flattened pixels into a random forest — scores 0.749 F1-macro. Higher is better."; for 48, "Baseline: a random policy scores 23.3 ± 9.8 mean episode reward; a linear policy scores 500.0, the environment's ceiling." Then register the same numbers in the external yaml from the verify-external finding so CI keeps them true.

---

## 10. [major / baseline / fix in courseware] competition 43 (LunarLander-v3)

**Problem.** The stated target sits below the median of its own leaderboard and roughly where a random agent already lands, so "solved around 200" reads as an aspiration and functions as a floor.

**Evidence.**
```
Overview: "the score is **mean episode reward**. The task is considered solved around **200**." Live board (708 entries): rank 505 is an agent literally named `RandomAgent` at 190.49; rank 632 `Baseline` at 62.90; rank 661 `__benchmark__` at -266.62; rank 1 at 293.05 ± 2.35 over 300 episodes.
```

**Proposed fix.** Requote against this board: "Baseline: a random policy scores ~190 mean episode reward over 100 episodes. Reference: -266.6 (an untrained agent that crashes). Solved, by the Gymnasium convention, is 200. Passing this lab: 250. Higher is better; the leaderboard shows a 95% CI, and two agents whose intervals overlap are tied." The board already reports RewardCi95 (2.35 at the top) so the resolution line is measurable, not invented.

---

## 11. [major / platform / fix in platform] backend/app/views/teacher/courses.py:95-98; backend/app/views/teacher/modules.py:286-289; frontend/src/pages/CourseAuthoring/AddCompetitionModal.tsx:49

**Problem.** Nothing warns a teacher that the competition they are attaching is invisible to students. The picker route deliberately includes the teacher's own non-public competitions but returns only {id, name}; the attach route validates existence only; the modal renders a bare Select. This is the (a) half of the attached-but-invisible problem, and it is exactly how courses 14's four 404s got shipped.

**Evidence.**
```
teacher/courses.py:90-98: `query = query.filter(visibility_filter())` (which is `is_public OR owner OR assistant`) then `return jsonify([{"id": c.id, "name": c.name} for c in competitions])`. teacher/modules.py:286-289 is the entire validation: `if Competition.query.get(payload.competition_id) is None: return 400`. AddCompetitionModal.tsx:49: `const options = competitions.map((c) => ({ value: String(c.id), label: c.name }));`.
```

**Proposed fix.** teacher/courses.py:96 -> `{"id": c.id, "name": c.name, "is_public": bool(c.is_public), "is_started": bool(c.is_started)}`. teacher/modules.py: after the existence check, if `not competition.is_public or not competition.is_started`, still create the link but return the 201 body with `"warnings": ["Competition <n> is not public — enrolled students will get a 404 until you publish it."]`. AddCompetitionModal.tsx:49 -> render the option with a `renderOption` badge 'Hidden' / 'Not started', and show a yellow Alert under the Select when the picked competition is not public. Surface the same warnings array in SDK attach_competition (client.py:1734) and MCP.

---

## 12. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 13. [major / baseline / fix in both] competition 174 (Clinical Note Triage)

**Problem.** The overview names a baseline but gives no number, so "you have done the lab" is undefined; and the leaderboard column is labelled `reward` rather than the F1-macro the overview promises, so a student cannot tell what 0.48 is or which direction is better. The number the student needs already exists on the board — it is just not written down anywhere a student reads.

**Evidence.**
```
Overview, verbatim and complete on this point: "A **TF-IDF + linear model** (logistic regression / linear SVM) is the starter baseline and is famously hard to beat on clinical text — a good lesson in itself." No figure appears anywhere in the overview. `client.leaderboard(174)`: `Metric == "reward"` and `MetricsSchema is None` for all 27 rows; values span 0.020908 (`__benchmark__`, the constant-class floor) to 0.608398 (`pubmedbert-logreg-v2`), with `tfidf-logreg-starter` at 0.479719. Compare competition 178, which carries a full `MetricsSchema` (`{'key':'reward','label':'Accuracy','higher_is_better':True,'precision':4,...}`) and a five-row measured baseline table in its overview.
```

**Proposed fix.** (a) Add to 174's overview, under "## Task", the table it is missing: `__benchmark__` (constant class) 0.0209 | TF-IDF + logistic regression starter 0.4797 | best entry to date 0.6084, followed by "F1-macro, higher is better, range 0 to 1. Below 0.4797 the transformer has not paid for itself." (b) Set `metrics_schema` on competition 174 so the leaderboard column reads "F1-macro" with `higher_is_better: True` instead of the generic `reward`.

---

## 14. [major / competition / fix in courseware] s9-reinforcement-learning-1 (competitions 48, 49) / course.yaml

**Problem.** Session 9 teaches tables only and says so explicitly, but both attached competitions have Box observation spaces, and the module label asserts the opposite. There is no path from Lab 9's agent to either competition: discretizing a continuous observation (binning, tile coding) is never taught in s9 or s10.

**Evidence.**
```
gymnasium.md: "`Discrete` observations index a table — this session. `Box` observations need a function approximator — Session 10." from-tables-to-networks.md table: "| CartPole | continuous, 4-dim | infinite |". course.yaml s9 label: "CartPole-v1 — discrete actions over a state you can discretize: where Lab 9's tabular agent lands". `grep -rniE "discretiz|discretis|tile coding|np.digitize" s9/*.md s10/*.md` returns exactly one hit, in from-tables-to-networks.md line 265, about continuous *action* spaces: "Discretising into bins works for one dimension and explodes combinatorially for six." MountainCar (comp 49) leaderboard: all 3 entries plus `__benchmark__` score exactly -200.0, the episode floor.
```

**Proposed fix.** Either (a) move competitions 48 and 49 to s10 in course.yaml and attach a Discrete-observation competition to s9 (a Frozen Lake or Taxi-v3 track, which is what Lab 9 actually builds), or (b) keep them in s9 and add a sixth lecture lesson `state-discretization` between `gymnasium` and `lab-9` covering `np.digitize` binning of CartPole's 4 floats and MountainCar's 2, with the bin counts that work (6x12 for MountainCar position/velocity, 6x6x12x12 for CartPole). Until one of those lands, replace the s9 CartPole label with "CartPole-v1 — optional: Lab 9's agent needs a state-discretization step this session does not teach" and drop the MountainCar attachment, since nothing on its leaderboard has ever beaten the floor.

---

## 15. [major / content / fix in courseware] s10-reinforcement-learning-2/making-deep-rl-work

**Problem.** Two code blocks do not run on gymnasium 1.3.0, the version pinned in every runtime image for these competitions, and the surrounding autoreset explanation teaches the Gym 0.x mental model, which is the reverse of 1.x behaviour. This is the lesson the speaker notes call "the lesson they will actually use in the lab".

**Evidence.**
```
`gym.wrappers.ClipObservation(env, -10, 10)` -> `AttributeError: module 'gymnasium.wrappers' has no attribute 'ClipObservation'` on gymnasium 1.3.0; `[n for n in dir(gym.wrappers) if "Clip" in n]` == `['ClipAction', 'ClipReward']`. `real_next_obs[done] = infos["final_observation"][done]` -> `KeyError: 'final_observation'`; at a vec-env termination `list(infos.keys())` == `[]`. And the claim "the observation you receive after a termination belongs to the **next** episode" is false: on `gym.make_vec("CartPole-v1", num_envs=2)` the terminating step returned pole angle 0.219 rad (past the 0.2095 failure threshold — the true final obs) and the reset observation `[0.004, 0.044, 0.032, -0.05]` arrived on the *following* step.
```

**Proposed fix.** Delete the `ClipObservation` line (or replace with `env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)`). Rewrite the "Autoreset, and the trap in it" section for 1.x: replace "Vector environments reset a finished sub-environment automatically, so the observation you receive after a termination belongs to the next episode" with "In Gymnasium 1.x the terminating step returns the true final observation; the reset observation arrives on the *next* step, where `terminated` and `truncated` are both False and the reward is 0. Skip that transition when writing to the buffer." and replace the `infos["final_observation"]` line with the flag-based skip: `mask = ~prev_done; buffer.add(obs[mask], action[mask], reward[mask], next_obs[mask], terminated[mask])`.

---

## 16. [major / content / fix in courseware] s9-reinforcement-learning-1/markov-decision-processes

**Problem.** The lesson states Frozen Lake's slip probabilities as 0.8/0.1/0.1 and they are 1/3 each. Lab 9 in the same module states 0.33, so the student meets a direct contradiction while debugging an agent, and the wrong number makes the DP lesson's `P` walkthrough and Part E's "why does this arrow point away from the goal" reasoning come out wrong.

**Evidence.**
```
markov-decision-processes.md: "Stochastic: `RIGHT` moves you right with probability 0.8 and sideways with probability 0.1 each, and $P$ is a distribution." immediately followed by "Frozen Lake is the second kind." Measured: `gym.make("FrozenLake-v1", is_slippery=True).unwrapped.P[0][2]` == `[(0.3333, 4, 0, False), (0.3333, 1, 0, False), (0.3333, 0, 0, False)]`. lab-9.md Part E: "the intended move has probability 0.33".
```

**Proposed fix.** Replace "Stochastic: `RIGHT` moves you right with probability 0.8 and sideways with probability 0.1 each" with "Stochastic: on slippery Frozen Lake `RIGHT` moves you right with probability 1/3 and slides you to each perpendicular square with probability 1/3 — the intended direction is only as likely as each accident." The following sentence ("why a policy which looks obviously optimal scores 0.7 rather than 1.0") is correct and should stay: I measured the DP-optimal policy at 0.740 over 20,000 episodes.

---

## 17. [major / validation / fix in courseware] s9-reinforcement-learning-1/lab-9

**Problem.** Nothing in Lab 9 tells the student what a correct agent scores, and the self-check the course promised is missing. The Dynamic Programming lesson sells value iteration as "the only ground truth you will ever have in RL, and the lab uses it" — Lab 9 contains no reference to value iteration, V*, or dynamic programming at all. The lab also never states how many episodes to train for, so "the same number of episodes" in Part C is unanchored and the Part D table is not comparable between students.

**Evidence.**
```
dynamic-programming.md: "It is the **reference implementation**. On a small MDP you can compute $V_*$ exactly, then check whether your Q-learning agent got close. That is the only ground truth you will ever have in RL, and the lab uses it." `grep -niE "value iteration|dynamic programming|V_\*|ground truth" lab-9.md` returns one line, and it is Part E's "the optimal policy frequently points away from the goal" — no DP anywhere. Running the lab as written at 5 seeds x 5000 episodes, alpha=0.1, eps decay 0.9995 floor 0.05: Q-learning 0.726 +/- 0.002, SARSA 0.680 +/- 0.092, wallclock 44s. I had to write value iteration myself to learn the ceiling is 0.740.
```

**Proposed fix.** Add a Part A' (5 min) to lab-9.md: "Run value iteration on `env.unwrapped.P` with gamma=0.99 to convergence, extract the greedy policy, and roll it out for 20,000 episodes. Write that success rate in REPORT.md — it is your ceiling. On 4x4 slippery Frozen Lake it is 0.74; a Q-learning agent that plateaus below 0.65 has a bug, not a hyperparameter problem." Reuse the `improve()` helper already printed in dynamic-programming.md so this costs six lines. In Part C, replace "for the same number of episodes" with "for 5,000 episodes each — enough to reach the ceiling with alpha=0.1 and eps decaying 0.9995 to a floor of 0.05, and about 45 seconds of compute for both agents across five seeds."

---

## 18. [major / baseline / fix in both] competitions 48, 49, 43, 65

**Problem.** None of the four competitions states, anywhere a student can reach, what score counts as done, which direction the metric runs, or what its range is. The published description is a one- or two-sentence stub lifted from the Farama docs; two of them are broken sentences where the hyperlink text was stripped. The only baseline visible is a `__benchmark__` leaderboard row that no page explains, and on comp 43 it sits at rank 661 of 708.

**Evidence.**
```
`c.competition(48)['description']` == "This environment is part of the Classic Control environments which contains general information about the environment.\n\nSource: https://gymnasium.farama.org/environments/classic_control/cart_pole/" (same stub for 49). `c.competition(43)['description']` == "A classic rocket trajectory optimization problem". `c.competition(65)['description']` == "This environment is part of the classic environments . Please read that page first for general information." — note the orphaned space before the period. The competition payload has no `overview`/`rules`/`evaluation`/`metric` key at all (`sorted(d.keys())` on all four). Measured from the leaderboards: comp 48 `__benchmark__` 20.7 (a random agent — `qa-cartpole-random` scores 23.3) against a 500 ceiling; comp 49 `__benchmark__` -200.0, which is the floor and also the score of every other entry; comp 43 `__benchmark__` -266.6 at rank 661/708 while 460 of 708 entries already score >= 200.
```

**Proposed fix.** Write an overview.md for each of the four and publish it, each ending with a Target line, in the shape the courseware/competitions/ packages already use: comp 48 "Metric: mean episode reward over seeded episodes, higher is better, range 0-500. A random policy scores about 21. Target: 475+ (CartPole-v1 is considered solved at 475 averaged over 100 episodes)." comp 49 "Metric: mean episode reward, higher is better, range -200 to about -90. Every episode costs -1 per step and truncates at 200, so -200 means you never reached the flag — it is both the random score and the floor. Target: -110." comp 43 "Metric: mean episode reward, higher is better. A random policy scores about -180; the median submission scores 218. Target: 200+ (LunarLander is considered solved at 200 averaged over 100 episodes)." comp 65 "Ranked by ELO against the live population, not by mean reward; 1200 is the starting rating and the benchmark sits at 1248. Target: beat 1248 over at least 50 games."

---

## 19. [major / content / fix in courseware] s10-reinforcement-learning-2/lab-10

**Problem.** Part A offers a PettingZoo track and Part D specifies its packaging contract, but Part B gives no way to train one. The only training recipe is `gym.make_vec` + SB3 PPO, which cannot consume a PettingZoo environment, and no lesson shows self-play, opponent pools or masked training — multi-agent-rl.md teaches masking at inference and then says a league is "enough to see the effect" without any code. Half the lab's stated options are unbuildable from the material.

**Evidence.**
```
lab-10.md Part A: "pick a **Gymnasium** track (single agent, ranked by mean reward) or a **PettingZoo** track (two-player, ranked by ELO)". Part B in full: "PPO unless you have a reason. Stable-Baselines3 unless you have a reason." + `envs = gym.make_vec(ENV_ID, num_envs=8)` / `model = PPO("MlpPolicy", envs, ...)`. Part D: "For a PettingZoo track you also implement `reset(env_player_name, episode_index)`". Nothing between them bridges the two. Comp 65's leaderboard has 4 entries total, the top one being `__benchmark__`.
```

**Proposed fix.** Either restrict Lab 10 to the Gymnasium track — change Part A to "pick a **Gymnasium** track (single agent, ranked by mean reward)" and move the PettingZoo packaging paragraph in Part D into an optional 'Going further' box — or add to Part B a PettingZoo branch with the missing three lines: `from pettingzoo.classic import connect_four_v3; import supersuit as ss; env = ss.pettingzoo_env_to_vec_env_v1(ss.black_death_v3(connect_four_v3.parallel_env()))`, plus a note that PPO must be trained against a frozen copy of itself and that logits must be masked during *training* as well as inference or the policy never learns the legal set.

---

## 20. [major / baseline / fix in both] competitions 8, 47, 48, 49, 43, 65, 165, 173, 174, 169, 170, 171

**Problem.** Twelve of the seventeen attached competitions state no numeric floor. Each says only that a Colab notebook holds a 'random-action baseline'. A student cannot tell a working submission from a broken one — the first CartPole agent scoring 21 has no way to know 21 IS the random floor.

**Evidence.**
```
Grepping every overview for a number: comp 48 (CartPole) and 49 (MountainCar) and 47 (CarRacing) and 170 contain only "a minimal **random-action** baseline you can deploy in a couple of minutes"; comp 173 says "The starter baseline flattens the pixels into a random forest" with no score; comp 174 says "A TF-IDF + linear model ... is the starter baseline and is famously hard to beat" with no score; comp 43 quotes "solved around 200" but never the floor. By contrast comp 177 carries a six-row measured table with error bars, 176 states 0.500 / 0.813, 172 states "always guessing alive scores ~0.59. That is the bar to beat."
```

**Proposed fix.** Add a '## Baselines' table to each overview in the exact form comp 177 already uses. Numbers I measured and that can be pasted in today: CartPole-v1 random = 20.98 (sd 10.89, 300 eps) and heuristic `0 if angle+0.5*angvel < 0 else 1` = 500.00; LunarLander-v3 random = -187.54 (sd 115.01, 200 eps); FrozenLake-v1 4x4 slippery random = 0.0120. For 173 and 174, run the shipped starter notebook once and paste its score. Enforce it by adding a `baselines` block to config.py that build_competitions.py refuses to publish without.

---

## 21. [major / competition / fix in courseware] ms2a-machine-learning-practice/s9-reinforcement-learning-1 (competitions 48, 49)

**Problem.** Lab 9 is titled 'Tabular Agent on Frozen Lake' and bans RL libraries, but the session's competitions are CartPole-v1 and MountainCar-v0 — both continuous-state, so the agent the student just wrote does not transfer without a discretisation step the lab never teaches. MountainCar is worse than useless as a first exercise: a random policy and a half-working policy both score exactly -200, so the leaderboard gives zero feedback.

**Evidence.**
```
s9-reinforcement-learning-1/_module.json lists competitions 48 (CartPole-v1) and 49 (MountainCar-v0). lab-9.md: "Implement Q-learning and SARSA from scratch ... **No RL library** — `gymnasium` and `numpy` only" over `gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)`. gymnasium.md already states the target floor: "On 4x4 Frozen Lake the floor is about 0.014 successes per episode." The FrozenLake competition the old module9_exercise2.ipynb links to (id 5) now returns CompetitionNotFoundError to a student.
```

**Proposed fix.** Build a FrozenLake-v1 4x4 is_slippery=True competition (flex_v1 gymnasium preset, mean episode reward over 1000 episodes, higher better) and attach it to s9 as the ramp; move 48 to s10 and drop 49. Measured for the overview: uniform-random = 0.0120 (sd 0.1089, 1000 eps); the Lab 9 tabular Q-learner (30k episodes, alpha 0.1, gamma 0.99, epsilon 1.0->0.05, greedy at eval) = 0.7290 (sd 0.4445). 8x8 stretch: random 0.0030, Q-learning 0.6260.

---

## 22. [minor / content / fix in both] competitions 65 (Connect-Four) and 169 (SuperTuxKart) — ELO-ranked

**Problem.** Both are ranked by ELO and neither page says so; the board shows a reward column that contradicts the rank, which makes any absolute baseline meaningless for these two.

**Evidence.**
```
Leaderboard payload: 65 `IsEloRanked: true`, rows ordered 1248/1200/1184/1184 Elo while MeanReward reads 0.40/0.0/-0.4/-0.8; 169 `IsEloRanked: true`, rank 1 `__benchmark__` MeanReward 946.41, rank 2 `Luigi` 7940.42, rank 3 `flexfix-kart-vmcheck` 11210.39 — reward increasing as rank worsens. Neither overview contains the string "ELO". project-tracks.md does explain it ("your rating moves when *other people* submit") but that lesson is in a different module from s10 and mlp-project only.
```

**Proposed fix.** Add to both overviews, in place of an absolute baseline: "Ranking: ELO against the current population, starting at 1200. Your rating moves when other people submit. There is no absolute bar — the target is to finish above the reference agent, which currently sits at <E>. The reward column is shown for information and does not determine rank." And suppress or de-emphasise the MeanReward column on ELO-ranked boards so it cannot be read as the score.

---

## 23. [minor / content / fix in courseware] s9-reinforcement-learning-1/gymnasium

**Problem.** The "ML-Arena shape" snippet puts `from flexkit.spaces import decode_space` at module level. flexkit ships only inside the platform runtime image — it is in no runtime's pip requirement list — so a student who copies this cannot `import agent` locally, which is what Lab 10's four required tests all do. The competition's own agent_template already solves this and the lesson did not copy the fix.

**Evidence.**
```
gymnasium.md "The ML-Arena shape": `from flexkit.spaces import decode_space` on its own line above `class Agent:`. Running `import agent` on that file locally: `ModuleNotFoundError: No module named 'flexkit'`. None of the five runtime_options for comps 48/49/43/65 lists flexkit in `requirement`. The PyPI package named `flexkit` is an unrelated 0.1.11 stub (summary: "Add your description here"), so the obvious recovery installs the wrong thing — the same trap as `mlarena` vs `mlarena-sdk`. Meanwhile the live agent_template for comp 48 does it correctly, with a comment: "# flexkit is provided by the platform runtime (not needed to run this notebook locally); import it here so `import agent` works anywhere."
```

**Proposed fix.** In gymnasium.md, move the import inside `setup` and carry the template's comment verbatim: `class Agent:` / `    def setup(self, observation_space, action_space):` / `        # flexkit ships in the platform runtime, not on PyPI — import it here` / `        # so "import agent" (and your tests) work on your laptop too.` / `        from flexkit.spaces import decode_space`. Do the same in lab-10.md Part D, which currently uses `decode_space` with no import shown at all.

---

## 24. [minor / platform / fix in platform] competitions 48, 49, 65 (leaderboard Metric column)

**Problem.** The leaderboard labels the score column `accuracy` on three of the four competitions while the number displayed is mean episode reward or an ELO-ranked mean outcome. A student who has just been taught "the only honest metric is episode return" sees a CartPole score of 500 labelled accuracy and an ELO-ranked Connect-Four score of 0.4 labelled accuracy.

**Evidence.**
```
`c.leaderboard(48)` -> `Metric` == 'accuracy' for every row, `MeanReward` 500.0 / 23.3 / 20.7. `c.leaderboard(49)` -> `Metric` == 'accuracy', `MeanReward` -200.0. `c.leaderboard(65)` -> `Metric` == 'accuracy', `IsEloRanked` True, `EloScore` 1248. Comp 43 is correct: `Metric` == 'reward'.
```

**Proposed fix.** Set the metric name on competitions 48, 49 and 65 to `reward` (matching comp 43) so the leaderboard column reads what it is. Comp 65 additionally displays both `EloScore` and a `MeanReward` labelled accuracy while ranking by ELO — surface only the ELO column when `IsEloRanked` is true.

---

## 25. [minor / structure / fix in courseware] course.yaml (s9-reinforcement-learning-1, s10-reinforcement-learning-2)

**Problem.** Both sessions budget more lecture time than the whole session has. The course description promises "ten 3-hour sessions" that are "half lecture, half lab"; s9 is 175 minutes of lecture plus a 45-minute lab and s10 is 170 + 45, so the lab starts 5 to 10 minutes before the session ends. The author's own speaker notes already concede this.

**Evidence.**
```
Summing `estimated_minutes` from course.yaml: s9 lecture=175 lab=45 total=220; s10 lecture=170 lab=45 total=215, against a 180-minute slot. Course description: "ten 3-hour sessions ... Every session is half lecture, half lab." lab-9.md speaker note: "Keep an eye on the clock, Part E is the part they skip and it is where the marks are." lab-10.md speaker note: "45 minutes and it will overrun". (The overrun is course-wide: every session totals 200-240 minutes.)
```

**Proposed fix.** Either restate the promise or cut the lecture. Cheapest honest fix for these two modules: mark `markov-decision-processes` (40) and `dynamic-programming` (35) in s9, and `multi-agent-rl` (35) in s10, as pre-session reading in course.yaml — that lands s9 at 100 lecture + 45 lab and s10 at 135 + 45 — and change the course description sentence to "Every session is roughly half lecture, half lab, with two lessons per session read before class."

---

## 26. [minor / baseline / fix in courseware] competition 43 (LunarLander-v3), s10-reinforcement-learning-2

**Problem.** LunarLander's overview quotes the solved threshold but not the floor, so the student has a 390-point gap with only one end marked and no intermediate rung inside the session.

**Evidence.**
```
Comp 43 overview: "the score is **mean episode reward**. The task is considered solved around **200**." No random-policy number. Measured: random policy = -187.54 (sd 115.01) over 200 episodes.
```

**Proposed fix.** Add to 43's overview: 'Random policy scores -187.54 (sd 115.01) over 200 episodes. Solved is ~200.' And attach CartPole-v1 (48) to s10-reinforcement-learning-2 as the ramp — a competition can be linked to two modules (172 already is, to s3-tabular-models and mlp-project). CartPole measured: random 20.98 (sd 10.89); the one-line heuristic `0 if angle + 0.5*angular_velocity < 0 else 1` = 500.00 (sd 0.00), which is the 'accepted submission in the first ten minutes' Lab 10's own notes demand.

---

