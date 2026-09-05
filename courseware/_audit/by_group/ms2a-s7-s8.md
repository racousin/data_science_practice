# Audit findings — ms2a-s7-s8

27 findings

## 1. [blocker / competition / fix in platform] competition 174 (Clinical Note Triage, attached to s7-nlp-1)

**Problem.** The only dataset attached to the Session 7 capstone competition is unreachable. Both files return 404 from GCS, so no student can start the competition, and the linked Colab starter dies on its data cell. This is the single external artefact that could prove a student did Lab 7.

**Evidence.**
```
`uv run python get.py` -> `requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://storage.googleapis.com/rlarena-417509-render-experiments-eu/prod/datasets/8/train.csv?...`. Direct GET on both signed URLs from `client.datasets(174)`:
  train.csv  id=28 size=5740082 -> HTTP 404  body: `<Code>NoSuchKey</Code>...No such object: rlarena-417509-render-experiments-eu/prod/datasets/8/train.csv`
  test.csv   id=29 size=1422816 -> HTTP 404  (same, .../prod/datasets/8/test.csv)
The DB rows still exist with their byte sizes. The overview's "Quick start" Colab (raw URL returns 200) has as cell 4: `client.download_dataset(COMPETITION_ID, "data/")` — it dies there.
```

**Proposed fix.** Re-upload the two objects to `prod/datasets/8/train.csv` and `prod/datasets/8/test.csv` in `rlarena-417509-render-experiments-eu` (or repoint `dataset_file` rows 28 and 29 at wherever they now live, e.g. the R2 bucket). Verify with `client.download_dataset(174, "data/")` returning two paths whose sizes match 5740082 / 1422816. Until that is done, do not point students at competition 174 from any lesson.

---

## 2. [blocker / content / fix in courseware] s8-nlp-2/evaluating-generation + s8-nlp-2/lab-8

**Problem.** Session 8's central instrument — the LLM judge — is never made runnable. The only worked judge in the course calls an undefined `client`, no provider is ever imported or named in any s8 lesson, and no lesson or lab says where an API key comes from. Lab 8 Part D (12 of its 45 minutes, 20% of the grade) plus Part E's ablation depend entirely on that call, and the lab lists "an API key in the source or in the history" as an automatic deduction — so it assumes a key exists that the course never provisions.

**Evidence.**
```
Running `evaluating-generation.md`'s judge block verbatim: `NameError: name 'client' is not defined`. `grep -rn "anthropic|openai|api_key|API key" s8-nlp-2/*.md` returns only `lab-8.md:178: - an API key in the source or in the history`. The only imports anywhere in s8 are `peft`, `sacrebleu`, `bert_score`, `sentence_transformers` — no model-provider client. `mlp-project/project-tracks.md` separately assumes "A local judge harness — an open-weights model scoring a set you labelled", which the lessons never show either.
```

**Proposed fix.** In `evaluating-generation.md`, replace the first line of the judge snippet (`JUDGE_PROMPT_V2 = Path(...).read_text()`) with a three-line preamble that makes it executable: `import os, json; from anthropic import Anthropic; client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])` followed by `JUDGE_PROMPT_V2 = Path("prompts/faithfulness_v2.txt").read_text()`. Then add one sentence under the snippet naming the fallback: "If you have no hosted key, the same four properties hold for a local judge — `AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen2.5-1.5B-Instruct\")` with `do_sample=False` — and Lab 8 accepts either, provided the model id is logged." Add to Lab 8's Setup block a line stating which of the two the school supplies, and where the key comes from if it is the hosted one.

---

## 3. [blocker / competition / fix in platform] COLLAPSE of findings 54, 68, 80 — dataset_file rows for datasets 6, 7, 8 (competitions 172, 173, 174)

**Problem.** Three agents independently concluded the data for three competitions is gone and the competitions are dead. The bytes are intact in R2; only the storage_backend column was never flipped during the R2 migration, so the backend signs GCS URLs for objects that were deleted after the copy. One root cause, one UPDATE, three blockers retired.

**Evidence.**
```
`gcloud storage ls gs://rlarena-417509-render-experiments-eu/prod/datasets/{6,7,8}/...` — "One or more URLs matched no objects" for all six keys. Same keys in R2 via the backend pod's own credentials: prod/datasets/6/ -> X_test.csv 624783, X_train.csv 2493985, y_train.csv 108474; prod/datasets/7/ -> test_images.npz 7439321, train_images.npz 29722203, y_train.csv 246129; prod/datasets/8/ -> test.csv 1422816, train.csv 5740082. Every size is byte-identical to the DB's file_size_bytes. DB shows these 8 rows at storage_backend='gcs' while datasets 9/10/11 are 'r2'. backend/app/dataset_storage.py:24-34 branches purely on that column.
```

**Proposed fix.** `UPDATE dataset_file SET storage_backend='r2' WHERE dataset_id IN (6,7,8)` — 8 rows — then GET /api/competitions/172/datasets with a user token and fetch one signed URL to confirm 200. Do not regenerate or re-upload anything.

---

## 4. [major / baseline / fix in courseware] all 21 competitions — the uniform contract to adopt

**Problem.** There is no uniform baseline block. Six pages carry a measured ladder (176/177/178/181/182 + 172's one-liner), four carry a number with nothing behind it (43/48/168 and 171's ceiling), nine carry only a metric name, and two carry nothing. A student cannot form a single habit for reading a target.

**Evidence.**
```
Compare 177 ("## Baselines — Measured, not asserted — 339 real 6-hourly runs replayed over June–August 2026, 119,520 scored samples. Reproduce them with `test_local.py`." + a 5-row table + "two agents within about 0.006 of each other are not distinguishable") against 173 ("The starter baseline flattens the pixels into a random forest.") and 170 ("Write your markdown here."). Same course, same student, same week.
```

**Proposed fix.** Adopt one block, placed immediately under the H1 of every overview.md, before any prose:\n\n> **Metric** — <metric name>. **<Higher|Lower> is better.**\n> **Baseline** — `<starter-name>`, the starter shipped with this competition, scores **<B>**. It is on the leaderboard under that name.\n> **Reference** — `<reference-name>` scores **<R>**.\n> **Passing this lab** — **<M>**.\n> Measured on <sample description, date>. Two scores closer than **±<e>** are not distinguishable.\n\nRules: every one of B, R, M is a number in the ranked metric's own units; the metric name must equal the competition's `evaluation_metric`; the direction is stated in words, never inferred; `<e>` is a real sampling-error estimate, not a guess (177 and 178 already do this). Where a competition has no meaningful passing bar (177, 178), the line becomes "**Passing this lab** — any scored submission above the baseline." Where a stated number is a ceiling rather than a baseline (171's $65M, 168's ~24), keep it but label it "**Ceiling**" on its own line so it cannot be mistaken for a target.

---

## 5. [major / validation / fix in courseware] the 17 competitions outside courseware/competitions/ (177 176 172 8 173 47 174 178 165 48 49 43 65 169 171 168 170)

**Problem.** The build-time benchmark guarantee only covers the four packaged competitions. The other 17 have no mechanism that keeps their overview's number true, and 13 of them have no number at all.

**Evidence.**
```
courseware/competitions/ contains exactly s1-textstats, s2-readability, s3-adult-income, s4-mnist-warmup. The 17 others are owned outside the repo, and their reference scores live only as an undocumented `__benchmark__` leaderboard row (present on all 17: e.g. 176 -> 0.64387, 172 -> 0.594419, 173 -> 0.040759, 174 -> 0.020908, 165 -> 0.0, 171 -> 27,100,000, 168 -> -99.946, 43 -> -266.62).
```

**Proposed fix.** Extend the same guarantee over the public read API instead of the creator benchmark. Add `courseware/competitions/external/<id>.yaml` per competition holding exactly {competition_id, metric, direction, baseline_agent_name, baseline_score, reference_score, target_score, measured_on, resolution} plus the overview.md body, and add a `build_competitions.py verify-external` subcommand that for each id: (a) GETs /api/competition_asset/{id}/markdown/overview and asserts the Baseline block parses and its numbers match the yaml to tolerance; (b) GETs /api/leaderboard/competition/{id} and asserts a row whose AgentName equals `baseline_agent_name` exists with MeanReward equal to `baseline_score` within tolerance. Both calls need only a student-scope token, so this runs in CI (.github/ already exists) on every content change and fails the build when a page drifts from the board. Wire it into `make publish` alongside the existing `build`.

---

## 6. [major / platform / fix in platform] backend leaderboard payload + evaluation.metrics_schema for competitions 172 8 173 47 174 165 48 49 43 65 169 171 168 170

**Problem.** 14 of the 17 reachable competitions return MetricsSchema: null, so the leaderboard's metric label is frequently wrong and the direction of the metric is declared nowhere a student or the SDK can read.

**Evidence.**
```
Leaderboard row fields, student token: 48 (CartPole) `Metric: "accuracy"` with MeanReward 500.0; 47 (CarRacing) `Metric: "accuracy"` with MeanReward -33.876; 49 `Metric: "accuracy"` with -200.0; 65 `Metric: "accuracy"` with 0.40; 173 and 174 `Metric: "reward"` while their overviews promise F1-macro; all fourteen have `MetricsSchema: null` and `MeanMetricsDetail: null`. By contrast 176/177/178 carry a populated schema, e.g. 177: `[{"key":"reward","label":"Skill","higher_is_better":true,"is_ranking":true,...}]`.
```

**Proposed fix.** Populate `evaluation_metrics_schema` for the fourteen via `client.update_settings(cid, evaluation_metrics_schema=[...])` with the correct `label` and an explicit `higher_is_better`, exactly as the courseware packages already do (config.py `metrics_schema`). Minimum per competition: one entry with `key: "reward"`, the true label ("Episode reward" for 43/47/48/49/168/169, "F1-macro" for 173/174, "Correct answers (of 50)" for 165, "USD raised" for 171, "Accuracy" for 172/8, "Elo" ranking flag for 65/169), `is_ranking: true`, and `higher_is_better`. This is also the field the overview contract should read its metric name from, so the page and the board can never disagree again.

---

## 7. [major / baseline / fix in courseware] competitions 8 173 174 48 165 171 (measured reference already on the board, absent from the page)

**Problem.** For six competitions the baseline number the overview omits is already sitting on the public leaderboard as an unlabelled `__benchmark__` or starter row. The information exists; it was simply never written down.

**Evidence.**
```
Live boards, student token: 173 `rf-pixel-baseline` = 0.748911 (rank 23/26) vs an overview that says only "The starter baseline flattens the pixels into a random forest"; 174 `tfidf-logreg-starter` = 0.479719 (rank 9/27) vs "a TF-IDF + linear model is the starter baseline"; 8 `random0` = 0.0958 and `__benchmark__` = 0.0989 vs no number at all; 48 `qa-cartpole-random` = 23.3 ± 9.81 and `qa-cartpole-linear` = 500.0 vs no number; 171 `__benchmark__` = 27,100,000 vs only a $65M ceiling; 165 `__benchmark__` = 0.0 vs "see Agent.py for a SmolLM3-3B baseline".
```

**Proposed fix.** Write these six numbers into the Baseline block of each overview, quoting the leaderboard row name so a student can verify it: e.g. for 173, "Baseline: `rf-pixel-baseline` — flattened pixels into a random forest — scores 0.749 F1-macro. Higher is better."; for 48, "Baseline: a random policy scores 23.3 ± 9.8 mean episode reward; a linear policy scores 500.0, the environment's ceiling." Then register the same numbers in the external yaml from the verify-external finding so CI keeps them true.

---

## 8. [major / baseline / fix in courseware] competition 165 (GSM8k)

**Problem.** The overview offers a named reference agent as the thing to measure against, and that agent scores zero. The score scale is also never stated in the open.

**Evidence.**
```
Overview: "See `agent_template.py` for the minimal interface and `Agent.py` for a SmolLM3-3B baseline with batched generation and 3-shot prompting." Live board: `__benchmark__` = 0.0 at rank 833 of 866; rank 1 = 18.0. The maximum is derivable only by multiplying two facts buried in a paragraph ("K=5 disjoint subsets of 10 questions each") — the page never says "out of 50".
```

**Proposed fix.** Either repair the reference agent and requote its real score, or drop the claim and quote what is actually measurable: "Metric: correct answers, out of a maximum of 50 (5 subsets x 10 questions). Higher is better. Baseline: random numeric guesses score 0. A cached 1.5B instruct model with 3-shot prompting scores <N>. Best so far: 18. Note the first failed or timed-out batch stops all later subsets, so a score of 0 usually means a crash, not a wrong answer." Also state the 50-question ceiling in the Scoring section.

---

## 9. [major / baseline / fix in both] competitions 8, 43, 47, 48, 49, 65, 165, 169, 170, 173, 174 (MS2A modules s1-s10, mlp-project, mlp-reference)

**Problem.** Eleven of the seventeen reachable MS2A competitions name a starter baseline but give no number. A student reads 'a minimal random-action baseline you can deploy in a couple of minutes' and has no way to tell whether their score is good, or even whether their pipeline is wired up. The three competitions that do it right (172, 176, 178) prove the house style exists and is not being applied.

**Evidence.**
```
Sweep of /api/competition_asset/<id>/markdown/overview for all 17: comp 8 -> only 'a minimal random-label baseline'; 173 -> 'The starter baseline flattens the pixels into a random forest'; 47/48/49/43 -> 'a minimal **random-action** baseline'; 65 -> 'a minimal baseline that plays a random legal move'; 174 -> 'TF-IDF + linear model ... is the starter baseline' with no score; 165 -> 'a minimal answer() baseline (random guesses)'. Contrast comp 178, which ships a five-row measured table (`hasard uniforme (1/30) | 0.0327`, `agent.py fourni | 0.2200`, `TF-IDF n-grammes | 0.3500`) plus a standard-error caveat, and comp 176 ('Repères': constante 0,500, Benchmark 0,644, forêt aléatoire 0,813, ±0,016).
```

**Proposed fix.** For each of the eleven, run the starter agent and one competent reference, then add a 'Baselines' table to overview.md in comp-178 style (row per strategy, measured score, and the sampling error on the eval size) and push with client.set_competition_markdown. Record the reference number in the new CompetitionConfiguration.benchmark_score field once it exists.

---

## 10. [major / content / fix in courseware] s7-nlp-1/lab-7 (Part D) and s7-nlp-1/the-transformer

**Problem.** Lab 7 Part D's code does not run, and the ~30 lines it silently assumes are taught nowhere in the course. `TrainingArguments` appears exactly once in all 73 published lessons — in this snippet — and `Trainer` appears zero times; no lesson shows a tokenised HF Dataset, a data collator, or `compute_metrics`. Session 4 taught a hand-written PyTorch loop instead, so a student following the course has no path from the snippet to a trained model.

**Evidence.**
```
On a clean env (`transformers` 5.16.1, `torch` 2.7.1), running only the two statements Part D gives: `ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=1.1.0`: Please run `pip install transformers[torch]`...`. `grep -rn "TrainingArguments|Trainer|load_dataset" --include="*.md"` over the whole course returns exactly one hit: `s7-nlp-1/lab-7.md:101`. Lab 7 has no environment or requirements block at all.
```

**Proposed fix.** Add a section "Fine-tuning with `Trainer`" to `the-transformer.md`, after "Encoder-only: BERT", showing the four missing pieces end to end: a `torch.utils.data.Dataset` wrapping `tok(texts, truncation=True, max_length=..., padding=False)`, `DataCollatorWithPadding(tok)`, a `compute_metrics(p)` returning macro-F1, and `Trainer(model=..., args=..., train_dataset=..., eval_dataset=..., data_collator=..., compute_metrics=...)` then `.train()`. Add to Lab 7's Setup block the line `pip install "transformers[torch]" scikit-learn matplotlib` (the `[torch]` extra is what pulls `accelerate`).

---

## 11. [major / content / fix in courseware] s7-nlp-1/lab-7 (Part E)

**Problem.** Part E's two-line attention-map snippet raises on any current transformers install, because the default `sdpa` attention kernel never materialises attention weights and returns an empty tuple rather than raising. The student gets a warning that scrolls past and then an IndexError with no connection to its cause. Part E is 6 minutes and 10% of the grade.

**Evidence.**
```
With `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)` (transformers 5.16.1):
  `[transformers] `sdpa` attention does not support `output_attentions=True`. Please set your attention to `eager`...`
  then `out.attentions[-1][0, head]` -> `IndexError: tuple index out of range` (`out.attentions == ()`).
Adding `attn_implementation="eager"` to `from_pretrained` fixes it — verified: `attentions[-1][0,4].shape == (9, 9)`, rows sum to 1.0.
```

**Proposed fix.** In Part D's model line, change `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_classes)` to `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_classes, attn_implementation="eager")`, and add one sentence to Part E above its snippet: "Attention weights come back only when the model was loaded with `attn_implementation=\"eager\"`; the default fused `sdpa` kernel never materialises the n x n matrix and returns `attentions=()` with a warning, not an error."

---

## 12. [major / competition / fix in both] s7-nlp-1/lab-7 and s8-nlp-2/lab-8 (competitions 174, 178, 165)

**Problem.** Neither lab mentions its attached competition. The deliverable of both labs is "a merged PR in your project repository" that the platform never sees, so a student has no externally-verified way to know they got it right — which is exactly the failure the course owner is asking about. The platform already has the mechanism to fix this and it is used nowhere.

**Evidence.**
```
`grep -ci "competition|174|178|leaderboard|ml-arena|submit" s7-nlp-1/lab-7.md` = 0. In `lab-8.md` the only hit is line 188, "Point it at your submission before the leaderboard does" — no id, no link. Meanwhile `_module.json` for s7 lists competitions 174 and 178 and s8 lists 165. `backend/app/services/lesson_directives.py:203-205` registers the directive types `competition`, `leaderboard` and `submit` (fenced as ```mlarena:competition id=174```), and `grep -rn '```mlarena:'` over all 73 published lesson bodies returns 0 hits.
```

**Proposed fix.** Add a "Part F — put it on the board (5 min)" to Lab 7: a ```mlarena:competition id=174``` directive, then "Run the same two models on the Clinical Note Triage data and submit `submission.csv`. The `tfidf-logreg-starter` on that board scores **0.4797 F1-macro** and the best current entry scores **0.6084**. You have done the lab when your fine-tuned model is above 0.4797 and you can reproduce that number locally within 0.02." Add the same shape to Lab 8 pointing at 165 with its own threshold. Do this only after competition 174's dataset is restored.

---

## 13. [major / baseline / fix in both] competition 174 (Clinical Note Triage)

**Problem.** The overview names a baseline but gives no number, so "you have done the lab" is undefined; and the leaderboard column is labelled `reward` rather than the F1-macro the overview promises, so a student cannot tell what 0.48 is or which direction is better. The number the student needs already exists on the board — it is just not written down anywhere a student reads.

**Evidence.**
```
Overview, verbatim and complete on this point: "A **TF-IDF + linear model** (logistic regression / linear SVM) is the starter baseline and is famously hard to beat on clinical text — a good lesson in itself." No figure appears anywhere in the overview. `client.leaderboard(174)`: `Metric == "reward"` and `MetricsSchema is None` for all 27 rows; values span 0.020908 (`__benchmark__`, the constant-class floor) to 0.608398 (`pubmedbert-logreg-v2`), with `tfidf-logreg-starter` at 0.479719. Compare competition 178, which carries a full `MetricsSchema` (`{'key':'reward','label':'Accuracy','higher_is_better':True,'precision':4,...}`) and a five-row measured baseline table in its overview.
```

**Proposed fix.** (a) Add to 174's overview, under "## Task", the table it is missing: `__benchmark__` (constant class) 0.0209 | TF-IDF + logistic regression starter 0.4797 | best entry to date 0.6084, followed by "F1-macro, higher is better, range 0 to 1. Below 0.4797 the transformer has not paid for itself." (b) Set `metrics_schema` on competition 174 so the leaderboard column reads "F1-macro" with `higher_is_better: True` instead of the generic `reward`.

---

## 14. [major / baseline / fix in both] competition 165 (GSM8k)

**Problem.** The GSM8k overview states no target score, no metric range and no direction; it documents a return contract that contradicts the template the platform actually serves; and it points at a reference solution file that is not distributed with the competition and names a checkpoint absent from its own cache list. A student has nothing to aim at and nothing to start from.

**Evidence.**
```
Overview: "Score = number of replies matching the gold answer to within `1e-6`, summed across all delivered subsets" — the 0..50 ceiling has to be inferred from "K=5 disjoint subsets of 10 questions"; the leaderboard's `Metric` is the bare string `score` and the best of 866 agents is 18.00. Contract drift: overview says `agent.answer(questions: list[str]) -> list[float]`, while `competition(165)["agent_template"]` says "expects a tuple: (solutions: list[float], thinking_traces: list[str])". Missing file: "See `agent_template.py` ... and `Agent.py` for a SmolLM3-3B baseline" — `client.datasets(165)` returns `{"datasets": []}`, and `SmolLM3-3B` is not in the overview's own pre-populated-cache list (which has `HuggingFaceTB/SmolLM2-1.7B-Instruct`).
```

**Proposed fix.** In 165's overview: (1) replace the scoring sentence's tail with "Score is the count of correct replies out of the 50 questions sent (5 subsets x 10), higher is better. The provided random-guess starter scores about 0; a cached 1.5B instruct model with 3-shot prompting scores around 10; the best entry to date is 18." (2) change the interface line to the tuple form the template serves: `agent.answer(questions: list[str]) -> tuple[list[float], list[str]]`, noting the bare `list[float]` is still accepted. (3) Either attach `Agent.py` as a competition dataset file or delete the reference and correct `SmolLM3-3B` to `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

---

## 15. [major / competition / fix in platform] competition 165 (GSM8k) — agent_template

**Problem.** The starter agent the platform serves is truncated mid-class: it calls `self._solve_one(q)` but that method was cut off, so the template raises on its very first call. The docstring tells the student to "override" a method that does not exist, which sends them looking for a base class rather than at the missing lines.

**Evidence.**
```
`competition(165)["agent_template"]` ends `'...return solutions, traces\n\n    \n'` (943 chars). Executing it as the platform would: `hasattr(Agent(), "_solve_one") -> False`; `Agent().answer(["Natalia sold clips to 48 friends..."]) -> AttributeError: 'Agent' object has no attribute '_solve_one'`. Its own docstring: "Override `_solve_one` with your own logic."
```

**Proposed fix.** Re-upload the template with the missing method appended, so it runs unchanged and scores the floor:
```
    def _solve_one(self, question: str) -> tuple[float, str]:
        """Replace this. Returns (numeric answer, reasoning trace)."""
        return float(self.rng.randint(0, 100)), "random guess"
```
Verify by exec'ing the served template and calling `answer([...])` without editing it.

---

## 16. [major / validation / fix in courseware] s8-nlp-2/rag and s8-nlp-2/lab-8 (required tests)

**Problem.** Lab 8 requires a test that the chunker in the RAG lesson cannot pass. The lesson's `chunk()` is the only chunker a student is given, the lab grades "Deterministic chunker with overlap and boundary handling" at 15% and "Four tests passing" at 10%, and two clauses of the required test are false against that reference implementation.

**Evidence.**
```
Ran the lesson's `chunk(text, 400, 50)` on a 360-word, 18-paragraph document:
  chunks = 2, word counts [360, 10]
  measured overlap between consecutive chunks: **10** (the test asserts `overlap`, i.e. 50)
  `round-trip == source exactly ?` **False**; `round-trip == " ".join(source.split())` True — `text.split()` destroys the paragraph breaks, so "concatenating the non-overlapping spans reproduces the source text exactly" can never hold, and the lab separately requires the chunker to "split on paragraph boundaries where possible", which makes exact reconstruction harder still. The second chunk is also wholly contained in the first — a duplicate that goes straight into the index.
```

**Proposed fix.** Change the RAG lesson's `chunk` to carry character offsets so reconstruction is over the original string, and stop emitting a tail chunk that is a subset of its predecessor:
```
def chunk(text, size=400, overlap=50):
    assert overlap < size
    words = text.split()
    step, out = size - overlap, []
    for i in range(0, max(len(words) - overlap, 1), step):
        out.append(" ".join(words[i:i + size]))
    return out
```
and rewrite the Lab 8 test docstring to what is actually true: "Consecutive chunks overlap by `overlap` words, and concatenating the non-overlapping spans reproduces the source text up to whitespace normalisation."

---

## 17. [major / content / fix in courseware] s7-nlp-1/lab-7 ("Carry it forward") -> s8-nlp-2/lab-8

**Problem.** Lab 7 closes by promising a Session 8 that does not exist. The promised comparison — a large model zero-shot against the fine-tuned encoder, on the same data — is the natural way a student would check that Session 7 was worth it, and it happens nowhere in Session 8.

**Evidence.**
```
lab-7.md, final lines: "Session 8 keeps the same dataset and asks a different question: what a much larger model gives you without any fine-tuning at all, and what it costs." lab-8.md Part A: "Pick 20–200 documents you care about: your notes, a library's docs, papers, a wiki export. Anything the model cannot already know is better." No s8 lesson or lab revisits a classification dataset, and no s8 lesson compares zero-shot generation against a fine-tuned encoder.
```

**Proposed fix.** Replace those two sentences in Lab 7 with what Session 8 actually does: "Session 8 drops classification and keeps the habits — a measured baseline before anything else, a versioned prompt, a test set you wrote by hand. You will need all three again, against a target where there is no accuracy to compute." If the comparison is wanted, the cheaper fix is to add it as Lab 8 Part A step 0: "Before the corpus, spend five minutes prompting a small instruct model zero-shot on 50 rows of your Lab 7 test set and record its macro-F1 next to the two numbers you already have."

---

## 18. [major / baseline / fix in both] competitions 8, 47, 48, 49, 43, 65, 165, 173, 174, 169, 170, 171

**Problem.** Twelve of the seventeen attached competitions state no numeric floor. Each says only that a Colab notebook holds a 'random-action baseline'. A student cannot tell a working submission from a broken one — the first CartPole agent scoring 21 has no way to know 21 IS the random floor.

**Evidence.**
```
Grepping every overview for a number: comp 48 (CartPole) and 49 (MountainCar) and 47 (CarRacing) and 170 contain only "a minimal **random-action** baseline you can deploy in a couple of minutes"; comp 173 says "The starter baseline flattens the pixels into a random forest" with no score; comp 174 says "A TF-IDF + linear model ... is the starter baseline and is famously hard to beat" with no score; comp 43 quotes "solved around 200" but never the floor. By contrast comp 177 carries a six-row measured table with error bars, 176 states 0.500 / 0.813, 172 states "always guessing alive scores ~0.59. That is the bar to beat."
```

**Proposed fix.** Add a '## Baselines' table to each overview in the exact form comp 177 already uses. Numbers I measured and that can be pasted in today: CartPole-v1 random = 20.98 (sd 10.89, 300 eps) and heuristic `0 if angle+0.5*angvel < 0 else 1` = 500.00; LunarLander-v3 random = -187.54 (sd 115.01, 200 eps); FrozenLake-v1 4x4 slippery random = 0.0120. For 173 and 174, run the shipped starter notebook once and paste its score. Enforce it by adding a `baselines` block to config.py that build_competitions.py refuses to publish without.

---

## 19. [major / competition / fix in courseware] ms2a-machine-learning-practice/s8-nlp-2 (competition 165)

**Problem.** GSM8k is the only S8 competition and it requires downloading and running a 0.5B-3B language model inside a 60-second-per-batch agent container. It is not something a student can attempt in the hour after the lecture, and the competition is known to be exploitable via the answer key on disk.

**Evidence.**
```
Comp 165 overview: "At test time Env draws K=5 disjoint subsets of 10 questions each ... Each subset has a 60-second timeout", followed by a 20-model allowlist (Qwen2.5-0.5B-Instruct through EXAONE-3.5-2.4B). Lab 8 is a RAG lab and produces no GSM8k artefact. Prior finding on record: comp 165 answer-key disk-read exploit, agent 6934.
```

**Proposed fix.** Build 'Math Word Problems' as the S8 ramp — flex_v1, the same `answer(questions) -> (list[float], list[str])` contract as 165 so it is a true dry run of the submission path. Data: scripts/generate_math_dataset.py regenerated at an unpublished seed, 2400 train / 600 private test over six balanced categories. Metric exact match within 1e-6, higher better. Measured: always-0 = 0.0050; last-number heuristic = 0.0150; a 20-minute regex solver = 0.2050; Qwen2.5-0.5B-Instruct 3-shot greedy (12 new tokens) = 0.4550 with 0 unparseable replies.

---

## 20. [major / content / fix in courseware] 92 of 102 lesson bodies across both courses (e.g. s5-computer-vision-1/training-cnns, s7-nlp-1/lab-7)

**Problem.** Teacher speaker notes are embedded as HTML comments in the published lesson bodies. The web renderer hides them, but the SDK and MCP hand students the raw markdown, so students read the instructor's private classroom management notes about them.

**Evidence.**
```
`grep -rl '<!-- notes:' --include='*.md' student_view/ | wc -l` -> 92 of 102 lessons. s5-computer-vision-1/training-cnns.md:8 served to a student token: "<!-- notes: 30 minutes. Show a batch of augmented images on screen before explaining any of it — half the room will spot an augmentation that destroys their own label. -->". s7-nlp-1/lab-7.md:9: "<!-- notes: They will want to start with the transformer. Do not let them ... -->".
```

**Proposed fix.** Strip `<!-- notes: ... -->` blocks in courseware/tools/publish_mlarena.py before the body is sent to the server (the deck builder already consumes them for the PPTX notes pane, so nothing is lost), then republish both courses. Do not rely on the web renderer hiding them — SDK and MCP consumers are first-class per the parity rule.

---

## 21. [major / competition / fix in courseware] ms2a-machine-learning-practice/s7-nlp-1 (competitions 174, 178)

**Problem.** Both S7 competitions are hard first exercises: 174 is clinical-transcription to specialty ranked on F1-macro, 178 is a 30-class daily-refreshed francophone news-source task. Neither states a number, so a student who runs the exact TF-IDF pipeline Lab 7 mandates has nothing to compare against.

**Evidence.**
```
s7-nlp-1/_module.json lists 174 and 178. 174's overview: "A **TF-IDF + linear model** (logistic regression / linear SVM) is the starter baseline and is famously hard to beat on clinical text" — named, never scored. Lab 7 Part B: "Score it on validation, then on test, and write both numbers to reports/metrics.json before you install anything else."
```

**Proposed fix.** Build 'Spooky Author Identification' (file_v1) as the S7 ramp using in-repo data at website/public/modules/llau/tp10/data.csv — 19,579 sentences, 3 authors (EAP 7900 / MWS 6044 / HPL 5635), no duplicate texts; stratified 80/20; metric macro-F1, higher better, accuracy second. Measured: majority class (EAP) = macro-F1 0.1917 / acc 0.4035; TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True) + LogisticRegression(max_iter=1000) = macro-F1 0.8227 / acc 0.8233; char_wb(2,5) = 0.8236. That reference is verbatim the pipeline in Lab 7 Part B.

---

## 22. [major / structure / fix in courseware] COLLAPSE of findings 9, 19, 26, 42, 92, 104, 178 — courseware/content/*/course.yaml estimated_minutes

**Problem.** Seven findings report the same arithmetic from seven angles and none produces the whole table, so the owner cannot see that this is universal rather than a few bad sessions. Not one taught session in either course fits its slot.

**Evidence.**
```
Summed estimated_minutes by lesson kind. Course 14 (billed '12-hour', description 'Every session is half lecture, half lab'): s1 170/60=230, s2 135/45=180, s3 140/45=185, s4 155/45=200, paie-reference 85/0=85; four taught sessions = 795 min = 13.25 h against 12 h, total authored 880 min = 14.7 h. Course 15 (billed 'ten 3-hour sessions', same half-and-half promise): s1 155/45=200, s2 185/45=230, s3 195/45=240, s4 185/45=230, s5 160/45=205, s6 160/45=205, s7 175/45=220, s8 180/45=225, s9 175/45=220, s10 170/45=215, mlp-project 65, mlp-reference 105; ten taught sessions = 2190 min = 36.5 h against 30 h, total 2360 min = 39.3 h. Lecture share is 74-81% everywhere, never 50%. Independently corroborated by the deck build: 83-91 slides per 3-hour session.
```

**Proposed fix.** This is a cut, not an edit — roughly 25% of lecture minutes across the board. Either cut to fit, or change both course descriptions to stop promising 'half lecture, half lab', since no session in either course is within 20 points of that ratio.

---

## 23. [minor / content / fix in courseware] competitions 176 and 178, attached to modules s2-data-preprocessing and s7-nlp-1 of course 15

**Problem.** Two competition briefs are written entirely in French inside a course whose description, all 73 lesson bodies and all module summaries are in English. A student following session 2 or session 7 hits a language switch at exactly the moment the task specification and the scoring contract are handed over.

**Evidence.**
```
comp 176 overview (7513 bytes): 'Ouvrir le notebook de départ dans Google Colab', '## Repères', 'Soumission constante | 0,500'. comp 178 overview (6404 bytes): '# Prédire la source', '## La tâche', '## Le score', 'hasard uniforme (1/30) | 0.0327'. Course 15 description and every lesson in student_view/ms2a-machine-learning-practice/ are English.
```

**Proposed fix.** Either translate 176 and 178 overview.md to English (they are the two best-written briefs on the platform — worth keeping) and push with client.set_competition_markdown, or state the language switch explicitly in the module summary for s2-data-preprocessing and s7-nlp-1 in courseware/content/ms2a-machine-learning-practice/course.yaml.

---

## 24. [minor / content / fix in courseware] s7-nlp-1/lab-7 (Setup vs Part D)

**Problem.** Lab 7 contradicts itself on epochs, and the 12-minute Part D budget is consumed by the training run alone before a line of the missing glue code is written.

**Evidence.**
```
Setup: "CPU is enough for a 3,000-example subset and one epoch." Part D: `num_train_epochs=2`. Measured on an Apple-silicon Mac using MPS (faster than the CPU the lab specifies), 3,000 examples, `max_length=256`, batch 16, DistilBERT: `TRAIN WALLCLOCK seconds: 846.0` (13m32s of steps plus a 60s eval per epoch). Result was worth it — test macro-F1 0.9329 against the TF-IDF baseline's 0.8998 — but not inside 12 minutes.
```

**Proposed fix.** Make Part D's code match its own Setup sentence: `num_train_epochs=1`, and change the Setup line to "One epoch over a 3,000-example subset is about 7 minutes on Apple-silicon MPS and 20+ on a plain CPU — start the run before you write the evaluation code." Re-budget Part D at 20 minutes and take the 8 minutes from Part A (5 -> 3) and Part B (8 -> 6), whose code is given complete.

---

## 25. [minor / content / fix in courseware] s7-nlp-1/lab-7 (Part A)

**Problem.** Part A tells the student to apply metric-selection rules from Session 3. Session 3 has no such rules — it never mentions F1 or class imbalance, and every one of its worked examples scores `roc_auc`. A student who goes back to check finds nothing, which is corrosive because Lab 7's whole grading depends on picking the metric correctly (and competition 174 is a 12-class imbalanced problem scored on F1-macro).

**Evidence.**
```
lab-7.md Part A: "Pick the metric now: macro-F1 if the classes are imbalanced, accuracy if they are not. Session 3's rules have not changed." `grep -rn "F1|f1_score" s3-tabular-models/` returns zero hits. The headings of `s3-tabular-models/model-selection-and-validation.md` are: Three sets three jobs / k-fold cross-validation / Read the spread / Stratified k-fold / Leave-one-out / Group k-fold / Nested cross-validation / Leakage 1 / Leakage 2 / Local CV and the leaderboard — no metric-choice section; `grep -n -i metric` on that file returns nothing. (The same dead pointer exists at `s5-computer-vision-1/training-cnns.md:176`.)
```

**Proposed fix.** Either add a short "Choosing the metric" section to `s3-tabular-models/model-selection-and-validation.md` (imbalanced -> macro-F1 or per-class recall; balanced -> accuracy; ranking-shaped -> ROC-AUC, which is what every s3 example uses and why), or change Lab 7's sentence to stand on its own: "Pick the metric now: macro-F1 if the classes are imbalanced, accuracy if they are not — and report per-class F1 either way, because a macro average hides which specialty you are failing."

---

## 26. [minor / structure / fix in courseware] s7-nlp-1 and s8-nlp-2 (course.yaml estimated_minutes)

**Problem.** Both sessions are budgeted at roughly twice the lecture time the session shape allows, so "half lecture, half lab" cannot happen as authored. This is systemic across all ten sessions, but s7 and s8 are among the worst.

**Evidence.**
```
`_course.json` description: "A 30-hour applied machine learning course in ten 3-hour sessions... Every session is half lecture, half lab." Summing `estimated_minutes` from `_index.tsv`: s7-nlp-1 = 220 min total (175 lecture + 45 lab); s8-nlp-2 = 225 min (180 + 45). Against a 180-minute session with a 90-minute lecture half, that is 1.94x and 2.00x. Every session lands between 200 and 240 minutes.
```

**Proposed fix.** Two concrete cuts that keep the arc: in s7, fold `recurrent-models` (25 min) into `attention` as a 10-minute "what attention replaced" preamble — the lesson's own speaker note already says "this lesson exists to set up the next one" — taking s7 to 160+45. In s8, move `agents` (30 min) into `mlp-reference` alongside the other self-study lessons, taking s8 to 150+45. If the intent is instead that lessons are pre-read, say so in the course description rather than leaving the arithmetic to the student.

---

## 27. [minor / platform / fix in platform] competition 165 (GSM8k), engine 19

**Problem.** The Session 8 competition's GPU VM has been reporting unhealthy all evening, so a student submitting today gets a queued agent and no score, with nothing on the competition page explaining why.

**Evidence.**
```
`client.competition(165)["engine"]` at 20:18 UTC: `{'id': 19, 'k8s_workload_value': 'local_vm', 'vm_health_checked_at': '2026-09-02T20:18:09.238110', 'vm_health_ok': False}`; re-polled at 20:35: `'vm_health_checked_at': '2026-09-02T20:35:55.183258', 'vm_health_ok': False`. The SDK's own docstring for `competition()`: "`vm_health_ok=False` means the GPU VM is currently unreachable and submissions will queue rather than run." The 866-row leaderboard's most recent runs are from 2026-06-23.
```

**Proposed fix.** Bring engine 19's local_vm back up (check the vmapi bridge on the GPU host and re-run the health poll), or surface the state to students: the competition detail already carries `vm_health_ok`, so render a banner on the competition page when it is False rather than silently accepting deployments that queue. Until then, do not attach 165 to a session whose lab week is running.

---

