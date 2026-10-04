# DS-Harness: a small model, a big toolbox

A language model of 1 to 3 billion parameters can read a question about a spreadsheet, but it
cannot compute a statistic over 600 rows in its head, fit a model or forecast a series. A
**harness** is the program around the model: it routes each task, gives the model tools, runs
the code the model writes, checks the result and recovers from errors. You build the harness.
Measured on the private set: the model answering directly scores **0.6 / 100**; the starter kit
(the model writes Python, one repair) scores **13.9**; an instructor harness with routing,
numpy fit/forecast tools and checks about **61**.

## The tasks and the score

Each task is a question in English, sometimes with CSV files, and one expected answer.

| Level | What it looks like | Scored |
|---|---|---|
| 1 | a unit conversion, a percentage, one statistic of a column | exact, with a tolerance |
| 2 | a word problem with distractors, a probability, filter → group → aggregate → rank over messy files | exact, with a tolerance |
| 3 | forecast a series; predict a target for the rows of a test file | 0 (trivial baseline) to 1 (reference model) |

**Score = 100 × (0.3·L1 + 0.4·L2 + 0.3·L3)**, each level being the mean over its tasks.
The private set has 119 tasks (37 / 67 / 15), with other wordings, other data domains, and
24 tasks from families that are not in `dev.json`: their mean is the *Unseen families* column.
A harness that reads the prompt and the files carries over to them. `schema.md` gives the
answer formats and tolerances. At level 3 the prompt states the method and the format: read it.

## The rules

- **Contract.** `agent.py` defines `class Agent`: `__init__` loads the model (**60 s at most**),
  `solve(tasks)` returns one `{"id", "answer", "trace"}` per task.
- **Time.** 21 `solve` calls (8 level-1/2 tasks or 2 level-3 tasks each), 40 s per full call.
  The job ends **399 s after it starts, model loading included**. A batch that no longer fits
  is not sent and scores 0. Plan for **330 s**: about 2 s per level-1/2 task, 8 s per level-3.
- **Failure = no score.** A `solve` call that raises or misses its timeout ends the run and the
  deployment fails. Wrap each task in `try/except`, return a placeholder, respect
  `time_budget_s`. Going over **3 GiB of RAM** kills the agent: plan for it as a failure too
  (catch `torch.cuda.OutOfMemoryError` per task; GPU memory is not the limit, RAM is).
- **Runtime: choose PyTorch** when you submit (it is not the default). It has torch,
  transformers, accelerate, pandas, numpy, sympy, matplotlib. **No scikit-learn, scipy or
  statsmodels**: Colab has them, the platform does not.
- **Hardware.** One 24 GB GPU (RTX 4090), 3 CPUs, 3 GiB RAM, 128 MB writable `/tmp`, no network.
- **Models.** Only the platform's offline cache, with a pinned `revision=` (`dsh.load_llm` does
  it). As-is: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`,
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2-1.5B-Instruct`, `Qwen/Qwen3-1.7B`,
  `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `HuggingFaceTB/SmolLM3-3B`,
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `allenai/OLMo-2-0425-1B-Instruct`,
  `stabilityai/stablelm-2-1_6b-chat`, `tiiuae/Falcon3-1B-Instruct`,
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. With `trust_remote_code=True`:
  `IndexTeam/Index-1.9B-Chat`, `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`,
  `internlm/internlm2_5-1_8b-chat`, `openbmb/MiniCPM-2B-sft-bf16`.
- **Uploads.** `agent.py` plus the modules it imports (e.g. `dsh.py`): up to 10 files, 100 MB,
  scanned by bandit.
- **Quota.** **2 deployments per person per rolling 24 h, across every ML-Arena challenge,
  failed ones included.** A deployment is a short test run, then the scored run: about 7–9 min
  plus the queue (one GPU, one job at a time). Test locally first. The queue is long in
  the evenings and near the freeze: a deployment **queued** before 2026-11-20 23:59 counts, but
  aim to have your final one in by **2026-11-19**.
- **Pairs.** Work in teams of two; create the team on this page before your first submission.
- **Privacy.** Do not log or store task prompts or files from platform runs.

## Start here

1. **Dataset.** Download `dev.json` from this challenge's dataset: 178 public tasks with answers.
2. **Repository.** [github.com/racousin/ds-harness](https://github.com/racousin/ds-harness):
   the scorer the leaderboard runs, `localtest.py` / `local_eval.py`, `schema.md`, and the kit
   (`dsh.py`, `agent_naive.py`, `agent_kit_baseline.py`).
3. **Notebook.** [Open the starter notebook in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-project-ds-harness.ipynb)
   on a T4 GPU, paste your `mlk_user_` key (Profile page), run it top to bottom.
4. **First submission.** Submit `agent_kit_baseline.py` as `agent.py`, with `dsh.py`:

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              wait=True, timeout_sec=1800)
```

## First levers that pay

- **Time first.** The kit answers 103 of 119 tasks before time runs out: make it faster
  (shorter generations, fewer retries) and it answers them all.
- **The model.** Swapping the kit's model is one line; `Qwen/Qwen3-1.7B` raised an earlier
  version of the kit from 19.5 to about 31.
- **Level 3.** The kit scores 0 there. A numpy fit/forecast tool (~40 lines) is worth a lot.
- **Answer format.** A right value in the wrong format scores 0: read `schema.md`.
- **Measure before adding work.** Adding a repair round *lowered* an earlier kit's score
  (19.5 → 14.7): tasks went unanswered. Measure seconds per task first.

## How it is graded

The project is **50 % of the course grade**. Within it: 25 % leaderboard, marked against fixed anchors on a
final private set regenerated after the freeze (**2026-11-20 23:59**), and 75 % oral. Details,
deliverables and calendar: the [course's Project module](https://ml-arena.com/courses/ms2a-machine-learning-practice/mlp-project).
