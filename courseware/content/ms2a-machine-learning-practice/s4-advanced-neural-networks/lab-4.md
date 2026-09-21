# Lab 4 — Building Blocks, Measured

One network, one training function, one Weights & Biases project. The notebook of
this lab changes one building block at a time — activation, normalisation, dropout
and weight decay, optimiser and schedule, loss, an embedding — trains, and sends you
to the W&B workspace to read the difference. The data is MNIST with the pixel
positions and the label meanings permuted, which is what challenge 8, *1 Minute
Permuted MNIST*, hands your agent. The last three parts turn what you measured into
an `agent.py` that trains and predicts inside two 60-second deadlines, test it with
the challenge's own harness, and put it on the board.

**Time:** 60 minutes in the room — about 15 of compute on a Colab CPU, the rest
reading your runs. **Deliverable:** your copy of the notebook run end to end with
`QUICK = False`, a one-sentence answer under each of its nine questions, the link to
your W&B project, and your `agent.py` on the leaderboard of challenge 8.

<!-- notes: The notebook is a solution, not a skeleton: the work is the nine
sentences, each with a number from the workspace, and the agent. Students need two
Colab secrets before the session: WANDB_API_KEY (a free account; the old anonymous
mode is a no-op in current wandb, so without a key the prompt's third choice is
offline logging) and MLARENA_API_KEY. Have them run once with QUICK = True to see
the end before the 15-minute full run. Where they stall: Part 3's train()-mode
BatchNorm cell (they think it is a bug), and Part 9 when they rename a parameter and
the upload is rejected. A submission takes a few minutes to settle; start Part 10
before the debrief, not after. -->

---

## Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s4-building-blocks.ipynb)

- **Two secrets** in Colab's *Secrets* panel, both granted to the notebook:
  `WANDB_API_KEY` (W&B, free account → *User settings* → *API keys*) and
  `MLARENA_API_KEY` (ML-Arena, Profile → API Keys, starts with `mlk_user_`). Never
  in a cell.
- **No W&B key?** `wandb.login()` asks once; *don't visualise* writes the runs to
  `./wandb/` in offline mode, and `wandb sync wandb/offline-run-*` uploads them once
  you have an account. The comparisons below need the workspace, so get the account.
- **Locally:** download the `.ipynb` from the GitHub path of the badge and
  `uv add torch torchvision wandb mlarena-sdk`. The first run downloads MNIST (and
  Fashion-MNIST for the canary) into `./data/`.
- **`QUICK`.** The third cell holds one flag. Run once with `QUICK = True` (a few
  minutes) to see the whole notebook work, then set it back to `False` for every
  number you report.

The notebook trains on a Colab CPU on purpose: the challenge's agent runs on **3 CPU
cores**, and the first line after the imports is `torch.set_num_threads(3)`.

---

## How the notebook is built

`CONFIG` names every block of the network and of its training:

```python
CONFIG = dict(hidden=(256, 256), activation="relu", norm="none", dropout=0.0,
              optimizer="adamw", lr=1e-3, momentum=0.0, weight_decay=0.0,
              schedule="none", loss="ce", label_smoothing=0.0,
              batch_size=128, epochs=8, seed=0)
```

`run({"activation": "gelu"}, name="p2-gelu")` merges the change, builds the model,
trains it, and logs once per epoch to the W&B project `ms2a-s4-building-blocks`:
`train/loss`, `train/acc`, `val/loss`, `val/acc`, `lr`, `grad_norm`, plus the
parameter and gradient histograms of `wandb.watch(model, log="all")`. It returns a
summary that `compare()` prints as a table. Every part is the same move: change one
key, run, open the workspace, compare.

The ablations of Parts 1–7 use 20,000 permuted training images and 10,000 validation
images from the MNIST training split; the 10,000 test images are touched for the
first time in Part 8, as the challenge's `X_test`. Run the whole notebook top to
bottom once; then go back to the part you want to explore.

---

## Part 1 — The baseline (5 min)

Two hidden layers of 256 ReLU units, AdamW at 1e-3, cross-entropy, nothing else.
Open the run link W&B prints and find the four panels every later part refers to:
`train/loss` with `val/loss`, `val/acc`, `grad_norm`, and under *gradients* the
histogram of `0.weight`, epoch by epoch. In the notebook's own run the baseline
reached **0.9635** validation accuracy after 8 epochs.

![Validation accuracy of every ablation of Parts 2 to 6, against the baseline](assets/nn/ablation-validation-accuracy.png)

The figure is every ablation of Parts 2–6 from one run of the notebook, against the
baseline's dashed line; the blue dot is the best of its group. Your numbers will
differ by a few thousandths — the seed is fixed but the platform is not — and the
ordering should not.

---

## Part 2 — Activations (5 min)

Change `activation` to `sigmoid`, `relu`, `gelu`. Put the three `grad_norm` curves on
one panel and set its y-axis to log scale: sigmoid's derivative is at most 0.25, so
its gradient shrinks at every layer. The `dead_units` column of the table is the
share of first-layer units active on fewer than 1% of the validation inputs; the
second cell multiplies the learning rate by ten and shows what that does to it.

![Mean gradient norm per epoch for sigmoid, ReLU and GELU, on a log scale](assets/nn/gradient-norm-by-activation.png)

In the notebook's run the sigmoid network's gradient norm was **2.3×
smaller** than ReLU's at the last epoch and it reached **0.9587** against
**0.9635**; at ten times the learning rate ReLU's dead-unit share went from
**3.9%** to **66.8%** and its accuracy to **0.9472**.

---

## Part 3 — Normalisation (5 min)

Four hidden layers and plain SGD at 0.1, where depth starts to hurt; `norm` set to
`none`, `batchnorm`, `layernorm`. Compare `val/acc` after the first epoch — the
un-normalised network reached **0.9534** after 8, BatchNorm
**0.9597**, LayerNorm **0.9676**.

The second cell is the trap of this session. In `train()` mode BatchNorm normalises
with the statistics of the batch it is given: the same network scored
**0.9597** under `model.eval()` and **0.6475** in `train()` mode
on batches of four, and raised `ValueError` on a batch of one. The `evaluate`
function calls `model.eval()` before every measurement; a validation number computed
without it depends on the batch size.

---

## Part 4 — Dropout and weight decay (5 min)

`dropout=0.3`, `weight_decay=5e-4`, both. The table's `gap` is `train_acc − val_acc`;
in the workspace put `train/loss` and `val/loss` of the four runs on one panel.

![Training and validation accuracy under dropout, weight decay, both and neither](assets/nn/train-validation-gap-regularisation.png)

Dropout closes the gap by lowering the training accuracy, which is the point: the
training number was measured with 30% of the units switched off. Weight decay in
AdamW is decoupled — applied to the weights, not through the gradient — and at
5e-4 it barely moves anything in 8 epochs. Whether the smaller gap comes with a
better validation accuracy is the question under the cell; at this data size it
barely does: 0.9635 with neither and 0.9659 with both, while the gap falls from +0.029 to +0.009. The gap is what you closed; the accuracy is what you wanted.

---

## Part 5 — Optimisation (10 min)

SGD at 0.1, SGD at 0.05 with momentum 0.9, AdamW at 1e-3, AdamW at 3e-3 under a
OneCycle schedule. Look at the `lr` panel of the OneCycle run: it rises for 30% of the
steps, then anneals to almost nothing, and the run ends on the lowest loss. Then the
range test:

![A learning-rate range test: the loss falls, flattens, then explodes](assets/nn/learning-rate-range-test.png)

One pass over the data with the learning rate multiplied by a constant at every
step, from 1e-5 to 1: the loss falls, flattens, and explodes. The rule of thumb is a
constant learning rate one decade below the minimum — here **0.0248** for
SGD with momentum — and a OneCycle peak a little higher. The notebook's momentum run
at 0.05 sits a factor of two above it, and it was the best of the three constant-rate runs — 0.9693 against 0.9614 for plain SGD and 0.9635 for AdamW. The OneCycle run beat all three at 0.9738.

---

## Part 6 — Losses (5 min)

`nn.CrossEntropyLoss` is `log_softmax` then `nll_loss`, in one stable step, and it
expects logits. The first cell shows the equivalence to six decimals, then the most
common PyTorch mistake, numerically: a network that is certain and right has a
cross-entropy of 0.0000, and of **1.4612** if a softmax is applied before the loss —
the floor $\log(1 + 9/e)$ that no amount of training gets under with 10 classes.
Trained that way (`loss="softmax_ce"`) the network still reached
**0.9575**, which is exactly why the bug survives code review. Label
smoothing 0.1 raised the validation loss and raised the accuracy with it, 0.9635 to 0.9755 — the loss is a different loss and is not comparable across the two runs, which is the reason the accuracy panel exists the accuracy.

The second cell fits a line to a target where 5% of the rows are wrong by +20: the
MSE slope came out at **3.374** against a truth of 3, L1 at **3.006**,
Huber at **3.035**.

---

## Part 7 — Embedding (5 min)

The electricity-demand table of Lab 2 has three categorical inputs: the weather
condition (4 values), the month (12) and the weekday (7). Two small networks predict
the demand from them plus temperature and humidity — one with one-hot inputs
(**1,729** parameters), one with `nn.Embedding(n, 2)` per column (**687**). The
counts are close because the cardinalities are tiny; the cell also prints them for
a 10,000-value column into 256 units: **2,560,000** weights one-hot, **164,096** with
an embedding of 16. That is where one-hot inputs stop being an option, not here.

![The learned two-dimensional embeddings of the month and of the day of the week](assets/nn/learned-embeddings-month-weekday.png)

An embedding is a lookup table trained with the rest of the network, so its rows end
up placed by what they do to the target. The notebook logs this figure to W&B with
`wandb.Image`. Do not over-read two dimensions: the weekdays put Saturday far from Wednesday and Friday but do not group the weekend, and the months do not come out in calendar order. What moved is the error — validation RMSE 22.10 with the embeddings against 22.83 one-hot, on a target whose constant-mean RMSE is 51.79.

---

## Part 8 — The 60-second budget (10 min)

From here the notebook uses the full task — 60,000 training and 10,000 test images,
permuted and noised exactly as `env.py` does it — and times what the challenge
times. Three candidates get the same training budget: linear softmax regression, the
baseline MLP, and a 512–256 MLP with BatchNorm. Each trains one epoch at a constant
learning rate to measure the machine, anneals the learning rate to zero over the
epochs that still fit, and stops at the budget whatever happens.

![Accuracy of six agents inside the 60-second budget, against the board's benchmark row](assets/nn/budget-train-seconds-vs-accuracy.png)

The grey dots are the notebook's three candidates on this machine; the coloured
ones are the ladder below, measured through the challenge's own `env.py`. The
deadline is a wall, not a target: the budget in the notebook is 40 s of the 60
because the grading pod is slower than a laptop, and the code adapts the number of
epochs to the machine rather than assuming one.

---

## Part 9 — `agent.py` (10 min)

The `%%writefile agent.py` cell is the winning candidate as an agent; the constants
at its top are the knobs Parts 1–8 were about. Five things in it are rules:

- **The template's signatures.** The upload validator compares your methods with the
  challenge's template by method name *and parameter names*. Keep
  `__init__(self, output_dim: int = 10, seed=None)`, `train(self, X_train, y_train)`
  and `predict(self, X_test)` spelled exactly like that; rename `X_test` and the
  upload is rejected with "wrong signature".
- **A fresh model on every `train()` call.** At 0.98 accuracy the challenge calls
  `train` and `predict` again on a permuted Fashion-MNIST task and expects at least
  0.40 there. A model kept from the previous call, or anything that assumes digits,
  scores −1 as a detected cheat.
- **Statistics from the arrays that arrive**, never hard-coded.
- **The wall-clock guard**: training stops at `TRAIN_BUDGET_S`, mid-epoch if needed.
- **`model.eval()` before predicting**, and `predict` returns a list of ints.

The harness cell is the challenge's `evaluate` loop: both calls timed against 60 s,
the accuracy, and the Fashion-MNIST canary when you pass 0.98 — so a second
`train()` on different data is part of the test. Report its three numbers **with
the thread count and the machine**.

---

## The board

```mlarena:challenge id=8
```

```python
import mlarena
client = mlarena.connect(api_key=MLARENA_API_KEY)          # from Colab Secrets, never a literal
submission = client.submit(8, files=["agent.py"], submission_name="lab4-mlp",
                           runtime={"language": "python", "framework": "torch"}, wait=True)
print(submission["status"]["status"], submission["status"]["last_status_message"])
print(client.leaderboard(8, top=5))
```

The rules, from `env.py`: one task per evaluation; `train(X_train, y_train)` with
`X_train` uint8 `(60000, 28, 28)` and `y_train` int64 `(60000, 1)`, then
`predict(X_test)` on `(10000, 28, 28)`, **each under its own 60 s deadline**; a
timeout or an exception scores 0; **3 CPU cores, 3 GiB, no GPU**; a fresh model per
`train()` call (the canary); no shipped weights — there is nothing to ship, the task
is new every time. The `torch` runtime is torch 2.12 on Python 3.12.

**The ladder**, three agents run through the challenge's own `env.py` on an
Apple-silicon laptop with `torch.set_num_threads(3)`, torch 2.14:

| agent | accuracy | train s | predict s |
|---|---|---|---|
| the template's random labels | 0.0993 | 0.0 | 0.0 |
| linear softmax regression, in torch | 0.9227 | 18.0 | 0.09 |
| the notebook's agent: MLP 512–256 with BatchNorm, AdamW, 40 s budget | **0.9821** | 36.2 | 0.33 |

**The bar is the board's `__benchmark__` row: 0.9256.** That row is the same linear
model written in pure numpy, measured on the grading pod, where it trains in 2.6 s —
the torch row above is the same idea at 18 s and lands in the same place. A linear
model on the pixels is where this session starts, not where it ends. Timings on a
laptop are not the pod's: the guard is what keeps you under 60 s there, and the
margin you leave is your choice.

---

## Grading

| Criterion | Weight |
|---|---|
| Notebook run end to end with `QUICK = False`, nine answers each with its number | 25% |
| W&B project linked, with the Part 2, 4 and 5 comparisons readable in the workspace | 20% |
| Part 8: the three candidates measured, thread count and machine stated | 15% |
| `agent.py`: template signatures, fresh model per call, guard, `eval()` before predict | 15% |
| Part 9 harness passed: both deadlines, canary when applicable | 10% |
| Submission `active` on challenge 8, above the `__benchmark__` row | 15% |

---

## Automatic deductions

- a softmax before `CrossEntropyLoss`, in the notebook or in the agent
- `model.eval()` missing before validation or prediction
- a model kept across `train()` calls, or statistics not computed from the arrays received
- a timing quoted without its thread count and machine
- a method or parameter renamed away from the template's `__init__`, `train`, `predict`
- a key pasted into a cell

---

## Did you validate this session?

- [ ] The notebook ran top to bottom with `QUICK = False`, and each of the nine
      questions has a one-sentence answer with a number from my runs
- [ ] My W&B project holds the runs of Parts 1–9 and I can open the `grad_norm`
      comparison of Part 2 and the loss panel of Part 4
- [ ] I can say why the BatchNorm network scores lower in `train()` mode
- [ ] Part 8's table states the budget, the thread count and the machine
- [ ] `agent.py` keeps the template's three signatures, builds a fresh model in
      `train()`, stops at its budget, and calls `model.eval()` before `predict`
- [ ] The Part 9 harness reports both calls under 60 s and, above 0.98, a canary
      accuracy above 0.40
- [ ] My submission is `active` on 1 Minute Permuted MNIST (#8)
- [ ] My score beats the `__benchmark__` row: **accuracy > 0.9256**

If the last two are not ticked you have not finished the lab, however good the
notebook is.
