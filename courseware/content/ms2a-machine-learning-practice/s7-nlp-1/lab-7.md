# Lab 7 — Fine-Tune a Text Classifier

Fine-tune a small pretrained encoder on a text classification task, against a
TF-IDF baseline that you build first and must beat.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository
reporting both scores on the same test set, and one scored run on the session's
competition (Part F).

<!-- notes: They will want to start with the transformer. Do not let them —
the baseline is 10 lines and it is the number the whole lab is judged against.
Circulate during Part C: the token-length histogram is where they discover
their max_length was throwing away half of every document. -->

---

## Setup

```text
src/nlp/
    data.py        <- load, split, stratify
    baseline.py    <- TF-IDF + logistic regression
    finetune.py    <- tokenize, train, evaluate
    explain.py     <- attention map for one example
tests/test_nlp.py
reports/           <- committed: metrics.json, lengths.png, attention.png
```

Any text classification dataset with two or more classes and at least 2,000
labelled examples. A text column from your own project data is the best choice;
a public set (IMDb, AG News, a French review corpus) is acceptable if the PR
says why.

```bash
uv pip install "transformers[torch]" scikit-learn matplotlib
```

The `[torch]` extra is what pulls `accelerate`; with bare `transformers`,
`TrainingArguments` raises an `ImportError` before a single step runs.

One epoch of DistilBERT over a 3,000-example subset at `max_length=256` and
batch 16 measures **5 minutes on Apple-silicon MPS and 21 minutes on the same
machine's CPU**. Start the run before you write the evaluation code, and if you
only have the CPU, cut the subset rather than the epoch.

---

## Part A — Data and splits (3 min)

Three splits, stratified, made **once** and written to disk:

```python
train, tmp = train_test_split(df, test_size=0.3,
                              stratify=df.label, random_state=0)
val, test = train_test_split(tmp, test_size=0.5,
                             stratify=tmp.label, random_state=0)
```

- the test set is touched exactly twice, at the end, once per model
- deduplicate **before** splitting — near-duplicate reviews across train and
  test is the classic text leak, and it inflates accuracy by several points
- record the class balance of each split in `reports/metrics.json`

Pick the metric now: macro-F1 if the classes are imbalanced, accuracy if they
are not — and report per-class F1 either way, because a macro average hides
*which* class you are failing.

---

## Part B — The baseline you must beat (6 min)

```python
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
pipe.fit(train.text, train.label)
```

Score it on validation, then on test, and write both numbers to
`reports/metrics.json` before you install anything else.

This is not a formality. TF-IDF plus logistic regression beats a badly
fine-tuned transformer on short, topical, single-language text often enough
that a lab without the comparison tells you nothing. If your transformer does
not beat it, that result is reportable — say so and explain why.

---

## Part C — Look at your tokens (7 min)

```python
lengths = [len(tok(t)["input_ids"]) for t in train.text]
```

Produce `reports/lengths.png` — a histogram of token lengths — and report in
the PR:

- the median, the 95th percentile and the maximum
- the fraction of documents truncated at your chosen `max_length`
- the fraction of **total tokens** discarded by that truncation

Choose `max_length` from this figure, not from the default. Then check one
tokenized example by eye: how your domain's vocabulary — product codes, medical
terms, accented names — is being split is information you cannot get any other
way.

---

## Part D — Fine-tune (20 min)

The four objects `Trainer` needs — a tokenizing `Dataset`, a
`DataCollatorWithPadding`, a `compute_metrics` returning macro-F1, and the
`Trainer` itself — are written out in *The Transformer*, section "Fine-tuning
with `Trainer`". Copy that skeleton; this part is the two statements it does not
write for you.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=n_classes,
    attn_implementation="eager")          # Part E needs this; see below
args = TrainingArguments(output_dir="runs", num_train_epochs=1,
                         per_device_train_batch_size=16,
                         learning_rate=2e-5, eval_strategy="epoch",
                         seed=0, report_to="none")
```

Requirements:

- learning rate between 1e-5 and 5e-5 — at 1e-3 the pretrained weights are
  destroyed in one epoch and the model predicts a single class
- evaluate on **validation** at the end of every epoch, and score test exactly
  once. One epoch leaves nothing to select; if yours finishes inside seven
  minutes, run two and keep the better validation macro-F1
- log per-epoch train and validation loss into `reports/metrics.json`
- seed everything, and record the seed

---

## Part E — One attention map (6 min)

Take one correctly classified test example and one misclassified one. For the
first, plot the last-layer attention of one head as a heatmap over its tokens,
save it as `reports/attention.png`, and write **one sentence** of honest
interpretation.

Attention weights come back only when the model was loaded with
`attn_implementation="eager"` — the default fused `sdpa` kernel never
materialises the $n \times n$ matrix and returns `attentions=()` with a warning,
not an error, so the line below fails with `IndexError: tuple index out of
range` and the warning has already scrolled past.

```python
out = model(**enc, output_attentions=True)
a = out.attentions[-1][0, head]          # (seq_len, seq_len)
assert torch.allclose(a.sum(-1), torch.ones(a.shape[0]))   # rows sum to 1
```

Honest means: describe the pattern you see and say what it does and does not
support. "Head 4 concentrates on `[SEP]`, which is an attention sink and tells
us nothing about the label" is a full-credit answer. "The model focused on the
word *terrible*, which is why it predicted negative" is a causal claim that one
heatmap does not license.

---

## Part F — Put it on the board (5 min here, the training run at home)

The session's competition is **Clinical Note Triage**, `competition_id=174` — a
free-text clinical transcription in, one of twelve specialty codes
(`spec_01 … spec_12`) out, ranked on **F1-macro**, higher is better, 0 to 1. It
is Part A to Part D of this lab on somebody else's data.

```bash
uv pip install mlarena-sdk
```

```python
import mlarena, pandas as pd

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.download_dataset(174, "data/clinical/")

train = pd.read_csv("data/clinical/train.csv")   # id, transcription, label
test  = pd.read_csv("data/clinical/test.csv")    # id, transcription

# fit on train, predict test, write submission.csv with the columns id,label
client.submit(competition_id=174, files=["submission.csv"])
print(client.leaderboard(174).head())
```

The board already tells you what a good score is. A constant-class submission —
the row named `__benchmark__` — scores **F1-macro = 0.021**, which is the floor
and not a target. The row `tfidf-logreg-starter` is the Part B pipeline, and it
scores **0.480**: that is the bar, and the point of the lab is that it is hard to
beat on clinical text. The best entry to date, `pubmedbert-logreg-v2`, scores
**0.608**. You have done Part F when your submission is above 0.480 and you can
reproduce that number locally within 0.02.

Two things about that leaderboard that will confuse you otherwise. Its metric
column is labelled `reward` rather than `F1-macro` — same number, generic label.
And if `download_dataset(174, ...)` answers 404, the dataset objects are not
reachable from your account: report it on the course channel rather than
assuming you mistyped the id, and submit to the session's other competition
instead.

That other one is **Prédire la source**, `competition_id=178` — 30 francophone
news sources, a fresh sample of the day's text scored every day, ranked on
**accuracy**. Its brief and its starter agent are written in French; the rest of
the course is not. Measured on its first real batch (100 texts, 2026-09-01):
uniform guessing over 30 classes scores **0.033**, the shipped rule-based
`agent.py` scores **0.220**, TF-IDF words plus logistic regression scores
**0.340**, and TF-IDF character n-grams (2–5) scores **0.350**. The standard
error on 100 texts at that level is about **0.048**, so 0.340 and 0.350 are the
same number — do not build on a gap that size. Unlike 174 this one takes an
`agent.py` with a `predict(request)` method, not a CSV.

---

## Required tests

```python
def test_tokenizer_round_trip():
    """decode(encode(s)) recovers s up to normalisation and special tokens."""

def test_padding_is_masked():
    """Two identical texts padded to different lengths give the same logits."""

def test_output_shape():
    """model(batch of B texts).logits has shape (B, n_classes)."""

def test_determinism():
    """Same seed, same input, two runs -> identical logits."""
```

The padding test is the one that matters: it fails loudly whenever the
attention mask is dropped, and that bug is otherwise invisible. Run the tests
on a fixture of a dozen rows, not on the dataset.

---

## Pull request

The description states:

- both scores on the same test set, baseline and fine-tuned, with the metric
  named
- `max_length`, how it was chosen, and how much text it discards
- one thing the tokenizer does to your domain vocabulary that surprised you
- the attention figure and its one-sentence interpretation
- your competition score, the leaderboard row it beat, and the gap between it
  and your local test number
- what you would try next with an hour of GPU time

---

## Grading

| Criterion | Weight |
|---|---|
| Correct three-way split, deduplicated, test used once per model | 15% |
| TF-IDF baseline built first and reported on the same test set | 20% |
| Token-length figure and a justified `max_length` | 15% |
| Fine-tuning runs, validation scored per epoch, test scored once, seeded | 20% |
| Attention figure with an honest interpretation | 10% |
| Four tests passing | 20% |

---

## Automatic deductions

- test set used for model selection, or scored more than once per model
- no baseline, or a baseline scored on a different split
- model weights or checkpoints committed to git
- a bare `except` around training or tokenization
- an interpretation sentence that claims causality from an attention map
- `max_length` left at the default with no figure behind it

---

## Carry it forward

You now have a fine-tuned encoder, a baseline it beats, and a measured
tokenization budget for your own text.

Session 8 drops classification and keeps the habits — a measured baseline before
anything else, a versioned prompt, a test set you wrote by hand. You will need
all three again, against a target where there is no accuracy to compute.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] `reports/metrics.json` records the class balance of all three splits, and deduplication happened before the split, not after (Part A)
- [ ] `reports/metrics.json` holds the TF-IDF baseline's validation *and* test score, and `git log` shows them committed before any fine-tuning code (Part B)
- [ ] `reports/lengths.png` is committed, and the PR states the median, the 95th percentile, the maximum, and the fraction of total tokens my `max_length` throws away (Part C)
- [ ] The fine-tuning run logged per-epoch train and validation loss into `reports/metrics.json`, the seed is recorded, the learning rate is between 1e-5 and 5e-5, and the test split was scored exactly once (Part D)
- [ ] `reports/attention.png` is committed, and `a.sum(-1)` on the map I plotted is a vector of ones (Part E)
- [ ] All four required tests pass, `test_padding_is_masked` included
- [ ] My submission is on the leaderboard of Clinical Note Triage (#174) — `client.leaderboard(174)` lists my agent name. If its dataset answers 404, Prédire la source (#178) instead.
- [ ] My score beats the baseline: **F1-macro > 0.480**, the `tfidf-logreg-starter` row. On #178 instead: **accuracy > 0.220**, the shipped `agent.py`.

If the last two are not ticked you have not finished the lab, however good the
code is.
