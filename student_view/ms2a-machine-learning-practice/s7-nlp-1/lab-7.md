# Lab 7 — Fine-Tune a Text Classifier

Fine-tune a small pretrained encoder on a text classification task, against a
TF-IDF baseline that you build first and must beat.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository,
reporting both scores on the same test set.

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
says why. CPU is enough for a 3,000-example subset and one epoch.

---

## Part A — Data and splits (5 min)

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
are not. Session 3's rules have not changed.

---

## Part B — The baseline you must beat (8 min)

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

## Part D — Fine-tune (12 min)

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=n_classes)
args = TrainingArguments(output_dir="runs", num_train_epochs=2,
                         per_device_train_batch_size=16,
                         learning_rate=2e-5, eval_strategy="epoch", seed=0)
```

Requirements:

- learning rate between 1e-5 and 5e-5 — at 1e-3 the pretrained weights are
  destroyed in one epoch and the model predicts a single class
- select the epoch on **validation**, then score test once
- log per-epoch train and validation loss into `reports/metrics.json`
- seed everything, and record the seed

---

## Part E — One attention map (6 min)

Take one correctly classified test example and one misclassified one. For the
first, plot the last-layer attention of one head as a heatmap over its tokens,
save it as `reports/attention.png`, and write **one sentence** of honest
interpretation.

```python
out = model(**enc, output_attentions=True)
a = out.attentions[-1][0, head]          # (seq_len, seq_len)
```

Honest means: describe the pattern you see and say what it does and does not
support. "Head 4 concentrates on `[SEP]`, which is an attention sink and tells
us nothing about the label" is a full-credit answer. "The model focused on the
word *terrible*, which is why it predicted negative" is a causal claim that one
heatmap does not license.

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
- what you would try next with an hour of GPU time

---

## Grading

| Criterion | Weight |
|---|---|
| Correct three-way split, deduplicated, test used once per model | 15% |
| TF-IDF baseline built first and reported on the same test set | 20% |
| Token-length figure and a justified `max_length` | 15% |
| Fine-tuning runs, epoch selected on validation, seeded | 20% |
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

Session 8 keeps the same dataset and asks a different question: what a much
larger model gives you without any fine-tuning at all, and what it costs.
