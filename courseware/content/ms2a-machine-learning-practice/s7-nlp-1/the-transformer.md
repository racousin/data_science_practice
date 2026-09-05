# The Transformer

One block, repeated. Attention mixes information across positions, a
feed-forward network transforms each position independently, residuals and
normalisation keep the stack trainable. Everything since 2017 is this block
with different masking and different scale.

<!-- notes: 40 minutes. Build the block on the board in the order of the
slides, then show the figure — the figure is unreadable as a first exposure.
The parameter-counting slide is the one they thank you for; do it live with
d=768 and let them check 110M against the BERT paper. -->

---

## The block

Two sublayers, in this order, each wrapped in a residual connection:

1. **Multi-head self-attention** — the only place positions talk to each other
2. **Position-wise feed-forward** — two linear layers with a non-linearity,
   applied identically and independently at every position

Everything else is plumbing that makes a deep stack of these trainable.

The feed-forward expands the width to $4d$ and projects back. It holds two
thirds of the block's parameters and is where most of the memorised knowledge
turns out to live.

---

## Residual connections

![A residual connection around a sublayer](assets/nlp/residual_connection.png)

Every sublayer computes $x + f(x)$, never $f(x)$.

The gradient then reaches the input through an identity path as well as through
$f$, so it does not have to survive a product of Jacobians — the same failure
that limited RNNs across time limits deep stacks across layers. Session 4 made
this argument for ResNets; it is the same argument.

It also means a block can learn to do nothing. Initialised near zero, $f$ is a
no-op and the stack starts as an identity function, which is a good place to
start from.

---

## Layer normalisation, and where to put it

$$
LN(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Statistics are taken over the feature dimension of **one position**, not over
the batch. That independence from batch composition is why transformers use
layer norm and not batch norm: sequence models have variable length and small
batches, and batch statistics over padding are meaningless.

| Placement | Form | Behaviour |
|---|---|---|
| Post-norm (2017) | $LN(x + f(x))$ | needs learning-rate warmup; unstable deep |
| Pre-norm (modern) | $x + f(LN(x))$ | trains without warmup, scales to 100+ layers |

Default to pre-norm. Post-norm exists in the paper and in BERT; you will read
it, you should not write it.

---

## Attention has no sense of order

![Positional encoding added to token embeddings](assets/nlp/position.png)

Self-attention is permutation-equivariant: shuffle the input tokens and the
outputs shuffle with them, unchanged. Without position information a
transformer cannot distinguish "dog bites man" from "man bites dog" — the exact
failure that ruled out bag-of-words.

The fix is to inject position into the representation. The embedding for
position $p$ is added to the token embedding before the first block.

---

## Three ways to encode position

**Sinusoidal** (original): fixed, not learned, one frequency per dimension pair.

$$
PE_{p, 2i} = \sin\left(\frac{p}{10000^{2i/d}}\right)
$$

Deterministic, needs no parameters, and extrapolates in principle to lengths
never seen.

**Learned absolute** (BERT, GPT-2): a plain `nn.Embedding(max_len, d)`. Simple,
slightly better in-distribution, and hard-capped at `max_len`.

**RoPE** (Llama, Mistral, Qwen — the current default): rotate $Q$ and $K$ by an
angle proportional to position, so the dot product depends only on the
*relative* offset. It is what makes context extension by interpolation
possible.

---

## The full architecture

![Encoder and decoder stacks](assets/nlp/transformer.png)

The encoder stacks $N$ blocks with unmasked self-attention: every source token
sees every other.

Each decoder block has **three** sublayers instead of two — masked
self-attention over what has been generated, then cross-attention whose queries
come from the decoder and whose keys and values come from the encoder output,
then the feed-forward.

Training uses teacher forcing, so the whole target sequence is processed in one
parallel pass. Inference is still one token at a time.

---

## Three families

![Encoder-only, decoder-only, encoder-decoder](assets/nlp/bertgpt.png)

| Family | Attention | Pretraining | Example |
|---|---|---|---|
| Encoder-only | bidirectional | masked LM | BERT, RoBERTa, DeBERTa |
| Decoder-only | causal | next token | GPT, Llama, Mistral |
| Encoder-decoder | both | span corruption | T5, BART, Whisper |

The families differ in exactly one thing: the mask. Every other component on
the previous slides is shared.

---

## Encoder-only: BERT

![The CLS token as a sequence representation](assets/nlp/cls.png)

Pretraining masks 15% of tokens and predicts them from **both** sides. That
bidirectionality is the point, and it is also why BERT cannot generate: there
is no left-to-right factorisation to sample from.

Fine-tuning puts a linear head on the final `[CLS]` state:

```python
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
```

BERT-base is 12 layers, $d = 768$, 12 heads, 110M parameters. For
classification, retrieval or extraction on a fixed label set, this family is
still the right default in 2026 — smaller, faster and usually more accurate
than prompting a large decoder.

---

## Fine-tuning with `Trainer`

The head is one line; the loop is four more objects. A `Dataset` that tokenizes
without padding, a collator that pads each batch to its own longest sequence, a
metric function, and the `Trainer` that puts them together.

```python
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, max_length=256):
        self.enc = tok(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item
```

Note what is *not* here: `padding`. Padding every document to `max_length` wastes
compute on sequences that are mostly `[PAD]`; the collator does it per batch.

---

## The other three pieces

```python
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

def compute_metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1),
                                 average="macro")}

args = TrainingArguments(output_dir="runs", num_train_epochs=1,
                         per_device_train_batch_size=16, learning_rate=2e-5,
                         eval_strategy="epoch", seed=0, report_to="none")

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=compute_metrics)
trainer.train()
```

`Trainer` needs `accelerate`, which arrives with the `[torch]` extra — install
`"transformers[torch]"`, not bare `transformers`. Without it `TrainingArguments`
raises an `ImportError` saying the Trainer requires `accelerate>=1.1.0`, before a
single step runs.

---

## Decoder-only: GPT

Causal mask, one objective: predict the next token. Every position in the
sequence is a training example, which makes the objective extremely
data-efficient and is most of why this family scaled.

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("gpt2")
m(**tok("The capital of France is", return_tensors="pt")).logits.shape
# (1, 6, 50257) — a distribution over the vocabulary at every position
```

Weight tying reuses the embedding matrix transposed as the output projection:
$|V| \times d$ fewer parameters, and it forces input and output to share one
semantic space. Session 8 takes this family and asks what happens at scale.

---

## Encoder-decoder: T5

Every task is cast as text in, text out:

```text
"translate English to German: Hello"  ->  "Hallo"
"sentiment: I loved it"               ->  "positive"
"summarize: <article>"                ->  "<summary>"
```

One model, one loss, one interface, and the encoder is bidirectional so the
input is read in full before generation starts.

Use this family when the input and output are both sequences and the input
deserves bidirectional reading — translation, summarisation, speech recognition
(Whisper is exactly this shape).

---

## Counting parameters

Per block, with a feed-forward width of $4d$ and biases ignored:

$$
P_{block} = 4 d^2 + 2 d\, d_{ff} = 12 d^2
$$

Four $d \times d$ matrices for $Q, K, V, W^O$; two $d \times 4d$ matrices for
the feed-forward.

For $d = 768$: 7.1M per block, 85M for 12 blocks. Add the embedding matrix,
$30522 \times 768 = 23$M, and you are at the published 110M for BERT-base.

Note what is absent: the number of heads and the sequence length. Heads split
$d$; they do not add parameters. Nothing in the parameter count depends on $n$.

---

## Counting FLOPs

A forward pass costs about $2P$ multiply-accumulates per token — one multiply
and one add per parameter. A training step adds the backward pass, roughly
twice the forward, giving the standard estimate:

$$
C \approx 6 P N
$$

for $P$ parameters and $N$ training tokens. On top of that, attention adds
$O(n^2 d)$ per block, which is negligible at $n = 512$ and dominant at
$n = 32{,}000$.

Use this before you launch anything: 110M parameters over 1B tokens is about
$6.6 \times 10^{17}$ FLOPs — hours on one GPU, not weeks. If your estimate says
weeks, redesign rather than start.

---

## Loading one

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
out = model(**tok("attention is all you need", return_tensors="pt"))
out.last_hidden_state.shape       # (1, 8, 768)
```

`AutoModel` gives the bare encoder stack. The `AutoModelFor...` variants add
the task head and the matching loss. Pick the head, do not write it.

---

## The rule

> Choose the family from the task, not from the leaderboard: bidirectional
> encoder for understanding a fixed input, causal decoder for generating a
> variable output.

The failure mode is a 7B decoder deployed to do binary classification: 60×
the cost of a fine-tuned DistilBERT, higher latency, and usually lower
accuracy — because the small model was trained on your labels and the large one
was not.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   d, n_blocks, vocab = 768, 12, 30522
   print(12 * d ** 2)                        # -> 7077888    one block
   print(n_blocks * 12 * d ** 2)             # -> 84934656   the stack
   print(vocab * d)                          # -> 23440896   the embeddings
   print(n_blocks * 12 * d ** 2 + vocab * d) # -> 108375552  ~ BERT-base's 110M
   ```

2. Nothing in that arithmetic mentions the number of heads or the sequence
   length. Why not, and what *does* depend on the sequence length?

   **Answer.** Heads split $d$ into $h$ projections of width $d/h$ and are
   concatenated back, so they add no parameters. Nothing in the parameter count
   depends on $n$ at all. What depends on $n$ is compute and memory: attention
   costs $O(n^2 d)$ per block, negligible at $n = 512$ and dominant at
   $n = 32{,}000$.

3. Encoder-only, decoder-only and encoder-decoder differ in exactly one
   component. Which one?

   **Answer.** The mask. BERT attends bidirectionally and is pretrained by
   masked language modelling; GPT uses a causal mask and predicts the next
   token; T5 does both, encoder bidirectional and decoder causal with
   cross-attention. Every other component in this lesson is shared.

4. You fine-tune with `Trainer` and the run crashes on `TrainingArguments`
   before any step. What did you install?

   **Answer.** Bare `transformers`. `Trainer` needs `accelerate`, which comes
   with the `[torch]` extra — install `"transformers[torch]"`.
