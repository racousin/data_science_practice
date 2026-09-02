# Attention

Attention removes the bottleneck by refusing to compress. Instead of one
context vector, the model keeps every position and learns, for each output
step, which inputs to read. It is the single mechanism the rest of modern NLP
is built from.

<!-- notes: 40 minutes, the core of the session. Draw the QK^T matrix on the
board with a five-word sentence and fill in a few cells by hand before showing
any code. If you run short, cut the complexity slide, not the masking one. -->

---

## Stop compressing

A recurrent decoder sees one summary of the source. An attention decoder sees
all $n$ encoder states and forms a **different weighted average of them at
every output step**.

The weights are computed, not stored: they depend on what the decoder is
currently trying to produce. Nothing is forced through a fixed-size channel, so
nothing degrades with length in the way seq2seq did.

Bahdanau et al. (2015) introduced this as an add-on to an RNN. Vaswani et al.
(2017) removed the RNN.

---

## A soft dictionary lookup

A Python dict is a hard lookup: one key matches exactly, you get one value.

```python
d = {"paris": v1, "rome": v2}
d["paris"]        # exact match, all-or-nothing
```

Attention is the same operation made differentiable: compare the query against
*every* key, turn the comparisons into weights that sum to 1, and return the
weighted average of *all* values.

A hard lookup has no gradient with respect to the key. A soft one does, which
is why it can be learned.

---

## Queries, keys and values

![Query, key and value projections](assets/nlp/qkv.png)

Three roles, three learned linear projections of the same input $X$ of shape
$n \times d$:

| Symbol | Role | Read as |
|---|---|---|
| $Q = X W_Q$ | query | what this position is looking for |
| $K = X W_K$ | key | what each position offers as a match |
| $V = X W_V$ | value | what each position contributes if matched |

Splitting "what makes a match" ($K$) from "what gets returned" ($V$) is the
design decision. A single projection would force the matching signal and the
content to live in the same coordinates.

---

## Scaled dot-product attention

![Self-attention over a sequence](assets/nlp/self-attention.png)

$$
Z = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Read it right to left in shapes, with $Q, K, V$ each $n \times d_k$:

- $Q K^T$ is $n \times n$ — the raw compatibility of every position with every
  other
- dividing by $\sqrt{d_k}$ rescales it (next slide)
- row-wise softmax gives $A$, $n \times n$, each row summing to 1
- $A V$ is $n \times d_k$ — one output vector per position

Output shape equals input shape, which is what lets these stack.

---

## Why divide by the square root

For $q$ and $k$ with independent unit-variance components, the dot product has
mean 0 and variance $d_k$. At $d_k = 64$ the scores routinely reach $\pm 25$.

Softmax of a vector with one entry 25 above the rest is a one-hot vector, and
its Jacobian is approximately zero: the gradient dies before training starts.
Scaling restores unit variance:

$$
\mathrm{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

This is not a tuning constant. It is the fix for a saturation failure that
scales with $d_k$, which is why the same $\sqrt{d_k}$ appears in every
implementation.

---

## In code

```python
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v
```

Four lines. In production call `F.scaled_dot_product_attention`, which fuses
the same computation and never materialises the $n \times n$ matrix.

Mask **before** the softmax, with $-\infty$, not after with a multiply. Masking
after leaves the masked mass redistributed and the rows no longer sum to 1.

---

## Reading an attention map

![Twelve BERT heads on one sentence](assets/nlp/head.png)

Rows are queries, columns are keys, brightness is the weight. Each row sums to
1. Head 3 and head 11 above are attending to the next token; head 4 to the
previous one; several heads dump most of their mass on `[CLS]` or `[SEP]`.

That last pattern is an **attention sink**, not a linguistic insight: when a
head has nothing useful to attend to, it parks its probability mass on a
constant position. Attention maps are evidence about the computation, not an
explanation of the prediction. Say that out loud when you show one.

---

## Multi-head attention

![Heads run in parallel and are concatenated](assets/nlp/multihead.png)

Run $h$ attention operations in parallel on $d/h$-dimensional projections,
concatenate the outputs, and apply one more linear map $W^O$.

For $d = 512$ and $h = 8$, each head works in 64 dimensions. Total parameters
are unchanged — the heads split the width rather than adding to it.

One head computes one weighted average per position, so it can express one
relation. Eight heads express eight, and the layer's output is their
concatenation. The heads are not assigned roles; they differentiate because
their random initialisations diverge under the loss.

---

## Self-attention and cross-attention

![Cross-attention between two sequences](assets/nlp/cross_attention.png)

**Self-attention:** $Q$, $K$, $V$ all come from the same sequence. Every
position mixes information from every other. The attention matrix is
$n \times n$.

**Cross-attention:** $Q$ comes from sequence 1 (length $n_1$), $K$ and $V$ from
sequence 2 (length $n_2$). The matrix is $n_1 \times n_2$ and the two lengths
are unrelated.

Cross-attention is exactly the seq2seq fix: the decoder queries the encoder's
states at every step. It is also how a vision-language model lets text query
image patches — Session 6's features on one side, tokens on the other.

---

## Causal masking

![Padding and causal masks](assets/nlp/mask.png)

A language model predicting token $t$ must not see token $t+1$. Enforce it by
setting the upper triangle of the score matrix to $-\infty$ before the softmax:

```python
mask = torch.tril(torch.ones(n, n))     # 1 on and below the diagonal
```

Two distinct masks, both required and often confused:

- **causal** — a property of the *task*, same for every example in the batch
- **padding** — a property of the *batch*, different per row, and derived from
  `attention_mask`

Combine them with a logical AND. Dropping the padding mask lets a short
sequence attend to `[PAD]`; dropping the causal mask leaks the label and gives
you a training loss near zero and a useless model.

---

## The cost

| Step | Time | Memory |
|---|---|---|
| $Q, K, V$ projections | $O(n d^2)$ | $O(nd)$ |
| Scores $Q K^T$ | $O(n^2 d)$ | $O(n^2)$ |
| Weighted sum $AV$ | $O(n^2 d)$ | — |

Quadratic in sequence length, in both time and memory. Doubling the context
quadruples the attention cost. At $n = 512$ the $n^2$ term is minor next to
$n d^2$; at $n = 32{,}000$ it dominates everything else.

This is why context windows were 512 tokens in 2018 and why extending them is
an engineering programme — FlashAttention (never materialise $A$), sliding
windows, sparse and linear approximations. The vocabulary is worth knowing; the
mechanism above is unchanged in all of them.

---

## What it bought

- **Parallel over positions.** No recurrence, so a whole sequence trains in one
  pass. This, not accuracy, is why transformers displaced RNNs.
- **Constant path length.** Position 1 and position 500 are one hop apart, so
  there is no product of Jacobians and no vanishing gradient over distance.
- **Permutation invariance.** Attention treats its input as a *set*. Shuffle
  the tokens and the output is shuffled identically.

That last one is a bug, not a feature. Fixing it is the first thing the next
lesson does.

---

## The rule

> Mask with $-\infty$ before the softmax, and assert that every attention row
> sums to 1.

The failure mode is silent in both directions. A missing causal mask produces a
suspiciously good validation loss; a missing padding mask produces a model
whose output changes when you re-sort the batch. Neither raises. One assertion
on the mask shape at the boundary catches both.
