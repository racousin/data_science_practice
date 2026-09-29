# Attention

Attention removes the bottleneck by refusing to compress. Instead of one
context vector, the model keeps every position and learns, for each output
step, which inputs to read. It is the single mechanism the rest of modern NN
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

## Shapes and parameters


![image53.png](assets/nlp/image53.png)


<!-- placeholder: image to add (toy layer, n=2, d=4, d_k=3). Walk through it on
the board before the formula slide. -->

Input $X \in \mathbb{R}^{n \times d}$, with $n$ the sequence length (number of
tokens) and $d$ the embedding dimension.

| Weight | Shape | Parameters | Role |
|---|---|---|---|
| $W_Q$ | $d \times d_k$ | $d\,d_k$ | query projection |
| $W_K$ | $d \times d_k$ | $d\,d_k$ | key projection |
| $W_V$ | $d \times d_v$ | $d\,d_v$ | value projection |

| Result | Shape |
|---|---|
| $Q = X W_Q$ | $n \times d_k$ |
| $K = X W_K$ | $n \times d_k$ |
| $V = X W_V$ | $n \times d_v$ |

$Q$ and $K$ must share $d_k$ because they are compared by dot product; $V$ is
free to have its own $d_v$. A single-head layer typically uses
$d_k = d_v = d$; multi-head uses $d_k = d_v = d/h$.

In the figure, $n = 2$, $d = 4$, $d_k = 3$: each $W$ holds 12 parameters and
$Q, K, V$ are $2 \times 3$. No parameter count depends on $n$ — the same
weights process a sequence of any length.

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


![image10.png](assets/nlp/image10.png)


---


## Reading an attention map


![image44.png](assets/nlp/image44.png)

---

## Self-attention

$Q$, $K$, $V$ all come from the same sequence $X \in \mathbb{R}^{n \times d}$.
Every position mixes information from every other.

$$
S = Q K^T \in \mathbb{R}^{n \times n}, \qquad
A = \mathrm{softmax}\left(\frac{S}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}, \qquad
Z = A V \in \mathbb{R}^{n \times d_v}
$$

| Step | Shape | Meaning |
|---|---|---|
| $X$ | $n \times d$ | input tokens |
| $Q, K$ | $n \times d_k$ | what each token seeks / offers |
| $V$ | $n \times d_v$ | what each token carries |
| $S_{ij} = q_i \cdot k_j$ | $n \times n$ | how much token $i$ wants to attend to token $j$ |
| $A$ | $n \times n$ | same, normalised: each row sums to 1 |
| $Z$ | $n \times d_v$ | one mixed vector per token |

With $d_v = d$ the output has the input's shape. Each of the $n$ positions
attends to all $n$ positions.

---

## Cross-attention

![Cross-attention between two sequences](assets/nlp/cross_attention.png)

Queries from sequence 1, keys and values from sequence 2:

- $X_1 \in \mathbb{R}^{n_1 \times d}$ — the sequence asking (queries)
- $X_2 \in \mathbb{R}^{n_2 \times d}$ — the sequence being read (keys, values)

| Step | Definition | Shape |
|---|---|---|
| $Q$ | $X_1 W_Q$ | $n_1 \times d_k$ |
| $K$ | $X_2 W_K$ | $n_2 \times d_k$ |
| $V$ | $X_2 W_V$ | $n_2 \times d_v$ |
| $S$ | $Q K^T$ | $n_1 \times n_2$ |
| $A$ | $\mathrm{softmax}(S / \sqrt{d_k})$ | $n_1 \times n_2$ |
| $Z$ | $A V$ | $n_1 \times d_v$ |

The output length follows the queries: $n_2$ appears only inside $A$ and is
summed out by $AV$. The two lengths are unrelated. If $X_2$ has its own width
$d_2$, only $W_K, W_V$ change to $d_2 \times d_k$ and $d_2 \times d_v$.

Cross-attention is exactly the seq2seq fix: the decoder queries the encoder's
states at every step. It is also how a vision-language model lets text query
image patches — Session 6's features on one side, tokens on the other.

---

## Multi-head attention

![Heads run in parallel and are concatenated](assets/nlp/multihead.png)

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O,
\qquad
\mathrm{head}_i = \mathrm{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)
$$

- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d/h}$ — projections for head $i$
- $W^O \in \mathbb{R}^{d \times d}$ — output projection

One head computes one weighted average per position, so it can express one
relation. Eight heads express eight, and the layer's output is their
concatenation. The heads are not assigned roles; they differentiate because
their random initialisations diverge under the loss.

---

## Multi-head: dimensional flow

| Step | Shape | $d = 512$, $h = 8$ |
|---|---|---|
| $X$ | $n \times d$ | $n \times 512$ |
| $X W_i^Q,\ X W_i^K,\ X W_i^V$ | $n \times d/h$ | $n \times 64$ |
| $S_i,\ A_i$ | $n \times n$ | $n \times n$ (one per head) |
| $\mathrm{head}_i$ | $n \times d/h$ | $n \times 64$ |
| $\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)$ | $n \times d$ | $n \times 512$ |
| $\cdot\, W^O$ | $n \times d$ | $n \times 512$ |

In practice the $h$ projections are one $d \times d$ matrix, reshaped:

```python
q = (x @ W_q).view(B, n, h, d // h).transpose(1, 2)   # (B, h, n, d/h)
```

Each head sees a 64-dimensional subspace. The heads split the width rather
than adding to it — but each head gets its own $n \times n$ attention matrix.

---

## Multi-head: parameters

| Weights | Count | $d = 512$, $h = 8$ |
|---|---|---|
| $W_i^Q, W_i^K, W_i^V$ over $h$ heads | $3 \cdot h \cdot d \cdot \frac{d}{h} = 3d^2$ | 786,432 |
| $W^O$ | $d^2$ | 262,144 |
| **Total** | $4d^2$ | **1,048,576** |

Biases add $4d$ (2,048). The count depends on neither $h$ nor $n$: changing
the number of heads reshapes the same parameters, and sequence length never
enters.

For scale: the feed-forward block that follows (hidden width $4d$) holds
$8d^2$, so attention is about a third of a transformer layer's weights.

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

For sequence length $n$, width $d$, $h$ heads:

| Step | Time | Memory |
|---|---|---|
| $Q, K, V$ projections | $O(n d^2)$ | $O(nd)$ |
| Scores $Q K^T$ | $O(n^2 d)$ | $O(h\, n^2)$ |
| Softmax | $O(h\, n^2)$ | $O(h\, n^2)$ |
| Weighted sum $AV$ | $O(n^2 d)$ | $O(nd)$ |
| Output projection $W^O$ | $O(n d^2)$ | $O(nd)$ |
| **Total** | $O(n^2 d + n d^2)$ | $O(nd + h\, n^2)$ |

Splitting into heads leaves time unchanged ($h \cdot n^2 \cdot d/h = n^2 d$)
but stores $h$ attention matrices. Cross-attention costs $O(n_1 n_2 d)$.

Quadratic in sequence length, in both time and memory. Doubling the context
quadruples the attention cost. Counting constants, the two terms cross near
$n \approx 2d$: at $n = d = 512$ they are comparable; at $n = 32{,}000$ the
$n^2$ term is about 30× the projections. One fp16 attention matrix at that
length is $32{,}000^2 \times 2$ bytes $\approx$ 2 GB — per head, per layer.

This is why context windows were 512 tokens in 2018 and why extending them is
an engineering programme — FlashAttention (never materialise $A$), sliding
windows, sparse and linear approximations. The vocabulary is worth knowing; the
mechanism above is unchanged in all of them.

---

## Parallelization

An RNN computes $h_t = f(h_{t-1}, x_t)$: step $t$ cannot start before step
$t-1$ finishes. Attention has no such dependency — all $n$ outputs come from
the same few matrix multiplications.

| Layer | Time per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |

(Vaswani et al., 2017, Table 1.)

- **Training:** a whole sequence in one pass, as dense matmuls — the workload
  GPUs are built for. More FLOPs than an RNN, far less wall-clock time.
- **Inference:** autoregressive generation is still one token at a time. A
  **KV cache** keeps $K$ and $V$ of past tokens, so each new token computes a
  single query row: $O(nd)$ per step instead of recomputing $O(n^2 d)$, at a
  memory cost of $2nd$ values per layer.

---

## Exercise: attention on DNA

A model reads DNA with one token per nucleotide. It is trained as a **masked
language model** on **batches of reads of variable length**.

1. **Vocabulary.** What is the minimum vocabulary size? List the tokens.
2. **Embeddings.** With embedding size $d = 8$:
   - how many parameters in the token embedding table?
3. **Attention layer.** Single head, $d_k = d_v = d$.
   - shapes of $W_Q, W_K, W_V, W^O$?
   - total parameters, without and with biases?
4. **Forward pass.** Input read: `ACGTTGCAAT` (10 bases).
   Give the shape of: token ids, $X$, $Q$, $K$, $V$, $S = QK^T$, $A$, $Z$,
   and the output after $W^O$.

<!-- notes: give 10 minutes in pairs. The trap is in question 4: the sequence
the model sees is not 10 tokens long. -->

---

## Solution (1/2): vocabulary and parameters

**1. Vocabulary: 7 tokens**

| Token | Why it is required |
|---|---|
| `A` `C` `G` `T` | the data |
| `[PAD]` | variable-length reads in a batch |
| `[MASK]` | masked language modelling objective |
| `[CLS]` | sequence-level representation |


**2. Embeddings**

| Table | Shape | Parameters |
|---|---|---|
| Token embedding | $7 \times 8$ | 56 |

**3. Attention layer ($d = 8$)**

| Weight | Shape | Parameters |
|---|---|---|
| $W_Q, W_K, W_V$ | $8 \times 8$ each | $3 \times 64 = 192$ |
| $W^O$ | $8 \times 8$ | 64 |
| **Total** | | $4d^2 = $ **256** |
| + biases | $4d$ | **288** |



---

## Solution (2/2): forward pass on `ACGTTGCAAT`

The model sees `[CLS] A C G T T G C A A T`: **$n = 11$**, not 10.

| Step | Shape |
|---|---|
| Token ids | $11$ |
| $X$ (token + position embedding) | $11 \times 8$ |
| $Q, K, V$ | $11 \times 8$ |
| $S = QK^T$ | $11 \times 11$ |
| $A = \mathrm{softmax}(S / \sqrt{8})$ | $11 \times 11$, rows sum to 1 |
| $Z = AV$ | $11 \times 8$ |
| $Z W^O$ | $11 \times 8$ |


Output shape equals input shape.
