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

Core Components
The Transformer architecture consists of several fundamental building blocks:

Embedding
Positional Encoding: Injects sequence order information
Self-Attention Mechanism: Computes relationships between all positions in the sequence
Multi-Head Attention: Parallel attention operations with different learned projections
Feed-Forward Networks: Position-wise fully connected layers
Layer Normalization: Stabilizes training
Residual Connections: Enables gradient flow through deep networks

---

## From tokens to vectors






A tokenizer turns text into indices $t_1, \dots, t_n$ with
$t_i \in \{0, \dots, |V|-1\}$. The embedding is a learned lookup table:

$$
E \in \mathbb{R}^{|V| \times d}, \qquad
X = \begin{bmatrix} E[t_1] \\ \vdots \\ E[t_n] \end{bmatrix} \in \mathbb{R}^{n \times d}
$$

Mathematically $X = \mathrm{onehot}(t)\,E$, an $(n \times |V|)(|V| \times d)$
product; in practice it is an index, never a matrix multiplication.

| Vocabulary | $d$ | Parameters $V \cdot d$ |
|---|---|---|
| 30,000 | 512 | 15.36M |
| 30,522 (BERT) | 768 | 23.4M — about a fifth of BERT-base |
| 50,257 (GPT-2) | 768 | 38.6M — about a third of GPT-2 small |

![embedding.png](assets/nlp/embedding.png)

---


## Attention has no sense of order



Self-attention is permutation-equivariant: shuffle the input tokens and the
outputs shuffle with them, unchanged. For any permutation matrix $P$:

$$
\mathrm{SelfAttn}(P X) = \mathrm{SelfAttn}(X)
$$

Without position information a transformer cannot distinguish "dog bites man"
from "man bites dog" — the exact failure that ruled out bag-of-words.

The fix is to inject position into the representation. The embedding for
position $p$ is added to the token embedding before the first block:

$$
X_0 = E[t]  + PE \in \mathbb{R}^{n \times d}, \qquad PE \in \mathbb{R}^{n \times d}
$$

![image43.png](assets/nlp/image43.png)

![RotaryPE2.png](assets/nlp/RotaryPE2.png)

---

## Sinusoidal encoding

![Positional encoding added to token embeddings](assets/nlp/position.png)

<!-- placeholder: image to add (heatmap, positions on y, dimensions on x) -->

One frequency per pair of dimensions:

$$
PE_{p,\,2i} = \sin\left(\frac{p}{10000^{2i/d}}\right), \qquad
PE_{p,\,2i+1} = \cos\left(\frac{p}{10000^{2i/d}}\right)
$$

Pair $i$ has wavelength $2\pi \cdot 10000^{2i/d}$, from $2\pi$ at $i = 0$ to
about $2\pi \cdot 10^4$ at the last pair. Early dimensions oscillate fast and
resolve neighbouring positions; late dimensions move slowly and encode coarse
position — a clock with $d/2$ hands.

Because each pair is a rotation, $PE_{p+k}$ is a fixed linear function of
$PE_p$ for any offset $k$: relative position is linearly accessible to the
attention projections.

Zero parameters, deterministic, and extrapolates in principle to lengths never
seen — in practice, poorly.


---

## The feed-forward network

$$
\mathrm{FFN}(X) = \phi(X W_1 + b_1)\, W_2 + b_2
$$

| Step | Shape | $d = 512$, $d_{ff} = 2048$ |
|---|---|---|
| $X$ | $n \times d$ | $n \times 512$ |
| $W_1$, $b_1$ | $d \times d_{ff}$, $d_{ff}$ | $512 \times 2048$ |
| $\phi(X W_1 + b_1)$ | $n \times d_{ff}$ | $n \times 2048$ |
| $W_2$, $b_2$ | $d_{ff} \times d$, $d$ | $2048 \times 512$ |
| $\mathrm{FFN}(X)$ | $n \times d$ | $n \times 512$ |

**Position-wise** means row $i$ of the output depends only on row $i$ of the
input: the same two matrices applied to $n$ vectors independently. Only
attention moves information between rows.

---



## Layer normalisation, and where to put it

$$
LN(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Statistics are taken over the feature dimension of **one position**, not over
the batch. That independence from batch composition is why transformers use
layer norm and not batch norm: sequence models have variable length and small
batches, and batch statistics over padding are meaningless.

| Quantity | Shape | Computed over |
|---|---|---|
| input $X$ | $n \times d$ | — |
| $\mu, \sigma^2$ | $n$ (one pair per position) | the $d$ features |
| $\gamma, \beta$ | $d$ — learned, $2d$ parameters | — |
| $LN(X)$ | $n \times d$ | — |

| Placement | Form | Behaviour |
|---|---|---|
| Post-norm (2017) | $LN(x + f(x))$ | needs learning-rate warmup; unstable deep |
| Pre-norm (modern) | $x + f(LN(x))$ | trains without warmup, scales to 100+ layers |



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
## The block: dimensional flow

$X_b \in \mathbb{R}^{n \times d}$ is the sequence **entering** block $b$. For
the first block, $X_0 = E[t] + PE$.

**Step 1 — attention sublayer: tokens exchange information.**

$$
H_b = X_b + \mathrm{MHA}\big(LN_1(X_b)\big)
$$

$H_b$ is the **intermediate state** of block $b$: each token's vector, updated
with what it gathered from the other tokens. It is the output of the attention
sublayer, residual included.

**Step 2 — feed-forward sublayer: each token is transformed on its own.**

$$
X_{b+1} = H_b + \mathrm{FFN}\big(LN_2(H_b)\big)
$$

$X_{b+1}$ is the **output** of block $b$, which is also the input of block
$b+1$.

| # | Operation | Tensor | Shape |
|---|---|---|---|
| 1 | normalise | $LN_1(X_b)$ | $n \times d$ |
| 2 | project, per head | $Q_i, K_i, V_i$ | $n \times d/h$ |
| 3 | scores + softmax, per head | $A_i$ | $n \times n$ ($h$ of them) |
| 4 | weighted sum, concat, $W^O$ | $\mathrm{MHA}(\cdot)$ | $n \times d$ |
| 5 | add residual | $H_b$ | $n \times d$ |
| 6 | normalise | $LN_2(H_b)$ | $n \times d$ |
| 7 | expand + activation | $\phi(\cdot\, W_1 + b_1)$ | $n \times 4d$ |
| 8 | project back | $\mathrm{FFN}(\cdot)$ | $n \times d$ |
| 9 | add residual | $X_{b+1}$ | $n \times d$ |


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

## Encoder-decoder: dimensional flow

Source of length $n$, target of length $m$, independent of each other. To keep
the two sides apart, $X$ denotes the encoder and $Y$ the decoder.

**Encoder** — run once per source:

$$
X_0 = E[\mathrm{src}] + PE \in \mathbb{R}^{n \times d}
\;\xrightarrow{\;N \text{ encoder blocks}\;}\;
\mathrm{enc} = X_N \in \mathbb{R}^{n \times d}
$$

**Decoder input:** $Y_0 = E[\mathrm{tgt}] + PE \in \mathbb{R}^{m \times d}$.
This is the target shifted right by one token (teacher forcing).

**Decoder block $b$** takes $Y_b \in \mathbb{R}^{m \times d}$ through three
steps:

$$
\begin{aligned}
S_b &= Y_b + \mathrm{MaskedMHA}\big(LN(Y_b)\big)
  && \text{1. self: each target token reads earlier target tokens} \\
C_b &= S_b + \mathrm{CrossMHA}\big(LN(S_b),\ \mathrm{enc}\big)
  && \text{2. cross: each target token reads the source} \\
Y_{b+1} &= C_b + \mathrm{FFN}\big(LN(C_b)\big)
  && \text{3. transform each token on its own}
\end{aligned}
$$

$S_b$ and $C_b$ are the intermediate states after steps 1 and 2. $Y_{b+1}$ is
the output of decoder block $b$ and the input of block $b+1$. All are
$m \times d$.

| Attention | $Q$ from | $K, V$ from | Matrix | Mask |
|---|---|---|---|---|
| encoder self | source | source | $n \times n$ | padding |
| decoder self | target | target | $m \times m$ | causal + padding |
| cross | target | encoder output | $m \times n$ | source padding |

**Output:** $\text{logits} = Y_N E^T \in \mathbb{R}^{m \times |V|}$. Row $i$
predicts target token $i+1$.

The same `enc` feeds all $N$ decoder blocks, each through its own $W_K, W_V$.
At inference those cross-attention keys and values are computed once per
source and cached.

---

## Back to the vocabulary

The last hidden state $Z \in \mathbb{R}^{n \times d}$ becomes one distribution
over the vocabulary per position:

$$
\text{logits} = Z\,E^T \in \mathbb{R}^{n \times |V|}, \qquad
P(\text{token}_j \mid \text{position } i) = \frac{\exp(\text{logits}_{ij})}{\sum_{k=1}^{|V|} \exp(\text{logits}_{ik})}
$$



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

### Encoder-only: BERT

Pretraining masks 15% of tokens and predicts them from **both** sides. That
bidirectionality is the point, and it is also why BERT cannot generate: there
is no left-to-right factorisation to sample from.

Let $M$ be the set of selected positions ($|M| \approx 0.15\,n$) and
$\tilde{t}$ the corrupted sequence. Of the selected tokens, 80% become
`[MASK]`, 10% a random token, and 10% stay unchanged. The loss is:

$$
\mathcal{L}_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(t_i \mid \tilde{t}_1, \dots, \tilde{t}_n)
$$

Each prediction conditions on the whole corrupted sequence, left and right.
The model outputs logits $\in \mathbb{R}^{n \times |V|}$, but only the $|M|$
selected rows enter the loss. About 15% of positions give a training signal
per pass.

---

### Decoder-only: GPT

Causal mask, one objective: predict the next token. Every position in the
sequence is a training example, which makes the objective extremely
data-efficient and is most of why this family scaled.

$$
\mathcal{L} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(t_{i+1} \mid t_1, \dots, t_i)
$$

One forward pass over $n$ tokens yields logits $\in \mathbb{R}^{n \times |V|}$,
with row $i$ predicting token $i+1$. That gives $n-1$ training signals per
pass against about $0.15\,n$ for BERT.

---

## Exercise: from hidden states to tokens (1/2)

A toy transformer:

- Vocabulary, $|V| = 10$: `[PAD] [MASK] the a cat dog sat on mat rug`
- Width $d = 4$, tied embeddings $E \in \mathbb{R}^{10 \times 4}$
- The last block returns $Z \in \mathbb{R}^{n \times d}$

**A. GPT (next token)** on the sentence `the cat sat on the mat`

1. What is $n$? Shape of $Z$?
2. Logits $= Z E^T$: shape? The softmax is taken over which axis?
3. **Training:** which rows enter the loss, and what is the target of each?
4. **Inference:** prompt `the cat sat`. Which row gives the next token?
   Shape of what you actually use?

<!-- notes: 15 minutes for both parts. The key question is A2: most students
answer "softmax over the sequence". Let them argue before showing the
solution. -->

---

## Solution: GPT

**A1.** $n = 6$, $Z \in \mathbb{R}^{6 \times 4}$.

**A2.** $(6 \times 4)(4 \times 10) = 6 \times 10$. Softmax over the
**vocabulary**: each row is a distribution over 10 tokens.
(Inside attention, the softmax is over the $n$ keys.)

**A3.** Rows 1 to 5, target = the input shifted by one:

| Row | Has seen | Target |
|---|---|---|
| 1 | `the` | `cat` |
| 2 | `the cat` | `sat` |
| 3 | `the cat sat` | `on` |
| 4 | `… on` | `the` |
| 5 | `… the` | `mat` |
| 6 | `… mat` | — no target |

One sentence, 5 training signals ($n - 1$), in one pass. The causal mask
makes it legal: row 3 never sees `on`.

**A4.** $n = 3$, logits $3 \times 10$, but only the **last row** is used:
$1 \times 10$ → `on`. Append it, repeat.

---

## Exercise: from hidden states to tokens (2/2)

**B. BERT (masked tokens)** on the input `the cat [MASK] on the [MASK]`

1. What is $n$? Shape of $Z$? Shape of the logits?
2. **Training:** which rows enter the loss, and what are their targets?
   How many signals, against GPT on the same sentence?
3. **Inference:** input `the dog sat on a [MASK]`. Which row do you read?
   Shape of what you actually use?

---

## Solution: BERT

**B1.** $n = 6$, $Z \in \mathbb{R}^{6 \times 4}$, logits $6 \times 10$,
softmax over the vocabulary — same head as GPT.

**B2.** Only rows 3 and 6:

| Row | Input | Target |
|---|---|---|
| 3 | `[MASK]` | `sat` |
| 6 | `[MASK]` | `mat` |
| 1, 2, 4, 5 | visible tokens | — discarded |

**2 signals** against 5 for GPT. Each mask sees the whole sentence, left
and right.

**B3.** Row 6, the `[MASK]` position: $1 \times 10$ → `rug` (or `mat`).

**Takeaway.** Same head for both: $Z E^T$, softmax over the vocabulary,
one distribution per
