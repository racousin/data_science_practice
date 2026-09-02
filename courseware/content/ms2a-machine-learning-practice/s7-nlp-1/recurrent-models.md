# Recurrent Models and Their Limits

Recurrent networks were the standard answer to sequences for twenty years. They
are worth twenty-five minutes for two reasons: their state-passing idea is
still correct, and the specific way they fail is what attention was invented to
fix.

<!-- notes: 25 minutes, and hold to it — this lesson exists to set up the next
one. Do not implement an LSTM cell on the board. The two things they must leave
with: gradients through time vanish, and a fixed-size context vector is a
bottleneck. -->

---

## One vector carries the past

![Recurrent units unrolled over time](assets/nlp/rnns.png)

A recurrent layer reads one token at a time and keeps a **hidden state** $h_t$
that summarises everything seen so far. The same weights apply at every step —
parameter sharing across time, as a CNN shares across space.

Three consequences: it accepts any sequence length, its parameter count is
independent of that length, and it cannot be parallelised over time. The last
one is what killed it.

---

## The recurrence

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)
$$

```python
rnn = nn.RNN(input_size=300, hidden_size=128, batch_first=True)
out, h_n = rnn(torch.randn(32, 20, 300))
out.shape, h_n.shape       # (32, 20, 128)  (1, 32, 128)
```

`out` is the state at every position; `h_n` is the last one. For
classification you take `h_n`; for tagging you take `out`. Getting this wrong
is the most common first bug in recurrent code.

---

## Backpropagation through time

Training unrolls the loop and backpropagates through every step. The gradient
that reaches step $k$ from step $t$ is a product of Jacobians:

$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$

A product of $t-k$ matrices. If their typical scale is below 1 the gradient
**vanishes** exponentially — the model cannot learn a dependency 100 steps
back. Above 1 it **explodes** and the loss becomes `nan`.

Exploding is the easy one: clip. Vanishing needs an architecture.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## LSTM: a path the gradient can survive

The LSTM adds a **cell state** that is updated additively rather than by a
matrix multiply, so gradients can travel along it almost unchanged. Three
learned gates, each a sigmoid producing values in $[0, 1]$, decide the traffic:

| Gate | Question it answers |
|---|---|
| Forget | What in the cell state is no longer relevant? |
| Input | What of the new step is worth writing? |
| Output | What of the cell state is exposed as $h_t$? |

Four times the parameters of a plain RNN, and it learns dependencies a plain
RNN cannot. The gates are learned, not scheduled — nothing tells the model what
"relevant" means except the loss.

---

## GRU, and how to choose

The GRU merges cell and hidden state and uses two gates — **reset** (how much
past to ignore when proposing an update) and **update** (how much of the
proposal to accept).

| | Parameters | Speed | Long dependencies |
|---|---|---|---|
| RNN | $d_h(d_x + d_h + 1)$ | fast | poor |
| GRU | 3× that | medium | good |
| LSTM | 4× that | slower | best |

For $d_x = 300$, $d_h = 128$: roughly 71k, 214k and 285k parameters. Default to
GRU when you must use a recurrent model at all; the accuracy difference against
LSTM is usually inside the noise, and it trains faster.

---

## Bidirectional

![Forward and backward passes concatenated](assets/nlp/birnn.png)

```python
bi = nn.LSTM(300, 128, bidirectional=True, batch_first=True)
out, _ = bi(torch.randn(32, 20, 300))
out.shape        # (32, 20, 256) — the two directions concatenated
```

Two independent passes, left-to-right and right-to-left, concatenated per
position. Right context disambiguates: "the **bank** of the river" is only
resolvable after the fifth word.

Only legal when the whole sequence is available. Never for autoregressive
generation, and never for streaming — you would be reading the future.

---

## Sequence to sequence

![Encoder, context vector, decoder](assets/nlp/seq2seq.png)

Translation, summarisation and question answering all have output length
independent of input length. The encoder-decoder split handles that: an encoder
consumes the source and produces a **context vector**; a decoder is initialised
from it and emits tokens one at a time until it emits end-of-sequence.

Trained with teacher forcing — the decoder receives the true previous token,
not its own prediction. At inference it receives its own, which is why
generation drifts.

---

## The bottleneck

Everything the decoder will ever know about the source has to fit in that one
fixed-size context vector. A 5-token sentence and a 200-token paragraph get the
same 512 numbers.

The measured symptom, from the 2014–2015 translation literature: BLEU is flat
for short sentences and degrades steadily past roughly 30 source tokens.
Doubling the hidden size moves the cliff; it does not remove it.

> Compressing a variable-length input into a fixed-length vector destroys
> information proportional to the input length.

The fix is not a bigger vector. It is to stop compressing: let the decoder look
back at *all* encoder states and choose which ones matter at each step. That is
attention, and it is the next lesson.

---

## What survives

Recurrence is not gone — it is niche. Reach for a GRU when the sequence is long
and cheap per step (sensor streams, some time series from Session 3), when you
need constant memory per step, or when you must run on a device where an
$O(n^2)$ attention matrix does not fit.

For text, do not start here. A fine-tuned small transformer beats a
from-scratch LSTM on almost any classification task you will meet, with less
code and less tuning.
