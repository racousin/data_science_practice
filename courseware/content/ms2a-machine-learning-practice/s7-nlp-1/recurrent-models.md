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

## MLP Fails with Token Sequences

- **Fixed input size**: can't handle variable-length sequences
- **No positional/context awareness**: treats all inputs as independent features
- **Parameter explosion**: long sequences → massive parameter counts

![slide15_image23.png](assets/nlp/slide15_image23.png)

---

## One vector carries the past

![rnn.png](assets/nlp/rnn.png)

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

At each time step $t$, the RNN:

1. receives input $x_t$
2. uses the previous hidden state $h_{t-1}$
3. applies the same parameters to compute $h_t$
4. produces output $y_t$

This sequential dependency means information flows forward through time, any
sequence length is accepted, and time steps **cannot** be processed in parallel.

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

---

## LSTM/GRU: a path the gradient can survive

![Recurrent units unrolled over time](assets/nlp/rnns.png)

The LSTM adds a **cell state** $c_t$ that is updated additively rather than by
a matrix multiply, so gradients can travel along it almost unchanged. Three
learned gates, each a sigmoid producing values in $[0, 1]$, decide the traffic:

- **forget** $f_t$ — how much of $c_{t-1}$ to keep
- **input** $i_t$ — how much of the new candidate $\tilde c_t$ to write
- **output** $o_t$ — how much of the cell to expose as $h_t$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t)
$$

The GRU merges cell and hidden state and uses two gates — **reset** (how much
past to ignore when proposing an update) and **update** (how much of the
proposal to accept).

| | Parameters | Speed | Long dependencies |
|---|---|---|---|
| RNN | $d_h(d_x + d_h + 1)$ | fast | poor |
| GRU | 3× that | medium | good |
| LSTM | 4× that | slower | best |

For $d_x = 300$, $d_h = 128$: roughly 55k, 165k and 220k parameters. Default to
GRU when you must use a recurrent model at all; the accuracy difference against
LSTM is usually inside the noise, and it trains faster.

---

## Performance

![rnns-diff.png](assets/nlp/rnns-diff.png)

---

## Sequence to sequence: the length problem

An RNN emits one state per input token, so its output length is tied to the
input length $T$:

| Task | Input | Output | What we take |
|---|---|---|---|
| Classification | $T$ tokens | 1 label | $h_T$ |
| Tagging | $T$ tokens | $T$ labels | $h_1, \dots, h_T$ |
| Translation | $T$ tokens | $T'$ tokens, $T' \neq T$ | ? |

"Je suis étudiant" (3 tokens) → "I am a student" (4 tokens). $T'$ is unknown in
advance, and output token $j$ is not aligned with input token $j$.

We need a model of
$$
p(y_1, \dots, y_{T'} \mid x_1, \dots, x_T)
$$
where $T'$ is decided by the model itself.

---

## Encoder–decoder

![Encoder, context vector, decoder](assets/nlp/seq2seq.png)

**Encoder** — read the whole source, keep the last state as the **context vector**:
$$
h_t = f_{\text{enc}}(x_t, h_{t-1}), \qquad c = h_T \in \mathbb{R}^{d_h}
$$

**Decoder** — a second RNN initialised from $c$, fed its own previous token:
$$
s_0 = c, \qquad s_j = f_{\text{dec}}(y_{j-1}, s_{j-1}), \qquad
p(y_j \mid y_{<j}, x) = \mathrm{softmax}(W_o s_j + b_o)
$$

The output factorises token by token:
$$
p(y \mid x) = \prod_{j=1}^{T'} p(y_j \mid y_{<j}, c)
$$

Decoding starts from `<s>` and stops when the model emits `</s>`. That is how
$T'$ becomes independent of $T$: **the length is a prediction**.

---

## Training vs inference

**Training** — cross-entropy on the reference $y^*$, with **teacher forcing**
(the decoder receives the true previous token, not its own prediction):
$$
\mathcal{L} = -\sum_{j=1}^{T'} \log p(y^*_j \mid y^*_{<j}, c)
$$

**Inference** — no reference exists, so the decoder receives its own output:
$$
\hat y_j = \arg\max_{y}\, p(y \mid \hat y_{<j}, c) \quad \text{(greedy; beam search keeps the top } k\text{)}
$$

The mismatch between the two (*exposure bias*) is why generation drifts: one
early mistake is fed back and compounds.
