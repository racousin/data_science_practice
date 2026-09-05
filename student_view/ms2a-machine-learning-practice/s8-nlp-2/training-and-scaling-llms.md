# Training and Scaling LLMs

A large language model is a transformer trained on one task: guess the next
token. Everything else it appears to do — translate, summarise, write code — is
a side effect of doing that well over a large enough corpus.

<!-- notes: 35 minutes. The point to land is the FLOP arithmetic: they must
leave able to say, in one line, why pretraining is not an option for them and
what that forces. Do the 6ND calculation on the board with their laptop's
number. -->

---

## The objective

Session 7 built the transformer block. Stack it, add a causal mask, and train
it to minimise the negative log-likelihood of the next token:

$$
L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{<t}; \theta)
$$

That is ordinary cross-entropy, with the vocabulary as the class set. A 50k
vocabulary is a 50k-way classification problem repeated at every position.

No labels are needed. The text is its own supervision — which is why the
training set can be the internet.

---

## Cross-entropy over the vocabulary

![Softmax over the vocabulary](/api/academic_courses/assets/lessons/103/softmax.png)

An untrained model scores roughly $\log V$ — about 10.8 for a 50k vocabulary.
A trained one lands between 2 and 4. A loss stuck near 10.8 means nothing is
learning; check the data pipeline before the model.

---

## Teacher forcing

Training feeds the *true* prefix at every position, not the model's own output.
All T positions are computed in one forward pass:

```python
out = model(input_ids=ids, labels=ids)   # labels shifted internally
loss = out.loss
```

Position 3 predicts token 4 while conditioning on the real tokens 1–3, even if
the model would have generated something else. Training is parallel and cheap.

It also creates **exposure bias**: at inference the model conditions on its own
output, a distribution it never saw in training. Long generations drift for
exactly this reason.

---

## Weight tying

![Weight tying: input and output embeddings share one matrix](/api/academic_courses/assets/lessons/103/weight-tying.png)

The output projection maps a hidden state back to vocabulary logits. Reusing
the transposed input embedding matrix costs nothing extra:

$$
logits = h E^{T} \in \mathbb{R}^{V}
$$

For $V = 50257$ and $d = 768$ that is 38.6M parameters saved — a third of
GPT-2 small. It also regularises: a token's input and output representations
are forced into the same space. Default to tied embeddings.

---

## The corpus

| Source | Rough share | What it brings |
|---|---|---|
| Filtered web crawl | 60–80% | volume, breadth, noise |
| Code | 5–20% | structure, reasoning transfer |
| Books, papers | 5–10% | long-range coherence |
| Wikipedia, curated | 1–5% | factual density |

Llama 3 saw about 15 trillion tokens; a twenty-year-old human has heard roughly
150 million words. Five orders of magnitude — the strongest argument that these
models generalise differently from people.

---

## What is wrong with the corpus

- **Duplicates.** Near-duplicate documents inflate memorisation and waste
  compute. Deduplication is the single highest-value preprocessing step.
- **Contamination.** Benchmark test sets are on the web. A model that has read
  the answers scores well and teaches you nothing.
- **Bias and toxicity.** It samples what people wrote, not what is true.
- **Personal data.** It is in there, and it can be extracted.
- **Licensing.** Largely unresolved, and increasingly litigated.

None of this is fixed downstream. Instruction tuning changes behaviour; it does
not unlearn.

---

## Perplexity

$$
PPL = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{<t})\right)
$$

Perplexity is the exponential of the cross-entropy loss, and it has a unit you
can reason about: the effective number of equally likely choices the model is
hesitating between at each step.

```python
ppl = torch.exp(out.loss)
```

---

## Reading perplexity

![Perplexity as effective branching factor](/api/academic_courses/assets/lessons/103/perplixity.png)

Five candidates, each at probability 0.2, gives a perplexity of 5. A random
model over a 50k vocabulary has perplexity 50k. A good modern model on web text
sits near 10–20.

**Perplexity is only comparable within one tokenizer and one test set.** Two
models with different vocabularies produce numbers that cannot be ranked
against each other, because they are not counting the same events. This is the
most common misuse of the metric.

---

## Context length

Attention compares every token with every other token:

$$
cost_{attention} \propto L^{2} d \qquad cost_{cache} \propto L\, d
$$

Doubling the context quadruples the attention compute; 4k to 128k is a 1024×
increase in that term. Long context is a product feature, not a free parameter —
and not automatically better, since accuracy degrades in the middle of a long
window.

---

## Scaling laws

Loss falls as a smooth power law in parameters, data and compute. Given a fixed
compute budget C, there is one best split between model size N and tokens D.
The Chinchilla result:

$$
C \approx 6\,N\,D \qquad D \approx 20\,N
$$

The factor 6 is two floating-point operations per parameter for the forward
pass and roughly four for the backward pass. The 20 is empirical: a
compute-optimal model sees about twenty tokens per parameter.

GPT-3 (175B parameters, 300B tokens) was badly undertrained by this rule. A
70B model on 1.4T tokens beat it at a quarter of the inference cost.

---

## What a training run costs

![Training compute of large-scale models over time](/api/academic_courses/assets/lessons/103/flop.jpeg)

Llama 3.1 405B took about $3.8 \times 10^{25}$ FLOPs. A laptop at 200 GFLOP/s
would need six million years. The axis above is logarithmic, and frontier runs
have grown 4–5× per year for a decade.

| Parameters | GPUs | Wall clock | Order of cost |
|---|---|---|---|
| 1B | 10–50 | days | $10k–$100k |
| 10B | 100–500 | weeks | $100k–$1M |
| 100B+ | 1000+ | months | $1M–$100M+ |

---

## You will never pretrain one

> Pretraining is not on your budget. Every decision in this session is about
> what to do with someone else's pretrained weights.

That constraint fixes the order of operations:

1. Choose a checkpoint; read its licence and its training cutoff.
2. Steer it with prompts and retrieved context — no gradients.
3. Adapt it with PEFT only when steering demonstrably fails.
4. Full fine-tuning is a last resort with a hardware bill attached.

The failure mode is jumping to step 4 because it feels like the real work. It is
the most expensive way to get an answer you could have had from step 2.
