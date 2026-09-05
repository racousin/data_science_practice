# Generation and Decoding

The model gives you a distribution over the vocabulary. Turning that into text
is a separate algorithm with its own parameters, and it changes the output more
than swapping the model usually does.

<!-- notes: 25 minutes. Live demo: same prompt, same model, temperature 0.2 vs
1.2 vs 2.0. Then top_p 0.9. They should see the failure modes themselves rather
than be told about them. -->

---

## The loop

Generation is a `for` loop around a forward pass:

```python
for _ in range(max_new_tokens):
    logits = model(ids).logits[:, -1, :]
    next_id = pick(logits)                 # the decoding strategy
    ids = torch.cat([ids, next_id[:, None]], dim=-1)
    if next_id.item() == tokenizer.eos_token_id:
        break
```

Everything in this lesson is a different `pick`. The model is unchanged; only
the selection rule moves. Note the asymmetry with training: T positions in one
pass then, T sequential passes now. Generation is inherently serial.

---

## Greedy

$$
\hat{y} = \arg\max_{i} P(w_t = i \mid w_{<t})
$$

Take the highest-probability token, every time. Deterministic given the same
weights and the same numerics, and the cheapest option.

It is locally optimal and globally poor: a high-probability first token can lock
the sequence into a bad continuation, and greedy decoding loops — *the best way
to do this is to do this is to do this*.

Use it for classification-shaped outputs and structured extraction, where there
is one right answer and you want it reproducibly.

---

## Beam search

Keep the `num_beams` most likely *sequences* rather than the most likely token,
and score them by total log-probability.

```python
out = model.generate(ids, num_beams=4, early_stopping=True)
```

It finds higher-likelihood sequences than greedy, which is exactly the problem:
the highest-likelihood text is short, generic and repetitive, because likelihood
is not quality.

Still standard for translation and summarisation, where length is bounded and
fidelity matters. A poor default for open-ended chat, at `num_beams` times the
compute.

---

## Temperature

$$
P(w_t = i) = \frac{e^{z_i / \tau}}{\sum_{j=1}^{V} e^{z_j / \tau}}
$$

Temperature $\tau$ rescales the logits before the softmax. Below 1 it sharpens
the distribution toward the mode; above 1 it flattens it.

| $\tau$ | Effect |
|---|---|
| 0 | equivalent to greedy |
| 0.2–0.5 | factual answers, code, extraction |
| 0.7–1.0 | conversation, general writing |
| > 1.2 | creative, and increasingly incoherent |

Temperature adds no knowledge. It redistributes mass onto tokens the model
already considered — including the bad ones in the tail.

---

## Top-k and top-p

Truncate the tail *before* sampling, so raising temperature does not open the
door to nonsense.

- **Top-k**: keep the k highest-probability tokens, renormalise, sample. Fixed
  count. Too wide when the model is confident, too narrow when it is not.
- **Top-p (nucleus)**: keep the smallest set whose probabilities sum to p.
  Adaptive — one token when the next word is obvious, hundreds when it is open.

```python
out = model.generate(ids, do_sample=True, temperature=0.8, top_p=0.9)
```

Default to top-p at 0.9 with temperature 0.7–0.8. Reach for top-k only when you
want a hard cap on the candidate set.

---

## Repetition penalty

Degenerate repetition is the characteristic failure of sampling-based decoding
from a small model. Two blunt fixes:

- `repetition_penalty=1.1` divides the logit of any already-generated token.
- `no_repeat_ngram_size=3` forbids any trigram from appearing twice.

Both fight a symptom. `no_repeat_ngram_size` will happily break correct text —
it cannot emit "New York" twice in one document. And if you need a penalty above
about 1.2 to get readable output, the model is too small or the prompt too weak.
Fix that instead.

---

## Stopping

A generation stops for one of four reasons, and you must decide all four:

| Criterion | Setting |
|---|---|
| End-of-sequence token | `eos_token_id` — check the model actually emits it |
| Token budget | `max_new_tokens`, not `max_length` |
| A custom string | a stopping criterion on the decoded suffix |
| Wall clock | a timeout in your own code |

`max_length` counts the prompt; `max_new_tokens` does not. Using the first is
how a long prompt silently produces an empty completion. Every production call
needs a token budget and a timeout — an unbounded loop against a paid API is a
billing incident.

---

## KV caching

Each new token re-attends over the whole prefix. Recomputing the keys and
values of every previous token at every step is quadratic waste, so they are
cached:

```python
out = model.generate(ids, use_cache=True)   # the default
```

Step t then costs one token's forward pass plus attention against L cached
entries. The cache is `2 × layers × heads × head_dim × L` per sequence —
gigabytes at long context and large batch.

Generation is therefore **memory-bandwidth bound**: the GPU spends its time
moving weights and cache, not multiplying. Which is why batching raises
throughput almost for free, and why longer context slows generation even when
the prompt is unchanged.

---

## Determinism

`temperature=0` (or `do_sample=False`) removes the sampling randomness. It does
not make generation reproducible.

- Floating-point reductions on a GPU depend on batch size and kernel choice.
- Batched inference means your request shares a batch with other requests.
- Provider-side model versions change under a stable name.
- Ties in the argmax break arbitrarily.

The rule: **pin the model version, set the seed, log both — and still write
tests that assert properties, not exact strings.**

```python
assert json.loads(out)["label"] in {"positive", "negative"}
```

A test comparing generated text to a stored literal fails on a Tuesday for
reasons you cannot reconstruct. Assert the schema, the range, the invariant —
fail fast applied to a stochastic component.
