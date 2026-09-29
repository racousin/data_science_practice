# Embeddings

A token id is an arbitrary integer: id 4102 is not "closer" to id 4103 than to
id 9. An embedding replaces that integer with a dense vector whose geometry
carries meaning.

<!-- notes: 35 minutes. Run the gensim `most_similar` demo live — it takes
ninety seconds and it is the moment the idea becomes concrete. Spend real time
on the "what the king/queen analogy does not prove" slide; it is the one they
repeat wrongly in interviews. -->

---

## One-hot, and why it stops

A vocabulary of size $|V|$ maps token $i$ to the vector with a 1 in position
$i$ and zeros elsewhere.

- dimension $|V|$ — 30,000 upward, almost entirely zeros
- every pair of distinct tokens is orthogonal, so every cosine similarity is 0
- `cat` and `dog` are exactly as related as `cat` and `bureaucracy`
- nothing is shared between rare and frequent words

Session 2 used one-hot for categorical columns with ten levels. It does not
survive contact with a vocabulary.


![image12.png](assets/nlp/image12.png)


---

## Dense vectors instead


![image11.png](assets/nlp/image11.png)


An embedding is a learned function from the vocabulary to a continuous space:

$$
E: \mathcal{V} \rightarrow \mathbb{R}^d
$$

with $d$ typically between 100 and 1024. Every token becomes a row of a matrix
$E$ of shape $|V| \times d$. The coordinates are not interpretable one by one;
the *relative positions* are what the model uses.

---

## The distributional hypothesis

> A word is characterised by the company it keeps. — Firth, 1957

Words appearing in similar contexts get similar vectors. That is the entire
learning signal, and it requires no labels: the corpus supervises itself.

This is why embeddings were the first thing in NLP to scale — the training data
is every sentence ever written, and the target is already inside it.

---


## Vector arithmetic

![Analogies as directions in embedding space](assets/nlp/vector-arithmetic.png)

The famous demonstration: `king - man + woman` lands near `queen`, and
`Paris - France + Italy` lands near `Rome`.

```python
model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
# [('queen', 0.71)]
```

It proves that a consistent direction in the space correlates with a semantic
contrast, learned from raw co-occurrence. That is genuinely surprising.

---


## Cosine similarity

![Cosine similarity by angle](assets/nlp/cosine.png)

Direction carries the meaning; magnitude mostly tracks word frequency. So
compare angles, not distances:

$$
cos(u, v) = \frac{u \cdot v}{\sqrt{u \cdot u} \; \sqrt{v \cdot v}}
$$

Range $-1$ to $1$, with 0 meaning unrelated. In practice, normalise every
vector to unit length once, and the cosine becomes a plain dot product — which
is what every vector database actually computes.

---

## `nn.Embedding` is a lookup table

```python
emb = nn.Embedding(num_embeddings=30522, embedding_dim=768)
ids = torch.tensor([[101, 2054, 2003, 102]])   # (batch, seq_len)
x = emb(ids)                                    # (1, 4, 768)
```

No matrix multiply happens. `nn.Embedding` indexes rows of a parameter matrix —
one-hot times $E$, computed as a gather. The rows receive gradients only for
the ids in the batch, so rare tokens are updated rarely.

That matrix is trained by ordinary backpropagation from the task loss. Nothing
special about it: it is a `Linear` layer whose input happens to be one-hot.

---

## What we want from text


![image41.png](assets/nlp/image41.png)


<!-- placeholder: image to add (three columns: text -> label/number, text -> text, text -> next token) -->

Let $\mathcal{S}$ be the set of token sequences. Almost every NLP task is a
function out of it:

| Family | Signature | Examples |
|---|---|---|
| Text → class | $f: \mathcal{S} \to \{1, \dots, K\}$ | sentiment, topic, spam |
| Text → number | $f: \mathcal{S} \to \mathbb{R}^k$ | price, readability score |
| Text → text | $f: \mathcal{S}_{L_1} \to \mathcal{S}_{L_2}$ | translation, summarisation |
| Text → next token | $P(t_{i+1} \mid t_1, \dots, t_i)$ | generation, dialogue |

All four need the same thing first: a vector for each token that reflects
**the sentence it sits in**. Labels for the first three are scarce. Raw text
is not. The question for the rest of the session is how to get those vectors
without labels.

---

## One vector per word is not enough

A static embedding gives `bank` a single row $E[\texttt{bank}]$:

- *she sat on the **bank** of the river*
- *he deposited cash at the **bank***

Both sentences receive the same vector. Word2vec averages the senses into one
point, weighted by corpus frequency.

What we want is a vector per **occurrence**, computed from the whole sequence:

$$
H = f(t_1, \dots, t_n) \in \mathbb{R}^{n \times d}, \qquad
h_i = \text{contextual vector of token } i
$$

$E[t_i]$ is where token $i$ starts. $h_i$ is what it means *here*. The
transformer of the next lesson is this $f$. The objective below is how $f$
gets trained.

---

## Make the text supervise itself

Hide part of a sentence and ask the model to recover it. The target is already
in the corpus, so no labelling is needed. This is the distributional
hypothesis turned into a loss.

**Causal language modelling (CLM):** predict each token from the ones before it.

$$
\mathcal{L}_{CLM} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(t_{i+1} \mid t_1, \dots, t_i)
$$

**Masked language modelling (MLM):** hide about 15% of the tokens, and
predict each one from everything else, left and right.

$$
\mathcal{L}_{MLM} = -\frac{1}{|M|} \sum_{i \in M} \log P(t_i \mid \tilde{t}_1, \dots, \tilde{t}_n)
$$

Word2vec's CBOW was already a masked objective: predict the centre word from
a window of $\pm c$ neighbours, averaged and order-free. MLM keeps the idea
and lifts both limits. The context is the whole sequence, and word order is
kept.

---

## Why a fill-in-the-blank loss builds representations

> *she sat on the `[MASK]` of the river*

To put probability on `bank` rather than `table`, the vector at the masked
position must encode several things at once:

- **syntax:** a noun after *the*
- **semantics:** something one sits on
- **long-range context:** *river*, four tokens later

That vector is $h_i$. The only way to lower the loss across billions of
sentences is to make every $h_i$ a compressed summary of what its context
implies. Nobody asked for representations. They are the cheapest route to
good predictions.

The objective is a pretext. After pretraining, the prediction head is thrown
away and $H$ is kept.
