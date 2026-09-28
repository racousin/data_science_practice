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
