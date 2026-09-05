# Embeddings

A token id is an arbitrary integer: id 4102 is not "closer" to id 4103 than to
id 9. An embedding replaces that integer with a dense vector whose geometry
carries meaning, and it is the second half of turning text into tensors.

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

---

## Dense vectors instead

![Words positioned in a learned vector space](assets/nlp/word_embeddings.png)

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

## word2vec

![Skip-gram and CBOW](assets/nlp/word2vec.png)

Two shallow architectures over a sliding context window of 5 to 10 tokens:

- **Skip-gram** — given the centre word, predict the context words
- **CBOW** — given the averaged context, predict the centre word

Skip-gram maximises the log-likelihood of the context:

$$
\mathcal{L} = \sum_{t=1}^{T} \sum_{j \neq 0, |j| \leq c} \log P(w_{t+j} | w_t)
$$

Skip-gram is better on rare words; CBOW trains faster. Both keep the input
matrix $E$ and throw the model away.

---

## Making it tractable, and GloVe

The softmax over $|V|$ in that objective costs $O(|V|)$ per token.
**Negative sampling** replaces it with a binary discrimination between the true
context word and 5–20 sampled decoys — the only reason word2vec was trainable
on a billion words in 2013.

**GloVe** (2014) takes the other route: build the global co-occurrence matrix
first, then fit vectors to reproduce the log counts. The insight is that
*ratios* of co-occurrence probabilities encode meaning — `ice`/`steam` against
`solid` and `gas`.

Both give comparable vectors. **FastText** adds character n-grams, so it can
embed a word it has never seen.

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

## What the analogy does not prove

It does not prove the model reasons. Three caveats, all reproducible:

- the query **excludes the input words** from the answer set; without that
  exclusion the nearest neighbour of `king - man + woman` is usually `king`
- accuracy is high on gender and capital cities, and near chance on most other
  relations
- the same geometry encodes the corpus's biases as cleanly as its semantics —
  occupation analogies reproduce stereotypes, because the text did

Report analogy results with the exclusion rule stated. Otherwise the number
means nothing.

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

## Static versus contextual

![The same word, different vectors by context](assets/nlp/contextual_embed.png)

word2vec, GloVe and FastText give **one vector per word type**. `bank` has an
identical vector in "river bank" and "bank account" — the model averaged the
two senses into a point between them.

A transformer produces **one vector per token occurrence**: the hidden state at
position $i$ has attended to the rest of the sentence, so the two `bank`s
differ. That is the whole payoff of the next three lessons.

---

## Getting a contextual vector

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
out = model(**tok("I sat on the river bank", return_tensors="pt"))
out.last_hidden_state.shape     # (1, seq_len, 768)
```

`last_hidden_state[:, i, :]` is the contextual vector for token $i$. Position 0
is `[CLS]`.

Static embeddings are not obsolete: they are 1000× cheaper, need no GPU, and
still win when you have a vocabulary of product names and no context to speak
of.

---

## Sentence embeddings

![Pairwise cosine similarity of sentences](assets/nlp/Heatmap-cosine.png)

One vector for a whole sentence, obtained by pooling token states — mean over
the *unpadded* positions, or the `[CLS]` state.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
v = m.encode(["The cat sits on the mat", "A feline rests on the rug"])
```

Raw BERT mean-pooling is a weak sentence encoder. Models like MiniLM are
*fine-tuned on sentence pairs* for exactly this, and the gap is large. Use a
sentence-transformer, not a bare encoder, when the output is a similarity.

---

## Vector search

![Semantic clusters in embedding space](assets/nlp/semantic-clustering.png)

Embed a corpus once, store the vectors in an index, embed the query, return the
nearest neighbours. Exact search is a matrix product; above a few hundred
thousand vectors, use an approximate index (FAISS, HNSW) and accept ~99%
recall for a 100× speedup.

This is retrieval by meaning rather than by keyword, and it is the mechanism
underneath RAG — Session 8 builds on it.

---

## The rule

> Embed with the same model on both sides of a comparison, normalise, and
> compare with cosine.

The failure mode is an index built with one model and queried with another, or
queried with a differently normalised vector. Similarities come back in a
plausible range, the ranking is noise, and no exception is ever raised. Assert
the embedding dimension and the model name next to the index.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import torch
   from torch import nn
   emb = nn.Embedding(num_embeddings=30522, embedding_dim=768)
   ids = torch.tensor([[101, 2054, 2003, 102]])   # (batch, seq_len)
   print(emb(ids).shape)        # -> torch.Size([1, 4, 768])
   print(emb.weight.shape)      # -> torch.Size([30522, 768])
   ```

   **Answer.** No matrix multiply happened. `nn.Embedding` gathered four rows out
   of a `|V| x d` parameter matrix, and only those four rows will receive a
   gradient from this batch.

2. `most_similar(positive=["king", "woman"], negative=["man"])` returns `queen`.
   Name the one property of that query without which the result means nothing.

   **Answer.** The three input words are excluded from the answer set. Without
   the exclusion the nearest neighbour of `king - man + woman` is usually `king`
   itself. Report analogy results with the exclusion rule stated.

3. You build a vector index with `all-MiniLM-L6-v2` and query it with
   `bge-small-en-v1.5`. Both are 384-dimensional, so nothing raises. What do you
   get, and what stops it happening again?

   **Answer.** Similarity scores in a plausible range and a ranking that is
   noise — the two models put their vectors in unrelated spaces. Store the
   embedding model name beside the vectors and assert it, with the dimension, at
   query time.

4. Why does `bank` have two different vectors inside a transformer and only one
   under word2vec?

   **Answer.** word2vec learns one vector per word *type*; the two senses are
   averaged into a point between them. A transformer emits one vector per token
   *occurrence*, and that hidden state has already attended to the rest of the
   sentence.
