# Lab 7 — Fine-Tune a Text Classifier

Use a small pretrained encoder on a text classification task, against a
TF-IDF baseline that you build first and must beat.


<!-- notes: They will want to start with the transformer. Do not let them —
the baseline is 10 lines and it is the number the whole lab is judged against.
Circulate during Part C: the token-length histogram is where they discover
their max_length was throwing away half of every document. -->

[tokenization warmup](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-tokenization.ipynb)

[Colab starter](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-clinical-note-triage.ipynb)

[Competition](https://ml-arena.com/viewchallenge/174)


---

## Frozen encoder, trainable head


<!-- placeholder: image to add (note -> tokenizer -> ids -> model -> H -> pool -> e -> MLP -> specialty); adjust lesson id in path -->

The pretrained model is used as a **feature extractor**. Its weights never
change. Only a small classifier trained on its output does.

| Step | Object | Shape |
|---|---|---|
| tokenize | `input_ids`, `attention_mask` | $B \times n$ |
| encode | `last_hidden_state` $H$ | $B \times n \times d$ |
| pool | one vector per note $e$ | $B \times d$ |
| classify | logits over 12 specialties | $B \times 12$ |

$B$ is the batch, $n$ the tokens per note (at most 512 here), and $d$ the
model's hidden size (768 for GPT-2 and BERT-base).

The expensive step, encoding, runs **once**. Everything after it is ordinary
scikit-learn on a matrix of shape $N_{notes} \times d$.

---

## Loading a model from the Hub

```python
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
if tokenizer.pad_token is None:          # GPT-2 was pretrained without padding
    tokenizer.pad_token = tokenizer.eos_token
```

- `AutoModel` loads the **body** only: no language-modelling head and no
  classification head. Its output is $H$, not logits.
- `.eval()` switches dropout off. Without it, the same note gives a different
  vector on each run.
- The tokenizer and the model come as a pair. Never mix a tokenizer from one
  checkpoint with weights from another: the ids would index the wrong rows of
  $E$.

`MODEL_NAME` is one line. The same code runs BERT, RoBERTa, BioBERT or Qwen.

---

## Pooling: one vector per note

The model returns one vector per token. The classifier needs one per note.

**CLS pooling:** take the first row, $e = h_1$. In BERT-style encoders the
`[CLS]` token attends to the whole sequence and is trained to summarise it.

**Mean pooling:** average the real tokens and skip the padding.

$$
e = \frac{\sum_{t=1}^{n} m_t\, h_t}{\sum_{t=1}^{n} m_t}, \qquad m_t \in \{0, 1\} \text{ from } \texttt{attention\_mask}
$$

```python
h = model(**enc).last_hidden_state                 # (B, n, d)
e_cls  = h[:, 0]                                   # (B, d)
m = enc["attention_mask"].unsqueeze(-1)            # (B, n, 1)
e_mean = (h * m).sum(1) / m.sum(1)                 # (B, d)
```

| Model | Pooling | Why |
|---|---|---|
| GPT-2, Qwen (decoders) | mean | causal mask: token 1 has seen nothing of the note |
| BERT, BioBERT (raw encoders) | try both | `[CLS]` was pretrained on next-sentence prediction, not on similarity |
| `all-MiniLM-L6-v2` | mean | the pooling it was trained with |
| `bge-small-en-v1.5` | CLS | the pooling it was trained with |

For sentence-embedding models, the model card or `1_Pooling/config.json`
says which pooling the model was trained with. Use that one. Averaging
without the mask adds zero vectors for padding, so a note's embedding would
depend on how long the other notes in its batch are.

---

## Extract once, reuse everywhere

```python
@torch.no_grad()
def embed(texts, batch_size=16, max_length=512):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        e = h[:, 0] if POOLING == "cls" else masked_mean(h, enc["attention_mask"])
        out.append(e.float().cpu().numpy())
    return np.vstack(out)                          # (N_notes, d)

X, X_test = embed(train["transcription"].tolist()), embed(test["transcription"].tolist())
np.save(f"X_{MODEL_NAME.split('/')[-1]}_{POOLING}.npy", X)
```

- `@torch.no_grad()`: no activations are stored for backprop, which cuts
  memory several times over.
- `truncation=True, max_length=512`: everything past token 512 is **dropped**.
  Check the token-length histogram before trusting a score.
- Cache the matrix to disk. On a T4 encoding is minutes; the MLP is seconds.
  Iterate on the cheap part.

---

## Look before you classify

```python
Z = X - X[idx_tr].mean(0)                          # centre on the training mean
pca = PCA(n_components=2).fit(Z[idx_tr])
P = pca.transform(Z[idx_tr])                       # (N_train, 2)
for c in classes:
    plt.scatter(*P[y[idx_tr] == c].T, s=6, label=c)
```

- **Centre first.** Raw transformer embeddings share a large common
  direction, so every pair of notes looks similar. Subtracting the mean removes
  it, and cosine similarity becomes informative again.
- **Fit PCA on the training split only.** The validation notes are
  projected, never used to choose the axes.
- **Read the explained variance in the title.** Two axes out of 768 often
  keep well under a third of it. Clusters in the plot mean the classes are
  separable. No clusters means only that they are not separable *in two
  linear directions*.

PCA answers "is there structure?" The nearest-centroid score answers "is it
the structure we need?", with no training at all.

---

## An MLP on the embeddings

```python
mlp = make_pipeline(StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2,
                                  early_stopping=True, max_iter=500, random_state=0))
mlp.fit(X[idx_tr], labels.transform(y[idx_tr]))
f1_score(y[idx_val], labels.inverse_transform(mlp.predict(X[idx_val])), average="macro")
```

$$
\hat{y} = \mathrm{softmax}\big(W_2\, \mathrm{ReLU}(W_1 e + b_1) + b_2\big),
\qquad W_1 \in \mathbb{R}^{256 \times d}, \; W_2 \in \mathbb{R}^{12 \times 256}
$$

At $d = 768$ the classifier has $768 \cdot 256 + 256 + 256 \cdot 12 + 12 =$
**199,948** trainable parameters. GPT-2's 124M stay frozen.

- `StandardScaler`: a few embedding dimensions have huge variance and would
  dominate the gradient.
- `alpha` (L2) and `early_stopping`: a few thousand notes against 200k
  parameters is overfitting territory.
