# Text as Data and Tokenization

A model consumes tensors of floats. Text is a variable-length sequence of
discrete symbols with no numeric meaning. Everything in this session follows
from closing that gap; tokenization is the first half of it.

<!-- notes: 35 minutes. Open a tokenizer playground live and type their own
names, an accented word, and a JSON snippet — the "why is my surname four
tokens" moment lands better than any slide. Budget 8 minutes for BPE, and do
the merges on the whiteboard before showing the figure. -->

---

## Sequential Nature of Text

- Word order impacts meaning: "Dog bites man" ≠ "Man bites dog"
- Dependencies can span across long distances in a sequence
- Context is critical for disambiguation (e.g., "bank" can mean financial institution or river edge)

![Sequential structure in text](assets/nlp/NatureofText.png)

"Dog bites man" and "Man bites dog" contain identical tokens and describe
different events. Counting words cannot separate them, which is exactly where
bag-of-words models stop.

Dependencies also travel: the subject fixing a verb agreement may be thirty
tokens back, and a pronoun's referent may be in the previous paragraph.

---

## Language Diversity at Scale

**Languages and writing systems**

- ~7,000 living languages worldwide (Ethnologue 2023)
- ~300 writing systems across history, with ~150 currently in use

**Vocabulary and text volume**

- 170,000+ words in current English usage (Oxford English Dictionary)
- ~130 million books published in all languages throughout history
- 2.5 million+ books published annually worldwide
- Billions of web pages containing trillions of words across languages

![image5.png](assets/nlp/image5.png)

---

## Sequence Representations Beyond Natural Language

- **Biological sequences**: DNA sequences use four nucleotide bases: A (Adenine), T (Thymine), G (Guanine), C (Cytosine).
- **Programming code**: Programming languages like Python, Java, and C++ have finite token vocabularies including keywords (`if`, `while`, `return`), operators (`=`, `+`, `==`), and punctuation (braces, parentheses, semicolons).
- **Mathematical notation**: A symbolic language with operators (+, −, ×, ÷, =), variables (x, y, z), functions (sin, cos, log), Greek letters (α, β, γ), and special symbols (∫, ∑, ∂).
- **Musical sequences**: MIDI note numbers (0–127), pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B), duration values (whole, half, quarter notes).
- **Chemical representations**: SMILES (Simplified Molecular Input Line Entry System) notation represents molecular structures as text strings using characters for atoms (C, N, O, S), bonds (`-`, `=`, `#`), and branches (parentheses).

![image14.png](assets/nlp/image14.png)

![image17.png](assets/nlp/image17.png)

---

## What a tokenizer is

![From text to tokens to ids and back](assets/nlp/tokenize1.jpg)

Three objects, and every tokenizer has exactly these three:

- a **vocabulary** — a fixed set of tokens, each with an integer id
- a **split rule** — text to a list of tokens
- a **decode rule** — ids back to tokens, tokens back to text

Vocabulary size is a model hyperparameter: it fixes the height of the embedding
matrix and the width of the output layer. Changing it means retraining.

---

## Bytes before tokens

![ASCII code table](assets/nlp/ascii.png)

ASCII is 128 codepoints in 7 bits — `A` is 65, `a` is 97, a constant offset of
32. It covers unaccented English and nothing else.

UTF-8 encodes every Unicode codepoint in 1 to 4 bytes and is byte-compatible
with ASCII for the first 128. It is the only encoding you should be writing in
2026. You will still be *reading* others.

---

## Value encoding

![ASCII versus Unicode](assets/nlp/utf.png)

---

## Character level

Character tokenization breaks text into individual characters, offering a very
small vocabulary but requiring longer sequences.

**Vocabulary:** alphabet + special characters (~30 tokens)

```text
"The cats are running"
→ [T, h, e, ␣, c, a, t, s, ␣, a, r, e, ␣, r, u, n, n, i, n, g]
→ [20, 8, 5, 27, 3, 1, 20, 19, 27, 1, 18, 5, 27, 18, 21, 14, 14, 9, 14, 7]
```

| Token | ID | Token | ID |
|---|---|---|---|
| `"a"` | 1 | `"z"` | 26 |
| `"b"` | 2 | `" "` | 27 |
| `"c"` | 3 | `"."` | 28 |

❌ **Problems:** very long sequences + loss of word structure

```python
tokens = list("playing games")   # 13 tokens, vocabulary under 200
```

---

## Word level, and why it fails

Word tokenization splits text at word boundaries, typically using spaces and
punctuation as delimiters.

```python
tokens = "playing videogames".split()   # ['playing', 'videogames']
```

**Vocabulary:** all the words in the corpus (example: English Wikipedia ~13M tokens)

```text
"The cats are running" → [The, cats, are, running] → [1, 856432, 15, 2347]
```

| Token | ID |
|---|---|
| `"The"` | 1 |
| `"cats"` | 856,432 |
| `"running"` | 2,341,567 |

Semantically clean, and it breaks in two ways at once:

❌ **Problems:** gigantic vocabulary + rare words underrepresented + "eat" ≠ "eats"

`play`, `plays` and `playing` are three unrelated ids. The morphology has to be
relearned from data, if it is learned at all.

---
## Subword: Byte-Pair Encoding

![BPE: learning merges, then encoding an unseen word](assets/nlp/image3.png)

**Training** — learn the merges once, on a corpus

Start from single characters, count every adjacent pair, merge the most
frequent pair into a new token, repeat until the vocabulary reaches its
target size. The ordered list of merges *is* the tokenizer.

| Step | Vocabulary |
|---|---|
| Corpus | `lower`, `newest`, `widest`, `low` |
| Initial (characters) | `l o w e r n s t i d` |
| Merges, in order | `es` → `est` → `lo` → `low` → `ne` → `new` → `newest` → `wi` → `wid` → `widest` |

Frequent suffixes (`est`) and stems (`low`) become tokens of their own;
frequent whole words (`newest`, `widest`) end up as a single token.

**Encoding** — apply the merges to any word, even one never seen

`lowest` is not in the corpus (out-of-vocabulary):

```text
lowest → l o w e s t        split into characters
       → lo w es t          apply merges in learned order
       → low est            until no merge applies
```

No `[UNK]`: an unseen word falls back to known pieces, and in the worst case
to single characters (single bytes in byte-level BPE, as in GPT).
Morphology comes for free: `low` + `est` shares `est` with `newest` and `widest`.

---

## Special tokens

| Token | Purpose |
|---|---|
| `[CLS]` / `<s>` | Sequence start; its final state is the sentence vector |
| `[SEP]` / `</s>` | Segment boundary, or end of sequence |
| `[PAD]` | Filler that makes a batch rectangular — must be masked |
| `[MASK]` | The prediction target in masked language modelling |
| `[UNK]` | Only reachable with a word-level vocabulary |

These hold real ids. A model pretrained with `[CLS]` at position 0 and fed a
sequence without it returns confident nonsense, silently. `add_special_tokens`
defaults to `True` — leave it there.

---

## Text Data - Extended Sequence Application

![image8.png](assets/nlp/image8.png)

![image20.png](assets/nlp/image20.png)

---

## Check yourself

![image19.png](assets/nlp/image19.png)

[Open the tokenization notebook in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-tokenization.ipynb)
