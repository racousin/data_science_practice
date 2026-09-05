# Text as Data and Tokenization

A model consumes tensors of floats. Text is a variable-length sequence of
discrete symbols with no numeric meaning. Everything in this session follows
from closing that gap; tokenization is the first half of it.

<!-- notes: 35 minutes. Open a tokenizer playground live and type their own
names, an accented word, and a JSON snippet — the "why is my surname four
tokens" moment lands better than any slide. Budget 8 minutes for BPE, and do
the merges on the whiteboard before showing the figure. -->

---

## What makes text hard

| Property | Consequence |
|---|---|
| **Discrete** | No gradient with respect to a symbol — you learn a lookup |
| **Variable length** | Batches need padding, and padding needs masking |
| **Compositional** | Meaning is assembled from parts, in order |
| **Ambiguous** | "bank" needs context, and so does "it" |
| **Long-tailed** | Roughly half the word types in a corpus occur once |

None of these apply to a twenty-column tabular row. Sessions 1 to 3 do not
prepare you for any of them.

---

## Order is the signal

![Sequential structure in text](assets/nlp/NatureofText.png)

"Dog bites man" and "Man bites dog" contain identical tokens and describe
different events. Counting words cannot separate them, which is exactly where
bag-of-words models stop.

Dependencies also travel: the subject fixing a verb agreement may be thirty
tokens back, and a pronoun's referent may be in the previous paragraph.

---

## Bytes before tokens

![ASCII code table](assets/nlp/ascii.png)

ASCII is 128 codepoints in 7 bits — `A` is 65, `a` is 97, a constant offset of
32. It covers unaccented English and nothing else.

UTF-8 encodes every Unicode codepoint in 1 to 4 bytes and is byte-compatible
with ASCII for the first 128. It is the only encoding you should be writing in
2026. You will still be *reading* others.

---

## The two encoding mistakes

![ASCII versus Unicode](assets/nlp/utf.png)

**Wrong codec on read.** A cp1252 export decoded as UTF-8 turns `é` into `Ã©`.
Never `errors="ignore"`: that deletes the character and you lose the evidence
that anything happened.

**Unnormalised equivalents.** `é` is either one codepoint or `e` plus a
combining accent. The two strings compare unequal and tokenize differently.

```python
import unicodedata
text = unicodedata.normalize("NFC", raw_bytes.decode("utf-8"))
```

Normalise once, at ingest, next to the assertions from Session 1.

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

## Character level

```python
tokens = list("playing games")   # 13 tokens, vocabulary under 200
```

Unknown tokens are impossible and the vocabulary is tiny. The cost is sequence
length — four to five times longer than word tokens — and the model has to
learn spelling before it can learn syntax.

Reasonable when the alphabet *is* the unit of meaning: DNA, phonemes, some code
models. Not reasonable for prose.

---

## Word level, and why it fails

```python
tokens = "playing videogames".split()   # ['playing', 'videogames']
```

Semantically clean, and it breaks in two ways at once:

- **Out of vocabulary.** Anything unseen becomes `<UNK>` and its content is
  gone. Names, typos, product codes and new words all collapse onto one id.
- **Vocabulary size.** Keeping OOV low in English needs 10^5 or more types. At
  $d = 768$ that is a 77M-parameter embedding matrix before a single layer.

`play`, `plays` and `playing` are three unrelated ids. The morphology has to be
relearned from data, if it is learned at all.

---

## Subword: the compromise everyone uses

Keep frequent words whole and split rare words into pieces that are themselves
frequent. `tokenization` becomes `token` + `ization`.

| Level | Vocabulary | Sequence length | OOV |
|---|---|---|---|
| Character | ~10^2 | ~5× | impossible |
| Word | 10^5–10^6 | 1× | constant |
| Subword | 3·10^4–1.3·10^5 | ~1.3× | impossible |

Every production model since 2016 is subword. This is not a live design
decision, it is a default you inherit from the checkpoint.

---

## Byte-Pair Encoding

![BPE merges on a small corpus](assets/nlp/bpe.png)

Training: start from single characters, count every adjacent pair, merge the
most frequent pair into a new token, repeat until the vocabulary reaches its
target size. The ordered list of merges *is* the tokenizer.

On `hello world hello`, the first four merges are `l+l → ll`, `e+ll → ell`,
`h+ell → hell`, `hell+o → hello`. Frequency alone discovered a whole word.

---

## Applying the merges

At inference you replay the learned merges, in the order they were learned,
until none applies.

```text
"hello"   -> ["hello"]           # fully merged
"helloes" -> ["hello", "es"]     # merged prefix, leftovers
"xqzt"    -> ["x", "q", "z", "t"]
```

Modern BPE runs on **bytes**, so the base vocabulary is 256 symbols and nothing
is ever unrepresentable. An emoji, a Cyrillic name and a corrupted byte all
tokenize — badly, but losslessly and without an `<UNK>`.

---

## WordPiece and SentencePiece

| Algorithm | Merge criterion | Used by |
|---|---|---|
| BPE | highest pair frequency | GPT-2/3/4, Llama, Mistral |
| WordPiece | highest likelihood gain | BERT, DistilBERT |
| SentencePiece | raw text, space is a token (`▁`) | T5, Llama, XLM-R |

WordPiece marks continuations: `playing` becomes `play`, `##ing`.
SentencePiece never assumes whitespace pre-tokenization, which is why it is the
one to reach for with Chinese, Japanese or Thai.

The differences matter for training a tokenizer. For using one, they do not.

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

## Padding, truncation, attention masks

```python
enc = tok(texts, padding=True, truncation=True, max_length=256,
          return_tensors="pt")
print(enc["input_ids"].shape, enc["attention_mask"].shape)
```

`attention_mask` is 1 on real tokens and 0 on padding. It is not decoration:
without it attention averages over `[PAD]` and a sentence vector starts to
depend on the longest sequence that happened to share its batch.

`truncation=True` discards the tail without a word. Plot the token-length
distribution and decide `max_length`; do not inherit 512 by accident.

---

## AutoTokenizer

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tok.tokenize("Tokenization is unavoidable."))
# ['token', '##ization', 'is', 'una', '##vo', '##ida', '##ble', '.']
```

The tokenizer ships with the checkpoint. Load it from the same name as the
model, always. A BERT tokenizer feeding a GPT-2 model produces ids that index
the wrong rows of the embedding table, and raises nothing.

---

## Tokens are the unit of cost

Context windows, API bills, attention memory and throughput are all priced in
tokens. For English under a modern BPE tokenizer: about 0.75 tokens per word,
about 4 characters per token. Code and JSON run roughly twice as dense; text in
a language the tokenizer was not trained on runs two to four times *longer* for
the same content, which is a cost and a quality penalty at once.

> Load the tokenizer from the model's checkpoint, normalise at ingest, and
> assert the token-length distribution before you train.

The failure mode is a truncation nobody looked at: a third of the documents cut
at 128 tokens, an accuracy ceiling with no explanation, and a model that is not
wrong — it simply never saw the evidence.

---

## Check yourself

1. You load `AutoModel.from_pretrained("bert-base-uncased")` but tokenize with a
   GPT-2 tokenizer. Both run without error. What have you actually built?

   **Answer.** Ids produced against one vocabulary indexing the rows of another
   embedding matrix — every token is looked up in the wrong row. Nothing is
   raised and the output is confident nonsense. Load the tokenizer from the same
   checkpoint name as the model, always.

2. Run this. You should get exactly the output shown.

   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   print(tok.tokenize("Tokenization is unavoidable."))
   # -> ['token', '##ization', 'is', 'una', '##vo', '##ida', '##ble', '.']
   print(len(tok("Tokenization is unavoidable.")["input_ids"]))
   # -> 10
   ```

3. The second number in that snippet is 10, not 8. Where do the two extra ids
   come from, and what happens to a model pretrained with them if you remove
   them?

   **Answer.** `add_special_tokens` defaults to `True`, so `[CLS]` and `[SEP]`
   are prepended and appended. A model pretrained with `[CLS]` at position 0 and
   fed a sequence without it returns confident nonsense, silently — leave the
   default alone.

4. Your documents have a 95th percentile of 400 tokens and you tokenize with
   `truncation=True, max_length=128`. What do you observe, and what do you not?

   **Answer.** You observe an accuracy ceiling with no explanation. You do not
   observe the cause: truncation discards the tail without a word and raises
   nothing. Plot the token-length distribution and choose `max_length` from that
   figure rather than inheriting a default.
