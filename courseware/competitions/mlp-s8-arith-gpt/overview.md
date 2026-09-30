# Arithmetic GPT — Session 8

Teach a tiny GPT to multiply two 4-digit numbers.

**The model is fixed.** You get a 0.42-million-parameter decoder-only
transformer and its greedy decoder. You cannot change the architecture, the
decoding or the answer parser. You train it from scratch and submit **its
weights**. What you control is the training data: which problems, and above
all **what the model writes before its answer**.

Trained to write the answer directly, this model gets **0%** of the products
right. The same model, the same training time, trained to write its
intermediate steps first, gets most of them right. This challenge is about
finding out why, and about designing those steps.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s8-arith-gpt.ipynb)

The starter notebook downloads the files, trains the answer-only model,
scores it on the dev set, and submits. Use a GPU runtime (Runtime → Change
runtime type → T4 GPU); a run of 5–10 minutes is enough to compare two
formats.

## The files

| file | contents |
|---|---|
| `arith_gpt.py` | the model, the decoder, the parser and the scorer: the same file the leaderboard imports |
| `dev_problems.csv` | 200 products with their answers, for scoring locally |

There is no training set. Every product has a known answer, so you generate
as much data as you want, in whatever format you choose.

## The model

| | |
|---|---|
| layers × heads × width | 2 × 4 × 128 |
| context | 160 tokens, learned positions |
| embeddings | input and output tied |
| vocabulary | 19 characters: `0`–`9`, `+ - * = \| > #`, `<pad>`, newline (end of sequence) |
| parameters | 0.42 M (1.7 MB in float32) |

## How a problem is scored

1. The prompt is `a*b=` with two 4-digit numbers (1000–9999), for example
   `1234*5678=`.
2. The model continues it with **greedy decoding** (always the most likely
   token), for at most **128 new tokens**.
3. It must write the end token (a newline) within those 128 tokens. A model
   that runs out of tokens has no answer.
4. The answer is what follows the **last `#`** before the end token, written
   as a plain integer: `7006652`, not `07006652`.

Everything the model writes before that `#` is ignored by the parser. You may
use it as a **scratchpad**:

```text
1234*5678=#7006652                       answer only
1234*5678=<steps you design>#7006652     a chain of thought, then the answer
```

`arith_gpt.evaluate_problems(model, problems)` applies exactly these rules.

## The score

**Exact-match accuracy** on 1000 private products: the leaderboard ranks on
it. **Per-digit accuracy** is shown beside it: the share of answer digits
that are right, aligned from the units digit. It shows progress before exact
answers appear, but note its floor: the answer-only model already reaches
about 45% per digit (the last digit and the first one are easy) while getting
no product right.

The answer-only model of the starter notebook is the benchmark: 0% exact.
The module counts the challenge validated at **50% exact**.

## Submitting

Save with `arith_gpt.save_weights(model, "weights.safetensors")` and upload
that single file. It is rejected, with a message saying why, if it is not a
safetensors file, if a tensor name or shape does not match the model, if it
contains NaN or infinity, or if it is larger than 5 MB. float32, float16 and
bfloat16 are accepted.

## Hints

Read them one at a time, and test each idea before reading the next.

1. **Give the model room to compute.** A transformer spends the same, fixed
   amount of computation on every token it writes. Seven digits of a product
   in seven tokens is too little for a 2-layer model. Tokens written *before*
   the `#` are extra computation, and the model can read its own previous
   tokens.
2. **Write what you would write on paper.** Long multiplication splits
   `1234*5678` into `1234*8`, `1234*7`, `1234*6`, `1234*5`, then adds the
   shifted partial products. Each step is small enough for the model; the
   whole product is not.
3. **Keep every step local.** A good step only needs a few tokens the model
   has already seen. A running sum after each partial product (`>`) is
   easier than one big addition of four numbers at the end.
4. **Think about the order of the digits.** The model writes left to right.
   In an addition, the carry comes from the digit to the *right*, which the
   model has not written yet. What if the numbers of the scratchpad were
   written in another order? Where would each digit then sit?
5. **Watch the loss, not only the accuracy.** Exact accuracy can stay at 0%
   for a few minutes while the loss falls, then climb quickly. Compare two
   formats at the same training time.
