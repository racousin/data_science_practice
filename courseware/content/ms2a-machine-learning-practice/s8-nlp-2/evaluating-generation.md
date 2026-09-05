# Evaluating Generation

Classification has one right answer, so accuracy works. Generation has a set of
acceptable answers whose size you do not know, and no automatic metric measures
the thing you care about. This lesson is about choosing which approximation to be
wrong with.

<!-- notes: 25 minutes. This is the lesson that governs the LLM-judged project
track — say so at the start. Have them score two summaries by hand before
showing BLEU, so the disagreement is theirs. -->

---

## Why accuracy does not apply

Reference: *The cat is on the mat.*

| Candidate | Correct? | Exact match |
|---|---|---|
| The cat is on the mat. | yes | 1 |
| A cat sits on the mat. | yes | 0 |
| The mat is on the cat. | no | 0 |
| The feline rests upon the rug. | yes | 0 |

Exact match scores three correct answers as failures and cannot tell row two
from row three. Every metric below closes part of that gap and still gets some
row wrong.

---

## BLEU

![N-gram overlap between candidate and reference](assets/nlp/bleu.png)

$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

$p_n$ is n-gram precision for n = 1..4, clipped at the reference count. $BP$ is
a brevity penalty, because precision alone rewards emitting the three words you
are sure of.

```python
from sacrebleu import corpus_bleu
score = corpus_bleu(hypotheses, [references]).score
```

Precision-oriented, built for machine translation, and reported corpus-level.
Sentence-level BLEU is noisy enough to be misleading — use `sacrebleu` on a
corpus, and report its version string.

---

## ROUGE

![ROUGE-N and ROUGE-L for summarisation](assets/nlp/rouge.jpeg)

$$
ROUGE_N = \frac{\sum_{g \in S} Count_{match}(g)}{\sum_{g \in S} Count(g)}
$$

The denominator counts n-grams in the *reference*, which makes ROUGE a recall
measure: did the summary cover the reference content. ROUGE-L uses the longest
common subsequence instead of fixed n-grams, so word order matters but gaps are
tolerated.

BLEU asks "is what you wrote in the reference?"; ROUGE asks "is the reference in
what you wrote?". Translation takes the first, summarisation the second.

---

## Why overlap metrics mislead

- **One reference, many valid outputs.** A correct paraphrase scores zero.
- **No semantics.** *The mat is on the cat* shares every unigram with the
  reference and inverts the meaning.
- **Length games.** Both metrics are gameable by tuning output length, which is
  why brevity penalties exist and why they are not enough.
- **Weak human correlation.** For strong modern systems, BLEU correlates poorly
  with human judgement — the gap the metric measures is smaller than its noise.

They are not useless: they are cheap, deterministic and comparable across runs.
Use them as a **regression alarm** on your own system, never as evidence that
one model is better than another.

---

## Embedding-based and intrinsic measures

**BERTScore** embeds both texts with a pretrained encoder and greedily matches
tokens by cosine similarity, then reports precision, recall and F1.

```python
from bert_score import score
P, R, F1 = score(cands, refs, lang="en")
```

It rewards paraphrase, which n-gram overlap cannot. It inherits the encoder's
blind spots, is not comparable across encoder versions, and still needs a
reference.

**Perplexity** needs no reference — it measures how surprised a model is by
held-out text. The right instrument for tracking a fine-tune, the wrong one for
judging quality: fluent nonsense has low perplexity.

---

## LLM-as-judge

![A model producing a structured critique of another model's output](assets/nlp/critic.png)

Ask a strong model to score the output against a rubric. On instruction
following and helpfulness, a good judge reaches roughly 80–90% agreement with
expert annotators — better than any n-gram metric, at a fraction of the cost of
humans.

A judge is only a measurement instrument if it is specified like one:

1. A written rubric with the levels defined, not "rate 1–5".
2. A fixed output schema — JSON with a score and a justification.
3. A pinned model version and `temperature=0`.
4. The prompt stored in the repository, under version control.

The same machinery in a loop is self-critique (Agents lesson); run once, offline,
against a fixed rubric, it is an evaluator.

---

## Pointwise, pairwise, and the biases

| Mode | Question | Good for |
|---|---|---|
| Pointwise | score this output 1–5 on the rubric | absolute thresholds, faithfulness |
| Pairwise | A or B, which is better? | comparing two systems |

Pairwise is more reliable — relative judgements are easier — but gives no number
you can track over time. Three biases to design around:

- **Position bias.** Judges favour the first option. Run every pair in both
  orders and count a disagreement as a tie.
- **Verbosity bias.** Longer answers score higher at equal quality. Put a length
  constraint in the rubric or control for it.
- **Self-preference.** A model prefers text from its own family. Do not judge a
  model with itself.

---

## A judge you can rerun

```python
import json
from pathlib import Path
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
JUDGE_PROMPT_V2 = Path("prompts/faithfulness_v2.txt").read_text()

def judge(question, context, answer, model="claude-sonnet-4-5-20250929"):
    out = client.messages.create(model=model, max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT_V2.format(
            question=question, context=context, answer=answer)}])
    return json.loads(out.content[0].text)   # raises on malformed output
```

Four properties make a judge reproducible: the prompt is a versioned file, the
model id is pinned to an exact dated snapshot and logged, sampling is off
wherever you can control it, and a malformed response raises instead of scoring
zero. The call above does three of them outright; the third is the one the next
slide is about.

The key never appears in the code. `anthropic.Anthropic()` with no argument reads
`ANTHROPIC_API_KEY` from the environment, which is the only form that survives a
`git log`.

---

## Where the temperature went

Notice what is *not* in that call. The Anthropic Python SDK has removed the
sampling parameters, so the line every tutorial still shows now fails before it
reaches the network:

```python
client.messages.create(model=..., temperature=0, ...)
# TypeError: Messages.create() got an unexpected keyword argument 'temperature'
```

Pinning the exact dated model id is what you have left, and it is why the id in
the snippet above carries a date. This is the same point the decoding lesson
made from the other side: `temperature=0` never bought reproducibility anyway —
assert properties, log the model, and do not compare across ids.

---

## A judge with no account

No hosted key? The same four properties hold locally, on CPU, in about a
gigabyte of weights:

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # log this string beside every score
tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.float32)

def judge_local(prompt):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=80, do_sample=False)
    text = tok.decode(out[0, enc["input_ids"].shape[-1]:],
                      skip_special_tokens=True)
    return json.loads(text)          # raises on malformed output
```

`do_sample=False` is greedy decoding from the previous lesson, and it is the
local equivalent of the temperature a hosted API no longer takes. A 0.5B judge
agrees with you less often than a frontier one — that is a number you can
measure, not a reason to skip the instrument.

> A judge prompt written inline in a notebook and edited between runs is not a
> metric. Yesterday's numbers cannot be compared to today's, and nobody will
> notice until the results stop making sense.

---

## Humans, contamination, and your project

Human evaluation is the ground truth every other metric approximates, and it is
slow and expensive: spend it on 50–100 sampled examples, a written rubric, two
annotators, and an agreement figure. A human score without one is an opinion.

**Benchmark contamination** is the reason a public score can be meaningless: if
the test set was on the web before the training cutoff, the model may have read
it. Trust a held-out set you built yourself over any leaderboard.

The project's LLM-judged `file_v1` track is scored exactly this way: a fixed
rubric, a pinned judge, a held-out prompt set. Build the harness now and you
evaluate your submission with the instrument that grades it.

---

## Check yourself

1. Reference: *The cat is on the mat.* Candidate: *The mat is on the cat.* What
   does exact match say, what does unigram BLEU say, and which one is right?

   **Answer.** Exact match scores 0, which happens to be right for the wrong
   reason. Unigram precision is 1.0 — every word of the candidate is in the
   reference — and the meaning is inverted. That is the "no semantics" failure:
   overlap metrics cannot separate a paraphrase from a reversal.

2. Run this. You should get exactly the output shown.

   ```python
   import json
   try:
       json.loads('Score: 2 - the answer is supported.')
   except json.JSONDecodeError as e:
       print(type(e).__name__, "|", e)
   # -> JSONDecodeError | Expecting value: line 1 column 1 (char 0)
   ```

   **Answer.** That raise is the design, not an accident. A judge that swallowed
   the parse failure and returned 0 would report a broken instrument as a bad
   answer, and the mean faithfulness would drop for a reason nobody could find.

3. You judge model A against model B pairwise, and A wins 62% of the time. Name
   the two biases that could produce that number on their own, and the fix for
   the first.

   **Answer.** Position bias — judges favour whichever option comes first — and
   verbosity bias, since longer answers score higher at equal quality. Fix the
   first by running every pair in both orders and counting a disagreement as a
   tie. (Self-preference is the third: never judge a model with itself.)

4. Your judge scored 1.62 last week and 1.41 today on the same 40 answers, and
   you edited the rubric in between. Which of the four properties did you break,
   and what is the cost?

   **Answer.** The versioned prompt. Once the rubric moves, last week's number
   and today's measure different things and cannot be compared — which is why the
   prompt lives in a file under version control and Lab 8 pins its sha256.
