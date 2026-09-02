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
JUDGE_PROMPT_V2 = Path("prompts/faithfulness_v2.txt").read_text()

def judge(question, context, answer, model="claude-sonnet-4-5-20250929"):
    out = client.messages.create(model=model, temperature=0, max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT_V2.format(
            question=question, context=context, answer=answer)}])
    return json.loads(out.content[0].text)   # raises on malformed output
```

Four properties make this reproducible: the prompt is a versioned file, the
model id is pinned and logged, the temperature is zero, and a malformed response
raises instead of scoring zero.

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
