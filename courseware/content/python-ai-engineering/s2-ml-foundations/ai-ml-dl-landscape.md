# The AI / ML / DL Landscape

<!-- notes: 20 minutes. Opening of the session. The goal is vocabulary, a shared
timeline, and the closing statement of the objective. Do not linger on the
history slide — the last section is the one the rest of the session builds on. -->

## Definitions

| Term | What it means |
|---|---|
| **AI** | Automation involving algorithms to reproduce cognitive capabilities such as reasoning, perception, and decision-making. |
| **ML** | Statistical algorithms that automatically learn patterns from data without being explicitly programmed for each task. |
| **DL** | Neural networks with multiple layers (millions of parameters) that learn complex representations from raw data. |

They nest: every deep learning system is machine learning, every machine
learning system is AI, and most of what is *called* AI in 2026 is deep generative models.

---

## Three nested sets, not three rivals

![AI contains ML contains DL](assets/s2-ml-foundations/ai-ml-dl-landscape/ai-ml-dl-venn.png)

Each ring is strictly inside the one around it. "AI" is the widest word and the
least informative one; say which ring you mean.

---

## Where you already meet it

![Applications of AI](assets/s2-ml-foundations/ai-ml-dl-landscape/ai-applications.png)

---

## What the work requires

![Data science as an intersection](assets/s2-ml-foundations/ai-ml-dl-landscape/data-science-venn.png)

You have most of the first column already. It is worth naming which parts.

---

## The three columns, concretely

| | What is actually used |
|---|---|
| **Mathematics** | Linear algebra — matrix products, rank, eigen/SVD decompositions, and matrix calculus. Multivariate calculus — gradients, Jacobians, the chain rule. Probability and statistics — estimators and their bias, maximum likelihood, conditional expectation, concentration and the bias–variance decomposition. Optimisation — convexity, first-order methods, and what their guarantees are worth when convexity is lost. |
| **Computer science** | Complexity and data structures, so you can tell $O(np)$ from $O(p^3)$ before you run it. Numerical computing — floating point, conditioning, and vectorisation instead of Python loops. Memory hierarchy and parallelism, which is why a GPU changes what is trainable. Software engineering — version control, tests, reproducible environments; Session 1. |
| **Domain expertise** | Which variables exist, which are available at prediction time, what the units mean, which errors are expensive. It is not decoration: it decides the target, the metric, and which columns you are allowed to use. |

---

## A short history

![Timeline of AI](assets/s2-ml-foundations/ai-ml-dl-landscape/ai-history-timeline.png)

Two "AI winters" (roughly 1974-1980 and 1987-1993) followed two waves of
over-promising. The field's current wave started in 2012 and has not broken yet.

---

## The deep learning revolution

Interest did not grow steadily — it broke upward around 2012 and again around
2023:

![Search interest in "deep learning"](assets/s2-ml-foundations/ai-ml-dl-landscape/deep-learning-trend.png)

Three factors converged. None of them alone would have been enough.

---

### 1. Big data

The internet era generated massive datasets: billions of images, text documents,
videos, and user interactions. This provided the fuel needed to train complex
models that require millions of examples to learn robust patterns.

![Data generated every minute](assets/s2-ml-foundations/ai-ml-dl-landscape/data-every-minute.png)

---

### 2. Computational power

GPUs originally designed for gaming proved perfect for neural network
computations. A single modern GPU performs trillions of operations per second,
which turns a training run that would have taken years into one that takes days.

![Growth in compute](assets/s2-ml-foundations/ai-ml-dl-landscape/compute-progress.png)

---

### 3. Algorithmic innovation

Breakthrough techniques — ReLU activations, batch normalization, dropout,
attention — solved training problems that had blocked deep networks for decades.
These made it practical to train networks dozens or hundreds of layers deep. Open
source, and the communities around it, spread each advance in months rather than
years.

![The transformer architecture](assets/s2-ml-foundations/ai-ml-dl-landscape/transformer-architecture.png)

The transformer (2017) is the architecture behind essentially every large
language model you have used.

---

## Where the rules come from

Classical AI is a human writing rules. It works, and it is still the right
answer when the rules are short and knowable.

**Rules are knowable.** Validating an IBAN: move the first four characters to
the end, map each letter to two digits, read the result as an integer and check
it is $\equiv 1 \pmod{97}$. Four lines, exact, no data required and no training
run. Writing a model for this would be strictly worse.

**Rules are not knowable.** Deciding whether a $28 \times 28$ grid of grey levels
is a handwritten 7. You know a 7 when you see one and you cannot say why: any
rule you write about strokes and angles is broken by the next person's
handwriting. There are $256^{784}$ possible images and no enumeration is
available. The knowledge is real and it is not expressible as rules.

---

## The objective, stated once

That gap is what machine learning is for. Instead of writing the rule, you
supply examples of the rule's behaviour and search for a function consistent
with them.

![Rules as input versus rules as output](assets/s2-ml-foundations/ai-ml-dl-landscape/rules-vs-learning.png)

| Classical programming | Machine learning |
|---|---|
| You write the rules; the data is the input. | You supply the data *and* the answers. |
| The program applies them mechanically. | Training returns the rules. |

> "The science of getting computers to learn without being explicitly
> programmed" — Arthur Samuel, 1959

So, for the rest of this course:

**Use data to learn the rule, and judge the rule by what it does on data you
have never seen.**

---

## On data you have never seen

![Predicting the future from historical data](assets/s2-ml-foundations/ai-ml-dl-landscape/temperature-extrapolation.png)

The circled points are the ones nobody has measured. Everything in this session
— the data, the model, the loss, the training — is machinery for putting them
there honestly.
