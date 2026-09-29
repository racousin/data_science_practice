# Post-Training: From Next Token to Reasoning

Pretraining produces a model that continues text. Everything that makes it an
assistant, and then a model that reasons before it answers, is added afterwards
by a few short training stages. Each one changes two things: the loss, and the
data the loss is computed on.

<!-- notes: 40 minutes. Keep the pipeline figure on screen as the map and come
back to it at every section. The two equations to land are the Bradley–Terry
reward loss and the GRPO advantage: the second is a z-score, which they already
know. Tie self-consistency to the GSM8k challenge: it is the cheapest lever they
have there. -->

---

## One model, five stages

![The LLM training pipeline, from pretraining to distillation](assets/nlp/post-training-pipeline.png)

| Stage | Data | Loss | What it adds |
|---|---|---|---|
| Pretraining | trillions of raw tokens | next-token cross-entropy | knowledge, language |
| SFT | 10k–1M (prompt, answer) pairs | same, on the answer only | format, following instructions |
| Preference | ranked answer pairs | reward model + RL, or DPO | helpful, harmless, the tone |
| Reasoning RL | problems with a checker | policy gradient on a 0/1 reward | long chains of thought |
| Distillation | a teacher's outputs | cross-entropy on the teacher | the same skills in a smaller model |

The objective is the same next-token loss until the preference stage. What
changes is **which text** the model is trained to produce. Post-training is
mostly a data problem.

---

## SFT: the data is the product

Supervised fine-tuning is next-token training on written answers, with the
prompt tokens masked out of the loss:

$$
L_{SFT}(\theta) = -\sum_{t \in \text{answer}} \log \pi_\theta(y_t \mid x, y_{<t})
$$

The model imitates its demonstrations exactly, style and mistakes included. So
the work is in the data:

- **Human-written** answers are expensive; tens of thousands is a large set.
- **Synthetic** answers from a stronger model are cheap and now dominate. Their
  errors become the student's errors.
- **Quality beats quantity.** About 1,000 carefully chosen examples can be
  enough to teach the chat format (LIMA, 2023).

SFT has a ceiling: the model learns to write like the demonstrations, never
better than them. To go past that, you need a signal that says which of two
answers is *better*. The next stages use that signal.

---

## RLHF: learn a reward, then optimise it

![The RLHF loop: policy, reward model, frozen reference](assets/nlp/rlhf-loop.png)

**Step 1, the reward model.** Annotators see two answers and pick one. A model
$r_\phi$ is trained so that the preferred answer $y_w$ scores higher than the
rejected one $y_l$ (the Bradley–Terry model):

$$
L(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) \right]
$$

**Step 2, the policy.** Sample answers, score them, and update the model with
PPO (Session 10) to maximise

$$
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ r_\phi(x, y) \big] - \beta \, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{ref}(\cdot \mid x)\big)
$$

The KL term keeps the policy close to the SFT model. Without it, the policy finds
text that the reward model scores highly but no human would want. This is
**reward hacking**: the reward model is only an approximation of human judgement.
Typical signs are answers that get longer, flatter the user, or sound confident
without being correct.

---

## DPO: the same target without RL

Direct Preference Optimization (Rafailov et al., 2023) shows that the optimum of
the RLHF objective can be written in closed form. Substituting it gives a
classification loss directly on the preference pairs:

$$
L_{DPO}(\theta) = -\,\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)} \right) \right]
$$

The loss increases the probability of the preferred answer and decreases the
probability of the rejected one, both relative to a frozen reference. There is
no reward model, no sampling and no value network: you train it with the same
tools as SFT (`trl.DPOTrainer`).

The preference data can also come from a model instead of people. In **RLAIF**
and Constitutional AI, a model compares the two answers against a written list
of principles.

---

## Chain of thought

A transformer does a fixed amount of computation per token: one forward pass. If
the answer has to come in the first token, the model gets one forward pass to
find it. If it can write intermediate steps first, every step is extra computation
that the next tokens can attend to.

```text
Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls?
A: He buys 2 × 3 = 6 balls. 5 + 6 = 11. The answer is 11.
```

- **Few-shot CoT** (Wei et al., 2022): examples with worked steps in the prompt.
- **Zero-shot CoT**: add *"Let's think step by step."*
- **Self-consistency**: sample $n$ chains at temperature > 0 and take a majority
  vote on the final answers.

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^{n} \mathbb{1}[a_i = a]
$$

This is **test-time compute**: more tokens at inference time buy accuracy, with
the weights unchanged. On GSM8k, this session's challenge, self-consistency is
the cheapest lever you have.

---

## Reasoning with RL: GRPO

Prompting asks the model to reason. RL on **verifiable rewards** trains it to.
Take problems where the answer can be checked by code: a math result, unit tests
for code. The reward is 1 if the check passes, 0 otherwise. No reward model is
needed, so there is nothing to hack except the checker.

GRPO (Group Relative Policy Optimization, DeepSeekMath 2024) samples $G$ answers
to the same prompt and uses the group as its own baseline:

$$
A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}
$$

![Group-normalised advantages for eight sampled answers](assets/nlp/grpo-advantages.png)

Each token of answer $i$ is pushed up or down by $A_i$ with PPO's clipped ratio
and a KL term. The group mean replaces PPO's critic network. If all answers are
right, or all wrong, every advantage is zero: training keeps problems the model
solves only sometimes.

DeepSeek-R1-Zero (2025) applied this to a base model with only a correctness and
a format reward. Its answers grew from hundreds to thousands of tokens, and
behaviours such as checking and restarting a solution appeared in the chains of
thought without being demonstrated.

---

## Distillation

![One-hot label versus a teacher's distribution](assets/nlp/distillation-soft-targets.png)

A one-hot label says the next word was *tea*. The teacher's distribution also
says that *coffee* was the likelier guess and that *beer* was unlikely. Knowledge
distillation (Hinton et al., 2015) trains a small student on those soft targets,
both softened by a temperature $T$:

$$
L_{KD} = T^2 \, \mathrm{KL}\big(\mathrm{softmax}(z_t / T) \,\|\, \mathrm{softmax}(z_s / T)\big)
$$

With LLMs the simpler form dominates: **sequence-level distillation**. Generate
answers with the teacher, keep the good ones, and run ordinary SFT on them. It
needs only the teacher's text, not its logits, so it also works through an API.

In the R1 report, a 32B model trained on 800k reasoning traces from R1 scored
72.6% on the AIME 2024 math benchmark. The same 32B base trained directly with
RL reached 47.0%. For a small model, copying the reasoning of a large one is
cheaper and better than discovering it again.

> Distillation is the reason small open models improve so quickly, and the
> reason many licences forbid training on the model's outputs.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   r = np.array([1, 0, 0, 1, 0, 0, 0, 1.])
   print(np.round((r - r.mean()) / r.std(), 2))
   # -> [ 1.29 -0.77 -0.77  1.29 -0.77 -0.77 -0.77  1.29]
   ```

   **Answer.** These are the GRPO advantages for eight samples with three
   correct. The correct answers are pushed up and the wrong ones down, and the
   mean is zero. With `r = np.ones(8)` the standard deviation is zero: in
   practice a small epsilon is added and every advantage is 0, so the prompt
   gives no gradient.

2. Your RLHF-trained model gives longer answers every epoch, and its reward
   keeps rising. What is happening, and which term of the objective should you
   look at?

   **Answer.** Reward hacking: the reward model prefers longer answers, and the
   policy exploits that preference. Look at the KL coefficient $\beta$. A
   larger value keeps the policy closer to the reference. Also look at the
   reward model's data: its length bias comes from the annotations.

3. Why can DPO be trained with SFT tooling while RLHF cannot?

   **Answer.** DPO's loss uses only log-probabilities of fixed answers from the
   dataset, under the model and under a frozen reference. RLHF has to sample new
   answers from the policy during training, score them with a separate reward
   model, and run PPO with a value network.

4. You have a 1B model and an API to a strong reasoning model. Which is the
   cheaper way to make the 1B model better at GSM8k: GRPO on the 1B model, or
   sequence-level distillation?

   **Answer.** Distillation. Generate worked solutions with the strong model,
   keep the ones whose final answer is correct, and run SFT on them. GRPO
   needs the 1B model to already solve some problems, and many samples per
   prompt. Check the API's terms before you train on its outputs.
