# Instruction Tuning and PEFT

A pretrained model completes text. It does not answer questions, follow
instructions or refuse anything. Turning a completion engine into an assistant
is a second training stage — and adapting that assistant to your domain is a
third, which you can actually afford.

<!-- notes: 35 minutes. Open by prompting a base checkpoint with a question and
letting the room watch it generate more questions. The adaptation-ladder slide
is the one they should photograph. -->

---

## Base versus instruct

Same architecture, same weights up to the last stage, completely different
behaviour.

| | Base (`Llama-3-8B`) | Instruct (`…-8B-Instruct`) |
|---|---|---|
| Trained on | raw corpus | corpus, then instruction pairs |
| Input | any text | a formatted conversation |
| "Capital of France?" | *"What is the capital of Spain? What…"* | *"Paris."* |
| Use for | pretraining, LoRA bases | anything you prompt |

The base model is not broken — it continues the pattern, exactly as trained: in
the corpus, a question is most often followed by another question.

---

## The chat template is not optional

An instruct model was tuned on one specific string format. Feed it raw text and
you get degraded, sometimes bizarre, output.

```python
msgs = [{"role": "user", "content": "Summarise this in one sentence."}]
prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
```

Never hand-write `<|im_start|>` markers. `apply_chat_template` reads the format
from the tokenizer config, so it survives a change of model. If a model that
works through an API behaves badly locally, check the template first.

---

## Supervised fine-tuning

SFT is ordinary next-token training on `(instruction, response)` pairs, masked so
that only the response tokens contribute to the loss.

```python
labels[: len(prompt_ids)] = -100     # do not train on the instruction
```

- 10k–100k pairs is enough, and quality dominates quantity: a curated 1k set
  beats a scraped 100k set.
- The model learns *format and behaviour*, not new facts. SFT is not how you
  teach knowledge; that is what retrieval is for.
- The `-100` mask is the forgotten step. Without it the model learns to generate
  instructions as readily as answers.

---

## Preference training: what it optimizes

SFT teaches one acceptable response per prompt. Preference training teaches a
*ranking* over responses, which is closer to what "helpful" means.

**RLHF**: humans mark A better than B; a reward model learns to predict that;
the policy maximises the reward under a KL penalty holding it near the SFT
model. The KL term is load-bearing — without it the policy finds text that
scores well and reads as nonsense: reward hacking, as in Sessions 9 and 10.

**DPO** removes the reward model and the RL loop, rewriting the same target as a
classification loss on preference pairs:

$$
\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi(y_l \mid x)}{\pi_{ref}(y_l \mid x)} \right) \right]
$$

Raise the winning response $y_w$, lower the losing one $y_l$, both relative to a
frozen reference. One model, one loss, ordinary supervised tooling — DPO is what
you would run; RLHF is what your downloaded checkpoint was trained with.

---

## The adaptation ladder

![Zero-shot and few-shot use of a pretrained model versus transfer](assets/nlp/zero-shot-vs-transfer.png)

Five rungs in increasing order of cost. Climb one at a time, stop when the task
is solved:

1. **Prompting** — a clear instruction, an output schema.
2. **Few-shot** — 3–10 examples in the prompt.
3. **RAG** — retrieve the facts the model lacks.
4. **PEFT** — LoRA on a few thousand examples.
5. **Full fine-tuning** — all parameters.

---

## Prompting and few-shot

![In-context learning: examples in the prompt shape the output](assets/nlp/prompteng.png)

In-context learning changes no weights. The examples in the prompt condition
the distribution for that call only.

```text
Text: The service was excellent.  ->  positive
Text: The food was terrible.      ->  negative
Text: The room was clean.         ->
```

Few-shot buys format compliance almost for free. The costs are real: the
examples occupy context on every call, and the effect is sensitive to order.

---

## Choosing a rung

| Method | Data needed | Compute | Changes weights | Latency cost |
|---|---|---|---|---|
| Prompting | 0 | none | no | none |
| Few-shot | 3–10 | none | no | longer prompt |
| RAG | a corpus | embedding pass | no | retrieval hop |
| LoRA | 1k–50k pairs | 1 GPU, hours | adapter only | none |
| Full fine-tune | 50k+ | many GPUs, days | all | none |

> Prompting and RAG fix **knowledge** problems. Fine-tuning fixes **behaviour**
> problems — tone, format, a vocabulary the tokenizer mangles, a task the model
> cannot be talked into.

Fine-tuning to inject facts is the classic waste of a week: they land as weak
statistical pressure, the model still hallucinates, and you own a checkpoint that
must be retrained whenever they change.

---

## Parameter-efficient fine-tuning

![Full fine-tuning versus parameter-efficient adaptation](assets/nlp/finetune2.jpeg)

Full fine-tuning of a 7B model in bf16 needs roughly 14 GB for weights, 14 GB
for gradients and 56 GB for Adam moments — past a consumer GPU before the first
batch.
PEFT freezes the pretrained weights and trains a small added set, so gradients
and optimiser state scale with the adapter — and one base model in memory serves
many tasks by swapping adapters.

---

## LoRA

![Low-rank decomposition of the weight update](assets/nlp/loracompute.png)

The observation: the weight *update* learned during fine-tuning has low
intrinsic rank. Parameterise it as a product of two thin matrices:

$$
W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

$W_0$ is frozen; $B$ is $d \times r$, $A$ is $r \times k$, and $r$ is 4 to 64.
For a 4096×4096 matrix at $r = 8$ that is 65,536 trainable parameters instead of
16.7 million — 0.39%.

---

## LoRA in the forward pass

![LoRA adapters alongside the frozen projection](assets/nlp/loraFlow.gif)

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

$A$ starts Gaussian and $B$ at zero, so the adapter is a no-op at step 0 and
training begins exactly at the pretrained model.

| Knob | What it does | Sane default |
|---|---|---|
| `r` | capacity of the update | 8–16; raise if underfitting |
| `lora_alpha` | scaling — effective LR of the adapter | 2× `r` |
| `target_modules` | which projections get adapters | `q_proj`, `v_proj` first |
| `lora_dropout` | regularisation | 0.05 |

Adapting all four attention projections plus the MLP costs more and helps on
harder tasks. Start narrow.

---

## Running it

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, cfg)
model.print_trainable_parameters()   # trainable: 0.24% of 6,738,415,616
```

Read that line, and better, assert on it — a misspelled entry in
`target_modules` attaches nothing, training runs, the loss barely moves, and an
afternoon goes to a silent no-op that never raised.

---

## QLoRA and quantization

Quantization stores weights at lower precision — 4-bit integers with per-block
scales instead of 16-bit floats — taking a 7B model from 14 GB to about 4 GB.

QLoRA loads the frozen base in 4-bit NF4 and keeps the adapters in bf16. The base
is never updated, so its quantization error is a fixed distortion rather than an
accumulating one.

```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
```

This is what makes fine-tuning a 7B model on a 16 GB GPU possible, at a small
quality loss and a slower step — compute traded for memory you do not have. The
adapter itself saves as tens of megabytes, not tens of gigabytes.

---

## Where all of this lives

![The Hugging Face ecosystem](assets/nlp/huggingface-ecosystem.png)

`transformers` for models, `datasets` for data, `peft` for adapters, `trl` for
SFT and DPO, `accelerate` for devices, `bitsandbytes` for quantization. One
convention, and every checkpoint on the Hub speaks it.

> Read the licence and the model card before you build on a checkpoint. "Open
> weights" is not "open source", and several popular licences forbid the use you
> are about to make.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   d, r = 4096, 8
   print(2 * d * r)                              # -> 65536
   print(d ** 2)                                 # -> 16777216
   print(round(100 * 2 * d * r / d ** 2, 2))     # -> 0.39
   ```

   **Answer.** For one 4096x4096 projection at rank 8, LoRA trains 65,536
   parameters instead of 16.7M — 0.39%. The frozen base still has to fit in
   memory; what collapses is the gradient and optimiser state, which is where
   full fine-tuning of a 7B model spends its 56 GB of Adam moments.

2. You prompt a base checkpoint with "What is the capital of France?" and it
   replies with three more questions. Is the model broken, and what do you do?

   **Answer.** No — it is doing exactly what it was trained to do, continuing the
   pattern, and in the corpus a question is most often followed by another
   question. Use the `-Instruct` variant, and feed it through
   `tokenizer.apply_chat_template`, never a hand-written prompt string.

3. You add a LoRA adapter, training runs cleanly for an hour, and the loss barely
   moves. Name the one line that would have caught it in the first ten seconds.

   **Answer.** `model.print_trainable_parameters()`. A misspelled entry in
   `target_modules` attaches no adapters at all: training proceeds, nothing
   raises, and the trainable percentage is the tell. Assert on it rather than
   reading it.

4. Your model answers fluently but gets your product catalogue wrong. Which rung
   of the adaptation ladder fixes that, and which one will not?

   **Answer.** Retrieval fixes it — the problem is missing *knowledge*. Fine-
   tuning will not: it fixes *behaviour* — tone, format, a task the model cannot
   be talked into — and facts injected as weights land as weak statistical
   pressure that must be retrained whenever they change.
