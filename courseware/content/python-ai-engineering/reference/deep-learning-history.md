# A Short History of Deep Learning

Reference only — not covered in class. Useful context for why the field looks
the way it does, and why some ideas took forty years to become practical.

---

## AI, ML, DL

Three nested sets, routinely confused:

- **Artificial intelligence** — any system performing tasks that would need human
  intelligence. Includes rule engines with no learning at all.
- **Machine learning** — systems that improve from data rather than from
  explicit rules.
- **Deep learning** — machine learning with multi-layer neural networks that
  learn their own feature representations.

The distinguishing property of the third is *learned features*. Classical ML
takes features you engineered; deep learning derives them from raw input.

---

## 1943–1969 — the first neural networks

| Year | Contribution |
|---|---|
| 1943 | McCulloch & Pitts — a mathematical model of a neuron |
| 1958 | Rosenblatt — the perceptron, with a learning rule |
| 1969 | Minsky & Papert — a single perceptron cannot learn XOR |

The 1969 result was correct and widely over-read. It applied to a *single* layer;
multi-layer networks were not excluded. Funding collapsed regardless — the first
"AI winter".

---

## 1974–1998 — backpropagation

| Year | Contribution |
|---|---|
| 1974 | Werbos — backpropagation, in a doctoral thesis |
| 1986 | Rumelhart, Hinton & Williams — popularised it |
| 1989 | LeCun — convolutional networks on handwritten digits |
| 1997 | Hochreiter & Schmidhuber — LSTM, addressing vanishing gradients |
| 1998 | LeCun — LeNet-5, deployed on US cheque reading |

The algorithms worked. What was missing was data and compute — a network that
takes weeks to train on a workstation cannot be iterated on.

---

## 2006–2012 — the conditions arrive

| Year | Contribution |
|---|---|
| 2006 | Hinton — layer-wise pretraining makes deep nets trainable |
| 2009 | Deng & Fei-Fei — ImageNet: 14M labelled images |
| 2010 | CUDA makes GPUs generally programmable |
| 2012 | Krizhevsky — **AlexNet** wins ImageNet by ~10 points |

AlexNet is the usual dividing line. Not because the architecture was novel — it
was a CNN, an idea from 1989 — but because ImageNet plus two GPUs plus ReLU plus
dropout made it *work*, publicly and by a wide margin.

---

## The three ingredients

Deep learning did not wait on a theoretical breakthrough. It waited on:

1. **Data** — ImageNet, and later the web at large.
2. **Compute** — GPUs, then TPUs; roughly $10^6\times$ more FLOPs per training
   run over the decade.
3. **Algorithmic detail** — ReLU, dropout, batch normalisation, better
   initialisation. Individually small; collectively the difference between
   trainable and not.

Ideas from 1986 became useful when the other two arrived.

---

## 2014–2017 — architectures

| Year | Contribution |
|---|---|
| 2014 | GANs; VAEs — generative modelling becomes practical |
| 2015 | ResNet — residual connections make 100+ layers trainable |
| 2015 | Batch normalisation |
| 2016 | AlphaGo — deep RL beats a professional Go player |
| 2017 | **Transformer** — "Attention Is All You Need" |

The transformer replaced recurrence with attention, which parallelises across
sequence positions. That made training on far more text feasible, which is what
the next era is built on.

---

## 2018–present — scale

| Year | Contribution |
|---|---|
| 2018 | BERT, GPT — pretrain on unlabelled text, then fine-tune |
| 2020 | GPT-3 — in-context learning at 175B parameters |
| 2020 | AlphaFold 2 — protein structure prediction at experimental accuracy |
| 2022 | Diffusion models; instruction-tuned assistants reach the public |
| 2023– | Multimodal models; long-context; tool-using and agentic systems |

The dominant pattern is now: pretrain a large model on a lot of unlabelled data,
then adapt it. You will do the adapting half in *ML en pratique*.

---

## What this means for you

Three consequences that are practical, not historical:

- **Start from a pretrained model.** Training a vision or language model from
  scratch is almost never the right move at your scale.
- **Old ideas are not dead ideas.** Gradient boosting still wins on tabular data;
  a linear baseline still tells you what your network has to beat.
- **The bottleneck is usually data, not architecture.** It has been for most of
  this timeline, and it is likely to be for your project.

---

## Further reading

- Goodfellow, Bengio & Courville, *Deep Learning* (2016) — free online
- Karpathy, *Neural Networks: Zero to Hero* — builds backprop and a transformer
  from scratch
- The original papers: AlexNet (2012), ResNet (2015), Attention Is All You Need
  (2017). All three are readable and short.
