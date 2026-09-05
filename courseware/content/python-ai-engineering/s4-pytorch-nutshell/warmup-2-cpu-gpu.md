# Warm-up 2 — CPU vs GPU

Ten minutes, and it answers a question you are about to have: **do I need a GPU
for this course?**

The answer is no, and this notebook is how you find that out rather than being
told. No API key needed.

[Open in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-cpu-gpu-benchmark.ipynb)

<!-- notes: 20 minutes. Have them switch the Colab runtime to GPU first —
Runtime > Change runtime type — and note who forgets and gets the n/a columns;
that is a useful mistake to make here rather than during the project. The
crossover section is the one worth discussing: most of them will assume a GPU is
strictly faster. Adapted from the SCAI-4EU workshop; its BERT benchmark is
replaced by the MLP from warm-up 1, which is in scope and does not download
440 MB. -->

---

## What it measures

On whatever machine you run it:

1. **A large matrix multiplication** — the operation a network is mostly made
   of. `nn.Linear(in, out)` computes `X @ W.T + b`; whatever the hardware does
   to a matmul, it does to your model.
2. **Training the MLP from warm-up 1**, scaled up. A real loop also moves data,
   computes a loss, runs backward and updates parameters, and the Python around
   it runs on the CPU either way — so the speedup is smaller than the matmul's.
3. **The crossover** — the model size below which the GPU is *slower*.

The third is the one that matters, and it is the one most people guess wrong.

---

## Two rules for timing GPU code

```python
if device == "cuda":
    torch.cuda.synchronize()
```

**CUDA calls are asynchronous.** They return immediately and the work happens
later. Time a GPU without synchronising and you measure how fast Python can
queue work — often a "1000× speedup" that evaporates the moment you read a
result back.

**Warm up before measuring.** The first call pays for context setup and kernel
selection; the next thousand do not.

---

## The conclusion, in advance

> A GPU pays when the arithmetic per step is large. It does not pay for small
> models, small batches, or work that is really a Python loop wearing a tensor
> costume.

The Session 4 challenge trains an 8-64-64-1 MLP — about 4,800 parameters, in
roughly ten seconds on a Colab CPU. That is on the wrong side of the crossover,
and moving it to a GPU would not measurably help.

You will need one later, for the image and language models in *MS2A - Machine
Learning Practice*. You do not need one this week, and now you can prove it.
