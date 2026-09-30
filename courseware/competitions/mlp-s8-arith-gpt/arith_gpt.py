"""Arithmetic GPT — the fixed model, the decoding rule and the answer parser.

This one file is both handed out to students and imported by the grader
(env.py), so what you run locally is exactly what the leaderboard runs.
Change nothing in it: the only thing you submit is a weights file.

    from arith_gpt import GPT, save_weights, load_weights, evaluate_problems
    model = GPT()                                     # random weights: train them
    save_weights(model, "weights.safetensors")        # what you upload

Grading, per problem:
  1. the prompt is "a*b=", two 4-digit numbers (1000-9999), e.g. "1234*5678=";
  2. greedy decoding (argmax), at most MAX_NEW_TOKENS new tokens;
  3. the completion must emit EOS ("\\n") within that budget;
  4. the answer is the text after the LAST "#" before EOS, written as a
     plain integer (no sign, no leading zero). Everything before that "#" is
     free scratchpad, so a chain of thought is allowed — it just costs tokens.
"""
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- vocabulary
VOCAB = ["<pad>", "\n"] + list("0123456789") + list("+-*=|>#")
STOI = {c: i for i, c in enumerate(VOCAB)}
PAD, EOS = 0, 1
MAX_NEW_TOKENS = 128


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[i] for i in ids)


# ---------------------------------------------------------------- the model
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = len(VOCAB)
    block_size: int = 160
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128


CONFIG = GPTConfig()


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        heads = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        if cache is not None:
            # Preallocated to block_size: writing in place, not concatenating,
            # keeps each decoding step's cost independent of the cached length.
            if "k" not in cache:
                shape = (B, self.n_head, cache["size"], C // self.n_head)
                cache["k"], cache["v"], cache["n"] = k.new_empty(shape), v.new_empty(shape), 0
            n = cache["n"]
            cache["k"][:, :, n : n + T] = k
            cache["v"][:, :, n : n + T] = v
            cache["n"] = n + T
            k, v = cache["k"][:, :, : n + T], cache["v"][:, :, : n + T]
        # Causal while reading the prompt; one new query attends to everything cached.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and k.shape[2] == T))
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only transformer: learned positions, pre-LN blocks, tied embeddings."""

    def __init__(self, cfg: GPTConfig = CONFIG):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied: one matrix reads and writes tokens
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, caches=None, pos0: int = 0):
        """idx (B, T) -> logits (B, T, vocab). Train with caches=None."""
        T = idx.shape[1]
        if pos0 + T > self.cfg.block_size:
            raise ValueError(f"sequence of {pos0 + T} tokens exceeds block_size {self.cfg.block_size}")
        x = self.tok(idx) + self.pos(torch.arange(pos0, pos0 + T, device=idx.device))
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = MAX_NEW_TOKENS):
        """Greedy decoding with a KV cache. idx: (B, T) prompts of one length.
        Returns (B, n) new token ids, n <= max_new_tokens; stops once every row emitted EOS."""
        caches = [{"size": self.cfg.block_size} for _ in self.blocks]
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        out, step_in, pos0 = [], idx, 0
        for _ in range(min(max_new_tokens, self.cfg.block_size - idx.shape[1])):
            logits = self(step_in, caches, pos0)
            pos0 += step_in.shape[1]
            nxt = logits[:, -1].argmax(-1)
            out.append(nxt)
            done |= nxt == EOS
            if done.all():
                break
            step_in = nxt[:, None]
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------- weights I/O
# The tied head is not stored: it IS tok.weight.
def expected_shapes(cfg: GPTConfig = CONFIG) -> dict:
    return {k: tuple(v.shape) for k, v in GPT(cfg).state_dict().items() if k != "head.weight"}


def save_weights(model: GPT, path: str, dtype=torch.float32) -> None:
    """Write the weights the grader accepts (safetensors; fp32, fp16 or bf16)."""
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu", dtype).contiguous()
          for k, v in model.state_dict().items() if k != "head.weight"}
    save_file(sd, path)


class WeightsError(ValueError):
    """The file is not a valid set of weights for this exact model."""


MAX_WEIGHTS_BYTES = 5 * 1024 * 1024
ALLOWED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def load_weights(path: str) -> GPT:
    """Load a safetensors file into the fixed GPT, or raise WeightsError saying why."""
    import os

    from safetensors import SafetensorError
    from safetensors.torch import load_file

    size = os.path.getsize(path)
    if size > MAX_WEIGHTS_BYTES:
        raise WeightsError(f"the weights file is {size / 1e6:.1f} MB, the limit is "
                           f"{MAX_WEIGHTS_BYTES / 1e6:.0f} MB (fp32 is ~1.7 MB)")
    try:
        sd = load_file(path, device="cpu")
    except (SafetensorError, OSError, ValueError) as exc:
        raise WeightsError(f"not a readable safetensors file ({exc}); save it with "
                           f"arith_gpt.save_weights(model, path)") from None
    expected = expected_shapes()
    missing = sorted(set(expected) - set(sd))
    unknown = sorted(set(sd) - set(expected))
    if missing or unknown:
        raise WeightsError(f"tensor names do not match the model: missing {missing[:5]}, "
                           f"unexpected {unknown[:5]} (the architecture is fixed; "
                           f"'head.weight' is tied to 'tok.weight' and must not be stored)")
    for k, shape in expected.items():
        t = sd[k]
        if tuple(t.shape) != shape:
            raise WeightsError(f"{k} has shape {tuple(t.shape)}, the model expects {shape}")
        if t.dtype not in ALLOWED_DTYPES:
            raise WeightsError(f"{k} is {t.dtype}; use float32, float16 or bfloat16")
        if not torch.isfinite(t).all():
            raise WeightsError(f"{k} contains NaN or infinity")
    model = GPT()
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    return model.eval()


# ---------------------------------------------------------------- the parser
ANSWER_RE = re.compile(r"0|[1-9][0-9]*")


def parse_answer(new_ids) -> int | None:
    """New token ids -> the integer answer, or None when there is no valid one."""
    new_ids = list(new_ids)
    if EOS not in new_ids:
        return None                      # ran out of tokens: no answer
    body = decode(new_ids[: new_ids.index(EOS)])
    if "#" not in body:
        return None
    answer = body.rsplit("#", 1)[1]
    return int(answer) if ANSWER_RE.fullmatch(answer) else None


# ---------------------------------------------------------------- scoring
def prompt_of(a: int, b: int) -> str:
    return f"{a}*{b}="


def digit_accuracy(got: int | None, truth: int) -> float:
    """Share of the answer's digits that are right, aligned from the units digit.
    A missing answer scores 0; extra or missing digits count as wrong ones."""
    if got is None:
        return 0.0
    g, t = str(got), str(truth)
    width = max(len(g), len(t))
    return sum(x == y for x, y in zip(g.zfill(width), t.zfill(width))) / width


@torch.no_grad()
def answer_prompts(model: GPT, prompts: list[str], batch_size: int = 500,
                   deadline: float | None = None) -> list[int | None]:
    """Greedy answers for every prompt, batched by prompt length (learned positions
    rule out left padding). Prompts not reached before `deadline` (time.monotonic())
    get None, i.e. score wrong."""
    import time

    model.eval()
    device = next(model.parameters()).device
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    answers: list[int | None] = [None] * len(prompts)
    for idxs in by_len.values():
        for s in range(0, len(idxs), batch_size):
            if deadline is not None and time.monotonic() > deadline:
                return answers
            chunk = idxs[s : s + batch_size]
            x = torch.tensor([encode(prompts[i]) for i in chunk], device=device)
            for i, row in zip(chunk, model.generate(x).tolist()):
                answers[i] = parse_answer(row)
    return answers


def evaluate_problems(model: GPT, problems, deadline: float | None = None) -> dict:
    """problems: iterable of (a, b, answer). Returns the leaderboard's numbers:
    exact-match accuracy (the ranking), per-digit accuracy, and how many
    problems got a valid answer at all."""
    problems = list(problems)
    answers = answer_prompts(model, [prompt_of(a, b) for a, b, _ in problems], deadline=deadline)
    n = len(problems)
    return {
        "accuracy": sum(got == c for (_, _, c), got in zip(problems, answers)) / n,
        "digit_accuracy": sum(digit_accuracy(got, c) for (_, _, c), got in zip(problems, answers)) / n,
        "answered": sum(a is not None for a in answers),
    }
