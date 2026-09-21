"""Shared plumbing for the Session 4 figure generators of
`ms2a-machine-learning-practice`: `ms2a_s4_initialization.py`,
`ms2a_s4_skip_connections.py`, `ms2a_s4_regularization.py` and
`ms2a_s4_dropout.py`. (`ms2a_s4_d2l.py`, the third-party diagrams, stands
alone.)

MNIST comes from torchvision into `courseware/build/data/` (gitignored), scaled
to [0, 1] and flattened to 784. The last 10,000 images of the training split
are the validation set every hyper-parameter is chosen on; the 10,000 official
test images are only ever scored. Training subsets are drawn from the first
50,000, always with the same generator, so "the same 2,000 images" is literal.

Every run caches its numbers in `courseware/build/figures-cache/<name>.json`,
keyed by its task, so adding a value to a sweep measures only that value;
delete the file (or pass `--rerun`) to measure everything again. Runs go five
at a time in a process pool, each on THREADS threads (default 4; 2 on a
10-core laptop). The lessons quote the printed numbers, not the plots.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COURSEWARE = pathlib.Path(__file__).resolve().parents[2]
OUT = COURSEWARE / "content" / "ms2a-machine-learning-practice" / "assets" / "nn"
DATA = COURSEWARE / "build" / "data"
CACHE = COURSEWARE / "build" / "figures-cache"

RERUN = "--rerun" in sys.argv

# The course palette: slots 1-3 of the validated categorical order, the neutral
# used for "train" markers, and the ink and rule colours of every Session 4
# figure already on the site.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b8ae"
LIGHT_BLUE, LIGHT_ORANGE = "#d8e6f7", "#f8d1c1"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SPINE, BAND = "#e1e0d9", "#c3c2b7", "#f1f1ed"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
        "axes.edgecolor": SPINE, "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 1.0, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "text.color": INK_2,
    })


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(COURSEWARE)}")


def cached(name: str, compute):
    """Run `compute()` once and keep its JSON-able result in build/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    if path.exists() and not RERUN:
        return json.loads(path.read_text())
    t0 = time.time()
    result = compute()
    path.write_text(json.dumps(result, indent=1))
    print(f"measured {name} in {time.time() - t0:.0f}s")
    return result


def run_all(name: str, fn, tasks: list[dict], workers: int = 5) -> list[dict]:
    """`fn` over `tasks` in a process pool, each result cached under its task,
    so extending a sweep measures only the tasks that are new."""
    from concurrent.futures import ProcessPoolExecutor

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    key = lambda t: json.dumps(t, sort_keys=True)  # noqa: E731
    done = dict(json.loads(path.read_text())) if path.exists() and not RERUN else {}
    todo = [t for t in tasks if key(t) not in done]
    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for t, r in zip(todo, ex.map(fn, todo)):
                done[key(t)] = r
                path.write_text(json.dumps(list(done.items())))
        print(f"measured {len(todo)} run(s) of {name} in {time.time() - t0:.0f}s")
    return [done[key(t)] for t in tasks]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

_MNIST = None


def mnist():
    """(X_pool, y_pool, X_val, y_val, X_test, y_test), float32 in [0, 1]."""
    global _MNIST
    if _MNIST is None:
        import torchvision

        tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
        te = torchvision.datasets.MNIST(DATA, train=False, download=True)
        X = tr.data.reshape(-1, 784).float() / 255.0
        Xt = te.data.reshape(-1, 784).float() / 255.0
        _MNIST = (X[:50_000], tr.targets[:50_000], X[50_000:], tr.targets[50_000:],
                  Xt, te.targets)
    return _MNIST


def subset(n: int):
    """The first `n` of one fixed permutation of the 50,000-image pool."""
    X, y, *_ = mnist()
    idx = torch.from_numpy(np.random.RandomState(0).permutation(len(X))[:n])
    return X[idx], y[idx]


def shift_and_rotate(xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random affine on a flat (B, 784) batch: ±10°, ±2 px, scale 0.9-1.1."""
    b = xb.shape[0]
    ang = (torch.rand(b, generator=gen) * 2 - 1) * (10 * np.pi / 180)
    sc = 0.9 + 0.2 * torch.rand(b, generator=gen)
    t = (torch.rand(b, 2, generator=gen) * 2 - 1) * (2 / 14)   # ±2 px of 28
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                         torch.stack([sin, cos, t[:, 1]], 1)], 1)
    img = xb.view(b, 1, 28, 28)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).view(b, 784)


# --------------------------------------------------------------------------- #
# Models and the one training loop
# --------------------------------------------------------------------------- #

def mlp(width: int = 512, depth: int = 2, p_drop: float = 0.0,
        p_in: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Dropout(p_in)] if p_in else []
    d = 784
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        if p_drop:
            layers.append(nn.Dropout(p_drop))
        d = width
    layers.append(nn.Linear(d, 10))
    return nn.Sequential(*layers)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()


@torch.no_grad()
def mean_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return F.cross_entropy(model(X), y).item()


def train(model: nn.Module, X, y, *, steps: int, seed: int, lr: float = 1e-3,
          weight_decay: float = 0.0, batch: int = 128, optimizer: str = "adamw",
          l1: float = 0.0, l1_prox: bool = False, noise: float = 0.0,
          augment: bool = False, label_smoothing: float = 0.0,
          eval_every: int = 0, X_val=None, y_val=None, log_every: int = 0):
    """Plain minibatch training for a fixed number of steps.

    With `eval_every`, returns the validation accuracy and loss at each
    evaluation, plus the state dict with the best validation accuracy (what
    early stopping with a restore keeps). With `log_every`, the history is
    instead (step, mean training loss over the last `log_every` steps), with
    step 0 the loss of the first batch before any update.
    """
    gen = torch.Generator().manual_seed(seed)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    weights = [p for p in model.parameters() if p.ndim > 1]
    history, best, window = [], (-1.0, None, 0), []
    n, perm, pos = len(X), None, len(X)
    for step in range(1, steps + 1):
        if pos + batch > n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + batch]
        pos += batch
        xb, yb = X[idx], y[idx]
        if augment:
            xb = shift_and_rotate(xb, gen)
        if noise:
            xb = xb + noise * torch.randn(xb.shape, generator=gen)
        model.train()
        loss = F.cross_entropy(model(xb), yb, label_smoothing=label_smoothing)
        if l1 and not l1_prox:
            loss = loss + l1 * sum(w.abs().sum() for w in weights)
        if log_every:
            if step == 1:
                history.append((0, loss.item()))
            window = (window if step % log_every != 1 else []) + [loss.item()]
            if step % log_every == 0:
                history.append((step, float(np.mean(window))))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if l1 and l1_prox:
            with torch.no_grad():                   # soft-threshold: exact zeros
                for w in weights:
                    w.copy_(w.sign() * (w.abs() - lr * l1).clamp_min(0))
        if eval_every and step % eval_every == 0:
            acc = accuracy(model, X_val, y_val)
            history.append((step, acc, mean_loss(model, X_val, y_val)))
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, step)
    return history, best


def set_threads() -> None:
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
