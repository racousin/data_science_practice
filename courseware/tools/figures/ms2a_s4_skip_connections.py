"""Figures for `s4-advanced-neural-networks/skip-connections.md` of
`ms2a-machine-learning-practice`. Re-run with:

    uv run --no-project --with torch --with torchvision --with matplotlib \
        --with numpy python courseware/tools/figures/ms2a_s4_skip_connections.py

(`--rerun` measures again instead of reading build/figures-cache/.)

The network is a stem `Linear(784, 128)`, a stack of blocks, and a head
`LayerNorm -> Linear(128, 10)`. A block is `Linear(128,128) -> ReLU ->
Linear(128,128)` with a LayerNorm placed by `norm`: "pre" normalises the
block's input (x + F(LN(x)), or F(LN(x)) without the skip), "post" normalises
after the addition (LN(x + F(x))), "none" has no LayerNorm. Data: 10,000 MNIST
training images, the 10,000 validation images for choosing, the 10,000 test
images for scoring. AdamW, batch 128, three seeds.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ms2a_s4_common import *  # noqa: E402,F403

N, WIDTH, SEEDS = 10_000, 128, (0, 1, 2)
DEPTHS = (1, 2, 4, 8, 16, 32)
LRS = (3e-4, 1e-3, 3e-3)
LR_SWEEP = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


class Block(nn.Module):
    def __init__(self, d: int, residual: bool, norm: str):
        super().__init__()
        self.residual, self.norm_at = residual, norm
        self.norm = nn.LayerNorm(d) if norm != "none" else nn.Identity()
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)

    def branch(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def forward(self, x):
        if self.norm_at == "post":
            return self.norm(x + self.branch(x) if self.residual else self.branch(x))
        h = self.branch(self.norm(x))
        return x + h if self.residual else h


class Net(nn.Module):
    def __init__(self, depth: int, residual: bool, norm: str = "pre", d: int = WIDTH):
        super().__init__()
        self.stem = nn.Linear(784, d)
        self.blocks = nn.ModuleList(Block(d, residual, norm) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 10))
        self.skip: set[int] = set()                 # blocks deleted at test time

    def forward(self, x):
        x = self.stem(x)
        for i, b in enumerate(self.blocks):
            if i not in self.skip:
                x = b(x)
        return self.head(x)


def fit(task: dict):
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(N)
    net = Net(task["depth"], task["residual"], task.get("norm", "pre"))
    train(net, X, y, steps=task["steps"], seed=task["seed"], lr=task["lr"])
    return net, X, y


def run(task: dict) -> dict:
    net, X, y = fit(task)
    _, _, Xv, yv, Xt, yt = mnist()
    out = dict(task, train_loss=mean_loss(net, X, y), val_acc=accuracy(net, Xv, yv),
               test_acc=accuracy(net, Xt, yt))
    if task.get("delete"):
        base = out["test_acc"]
        out["deleted"] = []
        for i in range(task["depth"]):
            net.skip = {i}
            out["deleted"].append(accuracy(net, Xt, yt))
        net.skip = set()
        assert accuracy(net, Xt, yt) == base
    return out


def depth_tasks():
    return [dict(depth=d, residual=r, lr=lr, seed=s, steps=2_000)
            for d in DEPTHS for r in (False, True) for lr in LRS for s in SEEDS]


def best_lr(rows, depth, residual):
    m = {}
    for r in rows:
        if r["depth"] == depth and r["residual"] == residual:
            m.setdefault(r["lr"], []).append(r["val_acc"])
    return max(m, key=lambda k: np.mean(m[k]))


def delete_tasks(rows, depth):
    return [dict(depth=depth, residual=r, lr=best_lr(rows, depth, r), seed=s,
                 steps=2_000, delete=True) for r in (False, True) for s in SEEDS]


def norm_tasks():
    return [dict(depth=24, residual=True, norm=nm, lr=lr, seed=s, steps=1_500)
            for nm in ("pre", "post", "none") for lr in LR_SWEEP for s in SEEDS]


@torch.no_grad()
def stream_std():
    """Std of the residual stream after each of 64 blocks, at initialization."""
    set_threads()
    X = subset(N)[0][:1024]
    out = {}
    for name, norm, scale in (("none", "none", None), ("pre", "pre", None),
                              ("pre_scaled", "pre", "sqrt"), ("pre_zero", "pre", "zero")):
        stds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = Net(64, True, norm, d=256)
            for b in net.blocks:
                if scale == "sqrt":
                    b.fc2.weight.mul_(1 / (2 * 64) ** 0.5)
                elif scale == "zero":
                    b.fc2.weight.zero_(); b.fc2.bias.zero_()
            x = net.stem(X)
            s = [x.std().item()]
            for b in net.blocks:
                x = b(x)
                s.append(x.std().item())
            stds.append(s)
        out[name] = np.mean(stds, 0).tolist()
    return out


def grad_by_block():
    """Gradient norm of each block's weights, one backward pass at init."""
    set_threads()
    X, y = subset(N)
    out = {}
    for norm in ("pre", "post"):
        g = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = Net(24, True, norm)
            F.cross_entropy(net(X[:512]), y[:512]).backward()
            g.append([torch.cat([b.fc1.weight.grad.flatten(), b.fc2.weight.grad.flatten()])
                      .norm().item() for b in net.blocks])
        out[norm] = np.mean(g, 0).tolist()
    return out


def delete_depth(rows):
    """The deepest stack whose plain version still trains to within a point and
    a half of the residual one, so that deleting a block compares two working
    networks."""
    ok = [d for d in DEPTHS if d > 2 and np.mean(
        [x["test_acc"] for x in rows if x["depth"] == d and not x["residual"]
         and x["lr"] == best_lr(rows, d, False)]) > np.mean(
        [x["test_acc"] for x in rows if x["depth"] == d and x["residual"]
         and x["lr"] == best_lr(rows, d, True)]) - 0.015]
    return max(ok)


def at_best(rows, d, residual, key):
    lr = best_lr(rows, d, residual)
    return [x[key] for x in rows if x["depth"] == d and x["residual"] == residual
            and x["lr"] == lr]


def fig_depth(rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for residual, color, label in ((False, ORANGE, "plain:  x = f(x)"),
                                   (True, BLUE, "residual:  x = x + f(x)")):
        for ax, key in ((a1, "train_loss"), (a2, "test_acc")):
            vals = [at_best(rows, d, residual, key) for d in DEPTHS]
            m = [np.mean(v) for v in vals]
            ax.fill_between(DEPTHS, [min(v) for v in vals], [max(v) for v in vals],
                            color=color, alpha=0.2, lw=0)
            ax.plot(DEPTHS, m, color=color, marker="o", ms=5, label=label)
    a1.set_yscale("log")
    a1.axhline(np.log(10), color=INK_2, lw=1, ls=(0, (4, 3)))
    a1.text(DEPTHS[0], np.log(10) * 1.25, "ln 10: guessing", fontsize=9.5, color=INK_2)
    a1.set_ylabel("final training cross-entropy")
    a2.set_ylabel("test accuracy")
    a2.legend(loc="lower left")
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEPTHS, [str(d) for d in DEPTHS])
        ax.minorticks_off()
        ax.set_xlabel("blocks (2 Linear layers each), best of 3 learning rates")
    fig.tight_layout()
    save(fig, "residual-depth-sweep.png")


def fig_delete(drows, depth):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    blocks = np.arange(1, depth + 1)
    for residual, color, label in ((False, ORANGE, "plain"), (True, BLUE, "residual")):
        sel = [r for r in drows if r["residual"] == residual]
        dele = np.array([r["deleted"] for r in sel])
        base = np.mean([r["test_acc"] for r in sel])
        ax.axhline(base, color=color, lw=1, ls=(0, (4, 3)))
        ax.plot(blocks, dele.mean(0), color=color, marker="o", ms=6,
                label=f"{label}: {base:.4f} with every block")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(blocks)
    ax.set_xlabel(f"the one block deleted at test time ({depth}-block networks, 3 seeds)")
    ax.set_ylabel("test accuracy (dashed: nothing deleted)")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "residual-delete-one-block.png")


def fig_stream(std):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    series = (("pre", BLUE, "pre-norm, default init"),
              ("pre_scaled", AQUA, "pre-norm, last layer of each branch × 1/√(2L)"),
              ("pre_zero", NEUTRAL, "pre-norm, last layer of each branch zero"))
    for key, color, label in series:
        ax.plot(range(len(std[key])), std[key], color=color, label=label)
        ax.text(len(std[key]) - 1 + 0.8, std[key][-1], f"{std[key][-1]:.2f}",
                va="center", fontsize=10, color=INK_2)
    ax.set_ylim(0, None)
    ax.set_xlabel("block (64 residual blocks of width 256, at initialization)")
    ax.set_ylabel("std of the residual stream")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "residual-stream-std-at-init.png")


def fig_norm(nrows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    series = (("pre", BLUE, "pre-norm"), ("post", ORANGE, "post-norm"),
              ("none", NEUTRAL, "no norm"))
    for nm, color, label in series:
        loss, acc = [], []
        for lr in LR_SWEEP:
            sel = [x for x in nrows if x["norm"] == nm and x["lr"] == lr]
            v = np.array([x["train_loss"] for x in sel], dtype=float)
            loss.append(np.exp(np.mean(np.log(np.clip(v, 1e-8, None))))
                        if np.isfinite(v).all() else np.nan)
            acc.append(np.mean([x["test_acc"] for x in sel]))
        a1.plot(LR_SWEEP, loss, color=color, marker="o", ms=5, label=label)
        a2.plot(LR_SWEEP, acc, color=color, marker="o", ms=5, label=label)
    a1.axhline(np.log(10), color=INK_2, lw=1, ls=(0, (4, 3)))
    a1.text(LR_SWEEP[0], np.log(10) * 1.2, "ln 10: guessing", fontsize=9, color=INK_2)
    blown = [nm for nm, _, _ in series if any(not np.isfinite(x["train_loss"]) for x in nrows
                                              if x["norm"] == nm and x["lr"] == LR_SWEEP[-1])]
    for nm, _, label in series:
        if nm in blown:
            a1.text(LR_SWEEP[-1], np.log(10) * 0.55, f"{label}: nan", ha="right",
                    fontsize=9, color=INK_2)
    a1.set_yscale("log")
    a1.set_ylabel("final training cross-entropy (geometric mean)")
    a2.set_ylabel("test accuracy")
    a2.set_ylim(0, 1)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(LR_SWEEP, [f"{v:g}" for v in LR_SWEEP]); ax.minorticks_off()
        ax.set_xlabel("AdamW learning rate, no warmup (24 residual blocks)")
    a2.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "pre-norm-vs-post-norm.png")


def main():
    style()
    rows = run_all("s4-skip-depth", run, depth_tasks())
    for d in DEPTHS:
        for r in (False, True):
            sel = [x for x in rows if x["depth"] == d and x["residual"] == r
                   and x["lr"] == best_lr(rows, d, r)]
            print(f"depth {d:2d} {'res  ' if r else 'plain'} lr {best_lr(rows, d, r):g}"
                  f" train_loss {np.mean([x['train_loss'] for x in sel]):.2e}"
                  f" test {np.mean([x['test_acc'] for x in sel]):.4f}"
                  f" ({min(x['test_acc'] for x in sel):.4f}-{max(x['test_acc'] for x in sel):.4f})")
    std = cached("s4-skip-stream-std", stream_std)
    grads = cached("s4-skip-grad-by-block", grad_by_block)
    print({k: [round(v[i], 3) for i in (0, 1, 8, 16, 32, 64)] for k, v in std.items()})
    print({k: [f"{v[i]:.2e}" for i in (0, 11, 23)] for k, v in grads.items()})
    depth = delete_depth(rows)
    drows = run_all("s4-skip-delete", run, delete_tasks(rows, depth))
    for r in drows:
        print("delete", depth, "res" if r["residual"] else "plain", round(r["test_acc"], 4),
              [round(v, 3) for v in r["deleted"]])
    nrows = run_all("s4-skip-norm", run, norm_tasks())
    for nm in ("pre", "post", "none"):
        for lr in LR_SWEEP:
            sel = [x for x in nrows if x["norm"] == nm and x["lr"] == lr]
            print(nm, lr, "loss", [f"{x['train_loss']:.2e}" for x in sel],
                  "test", [round(x["test_acc"], 4) for x in sel])
    fig_depth(rows)
    fig_delete(drows, depth)
    fig_stream(std)
    fig_norm(nrows)


if __name__ == "__main__":
    main()
