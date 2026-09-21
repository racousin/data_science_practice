"""Figures for `s4-advanced-neural-networks/initialization.md` of
`ms2a-machine-learning-practice`. Re-run with:

    uv run --no-project --with torch --with torchvision --with matplotlib \
        --with numpy python courseware/tools/figures/ms2a_s4_initialization.py

(`--rerun` measures again instead of reading build/figures-cache/.)

Plain ReLU MLPs of width 256 — `depth` hidden `Linear -> ReLU` layers, no
normalisation, no skip — on 10,000 MNIST training images, batch 128, 1,500
steps, three seeds, scored on the 10,000 test images. Five initializations of
every Linear weight (biases zero, except under PyTorch's default, which draws
them too): PyTorch's default, Xavier normal, He normal, N(0, 0.01²) "small"
and N(0, 1) "large". Two optimizers: AdamW at 1e-3 and SGD with momentum 0.9 at
0.01. The symmetry experiment is a 784-256-256-10 MLP under AdamW.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ms2a_s4_common import *  # noqa: E402,F403

N, WIDTH, STEPS, SEEDS = 10_000, 256, 1_500, (0, 1, 2)
DEPTHS = (2, 8, 16)
INITS = ("default", "xavier", "he", "small", "large")
OPTS = {"adamw": 1e-3, "sgd": 1e-2}


def plain(depth: int, bn: bool = False) -> nn.Sequential:
    layers, d = [], 784
    for _ in range(depth):
        layers += [nn.Linear(d, WIDTH, bias=not bn)]
        layers += [nn.BatchNorm1d(WIDTH)] if bn else []
        layers += [nn.ReLU()]
        d = WIDTH
    return nn.Sequential(*layers, nn.Linear(d, 10))


def initialize(model: nn.Module, how: str) -> None:
    if how == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            {"xavier": nn.init.xavier_normal_,
             "he": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
             "small": lambda w: nn.init.normal_(w, std=0.01),
             "large": lambda w: nn.init.normal_(w, std=1.0)}[how](m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def run(task: dict) -> dict:
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(N)
    *_, Xt, yt = mnist()
    model = plain(task["depth"], task.get("bn", False))
    initialize(model, task["init"])
    hist, _ = train(model, X, y, steps=STEPS, seed=task["seed"], lr=OPTS[task["opt"]],
                    optimizer=task["opt"], log_every=25)
    return dict(task, history=hist, test_acc=accuracy(model, Xt, yt))


def run_symmetry(task: dict) -> dict:
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(N)
    *_, Xt, yt = mnist()
    model = mlp(256, 2)
    for m in model.modules():
        if isinstance(m, nn.Linear) and task["init"] != "default":
            nn.init.constant_(m.weight, 0.0 if task["init"] == "zeros" else 0.01)
            nn.init.zeros_(m.bias)
    train(model, X, y, steps=STEPS, seed=task["seed"])
    w = model[0].weight.detach()
    return dict(task, test_acc=accuracy(model, Xt, yt),
                rank=int(torch.linalg.matrix_rank(w)),
                distinct_units=len({tuple(r) for r in torch.round(w * 1e5).tolist()}),
                weight_abs_max=w.abs().max().item())


def measure():
    grid = run_all("s4-init-grid", run, [
        dict(depth=d, init=i, opt=o, seed=s) for o in OPTS for d in DEPTHS
        for i in INITS for s in SEEDS])
    bn = run_all("s4-init-bn", run, [
        dict(depth=16, init=i, opt=o, bn=True, seed=s) for o in OPTS
        for i in INITS for s in SEEDS])
    sym = run_all("s4-init-symmetry", run_symmetry, [
        dict(init=i, seed=s) for i in ("zeros", "constant", "default") for s in SEEDS])
    return grid, bn, sym


def finite_mean(values):
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if len(v) == len(values) else float("nan")


def report(grid, bn, sym):
    for rows, tag in ((grid, ""), (bn, " +BN")):
        for o in OPTS:
            for d in sorted({r["depth"] for r in rows}):
                for i in INITS:
                    sel = [r for r in rows if r["opt"] == o and r["depth"] == d and r["init"] == i]
                    first = [r["history"][0][1] for r in sel]
                    last = [r["history"][-1][1] for r in sel]
                    print(f"{o:5s} depth {d:2d}{tag} {i:8s} loss@0 {finite_mean(first):10.4g}"
                          f" loss@end {finite_mean(last):10.4g}"
                          f" test {np.mean([r['test_acc'] for r in sel]):.4f}"
                          f" {[round(r['test_acc'], 4) for r in sel]}")
    for r in sym:
        print(r)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

# The colours of signal-and-gradient-std-across-depth-by-init.png, which the
# lesson shows first: default blue, Xavier orange, He green, sigma = 1 red.
COLORS = {"default": BLUE, "xavier": ORANGE, "he": AQUA, "small": "#eda100",
          "large": "#e34948"}
LABELS = {"default": "PyTorch default", "xavier": "Xavier", "he": "He (Kaiming)",
          "small": "N(0, 0.01²)", "large": "N(0, 1)"}


def fig_training(grid, bn, opt):
    panels = [(f"{d} hidden layers", [r for r in grid if r["depth"] == d]) for d in DEPTHS]
    panels.append(("16 hidden layers + BatchNorm", bn))
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    for ax, (title, rows) in zip(axes, panels):
        stuck, blown = [], []
        for i in INITS:
            sel = [r for r in rows if r["opt"] == opt and r["init"] == i]
            steps = [s for s, _ in sel[0]["history"]]
            curves = np.array([[l for _, l in r["history"]] for r in sel], dtype=float)
            curves[~np.isfinite(curves)] = np.nan
            if np.isnan(curves[:, -1]).any():
                blown.append(LABELS[i])
            elif np.all(curves[:, -1] > 2.0):
                stuck.append(LABELS[i])
            with np.errstate(all="ignore"):
                m = np.nanmean(curves, 0)
            ax.plot(steps[1:], m[1:], color=COLORS[i], label=LABELS[i], lw=2)
        ax.axhline(math.log(10), color=INK_2, lw=1, ls=(0, (4, 3)))
        note = "\n".join([f"at ln 10: {', '.join(stuck)}"] * bool(stuck)
                         + [f"nan: {', '.join(blown)}"] * bool(blown))
        if note:
            ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK_2, bbox=dict(fc="white", ec="none", pad=1))
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 30)
        ax.set_xlabel("step")
    axes[-1].text(1500, math.log(10) * 1.3, "ln 10: guessing", ha="right", fontsize=9,
                  color=INK_2)
    axes[0].set_ylabel("training cross-entropy (25-step mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "init-scale-by-depth.png")


def main():
    style()
    grid, bn, sym = measure()
    report(grid, bn, sym)
    fig_training(grid, bn, "sgd")      # the lesson quotes AdamW's numbers in text


if __name__ == "__main__":
    main()
