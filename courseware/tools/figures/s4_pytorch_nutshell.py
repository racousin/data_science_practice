"""Figures for `python-ai-engineering` Session 4 (PyTorch in a Nutshell).

Every image the Session 4 lessons show lives under
`content/python-ai-engineering/assets/s4-pytorch-nutshell/<lesson-slug>/`, and
this script accounts for all of them: it draws the authored ones and copies the
reused ones (the REUSED table below), so an image's provenance is readable here.
Re-run from `courseware/` with:

    uv run --with matplotlib --with numpy --with torch \
        python tools/figures/s4_pytorch_nutshell.py [figure ...]

With no argument it makes every figure; name functions to make only those
(`python ... lr_comparison early_stopping`).

Sizing: every authored figure is 10 in wide at 140 dpi (1400 px), between
2.25:1 and 2.6:1, with 13.5-20 pt text. The slide builder shows them at 0.74-0.97
of that size, so the smallest text still lands at 10 pt.

The numbers are computed, never typed in:
- the autograd and nn.Module figures read their values and shapes off torch;
- batch-and-epoch and the training curves (weights-and-biases, save-and-load)
  use the lab's own files, competitions/s4-taxi-eta/data/X.csv and y.csv, split
  and standardised as the lab's notebook does. data/ is not committed: run that
  package's prepare_data.py first;
- device-time is a benchmark. It refuses to run on anything but the Apple M4
  the lesson names, and its four labelled values are the ones the lesson
  quotes; they move a little from run to run, as any timing does.

Five images that used to be reused are drawn here instead (module1 is
website/public/assets/python-deep-learning/module1, S3 is
content/python-ai-engineering/assets/s3-models-and-tuning/training-neural-networks):

    why-tensors/cpu-vs-gpu.png                    was module1/cpu-vs-gpu.png, 749x399
    why-tensors/gpu-workflow.jpg                  was module1/gpuworkflow.jpg, 889x404
    why-tensors/device-time.png                   was module1/device-time.png, 1188x789
    training-loop-end-to-end/batch-and-epoch.png  was S3/batch-and-epoch.png, 850x278
    save-and-load/early-stopping.png              was S3/early-stopping.png, 1400x1134

At the height their slides give them, the first three had text under 7 pt
(device-time's labels about 4 pt, overlapping), and device-time showed an
unnamed NVIDIA GPU that beats the CPU at every size, where the slide's point is
the M4's GPU losing on small products. batch-and-epoch was enlarged to 85 px/in,
and numbered its batches 1..n where the slide's n counts rows. The
early-stopping still was a 1.2:1 textbook scan, too tall to share a slide.
"""

from __future__ import annotations

import functools
import itertools
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

COURSEWARE = pathlib.Path(__file__).resolve().parents[2]
REPO = COURSEWARE.parent
ASSETS = COURSEWARE / "content" / "python-ai-engineering" / "assets" / "s4-pytorch-nutshell"
LAB = COURSEWARE / "competitions" / "s4-taxi-eta" / "data"

# --------------------------------------------------------------------------- #
# REUSED — images this script copies rather than draws.
#   target (under ASSETS)                          <- source (under the repo root)
# --------------------------------------------------------------------------- #
REUSED = """
why-tensors/flop.jpeg                            <- website/public/assets/python-deep-learning/module1/flop.jpeg
training-loop-end-to-end/batch-sgd-minibatch.png <- courseware/content/python-ai-engineering/assets/s3-models-and-tuning/training-neural-networks/batch-sgd-minibatch.png
"""

# Same palette as the Session 1-3 scripts, so figures from different sessions
# sit on one slide without clashing.
INK = "#1f2933"
MUTED = "#6b7684"
ACCENT = "#2f6f9f"      # forward pass, parameters, the model
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"        # the loss, shapes, "late"
WARM_BG = "#f8e6e1"
GREEN = "#3f8f6f"       # backward pass, gradients
GREEN_BG = "#e2f0ea"
GOLD = "#b5892a"
GOLD_BG = "#faf0d9"
GREY_BG = "#f1f3f5"
RULE = "#dfe3e8"
PAPER = "#ffffff"
MONO = "DejaVu Sans Mono"

W = 10.0     # inches: 1400 px at 140 dpi
FS = 15      # body text, points

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.facecolor": PAPER,
        "font.family": "DejaVu Sans",
        "font.size": FS,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.labelsize": 15.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 1.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14.5,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------- #
# Output and drawing helpers
# --------------------------------------------------------------------------- #
def out(lesson: str, name: str) -> pathlib.Path:
    d = ASSETS / lesson
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def save(fig, lesson: str, name: str) -> None:
    """Write at the figure's exact size: no tight bbox, so every PNG is W x H."""
    p = out(lesson, name)
    kw = {"pil_kwargs": {"quality": 95}} if p.suffix in (".jpg", ".jpeg") else {}
    fig.savefig(p, facecolor=PAPER, **kw)
    w, h = (fig.get_size_inches() * fig.dpi).round().astype(int)
    plt.close(fig)
    print(f"  wrote {lesson}/{name}  ({w}x{h})")


def canvas(h: float, w: float = W):
    """A blank drawing surface whose data units are inches, origin bottom-left."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def chart(h: float, w: float = W):
    return plt.subplots(figsize=(w, h), layout="constrained")


def box(ax, cx, cy, w, h, *, fc=PAPER, ec=INK, lw=1.8, r=0.08, ls="-", z=2):
    ax.add_patch(
        FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    )


def lines(ax, cx, cy, items, *, spacing=1.3, ha="center"):
    """Stack text lines centred on (cx, cy); each item is (text, text-kwargs)."""
    heights = [kw.get("fontsize", FS) * spacing / 72 for _, kw in items]
    y = cy + sum(heights) / 2
    for (text, kw), h in zip(items, heights):
        y -= h
        ax.text(cx, y + h / 2, text, ha=ha, va="center", zorder=4,
                **{"color": INK, "fontsize": FS, **kw})


def arrow(ax, p0, p1, *, color=INK, lw=2.0, rad=0.0, ms=18, z=3, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                        lw=lw, shrinkA=0, shrinkB=0, zorder=z,
                        connectionstyle=f"arc3,rad={rad}")
    )


def end_labels(ax, items, x, min_gap, **kw) -> None:
    """Direct labels at the right end of lines. `items` are (y, text, color);
    labels closer than `min_gap` (data units) are pushed apart, order kept."""
    items = sorted(items)
    ys = [y for y, _, _ in items]
    for _ in range(200):
        moved = False
        for i in range(len(ys) - 1):
            over = min_gap - (ys[i + 1] - ys[i])
            if over > 1e-9:
                ys[i] -= over / 2
                ys[i + 1] += over / 2
                moved = True
        if not moved:
            break
    for y, (_, text, color) in zip(ys, items):
        ax.text(x, y, text, color=color, va="center", clip_on=False,
                **{"fontsize": 14.5, "weight": "bold", **kw})


def num(v) -> str:
    """A tensor or float as the slide writes it: 2, -3 with a real minus sign."""
    return f"{float(v):g}".replace("-", "−")


# --------------------------------------------------------------------------- #
# The lab's data and recipe — shared by batch-and-epoch and the curve figures
# --------------------------------------------------------------------------- #
FEATURES = ["trip_distance", "pickup_hour", "day_of_week", "is_weekend"] + [
    f"{side}_{b}" for side in ("pickup", "dropoff")
    for b in ("manhattan", "queens", "brooklyn", "bronx")
]
TAU = 0.9


def read_lab_csv(name: str, header: list[str]) -> np.ndarray:
    path = LAB / name
    if not path.is_file():
        raise SystemExit(f"{path} is missing. The lab's data/ is not committed: run "
                         f"competitions/s4-taxi-eta/prepare_data.py first.")
    with path.open() as fh:
        got = fh.readline().strip().split(",")
    if got != header:
        raise SystemExit(f"{path}: columns {got}, expected {header}")
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=str)


@functools.cache
def lab_data() -> dict:
    """The lab's X.csv and y.csv, split and standardised as its notebook does.

    41,844 trips in pickup order. The notebook validates on the last 20%, in time
    order (8,369 trips), and trains on the first 33,475, standardised with the
    training part's mean and standard deviation (StandardScaler).
    """
    x = read_lab_csv("X.csv", ["id", *FEATURES])
    y = read_lab_csv("y.csv", ["id", "prediction"])
    if x.shape != (41844, 13) or not np.array_equal(x[:, 0], y[:, 0]):
        raise SystemExit(f"{LAB}: expected 41,844 trips with the same ids in X.csv "
                         f"and y.csv; got {x.shape[0]} and {y.shape[0]} rows")
    X, target = x[:, 1:].astype(np.float64), y[:, 1].astype(np.float64)
    n_fit = int(0.8 * len(target))
    mean, std = X[:n_fit].mean(0), X[:n_fit].std(0)
    Xs = (X - mean) / std
    def as_t(a):
        return torch.tensor(a, dtype=torch.float32)

    return {
        "X_tr": as_t(Xs[:n_fit]), "y_tr": as_t(target[:n_fit]).view(-1, 1),
        "X_val": as_t(Xs[n_fit:]), "y_val": as_t(target[n_fit:]).view(-1, 1),
        "q_tr": float(np.quantile(target[:n_fit], TAU)),
        "q_val": float(np.quantile(target[n_fit:], TAU)),
    }


def pinball(pred, target, tau=TAU):
    d = target - pred
    return torch.maximum(tau * d, (tau - 1) * d).mean()


def make_model(hidden=64):
    return nn.Sequential(nn.Linear(12, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


@functools.cache
def lab_run(lr: float, epochs: int = 30, seed: int = 0) -> tuple[list, list]:
    """The lab's solution loop (lessons 7-9): Adam, batch 256, shuffled, pinball.

    Returns the per-epoch (train_loss, val_loss) the lab logs to W&B. Seeded once
    at the top; the notebook's own run differs in the third decimal because its
    earlier cells draw from the same generator first.
    """
    d = lab_data()
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(d["X_tr"], d["y_tr"]), batch_size=256, shuffle=True)
    train, val = [], []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = pinball(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val.append(pinball(model(d["X_val"]), d["y_val"]).item())
        train.append(total / len(loader.dataset))
    return train, val


# --------------------------------------------------------------------------- #
# why-tensors — few strong cores against thousands of simple ones
# --------------------------------------------------------------------------- #
def cpu_vs_gpu() -> None:
    lesson = "why-tensors"
    fig, ax = canvas(4.0)

    # CPU: four big cores.
    cx, cy, cw, ch = 1.95, 2.45, 2.9, 2.5
    ax.add_patch(Rectangle((cx - cw / 2, cy - ch / 2), cw, ch, fc=GREY_BG, ec=INK, lw=2.2))
    core_w, core_h, gap = 1.25, 1.02, 0.14
    for i in range(2):
        for j in range(2):
            x0 = cx - core_w - gap / 2 + j * (core_w + gap)
            y0 = cy - core_h - gap / 2 + i * (core_h + gap)
            ax.add_patch(Rectangle((x0, y0), core_w, core_h, fc=ACCENT, ec="none"))
            ax.text(x0 + core_w / 2, y0 + core_h / 2, "core", ha="center", va="center",
                    color=PAPER, fontsize=16, weight="bold")

    # GPU: four blocks of small cores, the way the chips group them.
    gx, gy, gw, gh = 6.75, 2.45, 6.1, 2.5
    ax.add_patch(Rectangle((gx - gw / 2, gy - gh / 2), gw, gh, fc=GREY_BG, ec=INK, lw=2.2))
    rows, cols, s, g, block_gap = 6, 16, 0.125, 0.045, 0.22
    bw, bh = cols * (s + g) - g, rows * (s + g) - g
    for bi in range(2):
        for bj in range(2):
            bx0 = gx - bw - block_gap / 2 + bj * (bw + block_gap)
            by0 = gy - bh - block_gap / 2 + bi * (bh + block_gap)
            for r in range(rows):
                for c in range(cols):
                    ax.add_patch(Rectangle((bx0 + c * (s + g), by0 + r * (s + g)), s, s,
                                           fc=ACCENT, ec="none"))

    for x, name, sub in ((cx, "CPU", "a few powerful cores"),
                         (gx, "GPU", "thousands of simple cores")):
        lines(ax, x, 0.55, [(name, {"fontsize": 17, "weight": "bold"}),
                            (sub, {"fontsize": 15, "color": MUTED})])
    save(fig, lesson, "cpu-vs-gpu.png")


# --------------------------------------------------------------------------- #
# why-tensors — two memories and the copy between them
# --------------------------------------------------------------------------- #
def gpu_workflow() -> None:
    lesson = "why-tensors"
    fig, ax = canvas(4.2)

    def step(x, y, n):
        ax.add_patch(plt.Circle((x, y), 0.19, fc=WARM, ec="none", zorder=5))
        ax.text(x, y, str(n), ha="center", va="center", color=PAPER, fontsize=14,
                weight="bold", zorder=6)

    # The two devices, each with its own memory.
    for cx, name, mem, sub, fc, ec in (
        (1.75, "CPU", "RAM", "CPU memory", GREY_BG, INK),
        (8.25, "GPU", "VRAM", "GPU memory, 16–80 GB", ACCENT_BG, ACCENT),
    ):
        box(ax, cx, 2.3, 3.1, 2.0, fc=fc, ec=ec, lw=2.2)
        ax.text(cx, 2.95, name, ha="center", va="center", fontsize=18, weight="bold")
        box(ax, cx, 1.95, 2.6, 0.9, fc=PAPER, ec=ec, lw=1.6, z=3)
        lines(ax, cx, 1.95, [(mem, {"fontsize": 15.5, "weight": "bold"}),
                             (sub, {"fontsize": 13.5, "color": MUTED})])

    # 2 and 4: the copies over the PCIe link.
    arrow(ax, (3.3, 2.75), (6.7, 2.75), color=ACCENT, lw=2.6, ms=22)
    arrow(ax, (6.7, 1.65), (3.3, 1.65), color=ACCENT, lw=2.6, ms=22)
    step(3.75, 3.12, 2)
    ax.text(4.05, 3.12, "copy to GPU memory", va="center", fontsize=15)
    step(3.75, 1.28, 4)
    ax.text(4.05, 1.28, "copy the results back", va="center", fontsize=15)
    lines(ax, 5.0, 2.2, [("PCIe link, ~30 GB/s", {"fontsize": 15, "weight": "bold", "color": WARM}),
                         ("often the slowest step", {"fontsize": 13.5, "color": WARM})])

    # 1: disk -> RAM.   3: compute on the GPU.   5: use the results.
    ax.add_patch(Ellipse((0.95, 0.28), 1.1, 0.22, fc=GOLD_BG, ec=GOLD, lw=1.8, zorder=2))
    ax.add_patch(Rectangle((0.4, 0.28), 1.1, 0.52, fc=GOLD_BG, ec="none", zorder=2))
    ax.plot([0.4, 0.4], [0.28, 0.8], color=GOLD, lw=1.8, zorder=3)
    ax.plot([1.5, 1.5], [0.28, 0.8], color=GOLD, lw=1.8, zorder=3)
    ax.add_patch(Ellipse((0.95, 0.8), 1.1, 0.22, fc=GOLD_BG, ec=GOLD, lw=1.8, zorder=3))
    ax.text(0.95, 0.5, "disk", ha="center", va="center", fontsize=14.5, zorder=4)
    arrow(ax, (1.75, 0.55), (1.75, 1.28), color=GOLD, lw=2.4, ms=20)
    step(2.2, 0.55, 1)
    ax.text(2.5, 0.55, "load the data into RAM", va="center", fontsize=15)

    step(6.95, 0.55, 3)
    ax.text(7.25, 0.55, "compute on the GPU", va="center", fontsize=15)
    step(0.45, 3.72, 5)
    ax.text(0.75, 3.72, "use the results on the CPU", va="center", fontsize=15)
    save(fig, lesson, "gpu-workflow.jpg")


# --------------------------------------------------------------------------- #
# why-tensors — measured: where the M4's GPU starts to pay (a benchmark)
# --------------------------------------------------------------------------- #
DT_SIZES = (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000)
DT_QUOTED = (100, 4000)          # the two sizes the lesson's text quotes
DT_SWEEPS = 3


def fmt_ms(v: float) -> str:
    """As the lesson writes a time: 0.004 ms, 0.2 ms, 44 ms."""
    return f"{v:.0f} ms" if v >= 10 else f"{v:.2g} ms" if v >= 1 else f"{v:.1g} ms"


def device_time() -> None:
    """Time per n x n float32 product on the CPU and on the GPU of an Apple M4,
    the operands already on each device. The GPU pays a fixed cost per call
    (launch the kernel, wait for it) that a small product cannot repay.

    Each point is the median of k synchronised repetitions after a warm-up (the
    protocol the lesson's numbers were measured with), and the lowest of those
    medians over DT_SWEEPS sweeps of all sizes: the median ignores a stray fast
    or slow call, the lowest sweep a background job. On a busy machine the CPU
    points still come out high; make this figure alone (`device_time`) when the
    machine is idle.
    """
    lesson = "why-tensors"
    brand = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                            capture_output=True, text=True, check=True).stdout.strip()
             if sys.platform == "darwin" else sys.platform)
    if brand != "Apple M4" or not torch.backends.mps.is_available():
        raise SystemExit(
            f"device_time measures an Apple M4 and its GPU (mps), the machine the "
            f"lesson's numbers come from; this machine is {brand!r}, mps "
            f"{torch.backends.mps.is_available()}. Make the other figures by name.")

    def median_ms(f, k):
        ts = []
        for _ in range(k):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    def on_gpu(a):
        def f():
            a @ a
            torch.mps.synchronize()
        return f

    warm = torch.randn(2000, 2000)
    warm_gpu = warm.to("mps")
    for _ in range(5):
        warm @ warm
        on_gpu(warm_gpu)()
    sizes = np.array(DT_SIZES, float)
    cpu, gpu = np.full(len(sizes), np.inf), np.full(len(sizes), np.inf)
    for _ in range(DT_SWEEPS):
        for j, n in enumerate(DT_SIZES):
            torch.manual_seed(0)
            a = torch.randn(n, n)
            a_gpu = a.to("mps")
            a @ a
            on_gpu(a_gpu)()
            # More repetitions for the short products: they cost nothing, and a
            # 4 µs call is the easiest one for the rest of the machine to disturb.
            k = 201 if n <= 300 else 51 if n <= 1000 else 21 if n <= 2000 else 9
            cpu[j] = min(cpu[j], median_ms(functools.partial(torch.matmul, a, a), k))
            gpu[j] = min(gpu[j], median_ms(on_gpu(a_gpu), k))
    faster = gpu < cpu
    if faster[0] or not faster[-1]:
        raise SystemExit(f"device_time: expected the GPU to lose at n = {DT_SIZES[0]} and "
                         f"win at n = {DT_SIZES[-1]}; cpu {cpu.round(4)}, gpu {gpu.round(4)} ms")
    # Crossover: where log(cpu / gpu) changes sign, interpolated in log-log.
    i = int(np.argmax(faster))
    r0, r1 = np.log(cpu[i - 1] / gpu[i - 1]), np.log(cpu[i] / gpu[i])
    cross = float(np.exp(np.log(sizes[i - 1]) + (np.log(sizes[i]) - np.log(sizes[i - 1])) * r0 / (r0 - r1)))

    fig, ax = chart(4.05)
    ax.axvspan(80, cross, color=GREY_BG, zorder=0)
    ax.plot(sizes, cpu, "o-", color=WARM, lw=2.6, ms=8, label="Apple M4 CPU", zorder=3)
    ax.plot(sizes, gpu, "s-", color=ACCENT, lw=2.6, ms=8, label="Apple M4 GPU (mps)", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 5200)
    ax.set_ylim(5e-4, 600)
    ax.xaxis.set_major_locator(FixedLocator([100, 300, 1000, 2000, 4000]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("matrix size n, for an n × n product")
    ax.set_ylabel("time per product (ms)")
    ax.text(88, 150, "GPU slower", fontsize=15, color=MUTED, va="center")
    ax.text(cross * 1.1, 150, "GPU faster", fontsize=15, color=MUTED, va="center")
    # The four values the lesson quotes: the faster device's below its point
    # (to the right of it at n = 100, clear of the y-axis), the slower one's above.
    for n in DT_QUOTED:
        j = DT_SIZES.index(n)
        (lo, c_lo), (hi, c_hi) = sorted([(cpu[j], WARM), (gpu[j], ACCENT)])
        x_lo, ha_lo = (n * 1.13, "left") if n == DT_SIZES[0] else (n, "center")
        ax.text(x_lo, lo * 0.7, fmt_ms(lo), color=c_lo, fontsize=15, weight="bold",
                ha=ha_lo, va="top")
        ax.text(n, hi * 1.6, fmt_ms(hi), color=c_hi, fontsize=15, weight="bold",
                ha="center", va="bottom")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.4)
    save(fig, lesson, "device-time.png")
    print("    n      " + " ".join(f"{n:>8}" for n in DT_SIZES))
    print("    cpu ms " + " ".join(f"{v:8.4g}" for v in cpu))
    print("    gpu ms " + " ".join(f"{v:8.4g}" for v in gpu))
    print(f"    GPU faster from n ≈ {cross:,.0f}")


# --------------------------------------------------------------------------- #
# autograd — the running example as a graph: forward values, backward gradients
# --------------------------------------------------------------------------- #
def autograd_graph() -> None:
    """Every number and node name on the figure is read off torch running the
    lesson's example (w = 1, x = 2, target 5), so the picture cannot drift from it."""
    lesson = "autograd"
    w = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    pred = w * x
    err = pred - 5.0
    loss = err ** 2
    for t in (pred, err, loss):
        t.retain_grad()
    loss.backward()
    ops = [type(t.grad_fn).__name__ for t in (pred, err, loss)]
    assert ops == ["MulBackward0", "SubBackward0", "PowBackward0"], ops
    assert (loss.item(), w.grad.item()) == (9.0, -12.0), (loss.item(), w.grad.item())

    fig, ax = canvas(4.15)
    body = {"fontsize": 15}
    name = {"fontsize": 13.5, "family": MONO, "color": MUTED}
    in_x, in_w = 1.02, 1.84
    op_w, xs = 2.3, [3.5, 6.1, 8.7]
    yf, yb, h = 2.7, 1.0, 0.95

    box(ax, in_x, 3.2, in_w, 0.8, fc=ACCENT_BG, ec=ACCENT)
    lines(ax, in_x, 3.2, [(f"w = {num(w)}", body), ("parameter", {"fontsize": 13.5, "color": MUTED})])
    box(ax, in_x, 2.2, in_w, 0.8, fc=GREY_BG, ec=INK)
    lines(ax, in_x, 2.2, [(f"x = {num(x)}", body), ("data", {"fontsize": 13.5, "color": MUTED})])

    forward = [f"pred = w·x = {num(pred)}", f"err = pred − 5 = {num(err)}", f"loss = err² = {num(loss)}"]
    for cx, op, text in zip(xs, ops, forward):
        box(ax, cx, yf, op_w, h, fc=ACCENT_BG, ec=ACCENT)
        lines(ax, cx, yf, [(op, name), (text, body)])
    arrow(ax, (in_x + in_w / 2, 3.2), (xs[0] - op_w / 2, yf + 0.2), color=ACCENT)
    arrow(ax, (in_x + in_w / 2, 2.2), (xs[0] - op_w / 2, yf - 0.2), color=ACCENT)
    for a, b in itertools.pairwise(xs):
        arrow(ax, (a + op_w / 2, yf), (b - op_w / 2, yf), color=ACCENT)

    backward = [
        (xs[2], op_w, "∂loss/∂loss", f"= {num(loss.grad)}"),
        (xs[1], op_w, "∂loss/∂err", f"= 2·err = {num(err.grad)}"),
        (xs[0], op_w, "∂loss/∂pred", f"= {num(err.grad)} × 1 = {num(pred.grad)}"),
        (in_x, in_w, "w.grad =", f"{num(pred.grad)} × x = {num(w.grad)}"),
    ]
    for cx, bw, top, text in backward:
        box(ax, cx, yb, bw, h, fc=GREEN_BG, ec=GREEN)
        weight = "bold" if cx == in_x else "normal"
        lines(ax, cx, yb, [(top, {"fontsize": 14.5, "weight": weight}),
                           (text, {**body, "weight": weight})])
    for (a, aw, *_), (b, bw, *_) in itertools.pairwise(backward):
        arrow(ax, (a - aw / 2, yb), (b + bw / 2, yb), color=GREEN)
    for cx in xs:
        ax.plot([cx, cx], [yf - h / 2, yb + h / 2], ls=":", color=MUTED, lw=1.6, zorder=1)

    left = xs[0] - op_w / 2
    ax.text(left, 3.72, "forward: each operation is recorded as it runs  →",
            fontsize=15, color=ACCENT, va="center")
    ax.text(left, 0.22, "←  backward(): the chain rule, applied right to left",
            fontsize=15, color=GREEN, va="center")
    save(fig, lesson, "autograd-graph.png")


# --------------------------------------------------------------------------- #
# modules-and-optimizers — the lab's MLP, with the shape on every arrow
# --------------------------------------------------------------------------- #
def mlp_shapes() -> None:
    """Shapes and parameter counts are measured on the real nn.Sequential."""
    lesson = "modules-and-optimizers"
    model = make_model()
    B = 256
    h = torch.zeros(B, 12)
    shapes = [tuple(h.shape)]
    for layer in model:
        h = layer(h)
        shapes.append(tuple(h.shape))
    assert all(s[0] == B for s in shapes)
    counts = [sum(p.numel() for p in layer.parameters()) for layer in model]
    assert sum(counts) == 5057, counts

    fig, ax = canvas(3.85)
    y, bh = 2.45, 0.8
    lin_w, relu_w, end_w, gap = 1.75, 0.85, 0.3, 0.36
    sub_fs = 14.5
    # One centre for every layer's notes: a ReLU's single line then sits level
    # with the middle line of its neighbours' three, and cannot run into them.
    sub_y = y - bh / 2 - 0.12 - 3 * sub_fs * 1.25 / 72 / 2
    items = [("X", end_w, None)] + [
        (type(m).__name__ if isinstance(m, nn.ReLU) else f"Linear({m.in_features}, {m.out_features})",
         relu_w if isinstance(m, nn.ReLU) else lin_w, m)
        for m in model
    ] + [("ŷ", end_w, None)]
    total = sum(w for _, w, _ in items) + gap * (len(items) - 1)
    x = (W - total) / 2
    centres = []
    for label, w, m in items:
        cx = x + w / 2
        centres.append((cx, w))
        if m is None:
            ax.text(cx, y, label, ha="center", va="center", fontsize=20, weight="bold")
        else:
            linear = isinstance(m, nn.Linear)
            box(ax, cx, y, w, bh, fc=ACCENT_BG if linear else GREY_BG, ec=ACCENT if linear else INK)
            ax.text(cx, y, label, ha="center", va="center", fontsize=16,
                    weight="bold" if not linear else "normal")
            sub = ([f"W {tuple(m.weight.shape)}", f"b ({m.bias.shape[0]})",
                    f"{sum(p.numel() for p in m.parameters()):,} parameters"]
                   if linear else ["no parameters"])
            lines(ax, cx, sub_y, [(s, {"fontsize": sub_fs, "color": MUTED}) for s in sub],
                  spacing=1.25)
        x += w + gap

    for (a, aw), (b, bw), s in zip(centres[:-1], centres[1:], shapes):
        x0, x1 = a + aw / 2 + 0.03, b - bw / 2 - 0.03
        arrow(ax, (x0, y), (x1, y), lw=2.2)
        ax.text((x0 + x1) / 2, y + bh / 2 + 0.3, f"(B, {s[1]})", ha="center", va="center",
                fontsize=15, color=WARM, family=MONO)

    params = " + ".join(f"{c:,}" for c in counts if c)
    ax.text(W / 2, 0.6, "B = rows in the batch: only the last dimension changes.",
            ha="center", va="center", fontsize=16)
    ax.text(W / 2, 0.2, f"Total: {params} = {sum(counts):,} parameters.",
            ha="center", va="center", fontsize=16, weight="bold")
    save(fig, lesson, "mlp-shapes.png")


# --------------------------------------------------------------------------- #
# losses — pinball against the two losses Session 2 taught
# --------------------------------------------------------------------------- #
def pinball_loss() -> None:
    lesson = "losses"
    r = torch.linspace(-10, 10, 801)                 # residual y - y_hat, minutes
    losses = {
        "squared (MSE)": (r ** 2, {"color": MUTED, "lw": 2.4, "ls": "--"}),
        "absolute (MAE)": (r.abs(), {"color": ACCENT, "lw": 2.4, "ls": "-."}),
        f"pinball, τ = {TAU}": (torch.maximum(TAU * r, (TAU - 1) * r), {"color": WARM, "lw": 3.6}),
    }
    ends = torch.tensor([-10.0, 10.0])
    early, late = torch.maximum(TAU * ends, (TAU - 1) * ends).tolist()

    fig, ax = chart(3.95)
    for label, (v, style) in losses.items():
        ax.plot(r, v, label=label, **style)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("residual  y − ŷ  (minutes)")
    ax.set_ylabel("loss for one trip")
    ax.scatter([-10, 10], [early, late], s=70, color=WARM, zorder=5, clip_on=False)
    ax.annotate(f"early by 10 min:\n{1 - TAU:.1f} × 10 = {early:.1f}", xy=(-10, early),
                xytext=(-9.6, 3.6), fontsize=15, color=WARM, va="center",
                arrowprops={"arrowstyle": "-|>", "color": WARM, "lw": 1.6, "mutation_scale": 16})
    ax.annotate(f"late by 10 min:\n{TAU:.1f} × 10 = {late:.1f}", xy=(10, late),
                xytext=(9.6, 3.6), fontsize=15, color=WARM, va="center", ha="right",
                arrowprops={"arrowstyle": "-|>", "color": WARM, "lw": 1.6, "mutation_scale": 16})
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.6)
    save(fig, lesson, "pinball-loss.png")


# --------------------------------------------------------------------------- #
# training-loop-end-to-end — the lab's epoch, cut into batches
# --------------------------------------------------------------------------- #
def batch_and_epoch() -> None:
    """The lab's 33,475 training trips, cut by the DataLoader the lesson uses.
    The row, batch and update counts are read off that DataLoader."""
    lesson = "training-loop-end-to-end"
    d = lab_data()
    B = 256
    loader = DataLoader(TensorDataset(d["X_tr"], d["y_tr"]), batch_size=B, shuffle=True)
    sizes = [len(xb) for xb, _ in loader]
    n, k = len(loader.dataset), len(loader)
    assert k == -(-n // B) == len(sizes) and sizes[0] == B, (n, k, sizes[:2])
    last = sizes[-1]

    fig, ax = canvas(4.0)
    bar_y, bar_h, up_y = 2.35, 0.9, 3.55
    x0, x1 = 1.85, 9.85
    ell_w = 1.35
    shown = [(1, B), (2, B), None, (k - 1, B), (k, last)]     # None: the batches not drawn
    unit = (x1 - x0 - ell_w) / (3 + last / B)
    lines(ax, 0.95, bar_y, [("training set", {"fontsize": 16, "weight": "bold"}),
                            (f"n = {n:,}", {"fontsize": 15}), ("rows", {"fontsize": 15})])
    x = x0
    for seg in shown:
        if seg is None:
            for yy in (bar_y + bar_h / 2, bar_y - bar_h / 2):
                ax.plot([x, x + ell_w], [yy, yy], color=INK, lw=1.8, ls=(0, (3, 3)))
            lines(ax, x + ell_w / 2, bar_y, [(f"{k - 4} more", {"fontsize": 14.5, "color": MUTED}),
                                             ("batches", {"fontsize": 14.5, "color": MUTED})])
            ax.text(x + ell_w / 2, up_y, "…", ha="center", va="center", fontsize=18, color=ACCENT)
            x += ell_w
            continue
        i, rows = seg
        w = unit * rows / B
        ax.add_patch(Rectangle((x, bar_y - bar_h / 2), w, bar_h, fc=GREY_BG, ec=INK, lw=1.8))
        sub = f"B = {rows} rows" if i == 1 else f"{rows} rows"
        lines(ax, x + w / 2, bar_y, [(f"batch {i}", {"fontsize": 15, "weight": "bold"}),
                                     (sub, {"fontsize": 14, "color": WARM if i == k else INK})])
        arrow(ax, (x + w / 2, bar_y + bar_h / 2 + 0.06), (x + w / 2, up_y - 0.25),
              color=ACCENT, lw=2.0, ms=16)
        ax.text(x + w / 2, up_y, f"update {i}", ha="center", va="center", fontsize=14.5,
                color=ACCENT, weight="bold")
        x += w
    ax.text(0.2, up_y, "opt.step()", va="center", fontsize=14.5, color=ACCENT, family=MONO)

    arrow(ax, (x0, 1.5), (x1, 1.5), style="<|-|>", lw=1.8, ms=16)
    n_tex = f"{n:,}".replace(",", "{,}")                  # no space after the comma
    ax.text((x0 + x1) / 2, 1.07,
            rf"1 epoch = every row once = $\lceil n\,/\,B \rceil$ = "
            rf"$\lceil {n_tex}\,/\,{B} \rceil$ = {k} updates",
            ha="center", va="center", fontsize=16)
    ax.text((x0 + x1) / 2, 0.6, f"the last batch holds the {last} rows left over",
            ha="center", va="center", fontsize=14, color=WARM)
    save(fig, lesson, "batch-and-epoch.png")


# --------------------------------------------------------------------------- #
# training-loop-end-to-end — the loop as one picture
# --------------------------------------------------------------------------- #
def training_loop() -> None:
    lesson = "training-loop-end-to-end"
    fig, ax = canvas(4.4)
    code = {"fontsize": 14, "family": MONO}

    # Dashed: once per epoch.  Blue: once per batch.  Below both: once.
    ep_y0, ep_y1 = 0.62, 4.35
    box(ax, W / 2, (ep_y0 + ep_y1) / 2, W - 0.1, ep_y1 - ep_y0, fc=PAPER, ec=MUTED,
        lw=1.6, ls="--", r=0.15, z=1)
    ax.text(0.3, 4.05, "for epoch in range(E):", va="center", color=MUTED,
            fontsize=15, family=MONO)
    bx0, bx1, by0, by1 = 0.25, 7.05, 0.8, 3.75
    box(ax, (bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0,
        fc="#eef4fa", ec=ACCENT, lw=1.8, r=0.12, z=1)
    ax.text(bx0 + 0.2, by1 - 0.3, "train  ·  model.train()  ·  for xb, yb in loader:",
            va="center", fontsize=14.5, color=ACCENT, weight="bold")

    # No lesson numbers in the boxes: the course reorders its lessons on the
    # website, and a number drawn into a PNG goes stale without anyone noticing.
    steps = ["zero_grad()", "model(xb)", "loss_fn", "backward()"]
    sw, sh, sgap, sy = 1.45, 0.82, 0.22, 2.58
    first = bx0 + ((bx1 - bx0) - (4 * sw + 3 * sgap)) / 2 + sw / 2
    xs = [first + i * (sw + sgap) for i in range(len(steps))]
    for x, text in zip(xs, steps):
        box(ax, x, sy, sw, sh, fc=PAPER, ec=ACCENT)
        lines(ax, x, sy, [(text, code)])
    for a, b in itertools.pairwise(xs):
        arrow(ax, (a + sw / 2, sy), (b - sw / 2, sy), color=ACCENT)
    step_y = 1.4
    box(ax, xs[-1], step_y, sw, sh, fc=PAPER, ec=ACCENT)
    lines(ax, xs[-1], step_y, [("step()", code)])
    arrow(ax, (xs[-1], sy - sh / 2), (xs[-1], step_y + sh / 2), color=ACCENT)
    arrow(ax, (xs[-1] - sw / 2, step_y), (xs[0], sy - sh / 2), color=ACCENT, rad=-0.18)
    ax.text(2.1, 1.12, "next batch", fontsize=14.5, color=ACCENT, va="center")

    # Once per epoch, after the batches.
    rx, rw, gap = 8.55, 2.5, 0.24
    heights = [1.14, 0.72, 0.66]
    centres = [by1 - heights[0] / 2]
    for above, h in itertools.pairwise(heights):
        centres.append(centres[-1] - above / 2 - gap - h / 2)
    contents = [
        (WARM_BG, WARM, [("validate", {"fontsize": 15, "weight": "bold"}),
                         ("model.eval()", code), ("torch.no_grad()", code),
                         ("→ val_loss", code)]),
        (GREY_BG, INK, [("best so far?", {"fontsize": 14.5}),
                        ("keep a deep copy", {"fontsize": 14.5})]),
        (GREY_BG, INK, [("log both losses", {"fontsize": 14.5})]),
    ]
    for cy, h, (fc, ec, items) in zip(centres, heights, contents):
        box(ax, rx, cy, rw, h, fc=fc, ec=ec)
        lines(ax, rx, cy, items, spacing=1.2)
    arrow(ax, (bx1, centres[0]), (rx - rw / 2, centres[0]))
    for (c0, h0), (c1, h1) in zip(zip(centres, heights), zip(centres[1:], heights[1:])):
        arrow(ax, (rx, c0 - h0 / 2), (rx, c1 + h1 / 2))

    ax.text(W / 2, 0.3, "after the loop, once:  model.load_state_dict(best_state), then predict",
            ha="center", va="center", fontsize=14.5)
    save(fig, lesson, "training-loop.png")


# --------------------------------------------------------------------------- #
# save-and-load — why the best epoch is kept, not the last (computed)
# --------------------------------------------------------------------------- #
ES_TRIPS, ES_LR, ES_EPOCHS = 200, 3e-3, 1500


def early_stopping() -> None:
    """The lab's network, make_model() unchanged, trained on only 200 of the lab's
    training trips (full batch: one epoch is one update) and validated on the
    lab's 8,369 validation trips. With so few rows it memorises them: training
    loss keeps falling while validation loss turns back up — the case best_state
    exists for."""
    lesson = "save-and-load"
    d = lab_data()
    torch.manual_seed(0)
    idx = torch.randperm(len(d["y_tr"]))[:ES_TRIPS]
    xs, ys = d["X_tr"][idx], d["y_tr"][idx]
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=ES_LR)
    train, val = [], []
    for _ in range(ES_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = pinball(model(xs), ys)
        loss.backward()
        opt.step()
        train.append(loss.item())
        model.eval()
        with torch.no_grad():
            val.append(pinball(model(d["X_val"]), d["y_val"]).item())
    ep = np.arange(1, ES_EPOCHS + 1)
    best = int(np.argmin(val))
    assert val[-1] > val[best] + 0.3, "validation did not turn back up: no overfitting to show"
    n_val = len(d["y_val"])

    fig, ax = chart(4.05)
    ax.plot(ep, train, color=ACCENT, lw=2.6)
    ax.plot(ep, val, color=WARM, lw=2.6)
    ax.axvline(ep[best], color=INK, ls="--", lw=1.6)
    ax.scatter([ep[best]], [val[best]], s=90, color=WARM, zorder=5)
    ax.text(ep[best] + 25, 2.55, f"best epoch ({ep[best]}): keep this state_dict",
            fontsize=15, va="center")
    ax.scatter([ep[-1]], [val[-1]], s=90, color=WARM, zorder=5, clip_on=False)
    ax.annotate(f"last epoch: {val[-1]:.2f}", xy=(ep[-1], val[-1]),
                xytext=(ep[-1] - 60, val[-1] + 0.5), ha="right", fontsize=15, color=WARM,
                va="center",
                arrowprops={"arrowstyle": "-|>", "color": WARM, "lw": 1.6, "mutation_scale": 16})
    ax.text(ep[-1], train[-1] + 0.12, f"training loss ({ES_TRIPS} trips)", color=ACCENT,
            fontsize=15, ha="right", va="bottom")
    ax.text(ep[best] + 60, val[best] - 0.18, f"validation loss ({n_val:,} trips)",
            color=WARM, fontsize=15, va="top")
    ax.set_xlim(0, ES_EPOCHS + 10)
    ax.set_ylim(0, 2.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("pinball loss (minutes)")
    save(fig, lesson, "early-stopping.png")
    print(f"    early stopping: best val {val[best]:.3f} at epoch {ep[best]}, "
          f"last {val[-1]:.3f}; train {train[best]:.3f} -> {train[-1]:.3f}")


# --------------------------------------------------------------------------- #
# weights-and-biases — the two charts of the lab's run (computed)
# --------------------------------------------------------------------------- #
def train_val_curves() -> None:
    """One run of the lab's solution at lr = 1e-3. The lesson reads three things
    off it: both losses at epoch 30, and a gap that does not grow."""
    lesson = "weights-and-biases"
    d = lab_data()
    train, val = lab_run(1e-3)
    ep = np.arange(1, len(train) + 1)
    top = 1.9

    fig, ax = chart(4.05)
    ax.plot(ep, train, color=ACCENT, lw=2.8, label="train_loss (mean over the epoch's batches)")
    ax.plot(ep, val, color=WARM, lw=2.8, label="val_loss (end of epoch)")
    # The gap between the curves, at epoch 10 and at epoch 30.
    for e, ha, dx in ((10, "center", 0.0), (ep[-1], "right", -0.4)):
        lo, hi = train[e - 1], val[e - 1]
        arrow(ax, (e, lo), (e, hi), style="<|-|>", lw=1.6, ms=12)
        ax.text(e + dx, hi + 0.05, f"gap {hi - lo:.2f}", ha=ha, va="bottom", fontsize=14.5)
    if train[0] > top:
        # Point at where the epoch 1 -> 2 segment of the train curve leaves the chart.
        exit_x = ep[0] + (train[0] - top) / (train[0] - train[1])
        ax.annotate(f"epoch 1 train_loss: {train[0]:.2f}, off the chart", xy=(exit_x, top - 0.01),
                    xytext=(3.4, 1.72), fontsize=14.5, color=ACCENT, va="center",
                    arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.6, "mutation_scale": 16})
    # The epoch-30 values, where the curves end: the lesson quotes them.
    end_labels(ax, [(train[-1], f"{train[-1]:.2f}", ACCENT), (val[-1], f"{val[-1]:.2f}", WARM)],
               ep[-1] + 0.5, 0.07)
    ax.set_xlim(0.5, ep[-1] + 2.6)
    ax.set_ylim(0.85, top)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("epoch")
    ax.set_ylabel("pinball loss (minutes)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2)
    save(fig, lesson, "train-val-curves.png")
    best = int(np.argmin(val))
    print(f"    90th percentile of y: training part {d['q_tr']:.1f}, validation part {d['q_val']:.1f}")
    print(f"    lr 1e-3: epoch 30 train {train[-1]:.3f} val {val[-1]:.3f}; gap at 10 "
          f"{val[9] - train[9]:.3f}, at 30 {val[-1] - train[-1]:.3f}; lowest val "
          f"{val[best]:.3f} at epoch {best + 1}; epoch 1 train {train[0]:.3f}")


LRS = {1e-4: ("lr = 1e-4", GOLD), 1e-3: ("lr = 1e-3", ACCENT),
       1e-2: ("lr = 1e-2", WARM), 1e-1: ("lr = 1e-1", GREEN)}


def lr_comparison() -> None:
    """The same run at four learning rates. The colours are the house palette in
    the order that keeps neighbouring legend entries apart for colour-blind
    readers; the labels at the right carry identity without colour."""
    lesson = "weights-and-biases"
    fig, ax = chart(4.05)
    ends = []
    for lr, (label, color) in LRS.items():
        _, val = lab_run(lr)
        ax.plot(np.arange(1, len(val) + 1), val, color=color, lw=2.6, label=label)
        ends.append((val[-1], label.removeprefix("lr = "), color))
        print(f"    {label}: last {val[-1]:.3f}, lowest {min(val):.3f} at epoch "
              f"{int(np.argmin(val)) + 1}, sd of the last 10 epoch-to-epoch steps "
              f"{np.std(np.diff(val[-10:])):.3f}")
    end_labels(ax, ends, 30.5, 0.045)
    ax.set_xlim(0.5, 32.6)
    ax.set_ylim(0.9, 1.6)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss (pinball, minutes)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=2.2)
    save(fig, lesson, "lr-comparison.png")


# --------------------------------------------------------------------------- #
# Reused images
# --------------------------------------------------------------------------- #
def copy_reused() -> None:
    for line in REUSED.strip().splitlines():
        target, source = (s.strip() for s in line.split("<-"))
        src = REPO / source
        if not src.is_file():
            raise SystemExit(f"reused image source missing: {source}")
        dst = ASSETS / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  copied {target}  <- {source}")


FIGURES = {f.__name__: f for f in (
    copy_reused, cpu_vs_gpu, gpu_workflow, device_time, autograd_graph, mlp_shapes,
    pinball_loss, batch_and_epoch, training_loop, early_stopping, train_val_curves,
    lr_comparison,
)}


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - set(FIGURES))
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    print("Session 4 figures ->", ASSETS)
    for name in names or FIGURES:
        FIGURES[name]()
    print("done.")


if __name__ == "__main__":
    main(sys.argv[1:])
