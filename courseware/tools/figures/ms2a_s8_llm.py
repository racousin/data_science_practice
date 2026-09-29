"""Figures for the Session 8 lessons of `ms2a-machine-learning-practice` that
cover post-training (`post-training.md`) and agent tooling
(`agent-tooling-and-mcp.md`).

Three are schematic diagrams drawn here, because no CC-licensed textbook figure
exists for them (the post-training pipeline, the RLHF loop, the MCP host /
client / server layout). Three are computed:

- `grpo-advantages.png`: the group-normalised advantages of eight sampled
  answers, from the GRPO formula in the lesson.
- `distillation-soft-targets.png`: GPT-2's real next-token distribution after
  "Every morning I drink a cup of", next to the one-hot label (" tea") that
  SFT on that sentence would train on. Needs `transformers` and the `gpt2` checkpoint (~500 MB).
- `pass-at-k.png`: pass@k against pass^k for three per-trial success rates.

Re-run with:

    uv run --with matplotlib --with torch --with transformers \
        python courseware/tools/figures/ms2a_s8_llm.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content"
       / "ms2a-machine-learning-practice" / "assets" / "nlp")

# The course palette (see ms2a_s4_common.py).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b8ae"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_AQUA = "#d8e6f7", "#f8d1c1", "#d3efe4"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SPINE, BAND = "#e1e0d9", "#c3c2b7", "#f1f1ed"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
        "axes.edgecolor": SPINE, "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 1.0, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 12, "axes.titlecolor": INK, "text.color": INK,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- #
# Diagram helpers
# --------------------------------------------------------------------------- #

def box(ax, x, y, w, h, title, body="", fill=BAND, edge=SPINE, size=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.4))
    if body:
        ax.text(x, y + h * 0.18, title, ha="center", va="center",
                fontsize=size, weight="bold", color=INK)
        ax.text(x, y - h * 0.2, body, ha="center", va="center",
                fontsize=size - 2, color=INK_2, linespacing=1.3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=size,
                weight="bold", color=INK)


def arrow(ax, x0, y0, x1, y1, label="", color=INK_2, rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center",
                va="bottom", fontsize=9, color=INK_2, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #
# Diagrams
# --------------------------------------------------------------------------- #

def pipeline() -> None:
    fig, ax = canvas(13, 3.6)
    stages = [
        ("Pretraining", "next-token loss\n~15T tokens of web,\ncode, books", LIGHT_BLUE),
        ("SFT", "imitate answers\n10k–1M written\ndemonstrations", BAND),
        ("Preference", "RLHF / DPO\n100k+ ranked\nanswer pairs", LIGHT_ORANGE),
        ("Reasoning RL", "GRPO on\nverifiable rewards\nmath, code, tests", LIGHT_AQUA),
        ("Distillation", "SFT a small model\non the big one's\nreasoning traces", BAND),
    ]
    xs = np.linspace(1.35, 11.65, len(stages))
    for x, (title, body, fill) in zip(xs, stages):
        box(ax, x, 2.0, 2.2, 1.9, title, body, fill=fill)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 1.12, 2.0, b - 1.12, 2.0)
    ax.text(xs[0], 0.62, "base model", ha="center", fontsize=10, color=MUTED)
    ax.text((xs[1] + xs[2]) / 2, 0.62, "assistant", ha="center", fontsize=10,
            color=MUTED)
    ax.text(xs[3], 0.62, "reasoning model", ha="center", fontsize=10, color=MUTED)
    ax.text(xs[4], 0.62, "small reasoning model", ha="center", fontsize=10,
            color=MUTED)
    ax.annotate("", xy=(xs[1] - 1.1, 0.4), xytext=(xs[4] + 1.1, 0.4),
                arrowprops=dict(arrowstyle="-", color=SPINE, lw=1))
    ax.text(6.5, 0.1, "post-training: 1–5% of the compute, most of the behaviour",
            ha="center", fontsize=10, color=INK_2)
    save(fig, "post-training-pipeline.png")


def rlhf_loop() -> None:
    fig, ax = canvas(11, 4.6)
    box(ax, 1.4, 2.3, 2.0, 1.1, "Prompt x", "from the dataset")
    box(ax, 4.6, 3.4, 2.4, 1.3, "Policy πθ", "being trained", fill=LIGHT_BLUE)
    box(ax, 4.6, 1.1, 2.4, 1.3, "Reference π_ref", "frozen SFT copy")
    box(ax, 8.3, 3.4, 2.6, 1.3, "Reward model rφ",
        "trained on human\npreference pairs", fill=LIGHT_ORANGE)
    box(ax, 8.3, 1.1, 2.6, 1.3, "Objective", "r(x, y) − β·KL(πθ ‖ π_ref)",
        fill=LIGHT_AQUA)
    arrow(ax, 2.4, 2.55, 3.4, 3.2)
    arrow(ax, 2.4, 2.05, 3.4, 1.3)
    arrow(ax, 5.8, 3.4, 7.0, 3.4, "answer y")
    arrow(ax, 8.3, 2.75, 8.3, 1.75, "score")
    arrow(ax, 5.8, 1.1, 7.0, 1.1, "log π_ref(y|x)")
    arrow(ax, 7.0, 1.4, 5.3, 2.75, color=AQUA, rad=-0.25)
    ax.text(5.55, 1.95, "PPO update", fontsize=9, color=AQUA, style="italic")
    save(fig, "rlhf-loop.png")


def mcp_architecture() -> None:
    fig, ax = canvas(12, 5.2)
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 4.4, 4.4,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=SPINE, lw=1.4, ls="--"))
    ax.text(2.5, 4.5, "Host", ha="center", fontsize=12, weight="bold")
    ax.text(2.5, 4.15, "Claude Code, an IDE, a chat app", ha="center",
            fontsize=9, color=INK_2)
    box(ax, 2.5, 3.2, 3.2, 0.8, "LLM + agent loop", fill=LIGHT_BLUE)
    clients = [2.4, 1.6, 0.8]
    for y in clients:
        box(ax, 3.4, y, 1.9, 0.55, "MCP client", size=10)
    ax.text(1.25, 1.6, "one client\nper server", ha="center", fontsize=9,
            color=MUTED)
    servers = [
        (2.4, "Filesystem server", "local process"),
        (1.6, "Postgres server", "local process"),
        (0.8, "Ticket-tracker server", "remote, OAuth"),
    ]
    for y, title, where in servers:
        box(ax, 9.6, y, 3.4, 0.62, title, fill=LIGHT_ORANGE, size=10)
        ax.text(11.45, y, where, ha="left", va="center", fontsize=8.5,
                color=MUTED)
    arrow(ax, 4.35, 2.4, 7.9, 2.4, "stdio (JSON-RPC)", dy=0.08)
    arrow(ax, 4.35, 1.6, 7.9, 1.6, "stdio (JSON-RPC)", dy=0.08)
    arrow(ax, 4.35, 0.8, 7.9, 0.8, "Streamable HTTP (JSON-RPC)", dy=0.08)
    ax.text(9.6, 3.35, "server exposes:  tools · resources · prompts",
            ha="center", fontsize=10, color=INK_2)
    ax.text(9.6, 3.0, "client offers:  sampling · roots · elicitation",
            ha="center", fontsize=10, color=INK_2)
    save(fig, "mcp-architecture.png")


# --------------------------------------------------------------------------- #
# Computed figures
# --------------------------------------------------------------------------- #

def grpo_advantages() -> None:
    rewards = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    adv = (rewards - rewards.mean()) / rewards.std()
    print("GRPO advantages:", np.round(adv, 2))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [AQUA if r else ORANGE for r in rewards]
    bars = ax.bar(np.arange(1, 9), adv, color=colors, width=0.62)
    ax.axhline(0, color=SPINE, lw=1)
    for b, r, a in zip(bars, rewards, adv):
        ax.text(b.get_x() + b.get_width() / 2, a + (0.08 if a > 0 else -0.08),
                f"r={int(r)}\n{a:+.2f}", ha="center",
                va="bottom" if a > 0 else "top", fontsize=9, color=INK_2)
    ax.set_xticks(np.arange(1, 9), [f"y{i}" for i in range(1, 9)])
    ax.set_ylim(-1.3, 2.1)
    ax.set_ylabel("advantage  A_i")
    ax.set_title("One prompt, eight sampled answers, 3 correct", loc="left")
    save(fig, "grpo-advantages.png")


def distillation_soft_targets() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok("Every morning I drink a cup of", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top = torch.topk(logits, 6).indices
    words = [repr(tok.decode(int(i)))[1:-1] for i in top]
    p1 = torch.softmax(logits, -1)[top].numpy()
    print("T=1:", dict(zip(words, np.round(p1, 3))))

    x = np.arange(len(words))
    w = 0.36
    onehot = np.zeros(len(words))
    onehot[words.index(" tea")] = 1.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2 - 0.02, onehot, w, color=NEUTRAL,
           label="one-hot label: what SFT trains on")
    ax.bar(x + w / 2 + 0.02, p1, w, color=BLUE,
           label="teacher distribution: what distillation trains on")
    ax.set_xticks(x, [f"'{s}'" for s in words], fontsize=9)
    ax.set_ylabel("probability")
    ax.set_title('GPT-2 next token after "Every morning I drink a cup of"',
                 loc="left")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "distillation-soft-targets.png")


def pass_at_k() -> None:
    k = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for p, c in [(0.5, ORANGE), (0.8, BLUE), (0.95, AQUA)]:
        ax.plot(k, 1 - (1 - p) ** k, color=c, lw=2)
        ax.plot(k, p ** k, color=c, lw=2, ls="--")
        ax.text(10.2, p ** 10, f"pass^k, p={p}", va="center", fontsize=9,
                color=INK_2)
    ax.text(10.2, 1.0, "pass@k (all three)", va="center", fontsize=9,
            color=INK_2)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k)
    ax.set_xlabel("k, number of trials")
    ax.set_ylabel("probability")
    ax.set_title("At least one success (solid) vs every trial succeeds (dashed)",
                 loc="left")
    save(fig, "pass-at-k.png")


def main() -> None:
    style()
    pipeline()
    rlhf_loop()
    mcp_architecture()
    grpo_advantages()
    pass_at_k()
    distillation_soft_targets()


if __name__ == "__main__":
    main()
