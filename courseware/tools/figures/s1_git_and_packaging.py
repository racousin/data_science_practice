"""Figures generated for `python-ai-engineering` Session 1.

Session 1 has no taught PPTX to lift stills from — unlike Sessions 2-4, whose
images come out of `pptxs/`. Every diagram it uses is therefore authored here,
which keeps the rule the rest of the course follows: a figure can always be
traced back to the code that drew it.

    uv run --with matplotlib --with numpy \
        python courseware/tools/figures/s1_git_and_packaging.py

Writes into
`content/python-ai-engineering/assets/s1-git-and-packaging/<lesson-slug>/`,
one directory per lesson, same convention as the rest of the course.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "python-ai-engineering"
    / "assets"
    / "s1-git-and-packaging"
)

# Same palette as tools/figures/s2_ml_foundations.py, so the two sessions'
# figures sit on a slide together without clashing.
INK = "#1f2933"
MUTED = "#6b7684"
ACCENT = "#2f6f9f"
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"
WARM_BG = "#f8e6e1"
GREEN = "#3f8f6f"
GREEN_BG = "#e2f0ea"
GOLD = "#b5892a"
GOLD_BG = "#faf0d9"
PAPER = "#ffffff"
GREY_BG = "#f1f3f5"
RULE = "#dfe3e8"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": PAPER,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)

MONO = "DejaVu Sans Mono"


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def out(lesson: str, name: str) -> pathlib.Path:
    d = ASSETS / lesson
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# Vertical extent actually drawn, in figure units. Every helper records what it
# covers so `save` can crop the canvas to the content — hand-tuning ylim on
# thirty diagrams is how you end up with one that clips its own caption.
_SPAN: list[float] = []
_WIDTH_IN = 10.5


def _note(*ys: float) -> None:
    _SPAN.extend(ys)


def canvas(w: float = 10.5, h: float = 5.2):
    """A blank drawing surface with square units, 0..100 across.

    `h` only seeds the aspect; `save` recomputes it from what was drawn.
    """
    global _SPAN, _WIDTH_IN
    _SPAN = []
    _WIDTH_IN = w
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, lesson: str, name: str, *, pad: float = 2.5) -> None:
    ax = fig.axes[0]
    lo, hi = (min(_SPAN) - pad, max(_SPAN) + pad) if _SPAN else (0.0, 50.0)
    ax.set_ylim(lo, hi)
    fig.set_size_inches(_WIDTH_IN, _WIDTH_IN * (hi - lo) / 100.0)
    path = out(lesson, name)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)
    print(f"  {lesson}/{name}")


def box(ax, x, y, w, h, *, fc=PAPER, ec=INK, lw=1.4, r=1.4, z=2, ls="-", alpha=1.0):
    _note(y, y + h)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            zorder=z, linestyle=ls, alpha=alpha,
        )
    )


def label(ax, x, y, text, *, fs=10, color=INK, weight="normal", ha="center",
          va="center", mono=False, z=4, style="normal"):
    # 1 unit is width/100 inches; a text line is ~1.45 * fs points tall.
    half = 1.45 * fs * (100.0 / (72.0 * _WIDTH_IN)) * (text.count("\n") + 1) / 2
    _note(y - half, y + half)
    ax.text(
        x, y, text, fontsize=fs, color=color, fontweight=weight,
        ha=ha, va=va, zorder=z, fontstyle=style,
        family=MONO if mono else None,
        linespacing=1.45,
    )


def titled_box(ax, x, y, w, h, title, body=None, *, fc=PAPER, ec=INK,
               tc=INK, fs_t=11, fs_b=9, mono_body=False, lw=1.4):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if body is None:
        label(ax, x + w / 2, y + h / 2, title, fs=fs_t, weight="bold", color=tc)
    else:
        label(ax, x + w / 2, y + h - h * 0.30, title, fs=fs_t, weight="bold", color=tc)
        label(ax, x + w / 2, y + h * 0.34, body, fs=fs_b, color=MUTED,
              mono=mono_body, va="center")


def arrow(ax, p0, p1, *, color=INK, lw=1.5, style="-|>", rad=0.0, z=3, ls="-",
          ms=9):
    _note(p0[1], p1[1])
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=ms,
            linewidth=lw, color=color, zorder=z, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2, shrinkB=2,
        )
    )


def caption(ax, x, y, text, *, fs=9, color=MUTED, ha="center", mono=False):
    label(ax, x, y, text, fs=fs, color=color, ha=ha, mono=mono, style="italic")


def commit(ax, x, y, sha, *, r=2.6, fc=ACCENT_BG, ec=ACCENT, tc=INK, fs=8.5):
    _note(y - r, y + r)
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    label(ax, x, y, sha, fs=fs, color=tc, mono=True, z=4)


def chain(ax, xs, y, shas, **kw):
    for i, (x, s) in enumerate(zip(xs, shas)):
        if i:
            arrow(ax, (xs[i - 1] + 2.9, y), (x - 2.9, y), color=kw.get("ec", ACCENT),
                  lw=1.4, ms=8)
        commit(ax, x, y, s, **kw)


# --------------------------------------------------------------------------- #
# session-map
# --------------------------------------------------------------------------- #

def session_map() -> None:
    lesson = "session-map"

    fig, ax = canvas(11, 5.0)
    stages = [
        ("Your laptop", "editor + shell\n+ .venv", ACCENT_BG, ACCENT),
        ("Git", "commits,\nbranches", GOLD_BG, GOLD),
        ("GitHub", "remote, PR,\nreview", WARM_BG, WARM),
        ("CI", "tests run on\nsomeone else's\nmachine", GREEN_BG, GREEN),
        ("Colab", "pip install\nfrom GitHub", ACCENT_BG, ACCENT),
    ]
    w, h, gap = 15.5, 15.0, 4.6
    x = 3.0
    y = 20.0
    for title, body, fc, ec in stages:
        titled_box(ax, x, y, w, h, title, body, fc=fc, ec=ec, tc=ec, fs_t=12, fs_b=8.5)
        if x > 3.0:
            arrow(ax, (x - gap + 0.4, y + h / 2), (x - 0.6, y + h / 2), color=MUTED, lw=1.6)
        x += w + gap

    label(ax, 50, 41.5, "One artifact, five places it has to survive",
          fs=13, weight="bold")
    caption(ax, 50, 13.5,
            "Session 1 is this line, left to right. Every later session assumes all five.")

    for x0, txt in zip(np.arange(3.0, 100, w + gap)[:5],
                       ["git init", "git push", "pull request", "green check",
                        "import mypkg"]):
        label(ax, x0 + w / 2, 16.5, txt, fs=8.5, mono=True, color=INK)
    save(fig, lesson, "session-map.png")

    # --- what you own at the end -------------------------------------------- #
    fig, ax = canvas(10, 5.4)
    label(ax, 50, 50, "The deliverable", fs=13, weight="bold")
    rows = [
        ("a GitHub repository", "public, with a README a stranger can follow"),
        ("an installable package", "src/ layout, pyproject.toml, version"),
        ("a test suite", "pytest, green, covering the failure paths"),
        ("a CI workflow", "runs on every push and every pull request"),
        ("a merged pull request", "with a review comment that is not 'LGTM'"),
        ("a Colab notebook", "that pip-installs the package from GitHub"),
        ("an agent on a leaderboard", "Connect-Four, rated against other people's"),
    ]
    y = 44
    for i, (what, detail) in enumerate(rows):
        fc = GREEN_BG if i in (0, 5, 6) else GREY_BG
        box(ax, 8, y - 4.2, 84, 5.2, fc=fc, ec=RULE, lw=1.0)
        label(ax, 12.5, y - 1.6, "✓", fs=12, color=GREEN, weight="bold")
        label(ax, 17, y - 1.6, what, fs=10.5, weight="bold", ha="left")
        label(ax, 49, y - 1.6, detail, fs=9, color=MUTED, ha="left")
        y -= 6.1
    caption(ax, 50, 1.5,
            "Nothing here is graded on machine learning. All seven are engineering.")
    save(fig, lesson, "the-deliverable.png")


# --------------------------------------------------------------------------- #
# accounts-and-tools
# --------------------------------------------------------------------------- #

def accounts_and_tools() -> None:
    lesson = "accounts-and-tools"

    fig, ax = canvas(11, 5.2)
    # A shell sits between git and uv: on Windows it is what the git installer
    # just handed you (Git Bash), and it is what the uv install line is typed
    # into. Six boxes need a narrower one than five did, so w/gap shrink to keep
    # the row centred on the same span.
    items = [
        ("GitHub", "account", "where the work\nis handed in", WARM_BG, WARM),
        ("VS Code", "install", "editor, terminal,\nnotebooks, agent", ACCENT_BG, ACCENT),
        ("git", "install", "the version\ncontrol client", GOLD_BG, GOLD),
        ("a shell", "set up", "where every\ncommand runs", GREY_BG, MUTED),
        ("uv", "install", "Python + envs\n+ packages", GREEN_BG, GREEN),
        ("Colab", "account", "a GPU you\ndo not own", ACCENT_BG, ACCENT),
    ]
    w, gap, h = 13.8, 2.6, 21.0
    x = (100.0 - (len(items) * w + (len(items) - 1) * gap)) / 2
    for name, kind, why, fc, ec in items:
        box(ax, x, 12, w, h, fc=fc, ec=ec, lw=1.5)
        label(ax, x + w / 2, 29.5, name, fs=12, weight="bold", color=ec)
        label(ax, x + w / 2, 25.6, kind, fs=8.5, color=MUTED, mono=True)
        label(ax, x + w / 2, 18.0, why, fs=8.5, color=INK)
        x += w + gap
    label(ax, 50, 42, "Six things, thirty minutes, once", fs=13, weight="bold")
    caption(ax, 50, 6.5,
            "If any one of these is missing you spend Session 3 debugging it "
            "instead of training a model.")
    save(fig, lesson, "toolchain.png")

    # --- three places code runs --------------------------------------------- #
    fig, ax = canvas(10.5, 5.0)
    cols = [
        ("Your laptop", ACCENT, ACCENT_BG,
         ["you install it", "you own the files", "no GPU", "survives a reboot"]),
        ("A CI runner", GREEN, GREEN_BG,
         ["GitHub installs it", "fresh every run", "no GPU", "destroyed at the end"]),
        ("A Colab VM", WARM, WARM_BG,
         ["Google installs it", "borrowed files", "free GPU", "destroyed in ~90 min"]),
    ]
    w, gap = 27.0, 4.0
    x = 5.0
    for name, ec, fc, bullets in cols:
        box(ax, x, 6, w, 32, fc=fc, ec=ec, lw=1.5)
        label(ax, x + w / 2, 34, name, fs=12, weight="bold", color=ec)
        yy = 28.5
        for b in bullets:
            label(ax, x + 3, yy, "•", fs=10, color=ec, ha="left")
            label(ax, x + 6, yy, b, fs=9.5, ha="left")
            yy -= 5.4
        x += w + gap
    label(ax, 50, 43, "The same code, three machines", fs=13, weight="bold")
    caption(ax, 50, 1.8,
            "\"It works on my machine\" names exactly one of these three.")
    save(fig, lesson, "where-code-runs.png")


# --------------------------------------------------------------------------- #
# the-shell
# --------------------------------------------------------------------------- #

def the_shell() -> None:
    lesson = "the-shell"

    # --- filesystem tree and paths ------------------------------------------ #
    fig, ax = canvas(10.5, 5.4)
    nodes = {
        "/": (10, 44),
        "home": (24, 44),
        "marie": (40, 44),
        "projects": (58, 50),
        "Downloads": (58, 38),
        "textstats": (83, 50),
    }
    for name, (x, y) in nodes.items():
        w = 4.0 + 2.0 * len(name)
        box(ax, x - w / 2, y - 3.0, w, 6.0, fc=GREY_BG, ec=MUTED, lw=1.2)
        label(ax, x, y, name, fs=9.5, mono=True)
    edges = [("/", "home"), ("home", "marie"), ("marie", "projects"),
             ("marie", "Downloads"), ("projects", "textstats")]
    for a, b in edges:
        ax.plot([nodes[a][0] + 2 + len(a), nodes[b][0] - 2 - len(b)],
                [nodes[a][1], nodes[b][1]], color=MUTED, lw=1.2, zorder=1)

    box(ax, 5, 2, 90, 26, fc=PAPER, ec=RULE, lw=1.0)
    label(ax, 50, 24.5, "You are in  ~/projects/textstats", fs=10.5,
          weight="bold", mono=True, color=ACCENT)
    rows = [
        ("core.py", "relative", "→  ~/projects/textstats/core.py"),
        ("./core.py", "relative, explicit", "→  the same file"),
        ("../core.py", "one level up", "→  ~/projects/core.py"),
        ("~/core.py", "home", "→  /home/marie/core.py"),
        ("/core.py", "absolute", "→  /core.py, the filesystem root"),
    ]
    y = 19.0
    for path, kind, resolves in rows:
        label(ax, 10, y, path, fs=9.5, mono=True, ha="left", color=INK)
        label(ax, 27, y, kind, fs=8.5, color=MUTED, ha="left", style="italic")
        label(ax, 52, y, resolves, fs=9, mono=True, ha="left", color=ACCENT)
        y -= 3.3
    label(ax, 50, 61, "Where a path points depends on where you are standing",
          fs=12, weight="bold")
    save(fig, lesson, "paths.png")

    # --- PATH resolution ---------------------------------------------------- #
    fig, ax = canvas(10.5, 4.6)
    label(ax, 50, 39, "What happens when you type  pytest", fs=12.5, weight="bold")
    box(ax, 6, 27, 20, 8, fc=ACCENT_BG, ec=ACCENT, lw=1.5)
    label(ax, 16, 31, "pytest", fs=11, mono=True, weight="bold", color=ACCENT)
    dirs = [
        ("/usr/local/bin", False),
        ("~/.venv/bin", True),
        ("/usr/bin", False),
        ("/bin", False),
    ]
    y = 30
    x = 34
    for i, (d, hit) in enumerate(dirs):
        fc = GREEN_BG if hit else GREY_BG
        ec = GREEN if hit else MUTED
        box(ax, x, y - 3.2, 30, 6.4, fc=fc, ec=ec, lw=1.4)
        label(ax, x + 15, y, d, fs=9, mono=True)
        if hit:
            label(ax, x + 32, y, "found → runs this one", fs=9,
                  color=GREEN, ha="left", weight="bold")
        y -= 8.0
    arrow(ax, (26.5, 31), (33.5, 30), color=MUTED)
    label(ax, 4, -2.5, "$PATH is searched left to right; the first match wins.",
          fs=9.5, ha="left", color=INK)
    label(ax, 4, -7.0,
          "command not found  =  no directory on $PATH holds a file with that name.",
          fs=9.5, ha="left", color=WARM)
    save(fig, lesson, "path-resolution.png")

    # --- streams, pipes, exit codes ----------------------------------------- #
    fig, ax = canvas(10.5, 4.4)
    label(ax, 50, 38, "One process, three channels and a number", fs=12.5, weight="bold")
    titled_box(ax, 33, 16, 22, 12, "uv run pytest", None, fc=ACCENT_BG, ec=ACCENT, fs_t=11)
    arrow(ax, (20, 22), (32, 22), color=MUTED)
    label(ax, 12, 22, "stdin", fs=10, mono=True, color=MUTED)
    arrow(ax, (56, 25), (72, 28), color=GREEN)
    label(ax, 80, 28.6, "stdout   the answer", fs=9.5, mono=True, color=GREEN)
    arrow(ax, (56, 19), (72, 15), color=WARM)
    label(ax, 80, 14.5, "stderr   the complaint", fs=9.5, mono=True, color=WARM)
    label(ax, 44, 12, "exit code  0 = success", fs=9.5, mono=True, color=INK)
    box(ax, 6, 2, 88, 7, fc=GREY_BG, ec=RULE, lw=1.0)
    label(ax, 50, 5.5,
          "uv run pytest -q  |  tail -20      # pipe: stdout of the left becomes stdin of the right",
          fs=8.5, mono=True)
    save(fig, lesson, "streams-and-pipes.png")


# --------------------------------------------------------------------------- #
# git-essentials — the motivation half (was the `why-version-control` lesson,
# merged in 2026-09-06; its two figures now live in the git-essentials folder)
# --------------------------------------------------------------------------- #

def why_version_control() -> None:
    lesson = "git-essentials"

    fig, ax = canvas(10.5, 4.2)
    label(ax, 50, 36, "A repository is a graph of snapshots", fs=12.5, weight="bold")
    chain(ax, [14, 30, 46, 62, 78], 20, ["a3f", "9c1", "77e", "b02", "de4"])
    for x, msg in zip([14, 30, 46, 62, 78],
                      ["init", "add core.py", "add tests", "fix tie-break", "add CI"]):
        label(ax, x, 12.5, msg, fs=8, color=MUTED, mono=True)
    label(ax, 78, 27, "HEAD → main", fs=9, color=ACCENT, weight="bold")
    caption(ax, 50, 6,
            "Each commit stores the whole project plus a pointer to its parent. "
            "Follow the arrows backwards and you have the entire history.")
    save(fig, lesson, "commit-graph.png")

    # --- the three trees ---------------------------------------------------- #
    fig, ax = canvas(10.5, 4.8)
    label(ax, 50, 43, "The three places a change can be", fs=12.5, weight="bold")
    trees = [
        ("Working directory", "the files you edit", ACCENT_BG, ACCENT, 6),
        ("Staging area", "chosen for the next commit", GOLD_BG, GOLD, 37),
        ("Repository (.git)", "committed, permanent", GREEN_BG, GREEN, 68),
    ]
    for name, sub, fc, ec, x in trees:
        box(ax, x, 14, 26, 20, fc=fc, ec=ec, lw=1.5)
        label(ax, x + 13, 29, name, fs=10.5, weight="bold", color=ec)
        label(ax, x + 13, 23, sub, fs=8.5, color=MUTED)
    arrow(ax, (32.5, 27), (36.5, 27), color=INK)
    label(ax, 34.5, 30, "git add", fs=9, mono=True, color=INK)
    arrow(ax, (63.5, 27), (67.5, 27), color=INK)
    label(ax, 65.5, 30, "git commit", fs=9, mono=True, color=INK)
    arrow(ax, (67.5, 19), (63.5, 19), color=WARM, rad=0.0)
    label(ax, 65.5, 15.6, "git restore --staged", fs=8, mono=True, color=WARM)
    arrow(ax, (36.5, 19), (32.5, 19), color=WARM)
    label(ax, 34.5, 15.6, "git restore", fs=8, mono=True, color=WARM)
    caption(ax, 50, 8,
            "Every command in the rest of this lesson moves a change between two of these boxes.")
    save(fig, lesson, "three-trees.png")


# --------------------------------------------------------------------------- #
# git-essentials
# --------------------------------------------------------------------------- #

def git_essentials() -> None:
    lesson = "git-essentials"

    fig, ax = canvas(10.5, 4.0)
    label(ax, 50, 33, "The three diffs", fs=12.5, weight="bold")
    xs = [8, 38, 68]
    names = ["Working\ndirectory", "Staging\narea", "Last commit\n(HEAD)"]
    cols = [(ACCENT_BG, ACCENT), (GOLD_BG, GOLD), (GREEN_BG, GREEN)]
    for x, n, (fc, ec) in zip(xs, names, cols):
        box(ax, x, 14, 24, 12, fc=fc, ec=ec, lw=1.5)
        label(ax, x + 12, 20, n, fs=9.5, weight="bold", color=ec)
    arrow(ax, (32.5, 23), (37.5, 23), color=INK, style="<|-|>")
    label(ax, 35, 27, "git diff", fs=9, mono=True)
    arrow(ax, (62.5, 23), (67.5, 23), color=INK, style="<|-|>")
    label(ax, 65, 27, "git diff --staged", fs=9, mono=True)
    # Left-to-right, so the sign is mirrored from the two above: rad=-0.12 was
    # the one lifting this span into the boxes (14..26). Positive drops it clear.
    arrow(ax, (20, 12), (80, 12), color=MUTED, style="<|-|>", rad=0.10)
    label(ax, 50, 5.5, "git diff HEAD", fs=9, mono=True, color=MUTED)
    caption(ax, 50, 1.0, "\"My change is not in the diff\" nearly always means it is already staged.")
    save(fig, lesson, "three-diffs.png")

    # --- undo decision map -------------------------------------------------- #
    fig, ax = canvas(10.0, 5.6)
    label(ax, 50, 52, "Which undo?  It depends how far the change has travelled",
          fs=12, weight="bold")
    steps = [
        ("Edited a file,\nnothing staged", "git restore <file>", GREEN, GREEN_BG),
        ("Staged it\nby mistake", "git restore --staged <file>", GOLD, GOLD_BG),
        ("Committed, not\npushed yet", "git commit --amend\ngit reset --soft HEAD~1", ACCENT, ACCENT_BG),
        ("Already pushed", "git revert <sha>", WARM, WARM_BG),
    ]
    y = 42
    for question, cmd, ec, fc in steps:
        box(ax, 6, y - 8, 32, 9.5, fc=fc, ec=ec, lw=1.4)
        label(ax, 22, y - 3.2, question, fs=9.5, color=INK)
        arrow(ax, (39, y - 3.2), (46, y - 3.2), color=MUTED)
        box(ax, 47, y - 8, 47, 9.5, fc=PAPER, ec=ec, lw=1.4)
        label(ax, 70.5, y - 3.2, cmd, fs=9, mono=True, color=ec, weight="bold")
        y -= 11.5
    label(ax, 50, -4.0,
          "git revert is the only one safe on shared history — it adds a commit, it rewrites nothing.",
          fs=9, color=MUTED, style="italic")
    save(fig, lesson, "undo-map.png")


# --------------------------------------------------------------------------- #
# branching-and-collaboration
# --------------------------------------------------------------------------- #

def branching() -> None:
    lesson = "branching-and-collaboration"

    fig, ax = canvas(10.5, 4.4)
    label(ax, 50, 37, "A branch is a pointer, not a copy", fs=12.5, weight="bold")
    chain(ax, [14, 28, 42], 20, ["A", "B", "C"])
    for i, (x, s) in enumerate(zip([56, 70], ["D", "E"])):
        commit(ax, x, 27, s, fc=WARM_BG, ec=WARM)
    arrow(ax, (44.5, 21.5), (53.4, 26), color=WARM, lw=1.4, ms=8)
    arrow(ax, (58.9, 27), (67.1, 27), color=WARM, lw=1.4, ms=8)
    box(ax, 36, 10, 14, 5.5, fc=ACCENT_BG, ec=ACCENT, lw=1.3)
    label(ax, 43, 12.7, "main", fs=9, mono=True, weight="bold", color=ACCENT)
    arrow(ax, (43, 15.8), (42, 17.2), color=ACCENT, lw=1.3)
    box(ax, 64, 34, 24, 5.5, fc=WARM_BG, ec=WARM, lw=1.3)
    label(ax, 76, 36.7, "feature/scaling", fs=9, mono=True, weight="bold", color=WARM)
    arrow(ax, (73, 33.8), (71, 29.7), color=WARM, lw=1.3)
    caption(ax, 50, 4.5,
            "Creating a branch writes one file containing one SHA. That is the whole implementation.")
    save(fig, lesson, "branch-pointers.png")

    # --- merge vs rebase ---------------------------------------------------- #
    fig, ax = canvas(10.5, 5.4)
    label(ax, 50, 49, "Two ways to catch up with main", fs=12.5, weight="bold")

    label(ax, 8, 42, "git merge main", fs=10.5, mono=True, weight="bold",
          color=ACCENT, ha="left")
    chain(ax, [12, 24, 36], 33, ["A", "B", "C"])
    commit(ax, 48, 33, "M", fc=ACCENT_BG, ec=ACCENT)
    for x, s in zip([30, 42], ["D", "E"]):
        commit(ax, x, 24, s, fc=WARM_BG, ec=WARM)
    arrow(ax, (26.4, 31.5), (27.8, 26), color=WARM, lw=1.3, ms=8)
    arrow(ax, (32.9, 24), (39.1, 24), color=WARM, lw=1.3, ms=8)
    arrow(ax, (38.9, 33), (45.1, 33), color=ACCENT, lw=1.3, ms=8)
    arrow(ax, (44.4, 25.5), (46.2, 30.6), color=ACCENT, lw=1.3, ms=8)
    label(ax, 56, 33, "history keeps both lines\n+ one merge commit",
          fs=9, color=MUTED, ha="left")

    ax.plot([5, 95], [18, 18], color=RULE, lw=1.2)

    label(ax, 8, 13.5, "git rebase main", fs=10.5, mono=True, weight="bold",
          color=GREEN, ha="left")
    chain(ax, [12, 24, 36], 5, ["A", "B", "C"])
    for x, s in zip([48, 60], ["D'", "E'"]):
        commit(ax, x, 5, s, fc=GREEN_BG, ec=GREEN)
    arrow(ax, (38.9, 5), (45.1, 5), color=GREEN, lw=1.3, ms=8)
    arrow(ax, (50.9, 5), (57.1, 5), color=GREEN, lw=1.3, ms=8)
    label(ax, 68, 5, "one straight line,\nnew SHAs for D and E",
          fs=9, color=MUTED, ha="left")
    save(fig, lesson, "merge-vs-rebase.png")


# --------------------------------------------------------------------------- #
# branching-and-collaboration — the review half (was the
# `pull-requests-and-review` lesson, merged in 2026-09-06)
# --------------------------------------------------------------------------- #

def pull_requests() -> None:
    lesson = "branching-and-collaboration"

    fig, ax = canvas(11, 4.6)
    label(ax, 50, 38, "The pull-request loop", fs=12.5, weight="bold")
    steps = [
        ("branch", "git switch -c", ACCENT_BG, ACCENT),
        ("commit", "small, readable", ACCENT_BG, ACCENT),
        ("push", "git push -u", ACCENT_BG, ACCENT),
        ("open PR", "title + why", GOLD_BG, GOLD),
        ("review", "CI + a human", WARM_BG, WARM),
        ("merge", "delete branch", GREEN_BG, GREEN),
    ]
    w, gap = 13.5, 2.6
    x = 3.0
    for name, sub, fc, ec in steps:
        box(ax, x, 14, w, 13, fc=fc, ec=ec, lw=1.5)
        label(ax, x + w / 2, 23, name, fs=10.5, weight="bold", color=ec)
        label(ax, x + w / 2, 18, sub, fs=8, color=MUTED, mono=True)
        if x > 3.0:
            arrow(ax, (x - gap + 0.2, 20.5), (x - 0.5, 20.5), color=MUTED, lw=1.4)
        x += w + gap
    # The rework arrow, review -> commit, routed UNDER the row. arc3 puts its
    # control point at midpoint + rad*(dy, -dx), so on a right-to-left arrow a
    # POSITIVE rad lifts the curve into the boxes (rad=0.25 landed it at y=23.5,
    # drawing the arc straight through "open PR" and "push"). Negative bows it
    # down into the empty band between the boxes (bottom y=14) and the caption.
    centre = lambda i: 3.0 + i * (w + gap) + w / 2
    arrow(ax, (centre(4), 13.2), (centre(1), 13.2),
          color=WARM, lw=1.4, rad=-0.12)
    label(ax, 51, 6.5, "reviewer asks for a change → you push to the same branch,\nthe PR updates itself",
          fs=9, color=WARM)
    save(fig, lesson, "pr-lifecycle.png")

    # --- what a reviewer actually reads ------------------------------------- #
    fig, axc = plt.subplots(figsize=(9.0, 4.6))
    sizes = ["< 50", "50-100", "100-300", "300-800", "> 800"]
    read = [95, 88, 55, 20, 5]
    colors = [GREEN, GREEN, GOLD, WARM, WARM]
    axc.bar(sizes, read, color=colors, edgecolor=INK, linewidth=1.0, width=0.62)
    for i, v in enumerate(read):
        axc.text(i, v + 3, f"{v}%", ha="center", fontsize=10, color=INK,
                 fontweight="bold")
    axc.set_ylim(0, 112)
    axc.set_ylabel("share of the diff a reviewer really reads", fontsize=9.5)
    axc.set_xlabel("size of the pull request, in changed lines", fontsize=9.5)
    axc.spines[["top", "right"]].set_visible(False)
    axc.tick_params(labelsize=9)
    axc.set_title("Reviewer attention is the scarce resource",
                  fontsize=13, fontweight="bold", color=INK, pad=12)
    fig.text(0.5, -0.02,
             "Illustrative. The shape is not controversial: a 900-line pull request "
             "gets approved, not read.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.savefig(out(lesson, "review-size.png"), facecolor=PAPER)
    plt.close(fig)
    print(f"  {lesson}/review-size.png")

    # --- anatomy of a good PR ------------------------------------------------ #
    fig, ax = canvas(10, 5.6)
    box(ax, 6, 6, 88, 44, fc=PAPER, ec=INK, lw=1.4)
    box(ax, 6, 42, 88, 8, fc=GREY_BG, ec=INK, lw=1.4)
    label(ax, 10, 46, "Add readability module (Flesch reading-ease)", fs=11,
          weight="bold", ha="left")
    label(ax, 78, 46, "#7  ·  +84 − 3", fs=9, mono=True, color=MUTED, ha="left")

    label(ax, 10, 37.5, "Why", fs=9.5, weight="bold", ha="left", color=ACCENT)
    label(ax, 10, 34,
          "Lab 2 needs a readability score. Rules are pinned in the lab brief;\n"
          "this implements them exactly, no interpretation.",
          fs=8.8, ha="left", color=INK)
    label(ax, 10, 28, "How to check it", fs=9.5, weight="bold", ha="left", color=ACCENT)
    label(ax, 10, 24.5, "uv sync && uv run pytest -q     →  14 passed",
          fs=8.8, mono=True, ha="left")
    label(ax, 10, 19, "Not in this PR", fs=9.5, weight="bold", ha="left", color=ACCENT)
    label(ax, 10, 15.5, "The CLI entry point — separate PR, it touches packaging.",
          fs=8.8, ha="left")

    box(ax, 10, 8, 36, 5.4, fc=GREEN_BG, ec=GREEN, lw=1.2)
    label(ax, 28, 10.7, "✓  CI / tests   passed", fs=9, mono=True, color=GREEN,
          weight="bold")
    box(ax, 50, 8, 40, 5.4, fc=GOLD_BG, ec=GOLD, lw=1.2)
    label(ax, 70, 10.7, "1 review requested", fs=9, mono=True, color=GOLD, weight="bold")

    label(ax, 50, 53, "A pull request a stranger can review in ten minutes",
          fs=12.5, weight="bold")
    save(fig, lesson, "pr-anatomy.png")


# --------------------------------------------------------------------------- #
# python-environments
# --------------------------------------------------------------------------- #

def python_environments() -> None:
    lesson = "python-environments"

    fig, ax = canvas(10.5, 4.8)
    label(ax, 50, 42, "One interpreter per project, none of them shared",
          fs=12.5, weight="bold")
    box(ax, 33, 4, 34, 8, fc=GREY_BG, ec=MUTED, lw=1.4)
    label(ax, 50, 8, "system python  (do not install into it)", fs=9, mono=True,
          color=MUTED)
    projects = [
        ("project-a/.venv", "numpy 1.26\npandas 2.1", ACCENT_BG, ACCENT, 8),
        ("project-b/.venv", "numpy 2.1\ntorch 2.5", GREEN_BG, GREEN, 37),
        ("scratch/.venv", "numpy 2.3\nnothing else", GOLD_BG, GOLD, 66),
    ]
    for name, deps, fc, ec, x in projects:
        box(ax, x, 17, 26, 18, fc=fc, ec=ec, lw=1.5)
        label(ax, x + 13, 31, name, fs=9.5, mono=True, weight="bold", color=ec)
        label(ax, x + 13, 23.5, deps, fs=9, color=INK)
        arrow(ax, (x + 13, 16.5), (50, 12.5), color=MUTED, lw=1.1, ls=":", ms=7)
    caption(ax, 50, 0.8,
            "Three incompatible numpy versions, no conflict. The .venv is never committed.")
    save(fig, lesson, "env-isolation.png")

    # --- declaration vs lock ------------------------------------------------- #
    fig, ax = canvas(10, 4.4)
    label(ax, 50, 37, "Two files, two different jobs", fs=12.5, weight="bold")
    box(ax, 6, 8, 40, 23, fc=ACCENT_BG, ec=ACCENT, lw=1.5)
    label(ax, 26, 27, "pyproject.toml", fs=11, mono=True, weight="bold", color=ACCENT)
    label(ax, 26, 22.5, "what the project supports", fs=9, color=MUTED)
    label(ax, 26, 15,
          'dependencies = [\n  "numpy>=1.26",\n  "pandas>=2.2",\n]',
          fs=8.5, mono=True)
    box(ax, 54, 8, 40, 23, fc=GREEN_BG, ec=GREEN, lw=1.5)
    label(ax, 74, 27, "uv.lock", fs=11, mono=True, weight="bold", color=GREEN)
    label(ax, 74, 22.5, "what you actually ran", fs=9, color=MUTED)
    label(ax, 74, 15,
          "numpy      2.1.3\npandas     2.2.2\n+ 41 transitive pins",
          fs=8.5, mono=True)
    arrow(ax, (46.5, 19.5), (53.5, 19.5), color=INK)
    label(ax, 50, 23, "uv lock", fs=8.5, mono=True)
    caption(ax, 50, 3.5, "Commit both. The first is intent; the second is reproducibility.")
    save(fig, lesson, "declaration-vs-lock.png")


# --------------------------------------------------------------------------- #
# packaging-and-tests
# --------------------------------------------------------------------------- #

def packaging_and_tests() -> None:
    lesson = "packaging-and-tests"

    fig, ax = canvas(10.5, 5.0)
    label(ax, 50, 45, "Why the src/ layout", fs=12.5, weight="bold")

    box(ax, 5, 8, 42, 32, fc=WARM_BG, ec=WARM, lw=1.5)
    label(ax, 26, 36, "flat layout", fs=11, weight="bold", color=WARM)
    label(ax, 10, 30,
          "textstats/\n  __init__.py\n  core.py\ntests/\npyproject.toml",
          fs=9, mono=True, ha="left", va="top")
    label(ax, 26, 13.5,
          "import textstats  finds the folder\nwhether or not the install works",
          fs=8.8, color=INK)
    label(ax, 26, 9.8, "→ breaks on someone else's machine", fs=8.8,
          color=WARM, weight="bold")

    box(ax, 53, 8, 42, 32, fc=GREEN_BG, ec=GREEN, lw=1.5)
    label(ax, 74, 36, "src/ layout", fs=11, weight="bold", color=GREEN)
    label(ax, 58, 30,
          "src/\n  textstats/\n    __init__.py\n    core.py\ntests/\npyproject.toml",
          fs=9, mono=True, ha="left", va="top")
    label(ax, 74, 13.5,
          "the only way to import it\nis to install it",
          fs=8.8, color=INK)
    label(ax, 74, 9.8, "→ your tests exercise what users get", fs=8.8,
          color=GREEN, weight="bold")
    save(fig, lesson, "src-layout.png")

    # --- test pyramid -------------------------------------------------------- #
    fig, ax = canvas(10, 5.0)
    label(ax, 50, 48, "How many of each", fs=12.5, weight="bold")
    tiers = [
        (8, 62, "unit — one function", "milliseconds  ·  many", GREEN_BG, GREEN),
        (20, 44, "integration — the wiring", "seconds  ·  a handful", GOLD_BG, GOLD),
        (32, 26, "end to end — the whole pipeline", "minutes  ·  one or two", WARM_BG, WARM),
    ]
    for y, w, name, cost, fc, ec in tiers:
        box(ax, 50 - w / 2, y, w, 9.5, fc=fc, ec=ec, lw=1.5)
        label(ax, 50, y + 6.2, name, fs=9.5, weight="bold", color=ec)
        label(ax, 50, y + 2.8, cost, fs=8.2, color=MUTED)
    label(ax, 4, 12.7, "fast,\nprecise", fs=9, color=GREEN, ha="left", weight="bold")
    label(ax, 4, 36.7, "slow,\nrealistic", fs=9, color=WARM, ha="left", weight="bold")
    caption(ax, 50, 1.5,
            "A test that loads a small CSV, trains, and asserts the score beats a constant "
            "baseline is an excellent integration test.")
    save(fig, lesson, "test-pyramid.png")


# --------------------------------------------------------------------------- #
# ide-syntax-linting  (taught: code quality)
# --------------------------------------------------------------------------- #

def code_quality() -> None:
    lesson = "ide-syntax-linting"

    fig, ax = canvas(11, 4.8)
    label(ax, 50, 41, "Five gates, and what each one costs to pass through",
          fs=12.5, weight="bold")
    gates = [
        ("editor", "as you type", "seconds", GREEN_BG, GREEN),
        ("pre-commit", "git commit", "seconds", GOLD_BG, GOLD),
        ("CI", "git push", "minutes", ACCENT_BG, ACCENT),
        ("reviewer", "pull request", "hours", WARM_BG, WARM),
        ("production", "next session", "a day", WARM_BG, WARM),
    ]
    w, gap = 16.0, 3.2
    x = 3.0
    for name, when, cost, fc, ec in gates:
        box(ax, x, 14, w, 16, fc=fc, ec=ec, lw=1.5)
        label(ax, x + w / 2, 26, name, fs=11, weight="bold", color=ec)
        label(ax, x + w / 2, 22, when, fs=8.5, mono=True, color=MUTED)
        label(ax, x + w / 2, 17.5, cost, fs=9, color=INK, weight="bold")
        if x > 3.0:
            arrow(ax, (x - gap + 0.2, 22), (x - 0.5, 22), color=MUTED, lw=1.4)
        x += w + gap
    label(ax, 3, 8, "cost of finding the same bug →", fs=9.5, ha="left",
          color=MUTED, style="italic")
    ax.plot([3, 96], [5, 5], color=RULE, lw=1.2)
    label(ax, 50, 1.5,
          "A linter is not about taste. F401 (unused import) and B006 (mutable default) are bugs.",
          fs=9, color=INK)
    save(fig, lesson, "quality-gates.png")


# --------------------------------------------------------------------------- #
# github-actions  (taught: CI/CD)
# --------------------------------------------------------------------------- #

def continuous_integration() -> None:
    lesson = "github-actions"

    fig, ax = canvas(11, 5.2)
    label(ax, 50, 54, "What happens after git push", fs=12.5, weight="bold")

    titled_box(ax, 3, 26, 17, 12, "git push", "or a PR", fc=ACCENT_BG, ec=ACCENT,
               tc=ACCENT, fs_t=10.5, fs_b=8.5)
    arrow(ax, (20.5, 32), (25.5, 32), color=MUTED)
    titled_box(ax, 26, 26, 19, 12, "GitHub", "reads the\nworkflow file",
               fc=GREY_BG, ec=MUTED, tc=INK, fs_t=10.5, fs_b=8)
    arrow(ax, (45.5, 32), (50.5, 32), color=MUTED)

    jobs = [
        ("ubuntu · py3.11", GREEN, GREEN_BG, 38),
        ("ubuntu · py3.12", GREEN, GREEN_BG, 26),
        ("lint · ruff", GOLD, GOLD_BG, 14),
    ]
    for name, ec, fc, y in jobs:
        box(ax, 51, y, 22, 10, fc=fc, ec=ec, lw=1.4)
        label(ax, 62, y + 5, name, fs=9, mono=True, color=ec, weight="bold")
        arrow(ax, (73.5, y + 5), (78.5, 30), color=MUTED, lw=1.1)
    label(ax, 62, 10.0, "a fresh machine per job", fs=8.5, color=MUTED, style="italic")

    box(ax, 79, 24, 18, 12, fc=GREEN_BG, ec=GREEN, lw=1.6)
    label(ax, 88, 32, "✓  green", fs=11, weight="bold", color=GREEN)
    label(ax, 88, 27.5, "merge allowed", fs=8.5, color=MUTED)

    box(ax, 3, -1, 94, 8, fc=PAPER, ec=RULE, lw=1.0)
    label(ax, 50, 3,
          "Branch protection turns the green tick from information into a rule: red blocks the merge.",
          fs=9.5, color=INK)
    save(fig, lesson, "ci-pipeline.png")

    # --- CI vs CD ------------------------------------------------------------ #
    fig, ax = canvas(10.5, 4.0)
    label(ax, 50, 33, "CI and CD are two halves of one pipeline", fs=12.5, weight="bold")
    box(ax, 5, 8, 43, 18, fc=GREEN_BG, ec=GREEN, lw=1.5)
    label(ax, 26.5, 21, "Continuous Integration", fs=11, weight="bold", color=GREEN)
    label(ax, 26.5, 14,
          "every push is built, linted, tested\non a machine that is not yours",
          fs=9, color=INK)
    box(ax, 52, 8, 43, 18, fc=ACCENT_BG, ec=ACCENT, lw=1.5)
    label(ax, 73.5, 21, "Continuous Delivery", fs=11, weight="bold", color=ACCENT)
    label(ax, 73.5, 14,
          "a green main is automatically published\n— a release, an image, a deployed model",
          fs=9, color=INK)
    arrow(ax, (48.5, 17), (51.5, 17), color=MUTED)
    caption(ax, 50, 3.0,
            "This course does CI properly and CD once, as a GitHub release you can pip install.")
    save(fig, lesson, "ci-vs-cd.png")


# --------------------------------------------------------------------------- #
# notebooks-and-colab
# --------------------------------------------------------------------------- #

def notebooks_and_colab() -> None:
    lesson = "notebooks-and-colab"

    # --- the kernel model ---------------------------------------------------- #
    fig, ax = canvas(10.5, 5.0)
    label(ax, 50, 45, "A notebook is a REPL with a scrollback", fs=12.5, weight="bold")
    box(ax, 5, 4, 42, 35, fc=PAPER, ec=INK, lw=1.4)
    label(ax, 26, 35, "what you see", fs=10, weight="bold", color=MUTED)
    cells = [("[1]", "import pandas as pd"), ("[3]", "df = load()"),
             ("[2]", "df.head()"), ("[ ]", "train(df)")]
    y = 29
    for n, src in cells:
        box(ax, 9, y - 3.2, 34, 5.2, fc=GREY_BG, ec=RULE, lw=1.0)
        label(ax, 12, y - 0.6, n, fs=8.5, mono=True, color=ACCENT, ha="left")
        label(ax, 18, y - 0.6, src, fs=8.5, mono=True, ha="left")
        y -= 6.4

    box(ax, 53, 4, 42, 35, fc=ACCENT_BG, ec=ACCENT, lw=1.4)
    label(ax, 74, 35, "what actually holds state", fs=10, weight="bold", color=ACCENT)
    box(ax, 58, 14, 32, 17, fc=PAPER, ec=ACCENT, lw=1.3)
    label(ax, 74, 27, "one kernel process", fs=10, weight="bold", color=ACCENT)
    label(ax, 74, 20.5, "pd  → module\ndf  → DataFrame(1212, 6)", fs=9, mono=True)
    label(ax, 74, 10.5, "cells are not a program;\nthey are edits to this memory",
          fs=8.8, color=INK)
    arrow(ax, (47.5, 22), (52.5, 22), color=MUTED)
    save(fig, lesson, "kernel-model.png")

    # --- out-of-order trap --------------------------------------------------- #
    fig, ax = canvas(10.5, 4.6)
    label(ax, 50, 40, "The bug that only exists in your kernel", fs=12.5, weight="bold")
    seq = [
        ("[1]", "df = load_csv('sales.csv')", GREEN),
        ("[2]", "df = df.dropna()", GREEN),
        ("[3]", "df['price'] *= 1.2", GOLD),
        ("[4]", "df['price'] *= 1.2", WARM),
    ]
    y = 32
    for n, src, col in seq:
        box(ax, 8, y - 3.4, 52, 5.6, fc=PAPER, ec=col, lw=1.3)
        label(ax, 11, y - 0.6, n, fs=9, mono=True, color=col, ha="left", weight="bold")
        label(ax, 17, y - 0.6, src, fs=9, mono=True, ha="left")
        y -= 7.0
    label(ax, 63, 21, "you re-ran cell 3\nbecause of a typo\n→ prices are now\n   1.44×, not 1.2×",
          fs=9.5, ha="left", color=WARM)
    box(ax, 8, 2.5, 84, 6.5, fc=GREEN_BG, ec=GREEN, lw=1.3)
    label(ax, 50, 5.7,
          "Restart & Run All  is the only honest check that your notebook still means what it shows.",
          fs=9.5, color=GREEN, weight="bold")
    save(fig, lesson, "out-of-order.png")

    # --- colab anatomy ------------------------------------------------------- #
    fig, ax = canvas(10.5, 5.2)
    label(ax, 50, 47, "What Colab actually gives you", fs=12.5, weight="bold")
    box(ax, 24, 8, 52, 33, fc=ACCENT_BG, ec=ACCENT, lw=1.6)
    label(ax, 50, 37, "a virtual machine at Google", fs=11, weight="bold", color=ACCENT)
    inner = [
        ("Python + CUDA preinstalled", GREEN, 30),
        ("a GPU, if you ask for one", GREEN, 24),
        ("a disk that is erased with the VM", WARM, 18),
        ("~90 min idle, then it is gone", WARM, 12),
    ]
    for txt, col, y in inner:
        box(ax, 28, y - 2.2, 44, 5.0, fc=PAPER, ec=col, lw=1.2)
        label(ax, 50, y + 0.3, txt, fs=9, color=col if col == WARM else INK)
    box(ax, 2, 26, 19, 12, fc=GREY_BG, ec=MUTED, lw=1.3)
    label(ax, 11.5, 34, "GitHub", fs=10, weight="bold")
    label(ax, 11.5, 29.5, "pip install\ngit+https://...", fs=8, mono=True, color=MUTED)
    arrow(ax, (21.5, 32), (27.5, 30), color=GREEN)
    box(ax, 2, 10, 19, 12, fc=GREY_BG, ec=MUTED, lw=1.3)
    label(ax, 11.5, 18, "Google Drive", fs=10, weight="bold")
    label(ax, 11.5, 13.5, "drive.mount()", fs=8, mono=True, color=MUTED)
    arrow(ax, (21.5, 16), (27.5, 18), color=GREEN)
    box(ax, 79, 18, 19, 12, fc=GOLD_BG, ec=GOLD, lw=1.3)
    label(ax, 88.5, 26, "your results", fs=10, weight="bold", color=GOLD)
    label(ax, 88.5, 21.5, "download or\nsave to Drive", fs=8, color=MUTED)
    arrow(ax, (76.5, 24), (78.5, 24), color=GOLD)
    caption(ax, 50, 3.0,
            "Everything inside the blue box is temporary. Anything you want to keep has to leave it.")
    save(fig, lesson, "colab-anatomy.png")

    # --- the through-line ---------------------------------------------------- #
    fig, ax = canvas(11, 4.4)
    label(ax, 50, 36, "Explore in the notebook, ship the tested module",
          fs=12.5, weight="bold")
    titled_box(ax, 4, 12, 24, 16, "notebook", "try things,\nplot, be wrong",
               fc=GOLD_BG, ec=GOLD, tc=GOLD, fs_t=11, fs_b=8.5)
    titled_box(ax, 38, 12, 24, 16, "src/ + tests/", "the version that\nhas to be right",
               fc=GREEN_BG, ec=GREEN, tc=GREEN, fs_t=11, fs_b=8.5)
    titled_box(ax, 72, 12, 24, 16, "notebook again", "import it,\nuse it, plot it",
               fc=GOLD_BG, ec=GOLD, tc=GOLD, fs_t=11, fs_b=8.5)
    arrow(ax, (28.5, 20), (37.5, 20), color=INK)
    label(ax, 33, 24, "extract", fs=8.5, mono=True)
    arrow(ax, (62.5, 20), (71.5, 20), color=INK)
    label(ax, 67, 24, "pip install", fs=8.5, mono=True)
    # Same arc3 trap as pr-lifecycle: right-to-left with a positive rad puts the
    # control point at y=21.9, inside the boxes (12..28), so the dotted loop drew
    # a strike-through across "the version that has to be right". Negative bows
    # it into the gap under the row instead.
    arrow(ax, (84, 11), (16, 11), color=MUTED, rad=-0.10, lw=1.2, ls=":")
    label(ax, 50, 4.5, "the loop, not a one-way trip", fs=9, color=MUTED, style="italic")
    save(fig, lesson, "notebook-to-package.png")


# --------------------------------------------------------------------------- #
# assistant-landscape
# --------------------------------------------------------------------------- #

def assistant_landscape() -> None:
    lesson = "assistant-landscape"

    fig, ax = canvas(11, 4.8)
    label(ax, 50, 41, "Three generations, and what each one can see",
          fs=12.5, weight="bold")
    gens = [
        ("autocomplete", "the current file", "suggests the\nnext few lines",
         "you accept or reject", GREY_BG, MUTED),
        ("chat", "what you paste", "answers in a\nside panel",
         "you copy it back", GOLD_BG, GOLD),
        ("agent", "your whole repo,\nyour shell", "reads, edits, runs,\nreads the failure, retries",
         "you specify and verify", GREEN_BG, GREEN),
    ]
    w, gap = 28.0, 4.0
    x = 4.0
    for name, sees, does, you, fc, ec in gens:
        box(ax, x, 8, w, 27, fc=fc, ec=ec, lw=1.5)
        label(ax, x + w / 2, 31, name, fs=12, weight="bold", color=ec)
        label(ax, x + w / 2, 26, "sees", fs=8, color=MUTED, weight="bold")
        label(ax, x + w / 2, 23, sees, fs=8.8)
        label(ax, x + w / 2, 19, "does", fs=8, color=MUTED, weight="bold")
        label(ax, x + w / 2, 15.5, does, fs=8.8)
        label(ax, x + w / 2, 11, you, fs=8.8, color=ec, weight="bold")
        if x > 4.0:
            arrow(ax, (x - gap + 0.3, 21), (x - 0.6, 21), color=MUTED, lw=1.5)
        x += w + gap
    caption(ax, 50, 3.5,
            "Only the third has a feedback loop: it finds out whether its answer worked.")
    save(fig, lesson, "generations.png")

    # --- the map -------------------------------------------------------------- #
    fig, ax = canvas(10.5, 5.6)
    ax.plot([14, 95], [7, 7], color=INK, lw=1.3)
    ax.plot([14, 14], [7, 48], color=INK, lw=1.3)
    label(ax, 55, 2.5, "proprietary product   →   open source", fs=9.5, color=MUTED)
    label(ax, 6, 27, "terminal / CI\n↑\n↓\neditor UI", fs=9.5, color=MUTED)

    tools = [
        ("Claude Code", 26, 43, GREEN),
        ("Codex CLI", 44, 38, GREEN),
        ("Gemini CLI", 30, 31, GREEN),
        ("Aider", 70, 43, ACCENT),
        ("aider + Ollama", 82, 34, ACCENT),
        ("GitHub Copilot", 26, 20, WARM),
        ("Windsurf", 48, 16, WARM),
        ("Cursor", 26, 12, WARM),
        ("Continue.dev", 76, 14, ACCENT),
    ]
    for name, x, y, col in tools:
        w = 5.0 + 1.0 * len(name)
        box(ax, x - w / 2, y - 2.6, w, 5.2, fc=PAPER, ec=col, lw=1.4)
        label(ax, x, y, name, fs=9, color=col, weight="bold")
    label(ax, 55, 57, "Where the tools sit", fs=12.5, weight="bold")
    caption(ax, 55, 52,
            "They differ in interface and in licence, not in the loop. Learn the loop once.")
    save(fig, lesson, "assistant-map.png")


# --------------------------------------------------------------------------- #
# assistant-landscape — the loop figure (was the `the-core-loop` lesson; the
# whole agentic block was merged into one lesson on 2026-09-06)
# --------------------------------------------------------------------------- #

def core_loop() -> None:
    lesson = "assistant-landscape"

    fig, ax = canvas(10.5, 5.4)
    label(ax, 48, 55, "The loop — and the two steps students skip",
          fs=12.5, weight="bold")
    steps = [
        ("EXPLORE", "read files, grep,\nunderstand", ACCENT_BG, ACCENT, 40),
        ("PLAN", "state the approach\nbefore any edit", GOLD_BG, GOLD, 29),
        ("ACT", "edit files,\nrun commands", ACCENT_BG, ACCENT, 18),
        ("VERIFY", "tests, linter,\nactually run it", GOLD_BG, GOLD, 7),
    ]
    for name, sub, fc, ec, y in steps:
        box(ax, 22, y, 34, 9.0, fc=fc, ec=ec, lw=1.6)
        label(ax, 39, y + 6.2, name, fs=11, weight="bold", color=ec)
        label(ax, 39, y + 2.8, sub, fs=8.5, color=MUTED)
    for y in (40, 29, 18):
        arrow(ax, (39, y - 0.6), (39, y - 1.4), color=INK, lw=1.6, ms=11)
    arrow(ax, (56.5, 11.5), (56.5, 22.5), color=WARM, lw=1.5, rad=-0.85)
    label(ax, 78, 17, "still failing →\nact again", fs=9, color=WARM)

    for y, txt in ((33.5, "your leverage"), (11.5, "your leverage")):
        label(ax, 20, y, txt, fs=9.5, color=GOLD, ha="right", weight="bold")
    caption(ax, 50, 2.0,
            "No tests means no VERIFY, and without VERIFY the loop stops when the code merely looks finished.")
    save(fig, lesson, "agent-loop.png")


# --------------------------------------------------------------------------- #

def main() -> None:
    print("writing Session 1 figures ->", ASSETS)
    session_map()
    accounts_and_tools()
    the_shell()
    why_version_control()
    git_essentials()
    branching()
    pull_requests()
    python_environments()
    packaging_and_tests()
    code_quality()
    continuous_integration()
    notebooks_and_colab()
    assistant_landscape()
    core_loop()
    print("done")


if __name__ == "__main__":
    main()
