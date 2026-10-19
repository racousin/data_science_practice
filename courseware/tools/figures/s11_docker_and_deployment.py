"""Figures for `ms2a-machine-learning-practice` Session 11 (Session 10, Model Deployment Workshop).

Every image the Session 11 lessons show lives under
`content/ms2a-machine-learning-practice/assets/deploy/`. This script draws the
diagrams; the screenshots next to them are stills, listed in SCREENSHOTS below
with how each was taken. Re-run from `courseware/` with:

    uv run --with matplotlib python tools/figures/s11_docker_and_deployment.py [figure ...]

With no argument it makes every figure; name functions to make only those.

Sizing follows the Session 4 script: 10 in wide at 140 dpi (1400 px), 13.5-20 pt
text, so the smallest label still lands near 10 pt on a slide.

The numbers drawn are measurements, not estimates. They were taken on
2026-10-19 on an Apple M4 laptop (OrbStack, docker 29.8.2, compose v5.5.0),
building the workshop's penguin API (labs/s11-docker-and-deployment/penguin-api,
re-measured 2026-10-19) with python:3.13-slim already pulled:
- first `docker build`: ~19 s; rebuild after editing main.py: < 1 s;
- the `RUN uv sync --locked --no-cache` layer: 335 MB (docker history); the
  image: 687 MB on disk (1.15 GB without `--no-cache`), 156 MB compressed on
  Docker Hub (linux/amd64);
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COURSEWARE = pathlib.Path(__file__).resolve().parents[2]
ASSETS = COURSEWARE / "content" / "ms2a-machine-learning-practice" / "assets" / "deploy"

# --------------------------------------------------------------------------- #
# SCREENSHOTS — stills, not drawn here.
# --------------------------------------------------------------------------- #
SCREENSHOTS = """
docs-penguin.png, html-form.png
    Chromium via Playwright, against the workshop's penguin API (fastapi 0.149.0)
    run with `uv run fastapi dev` on a spare port, 2026-10-19.
render-*.png
    The Render dashboard, 2026-10-19: render-project, render-service-types and
    render-existing-image taken by the course owner; render-config,
    render-logs-port-detect and render-service taken during the dry run that
    deployed racousin90/penguin-api:v1 on a Free instance (cropped).
dockerhub-repositories.png, dockerhub-tags.png
    hub.docker.com, namespace racousin90, 2026-10-19.
"""

# Same palette as the other courseware figure scripts.
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
GREY_BG = "#f1f3f5"
RULE = "#dfe3e8"
PAPER = "#ffffff"
MONO = "DejaVu Sans Mono"

W = 10.0
FS = 15

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "text.color": INK,
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, facecolor=PAPER)
    w, h = (fig.get_size_inches() * fig.dpi).round().astype(int)
    plt.close(fig)
    print(f"  wrote deploy/{name}  ({w}x{h})")


def canvas(h: float, w: float = W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, *, fc=PAPER, ec=INK, lw=1.8, r=0.08, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def lines(ax, cx, cy, items, *, spacing=1.3, ha="center"):
    """Stack text lines centred on (cx, cy); each item is (text, text-kwargs)."""
    heights = [kw.get("fontsize", FS) * spacing / 72 for _, kw in items]
    y = cy + sum(heights) / 2
    for (text, kw), h in zip(items, heights):
        y -= h
        ax.text(cx, y + h / 2, text, ha=ha, va="center", zorder=5,
                **{"color": INK, "fontsize": FS, **kw})


def arrow(ax, p0, p1, *, color=INK, lw=2.0, rad=0.0, ms=18, z=3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 0.1, y, y, y - 0.1], color=color, lw=2.0)
    ax.text((x0 + x1) / 2, y + 0.2, text, ha="center", va="center",
            color=color, fontsize=15.5, weight="bold")


# --------------------------------------------------------------------------- #
# docker-concepts — a VM carries an operating system, a container does not
# --------------------------------------------------------------------------- #
def vm_vs_container() -> None:
    fig, ax = canvas(4.6)

    def stack(x0, width, rows, title, caption):
        y = 0.55
        for label, fc, ec, split in rows:
            h = 0.5
            if split:
                half = (width - 0.12) / 2
                for k in range(2):
                    cx = x0 + half / 2 + k * (half + 0.12)
                    box(ax, cx, y + h / 2, half, h - 0.08, fc=fc, ec=ec, lw=1.6)
                    ax.text(cx, y + h / 2, label[k], ha="center", va="center", fontsize=14)
            else:
                box(ax, x0 + width / 2, y + h / 2, width, h - 0.08, fc=fc, ec=ec, lw=1.6)
                ax.text(x0 + width / 2, y + h / 2, label, ha="center", va="center", fontsize=14)
            y += h
        ax.text(x0 + width / 2, y + 0.3, title, ha="center", va="center",
                fontsize=18, weight="bold")
        ax.text(x0 + width / 2, 0.22, caption, ha="center", va="center",
                fontsize=14, color=MUTED)

    stack(0.4, 4.2, [
        ("hardware", GREY_BG, INK, False),
        ("host operating system", GREY_BG, INK, False),
        ("hypervisor", GREY_BG, INK, False),
        (("guest OS", "guest OS"), WARM_BG, WARM, True),
        (("Python + libs", "Python + libs"), GOLD_BG, GOLD, True),
        (("app A", "app B"), ACCENT_BG, ACCENT, True),
    ], "Virtual machines", "a whole OS each: GBs, boots in minutes")

    stack(5.4, 4.2, [
        ("hardware", GREY_BG, INK, False),
        ("host operating system", GREY_BG, INK, False),
        ("Docker Engine", GREEN_BG, GREEN, False),
        (("Python + libs", "Python + libs"), GOLD_BG, GOLD, True),
        (("app A", "app B"), ACCENT_BG, ACCENT, True),
    ], "Containers", "share the host's kernel: MBs, starts in seconds")
    save(fig, "vm-vs-container.png")


# --------------------------------------------------------------------------- #
# docker-concepts — Dockerfile, image, registry, container
# --------------------------------------------------------------------------- #
def build_push_pull_run() -> None:
    fig, ax = canvas(3.3)
    y = 1.75

    def layers(cx, cy, label):
        for k in range(3):
            box(ax, cx, cy - 0.3 + k * 0.3, 1.5, 0.26, fc=GREEN_BG, ec=GREEN, lw=1.5, r=0.04)
        ax.text(cx, cy - 0.78, label, ha="center", va="center", fontsize=14, color=MUTED)

    box(ax, 0.95, y, 1.5, 1.2, fc=GREY_BG, ec=INK, lw=2.0)
    lines(ax, 0.95, y, [("Dockerfile", {"fontsize": 15.5, "weight": "bold"}),
                        ("main.py", {"fontsize": 13, "color": MUTED, "family": MONO}),
                        ("model.joblib", {"fontsize": 13, "color": MUTED, "family": MONO})])
    ax.text(0.95, y + 0.9, "your laptop", ha="center", fontsize=14, weight="bold")

    layers(3.05, y, "image")
    ax.text(3.05, y + 0.9, "your laptop", ha="center", fontsize=14, weight="bold")

    box(ax, 5.2, y, 1.6, 1.2, fc=GOLD_BG, ec=GOLD, lw=2.2)
    lines(ax, 5.2, y, [("registry", {"fontsize": 16, "weight": "bold"}),
                       ("Docker Hub", {"fontsize": 13.5, "color": MUTED})])
    ax.text(5.2, y + 0.9, "the internet", ha="center", fontsize=14, weight="bold")

    layers(7.3, y, "same image")
    ax.text(8.35, y + 0.9, "any other machine", ha="center", fontsize=14, weight="bold")

    for k, dy in enumerate((0.33, -0.33)):
        box(ax, 9.3, y + dy, 1.05, 0.5, fc=ACCENT_BG, ec=ACCENT, lw=1.8)
        ax.text(9.3, y + dy, f"container {k + 1}", ha="center", va="center", fontsize=12.5)

    for x0, x1, verb in ((1.75, 2.25, "build"), (3.85, 4.35, "push"),
                         (6.05, 6.5, "pull"), (8.1, 8.72, "run")):
        arrow(ax, (x0, y), (x1, y), lw=2.2)
        ax.text((x0 + x1) / 2, y - 1.12, f"docker {verb}", ha="center", va="center",
                fontsize=13.5, family=MONO, color=WARM)
    ax.text(W / 2, 0.06, "build once, run the identical image anywhere, as many times as you like",
            ha="center", va="bottom", fontsize=14, color=MUTED)
    save(fig, "build-push-pull-run.png")


# --------------------------------------------------------------------------- #
# docker-install-and-run — publishing a port
# --------------------------------------------------------------------------- #
def port_mapping() -> None:
    fig, ax = canvas(3.7)
    box(ax, 5.0, 1.85, 9.5, 3.2, fc=GREY_BG, ec=INK, lw=2.0, r=0.12)
    ax.text(0.45, 3.2, "your laptop (the host)", fontsize=15.5, weight="bold", va="center")

    rows = [(2.35, "localhost:8000", "8000", "8000", "penguin-api", "uvicorn on 0.0.0.0:8000"),
            (1.0, "localhost:8080", "8080", "80", "nginx", "web server on port 80")]
    for y, url, host, cont, name, proc in rows:
        box(ax, 1.55, y, 2.0, 0.8, fc=PAPER, ec=INK, lw=1.8)
        lines(ax, 1.55, y, [("browser", {"fontsize": 14, "weight": "bold"}),
                            (url, {"fontsize": 13, "family": MONO, "color": ACCENT})])
        ax.add_patch(plt.Circle((3.55, y), 0.3, fc=GOLD_BG, ec=GOLD, lw=2, zorder=3))
        ax.text(3.55, y, host, ha="center", va="center", fontsize=12.5, weight="bold", zorder=4)
        arrow(ax, (2.58, y), (3.23, y), lw=2)
        box(ax, 7.75, y, 3.6, 0.95, fc=ACCENT_BG, ec=ACCENT, lw=2.0)
        ax.add_patch(plt.Circle((5.95, y), 0.3, fc=GOLD_BG, ec=GOLD, lw=2, zorder=3))
        ax.text(5.95, y, cont, ha="center", va="center", fontsize=12.5, weight="bold", zorder=4)
        arrow(ax, (3.87, y), (5.63, y), color=WARM, lw=2.4, ms=20)
        ax.text(4.75, y + 0.3, f"-p {host}:{cont}", ha="center", va="center",
                fontsize=13.5, family=MONO, color=WARM, weight="bold")
        lines(ax, 8.0, y, [(f"container: {name}", {"fontsize": 14.5, "weight": "bold"}),
                           (proc, {"fontsize": 13, "color": MUTED})])
    ax.text(3.55, 0.35, "host port", ha="center", fontsize=13, color=GOLD, weight="bold")
    ax.text(5.95, 0.35, "container port", ha="center", fontsize=13, color=GOLD, weight="bold")
    save(fig, "port-mapping.png")


# --------------------------------------------------------------------------- #
# apis-and-http — a request and its response
# --------------------------------------------------------------------------- #
def client_server() -> None:
    fig, ax = canvas(3.9)
    box(ax, 1.25, 2.0, 2.0, 2.6, fc=GREY_BG, ec=INK, lw=2.0)
    lines(ax, 1.25, 2.0, [("client", {"fontsize": 18, "weight": "bold"}),
                          ("browser", {"fontsize": 13.5, "color": MUTED}),
                          ("curl", {"fontsize": 13.5, "color": MUTED}),
                          ("Python requests", {"fontsize": 13.5, "color": MUTED}),
                          ("a mobile app", {"fontsize": 13.5, "color": MUTED})])
    box(ax, 8.75, 2.0, 2.0, 2.6, fc=ACCENT_BG, ec=ACCENT, lw=2.0)
    lines(ax, 8.75, 2.0, [("server", {"fontsize": 18, "weight": "bold"}),
                          ("uvicorn", {"fontsize": 13.5, "color": MUTED}),
                          ("runs main.py", {"fontsize": 13.5, "color": MUTED}),
                          ("waits for", {"fontsize": 13.5, "color": MUTED}),
                          ("requests", {"fontsize": 13.5, "color": MUTED})])

    arrow(ax, (2.35, 2.95), (7.65, 2.95), color=WARM, lw=2.6, ms=22)
    box(ax, 5.0, 3.42, 4.9, 0.62, fc=WARM_BG, ec=WARM, lw=1.6)
    lines(ax, 5.0, 3.42, [("request   POST /predict", {"fontsize": 14, "family": MONO, "weight": "bold"}),
                          ('{"bill_length_mm": 39.1, ...}', {"fontsize": 12.5, "family": MONO})],
          spacing=1.2)

    arrow(ax, (7.65, 1.05), (2.35, 1.05), color=GREEN, lw=2.6, ms=22)
    box(ax, 5.0, 0.58, 4.9, 0.62, fc=GREEN_BG, ec=GREEN, lw=1.6)
    lines(ax, 5.0, 0.58, [("response   200 OK", {"fontsize": 14, "family": MONO, "weight": "bold"}),
                          ('{"species": "Adelie"}', {"fontsize": 12.5, "family": MONO})],
          spacing=1.2)
    lines(ax, 5.0, 2.0, [("the model runs here, on the server:", {"fontsize": 14, "color": MUTED}),
                         ("the client never sees Python, sklearn or the model file",
                          {"fontsize": 14, "color": MUTED})])
    save(fig, "client-server.png")


# --------------------------------------------------------------------------- #
# writing-a-dockerfile — one instruction, one layer, and the cache
# --------------------------------------------------------------------------- #
def dockerfile_layers() -> None:
    fig, ax = canvas(4.7)
    rows = [
        ("FROM python:3.13-slim", "the base: Debian + Python", True),
        ("RUN pip install --no-cache-dir uv", "", True),
        ("WORKDIR /app", "", True),
        ("COPY pyproject.toml uv.lock ./", "", True),
        ("RUN uv sync --locked --no-cache", "335 MB, the slow step", True),
        ("COPY main.py model.joblib ./", "← you edited main.py", False),
        ("CMD [\"uv\", \"run\", \"fastapi\", ...]", "", False),
    ]
    h, x0, wbox = 0.52, 0.35, 5.6
    for k, (instr, note, cached) in enumerate(rows):
        y = 0.45 + k * h
        fc, ec = (GREEN_BG, GREEN) if cached else (WARM_BG, WARM)
        box(ax, x0 + wbox / 2, y + h / 2, wbox, h - 0.08, fc=fc, ec=ec, lw=1.7, r=0.05)
        ax.text(x0 + 0.15, y + h / 2, instr, fontsize=13, family=MONO, va="center")
        if note:
            ax.text(x0 + wbox + 0.15, y + h / 2, note, fontsize=13.5, va="center",
                    color=WARM if not cached else MUTED)
    ax.text(x0 + wbox / 2, 0.45 + len(rows) * h + 0.22, "image layers, built bottom-up",
            ha="center", fontsize=15, weight="bold")

    lx = 6.25
    box(ax, lx + 0.18, 3.83, 0.3, 0.3, fc=GREEN_BG, ec=GREEN, lw=1.6, r=0.03)
    ax.text(lx + 0.45, 3.83, "reused from cache", fontsize=14, va="center")
    box(ax, lx + 1.1 + 1.55, 3.83, 0.3, 0.3, fc=WARM_BG, ec=WARM, lw=1.6, r=0.03)
    ax.text(lx + 1.1 + 1.82, 3.83, "rebuilt", fontsize=14, va="center")
    lines(ax, lx + 0.0, 1.75, [("first build: ~19 s", {"fontsize": 14, "weight": "bold"}),
                               ("rebuild after the edit: < 1 s",
                                {"fontsize": 14, "weight": "bold", "color": GREEN})],
          ha="left")
    save(fig, "dockerfile-layers.png")


# --------------------------------------------------------------------------- #
# docker-compose — nginx spreads requests over three replicas of the API
# Addresses are the ones `curl -i` showed in X-Upstream on 2026-10-19
# (docker 29.8.2, compose v5.5.0, nginx 1.31.7, worker_processes 1).
# --------------------------------------------------------------------------- #
def compose_scaling() -> None:
    fig, ax = canvas(4.4)
    box(ax, 6.35, 2.15, 6.9, 3.55, fc=GREY_BG, ec=MUTED, lw=1.8, r=0.12, ls="--")
    ax.text(3.1, 3.65, "Compose network  (DNS: api → 3 addresses)", fontsize=14.5,
            weight="bold", va="center", color=MUTED)

    box(ax, 1.0, 2.0, 1.6, 0.95, fc=PAPER, ec=INK, lw=1.8)
    lines(ax, 1.0, 2.0, [("browser", {"fontsize": 14.5, "weight": "bold"}),
                         ("localhost:8080", {"fontsize": 12, "family": MONO, "color": ACCENT})])
    arrow(ax, (1.82, 2.0), (3.35, 2.0), lw=2.2)
    ax.text(2.35, 2.3, "8080:80", ha="center", fontsize=13, family=MONO, color=WARM,
            weight="bold")

    box(ax, 4.2, 2.0, 1.7, 1.15, fc=GOLD_BG, ec=GOLD, lw=2.2)
    lines(ax, 4.2, 2.0, [("nginx", {"fontsize": 16, "weight": "bold"}),
                         ("load balancer", {"fontsize": 12.5, "color": MUTED}),
                         ("round robin", {"fontsize": 12.5, "color": MUTED})])

    for k, (y, ip) in enumerate(((3.0, 4), (2.0, 3), (1.0, 2))):
        box(ax, 8.15, y, 2.9, 0.75, fc=ACCENT_BG, ec=ACCENT, lw=2.0)
        lines(ax, 8.15, y, [(f"api-{k + 1}", {"fontsize": 14.5, "weight": "bold"}),
                            (f"192.168.156.{ip}:8000",
                             {"fontsize": 12, "family": MONO, "color": MUTED})])
        arrow(ax, (5.07, 2.0 + (y - 2.0) * 0.25), (6.68, y), lw=2.0, color=ACCENT)
    ax.text(W / 2, 0.08, "one image, three containers: a request goes to the next replica in turn",
            ha="center", va="bottom", fontsize=14, color=MUTED)
    save(fig, "compose-scaling.png")


FIGURES = {f.__name__: f for f in (
    vm_vs_container, build_push_pull_run, port_mapping,
    client_server, dockerfile_layers, compose_scaling,
)}


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - set(FIGURES))
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    print("Session 11 figures ->", ASSETS)
    for name in names or FIGURES:
        FIGURES[name]()
    print("done.")


if __name__ == "__main__":
    main(sys.argv[1:])
