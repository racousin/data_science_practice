"""Figures for `ms2a-machine-learning-practice` Session 11 (Docker and Model Deployment).

Every image the Session 11 lessons show lives under
`content/ms2a-machine-learning-practice/assets/deploy/`. This script draws the
diagrams; the screenshots next to them are stills, listed in SCREENSHOTS below
with how each was taken. Re-run from `courseware/` with:

    uv run --with matplotlib python tools/figures/s11_docker_and_deployment.py [figure ...]

With no argument it makes every figure; name functions to make only those.

Sizing follows the Session 4 script: 10 in wide at 140 dpi (1400 px), 13.5-20 pt
text, so the smallest label still lands near 10 pt on a slide.

The numbers drawn are measurements, not estimates. They were taken on
2026-09-14 on an Apple M4 laptop (OrbStack, docker 29.4.0, compose v5.1.2),
building the lessons' Iris API with python:3.13-slim already pulled:
- first `docker build`: 28.5 s; rebuild after editing main.py: 2.8 s;
- the `RUN pip install` layer: 315 MB (docker history); the image: 603 MB on
  disk, 131 MB compressed;
- CVAT v2.75.0 `docker compose up -d`: the 18 containers named in cvat_services.
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
docs-overview.png, docs-try-it.png, docs-response.png, browser-health.png
    Chromium via Playwright 1.58, 1280x800 at 2x, against the lessons' Iris API
    (fastapi 0.141.1, uvicorn 0.52.4) running in its container on port 8000.
cvat-*.png
    CVAT v2.75.0 at http://localhost:8080 after `docker compose up -d`, 1440x900,
    labelling scikit-image's public-domain sample images.
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
# why-deploy — from a notebook to something other programs can call
# --------------------------------------------------------------------------- #
def deploy_path() -> None:
    fig, ax = canvas(3.0)
    y = 1.45
    stages = [
        (0.95, "notebook", "train.py", GREY_BG, INK),
        (2.95, "model.joblib", "the trained model", GOLD_BG, GOLD),
        (4.95, "main.py", "the API", ACCENT_BG, ACCENT),
        (6.95, "image", "iris-api:1.0", GREEN_BG, GREEN),
        (8.95, "container", "on any machine", GREEN_BG, GREEN),
    ]
    for cx, name, sub, fc, ec in stages:
        box(ax, cx, y, 1.72, 1.05, fc=fc, ec=ec, lw=2.2)
        lines(ax, cx, y, [(name, {"fontsize": 16, "weight": "bold"}),
                          (sub, {"fontsize": 13, "color": MUTED})])
    verbs = ["save", "load once", "docker build", "docker run"]
    for (x0, *_), (x1, *_), verb in zip(stages, stages[1:], verbs):
        arrow(ax, (x0 + 0.88, y), (x1 - 0.88, y), lw=2.2, ms=16)
        ax.text((x0 + x1) / 2, y - 0.72, verb, ha="center", va="center",
                fontsize=13.5, color=MUTED, family=MONO)

    bracket(ax, 2.15, 3.75, 2.3, "1. the model", GOLD)
    bracket(ax, 4.15, 5.75, 2.3, "2. the API", ACCENT)
    bracket(ax, 6.15, 9.75, 2.3, "3. the environment", GREEN)

    ax.text(8.7, 0.22, "browser · app · script: POST /predict", ha="right",
            va="center", fontsize=14, color=WARM, weight="bold")
    arrow(ax, (8.8, 0.22), (8.95, y - 0.54), color=WARM, lw=2.2, rad=0.35)
    save(fig, "deploy-path.png")


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

    rows = [(2.35, "localhost:8000", "8000", "8000", "iris-api", "uvicorn on 0.0.0.0:8000"),
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
# docker-compose-cvat — the 18 containers of CVAT v2.75.0
# --------------------------------------------------------------------------- #
def cvat_services() -> None:
    fig, ax = canvas(4.9)

    def group(cx, cy, w, h, title, ec, fc):
        box(ax, cx, cy, w, h, fc=fc, ec=ec, lw=1.8, ls="--", r=0.1, z=1)
        ax.text(cx - w / 2 + 0.12, cy + h / 2 - 0.2, title, fontsize=13.5, color=ec,
                weight="bold", va="center")

    def svc(cx, cy, name, sub, w=1.9, ec=INK):
        box(ax, cx, cy, w, 0.62, fc=PAPER, ec=ec, lw=1.6)
        lines(ax, cx, cy, [(name, {"fontsize": 12.5, "family": MONO, "weight": "bold"}),
                           (sub, {"fontsize": 11.5, "color": MUTED})], spacing=1.2)

    box(ax, 0.8, 3.6, 1.2, 0.8, fc=GREY_BG, ec=INK, lw=1.8)
    lines(ax, 0.8, 3.6, [("browser", {"fontsize": 14, "weight": "bold"}),
                         (":8080", {"fontsize": 13, "family": MONO, "color": ACCENT})])

    svc(2.75, 3.6, "traefik", "the front door", ec=GOLD)
    arrow(ax, (1.42, 3.6), (1.78, 3.6), lw=2)

    group(5.45, 3.6, 2.9, 1.9, "the application", ACCENT, ACCENT_BG)
    svc(5.45, 3.95, "cvat_ui", "the web page", w=2.3, ec=ACCENT)
    svc(5.45, 3.15, "cvat_server", "the REST API", w=2.3, ec=ACCENT)
    arrow(ax, (3.72, 3.6), (3.98, 3.6), lw=2)

    group(8.55, 3.6, 2.7, 1.9, "background jobs", ACCENT, ACCENT_BG)
    lines(ax, 8.55, 3.45, [("8 × cvat_worker_*", {"fontsize": 13, "family": MONO, "weight": "bold"}),
                           ("import, export, chunks,", {"fontsize": 12, "color": MUTED}),
                           ("annotation, quality, ...", {"fontsize": 12, "color": MUTED})])
    arrow(ax, (6.92, 3.6), (7.18, 3.6), lw=2)

    group(3.6, 1.35, 6.4, 1.75, "storage", GREEN, GREEN_BG)
    svc(1.55, 1.2, "cvat_db", "PostgreSQL", w=1.8, ec=GREEN)
    svc(3.6, 1.2, "cvat_redis_*", "2 caches / queues", w=2.1, ec=GREEN)
    svc(5.65, 1.2, "cvat_opa", "permissions", w=1.8, ec=GREEN)

    group(8.55, 1.35, 2.7, 1.75, "analytics", GOLD, GOLD_BG)
    lines(ax, 8.55, 1.15, [("vector → clickhouse", {"fontsize": 12.5, "family": MONO}),
                           ("→ grafana", {"fontsize": 12.5, "family": MONO})])

    arrow(ax, (5.45, 2.62), (4.3, 2.26), lw=1.8, color=MUTED)
    arrow(ax, (8.0, 2.62), (6.4, 2.26), lw=1.8, color=MUTED)
    ax.text(W / 2, 0.14, "18 containers, one docker-compose.yml, one command: docker compose up -d",
            ha="center", va="bottom", fontsize=14.5, weight="bold", color=WARM)
    save(fig, "cvat-services.png")


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
                          ('{"sepal_length": 6.7, ...}', {"fontsize": 12.5, "family": MONO})],
          spacing=1.2)

    arrow(ax, (7.65, 1.05), (2.35, 1.05), color=GREEN, lw=2.6, ms=22)
    box(ax, 5.0, 0.58, 4.9, 0.62, fc=GREEN_BG, ec=GREEN, lw=1.6)
    lines(ax, 5.0, 0.58, [("response   200 OK", {"fontsize": 14, "family": MONO, "weight": "bold"}),
                          ('{"species": "virginica"}', {"fontsize": 12.5, "family": MONO})],
          spacing=1.2)
    lines(ax, 5.0, 2.0, [("the model runs here, on the server:", {"fontsize": 14, "color": MUTED}),
                         ("the client never sees Python, sklearn or the model file",
                          {"fontsize": 14, "color": MUTED})])
    save(fig, "client-server.png")


# --------------------------------------------------------------------------- #
# apis-and-http — what each part of a URL chooses
# --------------------------------------------------------------------------- #
def url_anatomy() -> None:
    fig, ax = canvas(2.3)
    parts = [("http://", "scheme", "the protocol", MUTED),
             ("localhost", "host", "which machine", ACCENT),
             (":8000", "port", "which program", GOLD),
             ("/predict", "path", "which endpoint", WARM)]
    fs = 30
    char_w = 0.2505          # DejaVu Sans Mono advance at 30 pt, in inches
    total = sum(len(p) for p, *_ in parts) * char_w
    x = (W - total) / 2
    for text, name, sub, color in parts:
        w = len(text) * char_w
        ax.text(x, 1.6, text, fontsize=fs, family=MONO, color=color, weight="bold",
                va="center", ha="left")
        ax.plot([x + 0.04, x + w - 0.04], [1.18, 1.18], color=color, lw=3)
        lines(ax, x + w / 2, 0.68, [(name, {"fontsize": 15.5, "weight": "bold", "color": color}),
                                    (sub, {"fontsize": 13, "color": MUTED})])
        x += w
    save(fig, "url-anatomy.png")


# --------------------------------------------------------------------------- #
# writing-a-dockerfile — one instruction, one layer, and the cache
# --------------------------------------------------------------------------- #
def dockerfile_layers() -> None:
    fig, ax = canvas(4.1)
    rows = [
        ("FROM python:3.13-slim", "the base: Debian + Python", True),
        ("WORKDIR /app", "", True),
        ("COPY requirements.txt .", "", True),
        ("RUN pip install -r requirements.txt", "315 MB, the slow step", True),
        ("COPY main.py model.joblib ./", "← you edited main.py", False),
        ("EXPOSE 8000 · CMD [\"uvicorn\", ...]", "", False),
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
    box(ax, lx + 0.18, 3.55, 0.3, 0.3, fc=GREEN_BG, ec=GREEN, lw=1.6, r=0.03)
    ax.text(lx + 0.45, 3.55, "reused from cache", fontsize=14, va="center")
    box(ax, lx + 1.1 + 1.55, 3.55, 0.3, 0.3, fc=WARM_BG, ec=WARM, lw=1.6, r=0.03)
    ax.text(lx + 1.1 + 1.82, 3.55, "rebuilt", fontsize=14, va="center")
    lines(ax, lx + 0.0, 1.47, [("first build: 28.5 s", {"fontsize": 14, "weight": "bold"}),
                               ("rebuild after the edit: 2.8 s",
                                {"fontsize": 14, "weight": "bold", "color": GREEN})],
          ha="left")
    save(fig, "dockerfile-layers.png")


FIGURES = {f.__name__: f for f in (
    deploy_path, vm_vs_container, build_push_pull_run, port_mapping, cvat_services,
    client_server, url_anatomy, dockerfile_layers,
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
