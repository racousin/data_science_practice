"""Figures for the Session 10 workshop path and its latency measurement.

Drawn with the helpers and palette of `s11_docker_and_deployment.py`, into the
same `assets/deploy/` folder. Re-run from `courseware/` with:

    uv run --with matplotlib python tools/figures/ms2a_s10_workshop.py [figure ...]

`latency-local-vs-render.png` plots measurements, not estimates. They were
taken on 2026-10-19 with `LATENCY` below: 200 sequential `POST /predict`
requests (urllib, one bird) against the workshop image `penguin-api`
- in its container on an Apple M4 laptop (OrbStack, docker 29.8.2),
- on Render's Free instance (Frankfurt, 0.1 CPU) once warm, from a home
  connection in France,
plus the first request after 17 minutes without traffic (the cold start).
The raw latencies live in `latency_s10.json` next to this script.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s11_docker_and_deployment as f  # noqa: E402  (helpers only)

plt = f.plt
DATA = pathlib.Path(__file__).resolve().parent / "latency_s10.json"


def workshop_path() -> None:
    fig, ax = f.canvas(2.6, w=11.0)
    y = 1.3
    stages = [
        (0.95, "Colab", "train", "model.joblib", f.GOLD_BG, f.GOLD),
        (2.75, "laptop", "uv + FastAPI", "main.py", f.ACCENT_BG, f.ACCENT),
        (4.55, "laptop", "Docker", "image", f.GREEN_BG, f.GREEN),
        (6.35, "Docker Hub", "registry", "penguin-api:v1", f.GREEN_BG, f.GREEN),
        (8.15, "Render", "cloud", "….onrender.com", f.GREY_BG, f.INK),
        (9.95, "browser", "index.html", "the form", f.WARM_BG, f.WARM),
    ]
    for i, (cx, where, what, out, fc, ec) in zip((1, 2, 3, 3, 4, 5), stages):
        f.box(ax, cx, y, 1.6, 1.25, fc=fc, ec=ec, lw=2.2)
        f.lines(ax, cx, y, [(f"{i}. {where}", {"fontsize": 13.5, "weight": "bold"}),
                            (what, {"fontsize": 12.5, "color": f.MUTED})])
        ax.text(cx, y - 0.9, out, ha="center", va="center", fontsize=11,
                color=f.MUTED, family=f.MONO)
    verbs = ["download", "docker build", "docker push", "pull + run", "GET /"]
    for (x0, *_), (x1, *_), verb in zip(stages, stages[1:], verbs):
        f.arrow(ax, (x0 + 0.8, y), (x1 - 0.8, y), lw=2.0, ms=15)
        ax.text((x0 + x1) / 2, y + 0.85, verb, ha="center", va="center",
                fontsize=11, color=f.MUTED, family=f.MONO)
    f.save(fig, "workshop-path.png")


def latency_local_vs_render() -> None:
    data = json.loads(DATA.read_text())
    local = sorted(1000 * v for v in data["local"])
    render = sorted(1000 * v for v in data["render_warm"])
    cold = 1000 * data["render_cold"]

    def q(v, p):
        return v[int(p * len(v)) - 1]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = [("laptop container", local, f.ACCENT), ("Render, warm", render, f.GREEN)]
    for i, (name, v, color) in enumerate(rows):
        ax.scatter(v, [i] * len(v), s=12, alpha=0.35, color=color, zorder=2)
        for p, mk in [(0.5, "p50"), (0.95, "p95"), (0.99, "p99")]:
            ax.plot([q(v, p)] * 2, [i - 0.22, i + 0.22], color=f.INK, lw=1.8, zorder=3)
            ax.text(q(v, p), i + 0.32 + (0.16 if mk == "p95" else 0), f"{mk} {q(v, p):.0f} ms"
                    if q(v, p) >= 10 else f"{mk} {q(v, p):.1f} ms",
                    ha="center", fontsize=11, color=f.INK)
    ax.scatter([cold], [2], s=60, color=f.WARM, zorder=3)
    ax.text(cold, 2.3, f"cold start {cold / 1000:.0f} s", ha="center", fontsize=12,
            color=f.WARM, weight="bold")
    ax.set_xscale("log")
    ax.set_yticks([0, 1, 2], ["laptop container", "Render, warm", "Render, cold"])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("latency of one POST /predict (ms, log scale)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=f.RULE, lw=0.8, zorder=0)
    fig.tight_layout()
    f.save(fig, "latency-local-vs-render.png")


FIGURES = {
    "workshop_path": workshop_path,
    "latency_local_vs_render": latency_local_vs_render,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()
