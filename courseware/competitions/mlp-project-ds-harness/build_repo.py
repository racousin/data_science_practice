#!/usr/bin/env python3
"""Assemble the public GitHub repository of the challenge (racousin/ds-harness).

    python build_repo.py            # -> dist/ds-harness/

The evaluation code the leaderboard runs (env.py, scoring.py), the local
runners, the starter kit and the format docs. Sources stay where they are
(kit/, draft/public/, env.py, repo_README.md); this only copies them. No data:
dev.json comes from the challenge page, and the private split, its seed and
the generators never leave this folder.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist" / "ds-harness"

FILES = [  # (source, name in the repository)
    ("repo_README.md", "README.md"),
    ("env.py", "env.py"),
    ("draft/public/scoring.py", "scoring.py"),
    ("draft/public/schema.md", "schema.md"),
    ("draft/public/localtest.py", "localtest.py"),
    ("kit/local_eval.py", "local_eval.py"),
    ("kit/dsh.py", "dsh.py"),
    ("kit/agent_naive.py", "agent_naive.py"),
    ("kit/agent_kit_baseline.py", "agent_kit_baseline.py"),
    ("kit/README.md", "kit_README.md"),
    ("kit/test_dsh.py", "test_dsh.py"),
]
# the platform's packages (no scikit-learn, scipy or statsmodels there), plus pytest
REQUIREMENTS = "torch\ntransformers\naccelerate\npandas\nnumpy\nsympy\nmatplotlib\npytest\n"
GITIGNORE = ("# the public tasks come from the challenge page, runs are local\n"
             "dev.json\n*.json\n__pycache__/\n")
PRIVATE_MARKERS = ("private_seed", "build_dataset", "tasks.common", "HELDOUT_FAMILIES")


def heldout_markers():
    """The private set's unseen family ids, and their multi-word names, read from
    data/private.json so that this file never spells them out."""
    import json
    with open(HERE / "data" / "private.json") as f:
        fams = {t["family"] for t in json.load(f) if t["heldout_family"]}
    return tuple(sorted(fams | {f.split(".")[-1] for f in fams if "_" in f}))


def main():
    if OUT.exists():
        # keep an existing git history; replace every tracked file
        for p in OUT.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    markers = PRIVATE_MARKERS + heldout_markers()
    for src, dst in FILES:
        text = (HERE / src).read_text()
        leaked = [m for m in markers if m in text]
        if leaked:
            raise SystemExit(f"{src} mentions private material {leaked}: fix it before publishing")
        (OUT / dst).write_text(text)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / ".gitignore").write_text(GITIGNORE)
    for p in sorted(OUT.iterdir()):
        if p.name != ".git":
            print(f"  {p.name:24s} {p.stat().st_size / 1e3:7.1f} kB")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
