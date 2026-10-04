#!/usr/bin/env python3
"""Assemble the public GitHub repository of the challenge (racousin/ds-harness).

    python build_repo.py            # -> dist/ds-harness/ (its own git, pushed by hand)

The code the leaderboard runs (env.py, scoring.py), the local runner, the kit and
the format doc. Sources stay where they are (public/, kit/, env.py);
this only copies them. No data: dev.json comes from the
challenge's dataset, and the private split, its seed, the generators and the
anchors never leave this folder.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist" / "ds-harness"
sys.path.insert(0, str(HERE))

FILES = [  # (source, name in the repository)
    ("public/README.md", "README.md"),
    ("env.py", "env.py"),
    ("public/scoring.py", "scoring.py"),
    ("public/schema.md", "schema.md"),
    ("public/localtest.py", "localtest.py"),
    ("kit/dsh.py", "dsh.py"),
    ("kit/stage1_direct.py", "stage1_direct.py"),
    ("kit/stage2_tool_loop.py", "stage2_tool_loop.py"),
    ("kit/test_dsh.py", "test_dsh.py"),
    ("hf_models.json", "hf_models.json"),
]
# the platform's agent packages, plus pytest
REQUIREMENTS = "torch\ntransformers\naccelerate\npandas\nnumpy\nscipy\nscikit-learn\nsympy\npytest\n"
GITIGNORE = "# the tasks come from the challenge's dataset; runs are local\ndev.json\nrun*.json\n__pycache__/\n.pytest_cache/\n"


def private_markers():
    """Names that would give away private material: the private-only task types and
    the paths of the generators, seed and anchors."""
    from gen.tables import PRIVATE_TYPES
    return tuple(PRIVATE_TYPES) + ("private_seed", "private_specs", "from gen", "gen/", "anchors2", "stage2_solved")


def main():
    if OUT.exists():
        for p in OUT.iterdir():  # keep the git history, replace every file
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    markers = private_markers()
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
