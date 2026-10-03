#!/usr/bin/env python3
"""Build data/ for the DS-Harness challenge.

    uv run --with pandas --with numpy python prepare_data.py

data/ (gitignored) then holds:
  private.json          the private split (env file, never published), drawn with
                        the secret seed in data/private_seed.txt. The seed is
                        created on the first run and reused after, so the build
                        is re-runnable; delete it to draw a new private split.
  dev.json, scoring.py  the public task set (published) and the scorer (an env
                        file), from draft/public/, rebuilt by
                        draft/build_dataset.py dev, which is deterministic
The code students get is the GitHub repository, assembled by build_repo.py.
"""
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / "draft"
DATA = HERE / "data"
SEED_FILE = DATA / "private_seed.txt"

PUBLIC = ["dev.json", "scoring.py"]


def build(*args):
    subprocess.run([sys.executable, "build_dataset.py", *args], cwd=DRAFT, check=True)


def main():
    DATA.mkdir(exist_ok=True)
    if not SEED_FILE.exists():
        SEED_FILE.write_text(f"{secrets.randbits(31)}\n")
        print(f"new secret seed written to {SEED_FILE}")
    seed = int(SEED_FILE.read_text())

    build("dev")
    build("private", "--seed", str(seed), "--out", str(DATA / "private.json"))

    for name in PUBLIC:
        shutil.copy2(DRAFT / "public" / name, DATA / name)
    for p in sorted(DATA.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
